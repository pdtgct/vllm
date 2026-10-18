# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Head-major relative-position scores in the streaming encoder.

The candidate stores each prepared positional projection head-major and
computes the position scores as one head-batched GEMM over ``B*F`` rows,
with the relative shift taken as a strided view. The oracle below is the
pre-change attention: per-row ``(B, h, F, d_k) @ (1, h, d_k, 2T-1)`` with
the padded shift. FP32 must be bit-identical; FP16 GEMM shapes change, so
its difference is reported against a provisional bound.
"""

import copy

import pytest
import torch

from vllm.models.nemotron_asr import encoder as encoder_module
from vllm.models.nemotron_asr.encoder import (
    FastConformerEncoder,
    RelPositionMHA,
    StreamingCaches,
    _stream_attention_mask,
    _stream_cache_indices,
    _stream_rel_shift,
    stream_step,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]

_N_LAYERS = 2
_D_MODEL = 128
_HEADS = 8
_LEFT = 8
_KERNEL = 5
_BATCH = 4
_CHUNKS = 60
_MEL = 41

# PROVISIONAL, lead decision pending: no project tolerance governs an FP16
# encoder GEMM-shape change (parity-tolerances.json holds only exact zero).
# Measured 2**-9 on CPU for this tiny model; the bound is twice that and is
# not a qualification tolerance.
_PROVISIONAL_FP16_MAX_ABS = 2.0**-8


def _encoder(dtype: torch.dtype) -> FastConformerEncoder:
    torch.manual_seed(41)
    encoder = FastConformerEncoder(
        feat_in=16,
        d_model=_D_MODEL,
        d_ff=256,
        n_layers=_N_LAYERS,
        n_heads=_HEADS,
        conv_kernel=_KERNEL,
        subsampling_channels=16,
        att_context=(_LEFT, 1),
    )
    with torch.no_grad():
        for layer in encoder.layers:
            layer.self_attn.pos_bias_u.normal_(0.0, 0.1)
            layer.self_attn.pos_bias_v.normal_(0.0, 0.1)
    return encoder.to(dtype).eval()


def _pre_change_attention(
    layer,
    x,
    *,
    cache,
    valid,
    pos_emb,
    new_valid,
    new_lengths,
    projected_pos=None,
    mask=None,
    cache_indices=None,
    cache_out=None,
    sdpa=None,
):
    """The pre-change streaming attention, always from ``pos_emb``."""
    assert sdpa is None
    attn = layer.self_attn
    b, new_frames, _ = x.shape
    capacity = cache.shape[1]
    keys = torch.cat([cache.to(x.dtype), x], dim=1)
    t2 = keys.shape[1]
    q = attn.linear_q(x).view(b, new_frames, attn.h, attn.d_k)
    k = attn.linear_k(keys).view(b, t2, attn.h, attn.d_k).transpose(1, 2)
    v = attn.linear_v(keys).view(b, t2, attn.h, attn.d_k).transpose(1, 2)
    p = (
        attn.linear_pos(pos_emb)
        .view(pos_emb.size(0), -1, attn.h, attn.d_k)
        .transpose(1, 2)
    )
    q_u = (q + attn.pos_bias_u).transpose(1, 2)
    q_v = (q + attn.pos_bias_v).transpose(1, 2)
    matrix_bd = attn._rel_shift(torch.matmul(q_v, p.transpose(-2, -1)))
    matrix_ac = torch.matmul(q_u, k.transpose(-2, -1))
    scores = (matrix_ac + matrix_bd[:, :, :, : matrix_ac.size(-1)]) / attn.s_d_k
    if mask is None:
        mask = _stream_attention_mask(valid, new_valid, capacity)
    scores = scores.masked_fill(mask, -encoder_module._LOG_BASE)
    weights = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
    out = (
        torch.matmul(weights, v)
        .transpose(1, 2)
        .reshape(b, new_frames, attn.h * attn.d_k)
    )
    if cache_indices is None:
        cache_indices = (
            _stream_cache_indices(new_lengths, capacity)
            .unsqueeze(-1)
            .expand(b, capacity, keys.shape[2])
        )
    source = torch.cat([cache, x.to(cache.dtype)], dim=1)
    if cache_out is None:
        new_cache = source.gather(1, cache_indices)
    else:
        new_cache = torch.gather(source, 1, cache_indices, out=cache_out)
    return attn.linear_out(out), new_cache


def _caches(seed: int) -> StreamingCaches:
    generator = torch.Generator().manual_seed(seed)
    caches = StreamingCaches(
        n_layers=_N_LAYERS,
        batch=_BATCH,
        d_model=_D_MODEL,
        left_context=_LEFT,
        conv_kernel=_KERNEL,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    caches.channel.copy_(torch.randn(caches.channel.shape, generator=generator))
    caches.time.copy_(torch.randn(caches.time.shape, generator=generator))
    caches.valid = torch.randint(0, _LEFT + 1, (_BATCH,), generator=generator)
    return caches


def _out_width(encoder: FastConformerEncoder, dtype: torch.dtype) -> int:
    with torch.inference_mode():
        x, _ = encoder.pre_encode(
            torch.zeros(1, 16, _MEL, dtype=dtype), torch.tensor([_MEL])
        )
    return x.shape[1] - 2


def _schedule(compute: torch.dtype, out_width: int, seed: int):
    """Mixed per-row offsets/advance counts; every chunk has an f=0 row."""
    generator = torch.Generator().manual_seed(seed)
    for chunk in range(_CHUNKS):
        mel = torch.randn(_BATCH, 16, _MEL, generator=generator).to(compute)
        offsets = torch.randint(0, 3, (_BATCH,), generator=generator)
        lengths = torch.randint(0, out_width + 1, (_BATCH,), generator=generator)
        lengths[chunk % _BATCH] = 0
        yield mel, offsets, lengths


def _prepare(
    encoder: FastConformerEncoder, compute: torch.dtype, out_width: int
) -> None:
    encoder.prepare_stream_relative_position_projections(
        out_widths=(out_width,),
        cache_len=_LEFT,
        reference=torch.zeros(1, dtype=compute),
    )


@pytest.mark.parametrize(
    ("queries", "keys"), [(1, 1), (1, 9), (4, 12), (7, 15), (12, 12)]
)
def test_strided_rel_shift_equals_padded_shift_and_crop(
    queries: int, keys: int
) -> None:
    batch, heads = 3, 2
    raw = torch.randn(heads, batch * queries, 2 * keys - 1)
    per_row = raw.view(heads, batch, queries, -1).transpose(0, 1).contiguous()
    expected = RelPositionMHA._rel_shift(per_row)[..., :keys]
    actual = _stream_rel_shift(raw, batch=batch, queries=queries, keys=keys)
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


@torch.inference_mode()
def test_prepared_projection_is_contiguous_head_major() -> None:
    encoder = _encoder(torch.float32)
    out_width = _out_width(encoder, torch.float32)
    _prepare(encoder, torch.float32, out_width)
    projections = encoder.stream_relative_position_projections(
        out_width=out_width, cache_len=_LEFT, reference=torch.zeros(1)
    )
    assert projections is not None
    length = out_width + _LEFT
    for projection in projections:
        assert projection.shape == (_HEADS, 2 * length - 1, _D_MODEL // _HEADS)
        assert projection.is_contiguous()


def _compare(compute: torch.dtype, prepared: bool) -> tuple[float, int]:
    oracle_encoder = _encoder(compute)
    encoder = copy.deepcopy(oracle_encoder)
    out_width = _out_width(encoder, compute)
    if prepared:
        _prepare(encoder, compute, out_width)
    oracle, candidate = _caches(seed=5), _caches(seed=5)
    preserve = compute == torch.float16
    worst, zero_rows = 0.0, 0
    candidate_attention = encoder_module._stream_attention
    for mel, offsets, lengths in _schedule(compute, out_width, seed=11):
        zero_rows += int((lengths == 0).sum())
        kwargs = dict(
            out_offsets=offsets,
            out_lengths=lengths,
            out_width=out_width,
            preserve_conv_cache_precision=preserve,
        )
        before = (candidate.channel.clone(), candidate.time.clone())
        encoder_module._stream_attention = _pre_change_attention
        try:
            expected = stream_step(oracle_encoder, mel, oracle, **kwargs)
        finally:
            encoder_module._stream_attention = candidate_attention
        actual = stream_step(encoder, mel, candidate, **kwargs)
        # Zero-advance rows keep their caches bit-identical and emit zeros.
        for row in torch.nonzero(lengths == 0).flatten().tolist():
            assert torch.equal(candidate.channel[:, row], before[0][:, row])
            assert torch.equal(candidate.time[:, row], before[1][:, row])
            assert not actual[row].any()
        assert torch.equal(candidate.valid, oracle.valid)
        assert (
            candidate.channel.dtype == torch.float32
            and candidate.time.dtype == torch.float32
        )
        for got, want in (
            (actual, expected),
            (candidate.channel, oracle.channel),
            (candidate.time, oracle.time),
        ):
            worst = max(worst, (got.float() - want.float()).abs().max().item())
    return worst, zero_rows


@pytest.mark.parametrize(
    "prepared", [False, True], ids=["eager-projection", "prepared-projection"]
)
@torch.inference_mode()
def test_fp32_head_major_scores_bitwise_match_pre_change(prepared: bool) -> None:
    worst, zero_rows = _compare(torch.float32, prepared)
    assert zero_rows >= _CHUNKS
    assert worst == 0.0


@pytest.mark.parametrize(
    "prepared", [False, True], ids=["eager-projection", "prepared-projection"]
)
@torch.inference_mode()
def test_fp16_head_major_scores_within_provisional_bound(prepared: bool) -> None:
    worst, zero_rows = _compare(torch.float16, prepared)
    assert zero_rows >= _CHUNKS
    print(f"fp16 head-major max |delta| vs pre-change: {worst:.6g}")
    assert worst <= _PROVISIONAL_FP16_MAX_ABS


@pytest.mark.parametrize("compute", [torch.float32, torch.float16])
@torch.inference_mode()
def test_prepared_head_major_keeps_rows_isolated(compute: torch.dtype) -> None:
    """Perturbing one row's audio changes only that row's output and caches."""
    encoder = _encoder(compute)
    out_width = _out_width(encoder, compute)
    _prepare(encoder, compute, out_width)
    base, perturbed = _caches(seed=7), _caches(seed=7)
    victim = 2
    for mel, offsets, lengths in _schedule(compute, out_width, seed=13):
        other = mel.clone()
        other[victim] = other[victim] + 0.25
        kwargs = dict(
            out_offsets=offsets,
            out_lengths=lengths,
            out_width=out_width,
            preserve_conv_cache_precision=compute == torch.float16,
        )
        expected = stream_step(encoder, mel, base, **kwargs)
        actual = stream_step(encoder, other, perturbed, **kwargs)
        keep = [row for row in range(_BATCH) if row != victim]
        assert torch.equal(actual[keep], expected[keep])
        assert torch.equal(perturbed.channel[:, keep], base.channel[:, keep])
        assert torch.equal(perturbed.time[:, keep], base.time[:, keep])
        assert torch.equal(perturbed.valid, base.valid)
    assert not torch.equal(perturbed.channel[:, victim], base.channel[:, victim])
