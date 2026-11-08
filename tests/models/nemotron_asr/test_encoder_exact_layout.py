# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bitwise-exact in-place encoder cache advance.

The candidate ``stream_step`` gathers each cache advance straight into its
storage. The oracle below is the pre-change step: it materializes each
advanced cache and copies it back. Every comparison is exact.
"""

import copy

import pytest
import torch

from vllm.models.nemotron_asr.advance import _GatheredCaches
from vllm.models.nemotron_asr.encoder import (
    FastConformerEncoder,
    StreamingCaches,
    _stream_attention,
    _stream_attention_mask,
    _stream_cache_indices,
    _stream_conv,
    stream_step,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]

_N_LAYERS = 2
_D_MODEL = 32
_LEFT = 8
_KERNEL = 5
_BATCH = 4
_CHUNKS = 60
_MEL = 41


def _encoder(dtype: torch.dtype) -> FastConformerEncoder:
    torch.manual_seed(31)
    encoder = FastConformerEncoder(
        feat_in=16,
        d_model=_D_MODEL,
        d_ff=64,
        n_layers=_N_LAYERS,
        n_heads=4,
        conv_kernel=_KERNEL,
        subsampling_channels=16,
        att_context=(_LEFT, 1),
    )
    return encoder.to(dtype).eval()


def _reference_stream_step(
    encoder: FastConformerEncoder,
    chunk_mel: torch.Tensor,
    caches: StreamingCaches,
    *,
    out_offsets: torch.Tensor,
    out_lengths: torch.Tensor,
    out_width: int,
    preserve_conv_cache_precision: bool,
) -> torch.Tensor:
    """The pre-change step: materialize each advance, then copy it back."""
    b = chunk_mel.shape[0]
    device = chunk_mel.device
    lengths = torch.full((b,), chunk_mel.shape[2], device=device)
    x, _ = encoder.pre_encode(chunk_mel, lengths)
    gidx = (
        out_offsets.view(-1, 1) + torch.arange(out_width, device=device).unsqueeze(0)
    ).clamp(max=max(x.shape[1] - 1, 0))
    x = x.gather(1, gidx.unsqueeze(-1).expand(b, out_width, x.shape[2]))
    new_valid = torch.arange(out_width, device=device).unsqueeze(0) < out_lengths.view(
        -1, 1
    )
    cache_len = caches.channel.shape[2]
    mask = _stream_attention_mask(caches.valid, new_valid, cache_len)
    attn_indices = (
        _stream_cache_indices(out_lengths, cache_len)
        .unsqueeze(-1)
        .expand(b, cache_len, x.shape[2])
    )
    conv_len = caches.time.shape[-1]
    conv_indices = (
        _stream_cache_indices(out_lengths, conv_len)
        .unsqueeze(1)
        .expand(b, x.shape[2], conv_len)
    )
    pos_emb = encoder.pos_enc(
        torch.zeros(1, out_width + cache_len, x.shape[2], device=device, dtype=x.dtype)
    )
    for idx, layer in enumerate(encoder.layers):
        residual = x
        y = layer.norm_feed_forward1(x)
        residual = residual + 0.5 * layer.feed_forward1(y)
        y = layer.norm_self_att(residual)
        attn_out, caches.channel[idx] = _stream_attention(
            layer,
            y,
            cache=caches.channel[idx],
            valid=caches.valid,
            pos_emb=pos_emb,
            new_valid=new_valid,
            new_lengths=out_lengths,
            mask=mask,
            cache_indices=attn_indices,
        )
        residual = residual + attn_out
        y = layer.norm_conv(residual)
        conv_out, caches.time[idx] = _stream_conv(
            layer,
            y,
            caches.time[idx],
            new_lengths=out_lengths,
            cache_indices=conv_indices,
            preserve_cache_precision=preserve_conv_cache_precision,
        )
        residual = residual + conv_out
        y = layer.norm_feed_forward2(residual)
        residual = residual + 0.5 * layer.feed_forward2(y)
        x = layer.norm_out(residual)
    caches.valid = torch.clamp(
        caches.valid + out_lengths.to(caches.valid.dtype), max=caches.left_context
    )
    return torch.where(new_valid.unsqueeze(-1), x, x.new_zeros(()))


def _caches(state_dtype: torch.dtype, seed: int) -> StreamingCaches:
    generator = torch.Generator().manual_seed(seed)
    caches = StreamingCaches(
        n_layers=_N_LAYERS,
        batch=_BATCH,
        d_model=_D_MODEL,
        left_context=_LEFT,
        conv_kernel=_KERNEL,
        device=torch.device("cpu"),
        dtype=state_dtype,
    )
    caches.channel.copy_(torch.randn(caches.channel.shape, generator=generator))
    caches.time.copy_(torch.randn(caches.time.shape, generator=generator))
    caches.valid = torch.randint(0, _LEFT + 1, (_BATCH,), generator=generator)
    return caches


def _gathered(caches: StreamingCaches) -> _GatheredCaches:
    """Same values over per-layer NON-contiguous views (row axis strided)."""

    def strided(per_layer: torch.Tensor) -> torch.Tensor:
        storage = torch.empty(
            (per_layer.shape[1], per_layer.shape[0], *per_layer.shape[2:]),
            dtype=per_layer.dtype,
        )
        view = storage.transpose(0, 1)
        view.copy_(per_layer)
        assert not view.is_contiguous()
        return view

    return _GatheredCaches._from_tensors(
        channel=tuple(strided(caches.channel[layer]) for layer in range(_N_LAYERS)),
        time=tuple(strided(caches.time[layer]) for layer in range(_N_LAYERS)),
        valid=tuple(
            caches.valid.to(torch.int32).reshape(-1, 1).clone()
            for _ in range(_N_LAYERS)
        ),
        left_context=_LEFT,
    )


def _families(caches) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    channel = torch.stack([caches.channel[layer] for layer in range(_N_LAYERS)])
    time = torch.stack([caches.time[layer] for layer in range(_N_LAYERS)])
    return channel, time, caches.valid


def _schedule(compute_dtype: torch.dtype, out_width: int, seed: int):
    """Mixed per-row offsets/advance counts; every chunk has an f=0 row."""
    generator = torch.Generator().manual_seed(seed)
    for chunk in range(_CHUNKS):
        mel = torch.randn(_BATCH, 16, _MEL, generator=generator).to(compute_dtype)
        offsets = torch.randint(0, 3, (_BATCH,), generator=generator)
        lengths = torch.randint(0, out_width + 1, (_BATCH,), generator=generator)
        lengths[chunk % _BATCH] = 0
        yield mel, offsets, lengths


def _out_width(encoder: FastConformerEncoder, dtype: torch.dtype) -> int:
    with torch.inference_mode():
        x, _ = encoder.pre_encode(
            torch.zeros(1, 16, _MEL, dtype=dtype), torch.tensor([_MEL])
        )
    return x.shape[1] - 2


_LANES = {
    # name: (encoder compute dtype, cache state dtype, preserve conv history, gathered
    # views)
    "fp32": (torch.float32, torch.float32, False, False),
    "fp32-strided-views": (torch.float32, torch.float32, False, True),
    "fp16-encoder-preserve": (torch.float16, torch.float32, True, False),
    "fp16-encoder-castcopy": (torch.float16, torch.float32, False, True),
    "fp16-state": (torch.float16, torch.float16, False, False),
}


@pytest.mark.parametrize("lane", sorted(_LANES))
@torch.inference_mode()
def test_stream_step_bitwise_matches_pre_change_path(lane: str) -> None:
    """Outputs and every cache family are bit-identical over 60 chunks."""
    compute, state, preserve, gathered = _LANES[lane]
    oracle_encoder = _encoder(compute)
    encoder = copy.deepcopy(oracle_encoder)
    out_width = _out_width(encoder, compute)
    oracle_caches = _caches(state, seed=5)
    candidate = _gathered(oracle_caches) if gathered else copy.deepcopy(oracle_caches)
    if gathered:
        pointers = [candidate.channel[i].data_ptr() for i in range(_N_LAYERS)]
        pointers += [candidate.time[i].data_ptr() for i in range(_N_LAYERS)]
    else:
        pointers = [candidate.channel.data_ptr(), candidate.time.data_ptr()]
    zero_rows = 0
    for mel, offsets, lengths in _schedule(compute, out_width, seed=11):
        zero_rows += int((lengths == 0).sum())
        expected = _reference_stream_step(
            oracle_encoder,
            mel,
            oracle_caches,
            out_offsets=offsets,
            out_lengths=lengths,
            out_width=out_width,
            preserve_conv_cache_precision=preserve,
        )
        actual = stream_step(
            encoder,
            mel,
            candidate,
            out_offsets=offsets,
            out_lengths=lengths,
            out_width=out_width,
            preserve_conv_cache_precision=preserve,
        )
        assert torch.equal(actual, expected)
        for got, want in zip(_families(candidate), _families(oracle_caches)):
            assert got.dtype == want.dtype
            assert torch.equal(got, want)
    assert zero_rows >= _CHUNKS
    if gathered:
        now = [candidate.channel[i].data_ptr() for i in range(_N_LAYERS)]
        now += [candidate.time[i].data_ptr() for i in range(_N_LAYERS)]
    else:
        now = [candidate.channel.data_ptr(), candidate.time.data_ptr()]
    assert now == pointers


@pytest.mark.parametrize("compute", [torch.float32, torch.float16])
@torch.inference_mode()
def test_in_place_advance_keeps_rows_isolated(compute: torch.dtype) -> None:
    """Perturbing one row's audio changes only that row's output and caches."""
    encoder = _encoder(compute)
    out_width = _out_width(encoder, compute)
    base, perturbed = _caches(torch.float32, seed=7), _caches(torch.float32, seed=7)
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


@torch.inference_mode()
def test_zero_length_rows_leave_caches_bit_identical_in_place() -> None:
    encoder = _encoder(torch.float32)
    out_width = _out_width(encoder, torch.float32)
    caches = _caches(torch.float32, seed=3)
    channel, time = caches.channel.clone(), caches.time.clone()
    stream_step(
        encoder,
        torch.randn(_BATCH, 16, _MEL),
        caches,
        out_offsets=torch.zeros(_BATCH, dtype=torch.long),
        out_lengths=torch.tensor([0, 3, 0, 1]),
        out_width=out_width,
    )
    for row in (0, 2):
        assert torch.equal(caches.channel[:, row], channel[:, row])
        assert torch.equal(caches.time[:, row], time[:, row])
    for row in (1, 3):
        assert not torch.equal(caches.channel[:, row], channel[:, row])
