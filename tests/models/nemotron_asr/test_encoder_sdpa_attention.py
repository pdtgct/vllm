# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused SDPA attention core of the FP16 streaming encoder lane.

The candidate replaces the content-score GEMM, add, scale, mask, softmax,
weight mask and value GEMM with one ``scaled_dot_product_attention`` call
whose float bias is the scaled, additively masked position score, then
zeroes fully masked query rows (NeMo's public SDPA formulation). SDPA
reorders the reduction, so FP32 is not bitwise and keeps the eager core;
only the encoder FP16 policy selects SDPA. The FP16 lane is checked by a
regression guard against the FP32 reference (lead decision 2026-10-05),
not by drift from the previous FP16 implementation.
"""

import copy

import pytest
import torch
from test_encoder_position_head_major import (
    _BATCH,
    _CHUNKS,
    _D_MODEL,
    _KERNEL,
    _LEFT,
    _MEL,
    _N_LAYERS,
    _caches,
    _encoder,
    _out_width,
    _prepare,
    _schedule,
)

from vllm.models.nemotron_asr import encoder as encoder_module
from vllm.models.nemotron_asr.encoder import (
    FastConformerEncoder,
    StreamSDPAMask,
    _sdpa_bias_width,
    _stream_attention,
    _stream_attention_mask,
    _stream_cache_indices,
    _stream_rel_shift,
    stream_step,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]

# Real-dimension chunks kept within the ~60 s CPU budget for three arms (60
# chunks take ~210 s; measured 2026-10-05 at 60 chunks, err_new/err_old max
# 0.01476/0.01547, mean 0.0010025/0.0010029). The tiny-model case runs 60.
_REAL_DIMENSION_CHUNKS = 16


def _compare(compute: torch.dtype) -> tuple[float, int]:
    """Run eager and SDPA arms over one mixed schedule; return max |delta|."""
    encoder = _encoder(compute)
    out_width = _out_width(encoder, compute)
    _prepare(encoder, compute, out_width)
    oracle, candidate = _caches(seed=5), _caches(seed=5)
    worst, zero_rows = 0.0, 0
    for mel, offsets, lengths in _schedule(compute, out_width, seed=11):
        zero_rows += int((lengths == 0).sum())
        kwargs = dict(
            out_offsets=offsets,
            out_lengths=lengths,
            out_width=out_width,
            preserve_conv_cache_precision=compute == torch.float16,
        )
        before = (candidate.channel.clone(), candidate.time.clone())
        expected = stream_step(encoder, mel, oracle, **kwargs)
        actual = stream_step(encoder, mel, candidate, sdpa_attention=True, **kwargs)
        # Zero-advance rows keep their caches bit-identical and emit zeros.
        for row in torch.nonzero(lengths == 0).flatten().tolist():
            assert torch.equal(candidate.channel[:, row], before[0][:, row])
            assert torch.equal(candidate.time[:, row], before[1][:, row])
            assert not actual[row].any()
        assert torch.isfinite(actual).all()
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


# Regression-guard factors -- lead decision 2026-10-05: the FP16 SDPA lane is
# judged against the FP32 reference, not against drift from the previous FP16
# implementation. The SDPA error may exceed the previous eager-FP16 error by at
# most these factors. This is a regression guard, NOT a qualification: any FP16
# numerics change still requires the pre-registered WER gate before promotion.
_FP16_SDPA_MAX_ERROR_FACTOR = 1.25
_FP16_SDPA_MEAN_ERROR_FACTOR = 1.10


def _fp32_referenced_errors(
    make_encoder,
    *,
    batch: int,
    left: int,
    n_layers: int,
    d_model: int,
    conv_kernel: int,
    feat_in: int,
    mel_width: int,
    chunks: int,
) -> dict[str, tuple[float, float]]:
    """Max/mean |arm - FP32 reference| for eager-FP16 and SDPA-FP16 arms.

    Every arm uses the same weights (FP16 arms are casts of the FP32
    encoder), the same FP32 initial caches and the same audio; errors pool
    the outputs and both cache families over a mixed schedule in which every
    chunk has a zero-advance row.
    """
    from vllm.models.nemotron_asr.encoder import StreamingCaches

    base = make_encoder()
    arms = {
        "fp32": (torch.float32, False),
        "eager16": (torch.float16, False),
        "sdpa16": (torch.float16, True),
    }
    encoders = {
        dtype: copy.deepcopy(base).to(dtype).eval()
        for dtype in (torch.float32, torch.float16)
    }
    with torch.inference_mode():
        x, _ = encoders[torch.float32].pre_encode(
            torch.zeros(1, feat_in, mel_width), torch.tensor([mel_width])
        )
        out_width = x.shape[1] - 2
        for dtype, encoder in encoders.items():
            encoder.prepare_stream_relative_position_projections(
                out_widths=(out_width,),
                cache_len=left,
                reference=torch.zeros(1, dtype=dtype),
            )

    def caches() -> StreamingCaches:
        generator = torch.Generator().manual_seed(5)
        state = StreamingCaches(
            n_layers=n_layers,
            batch=batch,
            d_model=d_model,
            left_context=left,
            conv_kernel=conv_kernel,
            device=torch.device("cpu"),
        )
        state.channel.copy_(torch.randn(state.channel.shape, generator=generator))
        state.time.copy_(torch.randn(state.time.shape, generator=generator))
        state.valid = torch.randint(0, left + 1, (batch,), generator=generator)
        return state

    states = {name: caches() for name in arms}
    errors: dict[str, list[torch.Tensor]] = {"eager16": [], "sdpa16": []}
    generator = torch.Generator().manual_seed(11)
    zero_rows = 0
    with torch.inference_mode():
        for chunk in range(chunks):
            mel = torch.randn(batch, feat_in, mel_width, generator=generator)
            offsets = torch.randint(0, 3, (batch,), generator=generator)
            lengths = torch.randint(0, out_width + 1, (batch,), generator=generator)
            lengths[chunk % batch] = 0
            zero_rows += int((lengths == 0).sum())
            outputs = {}
            for name, (dtype, sdpa) in arms.items():
                before = (states[name].channel.clone(), states[name].time.clone())
                outputs[name] = stream_step(
                    encoders[dtype],
                    mel.to(dtype),
                    states[name],
                    out_offsets=offsets,
                    out_lengths=lengths,
                    out_width=out_width,
                    preserve_conv_cache_precision=dtype == torch.float16,
                    sdpa_attention=sdpa,
                ).float()
                for row in torch.nonzero(lengths == 0).flatten().tolist():
                    assert torch.equal(states[name].channel[:, row], before[0][:, row])
                    assert torch.equal(states[name].time[:, row], before[1][:, row])
                    assert not outputs[name][row].any()
                assert torch.isfinite(outputs[name]).all()
            reference = states["fp32"]
            for name in errors:
                state = states[name]
                assert torch.equal(state.valid, reference.valid)
                assert (
                    state.channel.dtype == torch.float32
                    and state.time.dtype == torch.float32
                )
                errors[name] += [
                    (outputs[name] - outputs["fp32"]).abs().flatten(),
                    (state.channel - reference.channel).abs().flatten(),
                    (state.time - reference.time).abs().flatten(),
                ]
    assert zero_rows >= chunks
    pooled = {name: torch.cat(parts) for name, parts in errors.items()}
    return {name: (err.max().item(), err.mean().item()) for name, err in pooled.items()}


def _assert_no_regression(errors: dict[str, tuple[float, float]], label: str) -> None:
    (old_max, old_mean), (new_max, new_mean) = errors["eager16"], errors["sdpa16"]
    print(
        f"{label}: err_old max {old_max:.6g} mean {old_mean:.6g}; err_new max "
        f"{new_max:.6g} mean {new_mean:.6g}"
    )
    assert new_max <= _FP16_SDPA_MAX_ERROR_FACTOR * old_max
    assert new_mean <= _FP16_SDPA_MEAN_ERROR_FACTOR * old_mean


@torch.inference_mode()
def test_fp16_sdpa_error_vs_fp32_reference_does_not_regress() -> None:
    """Regression guard (not a qualification) on the tiny model, 60 chunks."""
    errors = _fp32_referenced_errors(
        lambda: _encoder(torch.float32),
        batch=_BATCH,
        left=_LEFT,
        n_layers=_N_LAYERS,
        d_model=_D_MODEL,
        conv_kernel=_KERNEL,
        feat_in=16,
        mel_width=_MEL,
        chunks=_CHUNKS,
    )
    _assert_no_regression(errors, "tiny")


def _real_dimension_encoder() -> FastConformerEncoder:
    torch.manual_seed(41)
    encoder = FastConformerEncoder(att_context=(56, 13))
    with torch.no_grad():
        for layer in encoder.layers:
            layer.self_attn.pos_bias_u.normal_(0.0, 0.1)
            layer.self_attn.pos_bias_v.normal_(0.0, 0.1)
    return encoder.eval()


@torch.inference_mode()
def test_fp16_sdpa_error_vs_fp32_reference_does_not_regress_real_dimensions() -> None:
    """Regression guard (not a qualification) at 1024/8 heads/24 layers/56."""
    errors = _fp32_referenced_errors(
        _real_dimension_encoder,
        batch=4,
        left=56,
        n_layers=24,
        d_model=1024,
        conv_kernel=9,
        feat_in=128,
        mel_width=65,
        chunks=_REAL_DIMENSION_CHUNKS,
    )
    _assert_no_regression(errors, "real-dimension")


@torch.inference_mode()
def test_fp32_sdpa_attention_is_the_same_math() -> None:
    """FP32 SDPA agrees to rounding; it is not bitwise, hence FP16-only."""
    worst, zero_rows = _compare(torch.float32)
    assert zero_rows >= _CHUNKS
    print(f"fp32 sdpa max |delta| vs eager attention: {worst:.6g}")
    assert worst <= 1e-4


@torch.inference_mode()
def test_fp32_default_never_calls_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default (FP32 policy) step stays on the bitwise eager core."""

    def forbidden(*args, **kwargs):
        raise AssertionError("the FP32 encoder lane must not use SDPA")

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", forbidden)
    encoder = _encoder(torch.float32)
    out_width = _out_width(encoder, torch.float32)
    _prepare(encoder, torch.float32, out_width)
    caches = _caches(seed=5)
    for mel, offsets, lengths in _schedule(torch.float32, out_width, seed=11):
        stream_step(
            encoder,
            mel,
            caches,
            out_offsets=offsets,
            out_lengths=lengths,
            out_width=out_width,
        )
        break


def test_encoder_transition_selects_sdpa_only_under_fp16_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from vllm.models.nemotron_asr import encoder_execution

    seen: list[tuple[bool, bool]] = []

    def fake_stream_step(
        *args, preserve_conv_cache_precision, sdpa_attention, **kwargs
    ):
        seen.append((preserve_conv_cache_precision, sdpa_attention))
        raise StopIteration

    monkeypatch.setattr(encoder_execution, "stream_step", fake_stream_step)
    bi_encoder = SimpleNamespace(_batch_invariant_enabled=True)
    for encoder, override in (
        (None, None),
        (None, "fp16"),
        (bi_encoder, None),
        (bi_encoder, "fp16"),
    ):
        core = SimpleNamespace(
            encoder=encoder, policy=SimpleNamespace(encoder_compute_override=override)
        )
        with pytest.raises(StopIteration):
            encoder_execution.execute_encoder_transition(
                core, torch.zeros(1), None, None, None, 1, None
            )
    # Batch-invariant mode does not toggle the FP16 SDPA lane (PORT-PREC-027).
    assert seen == [(False, False), (True, True), (False, False), (True, True)]


@pytest.mark.parametrize("compute", [torch.float32, torch.float16])
@torch.inference_mode()
def test_sdpa_attention_keeps_rows_isolated(compute: torch.dtype) -> None:
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
            sdpa_attention=True,
        )
        expected = stream_step(encoder, mel, base, **kwargs)
        actual = stream_step(encoder, other, perturbed, **kwargs)
        keep = [row for row in range(_BATCH) if row != victim]
        assert torch.equal(actual[keep], expected[keep])
        assert torch.equal(perturbed.channel[:, keep], base.channel[:, keep])
        assert torch.equal(perturbed.time[:, keep], base.time[:, keep])
        assert torch.equal(perturbed.valid, base.valid)
    assert not torch.equal(perturbed.channel[:, victim], base.channel[:, victim])


@pytest.mark.parametrize(
    ("queries", "keys"), [(1, 2), (4, 12), (7, 15), (7, 63), (12, 12)]
)
def test_widened_rel_shift_crops_to_the_exact_shift(queries: int, keys: int) -> None:
    batch, heads = 3, 2
    raw = torch.randn(heads, batch * queries, 2 * keys - 1)
    exact = _stream_rel_shift(raw, batch=batch, queries=queries, keys=keys)
    width = _sdpa_bias_width(keys)
    assert keys <= width <= max(keys, 2 * keys - 2)
    wide = _stream_rel_shift(raw, batch=batch, queries=queries, keys=keys, width=width)
    assert torch.equal(wide[..., :keys], exact)


@pytest.mark.parametrize("compute", [torch.float32, torch.float16])
@torch.inference_mode()
def test_sdpa_fully_masked_query_rows_are_exact_zeros(
    compute: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoder = _encoder(compute)
    layer = encoder.layers[0]
    frames = 5
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(_BATCH, frames, _D_MODEL, generator=generator).to(compute)
    cache = torch.randn(_BATCH, _LEFT, _D_MODEL, generator=generator)
    # Row 0 all padded, row 1 partially padded with no history, rows 2-3 full.
    lengths = torch.tensor([0, 2, frames, frames])
    valid = torch.tensor([_LEFT, 0, 3, _LEFT])
    new_valid = torch.arange(frames).unsqueeze(0) < lengths.unsqueeze(1)
    mask = _stream_attention_mask(valid, new_valid, _LEFT)
    sdpa = StreamSDPAMask(mask, dtype=compute)
    assert sdpa.additive.shape[-1] % encoder_module._SDPA_BIAS_ALIGNMENT == 0
    assert torch.equal(sdpa.dead_queries.squeeze(), ~new_valid)
    pos_emb = encoder.pos_enc(torch.zeros(1, frames + _LEFT, _D_MODEL, dtype=compute))
    sdpa_call = torch.nn.functional.scaled_dot_product_attention
    biases: list[torch.Tensor] = []

    def recording_sdpa(*args, attn_mask, **kwargs):
        biases.append(attn_mask)
        return sdpa_call(*args, attn_mask=attn_mask, **kwargs)

    monkeypatch.setattr(
        torch.nn.functional, "scaled_dot_product_attention", recording_sdpa
    )
    out, _ = _stream_attention(
        layer,
        x,
        cache=cache,
        valid=valid,
        pos_emb=pos_emb,
        new_valid=new_valid,
        new_lengths=lengths,
        mask=mask,
        cache_indices=_stream_cache_indices(lengths, _LEFT)
        .unsqueeze(-1)
        .expand(_BATCH, _LEFT, _D_MODEL),
        sdpa=sdpa,
    )
    # The bias reaches SDPA with every non-unit stride on the alignment, so
    # the memory-efficient kernel needs no padding copy.
    (bias,) = biases
    assert bias.dtype == compute and bias.stride(-1) == 1
    assert all(
        stride % encoder_module._SDPA_BIAS_ALIGNMENT == 0
        for stride in bias.stride()[:-1]
    )
    assert torch.isfinite(out).all()
    assert not out[~new_valid].any()
    assert out[new_valid].abs().amax() > 0


@torch.inference_mode()
def test_fp16_sdpa_step_captures_as_one_fullgraph() -> None:
    """Dynamo traces the SDPA step without a break and replays it exactly."""
    compute = torch.float16
    encoder = _encoder(compute)
    out_width = _out_width(encoder, compute)
    _prepare(encoder, compute, out_width)
    eager_caches, graph_caches = _caches(seed=9), _caches(seed=9)

    def step(mel, caches, offsets, lengths):
        return stream_step(
            encoder,
            mel,
            caches,
            out_offsets=offsets,
            out_lengths=lengths,
            out_width=out_width,
            preserve_conv_cache_precision=True,
            sdpa_attention=True,
        )

    torch._dynamo.reset()
    compiled = torch.compile(step, backend="eager", fullgraph=True, dynamic=False)
    for index, (mel, offsets, lengths) in enumerate(
        _schedule(compute, out_width, seed=17)
    ):
        expected = step(mel, eager_caches, offsets, lengths)
        actual = compiled(mel, graph_caches, offsets, lengths)
        assert torch.equal(actual, expected)
        assert torch.equal(graph_caches.channel, eager_caches.channel)
        assert torch.equal(graph_caches.time, eager_caches.time)
        assert torch.equal(graph_caches.valid, eager_caches.valid)
        if index == 3:
            break
    torch._dynamo.reset()
