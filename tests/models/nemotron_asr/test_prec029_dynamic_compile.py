# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PORT-PREC-029 tests-first contracts for the production execution authority.

The named acceptance test uses the real encoder/conditioner at reduced dimensions.
Supporting guard tests use a tiny stateful transition. Dynamo and Inductor are real;
the CPU platform graph adapter is explicitly a test double, not CUDA evidence.
The CUDA case uses the same reduced encoder with real capture/replay. Existing
PREC-020 qualification tests remain responsible for full encoder arithmetic.
"""

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_encoder_execution import _compiled_core, _graph_runtime, _GraphCaches
from torch._dynamo.backends.registry import lookup_backend

from vllm.models.nemotron_asr import encoder_execution as subject
from vllm.models.nemotron_asr.batch_invariance import (
    BatchInvariantExecution,
    bind_batch_invariant_mode,
)
from vllm.models.nemotron_asr.encoder import FastConformerEncoder, StreamingCaches
from vllm.models.nemotron_asr.lid import PromptConditioner
from vllm.models.nemotron_asr.precision import FP32_BRINGUP

pytestmark = [pytest.mark.core_model]
POPULATIONS = (1, 2, 31, 63, 64, 128)
GEOMETRIES = (0, 1)
_REAL_TRANSITION = subject.execute_encoder_transition


@pytest.fixture(autouse=True)
def reset_compiler():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _transition(core, mel, caches, offsets, lengths, out_width, prompt):
    # Exercise independent static geometry, B-derived view extents, conditioning,
    # padding, every layer's history/tail and all committed valid control slots.
    raw = mel[:, :3, :out_width].transpose(1, 2).contiguous()
    raw = raw + offsets[:, None, None].to(raw.dtype)
    for channel, tail in zip(caches.channel, caches.time):
        raw = raw + channel[:, :1, :] + tail[:, :, :1].transpose(1, 2)
        channel.add_(1)
        tail.add_(2)
    valid = torch.arange(out_width, device=mel.device)[None, :] < lengths[:, None]
    raw = torch.where(valid[:, :, None], raw, 0)
    conditioned = torch.where(valid[:, :, None], raw + prompt[:, None, None], 0)
    caches.valid = (caches.valid + lengths).clamp_max(caches.left_context)
    return raw, conditioned


def _args(execution, geometry, population, device="cpu"):
    shape = execution._geometry_shapes[geometry]
    rows = torch.arange(population, device=device)
    feat, width, tail_width = getattr(execution._core, "_prec029_dims", (5, 3, 2))
    channel = tuple(
        torch.full((population, 56, width), float(i + 1), device=device)
        for i in range(2)
    )
    tail = tuple(
        torch.full((population, width, tail_width), float(i + 3), device=device)
        for i in range(2)
    )
    caches = _GraphCaches(
        channel,
        tail,
        tuple((rows % 3).to(torch.int32).reshape(-1, 1) for _ in range(2)),
    )
    if hasattr(execution._core, "_prec029_dims"):
        from vllm.models.nemotron_asr.advance import _GatheredCaches

        storage = caches.graph_storage()
        caches = _GatheredCaches._from_tensors(
            channel=storage.channel,
            time=storage.time,
            valid=storage.valid,
            left_context=56,
        )
    return (
        torch.ones(population, feat, shape.mel_width, device=device),
        caches,
        rows % 2,
        rows % (shape.out_width + 1),
        shape.out_width,
        rows % 4,
    )


def _snapshot(outputs, args):
    storage = args[1].graph_storage()
    return tuple(
        t.detach().clone()
        for t in (*outputs, *storage.channel, *storage.time, *storage.valid)
    )


def _bitwise(actual, expected):
    assert len(actual) == len(expected)
    for index, (left, right) in enumerate(zip(actual, expected)):
        assert left.shape == right.shape and left.dtype == right.dtype, index
        assert torch.equal(
            left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)
        ), index


class _Frames:
    """Count actual backend entry, preserving any PORT backend/counter wrapper."""

    def __init__(self, monkeypatch):
        self.events: list[tuple[int, str, str]] = []
        self.phase = "profile"
        self.current: tuple[int, str] = (0, "unattributed")
        original = torch.compile

        def compile_counted(fn, **kwargs):
            assert kwargs.get("fullgraph") is True
            assert kwargs.get("dynamic") is False
            options = kwargs.pop("options", {})
            assert options.get("triton.cudagraphs") is False
            backend = lookup_backend(kwargs.pop("backend", "inductor"))

            def counted(gm, example_inputs, **backend_kwargs):
                with torch._inductor.config.patch(options):
                    result = backend(gm, example_inputs, **backend_kwargs)
                self.events.append((*self.current, self.phase))
                return result

            compiled = original(fn, backend=counted, **kwargs)

            def invoke(*args):
                self.current = (
                    args[4],
                    "singleton" if args[0].shape[0] == 1 else "dynamic",
                )
                return compiled(*args)

            return invoke

        monkeypatch.setattr(torch, "compile", compile_counted)

    def counts(self):
        return Counter((width, kind) for width, kind, _phase in self.events)


def _build(
    monkeypatch,
    *,
    enabled=True,
    cap=128,
    arm="compiled-static",
    bucketed=False,
    device="cpu",
    real=False,
):
    monkeypatch.setattr(
        subject, "execute_encoder_transition", _REAL_TRANSITION if real else _transition
    )
    if real:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(29029)
            core = torch.nn.Module()
            core.encoder = FastConformerEncoder(
                feat_in=16,
                d_model=16,
                d_ff=32,
                n_layers=2,
                n_heads=2,
                conv_kernel=5,
                subsampling_channels=4,
                att_context=(56, 1),
            )
            core.lid = PromptConditioner(enc_hidden=16, num_prompts=4)
            core.policy = FP32_BRINGUP
            core._prec029_dims = (16, 16, 4)
            core.eval().to(device)
        bind_batch_invariant_mode(core, BatchInvariantExecution(enabled))
    else:
        core = _compiled_core().to(device)
        core.batch_invariant_mode = BatchInvariantExecution(enabled)
    config = SimpleNamespace(
        encoder_execution_arm=arm,
        att_context_left=56,
        encoder_population_bucketing=bucketed,
    )
    vllm_config = SimpleNamespace()
    runtime = _graph_runtime()
    if device == "cuda":
        from test_batch_invariant_cuda import _capture_stream

        from vllm.config import VllmConfig

        vllm_config = VllmConfig()
        runtime = replace(
            subject.platform_graph_runtime(), capture_context=_capture_stream
        )
    return subject.build_encoder_execution(
        core,
        config,
        maximum_population=cap,
        warmup_geometries=GEOMETRIES,
        vllm_config=vllm_config,
        graph_runtime=runtime,
    )


def _exercise(monkeypatch, *, order, arm, bucketed, cap, device="cpu"):
    frames = _Frames(monkeypatch)
    execution = _build(
        monkeypatch, cap=cap, arm=arm, bucketed=bucketed, device=device, real=True
    )
    populations = execution.warmup_populations
    if bucketed:
        assert populations == tuple(dict.fromkeys((1, 2, 4, 8, 16, 32, 64, cap)))
    ordered = (
        populations if order == "singleton-first" else tuple(reversed(populations))
    )
    cells = tuple((g, p) for g in GEOMETRIES for p in ordered)

    def invoke(g, p):
        output = execution.transition(*_args(execution, g, p, device))
        assert all(count == 1 for count in frames.counts().values()), (
            f"extra whole-transition frame: {frames.events}"
        )
        return output

    with torch.inference_mode():
        execution.profile_cell(
            geometry=GEOMETRIES[-1],
            population=ordered[0],
            invoke=lambda: invoke(GEOMETRIES[-1], ordered[0]),
        )
        frames.phase = "warmup"
        original_capture = execution._capture_graph_domain

        def capture(**kwargs):
            frames.phase = "capture"
            return original_capture(**kwargs)

        monkeypatch.setattr(execution, "_capture_graph_domain", capture)
        execution.warmup_domain(expected_cells=cells, invoke=invoke)
        if arm == "dense-graphed":
            assert len(execution._graph_entries) == len(cells)
            assert set(execution._graph_entries) == {
                execution.graph_key("cuda", cell) for cell in cells
            }
        frames.phase = "serving"
        results = {}
        for g in GEOMETRIES:
            for p in POPULATIONS:
                if p <= cap:
                    args = _args(execution, g, p, device)
                    results[g, p] = _snapshot(execution.transition(*args), args)
        assert execution.ready
    expected = Counter(
        {
            (execution._geometry_shapes[g].out_width, kind): 1
            for g in GEOMETRIES
            for kind in ("singleton", "dynamic")
        }
    )
    return execution, frames, expected, results


def _static_oracle(monkeypatch, results, cap, device="cpu"):
    # Same mode-on arithmetic, isolated reset cache, no annotations on fresh
    # inputs, and its own Cartesian budget. Never borrow the dynamic callable.
    torch._dynamo.reset()
    with monkeypatch.context() as oracle_patch, torch.inference_mode():
        execution = _build(oracle_patch, cap=cap, device=device, real=True)
        execution._materialize_preprofile_state(
            execution._geometry_shapes[GEOMETRIES[-1]].out_width
        )
        oracle = torch.compile(
            lambda mel, caches, offsets, lengths, width, prompt: _REAL_TRANSITION(
                execution._core, mel, caches, offsets, lengths, width, prompt
            ),
            fullgraph=True,
            dynamic=False,
            # PORT-PREC-029: the oracle is the same mode-on arm compiled
            # statically, so it uses the production mode-on compiler options.
            options={
                "triton.cudagraphs": False,
                "selective_decompose": True,
                "fallback_by_default": True,
            },
        )
        budget = len(GEOMETRIES) * cap
        with torch._dynamo.config.patch(
            cache_size_limit=budget, accumulated_cache_size_limit=budget + 256
        ):
            for (g, p), expected in results.items():
                args = _args(execution, g, p, device)
                _bitwise(_snapshot(oracle(*args), args), expected)


@pytest.mark.cpu_test
@pytest.mark.parametrize("order", ["singleton-first", "bulk-first"])
@pytest.mark.parametrize(
    "arm,bucketed,cap", [("compiled-static", False, 64), ("dense-graphed", True, 128)]
)
def test_prec029_mode_on_dynamic_batch_compile(monkeypatch, order, arm, bucketed, cap):
    """@spec PORT-PREC-029, PORT-PERF-009, PORT-PERF-011: two real frames and exact
    state."""
    with monkeypatch.context() as run_patch:
        execution, frames, expected, results = _exercise(
            run_patch, order=order, arm=arm, bucketed=bucketed, cap=cap
        )
        observed = frames.counts()
    _static_oracle(monkeypatch, results, cap)
    assert observed == expected, (
        f"whole-transition frames: {observed}; events={frames.events}"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA capture/replay"
)
@pytest.mark.parametrize("order", ["singleton-first", "bulk-first"])
def test_prec029_dynamic_batch_cuda_capture(
    monkeypatch, order, isolated_batch_invariance
):
    """@spec PORT-PREC-029, PORT-PERF-011:
    real CUDA tier captures use the two frames.
    """
    from vllm.model_executor.determinism import batch_invariant
    from vllm.models.nemotron_asr.batch_invariance import installed_batch_invariant_mode

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    # The worker initializer owns installation; the model only reads it.
    assert installed_batch_invariant_mode(
        module=batch_invariant, initialize=batch_invariant.init_batch_invariance
    ).enabled
    with monkeypatch.context() as run_patch:
        _, frames, expected, results = _exercise(
            run_patch,
            order=order,
            arm="dense-graphed",
            bucketed=True,
            cap=128,
            device="cuda",
        )
        observed = frames.counts()
    _static_oracle(monkeypatch, results, 128, "cuda")
    assert observed == expected


@pytest.mark.cpu_test
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("layout", ["gathered", "stacked"])
@pytest.mark.parametrize("population", [1, 2])
def test_prec029_named_population_axes_before_profile(
    monkeypatch, enabled, layout, population
):
    """@spec PORT-PREC-029: mark only population axes before the first compiler
    invocation."""
    calls = []
    # B>=2 invocations mark with bounds; B=1 is never marked (size-1 frame).
    monkeypatch.setattr(
        torch._dynamo,
        "mark_dynamic",
        lambda tensor, dim, **kw: calls.append((id(tensor), dim, kw)),
    )
    monkeypatch.setattr(
        torch._dynamo,
        "maybe_mark_dynamic",
        lambda *_a, **_k: pytest.fail("unbounded mark forbidden"),
    )
    expected: set[tuple[int, int]] = set()

    def fake_compile(fn, **kwargs):
        assert kwargs["fullgraph"] is True and kwargs["dynamic"] is False
        assert kwargs["options"]["triton.cudagraphs"] is False

        def invoke(*args):
            assert {(ident, dim) for ident, dim, _kw in calls} == (
                expected if enabled and population >= 2 else set()
            )
            assert all(kw == {"min": 2, "max": 64} for _ident, _dim, kw in calls)
            return args[0], args[0]

        return invoke

    monkeypatch.setattr(torch, "compile", fake_compile)
    execution = _build(monkeypatch, enabled=enabled, cap=64)
    with torch.inference_mode():
        args = list(_args(execution, 1, population))
        if layout == "stacked":
            args[1] = StreamingCaches(
                n_layers=2,
                batch=population,
                d_model=3,
                left_context=56,
                conv_kernel=3,
                device=torch.device("cpu"),
            )
        caches = args[1]
        expected.update((id(args[i]), 0) for i in (0, 2, 3, 5))
        if layout == "stacked":
            expected.update(
                ((id(caches.channel), 1), (id(caches.time), 1), (id(caches.valid), 0))
            )
        else:
            expected.update(
                (id(t), 0) for family in caches.graph_storage() for t in family
            )
        execution.profile_cell(
            geometry=1,
            population=population,
            invoke=lambda args=args: execution.transition(*args),
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_prec029_scoped_region_budget(monkeypatch, enabled, fail):
    """@spec PORT-PREC-029, PORT-PERF-009:
    exact mode-on cap; restoration on both exits.
    """
    seen = []

    def compile_probe(fn, **kwargs):
        def invoke(*args):
            seen.append(
                (
                    torch._dynamo.config.cache_size_limit,
                    torch._dynamo.config.accumulated_cache_size_limit,
                )
            )
            if fail:
                raise RuntimeError("injected compiler failure")
            return args[0], args[0]

        return invoke

    monkeypatch.setattr(torch, "compile", compile_probe)
    execution = _build(monkeypatch, enabled=enabled, cap=8)
    with (
        torch._dynamo.config.patch(
            cache_size_limit=100, accumulated_cache_size_limit=200
        ),
        torch.inference_mode(),
    ):
        args = _args(execution, 1, 2)

        def invoke():
            return execution.profile_cell(
                geometry=1,
                population=2,
                invoke=lambda args=args: execution.transition(*args),
            )

        if fail:
            with pytest.raises(RuntimeError, match="injected"):
                invoke()
            assert not execution.ready and execution._failed
        else:
            invoke()
        assert torch._dynamo.config.cache_size_limit == 100
        assert torch._dynamo.config.accumulated_cache_size_limit == 200
    assert seen[0][0] == (4 if enabled else 100)
    assert seen[0][1] >= 200


@pytest.mark.cpu_test
def test_prec029_extra_variant_rejected_during_warmup(monkeypatch):
    """@spec PORT-PREC-029: reject a second dynamic frame while other classes are
    unused."""
    frames = _Frames(monkeypatch)
    execution = _build(monkeypatch, cap=8)
    with torch.inference_mode():

        def warm(g, p, transposed=False):
            args = list(_args(execution, g, p))
            if transposed:
                args[0] = args[0].transpose(1, 2).contiguous().transpose(1, 2)
            execution.warmup_cell(
                geometry=g,
                population=p,
                invoke=lambda args=args: execution.transition(*args),
            )

        warm(0, 2)
        # Distinct stride guard requests an extra variant before geometry 1 or
        # singleton has consumed any region capacity. Aggregate cap cannot help.
        with pytest.raises(
            (ValueError, RuntimeError),
            match="(?i)(variant|specialization|frame|recompil)",
        ):
            warm(0, 3, transposed=True)
        assert not execution.ready and execution._failed
        assert (
            frames.counts()[(execution._geometry_shapes[0].out_width, "dynamic")] == 1
        )


@pytest.mark.cpu_test
def test_prec029_sealed_unwarmed_population_rejected(monkeypatch):
    """@spec PORT-PREC-029, PORT-PERF-009: tensor admission remains stricter than
    dynamic range."""
    frames = _Frames(monkeypatch)
    execution = _build(monkeypatch, cap=4)
    with torch.inference_mode():
        for g in GEOMETRIES:
            for p in execution.warmup_populations:
                args = _args(execution, g, p)
                execution.warmup_cell(
                    geometry=g,
                    population=p,
                    invoke=lambda args=args: execution.transition(*args),
                )
        execution._seal()
        omitted = execution._warmup_signatures[(0, 3)]
        execution._allowed_signatures = execution._allowed_signatures - {omitted}
        before = list(frames.events)
        with pytest.raises(ValueError, match="signature was not warmed"):
            execution.transition(*_args(execution, 0, 3))
        with pytest.raises(ValueError, match="population 0"):
            execution.transition(*_args(execution, 0, 0))
        assert frames.events == before


@pytest.mark.cpu_test
def test_prec029_accumulated_budget_preserves_existing_frames(monkeypatch):
    """@spec PORT-PREC-029:
    another region's existing entries leave room for the pair.
    """
    with monkeypatch.context() as seed_patch, torch.inference_mode():
        frames = _Frames(seed_patch)
        seed = _build(seed_patch, enabled=False, cap=3)
        for g in GEOMETRIES:
            for p in (1, 2, 3):
                args = _args(seed, g, p)
                seed.warmup_cell(
                    geometry=g,
                    population=p,
                    invoke=lambda args=args: seed.transition(*args),
                )
        entries_taken = len(frames.events)
        assert (
            entries_taken == 6
        )  # actual Inductor frames; no reset before the new region

    def compile_probe(fn, **kwargs):
        def invoke(*args):
            assert (
                torch._dynamo.config.accumulated_cache_size_limit
                >= entries_taken + 2 * len(GEOMETRIES)
            )
            return args[0], args[0]

        return invoke

    monkeypatch.setattr(torch, "compile", compile_probe)
    execution = _build(monkeypatch, cap=2)
    with (
        torch._dynamo.config.patch(accumulated_cache_size_limit=entries_taken),
        torch.inference_mode(),
    ):
        try:
            args = _args(execution, 1, 2)
            execution.profile_cell(
                geometry=1,
                population=2,
                invoke=lambda args=args: execution.transition(*args),
            )
        finally:
            assert torch._dynamo.config.accumulated_cache_size_limit == entries_taken


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "kind,key", [("cuda", (1, 31)), ("chunk", (1, 31)), ("decode_fn", (1, 31))]
)
def test_prec029_mode_off_keys_byte_identical(kind, key):
    """@spec PORT-PREC-029, PORT-PREC-019:
    no mode marker or tuple extension when off.
    """
    import json

    mode = BatchInvariantExecution(False)
    observed = mode.graph_key(kind, key)
    assert type(observed) is tuple
    assert json.dumps(observed, separators=(",", ":")).encode() == b"[1,31]"


@pytest.mark.cpu_test
@pytest.mark.parametrize("bucketed", [False, True])
def test_prec029_mode_off_cartesian_budget(monkeypatch, bucketed):
    """@spec PORT-PREC-029, PORT-PERF-009: mode-off budget follows the selected
    population set."""
    seen = []

    def compile_probe(fn, **kwargs):
        def invoke(*args):
            seen.append(torch._dynamo.config.cache_size_limit)
            return args[0], args[0]

        return invoke

    monkeypatch.setattr(torch, "compile", compile_probe)
    execution = _build(
        monkeypatch,
        enabled=False,
        cap=5,
        arm="dense-graphed" if bucketed else "compiled-static",
        bucketed=bucketed,
    )
    assert execution.warmup_populations == (
        (1, 2, 4, 5) if bucketed else (1, 2, 3, 4, 5)
    )
    with torch._dynamo.config.patch(cache_size_limit=1), torch.inference_mode():
        args = _args(execution, 1, 5)
        execution.profile_cell(
            geometry=1,
            population=5,
            invoke=lambda args=args: execution.transition(*args),
        )
        assert torch._dynamo.config.cache_size_limit == 1
    assert seen == [len(GEOMETRIES) * len(execution.warmup_populations)]


@pytest.mark.cpu_test
@pytest.mark.parametrize("enabled", [False, True])
def test_prec029_bucketed_graph_replays_partial_mel_widths(monkeypatch, enabled):
    """@spec PORT-PERF-010, PORT-PERF-011: valid mel widths are staged graph inputs.

    The real reduced encoder runs the length-masked subsampling through the
    compiled, population-bucketed graph arm; an underfilled live prefix with
    partial widths must match the uncompiled transition, and the widths must
    change the result (they are replayed values, not captured constants).
    """
    _assert_bucketed_partial_mel_widths(monkeypatch, enabled=enabled, device="cpu")


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA capture/replay"
)
def test_prec029_bucketed_graph_replays_partial_mel_widths_cuda(monkeypatch):
    """@spec PORT-PERF-010, PORT-PERF-011: the same contract under real CUDA capture."""
    _assert_bucketed_partial_mel_widths(monkeypatch, enabled=False, device="cuda")


def _assert_bucketed_partial_mel_widths(monkeypatch, *, enabled, device):
    execution = _build(
        monkeypatch,
        enabled=enabled,
        cap=4,
        arm="dense-graphed",
        bucketed=True,
        device=device,
        real=True,
    )
    populations = execution.warmup_populations
    assert populations == (1, 2, 4)

    def args(geometry, population, widths):
        mel_lengths = torch.tensor(widths, dtype=torch.int64, device=device)
        mel, caches, _offsets, _lengths, out_width, prompt = _args(
            execution, geometry, population, device
        )
        # Session-first rows (no pre-encode drop) emitting every valid frame,
        # so each row's last frame reads its subsampling's padded tail.
        offsets = torch.zeros(population, dtype=torch.int64, device=device)
        lengths = execution._core.encoder.pre_encode.output_lengths(mel_lengths).clamp(
            max=out_width
        )
        return mel, caches, offsets, lengths, out_width, prompt, mel_lengths

    def full(geometry, population):
        return [execution._geometry_shapes[geometry].mel_width] * population

    with torch.inference_mode():
        execution.warmup_domain(
            expected_cells=tuple((g, p) for g in GEOMETRIES for p in populations),
            invoke=lambda g, p: execution.transition(*args(g, p, full(g, p))),
        )
        assert execution.ready
        geometry = GEOMETRIES[-1]
        width = execution._geometry_shapes[geometry].mel_width
        partial = [width - 3, width, width - 6]
        replayed = execution.transition(*args(geometry, 3, partial))
        eager_args = args(geometry, 3, partial)
        expected = _REAL_TRANSITION(execution._core, *eager_args)
        for actual, oracle in zip(replayed, expected, strict=True):
            # CUDA convolutions may use TF32 in either execution; the
            # mask's effect below is far larger than that tolerance.
            torch.testing.assert_close(
                actual, oracle, rtol=0, atol=1e-5 if device == "cpu" else 1e-3
            )
        # The same rows and emitted frames with the mask disabled (full
        # widths) differ only where a width is partial.
        unmasked_args = list(args(geometry, 3, partial))
        unmasked_args[6] = torch.tensor(
            full(geometry, 3), dtype=torch.int64, device=device
        )
        unmasked = execution.transition(*unmasked_args)
        assert not torch.equal(replayed[0][0], unmasked[0][0])
        assert not torch.equal(replayed[0][2], unmasked[0][2])
        assert torch.equal(replayed[0][1], unmasked[0][1])
