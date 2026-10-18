# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-CUDA batch-invariance falsifier for the encoder-only FP16 policy.

Question: is one session's encoder transition bitwise independent of the other
rows launched with it? One fixed probe session runs a multi-chunk 160 ms
sequence through the real-dimension encoder (d_model 1024, 24 layers, 8 heads,
FF 4096, conv kernel 9, 56-frame history; seeded random weights) at populations
1, 2, 31, 63, 64 and 128. Every chunk's probe-row ``encoded`` and
``conditioned`` frames, and every layer's channel/time cache row and valid
count, must ``torch.equal`` the population-1 run of the same execution path.

* ``mixed``: the other rows carry different audio, prompts, cache history and
  session phases (fresh, steady and idle rows); the probe is the last row.
* ``repeated``: the probe is repeated in every row; every row must equal B=1.
* ``eager``: the uncompiled transition. ``graph``: the same uncompiled
  transition captured once per population with ``torch.cuda.CUDAGraph`` and
  replayed, i.e. what the ``eager-graphed`` arm captures (it does not prepare
  the dense arm's relative-position projections). Graph B=1 must also equal
  eager B=1.
* ``run-to-run``: B=128 ``mixed`` twice; all rows and the final state must be
  bitwise equal across the two runs.

On an eager divergence the probe is re-run with leaf-module hooks on the first
divergent chunk and the first divergent module is printed (``*.linear_out
[input]`` isolates the attention core: QK/PV matmuls and softmax; the
``pointwise_conv1`` row is unfolded from the ``(1, C, B*frames)`` cuDNN fold).
A graph-only divergence is localized to the first divergent layer cache.

``BI_MODE`` (read by this test/script only; production code is unchanged) is
applied once per process before any model or CUDA work:

* ``none``: process defaults.
* ``flags``: option (b): ``CUBLAS_WORKSPACE_CONFIG=:4096:8``, cuBLASLt, and
  ``allow_{fp16,bf16}_reduced_precision_reduction = (False, False)``.
* ``vllm``: ``VLLM_BATCH_INVARIANT=1`` and vLLM's own worker initializer
  ``vllm.model_executor.determinism.batch_invariant.init_batch_invariance``
  (called by ``init_worker_distributed_environment``), which on SM8x also
  routes ``mm/addmm/matmul/linear`` through Triton.

The cuBLAS workspace is fixed when the first cuBLAS handle is created, so run
each mode in a fresh process (the script's ``--all-modes`` does this).

Operator commands (from the vllm checkout on the pod)::

    T=tests/models/nemotron_asr/test_encoder_batch_invariance.py
    # Full matrix + timing for all three modes, one subprocess per mode;
    # exit 1 if any divergence (expected for BI_MODE=none FP16 if the A100
    # finding reproduces). Optional: --json /workspace/bi.json
    python $T --all-modes --timing
    # One mode only:
    python $T --mode vllm --timing
    # Timing only (median CUDA-event replay time, B=20 and B=64):
    python $T --all-modes --timing-only --iters 200
    # Pytest form (strict assertions; one mode per process):
    BI_MODE=flags python -m pytest $T -m cuda -s -q

Expected runtime on A100/A40 (estimate, not yet measured): about 1-2 min per
mode for the matrix (weights are initialized on device; captures are 6 per
policy and composition), plus up to about 1 min of Triton JIT the first time
``vllm`` mode runs; ``--all-modes --timing`` about 5-8 min. Peak memory is
under 10 GiB.

@spec PORT-PREC-016
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from torch import nn

from vllm.models.nemotron_asr.advance import PRE_ENCODE_DROP, _GatheredCaches
from vllm.models.nemotron_asr.batch_invariance import (
    MODE_OFF,
    BatchInvariantExecution,
    bind_batch_invariant_mode,
    installed_batch_invariant_mode,
)
from vllm.models.nemotron_asr.encoder import FastConformerEncoder
from vllm.models.nemotron_asr.encoder_execution import (
    encoder_geometry_shape,
    execute_encoder_transition,
)
from vllm.models.nemotron_asr.lid import PromptConditioner
from vllm.models.nemotron_asr.nemotron_asr import apply_policy_dtypes
from vllm.models.nemotron_asr.precision import (
    FP16_ENCODER_EXPERIMENT,
    PrecisionPolicy,
    reduced_precision_state,
)

if TYPE_CHECKING:
    from vllm.models.nemotron_asr.advance import SessionStateBatch
    from vllm.models.nemotron_asr.nemotron_asr import NemotronASRCore

pytestmark = [pytest.mark.core_model]

BI_MODES = ("none", "flags", "vllm")
POLICIES = ("fp16-encoder", "fp32")
EXECUTIONS = ("eager", "graph")
COMPOSITIONS = ("mixed", "repeated")
POPULATIONS = (1, 2, 31, 63, 64, 128)
TIMING_POPULATIONS = (20, 64)
DEFAULT_CHUNKS = 8
GEOMETRY = 1  # 160 ms: 2 encoder frames per steady chunk.
HISTORY = 56
PROBE_PROMPT = 3
SEED = 20261006
RESULT_MARKER = "BI_RESULT_JSON "

REAL_DIMENSIONS: dict[str, int] = {
    "feat_in": 128,
    "d_model": 1024,
    "d_ff": 4096,
    "n_layers": 24,
    "n_heads": 8,
    "conv_kernel": 9,
    "subsampling_channels": 256,
    "num_prompts": 128,
}

ChunkHook = Callable[[int, torch.Tensor, torch.Tensor, SimpleNamespace], None]

# --------------------------------------------------------------------------
# BI_MODE (test-only, process-wide)
# --------------------------------------------------------------------------

_APPLIED_MODE: str | None = None
_EXECUTION_MODE = MODE_OFF
_CUDA_INITIALIZED_BEFORE_MODE: bool | None = None
_REDUCED_PRECISION_NOTE = ""


def apply_bi_mode(mode: str) -> dict[str, Any]:
    """Apply one batch-invariance mode once per process, before capture."""
    global \
        _APPLIED_MODE, \
        _CUDA_INITIALIZED_BEFORE_MODE, \
        _REDUCED_PRECISION_NOTE, \
        _EXECUTION_MODE
    if mode not in BI_MODES:
        raise ValueError(f"BI_MODE must be one of {BI_MODES}, got {mode!r}")
    if _APPLIED_MODE is not None:
        if mode != _APPLIED_MODE:
            raise RuntimeError(
                f"BI_MODE {_APPLIED_MODE!r} is already applied; use a fresh process "
                f"for {mode!r}"
            )
        return bi_settings()
    _CUDA_INITIALIZED_BEFORE_MODE = torch.cuda.is_initialized()
    if mode == "flags":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cuda.preferred_blas_library(backend="cublaslt")
        try:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
                False,
                False,
            )  # type: ignore[assignment]
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                False,
                False,
            )  # type: ignore[assignment]
        except (TypeError, ValueError, RuntimeError):
            # Older torch accepts only a bool, which leaves split-K enabled.
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            _REDUCED_PRECISION_NOTE = (
                "torch rejected (False, False); bool False leaves split-K on"
            )
    elif mode == "vllm":
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        from vllm.model_executor.determinism import batch_invariant

        _EXECUTION_MODE = installed_batch_invariant_mode(
            module=batch_invariant, initialize=batch_invariant.init_batch_invariance
        )
        if not _EXECUTION_MODE.enabled:
            raise RuntimeError("vLLM batch-invariant mode did not enable")
    _APPLIED_MODE = mode
    return bi_settings()


def bi_settings() -> dict[str, Any]:
    state = reduced_precision_state()
    settings: dict[str, Any] = {
        "mode": _APPLIED_MODE,
        "torch": torch.__version__,
        "cuda_initialized_before_mode": _CUDA_INITIALIZED_BEFORE_MODE,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "VLLM_BATCH_INVARIANT": os.environ.get("VLLM_BATCH_INVARIANT"),
        # vLLM sets the per-operation precision API; the helper never reads
        # the legacy aggregate allow_tf32 flags unguarded (they raise on
        # mixed backend settings).
        "allow_fp16_reduced_precision_reduction": repr(
            state["allow_fp16_reduced_precision_reduction"]
        ),
        "matmul_fp32_precision": str(state["matmul_fp32_precision"]),
        "cudnn_conv_fp32_precision": str(state["cudnn_conv_fp32_precision"]),
        "cudnn_rnn_fp32_precision": str(state["cudnn_rnn_fp32_precision"]),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    if _REDUCED_PRECISION_NOTE:
        settings["note"] = _REDUCED_PRECISION_NOTE
    if torch.cuda.is_available():
        settings["preferred_blas_library"] = str(
            torch.backends.cuda.preferred_blas_library()
        )
        index = torch.accelerator.current_device_index()
        settings["device"] = torch.cuda.get_device_name(index)
        settings["capability"] = ".".join(
            map(str, torch.cuda.get_device_capability(index))
        )
    return settings


# --------------------------------------------------------------------------
# Model, schedule and session inputs
# --------------------------------------------------------------------------


def build_core(
    policy_name: str,
    device: torch.device,
    dims: dict[str, int] | None = None,
    *,
    mode: BatchInvariantExecution | None = None,
) -> nn.Module:
    """Encoder + conditioner with seeded random weights at realistic scale."""
    dims = dict(REAL_DIMENSIONS if dims is None else dims)
    torch.manual_seed(SEED)
    core = nn.Module()
    with torch.device(device):
        core.encoder = FastConformerEncoder(
            feat_in=dims["feat_in"],
            d_model=dims["d_model"],
            d_ff=dims["d_ff"],
            n_layers=dims["n_layers"],
            n_heads=dims["n_heads"],
            conv_kernel=dims["conv_kernel"],
            subsampling_channels=dims["subsampling_channels"],
            att_context=(HISTORY, GEOMETRY),
        )
        core.lid = PromptConditioner(
            enc_hidden=dims["d_model"], num_prompts=dims["num_prompts"]
        )
    with torch.no_grad():
        # Default Linear/Conv init is already fan-in scaled; move the zero/one
        # initialized biases and norms off their degenerate values.
        for name, parameter in core.encoder.named_parameters():
            if name.endswith(("pos_bias_u", "pos_bias_v")):
                parameter.normal_(0.0, 0.1)
            elif ".norm" in name or "batch_norm" in name:
                if name.endswith("weight"):
                    parameter.normal_(1.0, 0.1)
                else:
                    parameter.normal_(0.0, 0.05)
    if policy_name == "fp32":
        core.policy = PrecisionPolicy({"*": "fp32"})
    elif policy_name == "fp16-encoder":
        core.policy = FP16_ENCODER_EXPERIMENT
        # Decoder modules are absent; tensorless stand-ins satisfy realization.
        core.predictor = nn.Module()
        core.joint = nn.Module()
        apply_policy_dtypes(cast("NemotronASRCore", core))
    else:
        raise ValueError(f"unknown policy {policy_name!r}")
    core.bi_dims = dims
    bind_batch_invariant_mode(core, _EXECUTION_MODE if mode is None else mode)
    return core.eval()


@dataclass(frozen=True)
class Schedule:
    feat_in: int
    d_model: int
    n_layers: int
    conv_tail: int
    num_prompts: int
    mel_width: int
    out_width: int
    first_mel_len: int
    first_len: int
    steady_len: int
    drop: int


def schedule_for(core: nn.Module) -> Schedule:
    dims = cast(dict[str, int], core.bi_dims)
    encoder = cast(FastConformerEncoder, core.encoder)
    shape = encoder_geometry_shape(cast("NemotronASRCore", core), GEOMETRY)
    out_len = encoder.pre_encode.output_lengths
    first_len = int(out_len(torch.tensor([shape.cadence_frames]))[0])
    steady_len = int(out_len(torch.tensor([shape.mel_width]))[0]) - PRE_ENCODE_DROP
    return Schedule(
        feat_in=dims["feat_in"],
        d_model=dims["d_model"],
        n_layers=dims["n_layers"],
        conv_tail=dims["conv_kernel"] - 1,
        num_prompts=dims["num_prompts"],
        mel_width=shape.mel_width,
        out_width=shape.out_width,
        first_mel_len=shape.cadence_frames,
        first_len=min(first_len, shape.out_width),
        steady_len=steady_len,
        drop=PRE_ENCODE_DROP,
    )


def new_state(
    sched: Schedule,
    *,
    population: int,
    composition: str,
    probe_row: int,
    device: torch.device,
) -> SimpleNamespace:
    """Production-layout gathered state: per-layer channel/time/window_valid."""
    generator = torch.Generator(device=device).manual_seed(SEED + 7 * population + 1)
    channel, time, valid = [], [], []
    others = composition == "mixed" and population > 1
    history = (
        torch.randint(
            0,
            HISTORY + 1,
            (population, 1),
            generator=generator,
            device=device,
            dtype=torch.int32,
        )
        if others
        else torch.zeros(population, 1, dtype=torch.int32, device=device)
    )
    history[probe_row] = 0
    for _ in range(sched.n_layers):
        c = torch.zeros(population, HISTORY, sched.d_model, device=device)
        t = torch.zeros(population, sched.d_model, sched.conv_tail, device=device)
        if others:
            c.normal_(generator=generator)
            t.normal_(generator=generator)
            c[probe_row] = 0
            t[probe_row] = 0
        channel.append(c)
        time.append(t)
        valid.append(history.clone())
    return SimpleNamespace(channel=channel, time=time, window_valid=valid)


def gathered(state: SimpleNamespace) -> _GatheredCaches:
    return _GatheredCaches(cast("SessionStateBatch", state))


def chunk_inputs(
    sched: Schedule,
    *,
    population: int,
    chunk: int,
    composition: str,
    probe_row: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(mel, out_offsets, out_lengths, prompt_index)`` for one CHUNK."""
    probe_generator = torch.Generator().manual_seed(SEED + 1000 + chunk)
    probe_mel = torch.randn(sched.feat_in, sched.mel_width, generator=probe_generator)
    probe_first = chunk == 0
    mel = probe_mel.expand(population, -1, -1).clone()
    # Row phase: 0 fresh (first chunk), 1 steady, 2 idle (zero-length).
    phase = torch.full((population,), 0 if probe_first else 1, dtype=torch.long)
    prompt = torch.full((population,), PROBE_PROMPT, dtype=torch.long)
    if composition == "mixed" and population > 1:
        other_generator = torch.Generator().manual_seed(
            SEED + 100_003 * population + chunk
        )
        others = torch.ones(population, dtype=torch.bool)
        others[probe_row] = False
        count = int(others.sum())
        mel[others] = torch.randn(
            count, sched.feat_in, sched.mel_width, generator=other_generator
        )
        phase[others] = torch.multinomial(
            torch.tensor([0.15, 0.75, 0.10]),
            count,
            replacement=True,
            generator=other_generator,
        )
        prompt[others] = torch.randint(
            0, sched.num_prompts, (count,), generator=other_generator
        )
    fresh = phase == 0
    column = torch.arange(sched.mel_width).view(1, 1, -1)
    mel = torch.where(fresh.view(-1, 1, 1) & (column >= sched.first_mel_len), 0.0, mel)
    offsets = torch.where(fresh, 0, sched.drop)
    lengths = torch.where(
        fresh, sched.first_len, torch.where(phase == 1, sched.steady_len, 0)
    )
    return mel.to(device), offsets.to(device), lengths.to(device), prompt.to(device)


# --------------------------------------------------------------------------
# Runners: eager transition and one CUDA graph per population
# --------------------------------------------------------------------------


def _state_tensors(state: SimpleNamespace) -> list[torch.Tensor]:
    return [*state.channel, *state.time, *state.window_valid]


class EagerRunner:
    def __init__(self, core: nn.Module, state: SimpleNamespace, out_width: int) -> None:
        self.core, self.state, self.out_width = core, state, out_width

    def step(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mel, offsets, lengths, prompt = inputs
        return execute_encoder_transition(
            cast("NemotronASRCore", self.core),
            mel,
            gathered(self.state),
            offsets,
            lengths,
            self.out_width,
            prompt,
        )


class GraphRunner:
    """Capture the uncompiled transition once; replay over static buffers.

    The static state tensors are the session state carried across chunks.
    """

    def __init__(
        self,
        core: nn.Module,
        state: SimpleNamespace,
        out_width: int,
        example: Sequence[torch.Tensor],
        pool: Any,
    ) -> None:
        self.state = state
        self.static = [tensor.clone() for tensor in example]
        eager = EagerRunner(core, state, out_width)
        snapshot = [tensor.clone() for tensor in _state_tensors(state)]

        def restore() -> None:
            for destination, source in zip(
                _state_tensors(state), snapshot, strict=True
            ):
                destination.copy_(source)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                eager.step(*self.static)
                restore()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool):
            self.outputs = eager.step(*self.static)
        restore()

    def step(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for destination, source in zip(self.static, inputs, strict=True):
            destination.copy_(source)
        self.graph.replay()
        return self.outputs


def run_sequence(
    core: nn.Module,
    sched: Schedule,
    *,
    population: int,
    composition: str,
    execution: str,
    chunks: int,
    device: torch.device,
    on_chunk: ChunkHook,
    pool: Any = None,
    before_chunk: Callable[[int], None] | None = None,
) -> None:
    probe_row = population - 1
    state = new_state(
        sched,
        population=population,
        composition=composition,
        probe_row=probe_row,
        device=device,
    )

    def inputs(chunk: int) -> tuple[torch.Tensor, ...]:
        return chunk_inputs(
            sched,
            population=population,
            chunk=chunk,
            composition=composition,
            probe_row=probe_row,
            device=device,
        )

    runner: EagerRunner | GraphRunner
    if execution == "eager":
        runner = EagerRunner(core, state, sched.out_width)
    elif execution == "graph":
        runner = GraphRunner(core, state, sched.out_width, inputs(0), pool)
    else:
        raise ValueError(f"unknown execution {execution!r}")
    for chunk in range(chunks):
        if before_chunk is not None:
            before_chunk(chunk)
        encoded, conditioned = runner.step(*inputs(chunk))
        on_chunk(chunk, encoded, conditioned, state)


# --------------------------------------------------------------------------
# Probe snapshots and comparison
# --------------------------------------------------------------------------

Snapshot = list[tuple[str, torch.Tensor]]


def row_views(
    encoded: torch.Tensor, conditioned: torch.Tensor, state: SimpleNamespace
) -> list[tuple[str, torch.Tensor]]:
    """All-row tensors in dependency order (layer caches first)."""
    views: list[tuple[str, torch.Tensor]] = []
    for layer, (channel, time) in enumerate(
        zip(state.channel, state.time, strict=True)
    ):
        views.append((f"layer{layer:02d}.channel_cache", channel))
        views.append((f"layer{layer:02d}.time_cache", time))
    views += [
        ("encoded", encoded),
        ("conditioned", conditioned),
        ("valid", state.window_valid[0]),
    ]
    return views


def probe_snapshot(views: list[tuple[str, torch.Tensor]], row: int) -> Snapshot:
    return [(name, tensor[row].clone()) for name, tensor in views]


@dataclass
class Divergence:
    chunk: int
    tensor: str
    max_abs: float
    differing_elements: int
    differing_rows: list[int]
    localized: str = ""


def _diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, int]:
    delta = (actual.double() - expected.double()).abs()
    return float(delta.max()), int((actual != expected).sum())


def compare_rows(
    chunk: int,
    views: list[tuple[str, torch.Tensor]],
    reference: Snapshot,
    rows: Sequence[int],
) -> Divergence | None:
    """First tensor whose selected rows are not bitwise equal to the reference."""
    index = torch.as_tensor(list(rows), device=views[0][1].device)
    for (name, tensor), (ref_name, ref) in zip(views, reference, strict=True):
        assert name == ref_name
        selected = tensor.index_select(0, index)
        expected = ref.unsqueeze(0).expand_as(selected)
        if torch.equal(selected, expected):
            continue
        bad = (selected != expected).flatten(1).any(1).nonzero().flatten().tolist()
        max_abs, count = _diff(selected, expected)
        return Divergence(chunk, name, max_abs, count, [int(rows[i]) for i in bad][:16])
    return None


def record_reference(
    core: nn.Module,
    sched: Schedule,
    *,
    execution: str,
    composition: str,
    chunks: int,
    device: torch.device,
    pool: Any = None,
) -> list[Snapshot]:
    snapshots: list[Snapshot] = []

    def on_chunk(
        _chunk: int,
        encoded: torch.Tensor,
        conditioned: torch.Tensor,
        state: SimpleNamespace,
    ) -> None:
        snapshots.append(probe_snapshot(row_views(encoded, conditioned, state), 0))

    run_sequence(
        core,
        sched,
        population=1,
        composition=composition,
        execution=execution,
        chunks=chunks,
        device=device,
        on_chunk=on_chunk,
        pool=pool,
    )
    return snapshots


def check_population(
    core: nn.Module,
    sched: Schedule,
    reference: list[Snapshot],
    *,
    population: int,
    execution: str,
    composition: str,
    chunks: int,
    device: torch.device,
    pool: Any = None,
) -> Divergence | None:
    rows = list(range(population)) if composition == "repeated" else [population - 1]
    found: list[Divergence] = []

    def on_chunk(
        chunk: int,
        encoded: torch.Tensor,
        conditioned: torch.Tensor,
        state: SimpleNamespace,
    ) -> None:
        if not found:
            divergence = compare_rows(
                chunk, row_views(encoded, conditioned, state), reference[chunk], rows
            )
            if divergence is not None:
                found.append(divergence)

    run_sequence(
        core,
        sched,
        population=population,
        composition=composition,
        execution=execution,
        chunks=chunks,
        device=device,
        on_chunk=on_chunk,
        pool=pool,
    )
    return found[0] if found else None


# --------------------------------------------------------------------------
# Op-level localization (eager, leaf-module hooks on one chunk)
# --------------------------------------------------------------------------


def _probe_slice(
    name: str, tensor: torch.Tensor, population: int, row: int
) -> torch.Tensor | None:
    if name.endswith("pointwise_conv1") and tensor.dim() == 3 and tensor.shape[0] == 1:
        # _stream_conv folds (B, F, d) to (1, d, B*F): unfold the probe's frames.
        channels, folded = tensor.shape[1], tensor.shape[2]
        if folded % population == 0:
            return tensor[0].view(channels, population, folded // population)[:, row]
    if tensor.dim() > 0 and tensor.shape[0] == population:
        return tensor[row]
    return None


@contextmanager
def record_module_rows(
    core: nn.Module,
    population: int,
    row: int,
    sink: list[tuple[str, torch.Tensor]],
) -> Iterator[list[bool]]:
    """Record the probe row of every leaf module output while ``enabled[0]``.

    ``linear_out`` also records its input: the attention core (QK and PV
    matmuls plus softmax) that sits between the hooked projections.
    """
    enabled = [False]
    handles = []

    def record(name: str, value: Any) -> None:
        if enabled[0] and isinstance(value, torch.Tensor):
            piece = _probe_slice(name, value, population, row)
            if piece is not None:
                sink.append((name, piece.detach().clone()))

    for name, module in core.named_modules():
        if (
            not name
            or any(True for _ in module.children())
            or name.endswith(("linear_pos", "pos_enc"))
        ):
            continue
        if name.endswith("linear_out"):
            handles.append(
                module.register_forward_pre_hook(
                    lambda _m, args, name=name: record(
                        f"{name}[input: QK/PV matmul + softmax]", args[0]
                    )
                )
            )
        handles.append(
            module.register_forward_hook(
                lambda _m, _args, out, name=name: record(name, out)
            )
        )
    try:
        yield enabled
    finally:
        for handle in handles:
            handle.remove()


def localize(
    core: nn.Module,
    sched: Schedule,
    *,
    population: int,
    composition: str,
    chunk: int,
    device: torch.device,
) -> str:
    """Name the first leaf module whose probe-row output differs at ``chunk``.

    Inputs to that chunk are equal (the divergence starts there), so the first
    differing module is a batch-variant kernel, not inherited drift.
    """

    def trace(pop: int) -> list[tuple[str, torch.Tensor]]:
        sink: list[tuple[str, torch.Tensor]] = []
        with record_module_rows(core, pop, pop - 1, sink) as enabled:

            def before(k: int) -> None:
                enabled[0] = k == chunk

            run_sequence(
                core,
                sched,
                population=pop,
                composition=composition,
                execution="eager",
                chunks=chunk + 1,
                device=device,
                on_chunk=lambda *_args: None,
                before_chunk=before,
            )
        return sink

    reference, actual = trace(1), trace(population)
    if len(reference) != len(actual):
        return f"module call counts differ ({len(reference)} vs {len(actual)})"
    for call, ((name, expected), (_, got)) in enumerate(
        zip(reference, actual, strict=True)
    ):
        if expected.shape != got.shape:
            return (
                f"{name} (call {call}): probe slice shape "
                f"{tuple(expected.shape)} vs {tuple(got.shape)}"
            )
        if not torch.equal(expected, got):
            max_abs, count = _diff(got, expected)
            return f"{name} (call {call}): max|diff|={max_abs:.3e}, {count} elements"
    return (
        "no leaf-module divergence on that chunk "
        "(unhooked glue op: gather/cat/masked_fill/residual)"
    )


# --------------------------------------------------------------------------
# Matrix, run-to-run determinism and timing
# --------------------------------------------------------------------------


@dataclass
class CaseResult:
    mode: str
    policy: str
    execution: str
    composition: str
    population: int | str
    equal: bool
    detail: str = ""


def _describe(divergence: Divergence | None) -> str:
    if divergence is None:
        return ""
    text = (
        f"chunk {divergence.chunk} {divergence.tensor}: "
        f"max|diff|={divergence.max_abs:.3e}, "
        f"{divergence.differing_elements} elements, rows {divergence.differing_rows}"
    )
    if divergence.localized:
        text += f"; first divergent op: {divergence.localized}"
    return text


def invariance_cases(
    core: nn.Module,
    *,
    mode: str,
    policy: str,
    execution: str,
    composition: str,
    populations: Sequence[int],
    chunks: int,
    device: torch.device,
    pool: Any = None,
) -> list[CaseResult]:
    """Probe-row bitwise invariance against B=1 for one execution path."""
    sched = schedule_for(core)
    reference = record_reference(
        core,
        sched,
        execution=execution,
        composition=composition,
        chunks=chunks,
        device=device,
        pool=pool,
    )
    results: list[CaseResult] = []
    if execution == "graph":
        eager_reference = record_reference(
            core,
            sched,
            execution="eager",
            composition=composition,
            chunks=chunks,
            device=device,
        )
        divergence = None
        for chunk, (graph_snapshot, eager_snapshot) in enumerate(
            zip(reference, eager_reference, strict=True)
        ):
            views = [(name, tensor.unsqueeze(0)) for name, tensor in graph_snapshot]
            divergence = compare_rows(chunk, views, eager_snapshot, [0])
            if divergence is not None:
                break
        results.append(
            CaseResult(
                mode,
                policy,
                execution,
                composition,
                "1 vs eager",
                divergence is None,
                _describe(divergence),
            )
        )
    for population in populations:
        if population == 1:
            continue
        divergence = check_population(
            core,
            sched,
            reference,
            population=population,
            execution=execution,
            composition=composition,
            chunks=chunks,
            device=device,
            pool=pool,
        )
        if divergence is not None and execution == "eager":
            divergence.localized = localize(
                core,
                sched,
                population=population,
                composition=composition,
                chunk=divergence.chunk,
                device=device,
            )
        elif divergence is not None:
            divergence.localized = (
                f"graph replay; first divergent layer state: {divergence.tensor}"
            )
        results.append(
            CaseResult(
                mode,
                policy,
                execution,
                composition,
                population,
                divergence is None,
                _describe(divergence),
            )
        )
    return results


def run_to_run_case(
    core: nn.Module,
    *,
    mode: str,
    policy: str,
    execution: str,
    population: int,
    chunks: int,
    device: torch.device,
    pool: Any = None,
) -> CaseResult:
    """The same mixed B=``population`` sequence twice: all rows bitwise equal."""
    sched = schedule_for(core)
    first: list[Snapshot] = []
    final_state: list[torch.Tensor] = []
    found: list[Divergence] = []

    def record(
        chunk: int,
        encoded: torch.Tensor,
        conditioned: torch.Tensor,
        state: SimpleNamespace,
    ) -> None:
        first.append(
            [("encoded", encoded.clone()), ("conditioned", conditioned.clone())]
        )
        if chunk == chunks - 1:
            final_state.extend(tensor.clone() for tensor in _state_tensors(state))

    def compare(
        chunk: int,
        encoded: torch.Tensor,
        conditioned: torch.Tensor,
        state: SimpleNamespace,
    ) -> None:
        if found:
            return
        for (name, expected), actual in zip(
            first[chunk], (encoded, conditioned), strict=True
        ):
            if not torch.equal(actual, expected):
                bad = (
                    (actual != expected).flatten(1).any(1).nonzero().flatten().tolist()
                )
                max_abs, count = _diff(actual, expected)
                found.append(Divergence(chunk, name, max_abs, count, bad[:16]))
                return
        if chunk == chunks - 1:
            for index, (actual, expected) in enumerate(
                zip(_state_tensors(state), final_state, strict=True)
            ):
                if not torch.equal(actual, expected):
                    max_abs, count = _diff(actual, expected)
                    found.append(
                        Divergence(
                            chunk, f"final state tensor {index}", max_abs, count, []
                        )
                    )
                    return

    for on_chunk in (record, compare):
        run_sequence(
            core,
            sched,
            population=population,
            composition="mixed",
            execution=execution,
            chunks=chunks,
            device=device,
            on_chunk=on_chunk,
            pool=pool,
        )
    divergence = found[0] if found else None
    return CaseResult(
        mode,
        policy,
        execution,
        "run-to-run",
        population,
        divergence is None,
        _describe(divergence),
    )


@dataclass
class TimingResult:
    mode: str
    policy: str
    population: int
    median_ms: float
    p10_ms: float
    p90_ms: float
    iters: int


def time_graph_replay(
    core: nn.Module,
    *,
    mode: str,
    policy: str,
    population: int,
    iters: int,
    warmup: int,
    device: torch.device,
    pool: Any = None,
) -> TimingResult:
    """Median CUDA-event time of one steady-state encoder graph replay."""
    sched = schedule_for(core)
    # Steady rows with full history: the serving steady state.
    state = new_state(
        sched, population=population, composition="mixed", probe_row=0, device=device
    )
    for valid in state.window_valid:
        valid.fill_(HISTORY)
    mel, _offsets, _lengths, prompt = chunk_inputs(
        sched,
        population=population,
        chunk=1,
        composition="mixed",
        probe_row=0,
        device=device,
    )
    offsets = torch.full_like(_offsets, sched.drop)
    lengths = torch.full_like(_lengths, sched.steady_len)
    runner = GraphRunner(
        core, state, sched.out_width, (mel, offsets, lengths, prompt), pool
    )
    for _ in range(warmup):
        runner.graph.replay()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for start, end in events:
        start.record()
        runner.graph.replay()
        end.record()
    torch.accelerator.synchronize()
    times = sorted(start.elapsed_time(end) for start, end in events)
    quantiles = statistics.quantiles(times, n=10)
    return TimingResult(
        mode,
        policy,
        population,
        statistics.median(times),
        quantiles[0],
        quantiles[-1],
        iters,
    )


# --------------------------------------------------------------------------
# Reporting and driver
# --------------------------------------------------------------------------


def format_cases(cases: Sequence[CaseResult]) -> str:
    lines = [
        f"{'mode':5} {'policy':12} {'exec':5} {'composition':10} {'B':>10}  result"
    ]
    for case in cases:
        verdict = "equal" if case.equal else f"DIVERGED  {case.detail}"
        lines.append(
            f"{case.mode:5} {case.policy:12} {case.execution:5} {case.composition:10} "
            f"{case.population!s:>10}  {verdict}"
        )
    return "\n".join(lines)


def format_timings(timings: Sequence[TimingResult]) -> str:
    lines = [
        f"{'mode':5} {'policy':12} {'B':>4} {'median ms':>10} {'p10 ms':>8} "
        f"{'p90 ms':>8} {'iters':>6}"
    ]
    for t in sorted(
        timings,
        key=lambda item: (item.policy, item.population, BI_MODES.index(item.mode)),
    ):
        lines.append(
            f"{t.mode:5} {t.policy:12} {t.population:>4} {t.median_ms:>10.3f} "
            f"{t.p10_ms:>8.3f} {t.p90_ms:>8.3f} {t.iters:>6}"
        )
    return "\n".join(lines)


def _device() -> torch.device:
    return torch.device("cuda", torch.accelerator.current_device_index())


@torch.inference_mode()
def run_mode(args: argparse.Namespace) -> dict[str, Any]:
    settings = apply_bi_mode(args.mode)
    device = _device()
    pool = torch.cuda.graph_pool_handle()
    cases: list[CaseResult] = []
    timings: list[TimingResult] = []
    for policy in args.policies:
        core = build_core(policy, device)
        if not args.timing_only:
            for execution in args.executions:
                for composition in COMPOSITIONS:
                    cases += invariance_cases(
                        core,
                        mode=args.mode,
                        policy=policy,
                        execution=execution,
                        composition=composition,
                        populations=args.populations,
                        chunks=args.chunks,
                        device=device,
                        pool=pool,
                    )
                cases.append(
                    run_to_run_case(
                        core,
                        mode=args.mode,
                        policy=policy,
                        execution=execution,
                        population=max(args.populations),
                        chunks=args.chunks,
                        device=device,
                        pool=pool,
                    )
                )
        if args.timing or args.timing_only:
            for population in TIMING_POPULATIONS:
                timings.append(
                    time_graph_replay(
                        core,
                        mode=args.mode,
                        policy=policy,
                        population=population,
                        iters=args.iters,
                        warmup=args.warmup,
                        device=device,
                        pool=pool,
                    )
                )
        del core
    return {
        "settings": settings,
        "cases": [asdict(c) for c in cases],
        "timings": [asdict(t) for t in timings],
    }


def _print_report(reports: Sequence[dict[str, Any]]) -> bool:
    cases = [CaseResult(**case) for report in reports for case in report["cases"]]
    timings = [
        TimingResult(**timing) for report in reports for timing in report["timings"]
    ]
    for report in reports:
        print("settings:", json.dumps(report["settings"], sort_keys=True))
    if cases:
        print(
            "\n# Probe-row bitwise invariance (reference: same execution path at B=1)"
        )
        print(format_cases(cases))
    if timings:
        print("\n# Encoder graph replay (160 ms steady chunk, CUDA events)")
        print(format_timings(timings))
    return all(case.equal for case in cases)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else None
    )
    parser.add_argument(
        "--mode", choices=BI_MODES, default=os.environ.get("BI_MODE", "none")
    )
    parser.add_argument(
        "--all-modes", action="store_true", help="one fresh subprocess per BI_MODE"
    )
    parser.add_argument(
        "--policies", type=lambda v: tuple(v.split(",")), default=POLICIES
    )
    parser.add_argument(
        "--executions", type=lambda v: tuple(v.split(",")), default=EXECUTIONS
    )
    parser.add_argument(
        "--populations",
        type=lambda v: tuple(int(p) for p in v.split(",")),
        default=POPULATIONS,
    )
    parser.add_argument("--chunks", type=int, default=DEFAULT_CHUNKS)
    parser.add_argument(
        "--timing", action="store_true", help="also time graph replay at B=20 and B=64"
    )
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--json", help="write all results to this path")
    args = parser.parse_args(argv)
    if args.iters < 50:
        parser.error("--iters must be at least 50")
    if not torch.cuda.is_available():
        print("CUDA is unavailable; this falsifier needs a GPU.", file=sys.stderr)
        return 2
    reports: list[dict[str, Any]] = []
    if args.all_modes:
        forwarded = [
            arg
            for arg in (argv if argv is not None else sys.argv[1:])
            if arg != "--all-modes"
        ]
        for mode in BI_MODES:
            child = [sys.executable, __file__, *forwarded, "--mode", mode]
            env = {**os.environ, "BI_MODE": mode}
            env.pop("VLLM_BATCH_INVARIANT", None)
            completed = subprocess.run(
                child, env=env, capture_output=True, text=True, check=False
            )
            sys.stderr.write(completed.stderr[-4000:])
            payload = [
                line
                for line in completed.stdout.splitlines()
                if line.startswith(RESULT_MARKER)
            ]
            if not payload:
                print(completed.stdout)
                print(
                    f"mode {mode} produced no result (exit {completed.returncode})",
                    file=sys.stderr,
                )
                return 2
            reports.append(json.loads(payload[-1][len(RESULT_MARKER) :]))
    else:
        reports.append(run_mode(args))
        print(RESULT_MARKER + json.dumps(reports[-1]))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(reports, handle, indent=2)
    return 0 if _print_report(reports) else 1


# --------------------------------------------------------------------------
# Pytest entry points
# --------------------------------------------------------------------------

_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires real CUDA"
)


@pytest.fixture
def bi_mode(isolated_batch_invariance) -> str:
    mode = os.environ.get("BI_MODE", "none")
    print("\nsettings:", json.dumps(apply_bi_mode(mode), sort_keys=True))
    return mode


@pytest.fixture(scope="module")
def graph_pool() -> Any:
    return torch.cuda.graph_pool_handle()


@_requires_cuda
@torch.inference_mode()
@pytest.mark.parametrize("composition", COMPOSITIONS)
@pytest.mark.parametrize("execution", EXECUTIONS)
@pytest.mark.parametrize("policy", POLICIES)
def test_probe_row_is_bitwise_batch_invariant(
    bi_mode: str, graph_pool: Any, policy: str, execution: str, composition: str
) -> None:
    """@spec PORT-PREC-019, PORT-PREC-020: only mode-on promises invariance."""
    core = build_core(policy, _device())
    cases = invariance_cases(
        core,
        mode=bi_mode,
        policy=policy,
        execution=execution,
        composition=composition,
        populations=POPULATIONS,
        chunks=DEFAULT_CHUNKS,
        device=_device(),
        pool=graph_pool,
    )
    report = format_cases(cases)
    print("\n" + report)
    if _EXECUTION_MODE.enabled:
        assert all(case.equal for case in cases), report


@_requires_cuda
@torch.inference_mode()
@pytest.mark.parametrize("execution", EXECUTIONS)
@pytest.mark.parametrize("policy", POLICIES)
def test_population_128_is_run_to_run_deterministic(
    bi_mode: str, graph_pool: Any, policy: str, execution: str
) -> None:
    """@spec PORT-PREC-016: the same B=128 sequence twice is bitwise identical."""
    core = build_core(policy, _device())
    case = run_to_run_case(
        core,
        mode=bi_mode,
        policy=policy,
        execution=execution,
        population=max(POPULATIONS),
        chunks=DEFAULT_CHUNKS,
        device=_device(),
        pool=graph_pool,
    )
    print("\n" + format_cases([case]))
    assert case.equal, case.detail


_TINY = {
    "feat_in": 16,
    "d_model": 32,
    "d_ff": 64,
    "n_layers": 3,
    "n_heads": 4,
    "conv_kernel": 9,
    "subsampling_channels": 16,
    "num_prompts": 8,
}


@pytest.mark.cpu_test
def test_bi_settings_uses_per_operation_precision(monkeypatch) -> None:
    """Reporting vLLM's precision settings must not read legacy TF32 flags."""

    class CudnnSettings:
        conv = SimpleNamespace(fp32_precision="ieee")
        rnn = SimpleNamespace(fp32_precision="ieee")
        benchmark = False

        @property
        def allow_tf32(self):
            raise RuntimeError(
                "legacy and per-operation precision APIs cannot be mixed"
            )

    monkeypatch.setattr(torch.backends, "cudnn", CudnnSettings())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    settings = bi_settings()
    assert settings["cudnn_conv_fp32_precision"] == "ieee"
    assert settings["cudnn_rnn_fp32_precision"] == "ieee"


@pytest.mark.cpu_test
@torch.inference_mode()
def test_harness_localizes_injected_batch_dependence() -> None:
    """The comparison and localization path detects one batch-coupled module."""
    device = torch.device("cpu")
    core = build_core("fp32", device, _TINY)
    sched = schedule_for(core)
    # Couple every row to the batch at the first leaf module: a no-op at B=1,
    # a perturbation otherwise. It precedes any natural CPU GEMM variance.
    target = cast(FastConformerEncoder, core.encoder).pre_encode.conv[0]
    handle = target.register_forward_hook(
        lambda _m, _a, out: out + 1e-2 * (out.mean(0, keepdim=True) - out)
    )
    try:
        reference = record_reference(
            core, sched, execution="eager", composition="mixed", chunks=3, device=device
        )
        divergence = check_population(
            core,
            sched,
            reference,
            population=3,
            execution="eager",
            composition="mixed",
            chunks=3,
            device=device,
        )
        assert divergence is not None
        assert divergence.chunk == 0 and divergence.tensor == "layer00.channel_cache"
        assert divergence.differing_rows == [2]
        localized = localize(
            core, sched, population=3, composition="mixed", chunk=0, device=device
        )
        assert localized.startswith("encoder.pre_encode.conv.0 (call 0)"), localized
    finally:
        handle.remove()
    # Without the injected coupling the same sequence reproduces itself.
    repeat = record_reference(
        core, sched, execution="eager", composition="repeated", chunks=3, device=device
    )
    assert (
        check_population(
            core,
            sched,
            repeat,
            population=1,
            execution="eager",
            composition="repeated",
            chunks=3,
            device=device,
        )
        is None
    )


if __name__ == "__main__":
    sys.exit(main())
