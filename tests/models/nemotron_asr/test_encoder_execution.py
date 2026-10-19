# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feature-gated static encoder execution contracts."""

from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from torch import nn

from vllm.models.nemotron_asr import (
    encoder_execution as encoder_execution_module,
)
from vllm.models.nemotron_asr.decode_graph import (
    GraphRuntime,
    platform_graph_runtime,
)
from vllm.models.nemotron_asr.encoder import (
    RelPositionalEncoding,
)
from vllm.models.nemotron_asr.encoder_execution import (
    ResolvedEncoderExecution,
)
from vllm.models.nemotron_asr.encoder_execution import (
    build_encoder_execution as _build_encoder_execution,
)

if TYPE_CHECKING:
    from vllm.models.nemotron_asr.advance import SessionStateBatch
    from vllm.models.nemotron_asr.nemotron_asr import NemotronASRCore

pytestmark = [pytest.mark.core_model]


def build_encoder_execution(
    core: nn.Module | SimpleNamespace,
    hf_config: object,
    *,
    maximum_population: int,
    warmup_geometries: tuple[int, ...],
    vllm_config: object | None = None,
    graph_runtime: GraphRuntime | None = None,
) -> ResolvedEncoderExecution:
    # Dynamic test boundary: compiler doubles provide encoder metadata and
    # policy while each test substitutes the transition it exercises. The
    # real-CUDA fixture provides the actual encoder, conditioner, and policy.
    # Neither fixture claims the unrelated decoder/resident model surface.
    return _build_encoder_execution(
        cast("NemotronASRCore", core),
        hf_config,
        maximum_population=maximum_population,
        warmup_geometries=warmup_geometries,
        vllm_config=vllm_config,
        graph_runtime=graph_runtime,
    )


class _PreEncode(nn.Module):
    """Small host-shape authority with the production 8x length rule."""

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        out = lengths
        for _ in range(3):
            out = torch.div(out, 2, rounding_mode="floor") + 1
        return out.to(torch.int64)


class _CompilerEncoder(nn.Module):
    running: torch.Tensor
    guard_bias: float

    def __init__(self, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.pre_encode = _PreEncode()
        self.pos_enc = RelPositionalEncoding(8)
        self.weight = nn.Parameter(torch.ones(1, dtype=dtype))
        self.register_buffer("running", torch.zeros(2, 3))


class _CompilerCore(nn.Module):
    def __init__(
        self,
        *,
        activation_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.policy = SimpleNamespace(
            encoder_compute_override=None,
            dtype_for=lambda tensor_class: (
                activation_dtype
                if tensor_class in ("activations", "encoder_compute")
                else torch.float32
            ),
        )
        self.encoder = _CompilerEncoder(dtype=activation_dtype)
        self.lid = nn.Linear(1, 1, bias=False).to(activation_dtype)
        self.lid.num_prompts = 4
        self.predictor = nn.Linear(1, 1, bias=False)


def _compiled_core(
    *,
    activation_dtype: torch.dtype = torch.float32,
) -> _CompilerCore:
    return _CompilerCore(activation_dtype=activation_dtype).eval()


def _compiled_config() -> SimpleNamespace:
    return SimpleNamespace(
        encoder_execution_arm="compiled-static",
        att_context_left=56,
    )


def _dense_graphed_config(arm: str = "dense-graphed") -> SimpleNamespace:
    return SimpleNamespace(
        encoder_execution_arm=arm,
        att_context_left=56,
    )


class _GraphCaches:
    """Small test adapter with the production graph-storage protocol."""

    def __init__(
        self,
        channel: tuple[torch.Tensor, ...],
        time: tuple[torch.Tensor, ...],
        valid_slots: tuple[torch.Tensor, ...],
    ) -> None:
        self.channel = channel
        self.time = time
        self._valid_slots = valid_slots
        self.left_context = int(channel[0].shape[1])

    @property
    def valid(self) -> torch.Tensor:
        return self._valid_slots[0].reshape(-1).to(torch.long)

    @valid.setter
    def valid(self, value: torch.Tensor) -> None:
        for slot in self._valid_slots:
            slot.copy_(value.reshape(slot.shape).to(slot.dtype))

    def graph_storage(self) -> Any:
        storage_type = encoder_execution_module.EncoderCacheStorage
        return storage_type(
            channel=self.channel,
            time=self.time,
            valid=self._valid_slots,
        )

    def empty_like(self) -> "_GraphCaches":
        return type(self)(
            tuple(torch.empty_like(tensor) for tensor in self.channel),
            tuple(torch.empty_like(tensor) for tensor in self.time),
            tuple(torch.empty_like(tensor) for tensor in self._valid_slots),
        )

    def narrow_rows(self, rows: int) -> "_GraphCaches":
        return type(self)(
            tuple(tensor.narrow(0, 0, rows) for tensor in self.channel),
            tuple(tensor.narrow(0, 0, rows) for tensor in self.time),
            tuple(tensor.narrow(0, 0, rows) for tensor in self._valid_slots),
        )


class _FakeGraphWrapper:
    """Record-only first capture, then Python execution for CPU contracts.

    Subsequent calls do not emulate captured execution: only the real CUDA
    discriminator can detect dependencies on Python work absent from replay.
    """

    def __init__(self, runnable: Any, graph_mode: Any) -> None:
        self.runnable = runnable
        self.graph_mode = graph_mode
        self.captured = False
        self.output: tuple[torch.Tensor, ...] | None = None

    def __call__(self) -> Any:
        if self.graph_mode() and not self.captured:
            # Native stream capture records operations without running them.
            # Returning the stable buffers leaves their seeded values intact.
            assert self.output is not None
            self.captured = True
            return self.output
        self.output = self.runnable()
        return self.output


def _graph_runtime(
    *,
    fail_graph_call: int | None = None,
    graph_error: Exception | None = None,
) -> GraphRuntime:
    graph_calls = 0
    current_mode = "none"

    @contextmanager
    def context(
        _metadata: Any,
        _config: Any,
        *,
        cudagraph_runtime_mode: str,
        batch_descriptor: Any,
    ) -> Any:
        nonlocal graph_calls, current_mode
        del batch_descriptor
        if cudagraph_runtime_mode == "graph":
            graph_calls += 1
            if fail_graph_call == graph_calls:
                raise (
                    graph_error
                    if graph_error is not None
                    else RuntimeError("synthetic encoder graph failure")
                )
        previous_mode = current_mode
        current_mode = cudagraph_runtime_mode
        try:
            yield
        finally:
            current_mode = previous_mode

    return GraphRuntime(
        wrapper_factory=lambda run, *_args, **_kwargs: _FakeGraphWrapper(
            run, lambda: current_mode == "graph"
        ),
        forward_context=context,
        capture_context=lambda _device: nullcontext(),
        descriptor_factory=lambda population: (population, population),
        eager_mode="none",
        graph_mode="graph",
        synchronize=lambda _device: None,
        set_capture_enabled=lambda _enabled: None,
    )


def _graph_transition_args(
    *,
    population: int = 2,
    mel_width: int = 17,
    out_width: int = 3,
    base: float = 1.0,
) -> tuple[Any, ...]:
    channel = (
        torch.full((population, 56, 3), base),
        torch.full((population, 56, 3), base + 1),
    )
    time = (
        torch.full((population, 3, 2), base + 2),
        torch.full((population, 3, 2), base + 3),
    )
    valid_slots = (
        torch.arange(population, dtype=torch.int32).reshape(population, 1),
        torch.arange(population, dtype=torch.int32).reshape(population, 1),
    )
    return (
        torch.full((population, 5, mel_width), base),
        _GraphCaches(channel, time, valid_slots),
        torch.arange(population, dtype=torch.long) % 2,
        torch.arange(population, dtype=torch.long) % (out_width + 1),
        out_width,
        torch.arange(population, dtype=torch.long),
    )


def _functional_graph_transition(
    _core: Any,
    mel: torch.Tensor,
    caches: _GraphCaches,
    out_offsets: torch.Tensor,
    out_lengths: torch.Tensor,
    out_width: int,
    prompt_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = (mel[:, 0, 0] * (out_lengths > 0)).reshape(-1, 1, 1)
    for tensor in caches.channel:
        tensor.add_(delta)
    for tensor in caches.time:
        tensor.add_(delta)
    caches.valid = caches.valid + out_lengths
    raw = mel[:, 0, :out_width].unsqueeze(-1) + out_offsets.reshape(-1, 1, 1)
    frame = torch.arange(out_width).reshape(1, -1, 1)
    raw = torch.where(frame < out_lengths.reshape(-1, 1, 1), raw, torch.zeros_like(raw))
    conditioned = raw + prompt_index.reshape(-1, 1, 1)
    conditioned = torch.where(
        frame < out_lengths.reshape(-1, 1, 1),
        conditioned,
        torch.zeros_like(conditioned),
    )
    return raw, conditioned


def _transition_args(
    *,
    population: int = 1,
    mel_width: int = 17,
    out_width: int = 3,
    device: torch.device | str = "cpu",
) -> tuple[Any, ...]:
    with torch.inference_mode():
        caches = SimpleNamespace(
            channel=(torch.zeros(population, 2, 3, device=device),),
            time=(torch.zeros(population, 2, 3, device=device),),
            valid=torch.zeros(population, dtype=torch.long, device=device),
            left_context=56,
        )
        return (
            torch.zeros(population, 5, mel_width, device=device),
            caches,
            torch.zeros(population, dtype=torch.long, device=device),
            torch.ones(population, dtype=torch.long, device=device),
            out_width,
            torch.zeros(population, dtype=torch.long, device=device),
        )


@pytest.mark.cpu_test
def test_compiled_static_declares_capacity_from_shared_geometry_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-010
    geometry_shape = getattr(
        encoder_execution_module,
        "encoder_geometry_shape",
        None,
    )
    assert callable(geometry_shape), "PORT-PERF-010 missing encoder_geometry_shape"
    calls: list[int] = []

    def fake_shape(_core: Any, geometry: int) -> Any:
        calls.append(geometry)
        return SimpleNamespace(
            cadence_frames=geometry + 1,
            mel_width=geometry + 10,
            out_width=geometry + 20,
        )

    monkeypatch.setattr(
        encoder_execution_module,
        "encoder_geometry_shape",
        fake_shape,
    )
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    execution = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0, 2, 4),
    )

    assert calls == [0, 2, 4]
    assert execution.t_cap == 4 + 20 + 56
    assert not hasattr(execution, "positional_width")


@pytest.mark.cpu_test
def test_encoder_geometry_shape_reads_the_canonical_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-010
    from vllm.models.nemotron_asr import manifests

    monkeypatch.setattr(manifests, "CADENCES", {"probe": (56, 2)})
    shape = encoder_execution_module.encoder_geometry_shape(
        cast("NemotronASRCore", _compiled_core()),
        0,
    )

    assert shape.cadence_frames == 24
    assert shape.mel_width == 33
    assert shape.out_width == int(_PreEncode().output_lengths(torch.tensor([33]))[0])


@pytest.mark.cpu_test
def test_encoder_geometry_shape_stays_on_host_under_non_cpu_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-010
    from vllm.models.nemotron_asr import manifests

    monkeypatch.setattr(manifests, "CADENCES", {"probe": (56, 2)})
    core = _compiled_core()

    with torch.device("meta"):
        shape = encoder_execution_module.encoder_geometry_shape(
            cast("NemotronASRCore", core), 0
        )

    assert shape.mel_width == 33
    assert shape.out_width == int(_PreEncode().output_lengths(torch.tensor([33]))[0])


@pytest.mark.cpu_test
def test_compiled_static_preprofile_materializes_before_compiler_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009, PORT-PERF-010
    core = _compiled_core(activation_dtype=torch.bfloat16)
    compiler_calls = 0
    positional_ids: list[int] = []

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        def compiled(*args: Any) -> Any:
            nonlocal compiler_calls
            compiler_calls += 1
            assert torch.is_inference_mode_enabled()
            assert core.encoder.pos_enc.pe.shape == (
                1,
                2 * execution.t_cap - 1,
                8,
            )
            assert args[0].dtype == torch.float32
            assert core.encoder.weight.dtype == torch.bfloat16
            assert core.encoder.pos_enc.pe.dtype == torch.bfloat16
            assert core.encoder.pos_enc.pe.device == args[0].device
            assert args[0].layout == torch.strided
            dispatch_keys = str(torch._C._dispatch_keys(args[0]))
            assert "Autograd" not in dispatch_keys
            assert "ADInplaceOrView" not in dispatch_keys
            positional_ids.append(id(core.encoder.pos_enc.pe))
            return fn(*args)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        lambda *_args: (torch.zeros(1), torch.zeros(1)),
    )
    execution = build_encoder_execution(
        core,
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0,),
    )
    args = _transition_args(out_width=execution.t_cap - 56)

    execution.profile_cell(
        geometry=0,
        population=1,
        invoke=lambda: execution.transition(*args),
    )
    execution.warmup_domain(
        expected_cells=((0, 1),),
        invoke=lambda _geometry, _population: execution.transition(*args),
    )

    assert compiler_calls == 2
    assert len(set(positional_ids)) == 1


@pytest.mark.cpu_test
def test_missing_core_authority_fails_before_compiled_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009, PORT-PERF-010
    compiler_calls = 0

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        def compiled(*args: Any) -> Any:
            nonlocal compiler_calls
            compiler_calls += 1
            return fn(*args)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    execution = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0,),
    )
    execution._core = None
    with pytest.raises(ValueError, match="positional capacity is unavailable"):
        execution.profile_cell(
            geometry=0,
            population=1,
            invoke=lambda: execution.transition(*_transition_args()),
        )
    assert compiler_calls == 0
    assert not execution.ready


@pytest.mark.cpu_test
def test_encoder_fp16_materialization_receipt_and_policy_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.models.nemotron_asr.precision import FP16_ENCODER_EXPERIMENT, FP32_BRINGUP

    core = _compiled_core()
    core.policy = FP16_ENCODER_EXPERIMENT
    core.encoder.half()
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        lambda *_args: (torch.zeros(1), torch.zeros(1)),
    )
    execution = build_encoder_execution(
        core,
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0,),
    )
    args = _transition_args(out_width=execution.t_cap - 56)
    execution.profile_cell(
        geometry=0, population=1, invoke=lambda: execution.transition(*args)
    )
    execution.warmup_domain(
        expected_cells=((0, 1),),
        invoke=lambda _geometry, _population: execution.transition(*args),
    )
    assert core.encoder.pos_enc.pe.dtype == torch.float16
    assert core.lid.weight.dtype == torch.float32
    assert (
        execution.ready_receipt()["precision_policy_hash"]
        == FP16_ENCODER_EXPERIMENT.content_hash
    )
    core.policy = FP32_BRINGUP
    with pytest.raises(ValueError, match="precision_policy"):
        execution.profile_ready_cell(
            geometry=0, population=1, invoke=lambda: execution.transition(*args)
        )
    assert not execution.ready


@pytest.mark.cpu_test
def test_compiled_static_rejects_finalized_parameter_dtype_mismatch_before_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-010
    compiler_calls = 0

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        def compiled(*args: Any) -> Any:
            nonlocal compiler_calls
            compiler_calls += 1
            return fn(*args)

        return compiled

    core = _compiled_core(activation_dtype=torch.bfloat16)
    core.lid.to(torch.float32)
    monkeypatch.setattr(torch, "compile", fake_compile)
    execution = build_encoder_execution(
        core,
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0,),
    )

    with pytest.raises(
        ValueError,
        match="parameter dtype differs from activation policy",
    ):
        execution.profile_cell(
            geometry=0,
            population=1,
            invoke=lambda: execution.transition(*_transition_args()),
        )

    assert compiler_calls == 0
    assert not execution.ready


@pytest.mark.cpu_test
def test_compiled_static_undersized_capacity_fails_before_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-010
    geometry_shape = getattr(
        encoder_execution_module,
        "encoder_geometry_shape",
        None,
    )
    assert callable(geometry_shape), "PORT-PERF-010 missing encoder_geometry_shape"
    compiler_calls = 0

    def fake_shape(_core: Any, _geometry: int) -> Any:
        return SimpleNamespace(
            cadence_frames=1,
            mel_width=8,
            out_width=7,
        )

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        def compiled(*args: Any) -> Any:
            nonlocal compiler_calls
            compiler_calls += 1
            return fn(*args)

        return compiled

    monkeypatch.setattr(
        encoder_execution_module,
        "encoder_geometry_shape",
        fake_shape,
    )
    monkeypatch.setattr(torch, "compile", fake_compile)
    execution = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0,),
    )

    with pytest.raises(
        ValueError,
        match=r"encoder\.pos_enc\.pe.*capacity",
    ):
        execution.profile_cell(
            geometry=0,
            population=1,
            invoke=lambda: execution.transition(
                *_transition_args(out_width=8),
            ),
        )
    assert compiler_calls == 0


@pytest.mark.cpu_test
def test_compiled_static_rechecks_capacity_after_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-010
    compiler_calls = 0

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        def compiled(*args: Any) -> Any:
            nonlocal compiler_calls
            compiler_calls += 1
            return fn(*args)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        lambda *_args: (torch.zeros(1), torch.zeros(1)),
    )
    execution = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0,),
    )
    execution.profile_cell(
        geometry=0,
        population=1,
        invoke=lambda: execution.transition(*_transition_args()),
    )

    with pytest.raises(ValueError, match=r"encoder\.pos_enc\.pe.*capacity"):
        execution.warmup_domain(
            expected_cells=((0, 1),),
            invoke=lambda _geometry, _population: execution.transition(
                *_transition_args(out_width=execution.t_cap),
            ),
        )

    assert compiler_calls == 1
    assert not execution.ready


@pytest.mark.cpu_test
def test_compiled_static_rejects_mislabeled_geometry_before_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009, PORT-PERF-010
    compiler_calls = 0

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        def compiled(*args: Any) -> Any:
            nonlocal compiler_calls
            compiler_calls += 1
            return fn(*args)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    execution = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0, 1),
    )

    with pytest.raises(ValueError, match=r"declared geometry 1"):
        execution.profile_cell(
            geometry=1,
            population=1,
            invoke=lambda: execution.transition(*_transition_args()),
        )

    assert compiler_calls == 0
    assert not execution.ready


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("mutation", "expected_name"),
    (
        ("device", r"out_offsets.*runner device"),
        ("all-wrong-device", r"mel.*runner device"),
        ("layout", r"caches\.channel\[0\].*strided layout"),
        ("mel-dtype", r"mel.*torch\.float32"),
        ("cache-dtype", r"caches\.channel\[0\].*torch\.float32"),
        ("integer-dtype", r"prompt_index.*torch\.int64"),
        ("dispatch-keys", r"caches\.time\[0\].*inference mode"),
    ),
)
def test_compiled_static_validates_complete_transition_tuple_before_compiler(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_name: str,
) -> None:
    # @spec PORT-PERF-010
    compiler_calls = 0

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        def compiled(*args: Any) -> Any:
            nonlocal compiler_calls
            compiler_calls += 1
            return fn(*args)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    execution = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0,),
    )
    args = list(_transition_args())
    caches = args[1]
    if mutation == "device":
        args[2] = torch.zeros(1, dtype=torch.long, device="meta")
    elif mutation == "all-wrong-device":
        args = list(_transition_args(device="meta"))
        caches = args[1]
    elif mutation == "layout":
        with torch.inference_mode():
            caches.channel = (torch.zeros(1, 2, 3).to_sparse(),)
    elif mutation == "mel-dtype":
        with torch.inference_mode():
            args[0] = torch.zeros(1, 5, 17, dtype=torch.bfloat16)
    elif mutation == "cache-dtype":
        with torch.inference_mode():
            caches.channel = (torch.zeros(1, 2, 3, dtype=torch.bfloat16),)
    elif mutation == "integer-dtype":
        with torch.inference_mode():
            args[5] = torch.zeros(1, dtype=torch.int32)
    elif mutation == "dispatch-keys":
        caches.time = (torch.zeros(1, 2, 3),)
    else:  # pragma: no cover - the parametrization is closed.
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=expected_name):
        execution.profile_cell(
            geometry=0,
            population=1,
            invoke=lambda: execution.transition(*args),
        )

    assert compiler_calls == 0
    assert not execution.ready


@pytest.mark.cpu_test
def test_compiled_static_uses_warmed_signature_after_seal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009, PORT-PERF-010
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        lambda *_args: (torch.zeros(1), torch.zeros(1)),
    )
    execution = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0,),
    )
    args = _transition_args()
    execution.warmup_domain(
        expected_cells=((0, 1),),
        invoke=lambda _geometry, _population: execution.transition(*args),
    )

    monkeypatch.setattr(
        encoder_execution_module,
        "_validate_transition_tensor_contract",
        lambda *_args, **_kwargs: pytest.fail(
            "sealed serving must use its warmed tensor signature"
        ),
    )

    execution.transition(*args)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("mutation", "expected_name"),
    (
        ("add-parameter", r"encoder\.late_weight"),
        ("remove-parameter", r"encoder\.weight"),
        ("add-buffer", r"encoder\.late_buffer"),
        ("remove-buffer", r"encoder\.running"),
        ("add-module", r"encoder\.late_module"),
        ("remove-module", r"encoder\.pre_encode"),
        ("replace-buffer", r"encoder\.running"),
        ("reshape-buffer", r"encoder\.running"),
        ("restride-buffer", r"encoder\.running"),
        ("relayout-buffer", r"encoder\.running"),
        ("recast-parameter", r"encoder\.weight"),
        ("requires-grad", r"encoder\.weight"),
        ("move-buffer", r"encoder\.running"),
        ("dispatch-keys", r"encoder\.running"),
        ("training", r"encoder.*training"),
    ),
)
def test_compiled_static_profile_detects_compiler_visible_model_mutation(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_name: str,
) -> None:
    # @spec PORT-PERF-010
    core = _compiled_core()

    def mutate() -> None:
        if mutation == "add-parameter":
            core.encoder.register_parameter(
                "late_weight",
                nn.Parameter(torch.ones(1)),
            )
        elif mutation == "remove-parameter":
            del core.encoder.weight
        elif mutation == "add-buffer":
            core.encoder.register_buffer("late_buffer", torch.ones(1))
        elif mutation == "remove-buffer":
            del core.encoder.running
        elif mutation == "add-module":
            core.encoder.add_module("late_module", nn.Identity())
        elif mutation == "remove-module":
            del core.encoder.pre_encode
        elif mutation == "replace-buffer":
            core.encoder.running = torch.ones(2, 3)
        elif mutation == "reshape-buffer":
            core.encoder.running.data = torch.ones(3, 2)
        elif mutation == "restride-buffer":
            core.encoder.running.data = torch.ones(3, 2).t()
            assert core.encoder.running.stride() == (1, 2)
        elif mutation == "relayout-buffer":
            core.encoder.running = torch.ones(2, 3).to_sparse()
            assert core.encoder.running.layout == torch.sparse_coo
        elif mutation == "recast-parameter":
            core.encoder.weight.data = core.encoder.weight.data.to(torch.float64)
        elif mutation == "requires-grad":
            core.encoder.weight.requires_grad_(False)
        elif mutation == "move-buffer":
            core.encoder.running = core.encoder.running.to("meta")
        elif mutation == "dispatch-keys":
            original_id = id(core.encoder.running)
            with torch.inference_mode():
                inference_tensor = torch.ones_like(core.encoder.running)
            core.encoder.running.data = inference_tensor
            assert id(core.encoder.running) == original_id
        elif mutation == "training":
            core.encoder.train()
        else:  # pragma: no cover - the parametrization is closed.
            raise AssertionError(mutation)

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        def compiled(*args: Any) -> Any:
            result = fn(*args)
            mutate()
            return result

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        lambda *_args: (torch.zeros(1), torch.zeros(1)),
    )
    execution = build_encoder_execution(
        core,
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0,),
    )

    with pytest.raises(ValueError, match=expected_name):
        execution.profile_cell(
            geometry=0,
            population=1,
            invoke=lambda: execution.transition(
                *_transition_args(),
            ),
        )


@pytest.mark.cpu_test
def test_compiled_static_rechecks_model_state_before_sealing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-010
    core = _compiled_core()
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        lambda *_args: (torch.zeros(1), torch.zeros(1)),
    )
    execution = build_encoder_execution(
        core,
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0,),
    )
    args = _transition_args()
    execution.profile_cell(
        geometry=0,
        population=1,
        invoke=lambda: execution.transition(*args),
    )
    core.encoder.running = torch.ones(2, 3)

    with pytest.raises(ValueError, match=r"encoder\.running"):
        execution.warmup_domain(
            expected_cells=((0, 1),),
            invoke=lambda _geometry, _population: execution.transition(
                *args,
            ),
        )
    assert not execution.ready


@pytest.mark.cpu_test
def test_compiled_static_ignores_unrelated_model_sibling_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-010
    core = _compiled_core()
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        lambda *_args: (torch.zeros(1), torch.zeros(1)),
    )
    execution = build_encoder_execution(
        core,
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0,),
    )
    args = _transition_args()
    execution.profile_cell(
        geometry=0,
        population=1,
        invoke=lambda: execution.transition(*args),
    )
    core.predictor.weight.data = core.predictor.weight.data.to(torch.float64)

    execution.warmup_domain(
        expected_cells=((0, 1),),
        invoke=lambda _geometry, _population: execution.transition(*args),
    )

    assert execution.ready


@pytest.mark.cpu_test
def test_absent_encoder_gate_is_the_unchanged_eager_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    def forbidden_compile(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("the absent baseline must not initialize torch.compile")

    monkeypatch.setattr(torch, "compile", forbidden_compile)
    resolved = build_encoder_execution(
        SimpleNamespace(),
        SimpleNamespace(),
        maximum_population=4,
        warmup_geometries=(0,),
    )
    assert resolved.arm == "eager"
    assert resolved.ready_receipt() == {
        "arm": "eager",
        "ready": True,
        "warmup_cells": [],
        "warmup_geometries": [],
        "warmup_populations": [],
    }


@pytest.mark.cpu_test
def test_explicit_eager_encoder_gate_does_not_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    def forbidden_compile(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("the explicit eager arm must not initialize torch.compile")

    monkeypatch.setattr(torch, "compile", forbidden_compile)
    resolved = build_encoder_execution(
        SimpleNamespace(),
        SimpleNamespace(encoder_execution_arm="eager"),
        maximum_population=4,
        warmup_geometries=(0,),
    )
    assert resolved.arm == "eager"


@pytest.mark.cpu_test
def test_compiled_static_uses_one_fail_closed_fullgraph_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    calls: list[tuple[Any, dict[str, Any]]] = []

    def fake_compile(fn: Any, **kwargs: Any) -> Any:
        calls.append((fn, kwargs))
        return "compiled-transition"

    monkeypatch.setattr(torch, "compile", fake_compile)
    resolved = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=4,
        warmup_geometries=(0, 1, 2, 3, 4),
    )
    assert resolved.arm == "compiled-static"
    assert callable(resolved.transition)
    assert len(calls) == 1
    assert resolved.warmup_populations == (1, 2, 3, 4)
    assert calls[0][1] == {
        "fullgraph": True,
        "dynamic": False,
        "options": {"triton.cudagraphs": False},
    }


@pytest.mark.cpu_test
def test_compiler_initialization_failure_is_not_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    def fail_compile(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("synthetic compiler failure")

    monkeypatch.setattr(torch, "compile", fail_compile)
    with pytest.raises(RuntimeError, match="synthetic compiler failure"):
        build_encoder_execution(
            _compiled_core(),
            _compiled_config(),
            maximum_population=1,
            warmup_geometries=(0,),
        )


@pytest.mark.cpu_test
def test_unknown_encoder_execution_arm_fails_closed() -> None:
    # @spec PORT-PERF-009
    with pytest.raises(ValueError, match="unknown encoder_execution_arm"):
        build_encoder_execution(
            SimpleNamespace(),
            SimpleNamespace(encoder_execution_arm="auto-magic"),
            maximum_population=4,
            warmup_geometries=(0,),
        )


@pytest.mark.cpu_test
def test_compiled_static_derives_complete_population_authority_from_runner_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    resolved = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=4,
        warmup_geometries=(0, 2, 4),
    )
    assert resolved.warmup_populations == (1, 2, 3, 4)
    assert resolved.warmup_geometries == (0, 2, 4)


@pytest.mark.cpu_test
def test_compiled_static_seals_observed_signatures_and_rejects_lazy_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    compiled_calls: list[tuple[int, int]] = []

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        def compiled(*args: Any) -> Any:
            compiled_calls.append((int(args[0].shape[0]), int(args[4])))
            return fn(*args)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    execution = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=2,
        warmup_geometries=(0,),
    )

    def run(population: int, *, mel_width: int = 17) -> None:
        with torch.inference_mode():
            caches = SimpleNamespace(
                channel=(torch.zeros(population, 2, 3),),
                time=(torch.zeros(population, 2, 3),),
                valid=torch.zeros(population, dtype=torch.long),
                left_context=56,
            )
            execution.transition(
                torch.zeros(population, 5, mel_width),
                caches,
                torch.zeros(population, dtype=torch.long),
                torch.ones(population, dtype=torch.long),
                3,
                torch.zeros(population, dtype=torch.long),
            )

    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        lambda *_args: (torch.zeros(1), torch.zeros(1)),
    )
    execution.warmup_domain(
        expected_cells=((0, 1), (0, 2)),
        invoke=lambda _geometry, population: run(population),
    )
    run(1)
    assert compiled_calls == [(1, 3), (2, 3), (1, 3)]
    assert execution.ready_receipt() == {
        "arm": "compiled-static",
        "ready": True,
        "warmup_cells": [[0, 1], [0, 2]],
        "warmup_geometries": [0],
        "warmup_populations": [1, 2],
    }

    with pytest.raises(ValueError, match="was not warmed"):
        run(1, mel_width=18)
    assert compiled_calls == [(1, 3), (2, 3), (1, 3)]

    with pytest.raises(ValueError, match="population 3 was not declared"):
        execution.transition(
            torch.zeros(3, 5, 17),
            SimpleNamespace(
                channel=(torch.zeros(3, 2, 3),),
                time=(torch.zeros(3, 2, 3),),
                valid=torch.zeros(3, dtype=torch.long),
                left_context=56,
            ),
            torch.zeros(3, dtype=torch.long),
            torch.ones(3, dtype=torch.long),
            3,
            torch.zeros(3, dtype=torch.long),
        )
    assert compiled_calls == [(1, 3), (2, 3), (1, 3)]


@pytest.mark.cpu_test
def test_compiled_static_warmup_scopes_cache_budget_to_declared_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    observed_limits: list[tuple[int, int]] = []

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        def compiled(*args: Any) -> Any:
            observed_limits.append(
                (
                    int(torch._dynamo.config.cache_size_limit),
                    int(torch._dynamo.config.accumulated_cache_size_limit),
                )
            )
            return fn(*args)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(torch._dynamo.config, "cache_size_limit", 8)
    monkeypatch.setattr(
        torch._dynamo.config,
        "accumulated_cache_size_limit",
        12,
    )
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        lambda *_args: (torch.zeros(1), torch.zeros(1)),
    )
    execution = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=4,
        warmup_geometries=(0, 1, 2, 3, 4),
    )
    cells = tuple(
        (geometry, population) for geometry in range(5) for population in range(1, 5)
    )

    def invoke(geometry: int, population: int) -> None:
        shape = execution._geometry_shapes[geometry]
        caches = SimpleNamespace(
            channel=(torch.zeros(population, 2, 3),),
            time=(torch.zeros(population, 2, 3),),
            valid=torch.zeros(population, dtype=torch.long),
            left_context=56,
        )
        execution.transition(
            torch.zeros(population, 5, shape.mel_width),
            caches,
            torch.zeros(population, dtype=torch.long),
            torch.ones(population, dtype=torch.long),
            shape.out_width,
            torch.zeros(population, dtype=torch.long),
        )

    execution.profile_cell(
        geometry=4,
        population=4,
        invoke=lambda: invoke(4, 4),
    )
    assert observed_limits == [(20, 20)]
    assert torch._dynamo.config.cache_size_limit == 8
    assert torch._dynamo.config.accumulated_cache_size_limit == 12

    execution.warmup_domain(expected_cells=cells, invoke=invoke)

    assert observed_limits == [(20, 20)] * 21
    assert torch._dynamo.config.cache_size_limit == 8
    assert torch._dynamo.config.accumulated_cache_size_limit == 12
    assert execution.ready


@pytest.mark.cpu_test
def test_compiled_static_preseal_invocation_requires_declared_cell_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    compiled_calls = 0

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        def compiled(*args: Any) -> Any:
            nonlocal compiled_calls
            compiled_calls += 1
            return fn(*args)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    execution = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0,),
    )
    caches = SimpleNamespace(
        channel=(torch.zeros(1, 2, 3),),
        time=(torch.zeros(1, 2, 3),),
        valid=torch.zeros(1, dtype=torch.long),
        left_context=56,
    )

    with pytest.raises(ValueError, match="lacks declared cell authority"):
        execution.transition(
            torch.zeros(1, 5, 17),
            caches,
            torch.zeros(1, dtype=torch.long),
            torch.ones(1, dtype=torch.long),
            3,
            torch.zeros(1, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="geometry 1 was not declared"):
        execution.warmup_cell(
            geometry=1,
            population=1,
            invoke=lambda: None,
        )
    with torch.inference_mode():
        execution._materialize_preprofile_state(out_width=3)
    with pytest.raises(ValueError, match=r"missing=\[\(0, 1\)\]"):
        execution._seal()
    assert compiled_calls == 0


@pytest.mark.cpu_test
def test_compiled_static_failure_restores_budget_and_discards_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009
    compile_instances: list[dict[str, Any]] = []

    def fake_compile(fn: Any, **_kwargs: Any) -> Any:
        instance = {
            "calls": 0,
            "fail_second": not compile_instances,
        }
        compile_instances.append(instance)

        def compiled(*args: Any) -> Any:
            instance["calls"] += 1
            if instance["fail_second"] and instance["calls"] == 2:
                raise RuntimeError("synthetic specialization failure")
            return fn(*args)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(torch._dynamo.config, "cache_size_limit", 1)
    monkeypatch.setattr(
        torch._dynamo.config,
        "accumulated_cache_size_limit",
        1,
    )
    execution = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0, 1),
    )
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        lambda *_args: (torch.zeros(1), torch.zeros(1)),
    )

    def invoke(
        authority: Any,
        geometry: int,
        _population: int,
    ) -> Any:
        shape = authority._geometry_shapes[geometry]
        return authority.transition(
            torch.zeros(1, 5, shape.mel_width),
            SimpleNamespace(
                channel=(torch.zeros(1, 2, 3),),
                time=(torch.zeros(1, 2, 3),),
                valid=torch.zeros(1, dtype=torch.long),
                left_context=56,
            ),
            torch.zeros(1, dtype=torch.long),
            torch.ones(1, dtype=torch.long),
            shape.out_width,
            torch.zeros(1, dtype=torch.long),
        )

    with pytest.raises(RuntimeError, match="synthetic specialization failure"):
        execution.warmup_domain(
            expected_cells=((0, 1), (1, 1)),
            invoke=lambda geometry, population: invoke(
                execution,
                geometry,
                population,
            ),
        )

    assert torch._dynamo.config.cache_size_limit == 1
    assert torch._dynamo.config.accumulated_cache_size_limit == 1
    assert not execution.ready
    assert execution.ready_receipt()["warmup_cells"] == []
    failed_calls = compile_instances[0]["calls"]

    with pytest.raises(ValueError, match=r"failed|discarded|invalid"):
        execution.warmup_domain(
            expected_cells=((0, 1), (1, 1)),
            invoke=lambda geometry, population: invoke(
                execution,
                geometry,
                population,
            ),
        )
    assert compile_instances[0]["calls"] == failed_calls

    fresh_execution = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0, 1),
    )
    fresh_execution.warmup_domain(
        expected_cells=((0, 1), (1, 1)),
        invoke=lambda geometry, population: invoke(
            fresh_execution,
            geometry,
            population,
        ),
    )
    assert fresh_execution.ready
    assert compile_instances[1]["calls"] == 2


@pytest.mark.cpu_test
@torch.inference_mode()
@pytest.mark.parametrize("arm", ["dense-graphed", "eager-graphed"])
def test_dense_graphed_replays_outputs_and_every_cache_family_with_independent_lifetime(
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
) -> None:
    # @spec PORT-PERF-011
    seen_cache_types: list[type[Any]] = []

    def transition(*args: Any) -> tuple[torch.Tensor, torch.Tensor]:
        seen_cache_types.append(type(args[2]))
        return _functional_graph_transition(*args)

    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        transition,
    )
    execution = build_encoder_execution(
        _compiled_core(),
        _dense_graphed_config(arm),
        maximum_population=2,
        warmup_geometries=(0,),
        vllm_config=object(),
        graph_runtime=_graph_runtime(),
    )

    def invoke(_geometry: int, population: int) -> Any:
        return execution.transition(
            *_graph_transition_args(population=population),
        )

    execution.warmup_domain(
        expected_cells=((0, 1), (0, 2)),
        invoke=invoke,
    )
    assert execution.ready
    assert set(seen_cache_types) == {_GraphCaches}

    first = _graph_transition_args(population=2, base=7.0)
    expected_first = _graph_transition_args(population=2, base=7.0)
    expected_first_outputs = _functional_graph_transition(
        _compiled_core(),
        *expected_first,
    )
    first_outputs = execution.transition(*first)
    retained = tuple(tensor.clone() for tensor in first_outputs)
    assert all(
        torch.equal(left, right)
        for left, right in zip(first_outputs, expected_first_outputs)
    )
    first_storage = first[1].graph_storage()
    expected_first_storage = expected_first[1].graph_storage()
    for family in ("channel", "time", "valid"):
        assert all(
            torch.equal(left, right)
            for left, right in zip(
                getattr(first_storage, family),
                getattr(expected_first_storage, family),
            )
        )

    second = _graph_transition_args(population=2, base=19.0)
    expected_second = _graph_transition_args(population=2, base=19.0)
    expected_second_outputs = _functional_graph_transition(
        _compiled_core(),
        *expected_second,
    )
    second_outputs = execution.transition(*second)
    assert all(
        torch.equal(left, right)
        for left, right in zip(second_outputs, expected_second_outputs)
    )
    assert all(torch.equal(left, right) for left, right in zip(first_outputs, retained))
    assert first[0].data_ptr() != second[0].data_ptr()
    assert first_outputs[0].data_ptr() != second_outputs[0].data_ptr()


@pytest.mark.cpu_test
@torch.inference_mode()
@pytest.mark.parametrize("arm", ["dense-graphed", "eager-graphed"])
def test_dense_graphed_rejects_unknown_signature_before_mutating_caller_state(
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
) -> None:
    # @spec PORT-PERF-011
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        _functional_graph_transition,
    )
    execution = build_encoder_execution(
        _compiled_core(),
        _dense_graphed_config(arm),
        maximum_population=1,
        warmup_geometries=(0,),
        vllm_config=object(),
        graph_runtime=_graph_runtime(),
    )
    execution.warmup_domain(
        expected_cells=((0, 1),),
        invoke=lambda _geometry, _population: execution.transition(
            *_graph_transition_args(population=1),
        ),
    )
    unknown = _graph_transition_args(population=1, mel_width=18, base=31.0)
    before = tuple(
        tensor.clone() for family in unknown[1].graph_storage() for tensor in family
    )

    with pytest.raises(ValueError, match=r"not (warmed|captured)"):
        execution.transition(*unknown)

    after = tuple(tensor for family in unknown[1].graph_storage() for tensor in family)
    assert all(torch.equal(left, right) for left, right in zip(before, after))
    execution.transition(*_graph_transition_args(population=1, base=37.0))
    assert execution.ready


@pytest.mark.cpu_test
@torch.inference_mode()
@pytest.mark.parametrize("arm", ["dense-graphed", "eager-graphed"])
def test_dense_graphed_incomplete_capture_discards_all_keys_and_fresh_start_recovers(
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
) -> None:
    # @spec PORT-PERF-011
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        _functional_graph_transition,
    )

    def build(runtime: GraphRuntime) -> Any:
        return build_encoder_execution(
            _compiled_core(),
            _dense_graphed_config(arm),
            maximum_population=2,
            warmup_geometries=(0,),
            vllm_config=object(),
            graph_runtime=runtime,
        )

    failed = build(_graph_runtime(fail_graph_call=4))
    with pytest.raises(RuntimeError, match="synthetic encoder graph failure"):
        failed.warmup_domain(
            expected_cells=((0, 1), (0, 2)),
            invoke=lambda _geometry, population: failed.transition(
                *_graph_transition_args(population=population),
            ),
        )
    assert not failed.ready
    assert failed.ready_receipt()["captured_keys"] == []
    with pytest.raises(ValueError, match=r"failed|discarded"):
        failed.transition(*_graph_transition_args(population=1))

    fresh = build(_graph_runtime())
    fresh.warmup_domain(
        expected_cells=((0, 1), (0, 2)),
        invoke=lambda _geometry, population: fresh.transition(
            *_graph_transition_args(population=population),
        ),
    )
    assert fresh.ready_receipt() == {
        "arm": arm,
        "ready": True,
        "warmup_cells": [[0, 1], [0, 2]],
        "warmup_geometries": [0],
        "warmup_populations": [1, 2],
        "captured_keys": [[0, 1], [0, 2]],
    }


@pytest.mark.cpu_test
@torch.inference_mode()
def test_dense_graphed_real_dynamo_reuses_cache_adapter_guards_for_capture_staging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009, PORT-PERF-010, PORT-PERF-011
    unique_graphs = 0

    def compile_with_counting_backend(fn: Any, **_kwargs: Any) -> Any:
        def backend(graph_module: Any, _example_inputs: Any) -> Any:
            nonlocal unique_graphs
            unique_graphs += 1
            return graph_module.forward

        # Match the production fullgraph contract. Permissive optimize() can
        # compile a cache property as a separate frame and count graph breaks
        # as if they were extra transition specializations.
        return torch._dynamo.optimize(backend, dynamic=False, nopython=True)(fn)

    monkeypatch.setattr(torch, "compile", compile_with_counting_backend)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        _functional_graph_transition,
    )
    execution = build_encoder_execution(
        _compiled_core(),
        _dense_graphed_config(),
        maximum_population=2,
        warmup_geometries=(0,),
        vllm_config=object(),
        graph_runtime=_graph_runtime(),
    )
    execution.warmup_domain(
        expected_cells=((0, 1), (0, 2)),
        invoke=lambda _geometry, population: execution.transition(
            *_graph_transition_args(population=population),
        ),
    )
    graphs_after_capture = unique_graphs
    execution.transition(*_graph_transition_args(population=1, base=41.0))
    execution.transition(*_graph_transition_args(population=2, base=43.0))

    assert graphs_after_capture == 2
    assert unique_graphs == graphs_after_capture


@pytest.mark.cpu_test
@torch.inference_mode()
@pytest.mark.parametrize("arm", ["dense-graphed", "eager-graphed"])
def test_dense_graphed_model_drift_during_capture_discards_readiness(
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
) -> None:
    # @spec PORT-PERF-010, PORT-PERF-011
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        _functional_graph_transition,
    )
    core = _compiled_core()
    runtime = replace(
        _graph_runtime(),
        synchronize=lambda _device: core.encoder.register_buffer(
            "unexpected", torch.zeros(1)
        ),
    )
    execution = build_encoder_execution(
        core,
        _dense_graphed_config(arm),
        maximum_population=1,
        warmup_geometries=(0,),
        vllm_config=object(),
        graph_runtime=runtime,
    )
    with pytest.raises(ValueError, match="unexpected"):
        execution.warmup_domain(
            expected_cells=((0, 1),),
            invoke=lambda _geometry, population: execution.transition(
                *_graph_transition_args(population=population)
            ),
        )
    assert not execution.ready
    assert execution.ready_receipt()["captured_keys"] == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA capture")
@torch.inference_mode()
@pytest.mark.parametrize("policy_name", ["fp32", "fp16-encoder"])
@pytest.mark.parametrize("arm", ["dense-graphed", "eager-graphed"])
def test_graphed_real_cuda_encoder_matches_state_and_retained_outputs(
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
    policy_name: str,
) -> None:
    """Graph replay equals same-policy eager/compiled execution on real CUDA.

    The oracle is exact under both policies: replay and the uncaptured
    transition run the same kernels under the same policy, and the only
    published tolerance (parity-tolerances.json) is the FP32 exact entry; no
    looser graph/eager tolerance exists for the encoder-only FP16 policy.

    @spec PORT-PERF-009, PORT-PERF-010, PORT-PERF-011
    @spec PORT-PREC-011, PORT-PREC-012, PORT-PREC-013
    """
    from vllm.config import VllmConfig
    from vllm.models.nemotron_asr.advance import _GatheredCaches
    from vllm.models.nemotron_asr.encoder import FastConformerEncoder
    from vllm.models.nemotron_asr.lid import PromptConditioner
    from vllm.models.nemotron_asr.nemotron_asr import apply_policy_dtypes
    from vllm.models.nemotron_asr.precision import (
        FP16_ENCODER_EXPERIMENT,
        PrecisionPolicy,
    )

    if arm == "eager-graphed":
        monkeypatch.setattr(
            torch,
            "compile",
            lambda *_args, **_kwargs: pytest.fail("native graph must not compile"),
        )
    torch.manual_seed(17)
    # This fixture installs the three real transition members immediately;
    # the outer resident/decoder model is intentionally absent.
    core = cast("NemotronASRCore", nn.Module())
    core.encoder = FastConformerEncoder(
        feat_in=16,
        d_model=32,
        d_ff=64,
        n_layers=2,
        n_heads=4,
        conv_kernel=5,
        subsampling_channels=16,
        att_context=(56, 1),
    )
    core.lid = PromptConditioner(enc_hidden=32, num_prompts=4)
    other_policy: PrecisionPolicy
    if policy_name == "fp32":
        core.policy = PrecisionPolicy({"*": "fp32"})
        other_policy = FP16_ENCODER_EXPERIMENT
    else:
        core.policy = FP16_ENCODER_EXPERIMENT
        other_policy = PrecisionPolicy({"*": "fp32"})
        # Realize dtypes as production does for any non-bring-up policy. The
        # decoder modules are absent from this fixture, so tensorless stand-ins
        # satisfy the realization without changing the transition members.
        core.predictor = nn.Module()
        core.joint = nn.Module()
        apply_policy_dtypes(core)
        assert {parameter.dtype for parameter in core.encoder.parameters()} == {
            torch.float16
        }
        assert {parameter.dtype for parameter in core.lid.parameters()} == {
            torch.float32
        }
    assert core.policy.content_hash != other_policy.content_hash
    core.cuda().eval()
    device = torch.device("cuda", torch.accelerator.current_device_index())

    @contextmanager
    def capture_stream(_device: torch.device) -> Any:
        # Exercise the actual platform wrapper without requiring a distributed
        # worker process for the tiny encoder discriminator.
        current = torch.cuda.current_stream(device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            yield
        current.wait_stream(stream)

    runtime = replace(platform_graph_runtime(), capture_context=capture_stream)
    # Dynamo keys its specializations on the transition's code object, which
    # every build in this process shares. The production budget covers one
    # runtime's single immutable policy (PORT-PREC-013); without a reset, an
    # earlier parametrization's other-policy entries count against this
    # policy's exact 20-cell domain and the dense arm hits the recompile limit.
    torch._dynamo.reset()
    execution = build_encoder_execution(
        core,
        _dense_graphed_config(arm),
        maximum_population=4,
        warmup_geometries=(0, 1, 2, 3, 4),
        vllm_config=VllmConfig(),
        graph_runtime=runtime,
    )

    def args(
        geometry: int, population: int, *, base: float = 0.125, variant: int = 0
    ) -> tuple[Any, ...]:
        shape = encoder_execution_module.encoder_geometry_shape(core, geometry)
        rows = torch.arange(population, device=device)
        # _GatheredCaches reads exactly these three pool families.
        caches = _GatheredCaches(
            cast(
                "SessionStateBatch",
                SimpleNamespace(
                    channel=[
                        torch.full(
                            (population, 56, 32), base + layer * 0.01, device=device
                        )
                        for layer in range(2)
                    ],
                    time=[
                        torch.full(
                            (population, 32, 4), base + layer * 0.02, device=device
                        )
                        for layer in range(2)
                    ],
                    window_valid=[
                        (
                            torch.full_like(rows, 56)
                            if variant == 2
                            else (rows + variant) % 5
                        )
                        .to(torch.int32)
                        .reshape(-1, 1)
                        for _ in range(2)
                    ],
                ),
            )
        )
        offsets = torch.zeros_like(rows) if variant == 2 else (rows + variant) % 3
        lengths = torch.where(
            ((rows + variant) % 2 == 0) & (variant != 2),
            torch.zeros_like(rows),
            torch.full_like(rows, shape.out_width) - offsets,
        )
        # The production tuple carries each row's valid mel width: partial
        # widths exercise the captured, length-masked subsampling.
        mel_lengths = (
            torch.full_like(rows, shape.mel_width)
            if variant == 2
            else shape.mel_width - (rows + variant) % 8
        )
        return (
            torch.full((population, 16, shape.mel_width), base, device=device),
            caches,
            offsets,
            lengths,
            shape.out_width,
            (rows + variant) % 4,
            mel_lengths,
        )

    capture_graph_counts: list[int] = []
    original_capture = execution._capture_graph_domain

    def capture_sealed(**kwargs: Any) -> None:
        if arm == "dense-graphed":
            assert torch._dynamo.config.error_on_recompile
        capture_graph_counts.append(
            torch._dynamo.utils.counters["stats"]["unique_graphs"]
        )
        original_capture(**kwargs)
        capture_graph_counts.append(
            torch._dynamo.utils.counters["stats"]["unique_graphs"]
        )

    monkeypatch.setattr(execution, "_capture_graph_domain", capture_sealed)
    cells = tuple(
        (geometry, population) for geometry in range(5) for population in range(1, 5)
    )
    execution.warmup_domain(
        expected_cells=cells,
        invoke=lambda geometry, population: execution.transition(
            *args(geometry, population)
        ),
    )
    assert execution.ready
    assert capture_graph_counts[0] == capture_graph_counts[1]
    assert bool(core.encoder._stream_relative_position_lengths) == (
        arm == "dense-graphed"
    )
    assert execution.ready_receipt()["captured_keys"] == [list(cell) for cell in cells]
    # PORT-PREC-013: the captured domain is bound to this policy's identity.
    assert execution._model_state is not None
    assert execution._model_state.precision_policy == core.policy.content_hash
    if policy_name == "fp16-encoder":
        assert (
            execution.ready_receipt()["precision_policy_hash"]
            == FP16_ENCODER_EXPERIMENT.content_hash
        )
    else:
        assert "precision_policy_hash" not in execution.ready_receipt()
    # PORT-PREC-011/012: captured state buffers and downstream frames are FP32.
    for entry in execution._graph_entries.values():
        assert {
            t.dtype
            for family in entry.cache_storage()
            for t in family
            if t.is_floating_point()
        } == {torch.float32}
        assert entry.raw.dtype == torch.float32
        assert entry.conditioned.dtype == torch.float32
    retained: list[tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]] = []
    compiled = execution._compiled_transition
    assert compiled is not None
    with torch._dynamo.config.patch(error_on_recompile=True):
        # Revisiting every cell after every capture also catches cross-entry
        # reuse of pooled outputs and intermediates.
        for variant in (0, 1, 2):
            order = (
                (*cells[::2], *reversed(cells[1::2]))
                if variant == 1
                else tuple(reversed(cells))
            )
            for geometry, population in order:
                actual_args = args(
                    geometry, population, base=0.25 + variant, variant=variant
                )
                expected_args = args(
                    geometry, population, base=0.25 + variant, variant=variant
                )
                before = tuple(
                    t.clone()
                    for family in actual_args[1].graph_storage()
                    for t in family
                )
                expected_outputs = compiled(*expected_args)
                actual_outputs = execution.transition(*actual_args)
                for actual, expected in zip(
                    actual_outputs, expected_outputs, strict=True
                ):
                    assert actual.dtype == torch.float32
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual_state = tuple(
                    t for family in actual_args[1].graph_storage() for t in family
                )
                expected_state = tuple(
                    t for family in expected_args[1].graph_storage() for t in family
                )
                zero_rows = actual_args[3] == 0
                for actual, expected, original in zip(
                    actual_state, expected_state, before, strict=True
                ):
                    assert actual.dtype == original.dtype
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(
                        actual[zero_rows], original[zero_rows], rtol=0, atol=0
                    )
                for output in actual_outputs:
                    assert torch.count_nonzero(output[zero_rows]) == 0
                for old_outputs, old_values in retained:
                    for old_output, old_value in zip(
                        old_outputs, old_values, strict=True
                    ):
                        torch.testing.assert_close(
                            old_output, old_value, rtol=0, atol=0
                        )
                retained.append(
                    (actual_outputs, tuple(t.clone() for t in actual_outputs))
                )
    stages = [item["stage"] for item in execution.ready_receipt()["memory_diagnostics"]]
    assert stages == [
        "before-staging",
        "after-staging",
        *(["after-capture"] * len(cells)),
        "after-all-captures",
    ]
    # PORT-PREC-013: a different policy identity never reuses this captured
    # domain; the check precedes any replay and discards the domain.
    core.policy = other_policy
    with pytest.raises(ValueError, match="precision_policy"):
        execution.profile_ready_cell(
            geometry=0,
            population=1,
            invoke=lambda: pytest.fail("cross-policy replay must not execute"),
        )
    assert not execution.ready


@pytest.mark.cpu_test
@torch.inference_mode()
def test_dense_graphed_review_seeds_discriminate_rows_positions_and_full_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-011
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        _functional_graph_transition,
    )
    execution = build_encoder_execution(
        _compiled_core(),
        _dense_graphed_config(),
        maximum_population=2,
        warmup_geometries=(0,),
        vllm_config=object(),
        graph_runtime=_graph_runtime(),
    )
    execution.warmup_domain(
        expected_cells=((0, 1), (0, 2)),
        invoke=lambda _geometry, population: execution.transition(
            *_graph_transition_args(population=population)
        ),
    )
    for entry in execution._graph_entries.values():
        execution._seed_graph_entry(entry, variant="full")
        assert torch.all(entry.out_lengths > 0)
        for tensor in (
            entry.mel,
            *entry.cache_storage().channel,
            *entry.cache_storage().time,
        ):
            assert tensor.flatten()[0] != tensor.flatten()[1]
            if tensor.shape[0] > 1:
                assert not torch.equal(tensor[0], tensor[1])
        for valid in entry.cache_storage().valid:
            assert torch.all(valid == execution._history_frames)
        execution._seed_graph_entry(entry, variant="mixed")
        zero = entry.out_lengths == 0
        assert zero.any()
        assert torch.all(entry.out_offsets[zero] > 0)
        if entry.mel.shape[0] > 1:
            assert torch.any(entry.out_lengths > 0)


@pytest.mark.cpu_test
@torch.inference_mode()
@pytest.mark.parametrize("arm", ["dense-graphed", "eager-graphed"])
def test_dense_graphed_review_population_one_rejects_noop_graph_replay(
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
) -> None:
    # @spec PORT-PERF-011
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        _functional_graph_transition,
    )
    runtime = _graph_runtime()
    original_factory = runtime.wrapper_factory

    def no_replay_factory(*args: Any, **kwargs: Any) -> Any:
        wrapper = original_factory(*args, **kwargs)
        return (
            lambda: wrapper.output
            if wrapper.graph_mode() and wrapper.captured
            else wrapper()
        )

    execution = build_encoder_execution(
        _compiled_core(),
        _dense_graphed_config(arm),
        maximum_population=1,
        warmup_geometries=(0,),
        vllm_config=object(),
        graph_runtime=replace(runtime, wrapper_factory=no_replay_factory),
    )
    with pytest.raises(RuntimeError, match="capture/replay differs"):
        execution.warmup_domain(
            expected_cells=((0, 1),),
            invoke=lambda _geometry, population: execution.transition(
                *_graph_transition_args(population=population)
            ),
        )
    assert not execution.ready
    assert execution.ready_receipt()["captured_keys"] == []


@pytest.mark.cpu_test
@torch.inference_mode()
def test_dense_graphed_review_rejects_postseal_dynamo_recompile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-009, PORT-PERF-011
    core = _compiled_core()
    core.encoder.guard_bias = 1.0
    compile_count = 0

    def guarded(core: Any, *args: Any) -> tuple[torch.Tensor, torch.Tensor]:
        raw, conditioned = _functional_graph_transition(core, *args)
        return raw + core.encoder.guard_bias, conditioned

    def compile_counting(fn: Any, **_kwargs: Any) -> Any:
        def backend(graph: Any, _inputs: Any) -> Any:
            nonlocal compile_count
            compile_count += 1
            return graph.forward

        return torch._dynamo.optimize(backend, nopython=True, dynamic=False)(fn)

    monkeypatch.setattr(torch, "compile", compile_counting)
    monkeypatch.setattr(encoder_execution_module, "execute_encoder_transition", guarded)
    execution = build_encoder_execution(
        core,
        _dense_graphed_config(),
        maximum_population=1,
        warmup_geometries=(0,),
        vllm_config=object(),
        graph_runtime=_graph_runtime(),
    )
    original_new = execution._new_graph_entry
    prior_error_on_recompile = torch._dynamo.config.error_on_recompile

    def guard_drift(**kwargs: Any) -> Any:
        assert execution._sealed
        core.encoder.guard_bias = 2.0
        return original_new(**kwargs)

    monkeypatch.setattr(execution, "_new_graph_entry", guard_drift)
    with pytest.raises(torch._dynamo.exc.RecompileError):
        execution.warmup_domain(
            expected_cells=((0, 1),),
            invoke=lambda _geometry, population: execution.transition(
                *_graph_transition_args(population=population)
            ),
        )
    assert compile_count == 1
    assert torch._dynamo.config.error_on_recompile == prior_error_on_recompile
    assert not execution.ready
    assert execution.ready_receipt()["captured_keys"] == []


@pytest.mark.cpu_test
@pytest.mark.parametrize("snapshot_fails", [False, True])
@torch.inference_mode()
@pytest.mark.parametrize("arm", ["dense-graphed", "eager-graphed"])
def test_dense_graphed_review_oom_emits_memory_diagnostics_preserving_cause(
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
    caplog: pytest.LogCaptureFixture,
    snapshot_fails: bool,
) -> None:
    # @spec PORT-PERF-011
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        _functional_graph_transition,
    )

    def snapshot(_device: Any, *, stage: str, key: Any = None) -> dict[str, Any]:
        if snapshot_fails and stage.startswith("failure-"):
            raise RuntimeError("diagnostic unavailable")
        return {
            "stage": stage,
            "key": list(key) if key is not None else None,
            "free_bytes": 100,
            "total_bytes": 200,
            "allocated_bytes": 50,
            "reserved_bytes": 100,
        }

    monkeypatch.setattr(encoder_execution_module, "_cuda_memory_snapshot", snapshot)
    original = torch.cuda.OutOfMemoryError("injected capture OOM")
    execution = build_encoder_execution(
        _compiled_core(),
        _dense_graphed_config(arm),
        maximum_population=1,
        warmup_geometries=(0,),
        vllm_config=object(),
        graph_runtime=_graph_runtime(fail_graph_call=2, graph_error=original),
    )
    with pytest.raises(torch.cuda.OutOfMemoryError) as caught:
        execution.warmup_domain(
            expected_cells=((0, 1),),
            invoke=lambda _geometry, population: execution.transition(
                *_graph_transition_args(population=population)
            ),
        )
    assert caught.value is original
    assert not execution.ready
    assert execution.ready_receipt()["captured_keys"] == []
    assert "memory_diagnostics" in caplog.text
    assert "before-staging" in caplog.text and "after-staging" in caplog.text
    assert "failure-capture" in caplog.text and '"key": [0, 1]' in caplog.text
    if snapshot_fails:
        assert "diagnostic unavailable" in caplog.text


@pytest.mark.cpu_test
@torch.inference_mode()
@pytest.mark.parametrize("arm", ["dense-graphed", "eager-graphed"])
def test_dense_graphed_review_staging_failure_discards_partial_storage_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
) -> None:
    # @spec PORT-PERF-011
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        _functional_graph_transition,
    )
    original_empty_like = _GraphCaches.empty_like

    def wrong_layout(caches: _GraphCaches) -> _GraphCaches:
        clone = original_empty_like(caches)
        if caches.channel[0].shape[0] == 2:
            clone.channel = tuple(t.transpose(1, 2) for t in clone.channel)
        return clone

    def build() -> Any:
        return build_encoder_execution(
            _compiled_core(),
            _dense_graphed_config(arm),
            maximum_population=2,
            warmup_geometries=(0,),
            vllm_config=object(),
            graph_runtime=_graph_runtime(),
        )

    def warm(execution: Any) -> None:
        execution.warmup_domain(
            expected_cells=((0, 1), (0, 2)),
            invoke=lambda _geometry, population: execution.transition(
                *_graph_transition_args(population=population)
            ),
        )

    monkeypatch.setattr(_GraphCaches, "empty_like", wrong_layout)
    failed = build()
    with pytest.raises(ValueError, match="tensor layout"):
        warm(failed)
    assert not failed.ready
    assert failed._pending_graph_entries == {}
    assert failed.ready_receipt()["captured_keys"] == []
    monkeypatch.setattr(_GraphCaches, "empty_like", original_empty_like)
    fresh = build()
    warm(fresh)
    assert fresh.ready


@pytest.mark.cpu_test
@pytest.mark.parametrize("num_prompts", [None, 0])
def test_dense_graphed_review_requires_valid_prompt_count(
    num_prompts: int | None,
) -> None:
    # @spec PORT-PERF-011
    core = _compiled_core()
    if num_prompts is None:
        del core.lid.num_prompts
    else:
        core.lid.num_prompts = num_prompts
    with pytest.raises(ValueError, match="num_prompts"):
        build_encoder_execution(
            core,
            _dense_graphed_config(),
            maximum_population=1,
            warmup_geometries=(0,),
            vllm_config=object(),
            graph_runtime=_graph_runtime(),
        )


@pytest.mark.cpu_test
@torch.inference_mode()
def test_dense_graphed_review_profiles_copy_and_replay_subranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-011
    from vllm.models.nemotron_asr import profiling

    expected = ("port.encode.stage_in", "port.encode.replay", "port.encode.stage_out")
    assert all(name in profiling._RANGES for name in expected)
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        _functional_graph_transition,
    )
    observed = []

    def observe_phase(name: str) -> nullcontext[None]:
        observed.append(name)
        return nullcontext()

    monkeypatch.setattr(encoder_execution_module, "phase", observe_phase)
    execution = build_encoder_execution(
        _compiled_core(),
        _dense_graphed_config(),
        maximum_population=1,
        warmup_geometries=(0,),
        vllm_config=object(),
        graph_runtime=_graph_runtime(),
    )
    execution.warmup_domain(
        expected_cells=((0, 1),),
        invoke=lambda _geometry, population: execution.transition(
            *_graph_transition_args(population=population)
        ),
    )
    observed.clear()
    execution.transition(*_graph_transition_args(population=1))
    assert observed == list(expected)


@pytest.mark.cpu_test
def test_ready_profile_preserves_seal_and_rejects_changed_signature(monkeypatch):
    """@spec PORT-PERF-009: final headroom profile reuses the sealed domain."""
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        lambda *_args: (torch.zeros(1), torch.zeros(1)),
    )
    execution = build_encoder_execution(
        _compiled_core(),
        _compiled_config(),
        maximum_population=1,
        warmup_geometries=(0,),
    )

    def invoke(*, changed=False):
        shape = execution._geometry_shapes[0]
        caches = SimpleNamespace(
            channel=(torch.zeros(1, 2, 3),),
            time=(torch.zeros(1, 2, 3),),
            valid=torch.zeros(1, dtype=torch.long),
            left_context=56,
        )
        return execution.transition(
            torch.zeros(
                1, 5, shape.mel_width, dtype=torch.float64 if changed else torch.float32
            ),
            caches,
            torch.zeros(1, dtype=torch.long),
            torch.ones(1, dtype=torch.long),
            shape.out_width,
            torch.zeros(1, dtype=torch.long),
        )

    with pytest.raises(ValueError, match="ready"):
        execution.profile_ready_cell(geometry=0, population=1, invoke=invoke)
    execution.profile_cell(geometry=0, population=1, invoke=invoke)
    execution.warmup_domain(expected_cells=((0, 1),), invoke=lambda *_: invoke())
    original_profile = execution._profile_signature
    original_signatures = execution._allowed_signatures
    execution.profile_ready_cell(geometry=0, population=1, invoke=invoke)
    assert execution.ready
    assert execution._profile_signature is original_profile
    assert execution._allowed_signatures is original_signatures
    with pytest.raises(ValueError, match="signature"):
        execution.profile_ready_cell(
            geometry=0, population=1, invoke=lambda: invoke(changed=True)
        )
    assert not execution.ready


@pytest.mark.cpu_test
@torch.inference_mode()
def test_eager_graphed_exact_domain_never_enters_compiler_or_prepares_projections(
    monkeypatch,
):
    def forbidden(*_args, **_kwargs):
        pytest.fail("native graph must not compile or prepare learned projections")

    monkeypatch.setattr(torch, "compile", forbidden)
    monkeypatch.setattr(
        encoder_execution_module, "_compiler_specialization_budget", forbidden
    )
    monkeypatch.setattr(torch._dynamo, "config", SimpleNamespace(patch=forbidden))
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        _functional_graph_transition,
    )
    core = _compiled_core()
    core.encoder.prepare_stream_relative_position_projections = forbidden
    core.encoder._stream_relative_position_lengths = ()
    execution = build_encoder_execution(
        core,
        _dense_graphed_config("eager-graphed"),
        maximum_population=3,
        warmup_geometries=(0,),
        vllm_config=object(),
        graph_runtime=_graph_runtime(),
    )
    assert execution.warmup_populations == (1, 2, 3)
    assert not execution.ready
    with pytest.raises(ValueError, match="authority"):
        execution.transition(*_graph_transition_args(population=3))
    execution.profile_cell(
        geometry=0,
        population=3,
        invoke=lambda: execution.transition(*_graph_transition_args(population=3)),
    )
    assert core.encoder.pos_enc.pe.shape == (1, 2 * execution.t_cap - 1, 8)
    execution.warmup_domain(
        expected_cells=((0, 1), (0, 2), (0, 3)),
        invoke=lambda _geometry, population: execution.transition(
            *_graph_transition_args(population=population)
        ),
    )
    execution.profile_ready_cell(
        geometry=0,
        population=3,
        invoke=lambda: execution.transition(*_graph_transition_args(population=3)),
    )
    assert execution.ready_receipt()["captured_keys"] == [[0, 1], [0, 2], [0, 3]]
    assert core.encoder._stream_relative_position_lengths == ()
    for population in (3, 1, 2, 3):
        execution.transition(*_graph_transition_args(population=population))
    with pytest.raises(ValueError, match="population"):
        execution.transition(*_graph_transition_args(population=4))


@pytest.mark.cpu_test
@pytest.mark.parametrize("bucketing", [True, 1, "false"])
def test_eager_graphed_rejects_bucketing_and_nonboolean_flags(bucketing):
    config = _dense_graphed_config("eager-graphed")
    config.encoder_population_bucketing = bucketing
    with pytest.raises(ValueError, match="encoder_population_bucketing"):
        build_encoder_execution(
            _compiled_core(),
            config,
            maximum_population=3,
            warmup_geometries=(0,),
            vllm_config=object(),
            graph_runtime=_graph_runtime(),
        )


@pytest.mark.cpu_test
@torch.inference_mode()
@pytest.mark.parametrize("prepared", ["lengths", "buffer"])
def test_eager_graphed_rejects_prepared_learned_projection_state(monkeypatch, prepared):
    monkeypatch.setattr(
        torch,
        "compile",
        lambda *_args, **_kwargs: pytest.fail("native graph must not compile"),
    )
    core = _compiled_core()
    if prepared == "lengths":
        core.encoder._stream_relative_position_lengths = (59,)
    else:
        core.encoder.register_buffer("_stream_relative_position_59", torch.zeros(1))
    with pytest.raises(ValueError, match="prepared.*projection"):
        execution = build_encoder_execution(
            core,
            _dense_graphed_config("eager-graphed"),
            maximum_population=1,
            warmup_geometries=(0,),
            vllm_config=object(),
            graph_runtime=_graph_runtime(),
        )
        execution.profile_cell(
            geometry=0,
            population=1,
            invoke=lambda: execution.transition(*_graph_transition_args(population=1)),
        )


@pytest.mark.cpu_test
@torch.inference_mode()
def test_eager_graphed_native_profile_matches_unprepared_encoder(monkeypatch):
    from vllm.models.nemotron_asr.encoder import FastConformerEncoder, StreamingCaches
    from vllm.models.nemotron_asr.lid import PromptConditioner

    monkeypatch.setattr(
        torch,
        "compile",
        lambda *_args, **_kwargs: pytest.fail("native graph must not compile"),
    )
    core = _compiled_core()
    core.encoder = FastConformerEncoder(
        feat_in=16,
        d_model=32,
        d_ff=64,
        n_layers=2,
        n_heads=4,
        conv_kernel=5,
        subsampling_channels=16,
        att_context=(56, 1),
    )
    core.lid = PromptConditioner(enc_hidden=32, num_prompts=4)
    core.eval()
    execution = build_encoder_execution(
        core,
        _dense_graphed_config("eager-graphed"),
        maximum_population=3,
        warmup_geometries=(1,),
        vllm_config=object(),
        graph_runtime=_graph_runtime(),
    )
    shape = execution._geometry_shapes[1]
    mel = torch.randn(3, 16, shape.mel_width)
    offsets = torch.tensor([0, 1, 2])
    lengths = torch.tensor([shape.out_width, 0, 1])
    prompts = torch.tensor([0, 2, 1])
    caches = [
        StreamingCaches(
            n_layers=2,
            batch=3,
            left_context=56,
            d_model=32,
            conv_kernel=5,
            device=torch.device("cpu"),
        )
        for _ in range(2)
    ]
    calls = []
    hooks = [
        layer.self_attn.linear_pos.register_forward_hook(lambda *_: calls.append(True))
        for layer in core.encoder.layers
    ]
    try:
        actual = execution.profile_cell(
            geometry=1,
            population=3,
            invoke=lambda: execution.transition(
                mel, caches[0], offsets, lengths, shape.out_width, prompts
            ),
        )
        assert len(calls) == 2
        expected = encoder_execution_module.execute_encoder_transition(
            core,
            mel,
            caches[1],
            offsets,
            lengths,
            shape.out_width,
            prompts,
        )
    finally:
        for hook in hooks:
            hook.remove()
    assert len(calls) == 4
    assert core.encoder._stream_relative_position_lengths == ()
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for family in ("channel", "time", "valid"):
        torch.testing.assert_close(
            getattr(caches[0], family), getattr(caches[1], family), rtol=0, atol=0
        )


@pytest.mark.cpu_test
@torch.inference_mode()
@pytest.mark.parametrize("population,tier", [(1, 1), (31, 32)])
def test_exact_chunk_inventory_replaces_encoder_cell_and_requires_atomic_publication(
    monkeypatch, population, tier
):
    import json

    from vllm.models.nemotron_asr.chunk_bucket_graph import ExactChunkGraphBinding

    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        _functional_graph_transition,
    )
    execution = build_encoder_execution(
        _compiled_core(),
        _dense_graphed_config("eager-graphed"),
        maximum_population=31,
        warmup_geometries=(1,),
        vllm_config=object(),
        graph_runtime=_graph_runtime(),
    )
    binding = ExactChunkGraphBinding(
        execution._core,
        object(),
        execution,
        SimpleNamespace(execution_tier=lambda _n: tier),
        [population],
    )
    shape = execution._geometry_shapes[1]

    def arguments(population):
        values = _graph_transition_args(
            population=population, mel_width=shape.mel_width, out_width=shape.out_width
        )
        values[-1].remainder_(4)
        return values

    execution.warmup_domain(
        expected_cells=tuple((1, n) for n in range(1, 32)),
        invoke=lambda _geometry, population: execution.transition(
            *arguments(population)
        ),
    )
    assert set(execution._graph_entries) == {
        (1, n) for n in range(1, 32) if n != population
    }
    assert not execution.ready
    assert not execution._pending_graph_entries
    with pytest.raises(ValueError, match="inventory"):
        execution.publish_chunk_graphs({})
    assert not execution.ready
    assert not execution._chunk_graph_entries

    def _replay():
        return None

    replacement = cast(Any, _replay)
    replacement.replay_count = 4
    execution.publish_chunk_graphs({(1, population): replacement})
    assert execution.ready
    receipt = execution.ready_receipt()
    assert receipt["captured_keys"] == [[1, n] for n in range(1, 32) if n != population]
    assert receipt["chunk_graph_keys"] == [[1, population]]
    assert not set(execution._graph_entries) & set(execution._chunk_graph_entries)
    assert set(execution._graph_entries) | set(execution._chunk_graph_entries) == {
        (1, n) for n in range(1, 32)
    }
    first = binding.receipt()
    second = binding.receipt()
    assert first == second == json.loads(json.dumps(first))
    assert replacement.replay_count == 4
    first["cells"][0]["successful_replays"] = 999
    first["source_sha256"].clear()
    assert binding.receipt() == second
    execution.transition(*arguments(30))
    # A reserved cell can never fall through to a second or lazily built encoder graph.
    with pytest.raises(ValueError, match="not captured"):
        execution.transition(*arguments(population))
    execution._discard()
    assert not execution.ready
    assert not execution._chunk_graph_entries and not execution._graph_entries


def _retained_storage_bytes(tensors: Any) -> int:
    storages = {
        tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
        for tensor in tensors
    }
    return sum(storages.values())


@pytest.mark.cpu_test
@torch.inference_mode()
@pytest.mark.parametrize("arm", ["dense-graphed", "eager-graphed"])
def test_graphed_staging_borrows_one_max_population_bank_per_geometry(
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
) -> None:
    # @spec PORT-PERF-011
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        _functional_graph_transition,
    )
    core = _compiled_core()
    execution = build_encoder_execution(
        core,
        _dense_graphed_config(arm),
        maximum_population=8,
        warmup_geometries=(0, 1),
        vllm_config=object(),
        graph_runtime=_graph_runtime(),
    )
    shapes = {
        geometry: encoder_execution_module.encoder_geometry_shape(core, geometry)
        for geometry in (0, 1)
    }
    cells = tuple(
        (geometry, population) for geometry in (0, 1) for population in range(1, 9)
    )
    execution.warmup_domain(
        expected_cells=cells,
        invoke=lambda geometry, population: execution.transition(
            *_graph_transition_args(
                population=population,
                mel_width=shapes[geometry].mel_width,
                out_width=shapes[geometry].out_width,
            )
        ),
    )
    assert execution.ready
    assert execution.ready_receipt()["captured_keys"] == [
        list(cell) for cell in sorted(cells)
    ]
    entries = execution._graph_entries
    row_bytes = 56 * 3 * 4
    for geometry in (0, 1):
        small, large = entries[(geometry, 1)], entries[(geometry, 8)]
        for small_tensor, large_tensor in zip(
            (
                small.mel,
                small.out_offsets,
                small.out_lengths,
                small.prompt_index,
                small.raw,
                small.conditioned,
            ),
            (
                large.mel,
                large.out_offsets,
                large.out_lengths,
                large.prompt_index,
                large.raw,
                large.conditioned,
            ),
            strict=True,
        ):
            assert (
                small_tensor.untyped_storage().data_ptr()
                == large_tensor.untyped_storage().data_ptr()
            )
            assert small_tensor.storage_offset() == 0 and small_tensor.is_contiguous()
            assert small_tensor.shape[0] == 1 and large_tensor.shape[0] == 8
        for small_family, large_family in zip(
            small.cache_storage(), large.cache_storage(), strict=True
        ):
            for small_tensor, large_tensor in zip(
                small_family, large_family, strict=True
            ):
                assert (
                    small_tensor.untyped_storage().data_ptr()
                    == large_tensor.untyped_storage().data_ptr()
                )
                assert (
                    small_tensor.storage_offset() == 0 and small_tensor.is_contiguous()
                )
        channel = [
            tensor
            for (g, _), entry in entries.items()
            if g == geometry
            for tensor in entry.cache_storage().channel
        ]
        # Two channel layers retain 8 rows each, not the 1+...+8 = 36-row triangle.
        assert _retained_storage_bytes(channel) == 2 * 8 * row_bytes
    # Geometries never share a bank.
    assert (
        entries[(0, 8)].mel.untyped_storage().data_ptr()
        != entries[(1, 8)].mel.untyped_storage().data_ptr()
    )


@pytest.mark.cpu_test
@torch.inference_mode()
@pytest.mark.parametrize("arm", ["dense-graphed", "eager-graphed"])
def test_graphed_shared_bank_replays_are_isolated_from_prefix_neighbours(
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
) -> None:
    # @spec PORT-PERF-011
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        _functional_graph_transition,
    )
    execution = build_encoder_execution(
        _compiled_core(),
        _dense_graphed_config(arm),
        maximum_population=4,
        warmup_geometries=(0,),
        vllm_config=object(),
        graph_runtime=_graph_runtime(),
    )
    execution.warmup_domain(
        expected_cells=tuple((0, population) for population in range(1, 5)),
        invoke=lambda _geometry, population: execution.transition(
            *_graph_transition_args(population=population)
        ),
    )
    bank_pointers = {
        tensor.untyped_storage().data_ptr()
        for entry in execution._graph_entries.values()
        for tensor in (
            entry.raw,
            entry.conditioned,
            entry.mel,
            *(t for family in entry.cache_storage() for t in family),
        )
    }

    def state(caches: _GraphCaches) -> tuple[torch.Tensor, ...]:
        return tuple(
            tensor.clone() for family in caches.graph_storage() for tensor in family
        )

    retained: list[tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]] = []
    # small -> clone -> large -> small again; every result equals a no-sharing oracle.
    for population, base in ((2, 3.0), (4, 11.0), (2, 23.0), (1, 31.0), (3, 37.0)):
        actual = _graph_transition_args(population=population, base=base)
        oracle = _graph_transition_args(population=population, base=base)
        expected_outputs = _functional_graph_transition(_compiled_core(), *oracle)
        outputs = execution.transition(*actual)
        assert all(
            torch.equal(left, right)
            for left, right in zip(outputs, expected_outputs, strict=True)
        )
        assert all(
            torch.equal(left, right)
            for left, right in zip(state(actual[1]), state(oracle[1]), strict=True)
        )
        # Callers receive clones, never bank views.
        assert (
            not {tensor.untyped_storage().data_ptr() for tensor in outputs}
            & bank_pointers
        )
        for old_outputs, old_values in retained:
            assert all(
                torch.equal(left, right)
                for left, right in zip(old_outputs, old_values, strict=True)
            )
        retained.append((outputs, tuple(tensor.clone() for tensor in outputs)))


def _mel_length_graph_transition(
    core: Any,
    mel: torch.Tensor,
    caches: _GraphCaches,
    out_offsets: torch.Tensor,
    out_lengths: torch.Tensor,
    out_width: int,
    prompt_index: torch.Tensor,
    mel_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw, conditioned = _functional_graph_transition(
        core, mel, caches, out_offsets, out_lengths, out_width, prompt_index
    )
    shift = mel_lengths.reshape(-1, 1, 1).to(raw.dtype)
    return raw + shift, conditioned + shift


@pytest.mark.cpu_test
@torch.inference_mode()
@pytest.mark.parametrize("arm", ["dense-graphed", "eager-graphed"])
def test_graphed_arms_stage_valid_mel_widths_as_a_replayed_input(
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
) -> None:
    # @spec PORT-PERF-010, PORT-PERF-011
    # The per-row valid mel width rides the graph as staged storage: a replay
    # with new widths must observe them (no captured constant), and a caller
    # that omits them cannot reach a domain warmed with them.
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    monkeypatch.setattr(
        encoder_execution_module,
        "execute_encoder_transition",
        _mel_length_graph_transition,
    )
    execution = build_encoder_execution(
        _compiled_core(),
        _dense_graphed_config(arm),
        maximum_population=2,
        warmup_geometries=(0,),
        vllm_config=object(),
        graph_runtime=_graph_runtime(),
    )
    populations = execution.warmup_populations

    def args(population: int, base: float, widths: list[int]) -> tuple[Any, ...]:
        return (
            *_graph_transition_args(population=population, base=base),
            torch.tensor(widths),
        )

    execution.warmup_domain(
        expected_cells=tuple((0, population) for population in populations),
        invoke=lambda _geometry, population: execution.transition(
            *args(population, 1.0, [17] * population)
        ),
    )
    assert execution.ready

    live = 2
    for widths in ([5, 11], [17, 2]):
        actual = execution.transition(*args(live, 7.0, widths))
        expected = _mel_length_graph_transition(
            _compiled_core(), *args(live, 7.0, widths)
        )
        assert all(
            torch.equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )

    with pytest.raises(ValueError, match="signature"):
        execution.transition(*_graph_transition_args(population=live, base=7.0))
