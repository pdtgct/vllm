# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whole-transaction graph replay against the eager transaction.

CPU tests drive the binding through a runtime whose "replay" re-executes the
captured body over the static arena, which checks the arena plumbing, the
staging guard, key classification and fallback accounting. CUDA tests use real
``torch.cuda.CUDAGraph`` replay, where every host value outside the arena would
be frozen at capture.
"""

from __future__ import annotations

import gc
import os
import sys
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tests.models.nemotron_asr import test_encoder_execution as encoder_tests
from tests.models.nemotron_asr.test_advance_model_rows_local import (
    _BOOK,
    CAP,
    CARRIER_HIDDEN,
    D_MODEL,
    FEAT,
    FINAL_SAMPLES,
    GEOM_FINAL,
    GEOM_REG,
    KERNEL,
    N_LAYERS,
    PARK_ID,
    PLACEHOLDER_ID,
    REG_SAMPLES,
    VOCAB,
    WINDOW,
    _assert_pools_equal,
    _clone_pools,
    _CommitRecorder,
    _envelope,
    _fixed_resolver,
    _fresh_pools,
    _plan,
    _tiny_core,
    advance,
    rnnt,
)
from tests.models.nemotron_asr.test_advance_session_local import _fresh_state
from tests.models.nemotron_asr.test_decode_graph_binding import (
    _decode_calls,
)
from tests.models.nemotron_asr.test_decode_graph_binding import (
    _runtime as _decode_runtime,
)
from tests.models.nemotron_asr.test_plan_local import _bind, _feature, _row
from vllm.config.compilation import CUDAGraphMode
from vllm.models.nemotron_asr import profile_execution
from vllm.models.nemotron_asr import transaction_graph as tg
from vllm.models.nemotron_asr.batch_invariance import execution_mode
from vllm.models.nemotron_asr.capture_scope import (
    nested_graph_mode,
    outer_transaction_capture,
)
from vllm.models.nemotron_asr.chunk_bucket_graph import (
    ExactChunkGraphBinding,
    capture_chunk_bucket,
)
from vllm.models.nemotron_asr.commit_sink import (
    BoundedCommitSink,
    resolve_status_reports,
)
from vllm.models.nemotron_asr.decode_graph import DenseGraphBinding, GraphRuntime
from vllm.models.nemotron_asr.model_state_v2 import NemotronASRModelState
from vllm.models.nemotron_asr.plan import SessionRegistry
from vllm.models.nemotron_asr.state_profile import (
    allocate_state_scratch,
    build_nemotron_persistent_state_spec,
    project_nemotron_state_pools,
)
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, StateSlotHandle
from vllm.v1.worker.gpu.state_slots import StateSlotBindings, StateSlotViews

pytestmark = [pytest.mark.core_model]

EOU = VOCAB + 3
CPU = torch.device("cpu")
Pools = dict[str, Any]


# ---- runtimes --------------------------------------------------------------


class _GraphEntries:
    """Graph entries keyed by batch descriptor, as the platform wrapper keeps.

    ``concrete_cudagraph_entries`` maps a descriptor to an entry holding its
    ``cudagraph``; a graph-mode call whose descriptor has no graph captures
    and publishes one. ``clear_graphs()`` empties a wrapper's entries and
    ``_clear_all_graphs(mode)`` every live wrapper's of that runtime, as
    ``CUDAGraphWrapper.clear_all_graphs()`` does.
    """

    def __init__(self, mode: dict[str, Any]) -> None:
        self._mode = mode
        self.concrete_cudagraph_entries: dict[Any, SimpleNamespace] = {}
        mode.setdefault("wrappers", weakref.WeakSet()).add(self)

    def graph(self) -> Any:
        """The current descriptor's graph; ``None`` means this call captures."""
        entry = self.concrete_cudagraph_entries.get(self._mode["descriptor"])
        return None if entry is None else entry.cudagraph

    def publish(self, graph: Any) -> None:
        descriptor = self._mode["descriptor"]
        self.concrete_cudagraph_entries[descriptor] = SimpleNamespace(cudagraph=graph)

    def clear_graphs(self) -> None:
        self.concrete_cudagraph_entries.clear()


def _clear_all_graphs(mode: dict[str, Any]) -> None:
    """Empty every live wrapper that ``mode``'s runtime created."""
    for wrapper in list(mode.get("wrappers", ())):
        wrapper.clear_graphs()


def _runtime(wrapper_factory: Callable[..., Any], mode: dict[str, Any]) -> GraphRuntime:
    """A runtime whose forward context sets ``mode``'s value and descriptor."""

    def context(
        _metadata: object,
        _config: object,
        *,
        cudagraph_runtime_mode: str,
        batch_descriptor: object,
    ) -> Any:
        @contextmanager
        def scope() -> Iterator[None]:
            previous = mode["value"], mode["descriptor"]
            mode["value"], mode["descriptor"] = cudagraph_runtime_mode, batch_descriptor
            try:
                yield
            finally:
                mode["value"], mode["descriptor"] = previous

        return scope()

    return GraphRuntime(
        wrapper_factory=wrapper_factory,
        forward_context=context,
        capture_context=lambda _device: nullcontext(),
        descriptor_factory=lambda population: population,
        eager_mode="none",
        graph_mode="graph",
        synchronize=lambda device: (
            torch.cuda.synchronize(device) if device.type == "cuda" else None
        ),
        set_capture_enabled=lambda _enabled: None,
    )


def _rerun_runtime(
    on_capture: Callable[[int], None] | None = None,
) -> GraphRuntime:
    """Replay re-executes the body; used to test plumbing on CPU.

    ``on_capture`` receives the creation index of the wrapper whose
    graph-mode call is capturing; raising from it injects a capture failure.
    """
    mode: dict[str, Any] = {"value": "none", "descriptor": None}
    created = [0]

    class Wrapper(_GraphEntries):
        def __init__(self, runnable: Callable[[], Any], *_: Any, **__: Any):
            super().__init__(mode)
            self.runnable = runnable
            self.index = created[0]
            created[0] += 1

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            if mode["value"] == "graph" and self.graph() is None:
                if on_capture is not None:
                    on_capture(self.index)
                self.publish(object())
            return self.runnable(*args, **kwargs)

    return _runtime(Wrapper, mode)


def _cuda_runtime() -> GraphRuntime:
    """Capture on a descriptor's first graph-mode call, then replay its graph.

    Every cell's graph is captured into one shared memory pool, as the
    platform wrapper does with its global pool, unless the caller replaces a
    wrapper's ``graph_pool`` before its capture, as it may on the platform
    wrapper.
    """
    mode: dict[str, Any] = {"value": "none", "descriptor": None}
    pool = torch.cuda.graph_pool_handle()

    class Wrapper(_GraphEntries):
        def __init__(self, runnable: Callable[[], Any], *_: Any, **__: Any):
            super().__init__(mode)
            self.runnable = runnable
            self.graph_pool = pool
            self.output: Any = None

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            if mode["value"] != "graph":
                return self.runnable(*args, **kwargs)
            graph = self.graph()
            if graph is None:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self.graph_pool):
                    self.output = self.runnable(*args, **kwargs)
                self.publish(graph)
                return self.output
            graph.replay()
            return self.output

    return _runtime(Wrapper, mode)


def _logging_runtime(
    log: list[tuple[int, str]], corrupt: Callable[[int], bool] | None = None
) -> GraphRuntime:
    """The re-executing runtime, logging ``(wrapper index, mode)`` per call.

    When ``corrupt(index)`` holds, wrapper ``index``'s graph-mode calls after
    its capture return projection rows that differ from the eager body's.
    """
    mode: dict[str, Any] = {"value": "none", "descriptor": None}
    created = [0]

    class Wrapper(_GraphEntries):
        def __init__(self, runnable: Callable[[], Any], *_: Any, **__: Any):
            super().__init__(mode)
            self.runnable = runnable
            self.index = created[0]
            created[0] += 1

        def __call__(self) -> Any:
            log.append((self.index, mode["value"]))
            output = self.runnable()
            if mode["value"] == "graph":
                if self.graph() is None:
                    self.publish(object())
                elif corrupt is not None and corrupt(self.index):
                    output[0].add_(1)
            return output

    return _runtime(Wrapper, mode)


def _entry_runtime(captures: list[Any]) -> SimpleNamespace:
    """The re-executing runtime, recording every capture's batch descriptor.

    ``clear_all_graphs()`` empties every wrapper, as
    ``CUDAGraphWrapper.clear_all_graphs()`` does.
    """
    mode: dict[str, Any] = {"value": "none", "descriptor": None}

    class Wrapper(_GraphEntries):
        def __init__(self, runnable: Callable[[], Any], *_: Any, **__: Any):
            super().__init__(mode)
            self.runnable = runnable

        def __call__(self) -> Any:
            if mode["value"] == "graph" and self.graph() is None:
                captures.append(mode["descriptor"])
                self.publish(object())
            return self.runnable()

    return SimpleNamespace(
        runtime=_runtime(Wrapper, mode),
        clear_all_graphs=lambda: _clear_all_graphs(mode),
    )


def _recording(runtime: GraphRuntime, modes: list[Any]) -> GraphRuntime:
    """``runtime`` with every launch's graph mode appended to ``modes``."""
    forward = runtime.forward_context

    def context(*args: Any, cudagraph_runtime_mode: Any, **kwargs: Any) -> Any:
        modes.append(cudagraph_runtime_mode)
        return forward(*args, cudagraph_runtime_mode=cudagraph_runtime_mode, **kwargs)

    return replace(runtime, forward_context=context)


# ---- fixtures --------------------------------------------------------------


def _endpoint_pools(
    num_blocks: int, device: torch.device = CPU, *, endpoint: bool = True
) -> Pools:
    """Resident pools; ``endpoint`` adds the endpoint history and book pools."""
    pools = _fresh_pools(num_blocks)
    if endpoint:
        pools["endpoint_history_pool"] = torch.zeros(num_blocks, 8, dtype=torch.int32)
        pools["endpoint_book_pool"] = torch.zeros(num_blocks, 6, dtype=torch.int32)
    return {
        key: [tensor.to(device) for tensor in value]
        if isinstance(value, list)
        else value.to(device)
        for key, value in pools.items()
    }


def _transaction_pools(pools: Pools) -> advance.TransactionPools:
    return advance.TransactionPools(
        channel=pools["channel_pools"],
        time=pools["time_pools"],
        valid_length=pools["len_pools"],
        h=pools["h_pool"],
        c=pools["c_pool"],
        queue=pools["queue_pool"],
        book=pools["book_pool"],
        frontend_raw=pools["frontend_raw_pool"],
        frontend_mel=pools["frontend_mel_pool"],
        frontend_counter=pools["frontend_counter_pool"],
        endpoint_history=pools.get("endpoint_history_pool"),
        endpoint_book=pools.get("endpoint_book_pool"),
    )


def _core(device: torch.device = CPU) -> Any:
    core = _tiny_core(seed=1)
    for name in ("encoder", "lid", "predictor", "joint", "featurizer"):
        getattr(core, name).to(device)
    return core


def _call_bindings(core: Any, *, endpoint: bool = True) -> tg.TransactionCallBindings:
    """Call bindings; without ``endpoint`` the EOU control id is absent."""
    return tg.TransactionCallBindings(
        adapter=advance.make_mrv1_adapter(
            hidden_size=CARRIER_HIDDEN, park_id=PARK_ID, blank_id=core.blank_id
        ),
        decode_resolver=_fixed_resolver(rnnt.decode_dense_masked_frames),
        encoder_transition=None,
        bucket_transition=None,
        placeholder_id=PLACEHOLDER_ID,
        park_id=PARK_ID,
        eou_token_id=EOU if endpoint else None,
        graph_covers_decode=False,
    )


def _binding(
    core: Any,
    pools: Pools,
    call: tg.TransactionCallBindings,
    runtime: GraphRuntime,
    *,
    populations: tuple[int, ...] = (1, 2),
    device: torch.device = CPU,
) -> tg.TransactionGraphBinding:
    binding = tg.TransactionGraphBinding(
        populations=list(populations),
        geometries=(GEOM_REG,),
        maximum_population=4,
        vllm_config=object(),
        runtime=runtime,
    )
    if device.type == "cuda":
        advance.warmup_advance_model_rows_scatter(**pools)
    binding.capture(
        core,
        pools=_transaction_pools(pools),
        call=call,
        hidden_size=CARRIER_HIDDEN,
        device=device,
    )
    return binding


# ---- model, stand-in storage and memory-reading fixtures -------------------

#: Marks an ``hf_config`` field that the served configuration omits.
_ABSENT = object()

#: Receipt fields the admission check reports (Observability).
_ADMISSION_FIELDS = (
    "admission_bytes",
    "serving_bytes",
    "admission_release_verified",
    "admission_capture_seconds",
    "serving_capture_seconds",
)


def _seam(owner: Any, name: str) -> Any:
    """Return the production seam ``name``; a missing seam fails the test."""
    assert hasattr(owner, name), f"production seam {name!r} is missing"
    return getattr(owner, name)


def _model_config(**overrides: Any) -> Any:
    """A served configuration whose state layout matches the tiny core."""
    from vllm.models.nemotron_asr.configuration_nemotron_asr import (
        NemotronASRConfig,
    )

    fields: dict[str, Any] = {
        "n_layers": N_LAYERS,
        "d_model": D_MODEL,
        "pred_hidden": 16,
        "pred_rnn_layers": 2,
        "conv_kernel": KERNEL,
        "att_context_left": WINDOW,
        "att_context_right": 1,
        "num_asr_labels": VOCAB,
        "n_mels": FEAT,
        "num_prompts": 4,
        "vocab_size": PLACEHOLDER_ID + 8,
        "eos_token_id": PARK_ID,
        "audio_chunk_token_id": PLACEHOLDER_ID,
        "eou_token_id": EOU,
        "flush_token_id": PLACEHOLDER_ID + 2,
        "prompt_dictionary": {"en-US": 0},
        "hidden_size": CARRIER_HIDDEN,
        "decode_dispatch_arm": "dense-eager",
        "experimental_native_burst": False,
        "supported_num_lookahead_tokens": [0],
        "transaction_graph_populations": [1, 2],
    }
    fields.update(overrides)
    return NemotronASRConfig(
        **{key: value for key, value in fields.items() if value is not _ABSENT}
    )


def _engine_config(
    hf_config: Any,
    *,
    max_num_seqs: int = 2,
    max_num_batched_tokens: int = 8,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    cudagraph_mm_encoder: bool = False,
    num_gpu_blocks_override: int | None = None,
    kv_cache_memory_bytes: int | None = None,
) -> SimpleNamespace:
    """The engine configuration fields the model and its state consumer read."""
    return SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(
            hf_config=hf_config,
            dtype=torch.float32,
            max_model_len=143,
            logits_processors=None,
            get_inputs_embeds_size=lambda: CARRIER_HIDDEN,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            async_scheduling=False,
        ),
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=num_gpu_blocks_override,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
        ),
        compilation_config=SimpleNamespace(
            static_forward_context={},
            cudagraph_mode=cudagraph_mode,
            cudagraph_mm_encoder=cudagraph_mm_encoder,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_expert_parallel=False,
        ),
        speculative_config=None,
        attention_config=SimpleNamespace(hisparse_config=None),
    )


def _model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    runtime: Callable[[], GraphRuntime] = _rerun_runtime,
    device: torch.device = CPU,
    engine: dict[str, Any] | None = None,
    **hf: Any,
) -> tuple[Any, SimpleNamespace]:
    """Construct the served model around the tiny core.

    ``runtime`` builds the outer transaction runtime; ``engine`` overrides
    engine fields and ``hf`` the served configuration.
    """
    from vllm.models.nemotron_asr import nemotron_asr

    def make_core(**kwargs: Any) -> torch.nn.Module:
        core = torch.nn.Module()
        for name, value in vars(_core(device)).items():
            setattr(core, name, value)
        core.policy = kwargs["policy"]
        # The value an unbound owner reads; the dense decode binding needs it.
        core.batch_invariant_mode = execution_mode(None)
        return core

    monkeypatch.setattr(nemotron_asr, "NemotronASRCore", make_core)
    monkeypatch.setattr(tg, "platform_graph_runtime", runtime)
    config = _engine_config(_model_config(**hf), **(engine or {}))
    return nemotron_asr.NemotronASRForRNNT(vllm_config=config, prefix="asr"), config


def _model_state(
    model: Any, config: SimpleNamespace, *, num_blocks: int = 3
) -> tuple[NemotronASRModelState, StateSlotBindings]:
    """The model's state consumer and resident slot bindings of ``num_blocks``."""
    device = next(model.core.encoder.parameters()).device
    spec = build_nemotron_persistent_state_spec(model.config)
    layer = model._persistent_state_layer.prefix
    slots = StateSlotBindings(
        [KVCacheGroupSpec([layer], spec)],
        {
            layer: torch.zeros(
                num_blocks * spec.page_size_bytes, dtype=torch.uint8, device=device
            )
        },
    )
    return NemotronASRModelState(config, model, None, device), slots


def _transaction_view(storage: Any) -> advance.TransactionPools:
    """Project component views (``spec`` and ``views``) as transaction pools."""
    pools = project_nemotron_state_pools(storage, n_layers=N_LAYERS)
    return advance.TransactionPools(
        channel=list(pools.channel),
        time=list(pools.convolution),
        valid_length=list(pools.valid_length),
        h=pools.predictor_h,
        c=pools.predictor_c,
        queue=pools.replay_queue,
        book=pools.replay_book,
        frontend_raw=pools.frontend_raw,
        frontend_mel=pools.frontend_mel,
        frontend_counter=pools.frontend_counters,
        endpoint_history=pools.endpoint_history,
        endpoint_book=pools.endpoint_book,
    )


def _pool_arguments(pools: advance.TransactionPools) -> Pools:
    """The ``advance_model_rows`` pool keywords for ``pools``."""
    return {
        "channel_pools": pools.channel,
        "time_pools": pools.time,
        "len_pools": pools.valid_length,
        "h_pool": pools.h,
        "c_pool": pools.c,
        "queue_pool": pools.queue,
        "book_pool": pools.book,
        "frontend_raw_pool": pools.frontend_raw,
        "frontend_mel_pool": pools.frontend_mel,
        "frontend_counter_pool": pools.frontend_counter,
        "endpoint_history_pool": pools.endpoint_history,
        "endpoint_book_pool": pools.endpoint_book,
    }


def _scratch(num_blocks: int, device: torch.device = CPU) -> Any:
    """Real-layout state storage of ``num_blocks`` slots for the tiny core."""
    spec = build_nemotron_persistent_state_spec(_model_config())
    return allocate_state_scratch(spec, num_blocks, device)


class _StandIn:
    """Allocate admission stand-in storage and keep only weak references.

    Each call records the requested slot count, the storage size and weak
    references to the storage and every pool view. ``owner`` optionally keeps
    the storage alive, as an unexpected owning reference would.
    """

    def __init__(self, device: torch.device = CPU, owner: list[Any] | None = None):
        self.device = device
        self.owner = owner
        self.slots: list[int] = []
        self.nbytes: list[int] = []
        self.storage: list[weakref.ref[torch.Tensor]] = []
        self.views: list[weakref.ref[torch.Tensor]] = []

    def __call__(self, slots: int) -> tuple[torch.Tensor, advance.TransactionPools]:
        scratch = _scratch(slots, self.device)
        pools = _transaction_view(scratch)
        if self.device.type == "cuda":
            advance.warmup_advance_model_rows_scatter(**_pool_arguments(pools))
        self.slots.append(slots)
        self.nbytes.append(scratch.raw.nbytes)
        self.storage.append(weakref.ref(scratch.raw))
        self.views.extend(weakref.ref(tensor) for tensor in pools.sequence())
        if self.owner is not None:
            self.owner.append(scratch.raw)
        return scratch.raw, pools


def _free_memory(
    monkeypatch: pytest.MonkeyPatch,
    readings: Callable[[int], int],
    on_read: Callable[[int], None] | None = None,
) -> list[int]:
    """Script every normalized free-memory reading ``N()`` by call index.

    Like the real reading, each call collects garbage first; ``on_read`` then
    observes the process at that reading. Returns the call log.
    """
    _seam(tg, "normalized_free_memory")
    calls: list[int] = []

    def read(_device: torch.device) -> int:
        gc.collect()
        index = len(calls)
        calls.append(index)
        if on_read is not None:
            on_read(index)
        return readings(index)

    monkeypatch.setattr(tg, "normalized_free_memory", read)
    return calls


def _readings(*values: int) -> Callable[[int], int]:
    """Readings in call order; the last value repeats."""
    return lambda index: values[min(index, len(values) - 1)]


def _admission_binding(
    populations: tuple[int, ...] = (1, 2),
    runtime: GraphRuntime | None = None,
) -> tg.TransactionGraphBinding:
    return tg.TransactionGraphBinding(
        populations=list(populations),
        geometries=(GEOM_REG,),
        maximum_population=max(populations),
        vllm_config=object(),
        runtime=runtime or _rerun_runtime(),
    )


def _admit(
    binding: tg.TransactionGraphBinding,
    core: Any,
    call: tg.TransactionCallBindings,
    stand_in: Callable[[int], tuple[torch.Tensor, advance.TransactionPools]],
    device: torch.device = CPU,
) -> None:
    """Admission capture over stand-in storage, as ``prepare_execution_memory``."""
    _seam(binding, "admit")(
        core, allocate=stand_in, call=call, hidden_size=CARRIER_HIDDEN, device=device
    )


def _serve(
    binding: tg.TransactionGraphBinding,
    core: Any,
    call: tg.TransactionCallBindings,
    pools: advance.TransactionPools,
    device: torch.device = CPU,
) -> None:
    """Serving capture over resident pools, as ``capture_transaction_graphs``."""
    binding.capture(
        core, pools=pools, call=call, hidden_size=CARRIER_HIDDEN, device=device
    )


def _track_capture_cells(
    monkeypatch: pytest.MonkeyPatch, events: list[Any] | None = None
) -> list[SimpleNamespace]:
    """Record every cell the shared capture path prepares, holding nothing.

    Each record has the cell, weak references to its graph, arena, capture
    callbacks and every tensor copied while preparing it (page snapshots,
    expected outputs, capture-input copies and persistent buffers), and the
    shapes of the pools it was prepared over. ``events`` receives
    ``("prepare", label)`` per cell.
    """
    prepared: list[SimpleNamespace] = []
    capture_cell = tg.TransactionGraphBinding._capture_cell

    def track(self: Any, core: Any, cell: tg.TransactionCell, **kwargs: Any) -> Any:
        clones: list[weakref.ref[torch.Tensor]] = []
        with _tracking_clones(clones):
            graph = capture_cell(self, core, cell, **kwargs)
        prepared.append(
            SimpleNamespace(
                cell=cell,
                graph=weakref.ref(graph),
                arena=weakref.ref(graph.arena),
                callbacks=_capture_callbacks(graph),
                clones=clones,
                pool_shapes=tuple(
                    tuple(pool.shape) for pool in kwargs["pools"].sequence()
                ),
            )
        )
        if events is not None:
            events.append(("prepare", cell.label))
        return graph

    monkeypatch.setattr(tg.TransactionGraphBinding, "_capture_cell", track)
    return prepared


@contextmanager
def _tracking_clones(clones: list[weakref.ref[torch.Tensor]]) -> Iterator[None]:
    """Append a weak reference to every tensor copied in the scope.

    Covers ``Tensor.clone``, ``torch.clone`` and ``torch.empty_like``, the
    allocation an ``empty_like(...).copy_(...)`` copy starts from. A copy built
    from a bare ``torch.empty`` is not tracked; capture code copies through
    these three.
    """

    def tracked(allocate: Callable[..., torch.Tensor]) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> torch.Tensor:
            copy = allocate(*args, **kwargs)
            clones.append(weakref.ref(copy))
            return copy

        return call

    with pytest.MonkeyPatch.context() as patch:
        for owner, name in (
            (torch.Tensor, "clone"),
            (torch, "clone"),
            (torch, "empty_like"),
        ):
            patch.setattr(owner, name, tracked(getattr(owner, name)))
        yield


#: Cell buffers that persist after verification besides the arena's: static
#: inputs and owned outputs ("Inputs, outputs and signature").
_PERSISTENT_CELL_BUFFERS = (
    "static_ids",
    "static_embeds",
    "rows_out",
    "status_out",
    "progress_out",
)


def _persistent_storage(graph: Any) -> set[int]:
    """Storage addresses a published cell may own after verification.

    These are its arena's one device buffer and two pinned host buffers,
    static inputs and owned outputs; nothing else from capture may outlive it
    (Capture-lifetime rule). Arena views must alias the device buffer, so a
    copy kept among them still counts as live.
    """
    arena = graph.arena
    host = list(arena._host)
    assert len(host) == 2, "an arena owns exactly two host buffers"
    device = arena._device.untyped_storage().data_ptr()
    tensors = [getattr(graph, name) for name in _PERSISTENT_CELL_BUFFERS]
    tensors += [arena._device, *host]
    allowed = {
        tensor.untyped_storage().data_ptr()
        for tensor in tensors
        if tensor.untyped_storage().nbytes()
    }
    return allowed | {
        view.untyped_storage().data_ptr()
        for view in arena.views.values()
        if view.untyped_storage().data_ptr() == device
    }


def _live_capture_clones(records: list[SimpleNamespace]) -> int:
    """Count live copies made while preparing ``records`` outside persistent storage.

    A page snapshot, expected output or capture-input copy still alive counts
    wherever it is kept, including on the graph or its arena.
    """
    gc.collect()
    persistent: set[int] = set()
    for record in records:
        graph = record.graph()
        if graph is not None:
            persistent |= _persistent_storage(graph)
        del graph
    return sum(
        (tensor := reference()) is not None
        and tensor.untyped_storage().data_ptr() not in persistent
        for record in records
        for reference in record.clones
    )


def _track_capture_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> list[weakref.ref[Any]]:
    """Weak references to every capture ``TransactionInputs``."""
    inputs: list[weakref.ref[Any]] = []
    prepare = tg.prepare_transaction

    def track(*args: Any, **kwargs: Any) -> Any:
        prepared = prepare(*args, **kwargs)
        inputs.append(weakref.ref(prepared))
        return prepared

    monkeypatch.setattr(tg, "prepare_transaction", track)
    return inputs


def _capture_callbacks(graph: Any) -> list[weakref.ref[Any]]:
    """Weak references to a prepared cell's capture and verify callbacks."""
    return [
        weakref.ref(callback)
        for callback in (
            getattr(graph, "capture_graph", None),
            getattr(graph, "verify", None),
        )
        if callback is not None
    ]


def _budget(released: int, admitted: int, served: int) -> Callable[[int], int]:
    """Readings ``N0, Na, N1, Ns0, Ns1`` with ``N1 - Na == released``.

    ``A = N0 - N1 == admitted`` and ``S = Ns0 - Ns1 == served``.
    """
    base = 1 << 34
    return _readings(base + admitted, base - released, base, base, base - served)


@pytest.fixture
def cpu_readings(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterator[SimpleNamespace]:
    """Deterministic admission readings for one CPU startup.

    The readings ``N0, Na, N1, Ns0, Ns1`` release far more than any stand-in
    and admit as many bytes as serving uses, so admission passes without real
    device memory queries; later readings repeat ``Ns1``. A test whose sequence
    is fixed sets ``expected``, checked at teardown when the test passed.
    Tests of the readings patch their own.
    """
    readings = SimpleNamespace(
        calls=_free_memory(monkeypatch, _budget(1 << 30, 1 << 20, 1 << 20)),
        expected=None,
    )
    failed = request.session.testsfailed
    yield readings
    if readings.expected is not None and request.session.testsfailed == failed:
        assert len(readings.calls) == readings.expected, readings.calls


def _model_call(model: Any) -> tg.TransactionCallBindings:
    """The call bindings the model's serving capture uses."""
    return tg.TransactionCallBindings(
        adapter=model._emission_adapter,
        decode_resolver=model._decode_resolver,
        encoder_transition=model._encoder_execution.transition,
        bucket_transition=getattr(model, "_chunk_bucket_binding", None),
        placeholder_id=PLACEHOLDER_ID,
        park_id=PARK_ID,
        eou_token_id=EOU,
        graph_covers_decode=model._decode_graph_binding is not None,
    )


def _resident(model: Any, slots: StateSlotBindings) -> advance.TransactionPools:
    """The resident pools behind ``slots``, as serving capture reads them."""
    views = slots.views[model._persistent_state_layer.prefix]
    return _transaction_view(SimpleNamespace(spec=views.spec, views=views._views))


def _live(references: list[weakref.ref[Any]]) -> list[Any]:
    gc.collect()
    return [value for reference in references if (value := reference()) is not None]


def _schedule(
    slots: StateSlotBindings,
    layer: str,
    tokens: dict[str, int],
    *,
    generation: int = 1,
) -> SchedulerOutput:
    """Bind one scheduled token per request; request ``i`` owns slot ``i + 1``."""
    step = SchedulerOutput.make_empty()
    step.state_step_seq = generation
    step.state_slot_handles = {
        req_id: {layer: StateSlotHandle("epoch", index, generation)}
        for index, req_id in enumerate(tokens, 1)
    }
    step.num_scheduled_tokens = dict.fromkeys(tokens, 1)
    step.scheduled_cached_reqs.req_ids = list(tokens)
    step.scheduled_cached_reqs.new_token_ids = [[token] for token in tokens.values()]
    return step


def _serve_session(
    model: Any, state: NemotronASRModelState, slots: StateSlotBindings
) -> list[int]:
    """One g0 session: a fresh CHUNK, then its echoes until it parks."""
    layer = model._persistent_state_layer.prefix
    audio = _chunk(17, seq=0, geometry=GEOM_REG, final=False, prompt=0)
    token, emitted = PLACEHOLDER_ID, []
    for computed in range(CAP + 2):
        step = _schedule(slots, layer, {"left": token})
        state.prepare_step(step, dummy_run=False, is_profile=False)
        slots.begin_step(step, SimpleNamespace(wait_stream=lambda _: None))
        slots.activate_requests()
        if computed == 0:
            state.add_request(
                0,
                NewRequestData(
                    "left",
                    [PLACEHOLDER_ID],
                    [_feature(payload=audio)],
                    None,
                    None,
                    ([1],),
                    0,
                    None,
                    prefill_token_ids=[PLACEHOLDER_ID],
                ),
            )
        batch = SimpleNamespace(
            req_ids=["left"],
            num_scheduled_tokens=[1],
            input_ids=torch.tensor([token]),
            num_computed_tokens_np=[computed],
        )
        projection = state.prepare_inputs(
            batch, SimpleNamespace(req_id_to_index={"left": 0})
        )
        carrier = (
            audio.unsqueeze(0) if computed == 0 else torch.zeros(1, CARRIER_HIDDEN)
        )
        with torch.inference_mode():
            hidden = model.forward(
                input_ids=batch.input_ids, inputs_embeds=carrier, **projection
            )
            token = int(model.compute_logits(hidden).argmax(-1).item())
        state.postprocess_state(torch.tensor([0]), torch.tensor([1]))
        emitted.append(token)
        if token == PARK_ID:
            return emitted
    raise AssertionError("the session did not park")


def _chunk_forward(
    model: Any, state: NemotronASRModelState, slots: StateSlotBindings
) -> None:
    """Session ``left``'s fresh g0 CHUNK step, up to its commit collection."""
    layer = model._persistent_state_layer.prefix
    audio = _chunk(17, seq=0, geometry=GEOM_REG, final=False, prompt=0)
    step = _schedule(slots, layer, {"left": PLACEHOLDER_ID})
    state.prepare_step(step, dummy_run=False, is_profile=False)
    slots.begin_step(step, SimpleNamespace(wait_stream=lambda _: None))
    slots.activate_requests()
    state.add_request(
        0,
        NewRequestData(
            "left",
            [PLACEHOLDER_ID],
            [_feature(payload=audio)],
            None,
            None,
            ([1],),
            0,
            None,
            prefill_token_ids=[PLACEHOLDER_ID],
        ),
    )
    batch = SimpleNamespace(
        req_ids=["left"],
        num_scheduled_tokens=[1],
        input_ids=torch.tensor([PLACEHOLDER_ID]),
        num_computed_tokens_np=[0],
    )
    projection = state.prepare_inputs(
        batch, SimpleNamespace(req_id_to_index={"left": 0})
    )
    with torch.inference_mode():
        model.forward(
            input_ids=batch.input_ids, inputs_embeds=audio.unsqueeze(0), **projection
        )


def _policy(plan: Any, mode: int, threshold: int, residue: int) -> Any:
    rows = plan.num_decodes + plan.num_prefills
    return replace(
        plan,
        endpoint_mode=torch.full((rows,), mode, dtype=torch.int64),
        endpoint_threshold_frames=torch.full((rows,), threshold, dtype=torch.int64),
        endpoint_residue_frames=torch.full((rows,), residue, dtype=torch.int64),
    )


def _step(
    core: Any,
    pools: Pools,
    call: tg.TransactionCallBindings,
    ids: list[int],
    embeds: torch.Tensor,
    plan: Any,
    graph: Any,
    *,
    capture: bool = False,
    memory_profile: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sink = _CommitRecorder()
    out = _advance(
        core,
        pools,
        call,
        (ids, embeds, plan),
        graph,
        sink,
        capture=capture,
        memory_profile=memory_profile,
    )
    assert sink.log == ["reserve", "stage"]
    return (
        out.clone(),
        sink.staged[0].clone(),
        sink.plans[0].progress.clone(),
    )


Rows = tuple[list[int], torch.Tensor, Any]


def _rows(
    decodes: list[int],
    prefills: list[int],
    fresh: list[bool] | None = None,
    *,
    geometries: list[int] | None = None,
    chunk: list[bool] | None = None,
    num_pool_blocks: int = 4,
    device: torch.device = CPU,
) -> Rows:
    """Ids, carriers and plan: echo rows on ``decodes``, then CHUNK rows."""
    fresh = [True] * len(prefills) if fresh is None else fresh
    rows = len(decodes) + len(prefills)
    geometries = [GEOM_REG] * rows if geometries is None else geometries
    chunk = [False] * len(decodes) + [True] * len(prefills) if chunk is None else chunk
    carriers = [
        _chunk(5 + row, seq=0, geometry=geometries[row], final=False, prompt=0)
        if chunk[row]
        else torch.zeros(CARRIER_HIDDEN)
        for row in range(rows)
    ]
    plan = _plan(
        decodes=decodes,
        prefills=prefills,
        num_pool_blocks=num_pool_blocks,
        has_initial=[not value for value in fresh],
        geometries=geometries,
        chunk=chunk,
    )
    embeds = torch.stack(carriers) if carriers else torch.zeros(0, CARRIER_HIDDEN)
    ids = [PLACEHOLDER_ID if chunk[row] else VOCAB - 1 for row in range(rows)]
    return ids, embeds.to(device), plan


def _advance(
    core: Any,
    pools: Pools,
    call: tg.TransactionCallBindings,
    rows: Rows,
    graph: Any,
    sink: Any = None,
    **kwargs: Any,
) -> torch.Tensor:
    """One ``advance_model_rows`` call over ``rows`` and ``pools``."""
    ids, embeds, plan = rows
    return advance.advance_model_rows(
        core,
        torch.tensor(ids, dtype=torch.long, device=embeds.device),
        embeds,
        plan,
        adapter=call.adapter,
        decode_resolver=call.decode_resolver,
        encoder_transition=call.encoder_transition,
        bucket_transition=call.bucket_transition,
        placeholder_id=PLACEHOLDER_ID,
        park_id=PARK_ID,
        eou_token_id=call.eou_token_id,
        commit_sink=sink,
        graph_covers_decode=call.graph_covers_decode,
        transaction_graph=graph,
        **kwargs,
        **pools,
    )


def _prepared(
    core: Any, pools: Pools, call: tg.TransactionCallBindings, rows: Rows
) -> Any:
    """The host prologue's ``TransactionInputs`` for ``rows``."""
    ids, embeds, plan = rows
    return advance.prepare_transaction(
        core,
        torch.tensor(ids, dtype=torch.long, device=embeds.device),
        embeds,
        plan,
        pools=_transaction_pools(pools),
        eou_token_id=call.eou_token_id,
        adapter=call.adapter,
        decode_resolver=call.decode_resolver,
        encoder_transition=call.encoder_transition,
        bucket_transition=call.bucket_transition,
        placeholder_id=PLACEHOLDER_ID,
        park_id=PARK_ID,
        has_commit_sink=True,
        capture=False,
        graph_covers_decode=call.graph_covers_decode,
        memory_profile=False,
        native_burst_handoff=None,
    )


class _PermissiveLease:
    """A registry whose every lease is current, for sink-level collection."""

    def validate_lease(self, _bindings: Any) -> None:
        return None

    def lease_is_current(self, _bindings: Any) -> bool:
        return True

    def resolve_status(self, **_kwargs: Any) -> bool:
        return True


class _TrackedTicket:
    def __init__(self, ticket: Any, log: list[str]) -> None:
        self._ticket = ticket
        self._log = log

    def stage(self, row_status: torch.Tensor) -> None:
        self._log.append("stage")
        self._ticket.stage(row_status)

    def cancel(self) -> None:
        self._log.append("cancel")
        self._ticket.cancel()


class _TrackedSink(BoundedCommitSink):
    """The production sink, logging reserve, stage, cancel and collect.

    ``fail_stage`` makes staging raise after the ticket has marked itself
    done, as a ``CommitTicket.stage`` failure after ``_done`` would.
    """

    def __init__(
        self,
        registry: Any = None,
        *,
        rows: int = 4,
        device: torch.device = CPU,
        fail_stage: bool = False,
    ) -> None:
        super().__init__(registry or _PermissiveLease(), max_rows=rows, device=device)
        self.log: list[str] = []
        self._fail_stage = fail_stage

    def reserve(self, plan: Any) -> Any:
        ticket = super().reserve(plan)
        self.log.append("reserve")
        return _TrackedTicket(ticket, self.log)

    def _stage(self, plan: Any, row_status: torch.Tensor) -> None:
        if self._fail_stage:
            raise RuntimeError("injected stage failure")
        super()._stage(plan, row_status)

    def collect(self) -> Any:
        self.log.append("collect")
        return super().collect()

    def failed_request_ids(self) -> set[str]:
        """Collect the staged step as ``postprocess_state`` does."""
        reports, _records, lease_ok = self.collect()
        return resolve_status_reports(self._registry, reports, lease_ok=lease_ok)[1]


def _collected_step(
    core: Any,
    pools: Pools,
    call: tg.TransactionCallBindings,
    rows: Rows,
    graph: Any,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], set[str]]:
    """One step through the production sink and its collection.

    Returns the projected rows and the collected statuses and progress, and
    the failed request ids collection passes to ``state_slots.fail_requests``.
    """
    plan = rows[2]
    count = len(plan.bindings)
    sink = _TrackedSink(rows=count, device=rows[1].device)
    out = _advance(core, pools, call, rows, graph, sink).clone()
    assert sink.log == ["reserve", "stage"]
    failed = sink.failed_request_ids()
    status = sink._status_host[:count].clone()
    progress = sink._progress_host[:count].clone()
    return (out.cpu(), status, progress), failed


def _leased(request_id: str, block: int) -> tuple[SessionRegistry, _TrackedSink, Rows]:
    """A registry holding one fresh session's lease, its sink and CHUNK rows."""
    registry = SessionRegistry()
    [bound] = _bind(registry, [_row(request_id, block, chunk=True, prior=False)])
    ids, embeds, _plan_rows = _rows([], [block])
    plan = _plan(
        prefills=[block],
        num_pool_blocks=4,
        generations=[bound.admission_generation],
        request_ids=(request_id,),
    )
    return registry, _TrackedSink(registry), (ids, embeds, plan)


def _chunk(
    seed: int,
    *,
    seq: int,
    geometry: int,
    final: bool,
    prompt: int,
    samples: int | None = None,
) -> torch.Tensor:
    """One CHUNK envelope; ``samples`` defaults to the geometry's cadence."""
    torch.manual_seed(seed)
    if samples is None:
        samples = FINAL_SAMPLES if geometry == GEOM_FINAL else REG_SAMPLES
    return _envelope(
        torch.randn(samples) * 0.01,
        final=final,
        seq=seq,
        geometry=geometry,
        prompt=prompt,
    )


def _drive(
    core: Any,
    pools: Pools,
    call: tg.TransactionCallBindings,
    graph: Any,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Two sessions through fresh, continuing and final CHUNKs plus echoes.

    Rows use block ids, prompts, row orders and endpoint policies that differ
    from the capture inputs, and the populations change between steps.
    """
    device = pools["book_pool"].device
    records: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    sessions = {"a": (2, GEOM_REG, 1), "c": (3, GEOM_REG, 2)}
    policies = {"a": (1, 2, 1), "c": (0, 0, 0)}

    def echo(tokens: dict[str, int]) -> None:
        for _ in range(CAP + 2):
            active = [name for name, token in tokens.items() if token != PARK_ID]
            if not active:
                return
            plan = _plan(
                decodes=[sessions[name][0] for name in active],
                num_pool_blocks=4,
                geometries=[sessions[name][1] for name in active],
                prompts=[sessions[name][2] for name in active],
                request_ids=tuple(active),
            )
            mode, threshold, residue = policies[active[0]]
            record = _step(
                core,
                pools,
                call,
                [tokens[name] for name in active],
                torch.zeros(len(active), CARRIER_HIDDEN, device=device),
                _policy(plan, mode, threshold, residue),
                graph,
            )
            records.append(record)
            for index, name in enumerate(active):
                tokens[name] = int(record[0][index, 0])
        raise AssertionError("replay queue did not drain")

    def chunk(names: list[str], seq: int, *, fresh: bool, final: bool) -> None:
        embeds = torch.stack(
            [
                _chunk(
                    11 * seq + index,
                    seq=seq,
                    geometry=sessions[name][1],
                    final=final,
                    prompt=sessions[name][2],
                    samples=REG_SAMPLES // 2 if final else None,
                )
                for index, name in enumerate(names)
            ]
        ).to(device)
        plan = _plan(
            prefills=[sessions[name][0] for name in names],
            num_pool_blocks=4,
            has_initial=[not fresh] * len(names),
            geometries=[sessions[name][1] for name in names],
            prompts=[sessions[name][2] for name in names],
            request_ids=tuple(names),
        )
        mode, threshold, residue = policies[names[0]]
        record = _step(
            core,
            pools,
            call,
            [PLACEHOLDER_ID] * len(names),
            embeds,
            _policy(plan, mode, threshold, residue),
            graph,
        )
        records.append(record)
        echo({name: int(record[0][index, 0]) for index, name in enumerate(names)})

    chunk(["a", "c"], 0, fresh=True, final=False)
    chunk(["c", "a"], 1, fresh=False, final=False)
    chunk(["a"], 2, fresh=False, final=False)
    sessions["b"] = (1, GEOM_REG, 3)
    policies["b"] = (1, 1, 0)
    chunk(["b"], 0, fresh=True, final=True)
    return records


def _assert_records_equal(actual: list[Any], expected: list[Any]) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        for left, right in zip(got, want, strict=True):
            torch.testing.assert_close(left.cpu(), right.cpu(), atol=0, rtol=0)


def _replay_parity(device: torch.device, runtime: GraphRuntime) -> None:
    core = _core(device)
    call = _call_bindings(core)
    eager_pools = _endpoint_pools(4, device)
    graph_pools = _clone_pools(eager_pools)
    binding = _binding(core, graph_pools, call, runtime, device=device)
    _assert_pools_equal(graph_pools, eager_pools)
    assert binding.ready

    expected = _drive(core, eager_pools, call, None)
    actual = _drive(core, graph_pools, call, binding)

    _assert_records_equal(actual, expected)
    _assert_pools_equal(graph_pools, eager_pools)
    emitted = {
        int(token)
        for rows, _status, _progress in expected
        for token in rows[:, 0].tolist()
    }
    assert emitted - {PARK_ID}, "the trajectory must emit labels"
    assert all(not bool(status.any()) for _rows, status, _progress in expected)
    counts = binding.counters()
    assert counts["fallbacks"] == 0
    assert counts["replays"] == len(expected)
    for label in (
        "chunk-fresh:g0:b2",
        "chunk-continuing:g0:b2",
        "chunk-continuing:g0:b1",
        "chunk-fresh:g0:b1",
        "replay:b1",
    ):
        assert counts[f"replays:{label}"] >= 1, label


#: Endpoint policy (mode, threshold, residue) per CHUNK step of the drive.
_STEP_POLICIES = ((1, 2, 1), (0, 0, 0), (1, 1, 0))


def _drive_population(
    core: Any,
    pools: Pools,
    call: tg.TransactionCallBindings,
    graph: Any,
    population: int,
    *,
    chunks: int,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], int]:
    """``population`` sessions in lockstep: one fresh, then continuing CHUNKs.

    Sessions own descending block ids and mixed prompts, the row order changes
    every CHUNK and every CHUNK drains its echoes before the next one, so
    every CHUNK step is one declared cell. Echo steps shrink as rows park.
    Returns the records and the number of echo transactions.
    """
    device = pools["book_pool"].device
    names = [f"s{index}" for index in range(population)]
    block = {name: population - index for index, name in enumerate(names)}
    prompt = {name: index % 4 for index, name in enumerate(names)}
    records: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    echoes = 0
    for seq in range(chunks):
        shift = (seq * 7) % population
        order = names[shift:] + names[:shift]
        if seq % 2:
            order.reverse()
        mode, threshold, residue = _STEP_POLICIES[seq % len(_STEP_POLICIES)]
        embeds = torch.stack(
            [
                _chunk(
                    101 * seq + names.index(name),
                    seq=seq,
                    geometry=GEOM_REG,
                    final=False,
                    prompt=prompt[name],
                )
                for name in order
            ]
        ).to(device)
        plan = _plan(
            prefills=[block[name] for name in order],
            num_pool_blocks=population + 1,
            has_initial=[seq > 0] * population,
            prompts=[prompt[name] for name in order],
            request_ids=tuple(order),
        )
        record = _step(
            core,
            pools,
            call,
            [PLACEHOLDER_ID] * population,
            embeds,
            _policy(plan, mode, threshold, residue),
            graph,
        )
        records.append(record)
        tokens = {name: int(record[0][row, 0]) for row, name in enumerate(order)}
        for _ in range(CAP + 2):
            active = [name for name in order if tokens[name] != PARK_ID]
            if not active:
                break
            plan = _plan(
                decodes=[block[name] for name in active],
                num_pool_blocks=population + 1,
                prompts=[prompt[name] for name in active],
                request_ids=tuple(active),
            )
            record = _step(
                core,
                pools,
                call,
                [tokens[name] for name in active],
                torch.zeros(len(active), CARRIER_HIDDEN, device=device),
                _policy(plan, mode, threshold, residue),
                graph,
            )
            records.append(record)
            echoes += 1
            for row, name in enumerate(active):
                tokens[name] = int(record[0][row, 0])
        else:
            raise AssertionError("replay queue did not drain")
    return records, echoes


def _population_parity(
    device: torch.device,
    runtime: GraphRuntime,
    population: int,
    *,
    chunks: int,
    endpoint: bool = True,
) -> None:
    """Lockstep CHUNKs at ``population`` replay their cells bitwise exactly.

    Only ``population`` is declared, so every CHUNK must replay and only echo
    steps whose population shrank may fall back, as ``undeclared``.
    ``endpoint`` selects the ``endpoint_enabled`` pool configuration.
    """
    core = _core(device)
    call = _call_bindings(core, endpoint=endpoint)
    eager_pools = _endpoint_pools(population + 1, device, endpoint=endpoint)
    graph_pools = _clone_pools(eager_pools)
    binding = tg.TransactionGraphBinding(
        populations=[population],
        geometries=(GEOM_REG,),
        maximum_population=population,
        vllm_config=object(),
        runtime=runtime,
    )
    if device.type == "cuda":
        advance.warmup_advance_model_rows_scatter(**graph_pools)
    binding.capture(
        core,
        pools=_transaction_pools(graph_pools),
        call=call,
        hidden_size=CARRIER_HIDDEN,
        device=device,
    )
    _assert_pools_equal(graph_pools, eager_pools)
    assert binding.ready

    expected, echoes = _drive_population(
        core, eager_pools, call, None, population, chunks=chunks
    )
    actual, _ = _drive_population(
        core, graph_pools, call, binding, population, chunks=chunks
    )

    _assert_records_equal(actual, expected)
    _assert_pools_equal(graph_pools, eager_pools)
    assert all(not bool(status.any()) for _rows, status, _progress in expected)
    emitted = {
        int(token)
        for rows, _status, _progress in expected
        for token in rows[:, 0].tolist()
    }
    assert emitted - {PARK_ID}, "the trajectory must emit labels"
    counts = binding.counters()
    assert counts[f"replays:chunk-fresh:g0:b{population}"] == 1
    assert counts[f"replays:chunk-continuing:g0:b{population}"] == chunks - 1
    echo_replays = counts[f"replays:replay:b{population}"]
    fallbacks = {
        key: value
        for key, value in counts.items()
        if key.startswith("fallbacks:") and value
    }
    for key in fallbacks:
        assert key.startswith("fallbacks:replay:b") and key.endswith(":undeclared"), key
    assert echo_replays + sum(fallbacks.values()) == echoes
    assert counts["replays"] == chunks + echo_replays
    assert echoes > 0, "the trajectory must echo"


#: CHUNK steps per population; B63 runs fewer to keep runtime sane.
_POPULATION_CHUNKS = [(1, 3), (31, 3), (63, 2)]


# ---- CPU: arena, guard, classification and replay plumbing -----------------


@contextmanager
def _functional_encoder() -> Iterator[None]:
    """Run the encoder-execution tests' functional transition in this scope."""
    patch = pytest.MonkeyPatch()
    patch.setattr(
        encoder_tests.encoder_execution_module,
        "execute_encoder_transition",
        encoder_tests._functional_graph_transition,
    )
    try:
        yield
    finally:
        patch.undo()


def _nested_launchers(modes: list[Any]) -> dict[str, Callable[[], Any]]:
    """Captured encoder, decode and CHUNK-bucket entry points, by binding.

    Every launch through them appends its graph mode to ``modes``.
    """
    encoder = encoder_tests.build_encoder_execution(
        encoder_tests._compiled_core(),
        encoder_tests._dense_graphed_config("eager-graphed"),
        maximum_population=1,
        warmup_geometries=(0,),
        vllm_config=object(),
        graph_runtime=_recording(encoder_tests._graph_runtime(), modes),
    )

    def encode(population: int = 1) -> Any:
        with _functional_encoder():
            return encoder.transition(
                *encoder_tests._graph_transition_args(population=population)
            )

    with _functional_encoder():
        encoder.warmup_domain(
            expected_cells=((0, 1),),
            invoke=lambda _geometry, population: encode(population),
        )
    _, decode = _decode_calls()
    dense = DenseGraphBinding(
        decode_fn=decode,
        predictor=object(),
        joint=object(),
        vllm_config=object(),
        frame_widths=(2,),
        tiers=(1,),
        encoder_hidden=3,
        predictor_layers=2,
        predictor_hidden=5,
        blank_id=7,
        runtime=_recording(_decode_runtime(), modes),
    )
    dense.warmup(CPU, torch.float32)
    decode_fn = dense.decode_fn(geometry=0, tier=1)
    decode_state = rnnt.DecodeState(
        h=torch.ones(2, 1, 5), c=torch.ones(2, 1, 5), last_label=torch.tensor([4])
    )
    core = _core()
    env = _chunk(3, seq=0, geometry=1, final=False, prompt=0).unsqueeze(0)
    session = _fresh_state(1)
    bucket_args = dict(
        geometry=1,
        admitted_prompt=torch.zeros(1, dtype=torch.long),
        incoming_status=torch.zeros(1, dtype=torch.int32),
        queue_capacity=CAP,
    )
    bucket = capture_chunk_bucket(
        core,
        env,
        session,
        **bucket_args,
        vllm_config=object(),
        runtime=_recording(_rerun_runtime(), modes),
        capture_decode_fn=rnnt.decode_dense_masked_frames,
        decoder_tier=1,
    )
    return {
        "encoder": encode,
        "decode": lambda: decode_fn(
            torch.ones(1, 2, 3), torch.tensor([2]), object(), object(), decode_state
        ),
        "chunk-bucket": lambda: bucket(
            core,
            env,
            session,
            **bucket_args,
            decode_fn=rnnt.decode_dense_masked_frames,
        ),
    }


@pytest.mark.cpu_test
@torch.inference_mode()
def test_nested_bindings_run_eagerly_only_inside_the_capture_scope() -> None:
    runtime = _rerun_runtime()
    assert nested_graph_mode(runtime) == "graph"
    with outer_transaction_capture():
        assert nested_graph_mode(runtime) == "none"
    assert nested_graph_mode(runtime) == "graph"

    modes: list[Any] = []
    for name, launch in _nested_launchers(modes).items():
        modes.clear()
        launch()
        with outer_transaction_capture():
            launch()
        launch()
        assert modes == ["graph", "none", "graph"], name


@pytest.mark.cpu_test
def test_host_staging_is_rejected_inside_the_capture_scope() -> None:
    staging = advance.HostStaging(4)
    vector = torch.arange(2, dtype=torch.int64)
    with outer_transaction_capture():
        with pytest.raises(RuntimeError, match="frozen host source"):
            advance._h2d(vector, CPU)
        with pytest.raises(RuntimeError, match="frozen host source"):
            staging.stage("geometry", vector, CPU)
        with pytest.raises(RuntimeError, match="frozen host source"):
            staging.stage_bucket(0, vector, CPU)
    assert torch.equal(advance._h2d(vector, CPU), vector)


def _fresh_chunk_inputs(
    core: Any, pools: Pools, call: tg.TransactionCallBindings, blocks: list[int]
) -> Any:
    plan = _policy(
        _plan(prefills=blocks, num_pool_blocks=4, prompts=[1] * len(blocks)),
        1,
        3,
        2,
    )
    device = pools["book_pool"].device
    embeds = torch.stack(
        [_chunk(3, seq=0, geometry=GEOM_REG, final=False, prompt=1)] * len(blocks)
    ).to(device)
    return advance.prepare_transaction(
        core,
        torch.full((len(blocks),), PLACEHOLDER_ID, dtype=torch.long, device=device),
        embeds,
        plan,
        pools=_transaction_pools(pools),
        eou_token_id=EOU,
        adapter=call.adapter,
        decode_resolver=call.decode_resolver,
        encoder_transition=None,
        bucket_transition=None,
        placeholder_id=PLACEHOLDER_ID,
        park_id=PARK_ID,
        has_commit_sink=True,
        capture=False,
        graph_covers_decode=False,
        memory_profile=False,
        native_burst_handoff=None,
    )


@pytest.mark.cpu_test
def test_arena_stager_matches_live_staging_bit_identical() -> None:
    core = _core()
    call = _call_bindings(core)
    live_pools = _endpoint_pools(4)
    arena_pools = _clone_pools(live_pools)
    for pools in (live_pools, arena_pools):
        pools["book_pool"][1:, _BOOK["last_label"]] = core.blank_id
    inputs = _fresh_chunk_inputs(core, live_pools, call, [3, 1])
    arena_inputs = replace(inputs, pools=_transaction_pools(arena_pools))
    arena = tg.TransactionArena(tg._arena_layout(inputs), CPU)
    arena.upload(arena_inputs)
    for name, vector in inputs.staged_vectors():
        assert torch.equal(arena.views[name], vector), name

    ids = torch.full((2,), PLACEHOLDER_ID, dtype=torch.long)
    embeds = torch.stack(
        [_chunk(3, seq=0, geometry=GEOM_REG, final=False, prompt=1)] * 2
    )
    live = advance.run_transaction_body(
        core,
        ids,
        embeds,
        inputs,
        advance.LiveTransactionStager(inputs, advance.HostStaging(4)),
    )
    advance.execute_transaction_scatters(live.scatter_ops)
    with outer_transaction_capture():
        staged = advance.run_transaction_body(
            core, ids, embeds, arena_inputs, tg.ArenaTransactionStager(arena)
        )
        advance.execute_transaction_scatters(staged.scatter_ops)
    for left, right in (
        (staged.projection_rows, live.projection_rows),
        (staged.status, live.status),
        (staged.progress, live.progress),
    ):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    _assert_pools_equal(arena_pools, live_pools)


@pytest.mark.cpu_test
def test_arena_rejects_vectors_outside_its_layout() -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    arena = tg.TransactionArena(
        tg._arena_layout(_fresh_chunk_inputs(core, pools, call, [1])), CPU
    )
    with pytest.raises(ValueError, match="differs from the arena"):
        arena.upload(_fresh_chunk_inputs(core, pools, call, [1, 2]))
    with pytest.raises(ValueError, match="only int64 and bool"):
        tg.TransactionArena((("x", torch.float32, (1,)),), CPU)


class _FenceEvent:
    """A host event stub: logs its waits and records against ``log``.

    A wait also logs the fenced host buffer's slot id as the rewrite sees it.
    """

    def __init__(self, index: int, arena: Any, log: list[Any]) -> None:
        self._index = index
        self._arena = arena
        self._log = log

    def synchronize(self) -> None:
        held = int(self._arena._host_views[self._index]["composed_indices"][0])
        self._log.append(("wait", self._index, held))

    def record(self) -> None:
        self._log.append(("record", self._index))


@pytest.mark.cpu_test
def test_arena_alternates_host_buffers_across_calls() -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    calls = [_fresh_chunk_inputs(core, pools, call, [block]) for block in (1, 3, 2)]
    arena = tg.TransactionArena(tg._arena_layout(calls[0]), CPU)
    log: list[Any] = []
    arena._events = [_FenceEvent(index, arena, log) for index in range(2)]
    copy = arena._device.copy_

    def upload(source: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        log.append(("copy", int(source.data_ptr() == arena._host[1].data_ptr())))
        return copy(source, **kwargs)

    arena._device.copy_ = upload
    for inputs in calls:
        arena.upload(inputs)
        assert torch.equal(
            arena.views["composed_indices"], inputs.vector("composed_indices")
        )
    assert log == [
        ("wait", 0, 0),
        ("copy", 0),
        ("record", 0),
        ("wait", 1, 0),
        ("copy", 1),
        ("record", 1),
        ("wait", 0, 1),
        ("copy", 0),
        ("record", 0),
    ]
    assert int(arena._host_views[0]["composed_indices"][0]) == 2
    assert int(arena._host_views[1]["composed_indices"][0]) == 3


#: Classification cases over slots 2 and 4 of six; slots 1, 3 and 5 stay
#: bound without a row in the step: ``(decodes, prefills, fresh, options)``.
_CLASSIFIED = {
    "chunk-fresh": ([], [2, 4], [True, True], {}),
    "chunk-continuing": ([], [2, 4], [False, False], {}),
    "replay": ([2, 4], [], [], {}),
    "empty": ([], [], [], {}),
    "fresh-non-chunk": ([], [2], [True], {"chunk": [False]}),
    "multi-geometry": ([], [2, 4], [True, True], {"geometries": [0, GEOM_FINAL]}),
    "mixed-roles": ([2], [4], [True], {}),
    "mixed-fresh": ([], [2, 4], [True, False], {}),
}

_CELLS = {
    "chunk-fresh": tg.TransactionCell(tg.CELL_FRESH_CHUNK, GEOM_REG, 2),
    "chunk-continuing": tg.TransactionCell(tg.CELL_CONTINUING_CHUNK, GEOM_REG, 2),
    "replay": tg.TransactionCell(tg.CELL_REPLAY, -1, 2),
}


@pytest.mark.cpu_test
@pytest.mark.parametrize("case", list(_CLASSIFIED))
def test_classification_covers_declared_shapes_only(case: str) -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(6)
    decodes, prefills, fresh, options = _CLASSIFIED[case]
    rows = _rows(decodes, prefills, fresh, num_pool_blocks=6, **options)
    assert rows[2].live_block_ids.tolist() == [1, 2, 3, 4, 5]
    expected = (_CELLS[case], None) if case in _CELLS else (None, case)
    assert tg.classify_transaction(_prepared(core, pools, call, rows)) == expected


@pytest.mark.cpu_test
def test_replay_matches_eager_across_varied_plans_on_cpu() -> None:
    _replay_parity(CPU, _rerun_runtime())


@pytest.mark.cpu_test
@pytest.mark.parametrize(("population", "chunks"), _POPULATION_CHUNKS)
def test_population_lockstep_replays_every_chunk_on_cpu(
    population: int, chunks: int
) -> None:
    _population_parity(CPU, _rerun_runtime(), population, chunks=chunks)


def _corrupt_row_in_population(
    device: torch.device, runtime: GraphRuntime, population: int
) -> None:
    """One out-of-sequence row in a continuing population is rejected alone.

    Collection reports the same failed request under replay as under eager,
    and the rejection neither raises nor latches the binding.
    """
    core = _core(device)
    call = _call_bindings(core)
    eager_pools = _endpoint_pools(population + 1, device)
    graph_pools = _clone_pools(eager_pools)
    binding = tg.TransactionGraphBinding(
        populations=[population],
        geometries=(GEOM_REG,),
        maximum_population=population,
        vllm_config=object(),
        runtime=runtime,
    )
    if device.type == "cuda":
        advance.warmup_advance_model_rows_scatter(**graph_pools)
    binding.capture(
        core,
        pools=_transaction_pools(graph_pools),
        call=call,
        hidden_size=CARRIER_HIDDEN,
        device=device,
    )
    blocks = list(range(1, population + 1))
    bad = population // 2

    def run(pools: Pools, graph: Any) -> tuple[list[Any], list[set[str]]]:
        records, failures = [], []
        for seq in range(2):
            rows = []
            for index in range(population):
                stamped = seq + 3 if seq and index == bad else seq
                rows.append(
                    _chunk(
                        7 * seq + index,
                        seq=stamped,
                        geometry=GEOM_REG,
                        final=False,
                        prompt=0,
                    )
                )
            plan = _plan(
                prefills=blocks,
                num_pool_blocks=population + 1,
                has_initial=[seq > 0] * population,
            )
            before = _clone_pools(pools)
            record, failed = _collected_step(
                core,
                pools,
                call,
                ([PLACEHOLDER_ID] * population, torch.stack(rows).to(device), plan),
                graph,
            )
            records.append(record)
            failures.append(failed)
            if seq:
                for key in pools:
                    values = (
                        pools[key] if isinstance(pools[key], list) else [pools[key]]
                    )
                    olds = (
                        before[key] if isinstance(before[key], list) else [before[key]]
                    )
                    for value, old in zip(values, olds, strict=True):
                        torch.testing.assert_close(
                            value[blocks[bad]], old[blocks[bad]], atol=0, rtol=0
                        )
            # Drain echoes eagerly between CHUNKs (not under test here).
            tokens = records[-1][0][:, 0].long().tolist()
            for _ in range(CAP + 2):
                active = [
                    index
                    for index, token in enumerate(tokens)
                    if token != PARK_ID and not (seq and index == bad)
                ]
                if not active:
                    break
                out = _step(
                    core,
                    pools,
                    call,
                    [tokens[index] for index in active],
                    torch.zeros(len(active), CARRIER_HIDDEN, device=device),
                    _plan(
                        decodes=[blocks[index] for index in active],
                        num_pool_blocks=population + 1,
                    ),
                    None,
                )
                for row, index in enumerate(active):
                    tokens[index] = int(out[0][row, 0])
        return records, failures

    expected, expected_failed = run(eager_pools, None)
    actual, actual_failed = run(graph_pools, binding)
    _assert_records_equal(actual, expected)
    _assert_pools_equal(graph_pools, eager_pools)
    assert actual_failed == expected_failed == [set(), {f"req-{bad}"}]
    status = expected[1][1]
    assert int(status[bad]) != 0
    assert not bool(status[torch.arange(population) != bad].any())
    counts = binding.counters()
    assert counts["fallbacks"] == 0
    assert counts[f"replays:chunk-continuing:g0:b{population}"] == 1
    _collected_step(
        core,
        graph_pools,
        call,
        _rows(
            [],
            blocks,
            [False] * population,
            num_pool_blocks=population + 1,
            device=device,
        ),
        binding,
    )
    assert binding.counters()["replays"] == counts["replays"] + 1, "latched"


@pytest.mark.cpu_test
@pytest.mark.parametrize("population", [31, 63])
def test_one_corrupt_row_in_a_population_is_rejected_alone_on_cpu(
    population: int,
) -> None:
    _corrupt_row_in_population(CPU, _rerun_runtime(), population)


@pytest.mark.cpu_test
def test_capture_input_stores_no_page_and_publishes_every_cell() -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    for tensor in (pools["book_pool"], pools["queue_pool"]):
        tensor.fill_(3)
    before = _clone_pools(pools)
    launches: list[tuple[int, str]] = []
    binding = _binding(
        core, pools, call, _logging_runtime(launches), populations=(1, 2)
    )
    _assert_pools_equal(pools, before)
    assert binding.ready
    assert [cell.label for cell in binding.cells] == [
        "chunk-continuing:g0:b1",
        "chunk-continuing:g0:b2",
        "chunk-fresh:g0:b1",
        "chunk-fresh:g0:b2",
        "replay:b1",
        "replay:b2",
    ]
    wrappers = sorted({index for index, _mode in launches})
    assert len(wrappers) == len(binding.cells)
    for wrapper in wrappers:
        modes = [mode for index, mode in launches if index == wrapper]
        assert modes == ["none"] * 3 + ["graph"] * 2, wrapper
    assert advance.consume_batch_stats() is None


@pytest.mark.cpu_test
def test_capture_self_check_mismatch_publishes_nothing() -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    before = _clone_pools(pools)
    runtime = _logging_runtime([], corrupt=lambda _index: True)
    binding = tg.TransactionGraphBinding(
        populations=[1, 2],
        geometries=(GEOM_REG,),
        maximum_population=4,
        vllm_config=object(),
        runtime=runtime,
    )
    with pytest.raises(RuntimeError) as raised:
        _serve(binding, core, call, _transaction_pools(pools))
    assert any(cell.label in str(raised.value) for cell in binding.cells)
    assert not binding.ready
    assert binding.counters()["captured_cells"] == 0
    assert not any(entry["captured"] for entry in binding.receipt()["cells"])
    _assert_pools_equal(pools, before)


@pytest.mark.cpu_test
def test_cells_share_no_arena_static_input_or_owned_output() -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    binding = _binding(core, pools, call, _rerun_runtime())
    storages: list[int] = []
    for graph in binding._graphs.values():
        for tensor in (
            graph.arena._device,
            *graph.arena._host,
            graph.static_ids,
            graph.static_embeds,
            graph.rows_out,
            graph.status_out,
            graph.progress_out,
        ):
            assert tensor.numel() > 0
            storages.append(tensor.untyped_storage().data_ptr())
    assert len(binding._graphs) == len(binding.cells) > 1
    assert len(set(storages)) == len(storages)


@pytest.mark.cpu_test
def test_receipt_captures_largest_population_first_and_reports_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(5)
    binding = tg.TransactionGraphBinding(
        populations=[1, 4, 2],
        geometries=(GEOM_REG,),
        maximum_population=4,
        vllm_config=object(),
        runtime=_rerun_runtime(),
    )
    stand_in = _StandIn()
    admitted, served = 1 << 20, (1 << 20) - 4096
    _free_memory(
        monkeypatch,
        lambda index: _budget(
            stand_in.nbytes[0] if stand_in.nbytes else 0, admitted, served
        )(index),
    )
    _admit(binding, core, call, stand_in)
    assert stand_in.slots == [5]
    binding.capture(
        core,
        pools=_transaction_pools(pools),
        call=call,
        hidden_size=CARRIER_HIDDEN,
        device=CPU,
    )
    receipt = binding.receipt()
    cells = {entry["cell"]: entry for entry in receipt["cells"]}
    assert len(cells) == 9 and all(entry["captured"] for entry in cells.values())
    order = sorted(cells.values(), key=lambda entry: entry["capture_index"])
    populations = [int(entry["cell"].rsplit(":b", 1)[1]) for entry in order]
    assert populations == sorted(populations, reverse=True)
    assert sorted(entry["capture_index"] for entry in order) == list(range(9))
    for entry in order:
        assert entry["warmup_seconds"] > 0 and entry["capture_seconds"] >= 0
        assert entry["retained_bytes"] is None  # no device allocator on CPU
        assert entry["arena_bytes"] > 0
    assert receipt["retained_bytes_total"] is None
    assert receipt["ready"] is True
    assert receipt["admission_bytes"] == admitted
    assert receipt["serving_bytes"] == served
    assert receipt["admission_release_verified"] is True
    assert receipt["admission_capture_seconds"] > 0
    assert receipt["serving_capture_seconds"] > 0
    assert receipt["admission_capture_order"] == [entry["cell"] for entry in order]


def _first_step(
    graph: Any,
    pools: Pools,
    *,
    core: Any,
    call: tg.TransactionCallBindings,
    blocks: list[int],
    fresh: list[bool],
    capture: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embeds = torch.stack(
        [
            _chunk(5 + i, seq=0, geometry=GEOM_REG, final=False, prompt=0)
            for i in range(len(blocks))
        ]
    ).to(pools["book_pool"].device)
    plan = _plan(
        prefills=blocks,
        num_pool_blocks=4,
        has_initial=[not value for value in fresh],
    )
    return _step(
        core,
        pools,
        call,
        [PLACEHOLDER_ID] * len(blocks),
        embeds,
        plan,
        graph,
        capture=capture,
    )


#: Uncovered calls on ``_binding``'s cells, every reachable reason:
#: ``(decodes, prefills, fresh, options, key)``.
_UNCOVERED = {
    "mixed-fresh": (
        [],
        [1, 2],
        [True, False],
        {},
        "fallbacks:uncovered:mixed-fresh",
    ),
    "mixed-roles": ([2], [1], [True], {}, "fallbacks:uncovered:mixed-roles"),
    "multi-geometry": (
        [],
        [1, 2],
        [True, True],
        {"geometries": [GEOM_REG, GEOM_FINAL]},
        "fallbacks:uncovered:multi-geometry",
    ),
    "fresh-non-chunk": (
        [],
        [1],
        [True],
        {"chunk": [False]},
        "fallbacks:uncovered:fresh-non-chunk",
    ),
    "undeclared": (
        [],
        [1, 2, 3],
        [True] * 3,
        {},
        "fallbacks:chunk-fresh:g0:b3:undeclared",
    ),
    "capture": ([], [1], [True], {}, "fallbacks:chunk-fresh:g0:b1:capture"),
    "memory-profile": (
        [],
        [1],
        [True],
        {},
        "fallbacks:chunk-fresh:g0:b1:memory-profile",
    ),
}


@pytest.mark.cpu_test
@pytest.mark.parametrize("case", list(_UNCOVERED))
def test_uncovered_calls_run_eager_and_are_counted(case: str) -> None:
    core = _core()
    call = _call_bindings(core)
    graph_pools = _endpoint_pools(4)
    eager_pools = _clone_pools(graph_pools)
    binding = _binding(core, graph_pools, call, _rerun_runtime())
    for pools in (graph_pools, eager_pools):
        pools["book_pool"][1:, _BOOK["last_label"]] = core.blank_id
    decodes, prefills, fresh, options, key = _UNCOVERED[case]
    ids, embeds, plan = _rows(decodes, prefills, fresh, **options)
    flags = {"capture": case == "capture", "memory_profile": case == "memory-profile"}
    before = binding.counters()
    expected = _step(core, eager_pools, call, ids, embeds, plan, None, **flags)
    actual = _step(core, graph_pools, call, ids, embeds, plan, binding, **flags)
    _assert_records_equal([actual], [expected])
    _assert_pools_equal(graph_pools, eager_pools)
    counts = binding.counters()
    assert counts[key] == before.get(key, 0) + 1
    assert counts["fallbacks"] == before["fallbacks"] + 1
    assert {k: v for k, v in counts.items() if k.startswith("replays")} == {
        k: v for k, v in before.items() if k.startswith("replays")
    }
    assert not any(name.endswith((":uncaptured", ":native-burst")) for name in counts)


def _arena_snapshot(
    binding: tg.TransactionGraphBinding,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Every published arena buffer (device and both pinned host) and a copy."""
    return [
        (buffer, buffer.clone())
        for graph in binding._graphs.values()
        for buffer in (graph.arena._device, *graph.arena._host)
    ]


@pytest.mark.cpu_test
def test_relocated_pools_fall_back_instead_of_replaying() -> None:
    core = _core()
    call = _call_bindings(core)
    captured_pools = _endpoint_pools(4)
    binding = _binding(core, captured_pools, call, _rerun_runtime())
    captured = binding.counters()["captured_cells"]
    arenas = _arena_snapshot(binding)
    moved = _clone_pools(captured_pools)
    reference = _clone_pools(captured_pools)
    kwargs = dict(core=core, call=call, blocks=[1], fresh=[True])
    actual = _first_step(binding, moved, **kwargs)
    expected = _first_step(None, reference, **kwargs)
    _assert_records_equal([actual], [expected])
    _assert_pools_equal(moved, reference)
    _assert_pools_equal(captured_pools, _endpoint_pools(4))
    assert all(torch.equal(buffer, saved) for buffer, saved in arenas)
    assert binding.counters()["fallbacks:chunk-fresh:g0:b1:signature"] == 1
    assert binding.counters()["captured_cells"] == captured


def _extend_arena_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every later call one more staged vector than its cell's arena."""
    staged = advance.TransactionInputs.staged_vectors

    def extended(self: Any) -> tuple[tuple[str, torch.Tensor], ...]:
        return (*staged(self), ("layout-probe", torch.zeros(1, dtype=torch.int64)))

    monkeypatch.setattr(advance.TransactionInputs, "staged_vectors", extended)


def _signature(base: SimpleNamespace, **changes: Any) -> tuple[Any, ...]:
    """The signature of ``base``'s call with ``changes`` to its fields."""
    call = {**vars(base), **changes}
    return tg.transaction_signature(
        call["core"],
        call["inputs"],
        call["ids"],
        call["embeds"],
        graph_covers_decode=False,
    )


def _with_bucket(inputs: Any, **changes: Any) -> Any:
    return replace(inputs, buckets=tuple(replace(b, **changes) for b in inputs.buckets))


def _with_book(inputs: Any, book: torch.Tensor) -> Any:
    return replace(inputs, pools=replace(inputs.pools, book=book))


def _prepared_call(core: Any, pools: Pools, call: Any, rows: Rows) -> dict[str, Any]:
    ids, embeds, _plan_rows = rows
    return {
        "inputs": _prepared(core, pools, call, rows),
        "ids": torch.tensor(ids, dtype=torch.long),
        "embeds": embeds,
    }


def _resize_lazy_buffer(core: Any) -> dict[str, Any]:
    pe = core.encoder.pos_enc.pe
    core.encoder.pos_enc.pe = torch.zeros(
        pe.shape[0], pe.shape[1] + 8, pe.shape[2], dtype=pe.dtype
    )
    return {}


#: One structural input changed per case, as ``base -> field changes``.
_STRUCTURAL: dict[str, Callable[[SimpleNamespace], dict[str, Any]]] = {
    "population": lambda b: _prepared_call(b.core, b.pools, b.call, _rows([], [1])),
    "placeholder-id": lambda b: {
        "inputs": replace(b.inputs, placeholder_id=PLACEHOLDER_ID + 7)
    },
    "park-id": lambda b: {"inputs": replace(b.inputs, park_id=PARK_ID + 7)},
    "eou-token-id": lambda b: {"inputs": replace(b.inputs, eou_token_id=EOU + 1)},
    "endpoint-enabled": lambda b: {"inputs": replace(b.inputs, endpoint_enabled=False)},
    "n-fresh": lambda b: _prepared_call(
        b.core, b.pools, b.call, _rows([], [1, 2], [True, False])
    ),
    "decode-callable": lambda b: {
        "inputs": _with_bucket(
            b.inputs,
            resolved=replace(
                b.inputs.buckets[0].resolved, decode_fn=rnnt.decode_dense_masked
            ),
        )
    },
    "transition-callable": lambda b: {
        "inputs": _with_bucket(b.inputs, transition=lambda *_a, **_k: None)
    },
    "pool-address": lambda b: {
        "inputs": replace(b.inputs, pools=_transaction_pools(_clone_pools(b.pools)))
    },
    "pool-shape": lambda b: {"inputs": _with_book(b.inputs, b.pools["book_pool"][:-1])},
    "pool-stride": lambda b: {
        "inputs": _with_book(
            b.inputs,
            b.pools["book_pool"].as_strided(
                b.pools["book_pool"].shape, (1, b.pools["book_pool"].shape[0])
            ),
        )
    },
    "encoder-buffer": lambda b: _resize_lazy_buffer(b.core),
    "input-shape": lambda b: {"embeds": torch.zeros(2, CARRIER_HIDDEN + 1)},
}


@pytest.mark.cpu_test
@pytest.mark.parametrize("change", [*_STRUCTURAL, "arena-layout", "arena-data"])
def test_signature_covers_every_structural_input(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    base = SimpleNamespace(
        core=core,
        pools=pools,
        call=call,
        **_prepared_call(core, pools, call, _rows([], [1, 2])),
    )
    expected = _signature(base)
    if change == "arena-data":
        ids = [PLACEHOLDER_ID] * 2
        plan = _plan(prefills=[3, 1], num_pool_blocks=4, prompts=[2, 1])
        embeds = torch.stack(
            [
                _chunk(seed, seq=0, geometry=GEOM_REG, final=False, prompt=prompt)
                for seed, prompt in ((41, 2), (43, 1))
            ]
        )
        moved = _prepared_call(core, pools, call, (ids, embeds, plan))
        assert _signature(base, **moved) == expected
        return
    if change == "arena-layout":
        _extend_arena_layout(monkeypatch)
        assert _signature(base) != expected
        return
    assert _signature(base, **_STRUCTURAL[change](base)) != expected


@pytest.mark.cpu_test
def test_arena_layout_mismatch_is_a_signature_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _core()
    call = _call_bindings(core)
    graph_pools = _endpoint_pools(4)
    binding = _binding(core, graph_pools, call, _rerun_runtime())
    eager_pools = _clone_pools(graph_pools)
    for pools in (graph_pools, eager_pools):
        pools["book_pool"][1:, _BOOK["last_label"]] = core.blank_id
    arenas = _arena_snapshot(binding)
    captured = binding.counters()["captured_cells"]
    _extend_arena_layout(monkeypatch)
    kwargs = dict(core=core, call=call, blocks=[1], fresh=[True])
    expected = _first_step(None, eager_pools, **kwargs)
    actual = _first_step(binding, graph_pools, **kwargs)
    _assert_records_equal([actual], [expected])
    _assert_pools_equal(graph_pools, eager_pools)
    assert all(torch.equal(buffer, saved) for buffer, saved in arenas)
    counts = binding.counters()
    assert counts["fallbacks:chunk-fresh:g0:b1:signature"] == 1
    assert counts["fallbacks"] == 1 and counts["replays"] == 0
    assert counts["captured_cells"] == captured


@pytest.mark.cpu_test
def test_signature_callable_identities_are_stable_across_calls() -> None:
    _, decode = _decode_calls()
    dense = DenseGraphBinding(
        decode_fn=decode,
        predictor=object(),
        joint=object(),
        vllm_config=object(),
        frame_widths=(2, 2),
        tiers=(1, 2),
        encoder_hidden=3,
        predictor_layers=2,
        predictor_hidden=5,
        blank_id=7,
        runtime=_decode_runtime(),
    )
    dense.warmup(CPU, torch.float32)
    decode_fn = dense.decode_fn(geometry=1, tier=1)
    assert all(dense.decode_fn(geometry=1, tier=1) is decode_fn for _ in range(3))

    core = _core()

    def sealed(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("signature checks never run the sealed cell")

    encoder = SimpleNamespace(
        ready=True,
        transition=lambda *_args, **_kwargs: None,
        _chunk_graph_entries={},
        reserve_chunk_graph_cells=lambda _cells: None,
    )
    chunk = ExactChunkGraphBinding(core, object(), encoder, dense, [1])
    encoder._chunk_graph_entries[execution_mode(core).graph_key("chunk", (1, 1))] = (
        sealed
    )

    def resolve(fn: Any) -> Any:
        return chunk.resolve(
            geometry=1,
            population=1,
            decode_fn=fn,
            encoder_transition=encoder.transition,
            capture=False,
        )

    assert all(resolve(dense.decode_fn(geometry=1, tier=1)) is sealed for _ in range(3))
    with pytest.raises(ValueError, match="sealed"):
        resolve(lambda *_args: None)

    def resolver(request: Any) -> Any:
        tier = dense.execution_tier(request.execution_batch_size)
        return advance.ResolvedDecode(
            arm="dense-graphed",
            decode_fn=dense.decode_fn(geometry=request.geometry, tier=tier),
        )

    call = replace(
        _call_bindings(core),
        decode_resolver=resolver,
        encoder_transition=encoder.transition,
        bucket_transition=chunk,
        graph_covers_decode=True,
    )
    pools = _endpoint_pools(4)
    signatures = []
    for block in (1, 2):
        ids, embeds, plan = _rows([], [block], geometries=[1])
        inputs = _prepared(
            core, pools, call, (ids, embeds, replace(plan, execution_tier=1))
        )
        assert inputs.buckets[0].transition is sealed
        signatures.append(
            tg.transaction_signature(
                core,
                inputs,
                torch.tensor(ids, dtype=torch.long),
                embeds,
                graph_covers_decode=True,
            )
        )
    assert signatures[0] == signatures[1]


#: Per-operation TF32 enablement; each case flips one ``fp32_precision``.
_TF32_SETTINGS: dict[str, Callable[[], Any]] = {
    "tf32-matmul": lambda: torch.backends.cuda.matmul,
    "tf32-cudnn-conv": lambda: torch.backends.cudnn.conv,
    "tf32-cudnn-rnn": lambda: torch.backends.cudnn.rnn,
}


@contextmanager
def _changed_process_constant(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Change the process constant ``name`` for the scope."""
    if name == "batch-invariant":
        from vllm.model_executor.determinism import batch_invariant

        mode = batch_invariant._batch_invariant_MODE
        monkeypatch.setattr(batch_invariant, "_batch_invariant_MODE", not mode)
        yield
    elif name in _TF32_SETTINGS:
        # One API family, the per-operation settings ``exact_fp32`` sets;
        # PyTorch rejects mixing them with the legacy ``allow_tf32`` flags.
        target = _TF32_SETTINGS[name]()
        precision = target.fp32_precision
        target.fp32_precision = "ieee" if precision == "tf32" else "tf32"
        try:
            yield
        finally:
            target.fp32_precision = precision
    elif name == "float32_matmul_precision":
        precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision(
            "medium" if precision != "medium" else "highest"
        )
        try:
            yield
        finally:
            torch.set_float32_matmul_precision(precision)
    else:
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        monkeypatch.setenv(
            "CUBLAS_WORKSPACE_CONFIG", ":16:8" if workspace != ":16:8" else ":4096:8"
        )
        yield


#: Error-message pattern naming each process constant. The legacy setter
#: also changes ``torch.backends.cuda.matmul.fp32_precision``, so either name
#: identifies that change.
_PROCESS_CONSTANTS = {
    "batch-invariant": r"(?i)batch.invariant",
    **dict.fromkeys(_TF32_SETTINGS, r"(?i)fp32_precision"),
    "float32_matmul_precision": r"fp32_precision|float32_matmul_precision",
    "CUBLAS_WORKSPACE_CONFIG": r"CUBLAS_WORKSPACE_CONFIG",
}


def _legacy_precision_getters_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every legacy TF32 and matmul-precision getter raise.

    The per-operation ``fp32_precision`` settings stay readable.
    """

    def legacy(*_args: Any) -> Any:
        raise AssertionError("a legacy precision getter was called")

    monkeypatch.setattr(torch, "get_float32_matmul_precision", legacy)
    monkeypatch.setattr(torch._C, "_get_cublas_allow_tf32", legacy)
    monkeypatch.setattr(
        type(torch.backends.cudnn), "allow_tf32", property(legacy, legacy)
    )


#: Calls on ``_binding``'s cells: the covered call and every fallback path,
#: ``(decodes, prefills, fresh)``; ``signature`` runs over relocated pools.
_PATHS = {
    "replay": ([], [1], [True]),
    "uncovered": ([], [1, 2], [True, False]),
    "mixed-row": ([2], [1], [True]),
    "undeclared": ([], [1, 2, 3], [True] * 3),
    "signature": ([], [1], [True]),
}


def _path_pools(path: str, pools: Pools) -> Pools:
    """The live pools a ``_PATHS`` call runs over."""
    return _clone_pools(pools) if path == "signature" else pools


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("constant", "path"),
    [(constant, path) for constant in _PROCESS_CONSTANTS for path in _PATHS]
    + [("unchanged", "replay")],
)
def test_process_constant_change_is_fatal_on_every_path(
    monkeypatch: pytest.MonkeyPatch, constant: str, path: str
) -> None:
    """A changed constant is fatal; the guard reads only per-operation settings.

    The ``unchanged`` case makes every legacy getter raise from before capture,
    so a guard that snapshots or checks through them fails a covered call.
    """
    if constant == "unchanged":
        _legacy_precision_getters_raise(monkeypatch)
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    binding = _binding(core, pools, call, _rerun_runtime())
    if constant == "unchanged":
        _advance(core, pools, call, _rows(*_PATHS[path]), binding, _CommitRecorder())
        assert binding.counters()["replays:chunk-fresh:g0:b1"] == 1
        return
    live = _path_pools(path, pools)
    before = _clone_pools(live)
    counts = binding.counters()
    sink = _CommitRecorder()
    with (
        _changed_process_constant(constant, monkeypatch),
        pytest.raises(RuntimeError, match=_PROCESS_CONSTANTS[constant]),
    ):
        _advance(core, live, call, _rows(*_PATHS[path]), binding, sink)
    assert sink.log == []
    _assert_pools_equal(live, before)
    assert binding.counters() == counts


@pytest.mark.cpu_test
def test_replay_reserves_before_any_page_write_and_stages_after() -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    binding = _binding(core, pools, call, _rerun_runtime())
    untouched = _clone_pools(pools)
    seen: list[str] = []

    class Sink(_CommitRecorder):
        def reserve(self, plan: Any) -> Any:
            _assert_pools_equal(pools, untouched)
            seen.append("reserve")
            return super().reserve(plan)

    sink = Sink()
    _advance(core, pools, call, _rows([], [1]), binding, sink)
    assert seen == ["reserve"] and sink.log == ["reserve", "stage"]
    assert int(sink.staged[0][0]) == 0
    assert int(pools["frontend_counter_pool"][1].abs().sum()) > 0
    assert binding.counters()["replays:chunk-fresh:g0:b1"] == 1

    registry, leased, rows = _leased("right", 2)
    _advance(core, pools, call, rows, binding, leased)
    assert binding.counters()["replays:chunk-fresh:g0:b1"] == 2
    registry.prune(())  # the request is aborted after its replay
    reports, _records, _lease_ok = leased.collect()
    assert [report.request_id for report in reports] == ["right"]
    assert leased.log == ["reserve", "stage", "collect"]


def _fail(message: str) -> Callable[..., Any]:
    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(message)

    return fail


def _assert_latched(
    binding: tg.TransactionGraphBinding,
    core: Any,
    call: tg.TransactionCallBindings,
    pools: Pools,
    monkeypatch: pytest.MonkeyPatch,
    paths: tuple[str, ...] = ("replay", "uncovered"),
) -> None:
    """Later covered and fallback calls fail at the latch, counting nothing."""
    classified: list[Any] = []
    classify = tg.classify_transaction

    def tracked(inputs: Any) -> Any:
        classified.append(inputs)
        return classify(inputs)

    monkeypatch.setattr(tg, "classify_transaction", tracked)
    counts = binding.counters()
    before = _clone_pools(pools)
    for path in paths:
        sink = _CommitRecorder()
        with pytest.raises(RuntimeError, match="a fresh worker is required"):
            _advance(core, pools, call, _rows(*_PATHS[path]), binding, sink)
        assert sink.log == [], path
    assert not classified
    assert binding.counters() == counts
    _assert_pools_equal(pools, before)


@pytest.mark.cpu_test
def test_replay_failure_cancels_the_reservation_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    binding = _binding(core, pools, call, _rerun_runtime())
    graph = binding._graphs[tg.TransactionCell(tg.CELL_FRESH_CHUNK, GEOM_REG, 1)]
    graph.replay = _fail("replay failed")
    sink = _CommitRecorder()
    with pytest.raises(RuntimeError, match="replay failed"):
        _advance(core, pools, call, _rows([], [1]), binding, sink)
    assert sink.log == ["reserve", "cancel"]
    assert binding.counters()["replays"] == 0
    _assert_latched(binding, core, call, pools, monkeypatch)


@pytest.mark.cpu_test
def test_stage_failure_after_done_latches_the_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    binding = _binding(core, pools, call, _rerun_runtime())
    sink = _TrackedSink(fail_stage=True)
    with pytest.raises(RuntimeError, match="injected stage failure"):
        _advance(core, pools, call, _rows([], [1]), binding, sink)
    assert sink.log == ["reserve", "stage", "cancel"]
    _assert_latched(binding, core, call, pools, monkeypatch)


#: Worker-fatal sources; each latches the binding (Failure Containment).
_FATAL_SOURCES = (
    "replay",
    "stage",
    "process-constant",
    "wrapper-entry",
    "post-warmup",
    "changed-reinitialization",
)


@pytest.mark.cpu_test
@pytest.mark.parametrize("later", ["covered", "fallback"])
@pytest.mark.parametrize("source", _FATAL_SOURCES)
@pytest.mark.usefixtures("cpu_readings")
def test_every_fatal_source_latches_before_classification(
    monkeypatch: pytest.MonkeyPatch, source: str, later: str
) -> None:
    platform = _entry_runtime([])
    model, config = _model(monkeypatch, runtime=lambda: platform.runtime)
    model.prepare_execution_memory()
    state, slots = _model_state(model, config, num_blocks=4)
    state.initialize_state_slots(slots)
    binding = model._transaction_graph
    assert binding.ready
    pools = _pool_arguments(_resident(model, slots))
    call = _model_call(model)
    rows = _rows(*_PATHS["replay"])
    if source == "replay":
        cell = tg.TransactionCell(tg.CELL_FRESH_CHUNK, GEOM_REG, 1)
        binding._graphs[cell].replay = _fail("injected replay failure")
        with pytest.raises(RuntimeError, match="injected replay failure"):
            _advance(model.core, pools, call, rows, binding, _CommitRecorder())
    elif source == "stage":
        with pytest.raises(RuntimeError, match="injected stage failure"):
            _advance(
                model.core, pools, call, rows, binding, _TrackedSink(fail_stage=True)
            )
    elif source == "process-constant":
        with (
            _changed_process_constant("CUBLAS_WORKSPACE_CONFIG", monkeypatch),
            pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"),
        ):
            _advance(model.core, pools, call, rows, binding, _CommitRecorder())
    elif source == "wrapper-entry":
        platform.clear_all_graphs()
        with pytest.raises(RuntimeError, match=r"(?i)wrapper entry"):
            _advance(model.core, pools, call, rows, binding, _CommitRecorder())
    elif source == "post-warmup":
        verify = _seam(model, "verify_transaction_graphs")
        _resize_positional_buffer(model)
        with pytest.raises(RuntimeError, match="transaction_graph_populations"):
            verify()
    else:
        _other, moved = _model_state(model, config, num_blocks=4)
        with pytest.raises(RuntimeError, match=r"(?i)pool layout|call binding"):
            state.initialize_state_slots(moved)
    path = "replay" if later == "covered" else "uncovered"
    _assert_latched(binding, model.core, call, pools, monkeypatch, (path,))
    with pytest.raises(RuntimeError, match="a fresh worker is required"):
        state.prepare_step(
            SchedulerOutput.make_empty(), dummy_run=False, is_profile=False
        )


class _StagelessSink(_CommitRecorder):
    """A sink whose reservation lacks ``stage()``; reserving must cancel it."""

    def reserve(self, plan: Any) -> Any:
        return SimpleNamespace(cancel=super().reserve(plan).cancel)


#: Failure points before ``graph.replay()``, in Execute order.
_PRE_BOUNDARY = (
    "classification",
    "signature",
    "arena-upload",
    "static-input-copy",
    "escaped-output-allocation",
    "commit-reservation",
)


def _inject_pre_boundary_fault(
    fault: str, binding: tg.TransactionGraphBinding, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Make step ``fault`` of every later ``execute`` raise."""
    message = f"injected {fault} failure"
    if fault == "classification":
        monkeypatch.setattr(tg, "classify_transaction", _fail(message))
    elif fault == "signature":
        monkeypatch.setattr(tg, "transaction_signature", _fail(message))
    elif fault == "escaped-output-allocation":
        _seam(tg, "allocate_escaped_rows")
        monkeypatch.setattr(tg, "allocate_escaped_rows", _fail(message))
    for graph in binding._graphs.values():
        if fault == "arena-upload":
            monkeypatch.setattr(graph.arena, "upload", _fail(message))
        elif fault == "static-input-copy":
            monkeypatch.setattr(graph.static_ids, "copy_", _fail(message))


@pytest.mark.cpu_test
@pytest.mark.parametrize("fault", _PRE_BOUNDARY)
def test_pre_boundary_failure_keeps_baseline_and_cancels_ticket_once(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    binding = _binding(core, pools, call, _rerun_runtime())
    rows = _rows(*_PATHS["replay"])
    if fault == "commit-reservation":
        sink: _CommitRecorder = _StagelessSink()
        with pytest.raises(ValueError) as eager:
            _advance(core, _clone_pools(pools), call, rows, None, _StagelessSink())
        expected = (ValueError, str(eager.value))
    else:
        sink = _CommitRecorder()
        expected = (RuntimeError, f"injected {fault} failure")
    _inject_pre_boundary_fault(fault, binding, monkeypatch)
    baseline = _clone_pools(pools)
    counts = binding.counters()
    with pytest.raises(expected[0]) as raised:
        _advance(core, pools, call, rows, binding, sink)
    assert str(raised.value) == expected[1]
    _assert_pools_equal(pools, baseline)
    assert sink.cancels == int(fault == "commit-reservation")
    after = binding.counters()
    for name in counts:
        if name.startswith(("replays", "fallbacks")):
            assert after[name] == counts[name], name


@pytest.mark.cpu_test
def test_stale_lease_fails_before_replay_writes_a_page() -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    binding = _binding(core, pools, call, _rerun_runtime())
    before = _clone_pools(pools)
    with pytest.raises(ValueError, match="not registered"):
        _advance(
            core,
            pools,
            call,
            _rows([], [1]),
            binding,
            BoundedCommitSink(SessionRegistry(), max_rows=4),
        )
    _assert_pools_equal(pools, before)
    assert binding.counters()["replays"] == 0


@pytest.mark.cpu_test
def test_cleared_wrapper_entry_is_fatal_and_never_captures() -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    captures: list[Any] = []
    platform = _entry_runtime(captures)
    binding = _binding(core, pools, call, platform.runtime)
    started = len(captures)
    assert started == len(binding.cells)
    captured = binding.counters()["captured_cells"]
    platform.clear_all_graphs()

    moved, reference = _clone_pools(pools), _clone_pools(pools)
    kwargs = dict(core=core, call=call, blocks=[1], fresh=[True])
    actual = _first_step(binding, moved, **kwargs)
    expected = _first_step(None, reference, **kwargs)
    _assert_records_equal([actual], [expected])
    assert binding.counters()["fallbacks:chunk-fresh:g0:b1:signature"] == 1

    arenas = _arena_snapshot(binding)
    before = _clone_pools(pools)
    sink = _CommitRecorder()
    with pytest.raises(RuntimeError, match=r"(?i)wrapper entry"):
        _advance(core, pools, call, _rows(*_PATHS["replay"]), binding, sink)
    assert len(captures) == started
    assert binding.counters()["captured_cells"] == captured
    assert sink.log == []
    _assert_pools_equal(pools, before)
    assert all(torch.equal(buffer, saved) for buffer, saved in arenas)


@pytest.mark.cpu_test
@pytest.mark.parametrize("cause", ["abort", "replay-raised"])
def test_abort_after_replay_does_not_roll_back_pages(cause: str) -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    binding = _binding(core, pools, call, _rerun_runtime())
    registry, sink, rows = _leased("left", 1)
    written: list[Pools] = []
    if cause == "replay-raised":
        graph = binding._graphs[tg.TransactionCell(tg.CELL_FRESH_CHUNK, GEOM_REG, 1)]
        replay = graph.replay

        def fail() -> None:
            replay()
            written.append(_clone_pools(pools))
            raise RuntimeError("injected failure after the replay")

        graph.replay = fail
        with pytest.raises(RuntimeError, match="after the replay"):
            _advance(core, pools, call, rows, binding, sink)
        assert sink.log == ["reserve", "cancel"]
    else:
        _advance(core, pools, call, rows, binding, sink)
        written.append(_clone_pools(pools))
        registry.prune(())  # the request is aborted after its replay
        sink.collect()
        assert sink.log == ["reserve", "stage", "collect"]
    assert int(written[0]["frontend_counter_pool"][1].abs().sum()) > 0
    _assert_pools_equal(pools, written[0])


@pytest.mark.cpu_test
def test_collection_failure_after_replay_publishes_no_progress(
    monkeypatch: pytest.MonkeyPatch,
    cpu_readings: SimpleNamespace,
) -> None:
    cpu_readings.expected = 5
    model, state, slots = _verified_model(monkeypatch, num_blocks=4)
    binding = model._transaction_graph
    replays = binding.counters()["replays"]
    _chunk_forward(model, state, slots)
    assert binding.counters()["replays"] == replays + 1
    progress = model._registry._sessions["left"].progress_key
    monkeypatch.setattr(
        model._commit_sink, "collect", _fail("injected collection failure")
    )
    with pytest.raises(RuntimeError, match="injected collection failure"):
        state.postprocess_state(torch.tensor([0]), torch.tensor([1]))
    assert model.consume_committed_progress() == {}
    assert model._registry._sessions["left"].progress_key == progress
    # Collection runs outside ``execute``, so the binding is not latched.
    _first_step(
        binding,
        _pool_arguments(_resident(model, slots)),
        core=model.core,
        call=_model_call(model),
        blocks=[2],
        fresh=[True],
    )
    assert binding.counters()["replays"] == replays + 2


@pytest.mark.cpu_test
def test_failed_capture_publishes_nothing_and_every_call_runs_eager() -> None:
    core = _core()
    call = _call_bindings(core)

    def broken(*_args: Any) -> Any:
        raise RuntimeError("adapter blew up")

    broken_call = replace(call, adapter=broken)
    pools = _endpoint_pools(4)
    binding = tg.TransactionGraphBinding(
        populations=[1],
        geometries=(GEOM_REG,),
        maximum_population=4,
        vllm_config=object(),
        runtime=_rerun_runtime(),
    )
    with pytest.raises(RuntimeError, match="transaction_graph_populations"):
        binding.capture(
            core,
            pools=_transaction_pools(pools),
            call=broken_call,
            hidden_size=CARRIER_HIDDEN,
            device=CPU,
        )
    assert not binding.ready and binding.counters()["captured_cells"] == 0
    before = _clone_pools(pools)
    with pytest.raises(RuntimeError):
        _first_step(binding, pools, core=core, call=call, blocks=[1], fresh=[True])
    _assert_pools_equal(pools, before)
    counts = binding.counters()
    assert counts["fallbacks"] == 0 and counts["replays"] == 0


@pytest.mark.cpu_test
def test_replayed_rows_are_owned_by_the_caller() -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    binding = _binding(core, pools, call, _rerun_runtime())
    for tensor in (pools["book_pool"],):
        tensor[1:, _BOOK["last_label"]] = core.blank_id
    first = _first_step(binding, pools, core=core, call=call, blocks=[1], fresh=[True])
    kept = first[0].clone()
    _first_step(binding, pools, core=core, call=call, blocks=[2, 3], fresh=[True] * 2)
    graph = binding._graphs[tg.TransactionCell(tg.CELL_FRESH_CHUNK, GEOM_REG, 1)]
    assert first[0].data_ptr() != graph.rows_out.data_ptr()
    torch.testing.assert_close(first[0], kept, atol=0, rtol=0)


@pytest.mark.cpu_test
def test_population_declaration_is_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    for populations in ([], [0], [5], [1, 1], ["1"]):
        with pytest.raises(ValueError, match="transaction_graph_populations"):
            tg.TransactionGraphBinding(
                populations=populations,
                geometries=(0,),
                maximum_population=4,
                vllm_config=object(),
            )
    engine = {"max_num_seqs": 4, "max_num_batched_tokens": 2}
    model, _config = _model(
        monkeypatch, engine=engine, transaction_graph_populations=[1, 2]
    )
    assert model._transaction_graph is not None
    for populations in ([3], [5], [0], [2, 2], [True]):
        with pytest.raises(ValueError, match="transaction_graph_populations"):
            _model(
                monkeypatch, engine=engine, transaction_graph_populations=populations
            )


@pytest.mark.cpu_test
def test_counters_reach_the_model_from_an_engine_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    binding = _binding(core, pools, call, _rerun_runtime(), populations=(1,))
    pools["book_pool"][1:, _BOOK["last_label"]] = core.blank_id
    _first_step(binding, pools, core=core, call=call, blocks=[1], fresh=[True])
    _first_step(binding, pools, core=core, call=call, blocks=[2, 3], fresh=[True] * 2)

    def synchronized(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("transaction graph counters synchronized the device")

    for owner in (torch.cuda, torch.accelerator, torch.cuda.Stream, torch.cuda.Event):
        monkeypatch.setattr(owner, "synchronize", synchronized)
    expected = binding.counters()
    assert expected["captured_cells"] == len(binding.cells)
    assert expected["replays:chunk-fresh:g0:b1"] == 1
    assert expected["fallbacks:chunk-fresh:g0:b2:undeclared"] == 1
    assert expected["prologue_ns_total"] >= expected["prologue_ns_last"] > 0
    for role, target in _owners(SimpleNamespace(_transaction_graph=binding)).items():
        assert tg.transaction_graph_counters(target) == expected, role
    for role, target in _owners(SimpleNamespace(_transaction_graph=None)).items():
        assert tg.transaction_graph_counters(target) == {}, role


# ---- CPU: configuration envelope -------------------------------------------


@pytest.mark.cpu_test
@pytest.mark.parametrize("declared", [_ABSENT, []], ids=["absent", "empty"])
def test_option_off_constructs_no_binding(
    monkeypatch: pytest.MonkeyPatch, declared: Any
) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("the option is off, yet a transaction graph binding was used")

    monkeypatch.setattr(tg.TransactionGraphBinding, "__init__", refuse)
    monkeypatch.setattr(tg.TransactionGraphBinding, "capture", refuse)
    model, config = _model(monkeypatch, transaction_graph_populations=declared)
    assert model._transaction_graph is None
    graphs: list[Any] = []
    eager = advance.advance_model_rows

    def record(*args: Any, transaction_graph: Any = None, **kwargs: Any) -> Any:
        graphs.append(transaction_graph)
        return eager(*args, transaction_graph=transaction_graph, **kwargs)

    monkeypatch.setattr(advance, "advance_model_rows", record)
    model.prepare_execution_memory()
    state, slots = _model_state(model, config)
    state.initialize_state_slots(slots)
    emitted = _serve_session(model, state, slots)
    assert len(graphs) == len(emitted) > 1
    assert all(graph is None for graph in graphs)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "hf",
    [
        {"experimental_native_burst": True},
        {"decode_dispatch_arm": "compact-eager"},
    ],
    ids=["native-burst", "compact-decode"],
)
def test_option_requires_native_burst_off_and_dense_decode(
    monkeypatch: pytest.MonkeyPatch, hf: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="transaction_graph_populations"):
        _model(monkeypatch, **hf)


#: Runner graph configurations the option rejects: every outer mode but
#: ``NONE``, and multimodal-encoder graphs under ``NONE``.
_RUNNER_GRAPHS = {
    **{
        mode.name: {"cudagraph_mode": mode}
        for mode in CUDAGraphMode
        if mode != CUDAGraphMode.NONE
    },
    "mm-encoder": {"cudagraph_mm_encoder": True},
}


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "engine", list(_RUNNER_GRAPHS.values()), ids=list(_RUNNER_GRAPHS)
)
def test_option_rejects_every_outer_cudagraph_mode_but_none(
    monkeypatch: pytest.MonkeyPatch, engine: dict[str, Any]
) -> None:
    model, _config = _model(monkeypatch)
    assert model._transaction_graph is not None
    with pytest.raises(ValueError, match="transaction_graph_populations"):
        _model(monkeypatch, engine=engine)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "lookaheads",
    [[1], [0, 1], [3], _ABSENT, [0.5], [True], ["0"]],
    ids=["g1", "g0-g1", "g2", "every", "fractional", "bool", "string"],
)
def test_option_rejects_served_geometries_other_than_g0(
    monkeypatch: pytest.MonkeyPatch, lookaheads: Any
) -> None:
    with pytest.raises(ValueError, match="transaction_graph_populations"):
        _model(monkeypatch, supported_num_lookahead_tokens=lookaheads)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("override", "populations", "accepted"),
    [(3, [1, 2], True), (2, [1, 2], False), (2, [1], True)],
)
def test_population_declaration_respects_block_override(
    monkeypatch: pytest.MonkeyPatch,
    override: int,
    populations: list[int],
    accepted: bool,
) -> None:
    engine = {"num_gpu_blocks_override": override}
    if accepted:
        model, _config = _model(
            monkeypatch, engine=engine, transaction_graph_populations=populations
        )
        assert model._transaction_graph is not None
        return
    with pytest.raises(ValueError, match="transaction_graph_populations"):
        _model(monkeypatch, engine=engine, transaction_graph_populations=populations)


# ---- CPU: memory admission -------------------------------------------------


def _stand_in_bytes(model: Any) -> int:
    """The stand-in storage size: ``max(populations) + 1`` real pages."""
    spec = build_nemotron_persistent_state_spec(model.config)
    return spec.page_size_bytes * (max(model.config.transaction_graph_populations) + 1)


@pytest.mark.cpu_test
@pytest.mark.parametrize("declared", [_ABSENT, []], ids=["absent", "empty"])
def test_option_off_performs_no_admission(
    monkeypatch: pytest.MonkeyPatch, declared: Any
) -> None:
    from vllm.models.nemotron_asr import state_profile

    def refuse(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("the option is off, yet admission work ran")

    monkeypatch.setattr(tg, "normalized_free_memory", refuse, raising=False)
    monkeypatch.setattr(tg.TransactionGraphBinding, "admit", refuse, raising=False)
    monkeypatch.setattr(tg.TransactionGraphBinding, "_capture_cell", refuse)
    monkeypatch.setattr(state_profile, "allocate_state_scratch", refuse)
    model, _config = _model(monkeypatch, transaction_graph_populations=declared)
    model.prepare_execution_memory()
    assert model._execution_memory_prepared
    assert model._transaction_graph is None


@pytest.mark.cpu_test
def test_admission_captures_full_inventory_before_eager_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[Any] = []
    model, _config = _model(
        monkeypatch,
        runtime=lambda: _rerun_runtime(lambda _index: events.append("capture")),
    )
    binding = model._transaction_graph
    warmup = profile_execution.warmup_static_encoder_execution
    profile = profile_execution.run_persistent_state_profile

    def encoder(value: Any, *, device: torch.device) -> None:
        warmup(value, device=device)
        events.append("encoder")

    def profiled(value: Any, **kwargs: Any) -> Any:
        events.append("profile")
        return profile(value, **kwargs)

    monkeypatch.setattr(profile_execution, "warmup_static_encoder_execution", encoder)
    monkeypatch.setattr(profile_execution, "run_persistent_state_profile", profiled)
    _free_memory(
        monkeypatch,
        _budget(_stand_in_bytes(model), 1 << 20, 1 << 20),
        lambda index: events.append(("N", index)),
    )
    prepared = _track_capture_cells(monkeypatch, events)
    model.prepare_execution_memory()

    labels = sorted(cell.label for cell in binding.cells)
    assert sorted(record.cell.label for record in prepared) == labels
    assert events.count("capture") == len(labels)
    first, last = events.index(("N", 0)), events.index(("N", 2))
    assert events.index("encoder") < first
    assert all(
        first < index < last
        for index, event in enumerate(events)
        if event == "capture" or (isinstance(event, tuple) and event[0] == "prepare")
    )
    assert last < events.index("profile")
    layout = tuple(
        tuple(pool.shape[1:])
        for pool in _transaction_view(
            allocate_state_scratch(
                build_nemotron_persistent_state_spec(model.config), 1, CPU
            )
        ).sequence()
    )
    slots = max(model.config.transaction_graph_populations) + 1
    for record in prepared:
        assert all(shape[0] == slots for shape in record.pool_shapes)
        assert tuple(shape[1:] for shape in record.pool_shapes) == layout
    receipt = model.execution_profile_receipt()["transaction"]
    assert sorted(receipt["admission_capture_order"]) == labels


@pytest.mark.cpu_test
def test_normalized_readings_synchronize_collect_and_empty_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device_calls: list[str] = []

    def record(name: str, result: Any = None) -> Callable[..., Any]:
        def call(*_args: Any, **_kwargs: Any) -> Any:
            device_calls.append(name)
            return result

        return call

    reading = _seam(tg, "normalized_free_memory")
    for owner, name, event in (
        (torch.cuda, "synchronize", "synchronize"),
        (torch.accelerator, "synchronize", "synchronize"),
        (gc, "collect", "collect"),
        (torch.cuda, "empty_cache", "empty_cache"),
        (torch.cuda.memory, "empty_cache", "empty_cache"),
        (torch.accelerator, "empty_cache", "empty_cache"),
    ):
        monkeypatch.setattr(owner, name, record(event), raising=False)
    for owner in (torch.cuda, torch.cuda.memory):
        monkeypatch.setattr(owner, "mem_get_info", record("mem_get_info", (7, 9)))
    assert reading(torch.device("cuda", 0)) == 7
    assert [
        name
        for index, name in enumerate(device_calls)
        if index == 0 or name != device_calls[index - 1]
    ] == ["synchronize", "collect", "empty_cache", "mem_get_info"]
    monkeypatch.undo()

    core = _core()
    call = _call_bindings(core)
    stand_in = _StandIn()
    inputs = _track_capture_inputs(monkeypatch)
    prepared = _track_capture_cells(monkeypatch)
    observed: dict[int, tuple[int, int, int]] = {}

    def observe(index: int) -> None:
        observed[index] = (
            len(_live(inputs)),
            len(_live([ref for record in prepared for ref in record.callbacks])),
            len(_live(stand_in.storage)),
        )

    budget = _budget(4096, 1 << 20, 1 << 19)
    _free_memory(
        monkeypatch,
        lambda index: (
            budget(index) if index != 1 else budget(1) + 4096 - stand_in.nbytes[0]
        ),
        observe,
    )
    binding = _admission_binding()
    _admit(binding, core, call, stand_in)
    assert observed[0][2] == 0, "N0 is read before the stand-in exists"
    assert observed[1] == (0, 0, 1), "Na: temporaries cleared, stand-in retained"
    assert observed[2][2] == 0, "N1 is read after the stand-in is released"
    resident = _transaction_view(_scratch(3))
    _serve(binding, core, call, resident)
    receipt = binding.receipt()
    assert receipt["admission_bytes"] == budget(0) - budget(2)
    assert receipt["serving_bytes"] == budget(3) - budget(4)


@pytest.mark.cpu_test
@pytest.mark.parametrize("cause", ["readings", "owner"])
def test_unverified_admission_release_fails_startup(
    monkeypatch: pytest.MonkeyPatch, cause: str
) -> None:
    core = _core()
    call = _call_bindings(core)
    owner: list[Any] = []
    stand_in = _StandIn(owner=owner if cause == "owner" else None)
    short = 1 if cause == "readings" else 0

    def readings(index: int) -> int:
        released = stand_in.nbytes[0] - short if stand_in.nbytes else 0
        return _budget(released, 1 << 20, 1 << 20)(index)

    _free_memory(monkeypatch, readings)
    binding = _admission_binding()
    with pytest.raises(RuntimeError, match="transaction_graph_populations") as raised:
        _admit(binding, core, call, stand_in)
    if cause == "readings":
        message = str(raised.value)
        assert str(stand_in.nbytes[0] - 1) in message
        assert str(stand_in.nbytes[0]) in message
    else:
        assert owner, "the injected owner kept the stand-in storage"
    assert not binding.ready


@pytest.mark.cpu_test
def test_serving_capture_releases_admission_before_measuring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _core()
    call = _call_bindings(core)
    stand_in = _StandIn()
    events: list[Any] = []
    prepared = _track_capture_cells(monkeypatch, events)
    binding = _admission_binding()
    admitted: list[Any] = []

    def observe(index: int) -> None:
        events.append(("N", index))
        if index == 3:
            admitted.extend(_live([record.graph for record in prepared]))
            admitted.extend(_live([record.arena for record in prepared]))

    _free_memory(
        monkeypatch,
        lambda index: _budget(
            stand_in.nbytes[0] if stand_in.nbytes else 0, 1 << 20, 1 << 20
        )(index),
        observe,
    )
    _admit(binding, core, call, stand_in)
    count = len(binding.cells)
    assert len(prepared) == count
    resident = _transaction_view(_scratch(3))
    _serve(binding, core, call, resident)
    assert not admitted, "admission graphs and arenas outlived Ns0"
    serving = [
        index
        for index, event in enumerate(events)
        if isinstance(event, tuple) and event[0] == "prepare"
    ][count:]
    assert len(serving) == count
    assert events.index(("N", 3)) < serving[0]
    assert binding.ready


@pytest.mark.cpu_test
@pytest.mark.parametrize("excess", [0, 1])
def test_serving_bytes_above_admission_fail_startup(
    monkeypatch: pytest.MonkeyPatch, excess: int
) -> None:
    core = _core()
    call = _call_bindings(core)
    stand_in = _StandIn()
    admitted = 1 << 20
    _free_memory(
        monkeypatch,
        lambda index: _budget(
            stand_in.nbytes[0] if stand_in.nbytes else 0,
            admitted,
            admitted + excess,
        )(index),
    )
    binding = _admission_binding()
    _admit(binding, core, call, stand_in)
    resident = _transaction_view(_scratch(3))
    if not excess:
        _serve(binding, core, call, resident)
        assert binding.ready
        return
    with pytest.raises(RuntimeError, match="transaction_graph_populations") as raised:
        _serve(binding, core, call, resident)
    assert str(admitted) in str(raised.value)
    assert str(admitted + 1) in str(raised.value)
    assert not binding.ready
    assert binding.counters()["captured_cells"] == 0


@pytest.mark.cpu_test
def test_admission_holder_lives_until_serving_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, config = _model(monkeypatch)
    prepared = _track_capture_cells(monkeypatch)
    _free_memory(monkeypatch, _budget(_stand_in_bytes(model), 1 << 20, 1 << 20))
    count = len(model._transaction_graph.cells)
    during_profiles: list[int] = []
    profile = profile_execution.run_persistent_state_profile

    def profiled(value: Any, **kwargs: Any) -> Any:
        during_profiles.append(len(_live([record.arena for record in prepared])))
        return profile(value, **kwargs)

    monkeypatch.setattr(profile_execution, "run_persistent_state_profile", profiled)
    model.prepare_execution_memory()
    arenas = [record.arena for record in prepared]
    assert during_profiles and all(alive == count for alive in during_profiles)
    assert len(_live(arenas)) == count, "released before prepare returned"
    # The profile window closes and capacity is fixed outside the model.
    state, slots = _model_state(model, config)
    assert len(_live(arenas)) == count, "released before serving capture"
    state.initialize_state_slots(slots)
    assert not _live(arenas), "admission outlived serving capture"


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "override",
    [{"num_gpu_blocks_override": 3}, {"kv_cache_memory_bytes": 1 << 30}],
    ids=["block-override", "kv-bytes"],
)
@pytest.mark.parametrize("outcome", ["fits", "exceeds", "capture-fails"])
def test_override_capacity_reports_admission_and_wraps_failures(
    monkeypatch: pytest.MonkeyPatch, override: dict[str, int], outcome: str
) -> None:
    armed = {"value": False}

    def on_capture(_index: int) -> None:
        if armed["value"]:
            raise torch.OutOfMemoryError("injected serving capture failure")

    model, config = _model(
        monkeypatch, runtime=lambda: _rerun_runtime(on_capture), engine=override
    )
    admitted = 1 << 20
    _free_memory(
        monkeypatch,
        _budget(_stand_in_bytes(model), admitted, admitted + (outcome == "exceeds")),
    )
    cache = dict(vars(config.cache_config))
    model.prepare_execution_memory()
    assert model.execution_profile_receipt()["transaction"]["admission_bytes"] == (
        admitted
    )
    state, slots = _model_state(model, config)
    binding = model._transaction_graph
    if outcome == "fits":
        state.initialize_state_slots(slots)
        assert binding.ready
        receipt = model.execution_profile_receipt()["transaction"]
        assert receipt["serving_bytes"] == admitted
    else:
        armed["value"] = outcome == "capture-fails"
        with pytest.raises(RuntimeError, match="transaction_graph_populations"):
            state.initialize_state_slots(slots)
        assert not binding.ready
    assert vars(config.cache_config) == cache


@pytest.mark.cpu_test
def test_admission_oom_latches_and_reports_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[int] = []
    failing: list[int] = []

    def on_capture(index: int) -> None:
        if len(captured) == 2:
            failing.append(index)
            raise torch.OutOfMemoryError("injected admission capture OOM")
        captured.append(index)

    model, _config = _model(monkeypatch, runtime=lambda: _rerun_runtime(on_capture))
    prepared = _track_capture_cells(monkeypatch)
    base = 1 << 34
    _free_memory(monkeypatch, _readings(base, base - 111_111_111))
    _seam(tg, "device_memory_info")
    monkeypatch.setattr(
        tg, "device_memory_info", lambda _device: (222_222_222, 333_333_333)
    )
    with pytest.raises(RuntimeError, match="transaction_graph_populations") as raised:
        model.prepare_execution_memory()
    assert model._execution_memory_failed
    message = str(raised.value)
    for figure in ("111111111", "222222222", "333333333"):
        assert figure in message
    assert prepared[failing[0]].cell.label in message
    captured_labels = [prepared[index].cell.label for index in captured]
    assert all(label in message for label in captured_labels) or (
        f"{len(captured_labels)} " in message
    )
    assert "projected" not in message


@pytest.mark.cpu_test
def test_serving_capture_rejects_too_few_usable_slots() -> None:
    core = _core()
    call = _call_bindings(core)
    binding = tg.TransactionGraphBinding(
        populations=[3],
        geometries=(GEOM_REG,),
        maximum_population=3,
        vllm_config=object(),
        runtime=_rerun_runtime(),
    )
    pools = _endpoint_pools(3)
    with pytest.raises(
        (RuntimeError, ValueError), match="transaction_graph_populations"
    ):
        _serve(binding, core, call, _transaction_pools(pools))
    assert not binding.ready
    assert binding.counters()["captured_cells"] == 0


# ---- CPU: capture lifetime -------------------------------------------------


@pytest.mark.cpu_test
def test_serving_capture_runs_once_from_initialize_state_slots(
    monkeypatch: pytest.MonkeyPatch,
    cpu_readings: SimpleNamespace,
) -> None:
    cpu_readings.expected = 5
    model, config = _model(monkeypatch)
    model.prepare_execution_memory()
    binding = model._transaction_graph
    captures: list[int] = []
    readiness: list[bool] = []
    capture = tg.TransactionGraphBinding.capture
    capture_cell = tg.TransactionGraphBinding._capture_cell

    def tracked(self: Any, core: Any, *, pools: Any, **kwargs: Any) -> None:
        captures.append(int(pools.queue.data_ptr()))
        capture(self, core, pools=pools, **kwargs)

    def cell(self: Any, *args: Any, **kwargs: Any) -> Any:
        readiness.append(self.ready)
        return capture_cell(self, *args, **kwargs)

    monkeypatch.setattr(tg.TransactionGraphBinding, "capture", tracked)
    monkeypatch.setattr(tg.TransactionGraphBinding, "_capture_cell", cell)
    state, slots = _model_state(model, config)
    resident = slots.views[model._persistent_state_layer.prefix]._tensor
    state.initialize_state_slots(slots)
    assert len(captures) == 1
    start = resident.data_ptr()
    assert start <= captures[0] < start + resident.nbytes
    assert readiness and not any(readiness)
    assert binding.ready
    assert binding.counters()["captured_cells"] == len(binding.cells)
    _serve_session(model, state, slots)
    assert len(captures) == 1 and len(readiness) == len(binding.cells)
    assert binding.counters()["replays"] > 0


@pytest.mark.cpu_test
def test_identical_reinitialize_is_a_verified_no_op(
    monkeypatch: pytest.MonkeyPatch,
    cpu_readings: SimpleNamespace,
) -> None:
    cpu_readings.expected = 5
    model, config = _model(monkeypatch)
    model.prepare_execution_memory()
    state, slots = _model_state(model, config)
    state.initialize_state_slots(slots)
    binding = model._transaction_graph
    graphs = dict(binding._graphs)
    captured = binding.counters()["captured_cells"]
    order = {e["cell"]: e["capture_index"] for e in binding.receipt()["cells"]}
    prepared = _track_capture_cells(monkeypatch)
    state.initialize_state_slots(slots)
    assert not prepared
    assert binding._graphs.keys() == graphs.keys()
    assert all(binding._graphs[cell] is graph for cell, graph in graphs.items())
    assert binding.counters()["captured_cells"] == captured
    assert {e["cell"]: e["capture_index"] for e in binding.receipt()["cells"]} == order


def _rebound(fn: Callable[..., Any]) -> Callable[..., Any]:
    """A distinct callable with ``fn``'s behaviour, as a rebuilt binding is."""

    def rebound(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    return rebound


@pytest.mark.cpu_test
@pytest.mark.usefixtures("cpu_readings")
@pytest.mark.parametrize(
    "change", ["pool-layout", "decode-callable", "transition-callable"]
)
def test_changed_pool_layout_on_reinitialize_is_fatal(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    model, config = _model(monkeypatch)
    model.prepare_execution_memory()
    state, slots = _model_state(model, config, num_blocks=4)
    state.initialize_state_slots(slots)
    binding = model._transaction_graph
    again = slots
    if change == "pool-layout":
        _other, again = _model_state(model, config, num_blocks=4)
    elif change == "decode-callable":
        monkeypatch.setattr(model, "_decode_resolver", _rebound(model._decode_resolver))
    else:
        execution = model._encoder_execution
        monkeypatch.setattr(execution, "transition", _rebound(execution.transition))
    prepared = _track_capture_cells(monkeypatch)
    with pytest.raises(RuntimeError, match=r"(?i)pool layout|call binding"):
        state.initialize_state_slots(again)
    assert not prepared
    with pytest.raises(RuntimeError, match="a fresh worker is required"):
        _first_step(
            binding,
            _pool_arguments(_resident(model, slots)),
            core=model.core,
            call=_model_call(model),
            blocks=[1],
            fresh=[True],
        )


@pytest.mark.cpu_test
def test_published_cell_retains_no_capture_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _core()
    call = _call_bindings(core)
    stand_in = _StandIn()
    inputs = _track_capture_inputs(monkeypatch)
    prepared = _track_capture_cells(monkeypatch)
    observed: dict[int, dict[str, int]] = {}

    def observe(index: int) -> None:
        if index not in (1, 2):
            return
        observed[index] = {
            "inputs": len(_live(inputs)),
            "callbacks": len(
                _live([ref for record in prepared for ref in record.callbacks])
            ),
            "clones": _live_capture_clones(prepared),
            "views": len(_live(stand_in.views)),
            "storage": len(_live(stand_in.storage)),
        }

    _free_memory(
        monkeypatch,
        lambda index: _budget(
            stand_in.nbytes[0] if stand_in.nbytes else 0, 1 << 20, 1 << 20
        )(index),
        observe,
    )
    binding = _admission_binding()
    _admit(binding, core, call, stand_in)
    admission = list(prepared)
    assert admission
    cleared = {"inputs": 0, "callbacks": 0, "clones": 0, "views": 0}
    assert observed[1] == {**cleared, "storage": 1}, "Na: only the holder's storage"
    assert observed[2] == {**cleared, "storage": 0}, "N1: the holder was the owner"
    resident = _transaction_view(_scratch(3))
    _serve(binding, core, call, resident)
    assert binding.ready
    serving = prepared[len(admission) :]
    assert len(serving) == len(binding.cells)
    assert not _live(inputs)
    assert not _live([ref for record in prepared for ref in record.callbacks])
    assert _live_capture_clones(serving) == 0
    assert _live_capture_clones(admission) == 0

    leaked = serving[0]
    graph = leaked.graph()
    page = resident.sequence()[0][1]
    with _tracking_clones(leaked.clones):
        graph.page_snapshots = (
            page.clone(),
            torch.clone(page),
            torch.empty_like(page).copy_(page),
        )
    assert _live_capture_clones([leaked]) == 3, "a snapshot kept on a graph counts"
    del graph.page_snapshots
    with _tracking_clones(leaked.clones):
        kept = page.clone()
    views = {**graph.arena.views, "snapshot": kept}
    graph.arena.views = views
    del kept
    assert _live_capture_clones([leaked]) == 1, "a snapshot among arena views counts"
    graph.arena.views = {k: v for k, v in views.items() if k != "snapshot"}
    del views
    with _tracking_clones(leaked.clones):
        graph.arena._host.append(page.clone())
    with pytest.raises(AssertionError, match="two host buffers"):
        _live_capture_clones([leaked])
    del graph.arena._host[-1], graph


@pytest.mark.cpu_test
def test_admission_holder_is_never_published(monkeypatch: pytest.MonkeyPatch) -> None:
    core = _core()
    call = _call_bindings(core)
    stand_in = _StandIn()
    _free_memory(
        monkeypatch,
        lambda index: _budget(
            stand_in.nbytes[0] if stand_in.nbytes else 0, 1 << 20, 1 << 20
        )(index),
    )
    binding = _admission_binding()
    _admit(binding, core, call, stand_in)
    assert not binding.ready
    assert binding._graphs == {} and binding._pool_signature is None
    counts = binding.counters()
    assert counts["captured_cells"] == 0 and counts["replays"] == 0
    graph_pools = _pool_arguments(_transaction_view(_scratch(4)))
    eager_pools = _clone_pools(graph_pools)
    kwargs = dict(core=core, call=call, blocks=[1], fresh=[True])
    expected = _first_step(None, eager_pools, **kwargs)
    actual = _first_step(binding, graph_pools, **kwargs)
    _assert_records_equal([actual], [expected])
    _assert_pools_equal(graph_pools, eager_pools)
    counts = binding.counters()
    assert counts["replays"] == 0
    assert counts["fallbacks:chunk-fresh:g0:b1:uncaptured"] == 1


@pytest.mark.cpu_test
def test_discard_leaves_no_published_cell_or_arena() -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    binding = _binding(core, pools, call, _rerun_runtime())
    assert binding.ready
    owned = [
        weakref.ref(value)
        for graph in binding._graphs.values()
        for value in (
            graph,
            graph.arena,
            graph.static_ids,
            graph.static_embeds,
            graph.rows_out,
            graph.status_out,
            graph.progress_out,
        )
    ]
    binding.discard()
    assert not binding.ready
    assert binding.counters()["captured_cells"] == 0
    assert not any(entry["captured"] for entry in binding.receipt()["cells"])
    assert not _live(owned)


@pytest.mark.cpu_test
@pytest.mark.xfail(
    strict=True,
    reason="Deferred: core persistent-state obligation 6 (design: Core Obligations)",
)
@pytest.mark.usefixtures("cpu_readings")
def test_capture_reaches_pools_only_through_the_startup_accessor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, config = _model(monkeypatch)
    model.prepare_execution_memory()
    state, slots = _model_state(model, config)

    def private_views(self: StateSlotViews) -> dict[str, torch.Tensor]:
        caller = sys._getframe(1).f_globals.get("__name__", "")
        assert not caller.startswith("vllm.models.nemotron_asr"), (
            f"{caller} read the private views._views"
        )
        return self.__dict__["_views"]

    monkeypatch.setattr(
        StateSlotViews, "_views", property(private_views), raising=False
    )
    state.initialize_state_slots(slots)
    assert model._transaction_graph.ready


# ---- CPU: post-warmup verification -----------------------------------------


def _verified_model(
    monkeypatch: pytest.MonkeyPatch, *, num_blocks: int = 3, graphed: bool = False
) -> tuple[Any, NemotronASRModelState, StateSlotBindings]:
    """A served model with every cell captured over resident slots.

    ``graphed`` serves the graphed encoder and decode arms over the CPU
    re-executing runtime, so the outer body reads their entries' scratch.
    """
    hf: dict[str, Any] = {}
    if graphed:
        from vllm.models.nemotron_asr import decode_graph, encoder_execution

        monkeypatch.setattr(encoder_execution, "platform_graph_runtime", _rerun_runtime)
        monkeypatch.setattr(decode_graph, "platform_graph_runtime", _rerun_runtime)
        hf = {
            "encoder_execution_arm": "eager-graphed",
            "decode_dispatch_arm": "dense-graphed",
        }
    model, config = _model(monkeypatch, **hf)
    model.prepare_execution_memory()
    state, slots = _model_state(model, config, num_blocks=num_blocks)
    state.initialize_state_slots(slots)
    assert model._transaction_graph.ready
    return model, state, slots


def _resize_positional_buffer(model: Any) -> None:
    """Grow the encoder's lazily extended positional buffer, as a warmup may."""
    encoder = model.core.encoder
    pe = encoder.pos_enc.pe
    encoder.pos_enc.pe = torch.zeros(
        pe.shape[0], pe.shape[1] + 8, pe.shape[2], dtype=pe.dtype, device=pe.device
    )


#: Post-warmup changes to the live inputs of every cell signature and to
#: binding-external buffers, each of which the verification must reject.
_VERIFY_MISMATCHES = (
    "resized-buffer",
    "relocated-buffer",
    "rebound-encoder-transition",
    "relocated-pool-view",
    "lazy-projection",
    "relocated-encoder-entry",
    "relocated-decoder-entry",
)
#: The mismatches in nested-binding scratch, which need the graphed arms.
_NESTED_MISMATCHES = ("relocated-encoder-entry", "relocated-decoder-entry")


def _verification_mismatch(
    model: Any, slots: StateSlotBindings, kind: str
) -> Callable[[str], None]:
    """Change one live input after capture; return a check of the error message.

    ``relocated-buffer`` moves the positional buffer to a same-size copy, so
    only its address changes. ``rebound-encoder-transition`` rebinds the
    encoder transition every call binds; ``relocated-pool-view`` moves one
    resident pool view; ``lazy-projection`` binds the encoder's lazily prepared
    positional projections. The nested cases move one encoder graph entry's
    mel storage or one decode graph entry's frame input.
    """

    def names(*parts: str) -> Callable[[str], None]:
        def check(message: str) -> None:
            assert "transaction_graph_populations" in message
            assert any(
                cell.label in message for cell in model._transaction_graph.cells
            ), message
            for part in parts:
                assert part in message, message

        return check

    encoder = model.core.encoder
    if kind == "resized-buffer":
        _resize_positional_buffer(model)
        return names("encoder.pos_enc.pe")
    if kind == "relocated-buffer":
        encoder.pos_enc.pe = encoder.pos_enc.pe.clone()
        return names("encoder.pos_enc.pe")
    if kind == "rebound-encoder-transition":
        execution = model._encoder_execution
        transition = execution.transition
        execution.transition = lambda *args, **kwargs: transition(*args, **kwargs)
        return names("encoder transition")
    if kind == "relocated-pool-view":
        views = slots.views[model._persistent_state_layer.prefix]._views
        name = "encoder.layers.0.window.channel"
        views[name] = views[name].clone()
        return names("resident pool channel.0")
    if kind == "lazy-projection":
        assert encoder._stream_relative_position_lengths == ()
        with torch.inference_mode():
            encoder.prepare_stream_relative_position_projections(
                out_widths=(2,), cache_len=WINDOW, reference=encoder.pos_enc.pe
            )
        assert encoder._stream_relative_position_lengths
        return names("lazily bound buffer encoder.")
    if kind == "relocated-encoder-entry":
        entries = model._encoder_execution._graph_entries
        assert entries
        entry = next(iter(entries.values()))
        entry.mel = entry.mel.clone()
        return names("buffer encoder.graph.", ".mel")
    assert kind == "relocated-decoder-entry"
    entries = model._decode_graph_binding._entries
    assert entries
    entry = next(iter(entries.values()))
    entry.enc_frames = entry.enc_frames.clone()
    return names("buffer decode.graph.", ".enc_frames")


@pytest.mark.cpu_test
@pytest.mark.parametrize("mismatch", _VERIFY_MISMATCHES)
def test_post_warmup_verification_rejects_resized_external_buffer(
    monkeypatch: pytest.MonkeyPatch, cpu_readings: SimpleNamespace, mismatch: str
) -> None:
    cpu_readings.expected = 5
    model, _state, slots = _verified_model(
        monkeypatch, graphed=mismatch in _NESTED_MISMATCHES
    )
    verify = _seam(model, "verify_transaction_graphs")
    verify()
    check = _verification_mismatch(model, slots, mismatch)
    with pytest.raises(RuntimeError, match="transaction_graph_populations") as raised:
        verify()
    check(str(raised.value))


@pytest.mark.cpu_test
def test_first_real_step_verifies_once_before_pruning(
    monkeypatch: pytest.MonkeyPatch,
    cpu_readings: SimpleNamespace,
) -> None:
    cpu_readings.expected = 5
    model, state, slots = _verified_model(monkeypatch, num_blocks=4)
    events: list[str] = []
    verify = _seam(model, "verify_transaction_graphs")
    prune = model._registry.prune

    def verified() -> None:
        events.append("verify")
        verify()

    def pruned(live: Any) -> Any:
        events.append("prune")
        return prune(live)

    monkeypatch.setattr(model, "verify_transaction_graphs", verified)
    monkeypatch.setattr(model._registry, "prune", pruned)
    state.prepare_step(SchedulerOutput.make_empty(), dummy_run=True, is_profile=False)
    assert events == []
    state.prepare_step(SchedulerOutput.make_empty(), dummy_run=False, is_profile=False)
    assert events == ["verify", "prune"]
    state.prepare_step(SchedulerOutput.make_empty(), dummy_run=False, is_profile=False)
    assert events == ["verify", "prune", "prune"]

    binding = model._transaction_graph
    graph = binding._graphs[tg.TransactionCell(tg.CELL_FRESH_CHUNK, GEOM_REG, 1)]

    def fail(*_args: Any) -> None:
        raise RuntimeError("replay failed")

    graph.replay = fail
    with pytest.raises(RuntimeError, match="replay failed"):
        _first_step(
            binding,
            _pool_arguments(_resident(model, slots)),
            core=model.core,
            call=_model_call(model),
            blocks=[1],
            fresh=[True],
        )
    with pytest.raises(RuntimeError, match="a fresh worker is required"):
        state.prepare_step(
            SchedulerOutput.make_empty(), dummy_run=False, is_profile=False
        )
    assert events == ["verify", "prune", "prune"]


@pytest.mark.cpu_test
@pytest.mark.parametrize("mismatch", _VERIFY_MISMATCHES)
def test_first_real_step_mismatch_is_fatal_before_any_state_change(
    monkeypatch: pytest.MonkeyPatch, cpu_readings: SimpleNamespace, mismatch: str
) -> None:
    cpu_readings.expected = 5
    model, state, slots = _verified_model(
        monkeypatch, graphed=mismatch in _NESTED_MISMATCHES
    )
    pruned: list[Any] = []
    monkeypatch.setattr(model._registry, "prune", pruned.append)
    resident = slots.views[model._persistent_state_layer.prefix]._tensor
    before = resident.clone()
    check = _verification_mismatch(model, slots, mismatch)
    with pytest.raises(RuntimeError, match="transaction_graph_populations") as raised:
        state.prepare_step(
            SchedulerOutput.make_empty(), dummy_run=False, is_profile=False
        )
    check(str(raised.value))
    assert not pruned
    assert torch.equal(resident, before)
    with pytest.raises(RuntimeError, match="a fresh worker is required"):
        state.prepare_step(
            SchedulerOutput.make_empty(), dummy_run=False, is_profile=False
        )
    assert not pruned


# ---- CPU: observability ----------------------------------------------------


def _owners(model: Any) -> dict[str, Any]:
    """Every in-process owner of ``model``, keyed by role."""
    runner = SimpleNamespace(model=model)
    worker = SimpleNamespace(model_runner=runner)
    wrapper = SimpleNamespace(worker=worker)
    executor = SimpleNamespace(driver_worker=wrapper)
    return {
        "model": model,
        "runner": runner,
        "worker": worker,
        "worker-wrapper": wrapper,
        "executor": executor,
        "engine-core": SimpleNamespace(model_executor=executor),
    }


@pytest.mark.cpu_test
def test_option_off_receipt_has_no_transaction_graph_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _config = _model(monkeypatch, transaction_graph_populations=[])
    model.prepare_execution_memory()
    receipt = model.execution_profile_receipt()
    assert receipt["schema"] == "nemotron-execution-profile/1"
    assert "transaction" not in receipt


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "target",
    [
        object(),
        SimpleNamespace(model_executor=SimpleNamespace(workers=())),
        SimpleNamespace(model_executor=SimpleNamespace(driver_worker=None)),
    ],
    ids=["foreign", "multiprocess-executor", "remote-worker"],
)
def test_counters_raise_when_the_model_is_unreachable(target: Any) -> None:
    with pytest.raises(Exception, match=r"(?i)reach"):
        tg.transaction_graph_counters(target)


# ---- CUDA: real graph replay -----------------------------------------------


@pytest.fixture
def exact_fp32() -> Iterator[None]:
    """Disable TF32 so eager and replay both run IEEE FP32 kernels."""
    targets = (
        torch.backends.cuda.matmul,
        torch.backends.cudnn.conv,
        torch.backends.cudnn.rnn,
    )
    saved = [target.fp32_precision for target in targets]
    for target in targets:
        target.fp32_precision = "ieee"
    try:
        yield
    finally:
        for target, value in zip(targets, saved, strict=True):
            target.fp32_precision = value


@pytest.mark.nemotron_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.usefixtures("exact_fp32")
@torch.inference_mode()
def test_cuda_graph_replay_matches_eager_across_varied_plans() -> None:
    _replay_parity(torch.device("cuda"), _cuda_runtime())


@pytest.mark.nemotron_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.usefixtures("exact_fp32")
@torch.inference_mode()
@pytest.mark.parametrize("endpoint", [True, False], ids=["endpoint", "no-endpoint"])
@pytest.mark.parametrize(("population", "chunks"), _POPULATION_CHUNKS)
def test_cuda_population_lockstep_replays_every_chunk(
    population: int, chunks: int, endpoint: bool
) -> None:
    _population_parity(
        torch.device("cuda"),
        _cuda_runtime(),
        population,
        chunks=chunks,
        endpoint=endpoint,
    )


@pytest.mark.nemotron_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.usefixtures("exact_fp32")
@torch.inference_mode()
@pytest.mark.parametrize("population", [31, 63])
def test_cuda_one_corrupt_row_in_a_population_is_rejected_alone(
    population: int,
) -> None:
    _corrupt_row_in_population(torch.device("cuda"), _cuda_runtime(), population)


@pytest.mark.nemotron_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.usefixtures("exact_fp32")
@torch.inference_mode()
@pytest.mark.parametrize("corruption", ["sequence", "echo", "book"])
def test_cuda_graph_replay_rejects_corrupt_rows_like_eager(corruption: str) -> None:
    _corrupt_rows_parity(torch.device("cuda"), _cuda_runtime(), corruption)


def _corrupt_rows_parity(
    device: torch.device, runtime: GraphRuntime, corruption: str
) -> None:
    """A corrupt row is rejected under replay exactly as under eager.

    Collection reports the same failed request in both arms, and the step
    neither raises nor latches the binding.
    """
    core = _core(device)
    call = _call_bindings(core)
    eager_pools = _endpoint_pools(4, device)
    graph_pools = _clone_pools(eager_pools)
    binding = _binding(core, graph_pools, call, runtime, device=device)

    def run(pools: Pools, graph: Any) -> tuple[list[Any], set[str]]:
        records = [
            _first_step(graph, pools, core=core, call=call, blocks=[2], fresh=[True])
        ]
        token = int(records[0][0][0, 0])
        if corruption == "sequence":
            embeds = _chunk(9, seq=5, geometry=GEOM_REG, final=False, prompt=0)
            plan = _plan(prefills=[2], num_pool_blocks=4, has_initial=[True])
            ids, rows = [PLACEHOLDER_ID], embeds.unsqueeze(0).to(device)
        else:
            if corruption == "book":
                pools["book_pool"][2, _BOOK["queue_head"]] = CAP + 5
            plan = _plan(decodes=[2], num_pool_blocks=4)
            wrong = token + 1 if token + 1 < VOCAB else 0
            ids = [wrong if corruption == "echo" else token]
            rows = torch.zeros(1, CARRIER_HIDDEN, device=device)
        before = _clone_pools(pools)
        record, failed = _collected_step(core, pools, call, (ids, rows, plan), graph)
        records.append(record)
        _assert_pools_equal(pools, before)
        return records, failed

    expected, expected_failed = run(eager_pools, None)
    actual, actual_failed = run(graph_pools, binding)
    _assert_records_equal(actual, expected)
    assert int(expected[1][1][0]) != 0
    assert actual_failed == expected_failed == {"req-0"}
    _assert_pools_equal(graph_pools, eager_pools)
    counts = binding.counters()
    assert counts["fallbacks"] == 0
    _first_step(binding, graph_pools, core=core, call=call, blocks=[3], fresh=[True])
    assert binding.counters()["replays"] == counts["replays"] + 1, "latched"


@pytest.mark.nemotron_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.usefixtures("exact_fp32")
@torch.inference_mode()
def test_cuda_replayed_rows_survive_another_cells_replay() -> None:
    device = torch.device("cuda")
    core = _core(device)
    call = _call_bindings(core)
    pools = _endpoint_pools(4, device)
    binding = _binding(core, pools, call, _cuda_runtime(), device=device)
    first = _first_step(binding, pools, core=core, call=call, blocks=[1], fresh=[True])
    kept = first[0].clone()
    _first_step(binding, pools, core=core, call=call, blocks=[2, 3], fresh=[True] * 2)
    torch.cuda.synchronize()
    torch.testing.assert_close(first[0], kept, atol=0, rtol=0)
    assert binding.counters()["replays"] == 2


def _track_graph_captures(
    monkeypatch: pytest.MonkeyPatch, phase: dict[str, str | None]
) -> list[tuple[str | None, int, Any]]:
    """Record ``(phase, capture stream, graph pool)`` for every graph capture."""
    captures: list[tuple[str | None, int, Any]] = []
    begin = torch.cuda.CUDAGraph.capture_begin

    def track(self: Any, *args: Any, **kwargs: Any) -> Any:
        pool = args[0] if args else kwargs.get("pool")
        stream = torch.cuda.current_stream().cuda_stream
        captures.append((phase["value"], stream, None if pool is None else tuple(pool)))
        return begin(self, *args, **kwargs)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "capture_begin", track)
    return captures


def _cuda_inventories(
    monkeypatch: pytest.MonkeyPatch,
    *,
    on_read: Callable[[int], None] | None = None,
) -> SimpleNamespace:
    """Admit, then serve, populations ``(1, 2)`` with real CUDA graphs.

    Readings are the real normalized readings, observed by ``on_read``.
    Returns the binding, its runtime, core, call bindings and resident pools,
    and every graph capture as ``(phase, stream, pool)``.
    """
    device = torch.device("cuda")
    core = _core(device)
    call = _call_bindings(core)
    runtime = _cuda_runtime()
    phase: dict[str, str | None] = {"value": None}
    captures = _track_graph_captures(monkeypatch, phase)
    if on_read is not None:
        reading = _seam(tg, "normalized_free_memory")
        calls: list[int] = []

        def observed(target: torch.device) -> int:
            value = reading(target)
            on_read(len(calls))
            calls.append(value)
            return value

        monkeypatch.setattr(tg, "normalized_free_memory", observed)
    binding = _admission_binding(runtime=runtime)
    phase["value"] = "admission"
    _admit(binding, core, call, _StandIn(device), device)
    resident = _transaction_view(_scratch(4, device))
    advance.warmup_advance_model_rows_scatter(**_pool_arguments(resident))
    phase["value"] = "serving"
    _serve(binding, core, call, resident, device)
    phase["value"] = None
    assert binding.ready
    return SimpleNamespace(
        binding=binding,
        runtime=runtime,
        core=core,
        call=call,
        resident=resident,
        captures=captures,
    )


def _segment_pools() -> set[Any]:
    return {
        tuple(segment["segment_pool_id"])
        for segment in torch.cuda.memory._snapshot()["segments"]
    }


@pytest.mark.nemotron_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.usefixtures("exact_fp32")
@torch.inference_mode()
def test_cuda_admission_and_serving_capture_share_cell_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.platforms import current_platform

    served = _cuda_inventories(monkeypatch)
    binding, captures = served.binding, served.captures
    admission = [capture for capture in captures if capture[0] == "admission"]
    serving = [capture for capture in captures if capture[0] == "serving"]
    assert len(admission) == len(serving) == len(binding.cells)
    assert len({stream for _phase, stream, _pool in captures}) == 1
    admission_pools = {pool for _phase, _stream, pool in admission}
    serving_pools = {pool for _phase, _stream, pool in serving}
    assert len(admission_pools) == len(serving_pools) == 1
    assert admission_pools.isdisjoint(serving_pools)
    nested = {
        tuple(current_platform.get_global_graph_pool()),
        tuple(served.runtime.wrapper_factory(lambda: None).graph_pool),
    }
    assert admission_pools.isdisjoint(nested)
    assert serving_pools.isdisjoint(nested)
    receipt = binding.receipt()
    serving_order = [
        entry["cell"]
        for entry in sorted(receipt["cells"], key=lambda entry: entry["capture_index"])
    ]
    assert receipt["admission_capture_order"] == serving_order


@pytest.mark.nemotron_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.usefixtures("exact_fp32")
@torch.inference_mode()
def test_cuda_serving_pool_is_distinct_from_released_admission_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pools_at: dict[int, set[Any]] = {}
    captures = _cuda_inventories(
        monkeypatch, on_read=lambda index: pools_at.setdefault(index, _segment_pools())
    ).captures
    (admission,) = {pool for phase, _stream, pool in captures if phase == "admission"}
    (serving,) = {pool for phase, _stream, pool in captures if phase == "serving"}
    assert admission != serving
    assert admission in pools_at[1], "Na is read while the admission pool is held"
    assert admission not in pools_at[3], "Ns0 is read after the admission pool"
    assert serving not in pools_at[3]


@pytest.mark.nemotron_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.usefixtures("exact_fp32")
@torch.inference_mode()
def test_cuda_receipt_reports_admission_and_serving_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    served = _cuda_inventories(monkeypatch)
    binding = served.binding
    receipt = binding.receipt()
    assert receipt["ready"] is True
    assert all(receipt[name] is not None for name in _ADMISSION_FIELDS)
    admitted, serving = receipt["admission_bytes"], receipt["serving_bytes"]
    assert isinstance(admitted, int) and isinstance(serving, int)
    assert 0 < serving <= admitted
    assert receipt["admission_release_verified"] is True
    assert receipt["admission_capture_seconds"] > 0
    assert receipt["serving_capture_seconds"] > 0
    for entry in receipt["cells"]:
        assert entry["captured"] and entry["arena_bytes"] > 0
        assert isinstance(entry["retained_bytes"], int)
        assert entry["warmup_seconds"] > 0 and entry["capture_seconds"] > 0
        assert isinstance(entry["capture_index"], int)
    _first_step(
        binding,
        _pool_arguments(served.resident),
        core=served.core,
        call=served.call,
        blocks=[1],
        fresh=[True],
    )
    model = SimpleNamespace(_transaction_graph=binding)
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        counts = {
            role: tg.transaction_graph_counters(target)
            for role, target in _owners(model).items()
        }
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    for role, value in counts.items():
        assert value["replays:chunk-fresh:g0:b1"] == 1, role


@pytest.mark.nemotron_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.usefixtures("exact_fp32")
@torch.inference_mode()
def test_cuda_replay_freezes_nested_counters_and_counts_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.models.nemotron_asr import encoder_execution

    device = torch.device("cuda")
    monkeypatch.setattr(encoder_execution, "platform_graph_runtime", _cuda_runtime)
    model, config = _model(
        monkeypatch,
        runtime=_cuda_runtime,
        device=device,
        encoder_execution_arm="eager-graphed",
    )
    model.prepare_execution_memory()
    state, slots = _model_state(model, config, num_blocks=4)
    state.initialize_state_slots(slots)
    binding = model._transaction_graph
    assert binding.ready
    execution = model._encoder_execution
    pools = _pool_arguments(_resident(model, slots))
    call = _model_call(model)
    label = "replays:chunk-fresh:g0:b1"
    advance.consume_batch_stats()
    invocations = execution._invocations
    before = binding.counters()
    _first_step(binding, pools, core=model.core, call=call, blocks=[1], fresh=[True])
    after = binding.counters()
    assert execution._invocations == invocations
    assert advance.consume_batch_stats() is not None
    assert after[label] == before[label] + 1
    assert after["replays"] == before["replays"] + 1
    assert after["fallbacks"] == before["fallbacks"]
    _first_step(None, pools, core=model.core, call=call, blocks=[2], fresh=[True])
    assert execution._invocations == invocations + 1


@pytest.mark.nemotron_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_cuda_arena_reuses_pinned_buffers_after_async_upload() -> None:
    device = torch.device("cuda")
    core = _core(device)
    call = _call_bindings(core)
    pools = _endpoint_pools(4, device)
    calls = [_fresh_chunk_inputs(core, pools, call, [block]) for block in (1, 3, 2, 1)]
    arena = tg.TransactionArena(tg._arena_layout(calls[0]), device)
    assert all(buffer.is_pinned() for buffer in arena._host)
    uploaded = []
    for inputs in calls:
        arena.upload(inputs)
        uploaded.append({name: view.clone() for name, view in arena.views.items()})
    torch.cuda.synchronize()
    for inputs, views in zip(calls, uploaded, strict=True):
        for name, vector in inputs.staged_vectors():
            assert torch.equal(views[name].cpu(), vector.cpu()), name


def _control_trajectory(
    core: Any, pools: Pools, call: tg.TransactionCallBindings, graph: Any
) -> tuple[list[Any], dict[str, int]]:
    """One session through every accepted control row kind on slot 1.

    A fresh CHUNK drains to its park, then a park-echo and a forced-endpoint
    row; a final CHUNK drains to its park, then a FLUSH. Returns the records
    and each control row's record index. A rejected control row must leave
    every page unchanged and has no emission to echo.
    """
    device = pools["book_pool"].device
    records: list[Any] = []
    controls: dict[str, int] = {}

    def step(ids: list[int], embeds: torch.Tensor, plan: Any) -> int:
        records.append(_step(core, pools, call, ids, embeds, plan, graph))
        return int(records[-1][0][0, 0])

    def echo(token: int | None) -> None:
        for _ in range(CAP + 2):
            if token is None or token == PARK_ID:
                return
            token = step(
                [token],
                torch.zeros(1, CARRIER_HIDDEN, device=device),
                _plan(decodes=[1], num_pool_blocks=4),
            )
        raise AssertionError("replay queue did not drain")

    def chunk(seq: int, *, final: bool) -> None:
        embeds = _chunk(
            23 + seq,
            seq=seq,
            geometry=GEOM_REG,
            final=final,
            prompt=0,
            samples=REG_SAMPLES // 2 if final else None,
        )
        plan = _plan(prefills=[1], num_pool_blocks=4, has_initial=[seq > 0])
        echo(step([PLACEHOLDER_ID], embeds.unsqueeze(0).to(device), plan))

    def control(name: str, token: int) -> int | None:
        controls[name] = len(records)
        before = _clone_pools(pools)
        emitted = step(
            [token],
            torch.zeros(1, CARRIER_HIDDEN, device=device),
            _plan(decodes=[1], num_pool_blocks=4),
        )
        if int(records[-1][1][0]):
            _assert_pools_equal(pools, before)
            return None
        return emitted

    chunk(0, final=False)
    echo(control("park-echo", PARK_ID))
    echo(control("forced-endpoint", EOU))
    chunk(1, final=True)
    control("flush", PARK_ID)
    return records, controls


def _control_parity(
    device: torch.device, runtime: GraphRuntime, endpoint: bool
) -> None:
    """Accepted control rows replay the ``replay`` cell bitwise exactly."""
    core = _core(device)
    call = _call_bindings(core, endpoint=endpoint)
    eager_pools = _endpoint_pools(4, device, endpoint=endpoint)
    graph_pools = _clone_pools(eager_pools)
    binding = _binding(
        core, graph_pools, call, runtime, populations=(1,), device=device
    )
    expected, controls = _control_trajectory(core, eager_pools, call, None)
    actual, _ = _control_trajectory(core, graph_pools, call, binding)
    _assert_records_equal(actual, expected)
    _assert_pools_equal(graph_pools, eager_pools)
    statuses = {name: int(expected[index][1][0]) for name, index in controls.items()}
    assert statuses["park-echo"] == 0 and statuses["flush"] == 0
    assert (statuses["forced-endpoint"] == 0) is endpoint
    counts = binding.counters()
    assert counts["fallbacks"] == 0
    assert counts["replays"] == len(expected)
    assert counts["replays:replay:b1"] >= len(controls)


@pytest.mark.nemotron_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.usefixtures("exact_fp32")
@torch.inference_mode()
@pytest.mark.parametrize("endpoint", [True, False], ids=["endpoint", "no-endpoint"])
def test_cuda_replay_cell_accepted_control_rows_match_eager(endpoint: bool) -> None:
    _control_parity(torch.device("cuda"), _cuda_runtime(), endpoint)


def _storage(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


class _MutationStreams(TorchDispatchMode):
    """Log ``(name, current stream)`` for every op that writes a named tensor.

    An op writes an argument when its schema marks that argument mutable
    (in-place, ``out=`` or ``Tensor(a!)`` lists); the target is named by the
    ``names`` storage it shares. The stream is read at dispatch, so work an
    implementation moves to another stream is logged on that stream. Writes
    to other storage, such as host staging, are not logged.
    """

    def __init__(self, names: dict[int, str], log: list[tuple[str, int]]) -> None:
        super().__init__()
        self._names = names
        self._log = log

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        for index, argument in enumerate(func._schema.arguments):
            if argument.alias_info is None or not argument.alias_info.is_write:
                continue
            value = args[index] if index < len(args) else kwargs.get(argument.name)
            for tensor in value if isinstance(value, list | tuple) else (value,):
                if not isinstance(tensor, torch.Tensor):
                    continue
                name = self._names.get(_storage(tensor))
                if name is not None:
                    stream = torch.cuda.current_stream().cuda_stream
                    self._log.append((name, stream))
        return func(*args, **kwargs)


@contextmanager
def _stream_trace(
    monkeypatch: pytest.MonkeyPatch, targets: dict[str, list[torch.Tensor]]
) -> Iterator[list[tuple[str, int]]]:
    """Log the current stream of every graph replay and every write to ``targets``.

    A write into storage shared with a ``targets[name]`` tensor is logged as
    ``name`` through ``_MutationStreams``; each ``CUDAGraph.replay`` as
    ``"replay"``.
    """
    log: list[tuple[str, int]] = []
    names = {
        _storage(tensor): name
        for name, tensors in targets.items()
        for tensor in tensors
    }
    replay = torch.cuda.CUDAGraph.replay

    def traced_replay(graph: Any, *args: Any, **kwargs: Any) -> Any:
        log.append(("replay", torch.cuda.current_stream().cuda_stream))
        return replay(graph, *args, **kwargs)

    with monkeypatch.context() as patch, _MutationStreams(names, log):
        patch.setattr(torch.cuda.CUDAGraph, "replay", traced_replay)
        yield log


def _assert_on_stream(
    log: list[tuple[str, int]], stream: int, required: set[str]
) -> None:
    """Every logged operation ran on ``stream``, and each ``required`` ran."""
    assert required <= {name for name, _stream in log}, log
    assert all(issued == stream for _name, issued in log), log


@pytest.mark.nemotron_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.usefixtures("exact_fp32")
@torch.inference_mode()
def test_cuda_replay_runs_on_the_current_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    core = _core(device)
    call = _call_bindings(core)
    eager_pools = _endpoint_pools(4, device)
    graph_pools = _clone_pools(eager_pools)
    binding = _binding(core, graph_pools, call, _cuda_runtime(), device=device)
    kwargs = dict(core=core, call=call, blocks=[1], fresh=[True])
    expected = _first_step(None, eager_pools, **kwargs)
    torch.cuda.synchronize()
    graphs = list(binding._graphs.values())
    targets = {
        "arena": [graph.arena._device for graph in graphs],
        "static_ids": [graph.static_ids for graph in graphs],
        "static_embeds": [graph.static_embeds for graph in graphs],
        "resident": [tensor for tensor in _transaction_pools(graph_pools).sequence()],
    }
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream())
    with _stream_trace(monkeypatch, targets) as log, torch.cuda.stream(side):
        actual = _first_step(binding, graph_pools, **kwargs)
    side.synchronize()
    _assert_records_equal([actual], [expected])
    _assert_pools_equal(graph_pools, eager_pools)
    required = {"replay", "arena", "static_ids", "static_embeds"}
    _assert_on_stream(log, side.cuda_stream, required)
    assert binding.counters()["replays:chunk-fresh:g0:b1"] == 1

    # The oracle fails on a resident write issued to another stream, whichever
    # operation issues it.
    pool = graph_pools["book_pool"]
    block = torch.tensor([1], device=device)
    saved = pool[1:2].clone()
    other = torch.cuda.Stream(device)
    other.wait_stream(side)
    other.wait_stream(torch.cuda.current_stream())
    writes = {
        "fill": lambda: pool[1:2].fill_(0),
        "indexed-assignment": lambda: pool.__setitem__(block, saved),
    }
    for write in writes.values():
        with _stream_trace(monkeypatch, targets) as stray, torch.cuda.stream(other):
            write()
        other.synchronize()
        assert ("resident", other.cuda_stream) in stray, stray
        with pytest.raises(AssertionError):
            _assert_on_stream(stray, side.cuda_stream, {"resident"})


def _segment_pool(tensor: torch.Tensor) -> tuple[int, int]:
    """The allocator pool id of the segment holding ``tensor``."""
    address = tensor.untyped_storage().data_ptr()
    for segment in torch.cuda.memory._snapshot()["segments"]:
        if segment["address"] <= address < segment["address"] + segment["total_size"]:
            return tuple(segment["segment_pool_id"])
    raise AssertionError("tensor storage lies in no allocator segment")


@pytest.mark.nemotron_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.usefixtures("exact_fp32")
@torch.inference_mode()
def test_cuda_owned_outputs_are_allocated_outside_the_graph_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    core = _core(device)
    call = _call_bindings(core)
    pools = _endpoint_pools(4, device)
    captures = _track_graph_captures(monkeypatch, {"value": "serving"})
    binding = _binding(core, pools, call, _cuda_runtime(), device=device)
    graph_pools = {pool for _phase, _stream, pool in captures}
    assert graph_pools and None not in graph_pools
    escaped = _advance(
        core, pools, call, _rows([], [1], device=device), binding, _CommitRecorder()
    )
    for graph in binding._graphs.values():
        for name in ("rows_out", "status_out", "progress_out"):
            assert _segment_pool(getattr(graph, name)) not in graph_pools, name
    assert _segment_pool(escaped) not in graph_pools


def _drive_interleaved(
    core: Any,
    pools: Pools,
    call: tg.TransactionCallBindings,
    graph: Any,
    tier: Callable[[int], int],
) -> list[Any]:
    """CHUNK steps that alternate replayed cells with eager fallbacks.

    Populations ``1`` and ``2`` are declared; a mixed-fresh step and a
    three-row step fall back. Every CHUNK step drains its echoes before the
    next, through the same arm.
    """
    device = pools["book_pool"].device
    sequence: dict[int, int] = {}
    records: list[Any] = []
    steps = ([1, 2], [3, 1], [2, 3], [1, 2, 3], [4])
    for index, blocks in enumerate(steps):
        seqs = [sequence.get(block, -1) + 1 for block in blocks]
        sequence.update(zip(blocks, seqs, strict=True))
        embeds = torch.stack(
            [
                _chunk(
                    31 * index + block,
                    seq=seq,
                    geometry=GEOM_REG,
                    final=False,
                    prompt=block % 4,
                )
                for block, seq in zip(blocks, seqs, strict=True)
            ]
        ).to(device)
        plan = _plan(
            prefills=blocks,
            num_pool_blocks=5,
            has_initial=[seq > 0 for seq in seqs],
            prompts=[block % 4 for block in blocks],
            execution_tier=tier(len(blocks)),
        )
        records.append(
            _step(
                core, pools, call, [PLACEHOLDER_ID] * len(blocks), embeds, plan, graph
            )
        )
        tokens = dict(zip(blocks, records[-1][0][:, 0].long().tolist(), strict=True))
        for _ in range(CAP + 2):
            active = [block for block in blocks if tokens[block] != PARK_ID]
            if not active:
                break
            plan = _plan(
                decodes=active,
                num_pool_blocks=5,
                prompts=[block % 4 for block in active],
                execution_tier=tier(len(active)),
            )
            records.append(
                _step(
                    core,
                    pools,
                    call,
                    [tokens[block] for block in active],
                    torch.zeros(len(active), CARRIER_HIDDEN, device=device),
                    plan,
                    graph,
                )
            )
            for row, block in enumerate(active):
                tokens[block] = int(records[-1][0][row, 0])
        else:
            raise AssertionError("replay queue did not drain")
    return records


def _interleaved_parity(model: Any, slots: StateSlotBindings) -> None:
    """Fallbacks through the nested bindings interleave with exact replays."""
    binding = model._transaction_graph
    assert binding.ready
    call = _model_call(model)
    graph_pools = _pool_arguments(_resident(model, slots))
    eager_pools = _clone_pools(graph_pools)
    advance.warmup_advance_model_rows_scatter(**eager_pools)
    decoder = model._decode_graph_binding

    def tier(rows: int) -> int:
        return 0 if decoder is None else decoder.execution_tier(rows)

    expected = _drive_interleaved(model.core, eager_pools, call, None, tier)
    actual = _drive_interleaved(model.core, graph_pools, call, binding, tier)
    _assert_records_equal(actual, expected)
    _assert_pools_equal(graph_pools, eager_pools)
    assert all(not bool(status.any()) for _rows, status, _progress in expected)
    counts = binding.counters()
    assert counts["fallbacks:uncovered:mixed-fresh"] == 1
    assert counts["fallbacks:chunk-continuing:g0:b3:undeclared"] == 1
    assert counts["replays:chunk-fresh:g0:b2"] == 1
    assert counts["replays:chunk-continuing:g0:b2"] == 1
    assert counts["replays:chunk-fresh:g0:b1"] == 1
    assert counts["replays"] + counts["fallbacks"] == len(expected)


@pytest.mark.nemotron_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.usefixtures("exact_fp32")
@torch.inference_mode()
def test_cuda_interleaved_fallbacks_and_replays_match_eager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.models.nemotron_asr import decode_graph, encoder_execution

    device = torch.device("cuda")
    monkeypatch.setattr(encoder_execution, "platform_graph_runtime", _cuda_runtime)
    monkeypatch.setattr(decode_graph, "platform_graph_runtime", _cuda_runtime)
    model, config = _model(
        monkeypatch,
        runtime=_cuda_runtime,
        device=device,
        engine={"max_num_seqs": 4},
        encoder_execution_arm="eager-graphed",
        decode_dispatch_arm="dense-graphed",
    )
    model.prepare_execution_memory()
    state, slots = _model_state(model, config, num_blocks=5)
    state.initialize_state_slots(slots)
    _interleaved_parity(model, slots)


@pytest.mark.nemotron_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.usefixtures("exact_fp32")
@torch.inference_mode()
def test_cuda_replay_allocates_nothing_after_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    core = _core(device)
    call = _call_bindings(core)
    pools = _endpoint_pools(4, device)
    binding = _binding(core, pools, call, _cuda_runtime(), device=device)
    allocated: list[int] = []
    replay = torch.cuda.CUDAGraph.replay

    def counted(self: Any) -> Any:
        allocated.append(torch.cuda.memory_stats(device)["allocation.all.allocated"])
        return replay(self)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", counted)
    torch.cuda.synchronize()
    escaped = _advance(
        core,
        pools,
        call,
        _rows([], [1], device=device),
        binding,
        _TrackedSink(rows=1, device=device),
    )
    returned = torch.cuda.memory_stats(device)["allocation.all.allocated"]
    assert len(allocated) == 1 and returned == allocated[0]
    assert escaped.shape == (1, CARRIER_HIDDEN) and escaped.dtype == torch.float32
    owned = {
        tensor.untyped_storage().data_ptr()
        for graph in binding._graphs.values()
        for tensor in (
            graph.arena._device,
            graph.static_ids,
            graph.static_embeds,
            graph.rows_out,
            graph.status_out,
            graph.progress_out,
        )
    }
    assert escaped.untyped_storage().data_ptr() not in owned
    kept = escaped.clone()
    _advance(
        core,
        pools,
        call,
        _rows([], [2], device=device),
        binding,
        _TrackedSink(rows=1, device=device),
    )
    torch.cuda.synchronize()
    assert len(allocated) == 2
    assert torch.equal(escaped, kept)


@pytest.mark.nemotron_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.usefixtures("exact_fp32")
@torch.inference_mode()
def test_cuda_status_and_progress_survive_another_cells_replay() -> None:
    device = torch.device("cuda")
    core = _core(device)
    call = _call_bindings(core)
    eager_pools = _endpoint_pools(4, device)
    graph_pools = _clone_pools(eager_pools)
    binding = _binding(core, graph_pools, call, _cuda_runtime(), device=device)
    for pools in (eager_pools, graph_pools):
        pools["book_pool"][1:, _BOOK["last_label"]] = core.blank_id
    first, second = (
        dict(blocks=[1], fresh=[True]),
        dict(blocks=[2, 3], fresh=[True] * 2),
    )
    expected = _first_step(None, eager_pools, core=core, call=call, **first)
    _first_step(None, eager_pools, core=core, call=call, **second)
    _first_step(binding, graph_pools, core=core, call=call, **first)
    _first_step(binding, graph_pools, core=core, call=call, **second)
    torch.cuda.synchronize()
    cell = binding._graphs[tg.TransactionCell(tg.CELL_FRESH_CHUNK, GEOM_REG, 1)]
    torch.testing.assert_close(cell.status_out.cpu(), expected[1].cpu(), atol=0, rtol=0)
    torch.testing.assert_close(
        cell.progress_out.cpu(), expected[2].cpu(), atol=0, rtol=0
    )
    counts = binding.counters()
    assert counts["replays:chunk-fresh:g0:b1"] == 1
    assert counts["replays:chunk-fresh:g0:b2"] == 1
