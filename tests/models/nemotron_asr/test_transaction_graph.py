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
import sys
import types
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

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
from tests.models.nemotron_asr.test_plan_local import _feature
from vllm.config.compilation import CUDAGraphMode
from vllm.models.nemotron_asr import profile_execution
from vllm.models.nemotron_asr import transaction_graph as tg
from vllm.models.nemotron_asr.capture_scope import (
    nested_graph_mode,
    outer_transaction_capture,
)
from vllm.models.nemotron_asr.decode_graph import GraphRuntime
from vllm.models.nemotron_asr.model_state_v2 import NemotronASRModelState
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


def _runtime(wrapper_factory: Callable[..., Any], mode: dict[str, str]) -> GraphRuntime:
    def context(
        _metadata: object,
        _config: object,
        *,
        cudagraph_runtime_mode: str,
        batch_descriptor: object,
    ) -> Any:
        del batch_descriptor

        @contextmanager
        def scope() -> Iterator[None]:
            previous = mode["value"]
            mode["value"] = cudagraph_runtime_mode
            try:
                yield
            finally:
                mode["value"] = previous

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

    ``on_capture`` receives the creation index of the wrapper whose first
    graph-mode call (its capture) is starting; raising from it injects a
    capture failure.
    """
    mode = {"value": "none"}
    created = [0]

    class Wrapper:
        def __init__(self, runnable: Callable[[], Any], *_: Any, **__: Any):
            self.runnable = runnable
            self.index = created[0]
            self.captured = False
            created[0] += 1

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            if mode["value"] == "graph" and not self.captured:
                if on_capture is not None:
                    on_capture(self.index)
                self.captured = True
            return self.runnable(*args, **kwargs)

    return _runtime(Wrapper, mode)


def _cuda_runtime() -> GraphRuntime:
    """Capture on the first graph-mode call, then replay the CUDA graph.

    Every cell's graph is captured into one shared memory pool, as the
    platform wrapper does with its global pool, unless the caller replaces a
    wrapper's ``graph_pool`` before its capture, as it may on the platform
    wrapper.
    """
    mode = {"value": "none"}
    pool = torch.cuda.graph_pool_handle()

    class Wrapper:
        def __init__(self, runnable: Callable[[], Any], *_: Any, **__: Any):
            self.runnable = runnable
            self.graph: torch.cuda.CUDAGraph | None = None
            self.graph_pool = pool
            self.output: Any = None

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            if mode["value"] != "graph":
                return self.runnable(*args, **kwargs)
            if self.graph is None:
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, pool=self.graph_pool):
                    self.output = self.runnable(*args, **kwargs)
                return self.output
            self.graph.replay()
            return self.output

    return _runtime(Wrapper, mode)


# ---- fixtures --------------------------------------------------------------


def _endpoint_pools(num_blocks: int, device: torch.device = CPU) -> Pools:
    pools = _fresh_pools(num_blocks)
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
        endpoint_history=pools["endpoint_history_pool"],
        endpoint_book=pools["endpoint_book_pool"],
    )


def _core(device: torch.device = CPU) -> Any:
    core = _tiny_core(seed=1)
    for name in ("encoder", "lid", "predictor", "joint", "featurizer"):
        getattr(core, name).to(device)
    return core


def _call_bindings(core: Any) -> tg.TransactionCallBindings:
    return tg.TransactionCallBindings(
        adapter=advance.make_mrv1_adapter(
            hidden_size=CARRIER_HIDDEN, park_id=PARK_ID, blank_id=core.blank_id
        ),
        decode_resolver=_fixed_resolver(rnnt.decode_dense_masked_frames),
        encoder_transition=None,
        bucket_transition=None,
        placeholder_id=PLACEHOLDER_ID,
        park_id=PARK_ID,
        eou_token_id=EOU,
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
        geometries=(GEOM_REG, GEOM_FINAL),
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
            cudagraph_mm_encoder=False,
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

    Each record has the cell, weak references to its graph, arena and capture
    callbacks, and the shapes of the pools it was prepared over. ``events``
    receives ``("prepare", label)`` per cell.
    """
    prepared: list[SimpleNamespace] = []
    capture_cell = tg.TransactionGraphBinding._capture_cell

    def track(self: Any, core: Any, cell: tg.TransactionCell, **kwargs: Any) -> Any:
        graph = capture_cell(self, core, cell, **kwargs)
        prepared.append(
            SimpleNamespace(
                cell=cell,
                graph=weakref.ref(graph),
                arena=weakref.ref(graph.arena),
                callbacks=_capture_callbacks(graph),
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = embeds.device
    sink = _CommitRecorder()
    out = advance.advance_model_rows(
        core,
        torch.tensor(ids, dtype=torch.long, device=device),
        embeds,
        plan,
        adapter=call.adapter,
        decode_resolver=call.decode_resolver,
        encoder_transition=call.encoder_transition,
        bucket_transition=call.bucket_transition,
        placeholder_id=PLACEHOLDER_ID,
        park_id=PARK_ID,
        eou_token_id=EOU,
        commit_sink=sink,
        capture=capture,
        graph_covers_decode=call.graph_covers_decode,
        transaction_graph=graph,
        **pools,
    )
    assert sink.log == ["reserve", "stage"]
    return (
        out.clone(),
        sink.staged[0].clone(),
        sink.plans[0].progress.clone(),
    )


def _chunk(
    seed: int, *, seq: int, geometry: int, final: bool, prompt: int
) -> torch.Tensor:
    torch.manual_seed(seed)
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
    sessions["b"] = (1, GEOM_FINAL, 3)
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
        "chunk-fresh:g2:b1",
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
    device: torch.device, runtime: GraphRuntime, population: int, *, chunks: int
) -> None:
    """Lockstep CHUNKs at ``population`` replay their cells bitwise exactly.

    Only ``population`` is declared, so every CHUNK must replay and only echo
    steps whose population shrank may fall back, as ``undeclared``.
    """
    core = _core(device)
    call = _call_bindings(core)
    eager_pools = _endpoint_pools(population + 1, device)
    graph_pools = _clone_pools(eager_pools)
    binding = tg.TransactionGraphBinding(
        populations=[population],
        geometries=(GEOM_REG, GEOM_FINAL),
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


@pytest.mark.cpu_test
def test_nested_bindings_run_eagerly_only_inside_the_capture_scope() -> None:
    runtime = _rerun_runtime()
    assert nested_graph_mode(runtime) == "graph"
    with outer_transaction_capture():
        assert nested_graph_mode(runtime) == "none"
    assert nested_graph_mode(runtime) == "graph"


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
    embeds = torch.stack(
        [_chunk(3, seq=0, geometry=GEOM_REG, final=False, prompt=1)] * len(blocks)
    )
    return advance.prepare_transaction(
        core,
        torch.full((len(blocks),), PLACEHOLDER_ID, dtype=torch.long),
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


@pytest.mark.cpu_test
def test_arena_alternates_host_buffers_across_calls() -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    first = _fresh_chunk_inputs(core, pools, call, [1])
    second = _fresh_chunk_inputs(core, pools, call, [3])
    arena = tg.TransactionArena(tg._arena_layout(first), CPU)
    arena.upload(first)
    arena.upload(second)
    assert int(arena._host_views[0]["composed_indices"][0]) == 1
    assert int(arena._host_views[1]["composed_indices"][0]) == 3
    assert int(arena.views["composed_indices"][0]) == 3


@pytest.mark.cpu_test
def test_classification_covers_declared_shapes_only() -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    cell, reason = tg.classify_transaction(
        _fresh_chunk_inputs(core, pools, call, [1, 2])
    )
    assert cell == tg.TransactionCell(tg.CELL_FRESH_CHUNK, GEOM_REG, 2)
    assert reason is None
    mixed = replace(
        _fresh_chunk_inputs(core, pools, call, [1, 2]),
        fresh_positions=torch.tensor([0]),
    )
    assert tg.classify_transaction(mixed) == (None, "mixed-fresh")


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
    """One out-of-sequence row in a continuing population is rejected alone."""
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

    def run(pools: Pools, graph: Any) -> list[Any]:
        records = []
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
            records.append(
                _step(
                    core,
                    pools,
                    call,
                    [PLACEHOLDER_ID] * population,
                    torch.stack(rows).to(device),
                    plan,
                    graph,
                )
            )
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
        return records

    expected = run(eager_pools, None)
    actual = run(graph_pools, binding)
    _assert_records_equal(actual, expected)
    _assert_pools_equal(graph_pools, eager_pools)
    status = expected[1][1]
    assert int(status[bad]) != 0
    assert not bool(status[torch.arange(population) != bad].any())
    counts = binding.counters()
    assert counts["fallbacks"] == 0
    assert counts[f"replays:chunk-continuing:g0:b{population}"] == 1


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
    binding = _binding(core, pools, call, _rerun_runtime(), populations=(1,))
    _assert_pools_equal(pools, before)
    assert binding.ready
    assert [cell.label for cell in binding.cells] == [
        "chunk-continuing:g0:b1",
        "chunk-continuing:g2:b1",
        "chunk-fresh:g0:b1",
        "chunk-fresh:g2:b1",
        "replay:b1",
    ]
    assert advance.consume_batch_stats() is None


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


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("blocks", "fresh", "capture", "key"),
    [
        ([1, 2], [True, False], False, "fallbacks:uncovered:mixed-fresh"),
        ([1, 2, 3], [True] * 3, False, "fallbacks:chunk-fresh:g0:b3:undeclared"),
        ([1], [True], True, "fallbacks:chunk-fresh:g0:b1:capture"),
    ],
)
def test_uncovered_calls_run_eager_and_are_counted(
    blocks: list[int], fresh: list[bool], capture: bool, key: str
) -> None:
    core = _core()
    call = _call_bindings(core)
    graph_pools = _endpoint_pools(4)
    eager_pools = _clone_pools(graph_pools)
    binding = _binding(core, graph_pools, call, _rerun_runtime())
    for pools in (graph_pools, eager_pools):
        pools["book_pool"][1:, _BOOK["last_label"]] = core.blank_id
    kwargs = dict(core=core, call=call, blocks=blocks, fresh=fresh, capture=capture)
    expected = _first_step(None, eager_pools, **kwargs)
    actual = _first_step(binding, graph_pools, **kwargs)
    _assert_records_equal([actual], [expected])
    _assert_pools_equal(graph_pools, eager_pools)
    counts = binding.counters()
    assert counts[key] == 1
    assert counts["fallbacks"] == 1 and counts["replays"] == 0


@pytest.mark.cpu_test
def test_relocated_pools_fall_back_instead_of_replaying() -> None:
    core = _core()
    call = _call_bindings(core)
    captured_pools = _endpoint_pools(4)
    binding = _binding(core, captured_pools, call, _rerun_runtime())
    captured = binding.counters()["captured_cells"]
    moved = _clone_pools(captured_pools)
    reference = _clone_pools(captured_pools)
    kwargs = dict(core=core, call=call, blocks=[1], fresh=[True])
    actual = _first_step(binding, moved, **kwargs)
    expected = _first_step(None, reference, **kwargs)
    _assert_records_equal([actual], [expected])
    _assert_pools_equal(captured_pools, _endpoint_pools(4))
    assert binding.counters()["fallbacks:chunk-fresh:g0:b1:signature"] == 1
    assert binding.counters()["captured_cells"] == captured


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
    advance.advance_model_rows(
        core,
        torch.tensor([PLACEHOLDER_ID]),
        _chunk(5, seq=0, geometry=GEOM_REG, final=False, prompt=0).unsqueeze(0),
        _plan(prefills=[1], num_pool_blocks=4),
        adapter=call.adapter,
        decode_resolver=call.decode_resolver,
        placeholder_id=PLACEHOLDER_ID,
        park_id=PARK_ID,
        eou_token_id=EOU,
        commit_sink=sink,
        transaction_graph=binding,
        **pools,
    )
    assert seen == ["reserve"] and sink.log == ["reserve", "stage"]
    assert int(sink.staged[0][0]) == 0
    assert int(pools["frontend_counter_pool"][1].abs().sum()) > 0
    assert binding.counters()["replays:chunk-fresh:g0:b1"] == 1


@pytest.mark.cpu_test
def test_replay_failure_cancels_the_reservation_once() -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    binding = _binding(core, pools, call, _rerun_runtime())
    graph = binding._graphs[tg.TransactionCell(tg.CELL_FRESH_CHUNK, GEOM_REG, 1)]

    def fail() -> None:
        raise RuntimeError("replay failed")

    graph.replay = fail
    sink = _CommitRecorder()
    with pytest.raises(RuntimeError, match="replay failed"):
        advance.advance_model_rows(
            core,
            torch.tensor([PLACEHOLDER_ID]),
            _chunk(5, seq=0, geometry=GEOM_REG, final=False, prompt=0).unsqueeze(0),
            _plan(prefills=[1], num_pool_blocks=4),
            adapter=call.adapter,
            decode_resolver=call.decode_resolver,
            placeholder_id=PLACEHOLDER_ID,
            park_id=PARK_ID,
            eou_token_id=EOU,
            commit_sink=sink,
            transaction_graph=binding,
            **pools,
        )
    assert sink.log == ["reserve", "cancel"]
    assert binding.counters()["replays"] == 0


@pytest.mark.cpu_test
def test_stale_lease_fails_before_replay_writes_a_page() -> None:
    from vllm.models.nemotron_asr.commit_sink import BoundedCommitSink
    from vllm.models.nemotron_asr.plan import SessionRegistry

    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    binding = _binding(core, pools, call, _rerun_runtime())
    before = _clone_pools(pools)
    with pytest.raises(ValueError, match="not registered"):
        advance.advance_model_rows(
            core,
            torch.tensor([PLACEHOLDER_ID]),
            _chunk(5, seq=0, geometry=GEOM_REG, final=False, prompt=0).unsqueeze(0),
            _plan(prefills=[1], num_pool_blocks=4),
            adapter=call.adapter,
            decode_resolver=call.decode_resolver,
            placeholder_id=PLACEHOLDER_ID,
            park_id=PARK_ID,
            eou_token_id=EOU,
            commit_sink=BoundedCommitSink(SessionRegistry(), max_rows=4),
            transaction_graph=binding,
            **pools,
        )
    _assert_pools_equal(pools, before)
    assert binding.counters()["replays"] == 0


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


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "mode",
    [mode for mode in CUDAGraphMode if mode != CUDAGraphMode.NONE],
    ids=lambda mode: mode.name,
)
def test_option_rejects_every_outer_cudagraph_mode_but_none(
    monkeypatch: pytest.MonkeyPatch, mode: CUDAGraphMode
) -> None:
    model, _config = _model(monkeypatch)
    assert model._transaction_graph is not None
    with pytest.raises(ValueError, match="transaction_graph_populations"):
        _model(monkeypatch, engine={"cudagraph_mode": mode})


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "lookaheads", [[1], [0, 1], [3], _ABSENT], ids=["g1", "g0-g1", "g2", "every"]
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
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda *_args: (222_222_222, 333_333_333)
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
) -> None:
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
) -> None:
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


@pytest.mark.cpu_test
def test_changed_pool_layout_on_reinitialize_is_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, config = _model(monkeypatch)
    model.prepare_execution_memory()
    state, slots = _model_state(model, config, num_blocks=4)
    state.initialize_state_slots(slots)
    binding = model._transaction_graph
    _other, moved = _model_state(model, config, num_blocks=4)
    prepared = _track_capture_cells(monkeypatch)
    with pytest.raises(RuntimeError):
        state.initialize_state_slots(moved)
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
    at_na: dict[str, int] = {}

    def observe(index: int) -> None:
        if index != 1:
            return
        storage = stand_in.storage[0]()
        assert storage is not None, "the stand-in storage was released before Na"
        owners = [
            referrer
            for referrer in gc.get_referrers(storage)
            if not isinstance(referrer, types.FrameType)
        ]
        at_na.update(
            inputs=len(_live(inputs)),
            callbacks=len(
                _live([ref for record in prepared for ref in record.callbacks])
            ),
            views=len(_live(stand_in.views)),
            owners=len(owners),
        )
        del storage, owners

    _free_memory(
        monkeypatch,
        lambda index: _budget(
            stand_in.nbytes[0] if stand_in.nbytes else 0, 1 << 20, 1 << 20
        )(index),
        observe,
    )
    binding = _admission_binding()
    _admit(binding, core, call, stand_in)
    assert at_na == {"inputs": 0, "callbacks": 0, "views": 0, "owners": 1}
    resident = _transaction_view(_scratch(3))
    _serve(binding, core, call, resident)
    assert binding.ready
    assert not _live(inputs)
    assert not _live([ref for record in prepared for ref in record.callbacks])


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
    monkeypatch: pytest.MonkeyPatch, *, num_blocks: int = 3
) -> tuple[Any, NemotronASRModelState, StateSlotBindings]:
    """A served model with every cell captured over resident slots."""
    model, config = _model(monkeypatch)
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


def _assert_names_cell_and_buffer(model: Any, message: str) -> None:
    assert "transaction_graph_populations" in message
    assert any(cell.label in message for cell in model._transaction_graph.cells)
    assert "pos_enc" in message


@pytest.mark.cpu_test
def test_post_warmup_verification_rejects_resized_external_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _state, _slots = _verified_model(monkeypatch)
    verify = _seam(model, "verify_transaction_graphs")
    verify()
    _resize_positional_buffer(model)
    with pytest.raises(RuntimeError, match="transaction_graph_populations") as raised:
        verify()
    _assert_names_cell_and_buffer(model, str(raised.value))


@pytest.mark.cpu_test
def test_first_real_step_verifies_once_before_pruning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
def test_first_real_step_mismatch_is_fatal_before_any_state_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, state, slots = _verified_model(monkeypatch)
    pruned: list[Any] = []
    monkeypatch.setattr(model._registry, "prune", pruned.append)
    resident = slots.views[model._persistent_state_layer.prefix]._tensor
    before = resident.clone()
    _resize_positional_buffer(model)
    with pytest.raises(RuntimeError, match="transaction_graph_populations") as raised:
        state.prepare_step(
            SchedulerOutput.make_empty(), dummy_run=False, is_profile=False
        )
    _assert_names_cell_and_buffer(model, str(raised.value))
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
    with pytest.raises(RuntimeError):
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
@pytest.mark.parametrize(("population", "chunks"), _POPULATION_CHUNKS)
def test_cuda_population_lockstep_replays_every_chunk(
    population: int, chunks: int
) -> None:
    _population_parity(torch.device("cuda"), _cuda_runtime(), population, chunks=chunks)


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
    device = torch.device("cuda")
    core = _core(device)
    call = _call_bindings(core)
    eager_pools = _endpoint_pools(4, device)
    graph_pools = _clone_pools(eager_pools)
    binding = _binding(core, graph_pools, call, _cuda_runtime(), device=device)

    def run(pools: Pools, graph: Any) -> list[Any]:
        records = [
            _first_step(graph, pools, core=core, call=call, blocks=[2], fresh=[True])
        ]
        token = int(records[0][0][0, 0])
        if corruption == "sequence":
            embeds = _chunk(9, seq=5, geometry=GEOM_REG, final=False, prompt=0)
            embeds = embeds.to(device)
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
        records.append(_step(core, pools, call, ids, rows, plan, graph))
        _assert_pools_equal(pools, before)
        return records

    expected = run(eager_pools, None)
    actual = run(graph_pools, binding)
    _assert_records_equal(actual, expected)
    assert int(expected[1][1][0]) != 0
    _assert_pools_equal(graph_pools, eager_pools)
    assert binding.counters()["fallbacks"] == 0


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
    admitted, served = receipt["admission_bytes"], receipt["serving_bytes"]
    assert isinstance(admitted, int) and isinstance(served, int)
    assert 0 < served <= admitted
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
