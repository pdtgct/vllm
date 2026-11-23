# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker, model-state and bounded-history behaviour of persistent state."""

import contextlib
import gc
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
from tests.v1.attention.test_mamba_metadata_builder import _make_vllm_config
from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from tests.v1.core.test_persistent_state_scheduler import declare_state_facts
from tests.v1.core.utils import (
    EOS_TOKEN_ID,
    create_requests,
    create_scheduler,
    new_kv_cache_group,
)
from tests.v1.engine.test_engine_core import _executor_calls, _pausable_engine_core_proc
from tests.v1.worker.test_gpu_model_runner_v2 import _make_capture_runner
from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.models.interfaces import SupportsPersistentState
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.utils.vmm_driver import vmm_unavailable_reason
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.output_processor import (
    OutputProcessor,
    StreamingInputOutputCollector,
)
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.extensible_kv_cache import ExtensibleKVCache
from vllm.v1.worker.gpu import warmup
from vllm.v1.worker.gpu.input_batch import prepare_pos_seq_lens
from vllm.v1.worker.gpu.model_runner import ExecuteModelState, GPUModelRunner
from vllm.v1.worker.utils import StatePageInitializer

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _state_spec(**kwargs) -> MambaSpec:
    return MambaSpec(
        block_size=16,
        shapes=((1,), (1,)),
        dtypes=(torch.int64, torch.int64),
        mamba_type=MambaAttentionBackendEnum.PERSISTENT_STATE,
        **kwargs,
    )


# Declarations


# @spec PSTATE-DECL-002
@pytest.mark.parametrize("initial_values", [None, (0, 7)])
def test_mamba_spec_carries_initial_values(initial_values):
    assert _state_spec(initial_values=initial_values).initial_values == initial_values


# @spec PSTATE-DECL-002
def test_mamba_spec_initial_values_default_to_none():
    assert _state_spec().initial_values is None


# @spec PSTATE-DECL-004
def test_state_model_declarations_default_to_a_resumable_ordinary_model():
    assert SupportsPersistentState.persistent_state_reads_positions is True
    assert SupportsPersistentState.persistent_state_parks_on_eos is True
    assert SupportsPersistentState.persistent_state_accepts_synthetic_warmup is True
    assert SupportsPersistentState.persistent_state_ignores_stopped_row_echoes is False


# @spec PSTATE-DECL-010
def test_mamba_spec_states_tensor_parallel_ownership():
    assert _state_spec(tp_replicated=True).tp_replicated is True
    assert _state_spec().tp_replicated is False


# Page initialization and reuse


def _build_state_metadata(seq_lens, query_lens, is_prefilling):
    builder_cls = MambaAttentionBackendEnum.PERSISTENT_STATE.get_class()
    config = _make_vllm_config(256, len(seq_lens), block_size=16)
    builder = builder_cls.get_builder_cls()(
        _state_spec(), ["layer0"], config, torch.device("cpu")
    )
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        block_size=16,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )
    common = common.replace(is_prefilling=torch.tensor(is_prefilling, dtype=torch.bool))
    return builder.build(0, common)


# @spec PSTATE-PAGE-003
def test_state_metadata_marks_a_first_turn_row_as_holding_initial_state():
    """A prefill row with no computed tokens still reads its page."""
    metadata = _build_state_metadata([4, 6], [4, 2], [True, True])
    assert metadata.has_initial_states_p is not None
    assert metadata.has_initial_states_p.tolist() == [True, True]


# @spec PSTATE-PAGE-003
def test_state_metadata_marks_a_single_token_prefill_as_holding_initial_state():
    metadata = _build_state_metadata([8, 9, 1], [1, 1, 1], [False, False, True])
    assert metadata.num_prefills == 1
    assert metadata.has_initial_states_p is not None
    assert metadata.has_initial_states_p.tolist() == [True]


def _two_state_groups():
    """Two state groups of one layer each, whose pages are separate CPU
    tensors; the second declares initial values, the first takes zeros."""
    first = MambaManager.layer_config(_state_spec(), recomputable=False)
    second = MambaManager.layer_config(
        _state_spec(initial_values=(5, 7)), recomputable=False
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            new_kv_cache_group(["a"], first),
            new_kv_cache_group(["b"], second),
        ]
    )
    context = {
        name: SimpleNamespace(
            kv_cache=[torch.full((4, 1), -1, dtype=torch.int64) for _ in range(2)]
        )
        for name in ("a", "b")
    }
    return kv_cache_config, context


# @spec PSTATE-PAGE-001
def test_initializer_writes_each_groups_values_into_its_own_pages_only():
    kv_cache_config, context = _two_state_groups()
    initializer = StatePageInitializer.from_kv_cache_config(
        torch.device("cpu"), kv_cache_config, context
    )
    assert initializer is not None and initializer.group_ids == [0, 1]

    initializer.initialize({0: [1], 1: [2, 3]})

    a_total, a_count = (t[:, 0].tolist() for t in context["a"].kv_cache)
    b_total, b_count = (t[:, 0].tolist() for t in context["b"].kv_cache)
    assert (a_total, a_count) == ([-1, 0, -1, -1], [-1, 0, -1, -1])
    assert (b_total, b_count) == ([-1, -1, 5, 5], [-1, -1, 7, 7])


# @spec PSTATE-PAGE-001
def test_initializer_is_none_without_a_state_group():
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[new_kv_cache_group(["m"], _state_spec())]
    )
    initializer = StatePageInitializer.from_kv_cache_config(
        torch.device("cpu"), kv_cache_config, {}
    )
    assert initializer is None


# @spec PSTATE-PREP-005
def test_warmup_initializes_each_groups_pages_from_its_requests_blocks():
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = [
        SimpleNamespace(block_ids=([1], [2, 3])),
        SimpleNamespace(block_ids=([4], [5])),
    ]
    model_runner = SimpleNamespace(
        state_page_initializer=SimpleNamespace(group_ids=[1])
    )

    warmup._initialize_state_pages(model_runner, output, range(1, 6))

    assert output.new_state_block_ids == {1: [2, 3, 5]}
    assert output.new_block_ids_to_zero is None


# @spec PSTATE-PAGE-006
def test_worker_frees_a_finished_row_before_it_initializes_a_new_owner(monkeypatch):
    """Old owners leave and new owners arrive in this order on the runner
    stream, so a page's new owner initializes it after the old owner's
    accesses."""
    events: list[str] = []
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.draft_tokens_handler = SimpleNamespace(
        begin_step=lambda scheduler_output: None
    )
    runner.encoder_cache = None
    runner.aux_output_connector = None
    runner.block_tables = SimpleNamespace(apply_staged_writes=lambda: None)
    for name in (
        "update_pp_decode_requests",
        "finish_requests",
        "free_states",
        "add_requests",
        "update_requests",
    ):
        monkeypatch.setattr(
            runner,
            name,
            lambda *args, name=name: events.append(name),
        )
    monkeypatch.setattr(runner, "_merge_ec_connector_no_forward", lambda *args: None)
    runner.kv_connector = SimpleNamespace(no_forward=lambda scheduler_output: None)

    runner.execute_model(SchedulerOutput.make_empty())

    assert events[1:] == [
        "finish_requests",
        "free_states",
        "add_requests",
        "update_requests",
    ]


# @spec PSTATE-PRES-008
@requires_cuda
def test_extensible_kv_cache_growth_keeps_state_pages_and_graphs_valid():
    """A captured graph that updates a held state page stays valid when the
    cache grows: the page keeps its address and contents."""
    if (reason := vmm_unavailable_reason()) is not None:
        pytest.skip(f"VMM unavailable: {reason}")
    num_blocks = 8
    spec = _state_spec()
    layer = MambaManager.layer_config(spec, recomputable=False)
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=16))
    config = get_kv_cache_config_from_groups(
        vllm_config,
        [new_kv_cache_group(["state"], layer)],
        available_memory=spec.page_size_bytes * num_blocks,
    )
    kv_cache = ExtensibleKVCache(config, torch.device("cuda"), KVCacheLayout.BLNHC)
    try:
        buf = kv_cache.allocate(kv_cache.size)
        page_size = spec.page_size_bytes
        kv_cache.commit(2)
        state = buf[page_size : 2 * page_size].view(torch.int64)
        ptr = state.data_ptr()
        graph = torch.cuda.CUDAGraph()
        state.add_(1)
        with torch.cuda.graph(graph):
            state.add_(1)
        state.zero_()
        graph.replay()
        torch.accelerator.synchronize()
        assert state.tolist() == [1, 1]

        kv_cache.commit(num_blocks)
        graph.replay()
        torch.accelerator.synchronize()

        assert state.data_ptr() == ptr
        assert state.tolist() == [2, 2]
    finally:
        kv_cache.free()


# Positions


# @spec PSTATE-HIST-004
@requires_cuda
def test_position_offset_shifts_positions_but_not_seq_lens():
    device = torch.device("cuda")
    positions = torch.full((8,), -1, dtype=torch.int64, device=device)
    seq_lens = torch.full((4,), -1, dtype=torch.int32, device=device)
    idx_mapping = torch.tensor([2, 0], dtype=torch.int64, device=device)
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32, device=device)
    num_computed_tokens = torch.tensor([1, 0, 5], dtype=torch.int32, device=device)
    position_offset = torch.tensor([100, 0, 7], dtype=torch.int32, device=device)

    prepare_pos_seq_lens(
        idx_mapping,
        query_start_loc,
        num_computed_tokens,
        positions,
        seq_lens,
        position_offset=position_offset,
    )
    torch.accelerator.synchronize()

    assert positions.tolist() == [12, 13, 101, -1, -1, -1, -1, -1]
    assert seq_lens.tolist() == [7, 2, 0, 0]


# @spec PSTATE-HIST-004
def test_new_request_data_carries_the_position_offset():
    request = create_requests(1)[0]
    request.position_offset = 42

    data = NewRequestData.from_request(request, ([0],), [1, 2, 3])

    assert data.position_offset == 42


# @spec PSTATE-HIST-004
def test_new_request_data_has_no_position_offset_by_default():
    request = create_requests(1)[0]

    data = NewRequestData.from_request(request, ([0],), [1, 2, 3])

    assert data.position_offset == 0


# Per-request failure


# @spec PSTATE-FAIL-001
def test_runner_attaches_failed_ids_after_the_state_postprocesses(monkeypatch):
    events: list[str] = []
    outputs: list[ModelRunnerOutput] = []

    class FakeAsyncOutput:
        def __init__(self, *, model_runner_output, **kwargs):
            self.model_runner_output = model_runner_output
            outputs.append(model_runner_output)

    class FakeModelState:
        def postprocess_state(self, *args):
            events.append("postprocess_state")

        def take_failed_requests(self):
            events.append("take_failed_requests")
            return {"failed"}

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.is_last_pp_rank = True
    runner.pcp_manager = None
    runner.pp_handler = None
    runner.speculator = None
    runner.aux_output_connector = None
    runner.num_speculative_steps = 0
    runner.main_stream = None
    runner.output_copy_stream = None
    runner.check_ep_fault = None
    runner.model = SimpleNamespace(compute_logits=None)
    runner.model_state = FakeModelState()
    runner.kv_connector = SimpleNamespace(post_forward=lambda *args, **kwargs: None)
    runner.prompt_logprobs_worker = SimpleNamespace(
        compute_prompt_logprobs=lambda *args: {},
        compute_prompt_token_id_logprobs=lambda *args: {},
    )
    runner.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=None),
        num_computed_tokens=SimpleNamespace(gpu=None),
        prompt_len=SimpleNamespace(np=None),
    )
    input_batch = SimpleNamespace(
        req_ids=["failed", "healthy"], idx_mapping=None, query_start_loc=None
    )
    runner.execute_model_state = ExecuteModelState(
        input_batch=input_batch,
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=object(),
        aux_hidden_states=None,
        dp_sync_state=None,
        finished_req_ids=set(),
        ec_connector_output=None,
        cudagraph_stats=None,
        num_spec_tokens_to_schedule=0,
    )
    sampler_output = SimpleNamespace(sampled_token_ids=None)
    monkeypatch.setattr(
        runner, "sample", lambda *args: (sampler_output, None, None), raising=False
    )
    monkeypatch.setattr(runner, "_get_num_drafted_and_verified", lambda b: (None, None))
    monkeypatch.setattr(
        runner,
        "postprocess_sampled",
        lambda *args: runner.model_state.postprocess_state(),
    )
    monkeypatch.setattr(model_runner_module, "AsyncOutput", FakeAsyncOutput)

    runner.sample_tokens(None)

    assert events == ["postprocess_state", "take_failed_requests"]
    assert outputs[0].failed_req_ids == {"failed"}


# Startup and resume preparation


PROFILED_PEAK = 1 << 20
"""The transient peak profiling left in the KV cache's reserved headroom."""

WARMUP_PEAK = 1 << 24
"""The larger transient peak the sizing warmup reaches."""


def _warmup_worker(monkeypatch, *, accepts_synthetic_warmup: bool):
    """A worker whose warmup steps are recorded in `calls`, and whose KV cache
    sizing records the reserved headroom it is given as `("size", bytes)`."""
    calls: list = []
    monkeypatch.setattr(
        gpu_worker, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False)
    )
    monkeypatch.setattr(gpu_worker, "kernel_warmup", lambda worker: None)
    monkeypatch.setattr(gpu_worker, "freeze_gc_heap", lambda: None)
    monkeypatch.setattr(gpu_worker, "maybe_attach_gc_debug_callback", lambda: None)
    monkeypatch.setattr(gpu_worker, "set_random_seed", lambda *args: None)
    monkeypatch.setattr(gpu_worker, "enable_gpu_sync_check", lambda: None)
    monkeypatch.setattr(gpu_worker, "_min_across_ranks", lambda value: value)

    def measure_kv_cache_blocks(kv_cache, *args, **kwargs):
        calls.append(("size", kv_cache.reserved_headroom_bytes))
        return 1

    monkeypatch.setattr(gpu_worker, "measure_kv_cache_blocks", measure_kv_cache_blocks)
    monkeypatch.setattr(
        torch.accelerator, "reset_peak_memory_stats", lambda device: None
    )
    monkeypatch.setattr(
        gpu_worker, "warmup_kernels", lambda *args: calls.append("warmup_kernels")
    )
    monkeypatch.setattr(
        gpu_worker,
        "run_kv_cache_sizing_warmup",
        lambda *args: calls.append("sizing_warmup"),
    )
    kv_cache = SimpleNamespace(
        min_committed_blocks=1,
        reserved_headroom_bytes=PROFILED_PEAK,
        commit=lambda *args, **kwargs: None,
    )

    def record_transient_peak():
        calls.append("record_transient_peak")
        kv_cache.reserved_headroom_bytes = max(
            kv_cache.reserved_headroom_bytes, WARMUP_PEAK
        )

    kv_cache.record_transient_peak = record_transient_peak
    compilation_config = SimpleNamespace(
        mode=gpu_worker.CompilationMode.NONE,
        compilation_time=0.0,
        encoder_compilation_time=0.0,
    )
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            compilation_config=compilation_config,
            kernel_config=SimpleNamespace(enable_jit_warmup=True),
        ),
        cache_config=SimpleNamespace(
            enable_extensible_kv_cache=True, kv_cache_memory_bytes=None
        ),
        model_config=SimpleNamespace(
            enforce_eager=True,
            seed=0,
            has_persistent_state=True,
            persistent_state_accepts_synthetic_warmup=accepts_synthetic_warmup,
        ),
        parallel_config=SimpleNamespace(data_parallel_index=0),
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=64),
        compilation_config=compilation_config,
        device=torch.device("cpu"),
        init_free_memory=0,
        requested_memory=0,
        model_runner=SimpleNamespace(
            is_pooling_model=False,
            maybe_remove_all_loras=lambda lora_config: None,
            lora_config=None,
            _dummy_run=lambda num_tokens, skip_eplb: (None, None),
            _dummy_sampler_run=lambda hidden_states: None,
            model_state=SimpleNamespace(post_warmup=lambda: None),
            extensible_kv_cache=kv_cache,
            block_tables=SimpleNamespace(clear_input_block_tables=lambda: None),
        ),
        _maybe_activate_jit_monitor=lambda: None,
        execute_model=lambda *args, **kwargs: None,
        sample_tokens=lambda *args, **kwargs: None,
    )
    return worker, calls


# @spec PSTATE-PREP-001
def test_worker_skips_both_synthetic_warmups_for_a_model_that_declines_them(
    monkeypatch,
):
    worker, calls = _warmup_worker(monkeypatch, accepts_synthetic_warmup=False)

    gpu_worker.Worker._compile_or_warm_up_model(worker)

    assert "warmup_kernels" not in calls
    assert "sizing_warmup" not in calls


# @spec PSTATE-PREP-001
def test_worker_runs_both_synthetic_warmups_for_a_model_that_accepts_them(
    monkeypatch,
):
    worker, calls = _warmup_worker(monkeypatch, accepts_synthetic_warmup=True)

    gpu_worker.Worker._compile_or_warm_up_model(worker)

    assert calls[:2] == ["warmup_kernels", "sizing_warmup"]


# @spec PSTATE-PREP-002
def test_extensible_kv_cache_is_sized_from_the_profiling_peak_without_warmup(
    monkeypatch,
):
    worker, calls = _warmup_worker(monkeypatch, accepts_synthetic_warmup=False)

    gpu_worker.Worker._compile_or_warm_up_model(worker)

    assert ("size", PROFILED_PEAK) in calls
    assert "record_transient_peak" not in calls


# @spec PSTATE-PREP-002
def test_extensible_kv_cache_keeps_the_warmup_peak_when_warmup_runs(monkeypatch):
    """The control of the test above: with the sizing warmup, the cache is
    sized from the warmup's larger peak."""
    worker, calls = _warmup_worker(monkeypatch, accepts_synthetic_warmup=True)

    gpu_worker.Worker._compile_or_warm_up_model(worker)

    assert ("size", WARMUP_PEAK) in calls


# @spec PSTATE-PREP-004
def test_model_state_prepares_its_workspace_inside_the_profiling_scope(monkeypatch):
    events: list[str] = []

    class Profiled(Exception):
        pass

    @contextlib.contextmanager
    def memory_profiling(*args, **kwargs):
        events.append("enter")
        try:
            yield SimpleNamespace()
        finally:
            events.append("exit")

    def profile_run(*, randomize_inputs=False):
        events.append("profile_run")
        raise Profiled

    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=4),
            speculative_config=None,
        ),
        randomize_dummy_inputs=False,
        cache_config=SimpleNamespace(kv_cache_memory_bytes=1),
        model_runner=SimpleNamespace(
            profile_run=profile_run,
            prepare_execution_memory=lambda: events.append("prepare"),
            model_memory_usage=0,
        ),
        init_snapshot=SimpleNamespace(free_memory=2),
        _scoped_allocator_max_split=lambda **kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(gpu_worker, "maybe_apply_startup_plan", lambda _: None)
    monkeypatch.setattr(gpu_worker, "memory_profiling", memory_profiling)

    with pytest.raises(Profiled):
        gpu_worker.Worker.determine_available_memory(worker)

    assert events == ["enter", "prepare", "profile_run", "exit"]


def _recording_model_state(events: list[tuple]):
    return SimpleNamespace(
        supports_mm_inputs=False,
        needs_cudagraph_capture=lambda: True,
        clear_cudagraphs=lambda: events.append(("clear",)),
        capture_cudagraphs=lambda *, graph_pool, stream, profile_only: events.append(
            ("capture", profile_only)
        ),
        capture_inner_cudagraphs=lambda *args: None,
    )


@contextlib.contextmanager
def _fake_graph_capture(device):
    yield SimpleNamespace(stream=None)


# @spec PSTATE-PREP-006 PSTATE-PREP-007
def test_runner_clears_then_recaptures_serving_graphs_of_the_model_state(
    monkeypatch,
):
    events: list[tuple] = []
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model_config = SimpleNamespace(enforce_eager=False)
    runner.model_state = _recording_model_state(events)
    runner.device = torch.device("cpu")
    monkeypatch.setattr(model_runner_module, "graph_capture", _fake_graph_capture)

    runner.recapture_model_state_graphs()

    assert events == [("clear",), ("capture", False)]


# @spec PSTATE-PREP-007
def test_serving_capture_does_not_clear_the_model_state_graphs(monkeypatch):
    events: list[tuple] = []
    runner = _make_capture_runner(monkeypatch, captured=False)
    runner.model_state = _recording_model_state(events)
    runner.device = torch.device("cpu")
    monkeypatch.setattr(model_runner_module, "graph_capture", _fake_graph_capture)
    monkeypatch.setattr(
        model_runner_module, "freeze_gc_for_cudagraph_capture", contextlib.nullcontext
    )
    monkeypatch.setattr(model_runner_module, "lock_workspace", lambda: None)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda: (1 << 30, 1 << 30)
    )

    runner.capture_model()

    assert events == [("capture", False)]


def _waking_core(*, fully_awake: bool, was_sleeping: bool = True):
    """A core whose executor is asleep at level 1 or above when `was_sleeping`
    and, after a wake, asleep still unless `fully_awake`."""
    calls: list[str] = []
    core = _pausable_engine_core_proc()
    core.scheduler.has_unrecomputable_state = True
    core.scheduler.held_state_streams.return_value = []
    core.model_executor.is_sleeping = was_sleeping

    def wake_up(tags):
        calls.append("wake_up")
        core.model_executor.is_sleeping = not fully_awake

    core.model_executor.wake_up.side_effect = wake_up
    core.model_executor.collective_rpc.side_effect = lambda method, *args, **kwargs: (
        calls.append(method)
    )
    core.scheduler.get_kv_connector.return_value = None
    core._set_kv_connector_handshake_metadata = MagicMock()
    core.resume_scheduler = MagicMock(side_effect=lambda: calls.append("resume"))
    return core, calls


# @spec PSTATE-PREP-006
def test_full_wake_recaptures_before_scheduling_resumes():
    core, calls = _waking_core(fully_awake=True)

    EngineCore.wake_up(core)

    assert calls == ["wake_up", "recapture_model_state_graphs", "resume"]


# @spec PSTATE-PREP-006
def test_partial_wake_does_not_recapture():
    core, calls = _waking_core(fully_awake=False)

    EngineCore.wake_up(core, ["weights"])

    assert calls == ["wake_up"]


# @spec PSTATE-PREP-009
def test_light_wake_keeps_captured_graphs():
    core, calls = _waking_core(fully_awake=True, was_sleeping=False)

    EngineCore.wake_up(core, ["scheduling"])

    assert calls == ["resume"]


# @spec PSTATE-PREP-009
@pytest.mark.parametrize("tags", [None, ["weights", "kv_cache"]])
def test_wake_of_an_awake_executor_does_not_recapture(tags):
    """After a level-0 sleep, or a repeated wake, nothing was remapped."""
    core, calls = _waking_core(fully_awake=True, was_sleeping=False)

    EngineCore.wake_up(core, tags)

    assert calls == ["wake_up", "resume"]


# @spec PSTATE-PREP-006
def test_checkpoint_restore_recaptures_after_the_kv_cache_wakes():
    core = _pausable_engine_core_proc()
    core.scheduler.has_unrecomputable_state = True
    core.scheduler.held_state_streams.return_value = []
    core.vllm_config = MagicMock()
    core.model_executor.is_sleeping = False
    core._reset_caches = MagicMock()
    core.reject_new_requests = False
    core.scheduler.get_kv_connector.return_value = None
    core.checkpoint_prepare("wait")
    core.model_executor.reset_mock()
    # Preparing released the KV cache; the restore's wake maps it again.
    core.model_executor.is_sleeping = True
    core.model_executor.wake_up.side_effect = lambda tags: setattr(
        core.model_executor, "is_sleeping", False
    )

    core.checkpoint_restore()

    assert _executor_calls(core) == [
        "checkpoint_restore",
        "wake_up",
        "recapture_model_state_graphs",
    ]


# Bounded history


SHARED_OBJECT_MODULES = ("vllm.config", "vllm.tokenizers", "transformers", "tokenizers")
"""Objects every request shares, which a walk stops at: they are not a
request's retained history, and they reach the rest of the engine."""


def retained_bytes(*roots, exclude=()) -> int:
    """Host bytes reachable from `roots`, not through `exclude`: `sys.getsizeof`
    of every object, plus the whole backing storage of each tensor and NumPy
    array once, deduplicated by address, so a small view that keeps a large
    buffer alive counts the buffer. The walk stops at classes, modules,
    functions and the shared config and tokenizer objects."""
    seen: set[int] = {id(obj) for obj in exclude}
    storages: dict[int, int] = {}
    total = 0
    stack = list(roots)
    skipped = (
        type,
        types.ModuleType,
        types.FunctionType,
        types.BuiltinFunctionType,
        types.MethodType,
    )
    while stack:
        obj = stack.pop()
        if (
            id(obj) in seen
            or isinstance(obj, skipped)
            or type(obj).__module__.startswith(SHARED_OBJECT_MODULES)
        ):
            continue
        seen.add(id(obj))
        total += sys.getsizeof(obj)
        if isinstance(obj, torch.Tensor):
            storage = obj.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
        elif isinstance(obj, np.ndarray):
            base = obj
            while isinstance(base.base, np.ndarray):
                base = base.base
            address = base.__array_interface__["data"][0]
            storages[address] = max(storages.get(address, 0), base.nbytes)
        stack.extend(gc.get_referents(obj))
    return total + sum(storages.values())


NUM_TURNS = 1024
"""Turns each stream runs; enough that history growth of even a few bytes a
turn would show, while the test runs in seconds."""
MEASURE_EVERY = 8
"""Turns between two walks of the retained structures."""
CALIBRATION_SAMPLES = 16
"""Walks that set each structure's bound; they cover every input and output
length, whose cycles are shorter than the turns they span."""
MAX_INPUT = 24
MAX_OUTPUT = 8
READ_EVERY = 4
"""Turns the client lets pile up before it reads them all."""


def _input_len(turn: int) -> int:
    return 1 + turn * 7 % MAX_INPUT


def _output_len(turn: int) -> int:
    return 1 + turn * 3 % MAX_OUTPUT


def _check_bounded(sizes: dict[str, list[int]]) -> None:
    """Each structure stays within twice its size over the first walks."""
    for name, structure_sizes in sizes.items():
        bound = 2 * max(structure_sizes[:CALIBRATION_SAMPLES]) + 4096
        assert max(structure_sizes) <= bound, (
            f"{name} retained {max(structure_sizes)} bytes by the last of "
            f"{len(structure_sizes)} walks; {bound} after the first "
            f"{CALIBRATION_SAMPLES}"
        )


def _measured(turn: int) -> bool:
    return turn % MEASURE_EVERY == MEASURE_EVERY - 1


# @spec PSTATE-HIST-008
@pytest.mark.cpu_test
@pytest.mark.parametrize("backing", ["tensor", "array"])
def test_retained_bytes_counts_the_whole_buffer_behind_a_view(backing):
    if backing == "tensor":
        view = torch.zeros(1 << 16, dtype=torch.uint8)[:8]
    else:
        view = np.zeros(1 << 16, dtype=np.uint8)[:8]

    assert retained_bytes(view) >= 1 << 16


# @spec PSTATE-HIST-008
@pytest.mark.cpu_test
@pytest.mark.parametrize("backing", ["tensor", "array"])
def test_retained_bytes_detects_a_planted_growing_buffer(backing):
    """A one-element view of a buffer that grows each turn, as a leak that a
    shallow size would miss."""
    zeros = torch.zeros if backing == "tensor" else np.zeros
    dtype = torch.uint8 if backing == "tensor" else np.uint8
    kept: list = []
    sizes = []
    for turn in range(CALIBRATION_SAMPLES * 16):
        kept[:] = [zeros(64 * (turn + 1), dtype=dtype)[:1]]
        sizes.append(retained_bytes(kept))

    with pytest.raises(AssertionError, match="retained"):
        _check_bounded({"planted": sizes})


def _chunk(prompt: list[int]) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="stream",
        prompt_token_ids=prompt,
        mm_features=None,
        sampling_params=SamplingParams(
            detokenize=False, output_kind=RequestOutputKind.DELTA
        ),
        pooling_params=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        resumable=True,
        external_req_id="stream-ext",
    )


# @spec PSTATE-HIST-008
@pytest.mark.cpu_test
def test_output_processor_history_stays_bounded_over_many_turns():
    processor = OutputProcessor(None, log_stats=False, persistent_state=True)
    queue = StreamingInputOutputCollector(RequestOutputKind.DELTA, "stream")
    processor.add_request(_chunk([1] * _input_len(0)), None, queue=queue)
    sizes: dict[str, list[int]] = {
        "request state": [],
        "input queue": [],
        "output backlog": [],
    }
    for turn in range(NUM_TURNS):
        processor.add_request(_chunk([2] * _input_len(turn + 1)), None)
        processor.process_outputs(
            [
                EngineCoreOutput(
                    request_id="stream",
                    new_token_ids=[3] * _output_len(turn),
                    finish_reason=FinishReason.STOP,
                )
            ]
        )
        if _measured(turn):
            state = processor.request_states["stream"]
            sizes["request state"].append(
                retained_bytes(state, exclude=(state.queue, state.input_chunk_queue))
            )
            sizes["input queue"].append(retained_bytes(state.input_chunk_queue))
            sizes["output backlog"].append(retained_bytes(queue))
        if turn % READ_EVERY == READ_EVERY - 1:
            while queue.get_nowait() is not None:
                pass

    _check_bounded(sizes)


def _stream_input(turn: int) -> Request:
    params = SamplingParams(max_tokens=64, output_kind=RequestOutputKind.DELTA)
    (request,) = create_requests(
        1,
        num_tokens=_input_len(turn),
        req_ids=["stream"],
        block_size=512,
        sampling_params=params,
    )
    request.resumable = True
    request.continuation = turn > 0
    return request


# @spec PSTATE-HIST-008
@pytest.mark.cpu_test
def test_scheduler_request_history_stays_bounded_over_many_turns(monkeypatch):
    """A stream of a model that ignores positions keeps one frontier token and
    one input in its request and in what the scheduler re-sends, however many
    turns it has run."""
    declare_state_facts(monkeypatch, persistent_state_reads_positions=False)
    spec = MambaSpec(
        block_size=512,
        shapes=((1,),),
        dtypes=(torch.int64,),
        mamba_type=MambaAttentionBackendEnum.PERSISTENT_STATE,
    )
    scheduler = create_scheduler(
        max_num_seqs=1,
        max_model_len=512,
        block_size=512,
        num_blocks=3,
        kv_cache_layer_config=MambaManager.layer_config(spec, recomputable=False),
    )
    sizes: dict[str, list[int]] = {
        "request": [],
        "re-sent history": [],
        "input queue": [],
    }
    for turn in range(NUM_TURNS):
        scheduler.add_request(_stream_input(turn))
        measured = _measured(turn)
        resent = 0
        for step in range(_output_len(turn)):
            output = scheduler.schedule()
            if step == 0 and measured:
                resent = retained_bytes(
                    output.scheduled_new_reqs, output.scheduled_cached_reqs
                )
            token = EOS_TOKEN_ID if step == _output_len(turn) - 1 else 7
            scheduler.update_from_output(
                output,
                ModelRunnerOutput(req_ids=["stream"], sampled_token_ids=[[token]]),
            )
        if not measured:
            continue
        request = scheduler.requests["stream"]
        sizes["request"].append(
            retained_bytes(request, exclude=(request.streaming_queue,))
        )
        sizes["re-sent history"].append(resent)
        sizes["input queue"].append(retained_bytes(request.streaming_queue))

    _check_bounded(sizes)


# Declined by design


# @spec PSTATE-ASYNC-005 PSTATE-ASYNC-006
@pytest.mark.skip(
    reason="the echo declaration is accepted only after the A40 exactly-once "
    "test and latency comparison of PSTATE-ASYNC-006, which need the plugin"
)
def test_echo_declaration_is_accepted_only_after_the_a40_gate():
    raise AssertionError


# @spec PSTATE-PAGE-005
@pytest.mark.skip(
    reason="a model's write on a side stream is its own duty; core cannot "
    "observe it, and no core behaviour changes"
)
def test_model_joins_side_streams_before_its_forward_returns():
    raise AssertionError
