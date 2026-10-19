# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
from vllm.config import CacheConfig, LoRAConfig, ParallelConfig
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.model_runner import ExecuteModelState, GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.standalone_ar import speculator as ar_module
from vllm.v1.worker.gpu.spec_decode.utils import get_drafter_hidden_states


def test_non_last_pp_rank_uses_global_batch_for_sample_feedback(monkeypatch):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.state_slots = None
    runner.is_last_pp_rank = False
    local_batch = object()
    global_batch = SimpleNamespace(idx_mapping=object())
    # Keep handles on the mocks: the runner's attributes are typed as the real
    # PCPManager / PPHandler / ModelState, which have no assert_* methods.
    restore_for_sampling = Mock()
    runner.pcp_manager = SimpleNamespace(
        global_batch=global_batch,
        restore_for_sampling=restore_for_sampling,
    )
    pp_receive = Mock(return_value=False)
    runner.pp_handler = SimpleNamespace(receive=pp_receive)
    postprocess_num_computed_tokens = Mock()
    monkeypatch.setattr(
        runner, "postprocess_num_computed_tokens", postprocess_num_computed_tokens
    )
    postprocess_state = Mock()
    runner.model_state = SimpleNamespace(postprocess_state=postprocess_state)
    runner.kv_connector = SimpleNamespace(post_forward=Mock(return_value=None))
    runner.eplb = SimpleNamespace(step=Mock())
    runner.execute_model_state = ExecuteModelState(
        input_batch=local_batch,
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=None,
        aux_hidden_states=None,
        dp_sync_state=None,
        finished_req_ids=set(),
        ec_connector_output=None,
        cudagraph_stats=None,
        num_spec_tokens_to_schedule=0,
    )

    runner.sample_tokens(None)

    pp_receive.assert_called_once_with(global_batch)
    postprocess_num_computed_tokens.assert_called_once_with(global_batch)
    postprocess_state.assert_called_once_with(global_batch.idx_mapping, 0)
    restore_for_sampling.assert_not_called()


def test_qsa_circular_group_uses_custom_slot_mapping(monkeypatch):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.state_slots = None
    runner.max_model_len = 262144
    runner.is_encoder_decoder = False
    runner.dcp_size = 1
    runner.dcp_rank = 0
    runner.cp_interleave = 1
    runner.cache_config = SimpleNamespace(enable_prefix_caching=True)
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=1,
        cp_kv_cache_interleave_size=1,
    )
    runner.parallel_config = parallel_config
    runner.vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
    )
    runner.jit_warmup_registry = JitWarmupRegistry(runner.vllm_config)
    runner.model_state = SimpleNamespace(
        get_additional_cg_support=lambda: (),
        num_new_sampled_tokens_per_step=1,
    )
    runner.speculator = None
    runner.req_states = []
    runner.input_buffers = SimpleNamespace(query_start_loc=None)
    runner.vocab_size = 1
    runner.max_num_reqs = 1
    runner.max_num_tokens = 2
    runner.device = torch.device("cuda")

    raw_spec = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    compressed_spec = FullAttentionSpec(
        block_size=262144,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["raw"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=8,
                    kv_cache_specs={"raw": raw_spec},
                ),
            ),
            KVCacheGroupSpec(layer_names=["compressed"], kv_cache_spec=compressed_spec),
        ],
    )

    class FakeAttnCGSupport:
        def narrow(self, *args):
            return self

    attn_cg_support = FakeAttnCGSupport()
    monkeypatch.setattr(
        model_runner_module,
        "init_attn_backend",
        lambda *args, **kwargs: ([], attn_cg_support, [8, 262144]),
    )
    monkeypatch.setattr(
        model_runner_module,
        "maybe_create_adaptive_verification_manager",
        lambda **kwargs: None,
    )

    captured = {}

    class BlockTablesCaptured(Exception):
        pass

    def capture_block_tables(**kwargs):
        captured.update(kwargs)
        raise BlockTablesCaptured

    monkeypatch.setattr(model_runner_module, "BlockTables", capture_block_tables)

    with pytest.raises(BlockTablesCaptured):
        runner.initialize_kv_cache(kv_cache_config)

    assert captured["max_num_blocks_per_group"] == [1, 1]
    assert captured["slot_mapping_enabled"] == [False, True]


@pytest.mark.parametrize(
    ("mamba_cache_mode", "num_speculative_blocks", "expected"),
    [
        pytest.param("align", 0, 65_536, id="align-prefix-cache"),
        pytest.param("none", 7, 8, id="no-prefix-cache-with-speculation"),
    ],
)
def test_initialize_kv_cache_does_not_dcp_shard_mamba_block_table(
    monkeypatch,
    mamba_cache_mode: str,
    num_speculative_blocks: int,
    expected: int,
):
    """Mamba/GDN block-table rows index global positions, unlike DCP KV."""
    max_model_len = 1_048_576
    attention_block_size = 1_536
    mamba_block_size = 16
    dcp_size = 8
    full_attention_spec = FullAttentionSpec(
        block_size=attention_block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.bfloat16,
    )
    mamba_spec = MambaSpec(
        shapes=((1,),),
        dtypes=(torch.bfloat16,),
        block_size=mamba_block_size,
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention"], full_attention_spec),
            KVCacheGroupSpec(["kda"], mamba_spec),
        ],
    )
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=dcp_size,
        cp_kv_cache_interleave_size=1,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_cache_mode),
    )
    runner = SimpleNamespace(
        max_model_len=max_model_len,
        is_encoder_decoder=False,
        vllm_config=vllm_config,
        parallel_config=parallel_config,
    )

    class _CapturedWidths(Exception):
        pass

    captured: list[int] = []

    def capture_width(max_num_blocks: int, *_args, **_kwargs) -> int:
        captured.append(max_num_blocks)
        if len(captured) == 2:
            raise _CapturedWidths
        return max_num_blocks

    monkeypatch.setattr(model_runner_module, "get_block_table_width", capture_width)

    with pytest.raises(_CapturedWidths):
        GPUModelRunner.initialize_kv_cache(runner, kv_cache_config)

    # Attention KV is local to one of eight DCP ranks; KDA state is replicated
    # and therefore needs one table entry for every global 16-token page.
    assert captured == [86, expected]


def test_append_block_ids_rejects_write_past_row_capacity():
    """Reject an oversized staged write before it can corrupt the next row."""

    class _BlockTable:
        gpu = torch.empty((2, 4), dtype=torch.int32)

        def stage_write(self, *_args):
            pytest.fail("an oversized write must not be staged")

    block_tables = BlockTables.__new__(BlockTables)
    block_tables.num_kv_cache_groups = 1
    # append_block_ids only touches .gpu and .stage_write on each table.
    block_tables.block_tables = [_BlockTable()]  # type: ignore[list-item]  # duck-typed stub
    block_tables.num_blocks = SimpleNamespace(
        np=torch.tensor([[0, 3]], dtype=torch.int32)
    )

    with pytest.raises(
        RuntimeError,
        match=r"request 1, group 0 exceeds row capacity \(5 > 4\)",
    ):
        block_tables.append_block_ids(
            req_index=1,
            new_block_ids=([4, 5],),
            overwrite=False,
        )

    assert block_tables.num_blocks.np[0, 1] == 3


@contextlib.contextmanager
def _no_dummy_loras(
    lora_config: LoRAConfig | None, remove_lora: bool = True
) -> Iterator[None]:
    """No-op stand-in for LoRAModelRunnerMixin.maybe_setup_dummy_loras."""
    yield


def _make_capture_runner(
    monkeypatch: pytest.MonkeyPatch, captured: bool
) -> GPUModelRunner:
    """Minimal V2 runner for capture_model: fakes everything except the
    cudagraph_manager's needs_capture decision."""
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.state_slots = None
    runner.model_state = SimpleNamespace(
        supports_mm_inputs=False, capture_inner_cudagraphs=lambda *args: None
    )
    runner.cudagraph_manager = SimpleNamespace(
        needs_capture=lambda: captured,
        capture=lambda *args, **kwargs: None,
    )
    runner.lora_config = None
    monkeypatch.setattr(runner, "maybe_setup_dummy_loras", _no_dummy_loras)
    runner.speculator = None
    runner.adaptive_verification = None
    runner.model = None
    runner.input_buffers = None
    runner.pcp_manager = None
    runner.intermediate_tensors = None
    runner.block_tables = None
    runner.attn_groups = None
    runner.kv_cache_config = None
    runner.use_aux_hidden_state_outputs = False
    runner.kv_connector = model_runner_module.NO_OP_KV_CONNECTOR
    return runner


def test_capture_model_locks_workspace_after_capture(monkeypatch):
    """A workspace resize after capture frees the buffer the captured graphs
    baked in, so capture_model must lock the workspace before returning
    (https://github.com/vllm-project/vllm/issues/55336)."""
    runner = _make_capture_runner(monkeypatch, captured=True)
    monkeypatch.setattr(
        model_runner_module, "freeze_gc_for_cudagraph_capture", contextlib.nullcontext
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda: (1 << 30, 1 << 30)
    )
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    runner.capture_model()

    assert lock_calls == ["lock"]


def test_capture_model_skips_lock_when_nothing_captured(monkeypatch):
    """With no graphs to capture (e.g. enforce_eager) there is nothing baked
    into the workspace, so the early return must not lock it."""
    runner = _make_capture_runner(monkeypatch, captured=False)
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    assert runner.capture_model() == 0
    assert lock_calls == []


def test_capture_model_profile_only_skips_lock(monkeypatch):
    """The memory-profiling capture pass runs before kernel warmup and the
    real capture; locking there would stop the warmup from growing the
    workspace to its scheduler-realistic size."""
    runner = _make_capture_runner(monkeypatch, captured=True)
    monkeypatch.setattr(
        model_runner_module, "freeze_gc_for_cudagraph_capture", contextlib.nullcontext
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda: (1 << 30, 1 << 30)
    )
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    runner.capture_model(profile_only=True)

    assert lock_calls == []


@pytest.mark.parametrize("target_buffer", ["absent", "none", "tensor"])
def test_get_drafter_hidden_states_tolerates_missing_target_buffer(target_buffer):
    """Targets allocate the MTP hidden buffer only for hidden-state drafters."""
    hidden_states = torch.zeros(4, 8)
    buffer = torch.arange(16 * 8, dtype=torch.float32).view(16, 8)
    if target_buffer == "absent":
        model = SimpleNamespace()
    else:
        returned = buffer if target_buffer == "tensor" else None
        model = SimpleNamespace(get_mtp_target_hidden_states=lambda: returned)

    out = get_drafter_hidden_states(model, hidden_states)

    if target_buffer == "tensor":
        assert torch.equal(out, buffer[:4])
    else:
        assert out is hidden_states


def test_async_copy_to_np_does_not_alias_reused_buffer():
    buffer = torch.zeros(4, dtype=torch.int64)

    snapshot = async_copy_to_np(buffer)
    buffer.fill_(1)

    assert snapshot.tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize(
    ("draft_dtype", "expected_dtype"),
    [("auto", "auto"), ("bfloat16", "bfloat16"), (None, "fp8_e4m3")],
)
def test_standalone_draft_kv_dtype_override(monkeypatch, draft_dtype, expected_dtype):
    @dataclass
    class Config:
        cache_config: CacheConfig
        parallel_config: ParallelConfig
        model_config: object
        quant_config: object | None = None

    target = Config(CacheConfig(cache_dtype="fp8_e4m3"), ParallelConfig(), object())
    speculator = object.__new__(ar_module.StandaloneARSpeculator)
    speculator.vllm_config = target
    speculator.draft_model_config = object()
    speculator.speculative_config = SimpleNamespace(
        kv_cache_dtype=draft_dtype, draft_parallel_config=ParallelConfig()
    )
    get_model = Mock(return_value=object())
    monkeypatch.setattr(ar_module, "get_model", get_model)
    monkeypatch.setattr(ar_module, "set_model_tag", lambda _: contextlib.nullcontext())

    speculator.load_draft_model(object(), set())

    draft_config = get_model.call_args.kwargs["vllm_config"]
    assert draft_config.cache_config.cache_dtype == expected_dtype
    assert speculator.attn_vllm_config.cache_config.cache_dtype == expected_dtype
    assert target.cache_config.cache_dtype == "fp8_e4m3"


def test_state_only_cache_reaches_mrv2_initialization(monkeypatch):
    """Real MRv2 initialization exposes typed per-slot views from core allocation."""
    from tests.v1.core.utils import create_scheduler, create_state_spec
    from vllm.config.compilation import CUDAGraphMode
    from vllm.platforms import current_platform
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_planning import get_kv_cache_configs
    from vllm.v1.worker.gpu import buffer_utils
    from vllm.v1.worker.gpu.model_states.interface import ModelState

    monkeypatch.setattr(buffer_utils, "is_uva_available", lambda: False)
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    config = create_scheduler(max_num_seqs=2, max_model_len=32).vllm_config
    config.cache_config.kv_cache_layout = "LBNHC"
    state = create_state_spec()
    plan = get_kv_cache_configs(config, [{"state": state}], [96])[0]
    from vllm.model_executor.layers.state import PersistentStateLayerBase
    from vllm.v1.worker.gpu.attn_utils import get_kv_cache_spec

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    with pytest.raises(ValueError, match="require Model Runner V2"):
        PersistentStateLayerBase(state, prefix="state", vllm_config=config)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    PersistentStateLayerBase(state, prefix="state", vllm_config=config)
    assert get_kv_cache_spec(config)["state"] == state

    class _ModelState:
        num_new_sampled_tokens_per_step = 1
        initialize_state_slots = ModelState.initialize_state_slots

        def get_additional_cg_support(self):
            return AttentionCGSupport.ALWAYS, None

        def initialize_kv_cache(self, cache_config, block_tables):
            assert len(cache_config.kv_cache_groups) == 1
            assert block_tables.num_kv_cache_groups == 1

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.state_slots = None
    runner.vllm_config = config
    runner.parallel_config = config.parallel_config
    runner.model_config = config.model_config
    runner.scheduler_config = config.scheduler_config
    runner.cache_config = config.cache_config
    runner.compilation_config = config.compilation_config
    runner.jit_warmup_registry = JitWarmupRegistry(config)
    runner.max_model_len = 32
    runner.is_encoder_decoder = False
    runner.dcp_size = 1
    runner.dcp_rank = 0
    runner.pcp_manager = None
    runner.speculator = None
    runner.req_states = []
    runner.input_buffers = SimpleNamespace(query_start_loc=None)
    runner.vocab_size = 100
    runner.max_num_reqs = 2
    runner.max_num_tokens = 32
    runner.device = torch.device("cpu")
    runner.model_state = _ModelState()
    runner.model = torch.nn.Module()
    runner.decode_query_len = 1
    runner.lora_capture_cases = []
    monkeypatch.setattr(
        config.compilation_config,
        "resolve_cudagraph_mode_and_sizes",
        lambda *a, **kw: CUDAGraphMode.NONE,
    )
    monkeypatch.setattr(model_runner_module, "ModelCudaGraphManager", Mock())
    runner.initialize_kv_cache(plan, is_profiling=True)
    assert runner.block_tables.block_sizes == [1]
    assert runner.state_slots is runner.model_state.state_slots
    allocator = KVCacheManager(
        kv_cache_config=plan,
        max_model_len=32,
        scheduler_block_size=1,
        hash_block_size=1,
        enable_caching=False,
    ).coordinator.state_managers["state"]
    a, b = allocator.allocate("a"), allocator.allocate("b")
    views = runner.state_slots.views["state"]
    views.bind("a", a)
    views.bind("b", b)
    av, bv = views.view("a", a), views.view("b", b)
    assert av["hidden"].dtype == torch.float32
    assert av["count"].dtype == torch.int64
    assert av["hidden"].shape == (4,)
    assert torch.all(av["hidden"] == 1)
    assert torch.all(av["count"] == 0)
    assert av["hidden"].untyped_storage() == bv["count"].untyped_storage()
    av["hidden"].fill_(11)
    bv["hidden"].fill_(13)
    assert torch.all(av["hidden"] == 11) and torch.all(bv["hidden"] == 13)
    allocator.retire("a", 0)
    assert allocator.release(a, 0)
    c = allocator.allocate("c")
    views.bind("c", c)
    with pytest.raises(ValueError, match="Stale"):
        views.view("a", a)
    with pytest.raises(ValueError, match="Stale"):
        views.bind("a", a)
    assert torch.all(bv["hidden"] == 13)
    assert torch.all(views.view("c", c)["hidden"] == 1)
    with pytest.raises(ValueError, match="Stale"):
        allocator.complete(a, 10)


def test_state_slot_rows_validate_step_handles_and_alias_resident_pools(monkeypatch):
    from dataclasses import replace

    from tests.v1.core.utils import create_state_spec
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, StateSlotHandle
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
    from vllm.v1.worker.gpu.state_slots import StateSlotBindings

    spec = create_state_spec()
    raw = torch.zeros(4 * spec.page_size_bytes, dtype=torch.uint8)
    slots = StateSlotBindings([KVCacheGroupSpec(["state"], spec)], {"state": raw})
    main = SimpleNamespace(wait_stream=lambda _: None)
    step = SchedulerOutput.make_empty()
    step.state_step_seq = 1
    a, b = (StateSlotHandle("epoch", i, 1) for i in (1, 3))
    step.state_slot_handles = {"a": {"state": a}, "b": {"state": b}}
    with pytest.raises(ValueError, match="live"):
        slots.bind_rows("state", ("b", "a"), (b, a))
    slots.begin_step(step, main)
    with pytest.raises(ValueError, match="live"):
        slots.bind_rows("state", ("b", "a"), (b, a))
    slots.activate_requests()
    access = slots.bind_rows("state", ("b", "a"), (b, a))
    assert access.step_seq == 1 and access.slot_indices_cpu == (3, 1)
    pools = access.component_pools
    assert pools["hidden"].untyped_storage() == raw.untyped_storage()
    assert pools["hidden"].shape == (4, 4)
    pools["hidden"][3].fill_(19)
    assert torch.all(slots.views["state"].view("b", b)["hidden"] == 19)
    assert torch.all(pools["hidden"][1] == 1)
    assert torch.all(pools["hidden"][[0, 2]] == 0)
    for ids, handles in [
        (("a", "a"), (a, a)),
        (("a", "b"), (b, a)),
        (("a",), (replace(a, engine_epoch="old"),)),
        (("a",), (replace(a, generation=2),)),
        (("a",), (replace(a, slot=0),)),
        (("a",), (replace(a, slot=4),)),
        (("a", "b"), (a,)),
    ]:
        with pytest.raises(ValueError):
            slots.bind_rows("state", ids, handles)
    with pytest.raises(ValueError, match="group"):
        slots.bind_rows("other", ("a",), (a,))
    slots.fail_requests({"a"})
    access.validate_current()  # Failure reporting does not invalidate the step.
    monkeypatch.setattr(
        torch.cuda, "Event", lambda **_: SimpleNamespace(record=lambda _: None)
    )
    slots.finish_step(EMPTY_MODEL_RUNNER_OUTPUT, main)
    with pytest.raises(ValueError, match="Stale"):
        access.validate_current()
    slots.begin_step(step, main)  # Even a repeated sequence/handle is a new scope.
    slots.activate_requests()
    with pytest.raises(ValueError, match="Stale"):
        access.validate_current()
    fresh = slots.bind_rows("state", ("a",), (a,))
    slots.views["state"].bind("replacement", replace(a, generation=2))
    before = raw.clone()
    with pytest.raises(ValueError, match="Stale"):
        fresh.validate_current()
    assert torch.equal(raw, before)


def test_state_completion_waits_for_postprocess_and_registered_stream(monkeypatch):
    """Completion follows postprocessing and all state-writing streams."""
    from tests.v1.core.utils import create_state_spec
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import StateSlotHandle
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
    from vllm.v1.worker.gpu import async_utils
    from vllm.v1.worker.gpu.state_slots import StateSlotBindings

    events = []
    main = SimpleNamespace(wait_stream=lambda stream: events.append("join"))

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self, stream):
            self.kind = "state" if stream is main else "copy"
            events.append(self.kind + "-record")

        def synchronize(self):
            events.append(self.kind + "-wait")

    monkeypatch.setattr(torch.cuda, "Event", Event)
    slots = StateSlotBindings(
        [KVCacheGroupSpec(["state"], create_state_spec())],
        {"state": torch.zeros(96, dtype=torch.int8)},
    )
    slots.register_writing_stream(
        SimpleNamespace(wait_stream=lambda stream: events.append("ready"))
    )
    scheduled = SchedulerOutput.make_empty()
    scheduled.state_step_seq = 7
    scheduled.state_slot_handles = {"a": {"state": StateSlotHandle("epoch", 1, 1)}}
    slots.begin_step(scheduled, main)
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.state_slots = slots
    runner.main_stream = main
    runner.output_copy_stream = SimpleNamespace(
        wait_stream=lambda stream: events.append("copy-join")
    )
    runner.is_last_pp_rank = True
    runner.pcp_manager = runner.pp_handler = runner.speculator = None
    runner.aux_output_connector = None
    runner.num_speculative_steps = 0
    runner.check_ep_fault = False
    runner.eplb = SimpleNamespace(step=Mock())
    runner.kv_connector = SimpleNamespace(post_forward=Mock(return_value=None))
    runner.model = SimpleNamespace(compute_logits=Mock())
    runner.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=None),
        num_computed_tokens=SimpleNamespace(gpu=None),
        prompt_len=SimpleNamespace(np=None),
    )
    runner.prompt_logprobs_worker = SimpleNamespace(
        compute_prompt_logprobs=Mock(return_value={}),
        compute_prompt_token_id_logprobs=Mock(return_value={}),
    )
    runner.sample = lambda *a: (
        SimpleNamespace(
            sampled_token_ids=torch.tensor([[1]]),
            logprobs_tensors=None,
            num_nans=None,
            sampling_mask_tensors=None,
        ),
        torch.tensor([1]),
        torch.tensor([0]),
    )
    runner.postprocess_sampled = lambda *a: events.append("postprocess")
    runner.execute_model_state = ExecuteModelState(
        input_batch=SimpleNamespace(
            req_ids=["a"], idx_mapping=None, query_start_loc=None
        ),
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=torch.empty(1, 1),
        aux_hidden_states=None,
        dp_sync_state=None,
        finished_req_ids=set(),
        ec_connector_output=None,
        cudagraph_stats=None,
        num_spec_tokens_to_schedule=0,
    )

    monkeypatch.setattr(async_utils, "stream", lambda *args: contextlib.nullcontext())
    monkeypatch.setattr(async_utils, "async_copy_to_np", lambda tensor: tensor.numpy())
    pending = runner.sample_tokens(None)
    assert EMPTY_MODEL_RUNNER_OUTPUT.state_completion is None
    expected = [
        "join",
        "ready",
        "copy-join",
        "copy-record",
        "postprocess",
        "join",
        "state-record",
    ]
    assert events == expected
    result = pending.get_output()
    assert events == expected + ["state-wait", "copy-wait"]
    assert result.sampled_token_ids == [[1]]
    assert result.state_completion.step_seq == 7
    assert result.state_completion.handles == scheduled.state_slot_handles
    assert EMPTY_MODEL_RUNNER_OUTPUT.state_completion is None


def test_zero_token_state_step_returns_completion(monkeypatch):
    """No-forward executes the same completion contract."""
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT

    runner = GPUModelRunner.__new__(GPUModelRunner)
    calls = []

    def finish(output, stream):
        calls.append("done")
        return output

    runner.state_slots = SimpleNamespace(
        begin_step=lambda step, stream: calls.append("begin"),
        finish_step=finish,
        activate_requests=lambda: None,
    )
    runner.model_state = SimpleNamespace(prepare_step=lambda *args, **kwargs: None)
    runner.main_stream = object()
    for name in (
        "update_pp_decode_requests",
        "finish_requests",
        "free_states",
        "add_requests",
        "update_requests",
    ):
        setattr(runner, name, lambda *args: None)
    runner.block_tables = SimpleNamespace(apply_staged_writes=lambda: None)
    runner.aux_output_connector = None
    runner.kv_connector = SimpleNamespace(
        no_forward=lambda step: EMPTY_MODEL_RUNNER_OUTPUT
    )
    runner._merge_ec_connector_no_forward = lambda step, output: output
    assert (
        runner.execute_model(SchedulerOutput.make_empty()) is EMPTY_MODEL_RUNNER_OUTPUT
    )
    assert calls == ["begin", "done"]


@pytest.mark.parametrize("override", [2, 4])
def test_persistent_state_profiling_ignores_block_override(override):
    from tests.v1.core.utils import create_scheduler, create_state_spec
    from vllm.v1.core.kv_cache_planning import get_kv_cache_configs
    from vllm.v1.worker.gpu.cudagraph_utils import _init_minimal_kv_cache_for_profiling

    config = create_scheduler(max_num_seqs=2, max_model_len=32).vllm_config
    config.cache_config.kv_cache_layout = "LBNHC"
    config.cache_config.num_gpu_blocks_override = override
    spec = create_state_spec()
    assert (
        get_kv_cache_configs(config, [{"state": spec}], [128])[0].num_blocks == override
    )
    runner = SimpleNamespace(
        vllm_config=config,
        cache_config=config.cache_config,
        max_num_reqs=2,
        get_kv_cache_spec=lambda: {"state": spec},
        initialize_kv_cache=Mock(),
    )
    _init_minimal_kv_cache_for_profiling(runner)
    assert runner.initialize_kv_cache.call_args.args[0].num_blocks == 3
    assert config.cache_config.num_gpu_blocks_override == override


@pytest.mark.parametrize("lose_ack", [False, True])
def test_state_cleanup_is_fenced_through_real_runner(monkeypatch, lose_ack):
    from tests.v1.core.test_scheduler import _state_output, _state_scheduler
    from tests.v1.core.utils import create_requests, create_state_spec
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
    from vllm.v1.request import RequestStatus
    from vllm.v1.worker.gpu.state_slots import StateSlotBindings

    scheduler = _state_scheduler()
    (req,) = create_requests(num_requests=1, num_tokens=4)
    scheduler.add_request(req)
    step = scheduler.schedule()
    slots = StateSlotBindings(
        [KVCacheGroupSpec(["state"], create_state_spec())],
        {"state": torch.zeros(96, dtype=torch.int8)},
    )
    events = []
    main = SimpleNamespace(wait_stream=lambda stream: events.append("join"))
    writer = SimpleNamespace(wait_stream=lambda stream: events.append("ready"))
    slots.register_writing_stream(writer)

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self, stream):
            events.append("record")

        def synchronize(self):
            events.append("wait")

    monkeypatch.setattr(torch.cuda, "Event", Event)
    slots.begin_step(step, main)
    slots.activate_requests()
    slots.finish_step(_state_output(step), main).get_output()
    scheduler.update_from_output(step, _state_output(step))
    scheduler.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)
    manager = scheduler.kv_cache_manager.coordinator.state_managers["state"]
    assert manager.num_free_slots == 1
    cleanup = scheduler.schedule()
    old = cleanup.state_cleanup_handles[req.request_id]["state"]
    events.clear()

    def remove(request_id):
        assert slots.get_handle(request_id, "state") == old
        slots.views["state"].view(request_id, old)["hidden"].fill_(17)
        events.append("cleanup")

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.state_slots = slots
    runner.main_stream = main
    runner.model_state = SimpleNamespace(
        remove_request=remove, prepare_step=lambda *args, **kwargs: None
    )
    runner.req_states = SimpleNamespace(remove_request=lambda _: 0)
    runner.lora_state = SimpleNamespace(remove_request=lambda _: None)
    runner.pooling_runner = runner.pp_handler = runner.encoder_cache = None
    runner.prompt_logprobs_worker = runner.aux_output_connector = None
    for name in ("update_pp_decode_requests", "add_requests", "update_requests"):
        setattr(runner, name, lambda *args: None)
    runner.block_tables = SimpleNamespace(apply_staged_writes=lambda: None)
    runner.kv_connector = SimpleNamespace(
        no_forward=lambda _: EMPTY_MODEL_RUNNER_OUTPUT
    )
    runner._merge_ec_connector_no_forward = lambda _, output: output
    pending = runner.execute_model(cleanup)
    assert events.index("cleanup") < events.index("record")
    assert events[-2:] == ["join", "record"]
    assert manager.num_free_slots == 1
    output = pending.get_output()
    assert events[-1] == "wait"
    assert output.state_completion.cleanup_handles == cleanup.state_cleanup_handles
    if lose_ack:
        output.state_completion = None
        scheduler.update_from_output(cleanup, output)
        assert manager.num_free_slots == 1
        assert scheduler.has_requests()
        cleanup = scheduler.schedule()
        assert not cleanup.finished_req_ids
        output = runner.execute_model(cleanup).get_output()
        assert events.count("cleanup") == 1
    scheduler.update_from_output(cleanup, output)
    assert manager.num_free_slots == 2
    assert not scheduler.has_requests()
    assert EMPTY_MODEL_RUNNER_OUTPUT.state_completion is None

    # Reuse the cleaned slot with the same request ID.
    replacement = manager.allocate(req.request_id)
    assert replacement.slot == old.slot
    assert replacement.generation > old.generation
    rebind = SchedulerOutput.make_empty()
    rebind.state_step_seq = cleanup.state_step_seq + 1
    rebind.state_slot_handles = {req.request_id: {"state": replacement}}

    def add_requests(step):
        assert slots.get_handle(req.request_id, "state") == replacement
        view = slots.views["state"].view(req.request_id, replacement)
        assert torch.all(view["hidden"] == 1)
        with pytest.raises(ValueError, match="Stale"):
            slots.views["state"].view(req.request_id, old)

    runner.add_requests = add_requests
    result = runner.execute_model(rebind).get_output()
    assert result.state_completion.handles == rebind.state_slot_handles
    assert events.count("cleanup") == 1


def test_nemotron_core_admission_park_cleanup_and_slot_reuse(monkeypatch):
    """Real core/runner lifecycle; CPU tensor writes stand in for GPU execution."""
    from tests.models.nemotron_asr.test_persistent_model_state_v2 import (
        _config,
        _input,
        _state,
    )
    from tests.v1.core.test_scheduler import (
        _admit_stream,
        _state_output,
        _stream_chunk,
        _stream_request,
    )
    from tests.v1.core.utils import create_scheduler
    from vllm.model_executor.layers.state import PersistentStateLayerBase
    from vllm.models.nemotron_asr.state_profile import (
        build_nemotron_persistent_state_spec,
    )
    from vllm.v1.core.kv_cache_planning import get_kv_cache_configs
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
    from vllm.v1.worker.gpu.attn_utils import get_kv_cache_spec
    from vllm.v1.worker.gpu.state_slots import StateSlotBindings

    base = create_scheduler(max_num_seqs=2, max_model_len=32)
    config = base.vllm_config
    config.cache_config.kv_cache_layout = "LBNHC"
    spec = build_nemotron_persistent_state_spec(_config())
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    PersistentStateLayerBase(spec, prefix="state", vllm_config=config)
    discovered = get_kv_cache_spec(config)
    assert discovered == {"state": spec}
    plan = get_kv_cache_configs(config, [discovered], [3 * spec.page_size_bytes])[0]
    assert plan.num_blocks == 3 and len(plan.kv_cache_groups) == 1
    scheduler = Scheduler(
        vllm_config=config,
        kv_cache_config=plan,
        block_size=1,
        log_stats=False,
        structured_output_manager=base.structured_output_manager,
    )
    core = object.__new__(EngineCore)
    core.scheduler, core.request_block_hasher = scheduler, None
    scheduler.stream_admissions.clock = lambda: 100.0
    assert _admit_stream(core).accepted
    assert _admit_stream(core, 2, _stream_request("peer")).accepted
    assert not _admit_stream(core, 3, _stream_request("full")).accepted
    slots = StateSlotBindings(
        plan.kv_cache_groups,
        {"state": torch.zeros(3 * spec.page_size_bytes, dtype=torch.uint8)},
    )
    state = _state()
    state.initialize_state_slots(slots)
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.state_slots, runner.model_state = slots, state
    runner.main_stream = SimpleNamespace(wait_stream=lambda _: None)
    indices: dict[str, int] = {}

    def add_request(**kwargs):
        indices[kwargs["req_id"]] = len(indices)

    runner.req_states = SimpleNamespace(
        req_id_to_index=indices,
        add_request=add_request,
        remove_request=lambda r: indices.pop(r, None),
        apply_staged_writes=lambda: None,
    )
    runner.block_tables = SimpleNamespace(
        append_block_ids=lambda *a, **kw: None, apply_staged_writes=lambda: None
    )
    runner.lora_state = SimpleNamespace(
        add_request=lambda *a: None, remove_request=lambda r: None
    )
    runner.pooling_runner = runner.pp_handler = runner.encoder_cache = None
    runner.prompt_logprobs_worker = runner.sampler = None
    runner.aux_output_connector = runner.adaptive_verification = None
    runner.is_last_pp_rank = False
    runner.update_pp_decode_requests = lambda: None
    runner.update_requests = lambda _: None
    runner.kv_connector = SimpleNamespace(
        no_forward=lambda _: EMPTY_MODEL_RUNNER_OUTPUT
    )
    runner._merge_ec_connector_no_forward = lambda _, output: output
    events = []

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self, stream):
            events.append("record")

        def synchronize(self):
            events.append("wait")

    monkeypatch.setattr(torch.cuda, "Event", Event)

    def feed():
        chunk = _stream_chunk(core)
        chunk.prompt_token_ids = [1]
        chunk._all_token_ids[:] = [1]
        chunk.num_prompt_tokens = 1
        scheduler.add_request(chunk)

    def run(token, value):
        step = scheduler.schedule()
        state.prepare_step(step, dummy_run=False, is_profile=False)
        slots.begin_step(step, runner.main_stream)
        runner.finish_requests(step)
        slots.activate_requests()
        runner.add_requests(step)
        batch = _input(("stream",), (token,))
        snapshot = state.prepare_inputs(batch, runner.req_states)[
            "persistent_state_projection"
        ]
        snapshot.state_batch.access.validate_current()
        snapshot.state_batch.pools.predictor_h[snapshot.bindings[0].slot_id].fill_(
            value
        )
        state.postprocess_state(torch.tensor([0]), torch.tensor([1]))
        output = _state_output(step)
        output.sampled_token_ids = [[token]]
        fenced = runner._finish_state_step(output)
        scheduler.update_from_output(step, fenced.get_output())
        return snapshot

    feed()
    first = run(6, 17)
    old = first.bindings[0].handle
    run(7, 19)
    assert scheduler.requests["stream"].status.name == "WAITING_FOR_STREAMING_REQ"
    peer_handle = scheduler.kv_cache_manager.coordinator.state_managers["state"].get(
        "peer"
    )
    feed()
    second = run(7, 23)
    assert second.bindings[0].handle == old and not second.bindings[0].fresh
    assert second.request_metadata["stream"].prompt_token_ids == [1]
    terminal = _stream_chunk(core)
    terminal.resumable = False
    scheduler.add_request(terminal)
    manager = scheduler.kv_cache_manager.coordinator.state_managers["state"]
    assert manager.num_free_slots == 0
    cleanup = scheduler.schedule()
    pending = runner.execute_model(cleanup)
    assert manager.num_free_slots == 0
    scheduler.update_from_output(cleanup, pending.get_output())
    assert manager.num_free_slots == 1
    assert manager.get("peer") == peer_handle
    assert _admit_stream(core, 4, _stream_request("replacement")).accepted
    replacement = manager.get("replacement")
    assert replacement.slot == old.slot and replacement.generation > old.generation
    assert "stream" not in state._initialized
    with pytest.raises(ValueError, match="Stale"):
        second.state_batch.access.validate_current()
    assert events.count("record") == events.count("wait") == 4
