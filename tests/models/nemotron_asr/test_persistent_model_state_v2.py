# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core-backed MRv2 projection, isolation, and streaming lifecycle contracts."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.models.nemotron_asr.model_state_v2 import (
    NemotronASRModelState,
    RowBinding,
    StateBatch,
)
from vllm.models.nemotron_asr.nemotron_asr import NemotronASRForRNNT
from vllm.models.nemotron_asr.plan import SessionRegistry
from vllm.models.nemotron_asr.state_profile import build_nemotron_persistent_state_spec
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, StateSlotHandle
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.state_slots import StateSlotBindings

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]


def _config():
    return SimpleNamespace(
        n_layers=2,
        att_context_left=8,
        d_model=32,
        conv_kernel=5,
        pred_rnn_layers=2,
        pred_hidden=8,
        n_mels=8,
        num_asr_labels=5,
        endpoint_history_capacity_frames=12,
    )


def _state():
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_inputs_embeds_size=lambda: 32, dtype=torch.float32, max_model_len=142
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=2),
    )
    model = SimpleNamespace(
        config=_config(),
        _persistent_state_layer=SimpleNamespace(prefix="state"),
        _native_burst_handoff=None,
        _registry=SessionRegistry(),
        collect_commit_status=lambda: ({}, set()),
        consume_batch_stats=lambda: None,
    )
    return NemotronASRModelState(config, model, None, torch.device("cpu"))


def _slots():
    spec = build_nemotron_persistent_state_spec(_config())
    slots = StateSlotBindings(
        [KVCacheGroupSpec(["state"], spec)],
        {"state": torch.zeros(3 * spec.page_size_bytes, dtype=torch.uint8)},
    )
    return slots


def _step(slots, ids=("left", "right"), generation=1):
    step = SchedulerOutput.make_empty()
    step.state_step_seq = generation
    step.state_slot_handles = {
        req_id: {"state": StateSlotHandle("epoch", i, generation)}
        for i, req_id in enumerate(ids, 1)
    }
    step.num_scheduled_tokens = dict.fromkeys(ids, 1)
    step.scheduled_cached_reqs.req_ids = list(ids)
    step.scheduled_cached_reqs.new_token_ids = [[i] for i in range(1, len(ids) + 1)]
    slots.begin_step(step, SimpleNamespace(wait_stream=lambda _: None))
    slots.activate_requests()
    return step


def _input(ids=("left", "right"), tokens=(1, 2)):
    return SimpleNamespace(
        req_ids=list(ids),
        num_scheduled_tokens=[1] * len(ids),
        input_ids=torch.tensor(tokens),
        num_computed_tokens_np=[0] * len(ids),
    )


def _metadata(state, ids=("left", "right")):
    for req_id in ids:
        state.add_request(
            0,
            NewRequestData(
                req_id, [9], [], None, None, ([1],), 0, None, prefill_token_ids=[9]
            ),
        )


def test_model_selects_direct_state_and_core_task_policy():
    assert NemotronASRForRNNT.get_model_state_cls() is NemotronASRModelState
    assert issubclass(NemotronASRModelState, ModelState)
    assert not NemotronASRModelState.__abstractmethods__
    state = _state()
    state.model = object.__new__(NemotronASRForRNNT)
    assert state.get_supported_generation_tasks() == ("generate", "realtime")
    assert state.num_new_sampled_tokens_per_step == 1
    assert state.custom_sampler(object()) is None


def test_dummy_uses_ephemeral_encoder_buffer_and_no_resident_authority():
    state = _state()
    backing = torch.zeros(7, 32)
    state.encoder_runner = SimpleNamespace(inputs_embeds=backing)
    assert state.dummy_inputs_embeds(3).data_ptr() == backing.data_ptr()
    state.prepare_step(SchedulerOutput.make_empty(), dummy_run=True, is_profile=True)
    snapshot = state.prepare_inputs(_input(), object())["persistent_state_projection"]
    assert snapshot.no_page_io and snapshot.dummy_run and snapshot.is_profile
    assert not snapshot.bindings
    assert snapshot.state_batch is None and snapshot.request_metadata is None


def test_profile_marker_requires_explicit_dummy_and_real_ids_never_select_dummy():
    state = _state()
    with pytest.raises(ValueError, match="dummy"):
        state.prepare_step(
            SchedulerOutput.make_empty(), dummy_run=False, is_profile=True
        )
    state.prepare_step(SchedulerOutput.make_empty(), dummy_run=False, is_profile=False)
    with pytest.raises(RuntimeError, match="not initialized"):
        state.prepare_inputs(_input(("_dummy_req_0",), (1,)), object())


@pytest.mark.parametrize(
    "mode,capture,ubatch",
    [
        (CUDAGraphMode.FULL, False, 0),
        (CUDAGraphMode.NONE, True, 0),
        (CUDAGraphMode.NONE, False, 1),
    ],
)
def test_outer_capture_and_dbo_fail_closed(mode, capture, ubatch):
    state = _state()
    with pytest.raises(RuntimeError, match="DBO|capture"):
        state.prepare_attn(
            _input(),
            mode,
            (),
            torch.empty(0),
            [],
            None,
            for_capture=capture,
            ubatch_idx=ubatch,
        )
    with pytest.raises(RuntimeError, match="capture"):
        state.prepare_dummy_inputs(1, 1)


@pytest.mark.parametrize("new_request", [False, True])
def test_projection_preserves_order_and_reads_current_replay_tokens(
    monkeypatch, new_request
):
    state, slots = _state(), _slots()
    state.initialize_state_slots(slots)
    step = _step(slots)
    step.scheduled_cached_reqs.new_token_ids = [[4], [3]]
    _metadata(state)
    if new_request:
        step.scheduled_cached_reqs.req_ids = ["right"]
        step.scheduled_cached_reqs.new_token_ids = [[3]]
        step.scheduled_new_reqs = [
            replace(
                state._metadata["left"], prefill_token_ids=[9, 4], num_computed_tokens=1
            )
        ]
    state.prepare_step(step, dummy_run=False, is_profile=False)
    batch = _input(("right", "left"), (3, 4))

    def forbid_cpu(*args, **kwargs):
        raise AssertionError("preparation must not read device tokens")

    monkeypatch.setattr(torch.Tensor, "cpu", forbid_cpu)
    snapshot = state.prepare_inputs(batch, object())["persistent_state_projection"]
    assert snapshot.req_ids == ("right", "left")
    assert snapshot.token_ids == (3, 4)  # Not the initial prefill token, 9.
    assert snapshot.block_ids == ((2,), (1,))
    assert all(b.fresh for b in snapshot.bindings)
    assert snapshot.input_batch is batch
    state.postprocess_state(torch.arange(2), torch.ones(2))
    state.prepare_step(step, dummy_run=False, is_profile=False)
    assert not any(
        b.fresh
        for b in state.prepare_inputs(batch, object())[
            "persistent_state_projection"
        ].bindings
    )


def test_resident_pools_alias_physical_slots_without_batch_copies():
    slots = _slots()
    _step(slots)
    bindings = tuple(
        RowBinding(r, slots.get_handle(r, "state"), True) for r in ("right", "left")
    )
    batch = StateBatch(slots, "state", bindings, 2)
    assert batch.access.slot_indices_cpu == (2, 1)
    batch.pools.replay_book[2, 4] = 12
    batch.pools.replay_book[1, 4] = 23
    field = "decode.layers.0.replay.book.geometry"
    for b, expected in zip(bindings, (12, 23)):
        assert (
            slots.views["state"].view(b.request_id, b.handle)[field].item() == expected
        )


def test_stale_batch_cannot_write_replacement_or_survivor():
    slots = _slots()
    _step(slots)
    bindings = tuple(
        RowBinding(r, slots.get_handle(r, "state"), True) for r in ("left", "right")
    )
    batch = StateBatch(slots, "state", bindings, 2)
    _step(slots, ("replacement", "right"), generation=2)
    before = slots.views["state"]._tensor.clone()
    with pytest.raises((KeyError, ValueError)):
        batch.access.validate_current()
    assert torch.equal(slots.views["state"]._tensor, before)


def test_terminal_pruning_does_not_remove_parked_state():
    state = _state()
    _metadata(state)
    state._initialized = {
        r: StateSlotHandle("epoch", i, 1) for i, r in enumerate(("left", "right"), 1)
    }
    state.remove_request("right")
    assert "right" in state._initialized
    step = SchedulerOutput.make_empty()
    step.finished_req_ids = {"left"}
    state.prepare_step(step, dummy_run=False, is_profile=False)
    assert set(state._initialized) == {"right"}
    assert not state._metadata


def test_failed_status_cannot_publish_freshness():
    state, slots = _state(), _slots()
    state.initialize_state_slots(slots)
    step = _step(slots)
    state.prepare_step(step, dummy_run=False, is_profile=False)
    _metadata(state)
    state.prepare_inputs(_input(), object())
    state.model.collect_commit_status = lambda: ({"left": 3}, {"left"})
    state.postprocess_state(torch.arange(2), torch.ones(2))
    assert set(state._initialized) == {"right"}
    assert slots._completion.failed_req_ids == frozenset({"left"})


def test_missing_binding_and_multitoken_row_fail_before_pool_access():
    state, slots = _state(), _slots()
    state.initialize_state_slots(slots)
    step = _step(slots, ("left",))
    state.prepare_step(step, dummy_run=False, is_profile=False)
    _metadata(state)
    with pytest.raises(KeyError):
        state.prepare_inputs(_input(), object())
    batch = _input(("left",), (1,))
    batch.num_scheduled_tokens = [2]
    with pytest.raises(ValueError, match="single-token"):
        state.prepare_inputs(batch, object())


@pytest.mark.parametrize("tokens", [[], [1, 2]])
def test_projection_rejects_missing_or_multitoken_host_slice(tokens):
    state, slots = _state(), _slots()
    state.initialize_state_slots(slots)
    step = _step(slots, ("left",))
    step.scheduled_cached_reqs.new_token_ids = [tokens]
    state.prepare_step(step, dummy_run=False, is_profile=False)
    _metadata(state, ("left",))
    with pytest.raises(ValueError, match="one scheduled host token"):
        state.prepare_inputs(_input(("left",), (1,)), object())


def test_commit_sink_retains_physical_bindings():
    from vllm.models.nemotron_asr.advance import CommitPlan, PreparedRowBinding
    from vllm.models.nemotron_asr.model_state_v2 import _ResidentCommitSink

    slots = _slots()
    _step(slots, ("r",))
    access = slots.bind_rows("state", ("r",), (slots.get_handle("r", "state"),))
    binding = PreparedRowBinding("r", 1, 2, 0, 0, 0, False, True, 1)
    physical = (binding,)
    seen: list[CommitPlan] = []
    sink = _ResidentCommitSink(SimpleNamespace(reserve=seen.append), access)
    sink.reserve(CommitPlan(physical, None))
    assert seen[0].bindings == physical
    with pytest.raises(ValueError, match="order"):
        sink.reserve(CommitPlan((replace(binding, request_id="other"),), None))
    with pytest.raises(ValueError, match="slot"):
        sink.reserve(CommitPlan((replace(binding, block_id=2),), None))
    _step(slots, ("r",), generation=2)
    with pytest.raises(ValueError, match="Stale"):
        sink.reserve(CommitPlan(physical, None))
    assert len(seen) == 1


@pytest.mark.parametrize("failure", [None, "step", "generation", "echo", "maximum"])
def test_real_cpu_transaction_replays_labels_and_preserves_parked_peer(
    monkeypatch, failure
):
    """Real resident commits isolate peers, failed echoes, and stale access."""
    from tests.models.nemotron_asr.test_advance_model_rows_local import (
        CARRIER_HIDDEN,
        FEAT,
        PARK_ID,
        PLACEHOLDER_ID,
        VOCAB,
        _envelope,
        _fixed_resolver,
        _tiny_core,
    )
    from tests.models.nemotron_asr.test_plan_local import _feature
    from vllm.models.nemotron_asr.advance import make_mrv1_adapter
    from vllm.models.nemotron_asr.plan import PlanContextSlot

    config = _config()
    config.n_mels, config.pred_rnn_layers, config.pred_hidden = FEAT, 2, 16
    config.num_asr_labels = VOCAB
    config.hidden_size = CARRIER_HIDDEN
    config.audio_chunk_token_id, config.eos_token_id = PLACEHOLDER_ID, PARK_ID
    config.eou_token_id = PARK_ID + 2
    config.num_prompts = 4
    model = object.__new__(NemotronASRForRNNT)
    torch.nn.Module.__init__(model)
    model.config, model.core = config, _tiny_core(seed=1)
    model.num_logits = PARK_ID + 4
    model._registry, model._plan_slot = SessionRegistry(), PlanContextSlot()
    model._plan_step, model._max_num_seqs = 0, 2
    model._commit_sink = model._host_staging = None
    model._decode_graph_binding = model._native_burst_handoff = None
    model._persistent_state_layer = SimpleNamespace(prefix="state")
    model._encoder_execution = SimpleNamespace(transition=None)
    from vllm.models.nemotron_asr.rnnt import decode_dense_masked_frames

    model._decode_resolver = _fixed_resolver(decode_dense_masked_frames)
    model._emission_adapter = make_mrv1_adapter(
        hidden_size=CARRIER_HIDDEN, park_id=PARK_ID, blank_id=VOCAB
    )
    state = _state()
    state.model = model
    spec = build_nemotron_persistent_state_spec(config)
    slots = StateSlotBindings(
        [KVCacheGroupSpec(["state"], spec)],
        {"state": torch.zeros(3 * spec.page_size_bytes, dtype=torch.uint8)},
    )
    state.initialize_state_slots(slots)
    step = _step(slots)
    peer = slots.views["state"].view("right", step.state_slot_handles["right"]["state"])
    peer["predictor.layers.0.lstm_state.h"].fill_(19)
    before = {k: v.clone() for k, v in peer.items()}
    audio = _envelope(torch.randn(3840) * 0.1, final=True, seq=0, geometry=2)
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
    token = PLACEHOLDER_ID
    outputs = []
    payload = []
    for computed in range(142):
        if computed == 1 and failure == "maximum":
            # Seed the declared 140-label + EOU payload; real replay drains it.
            payload = [token, *(i % VOCAB for i in range(139)), config.eou_token_id]
            resident = slots.views["state"].view(
                "left", step.state_slot_handles["left"]["state"]
            )
            resident["decode.layers.0.replay.queue"].copy_(torch.tensor(payload))
            for field, value in {
                "queue_head": 1,
                "queue_length": len(payload),
                "last_label": payload[-2],
                "pending_echo": 1,
                "expected_label": token,
            }.items():
                resident[f"decode.layers.0.replay.book.{field}"].fill_(value)
        if computed == 1 and failure == "echo":
            token = (token + 1) % VOCAB
        step.scheduled_cached_reqs.req_ids = ["left"]
        step.scheduled_cached_reqs.new_token_ids = [[token]]
        state.prepare_step(step, dummy_run=False, is_profile=False)
        batch = _input(("left",), (token,))
        batch.num_computed_tokens_np = [computed]
        projection = state.prepare_inputs(
            batch, SimpleNamespace(req_id_to_index={"left": 0, "right": 1})
        )
        carrier = (
            audio.unsqueeze(0) if computed == 0 else torch.zeros(1, CARRIER_HIDDEN)
        )
        if computed == 1 and failure in ("step", "generation"):
            from vllm.models.nemotron_asr.model_state_v2 import _ResidentCommitSink

            reserve = _ResidentCommitSink.reserve
            before_commit: list[torch.Tensor] = []

            def invalidate_at_reservation(
                sink, plan, before_commit=before_commit, reserve=reserve
            ):
                if failure == "step":
                    _step(slots)
                else:
                    slots.views["state"].bind(
                        "replacement",
                        replace(step.state_slot_handles["left"]["state"], generation=2),
                    )
                before_commit.append(slots.views["state"]._tensor.clone())
                return reserve(sink, plan)

            monkeypatch.setattr(
                _ResidentCommitSink, "reserve", invalidate_at_reservation
            )
            with torch.inference_mode(), pytest.raises(ValueError, match="Stale"):
                model.forward(
                    input_ids=batch.input_ids, inputs_embeds=carrier, **projection
                )
            assert torch.equal(slots.views["state"]._tensor, before_commit[0])
            assert all(torch.equal(peer[k], value) for k, value in before.items())
            return
        before_forward = slots.views["state"]._tensor.clone()
        with torch.inference_mode():
            hidden = model.forward(
                input_ids=batch.input_ids, inputs_embeds=carrier, **projection
            )
            token = int(model.compute_logits(hidden).argmax(-1).item())
        state.postprocess_state(torch.tensor([0]), torch.tensor([1]))
        if computed == 1 and failure == "echo":
            assert slots._completion.failed_req_ids == frozenset({"left"})
            assert torch.equal(slots.views["state"]._tensor, before_forward)
            return
        outputs.append(token)
        if token == PARK_ID:
            break
    assert outputs[-1] == PARK_ID
    if failure == "maximum":
        assert outputs == payload + [PARK_ID]
        assert len(outputs) + 1 == 143
    assert len(outputs) > 2 and len(set(outputs[:-1])) > 1
    assert all(torch.equal(peer[k], value) for k, value in before.items())
    assert model.consume_committed_progress()["left"][2] == 2


def test_recoverable_geometry_status_keeps_session_live():
    state, slots = _state(), _slots()
    state.initialize_state_slots(slots)
    step = _step(slots)
    state.prepare_step(step, dummy_run=False, is_profile=False)
    _metadata(state)
    state.prepare_inputs(_input(), object())
    state.model.collect_commit_status = lambda: ({"left": 1}, set())
    state.postprocess_state(torch.arange(2), torch.ones(2))
    assert set(state._initialized) == {"left", "right"}
    assert not slots._completion.failed_req_ids


def test_failure_reporting_requires_a_live_scheduled_request():
    slots = _slots()
    with pytest.raises(ValueError, match="No live state"):
        slots.fail_requests({"left"})
    _step(slots)
    with pytest.raises(ValueError, match="outside"):
        slots.fail_requests({"unknown"})
    assert not slots._completion.failed_req_ids


@pytest.mark.nemotron_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("new_request", [False, True])
def test_cuda_state_consumer_preparation_has_no_host_sync(new_request):
    """Real GPU bindings and input IDs use the scheduler's host token slice.

    This covers preparation, not forward/commit-status or output-fence waits.
    """
    state = _state()
    state.device = torch.device("cuda")
    spec = build_nemotron_persistent_state_spec(_config())
    slots = StateSlotBindings(
        [KVCacheGroupSpec(["state"], spec)],
        {
            "state": torch.zeros(
                3 * spec.page_size_bytes, dtype=torch.uint8, device="cuda"
            )
        },
    )
    state.initialize_state_slots(slots)
    step = SchedulerOutput.make_empty()
    step.state_step_seq = 1
    step.state_slot_handles = {"left": {"state": StateSlotHandle("epoch", 1, 1)}}
    step.num_scheduled_tokens = {"left": 1}
    step.scheduled_cached_reqs.req_ids = ["left"]
    step.scheduled_cached_reqs.new_token_ids = [[4]]
    _metadata(state, ("left",))
    if new_request:
        step.scheduled_cached_reqs.req_ids = []
        step.scheduled_cached_reqs.new_token_ids = []
        step.scheduled_new_reqs = [
            replace(state._metadata["left"], prefill_token_ids=[4])
        ]
    batch = _input(("left",), (4,))
    batch.input_ids = batch.input_ids.cuda()
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        slots.begin_step(step, torch.cuda.current_stream())
        slots.activate_requests()
        state.prepare_step(step, dummy_run=False, is_profile=False)
        snapshot = state.prepare_inputs(batch, object())["persistent_state_projection"]
        snapshot.state_batch.access.validate_current()
        assert snapshot.token_ids == (4,)
        assert snapshot.state_batch.pools.predictor_h.is_cuda
    finally:
        torch.cuda.set_sync_debug_mode(previous)
