# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler behaviour for requests that hold state their tokens cannot rebuild.

A state group is a KV cache group whose manager config is not recomputable. The
scheduler fixtures below hold one state group, or one attention group and one
state group for a hybrid model.
"""

import time

import pytest
import torch

from vllm.config import ModelConfig
from vllm.multimodal.inputs import PlaceholderRange
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.engine import FinishReason
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayerConfig, MambaSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from .utils import EOS_TOKEN_ID, create_requests, create_scheduler, full_layer

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16
MAX_MODEL_LEN = 64
EOS = EOS_TOKEN_ID
TOKEN = 5
STOP_TOKEN = 7


def state_layer() -> KVCacheLayerConfig:
    spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((1,),),
        dtypes=(torch.int64,),
        mamba_type=MambaAttentionBackendEnum.PERSISTENT_STATE,
    )
    return MambaManager.layer_config(spec, recomputable=False)


def attention_layer() -> KVCacheLayerConfig:
    return full_layer(
        AttentionSpec(
            block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32
        )
    )


def make_scheduler(*, hybrid: bool = False, max_num_seqs: int = 2, **kwargs):
    if hybrid:
        kwargs["kv_cache_layer_config"] = attention_layer()
        kwargs["mamba_layer_config"] = state_layer()
    else:
        kwargs["kv_cache_layer_config"] = state_layer()
    return create_scheduler(
        max_num_seqs=max_num_seqs,
        max_model_len=MAX_MODEL_LEN,
        block_size=BLOCK_SIZE,
        **kwargs,
    )


def declare(monkeypatch: pytest.MonkeyPatch, **facts: bool) -> None:
    """Set `SupportsPersistentState` facts as `ModelConfig` exposes them."""
    for name, value in facts.items():
        monkeypatch.setattr(
            ModelConfig, f"persistent_state_{name}", value, raising=False
        )


def stream_input(
    req_id: str, num_tokens: int, *, resumable: bool = True, **kwargs
) -> Request:
    kwargs.setdefault("max_tokens", 100)
    (request,) = create_requests(
        num_requests=1,
        num_tokens=num_tokens,
        req_ids=[req_id],
        block_size=BLOCK_SIZE,
        **kwargs,
    )
    request.resumable = resumable
    return request


def admission_request(req_id: str) -> Request:
    return stream_input(req_id, 0)


def step(scheduler, sampled: dict[str, int] | None = None, **runner_kwargs):
    """Schedule one step and report one sampled token per scheduled request."""
    sampled = sampled or {}
    output = scheduler.schedule()
    req_ids = list(output.num_scheduled_tokens)
    outputs = scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            sampled_token_ids=[[sampled.get(r, TOKEN)] for r in req_ids],
            **runner_kwargs,
        ),
    )
    return output, outputs


def run_turn(scheduler, req_id: str, tokens: list[int]) -> None:
    for token in tokens:
        step(scheduler, {req_id: token})


def open_stream(
    scheduler, req_id: str, num_tokens: int, tokens: list[int], **kwargs
) -> Request:
    request = stream_input(req_id, num_tokens, **kwargs)
    scheduler.add_request(request)
    run_turn(scheduler, req_id, tokens)
    return request


def park_after_two_turns(scheduler) -> Request:
    """A stream whose first turn took 20 tokens and second turn 10 more."""
    request = open_stream(scheduler, "s", 20, [EOS])
    scheduler.add_request(stream_input("s", 10))
    run_turn(scheduler, "s", [EOS])
    assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    return request


def free_blocks(scheduler) -> int:
    return scheduler.kv_cache_manager.block_pool.get_num_free_blocks()


def num_rows(scheduler) -> int:
    return (
        len(scheduler.running)
        + scheduler.num_waiting_for_streaming_input
        + len(scheduler.resumed_streaming_sessions)
    )


# @spec PSTATE-DECL-001
@pytest.mark.parametrize("hybrid", [False, True], ids=["state", "hybrid"])
def test_group_that_inputs_cannot_rebuild_makes_scheduler_a_state_model(hybrid):
    assert make_scheduler(hybrid=hybrid).has_unrecomputable_state


# @spec PSTATE-DECL-001
def test_group_that_inputs_can_rebuild_leaves_scheduler_ordinary():
    assert not create_scheduler().has_unrecomputable_state


# @spec PSTATE-DECL-012
@pytest.mark.parametrize(
    ("num_blocks", "fits"), [(10, False), (11, True)], ids=["short", "exact"]
)
def test_startup_needs_every_groups_footprint_for_every_active_row(num_blocks, fits):
    # Per row: 4 attention blocks at max_model_len plus 1 state block, for 2 rows,
    # plus the null block.
    if fits:
        make_scheduler(hybrid=True, num_blocks=num_blocks)
    else:
        with pytest.raises(ValueError):
            make_scheduler(hybrid=True, num_blocks=num_blocks)


# @spec PSTATE-PRES-001
def test_allocation_failure_does_not_preempt_state_request(monkeypatch):
    scheduler = make_scheduler()
    request = open_stream(scheduler, "s", 4, [TOKEN])
    monkeypatch.setattr(
        scheduler.kv_cache_manager, "allocate_slots", lambda *args, **kwargs: None
    )

    output = scheduler.schedule()

    assert request.status == RequestStatus.RUNNING
    assert not output.preempted_req_ids


# @spec PSTATE-PRES-002
def test_preempting_state_request_raises_naming_it():
    scheduler = make_scheduler()
    request = open_stream(scheduler, "stream-1", 4, [TOKEN])

    with pytest.raises(RuntimeError, match="stream-1"):
        scheduler._preempt_request(request, time.monotonic())


# @spec PSTATE-PRES-003
@pytest.mark.parametrize("tokens", [[TOKEN], [EOS]], ids=["running", "parked"])
def test_reset_running_requests_refused_while_state_stream_held(tokens):
    scheduler = make_scheduler()
    request = open_stream(scheduler, "stream-1", 4, tokens)

    with pytest.raises(RuntimeError, match="stream-1"):
        scheduler.reset_prefix_cache(reset_running_requests=True)

    assert request.status in (
        RequestStatus.RUNNING,
        RequestStatus.WAITING_FOR_STREAMING_REQ,
    )


# @spec PSTATE-PRES-003 PSTATE-DECL-013
def test_reset_running_requests_still_preempts_ordinary_stream():
    scheduler = create_scheduler(max_num_seqs=1)
    request = open_stream(scheduler, "s", 4, [TOKEN])

    assert scheduler.reset_prefix_cache(reset_running_requests=True)

    assert request.status == RequestStatus.PREEMPTED


# @spec PSTATE-PAGE-004
@pytest.mark.parametrize(
    ("request_kwargs", "tokens", "ending"),
    [
        ({}, [TOKEN], "abort"),
        ({}, [EOS], "abort"),
        ({}, [EOS], "close"),
        ({"stop_token_ids": [STOP_TOKEN]}, [TOKEN, STOP_TOKEN], None),
        ({"max_tokens": 2}, [TOKEN, TOKEN], None),
    ],
    ids=["abort_running", "abort_parked", "close_parked", "stop_token", "length"],
)
def test_terminal_finish_returns_state_page_to_pool(request_kwargs, tokens, ending):
    scheduler = make_scheduler()
    free_before = free_blocks(scheduler)
    open_stream(scheduler, "s", 4, tokens, **request_kwargs)
    assert free_blocks(scheduler) == free_before - (ending is not None)

    if ending == "abort":
        scheduler.finish_requests("s", RequestStatus.FINISHED_ABORTED)
    elif ending == "close":
        scheduler.add_request(stream_input("s", 4, resumable=False))

    assert "s" not in scheduler.requests
    assert free_blocks(scheduler) == free_before


# @spec PSTATE-PAGE-007
def test_first_step_names_request_whose_state_page_was_initialized():
    scheduler = make_scheduler()
    scheduler.add_request(stream_input("s", 4))

    first, _ = step(scheduler)
    second, _ = step(scheduler)

    assert first.new_state_page_req_ids == {"s"}
    assert second.new_state_page_req_ids == set()


# @spec PSTATE-PAGE-007
def test_resumed_turn_does_not_name_request_again():
    scheduler = make_scheduler()
    open_stream(scheduler, "s", 4, [EOS])
    scheduler.add_request(stream_input("s", 4))

    resumed, _ = step(scheduler)

    assert set(resumed.num_scheduled_tokens) == {"s"}
    assert resumed.new_state_page_req_ids == set()


# @spec PSTATE-PAGE-007
def test_reused_page_is_named_for_its_new_owner():
    scheduler = make_scheduler(max_num_seqs=1, num_blocks=2)
    open_stream(scheduler, "old", 4, [TOKEN])
    old_blocks = scheduler.kv_cache_manager.get_block_ids("old")
    scheduler.finish_requests("old", RequestStatus.FINISHED_ABORTED)
    scheduler.add_request(stream_input("new", 4))

    output, _ = step(scheduler)

    assert scheduler.kv_cache_manager.get_block_ids("new") == old_blocks
    assert output.new_state_page_req_ids == {"new"}


# @spec PSTATE-PAGE-007 PSTATE-DECL-013
def test_ordinary_model_names_no_initialized_state_pages():
    scheduler = create_scheduler()
    scheduler.add_request(stream_input("s", 4))

    output, _ = step(scheduler)

    assert not output.new_state_page_req_ids


# @spec PSTATE-ASYNC-001
def test_barrier_holds_state_request_while_its_step_is_in_flight():
    scheduler = make_scheduler(async_scheduling=True)
    scheduler.add_request(stream_input("s", 4))
    first = scheduler.schedule()
    assert scheduler.requests["s"].num_in_flight_tokens > 0

    second = scheduler.schedule()
    assert "s" not in second.num_scheduled_tokens

    scheduler.update_from_output(
        first, ModelRunnerOutput(req_ids=["s"], sampled_token_ids=[[TOKEN]])
    )
    assert "s" in scheduler.schedule().num_scheduled_tokens


# @spec PSTATE-ASYNC-001
def test_declared_echo_tolerance_lifts_the_barrier(monkeypatch):
    declare(monkeypatch, ignores_stopped_row_echoes=True)
    scheduler = make_scheduler(async_scheduling=True)
    scheduler.add_request(stream_input("s", 4))
    scheduler.schedule()

    assert "s" in scheduler.schedule().num_scheduled_tokens


# @spec PSTATE-ASYNC-002
def test_barrier_leaves_other_requests_schedulable():
    scheduler = make_scheduler(async_scheduling=True)
    scheduler.add_request(stream_input("held", 4))
    scheduler.schedule()
    scheduler.add_request(stream_input("other", 4))

    output = scheduler.schedule()

    assert set(output.num_scheduled_tokens) == {"other"}


# @spec PSTATE-ASYNC-002 PSTATE-DECL-013
def test_ordinary_model_keeps_async_overlap():
    scheduler = create_scheduler(async_scheduling=True)
    scheduler.add_request(stream_input("s", 4))
    scheduler.schedule()

    assert "s" in scheduler.schedule().num_scheduled_tokens


# @spec PSTATE-TURN-001
def test_eos_parks_state_stream_with_empty_queue():
    scheduler = make_scheduler()
    request = open_stream(scheduler, "s", 4, [EOS])

    assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert scheduler.num_waiting_for_streaming_input == 1
    assert any(scheduler.kv_cache_manager.get_block_ids("s"))


# @spec PSTATE-TURN-001
def test_eos_folds_queued_input_into_state_stream():
    scheduler = make_scheduler()
    request = open_stream(scheduler, "s", 4, [TOKEN])
    scheduler.add_request(stream_input("s", 4))

    run_turn(scheduler, "s", [EOS])

    assert request.status == RequestStatus.WAITING
    assert request in scheduler.resumed_streaming_sessions
    assert scheduler.num_waiting_for_streaming_input == 0


# @spec PSTATE-TURN-001
def test_eos_before_close_finishes_state_stream_with_stop():
    scheduler = make_scheduler()
    request = open_stream(scheduler, "s", 4, [TOKEN])
    scheduler.add_request(stream_input("s", 4, resumable=False))

    run_turn(scheduler, "s", [EOS])

    assert request.status == RequestStatus.FINISHED_STOPPED
    assert "s" not in scheduler.requests


# @spec PSTATE-TURN-001
def test_eos_finishes_stream_of_model_that_does_not_park_on_eos(monkeypatch):
    declare(monkeypatch, parks_on_eos=False)
    scheduler = make_scheduler()
    request = open_stream(scheduler, "s", 4, [EOS])

    assert request.status == RequestStatus.FINISHED_STOPPED
    assert "s" not in scheduler.requests


# @spec PSTATE-TURN-015
def test_inputs_accepted_before_close_run_in_order():
    scheduler = make_scheduler()
    request = open_stream(scheduler, "s", 4, [TOKEN])
    scheduler.add_request(stream_input("s", 5))
    scheduler.add_request(stream_input("s", 6))
    scheduler.add_request(stream_input("s", 4, resumable=False))

    run_turn(scheduler, "s", [EOS])
    assert len(request.prompt_token_ids) == 1 + 5
    run_turn(scheduler, "s", [EOS])
    assert len(request.prompt_token_ids) == 1 + 6
    run_turn(scheduler, "s", [EOS])

    assert request.status == RequestStatus.FINISHED_STOPPED
    assert "s" not in scheduler.requests


# @spec PSTATE-TURN-015
def test_input_after_close_is_not_accepted():
    scheduler = make_scheduler()
    request = open_stream(scheduler, "s", 4, [TOKEN])
    scheduler.add_request(stream_input("s", 4, resumable=False))

    scheduler.add_request(stream_input("s", 4))

    assert len(request.streaming_queue) == 1


# @spec PSTATE-TURN-002 PSTATE-HIST-011
@pytest.mark.parametrize(
    ("num_tokens", "request_kwargs", "tokens", "status"),
    [
        (4, {"max_tokens": 2}, [TOKEN, TOKEN], RequestStatus.FINISHED_LENGTH_CAPPED),
        (
            MAX_MODEL_LEN - 2,
            {},
            [TOKEN, TOKEN],
            RequestStatus.FINISHED_LENGTH_CAPPED,
        ),
        (
            4,
            {"stop_token_ids": [STOP_TOKEN]},
            [TOKEN, STOP_TOKEN],
            RequestStatus.FINISHED_STOPPED,
        ),
    ],
    ids=["max_tokens", "max_model_len", "stop_token"],
)
def test_non_resumable_stop_finishes_state_stream_despite_queued_input(
    num_tokens, request_kwargs, tokens, status
):
    scheduler = make_scheduler()
    request = open_stream(scheduler, "s", num_tokens, tokens[:-1], **request_kwargs)
    scheduler.add_request(stream_input("s", 4))

    run_turn(scheduler, "s", tokens[-1:])

    assert request.status == status
    assert "s" not in scheduler.requests


# @spec PSTATE-TURN-002
@pytest.mark.parametrize("tokens", [[TOKEN], [EOS]], ids=["running", "parked"])
def test_abort_finishes_state_stream(tokens):
    scheduler = make_scheduler()
    request = open_stream(scheduler, "s", 4, tokens)

    scheduler.finish_requests("s", RequestStatus.FINISHED_ABORTED)

    assert request.status == RequestStatus.FINISHED_ABORTED
    assert "s" not in scheduler.requests


# @spec PSTATE-TURN-008 PSTATE-TURN-009 PSTATE-HIST-003
def test_update_past_position_limit_finishes_state_stream_length_capped():
    scheduler = make_scheduler()
    request = park_after_two_turns(scheduler)
    free_before = free_blocks(scheduler) + 1

    scheduler.add_request(stream_input("s", 40))

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert "s" not in scheduler.requests
    assert request not in scheduler.dropped_streaming_updates
    assert free_blocks(scheduler) == free_before


# @spec PSTATE-TURN-008 PSTATE-HIST-003
def test_refused_update_leaves_request_unchanged():
    scheduler = make_scheduler()
    request = park_after_two_turns(scheduler)
    prompt = list(request.prompt_token_ids)
    offset = request.position_offset

    scheduler.add_request(stream_input("s", 40))

    assert list(request.prompt_token_ids) == prompt
    assert request.position_offset == offset
    assert request.num_computed_tokens == len(prompt)


# @spec PSTATE-TURN-009 PSTATE-HIST-003
def test_position_limit_uses_compacted_length_for_model_without_positions(
    monkeypatch,
):
    declare(monkeypatch, reads_positions=False)
    scheduler = make_scheduler()
    request = park_after_two_turns(scheduler)

    scheduler.add_request(stream_input("s", 40))

    assert request.status == RequestStatus.WAITING
    assert len(request.prompt_token_ids) == 1 + 40


# @spec PSTATE-TURN-009
def test_position_limit_counts_full_length_for_hybrid_request():
    scheduler = make_scheduler(hybrid=True)
    request = park_after_two_turns(scheduler)

    scheduler.add_request(stream_input("s", 40))

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert "s" not in scheduler.requests


# @spec PSTATE-TURN-010
@pytest.mark.parametrize(
    ("reads_positions", "final_num_tokens"),
    [(True, MAX_MODEL_LEN - 19), (False, MAX_MODEL_LEN)],
    ids=["reads_positions", "ignores_positions"],
)
def test_decode_stops_at_position_limit(monkeypatch, reads_positions, final_num_tokens):
    declare(monkeypatch, reads_positions=reads_positions)
    scheduler = make_scheduler()
    open_stream(scheduler, "s", 20, [EOS])
    request = scheduler.requests["s"]
    scheduler.add_request(stream_input("s", 10))
    assert request.position_offset == 19

    for _ in range(MAX_MODEL_LEN):
        if request.is_finished():
            break
        step(scheduler)

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert request.num_tokens == final_num_tokens


# @spec PSTATE-TURN-011
def test_eos_at_position_limit_ends_state_stream():
    scheduler = make_scheduler()
    request = open_stream(scheduler, "s", MAX_MODEL_LEN - 1, [EOS])

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert "s" not in scheduler.requests


# @spec PSTATE-TURN-011
def test_eos_at_turn_budget_parks_state_stream():
    scheduler = make_scheduler()
    request = open_stream(scheduler, "s", 4, [EOS], max_tokens=1)

    assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ


# @spec PSTATE-TURN-011
def test_eos_at_position_limit_keeps_upstream_order_for_single_request():
    scheduler = make_scheduler()
    request = open_stream(scheduler, "s", MAX_MODEL_LEN - 1, [EOS], resumable=False)

    assert request.status == RequestStatus.FINISHED_STOPPED


# @spec PSTATE-TURN-014
def test_continuation_needing_reprefill_of_hybrid_request_finishes_it():
    scheduler = make_scheduler(hybrid=True)
    request = open_stream(scheduler, "s", 4, [EOS])
    # A parked hybrid request cannot lose its computed tokens through any
    # scheduler path, so the state a re-prefill would start from is set directly.
    request.num_computed_tokens = 0

    scheduler.add_request(stream_input("s", 4))

    assert request.is_finished()
    assert "s" not in scheduler.requests
    assert not scheduler.schedule().scheduled_new_reqs


# @spec PSTATE-HIST-001
@pytest.mark.parametrize("queued", [False, True], ids=["parked", "queued"])
def test_fold_compacts_history_to_frontier_and_new_input(queued):
    scheduler = make_scheduler()
    request = stream_input("s", 20)
    scheduler.add_request(request)
    step(scheduler, {"s": TOKEN})
    assert len(request.block_hashes) == 1

    if queued:
        scheduler.add_request(stream_input("s", 10))
    step(scheduler, {"s": EOS})
    if not queued:
        scheduler.add_request(stream_input("s", 10))

    assert list(request.prompt_token_ids) == [TOKEN] + [0] * 10
    assert list(request.all_token_ids) == [TOKEN] + [0] * 10
    assert request.num_prompt_tokens == 11
    assert request.num_computed_tokens == 1
    assert request.block_hashes == []
    assert request.position_offset == 20


# @spec PSTATE-HIST-001
def test_second_fold_adds_dropped_tokens_to_position_offset():
    scheduler = make_scheduler()
    request = park_after_two_turns(scheduler)

    assert request.position_offset == 19

    scheduler.add_request(stream_input("s", 5))

    assert request.position_offset == 19 + 10
    assert request.num_computed_tokens == 1
    assert len(request.prompt_token_ids) == 1 + 5


# @spec PSTATE-HIST-002
def test_fold_keeps_only_updates_features_relative_to_compacted_prompt():
    scheduler = make_scheduler(model="llava-hf/llava-1.5-7b-hf")
    (request,) = create_requests(
        num_requests=1,
        num_tokens=20,
        req_ids=["s"],
        max_tokens=100,
        block_size=BLOCK_SIZE,
        mm_hashes_list=[["opening-image"]],
        mm_positions=[[PlaceholderRange(offset=2, length=5)]],
    )
    request.resumable = True
    scheduler.add_request(request)
    run_turn(scheduler, "s", [EOS])
    assert [f.identifier for f in request.mm_features] == ["opening-image"]

    (update,) = create_requests(
        num_requests=1,
        num_tokens=10,
        req_ids=["s"],
        max_tokens=100,
        block_size=BLOCK_SIZE,
        mm_hashes_list=[["update-image"]],
        mm_positions=[[PlaceholderRange(offset=2, length=3)]],
    )
    update.resumable = True
    scheduler.add_request(update)

    assert [f.identifier for f in request.mm_features] == ["update-image"]
    assert request.mm_features[0].mm_position.offset == 1 + 2


# @spec PSTATE-HIST-006
def test_pending_stream_inputs_default_to_eight():
    scheduler = make_scheduler()

    assert scheduler.scheduler_config.max_pending_stream_inputs == 8


# @spec PSTATE-HIST-006
def test_stream_queue_holds_the_bound_of_pending_inputs():
    scheduler = make_scheduler()
    request = open_stream(scheduler, "s", 4, [TOKEN])
    bound = scheduler.scheduler_config.max_pending_stream_inputs

    for _ in range(bound):
        scheduler.add_request(stream_input("s", 4))

    assert len(request.streaming_queue) == bound
    assert not request.is_finished()


# @spec PSTATE-HIST-009
def test_hybrid_request_keeps_full_history_on_fold():
    scheduler = make_scheduler(hybrid=True)
    request = open_stream(scheduler, "s", 20, [TOKEN, EOS])

    scheduler.add_request(stream_input("s", 10))

    assert list(request.prompt_token_ids) == [0] * 20 + [TOKEN] + [0] * 10
    assert request.num_computed_tokens == 21
    assert request.position_offset == 0
    assert len(request.block_hashes) == 1


# @spec PSTATE-ADMIT-002
def test_admitted_stream_is_a_parked_request_with_an_empty_prompt():
    scheduler = make_scheduler()
    request = admission_request("a")

    assert scheduler.admit_stream(request) is None

    assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert scheduler.requests["a"] is request
    assert request in scheduler.waiting
    assert scheduler.num_waiting_for_streaming_input == 1
    assert len(request.prompt_token_ids) == 0


# @spec PSTATE-ADMIT-010
def test_first_input_becomes_prompt_of_admitted_request():
    scheduler = make_scheduler()
    request = admission_request("a")
    scheduler.admit_stream(request)

    scheduler.add_request(stream_input("a", 6))

    assert list(request.prompt_token_ids) == [0] * 6
    assert request.num_prompt_tokens == 6
    assert request.num_computed_tokens == 0
    assert request.position_offset == 0
    assert request.status == RequestStatus.WAITING


# @spec PSTATE-ADMIT-003
def test_first_input_moves_reservation_without_a_gap():
    scheduler = make_scheduler()
    request = admission_request("a")
    scheduler.admit_stream(request)
    assert num_rows(scheduler) == 1

    scheduler.add_request(stream_input("a", 6))

    assert scheduler.num_waiting_for_streaming_input == 0
    assert request in scheduler.resumed_streaming_sessions
    assert num_rows(scheduler) == 1

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {"a": 6}
    assert not scheduler.resumed_streaming_sessions
    assert num_rows(scheduler) == 1


# @spec PSTATE-PAGE-007 PSTATE-ADMIT-010
def test_admitted_stream_page_is_initialized_when_first_input_is_scheduled():
    scheduler = make_scheduler()
    scheduler.admit_stream(admission_request("a"))
    idle, _ = step(scheduler)
    scheduler.add_request(stream_input("a", 6))

    first, _ = step(scheduler)

    assert not idle.num_scheduled_tokens
    assert first.new_state_page_req_ids == {"a"}


def _hold_row(scheduler, kind: str, req_id: str) -> None:
    if kind == "admitted":
        scheduler.admit_stream(admission_request(req_id))
    elif kind == "resumed":
        scheduler.admit_stream(admission_request(req_id))
        scheduler.add_request(stream_input(req_id, 4))
    else:
        open_stream(scheduler, req_id, 4, [EOS if kind == "parked" else TOKEN])


# @spec PSTATE-ADMIT-004
@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("running", "admitted"),
        ("running", "resumed"),
        ("parked", "admitted"),
        ("parked", "resumed"),
        ("admitted", "resumed"),
    ],
)
def test_admission_counts_running_parked_and_resumed_rows(first, second):
    scheduler = make_scheduler(max_num_seqs=2)
    _hold_row(scheduler, first, "a")
    _hold_row(scheduler, second, "b")

    assert scheduler.admit_stream(admission_request("c")) == "active_limit"
    assert "c" not in scheduler.requests

    scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)

    assert scheduler.admit_stream(admission_request("c")) is None


# @spec PSTATE-FAIL-002 PSTATE-ADMIT-014
def test_reported_failure_finishes_only_its_request_before_its_tokens_and_stop():
    scheduler = make_scheduler()
    failed = stream_input("failed", 4)
    healthy = stream_input("healthy", 4)
    scheduler.add_request(failed)
    scheduler.add_request(healthy)

    _, outputs = step(
        scheduler,
        {"failed": EOS, "healthy": TOKEN},
        failed_req_ids={"failed"},
    )

    assert failed.status == RequestStatus.FINISHED_ERROR
    assert list(failed.output_token_ids) == []
    assert "failed" not in scheduler.requests
    assert healthy.status == RequestStatus.RUNNING
    assert list(healthy.output_token_ids) == [TOKEN]
    emitted = {o.request_id: o for o in outputs[0].outputs}
    assert emitted["failed"].finish_reason == FinishReason.ERROR
    assert emitted["failed"].new_token_ids == []
    assert emitted["healthy"].new_token_ids == [TOKEN]


# @spec PSTATE-FAIL-002
def test_reported_failure_finishes_parked_stream():
    scheduler = make_scheduler()
    parked = open_stream(scheduler, "parked", 4, [EOS])
    scheduler.add_request(stream_input("running", 4))

    step(scheduler, failed_req_ids={"parked"})

    assert parked.status == RequestStatus.FINISHED_ERROR
    assert "parked" not in scheduler.requests
    assert scheduler.requests["running"].status == RequestStatus.RUNNING


# @spec PSTATE-FAIL-003
def test_failure_for_request_not_held_is_ignored():
    scheduler = make_scheduler()
    gone = open_stream(scheduler, "gone", 4, [TOKEN])
    scheduler.finish_requests("gone", RequestStatus.FINISHED_ABORTED)
    scheduler.add_request(stream_input("held", 4))

    step(scheduler, failed_req_ids={"gone", "never-existed"})

    assert gone.status == RequestStatus.FINISHED_ABORTED
    assert list(scheduler.requests["held"].output_token_ids) == [TOKEN]


# @spec PSTATE-ADMIT-014
def test_abort_ahead_of_failure_report_fixes_aborted_outcome():
    scheduler = make_scheduler()
    request = stream_input("s", 4)
    scheduler.add_request(request)
    output = scheduler.schedule()
    scheduler.finish_requests("s", RequestStatus.FINISHED_ABORTED)

    outputs = scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=["s"], sampled_token_ids=[[TOKEN]], failed_req_ids={"s"}
        ),
    )

    assert request.status == RequestStatus.FINISHED_ABORTED
    assert not outputs.get(0) or not outputs[0].outputs


# @spec PSTATE-ADMIT-014
def test_abort_after_failure_keeps_error_outcome():
    scheduler = make_scheduler()
    request = stream_input("s", 4)
    scheduler.add_request(request)
    step(scheduler, failed_req_ids={"s"})

    scheduler.finish_requests("s", RequestStatus.FINISHED_ABORTED)

    assert request.status == RequestStatus.FINISHED_ERROR


# @spec PSTATE-DECL-013
def test_ordinary_stream_parks_on_length_stop():
    scheduler = create_scheduler()
    request = open_stream(scheduler, "s", 4, [TOKEN, TOKEN], max_tokens=2)

    assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert scheduler.num_waiting_for_streaming_input == 1


# @spec PSTATE-DECL-013
def test_ordinary_stream_folds_full_history():
    scheduler = create_scheduler()
    request = open_stream(scheduler, "s", 20, [TOKEN, EOS])

    scheduler.add_request(stream_input("s", 10))

    assert list(request.prompt_token_ids) == [0] * 20 + [TOKEN] + [0] * 10
    assert request.num_computed_tokens == 21
    assert len(request.block_hashes) == 1


# @spec PSTATE-DECL-013
def test_ordinary_stream_parks_on_eos_at_max_model_len():
    scheduler = create_scheduler(max_model_len=MAX_MODEL_LEN, block_size=BLOCK_SIZE)
    request = open_stream(scheduler, "s", MAX_MODEL_LEN - 1, [EOS])

    assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
