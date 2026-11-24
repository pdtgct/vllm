# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler behaviour for requests that hold state their tokens cannot rebuild.

A state group is a KV cache group whose manager config is not recomputable. The
scheduler fixtures below hold one state group, or one attention group and one
state group for a hybrid model.
"""

import pytest
import torch

from vllm.config import ModelConfig
from vllm.multimodal.inputs import PlaceholderRange
from vllm.sampling_params import (
    RepetitionDetectionParams,
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.engine import FinishReason
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayerConfig, MambaSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output.request import StructuredOutputRequest

from .utils import EOS_TOKEN_ID, create_requests, create_scheduler, full_layer

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16
MAX_MODEL_LEN = 64
EOS = EOS_TOKEN_ID
TOKEN = 5
STOP_TOKEN = 7
REPEATS = RepetitionDetectionParams(max_pattern_size=1, min_count=2)
"""Ends a turn once one token is sampled twice in a row."""


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


STATE_FACTS = {
    "has_persistent_state": True,
    "persistent_state_reads_positions": True,
    "persistent_state_accepts_synthetic_warmup": True,
    "persistent_state_resumable_stops": frozenset({"eos"}),
}
"""The `SupportsPersistentState` defaults, as `ModelConfig` exposes them."""


def declare_state_facts(monkeypatch: pytest.MonkeyPatch, **facts: object) -> None:
    """Make `ModelConfig` describe a state model, with `facts` overriding the
    protocol defaults."""
    for name, value in {**STATE_FACTS, **facts}.items():
        monkeypatch.setattr(ModelConfig, name, value, raising=False)


def make_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    *,
    hybrid: bool = False,
    max_num_seqs: int = 2,
    facts: dict[str, object] | None = None,
    **kwargs,
):
    declare_state_facts(monkeypatch, **(facts or {}))
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


def stream_input(
    req_id: str,
    num_tokens: int,
    *,
    resumable: bool = True,
    continuation: bool = False,
    max_tokens: int = 100,
    stop_token_ids: list[int] | None = None,
    repetition_detection: RepetitionDetectionParams | None = None,
    **kwargs,
) -> Request:
    params = SamplingParams(
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids,
        repetition_detection=repetition_detection,
        output_kind=RequestOutputKind.DELTA,
    )
    (request,) = create_requests(
        num_requests=1,
        num_tokens=num_tokens,
        req_ids=[req_id],
        block_size=BLOCK_SIZE,
        sampling_params=params,
        **kwargs,
    )
    request.resumable = resumable
    request.continuation = continuation
    return request


def update(req_id: str, num_tokens: int, **kwargs) -> Request:
    """An input that continues the stream `req_id`."""
    return stream_input(req_id, num_tokens, continuation=True, **kwargs)


def close(req_id: str) -> Request:
    """The close signal of the stream `req_id`."""
    return stream_input(req_id, 4, resumable=False, continuation=True)


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
    scheduler.add_request(update("s", 10))
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


def finished_output(outputs, req_id: str):
    """The finished output that one step's `outputs` carry for `req_id`."""
    [output] = [
        o
        for eco in outputs.values()
        for o in eco.outputs
        if o.request_id == req_id and o.finish_reason is not None
    ]
    return output


def new_request_data(output, req_id: str):
    [data] = [r for r in output.scheduled_new_reqs if r.req_id == req_id]
    return data


# Preservation


# @spec PSTATE-PRES-001
def test_allocation_failure_does_not_preempt_state_request(monkeypatch):
    scheduler = make_scheduler(monkeypatch)
    request = open_stream(scheduler, "s", 4, [TOKEN])
    monkeypatch.setattr(
        scheduler.kv_cache_manager, "allocate_slots", lambda *args, **kwargs: None
    )

    output = scheduler.schedule()

    assert request.status == RequestStatus.RUNNING
    assert not output.preempted_req_ids


# @spec PSTATE-PRES-004
@pytest.mark.parametrize(
    ("tokens", "resumable"),
    [([TOKEN], True), ([EOS], True), ([TOKEN], False)],
    ids=["running_stream", "parked_stream", "single_request"],
)
def test_reset_running_requests_refused_while_state_request_held(
    monkeypatch, tokens, resumable
):
    scheduler = make_scheduler(monkeypatch)
    request = open_stream(scheduler, "held-1", 4, tokens, resumable=resumable)

    with pytest.raises(RuntimeError, match="held-1"):
        scheduler.reset_prefix_cache(reset_running_requests=True)

    assert scheduler.requests["held-1"] is request
    assert request.status in (
        RequestStatus.RUNNING,
        RequestStatus.WAITING_FOR_STREAMING_REQ,
    )


# @spec PSTATE-PRES-004 PSTATE-DECL-013
def test_reset_running_requests_still_preempts_ordinary_stream():
    scheduler = create_scheduler(max_num_seqs=1)
    request = open_stream(scheduler, "s", 4, [TOKEN])

    assert scheduler.reset_prefix_cache(reset_running_requests=True)

    assert request.status == RequestStatus.PREEMPTED


# Page initialization and reuse


# @spec PSTATE-PAGE-004 PSTATE-TURN-002
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
def test_terminal_finish_returns_state_page_and_row(
    monkeypatch, request_kwargs, tokens, ending
):
    scheduler = make_scheduler(monkeypatch)
    free_before = free_blocks(scheduler)
    open_stream(scheduler, "s", 4, tokens, **request_kwargs)
    assert free_blocks(scheduler) == free_before - (ending is not None)

    if ending == "abort":
        scheduler.finish_requests("s", RequestStatus.FINISHED_ABORTED)
    elif ending == "close":
        scheduler.add_request(close("s"))

    assert "s" not in scheduler.requests
    assert free_blocks(scheduler) == free_before
    assert num_rows(scheduler) == 0


# @spec PSTATE-PAGE-004 PSTATE-ADMIT-002
def test_aborted_admission_releases_its_row(monkeypatch):
    scheduler = make_scheduler(monkeypatch)
    free_before = free_blocks(scheduler)
    scheduler.admit_stream(admission_request("a"))

    scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)

    assert "a" not in scheduler.requests
    assert num_rows(scheduler) == 0
    assert free_blocks(scheduler) == free_before


# @spec PSTATE-PAGE-007
@pytest.mark.parametrize("admitted", [False, True], ids=["opened", "admitted"])
def test_first_schedule_of_a_stream_carries_no_computed_tokens(monkeypatch, admitted):
    scheduler = make_scheduler(monkeypatch)
    if admitted:
        scheduler.admit_stream(admission_request("s"))
        idle, _ = step(scheduler)
        assert not idle.num_scheduled_tokens
        scheduler.add_request(update("s", 6))
    else:
        scheduler.add_request(stream_input("s", 6))

    first, _ = step(scheduler)

    assert new_request_data(first, "s").num_computed_tokens == 0


# @spec PSTATE-PAGE-007 PSTATE-HIST-001
@pytest.mark.parametrize(
    ("hybrid", "num_computed_tokens"), [(False, 1), (True, 5)], ids=["state", "hybrid"]
)
def test_resumed_turn_carries_its_computed_frontier(
    monkeypatch, hybrid, num_computed_tokens
):
    scheduler = make_scheduler(monkeypatch, hybrid=hybrid)
    open_stream(scheduler, "s", 4, [TOKEN, EOS])
    scheduler.add_request(update("s", 4))

    resumed, _ = step(scheduler)

    assert new_request_data(resumed, "s").num_computed_tokens == num_computed_tokens


# @spec PSTATE-PAGE-007
def test_reused_page_is_fresh_for_its_new_owner(monkeypatch):
    scheduler = make_scheduler(monkeypatch, max_num_seqs=1, num_blocks=2)
    open_stream(scheduler, "old", 4, [TOKEN])
    old_blocks = scheduler.kv_cache_manager.get_block_ids("old")
    scheduler.finish_requests("old", RequestStatus.FINISHED_ABORTED)
    scheduler.add_request(stream_input("new", 4))

    output, _ = step(scheduler)

    assert scheduler.kv_cache_manager.get_block_ids("new") == old_blocks
    assert new_request_data(output, "new").num_computed_tokens == 0


# Per-request barrier


# @spec PSTATE-ASYNC-001
def test_barrier_holds_state_request_while_its_step_is_in_flight(monkeypatch):
    scheduler = make_scheduler(monkeypatch, async_scheduling=True)
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
def test_waiting_loop_holds_state_request_with_a_step_in_flight(monkeypatch):
    """The running-loop barrier keeps a parked or resumed state request from
    having a step in flight; the waiting loop checks the same predicate in case
    another path ever lets one through, so the count is set directly."""
    scheduler = make_scheduler(monkeypatch, async_scheduling=True)
    request = open_stream(scheduler, "s", 4, [EOS])
    scheduler.add_request(update("s", 4))
    # A resumed request still holds its pages, so it waits in the queue the
    # waiting loop drains first.
    assert request in scheduler.kv_holding_waiting
    request.num_in_flight_tokens = 1

    assert "s" not in scheduler.schedule().num_scheduled_tokens

    request.num_in_flight_tokens = 0
    assert "s" in scheduler.schedule().num_scheduled_tokens


# @spec PSTATE-ASYNC-002
def test_barrier_leaves_other_requests_schedulable(monkeypatch):
    scheduler = make_scheduler(monkeypatch, async_scheduling=True)
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


# Turn boundary and position limit


# @spec PSTATE-TURN-001
def test_eos_parks_state_stream_with_empty_queue(monkeypatch):
    scheduler = make_scheduler(monkeypatch)
    request = open_stream(scheduler, "s", 4, [EOS])

    assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert scheduler.num_waiting_for_streaming_input == 1
    assert any(scheduler.kv_cache_manager.get_block_ids("s"))


# @spec PSTATE-TURN-001
def test_eos_folds_queued_input_into_state_stream(monkeypatch):
    scheduler = make_scheduler(monkeypatch)
    request = open_stream(scheduler, "s", 4, [TOKEN])
    scheduler.add_request(update("s", 4))

    run_turn(scheduler, "s", [EOS])

    assert request.status == RequestStatus.WAITING
    assert request in scheduler.resumed_streaming_sessions
    assert scheduler.num_waiting_for_streaming_input == 0


# @spec PSTATE-TURN-001
def test_eos_before_close_finishes_state_stream_with_stop(monkeypatch):
    scheduler = make_scheduler(monkeypatch)
    request = open_stream(scheduler, "s", 4, [TOKEN])
    scheduler.add_request(close("s"))

    run_turn(scheduler, "s", [EOS])

    assert request.status == RequestStatus.FINISHED_STOPPED
    assert "s" not in scheduler.requests


# @spec PSTATE-TURN-015
def test_inputs_accepted_before_close_run_in_order(monkeypatch):
    scheduler = make_scheduler(monkeypatch)
    request = open_stream(scheduler, "s", 4, [TOKEN])
    scheduler.add_request(update("s", 5))
    scheduler.add_request(update("s", 6))
    scheduler.add_request(close("s"))

    run_turn(scheduler, "s", [EOS])
    assert len(request.prompt_token_ids) == 1 + 5
    run_turn(scheduler, "s", [EOS])
    assert len(request.prompt_token_ids) == 1 + 6
    run_turn(scheduler, "s", [EOS])

    assert request.status == RequestStatus.FINISHED_STOPPED
    assert "s" not in scheduler.requests


# @spec PSTATE-TURN-015 PSTATE-TURN-002
@pytest.mark.parametrize("admitted", [False, True], ids=["parked", "never_fed"])
def test_close_reaching_parked_stream_finishes_it_stopped(monkeypatch, admitted):
    scheduler = make_scheduler(monkeypatch)
    free_before = free_blocks(scheduler)
    if admitted:
        request = admission_request("s")
        scheduler.admit_stream(request)
    else:
        request = open_stream(scheduler, "s", 4, [EOS])

    scheduler.add_request(close("s"))

    assert request.status == RequestStatus.FINISHED_STOPPED
    assert "s" not in scheduler.requests
    assert num_rows(scheduler) == 0
    assert free_blocks(scheduler) == free_before


# @spec PSTATE-TURN-003 PSTATE-FAIL-002
@pytest.mark.parametrize(
    ("opening", "then", "sampled", "finish_reason", "terminal"),
    [
        ({}, None, EOS, FinishReason.STOP, False),
        ({}, "update", EOS, FinishReason.STOP, False),
        ({}, "close", EOS, FinishReason.STOP, True),
        ({}, "fold_past_limit", EOS, FinishReason.LENGTH, True),
        ({"max_tokens": 1}, None, TOKEN, FinishReason.LENGTH, True),
        ({"num_tokens": MAX_MODEL_LEN - 1}, None, TOKEN, FinishReason.LENGTH, True),
        ({"stop_token_ids": [STOP_TOKEN]}, None, STOP_TOKEN, FinishReason.STOP, True),
        (
            {"repetition_detection": REPEATS},
            "repeat",
            TOKEN,
            FinishReason.REPETITION,
            True,
        ),
        ({}, "model_failure", TOKEN, FinishReason.ERROR, True),
    ],
    ids=[
        "eos_parks",
        "eos_folds",
        "eos_before_close",
        "rejected_fold",
        "max_tokens",
        "max_model_len",
        "stop_token",
        "repetition",
        "model_failure",
    ],
)
def test_finished_output_marks_whether_it_ends_the_stream(
    monkeypatch, opening, then, sampled, finish_reason, terminal
):
    scheduler = make_scheduler(monkeypatch)
    opening = dict(opening)
    num_tokens = opening.pop("num_tokens", 4)
    if then == "fold_past_limit":
        open_stream(scheduler, "s", 20, [EOS])
        scheduler.add_request(update("s", 10))
        scheduler.add_request(update("s", 40))
    else:
        scheduler.add_request(stream_input("s", num_tokens, **opening))
        if then == "update":
            scheduler.add_request(update("s", 4))
        elif then == "close":
            scheduler.add_request(close("s"))
        elif then == "repeat":
            step(scheduler, {"s": sampled})
    runner_kwargs = {"failed_req_ids": {"s"}} if then == "model_failure" else {}

    _, outputs = step(scheduler, {"s": sampled}, **runner_kwargs)

    output = finished_output(outputs, "s")
    assert output.finish_reason == finish_reason
    assert output.terminal is terminal
    assert ("s" in scheduler.requests) is not terminal


# @spec PSTATE-TURN-002
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
        (
            4,
            {"repetition_detection": REPEATS},
            [TOKEN, TOKEN],
            RequestStatus.FINISHED_REPETITION,
        ),
    ],
    ids=["max_tokens", "max_model_len", "stop_token", "repetition"],
)
def test_non_resumable_stop_finishes_state_stream_despite_queued_input(
    monkeypatch, num_tokens, request_kwargs, tokens, status
):
    scheduler = make_scheduler(monkeypatch)
    request = open_stream(scheduler, "s", num_tokens, tokens[:-1], **request_kwargs)
    scheduler.add_request(update("s", 4))

    run_turn(scheduler, "s", tokens[-1:])

    assert request.status == status
    assert "s" not in scheduler.requests


# @spec PSTATE-TURN-002
@pytest.mark.parametrize("tokens", [[TOKEN], [EOS]], ids=["running", "parked"])
def test_abort_finishes_state_stream(monkeypatch, tokens):
    scheduler = make_scheduler(monkeypatch)
    request = open_stream(scheduler, "s", 4, tokens)

    scheduler.finish_requests("s", RequestStatus.FINISHED_ABORTED)

    assert request.status == RequestStatus.FINISHED_ABORTED
    assert "s" not in scheduler.requests


# @spec PSTATE-TURN-008 PSTATE-TURN-009 PSTATE-HIST-003
def test_update_past_position_limit_finishes_state_stream_length_capped(
    monkeypatch,
):
    scheduler = make_scheduler(monkeypatch)
    free_before = free_blocks(scheduler)
    request = park_after_two_turns(scheduler)
    assert free_blocks(scheduler) < free_before

    scheduler.add_request(update("s", 40))

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert "s" not in scheduler.requests
    assert request not in scheduler.dropped_streaming_updates
    assert free_blocks(scheduler) == free_before


# @spec PSTATE-TURN-008 PSTATE-HIST-003
def test_refused_update_leaves_request_unchanged(monkeypatch):
    scheduler = make_scheduler(monkeypatch)
    request = park_after_two_turns(scheduler)
    prompt = list(request.prompt_token_ids)
    offset = request.position_offset

    scheduler.add_request(update("s", 40))

    assert list(request.prompt_token_ids) == prompt
    assert request.position_offset == offset
    assert request.num_computed_tokens == len(prompt)


# @spec PSTATE-TURN-009 PSTATE-HIST-003 PSTATE-HIST-001
def test_position_limit_uses_compacted_length_for_model_without_positions(
    monkeypatch,
):
    scheduler = make_scheduler(
        monkeypatch, facts={"persistent_state_reads_positions": False}
    )
    request = park_after_two_turns(scheduler)
    assert request.position_offset == 0

    scheduler.add_request(update("s", 40))

    assert request.status == RequestStatus.WAITING
    assert len(request.prompt_token_ids) == 1 + 40
    assert request.position_offset == 0


# @spec PSTATE-TURN-009
def test_position_limit_counts_full_length_for_hybrid_request(monkeypatch):
    scheduler = make_scheduler(monkeypatch, hybrid=True)
    request = park_after_two_turns(scheduler)
    assert request.position_offset == 0

    scheduler.add_request(update("s", 40))

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert "s" not in scheduler.requests


# @spec PSTATE-TURN-010 PSTATE-TURN-009
@pytest.mark.parametrize(
    ("reads_positions", "position_offset", "final_num_tokens"),
    [(True, 19, MAX_MODEL_LEN - 19), (False, 0, MAX_MODEL_LEN)],
    ids=["reads_positions", "ignores_positions"],
)
def test_decode_stops_at_position_limit(
    monkeypatch, reads_positions, position_offset, final_num_tokens
):
    scheduler = make_scheduler(
        monkeypatch, facts={"persistent_state_reads_positions": reads_positions}
    )
    open_stream(scheduler, "s", 20, [EOS])
    request = scheduler.requests["s"]
    scheduler.add_request(update("s", 10))
    assert request.position_offset == position_offset

    for _ in range(MAX_MODEL_LEN):
        if request.is_finished():
            break
        step(scheduler)

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert request.num_tokens == final_num_tokens


# @spec PSTATE-TURN-010
def test_running_loop_clamps_a_multi_token_step_to_the_position_length(monkeypatch):
    """A chunked prefill continuing in the running loop: with 52 dropped
    positions, 8 computed tokens and one sampled token a step, only 3 more
    fit below the position limit, although the budget allows 8. Compaction
    leaves no request this close to the limit mid-prefill, so the offset is
    set directly."""
    scheduler = make_scheduler(monkeypatch, max_num_batched_tokens=8)
    request = stream_input("s", 20)
    scheduler.add_request(request)
    first = scheduler.schedule()
    assert first.num_scheduled_tokens == {"s": 8}
    # A partial prefill samples nothing.
    scheduler.update_from_output(
        first, ModelRunnerOutput(req_ids=["s"], sampled_token_ids=[[]])
    )
    request.position_offset = MAX_MODEL_LEN - 8 - 1 - 3

    second = scheduler.schedule()

    assert second.num_scheduled_tokens == {"s": 3}


# @spec PSTATE-TURN-011 PSTATE-TURN-008
@pytest.mark.parametrize(
    ("hybrid", "reads_positions", "continues"),
    [(False, True, False), (False, False, True), (True, True, False)],
    ids=["reads_positions", "ignores_positions", "hybrid"],
)
def test_eos_at_position_limit_parks_and_next_update_meets_the_limit(
    monkeypatch, hybrid, reads_positions, continues
):
    scheduler = make_scheduler(
        monkeypatch,
        hybrid=hybrid,
        facts={"persistent_state_reads_positions": reads_positions},
    )
    request = open_stream(scheduler, "s", MAX_MODEL_LEN - 1, [EOS])
    assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ

    scheduler.add_request(update("s", 4))

    if continues:
        assert request.status == RequestStatus.WAITING
        assert len(request.prompt_token_ids) == 1 + 4
    else:
        assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
        assert "s" not in scheduler.requests


# @spec PSTATE-TURN-011
def test_eos_at_turn_budget_parks_state_stream(monkeypatch):
    scheduler = make_scheduler(monkeypatch)
    request = open_stream(scheduler, "s", 4, [EOS], max_tokens=1)

    assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ


# @spec PSTATE-TURN-011
def test_eos_at_position_limit_keeps_upstream_order_for_single_request(
    monkeypatch,
):
    scheduler = make_scheduler(monkeypatch)
    request = open_stream(scheduler, "s", MAX_MODEL_LEN - 1, [EOS], resumable=False)

    assert request.status == RequestStatus.FINISHED_STOPPED


DUPLEX_FACTS = {"persistent_state_resumable_stops": frozenset({"eos", "max_tokens"})}
"""A model that also declares its per-turn `max_tokens` stop resumable."""


# @spec PSTATE-TURN-012 PSTATE-TURN-001 PSTATE-TURN-003
def test_declared_max_tokens_stop_parks_state_stream(monkeypatch):
    scheduler = make_scheduler(monkeypatch, facts=DUPLEX_FACTS)
    request = stream_input("s", 4, max_tokens=1)
    scheduler.add_request(request)

    _, outputs = step(scheduler, {"s": TOKEN})

    assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert any(scheduler.kv_cache_manager.get_block_ids("s"))
    output = finished_output(outputs, "s")
    assert (output.finish_reason, output.terminal) == (FinishReason.LENGTH, False)


# @spec PSTATE-TURN-012 PSTATE-TURN-001 PSTATE-HIST-001
def test_declared_max_tokens_stop_folds_one_frame_per_turn(monkeypatch):
    """One token per frame: each input folds at the previous frame's stop, and
    history stays one frontier token plus the frame."""
    scheduler = make_scheduler(monkeypatch, facts=DUPLEX_FACTS)
    request = open_stream(scheduler, "s", 2, [], max_tokens=1)
    for _ in range(3):
        scheduler.add_request(update("s", 2, max_tokens=1))

    for _ in range(3):
        run_turn(scheduler, "s", [TOKEN])
        assert request.status == RequestStatus.WAITING
        assert len(request.prompt_token_ids) == 1 + 2
    run_turn(scheduler, "s", [TOKEN])

    assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ


# @spec PSTATE-TURN-012 PSTATE-TURN-002
def test_undeclared_max_tokens_stop_ends_state_stream(monkeypatch):
    scheduler = make_scheduler(monkeypatch)
    request = open_stream(scheduler, "s", 4, [TOKEN], max_tokens=1)

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert "s" not in scheduler.requests


# @spec PSTATE-TURN-012 PSTATE-TURN-002
def test_max_tokens_stop_at_the_position_limit_ends_state_stream(monkeypatch):
    scheduler = make_scheduler(monkeypatch, facts=DUPLEX_FACTS)
    request = open_stream(scheduler, "s", MAX_MODEL_LEN - 1, [TOKEN], max_tokens=1)

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert "s" not in scheduler.requests


# @spec PSTATE-TURN-012
def test_model_declaring_only_max_tokens_ends_on_eos(monkeypatch):
    scheduler = make_scheduler(
        monkeypatch,
        facts={"persistent_state_resumable_stops": frozenset({"max_tokens"})},
    )
    request = open_stream(scheduler, "s", 4, [EOS])

    assert request.status == RequestStatus.FINISHED_STOPPED
    assert "s" not in scheduler.requests


# @spec PSTATE-TURN-015 PSTATE-TURN-012
@pytest.mark.parametrize(
    ("parked", "status"),
    [
        (False, RequestStatus.FINISHED_LENGTH_CAPPED),
        (True, RequestStatus.FINISHED_STOPPED),
    ],
    ids=["queued_close", "parked_close"],
)
def test_close_after_max_tokens_stop_takes_its_status_from_where_it_waits(
    monkeypatch, parked, status
):
    scheduler = make_scheduler(monkeypatch, facts=DUPLEX_FACTS)
    request = stream_input("s", 4, max_tokens=1)
    scheduler.add_request(request)
    if not parked:
        scheduler.add_request(close("s"))

    _, outputs = step(scheduler, {"s": TOKEN})
    turn_end = finished_output(outputs, "s")
    if parked:
        assert turn_end.terminal is False
        scheduler.add_request(close("s"))
        # The engine core publishes this stream's terminal output.
        assert scheduler.ended_state_streams == [(request, None)]
    else:
        assert (turn_end.finish_reason, turn_end.terminal) == (
            FinishReason.LENGTH,
            True,
        )

    assert request.status == status
    assert "s" not in scheduler.requests


class _ReadyGrammar:
    """Stands for a compiled grammar that accepts every token."""

    def accept_tokens(self, request_id, tokens):
        return True

    def is_terminated(self):
        return False


# @spec PSTATE-TURN-020
@pytest.mark.parametrize("compiled", ["ready", "failed"])
def test_first_input_of_an_admitted_stream_waits_for_its_grammar(monkeypatch, compiled):
    scheduler = make_scheduler(monkeypatch)
    admission = stream_input("s", 0)
    admission.sampling_params = SamplingParams(
        max_tokens=100,
        output_kind=RequestOutputKind.DELTA,
        structured_outputs=StructuredOutputsParams(regex="a+"),
    )
    admission.structured_output_request = StructuredOutputRequest.from_sampling_params(
        admission.sampling_params
    )
    scheduler.admit_stream(admission)
    first = update("s", 4)
    first.sampling_params = admission.sampling_params

    scheduler.add_request(first)

    assert admission.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    assert "s" not in scheduler.schedule().num_scheduled_tokens

    if compiled == "ready":
        admission.structured_output_request.grammar = _ReadyGrammar()
        assert "s" in scheduler.schedule().num_scheduled_tokens
    else:
        admission.structured_output_request.grammar = ValueError("bad schema")
        _, outputs = step(scheduler)
        output = finished_output(outputs, "s")
        assert (output.finish_reason, output.terminal) == (FinishReason.ERROR, True)
        assert output.error is not None and output.error.code == "validation_error"
        assert "s" not in scheduler.requests


def structured_stream_input(req_id: str, num_tokens: int, **kwargs) -> Request:
    request = stream_input(req_id, num_tokens, **kwargs)
    request.sampling_params = SamplingParams(
        max_tokens=100,
        output_kind=RequestOutputKind.DELTA,
        structured_outputs=StructuredOutputsParams(regex="a+"),
    )
    request.structured_output_request = StructuredOutputRequest.from_sampling_params(
        request.sampling_params
    )
    return request


# @spec PSTATE-TURN-015 PSTATE-TURN-020
@pytest.mark.parametrize("compiled", ["ready", "failed"])
def test_close_of_a_stream_whose_grammar_failed_ends_it_with_an_error(
    monkeypatch, compiled
):
    scheduler = make_scheduler(monkeypatch)
    admission = structured_stream_input("s", 0)
    assert scheduler.admit_stream(admission) is None
    admission.structured_output_request.grammar = (
        ValueError("bad schema") if compiled == "failed" else _ReadyGrammar()
    )

    scheduler.add_request(close("s"))

    [(ended, error)] = scheduler.ended_state_streams
    assert ended is admission
    if compiled == "failed":
        assert admission.status == RequestStatus.FINISHED_ERROR
        assert error is not None and error.code == "validation_error"
        assert "bad schema" in error.message
    else:
        assert admission.status == RequestStatus.FINISHED_STOPPED
        assert error is None
    assert "s" not in scheduler.requests


# @spec PSTATE-TURN-020
@pytest.mark.parametrize("state_stream", [True, False], ids=["stream", "single"])
def test_grammar_rejection_ends_a_state_stream_terminally(monkeypatch, state_stream):
    scheduler = make_scheduler(monkeypatch)
    request = stream_input("s", 4, resumable=state_stream)
    scheduler.add_request(request)
    monkeypatch.setattr(
        scheduler.structured_output_manager,
        "accept_tokens",
        lambda request, tokens: False,
    )

    _, outputs = step(scheduler)

    output = finished_output(outputs, "s")
    assert request.status == RequestStatus.FINISHED_ERROR
    assert output.finish_reason == FinishReason.ERROR
    # Only a stream's output ends it for the frontend; a single request ends
    # with its finish reason alone.
    assert output.terminal is state_stream
    assert "s" not in scheduler.requests


# @spec PSTATE-TURN-014
def test_hybrid_request_keeps_full_history_on_fold(monkeypatch):
    scheduler = make_scheduler(monkeypatch, hybrid=True)
    request = open_stream(scheduler, "s", 20, [TOKEN, EOS])

    scheduler.add_request(update("s", 10))

    assert list(request.prompt_token_ids) == [0] * 20 + [TOKEN] + [0] * 10
    assert request.num_computed_tokens == 21
    assert request.position_offset == 0
    assert len(request.block_hashes) == 1


# Bounded history


# @spec PSTATE-HIST-001
@pytest.mark.parametrize("queued", [False, True], ids=["parked", "queued"])
def test_fold_compacts_history_to_frontier_and_new_input(monkeypatch, queued):
    scheduler = make_scheduler(monkeypatch)
    request = stream_input("s", 20)
    scheduler.add_request(request)
    step(scheduler, {"s": TOKEN})
    assert len(request.block_hashes) == 1

    if queued:
        scheduler.add_request(update("s", 10))
    step(scheduler, {"s": EOS})
    if not queued:
        scheduler.add_request(update("s", 10))

    assert list(request.prompt_token_ids) == [TOKEN] + [0] * 10
    assert list(request.all_token_ids) == [TOKEN] + [0] * 10
    assert request.num_prompt_tokens == 11
    assert request.num_computed_tokens == 1
    assert request.block_hashes == []
    assert request.position_offset == 20


# @spec PSTATE-HIST-001
def test_second_fold_adds_dropped_tokens_to_position_offset(monkeypatch):
    scheduler = make_scheduler(monkeypatch)
    request = park_after_two_turns(scheduler)

    assert request.position_offset == 19

    scheduler.add_request(update("s", 5))

    assert request.position_offset == 19 + 10
    assert request.num_computed_tokens == 1
    assert len(request.prompt_token_ids) == 1 + 5


# @spec PSTATE-HIST-002
def test_fold_keeps_only_updates_features_relative_to_compacted_prompt(
    monkeypatch,
):
    scheduler = make_scheduler(monkeypatch, model="llava-hf/llava-1.5-7b-hf")
    request = stream_input(
        "s",
        20,
        mm_hashes_list=[["opening-image"]],
        mm_positions=[[PlaceholderRange(offset=2, length=5)]],
    )
    scheduler.add_request(request)
    run_turn(scheduler, "s", [EOS])
    assert [f.identifier for f in request.mm_features] == ["opening-image"]

    scheduler.add_request(
        update(
            "s",
            10,
            mm_hashes_list=[["update-image"]],
            mm_positions=[[PlaceholderRange(offset=2, length=3)]],
        )
    )

    assert [f.identifier for f in request.mm_features] == ["update-image"]
    assert request.mm_features[0].mm_position.offset == 1 + 2

    # The scheduler drops a request's reference to an encoder input in the same
    # `schedule()` that covers its placeholder, so record what it allocates.
    encoder_cache = scheduler.encoder_cache_manager
    allocated: list[tuple[str, int, str, set[int], set[str], set[str]]] = []
    allocate = encoder_cache.allocate

    def recording_allocate(alloc_request, input_id):
        allocate(alloc_request, input_id)
        allocated.append(
            (
                alloc_request.request_id,
                input_id,
                alloc_request.mm_features[input_id].identifier,
                set(encoder_cache.get_cached_input_ids(alloc_request)),
                set(encoder_cache.request_hash_refs[alloc_request.request_id]),
                set(encoder_cache.cached["update-image"]),
            )
        )

    monkeypatch.setattr(encoder_cache, "allocate", recording_allocate)

    output = scheduler.schedule()

    # The update's image is the request's only encoder input, at its rebased
    # offset inside the compacted prompt.
    assert output.num_scheduled_tokens == {"s": 1 + 10 - 1}
    assert output.scheduled_encoder_inputs == {"s": [0]}
    assert allocated == [("s", 0, "update-image", {0}, {"update-image"}, {"s"})]
    assert "s" not in encoder_cache.cached.get("opening-image", set())
    assert "opening-image" not in encoder_cache.request_hash_refs.get("s", {})


# @spec PSTATE-HIST-006
def test_scheduler_queues_stream_inputs_beyond_the_pending_bound(monkeypatch):
    """`AsyncLLM` bounds pending inputs by backpressure; the scheduler ends no
    stream for the length of its queue."""
    scheduler = make_scheduler(monkeypatch)
    request = open_stream(scheduler, "s", 4, [TOKEN])
    bound = scheduler.scheduler_config.max_pending_stream_inputs

    for _ in range(bound + 2):
        scheduler.add_request(update("s", 4))
    scheduler.add_request(close("s"))

    assert len(request.streaming_queue) == bound + 3
    assert not request.is_finished()


# Admission


# @spec PSTATE-ADMIT-002
def test_admitted_stream_is_a_parked_request_with_an_empty_prompt(monkeypatch):
    scheduler = make_scheduler(monkeypatch)
    request = admission_request("a")

    assert scheduler.admit_stream(request) is None

    assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert scheduler.requests["a"] is request
    assert request in scheduler.waiting
    assert scheduler.num_waiting_for_streaming_input == 1
    assert len(request.prompt_token_ids) == 0


# @spec PSTATE-ADMIT-002
def test_first_input_becomes_prompt_and_moves_reservation_without_a_gap(
    monkeypatch,
):
    scheduler = make_scheduler(monkeypatch)
    request = admission_request("a")
    scheduler.admit_stream(request)
    assert num_rows(scheduler) == 1

    scheduler.add_request(update("a", 6))

    assert list(request.prompt_token_ids) == [0] * 6
    assert request.num_prompt_tokens == 6
    assert request.num_computed_tokens == 0
    assert request.position_offset == 0
    assert request.status == RequestStatus.WAITING
    assert scheduler.num_waiting_for_streaming_input == 0
    assert request in scheduler.resumed_streaming_sessions
    assert num_rows(scheduler) == 1

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {"a": 6}
    assert not scheduler.resumed_streaming_sessions
    assert num_rows(scheduler) == 1


def _hold_row(scheduler, kind: str, req_id: str) -> None:
    if kind == "admitted":
        scheduler.admit_stream(admission_request(req_id))
    elif kind == "resumed":
        scheduler.admit_stream(admission_request(req_id))
        scheduler.add_request(update(req_id, 4))
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
def test_admission_counts_running_parked_and_resumed_rows(monkeypatch, first, second):
    scheduler = make_scheduler(monkeypatch, max_num_seqs=2)
    _hold_row(scheduler, first, "a")
    _hold_row(scheduler, second, "b")

    assert scheduler.admit_stream(admission_request("c")) == "active_limit"
    assert "c" not in scheduler.requests

    scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)

    assert scheduler.admit_stream(admission_request("c")) is None


# Per-request failure and terminal precedence


# @spec PSTATE-FAIL-002 PSTATE-ADMIT-014
def test_reported_failure_finishes_only_its_request_before_its_tokens_and_stop(
    monkeypatch,
):
    scheduler = make_scheduler(monkeypatch)
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
def test_reported_failure_finishes_parked_stream(monkeypatch):
    scheduler = make_scheduler(monkeypatch)
    parked = open_stream(scheduler, "parked", 4, [EOS])
    scheduler.add_request(stream_input("running", 4))

    step(scheduler, failed_req_ids={"parked"})

    assert parked.status == RequestStatus.FINISHED_ERROR
    assert "parked" not in scheduler.requests
    assert scheduler.requests["running"].status == RequestStatus.RUNNING


# @spec PSTATE-ADMIT-014
def test_abort_ahead_of_failure_report_fixes_aborted_outcome(monkeypatch):
    scheduler = make_scheduler(monkeypatch)
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
def test_abort_after_failure_keeps_error_outcome(monkeypatch):
    scheduler = make_scheduler(monkeypatch)
    request = stream_input("s", 4)
    scheduler.add_request(request)
    step(scheduler, failed_req_ids={"s"})

    scheduler.finish_requests("s", RequestStatus.FINISHED_ABORTED)

    assert request.status == RequestStatus.FINISHED_ERROR


# Ordinary models


# @spec PSTATE-DECL-013
@pytest.mark.parametrize("max_tokens", [1, 2])
def test_ordinary_stream_parks_on_length_stop(max_tokens):
    scheduler = create_scheduler()
    request = open_stream(
        scheduler, "s", 4, [TOKEN] * max_tokens, max_tokens=max_tokens
    )

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


# @spec PSTATE-DECL-013
def test_ordinary_stream_closed_while_parked_is_aborted_without_output():
    scheduler = create_scheduler()
    request = open_stream(scheduler, "s", 4, [EOS])

    scheduler.add_request(stream_input("s", 4, resumable=False))
    _, outputs = step(scheduler)

    assert request.status == RequestStatus.FINISHED_ABORTED
    assert "s" not in scheduler.requests
    assert not any(o.request_id == "s" for eco in outputs.values() for o in eco.outputs)
