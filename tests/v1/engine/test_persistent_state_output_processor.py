# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent state in the output processor, detokenizer and output collector.

A state request's history is replaced, not appended, on each update; a state
stream ends only on a core output marked terminal; and the output processor sets
no bound of its own on pending inputs or undelivered turns.
"""

from collections import deque

import numpy as np
import pytest

from vllm.outputs import STREAM_FINISHED
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tokenizers import get_tokenizer
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import (
    OutputProcessor,
    StreamingInputOutputCollector,
)
from vllm.v1.outputs import LogprobsLists

MAX_PENDING = 8
"""The default `max_pending_stream_inputs`."""


def _processor(tokenizer=None, persistent_state: bool = True) -> OutputProcessor:
    return OutputProcessor(
        tokenizer, log_stats=False, persistent_state=persistent_state
    )


def _input(
    request_id: str,
    prompt_token_ids: list[int],
    *,
    resumable: bool = True,
    **sampling_kwargs,
) -> EngineCoreRequest:
    sampling_kwargs.setdefault("detokenize", False)
    return EngineCoreRequest(
        request_id=request_id,
        external_req_id=f"{request_id}-ext",
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=SamplingParams(
            output_kind=RequestOutputKind.DELTA, **sampling_kwargs
        ),
        pooling_params=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        resumable=resumable,
    )


def _open(
    processor: OutputProcessor,
    request_id: str = "s",
    prompt_token_ids: list[int] | None = None,
    prompt: str | None = None,
    **sampling_kwargs,
) -> StreamingInputOutputCollector:
    queue = StreamingInputOutputCollector(RequestOutputKind.DELTA, request_id)
    request = _input(request_id, prompt_token_ids or [1, 2, 3], **sampling_kwargs)
    processor.add_request(request, prompt, queue=queue)
    return queue


def _queue_input(processor: OutputProcessor, token_ids: list[int], prompt=None):
    processor.add_request(_input("s", token_ids), prompt)


def _turn_end(
    processor: OutputProcessor,
    token: int | None = 4,
    finish_reason: FinishReason = FinishReason.STOP,
    **fields,
):
    return processor.process_outputs(
        [
            EngineCoreOutput(
                request_id="s",
                new_token_ids=[] if token is None else [token],
                finish_reason=finish_reason,
                **fields,
            )
        ]
    )


# @spec PSTATE-TURN-016
def test_terminal_finish_ends_the_stream_instead_of_starting_the_next_turn():
    processor = _processor()
    queue = _open(processor)
    _queue_input(processor, [7, 8])

    _turn_end(processor, terminal=True)

    assert processor.get_num_unfinished_requests() == 0
    output = queue.get_nowait()
    assert output is not None and output.finished is True


def _close(processor: OutputProcessor) -> None:
    processor.add_request(_input("s", [0], resumable=False), None)


def _admit(processor: OutputProcessor) -> StreamingInputOutputCollector:
    """An admitted stream: its opening request carries no prompt."""
    queue = StreamingInputOutputCollector(RequestOutputKind.DELTA, "s")
    processor.add_request(_input("s", []), None, queue=queue)
    return queue


# @spec PSTATE-TURN-016 PSTATE-TURN-015
@pytest.mark.parametrize("opening", ["parked", "admitted"])
def test_close_of_a_parked_stream_waits_for_the_terminal_output(opening):
    """The output processor synthesizes no completion for a state stream; the
    core's terminal output ends it."""
    processor = _processor()
    if opening == "admitted":
        queue = _admit(processor)
    else:
        queue = _open(processor)
        _turn_end(processor)
        assert queue.get_nowait().finished is False

    _close(processor)

    assert processor.has_request("s")
    assert queue.get_nowait() is None

    _turn_end(processor, token=None, terminal=True)

    assert not processor.has_request("s")
    output = queue.get_nowait()
    assert output is not STREAM_FINISHED and output.finished is True


# @spec PSTATE-TURN-016
def test_eos_boundary_in_flight_when_the_close_is_sent_stays_a_boundary():
    """The close can reach the frontend before core applies it: the turn end
    that core marks resumable keeps the stream, and core's terminal output
    after it ends the stream."""
    processor = _processor()
    queue = _open(processor)
    _close(processor)

    _turn_end(processor, terminal=False)

    assert processor.has_request("s")
    assert queue.get_nowait().finished is False

    _turn_end(processor, token=None, terminal=True)

    assert not processor.has_request("s")
    assert queue.get_nowait().finished is True


# @spec PSTATE-DECL-013
def test_ordinary_stream_closed_while_parked_gets_a_synthesized_completion():
    processor = _processor(persistent_state=False)
    queue = _open(processor)
    _turn_end(processor)
    assert queue.get_nowait().finished is False

    _close(processor)

    assert not processor.has_request("s")
    assert queue.get_nowait() is STREAM_FINISHED


# @spec PSTATE-ADMIT-002
def test_admitted_streams_first_input_becomes_its_first_turn():
    processor = _processor()
    _admit(processor)

    processor.add_request(_input("s", [7, 8]), "ab")

    state = processor.request_states["s"]
    assert (state.prompt, state.prompt_token_ids, state.prompt_len) == ("ab", [7, 8], 2)
    assert state.streaming_input and state.input_chunk_queue == deque()


# @spec PSTATE-HIST-005
def test_update_replaces_the_prompt_text_and_ids():
    processor = _processor()
    _open(processor, prompt_token_ids=[1, 2, 3], prompt="abc")
    _queue_input(processor, [7, 8], prompt="de")

    _turn_end(processor)

    state = processor.request_states["s"]
    assert state.prompt == "de"
    assert state.prompt_token_ids == [7, 8]
    assert state.prompt_len == 2


_TURN_TEXT = (
    "The quick brown fox jumps over the lazy dog, and then it runs into the "
    "forest where the trees are tall and the river is cold and clear."
)


def _turn_texts(tokenizer, persistent_state: bool) -> tuple[list[str], list[int]]:
    """Stream six turns, each input sent once the previous turn ends, and
    return each turn's text and the tokens the detokenizer retains at the
    turn's end, before the next input starts the next turn."""
    processor = _processor(tokenizer, persistent_state=persistent_state)
    queue = StreamingInputOutputCollector(RequestOutputKind.DELTA, "s")
    prompts = [
        tokenizer.encode(text, add_special_tokens=False)
        for text in ("Hello there.", "Next part:", "And then")
    ]
    generated = tokenizer.encode(_TURN_TEXT, add_special_tokens=False) * 2
    assert len(generated) >= 48
    turns = [generated[i : i + 8] for i in range(0, 48, 8)]
    processor.add_request(_input("s", prompts[0], detokenize=True), None, queue=queue)
    texts, retained = [], []
    for i, turn in enumerate(turns):
        processor.process_outputs(
            [EngineCoreOutput("s", turn, finish_reason=FinishReason.STOP)]
        )
        output = queue.get_nowait()
        assert output is not None
        texts.append(output.outputs[0].text)
        retained.append(len(processor.request_states["s"].detokenizer.token_ids))
        if i + 1 < len(turns):
            next_prompt = prompts[(i + 1) % len(prompts)]
            processor.add_request(_input("s", next_prompt, detokenize=True), None)
    return texts, retained


# @spec PSTATE-HIST-005
def test_detokenizer_tail_decodes_each_turn_as_full_history_does():
    tokenizer = get_tokenizer("facebook/opt-125m")
    expected, _ = _turn_texts(tokenizer, persistent_state=False)

    texts, retained = _turn_texts(tokenizer, persistent_state=True)

    assert texts == expected
    assert all(text for text in texts)
    assert max(retained[2:]) <= max(retained[:2])


# @spec PSTATE-HIST-005
def test_logprobs_reset_each_turn():
    processor = _processor()
    _open(processor, logprobs=1)
    _queue_input(processor, [7, 8])
    logprobs = LogprobsLists(
        logprob_token_ids=np.array([[4, 5]]),
        logprobs=np.array([[-0.5, -1.0]]),
        sampled_token_ranks=np.array([1]),
    )

    _turn_end(processor, new_logprobs=logprobs)

    processor_state = processor.request_states["s"].logprobs_processor
    assert len(processor_state.logprobs) == 0
    assert processor_state.cumulative_logprob == 0.0


# @spec PSTATE-HIST-005
def test_per_request_accumulators_reset_each_turn():
    processor = _processor()
    _open(processor)
    _queue_input(processor, [7, 8])
    state = processor.request_states["s"]
    state.routed_experts_chunks.append(np.zeros((1, 1, 1), dtype=np.int16))
    state.weight_version_starts.append(("v1", 0))

    _turn_end(processor)

    assert state.routed_experts_chunks == []
    assert state.weight_version_starts == []


def _run_turns(processor: OutputProcessor, num_turns: int) -> None:
    for token in range(num_turns):
        _turn_end(processor, token)


# @spec PSTATE-HIST-006
def test_output_processor_sets_no_bound_of_its_own():
    """`AsyncLLM` bounds pending inputs and undelivered turns by credits; past
    that bound the output processor still queues inputs and delivers turns."""
    processor = _processor()
    queue = _open(processor)
    for i in range(MAX_PENDING + 2):
        _queue_input(processor, [10 + i])
    _close(processor)
    assert len(processor.request_states["s"].input_chunk_queue) == MAX_PENDING + 2

    _run_turns(processor, MAX_PENDING + 2)

    assert processor.has_request("s")
    outputs = [queue.get_nowait() for _ in range(MAX_PENDING + 2)]
    assert [output.outputs[0].token_ids for output in outputs] == [
        [token] for token in range(MAX_PENDING + 2)
    ]


# @spec PSTATE-TURN-016 PSTATE-HIST-006
def test_terminal_output_is_delivered_behind_undelivered_turns_and_drops_inputs():
    processor = _processor()
    queue = _open(processor)
    for i in range(MAX_PENDING):
        _queue_input(processor, [10 + i])
    _run_turns(processor, MAX_PENDING)
    _queue_input(processor, [50])

    _turn_end(processor, terminal=True)

    assert not processor.has_request("s")
    outputs = [queue.get_nowait() for _ in range(MAX_PENDING + 1)]
    assert [output.finished for output in outputs] == [False] * MAX_PENDING + [True]
