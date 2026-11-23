# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent state in the output processor, detokenizer and output collector.

A state request's history is replaced, not appended, on each update; its pending
inputs and undelivered turns are bounded; and a terminal finish ends the stream.
"""

import numpy as np
import pytest

from vllm.exceptions import GenerationError
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import (
    OutputProcessor,
    StreamingInputOutputCollector,
)
from vllm.v1.outputs import LogprobsLists

MAX_PENDING = 8


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
    token: int = 4,
    finish_reason: FinishReason = FinishReason.STOP,
    **fields,
):
    return processor.process_outputs(
        [
            EngineCoreOutput(
                request_id="s",
                new_token_ids=[token],
                finish_reason=finish_reason,
                **fields,
            )
        ]
    )


# @spec PSTATE-TURN-003
def test_terminal_finish_ends_the_stream_instead_of_starting_the_next_turn():
    processor = _processor()
    queue = _open(processor)
    _queue_input(processor, [7, 8])

    _turn_end(processor, terminal=True)

    assert processor.get_num_unfinished_requests() == 0
    output = queue.get_nowait()
    assert output is not None and output.finished is True


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


def _turn_texts(vectors, persistent_state: bool) -> tuple[list[str], list[int]]:
    """Stream six turns, each queued before the previous one ends, and return
    each turn's text and the tokens the detokenizer retains after it."""
    processor = _processor(vectors.tokenizer, persistent_state=persistent_state)
    queue = StreamingInputOutputCollector(RequestOutputKind.DELTA, "s")
    prompts = vectors.prompt_tokens
    generated = list(vectors.generation_tokens[0]) * 8
    assert len(generated) >= 48
    turns = [generated[i : i + 8] for i in range(0, 48, 8)]
    processor.add_request(_input("s", prompts[0], detokenize=True), None, queue=queue)
    texts, retained = [], []
    for i, turn in enumerate(turns):
        if i + 1 < len(turns):
            next_prompt = prompts[(i + 1) % len(prompts)]
            processor.add_request(_input("s", next_prompt, detokenize=True), None)
        processor.process_outputs(
            [EngineCoreOutput("s", turn, finish_reason=FinishReason.STOP)]
        )
        output = queue.get_nowait()
        assert output is not None
        texts.append(output.outputs[0].text)
        retained.append(len(processor.request_states["s"].detokenizer.token_ids))
    return texts, retained


# @spec PSTATE-HIST-005
def test_detokenizer_tail_decodes_each_turn_as_full_history_does(
    dummy_test_vectors,
):
    expected, _ = _turn_texts(dummy_test_vectors, persistent_state=False)

    texts, retained = _turn_texts(dummy_test_vectors, persistent_state=True)

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
    assert state.sampling_mask_chunks == []
    assert state.weight_version_starts == []


# @spec PSTATE-HIST-006
def test_input_chunk_queue_holds_the_pending_bound():
    processor = _processor()
    _open(processor)
    for i in range(MAX_PENDING):
        _queue_input(processor, [10 + i])

    assert len(processor.request_states["s"].input_chunk_queue) == MAX_PENDING
    assert processor.has_request("s")


# @spec PSTATE-HIST-007
@pytest.mark.xfail(strict=False, reason="pending Pete: backpressure vs error")
def test_input_beyond_the_pending_bound_ends_the_stream():
    processor = _processor()
    queue = _open(processor)
    for i in range(MAX_PENDING):
        _queue_input(processor, [10 + i])

    _queue_input(processor, [99])

    assert not processor.has_request("s")
    with pytest.raises(GenerationError, match="pending_input_limit"):
        queue.get_nowait()


# @spec PSTATE-HIST-006
def test_close_signal_does_not_count_against_the_pending_bound():
    processor = _processor()
    _open(processor)
    for i in range(MAX_PENDING):
        _queue_input(processor, [10 + i])

    processor.add_request(_input("s", [0], resumable=False), None)

    assert processor.has_request("s")


def _run_turns(processor: OutputProcessor, num_turns: int) -> None:
    for token in range(num_turns):
        _turn_end(processor, token)


# @spec PSTATE-HIST-006
def test_collector_backlog_holds_at_most_the_pending_bound_of_turns():
    processor = _processor()
    queue = _open(processor)
    for i in range(MAX_PENDING):
        _queue_input(processor, [10 + i])

    _run_turns(processor, MAX_PENDING)
    assert processor.has_request("s")

    processed = _turn_end(processor)
    assert not processor.has_request("s")
    assert processed.reqs_to_abort == ["s"]
    with pytest.raises(GenerationError, match="output_backlog_limit"):
        queue.get_nowait()


# @spec PSTATE-HIST-010
def test_unread_turns_beyond_the_bound_finish_the_stream_with_a_named_error():
    processor = _processor()
    queue = _open(processor)
    for i in range(MAX_PENDING):
        _queue_input(processor, [10 + i])
    _run_turns(processor, MAX_PENDING)

    processed = _turn_end(processor)

    assert processed.reqs_to_abort == ["s"]
    assert not processor.has_request("s")
    with pytest.raises(GenerationError) as excinfo:
        queue.get_nowait()
    # Not retryable: a client that stopped reading has lost turns.
    assert type(excinfo.value) is GenerationError
    assert "output_backlog_limit" in str(excinfo.value)


# @spec PSTATE-HIST-011
def test_input_queue_and_backlog_bounds_apply_independently():
    processor = _processor()
    _open(processor)
    for i in range(MAX_PENDING):
        _queue_input(processor, [10 + i])
    _run_turns(processor, MAX_PENDING)

    for i in range(MAX_PENDING):
        _queue_input(processor, [30 + i])

    assert processor.has_request("s")
    assert len(processor.request_states["s"].input_chunk_queue) == MAX_PENDING


# @spec PSTATE-HIST-011
def test_terminal_output_is_delivered_with_a_full_backlog_and_drops_inputs():
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
