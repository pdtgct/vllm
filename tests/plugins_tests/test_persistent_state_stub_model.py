# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request GPU state held by a standard state layer, end to end.

The dummy-model plugin's `StubPersistentStateForCausalLM` emits a function of
every token the request has fed it, so a lost, shared or uncleared slot changes
the tokens.
"""

import asyncio

import pytest
from vllm_add_dummy_model.my_persistent_state import expected_turns

from tests.utils import create_new_process_for_each_test
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import StreamingInput
from vllm.inputs import TokensPrompt
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

INPUTS = [[10, 20, 30], [40, 50], [60]]
MAX_TOKENS = 3
ENGINE_KWARGS = dict(
    model="facebook/opt-125m",
    hf_overrides={"architectures": ["StubPersistentStateForCausalLM"]},
    load_format="dummy",
    max_model_len=256,
    max_num_seqs=2,
    # The null block plus one 8-byte state page per running request.
    num_gpu_blocks_override=3,
)


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_llm_reuses_slots_from_a_fresh_state(enforce_eager):
    """Six prompts through two slots: each produces the tokens of a fresh
    state, so a reused slot never carries its previous owner's sum."""
    llm = LLM(**ENGINE_KWARGS, enforce_eager=enforce_eager)
    vocab_size = llm.llm_engine.model_config.get_vocab_size()
    prompts = [[11 * i + 1, 11 * i + 2, 11 * i + 3] for i in range(6)]
    params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    outputs = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
    assert [list(output.outputs[0].token_ids) for output in outputs] == [
        expected_turns([prompt], MAX_TOKENS, vocab_size)[0] for prompt in prompts
    ]


async def _stream(llm: AsyncLLM, request_id: str) -> list[list[int]]:
    async def inputs():
        for prompt in INPUTS:
            # Copies: the output processor extends a request's prompt in place.
            yield StreamingInput(prompt=TokensPrompt(prompt_token_ids=list(prompt)))

    params = SamplingParams(
        temperature=0, max_tokens=MAX_TOKENS, output_kind=RequestOutputKind.DELTA
    )
    turns: list[list[int]] = [[]]
    async for output in llm.generate(inputs(), params, request_id):
        completion = output.outputs[0]
        turns[-1].extend(completion.token_ids)
        if completion.finish_reason is not None and not output.finished:
            turns.append([])
    return [turn for turn in turns if turn]


@pytest.mark.asyncio
async def test_async_llm_streams_keep_their_state_across_inputs():
    """Three streaming sessions share two slots: each keeps its state from
    one input to the next, and the third starts from a fresh slot once one
    of the first two ends."""
    llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**ENGINE_KWARGS))
    try:
        expected = expected_turns(INPUTS, MAX_TOKENS, llm.model_config.get_vocab_size())
        results = await asyncio.wait_for(
            asyncio.gather(*(_stream(llm, f"stream-{i}") for i in range(3))), 60
        )
        assert results == [expected] * 3
    finally:
        llm.shutdown()


def _stop_token_case(vocab_size: int) -> tuple[list[int], list[int], int, int, int]:
    """A first input whose turn stops on its second token, before
    `max_tokens`, and whose stop token changes the second turn's first token
    if it is fed back: the stub emits `total` modulo `vocab_size - 3`, so a stop
    token congruent to 0 would hide the extra write."""
    from vllm_add_dummy_model.my_persistent_state import emit

    second = [7]
    for first_token in range(1, 10_000):
        first = [first_token]
        t0 = emit(sum(first), vocab_size)
        stop = emit(sum(first) + t0, vocab_size)
        clean = emit(sum(first) + t0 + sum(second), vocab_size)
        extra = emit(sum(first) + t0 + stop + sum(second), vocab_size)
        if stop != t0 and stop % (vocab_size - 3) and clean != extra:
            return first, second, stop, clean, extra
    raise AssertionError("no observable stop token")


@pytest.mark.asyncio
@pytest.mark.parametrize("async_scheduling", [False, True])
async def test_stop_token_is_not_fed_back(async_scheduling):
    """A turn that stops on a sampled stop token must not feed that token
    into the state. Under async scheduling the step after the stop is already
    scheduled with the stop token as its input; this test records which of
    the two predictions the second turn matches."""
    llm = AsyncLLM.from_engine_args(
        AsyncEngineArgs(**ENGINE_KWARGS, async_scheduling=async_scheduling)
    )
    try:
        vocab_size = llm.model_config.get_vocab_size()
        first, second, stop, clean, extra = _stop_token_case(vocab_size)

        async def inputs():
            for prompt in (first, second):
                yield StreamingInput(prompt=TokensPrompt(prompt_token_ids=list(prompt)))

        params = SamplingParams(
            temperature=0,
            max_tokens=4,
            stop_token_ids=[stop],
            output_kind=RequestOutputKind.DELTA,
        )
        turns: list[list[int]] = [[]]
        async for output in llm.generate(inputs(), params, "stop-stream"):
            completion = output.outputs[0]
            turns[-1].extend(completion.token_ids)
            if completion.finish_reason is not None and not output.finished:
                turns.append([])
        turns = [turn for turn in turns if turn]
        assert turns[0][-1] == stop and len(turns[0]) == 2, turns
        observed = turns[1][0]
        assert observed in (clean, extra), (turns, clean, extra)
        print(
            f"async_scheduling={async_scheduling} second turn starts {observed}: "
            f"{'clean' if observed == clean else 'stop token fed back'}"
        )
        assert observed == (extra if async_scheduling else clean)
    finally:
        llm.shutdown()
