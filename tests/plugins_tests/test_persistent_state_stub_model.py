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
