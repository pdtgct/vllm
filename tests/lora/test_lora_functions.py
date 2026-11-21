# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Script to test add_lora, remove_lora, pin_lora, list_loras functions."""

import asyncio
from collections.abc import AsyncGenerator

import pytest

from vllm import LLM, RequestOutput, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.engine.protocol import StreamingInput
from vllm.entrypoints.launchers.api_server.entry import (
    build_async_engine_client_from_engine_args,
)
from vllm.exceptions import VLLMValidationError
from vllm.lora.request import LoRARequest
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.llm_engine import LLMEngine

MODEL_PATH = "Qwen/Qwen3-0.6B"
LORA_MODULE_PATH = "vllm-project/self_cognition_Alice"
LORA_RANK = 8


def make_lora_request(lora_id: int):
    return LoRARequest(
        lora_name=f"{lora_id}", lora_int_id=lora_id, lora_path=LORA_MODULE_PATH
    )


def test_lora_functions_sync():
    max_loras = 4
    # Create engine in eager-mode. Due to high max_loras, the CI can
    # OOM during cuda-graph capture.
    engine_args = EngineArgs(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=max_loras,
        max_lora_rank=LORA_RANK,
        max_model_len=128,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )

    llm = LLMEngine.from_engine_args(engine_args)

    def run_check(fn, args, expected: list):
        fn(args)
        assert set(llm.list_loras()) == set(expected)

    run_check(llm.add_lora, make_lora_request(1), [1])
    run_check(llm.add_lora, make_lora_request(2), [1, 2])

    # Pin LoRA 1 and test that it is never removed on subsequent adds.
    run_check(llm.pin_lora, 1, [1, 2])
    run_check(llm.add_lora, make_lora_request(3), [1, 2, 3])
    run_check(llm.add_lora, make_lora_request(4), [1, 2, 3, 4])
    run_check(llm.add_lora, make_lora_request(5), [1, 5, 3, 4])
    run_check(llm.add_lora, make_lora_request(6), [1, 5, 6, 4])
    run_check(llm.add_lora, make_lora_request(7), [1, 5, 6, 7])
    run_check(llm.add_lora, make_lora_request(8), [1, 8, 6, 7])
    run_check(llm.add_lora, make_lora_request(9), [1, 8, 9, 7])
    run_check(llm.add_lora, make_lora_request(10), [1, 8, 9, 10])

    # Remove LoRA 1 and continue adding.
    run_check(llm.remove_lora, 1, [8, 9, 10])
    run_check(llm.add_lora, make_lora_request(11), [8, 9, 10, 11])
    run_check(llm.add_lora, make_lora_request(12), [12, 9, 10, 11])
    run_check(llm.add_lora, make_lora_request(13), [12, 13, 10, 11])

    # Remove all LoRAs.
    run_check(llm.remove_lora, 13, [12, 10, 11])
    run_check(llm.remove_lora, 12, [10, 11])
    run_check(llm.remove_lora, 11, [10])
    run_check(llm.remove_lora, 10, [])

    # An id in use by another adapter is rejected instead of silently serving
    # that adapter's weights; removing the adapter frees the id.
    run_check(llm.add_lora, make_lora_request(1), [1])
    other = LoRARequest("other", lora_int_id=1, lora_path=LORA_MODULE_PATH)
    with pytest.raises(VLLMValidationError, match="already used"):
        llm.add_lora(other)
    with pytest.raises(VLLMValidationError, match="already used"):
        llm.add_request("req", "Hi", SamplingParams(max_tokens=1), lora_request=other)
    run_check(llm.remove_lora, 1, [])
    run_check(llm.add_lora, other, [1])


@pytest.mark.asyncio
async def test_lora_functions_async():
    max_loras = 4
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=max_loras,
        max_lora_rank=LORA_RANK,
        max_model_len=128,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )

    async with build_async_engine_client_from_engine_args(engine_args) as llm:
        # The builder always yields an AsyncLLM; EngineClient only declares add_lora.
        assert isinstance(llm, AsyncLLM)

        async def run_check(fn, args, expected: list):
            await fn(args)
            assert set(await llm.list_loras()) == set(expected)

        await run_check(llm.add_lora, make_lora_request(1), [1])
        await run_check(llm.add_lora, make_lora_request(2), [1, 2])

        # Pin LoRA 1 and test that it is never removed on subsequent adds.
        await run_check(llm.pin_lora, 1, [1, 2])
        await run_check(llm.add_lora, make_lora_request(3), [1, 2, 3])
        await run_check(llm.add_lora, make_lora_request(4), [1, 2, 3, 4])
        await run_check(llm.add_lora, make_lora_request(5), [1, 5, 3, 4])
        await run_check(llm.add_lora, make_lora_request(6), [1, 5, 6, 4])
        await run_check(llm.add_lora, make_lora_request(7), [1, 5, 6, 7])
        await run_check(llm.add_lora, make_lora_request(8), [1, 8, 6, 7])
        await run_check(llm.add_lora, make_lora_request(9), [1, 8, 9, 7])
        await run_check(llm.add_lora, make_lora_request(10), [1, 8, 9, 10])

        # Remove LoRA 1 and continue adding.
        await run_check(llm.remove_lora, 1, [8, 9, 10])
        await run_check(llm.add_lora, make_lora_request(11), [8, 9, 10, 11])
        await run_check(llm.add_lora, make_lora_request(12), [12, 9, 10, 11])
        await run_check(llm.add_lora, make_lora_request(13), [12, 13, 10, 11])

        # Remove all LoRAs
        await run_check(llm.remove_lora, 13, [12, 10, 11])
        await run_check(llm.remove_lora, 12, [10, 11])
        await run_check(llm.remove_lora, 11, [10])
        await run_check(llm.remove_lora, 10, [])


def test_same_lora_name_with_different_ids(
    qwen3_meowing_lora_files, qwen3_woofing_lora_files
):
    """An adapter unloaded and loaded again under the same name gets a new id
    while requests for the old id may still be running; each request in the
    shared batch must use its own adapter."""
    llm = LLM(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=LORA_RANK,
        max_model_len=128,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )
    messages = [
        {"role": "system", "content": "Follow the instructions to make animal noises"},
        {"role": "user", "content": "Make your favorite animal noise."},
    ]
    outputs = llm.chat(
        [messages, messages],
        SamplingParams(temperature=0, max_tokens=10),
        chat_template_kwargs={"enable_thinking": False},
        lora_request=[
            LoRARequest("pet", 1, qwen3_meowing_lora_files),
            LoRARequest("pet", 2, qwen3_woofing_lora_files),
        ],
    )
    assert "Meow Meow Meow" in outputs[0].outputs[0].text
    assert "Woof Woof Woof" in outputs[1].outputs[0].text


@pytest.mark.parametrize(
    ("specialize_active_lora", "max_loras"), [(True, 1), (True, 4), (False, 1)]
)
def test_lora_cuda_graphs_with_base_requests(specialize_active_lora, max_loras):
    """CUDA graphs must apply an adapter whether its request runs alone or
    shares the batch with a base model request."""
    engine_args = EngineArgs(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=max_loras,
        max_lora_rank=LORA_RANK,
        max_model_len=128,
        gpu_memory_utilization=0.8,
        specialize_active_lora=specialize_active_lora,
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4]},
    )
    llm = LLMEngine.from_engine_args(engine_args)
    prompt = llm.get_tokenizer().apply_chat_template(
        [{"role": "user", "content": "Who are you?"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=16)

    def run(lora_by_request_id: dict[str, LoRARequest | None]) -> dict[str, str]:
        for request_id, lora_request in lora_by_request_id.items():
            llm.add_request(
                request_id, prompt, sampling_params, lora_request=lora_request
            )
        texts = {}
        while llm.has_unfinished_requests():
            for output in llm.step():
                assert isinstance(output, RequestOutput)
                if output.finished:
                    texts[output.request_id] = output.outputs[0].text
        return texts

    assert "Alice" in run({"alone": make_lora_request(1)})["alone"]
    mixed = run({"lora": make_lora_request(1), "base": None})
    assert "Alice" in mixed["lora"]
    assert "Alice" not in mixed["base"]


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_dummy_batch_keeps_loras(enforce_eager):
    """Idle data parallel ranks run dummy batches between steps. Those must
    not unload adapters, including ones whose ids match the dummy LoRA ids."""
    engine_args = EngineArgs(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=2,
        max_cpu_loras=3,
        max_lora_rank=LORA_RANK,
        max_model_len=128,
        max_num_seqs=8,
        gpu_memory_utilization=0.8,
        enforce_eager=enforce_eager,
    )
    llm = LLMEngine.from_engine_args(engine_args)

    llm.add_lora(make_lora_request(1))
    llm.add_lora(make_lora_request(3))
    assert set(llm.list_loras()) == {1, 3}

    for _ in range(3):
        llm.engine_core.execute_dummy_batch()
    assert set(llm.list_loras()) == {1, 3}

    llm.add_request(
        "req-1",
        "Who are you?",
        SamplingParams(max_tokens=4),
        lora_request=make_lora_request(1),
    )
    while llm.has_unfinished_requests():
        llm.step()
    assert set(llm.list_loras()) == {1, 3}


@pytest.mark.asyncio
async def test_streaming_session_held_back_by_max_loras_keeps_its_slot(
    qwen3_lora_files,
):
    """A streaming session given its next input keeps its model runner slot
    while max_loras holds it back, so a new request must wait for that slot.
    Admitting the new request made the model runner fail with "No free indices"
    and killed the engine."""
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=LORA_RANK,
        max_model_len=256,
        max_num_seqs=2,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )
    turn_params = SamplingParams(
        temperature=0, max_tokens=4, output_kind=RequestOutputKind.DELTA
    )
    # Long enough that the busy request still runs when the others arrive.
    busy_params = SamplingParams(temperature=0, max_tokens=200, ignore_eos=True)
    next_input = asyncio.Event()

    async def session_inputs() -> AsyncGenerator[StreamingInput, None]:
        yield StreamingInput(prompt="Hello, my name is")
        await next_input.wait()
        yield StreamingInput(prompt=" and I like")

    async def drain(
        outputs: AsyncGenerator[RequestOutput, None],
    ) -> list[RequestOutput]:
        return [output async for output in outputs]

    async with build_async_engine_client_from_engine_args(engine_args) as llm:
        session = llm.generate(
            session_inputs(),
            turn_params,
            "session",
            lora_request=LoRARequest("2", 2, qwen3_lora_files),
        )
        # After its first turn the session waits for input, holding its slot.
        output = await asyncio.wait_for(anext(session), timeout=60)
        while not (output.outputs and output.outputs[0].finish_reason):
            output = await asyncio.wait_for(anext(session), timeout=60)

        # The busy request's adapter fills max_loras, holding back the session.
        busy = llm.generate(
            "Count to two hundred:",
            busy_params,
            "busy",
            lora_request=LoRARequest("1", 1, qwen3_lora_files),
        )
        await asyncio.wait_for(anext(busy), timeout=60)
        next_input.set()
        new = llm.generate("Hello", turn_params, "new")
        outputs = await asyncio.wait_for(
            asyncio.gather(drain(session), drain(busy), drain(new)), timeout=120
        )

    assert all(request_outputs[-1].finished for request_outputs in outputs)
