# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request GPU state held by a standard state layer, end to end.

The dummy-model plugin's `StubPersistentStateForCausalLM` emits a function of
every token the request has fed it and ends each turn with EOS, so a lost,
shared, uninitialized or doubly written state changes the tokens.
"""

import asyncio
import os
from types import SimpleNamespace

import pytest
from vllm_add_dummy_model import my_persistent_state as stub
from vllm_add_dummy_model.my_persistent_state import expected_turns, without_eos

from tests.utils import create_new_process_for_each_test
from vllm import LLM, ModelRegistry, SamplingParams
from vllm.config import ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import StreamingInput
from vllm.inputs import TokensPrompt
from vllm.plugins import load_general_plugins
from vllm.sampling_params import RequestOutputKind
from vllm.utils.vmm_driver import vmm_unavailable_reason
from vllm.v1.engine.async_llm import AsyncLLM

INPUTS = [[10, 20, 30], [40, 50], [60]]
MAX_TOKENS = 16
STUB_ARCH = "StubPersistentStateForCausalLM"
NO_WARMUP_ARCH = "StubNoWarmupPersistentStateForCausalLM"
INITIAL_TOTAL = 5


def _engine_kwargs(arch: str = STUB_ARCH, **hf_overrides) -> dict:
    return dict(
        model="facebook/opt-125m",
        hf_overrides={"architectures": [arch], **hf_overrides},
        load_format="dummy",
        max_model_len=256,
        max_num_seqs=2,
        # The null block plus one 16-byte state page per running request.
        num_gpu_blocks_override=3,
    )


def _register_no_warmup_arch() -> None:
    if NO_WARMUP_ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            NO_WARMUP_ARCH,
            f"vllm_add_dummy_model.my_persistent_state:{NO_WARMUP_ARCH}",
        )


def _in_process_llm(arch: str = STUB_ARCH, **kwargs) -> LLM:
    """An engine in this process, so a test can read the stub's `EVENTS`."""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    _register_no_warmup_arch()
    stub.EVENTS.clear()
    stub.PREFILL_STARTS.clear()
    return LLM(**_engine_kwargs(arch, stub_initial_total=INITIAL_TOTAL), **kwargs)


def _requires_extensible_kv_cache() -> None:
    if (reason := vmm_unavailable_reason()) is not None:
        pytest.skip(f"VMM unavailable: {reason}")


@pytest.mark.parametrize(
    ("arch", "declared"),
    [
        (STUB_ARCH, (True, True, True, False)),
        (NO_WARMUP_ARCH, (False, True, False, False)),
    ],
)
# @spec PSTATE-DECL-004
def test_model_config_exposes_the_state_declarations(arch, declared):
    load_general_plugins()
    _register_no_warmup_arch()
    config = ModelConfig(
        model="facebook/opt-125m", hf_overrides={"architectures": [arch]}
    )
    assert config.has_persistent_state
    assert (
        config.persistent_state_reads_positions,
        config.persistent_state_parks_on_eos,
        config.persistent_state_accepts_synthetic_warmup,
        config.persistent_state_ignores_stopped_row_echoes,
    ) == declared


def _stub_layer_spec(**hf_config):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_vocab_size=lambda: 64,
            hf_config=SimpleNamespace(**hf_config),
            max_model_len=256,
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    layer = stub.StubStateLayer(vllm_config, prefix="state")
    return layer.get_kv_cache_config(vllm_config).spec


@pytest.mark.parametrize("initial_total", [0, INITIAL_TOTAL])
# @spec PSTATE-DECL-002
def test_stub_state_declares_one_initial_value_per_component(initial_total):
    spec = _stub_layer_spec(stub_initial_total=initial_total)
    assert spec.initial_values == (initial_total, 0)
    assert len(spec.initial_values) == len(spec.shapes)


# @spec PSTATE-DECL-010
def test_stub_state_declares_its_tensor_parallel_ownership():
    assert _stub_layer_spec().tp_replicated is True


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.parametrize("initial_total", [0, INITIAL_TOTAL])
# @spec PSTATE-PAGE-001
def test_llm_reuses_slots_from_a_fresh_state(enforce_eager, initial_total):
    """Six prompts through two slots: each produces the tokens of a page that
    holds its initial values, so a reused slot never carries its previous
    owner's sum."""
    llm = LLM(
        **_engine_kwargs(stub_initial_total=initial_total),
        enforce_eager=enforce_eager,
    )
    vocab_size = llm.llm_engine.model_config.get_vocab_size()
    prompts = [[11 * i + 1, 11 * i + 2, 11 * i + 3] for i in range(6)]
    params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    outputs = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
    assert [without_eos(output.outputs[0].token_ids) for output in outputs] == [
        expected_turns([prompt], vocab_size, initial_total=initial_total)[0]
        for prompt in prompts
    ]


async def _stream(
    llm: AsyncLLM, request_id: str, inputs: list[list[int]] = INPUTS
) -> list[list[int]]:
    async def stream_inputs():
        for prompt in inputs:
            # Copies: the output processor extends a request's prompt in place.
            yield StreamingInput(prompt=TokensPrompt(prompt_token_ids=list(prompt)))

    params = SamplingParams(
        temperature=0, max_tokens=MAX_TOKENS, output_kind=RequestOutputKind.DELTA
    )
    turns: list[list[int]] = [[]]
    async for output in llm.generate(stream_inputs(), params, request_id):
        completion = output.outputs[0]
        turns[-1].extend(completion.token_ids)
        if completion.finish_reason is not None and not output.finished:
            turns.append([])
    return [without_eos(turn) for turn in turns if turn]


@pytest.mark.asyncio
async def test_async_llm_streams_keep_their_state_across_inputs():
    """Three streaming sessions share two slots: each keeps its state from
    one input to the next, and the third starts from a fresh slot once one
    of the first two ends."""
    llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**_engine_kwargs()))
    try:
        expected = expected_turns(INPUTS, llm.model_config.get_vocab_size())
        results = await asyncio.wait_for(
            asyncio.gather(*(_stream(llm, f"stream-{i}") for i in range(3))), 60
        )
        assert results == [expected] * 3
    finally:
        llm.shutdown()


@pytest.mark.asyncio
# @spec PSTATE-PAGE-002
async def test_page_is_initialized_once_across_the_turns_of_a_stream():
    """A later turn, which core compacts, continues from its page rather than
    from the initial values."""
    llm = AsyncLLM.from_engine_args(
        AsyncEngineArgs(**_engine_kwargs(stub_initial_total=INITIAL_TOTAL))
    )
    try:
        expected = expected_turns(
            INPUTS, llm.model_config.get_vocab_size(), initial_total=INITIAL_TOTAL
        )
        result = await asyncio.wait_for(_stream(llm, "stream"), 60)
        assert result == expected
    finally:
        llm.shutdown()


@pytest.mark.asyncio
# @spec PSTATE-ASYNC-003
async def test_async_scheduling_applies_each_committed_input_exactly_once():
    """Four sessions share two slots under async scheduling. A step run for a
    row after its stop would add the EOS token to the session's state and
    shift every later turn."""
    llm = AsyncLLM.from_engine_args(
        AsyncEngineArgs(**_engine_kwargs(), async_scheduling=True)
    )
    try:
        assert llm.vllm_config.scheduler_config.async_scheduling
        inputs = [[[i + 1, i + 2], [i + 3], [i + 4, i + 5, i + 6]] for i in range(4)]
        vocab_size = llm.model_config.get_vocab_size()
        results = await asyncio.wait_for(
            asyncio.gather(*(_stream(llm, f"stream-{i}", inputs[i]) for i in range(4))),
            60,
        )
        assert results == [expected_turns(i, vocab_size) for i in inputs]
    finally:
        llm.shutdown()


@pytest.mark.asyncio
# @spec PSTATE-HIST-004
async def test_positions_stay_absolute_across_compacted_turns():
    llm = AsyncLLM.from_engine_args(
        AsyncEngineArgs(**_engine_kwargs(stub_reads_positions=True))
    )
    try:
        expected = expected_turns(
            INPUTS, llm.model_config.get_vocab_size(), reads_positions=True
        )
        result = await asyncio.wait_for(_stream(llm, "stream"), 60)
        assert result == expected
    finally:
        llm.shutdown()


@create_new_process_for_each_test("spawn")
# @spec PSTATE-FAIL-001
def test_model_reported_failure_ends_only_its_request():
    llm = LLM(**_engine_kwargs())
    vocab_size = llm.llm_engine.model_config.get_vocab_size()
    params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    healthy = [1, 2, 3]
    failing = [4, stub.FAIL_TOKEN, 6]
    outputs = llm.generate(
        [{"prompt_token_ids": healthy}, {"prompt_token_ids": failing}], params
    )
    assert outputs[0].outputs[0].finish_reason == "stop"
    assert (
        without_eos(outputs[0].outputs[0].token_ids)
        == expected_turns([healthy], vocab_size)[0]
    )
    assert outputs[1].outputs[0].finish_reason == "error"


@create_new_process_for_each_test("spawn")
# @spec PSTATE-PREP-001, PSTATE-PREP-002, PSTATE-PREP-003
def test_startup_runs_no_request_through_a_model_that_declines_warmup():
    _requires_extensible_kv_cache()
    _in_process_llm(NO_WARMUP_ARCH, enable_extensible_kv_cache=True)
    assert [event for event in stub.EVENTS if event[0] == "add"] == []


@create_new_process_for_each_test("spawn")
# @spec PSTATE-PREP-001
def test_startup_warms_up_a_model_that_accepts_it():
    """The control of the test above: the same startup does run warmup
    requests through a model that accepts them."""
    _requires_extensible_kv_cache()
    _in_process_llm(enable_extensible_kv_cache=True)
    assert [event for event in stub.EVENTS if event[0] == "add"] != []


@create_new_process_for_each_test("spawn")
# @spec PSTATE-PREP-005
def test_warmup_requests_read_initialized_pages():
    _requires_extensible_kv_cache()
    llm = _in_process_llm(enable_extensible_kv_cache=True)
    assert stub.PREFILL_STARTS
    assert set(stub.PREFILL_STARTS) == {INITIAL_TOTAL}
    del llm


@create_new_process_for_each_test("spawn")
# @spec PSTATE-PREP-004, PSTATE-PREP-008
def test_engine_is_ready_after_the_model_state_is_prepared_and_captured():
    llm = _in_process_llm()
    assert stub.EVENTS.count(("prepare_execution_memory",)) == 1
    assert ("capture", False) in stub.EVENTS
    adds = [i for i, event in enumerate(stub.EVENTS) if event[0] == "add"]
    assert all(i > stub.EVENTS.index(("prepare_execution_memory",)) for i in adds)
    assert llm.llm_engine.engine_core.engine_core.is_inference_ready()


@create_new_process_for_each_test("spawn")
# @spec PSTATE-PREP-006
def test_full_wake_after_deep_sleep_recaptures_model_state_graphs_once():
    llm = _in_process_llm(enable_sleep_mode=True)
    ready = len(stub.EVENTS)
    llm.sleep(level=1)
    llm.wake_up(tags=["weights"])
    assert stub.EVENTS[ready:] == []
    llm.wake_up(tags=["kv_cache"])
    assert stub.EVENTS[ready:] == [("clear",), ("capture", False)]


@create_new_process_for_each_test("spawn")
# @spec PSTATE-PREP-009
def test_light_sleep_and_cache_reset_keep_model_state_graphs():
    llm = _in_process_llm(enable_sleep_mode=True)
    ready = len(stub.EVENTS)
    llm.sleep(level=0)
    llm.wake_up(tags=["scheduling"])
    assert llm.reset_prefix_cache()
    assert stub.EVENTS[ready:] == []
