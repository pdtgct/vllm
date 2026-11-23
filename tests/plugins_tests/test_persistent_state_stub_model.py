# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request GPU state held by a standard state layer, end to end.

The dummy-model plugin's `StubPersistentStateForCausalLM` emits a function of
every token the request has fed it and ends each turn with EOS, so a lost,
shared, uninitialized or doubly written state changes the tokens.
"""

import asyncio
import gc
import os
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

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
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.exceptions import StreamAdmissionError

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
        # The null block plus one 16-byte state page per running request. The
        # cap is load-bearing: sized from device memory, a 16-byte page gives
        # a block count in the billions, and the scheduler's block pool holds
        # one host object per block.
        num_gpu_blocks_override=3,
    )


def _shutdown(llm: LLM) -> None:
    """End the engine core this `LLM` started, then drop the object."""
    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()


@contextmanager
def _llm(**kwargs) -> Iterator[LLM]:
    llm = LLM(**kwargs)
    try:
        yield llm
    finally:
        _shutdown(llm)


def _register_no_warmup_arch() -> None:
    if NO_WARMUP_ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            NO_WARMUP_ARCH,
            f"vllm_add_dummy_model.my_persistent_state:{NO_WARMUP_ARCH}",
        )


@contextmanager
def _in_process_llm(arch: str = STUB_ARCH, **kwargs) -> Iterator[LLM]:
    """An engine in this process, so a test can read the stub's `EVENTS`.

    Without an extensible KV cache, which refuses the block-count override
    the stub's 16-byte page needs; the sizing warmup is covered by the worker
    tests.
    """
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    _register_no_warmup_arch()
    stub.EVENTS.clear()
    stub.PREFILL_STARTS.clear()
    engine_kwargs = _engine_kwargs(arch, stub_initial_total=INITIAL_TOTAL)
    with _llm(**engine_kwargs, **kwargs) as llm:
        yield llm


# @spec PSTATE-DECL-004
@pytest.mark.parametrize(
    ("arch", "declared"),
    [
        (STUB_ARCH, (True, True)),
        (NO_WARMUP_ARCH, (False, False)),
    ],
)
def test_model_config_exposes_the_state_declarations(arch, declared):
    load_general_plugins()
    _register_no_warmup_arch()
    config = ModelConfig(
        model="facebook/opt-125m", hf_overrides={"architectures": [arch]}
    )
    assert config.has_persistent_state
    assert (
        config.persistent_state_reads_positions,
        config.persistent_state_accepts_synthetic_warmup,
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


# @spec PSTATE-DECL-002
@pytest.mark.parametrize("initial_total", [0, INITIAL_TOTAL])
def test_stub_state_declares_one_initial_value_per_component(initial_total):
    spec = _stub_layer_spec(stub_initial_total=initial_total)
    assert spec.initial_values == (initial_total, stub.TURN_LEN - 1)
    assert len(spec.initial_values) == len(spec.shapes)


# @spec PSTATE-PAGE-001 PSTATE-PAGE-005
@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.parametrize("initial_total", [0, INITIAL_TOTAL])
def test_llm_reuses_slots_from_a_fresh_state(
    enforce_eager, initial_total, async_scheduling
):
    """Six prompts through two slots: each produces the tokens of a page that
    holds its initial values, so a reused slot never carries its previous
    owner's sum. Under async scheduling the step after a request's stop is
    enqueued before core frees its page, so the old owner's work is still on
    the runner stream when the new owner initializes the page."""
    with _llm(
        **_engine_kwargs(stub_initial_total=initial_total),
        enforce_eager=enforce_eager,
        async_scheduling=async_scheduling,
    ) as llm:
        vocab_size = llm.llm_engine.model_config.get_vocab_size()
        prompts = [[11 * i + 1, 11 * i + 2, 11 * i + 3] for i in range(6)]
        params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
        outputs = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
    assert [without_eos(output.outputs[0].token_ids) for output in outputs] == [
        expected_turns([prompt], vocab_size, initial_total=initial_total)[0]
        for prompt in prompts
    ]


WATCHDOG = 120
"""Seconds before a streaming test counts as hung."""


async def _stream(
    llm: AsyncLLM,
    request_id: str,
    inputs: list[list[int]] = INPUTS,
    *,
    hold: asyncio.Event | None = None,
    turns_done: asyncio.Event | None = None,
) -> list[list[int]]:
    """Stream `inputs` and return each turn's tokens. With `hold`, the input
    iterator stays open after its last input until `hold` is set, so the
    stream stays held, parked; `turns_done` is set once every input's turn has
    ended."""

    async def stream_inputs():
        for prompt in inputs:
            # Copies: the output processor extends a request's prompt in place.
            yield StreamingInput(prompt=TokensPrompt(prompt_token_ids=list(prompt)))
        if hold is not None:
            await hold.wait()

    params = SamplingParams(
        temperature=0, max_tokens=MAX_TOKENS, output_kind=RequestOutputKind.DELTA
    )
    turns: list[list[int]] = [[]]
    async for output in llm.generate(stream_inputs(), params, request_id):
        completion = output.outputs[0]
        turns[-1].extend(completion.token_ids)
        if completion.finish_reason is not None and not output.finished:
            turns.append([])
            if turns_done is not None and len(turns) == len(inputs) + 1:
                turns_done.set()
    return [without_eos(turn) for turn in turns if turn]


async def _stream_in_waves(
    llm: AsyncLLM, inputs: list[list[list[int]]]
) -> list[list[list[int]]]:
    """Stream each entry of `inputs`, at most `max_num_seqs` (two) at a time.
    While two streams are held, a third is refused with `active_limit`; once
    the oldest ends, the next stream takes its row."""
    results: list = [None] * len(inputs)
    pending = list(range(len(inputs)))
    held: dict[int, tuple[asyncio.Future, asyncio.Event]] = {}
    while pending or held:
        while pending and len(held) < 2:
            index = pending.pop(0)
            hold, turns_done = asyncio.Event(), asyncio.Event()
            task = asyncio.ensure_future(
                _stream(
                    llm,
                    f"stream-{index}",
                    inputs[index],
                    hold=hold,
                    turns_done=turns_done,
                )
            )
            waiter = asyncio.ensure_future(turns_done.wait())
            await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
            if not waiter.done():
                waiter.cancel()
                task.result()
                raise AssertionError(f"stream-{index} ended before its turns")
            held[index] = (task, hold)
        if pending:
            with pytest.raises(StreamAdmissionError) as excinfo:
                await _stream(llm, "refused", inputs[pending[0]])
            assert excinfo.value.reason == "active_limit"
        oldest = min(held)
        task, hold = held.pop(oldest)
        hold.set()
        results[oldest] = await task
    return results


@pytest.mark.asyncio
async def test_async_llm_streams_keep_their_state_across_inputs():
    """Three streaming sessions through two rows: each keeps its state from
    one input to the next, the third is refused while both rows are held, and
    it starts from a fresh page once one of the first two ends."""
    llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**_engine_kwargs()))
    try:
        expected = expected_turns(INPUTS, llm.model_config.get_vocab_size())
        results = await asyncio.wait_for(_stream_in_waves(llm, [INPUTS] * 3), WATCHDOG)
        assert results == [expected] * 3
    finally:
        llm.shutdown()


# @spec PSTATE-PAGE-001
@pytest.mark.asyncio
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
        result = await asyncio.wait_for(_stream(llm, "stream"), WATCHDOG)
        assert result == expected
    finally:
        llm.shutdown()


# @spec PSTATE-ASYNC-003
@pytest.mark.asyncio
async def test_async_scheduling_applies_each_committed_input_exactly_once():
    """Four sessions through two rows under async scheduling, in waves. A step
    run for a row after its stop would add the EOS token to the session's
    state and shift every later turn."""
    llm = AsyncLLM.from_engine_args(
        AsyncEngineArgs(**_engine_kwargs(), async_scheduling=True)
    )
    try:
        assert llm.vllm_config.scheduler_config.async_scheduling
        inputs = [[[i + 1, i + 2], [i + 3], [i + 4, i + 5, i + 6]] for i in range(4)]
        vocab_size = llm.model_config.get_vocab_size()
        results = await asyncio.wait_for(_stream_in_waves(llm, inputs), WATCHDOG)
        assert results == [expected_turns(i, vocab_size) for i in inputs]
    finally:
        llm.shutdown()


# @spec PSTATE-HIST-004
@pytest.mark.asyncio
async def test_positions_stay_absolute_across_compacted_turns():
    llm = AsyncLLM.from_engine_args(
        AsyncEngineArgs(**_engine_kwargs(stub_reads_positions=True))
    )
    try:
        expected = expected_turns(
            INPUTS, llm.model_config.get_vocab_size(), reads_positions=True
        )
        result = await asyncio.wait_for(_stream(llm, "stream"), WATCHDOG)
        assert result == expected
    finally:
        llm.shutdown()


# @spec PSTATE-FAIL-001
@create_new_process_for_each_test("spawn")
def test_model_reported_failure_ends_only_its_request():
    with _llm(**_engine_kwargs()) as llm:
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


# @spec PSTATE-PREP-001 PSTATE-PREP-003
@create_new_process_for_each_test("spawn")
def test_startup_runs_no_request_through_a_model_that_declines_warmup():
    with _in_process_llm(NO_WARMUP_ARCH):
        assert [event for event in stub.EVENTS if event[0] == "add"] == []


# @spec PSTATE-PREP-001
@create_new_process_for_each_test("spawn")
def test_startup_warms_up_a_model_that_accepts_it():
    """The control of the test above: the same startup does run warmup
    requests through a model that accepts them."""
    with _in_process_llm():
        assert [event for event in stub.EVENTS if event[0] == "add"] != []


# @spec PSTATE-PREP-005
@create_new_process_for_each_test("spawn")
def test_warmup_requests_read_initialized_pages():
    with _in_process_llm():
        assert stub.PREFILL_STARTS
        assert set(stub.PREFILL_STARTS) == {INITIAL_TOTAL}


# @spec PSTATE-PREP-004 PSTATE-PREP-008
@create_new_process_for_each_test("spawn")
def test_engine_is_ready_after_the_model_state_is_prepared_and_captured():
    """The engine core exists, and so can report ready, only once its startup
    has prepared the model state and captured its serving graphs."""
    construct = EngineCore.__init__

    def recording_construct(self, *args, **kwargs):
        construct(self, *args, **kwargs)
        stub.EVENTS.append(("ready",))

    with (
        patch.object(EngineCore, "__init__", recording_construct),
        _in_process_llm() as llm,
    ):
        events = list(stub.EVENTS)
        assert events.count(("prepare_execution_memory",)) == 1
        assert events.count(("ready",)) == 1
        ready = events.index(("ready",))
        assert events.index(("prepare_execution_memory",)) < ready
        assert events.index(("capture", False)) < ready
        adds = [i for i, event in enumerate(events) if event[0] == "add"]
        assert all(i > events.index(("prepare_execution_memory",)) for i in adds)
        assert llm.llm_engine.engine_core.engine_core.is_inference_ready()


# @spec PSTATE-PREP-006
@create_new_process_for_each_test("spawn")
def test_full_wake_after_deep_sleep_recaptures_model_state_graphs_once():
    with _in_process_llm(enable_sleep_mode=True) as llm:
        ready = len(stub.EVENTS)
        llm.sleep(level=1)
        llm.wake_up(tags=["weights"])
        assert stub.EVENTS[ready:] == []
        llm.wake_up(tags=["kv_cache"])
        assert stub.EVENTS[ready:] == [("clear",), ("capture", False)]
        # A wake of an awake engine remaps nothing.
        llm.wake_up()
        assert stub.EVENTS[ready:] == [("clear",), ("capture", False)]


# @spec PSTATE-PREP-006
@create_new_process_for_each_test("spawn")
def test_light_sleep_and_cache_reset_keep_model_state_graphs():
    with _in_process_llm(enable_sleep_mode=True) as llm:
        ready = len(stub.EVENTS)
        llm.sleep(level=0)
        llm.wake_up(tags=["scheduling"])
        assert llm.reset_prefix_cache()
        llm.sleep(level=0)
        llm.wake_up()
        assert stub.EVENTS[ready:] == []
