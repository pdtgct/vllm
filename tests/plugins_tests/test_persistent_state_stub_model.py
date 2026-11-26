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
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm_add_dummy_model import my_persistent_state as stub
from vllm_add_dummy_model.my_persistent_state import expected_turns, without_eos

import vllm.envs as envs
import vllm.v1.worker.gpu.cudagraph_utils as cudagraph_utils
from tests.utils import create_new_process_for_each_test
from vllm import LLM, ModelRegistry, SamplingParams
from vllm.config import ModelConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import StreamingInput
from vllm.inputs import TokensPrompt
from vllm.plugins import load_general_plugins
from vllm.sampling_params import RequestOutputKind
from vllm.utils.vmm_driver import vmm_unavailable_reason
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core_client import InprocClient
from vllm.v1.engine.exceptions import StreamAdmissionError
from vllm.v1.request import RequestStatus
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

INPUTS = [[10, 20, 30], [40, 50], [60]]
MAX_TOKENS = 16
STUB_ARCH = "StubPersistentStateForCausalLM"
NO_WARMUP_ARCH = "StubNoWarmupPersistentStateForCausalLM"
INITIAL_TOTAL = 5

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


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
    _clear_stub_records()
    engine_kwargs = _engine_kwargs(arch, stub_initial_total=INITIAL_TOTAL)
    with _llm(**{**engine_kwargs, **kwargs}) as llm:
        yield llm


PAGE_ELEMS = 1 << 16
"""Elements per state component in a page of an engine that sizes its own KV
cache: a 1 MiB page. A 16-byte page would give a block count in the billions,
and the scheduler's block pool holds one host object per block."""

SIZED_MEMORY_UTILIZATION = 0.3
"""The share of device memory the sized engines may use, so a page count stays
in the thousands on any device."""


@contextmanager
def _sized_llm(**kwargs) -> Iterator[LLM]:
    """An engine in this process whose KV cache core sizes from device memory:
    no block-count override and no `kv_cache_memory_bytes`, so with the
    extensible KV cache on, startup profiles, warms up, captures and then
    extends it as it does in serving. Needs a CUDA device."""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    _clear_stub_records()
    engine_kwargs = _engine_kwargs(
        stub_initial_total=INITIAL_TOTAL, stub_page_elems=PAGE_ELEMS
    )
    del engine_kwargs["num_gpu_blocks_override"]
    engine_kwargs["gpu_memory_utilization"] = SIZED_MEMORY_UTILIZATION
    with _llm(**{**engine_kwargs, **kwargs}) as llm:
        yield llm


def _skip_without_vmm() -> None:
    if (reason := vmm_unavailable_reason()) is not None:
        pytest.skip(f"VMM unavailable: {reason}")


def _clear_stub_records() -> None:
    for records in (
        stub.EVENTS,
        stub.PREFILL_STARTS,
        stub.PAGES,
        stub.PROMPTS,
        stub.LAYOUTS_AT_CAPTURE,
        stub.CAPTURES,
        stub.FORWARDS,
    ):
        records.clear()


def _runner(llm: LLM):
    client = llm.llm_engine.engine_core
    assert isinstance(client, InprocClient)
    return client.engine_core.model_executor.driver_worker.worker.model_runner


# @spec PSTATE-DECL-004
@pytest.mark.parametrize(
    ("arch", "declared"),
    [
        (STUB_ARCH, (True, True, frozenset({"eos"}))),
        (NO_WARMUP_ARCH, (False, False, frozenset({"eos"}))),
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
        config.persistent_state_resumable_stops,
    ) == declared


class _DuplexStub(stub.StubPersistentStateForCausalLM):
    persistent_state_resumable_stops = frozenset({"eos", "max_tokens"})


class _UnknownStopStub(stub.StubPersistentStateForCausalLM):
    persistent_state_resumable_stops = frozenset({"eos", "stop_token"})


def _config_for(model_cls) -> ModelConfig:
    arch = model_cls.__name__
    ModelRegistry.register_model(arch, model_cls)
    return ModelConfig(
        model="facebook/opt-125m", hf_overrides={"architectures": [arch]}
    )


# @spec PSTATE-DECL-004
def test_model_config_exposes_declared_resumable_stops():
    load_general_plugins()
    config = _config_for(_DuplexStub)
    assert config.persistent_state_resumable_stops == frozenset({"eos", "max_tokens"})


# @spec PSTATE-DECL-004
def test_configuration_refuses_an_unknown_resumable_stop():
    load_general_plugins()
    with pytest.raises(ValueError, match="persistent_state_resumable_stops"):
        _config_for(_UnknownStopStub)


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
    park_before: frozenset[int] = frozenset(),
) -> list[list[int]]:
    """Stream `inputs` and return each turn's tokens. With `hold`, the input
    iterator stays open after its last input until `hold` is set, so the
    stream stays held, parked; `turns_done` is set once every input's turn has
    ended. The input at each index in `park_before` waits until every earlier
    turn has ended, so the stream parks before it."""
    num_ended = [0]
    turn_ended = asyncio.Event()

    async def stream_inputs():
        for index, prompt in enumerate(inputs):
            while index in park_before and num_ended[0] < index:
                turn_ended.clear()
                await turn_ended.wait()
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
            num_ended[0] += 1
            turn_ended.set()
            if turns_done is not None and len(turns) == len(inputs) + 1:
                turns_done.set()
    return [without_eos(turn) for turn in turns if turn]


async def _stream_in_waves(
    llm: AsyncLLM,
    inputs: list[list[list[int]]],
    park_before: frozenset[int] = frozenset(),
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
                    park_before=park_before,
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


# Runner CUDA graphs over state pages


# @spec PSTATE-PREP-009
@requires_cuda
@create_new_process_for_each_test("spawn")
def test_model_state_holds_the_bound_state_layout_when_it_captures():
    """The serving capture sees the layout of the views the state layer reads,
    one page per block id."""
    with _in_process_llm() as llm:
        runner = _runner(llm)
        layouts = runner.model_state.state_group_layouts
        [layer] = [
            layer
            for layer in runner.compilation_config.static_forward_context.values()
            if isinstance(layer, stub.StubStateLayer)
        ]
        serving = [d for profile_only, d in stub.LAYOUTS_AT_CAPTURE if not profile_only]
        # Shutdown releases the layer's views, so check them while bound.
        assert serving[-1] == stub.describe_layouts(layouts)
        [layout] = layouts
        components = layout.layers[layer.prefix]
        assert len(components) == len(layer.kv_cache) == 2
        for component, tensor in zip(components, layer.kv_cache, strict=True):
            assert component.tensor is tensor
            assert (component.dtype, component.page_shape) == (torch.int64, (1,))
            assert component.page(2).data_ptr() == (
                tensor.data_ptr() + 2 * component.page_stride * tensor.element_size()
            )


PAGE_INPUTS = [[10, 20, 30], [40, 50], [60], [70, 80]]
"""A first turn, a park before the second input, and two inputs sent back to
back, so the fourth normally queues behind the third and folds into it."""


# @spec PSTATE-ACC-008 PSTATE-PRES-009
@requires_cuda
@pytest.mark.asyncio
async def test_stream_keeps_its_page_and_address_until_its_terminal_finish(
    monkeypatch,
):
    """Three streams through two pages: every turn of a stream, after a park
    and through compaction, runs on the block id and device addresses of its
    first turn; the two held streams use different pages, and the third gets
    a page only after a stream ends."""
    # The worker's page records come back through a function sent by RPC.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**_engine_kwargs()))
    try:
        expected = expected_turns(PAGE_INPUTS, llm.model_config.get_vocab_size())
        results = await asyncio.wait_for(
            _stream_in_waves(llm, [PAGE_INPUTS] * 3, park_before=frozenset({1})),
            WATCHDOG,
        )

        def recorded_pages(worker):
            from vllm_add_dummy_model import my_persistent_state

            return dict(my_persistent_state.PAGES)

        [pages] = await llm.engine_core.collective_rpc_async(recorded_pages)
    finally:
        llm.shutdown()
    assert results == [expected] * 3
    per_stream = []
    for index in range(3):
        [entries] = [v for k, v in pages.items() if k.startswith(f"stream-{index}")]
        # One entry per turn; RPC returns each tuple as a list.
        assert len(entries) == len(PAGE_INPUTS)
        offsets = [offset for _, _, offset in entries]
        # Every later turn was compacted, and the page stayed.
        assert offsets[0] == 0 and all(o > 0 for o in offsets[1:]), offsets
        pages_held = {
            (block_id, tuple(addresses)) for block_id, addresses, _ in entries
        }
        assert len(pages_held) == 1, entries
        per_stream.append(pages_held.pop())
    first, second, third = per_stream
    assert first != second
    # The third opened once the first ended, on the page the first released.
    assert third == first


# An engine core driven in this process, one step at a time


def _engine_core(llm: LLM) -> EngineCore:
    client = llm.llm_engine.engine_core
    assert isinstance(client, InprocClient)
    return client.engine_core


def _core_input(
    request_id: str,
    prompt: list[int],
    *,
    resumable: bool = True,
    continuation: bool = False,
) -> EngineCoreRequest:
    """A greedy input as the engine core receives it: an opener, a
    continuation, a close (`resumable=False, continuation=True`) or, with
    neither flag, a single request."""
    params = SamplingParams(
        temperature=0, max_tokens=MAX_TOKENS, output_kind=RequestOutputKind.DELTA
    )
    params.update_from_generation_config({}, eos_token_id=stub.EOS_TOKEN_ID)
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=list(prompt),
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        resumable=resumable,
        continuation=continuation,
    )


def _close(request_id: str) -> EngineCoreRequest:
    """The close as `AsyncLLM` sends it: a one-token `[0]` final request."""
    return _core_input(request_id, [0], resumable=False, continuation=True)


class _CoreDriver:
    """Steps an in-process engine core and collects each request's tokens per
    turn and the request ids each nonempty step scheduled.

    Streams are routed as `AsyncLLM` routes them: `admit` admits the stream
    with no prompt, and every input, the first and the close included, is a
    continuation of it, so the first input resumes the admitted stream and
    records its row reservation. Because the test adds each input between
    steps, the order of every input relative to every stop is fixed: an
    input added while its stream runs queues and folds at the turn's stop,
    and one added while the stream is parked resumes it. `InprocClient`
    refuses streaming inputs for a state model because the engine-core
    process owns the stream guards and terminal outputs; these tests read
    neither.
    """

    def __init__(self, core: EngineCore) -> None:
        self.core = core
        self.scheduler = core.scheduler
        self.turns: dict[str, list[list[int]]] = {}
        self.batches: list[frozenset[str]] = []
        self.num_steps = 0
        schedule = self.scheduler.schedule

        def recording_schedule(*args, **kwargs):
            output = schedule(*args, **kwargs)
            if output.num_scheduled_tokens:
                self.batches.append(frozenset(output.num_scheduled_tokens))
            return output

        self.scheduler.schedule = recording_schedule

    def add(self, request: EngineCoreRequest) -> None:
        req, wave = self.core.preprocess_add_request(request)
        self.core.add_request(req, wave)

    def admit(self, request_id: str) -> None:
        assert self.core.admit_stream(_core_input(request_id, [])) is None

    def send(self, request_id: str, prompt: list[int]) -> None:
        """The next input of the admitted stream `request_id`."""
        self.add(_core_input(request_id, prompt, continuation=True))

    def step(self) -> None:
        outputs, model_executed = self.core.step_fn()
        self.core.post_step(model_executed)
        self.num_steps += 1
        for engine_core_outputs in (outputs or {}).values():
            for output in engine_core_outputs.outputs:
                turns = self.turns.setdefault(output.request_id, [[]])
                turns[-1].extend(output.new_token_ids)
                if output.finish_reason is not None:
                    turns.append([])

    def run_until(self, done: Callable[[], bool], max_steps: int = 200) -> None:
        for _ in range(max_steps):
            if done():
                return
            self.step()
        raise AssertionError(f"not done after {max_steps} steps")

    def run_until_idle(self) -> None:
        self.run_until(lambda: not self.scheduler.has_requests())

    def num_turns_ended(self, request_id: str) -> int:
        return len(self.turns.get(request_id, [[]])) - 1

    def tokens(self, request_id: str) -> list[list[int]]:
        """Each ended turn's tokens, EOS excluded."""
        return [without_eos(turn) for turn in self.turns[request_id][:-1]]


FOLD_INPUTS = [[10, 20, 30], [40, 50], [60], [70, 80]]
"""A first turn that parks, a second input that resumes it, and two inputs
that reach core while the stream runs and so fold."""


# @spec PSTATE-PRES-009 PSTATE-ACC-008
@requires_cuda
@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("async_scheduling", [False, True])
def test_stream_keeps_its_page_and_address_through_forced_folds(async_scheduling):
    """A stream parks after its first turn, resumes, and then folds twice and
    ends on a close it folds the same way: core adds each later input while
    the previous turn runs, so the stream never parks between those turns.
    Every turn, compacted after the first, runs on the block id and device
    addresses of the first turn; a request running beside the stream gets
    another page, and the stream's page goes to another request only after
    the stream's terminal finish."""
    with _in_process_llm(async_scheduling=async_scheduling) as llm:
        vocab_size = llm.llm_engine.model_config.get_vocab_size()
        driver = _CoreDriver(_engine_core(llm))
        scheduler = driver.scheduler
        stub.PAGES.clear()

        pool = scheduler.kv_cache_manager.block_pool

        def assert_page_held(block_id: int) -> None:
            """The stream still references its page, and the pool cannot hand
            it to another request."""
            assert scheduler.kv_cache_manager.get_block_ids("s") == ([block_id],)
            assert pool.blocks[block_id].ref_cnt > 0
            free = pool.free_block_queue.get_all_free_blocks()
            assert block_id not in {block.block_id for block in free}

        driver.admit("s")
        stream = scheduler.requests["s"]
        driver.send("s", FOLD_INPUTS[0])
        assert stream in scheduler.resumed_streaming_sessions
        driver.run_until(lambda: driver.num_turns_ended("s") == 1)
        assert stream.status == RequestStatus.WAITING_FOR_STREAMING_REQ
        [[page]] = scheduler.kv_cache_manager.get_block_ids("s")
        assert_page_held(page)

        driver.send("s", FOLD_INPUTS[1])
        assert stream.status == RequestStatus.WAITING
        for prompt in FOLD_INPUTS[2:]:
            driver.send("s", prompt)
        assert stream.streaming_queue is not None
        assert len(stream.streaming_queue) == 2
        driver.add(_core_input("beside", [1, 2, 3], resumable=False))

        # Each stop folds the next queued input: the stream is waiting to be
        # scheduled again, not parked, and the queue is one shorter.
        for turn, num_queued in ((2, 1), (3, 0)):
            driver.run_until(lambda turn=turn: driver.num_turns_ended("s") == turn)
            assert stream.status != RequestStatus.WAITING_FOR_STREAMING_REQ
            assert scheduler.num_waiting_for_streaming_input == 0
            assert len(stream.streaming_queue) == num_queued
            assert_page_held(page)
        driver.add(_close("s"))
        assert len(stream.streaming_queue) == 1
        driver.run_until(lambda: "s" not in scheduler.requests)
        assert stream.status == RequestStatus.FINISHED_STOPPED
        assert driver.num_turns_ended("s") == len(FOLD_INPUTS)
        driver.run_until_idle()
        held = {req_id: list(entries) for req_id, entries in stub.PAGES.items()}

        # Two requests take both pages, so one of them takes the stream's.
        driver.add(_core_input("after-0", [4, 5], resumable=False))
        driver.add(_core_input("after-1", [6, 7], resumable=False))
        driver.run_until_idle()
        after = {stub.PAGES[r][0][0] for r in ("after-0", "after-1")}

    assert driver.tokens("s") == expected_turns(
        FOLD_INPUTS, vocab_size, initial_total=INITIAL_TOTAL
    )
    assert driver.tokens("beside") == expected_turns(
        [[1, 2, 3]], vocab_size, initial_total=INITIAL_TOTAL
    )
    entries = held["s"]
    # One entry per turn: the park, the resume and both folds re-add the row.
    assert len(entries) == len(FOLD_INPUTS)
    offsets = [offset for _, _, offset in entries]
    assert offsets[0] == 0 and all(o > 0 for o in offsets[1:]), offsets
    assert len({(block_id, addresses) for block_id, addresses, _ in entries}) == 1
    assert entries[0][0] == page
    assert held["beside"][0][0] != page
    assert page in after


GRAPH_MODES = ("PIECEWISE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE")
GRAPH_PROMPTS = [[1, 2, 3], [40, 50]]


def _page_values(runner, block_ids) -> dict[int, tuple[int, ...]]:
    [layout] = runner.model_state.state_group_layouts
    [components] = layout.layers.values()
    return {
        block_id: tuple(int(c.page(block_id).item()) for c in components)
        for block_id in block_ids
    }


RESOLVED_MODES = {
    "PIECEWISE": CUDAGraphMode.PIECEWISE,
    # A state group's metadata builder supports full graphs only for uniform
    # batches, so with compilation off (as here) core keeps full graphs for
    # decode only. With torch.compile and attention as a splitting op, core
    # resolves FULL to FULL_AND_PIECEWISE instead.
    "FULL": CUDAGraphMode.FULL_DECODE_ONLY,
    "FULL_DECODE_ONLY": CUDAGraphMode.FULL_DECODE_ONLY,
    "FULL_AND_PIECEWISE": CUDAGraphMode.FULL_AND_PIECEWISE,
}


# @spec PSTATE-ACC-005 PSTATE-DECL-006
@requires_cuda
@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("cudagraph_mode", [None, *GRAPH_MODES])
def test_runner_graph_replays_each_row_on_its_own_page(cudagraph_mode):
    """Two requests on different pages decode together through a runner graph
    captured with compilation off, and produce the tokens and page contents of
    the stub's eager semantics, which the eager arm (`None`) checks too; the
    page no request holds keeps its contents. A piecewise mode uses breakable
    CUDA graphs, around which the stub's state update runs eagerly."""
    if cudagraph_mode is None:
        graph_kwargs: dict = {"enforce_eager": True}
    else:
        if CUDAGraphMode[cudagraph_mode].requires_piecewise_compilation():
            os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
        graph_kwargs = {
            "compilation_config": {
                "mode": CompilationMode.NONE,
                "cudagraph_mode": cudagraph_mode,
            }
        }
    # The null block, a page per row, and a page no request is given.
    with _in_process_llm(num_gpu_blocks_override=4, **graph_kwargs) as llm:
        runner = _runner(llm)
        resolved = runner.compilation_config.cudagraph_mode
        vocab_size = llm.llm_engine.model_config.get_vocab_size()
        all_pages = [1, 2, 3]
        before = _page_values(runner, all_pages)
        stub.PAGES.clear()
        stub.PROMPTS.clear()
        stub.FORWARDS.clear()
        execute_model = runner.execute_model
        num_steps = [0]

        def counting_execute_model(*args, **kwargs):
            num_steps[0] += 1
            return execute_model(*args, **kwargs)

        runner.execute_model = counting_execute_model
        params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
        outputs = llm.generate([{"prompt_token_ids": p} for p in GRAPH_PROMPTS], params)
        torch.accelerator.synchronize()
        page_of = {
            stub.PROMPTS[req_id]: entries[0][0]
            for req_id, entries in stub.PAGES.items()
        }
        after = _page_values(runner, all_pages)
        forwards = list(stub.FORWARDS)

    turns = [
        expected_turns([prompt], vocab_size, initial_total=INITIAL_TOTAL)[0]
        for prompt in GRAPH_PROMPTS
    ]
    assert [without_eos(o.outputs[0].token_ids) for o in outputs] == turns
    assert sorted(page_of) == sorted(tuple(p) for p in GRAPH_PROMPTS)
    assert len(set(page_of.values())) == len(GRAPH_PROMPTS)
    for prompt, turn in zip(GRAPH_PROMPTS, turns, strict=True):
        expected_page = (INITIAL_TOTAL + sum(prompt) + sum(turn), stub.TURN_LEN - 1)
        assert after[page_of[tuple(prompt)]] == expected_page, prompt
    [spare] = set(all_pages) - set(page_of.values())
    assert after[spare] == before[spare]

    decode_only_eager = [f for f in forwards if f[0] > 0 and f[1] == 0]
    if cudagraph_mode is None:
        assert decode_only_eager
        return
    assert resolved == RESOLVED_MODES[cudagraph_mode], resolved
    # Unverified on hardware for breakable piecewise graphs: assumes a
    # replayed graph runs no Python model forward.
    assert decode_only_eager == [], (
        "a decode-only step ran the Python model forward, so it did not "
        f"replay a graph (resolved {resolved}, forwards {forwards})"
    )
    assert len(forwards) < num_steps[0], (
        f"every one of {num_steps[0]} steps ran the Python model forward, so "
        f"none replayed a graph (resolved {resolved}, forwards {forwards})"
    )


# Model-state graphs over state pages


def _state_layer(runner) -> stub.StubStateLayer:
    [layer] = [
        layer
        for layer in runner.compilation_config.static_forward_context.values()
        if isinstance(layer, stub.StubStateLayer)
    ]
    return layer


def _page_counts(description: tuple) -> tuple[int, ...]:
    """The page count of each component in `stub.describe_layouts` output."""
    return tuple(
        component[2]
        for _, layers in description
        for _, components in layers
        for component in components
    )


def _generate_two_pages(llm: LLM, runner) -> tuple[list[int], list[list[int]]]:
    """Run `GRAPH_PROMPTS` as two requests and return the block ids of their
    pages, in prompt order, and each prompt's tokens."""
    stub.PAGES.clear()
    stub.PROMPTS.clear()
    params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    outputs = llm.generate([{"prompt_token_ids": p} for p in GRAPH_PROMPTS], params)
    torch.accelerator.synchronize()
    page_of = {
        stub.PROMPTS[req_id]: entries[0][0] for req_id, entries in stub.PAGES.items()
    }
    assert sorted(page_of) == sorted(tuple(p) for p in GRAPH_PROMPTS)
    return (
        [page_of[tuple(p)] for p in GRAPH_PROMPTS],
        [without_eos(o.outputs[0].token_ids) for o in outputs],
    )


# @spec PSTATE-PREP-009 PSTATE-PREP-010
@requires_cuda
@create_new_process_for_each_test("spawn")
def test_extensible_kv_cache_keeps_the_layout_of_a_block_zero_capture():
    """With the extensible KV cache on and its size left to core, the model
    state captures a graph over block id 0 while only the minimum blocks are
    backed. `extend_kv_cache` then backs every assignable page and rebinds the
    layers to views of the same pages with the final count; the layout keeps
    the reservation capacity as its page count, its base address and strides,
    and the scheduler assigns only block ids below it."""
    _skip_without_vmm()
    extend = gpu_worker.extend_kv_cache
    capture = GPUModelRunner._capture_model_state_cudagraphs

    def recording_extend(runner, kv_cache_config, num_blocks):
        committed = runner.extensible_kv_cache.num_committed_blocks
        stub.EVENTS.append(("extend", committed, num_blocks))
        extend(runner, kv_cache_config, num_blocks)

    def recording_capture(self, profile_only):
        committed = self.extensible_kv_cache.num_committed_blocks
        stub.EVENTS.append(("committed_at_capture", committed))
        capture(self, profile_only)

    with (
        patch.object(gpu_worker, "extend_kv_cache", recording_extend),
        patch.object(
            GPUModelRunner, "_capture_model_state_cudagraphs", recording_capture
        ),
        _sized_llm(enable_extensible_kv_cache=True) as llm,
    ):
        runner = _runner(llm)
        kv_cache = runner.extensible_kv_cache
        assert kv_cache is not None
        assert runner.cache_config.enable_extensible_kv_cache
        prompts = [[11 * i + 1, 11 * i + 2, 11 * i + 3] for i in range(4)]
        params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
        started = len(stub.EVENTS)
        llm.generate([{"prompt_token_ids": p} for p in prompts], params)
        torch.accelerator.synchronize()

        events = list(stub.EVENTS)
        # Profiling captures nothing under the extensible KV cache, which
        # measures after warmup: the one capture is the serving capture.
        assert [e for e in events if e[0] == "capture"] == [("capture", False)]
        [extend_at] = [i for i, e in enumerate(events) if e[0] == "extend"]
        assert events.index(("capture", False)) < extend_at
        # Only the minimum is backed at the capture, and still at the extension.
        assert [e[1] for e in events if e[0] == "committed_at_capture"] == [
            kv_cache.min_committed_blocks
        ]
        assert events[extend_at][1] == kv_cache.min_committed_blocks

        layouts = runner.model_state.state_group_layouts
        [(profile_only, at_capture)] = stub.LAYOUTS_AT_CAPTURE
        assert not profile_only
        # The layout the graph was captured through is the layout now: the
        # same addresses, dtypes, page counts, shapes and strides.
        assert stub.describe_layouts(layouts) == at_capture
        capacity = kv_cache.capacity_blocks
        assert set(_page_counts(at_capture)) == {capacity}
        [layout] = layouts
        [components] = layout.layers.values()
        layer = _state_layer(runner)
        num_blocks = runner.kv_cache_config.num_blocks
        assert num_blocks == kv_cache.num_committed_blocks
        # Core backs fewer pages than it reserved (it keeps a margin of at
        # least 1 GiB, about a thousand pages, out of the measured budget), so
        # a layout refreshed to the final count would fail below.
        assert num_blocks < capacity
        for component, view in zip(components, layer.kv_cache, strict=True):
            # The layer reads the final count, from the same base and stride.
            assert view.shape[0] == num_blocks
            assert view.data_ptr() == component.tensor.data_ptr()
            assert view.stride(0) == component.page_stride
            assert (view.dtype, tuple(view.shape[1:])) == (
                component.dtype,
                component.page_shape,
            )
        [(_, block_ids, num_pages)] = stub.CAPTURES
        assert block_ids == (0,) * runner.max_num_reqs
        assert num_pages == (capacity,) * len(components)

        scheduler = llm.llm_engine.engine_core.engine_core.scheduler
        assert scheduler.kv_cache_manager.block_pool.num_gpu_blocks <= capacity
        assigned = {
            block_id for entries in stub.PAGES.values() for block_id, _, _ in entries
        }
        # Core backed every page before it scheduled the first request: the
        # extension came before the prompts' requests were added. (Startup's
        # synthetic warmup requests run earlier, before the capture.)
        assert extend_at < started
        assert [e for e in events[started:] if e[0] == "add"] != []
        # The warmup and the four prompts' requests each held a page.
        assert assigned and 0 not in assigned
        assert max(assigned) < capacity


# @spec PSTATE-PREP-010
@requires_cuda
@create_new_process_for_each_test("spawn")
def test_profiling_cache_layout_is_captured_over_block_zero_then_replaced():
    """Without the extensible KV cache, with profiled sizing, no block-count
    override and a capture size of 0, the profiling cache has one block. The
    stub captures over block id 0 of it, profiling teardown clears the layout,
    and the serving capture runs over the serving layout, whose page count is
    the allocated block count."""
    teardown = cudagraph_utils._teardown_profiling_state

    def recording_teardown(runner):
        teardown(runner)
        stub.EVENTS.append(("teardown", len(runner.model_state.state_group_layouts)))

    with (
        patch.object(cudagraph_utils, "_teardown_profiling_state", recording_teardown),
        _sized_llm(
            enable_extensible_kv_cache=False,
            compilation_config={
                "mode": CompilationMode.NONE,
                "cudagraph_mode": "NONE",
            },
        ) as llm,
    ):
        runner = _runner(llm)
        assert getattr(runner, "extensible_kv_cache", None) is None
        assert not runner.compilation_config.max_cudagraph_capture_size
        events = list(stub.EVENTS)
        # The profiling capture, its release, the teardown that leaves no
        # layout, and then the serving capture.
        order = [
            events.index(("capture", True)),
            events.index(("clear",)),
            events.index(("teardown", 0)),
            events.index(("capture", False)),
        ]
        assert order == sorted(order)
        assert [e for e in events if e[0] == "capture"] == [
            ("capture", True),
            ("capture", False),
        ]

        num_blocks = runner.kv_cache_config.num_blocks
        assert num_blocks > 1
        layouts = runner.model_state.state_group_layouts
        [(_, profiling), (_, serving)] = stub.LAYOUTS_AT_CAPTURE
        assert set(_page_counts(profiling)) == {1}
        assert set(_page_counts(serving)) == {num_blocks}
        assert stub.describe_layouts(layouts) == serving
        for view in _state_layer(runner).kv_cache:
            assert view.shape[0] == num_blocks
        rows = (0,) * runner.max_num_reqs
        assert [(True, rows, (1, 1)), (False, rows, (num_blocks,) * 2)] == stub.CAPTURES


# @spec PSTATE-PRES-008 PSTATE-PREP-010
@requires_cuda
@create_new_process_for_each_test("spawn")
def test_block_zero_graph_replays_on_requests_pages_and_spares_witness_pages():
    """The graph captured over block id 0 before the extensible KV cache was
    extended replays on the pages two requests used, which `extend_kv_cache`
    backed: it reads and updates exactly those pages, as an eager run does,
    leaves the sampled witness pages alone, and is not captured again."""
    _skip_without_vmm()
    with _sized_llm(enable_extensible_kv_cache=True) as llm:
        runner = _runner(llm)
        # Fail, rather than pass without extending, if core disabled the feature.
        assert runner.extensible_kv_cache is not None
        state = runner.model_state
        graph = state.state_graph
        assert graph is not None and graph.graph is not None
        captured = graph.graph
        [first, second], tokens = _generate_two_pages(llm, runner)
        vocab_size = llm.llm_engine.model_config.get_vocab_size()
        turns = [
            expected_turns([p], vocab_size, initial_total=INITIAL_TOTAL)[0]
            for p in GRAPH_PROMPTS
        ]
        assert tokens == turns
        assert 0 not in (first, second) and first != second

        tensors = graph.tensors
        backed = _state_layer(runner).kv_cache[0].shape[0]
        targets = [first, second]
        # Sampled witness pages: the null block, low pages, and the middle and
        # last backed pages.
        witnesses = sorted({0, 1, 2, 3, backed // 2, backed - 1}.difference(targets))

        def snapshot() -> dict[int, list[torch.Tensor]]:
            return {i: [t[i].clone() for t in tensors] for i in targets + witnesses}

        before = snapshot()
        # Each page holds its request's state.
        for prompt, turn, block_id in zip(GRAPH_PROMPTS, turns, targets, strict=True):
            assert before[block_id][0][0].item() == (
                INITIAL_TOTAL + sum(prompt) + sum(turn)
            )
            assert before[block_id][1][0].item() == stub.TURN_LEN - 1

        deltas = [3, 5]
        graph.set_inputs(targets, deltas)
        graph.replay()
        torch.accelerator.synchronize()
        replayed, replayed_out = snapshot(), graph.out.clone()

        # The eager arm: the same update from the same page contents.
        with torch.inference_mode():
            for block_id in targets:
                for tensor, saved in zip(tensors, before[block_id], strict=True):
                    tensor[block_id].copy_(saved)
        graph.run()
        torch.accelerator.synchronize()
        eager, eager_out = snapshot(), graph.out.clone()

        assert torch.equal(replayed_out, eager_out)
        for block_id in targets + witnesses:
            for got, want in zip(replayed[block_id], eager[block_id], strict=True):
                assert torch.equal(got, want), block_id
        # Each row read and updated its own page, by the block id it was given.
        for row, (block_id, delta) in enumerate(zip(targets, deltas, strict=True)):
            for k in range(len(tensors)):
                assert torch.equal(replayed[block_id][k], before[block_id][k] + delta)
                assert replayed_out[k, row].item() == before[block_id][k][0].item()
        # No sampled witness page changed, including block id 0, which the
        # capture named.
        for block_id in witnesses:
            for got, want in zip(replayed[block_id], before[block_id], strict=True):
                assert torch.equal(got, want), block_id

        # No recapture or release since startup.
        assert [e for e in stub.EVENTS if e[0] in ("capture", "clear")] == [
            ("capture", False)
        ]
        assert len(stub.CAPTURES) == 1
        assert state.state_graph is graph and graph.graph is captured


# @spec PSTATE-PREP-010 PSTATE-PREP-006
@requires_cuda
@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("extensible", [True, False])
def test_layout_is_unchanged_by_sleep_wake_and_the_recapture(extensible):
    """With no state held, a level 1 sleep and a full wake remap the KV cache
    at the same addresses, and the recapture runs over a layout with the
    startup layout's addresses, dtypes, page counts, shapes and strides."""
    if extensible:
        _skip_without_vmm()
        engine = _sized_llm(enable_extensible_kv_cache=True, enable_sleep_mode=True)
    else:
        engine = _in_process_llm(enable_sleep_mode=True)
    with engine as llm:
        runner = _runner(llm)
        if extensible:
            # Fail, rather than pass without it, if core disabled the feature.
            assert runner.extensible_kv_cache is not None
        before = stub.describe_layouts(runner.model_state.state_group_layouts)
        assert [d for profile_only, d in stub.LAYOUTS_AT_CAPTURE if not profile_only][
            -1
        ] == before
        ready = len(stub.EVENTS)
        num_captures = len(stub.LAYOUTS_AT_CAPTURE)

        llm.sleep(level=1)
        assert stub.describe_layouts(runner.model_state.state_group_layouts) == before
        llm.wake_up(tags=["weights"])
        # A partial wake keeps the layout and recaptures nothing.
        assert stub.EVENTS[ready:] == []
        assert len(stub.LAYOUTS_AT_CAPTURE) == num_captures
        assert stub.describe_layouts(runner.model_state.state_group_layouts) == before
        llm.wake_up(tags=["kv_cache"])

        assert stub.EVENTS[ready:] == [("clear",), ("capture", False)]
        assert len(stub.LAYOUTS_AT_CAPTURE) == num_captures + 1
        profile_only, at_recapture = stub.LAYOUTS_AT_CAPTURE[-1]
        assert not profile_only
        assert at_recapture == before
        assert stub.describe_layouts(runner.model_state.state_group_layouts) == before
        # The recapture addressed block id 0 again.
        assert stub.CAPTURES[-1][1] == (0,) * runner.max_num_reqs
        assert runner.model_state.state_graph is not None


# Batch invariance

BATCH_INVARIANT_SETTINGS = """What `VLLM_BATCH_INVARIANT=1` selects here, from the code.

On a CUDA device of capability 8.x (A40 is 8.6, L40S 8.9) the worker's
`init_batch_invariance` installs the Triton matmul, linear, softmax,
log_softmax, mean and bmm overrides, sets `VLLM_USE_AOT_COMPILE=0` and
inductor's `batch_invariant` flag, and turns TF32 off. The mode changes the
graph settings only through `VllmConfig._maybe_enable_breakable_cudagraph`,
which turns breakable CUDA graphs on, and compilation mode NONE, for an
unquantized bf16 model with a matmul shape in the device family's tuned table
(none for 8.6, `ada` for 8.9). The stub's dtype is the fp16 of
facebook/opt-125m's config, so the stub keeps the engine's default
compilation and graph settings; the test prints what they resolved to. The
`PERSISTENT_STATE` backend inherits `supports_batch_invariance` from the
short-conv backend, so the mode does not refuse it."""

BI_STREAMS = {"stream": [[10, 20, 30], [40, 50], [60]], "stream-b": [[1, 2], [3, 4, 5]]}
BI_SINGLES = {"single": [7, 8, 9], "single-b": [11, 12], "single-c": [13, 14, 15, 16]}

BI_RUNS: dict[str, dict[str, int]] = {
    "stream_alone": {"stream": 0},
    "single_alone": {"single": 0},
    "pair": {"stream": 0, "single": 0},
    "four": {"stream": 0, "single": 0, "stream-b": 0, "single-b": 0},
    "staggered": {"stream": 0, "single-b": 1, "single": 2, "stream-b": 4},
    "singles": {"single": 0, "single-b": 0, "single-c": 0},
}
"""Each run's requests, by name, and the step of the run before which core
receives each. A stream's inputs and its close all arrive with its admission,
so each input after the first folds at the previous turn's stop."""


def _add_bi_request(driver: _CoreDriver, run: str, name: str) -> None:
    request_id = f"{run}/{name}"
    if name in BI_SINGLES:
        driver.add(_core_input(request_id, BI_SINGLES[name], resumable=False))
        return
    driver.admit(request_id)
    for prompt in BI_STREAMS[name]:
        driver.send(request_id, prompt)
    driver.add(_close(request_id))


# @spec PSTATE-DECL-006
@requires_cuda
@create_new_process_for_each_test("spawn")
def test_batch_invariant_mode_gives_state_requests_the_same_tokens_in_any_batch():
    """Supplementary evidence, not an arm of PSTATE-ACC-007, which names the
    Nemotron streaming plugin: with batch-invariant mode on, core accepts a
    state model (PSTATE-DECL-006), and one state stream and one single state
    request produce the same token ids alone and in batches of two, three and
    four rows that mix streams and single requests, including rows that join
    mid-stream, so each row stays on its own state. The stub computes exact
    int64 sums, which no reduction order changes, so this test cannot reveal
    a batch-variance failure. `BATCH_INVARIANT_SETTINGS` records the
    settings."""
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    with _in_process_llm(max_num_seqs=4, num_gpu_blocks_override=5) as llm:
        assert envs.VLLM_BATCH_INVARIANT
        # Set by the worker's `init_batch_invariance`, in this process.
        assert os.environ["VLLM_USE_AOT_COMPILE"] == "0"
        compilation = _runner(llm).compilation_config
        settings = {
            "device": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "compilation_mode": compilation.mode,
            "cudagraph_mode": compilation.cudagraph_mode,
            "use_breakable_cudagraph": compilation.use_breakable_cudagraph,
            "async_scheduling": (
                llm.llm_engine.vllm_config.scheduler_config.async_scheduling
            ),
        }
        print(f"batch-invariant settings: {settings}")
        assert not compilation.use_breakable_cudagraph, settings
        vocab_size = llm.llm_engine.model_config.get_vocab_size()
        driver = _CoreDriver(_engine_core(llm))
        batches: dict[str, list[frozenset[str]]] = {}
        for run, starts in BI_RUNS.items():
            pending = dict(starts)
            first_batch = len(driver.batches)
            for step in range(200):
                for name in [n for n, at in pending.items() if at <= step]:
                    _add_bi_request(driver, run, name)
                    del pending[name]
                if not pending and not driver.scheduler.has_requests():
                    break
                driver.step()
            else:
                raise AssertionError(f"run {run} did not finish")
            batches[run] = driver.batches[first_batch:]

    def rows(run: str) -> list[int]:
        return [len(batch) for batch in batches[run]]

    # The rows really shared steps, so each run is the batch it names.
    assert max(rows("pair")) == 2
    assert max(rows("four")) == 4
    assert max(rows("singles")) == 3
    assert len(set(rows("staggered"))) > 1, rows("staggered")
    # Each request that joined mid-stream shared a step with the stream.
    for name in ("single-b", "single", "stream-b"):
        pair = {"staggered/stream", f"staggered/{name}"}
        assert any(pair <= batch for batch in batches["staggered"]), name

    stream = driver.tokens("stream_alone/stream")
    single = driver.tokens("single_alone/single")
    assert stream == expected_turns(
        BI_STREAMS["stream"], vocab_size, initial_total=INITIAL_TOTAL
    )
    assert single == expected_turns(
        [BI_SINGLES["single"]], vocab_size, initial_total=INITIAL_TOTAL
    )
    for run, starts in BI_RUNS.items():
        if "stream" in starts:
            assert driver.tokens(f"{run}/stream") == stream, run
        if "single" in starts:
            assert driver.tokens(f"{run}/single") == single, run


ORDINARY_PROMPTS = [
    list(range(100, 105)),
    list(range(200, 217)),
    list(range(300, 333)),
    list(range(400, 409)),
]


# @spec PSTATE-DECL-013
@requires_cuda
@create_new_process_for_each_test("spawn")
def test_batch_invariant_mode_is_consistent_for_an_ordinary_model():
    """An internal-consistency check, not the upstream regression: with
    batch-invariant mode on, an ordinary model (facebook/opt-125m with dummy
    weights) gives each prompt the same greedy token ids alone and in batches
    of two and four in either order, and no persistent-state check is active.
    It pins no upstream token ids. Prefix caching is off, as in the upstream
    batch-invariance tests; the upstream regression arm is
    `tests/v1/determinism/test_batch_invariance.py`."""
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    with _llm(
        model="facebook/opt-125m",
        load_format="dummy",
        max_model_len=256,
        max_num_seqs=4,
        enable_prefix_caching=False,
        gpu_memory_utilization=SIZED_MEMORY_UTILIZATION,
    ) as llm:
        assert envs.VLLM_BATCH_INVARIANT
        assert not llm.llm_engine.model_config.has_persistent_state
        assert not _engine_core(llm).scheduler.has_unrecomputable_state
        params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS, ignore_eos=True)

        def tokens(prompts: list[list[int]]) -> list[list[int]]:
            outputs = llm.generate(
                [{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False
            )
            return [list(output.outputs[0].token_ids) for output in outputs]

        alone = [tokens([prompt])[0] for prompt in ORDINARY_PROMPTS]
        assert tokens(ORDINARY_PROMPTS[:2]) == alone[:2]
        assert tokens(ORDINARY_PROMPTS) == alone
        assert tokens(ORDINARY_PROMPTS[::-1]) == alone[::-1]


# Chunked prefill of a state request (design Pending question 2; no spec yet)

CHUNK_BUDGET = 4
"""Tokens per step in the chunked engine, so a first input of 9 or 10 tokens
is prefilled as chunks of 4, 4 and the remainder."""


@requires_cuda
@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize(
    "prompt_len",
    [
        pytest.param(10, id="tail_2"),
        pytest.param(
            9,
            id="tail_1",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "Pending question 2: one-token final prefill chunk takes "
                    "the decode path"
                ),
            ),
        ),
    ],
)
def test_chunked_first_input_gives_the_unchunked_output_and_page(prompt_len):
    """A state request whose first input is prefilled in chunks produces the
    tokens and final page contents of the same request prefilled in one step.
    The second chunk starts from the state the first chunk wrote, which
    `PREFILL_STARTS` shows.

    `tail_1` is a strict expected failure until Pending question 2 is
    decided: core runs a one-token prefill tail on a row
    with prior state through the state layer's decode path
    (`BaseMambaAttentionMetadataBuilder._compute_common_metadata`), and the
    stub's decode path continues the turn where its prefill path starts one,
    so the turn ends a token early."""
    prompt = list(range(1, prompt_len + 1))
    params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    results = {}
    for chunked in (False, True):
        kwargs: dict = {"enforce_eager": True}
        if chunked:
            kwargs.update(
                enable_chunked_prefill=True, max_num_batched_tokens=CHUNK_BUDGET
            )
        with _in_process_llm(**kwargs) as llm:
            runner = _runner(llm)
            vocab_size = llm.llm_engine.model_config.get_vocab_size()
            stub.PAGES.clear()
            stub.PREFILL_STARTS.clear()
            [output] = llm.generate([{"prompt_token_ids": prompt}], params)
            torch.accelerator.synchronize()
            [entries] = stub.PAGES.values()
            block_id = entries[0][0]
            results[chunked] = (
                without_eos(output.outputs[0].token_ids),
                _page_values(runner, [block_id])[block_id],
                list(stub.PREFILL_STARTS),
            )

    (tokens, page, starts), (chunked_tokens, chunked_page, chunked_starts) = (
        results[False],
        results[True],
    )
    assert (
        tokens == expected_turns([prompt], vocab_size, initial_total=INITIAL_TOTAL)[0]
    )
    assert starts == [INITIAL_TOTAL]
    # The prompt did run in chunks, and the second resumed from the page.
    assert chunked_starts[:2] == [
        INITIAL_TOTAL,
        INITIAL_TOTAL + sum(prompt[:CHUNK_BUDGET]),
    ]
    assert chunked_tokens == tokens
    assert chunked_page == page
