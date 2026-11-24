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
import torch
from vllm_add_dummy_model import my_persistent_state as stub
from vllm_add_dummy_model.my_persistent_state import expected_turns, without_eos

from tests.utils import create_new_process_for_each_test
from vllm import LLM, ModelRegistry, SamplingParams
from vllm.config import ModelConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import StreamingInput
from vllm.inputs import TokensPrompt
from vllm.plugins import load_general_plugins
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core_client import InprocClient
from vllm.v1.engine.exceptions import StreamAdmissionError

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


def _clear_stub_records() -> None:
    for records in (
        stub.EVENTS,
        stub.PREFILL_STARTS,
        stub.PAGES,
        stub.PROMPTS,
        stub.LAYOUTS_AT_CAPTURE,
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
