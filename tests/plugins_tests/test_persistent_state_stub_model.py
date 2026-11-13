# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent state end to end, with the dummy-model plugin's stub.

Every engine here is built with its real constructors from the plugin's
`StubPersistentStateForCausalLM` (facebook/opt-125m config and tokenizer,
dummy weights), whose output depends only on the state slot it is given and
its inputs. The in-process engine core exposes the scheduler and the worker;
`LLM` and `AsyncLLM` are the public entries. See
docs/design/persistent_state.md for the design.

Each in-process engine test starts its own engine in a new process; behaviours
that need the same stream state share one test where a second startup would
only repeat the setup.
"""

import asyncio
import gc
import logging
import os
import re
import time
from contextlib import contextmanager

import pytest
import torch

from tests.utils import create_new_process_for_each_test
from tests.v1.core.utils import mentions_number
from vllm import LLM, SamplingParams
from vllm.config import CUDAGraphMode
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.platforms import current_platform
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core_client import InprocClient
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.request import RequestStatus

if not current_platform.is_cuda():
    pytest.skip(reason="runs engines on a CUDA device", allow_module_level=True)

MiB = 1 << 20
STATE_LAYER = "state"
FIRST_INPUT = [10, 20, 30]
SECOND_INPUT = [40, 50]
MAX_STEPS = 64
TIMEOUT = 60
RESET_REFUSAL = (
    "persistent state: {n} state slot(s) are held, and resetting running "
    "requests would end their streams. Abort the streams and wait until they "
    "have retired, then reset again."
)
PAUSE_REFUSAL = (
    "persistent state: {n} state slot(s) are held, and clearing the cache would "
    "end their streams. Pause with clear_cache=False; to clear the cache, abort "
    "the streams, wait until they have retired, then reset the prefix cache."
)
RETRY_REFUSAL = (
    "persistent state: {n} state slot(s) are still retiring; retry the reset "
    "once they have retired."
)
RELEASE_REFUSAL = (
    "persistent state: the KV cache holds per-stream state, so it cannot be "
    "released while the engine runs."
)


@pytest.fixture(autouse=True)
def _dummy_model_plugin(monkeypatch):
    monkeypatch.setenv("VLLM_PLUGINS", "register_dummy_model")
    # In-process engines: the engine core, the executor and the worker share
    # this process, so the scheduler and the model are reachable.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


# What the persistent-state defaults resolve to, set explicitly where a test
# is about something else.
RESOLVED_DEFAULTS = dict(
    async_scheduling=False, enable_extensible_kv_cache=False, enforce_eager=True
)


def _engine_kwargs(path, **overrides) -> dict:
    """Arguments for LLM, EngineArgs and AsyncEngineArgs alike."""
    return dict(
        model=path,
        load_format="dummy",
        enable_prefix_caching=False,
        max_model_len=64,
        max_num_batched_tokens=512,
        **{"max_num_seqs": 2, **overrides},
    )


def _engine_core(path, **overrides) -> EngineCore:
    args = EngineArgs(**_engine_kwargs(path, **{**RESOLVED_DEFAULTS, **overrides}))
    return EngineCore(
        vllm_config=args.create_engine_config(),
        executor_class=UniProcExecutor,
        log_stats=False,
    )


def _model(core: EngineCore):
    return core.model_executor.driver_worker.worker.model_runner.model


def _stub_events() -> list:
    from vllm_add_dummy_model.my_persistent_state import EVENTS

    return EVENTS


def _expected_turns(vocab_size: int, eos_token_id: int) -> list[list[int]]:
    """The tokens the stub emits for FIRST_INPUT then SECOND_INPUT on one slot."""
    from vllm_add_dummy_model.my_persistent_state import expected_tokens

    first = expected_tokens(0, FIRST_INPUT[-1], vocab_size, eos_token_id)
    second = expected_tokens(len(first), SECOND_INPUT[-1], vocab_size, eos_token_id)
    return [first, second]


# The LLM regression: slots reused across prompts, with the default
# configuration and with the values it resolves to passed explicitly.


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize(
    "configuration",
    [
        # The persistent-state defaults left unset.
        "defaults",
        # Generation and slot reuse only, with nothing to resolve.
        "explicit",
    ],
)
def test_llm_generates_the_stub_tokens_on_reused_slots(
    dummy_state_model_path, configuration
):
    """With max_num_seqs=2, six greedy prompts each produce the stub's tokens
    from a fresh counter: a slot reused without its initializer, or held by
    two requests at once, would change them. With the persistent-state
    defaults left unset, they resolve to synchronous scheduling, no extensible
    KV cache and no runner CUDA graphs, and releasing the KV cache is refused.
    With those values passed explicitly (and a pool of three pages), only the
    generation and the slot reuse are checked."""
    from vllm_add_dummy_model.my_persistent_state import expected_tokens

    explicit = configuration == "explicit"
    # What the defaults resolve to on the non-eager path, with the pool that
    # planning sizes for max_num_seqs=2 (max_num_seqs + 1 pages).
    overrides: dict = (
        dict(
            async_scheduling=False,
            enable_extensible_kv_cache=False,
            compilation_config={"cudagraph_mode": "NONE"},
            enforce_eager=False,
            num_gpu_blocks_override=3,
        )
        if explicit
        else {}
    )
    llm = LLM(**_engine_kwargs(dummy_state_model_path, **overrides))
    config = llm.llm_engine.vllm_config
    assert config.scheduler_config.async_scheduling is False
    assert config.cache_config.enable_extensible_kv_cache is False
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    assert config.model_config.enforce_eager is False

    # Without multiprocessing, LLM runs the engine core in this process.
    client = llm.llm_engine.engine_core
    assert isinstance(client, InprocClient)
    model = _model(client.engine_core)
    prompts = [[11 * i + 1, 11 * i + 2, 11 * i + 3] for i in range(6)]
    params = SamplingParams(temperature=0, max_tokens=8)
    outputs = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
    assert [list(output.outputs[0].token_ids) for output in outputs] == [
        expected_tokens(0, prompt[-1], model.vocab_size, model.eos_token_id)
        for prompt in prompts
    ]
    assert {output.outputs[0].finish_reason for output in outputs} == {"stop"}

    if not explicit:
        with pytest.raises(RuntimeError) as refused:
            llm.release_kv_cache_memory()
        assert str(refused.value) == RELEASE_REFUSAL


# Startup.


@create_new_process_for_each_test("spawn")
def test_engine_refuses_more_slots_than_the_fixed_budget_holds(
    dummy_state_model_path,
):
    """kv_cache_memory_bytes of three 1 MiB pages under max_num_seqs=4 (five
    pages) fails startup; the message states the planned and fitting pages,
    the page size and the largest max_num_seqs that fits."""
    with pytest.raises(ValueError) as rejected:
        _engine_core(
            dummy_state_model_path, max_num_seqs=4, kv_cache_memory_bytes=3 * MiB
        )
    message = str(rejected.value)
    for figure in (5, 3, 2):
        assert mentions_number(message, figure), (figure, message)
    assert "max_num_seqs" in message, message


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("override", [0, 1])
def test_engine_refuses_an_override_below_two_pages_before_profiling(
    dummy_state_model_path, override
):
    """An override below two pages holds no slot: startup fails with a
    ValueError saying so, before memory profiling allocates any temporary
    pool, so no state pool, temporary or serving, was ever bound."""
    events = _stub_events()
    events.clear()
    with pytest.raises(ValueError) as rejected:
        _engine_core(dummy_state_model_path, num_gpu_blocks_override=override)
    lower = str(rejected.value).lower()
    assert "slot" in lower and re.search(r"\b(no|zero|0|none)\b", lower), lower
    assert [e for e in events if e[0] == "initialize_state_slots"] == []


@create_new_process_for_each_test("spawn")
def test_engine_refuses_an_undeclared_state_layer(dummy_undeclared_state_model_path):
    """A model that builds a persistent-state layer without declaring
    `supports_persistent_state` fails startup with a ValueError naming the
    class attribute, before memory profiling runs the model."""
    events = _stub_events()
    events.clear()
    with pytest.raises(ValueError, match="supports_persistent_state"):
        _engine_core(dummy_undeclared_state_model_path)
    assert [e for e in events if e[0] == "forward"] == []


@create_new_process_for_each_test("spawn")
def test_declared_model_gets_prepare_step_before_its_slots_are_bound(
    dummy_state_model_path,
):
    """The profiling and dummy runs before the serving bindings exist call
    `prepare_step` on a declared model, with no slots bound."""
    events = _stub_events()
    events.clear()
    core = _engine_core(dummy_state_model_path)
    try:
        serving = events.index(("initialize_state_slots",))
        early = [e for e in events[:serving] if e[0] == "prepare_step"]
        assert early, events
        assert all(e[1] or e[2] for e in early), early
        assert all(e[3] is False for e in early), early
    finally:
        core.shutdown()


# A resumable stream through the in-process engine core: cache clears never
# end a live stream.


def _params(core: EngineCore, **kwargs) -> SamplingParams:
    params = SamplingParams(temperature=0, **{"max_tokens": 8, **kwargs})
    params.update_from_generation_config({}, _model(core).eos_token_id)
    return params


def _feed(core, request_id, tokens, resumable=True, mm_features=None, **params):
    request = EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=list(tokens),
        mm_features=mm_features,
        sampling_params=_params(core, **params),
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        resumable=resumable,
    )
    core.add_request(*core.preprocess_add_request(request))


def _run_turn(core, request_id) -> list[int]:
    """Step until the stream's turn ends (an output with a finish reason)."""
    tokens: list[int] = []
    for _ in range(MAX_STEPS):
        outputs, _ = core.step_fn()
        for batch in outputs.values():
            for output in batch.outputs:
                if output.request_id != request_id:
                    continue
                tokens.extend(output.new_token_ids)
                if output.finish_reason is not None:
                    return tokens
    raise AssertionError(f"{request_id} did not end its turn in {MAX_STEPS} steps")


def _retiring(core) -> list[str]:
    return [r.request_id for r in core.scheduler.state_retirements]


def _manager(core):
    return core.scheduler.kv_cache_manager.coordinator.state_managers[STATE_LAYER]


def _snapshot(core):
    """Everything a refused clear must leave as it is."""
    scheduler = core.scheduler
    manager = _manager(core)
    return (
        {rid: manager.get(rid) for rid in scheduler.requests},
        manager.num_free_slots,
        {rid: request.status for rid, request in scheduler.requests.items()},
        [request.request_id for request in scheduler.running],
        _retiring(core),
        scheduler.pause_state,
    )


def _park(core, request_id="stream"):
    _feed(core, request_id, FIRST_INPUT)
    turn = _run_turn(core, request_id)
    assert core.scheduler.requests[request_id].status == (
        RequestStatus.WAITING_FOR_STREAMING_REQ
    )
    return turn


def _bind(core, request_id="stream"):
    """One step of a long request: it is running and its slot is bound."""
    _feed(
        core, request_id, FIRST_INPUT, resumable=False, max_tokens=50, ignore_eos=True
    )
    core.step_fn()
    assert core.scheduler.requests[request_id].status == RequestStatus.RUNNING
    assert _manager(core).get(request_id) is not None


def _retire(core, request_id="ended"):
    """End a parked stream; its slot is retiring until its fence is acked."""
    _park(core, request_id)
    core.abort_requests([request_id])
    assert _retiring(core) == [request_id]


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("held", ["parked", "bound"])
def test_in_process_keep_pause_clearing_the_cache_refused(dummy_state_model_path, held):
    """A keep pause that clears the cache is refused with the exact message
    while a slot is parked or bound, before anything changes."""
    core = _engine_core(dummy_state_model_path)
    try:
        if held == "parked":
            _park(core)
        else:
            _bind(core)
        before = _snapshot(core)
        with pytest.raises(RuntimeError) as refused:
            core.pause_scheduler(mode="keep", clear_cache=True)
        assert str(refused.value) == PAUSE_REFUSAL.format(n=1)
        assert _snapshot(core) == before
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("state", ["parked", "paused-without-slots"])
def test_in_process_release_refused(dummy_state_model_path, state):
    """Releasing the KV cache is refused before its pause check, while a stream
    is parked and also once a completed pause has left no slot held."""
    core = _engine_core(dummy_state_model_path)
    try:
        free = _manager(core).num_free_slots
        _park(core)
        if state == "paused-without-slots":
            assert core.pause_scheduler() is None
            assert _manager(core).num_free_slots == free
        before = _snapshot(core)
        with pytest.raises(RuntimeError) as refused:
            core.release_kv_cache_memory()
        assert str(refused.value) == RELEASE_REFUSAL
        assert _snapshot(core) == before
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
def test_in_process_keep_pause_without_clear_preserves_the_stream(
    dummy_state_model_path,
):
    """A stream parked across a keep pause without a cache clear resumes under
    the same handle with the tokens of an unpaused stream."""
    core = _engine_core(dummy_state_model_path)
    try:
        model = _model(core)
        first = _park(core)
        handle = _manager(core).get("stream")
        assert core.pause_scheduler(mode="keep", clear_cache=False) is None
        core.resume_scheduler()
        assert _manager(core).get("stream") == handle
        _feed(core, "stream", SECOND_INPUT)
        second = _run_turn(core, "stream")
        assert [first, second] == _expected_turns(model.vocab_size, model.eos_token_id)
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("reset_connector", [True, False])
def test_in_process_abort_pause_drains_and_clears(
    dummy_state_model_path, reset_connector
):
    """The default abort pause, whatever reset_connector is, ends a parked
    stream, steps the engine until its slot has retired and clears the cache;
    a new stream then starts from the initializer."""
    core = _engine_core(dummy_state_model_path)
    try:
        model = _model(core)
        free = _manager(core).num_free_slots
        _park(core, "aborted")
        assert core.pause_scheduler(reset_connector=reset_connector) is None
        assert isinstance(core.scheduler, Scheduler)
        assert "aborted" not in core.scheduler.requests
        assert _retiring(core) == []
        assert _manager(core).num_free_slots == free
        core.resume_scheduler()
        first = _park(core, "next")
        _feed(core, "next", SECOND_INPUT)
        second = _run_turn(core, "next")
        assert [first, second] == _expected_turns(model.vocab_size, model.eos_token_id)
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("reset_running_requests", [True, False])
def test_in_process_reset_drains_retiring_slots(
    dummy_state_model_path, reset_running_requests
):
    """With no unfinished request and only retiring slots, the in-process
    reset steps the engine until no slot is retiring and then resets."""
    core = _engine_core(dummy_state_model_path)
    try:
        free = _manager(core).num_free_slots
        _retire(core)
        assert core.reset_prefix_cache(reset_running_requests=reset_running_requests)
        assert _retiring(core) == []
        assert _manager(core).num_free_slots == free
    finally:
        core.shutdown()


# A pause without a cache clear leaves the slot retiring; the queued request
# then waits in the scheduler, which a paused scheduler does not report as
# unfinished.
_ALREADY_PAUSED = pytest.mark.parametrize("paused", [None, "keep", "abort"])


def _pause_without_clear(core, paused):
    if paused is not None:
        assert core.pause_scheduler(mode=paused, clear_cache=False) is None


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("reset_running_requests", [True, False])
@pytest.mark.parametrize("paused", [None, "keep"])
def test_in_process_reset_with_a_queued_request_does_not_step(
    dummy_state_model_path, reset_running_requests, paused
):
    """With a slot retiring and a request queued, the in-process reset of
    running requests raises the retry message, and the reset without them
    returns False; neither steps the engine or changes anything, also when
    the engine is already paused and so reports no unfinished request."""
    core = _engine_core(dummy_state_model_path)
    try:
        _retire(core)
        _pause_without_clear(core, paused)
        _feed(core, "queued", FIRST_INPUT, resumable=False)
        before = _snapshot(core)
        if reset_running_requests:
            with pytest.raises(RuntimeError) as refused:
                core.reset_prefix_cache(reset_running_requests=True)
            assert str(refused.value) == RETRY_REFUSAL.format(n=1)
        else:
            assert core.reset_prefix_cache(reset_running_requests=False) is False
        assert _snapshot(core) == before
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
@_ALREADY_PAUSED
def test_in_process_clearing_keep_pause_with_a_queued_request_asks_for_a_retry(
    dummy_state_model_path, paused
):
    """With a slot retiring and a request queued, a keep pause that clears the
    cache raises the retry message before any change, whether or not the
    engine is already paused."""
    core = _engine_core(dummy_state_model_path)
    try:
        _retire(core)
        _pause_without_clear(core, paused)
        _feed(core, "queued", FIRST_INPUT, resumable=False)
        before = _snapshot(core)
        with pytest.raises(RuntimeError) as refused:
            core.pause_scheduler(mode="keep", clear_cache=True)
        assert str(refused.value) == RETRY_REFUSAL.format(n=1)
        assert _snapshot(core) == before
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
def test_in_process_clearing_keep_pause_with_only_retiring_slots_drains(
    dummy_state_model_path,
):
    """With only a retiring slot, a keep pause that clears the cache steps the
    engine until the slot has retired, then completes."""
    core = _engine_core(dummy_state_model_path)
    try:
        free = _manager(core).num_free_slots
        _retire(core)
        assert core.pause_scheduler(mode="keep", clear_cache=True) is None
        assert _retiring(core) == []
        assert _manager(core).num_free_slots == free
        assert core.scheduler.pause_state == PauseState.PAUSED_ALL
    finally:
        core.shutdown()


def _end_with_misplaced_input(core, request_id="stream"):
    """End a parked stream with an input whose image lies past its prompt: its
    slot retires and its error output is queued for the next step."""
    from vllm.multimodal.inputs import (
        MultiModalFeatureSpec,
        MultiModalKwargsItem,
        PlaceholderRange,
    )

    image = MultiModalFeatureSpec(
        data=MultiModalKwargsItem.dummy(),
        mm_position=PlaceholderRange(offset=len(SECOND_INPUT), length=1),
        identifier="image",
        modality="image",
    )
    _feed(core, request_id, SECOND_INPUT, mm_features=[image])
    assert request_id not in core.scheduler.requests
    assert _retiring(core) == [request_id]


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize(
    ("mode", "reset_connector"),
    [(None, True), ("keep", True), ("keep", False), ("abort", False)],
    ids=["reset", "keep-pause", "keep-pause-keeping-connector", "abort-pause"],
)
def test_in_process_cache_clear_delivers_a_queued_error_once(
    dummy_state_model_path, mode, reset_connector
):
    """A stream that ended with an error just before a cache clear gets its
    error output exactly once, after the clear: the steps that retire its slot
    during the reset or the clearing pause deliver nothing and drop nothing."""
    from vllm.v1.engine import FinishReason

    core = _engine_core(dummy_state_model_path)
    try:
        _park(core)
        _end_with_misplaced_input(core)
        if mode is None:
            assert core.reset_prefix_cache(reset_running_requests=True)
        else:
            assert (
                core.pause_scheduler(
                    mode=mode, clear_cache=True, reset_connector=reset_connector
                )
                is None
            )
            core.resume_scheduler()
        assert _retiring(core) == []
        ends = []
        for _ in range(3):
            outputs, _ = core.step_fn()
            ends += [
                output.finish_reason
                for batch in outputs.values()
                for output in batch.outputs
                if output.request_id == "stream" and output.finish_reason is not None
            ]
        assert ends == [FinishReason.ERROR]
    finally:
        core.shutdown()


# The multiprocess engine through AsyncLLM. A running request's slot
# stays bound from step to step, but a multiprocess client cannot order its
# pause or reset before the request finishes, so bound-slot refusals are
# tested in-process above.


def _async_llm(path, monkeypatch):
    from vllm.v1.engine.async_llm import AsyncLLM

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    return AsyncLLM.from_engine_args(
        AsyncEngineArgs(**_engine_kwargs(path, **RESOLVED_DEFAULTS))
    )


class _Stream:
    """One streaming-input request: FIRST_INPUT, then SECOND_INPUT once
    `release()` is called; `parked` is set when the first turn ends."""

    def __init__(self, llm, request_id):
        from vllm.sampling_params import RequestOutputKind

        self.llm = llm
        self.request_id = request_id
        self.gate = asyncio.Event()
        self.parked = asyncio.Event()
        self.params = SamplingParams(
            max_tokens=8, temperature=0, output_kind=RequestOutputKind.DELTA
        )
        self.task = asyncio.create_task(self._run())

    async def _inputs(self):
        # Copies: the output processor extends a request's prompt list in place.
        from vllm.engine.protocol import StreamingInput
        from vllm.inputs import TokensPrompt

        yield StreamingInput(prompt=TokensPrompt(prompt_token_ids=list(FIRST_INPUT)))
        await self.gate.wait()
        yield StreamingInput(prompt=TokensPrompt(prompt_token_ids=list(SECOND_INPUT)))

    async def _run(self):
        turns: list[list[int]] = [[]]
        final = None
        async for output in self.llm.generate(
            self._inputs(), self.params, self.request_id
        ):
            completion = output.outputs[0]
            turns[-1].extend(completion.token_ids)
            if completion.finish_reason is not None and not output.finished:
                turns.append([])
                self.parked.set()
            if output.finished:
                final = completion.finish_reason
        return [turn for turn in turns if turn], final

    async def wait_parked(self):
        await asyncio.wait_for(self.parked.wait(), TIMEOUT)

    async def finish(self):
        self.gate.set()
        return await asyncio.wait_for(self.task, TIMEOUT)


def _llm_expected_turns(llm) -> list[list[int]]:
    hf = llm.model_config.hf_config
    return _expected_turns(llm.model_config.get_vocab_size(), int(hf.eos_token_id))


@pytest.mark.asyncio
async def test_async_llm_clearing_pauses_and_reset_refused_while_parked(
    dummy_state_model_path, monkeypatch
):
    """While a stream is parked, keep and wait pauses that clear the cache and
    a reset of running requests are refused with the exact messages; the
    stream then continues with the tokens of an unrefused one."""
    llm = _async_llm(dummy_state_model_path, monkeypatch)
    try:
        stream = _Stream(llm, "parked")
        await stream.wait_parked()
        for mode in ("keep", "wait"):
            with pytest.raises(Exception, match=re.escape(PAUSE_REFUSAL.format(n=1))):
                await asyncio.wait_for(
                    llm.pause_generation(mode=mode, clear_cache=True), TIMEOUT
                )
        with pytest.raises(Exception, match=re.escape(RESET_REFUSAL.format(n=1))):
            await asyncio.wait_for(
                llm.reset_prefix_cache(reset_running_requests=True), TIMEOUT
            )
        turns, _ = await stream.finish()
        assert turns == _llm_expected_turns(llm)
    finally:
        llm.shutdown()


@pytest.mark.asyncio
async def test_async_llm_keep_pause_without_clear_preserves_the_stream(
    dummy_state_model_path, monkeypatch
):
    llm = _async_llm(dummy_state_model_path, monkeypatch)
    try:
        stream = _Stream(llm, "paused")
        await stream.wait_parked()
        await asyncio.wait_for(
            llm.pause_generation(mode="keep", clear_cache=False), TIMEOUT
        )
        await asyncio.wait_for(llm.resume_generation(), TIMEOUT)
        turns, _ = await stream.finish()
        assert turns == _llm_expected_turns(llm)
    finally:
        llm.shutdown()


@pytest.mark.asyncio
async def test_async_llm_abort_pause_drains_a_parked_stream(
    dummy_state_model_path, monkeypatch
):
    """The default abort pause ends a parked stream with an abort, completes
    once its slot has retired, and a new stream then starts from the
    initializer."""
    llm = _async_llm(dummy_state_model_path, monkeypatch)
    try:
        stream = _Stream(llm, "aborted")
        await stream.wait_parked()
        await asyncio.wait_for(llm.pause_generation(), TIMEOUT)
        turns, final = await asyncio.wait_for(stream.task, TIMEOUT)
        assert final == "abort"
        await asyncio.wait_for(llm.resume_generation(), TIMEOUT)
        after = _Stream(llm, "after")
        await after.wait_parked()
        assert (await after.finish())[0] == _llm_expected_turns(llm)
    finally:
        llm.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["keep", "wait"])
async def test_async_llm_clearing_pause_with_only_retiring_slots_completes(
    dummy_state_model_path, monkeypatch, mode
):
    """After a parked stream is aborted, a pause that clears the cache
    completes (once its slot has retired) and a new stream starts from the
    initializer. Whether the pause arrives before or after the retirement is
    not controlled, so the pause's repeated check at its deferred finish is
    not claimed here."""
    llm = _async_llm(dummy_state_model_path, monkeypatch)
    try:
        stream = _Stream(llm, "ended")
        await stream.wait_parked()
        await asyncio.wait_for(llm.abort("ended"), TIMEOUT)
        await asyncio.wait_for(
            llm.pause_generation(mode=mode, clear_cache=True), TIMEOUT
        )
        await asyncio.wait_for(llm.resume_generation(), TIMEOUT)
        # The abort ended the stream.
        assert (await asyncio.wait_for(stream.task, TIMEOUT))[1] == "abort"
        after = _Stream(llm, "after")
        await after.wait_parked()
        assert (await after.finish())[0] == _llm_expected_turns(llm)
    finally:
        llm.shutdown()


def _record_engine_inputs(llm) -> list[tuple[list[int], bool]]:
    """The prompt and resumable flag of each request AsyncLLM sends to the
    engine core, in order, copied when it is sent."""
    sent: list[tuple[list[int], bool]] = []
    add = llm.engine_core.add_request_async

    async def record(request):
        sent.append((list(request.prompt_token_ids or ()), request.resumable))
        return await add(request)

    llm.engine_core.add_request_async = record
    return sent


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["length", "stop_token"])
@pytest.mark.parametrize("later", ["after_the_end", "queued"])
async def test_async_llm_stream_ends_with_a_turn_that_stops_off_eos(
    dummy_state_model_path, monkeypatch, stop, later
):
    """A turn that stops on the length limit or a stop token finishes the
    client's stream with that finish reason, so `generate` returns. An input
    offered after the client has seen the end never reaches the engine, and
    no end-of-input request follows it; an input queued before the end is
    dropped with the request. Both slots are then free for two new streams.
    (Repetition cannot end a stub turn: the stub never emits one token twice
    in a row, and a turn has two tokens before its EOS.)"""
    from vllm_add_dummy_model.my_persistent_state import expected_tokens

    from vllm.engine.protocol import StreamingInput
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind

    llm = _async_llm(dummy_state_model_path, monkeypatch)
    try:
        hf = llm.model_config.hf_config
        turn = expected_tokens(
            0, FIRST_INPUT[-1], llm.model_config.get_vocab_size(), int(hf.eos_token_id)
        )
        if stop == "length":
            params: dict = {"max_tokens": 1}
            expected = (turn[:1], "length", None)
        else:
            params = {"max_tokens": 8, "stop_token_ids": [turn[1]]}
            expected = (turn[:2], "stop", turn[1])
        sent = _record_engine_inputs(llm)
        ended = asyncio.Event()

        async def inputs():
            yield StreamingInput(
                prompt=TokensPrompt(prompt_token_ids=list(FIRST_INPUT))
            )
            if later == "after_the_end":
                await ended.wait()
            yield StreamingInput(
                prompt=TokensPrompt(prompt_token_ids=list(SECOND_INPUT))
            )

        outputs = []

        async def run():
            sampling = SamplingParams(
                temperature=0, output_kind=RequestOutputKind.DELTA, **params
            )
            async for output in llm.generate(inputs(), sampling, "ended"):
                outputs.append(output)
                if output.outputs[0].finish_reason is not None:
                    ended.set()

        await asyncio.wait_for(run(), TIMEOUT)
        final = outputs[-1].outputs[0]
        assert [t for o in outputs for t in o.outputs[0].token_ids] == expected[0]
        assert outputs[-1].finished
        assert (final.finish_reason, final.stop_reason) == expected[1:]
        assert not any(o.finished for o in outputs[:-1])
        if later == "after_the_end":
            assert sent == [(FIRST_INPUT, True)]
        streams = [_Stream(llm, f"after-{i}") for i in range(2)]
        for after in streams:
            await after.wait_parked()
        for after in streams:
            assert (await after.finish())[0] == _llm_expected_turns(llm)
    finally:
        llm.shutdown()


# Stream admission: the engine reserves a stream's slot before any input.


def _admission(core, request_id="stream", **fields) -> EngineCoreRequest:
    return EngineCoreRequest(
        **{
            "request_id": request_id,
            "prompt_token_ids": None,
            "mm_features": None,
            "sampling_params": _params(core),
            "pooling_params": None,
            "arrival_time": time.time(),
            "lora_request": None,
            "cache_salt": None,
            "data_parallel_rank": None,
            "resumable": True,
            **fields,
        }
    )


def _feed_admitted(core, update_seq, tokens, *, resumable=True, request_id="stream"):
    """Send input `update_seq` of an admitted stream, or with
    `resumable=False` the close that carries no input."""
    from vllm.v1.engine import StreamAdmissionContext

    request = _admission(
        core,
        request_id,
        prompt_token_ids=list(tokens),
        resumable=resumable,
        stream_admission=StreamAdmissionContext(
            input_closed=not resumable, update_seq=update_seq
        ),
    )
    core.add_request(*core.preprocess_add_request(request))


def _steps_until_retired(core, request_id="stream"):
    """Step until a batch carries the stream's retirement notice; returns
    every output of the stream seen on the way."""
    seen = []
    for _ in range(MAX_STEPS):
        outputs, _ = core.step_fn()
        for batch in outputs.values():
            seen += [o for o in batch.outputs if o.request_id == request_id]
            if request_id in (batch.retired_streams or ()):
                return seen
    raise AssertionError(f"{request_id} did not retire in {MAX_STEPS} steps")


@create_new_process_for_each_test("spawn")
def test_engine_core_admitted_stream_runs_on_the_slot_from_admission(
    dummy_state_model_path,
):
    """An admission reserves a slot and leaves the engine with no work; the
    stream's inputs then run on that slot with the tokens of an ordinary
    stream, and after the close the engine's steps deliver the retirement
    notice to the caller of admit_stream."""
    from vllm.v1.engine import StreamAdmissionResult

    core = _engine_core(dummy_state_model_path)
    try:
        model = _model(core)
        result = core.admit_stream(_admission(core))
        assert result == StreamAdmissionResult(True, "stream")
        handle = _manager(core).get("stream")
        assert handle is not None
        assert core.step_fn() == ({}, False)
        _feed_admitted(core, 1, FIRST_INPUT)
        first = _run_turn(core, "stream")
        _feed_admitted(core, 2, SECOND_INPUT)
        second = _run_turn(core, "stream")
        assert [first, second] == _expected_turns(model.vocab_size, model.eos_token_id)
        assert _manager(core).get("stream") == handle
        _feed_admitted(core, 3, [0], resumable=False)
        (end,) = _steps_until_retired(core)
        assert end.finish_reason is not None and end.ends_stream is True
        assert _manager(core).num_free_slots == 2
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
def test_in_process_abort_pause_that_clears_the_cache_keeps_one_retirement_notice(
    dummy_state_model_path,
):
    """An abort pause ends an admitted stream as it ends other requests: the
    in-process engine sends no output for it and steps until its scheduled
    slot has retired, dropping those steps' outputs; the stream's retirement
    notice is not dropped with them, and the steps after the pause deliver
    it exactly once."""
    core = _engine_core(dummy_state_model_path)
    try:
        assert core.admit_stream(_admission(core)).accepted
        _feed_admitted(core, 1, FIRST_INPUT)
        _run_turn(core, "stream")
        assert core.pause_scheduler() is None
        assert "stream" not in core.scheduler.requests
        assert _retiring(core) == []
        core.resume_scheduler()
        notices = []
        outputs_of_stream = []
        for _ in range(4):
            outputs, _ = core.step_fn()
            for batch in outputs.values():
                notices += batch.retired_streams or []
                outputs_of_stream += [
                    o for o in batch.outputs if o.request_id == "stream"
                ]
        assert notices == ["stream"]
        assert outputs_of_stream == []
        assert _manager(core).num_free_slots == 2
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("multiprocessing", ["0", "1"], ids=["in_process", "sync_mp"])
def test_synchronous_clients_do_not_implement_stream_admission(
    dummy_state_model_path, monkeypatch, multiprocessing
):
    """Only the async multiprocess client implements admission and the wait
    for retirement: the in-process client and the synchronous multiprocess
    client, which LLM uses, raise NotImplementedError."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", multiprocessing)
    llm = LLM(**_engine_kwargs(dummy_state_model_path, **RESOLVED_DEFAULTS))
    client = llm.llm_engine.engine_core
    assert isinstance(client, InprocClient) == (multiprocessing == "0")
    request = _raw_admission(llm.llm_engine.vllm_config)
    with pytest.raises(NotImplementedError):
        asyncio.run(client.admit_stream_async(request))
    with pytest.raises(NotImplementedError):
        asyncio.run(client.wait_stream_retired_async("raw"))


class _AdmittedStream:
    """One admitted streaming-input request on AsyncLLM: each `feed()` sends
    the next input, `close()` ends the generator; `task` collects the turns
    and the finish reason of the output that ends the stream. `admitted` is
    set when the frontend first reads the generator, which it does only
    after the admission has succeeded."""

    def __init__(self, llm, request_id):
        from vllm.sampling_params import RequestOutputKind

        self.llm = llm
        self.request_id = request_id
        self.inputs: asyncio.Queue = asyncio.Queue()
        self.turn_ended = asyncio.Event()
        self.admitted = asyncio.Event()
        self.params = SamplingParams(
            max_tokens=8, temperature=0, output_kind=RequestOutputKind.DELTA
        )
        self.task = asyncio.create_task(self._run())

    async def _inputs(self):
        from vllm.engine.protocol import StreamingInput
        from vllm.inputs import TokensPrompt

        self.admitted.set()
        while (tokens := await self.inputs.get()) is not None:
            yield StreamingInput(prompt=TokensPrompt(prompt_token_ids=list(tokens)))

    async def _run(self):
        turns: list[list[int]] = [[]]
        final = None
        async for output in self.llm.generate(
            self._inputs(), self.params, self.request_id, stream_admission=True
        ):
            completion = output.outputs[0]
            turns[-1].extend(completion.token_ids)
            if completion.finish_reason is not None and not output.finished:
                turns.append([])
                self.turn_ended.set()
            if output.finished:
                final = completion.finish_reason
        return [turn for turn in turns if turn], final

    async def feed(self, tokens):
        self.turn_ended.clear()
        self.inputs.put_nowait(tokens)
        await asyncio.wait_for(self.turn_ended.wait(), TIMEOUT)

    async def close(self):
        self.inputs.put_nowait(None)
        return await asyncio.wait_for(self.task, TIMEOUT)


@pytest.mark.asyncio
async def test_async_llm_admitted_stream_runs_closes_and_retires(
    dummy_state_model_path, monkeypatch
):
    """An admitted stream produces the tokens of an ordinary stream for each
    input, ends with a stop output once its generator ends, and
    wait_stream_retired returns once its slot is free; the caller's request
    ID can then be admitted again."""
    llm = _async_llm(dummy_state_model_path, monkeypatch)
    try:
        stream = _AdmittedStream(llm, "stream")
        await stream.feed(FIRST_INPUT)
        await stream.feed(SECOND_INPUT)
        turns, final = await stream.close()
        assert turns == _llm_expected_turns(llm)
        assert final == "stop"
        await asyncio.wait_for(llm.wait_stream_retired("stream"), TIMEOUT)
        again = _AdmittedStream(llm, "stream")
        await again.feed(FIRST_INPUT)
        assert (await again.close())[0] == _llm_expected_turns(llm)[:1]
    finally:
        llm.shutdown()


@pytest.mark.asyncio
async def test_async_llm_refuses_a_stream_when_every_slot_is_held(
    dummy_state_model_path, monkeypatch
):
    """With both slots held by streams that have no input yet, a third
    admission raises StreamAdmissionError with reason capacity and HTTP
    status 503; ending a stream frees its slot for the next admission."""
    from http import HTTPStatus

    from vllm.v1.engine import StreamAdmissionError

    llm = _async_llm(dummy_state_model_path, monkeypatch)
    try:
        held = [_AdmittedStream(llm, f"held-{i}") for i in range(2)]
        for stream in held:
            await asyncio.wait_for(stream.admitted.wait(), TIMEOUT)
        refused = _AdmittedStream(llm, "refused")
        with pytest.raises(StreamAdmissionError) as error:
            await asyncio.wait_for(refused.task, TIMEOUT)
        assert error.value.result.reason == "capacity"
        assert error.value.http_status == HTTPStatus.SERVICE_UNAVAILABLE
        await held[0].close()
        await asyncio.wait_for(llm.wait_stream_retired("held-0"), TIMEOUT)
        admitted = _AdmittedStream(llm, "admitted")
        await admitted.feed(FIRST_INPUT)
        await admitted.close()
        await held[1].close()
    finally:
        llm.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["keep", "wait"])
async def test_async_llm_never_fed_stream_leaves_the_engine_idle_for_a_pause(
    dummy_state_model_path, monkeypatch, mode
):
    """A never-fed stream is not unfinished work: a pause that keeps or waits
    for requests completes while it holds its slot, and an admission during
    the pause is refused as engine_paused."""
    from vllm.v1.engine import StreamAdmissionError

    llm = _async_llm(dummy_state_model_path, monkeypatch)
    try:
        idle = _AdmittedStream(llm, "idle")
        await asyncio.wait_for(idle.admitted.wait(), TIMEOUT)
        await asyncio.wait_for(
            llm.pause_generation(mode=mode, clear_cache=False), TIMEOUT
        )
        refused = _AdmittedStream(llm, "refused")
        with pytest.raises(StreamAdmissionError) as error:
            await asyncio.wait_for(refused.task, TIMEOUT)
        assert error.value.result.reason == "engine_paused"
        await asyncio.wait_for(llm.resume_generation(), TIMEOUT)
        await idle.feed(FIRST_INPUT)
        assert (await idle.close())[0] == _llm_expected_turns(llm)[:1]
    finally:
        llm.shutdown()


@pytest.mark.asyncio
async def test_async_llm_never_fed_stream_expires_while_the_engine_is_idle(
    dummy_state_model_path, monkeypatch
):
    """The idle engine wakes at the earliest stream deadline: a stream that
    gets no input within the configured never-fed timeout ends with one abort
    output, and its slot retires."""
    from vllm.v1.engine.async_llm import AsyncLLM

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    llm = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            **_engine_kwargs(
                dummy_state_model_path,
                **RESOLVED_DEFAULTS,
                stream_admission_timeouts={"never_fed_timeout": 0.5},
            )
        )
    )
    try:
        stream = _AdmittedStream(llm, "idle")
        turns, final = await asyncio.wait_for(stream.task, TIMEOUT)
        assert (turns, final) == ([], "abort")
        await asyncio.wait_for(llm.wait_stream_retired("idle"), TIMEOUT)
    finally:
        llm.shutdown()


async def _next_output(client, request_id):
    """The next engine output for `request_id`, with its batch."""
    while True:
        batch = await asyncio.wait_for(client.get_output_async(), TIMEOUT)
        for output in batch.outputs:
            if output.request_id == request_id:
                return output, batch


# The two tests below read the engine's outputs from the multiprocess client
# itself, so they build the client without AsyncLLM, whose output handler
# would consume them.


def _async_mp_client(path, monkeypatch, **overrides):
    from vllm.v1.engine.core_client import EngineCoreClient
    from vllm.v1.executor import Executor

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    config = AsyncEngineArgs(
        **_engine_kwargs(path, **RESOLVED_DEFAULTS, **overrides)
    ).create_engine_config()
    return config, EngineCoreClient.make_async_mp_client(
        config, Executor.get_class(config), log_stats=False
    )


def _raw_admission(config, request_id="raw", **fields) -> EngineCoreRequest:
    params = SamplingParams(temperature=0, max_tokens=8)
    params.update_from_generation_config(
        {}, int(config.model_config.hf_config.eos_token_id)
    )
    return EngineCoreRequest(
        **{
            "request_id": request_id,
            "prompt_token_ids": None,
            "mm_features": None,
            "sampling_params": params,
            "pooling_params": None,
            "arrival_time": time.time(),
            "lora_request": None,
            "cache_salt": None,
            "data_parallel_rank": None,
            "resumable": True,
            "external_req_id": request_id,
            **fields,
        }
    )


@pytest.mark.asyncio
async def test_multiprocess_abort_pause_output_carries_no_stream_marking(
    dummy_state_model_path, monkeypatch
):
    """The abort output that the multiprocess engine core sends for an
    admitted stream an abort pause ends has neither ends_stream nor
    stream_output; the stream's retirement notice follows."""
    from vllm.v1.engine import FinishReason

    config, client = _async_mp_client(dummy_state_model_path, monkeypatch)
    try:
        result = await asyncio.wait_for(
            client.admit_stream_async(_raw_admission(config)), TIMEOUT
        )
        assert result.accepted
        await asyncio.wait_for(
            client.pause_scheduler_async(mode="abort", clear_cache=False), TIMEOUT
        )
        output, _ = await _next_output(client, "raw")
        assert output.finish_reason == FinishReason.ABORT
        assert output.ends_stream is False
        assert output.stream_output is None
        await asyncio.wait_for(client.wait_stream_retired_async("raw"), TIMEOUT)
    finally:
        client.shutdown()


@pytest.mark.asyncio
async def test_multiprocess_engine_rejects_an_input_it_cannot_validate(
    dummy_state_model_path, monkeypatch
):
    """An input of an admitted stream that fails the engine's validation
    before preprocessing gets one rejection output with its own context, a
    validation_error RequestError and no finished request; the session then
    runs a valid input."""
    from vllm.v1.engine import FinishReason, StreamAdmissionContext, StreamOutput

    config, client = _async_mp_client(dummy_state_model_path, monkeypatch)
    try:
        await asyncio.wait_for(
            client.admit_stream_async(_raw_admission(config)), TIMEOUT
        )
        context = StreamAdmissionContext(update_seq=1)
        invalid = _raw_admission(
            config,
            prompt_token_ids=list(FIRST_INPUT),
            stream_admission=context,
            sampling_params=SamplingParams(n=2),
        )
        await client.add_request_async(invalid)
        rejection, batch = await _next_output(client, "raw")
        assert rejection.new_token_ids == []
        assert rejection.finish_reason == FinishReason.ERROR
        assert rejection.ends_stream is False
        assert rejection.stream_output == StreamOutput(context, "validation_error")
        assert rejection.error.code == "validation_error"
        assert not batch.finished_requests

        valid = _raw_admission(
            config,
            prompt_token_ids=list(FIRST_INPUT),
            stream_admission=StreamAdmissionContext(update_seq=2),
        )
        await client.add_request_async(valid)
        output, _ = await _next_output(client, "raw")
        assert output.stream_output == StreamOutput(
            StreamAdmissionContext(update_seq=2)
        )
        await client.abort_requests_async(["raw"])
        await asyncio.wait_for(client.wait_stream_retired_async("raw"), TIMEOUT)
    finally:
        client.shutdown()


def _raw_input(config, update_seq, tokens=FIRST_INPUT, **fields):
    from vllm.v1.engine import StreamAdmissionContext

    return _raw_admission(
        config,
        prompt_token_ids=list(tokens),
        stream_admission=StreamAdmissionContext(update_seq=update_seq),
        **fields,
    )


@pytest.mark.asyncio
async def test_multiprocess_rejected_input_with_abort_immediately_keeps_its_session(
    dummy_state_model_path, monkeypatch
):
    """An input that the scheduler rejects leaves its session unchanged also
    when it has abort_immediately set: the session then runs a valid input."""
    from vllm.v1.engine import StreamAdmissionContext, StreamOutput

    config, client = _async_mp_client(dummy_state_model_path, monkeypatch)
    try:
        await asyncio.wait_for(
            client.admit_stream_async(_raw_admission(config)), TIMEOUT
        )
        empty = _raw_input(config, 1, tokens=(), abort_immediately=True)
        await client.add_request_async(empty)
        rejection, batch = await _next_output(client, "raw")
        assert rejection.stream_output == StreamOutput(
            empty.stream_admission, "empty_update"
        )
        assert not batch.finished_requests
        await client.add_request_async(_raw_input(config, 2))
        output, _ = await _next_output(client, "raw")
        assert output.stream_output == StreamOutput(
            StreamAdmissionContext(update_seq=2)
        )
        await client.abort_requests_async(["raw"])
        await asyncio.wait_for(client.wait_stream_retired_async("raw"), TIMEOUT)
    finally:
        client.shutdown()


@pytest.mark.asyncio
async def test_multiprocess_paused_engine_rejections_and_their_precedence(
    dummy_state_model_path, monkeypatch
):
    """While the engine rejects new requests, an input whose session has
    ended is rejected as engine_paused with a retryable RequestError, not as
    unknown_admission, and an input that also fails validation is rejected
    as validation_error, which preprocessing reports first. A new request
    without a context keeps the ordinary paused-engine error output."""
    from vllm.v1.engine import FinishReason, StreamOutput

    config, client = _async_mp_client(dummy_state_model_path, monkeypatch)
    try:
        await asyncio.wait_for(
            client.admit_stream_async(_raw_admission(config)), TIMEOUT
        )
        await client.abort_requests_async(["raw"])
        await asyncio.wait_for(client.wait_stream_retired_async("raw"), TIMEOUT)
        await asyncio.wait_for(
            client.pause_generation_async(mode="wait", clear_cache=False), TIMEOUT
        )

        late = _raw_input(config, 1)
        await client.add_request_async(late)
        paused, batch = await _next_output(client, "raw")
        assert paused.finish_reason == FinishReason.ERROR
        assert paused.ends_stream is False
        assert paused.stream_output == StreamOutput(
            late.stream_admission, "engine_paused"
        )
        assert paused.error.code == "engine_paused" and paused.error.retryable
        assert not batch.finished_requests

        invalid = _raw_input(config, 2, sampling_params=SamplingParams(n=2))
        await client.add_request_async(invalid)
        refused, _ = await _next_output(client, "raw")
        assert refused.stream_output == StreamOutput(
            invalid.stream_admission, "validation_error"
        )
        assert refused.error.code == "validation_error"

        plain = _raw_admission(
            config, "plain", prompt_token_ids=list(FIRST_INPUT), resumable=False
        )
        await client.add_request_async(plain)
        ordinary, batch = await _next_output(client, "plain")
        assert ordinary.finish_reason == FinishReason.ERROR
        assert ordinary.stream_output is None
        assert ordinary.error.code == "engine_paused"
        assert "plain" in batch.finished_requests
    finally:
        client.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("context", [True, False], ids=["input", "context_free"])
async def test_multiprocess_preprocessing_failure_leaves_the_session_unchanged(
    dummy_state_model_path, monkeypatch, context
):
    """A request under an admitted ID whose preprocessing raises an
    unexpected error changes nothing in the session, whose next valid input
    then runs its whole turn with the tokens of a first input and parks.
    With a context the request is rejected as preprocessing_error; without
    one it gets the ordinary error output, with no stream_output and its ID
    in finished_requests.

    The trigger is sampling parameters without max_tokens: they pass the
    stream-input validation, and `Request.__init__` then fails its assert
    that max_tokens is set. If that assert goes, this test needs another
    unexpected preprocessing error."""
    from vllm.v1.engine import FinishReason, StreamAdmissionContext, StreamOutput

    config, client = _async_mp_client(dummy_state_model_path, monkeypatch)
    try:
        await asyncio.wait_for(
            client.admit_stream_async(_raw_admission(config)), TIMEOUT
        )
        failing = _raw_input(config, 1, sampling_params=SamplingParams(max_tokens=None))
        if not context:
            failing.stream_admission = None
        await client.add_request_async(failing)
        refused, batch = await _next_output(client, "raw")
        assert refused.new_token_ids == []
        assert refused.finish_reason == FinishReason.ERROR
        assert refused.ends_stream is False
        if context:
            assert refused.stream_output == StreamOutput(
                failing.stream_admission, "preprocessing_error"
            )
            assert not batch.finished_requests
        else:
            assert refused.stream_output is None
            assert "raw" in batch.finished_requests

        await client.add_request_async(_raw_input(config, 2))
        tokens: list[int] = []
        for _ in range(MAX_STEPS):
            output, _ = await _next_output(client, "raw")
            assert output.error is None
            assert output.stream_output == StreamOutput(
                StreamAdmissionContext(update_seq=2)
            )
            tokens.extend(output.new_token_ids)
            if output.finish_reason is not None:
                break
        hf_config = config.model_config.hf_config
        assert (
            tokens
            == _expected_turns(
                config.model_config.get_vocab_size(), int(hf_config.eos_token_id)
            )[0]
        )
        assert output.finish_reason == FinishReason.STOP
        assert output.ends_stream is False
        await client.abort_requests_async(["raw"])
        await asyncio.wait_for(client.wait_stream_retired_async("raw"), TIMEOUT)
    finally:
        client.shutdown()


@pytest.mark.asyncio
async def test_multiprocess_draining_shutdown_does_not_wait_for_a_never_fed_stream(
    dummy_state_model_path, monkeypatch
):
    """A never-fed stream is not unfinished work, so a shutdown that drains
    does not wait for it: on SIGTERM the engine process exits by itself with
    code 0, well inside the shutdown window and long before the stream's
    never-fed deadline; an engine that waited for the session would still be
    running when the window closes. The client gets no output for the
    session, only the engine's end.

    The never-fed deadline is set through stream_admission_timeouts so that
    it cannot end the session during the test."""
    from vllm.v1.engine.exceptions import EngineDeadError

    window = 60
    config, client = _async_mp_client(
        dummy_state_model_path,
        monkeypatch,
        shutdown_timeout=window,
        stream_admission_timeouts={"never_fed_timeout": 10 * window},
    )
    try:
        result = await asyncio.wait_for(
            client.admit_stream_async(_raw_admission(config)), TIMEOUT
        )
        assert result.accepted
        (process,) = client.resources.engine_manager.processes
        batches = []

        async def read_outputs():
            while True:
                batches.append(await client.get_output_async())

        reader = asyncio.create_task(read_outputs())
        process.terminate()
        await asyncio.to_thread(process.join, window)
        assert process.exitcode == 0
        with pytest.raises(EngineDeadError):
            await asyncio.wait_for(reader, TIMEOUT)
        assert [o for batch in batches for o in batch.outputs] == []
    finally:
        client.shutdown()


# Model-state hooks: execution memory, the model state's own CUDA graphs,
# post-warmup and shutdown. Each test starts one in-process engine and reads
# what the stub's hooks recorded. An engine here is not eager unless the test
# says so; the persistent-state defaults then resolve `cudagraph_mode` to
# NONE, so the only graph is the one the stub captures when asked.

CAPTURING = {"capture_graphs": True}
CONTRIBUTORS = ["stub_capture_mib", "stub_execution_mib"]
FIXED_BUDGET = dict(kv_cache_memory_bytes=8 * MiB)
# 64 pages of 16 GiB: a pool that no device holds.
OVERSIZED_POOL = dict(max_num_seqs=63)
OVERSIZED_PAGE_MIB = 16 * 1024


def _stub_module():
    """The stub's module, with the records of earlier engines of this process
    cleared."""
    import vllm_add_dummy_model.my_persistent_state as stub

    for record in (stub.EVENTS, stub.BINDINGS, stub.RAISED):
        record.clear()
    return stub


def _hook_core(path, options=None, **overrides) -> EngineCore:
    """An in-process engine whose stub has `options`: the stub's `hf_overrides`
    fields, named without their `stub_` prefix."""
    fields = {f"stub_{name}": value for name, value in (options or {}).items()}
    return _engine_core(
        path, **{"enforce_eager": False, "hf_overrides": fields, **overrides}
    )


def _worker(core: EngineCore):
    return core.model_executor.driver_worker.worker


def _calls(stub, hook: str) -> list[dict]:
    """What each call of `hook` recorded, in call order."""
    return [event[1] for event in stub.EVENTS if event[0] == hook]


def _positions(stub, kind: str) -> list[int]:
    return [index for index, event in enumerate(stub.EVENTS) if event[0] == kind]


def _scheduled_steps(stub) -> list:
    """The steps the model ran with `dummy_run=False`."""
    return [e for e in stub.EVENTS if e[0] == "prepare_step" and not e[1]]


@contextmanager
def _vllm_log_records():
    """The records vLLM's loggers emit inside the block."""
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger = logging.getLogger("vllm")
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


def _contributor_notes(error: BaseException) -> list[str]:
    """The notes on `error` that name every declared contributor."""
    return [
        note
        for note in getattr(error, "__notes__", [])
        if all(name in note for name in CONTRIBUTORS)
    ]


# Execution memory.


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("budget", [{}, FIXED_BUDGET], ids=["profiled", "fixed"])
def test_execution_memory_is_prepared_once_before_the_profiling_pass(
    dummy_state_model_path, budget
):
    """The model state's `prepare_execution_memory` runs once, before the
    profiling forward pass and before any state pool exists, whether the KV
    cache budget is profiled or fixed."""
    stub = _stub_module()
    core = _hook_core(dummy_state_model_path, **budget)
    try:
        (prepared,) = _calls(stub, "prepare_execution_memory")
        assert prepared["slots_bound"] is False
        (position,) = _positions(stub, "prepare_execution_memory")
        assert position < _positions(stub, "profile_run")[0]
        assert position < _positions(stub, "forward")[0]
        assert position < _positions(stub, "initialize_state_slots")[0]
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
def test_execution_memory_is_counted_in_the_profiled_budget(dummy_state_model_path):
    """With a profiled budget the hook runs with the current vLLM
    configuration set and before the worker limits allocator block splitting,
    which it does for the profiling pass only; the 1 GiB that the hook leaves
    allocated is not available to the KV cache."""
    stub = _stub_module()
    execution_mib = 1024
    core = _hook_core(dummy_state_model_path, {"execution_mib": execution_mib})
    try:
        (prepared,) = _calls(stub, "prepare_execution_memory")
        assert prepared["config_set"] is True
        profiling = _calls(stub, "profile_run")[0]
        assert "max_split_size_mb" in profiling["allocator_settings"]
        assert prepared["allocator_settings"] != profiling["allocator_settings"]
        assert prepared["allocator_settings"] == stub.allocator_settings()
        worker = _worker(core)
        unavailable = worker.requested_memory - worker.available_kv_cache_memory_bytes
        assert unavailable >= execution_mib * MiB
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
def test_execution_memory_failure_fails_startup(dummy_state_model_path):
    """An exception from `prepare_execution_memory` is the one that fails
    startup, before the model runs."""
    stub = _stub_module()
    with pytest.raises(stub.StubHookError) as failed:
        _hook_core(dummy_state_model_path, {"fail": ["prepare_execution_memory"]})
    assert failed.value is stub.RAISED[0]
    assert _positions(stub, "forward") == []


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("profiling_pass", ["ends", "raises"])
def test_allocator_settings_are_restored_after_the_profiling_pass(
    dummy_state_model_path, profiling_pass
):
    """Once the profiling pass has ended or raised, the allocator settings
    are again those in effect when `prepare_execution_memory` ran."""
    stub = _stub_module()
    if profiling_pass == "raises":
        with pytest.raises(stub.StubHookError):
            _hook_core(dummy_state_model_path, {"fail": ["profile_run"]})
        core = None
    else:
        core = _hook_core(dummy_state_model_path)
    try:
        (prepared,) = _calls(stub, "prepare_execution_memory")
        assert stub.allocator_settings() == prepared["allocator_settings"]
    finally:
        if core is not None:
            core.shutdown()


# The model state's own graphs: whether the runner captures.


@create_new_process_for_each_test("spawn")
def test_capture_predicate_is_first_asked_before_any_state_pool(
    dummy_state_model_path,
):
    """Under NONE with the graph-memory estimate, the runner first asks the
    model state whether it captures before any state binding exists. The
    stub's predicate takes no argument, so the runner passes none."""
    stub = _stub_module()
    core = _hook_core(dummy_state_model_path, CAPTURING)
    try:
        first = _positions(stub, "needs_cudagraph_capture")[0]
        assert first < _positions(stub, "initialize_state_slots")[0]
        assert _calls(stub, "needs_cudagraph_capture")[0]["slots_bound"] is False
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("captures", [True, False])
def test_runner_needs_capture_when_the_model_state_does(
    dummy_state_model_path, captures
):
    """Without eager mode, and with no encoder or decoder graph to capture,
    the runner has graphs to capture exactly when the model state says it
    captures, although `cudagraph_mode` is NONE."""
    _stub_module()
    core = _hook_core(dummy_state_model_path, {"capture_graphs": captures})
    try:
        config = core.vllm_config
        assert config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
        assert _worker(core).model_runner.needs_cudagraph_capture() is captures
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
def test_eager_mode_never_asks_the_capture_predicate(dummy_state_model_path):
    """With `enforce_eager` the runner has nothing to capture and does not
    ask the model state, at startup or later."""
    stub = _stub_module()
    core = _hook_core(dummy_state_model_path, CAPTURING, enforce_eager=True)
    try:
        assert _worker(core).model_runner.needs_cudagraph_capture() is False
        assert _calls(stub, "needs_cudagraph_capture") == []
    finally:
        core.shutdown()


# The capture calls.


@create_new_process_for_each_test("spawn")
def test_estimate_and_warmup_each_capture_exactly_once(dummy_state_model_path):
    """With no encoder or decoder graph, each `capture_model` call still
    invokes `capture_cudagraphs` once: one call from the graph-memory
    estimate, then one from the worker's warmup."""
    stub = _stub_module()
    core = _hook_core(dummy_state_model_path, CAPTURING)
    try:
        captures = _calls(stub, "capture_cudagraphs")
        assert [capture["profile_only"] for capture in captures] == [True, False]
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
def test_estimate_captures_into_its_throwaway_pool(dummy_state_model_path):
    """The estimate's capture call has `profile_only=True` and runs while the
    platform's global graph pool is the estimate's throwaway pool, which is
    not the pool that serves afterwards."""
    stub = _stub_module()
    core = _hook_core(dummy_state_model_path, CAPTURING)
    try:
        profiling = _calls(stub, "capture_cudagraphs")[0]
        assert profiling["profile_only"] is True
        assert profiling["graph_pool"] == profiling["global_graph_pool"]
        assert profiling["graph_pool"] != current_platform.get_global_graph_pool()
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
def test_serving_capture_gets_the_global_pool_and_a_capture_stream(
    dummy_state_model_path,
):
    """The serving call receives `profile_only=False`, the platform's global
    graph pool, and a stream that is current during the call and is not the
    main stream; the runner has left that stream when the next hook runs."""
    stub = _stub_module()
    core = _hook_core(dummy_state_model_path, CAPTURING)
    try:
        serving = _calls(stub, "capture_cudagraphs")[-1]
        assert serving["profile_only"] is False
        assert serving["graph_pool"] == current_platform.get_global_graph_pool()
        assert serving["stream_is_current"] is True
        assert serving["stream_is_main"] is False
        (after,) = _calls(stub, "post_warmup")
        assert after["on_main_stream"] is True
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("budget", [{}, FIXED_BUDGET], ids=["estimate", "no-estimate"])
def test_warmup_captures_once_over_the_serving_bindings(dummy_state_model_path, budget):
    """Whether or not the estimate runs, the worker's warmup makes exactly one
    `profile_only=False` call, while the model state holds the serving
    bindings, before the final dummy run and before `post_warmup`."""
    stub = _stub_module()
    core = _hook_core(dummy_state_model_path, CAPTURING, **budget)
    try:
        captures = _calls(stub, "capture_cudagraphs")
        (serving,) = [c for c in captures if not c["profile_only"]]
        assert serving["bindings"] == len(stub.BINDINGS) - 1
        position = _positions(stub, "capture_cudagraphs")[-1]
        assert position > _positions(stub, "initialize_state_slots")[-1]
        assert position < _positions(stub, "prepare_step")[-1]
        assert position < _positions(stub, "post_warmup")[0]
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
def test_model_state_graphs_are_part_of_the_graph_memory_estimate(
    dummy_state_model_path,
):
    """The 64 MiB that the profiling capture kept is in the worker's
    graph-memory estimate, and the estimate is taken from the KV cache
    budget."""
    stub = _stub_module()
    core = _hook_core(dummy_state_model_path, {**CAPTURING, "capture_mib": 64})
    try:
        profiling = _calls(stub, "capture_cudagraphs")[0]
        assert profiling["profile_only"] is True
        assert profiling["scratch_bytes"] == 64 * MiB
        worker = _worker(core)
        assert worker.cudagraph_memory_estimate >= profiling["scratch_bytes"]
        assert worker.available_kv_cache_memory_bytes <= (
            worker.requested_memory - worker.cudagraph_memory_estimate
        )
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("serving_capture", ["returns", "raises"])
def test_stub_capture_finishes_its_device_work(dummy_state_model_path, serving_capture):
    """The stub keeps its side of the capture contract: when a capture call
    returns or raises, the capture on the runner's stream and the replay on
    the stub's own stream have completed."""
    stub = _stub_module()
    if serving_capture == "raises":
        with pytest.raises(stub.StubHookError):
            _hook_core(
                dummy_state_model_path, {**CAPTURING, "fail": ["capture_serving"]}
            )
        core = None
    else:
        core = _hook_core(dummy_state_model_path, CAPTURING)
    try:
        captures = _calls(stub, "capture_cudagraphs")
        assert [capture["profile_only"] for capture in captures] == [True, False]
        assert [capture["work_completed"] for capture in captures] == [True, True]
    finally:
        if core is not None:
            core.shutdown()


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("call", ["capture_profile", "capture_serving"])
def test_capture_failure_fails_startup(dummy_state_model_path, call):
    """An exception from `capture_cudagraphs`, in the estimate or in the
    warmup, is the one that fails startup."""
    stub = _stub_module()
    with pytest.raises(stub.StubHookError) as failed:
        _hook_core(dummy_state_model_path, {**CAPTURING, "fail": [call]})
    assert failed.value is stub.RAISED[0]
    assert call in str(failed.value)


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("captures", [True, False])
def test_workspace_is_locked_by_a_serving_capture_only(
    dummy_state_model_path, captures
):
    """After the serving capture the workspace is locked, so `post_warmup`
    sees it locked and a new workspace resource is refused; the profiling
    capture leaves it unlocked. When nothing is captured it stays unlocked."""
    from vllm.v1.worker.workspace import current_workspace_manager

    stub = _stub_module()
    core = _hook_core(dummy_state_model_path, {"capture_graphs": captures})
    try:
        assert [c["workspace_locked"] for c in _calls(stub, "clear_cudagraphs")] == (
            [False] if captures else []
        )
        (after,) = _calls(stub, "post_warmup")
        assert after["workspace_locked"] is captures
        manager = current_workspace_manager()
        if captures:
            with pytest.raises(AssertionError, match="locked"):
                manager.get_persistent_resource(("stub", "growth"), list)
        else:
            assert manager.get_persistent_resource(("stub", "growth"), list) == []
    finally:
        core.shutdown()


# The profiling capture: its pool, its release and its failures.


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize(
    ("sizing", "pages"),
    [(dict(max_num_seqs=4), 5), (dict(num_gpu_blocks_override=7), 7)],
    ids=["max_num_seqs", "override"],
)
def test_profiling_pool_has_the_planned_page_count(
    dummy_state_model_path, sizing, pages
):
    """When the model state captures, the estimate's temporary state pool has
    the planned page count, `max_num_seqs + 1` or the override, so the
    profiling capture sees the pool shape of the serving capture."""
    stub = _stub_module()
    core = _hook_core(dummy_state_model_path, CAPTURING, **sizing)
    try:
        profiling, serving = _calls(stub, "capture_cudagraphs")
        assert profiling["profile_only"] is True
        assert profiling["pool_pages"] == pages
        assert serving["pool_pages"] == pages
        assert [binding["pages"] for binding in stub.BINDINGS] == [pages, pages]
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
def test_profiling_pool_allocation_failure_fails_startup_as_it_is(
    dummy_state_model_path,
):
    """When the device cannot hold the temporary pool of the planned size,
    the allocation's own exception fails startup: it is not the
    planned-capacity `ValueError`, it states no `max_num_seqs`, and neither
    the capture nor `clear_cudagraphs` runs."""
    stub = _stub_module()
    with pytest.raises(torch.OutOfMemoryError) as failed:
        _hook_core(
            dummy_state_model_path,
            {**CAPTURING, "page_mib": OVERSIZED_PAGE_MIB},
            **OVERSIZED_POOL,
        )
    assert "max_num_seqs" not in str(failed.value)
    assert _positions(stub, "initialize_state_slots") == []
    assert _calls(stub, "capture_cudagraphs") == []
    assert _calls(stub, "clear_cudagraphs") == []


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("profiling_capture", ["returns", "raises"])
def test_estimate_teardown_clears_the_profiling_capture_once(
    dummy_state_model_path, profiling_capture
):
    """The estimate's teardown calls `clear_cudagraphs` once, after the
    profiling capture returned or raised, while the model state still holds
    the profiling bindings and before the serving pool is bound."""
    stub = _stub_module()
    if profiling_capture == "raises":
        with pytest.raises(stub.StubHookError):
            _hook_core(
                dummy_state_model_path, {**CAPTURING, "fail": ["capture_profile"]}
            )
        core = None
    else:
        core = _hook_core(dummy_state_model_path, CAPTURING)
    try:
        (cleared,) = _calls(stub, "clear_cudagraphs")
        assert cleared["held_graph"] is True
        assert cleared["bindings"] == 0
        (position,) = _positions(stub, "clear_cudagraphs")
        assert position > _positions(stub, "capture_cudagraphs")[0]
        later_bindings = _positions(stub, "initialize_state_slots")[1:]
        assert all(position < later for later in later_bindings)
        assert len(later_bindings) == (0 if core is None else 1)
    finally:
        if core is not None:
            core.shutdown()


@create_new_process_for_each_test("spawn")
def test_failed_release_fails_startup(dummy_state_model_path):
    """With no other exception propagating, the exception from
    `clear_cudagraphs` is the one that fails startup."""
    stub = _stub_module()
    with pytest.raises(stub.StubHookError) as failed:
        _hook_core(dummy_state_model_path, {**CAPTURING, "fail": ["clear_cudagraphs"]})
    assert failed.value is stub.RAISED[0]
    assert "clear_cudagraphs" in str(failed.value)


@create_new_process_for_each_test("spawn")
def test_estimate_teardown_completes_after_a_failed_release(dummy_state_model_path):
    """When `clear_cudagraphs` raises, the teardown still drops the profiling
    bindings and restores the platform's global graph pool: the pool is no
    longer the throwaway pool that the profiling capture was given."""
    stub = _stub_module()
    with pytest.raises(stub.StubHookError):
        _hook_core(dummy_state_model_path, {**CAPTURING, "fail": ["clear_cudagraphs"]})
    (profiling,) = _calls(stub, "capture_cudagraphs")
    assert profiling["graph_pool"] == profiling["global_graph_pool"]
    assert current_platform.get_global_graph_pool() != profiling["graph_pool"]
    gc.collect()
    assert [binding["ref"]() for binding in stub.BINDINGS] == [None]


@create_new_process_for_each_test("spawn")
def test_earlier_failure_outranks_a_failed_release(dummy_state_model_path):
    """When the profiling capture raised and `clear_cudagraphs` then raises
    too, the capture's exception fails startup, with its contributor note,
    and the exception from `clear_cudagraphs` is logged with its traceback."""
    stub = _stub_module()
    options = {
        **CAPTURING,
        "memory_contributors": CONTRIBUTORS,
        "fail": ["capture_profile", "clear_cudagraphs"],
    }
    with _vllm_log_records() as records, pytest.raises(stub.StubHookError) as failed:
        _hook_core(dummy_state_model_path, options)
    capture_error, clear_error = stub.RAISED
    assert failed.value is capture_error
    assert len(_contributor_notes(capture_error)) == 1
    logged = [record for record in records if record.exc_info]
    assert [r for r in logged if r.exc_info[1] is clear_error]
    assert [r for r in logged if r.exc_info[1] is clear_error][0].exc_info[2]


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("estimate", ["fixed-budget", "switched-off"])
def test_without_the_estimate_only_the_serving_capture_runs(
    dummy_state_model_path, estimate
):
    """With a fixed budget, or with the graph-memory estimate switched off,
    there is no `profile_only=True` call and no `clear_cudagraphs` call; the
    serving capture still runs."""
    stub = _stub_module()
    budget = {}
    if estimate == "fixed-budget":
        budget = FIXED_BUDGET
    else:
        os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
    core = _hook_core(dummy_state_model_path, CAPTURING, **budget)
    try:
        captures = _calls(stub, "capture_cudagraphs")
        assert [capture["profile_only"] for capture in captures] == [False]
        assert _calls(stub, "clear_cudagraphs") == []
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
def test_eager_mode_runs_the_other_hooks_but_no_capture(dummy_state_model_path):
    """With `enforce_eager`, a model state that would capture gets neither
    `capture_cudagraphs` nor `clear_cudagraphs`; it still gets
    `prepare_execution_memory`, `post_warmup` and `shutdown`, once each."""
    stub = _stub_module()
    core = _hook_core(dummy_state_model_path, CAPTURING, enforce_eager=True)
    core.shutdown()
    assert _calls(stub, "capture_cudagraphs") == []
    assert _calls(stub, "clear_cudagraphs") == []
    for hook in ("prepare_execution_memory", "post_warmup", "shutdown"):
        assert len(_calls(stub, hook)) == 1, hook


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("captures", [True, False])
def test_serving_bindings_replace_the_profiling_bindings(
    dummy_state_model_path, captures
):
    """When the runner has graphs to capture, `initialize_state_slots` gets
    bindings over the temporary pool and then, once, new bindings over the
    serving pool; by then the runner has dropped the profiling bindings and
    cleared the model state's. Otherwise there is no temporary pool and the
    serving bindings are the only ones."""
    stub = _stub_module()
    core = _hook_core(dummy_state_model_path, {"capture_graphs": captures})
    try:
        if not captures:
            assert len(stub.BINDINGS) == 1
            return
        profiling, serving = stub.BINDINGS
        assert profiling["ref"]() is None
        assert serving["previous_alive"] is False
        assert serving["slots_cleared"] is True
        assert serving["ref"]() is _worker(core).model_runner.state_slots
    finally:
        core.shutdown()


# Startup pool access during the capture calls.


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("serving_capture", ["returns", "raises"])
def test_startup_access_is_open_during_each_capture_call_only(
    dummy_state_model_path, serving_capture
):
    """Each capture call can take startup access to the pools of its
    bindings, and that access is refused once the call has returned or
    raised: at `clear_cudagraphs` after the profiling call, and at
    `post_warmup`, or at `shutdown` after a failure, for the serving call."""
    stub = _stub_module()
    if serving_capture == "raises":
        with pytest.raises(stub.StubHookError):
            _hook_core(
                dummy_state_model_path, {**CAPTURING, "fail": ["capture_serving"]}
            )
        core = None
    else:
        core = _hook_core(dummy_state_model_path, CAPTURING)
    try:
        captures = _calls(stub, "capture_cudagraphs")
        assert [capture["pool_pages"] for capture in captures] == [3, 3]
        (cleared,) = _calls(stub, "clear_cudagraphs")
        assert cleared["stale_access_refused"] is True
        after = "shutdown" if core is None else "post_warmup"
        assert _calls(stub, after)[0]["stale_access_refused"] is True
    finally:
        if core is not None:
            core.shutdown()


# Memory contributors on allocation and capture failures.


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("failure", ["error", "oom"])
def test_capture_failure_notes_the_memory_contributors(dummy_state_model_path, failure):
    """When `capture_cudagraphs` raises, whatever the exception's type, the
    same exception object fails startup with one note that names each memory
    contributor the spec declares. The note gives no advice: it names none of
    the options that size the pool or the budget."""
    stub = _stub_module()
    options = {
        **CAPTURING,
        "memory_contributors": CONTRIBUTORS,
        "fail": ["capture_serving"],
        "fail_oom": failure == "oom",
    }
    expected = torch.OutOfMemoryError if failure == "oom" else stub.StubHookError
    with pytest.raises(expected) as failed:
        _hook_core(dummy_state_model_path, options, **FIXED_BUDGET)
    assert failed.value is stub.RAISED[0]
    (note,) = _contributor_notes(failed.value)
    assert "memory" in note
    for advice in ("max_num_seqs", "gpu_memory_utilization", "kv_cache_memory_bytes"):
        assert advice not in note


@create_new_process_for_each_test("spawn")
def test_capture_failure_without_contributors_gets_no_note(dummy_state_model_path):
    """When the spec declares no memory contributors, the exception from
    `capture_cudagraphs` reaches the caller without a note."""
    stub = _stub_module()
    with pytest.raises(stub.StubHookError) as failed:
        _hook_core(
            dummy_state_model_path,
            {**CAPTURING, "fail": ["capture_serving"]},
            **FIXED_BUDGET,
        )
    assert failed.value is stub.RAISED[0]
    assert getattr(failed.value, "__notes__", []) == []


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("pool", ["temporary", "serving"])
@pytest.mark.parametrize("contributors", [CONTRIBUTORS, []], ids=["named", "none"])
def test_state_pool_allocation_failure_notes_the_memory_contributors(
    dummy_state_model_path, pool, contributors
):
    """When the device cannot hold a state pool, the estimate's temporary
    pool or the serving pool, the allocation's exception carries one note
    naming each declared memory contributor, and none that names a model
    option when the spec declares no contributors."""
    _stub_module()
    options = {
        "page_mib": OVERSIZED_PAGE_MIB,
        "memory_contributors": contributors,
        # Only a model state that captures gets a temporary pool of this size.
        "capture_graphs": pool == "temporary",
    }
    budget = {}
    if pool == "serving":
        # A fixed budget that holds the planned pages, so planning passes.
        budget = dict(kv_cache_memory_bytes=64 * OVERSIZED_PAGE_MIB * MiB)
    with pytest.raises(torch.OutOfMemoryError) as failed:
        _hook_core(dummy_state_model_path, options, **OVERSIZED_POOL, **budget)
    assert len(_contributor_notes(failed.value)) == (1 if contributors else 0)
    if not contributors:
        notes = getattr(failed.value, "__notes__", [])
        assert not [note for note in notes if "stub_" in note]


# Warmup under a persistent-state layout.


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize("captures", [True, False])
def test_warmup_executes_no_scheduled_step(dummy_state_model_path, captures):
    """Under NONE without `enforce_eager`, JIT warmup is enabled, yet the
    model runs no step with `dummy_run=False` during startup, whatever the
    model state's capture predicate returns. Called again on the started
    runner, `warmup_kernels` executes no step, and
    `run_mixed_prefill_decode_warmup` executes none and returns `False`."""
    from vllm.v1.worker.gpu.warmup import (
        run_mixed_prefill_decode_warmup,
        warmup_kernels,
    )

    stub = _stub_module()
    core = _hook_core(dummy_state_model_path, {"capture_graphs": captures})
    try:
        assert core.vllm_config.kernel_config.enable_jit_warmup is True
        assert _scheduled_steps(stub) == []
        worker = _worker(core)
        runner = worker.model_runner
        warmup_kernels(runner, worker.execute_model, worker.sample_tokens)
        assert _scheduled_steps(stub) == []
        ran = run_mixed_prefill_decode_warmup(
            runner,
            worker.execute_model,
            worker.sample_tokens,
            num_tokens=runner.max_num_tokens,
        )
        assert ran is False
        assert _scheduled_steps(stub) == []
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
def test_capturing_model_starts_and_serves_without_eager_mode(dummy_state_model_path):
    """With NONE and without `enforce_eager`, a model state that captures its
    own graph starts through `LLM` and generates the stub's tokens. The
    remaining warmup stages ran: the final dummy run follows the serving
    capture, and JIT monitoring is on, as for any layout with JIT warmup."""
    from vllm_add_dummy_model.my_persistent_state import expected_tokens

    from vllm.utils import jit_monitor

    stub = _stub_module()
    llm = LLM(
        **_engine_kwargs(
            dummy_state_model_path,
            enforce_eager=False,
            compilation_config={"cudagraph_mode": "NONE"},
            hf_overrides={"stub_capture_graphs": True},
        )
    )
    capture = _positions(stub, "capture_cudagraphs")[-1]
    dummy_runs = [i for i, e in enumerate(stub.EVENTS) if e[0] == "forward" and e[1]]
    assert capture > _positions(stub, "initialize_state_slots")[-1]
    assert [i for i in dummy_runs if i > capture]
    assert jit_monitor.is_active() is True

    model = _model(llm.llm_engine.engine_core.engine_core)
    prompts = [[11 * i + 1, 11 * i + 2, 11 * i + 3] for i in range(4)]
    params = SamplingParams(temperature=0, max_tokens=8)
    outputs = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
    assert [list(output.outputs[0].token_ids) for output in outputs] == [
        expected_tokens(0, prompt[-1], model.vocab_size, model.eos_token_id)
        for prompt in prompts
    ]


# Post-warmup.


@create_new_process_for_each_test("spawn")
@pytest.mark.parametrize(
    "configuration",
    [
        dict(enforce_eager=True),
        dict(enforce_eager=False),
        dict(enforce_eager=False, kernel_config={"enable_jit_warmup": False}),
    ],
    ids=["eager", "jit-warmup", "no-jit-warmup"],
)
def test_warmup_calls_post_warmup_once(dummy_state_model_path, configuration):
    """A successful startup calls `post_warmup` exactly once, with or without
    eager mode and JIT warmup."""
    stub = _stub_module()
    core = _hook_core(dummy_state_model_path, CAPTURING, **configuration)
    try:
        assert len(_calls(stub, "post_warmup")) == 1
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
def test_post_warmup_follows_the_model_runs_and_precedes_the_seed_reset(
    dummy_state_model_path,
):
    """`post_warmup` runs after the capture, the final dummy run and the
    sampler warmup: no model run or capture follows it at startup. It runs
    before the seed reset, so the random numbers the stub draws there leave
    the generator as a reset leaves it; and JIT monitoring and GC freezing,
    which follow the reset, are not yet on."""
    from vllm.utils import jit_monitor
    from vllm.utils.torch_utils import set_random_seed

    stub = _stub_module()
    core = _hook_core(dummy_state_model_path, CAPTURING)
    try:
        (position,) = _positions(stub, "post_warmup")
        assert position > _positions(stub, "capture_cudagraphs")[-1]
        assert position > _positions(stub, "forward")[-1]
        assert position > _positions(stub, "prepare_step")[-1]

        (hook,) = _calls(stub, "post_warmup")
        assert hook["jit_monitor_active"] is False
        assert jit_monitor.is_active() is True
        assert gc.get_freeze_count() > hook["gc_frozen_objects"]

        config = core.vllm_config
        after_startup = torch.get_rng_state()
        set_random_seed(
            config.model_config.seed, config.parallel_config.data_parallel_index
        )
        assert torch.equal(after_startup, torch.get_rng_state())
    finally:
        core.shutdown()


@create_new_process_for_each_test("spawn")
def test_post_warmup_failure_fails_startup(dummy_state_model_path):
    """An exception from `post_warmup` is the one that fails startup."""
    stub = _stub_module()
    with pytest.raises(stub.StubHookError) as failed:
        _hook_core(dummy_state_model_path, {"fail": ["post_warmup"]})
    assert failed.value is stub.RAISED[0]
    assert "post_warmup" in str(failed.value)


@create_new_process_for_each_test("spawn")
def test_post_warmup_sees_the_pools_where_the_capture_saw_them(
    dummy_state_model_path,
):
    """The layout snapshot the stub takes in `post_warmup` equals the one it
    took in the serving capture: the pages did not move during warmup."""
    stub = _stub_module()
    core = _hook_core(dummy_state_model_path, CAPTURING)
    try:
        (hook,) = _calls(stub, "post_warmup")
        assert hook["layout_unchanged"] is True
        assert hook["bindings"] == len(stub.BINDINGS) - 1
    finally:
        core.shutdown()


# Shutdown.


@create_new_process_for_each_test("spawn")
def test_runner_shutdown_calls_the_hook_before_it_releases_the_state(
    dummy_state_model_path,
):
    """`GPUModelRunner.shutdown` calls the model state's `shutdown` once the
    device is idle and while the state bindings and their pool still exist."""
    stub = _stub_module()
    core = _hook_core(dummy_state_model_path)
    assert _calls(stub, "shutdown") == []
    core.shutdown()
    (hook,) = _calls(stub, "shutdown")
    assert hook["slots_bound"] is True
    assert hook["pool_pages"] == 3
    assert hook["device_idle"] is True


@create_new_process_for_each_test("spawn")
def test_second_runner_shutdown_does_not_call_the_hook_again(dummy_state_model_path):
    """Once a shutdown has released the model state, shutting the runner down
    again calls no hook."""
    stub = _stub_module()
    core = _hook_core(dummy_state_model_path)
    runner = _worker(core).model_runner
    core.shutdown()
    runner.shutdown()
    assert len(_calls(stub, "shutdown")) == 1


@create_new_process_for_each_test("spawn")
def test_failed_startup_shuts_the_model_state_down(dummy_state_model_path):
    """After a startup that fails once the model state exists, the runner's
    shutdown calls the hook on that model state, once, after everything else
    it did."""
    stub = _stub_module()
    with pytest.raises(stub.StubHookError):
        _hook_core(dummy_state_model_path, {"fail": ["post_warmup"]})
    assert _positions(stub, "shutdown") == [len(stub.EVENTS) - 1]


@create_new_process_for_each_test("spawn")
def test_shutdown_hook_failure_is_logged_and_cleanup_continues(
    dummy_state_model_path,
):
    """An exception from the model state's `shutdown` does not leave
    `GPUModelRunner.shutdown`: it is logged with its traceback, and the runner
    still releases the state bindings and the model state."""
    stub = _stub_module()
    core = _hook_core(dummy_state_model_path, {"fail": ["shutdown"]})
    runner = _worker(core).model_runner
    with _vllm_log_records() as records:
        core.shutdown()
    (error,) = stub.RAISED
    logged = [r for r in records if r.exc_info and r.exc_info[1] is error]
    assert logged and logged[0].exc_info[2] is not None
    assert runner.state_slots is None
    assert not hasattr(runner, "model_state")
