# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent state end to end, with the dummy-model plugin's stub.

Every engine here is built with its real constructors from the plugin's
`StubPersistentStateForCausalLM` (facebook/opt-125m config and tokenizer,
dummy weights), whose output depends only on the state slot it is given and
its inputs. The in-process engine core exposes the scheduler and the worker;
`LLM` and `AsyncLLM` are the public entries. See
docs/design/persistent_state.md for the criteria named in `@spec`.
"""

import asyncio
import re
import time

import pytest
import torch

from tests.utils import create_new_process_for_each_test
from vllm import LLM, SamplingParams
from vllm.config import CUDAGraphMode
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.platforms import current_platform
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.uniproc_executor import UniProcExecutor

if not current_platform.is_cuda():
    pytest.skip(reason="runs engines on a CUDA device", allow_module_level=True)

MiB = 1 << 20
STATE_LAYER = "state"
FIRST_INPUT = [10, 20, 30]
SECOND_INPUT = [40, 50]
MAX_STEPS = 64
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


def _expected_turns(model) -> list[list[int]]:
    """The tokens the stub emits for FIRST_INPUT then SECOND_INPUT on one slot."""
    from vllm_add_dummy_model.my_persistent_state import expected_tokens

    first = expected_tokens(0, FIRST_INPUT[-1], model.vocab_size, model.eos_token_id)
    second = expected_tokens(
        len(first), SECOND_INPUT[-1], model.vocab_size, model.eos_token_id
    )
    return [first, second]


# The LLM regression: default configuration, slots reused across prompts.


# @spec F1
# @spec M5
# @spec M17
# @spec M28
# @spec M29
# @spec M33
@create_new_process_for_each_test()
def test_llm_generates_the_stub_tokens_on_reused_slots(dummy_state_model_path):
    """With max_num_seqs=2 and the persistent-state defaults left unset, six
    greedy prompts each produce the stub's tokens from a fresh counter: a slot
    reused without its release fence or its initializer would change them.
    The defaults resolve to synchronous scheduling, no extensible KV cache and
    no runner CUDA graphs, and releasing the KV cache is refused."""
    from vllm_add_dummy_model.my_persistent_state import expected_tokens

    llm = LLM(**_engine_kwargs(dummy_state_model_path))
    config = llm.llm_engine.vllm_config
    assert config.scheduler_config.async_scheduling is False
    assert config.cache_config.enable_extensible_kv_cache is False
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    assert config.model_config.enforce_eager is False

    model = _model(llm.llm_engine.engine_core.engine_core)
    prompts = [[11 * i + 1, 11 * i + 2, 11 * i + 3] for i in range(6)]
    params = SamplingParams(temperature=0, max_tokens=8)
    outputs = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
    assert [list(output.outputs[0].token_ids) for output in outputs] == [
        expected_tokens(0, prompt[-1], model.vocab_size, model.eos_token_id)
        for prompt in prompts
    ]
    assert {output.outputs[0].finish_reason for output in outputs} == {"stop"}

    with pytest.raises(RuntimeError) as refused:
        llm.release_kv_cache_memory()
    assert str(refused.value) == RELEASE_REFUSAL
    assert "remote KV" not in str(refused.value)


# Startup.


# @spec M7
@create_new_process_for_each_test()
def test_engine_refuses_more_slots_than_the_fixed_budget_holds(
    dummy_state_model_path,
):
    """kv_cache_memory_bytes of three 1 MiB pages under max_num_seqs=4 (five
    pages) fails startup, naming the largest max_num_seqs that fits."""
    with pytest.raises(ValueError) as rejected:
        _engine_core(
            dummy_state_model_path,
            max_num_seqs=4,
            kv_cache_memory_bytes=3 * MiB,
        )
    message = str(rejected.value)
    assert "5 pages" in message and "3 fit" in message, message
    assert "largest max_num_seqs that fits is 2" in message, message


# @spec M31
@create_new_process_for_each_test()
def test_engine_refuses_an_undeclared_state_layer(dummy_undeclared_state_model_path):
    """A model that builds a persistent-state layer without declaring
    `supports_persistent_state` fails startup with a ValueError naming the
    class attribute."""
    with pytest.raises(ValueError, match="supports_persistent_state"):
        _engine_core(dummy_undeclared_state_model_path)


# A resumable stream through the in-process engine core (M17).


def _params(core: EngineCore) -> SamplingParams:
    params = SamplingParams(max_tokens=8, temperature=0)
    params.update_from_generation_config({}, _model(core).eos_token_id)
    return params


def _feed(core, request_id, tokens, resumable=True):
    request = EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=list(tokens),
        mm_features=None,
        sampling_params=_params(core),
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


def _step_until_retired(core, request_id):
    for _ in range(MAX_STEPS):
        if not any(
            r.request_id == request_id for r in core.scheduler.state_retirements
        ):
            return
        core.step_fn()
    raise AssertionError(f"{request_id} did not retire in {MAX_STEPS} steps")


def _snapshot(core):
    """Everything a refused clear must leave as it is."""
    scheduler = core.scheduler
    manager = scheduler.kv_cache_manager.coordinator.state_managers[STATE_LAYER]
    return (
        {rid: manager.get(rid) for rid in scheduler.requests},
        manager.num_free_slots,
        {rid: request.status for rid, request in scheduler.requests.items()},
        [request.request_id for request in scheduler.running],
        len(scheduler.state_retirements),
        scheduler.pause_state,
    )


def _stream_both_turns(core, request_id, between_turns=None) -> list[list[int]]:
    _feed(core, request_id, FIRST_INPUT)
    first = _run_turn(core, request_id)
    if between_turns is not None:
        between_turns()
    _feed(core, request_id, SECOND_INPUT)
    second = _run_turn(core, request_id)
    core.abort_requests([request_id])
    _step_until_retired(core, request_id)
    return [first, second]


def _park(core, request_id):
    _feed(core, request_id, FIRST_INPUT)
    _run_turn(core, request_id)


# @spec M17
@create_new_process_for_each_test()
def test_in_process_parked_stream_refusals_and_keep_pause(dummy_state_model_path):
    """While a stream is parked, a keep pause that clears the cache, a reset
    of running requests and a release are refused with the exact messages
    before anything changes; a keep pause without a clear is taken and the
    stream then continues with the tokens of an unpaused stream."""
    core = _engine_core(dummy_state_model_path)
    try:
        reference = _stream_both_turns(core, "reference")
        assert reference == _expected_turns(_model(core))

        def pause_and_resume():
            before = _snapshot(core)
            with pytest.raises(RuntimeError) as refused:
                core.pause_scheduler(mode="keep", clear_cache=True)
            assert str(refused.value) == PAUSE_REFUSAL.format(n=1)
            assert _snapshot(core) == before
            with pytest.raises(RuntimeError) as refused:
                core.reset_prefix_cache(reset_running_requests=True)
            assert str(refused.value) == RESET_REFUSAL.format(n=1)
            assert _snapshot(core) == before
            with pytest.raises(RuntimeError) as refused:
                core.release_kv_cache_memory()
            assert str(refused.value) == RELEASE_REFUSAL
            assert _snapshot(core) == before
            assert core.pause_scheduler(mode="keep", clear_cache=False) is None
            core.resume_scheduler()
            assert _snapshot(core) == before

        assert _stream_both_turns(core, "paused", pause_and_resume) == reference
    finally:
        core.shutdown()


# @spec M17
@create_new_process_for_each_test()
def test_in_process_abort_pause_drains_and_clears(dummy_state_model_path):
    """The default abort pause ends a parked stream, steps the engine until
    its slot has retired and clears the cache; a new stream then starts from
    the initializer."""
    core = _engine_core(dummy_state_model_path)
    try:
        _park(core, "aborted")
        assert core.pause_scheduler() is None
        assert "aborted" not in core.scheduler.requests
        assert not core.scheduler.state_retirements
        manager = core.scheduler.kv_cache_manager.coordinator.state_managers[
            STATE_LAYER
        ]
        assert manager.num_free_slots == 2
        core.resume_scheduler()
        assert _stream_both_turns(core, "next") == _expected_turns(_model(core))
    finally:
        core.shutdown()


# @spec M17
@create_new_process_for_each_test()
@pytest.mark.parametrize("reset_running_requests", [True, False])
def test_in_process_reset_drains_retiring_slots(
    dummy_state_model_path, reset_running_requests
):
    """With no unfinished request and only retiring slots, the in-process
    reset steps the engine until no slot is retiring and then resets."""
    core = _engine_core(dummy_state_model_path)
    try:
        _park(core, "ended")
        core.abort_requests(["ended"])
        assert core.scheduler.state_retirements
        assert core.reset_prefix_cache(reset_running_requests=reset_running_requests)
        assert not core.scheduler.state_retirements
    finally:
        core.shutdown()


# @spec M17
@create_new_process_for_each_test()
def test_in_process_reset_with_a_queued_request_asks_for_a_retry(
    dummy_state_model_path,
):
    """With a slot retiring and a request queued, the in-process reset raises
    the retry message without stepping."""
    core = _engine_core(dummy_state_model_path)
    try:
        _park(core, "ended")
        core.abort_requests(["ended"])
        _feed(core, "queued", FIRST_INPUT, resumable=False)
        before = _snapshot(core)
        with pytest.raises(RuntimeError) as refused:
            core.reset_prefix_cache(reset_running_requests=True)
        assert str(refused.value) == (
            "persistent state: 1 state slot(s) are still retiring; retry the "
            "reset once they have retired."
        )
        assert _snapshot(core) == before
    finally:
        core.shutdown()


# The multiprocess engine through AsyncLLM (M17).


@pytest.mark.asyncio
async def test_async_llm_parked_stream_across_pauses(
    dummy_state_model_path, monkeypatch
):
    """While a stream is parked, keep and wait pauses that clear the cache and
    a reset of running requests are refused; a keep pause without a clear
    preserves the stream. After the stream ends, keep and wait pauses that
    clear the cache complete once its slot has retired, and a new stream then
    starts from the initializer."""
    from vllm_add_dummy_model.my_persistent_state import expected_tokens

    from vllm.engine.protocol import StreamingInput
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    args = AsyncEngineArgs(
        **_engine_kwargs(dummy_state_model_path, **RESOLVED_DEFAULTS)
    )
    llm = AsyncLLM.from_engine_args(args)
    try:
        hf = llm.model_config.hf_config
        vocab, eos = llm.model_config.get_vocab_size(), int(hf.eos_token_id)
        first = expected_tokens(0, FIRST_INPUT[-1], vocab, eos)
        second = expected_tokens(len(first), SECOND_INPUT[-1], vocab, eos)
        params = SamplingParams(
            max_tokens=8, temperature=0, output_kind=RequestOutputKind.DELTA
        )

        async def stream(request_id, gate: asyncio.Event, parked: asyncio.Event):
            async def inputs():
                yield StreamingInput(prompt=TokensPrompt(prompt_token_ids=FIRST_INPUT))
                await gate.wait()
                yield StreamingInput(prompt=TokensPrompt(prompt_token_ids=SECOND_INPUT))

            turns: list[list[int]] = [[]]
            async for output in llm.generate(inputs(), params, request_id):
                completion = output.outputs[0]
                turns[-1].extend(completion.token_ids)
                if completion.finish_reason is not None and not output.finished:
                    turns.append([])
                    parked.set()
            return [turn for turn in turns if turn]

        async def run(request_id, while_parked):
            gate, parked = asyncio.Event(), asyncio.Event()
            task = asyncio.create_task(stream(request_id, gate, parked))
            await asyncio.wait_for(parked.wait(), 60)
            await while_parked()
            gate.set()
            return await asyncio.wait_for(task, 60)

        async def nothing():
            return None

        assert (await run("reference", nothing))[:2] == [first, second]

        async def refused_then_kept():
            pause_refusal = re.escape(PAUSE_REFUSAL.format(n=1))
            for mode in ("keep", "wait"):
                with pytest.raises(Exception, match=pause_refusal):
                    await llm.pause_generation(mode=mode, clear_cache=True)
            with pytest.raises(Exception, match=re.escape(RESET_REFUSAL.format(n=1))):
                await llm.reset_prefix_cache(reset_running_requests=True)
            await llm.pause_generation(mode="keep", clear_cache=False)
            await llm.resume_generation()

        assert (await run("paused", refused_then_kept))[:2] == [first, second]

        # Only retiring slots: a clearing pause completes once they retire,
        # including a keep pause whose finish waits for the engine to idle.
        for mode in ("keep", "wait"):
            gate, parked = asyncio.Event(), asyncio.Event()
            task = asyncio.create_task(stream(f"ended-{mode}", gate, parked))
            await asyncio.wait_for(parked.wait(), 60)
            await llm.abort(f"ended-{mode}")
            await asyncio.wait_for(
                llm.pause_generation(mode=mode, clear_cache=True), 60
            )
            await llm.resume_generation()
            task.cancel()
            assert (await run(f"after-{mode}", nothing))[:2] == [first, second]
    finally:
        llm.shutdown()


# The release fence on the worker (F3).


# @spec F3
def test_completion_waits_for_a_registered_writing_stream():
    """A write enqueued on a registered auxiliary stream after a delay is
    complete by the time the step's completion is collected: the worker joins
    every registered stream into the main stream before recording the event."""
    from tests.v1.core.utils import create_state_spec
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, StateSlotHandle
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.worker.gpu.state_slots import StateSlotBindings

    spec = create_state_spec()
    pool = torch.zeros(3 * spec.page_size_bytes, dtype=torch.uint8, device="cuda")
    bindings = StateSlotBindings(
        [KVCacheGroupSpec([STATE_LAYER], spec)], {STATE_LAYER: pool}
    )
    main = torch.cuda.current_stream()
    aux = torch.cuda.Stream()
    bindings.register_writing_stream(aux)
    handle = StateSlotHandle("epoch", 1, 1)
    step = SchedulerOutput.make_empty()
    step.state_step_seq = 1
    step.state_slot_handles = {"r": {STATE_LAYER: handle}}
    step.state_cleanup_handles = {}
    bindings.begin_step(step, main)
    bindings.activate_requests()
    rows = bindings.bind_rows(STATE_LAYER, ("r",), (handle,))
    count = rows.component_pools["count"]
    written = torch.cuda.Event()
    with torch.cuda.stream(aux):
        torch.cuda._sleep(200_000_000)
        count[1].fill_(7)
        written.record(aux)
    completion = bindings.finish_step(ModelRunnerOutput(req_ids=["r"]), main)
    output = completion.get_output()
    assert written.query()
    assert output.state_completion.handles == {"r": {STATE_LAYER: handle}}
    assert count[1].tolist() == [7, 7]
