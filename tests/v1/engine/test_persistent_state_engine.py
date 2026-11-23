# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent state at the engine layer: configuration refusals, `AsyncLLM`
stream handling, engine-core admission, preservation and deadlines.

Scheduler-side rules run on a real scheduler whose only KV cache group is one
`MambaSpec` state group; engine-core rules run on a bare `EngineCoreProc`.
"""

import asyncio
import queue
import re
from collections import deque
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from tests.v1.core.utils import EOS_TOKEN_ID, create_requests, create_scheduler
from vllm.config import SchedulerConfig
from vllm.config.kv_transfer import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.protocol import StreamingInput
from vllm.exceptions import VLLMValidationError
from vllm.inputs import TokensPrompt
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.outputs import RequestError
from vllm.sampling_params import (
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType, FinishReason
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import (
    CheckpointState,
    EngineCore,
    EngineCoreProc,
    EngineShutdownState,
)
from vllm.v1.engine.exceptions import StreamAdmissionError
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

STATE_ENGINE_KWARGS = dict(
    model="facebook/opt-125m",
    hf_overrides={"architectures": ["StubPersistentStateForCausalLM"]},
    load_format="dummy",
    max_model_len=256,
    max_num_seqs=2,
)


def _state_spec(**kwargs) -> MambaSpec:
    kwargs.setdefault("shapes", ((1,),))
    kwargs.setdefault("dtypes", (torch.int64,))
    kwargs.setdefault("mamba_type", MambaAttentionBackendEnum.PERSISTENT_STATE)
    return MambaSpec(block_size=64, **kwargs)


# Configuration


@pytest.fixture(scope="module", autouse=True)
def _load_stub_model_plugin():
    from vllm.plugins import load_general_plugins

    load_general_plugins()


# @spec PSTATE-DECL-011
def test_state_model_disables_prefix_caching_by_default():
    config = EngineArgs(**STATE_ENGINE_KWARGS).create_engine_config()
    assert config.cache_config.enable_prefix_caching is False


# @spec PSTATE-DECL-005
@pytest.mark.parametrize(
    ("option", "kwargs"),
    [
        ("enable_prefix_caching", dict(enable_prefix_caching=True)),
        (
            "kv_transfer_config",
            dict(
                kv_transfer_config=KVTransferConfig(
                    kv_connector="ExampleConnector", kv_role="kv_both"
                )
            ),
        ),
        (
            "speculative_config",
            dict(speculative_config={"method": "ngram", "num_speculative_tokens": 2}),
        ),
        ("pipeline_parallel_size", dict(pipeline_parallel_size=2)),
        ("data_parallel_size", dict(data_parallel_size=2)),
        (
            "tensor_parallel_size",
            dict(tensor_parallel_size=2, distributed_executor_backend="mp"),
        ),
        (
            "multi-dimensional rotary positions",
            dict(
                hf_overrides={
                    "architectures": ["StubPersistentStateForCausalLM"],
                    "mrope_section": [1, 1, 1],
                }
            ),
        ),
    ],
)
def test_state_model_refuses_unsupported_option(option, kwargs):
    args = {**STATE_ENGINE_KWARGS, **kwargs}
    with pytest.raises(ValueError, match=f"{option}.* is not supported for this model"):
        EngineArgs(**args).create_engine_config()


# @spec PSTATE-DECL-006
def test_state_model_accepts_async_scheduling_and_cuda_graphs():
    config = EngineArgs(
        **STATE_ENGINE_KWARGS,
        async_scheduling=True,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
    ).create_engine_config()
    assert config.scheduler_config.async_scheduling is True


# @spec PSTATE-DECL-003
def test_state_group_refuses_other_mamba_types():
    spec = _state_spec(mamba_type=MambaAttentionBackendEnum.SHORT_CONV)
    with pytest.raises(ValueError, match="PERSISTENT_STATE"):
        MambaManager.layer_config(spec, recomputable=False)


# @spec PSTATE-DECL-009
def test_state_group_refuses_initial_values_of_the_wrong_length():
    spec = _state_spec(initial_values=(0, 1))
    with pytest.raises(ValueError, match="initial_values"):
        MambaManager.layer_config(spec, recomputable=False)


# @spec PSTATE-HIST-006
def test_pending_stream_input_bound_defaults_to_eight():
    assert SchedulerConfig().max_pending_stream_inputs == 8


# @spec PSTATE-ADMIT-007
def test_stream_admission_timeouts_default():
    assert SchedulerConfig().stream_admission_timeouts == {
        "never_fed_timeout": 15.0,
        "parked_idle_timeout": 60.0,
        "draining_timeout": 40.0,
    }


# AsyncLLM streams


def _core_request(request_id: str, prompt_token_ids: list[int], **kwargs):
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=kwargs.pop("mm_features", None),
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        **kwargs,
    )


def _state_llm(monkeypatch: pytest.MonkeyPatch, mm_features=None) -> AsyncLLM:
    """An `AsyncLLM` over a state model whose requests go to a mocked engine."""
    llm = AsyncLLM.__new__(AsyncLLM)
    monkeypatch.setattr(
        llm, "get_supported_tasks", AsyncMock(return_value=("generate",))
    )
    monkeypatch.setattr(llm, "_run_output_handler", MagicMock())
    monkeypatch.setattr(llm, "_add_request", AsyncMock())
    monkeypatch.setattr(
        "vllm.v1.engine.async_llm.extract_prompt_components",
        lambda model_config, prompt: ("", None, None),
    )
    llm.vllm_config = MagicMock(is_mm_encoder_only=False)
    llm.model_config = MagicMock(is_diffusion=False, has_persistent_state=True)
    llm.engine_core = MagicMock(client_index=3)
    llm.engine_core.wait_stream_retired_async = AsyncMock()
    llm.output_processor = MagicMock()
    llm.output_processor.has_request.return_value = True
    llm.input_processor = MagicMock()

    def process_inputs(*, request_id, prompt, resumable=False, **_):
        ids = list(prompt["prompt_token_ids"]) if resumable else [0]
        return _core_request(
            request_id, ids, resumable=resumable, mm_features=mm_features
        )

    llm.input_processor.process_inputs.side_effect = process_inputs
    return llm


def _chunks(*token_ids: list[int]):
    async def chunks():
        for ids in token_ids:
            yield StreamingInput(prompt=TokensPrompt(prompt_token_ids=list(ids)))

    return chunks()


async def _open_stream(llm: AsyncLLM, external_id: str, chunks, **kwargs):
    queue = await llm._add_streaming_input_request(
        external_id, chunks, SamplingParams(max_tokens=4), **kwargs
    )
    task = queue._input_stream_task
    assert task is not None
    await task
    return queue


def _sent(llm: AsyncLLM) -> list[EngineCoreRequest]:
    return [call.args[0] for call in llm._add_request.await_args_list]


# @spec PSTATE-DECL-007
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("parameter", "params"),
    [
        ("presence_penalty", dict(presence_penalty=0.5)),
        ("frequency_penalty", dict(frequency_penalty=0.5)),
        ("repetition_penalty", dict(repetition_penalty=1.1)),
        ("bad_words", dict(bad_words=["no"])),
        (
            "structured_outputs",
            dict(structured_outputs=StructuredOutputsParams(json_object=True)),
        ),
        ("prompt_logprobs", dict(prompt_logprobs=1)),
        ("thinking_token_budget", dict(thinking_token_budget=8)),
        ("n", dict(n=2)),
        ("output_kind", dict(output_kind=RequestOutputKind.FINAL_ONLY)),
        ("output_kind", dict(output_kind=RequestOutputKind.CUMULATIVE)),
    ],
)
async def test_state_stream_refuses_sampling_parameter_before_reading_input(
    monkeypatch, parameter, params
):
    llm = _state_llm(monkeypatch)
    consumed = []

    async def chunks():
        consumed.append(True)
        yield StreamingInput(prompt=TokensPrompt(prompt_token_ids=[1]))

    sampling_params = SamplingParams(max_tokens=4, **params)
    with pytest.raises(
        VLLMValidationError, match=f"{parameter}.* is not supported for this model"
    ):
        await llm._add_streaming_input_request("ext", chunks(), sampling_params)
    assert not consumed
    llm._add_request.assert_not_awaited()


# @spec PSTATE-ADMIT-001
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason", ["request_id_in_use", "engine_paused", "active_limit"]
)
async def test_stream_admission_refusal_raises_before_reading_input(
    monkeypatch, reason
):
    llm = _state_llm(monkeypatch)
    consumed = []

    async def refuse(request, prompt, parent_req, index, collector):
        collector.put(StreamAdmissionError(reason))

    llm._add_request.side_effect = refuse

    async def chunks():
        consumed.append(True)
        yield StreamingInput(prompt=TokensPrompt(prompt_token_ids=[1]))

    with pytest.raises(StreamAdmissionError) as excinfo:
        await llm._add_streaming_input_request(
            "ext", chunks(), SamplingParams(max_tokens=4), stream_admission=True
        )
    assert excinfo.value.reason == reason
    assert not consumed


# @spec PSTATE-ADMIT-006
@pytest.mark.asyncio
async def test_state_stream_id_is_external_id_client_index_and_counter(monkeypatch):
    llm = _state_llm(monkeypatch)
    ids = []
    for _ in range(2):
        queue = await _open_stream(llm, "ext", _chunks([1]))
        ids.append(queue.request_id)
    numbers = []
    for request_id in ids:
        match = re.fullmatch(r"ext-3-(\d+)", request_id)
        assert match, request_id
        numbers.append(int(match.group(1)))
    assert numbers[1] == numbers[0] + 1


# @spec PSTATE-ADMIT-006
@pytest.mark.asyncio
async def test_state_stream_inputs_and_close_carry_the_stream_id(monkeypatch):
    llm = _state_llm(monkeypatch)
    queue = await _open_stream(llm, "ext", _chunks([1], [2]))
    sent = _sent(llm)
    assert len(sent) == 3
    assert {request.request_id for request in sent} == {queue.request_id}


# @spec PSTATE-ADMIT-005
@pytest.mark.asyncio
async def test_wait_stream_retired_waits_for_the_bound_stream(monkeypatch):
    llm = _state_llm(monkeypatch)
    queue = await _open_stream(llm, "ext", _chunks([1]))
    await asyncio.wait_for(llm.wait_stream_retired("ext"), 1)
    llm.engine_core.wait_stream_retired_async.assert_awaited_once_with(queue.request_id)


# @spec PSTATE-ADMIT-005
@pytest.mark.asyncio
async def test_wait_stream_retired_resolves_at_once_for_an_unknown_id(monkeypatch):
    llm = _state_llm(monkeypatch)
    await asyncio.wait_for(llm.wait_stream_retired("never-admitted"), 1)
    llm.engine_core.wait_stream_retired_async.assert_not_awaited()


# @spec PSTATE-TURN-005
@pytest.mark.asyncio
async def test_inputs_after_the_opener_and_close_carry_continuation(monkeypatch):
    llm = _state_llm(monkeypatch)
    await _open_stream(llm, "ext", _chunks([1], [2], [3]))
    sent = _sent(llm)
    assert [(r.resumable, r.continuation) for r in sent] == [
        (True, False),
        (True, True),
        (True, True),
        (False, True),
    ]


# @spec PSTATE-TURN-004
@pytest.mark.asyncio
async def test_input_for_a_terminally_finished_stream_is_not_forwarded(monkeypatch):
    llm = _state_llm(monkeypatch)

    async def finish_after_first_input(request, *args):
        llm.output_processor.has_request.return_value = False

    llm._add_request.side_effect = finish_after_first_input
    await _open_stream(llm, "ext", _chunks([1], [2], [3]))
    assert [request.prompt_token_ids for request in _sent(llm)] == [[1]]


# @spec PSTATE-TURN-007
@pytest.mark.asyncio
async def test_empty_update_never_reaches_the_output_processor_or_engine(
    monkeypatch,
):
    llm = _state_llm(monkeypatch)
    await _open_stream(llm, "ext", _chunks([1], [], [2]))
    assert [request.prompt_token_ids for request in _sent(llm)] == [[1], [2], [0]]


# @spec PSTATE-TURN-007
@pytest.mark.asyncio
async def test_update_with_only_features_is_forwarded(monkeypatch):
    feature = MagicMock(spec=MultiModalFeatureSpec)
    llm = _state_llm(monkeypatch, mm_features=[feature])
    await _open_stream(llm, "ext", _chunks([1], []))
    assert len(_sent(llm)) == 3


# Scheduler-side stream rules


def _state_scheduler(**kwargs):
    layer_config = MambaManager.layer_config(_state_spec(), recomputable=False)
    return create_scheduler(
        max_model_len=64, block_size=64, kv_cache_layer_config=layer_config, **kwargs
    )


def _stream_request(request_id: str, num_tokens: int = 3, **kwargs):
    continuation = kwargs.pop("continuation", False)
    resumable = kwargs.pop("resumable", True)
    (request,) = create_requests(
        num_requests=1,
        num_tokens=num_tokens,
        req_ids=[request_id],
        block_size=64,
        **kwargs,
    )
    request.resumable = resumable
    request.continuation = continuation
    return request


def _runner_output(req_ids: list[str], sampled: list[list[int]]) -> ModelRunnerOutput:
    return ModelRunnerOutput(req_ids=req_ids, sampled_token_ids=sampled)


def _step(scheduler, req_ids: list[str], sampled: list[list[int]]):
    output = scheduler.schedule()
    return scheduler.update_from_output(output, _runner_output(req_ids, sampled))


def _park(scheduler, request_id: str):
    request = _stream_request(request_id)
    scheduler.add_request(request)
    _step(scheduler, [request_id], [[EOS_TOKEN_ID]])
    assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    return request


# @spec PSTATE-DECL-014
def test_update_with_refused_sampling_parameters_finishes_the_stream():
    scheduler = _state_scheduler()
    request = _park(scheduler, "s")
    update = _stream_request(
        "s",
        2,
        continuation=True,
        sampling_params=SamplingParams(presence_penalty=0.5, max_tokens=4),
    )
    scheduler.add_request(update)

    outputs = _step(scheduler, [], [])[0].outputs

    [output] = [o for o in outputs if o.request_id == "s"]
    assert output.finish_reason == FinishReason.ERROR
    assert output.error.code == "unsupported_sampling_parameter"
    assert not output.error.retryable
    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.num_prompt_tokens == 3


# @spec PSTATE-TURN-003
@pytest.mark.parametrize(
    ("max_tokens", "token", "terminal"),
    [(1, 5, True), (16, EOS_TOKEN_ID, False)],
    ids=["length", "eos-parks"],
)
def test_terminal_finish_carries_the_terminal_flag(max_tokens, token, terminal):
    scheduler = _state_scheduler()
    scheduler.add_request(_stream_request("s", max_tokens=max_tokens))
    [output] = _step(scheduler, ["s"], [[token]])[0].outputs
    assert output.terminal is terminal


# @spec PSTATE-ADMIT-001
def test_admission_refuses_beyond_the_active_limit():
    scheduler = _state_scheduler(max_num_seqs=1)
    assert scheduler.admit_stream(_stream_request("a", 0)) is None
    assert scheduler.admit_stream(_stream_request("b", 0)) == "active_limit"


# @spec PSTATE-ADMIT-012
def test_admission_refuses_an_external_id_held_by_a_stream():
    scheduler = _state_scheduler()
    first = _stream_request("ext-0-1", 0)
    first.external_req_id = "ext"
    second = _stream_request("ext-0-2", 0)
    second.external_req_id = "ext"
    assert scheduler.admit_stream(first) is None
    assert scheduler.admit_stream(second) == "request_id_in_use"


# @spec PSTATE-ADMIT-012
def test_admission_refuses_an_internal_id_held_by_a_stream():
    scheduler = _state_scheduler()
    assert scheduler.admit_stream(_stream_request("s", 0)) is None
    assert scheduler.admit_stream(_stream_request("s", 0)) == "request_id_in_use"


# @spec PSTATE-ADMIT-013
def test_input_beyond_the_pending_bound_ends_the_stream_with_a_named_error():
    scheduler = _state_scheduler()
    request = _stream_request("s")
    scheduler.add_request(request)
    scheduler.schedule()
    bound = scheduler.scheduler_config.max_pending_stream_inputs
    for _ in range(bound + 1):
        scheduler.add_request(_stream_request("s", 1, continuation=True))

    outputs = _step(scheduler, [], [])[0].outputs

    [output] = [o for o in outputs if o.request_id == "s"]
    assert output.finish_reason == FinishReason.ERROR
    assert output.error.code == "pending_input_limit"
    assert not scheduler.has_request("s")


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _clocked_scheduler(**kwargs):
    scheduler = _state_scheduler(**kwargs)
    scheduler.clock = _Clock()
    return scheduler


# @spec PSTATE-ADMIT-007
def test_never_fed_stream_is_aborted_after_its_deadline():
    scheduler = _clocked_scheduler()
    request = _stream_request("s", 0)
    scheduler.admit_stream(request)

    scheduler.clock.now += 14.9
    assert scheduler.sweep_stream_deadlines() == []
    scheduler.clock.now += 0.2
    assert scheduler.sweep_stream_deadlines() == [request]
    assert request.status == RequestStatus.FINISHED_ABORTED
    assert request.stop_reason == "never_fed_timeout"


# @spec PSTATE-ADMIT-007
def test_parked_stream_is_aborted_after_its_idle_deadline():
    scheduler = _clocked_scheduler()
    request = _park(scheduler, "s")

    scheduler.clock.now += 59.9
    assert scheduler.sweep_stream_deadlines() == []
    scheduler.clock.now += 0.2
    assert scheduler.sweep_stream_deadlines() == [request]
    assert request.status == RequestStatus.FINISHED_ABORTED
    assert request.stop_reason == "parked_idle_timeout"


# @spec PSTATE-ADMIT-007
def test_stream_unfinished_after_close_is_aborted_after_its_draining_deadline():
    scheduler = _clocked_scheduler()
    request = _stream_request("s")
    scheduler.add_request(request)
    scheduler.schedule()
    scheduler.add_request(_stream_request("s", 1, resumable=False, continuation=True))

    scheduler.clock.now += 39.9
    assert scheduler.sweep_stream_deadlines() == []
    scheduler.clock.now += 0.2
    assert scheduler.sweep_stream_deadlines() == [request]
    assert request.stop_reason == "draining_timeout"


# @spec PSTATE-ADMIT-009
@pytest.mark.parametrize("pause_state", [PauseState.PAUSED_NEW, PauseState.PAUSED_ALL])
def test_deadlines_do_not_advance_while_paused(pause_state):
    scheduler = _clocked_scheduler()
    request = _stream_request("s", 0)
    scheduler.admit_stream(request)

    scheduler.clock.now += 10
    scheduler.set_pause_state(pause_state)
    scheduler.clock.now += 500
    scheduler.set_pause_state(PauseState.UNPAUSED)

    assert scheduler.sweep_stream_deadlines() == []
    scheduler.clock.now += 5.1
    assert scheduler.sweep_stream_deadlines() == [request]


# Engine core


def _state_core(held: list[str] | None = None) -> EngineCoreProc:
    """A bare engine core over a mocked scheduler that holds `held` streams."""
    held = [] if held is None else held
    core = object.__new__(EngineCoreProc)
    core._checkpoint_state = CheckpointState.RUNNING
    core._checkpoint_error = None
    core._kv_cache_reclaimable = False
    core.vllm_config = MagicMock()
    core.model_executor = MagicMock()
    core.model_executor.is_sleeping = False
    core.model_executor.vllm_config.cache_config.enable_extensible_kv_cache = False
    core.scheduler = MagicMock()
    core.scheduler.has_unrecomputable_state = True
    core.scheduler.has_requests.return_value = False
    core.scheduler.has_request.return_value = False
    core.scheduler.held_state_streams.side_effect = lambda: list(held)
    core.scheduler.reset_prefix_cache.return_value = True
    core.scheduler.admit_stream.return_value = None
    core.batch_queue = None
    core.engines_running = False
    core.reject_new_requests = False
    core.shutdown_state = EngineShutdownState.RUNNING
    core._idle_state_callbacks = []
    core.add_request = MagicMock()
    core._send_finish_outputs_to_client = MagicMock()
    core.held = held
    return core


def _add(core: EngineCoreProc, **fields) -> MagicMock:
    request = MagicMock(
        request_id="s",
        client_index=1,
        abort_immediately=False,
        kv_transfer_params=None,
        continuation=False,
        stream_admission=False,
        resumable=True,
    )
    for name, value in fields.items():
        setattr(request, name, value)
    core._handle_client_request(EngineCoreRequestType.ADD, (request, 0))
    return request


def _refusal(core: EngineCoreProc) -> tuple[list[str], int, FinishReason, RequestError]:
    [call] = core._send_finish_outputs_to_client.call_args_list
    return call.args


# @spec PSTATE-TURN-006
def test_continuation_for_an_unheld_stream_is_refused():
    core = _state_core()
    _add(core, continuation=True)
    req_ids, client_index, reason, error = _refusal(core)
    assert (req_ids, client_index, reason) == (["s"], 1, FinishReason.ERROR)
    assert error.code == "stream_not_found"
    core.add_request.assert_not_called()


# @spec PSTATE-TURN-006
def test_continuation_for_a_held_stream_is_added():
    core = _state_core()
    core.scheduler.has_request.return_value = True
    request = _add(core, continuation=True)
    core.add_request.assert_called_once_with(request, 0)
    core._send_finish_outputs_to_client.assert_not_called()


# @spec PSTATE-TURN-013
def test_close_for_an_unheld_stream_is_dropped_silently():
    core = _state_core()
    _add(core, continuation=True, resumable=False)
    core.add_request.assert_not_called()
    core._send_finish_outputs_to_client.assert_not_called()


# @spec PSTATE-ADMIT-001
@pytest.mark.parametrize(
    ("reason", "retryable"),
    [("request_id_in_use", False), ("active_limit", False), ("engine_paused", True)],
)
def test_refused_stream_admission_is_delivered_with_its_reason(reason, retryable):
    core = _state_core()
    core.scheduler.admit_stream.return_value = reason
    _add(core, stream_admission=True)
    req_ids, client_index, finish_reason, error = _refusal(core)
    assert (req_ids, client_index, finish_reason) == (["s"], 1, FinishReason.ERROR)
    assert (error.code, error.retryable) == (reason, retryable)


# @spec PSTATE-ADMIT-001
def test_accepted_stream_admission_sends_no_output():
    core = _state_core()
    _add(core, stream_admission=True)
    core.scheduler.admit_stream.assert_called_once()
    core._send_finish_outputs_to_client.assert_not_called()


# @spec PSTATE-ADMIT-011
def test_stream_admission_is_refused_for_an_ordinary_model():
    core = _state_core()
    core.scheduler.has_unrecomputable_state = False
    _add(core, stream_admission=True)
    *_, error = _refusal(core)
    assert error.code == "validation_error"
    assert "stream admission is not supported for this model" in error.message
    core.scheduler.admit_stream.assert_not_called()


# @spec PSTATE-PRES-004
def test_cache_reset_is_refused_while_a_stream_is_held():
    core = _state_core(["stream-1"])
    with pytest.raises(RuntimeError, match="stream-1"):
        core._reset_caches()
    core.scheduler.reset_prefix_cache.assert_not_called()


# @spec PSTATE-PRES-004
@pytest.mark.parametrize("mode", ["wait", "keep"])
def test_clearing_pause_is_refused_before_any_pause_state_changes(mode):
    core = _state_core(["stream-1"])
    with pytest.raises(RuntimeError, match="stream-1"):
        core.pause_scheduler(mode=mode, clear_cache=True)
    core.scheduler.set_pause_state.assert_not_called()


# @spec PSTATE-PRES-004
@pytest.mark.parametrize("mode", ["wait", "keep"])
def test_clearing_sleep_is_refused_before_any_pause_state_changes(mode):
    core = _state_core(["stream-1"])
    with pytest.raises(RuntimeError, match="stream-1"):
        core.sleep(level=1, mode=mode)
    core.scheduler.set_pause_state.assert_not_called()
    core.model_executor.sleep.assert_not_called()


# @spec PSTATE-PRES-004
@pytest.mark.parametrize("mode", ["wait", "keep"])
def test_checkpoint_prepare_names_the_held_streams_and_changes_nothing(mode):
    core = _state_core(["stream-1"])
    reason = core.checkpoint_prepare(mode)
    assert reason is not None and "stream-1" in reason
    assert core._checkpoint_state == CheckpointState.RUNNING
    core.scheduler.set_pause_state.assert_not_called()


# @spec PSTATE-PRES-004
def test_kv_cache_release_is_refused_while_a_stream_is_held():
    core = _state_core(["stream-1"])
    core.scheduler.pause_state = PauseState.PAUSED_ALL
    with pytest.raises(RuntimeError, match="stream-1"):
        core.release_kv_cache_memory()
    core.model_executor.discard.assert_not_called()


# @spec PSTATE-PRES-005
def test_draining_clearing_pause_refuses_admissions_then_rechecks_when_idle():
    core = _state_core()
    core.engines_running = True
    future = core.pause_scheduler(mode="wait", clear_cache=True)
    assert future is not None and not future.done()

    _add(core, stream_admission=True)
    *_, error = _refusal(core)
    assert error.code == "engine_paused"
    core.scheduler.admit_stream.assert_not_called()

    core.held.append("stream-1")
    core.engines_running = False
    core._notify_idle_state_callbacks()
    with pytest.raises(RuntimeError, match="stream-1"):
        future.result(timeout=0)
    core.scheduler.reset_prefix_cache.assert_not_called()


def _pause_abort(core):
    core.pause_scheduler(mode="abort", clear_cache=True)


def _sleep_abort(core):
    core.sleep(level=1, mode="abort")


def _checkpoint_abort(core):
    assert core.checkpoint_prepare("abort") is None


# @spec PSTATE-PRES-006
@pytest.mark.parametrize("call", [_pause_abort, _sleep_abort, _checkpoint_abort])
def test_abort_mode_finishes_every_stream_then_performs_the_operation(call):
    core = _state_core(["stream-1"])
    core.scheduler.finish_requests.side_effect = lambda *args: core.held.clear() or []
    call(core)
    core.scheduler.finish_requests.assert_called_once()
    core.scheduler.reset_prefix_cache.assert_called_once()


# @spec PSTATE-PRES-007
@pytest.mark.parametrize("mode", ["wait", "keep"])
def test_sleep_level_zero_is_allowed_while_a_stream_is_held(mode):
    core = _state_core(["stream-1"])
    core.sleep(level=0, mode=mode)
    core.scheduler.set_pause_state.assert_called_once()
    core.model_executor.sleep.assert_not_called()


# @spec PSTATE-ADMIT-008
def test_deadlines_are_swept_after_the_input_drain_and_before_the_work_check():
    core = _state_core()
    events = []
    core._deferred_inputs = deque()
    core.input_queue = queue.Queue()
    core.input_queue.put((EngineCoreRequestType.WAKEUP, None))
    core.process_input_queue_block = True
    core._handle_client_request = lambda *args: events.append("input")
    core._sweep_stream_deadlines = lambda: events.append("sweep")
    core.has_work = lambda: events.append("work") or True

    core._process_input_queue()

    assert events.index("input") < events.index("sweep") < events.index("work")


# @spec PSTATE-ADMIT-008
def test_idle_wait_times_out_at_the_nearest_deadline():
    core = _state_core()
    core._deferred_inputs = deque()
    core.input_queue = MagicMock()
    core.input_queue.empty.return_value = True
    core.input_queue.get.side_effect = queue.Empty
    core.aborts_queue = queue.Queue()
    core.process_input_queue_block = True
    core.has_work = lambda: False
    core.is_running = lambda: True
    core._notify_idle_state_callbacks = MagicMock()
    core._sweep_stream_deadlines = MagicMock()
    core.scheduler.next_stream_deadline.return_value = 5.0

    core._process_input_queue()

    [call] = core.input_queue.get.call_args_list
    assert call.kwargs["block"] is True
    assert call.kwargs["timeout"] == pytest.approx(5.0)


# @spec PSTATE-ADMIT-007
def test_swept_streams_are_aborted_through_the_abort_outputs():
    core = _state_core()
    expired = [MagicMock(request_id="s", client_index=1)]
    core.scheduler.sweep_stream_deadlines.return_value = expired
    core._send_abort_outputs = MagicMock()

    core._sweep_stream_deadlines()

    core._send_abort_outputs.assert_called_once_with(expired)


# @spec PSTATE-FAIL-004
def test_unreported_forward_exception_is_fatal_for_a_state_model():
    core = object.__new__(EngineCore)
    core.scheduler = MagicMock()
    core.scheduler.has_unrecomputable_state = True
    core.scheduler.has_requests.return_value = True
    core.scheduler.schedule.return_value.total_num_scheduled_tokens = 1
    core.num_executed_steps = 0
    core._should_throttle_prefills = MagicMock(return_value=False)
    core.model_executor = MagicMock()
    core.model_executor.execute_model.side_effect = RuntimeError("forward failed")

    with pytest.raises(RuntimeError, match="forward failed"):
        core.step()

    core.scheduler.update_from_output.assert_not_called()
    core.scheduler.finish_requests.assert_not_called()
