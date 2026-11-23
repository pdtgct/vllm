# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent state at the engine layer: configuration refusals, `AsyncLLM`
stream handling, engine-core admission, preservation and deadlines.

Scheduler-side rules run on a real scheduler whose only KV cache group is one
`MambaSpec` state group; engine-core rules run on a bare `EngineCoreProc`.
"""

import asyncio
import itertools
import queue
import re
import time
from collections import deque
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from tests.v1.core.test_persistent_state_scheduler import declare_state_facts
from tests.v1.core.utils import (
    EOS_TOKEN_ID,
    create_requests,
    create_scheduler,
    new_kv_cache_group,
)
from vllm import ModelRegistry
from vllm.config import (
    ECTransferConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
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
from vllm.v1.core.kv_cache_utils import verify_persistent_state_groups
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
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, MambaSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

DELTA = RequestOutputKind.DELTA
STUB_ARCH = "StubPersistentStateForCausalLM"
NO_WARMUP_ARCH = "StubNoWarmupPersistentStateForCausalLM"


def _engine_kwargs(arch: str = STUB_ARCH, **hf_overrides) -> dict:
    return dict(
        model="facebook/opt-125m",
        hf_overrides={"architectures": [arch], **hf_overrides},
        load_format="dummy",
        max_model_len=256,
        max_num_seqs=2,
    )


STATE_ENGINE_KWARGS = _engine_kwargs()


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
    if NO_WARMUP_ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            NO_WARMUP_ARCH,
            f"vllm_add_dummy_model.my_persistent_state:{NO_WARMUP_ARCH}",
        )


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
            "ec_transfer_config",
            dict(
                ec_transfer_config=ECTransferConfig(
                    ec_connector="ECExampleConnector",
                    ec_role="ec_both",
                    ec_connector_extra_config={"shared_storage_path": "/tmp/ec"},
                )
            ),
        ),
        (
            "speculative_config",
            dict(speculative_config={"method": "ngram", "num_speculative_tokens": 2}),
        ),
        # An explicit backend skips the check of the world size against the
        # GPUs present, so the refusal fires on a single-GPU host.
        (
            "pipeline_parallel_size",
            dict(pipeline_parallel_size=2, distributed_executor_backend="mp"),
        ),
        (
            "data_parallel_size",
            dict(data_parallel_size=2, distributed_executor_backend="mp"),
        ),
        (
            "tensor_parallel_size",
            dict(tensor_parallel_size=2, distributed_executor_backend="mp"),
        ),
        (
            "multi-dimensional rotary positions",
            dict(
                hf_overrides={
                    "architectures": [STUB_ARCH],
                    "rope_parameters": {
                        "rope_type": "default",
                        "mrope_section": [1, 1, 1],
                    },
                }
            ),
        ),
    ],
)
def test_state_model_refuses_unsupported_option(option, kwargs):
    args = {**STATE_ENGINE_KWARGS, **kwargs}
    with pytest.raises(ValueError, match=f"{option}.* is not supported for this model"):
        EngineArgs(**args).create_engine_config()


def _config_with_parallel_fields(arch: str, **fields) -> VllmConfig:
    """A config of `arch` whose parallel config has `fields` set, built from
    config objects so that no other check of those modes runs first."""
    base = EngineArgs(**_engine_kwargs(arch)).create_engine_config()
    parallel_config = ParallelConfig()
    for name, value in fields.items():
        setattr(parallel_config, name, value)
    return VllmConfig(
        model_config=base.model_config,
        cache_config=base.cache_config,
        scheduler_config=base.scheduler_config,
        parallel_config=parallel_config,
    )


# @spec PSTATE-DECL-005
@pytest.mark.parametrize("option", ["enable_fault_tolerance", "enable_elastic_ep"])
def test_model_declining_warmup_refuses_modes_that_rerun_it(option):
    with pytest.raises(ValueError, match=f"{option}.* is not supported for this model"):
        _config_with_parallel_fields(NO_WARMUP_ARCH, **{option: True})


# @spec PSTATE-DECL-005
@pytest.mark.parametrize("option", ["enable_fault_tolerance", "enable_elastic_ep"])
def test_model_accepting_warmup_keeps_modes_that_rerun_it(option):
    config = _config_with_parallel_fields(STUB_ARCH, **{option: True})
    assert getattr(config.parallel_config, option) is True


# @spec PSTATE-DECL-006
def test_state_model_accepts_async_scheduling_and_cuda_graphs():
    config = EngineArgs(
        **STATE_ENGINE_KWARGS,
        async_scheduling=True,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
    ).create_engine_config()
    assert config.scheduler_config.async_scheduling is True


def _state_group(spec: MambaSpec | None = None):
    layer = MambaManager.layer_config(spec or _state_spec(), recomputable=False)
    return new_kv_cache_group(["state"], layer)


def _attention_group():
    return new_kv_cache_group(
        ["attn"],
        AttentionSpec(block_size=64, num_kv_heads=1, head_size=1, dtype=torch.float32),
    )


def _verify_groups(*groups, declared: bool = True) -> None:
    """The engine's startup check over `groups`, for a model that declares
    `SupportsPersistentState` when `declared`."""
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(has_persistent_state=declared)
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=4, kv_cache_tensors=[], kv_cache_groups=list(groups)
    )
    verify_persistent_state_groups(vllm_config, kv_cache_config)


# @spec PSTATE-DECL-003
def test_state_group_refuses_other_mamba_types():
    spec = _state_spec(mamba_type=MambaAttentionBackendEnum.SHORT_CONV)
    with pytest.raises(ValueError, match="PERSISTENT_STATE"):
        _verify_groups(_state_group(spec))


# @spec PSTATE-DECL-009
def test_state_group_refuses_initial_values_of_the_wrong_length():
    with pytest.raises(ValueError, match="initial_values"):
        _verify_groups(_state_group(_state_spec(initial_values=(0, 1))))


# @spec PSTATE-DECL-004 PSTATE-DECL-013
def test_state_declaration_and_state_groups_must_agree():
    """A model has a state group if and only if it declares persistent state,
    so every PSTATE check keys on one fact."""
    _verify_groups(_state_group(), declared=True)
    _verify_groups(_attention_group(), _state_group(), declared=True)
    _verify_groups(_attention_group(), declared=False)
    with pytest.raises(ValueError, match="SupportsPersistentState"):
        _verify_groups(_attention_group(), declared=True)
    with pytest.raises(ValueError, match="SupportsPersistentState"):
        _verify_groups(_state_group(), declared=False)


# @spec PSTATE-HIST-006 PSTATE-HIST-007
def test_pending_stream_input_bound_defaults_to_eight():
    assert SchedulerConfig.default_factory().max_pending_stream_inputs == 8


# @spec PSTATE-ADMIT-007
def test_stream_admission_timeouts_default():
    assert SchedulerConfig.default_factory().stream_admission_timeouts == {
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
        sampling_params=kwargs.pop(
            "sampling_params", SamplingParams(max_tokens=4, output_kind=DELTA)
        ),
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
    llm.output_handler = None
    llm.log_requests = False
    llm.observability_config = MagicMock(otlp_traces_endpoint=None)
    llm.vllm_config = MagicMock(is_mm_encoder_only=False)
    llm.vllm_config.cache_config.kv_sharing_fast_prefill = False
    llm.vllm_config.parallel_config.enable_fault_tolerance = False
    llm.scheduler_config = MagicMock(
        max_num_queued_reqs=None, max_num_queued_tokens=None
    )
    llm.model_config = MagicMock(is_diffusion=False, has_persistent_state=True)
    llm._state_stream_numbers = itertools.count(1)
    llm._state_stream_ids = {}
    llm.engine_core = MagicMock(client_index=3)
    llm.engine_core.resources.engine_dead = False
    llm.engine_core.admit_stream_async = AsyncMock(return_value=None)
    llm.engine_core.wait_stream_retired_async = AsyncMock()
    llm.output_processor = MagicMock()
    llm.output_processor.has_request.return_value = True
    llm.input_processor = MagicMock()

    def process_inputs(request_id, prompt, params, *, resumable=False, **kwargs):
        ids = list(prompt["prompt_token_ids"]) if resumable else [0]
        fields = {
            name: kwargs[name]
            for name in ("continuation", "external_req_id")
            if name in kwargs
        }
        return _core_request(
            request_id,
            ids,
            resumable=resumable,
            mm_features=mm_features,
            sampling_params=params,
            **fields,
        )

    def process_inputs_keywords(*, request_id, prompt, params, **kwargs):
        return process_inputs(request_id, prompt, params, **kwargs)

    llm.input_processor.process_inputs.side_effect = process_inputs_keywords
    llm.input_processor.process_inputs_async = AsyncMock(side_effect=process_inputs)
    return llm


def _chunks(*token_ids: list[int]):
    async def chunks():
        for ids in token_ids:
            yield StreamingInput(prompt=TokensPrompt(prompt_token_ids=list(ids)))

    return chunks()


async def _open_stream(llm: AsyncLLM, external_id: str, chunks, **kwargs):
    queue = await llm._add_streaming_input_request(
        external_id,
        chunks,
        SamplingParams(max_tokens=4, output_kind=DELTA),
        **kwargs,
    )
    task = queue._input_stream_task
    assert task is not None
    await task
    return queue


def _sent(llm: AsyncLLM) -> list[EngineCoreRequest]:
    return [call.args[0] for call in llm._add_request.await_args_list]


REFUSED_SAMPLING_PARAMETERS = [
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
]
"""Each refused parameter alone. `SamplingParams` has no per-request logits
processors field, so none is listed."""


# @spec PSTATE-DECL-007
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("parameter", "params"),
    [
        *REFUSED_SAMPLING_PARAMETERS,
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

    sampling_params = SamplingParams(
        **{"max_tokens": 4, "output_kind": DELTA, **params}
    )
    with pytest.raises(
        VLLMValidationError, match=f"{parameter}.* is not supported for this model"
    ):
        await llm._add_streaming_input_request("ext", chunks(), sampling_params)
    assert not consumed
    llm._add_request.assert_not_awaited()


# @spec PSTATE-DECL-007
@pytest.mark.asyncio
@pytest.mark.parametrize(("parameter", "params"), REFUSED_SAMPLING_PARAMETERS)
async def test_single_state_request_refuses_sampling_parameter(
    monkeypatch, parameter, params
):
    llm = _state_llm(monkeypatch)
    sampling_params = SamplingParams(
        **{"max_tokens": 4, "output_kind": RequestOutputKind.FINAL_ONLY, **params}
    )
    with pytest.raises(
        VLLMValidationError, match=f"{parameter}.* is not supported for this model"
    ):
        await llm.add_request(
            "single", TokensPrompt(prompt_token_ids=[1]), sampling_params
        )
    llm._add_request.assert_not_awaited()
    llm.engine_core.add_request_async.assert_not_called()


# @spec PSTATE-DECL-007
@pytest.mark.asyncio
@pytest.mark.parametrize("output_kind", list(RequestOutputKind))
async def test_single_state_request_allows_every_output_kind(monkeypatch, output_kind):
    llm = _state_llm(monkeypatch)
    await llm.add_request(
        "single",
        TokensPrompt(prompt_token_ids=[1]),
        SamplingParams(max_tokens=4, output_kind=output_kind),
    )
    llm._add_request.assert_awaited_once()


# @spec PSTATE-ADMIT-001
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason", ["request_id_in_use", "engine_paused", "active_limit"]
)
async def test_stream_admission_refusal_raises_before_reading_input(
    monkeypatch, reason
):
    llm = _state_llm(monkeypatch)
    llm.engine_core.admit_stream_async.return_value = reason
    consumed = []

    async def chunks():
        consumed.append(True)
        yield StreamingInput(prompt=TokensPrompt(prompt_token_ids=[1]))

    with pytest.raises(StreamAdmissionError) as excinfo:
        await llm._add_streaming_input_request(
            "ext",
            chunks(),
            SamplingParams(max_tokens=4, output_kind=DELTA),
            stream_admission=True,
        )
    assert excinfo.value.reason == reason
    assert not consumed
    llm.engine_core.admit_stream_async.assert_awaited_once()
    llm._add_request.assert_not_awaited()


# @spec PSTATE-ADMIT-001 PSTATE-TURN-005
@pytest.mark.asyncio
async def test_admitted_stream_reads_inputs_after_admission_as_continuations(
    monkeypatch,
):
    llm = _state_llm(monkeypatch)
    events = []

    async def admit(request, *args, **kwargs):
        events.append("admit")

    llm.engine_core.admit_stream_async.side_effect = admit

    async def chunks():
        events.append("read")
        yield StreamingInput(prompt=TokensPrompt(prompt_token_ids=[1]))

    queue = await _open_stream(llm, "ext", chunks(), stream_admission=True)

    assert events == ["admit", "read"]
    [admission, *_] = llm.engine_core.admit_stream_async.await_args.args
    assert admission.request_id == queue.request_id
    sent = _sent(llm)
    assert len(sent) == 2
    assert all(request.continuation for request in sent)


# @spec PSTATE-ADMIT-006
@pytest.mark.asyncio
async def test_state_stream_id_is_external_id_client_index_and_counter(monkeypatch):
    llm = _state_llm(monkeypatch)
    ids = []
    for _ in range(3):
        queue = await _open_stream(llm, "ext", _chunks([1]))
        ids.append(queue.request_id)

    assert all("ext" in request_id and "3" in request_id for request_id in ids)
    numbers = [
        tuple(int(n) for n in re.findall(r"\d+", request_id.replace("ext", "", 1)))
        for request_id in ids
    ]
    assert numbers[0] < numbers[1] < numbers[2]


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
async def test_wait_stream_retired_resolves_only_when_core_retires_the_stream(
    monkeypatch,
):
    llm = _state_llm(monkeypatch)
    retired = asyncio.Event()

    async def wait_retired(request_id):
        await retired.wait()

    llm.engine_core.wait_stream_retired_async.side_effect = wait_retired
    queue = await _open_stream(llm, "ext", _chunks([1]))

    waiter = asyncio.ensure_future(llm.wait_stream_retired("ext"))
    await asyncio.sleep(0.05)
    assert not waiter.done()

    retired.set()
    await asyncio.wait_for(waiter, 1)
    llm.engine_core.wait_stream_retired_async.assert_awaited_once_with(queue.request_id)


# @spec PSTATE-ADMIT-005 PSTATE-ADMIT-012
@pytest.mark.asyncio
async def test_refused_admission_keeps_the_live_streams_binding(monkeypatch):
    """Opening a second stream under a held external id is refused by the
    engine; the first stream keeps the binding `wait_stream_retired` uses."""
    llm = _state_llm(monkeypatch)
    live = await _open_stream(llm, "ext", _chunks([1]))
    llm.engine_core.admit_stream_async.return_value = "request_id_in_use"

    with pytest.raises(StreamAdmissionError):
        await llm._add_streaming_input_request(
            "ext",
            _chunks([1]),
            SamplingParams(max_tokens=4, output_kind=DELTA),
            stream_admission=True,
        )

    assert llm._state_stream_ids["ext"] == live.request_id
    await llm.wait_stream_retired("ext")
    llm.engine_core.wait_stream_retired_async.assert_awaited_once_with(live.request_id)


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

    async def chunks():
        yield StreamingInput(prompt=TokensPrompt(prompt_token_ids=[1]))
        yield StreamingInput(
            prompt=TokensPrompt(
                prompt_token_ids=[], multi_modal_data={"audio": object()}
            )
        )

    await _open_stream(llm, "ext", chunks())
    assert len(_sent(llm)) == 3


# @spec PSTATE-DECL-014
@pytest.mark.asyncio
async def test_unsupported_update_of_a_held_stream_is_left_to_the_scheduler(
    monkeypatch,
):
    """Only the opening parameters are validated here; an update's are the
    scheduler's to refuse, ending the stream with a named code."""
    llm = _state_llm(monkeypatch)

    async def chunks():
        yield StreamingInput(prompt=TokensPrompt(prompt_token_ids=[1]))
        yield StreamingInput(
            prompt=TokensPrompt(prompt_token_ids=[2]),
            sampling_params=SamplingParams(
                presence_penalty=0.5, max_tokens=4, output_kind=DELTA
            ),
        )

    queue = await _open_stream(llm, "ext", chunks())

    assert [request.prompt_token_ids for request in _sent(llm)] == [[1], [2], [0]]
    assert queue.get_nowait() is None


# Scheduler-side stream rules


def _state_scheduler(monkeypatch: pytest.MonkeyPatch, **kwargs):
    declare_state_facts(monkeypatch)
    layer_config = MambaManager.layer_config(_state_spec(), recomputable=False)
    return create_scheduler(
        max_model_len=64, block_size=64, kv_cache_layer_config=layer_config, **kwargs
    )


def _stream_request(request_id: str, num_tokens: int = 3, **kwargs):
    continuation = kwargs.pop("continuation", False)
    resumable = kwargs.pop("resumable", True)
    kwargs.setdefault(
        "sampling_params",
        SamplingParams(max_tokens=kwargs.pop("max_tokens", 16), output_kind=DELTA),
    )
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
def test_update_with_refused_sampling_parameters_finishes_the_stream(monkeypatch):
    scheduler = _state_scheduler(monkeypatch)
    request = _park(scheduler, "s")
    update = _stream_request(
        "s",
        2,
        continuation=True,
        sampling_params=SamplingParams(
            presence_penalty=0.5, max_tokens=4, output_kind=DELTA
        ),
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
def test_terminal_finish_carries_the_terminal_flag(
    monkeypatch, max_tokens, token, terminal
):
    scheduler = _state_scheduler(monkeypatch)
    scheduler.add_request(_stream_request("s", max_tokens=max_tokens))
    [output] = _step(scheduler, ["s"], [[token]])[0].outputs
    assert output.terminal is terminal


# @spec PSTATE-ADMIT-001
def test_admission_refuses_beyond_the_active_limit(monkeypatch):
    scheduler = _state_scheduler(monkeypatch, max_num_seqs=1)
    assert scheduler.admit_stream(_stream_request("a", 0)) is None
    assert scheduler.admit_stream(_stream_request("b", 0)) == "active_limit"


# @spec PSTATE-ADMIT-001 PSTATE-PRES-005
@pytest.mark.parametrize("pause_state", [PauseState.PAUSED_NEW, PauseState.PAUSED_ALL])
def test_admission_refuses_while_paused(monkeypatch, pause_state):
    scheduler = _state_scheduler(monkeypatch)
    scheduler.set_pause_state(pause_state)
    assert scheduler.admit_stream(_stream_request("a", 0)) == "engine_paused"
    assert not scheduler.has_request("a")


# @spec PSTATE-ADMIT-012
def test_admission_refuses_an_external_id_held_by_a_stream(monkeypatch):
    scheduler = _state_scheduler(monkeypatch)
    first = _stream_request("ext-0-1", 0)
    first.external_req_id = "ext"
    second = _stream_request("ext-0-2", 0)
    second.external_req_id = "ext"
    assert scheduler.admit_stream(first) is None
    assert scheduler.admit_stream(second) == "request_id_in_use"


# @spec PSTATE-ADMIT-012
def test_admission_refuses_an_internal_id_held_by_a_stream(monkeypatch):
    scheduler = _state_scheduler(monkeypatch)
    assert scheduler.admit_stream(_stream_request("s", 0)) is None
    assert scheduler.admit_stream(_stream_request("s", 0)) == "request_id_in_use"


# @spec PSTATE-HIST-007 PSTATE-ADMIT-013
@pytest.mark.xfail(strict=False, reason="pending Pete: backpressure vs error")
def test_input_beyond_the_pending_bound_ends_the_stream_with_a_named_error(
    monkeypatch,
):
    scheduler = _state_scheduler(monkeypatch)
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


def _clocked_scheduler(monkeypatch: pytest.MonkeyPatch, **kwargs):
    """A state scheduler and the monotonic clock its deadlines run on."""
    scheduler = _state_scheduler(monkeypatch, **kwargs)
    clock = _Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    return scheduler, clock


# @spec PSTATE-ADMIT-007
def test_never_fed_stream_is_aborted_after_its_deadline(monkeypatch):
    scheduler, clock = _clocked_scheduler(monkeypatch)
    request = _stream_request("s", 0)
    scheduler.admit_stream(request)

    clock.now += 14.9
    assert scheduler.sweep_stream_deadlines() == []
    clock.now += 0.2
    assert scheduler.sweep_stream_deadlines() == [request]
    assert request.status == RequestStatus.FINISHED_ABORTED
    assert request.stop_reason == "never_fed_timeout"


# @spec PSTATE-ADMIT-007
def test_parked_stream_is_aborted_after_its_idle_deadline(monkeypatch):
    scheduler, clock = _clocked_scheduler(monkeypatch)
    request = _park(scheduler, "s")

    clock.now += 59.9
    assert scheduler.sweep_stream_deadlines() == []
    clock.now += 0.2
    assert scheduler.sweep_stream_deadlines() == [request]
    assert request.status == RequestStatus.FINISHED_ABORTED
    assert request.stop_reason == "parked_idle_timeout"


# @spec PSTATE-ADMIT-007
def test_stream_unfinished_after_close_is_aborted_after_its_draining_deadline(
    monkeypatch,
):
    scheduler, clock = _clocked_scheduler(monkeypatch)
    request = _stream_request("s")
    scheduler.add_request(request)
    scheduler.schedule()
    scheduler.add_request(_stream_request("s", 1, resumable=False, continuation=True))

    clock.now += 39.9
    assert scheduler.sweep_stream_deadlines() == []
    clock.now += 0.2
    assert scheduler.sweep_stream_deadlines() == [request]
    assert request.stop_reason == "draining_timeout"


# @spec PSTATE-ADMIT-009
@pytest.mark.parametrize("pause_state", [PauseState.PAUSED_NEW, PauseState.PAUSED_ALL])
def test_deadlines_do_not_advance_while_paused(monkeypatch, pause_state):
    scheduler, clock = _clocked_scheduler(monkeypatch)
    request = _stream_request("s", 0)
    scheduler.admit_stream(request)

    clock.now += 10
    scheduler.set_pause_state(pause_state)
    clock.now += 500
    scheduler.set_pause_state(PauseState.UNPAUSED)

    assert scheduler.sweep_stream_deadlines() == []
    clock.now += 5.1
    assert scheduler.sweep_stream_deadlines() == [request]


# Engine core


def _state_core(held: list[str] | None = None) -> EngineCoreProc:
    """A bare engine core over a mocked scheduler that holds `held` streams.

    The scheduler mock keeps its pause state and refuses admissions while
    paused, as the scheduler does; `core.events` records the cache, sleep and
    release operations with the streams held when each ran.
    """
    held = [] if held is None else held
    events: list[tuple] = []
    core = object.__new__(EngineCoreProc)
    core._checkpoint_state = CheckpointState.RUNNING
    core._checkpoint_error = None
    core._kv_cache_reclaimable = False
    core.vllm_config = MagicMock()
    core.vllm_config.watermark_config = None
    core.prefix_hash_config = None
    core.request_block_hasher = None
    core.mm_receiver_cache = None
    core.model_executor = MagicMock()
    core.model_executor.is_sleeping = False
    core.model_executor.vllm_config.cache_config.enable_extensible_kv_cache = False
    core.model_executor.sleep.side_effect = lambda level: events.append(
        ("sleep", tuple(held))
    )
    core.model_executor.discard.side_effect = lambda tags: events.append(
        ("discard", tuple(held))
    )
    scheduler = MagicMock()
    scheduler.has_unrecomputable_state = True
    scheduler.has_requests.return_value = False
    scheduler.has_request.return_value = False
    scheduler.stream_id_in_use.return_value = False
    scheduler.stream_finish_outputs = []
    scheduler.held_state_streams.side_effect = lambda: list(held)
    scheduler.pause_state = PauseState.UNPAUSED

    def set_pause_state(state):
        scheduler.pause_state = state

    def reset_prefix_cache(*args, **kwargs):
        events.append(("reset", tuple(held)))
        return True

    def finish_requests(*args):
        events.append(("finish", tuple(held)))
        held.clear()
        return []

    scheduler.set_pause_state.side_effect = set_pause_state
    scheduler.reset_prefix_cache.side_effect = reset_prefix_cache
    scheduler.finish_requests.side_effect = finish_requests
    scheduler.admit_stream.side_effect = lambda request: (
        None if scheduler.pause_state == PauseState.UNPAUSED else "engine_paused"
    )
    core.scheduler = scheduler
    core.batch_queue = None
    core.engines_running = False
    core.has_coordinator = False
    core.reject_new_requests = False
    core.shutdown_state = EngineShutdownState.RUNNING
    core._idle_state_callbacks = []
    core.output_queue = queue.Queue()
    core._send_finish_outputs_to_client = MagicMock()
    core._send_abort_outputs_to_client = MagicMock()
    core.held = held
    core.events = events
    return core


def _add(core: EngineCoreProc, **fields) -> MagicMock:
    request = MagicMock(
        request_id="s",
        client_index=1,
        abort_immediately=False,
        kv_transfer_params=None,
        ec_transfer_params=None,
        pooling_params=None,
        continuation=False,
        resumable=True,
    )
    for name, value in fields.items():
        setattr(request, name, value)
    core._handle_client_request(EngineCoreRequestType.ADD, (request, 0))
    return request


def _refusal(core: EngineCoreProc) -> tuple[list[str], int, FinishReason, RequestError]:
    [call] = core._send_finish_outputs_to_client.call_args_list
    return call.args


def _admission() -> EngineCoreRequest:
    return _core_request("s", [], resumable=True, client_index=1)


# @spec PSTATE-TURN-006
def test_continuation_for_an_unheld_stream_is_refused():
    core = _state_core()
    _add(core, continuation=True)
    req_ids, client_index, reason, error = _refusal(core)
    assert (req_ids, client_index, reason) == (["s"], 1, FinishReason.ERROR)
    assert error.code == "stream_not_found"
    core.scheduler.add_request.assert_not_called()


# @spec PSTATE-TURN-006
def test_continuation_for_a_held_stream_is_added():
    core = _state_core()
    core.scheduler.has_request.return_value = True
    request = _add(core, continuation=True)
    core.scheduler.add_request.assert_called_once_with(request)
    core._send_finish_outputs_to_client.assert_not_called()


# @spec PSTATE-TURN-013
def test_close_for_an_unheld_stream_is_dropped_silently():
    core = _state_core()
    _add(core, continuation=True, resumable=False)
    core.scheduler.add_request.assert_not_called()
    core._send_finish_outputs_to_client.assert_not_called()
    core._send_abort_outputs_to_client.assert_not_called()


# @spec PSTATE-ADMIT-012
def test_opener_whose_external_id_a_held_stream_has_is_refused(monkeypatch):
    core = _state_core()
    core.scheduler = _state_scheduler(monkeypatch)
    held = _stream_request("ext-1-1", 0)
    held.external_req_id = "ext"
    assert core.scheduler.admit_stream(held) is None
    opener = _stream_request("ext-1-2")
    opener.external_req_id = "ext"
    opener.client_index = 1

    core._handle_client_request(EngineCoreRequestType.ADD, (opener, 0))

    assert not core.scheduler.has_request("ext-1-2")
    assert core.scheduler.requests["ext-1-1"] is held
    req_ids, client_index, reason, error = _refusal(core)
    assert (req_ids, client_index, reason) == (["ext-1-2"], 1, FinishReason.ERROR)
    assert error.code == "request_id_in_use"


# @spec PSTATE-ADMIT-012
def test_opener_whose_internal_id_a_held_stream_has_is_refused(monkeypatch):
    core = _state_core()
    core.scheduler = _state_scheduler(monkeypatch)
    held = _stream_request("s")
    core.scheduler.add_request(held)
    opener = _stream_request("s")
    opener.client_index = 1

    core._handle_client_request(EngineCoreRequestType.ADD, (opener, 0))

    assert core.scheduler.requests["s"] is held
    req_ids, client_index, reason, error = _refusal(core)
    assert (req_ids, client_index, reason) == (["s"], 1, FinishReason.ERROR)
    assert error.code == "request_id_in_use"


# @spec PSTATE-TURN-006 PSTATE-TURN-013
def test_paused_engine_still_refuses_unknown_continuations_by_name():
    """A paused engine's retryable refusal does not replace `stream_not_found`,
    and an unknown close signal still draws no output."""
    core = _state_core()
    core.reject_new_requests = True

    _add(core, continuation=True)
    *_, error = _refusal(core)
    assert error.code == "stream_not_found"

    core._send_finish_outputs_to_client.reset_mock()
    _add(core, continuation=True, resumable=False)
    core._send_finish_outputs_to_client.assert_not_called()
    core._send_abort_outputs_to_client.assert_not_called()


# @spec PSTATE-ADMIT-014
def test_stream_ended_by_an_input_is_reported_before_a_later_refusal(monkeypatch):
    """An input that ends its stream between steps reports that at once, so
    a later input's refusal in the same drain cannot overtake it."""
    core = _state_core()
    core.scheduler = _state_scheduler(monkeypatch)
    core._send_finish_outputs_to_client = partial(
        EngineCoreProc._send_finish_outputs_to_client, core
    )

    def add(request):
        request.client_index = 1
        core._handle_client_request(EngineCoreRequestType.ADD, (request, 0))

    add(_stream_request("s"))
    core.scheduler.schedule()
    add(_stream_request("s", 1, resumable=False, continuation=True))
    add(_stream_request("s", 1, continuation=True))
    add(_stream_request("t", 1, continuation=True))

    sent = [core.output_queue.get_nowait() for _ in range(2)]
    assert core.output_queue.empty()
    assert [
        (client_index, outputs.finished_requests, outputs.outputs[0].error.code)
        for client_index, outputs in sent
    ] == [(1, ["s"], "input_after_close"), (1, ["t"], "stream_not_found")]
    assert all(outputs.outputs[0].terminal for _, outputs in sent[:1])


# @spec PSTATE-ADMIT-001
@pytest.mark.parametrize(
    "reason", ["request_id_in_use", "active_limit", "engine_paused"]
)
def test_admit_stream_returns_the_refusal_reason(reason):
    core = _state_core()
    core.scheduler.admit_stream.side_effect = None
    core.scheduler.admit_stream.return_value = reason

    assert core.admit_stream(_admission()) == reason

    [call] = core.scheduler.admit_stream.call_args_list
    assert call.args[0].request_id == "s"
    core._send_finish_outputs_to_client.assert_not_called()


# @spec PSTATE-ADMIT-001
def test_admit_stream_returns_none_when_admitted():
    core = _state_core()

    assert core.admit_stream(_admission()) is None

    core.scheduler.admit_stream.assert_called_once()
    core._send_finish_outputs_to_client.assert_not_called()


# @spec PSTATE-ADMIT-011
def test_stream_admission_is_refused_for_an_ordinary_model():
    core = _state_core()
    core.scheduler.has_unrecomputable_state = False

    with pytest.raises(
        VLLMValidationError, match="stream admission is not supported for this model"
    ):
        core.admit_stream(_admission())
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

    assert core.admit_stream(_admission()) == "engine_paused"

    core.held.append("stream-1")
    core.engines_running = False
    core._notify_idle_state_callbacks()
    with pytest.raises(RuntimeError, match="stream-1"):
        future.result(timeout=0)
    core.scheduler.reset_prefix_cache.assert_not_called()
    # The refusal leaves the engine as it was before the pause.
    assert core.scheduler.pause_state == PauseState.UNPAUSED
    assert core.reject_new_requests is False


def _pause_abort(core):
    core.pause_scheduler(mode="abort", clear_cache=True)
    return "reset"


def _sleep_abort(core):
    core.sleep(level=1, mode="abort")
    return "sleep"


def _checkpoint_abort(core):
    assert core.checkpoint_prepare("abort") is None
    assert core._checkpoint_state == CheckpointState.PREPARED
    return "discard"


# @spec PSTATE-PRES-006
@pytest.mark.parametrize("call", [_pause_abort, _sleep_abort, _checkpoint_abort])
def test_abort_mode_finishes_every_stream_then_performs_the_operation(call):
    core = _state_core(["stream-1"])

    operation = call(core)

    assert core.events[0] == ("finish", ("stream-1",))
    assert (operation, ()) in core.events
    assert all(held == () for _, held in core.events[1:])


# @spec PSTATE-PRES-007
@pytest.mark.parametrize("mode", ["wait", "keep"])
def test_sleep_level_zero_is_allowed_while_a_stream_is_held(mode):
    core = _state_core(["stream-1"])
    core.sleep(level=0, mode=mode)
    core.scheduler.set_pause_state.assert_called_once()
    core.model_executor.sleep.assert_not_called()


# @spec PSTATE-ADMIT-008
def test_deadlines_are_swept_after_the_input_drain_and_before_the_step():
    """The sweep is the last thing before the loop hands over to the step,
    after every queued input has been handled."""
    core = _state_core()
    events = []
    core._deferred_inputs = deque()
    core.input_queue = queue.Queue()
    core.input_queue.put((EngineCoreRequestType.WAKEUP, None))
    core.process_input_queue_block = True
    core._handle_client_request = lambda *args: events.append("input")
    core._sweep_stream_deadlines = lambda: events.append("sweep")
    core.has_work = lambda: True

    core._process_input_queue()

    assert events == ["input", "sweep"]


def _idle_core(core: EngineCoreProc) -> EngineCoreProc:
    """`core` waiting idle on an empty input queue."""
    core._deferred_inputs = deque()
    core.input_queue = MagicMock()
    core.input_queue.empty.return_value = True
    core.input_queue.get.side_effect = queue.Empty
    core.aborts_queue = queue.Queue()
    core.process_input_queue_block = True
    core.has_work = lambda: False
    core.is_running = lambda: True
    core._notify_idle_state_callbacks = MagicMock()
    return core


# @spec PSTATE-ADMIT-008
def test_idle_wait_times_out_at_the_nearest_deadline():
    core = _idle_core(_state_core())
    core._sweep_stream_deadlines = MagicMock()
    core.scheduler.next_stream_deadline.return_value = 5.0

    core._process_input_queue()

    [call] = core.input_queue.get.call_args_list
    assert call.kwargs["block"] is True
    assert call.kwargs["timeout"] == pytest.approx(5.0)


# @spec PSTATE-DECL-013
def test_ordinary_engine_iterations_traverse_no_deadlines():
    core = _idle_core(_state_core())
    core.scheduler.has_unrecomputable_state = False

    core._process_input_queue()

    [call] = core.input_queue.get.call_args_list
    assert call.kwargs.get("timeout") is None
    core.scheduler.sweep_stream_deadlines.assert_not_called()
    core.scheduler.next_stream_deadline.assert_not_called()


# @spec PSTATE-ADMIT-007
def test_abort_outputs_name_the_deadline_that_expired():
    core = _state_core()
    expired = MagicMock(
        request_id="s", client_index=1, stop_reason="parked_idle_timeout"
    )

    core._send_abort_outputs([expired])

    client_index, outputs = core.output_queue.get_nowait()
    assert (client_index, outputs.finished_requests) == (1, ["s"])
    [output] = outputs.outputs
    assert (output.finish_reason, output.stop_reason) == (
        FinishReason.ABORT,
        "parked_idle_timeout",
    )


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
