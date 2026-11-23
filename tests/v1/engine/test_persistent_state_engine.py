# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent state at the engine layer: configuration refusals, `AsyncLLM`
stream admission, validation, backpressure and lifetime, and the engine core's
preservation refusals and stream endings.

Scheduler-side rules run on a real scheduler whose only KV cache group is one
`MambaSpec` state group; engine-core rules run on a bare `EngineCoreProc`;
`AsyncLLM` runs with its real output processor over a mocked engine core whose
outputs each test feeds.
"""

import asyncio
import queue
import re
from collections import deque
from collections.abc import Mapping
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

import vllm.v1.engine.async_llm as async_llm_module
import vllm.v1.worker.gpu.sample.logits_processor.loader as logits_loader
from tests.v1.core.test_persistent_state_scheduler import declare_state_facts
from tests.v1.core.utils import (
    EOS_TOKEN_ID,
    create_requests,
    create_scheduler,
    new_kv_cache_group,
)
from vllm import ModelRegistry
from vllm.config import (
    CUDAGraphMode,
    ECTransferConfig,
    LoggingConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.config.fault_tolerance import FaultToleranceConfig
from vllm.config.kv_transfer import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.protocol import StreamingInput
from vllm.exceptions import VLLMValidationError
from vllm.inputs import TokensPrompt
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import (
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.kv_cache_utils import verify_persistent_state_groups
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
    FinishReason,
)
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import (
    CheckpointState,
    EngineCoreProc,
    EngineShutdownState,
)
from vllm.v1.engine.exceptions import EngineDeadError, StreamAdmissionError
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, MambaSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsProcessor

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


# @spec PSTATE-DECL-005
def test_state_model_refuses_recompute_on_fault_recovery():
    """Recompute recovery would preempt state its tokens cannot rebuild."""
    with pytest.raises(
        ValueError, match="request_failure_policy.* is not supported for this model"
    ):
        _config_with_parallel_fields(
            STUB_ARCH,
            enable_fault_tolerance=True,
            fault_tolerance_config=FaultToleranceConfig(
                request_failure_policy="recompute"
            ),
        )


# @spec PSTATE-DECL-006
def test_state_model_accepts_async_scheduling_and_cuda_graphs():
    config = EngineArgs(
        **STATE_ENGINE_KWARGS,
        async_scheduling=True,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
    ).create_engine_config()
    assert config.scheduler_config.async_scheduling is True
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY


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


# @spec PSTATE-DECL-005
def test_state_group_refuses_prefix_caching():
    """The prefix-caching allocator does not record the pages to initialize."""
    with pytest.raises(ValueError, match="enable_prefix_caching is not supported"):
        MambaManager.layer_config(
            _state_spec(), recomputable=False, enable_prefix_caching=True
        )
    MambaManager.layer_config(_state_spec(), enable_prefix_caching=True)


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


# @spec PSTATE-HIST-006
def test_pending_stream_input_credits_default_to_eight():
    assert SchedulerConfig.default_factory().max_pending_stream_inputs == 8


# AsyncLLM streams


class _CustomLogitsProcessor(LogitsProcessor):
    """Stands for a processor installed under `vllm.logits_processors`."""


def _request_from(*, request_id, prompt, params, resumable=False, **kwargs):
    """What the input processor makes of `prompt`: its tokens, or one token
    for any other content."""
    ids = prompt.get("prompt_token_ids") if isinstance(prompt, Mapping) else None
    embeds = prompt.get("prompt_embeds") if isinstance(prompt, Mapping) else None
    fields = {k: kwargs[k] for k in ("continuation", "external_req_id") if k in kwargs}
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=list(ids or [1]),
        prompt_embeds=embeds,
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        resumable=resumable,
        **fields,
    )


def _state_llm(
    monkeypatch: pytest.MonkeyPatch,
    *,
    persistent_state: bool = True,
    max_pending: int = 8,
    logits_processors: list[str] | None = None,
    installed_processors: bool = False,
) -> AsyncLLM:
    """An `AsyncLLM` with its real output processor over a mocked engine core.

    `llm.engine_core.feed` takes the `EngineCoreOutputs`, or an exception to
    raise, that the engine core returns next.
    """
    engine_core = MagicMock(client_index=3)
    engine_core.resources.engine_dead = False
    engine_core.resources.stop_requested = False
    engine_core.admit_stream_async = AsyncMock(return_value=None)
    engine_core.add_request_async = AsyncMock()
    engine_core.abort_requests_async = AsyncMock()
    engine_core.get_supported_tasks_async = AsyncMock(return_value=("generate",))
    engine_core.feed = asyncio.Queue()

    async def get_output_async():
        item = await engine_core.feed.get()
        if isinstance(item, BaseException):
            raise item
        return item

    engine_core.get_output_async = get_output_async
    input_processor = MagicMock()
    input_processor.process_inputs.side_effect = _request_from
    input_processor.process_inputs_async = AsyncMock(
        side_effect=lambda request_id, prompt, params, **kwargs: _request_from(
            request_id=request_id, prompt=prompt, params=params, **kwargs
        )
    )
    renderer = MagicMock(tokenizer=None, mm_processor_cache=None)
    for name, value in (
        ("maybe_register_config_serialize_by_value", MagicMock()),
        ("load_stat_logger_plugin_factories", MagicMock(return_value=[])),
        ("renderer_from_config", MagicMock(return_value=renderer)),
        ("InputProcessor", MagicMock(return_value=input_processor)),
        ("extract_prompt_components", lambda model_config, prompt: ("", None, None)),
    ):
        monkeypatch.setattr(async_llm_module, name, value)
    monkeypatch.setattr(
        async_llm_module.EngineCoreClient,
        "make_async_mp_client",
        MagicMock(return_value=engine_core),
    )
    installed = [
        SimpleNamespace(
            name="custom",
            value=f"{__name__}:_CustomLogitsProcessor",
            group=logits_loader.LOGITSPROCS_GROUP,
            load=lambda: _CustomLogitsProcessor,
        )
    ]
    monkeypatch.setattr(
        logits_loader,
        "entry_points",
        lambda **kwargs: installed if installed_processors else [],
    )

    config = MagicMock()
    config.logging_config = LoggingConfig()
    config.observability_config.otlp_traces_endpoint = None
    config.profiler_config.should_profile_frontend = False
    config.is_mm_encoder_only = False
    config.cache_config.kv_sharing_fast_prefill = False
    config.parallel_config.enable_fault_tolerance = False
    config.scheduler_config.stream_interval = 1
    config.scheduler_config.max_pending_stream_inputs = max_pending
    config.scheduler_config.max_num_queued_reqs = None
    config.scheduler_config.max_num_queued_tokens = None
    config.model_config.has_persistent_state = persistent_state
    config.model_config.is_diffusion = False
    config.model_config.logits_processors = logits_processors
    return AsyncLLM(config, MagicMock(), log_stats=False, log_requests=False)


def _params(**kwargs) -> SamplingParams:
    return SamplingParams(**{"max_tokens": 4, "output_kind": DELTA, **kwargs})


def _chunks(*prompts, reads: list | None = None, then: Exception | None = None):
    """An input iterator over `prompts` (token lists, prompts or
    `StreamingInput`s) that records each input it yields in `reads` and raises
    `then` after the last."""

    async def chunks():
        for prompt in prompts:
            await asyncio.sleep(0)
            if isinstance(prompt, list):
                prompt = TokensPrompt(prompt_token_ids=list(prompt))
            if not isinstance(prompt, StreamingInput):
                prompt = StreamingInput(prompt=prompt)
            if reads is not None:
                reads.append(prompt)
            yield prompt
        if then is not None:
            raise then

    return chunks()


WATCHDOG = 30.0
"""Seconds before a wait counts as a hang; no test relies on it for timing."""


async def _until(predicate, what: str) -> None:
    """Wait until `predicate()` holds."""

    async def poll():
        while not predicate():
            await asyncio.sleep(0.001)

    try:
        await asyncio.wait_for(poll(), WATCHDOG)
    except TimeoutError:
        pytest.fail(f"timed out waiting for {what}")


async def _let_ready_tasks_run() -> None:
    """Before a check that something did not happen: give every ready task the
    chance to do it."""
    for _ in range(50):
        await asyncio.sleep(0)


async def _open(llm: AsyncLLM, chunks, params: SamplingParams | None = None):
    """Open a stream and return its output collector."""
    return await llm._add_streaming_input_request("ext", chunks, params or _params())


def _sent(llm: AsyncLLM) -> list[EngineCoreRequest]:
    return [call.args[0] for call in llm.engine_core.add_request_async.await_args_list]


def _sent_prompts(llm: AsyncLLM) -> list[list[int]]:
    return [request.prompt_token_ids for request in _sent(llm)]


def _stream_id(llm: AsyncLLM) -> str:
    """The internal request id of the stream `llm` opened."""
    admission = llm.engine_core.admit_stream_async.await_args
    return admission.args[0].request_id if admission else _sent(llm)[0].request_id


def _aborted(llm: AsyncLLM) -> list[str]:
    return [
        request_id
        for call in llm.engine_core.abort_requests_async.await_args_list
        for request_id in call.args[0]
    ]


def _emit(
    llm: AsyncLLM,
    request_id: str,
    token_ids: list[int] | None = None,
    finish_reason: FinishReason | None = None,
    terminal: bool = False,
) -> None:
    llm.engine_core.feed.put_nowait(
        EngineCoreOutputs(
            outputs=[
                EngineCoreOutput(
                    request_id=request_id,
                    new_token_ids=token_ids or [],
                    finish_reason=finish_reason,
                    terminal=terminal,
                )
            ]
        )
    )


async def _ended(collector, errors: tuple[type[BaseException], ...] = ()) -> bool:
    """Whether the collector's next output ends its stream: a finished output,
    or one of the `errors` it was failed with."""
    try:
        output = await asyncio.wait_for(collector.get(), WATCHDOG)
    except errors:
        return True
    return output.finished


async def _consume(llm: AsyncLLM, chunks, params: SamplingParams | None = None):
    """Run `generate` over `chunks` to its end."""

    async def consume():
        return [out async for out in llm.generate(chunks, params or _params(), "ext")]

    return await asyncio.wait_for(consume(), WATCHDOG)


# @spec PSTATE-ADMIT-020
@pytest.mark.asyncio
async def test_state_stream_is_admitted_before_its_inputs_are_read(monkeypatch):
    llm = _state_llm(monkeypatch)
    events = []

    async def admit(request, *args, **kwargs):
        events.append("admit")

    llm.engine_core.admit_stream_async.side_effect = admit

    async def chunks():
        events.append("read")
        yield StreamingInput(prompt=TokensPrompt(prompt_token_ids=[1]))

    collector = await _open(llm, chunks())
    await _until(lambda: len(_sent(llm)) == 2, "the input and the close")

    assert events == ["admit", "read"]
    assert _stream_id(llm) == collector.request_id
    assert {request.request_id for request in _sent(llm)} == {collector.request_id}


# @spec PSTATE-ADMIT-020 PSTATE-ADMIT-017
@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["engine_paused", "active_limit"])
async def test_refused_admission_raises_before_reading_input(monkeypatch, reason):
    llm = _state_llm(monkeypatch)
    llm.engine_core.admit_stream_async.return_value = reason
    reads: list = []

    with pytest.raises(StreamAdmissionError) as excinfo:
        await _consume(llm, _chunks([1], reads=reads))

    assert excinfo.value.reason == reason
    assert not reads
    assert not _sent(llm)
    assert not _aborted(llm)
    assert llm.output_processor.get_num_unfinished_requests() == 0


# @spec PSTATE-ADMIT-020 PSTATE-DECL-013
@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["single_request", "ordinary_stream"])
async def test_single_requests_and_ordinary_streams_open_without_admission(
    monkeypatch, kind
):
    llm = _state_llm(monkeypatch, persistent_state=kind == "single_request")
    if kind == "single_request":
        await llm.add_request("single", TokensPrompt(prompt_token_ids=[1]), _params())
    else:
        await _open(llm, _chunks([1]))
        await _until(lambda: _sent(llm), "the input")

    llm.engine_core.admit_stream_async.assert_not_awaited()
    assert _sent_prompts(llm)[0] == [1]


def _id_fields(request_id: str) -> list[str]:
    """The fields an internal id adds to the external id `ext`."""
    assert request_id.startswith("ext")
    return re.findall(r"[0-9A-Za-z]+", request_id.removeprefix("ext"))


# @spec PSTATE-ADMIT-006
@pytest.mark.asyncio
async def test_state_stream_ids_share_a_nonce_per_frontend_and_count_up(
    monkeypatch,
):
    """Streams of one `AsyncLLM` differ only in a counter that increases; a
    second `AsyncLLM` on the same client index, as after a frontend restart,
    draws another nonce, so its first id differs from the first one's."""
    first = _state_llm(monkeypatch)
    ids = [(await _open(first, _chunks([1]))).request_id for _ in range(3)]
    second = _state_llm(monkeypatch)
    restarted = (await _open(second, _chunks([1]))).request_id

    fields = [_id_fields(request_id) for request_id in ids]
    assert len({len(f) for f in fields}) == 1
    [counter] = [i for i in range(len(fields[0])) if len({f[i] for f in fields}) > 1]
    assert [int(f[counter]) for f in fields] == sorted(
        {int(f[counter]) for f in fields}
    )
    assert "3" in fields[0]
    restarted_fields = _id_fields(restarted)
    assert len(restarted_fields) == len(fields[0])
    assert restarted != ids[0]
    assert any(
        restarted_fields[i] != fields[0][i]
        for i in range(len(restarted_fields))
        if i != counter
    )


# @spec PSTATE-ADMIT-006
@pytest.mark.asyncio
async def test_state_stream_inputs_and_close_carry_the_stream_id(monkeypatch):
    llm = _state_llm(monkeypatch)
    collector = await _open(llm, _chunks([1], [2]))
    await _until(lambda: len(_sent(llm)) == 3, "two inputs and the close")

    assert {request.request_id for request in _sent(llm)} == {collector.request_id}


# @spec PSTATE-TURN-005
@pytest.mark.asyncio
async def test_every_input_after_admission_carries_continuation(monkeypatch):
    llm = _state_llm(monkeypatch)
    await _open(llm, _chunks([1], [2], [3]))
    await _until(lambda: len(_sent(llm)) == 4, "three inputs and the close")

    assert [(r.resumable, r.continuation) for r in _sent(llm)] == [
        (True, True),
        (True, True),
        (True, True),
        (False, True),
    ]


# @spec PSTATE-TURN-017
@pytest.mark.asyncio
async def test_inputs_after_the_stream_ended_are_not_forwarded(monkeypatch):
    llm = _state_llm(monkeypatch)
    ended = asyncio.Event()

    async def chunks():
        yield StreamingInput(prompt=TokensPrompt(prompt_token_ids=[1]))
        await ended.wait()
        yield StreamingInput(prompt=TokensPrompt(prompt_token_ids=[2]))
        yield StreamingInput(prompt=TokensPrompt(prompt_token_ids=[3]))

    collector = await _open(llm, chunks())
    input_task = collector._input_stream_task
    await _until(lambda: _sent(llm), "the first input")
    stream_id = _stream_id(llm)
    _emit(llm, stream_id, [5], FinishReason.STOP, terminal=True)
    await _until(
        lambda: not llm.output_processor.has_request(stream_id), "the stream's end"
    )
    ended.set()
    await _until(input_task.done, "the input task to end")

    assert _sent_prompts(llm) == [[1]]
    assert await _ended(collector)


EMPTY_INPUTS = {
    "empty-text": "",
    "no-tokens": TokensPrompt(prompt_token_ids=[]),
    "empty-encoder-decoder": {
        "encoder_prompt": TokensPrompt(prompt_token_ids=[]),
        "decoder_prompt": TokensPrompt(prompt_token_ids=[]),
    },
    "rendered-empty": {
        "type": "multimodal",
        "prompt_token_ids": [],
        "mm_kwargs": {},
        "mm_hashes": {},
        "mm_placeholders": {},
    },
}

NON_EMPTY_INPUTS = {
    "text": "hi",
    "features": TokensPrompt(prompt_token_ids=[], multi_modal_data={"audio": [1]}),
    "encoder-tokens": {
        "encoder_prompt": TokensPrompt(prompt_token_ids=[1]),
        "decoder_prompt": None,
    },
    "rendered-decoder-tokens": {
        "type": "enc_dec",
        "encoder_prompt": {"type": "token", "prompt_token_ids": []},
        "decoder_prompt": {"type": "token", "prompt_token_ids": [7]},
    },
    "rendered-features": {
        "type": "multimodal",
        "prompt_token_ids": [],
        "mm_kwargs": {"audio": [object()]},
        "mm_hashes": {},
        "mm_placeholders": {},
    },
}


# @spec PSTATE-TURN-007 PSTATE-ADMIT-015
@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", EMPTY_INPUTS.values(), ids=EMPTY_INPUTS.keys())
async def test_empty_input_ends_the_stream_with_a_validation_error(monkeypatch, prompt):
    llm = _state_llm(monkeypatch)

    with pytest.raises(VLLMValidationError):
        await _consume(llm, _chunks([1], prompt, [2]))

    assert _sent_prompts(llm) == [[1]]
    assert _stream_id(llm) in _aborted(llm)


# @spec PSTATE-TURN-007
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt", NON_EMPTY_INPUTS.values(), ids=NON_EMPTY_INPUTS.keys()
)
async def test_input_with_any_content_is_forwarded(monkeypatch, prompt):
    llm = _state_llm(monkeypatch)

    await _open(llm, _chunks([1], prompt))
    await _until(lambda: len(_sent(llm)) == 3, "two inputs and the close")

    assert [r.resumable for r in _sent(llm)] == [True, True, False]


# @spec PSTATE-TURN-007 PSTATE-ADMIT-015
@pytest.mark.asyncio
@pytest.mark.parametrize("with_tokens", [False, True], ids=["embeddings", "mixed"])
async def test_input_with_prompt_embeddings_ends_the_stream(monkeypatch, with_tokens):
    llm = _state_llm(monkeypatch)
    prompt = {"prompt_embeds": torch.zeros(2, 4)}
    if with_tokens:
        prompt["prompt_token_ids"] = [7, 7]

    with pytest.raises(VLLMValidationError, match="prompt_embeds"):
        await _consume(llm, _chunks([1], prompt, [2]))

    assert _sent_prompts(llm) == [[1]]
    assert _stream_id(llm) in _aborted(llm)


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
]
"""Parameters whose effect outlives a turn, each alone."""

UPSTREAM_STREAMING_REFUSALS = [
    ("n", dict(n=2)),
    ("stop", dict(stop=["."])),
    ("output_kind", dict(output_kind=RequestOutputKind.FINAL_ONLY)),
]
"""What upstream input streaming refuses for every model."""

REFUSED_STREAM_PARAMETERS = [
    *REFUSED_SAMPLING_PARAMETERS,
    *UPSTREAM_STREAMING_REFUSALS,
    ("output_kind", dict(output_kind=RequestOutputKind.CUMULATIVE)),
]
"""Every parameter a state stream's input cannot carry."""

LOADED_PROCESSORS = [
    dict(logits_processors=["pkg.module:Processor"]),
    dict(installed_processors=True),
]
"""Custom logits processors from the model config, and from an entry point."""


# @spec PSTATE-DECL-007
@pytest.mark.asyncio
@pytest.mark.parametrize(("parameter", "params"), REFUSED_STREAM_PARAMETERS)
async def test_state_stream_refuses_generate_parameter_before_reading_input(
    monkeypatch, parameter, params
):
    llm = _state_llm(monkeypatch)
    reads: list = []

    with pytest.raises(
        VLLMValidationError, match=f"{parameter}.* is not supported for this model"
    ):
        await _consume(llm, _chunks([1], reads=reads), _params(**params))

    assert not reads
    assert not _sent(llm)


# @spec PSTATE-DECL-007 PSTATE-ADMIT-015
@pytest.mark.asyncio
@pytest.mark.parametrize("position", ["first", "update"])
@pytest.mark.parametrize(("parameter", "params"), REFUSED_STREAM_PARAMETERS)
async def test_input_carrying_a_refused_parameter_ends_the_stream(
    monkeypatch, parameter, params, position
):
    llm = _state_llm(monkeypatch)
    refused = StreamingInput(
        prompt=TokensPrompt(prompt_token_ids=[2]), sampling_params=_params(**params)
    )
    inputs = [refused, [3]] if position == "first" else [[1], refused, [3]]

    with pytest.raises(VLLMValidationError, match=parameter):
        await _consume(llm, _chunks(*inputs))

    assert _sent_prompts(llm) == ([] if position == "first" else [[1]])
    assert _stream_id(llm) in _aborted(llm)


# @spec PSTATE-DECL-007
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "loaded", LOADED_PROCESSORS, ids=["model_config", "entry_point"]
)
async def test_state_stream_refused_while_custom_logits_processors_are_loaded(
    monkeypatch, loaded
):
    llm = _state_llm(monkeypatch, **loaded)
    reads: list = []

    with pytest.raises(VLLMValidationError, match="logits_processors"):
        await _consume(llm, _chunks([1], reads=reads))

    assert not reads
    assert not _sent(llm)


# @spec PSTATE-DECL-007
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("loaded", "params"),
    [
        *[(loaded, {}) for loaded in LOADED_PROCESSORS],
        *[({}, params) for _, params in REFUSED_SAMPLING_PARAMETERS],
        *[({}, params) for _, params in UPSTREAM_STREAMING_REFUSALS],
        *[({}, dict(output_kind=kind)) for kind in RequestOutputKind],
    ],
)
async def test_single_state_request_keeps_upstream_validation(
    monkeypatch, loaded, params
):
    """A single request is one turn: nothing it carries outlives it."""
    llm = _state_llm(monkeypatch, **loaded)

    await llm.add_request(
        "single", TokensPrompt(prompt_token_ids=[1]), SamplingParams(**params)
    )

    assert len(_sent(llm)) == params.get("n", 1)


# @spec PSTATE-DECL-013 PSTATE-DECL-007 PSTATE-TURN-007
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("loaded", "params", "middle"),
    [
        *[({}, params, [2]) for _, params in REFUSED_SAMPLING_PARAMETERS],
        ({}, dict(output_kind=RequestOutputKind.CUMULATIVE), [2]),
        *[(loaded, {}, [2]) for loaded in LOADED_PROCESSORS],
        ({}, {}, TokensPrompt(prompt_token_ids=[])),
    ],
)
async def test_ordinary_stream_takes_what_state_streams_refuse(
    monkeypatch, loaded, params, middle
):
    """Every state-stream refusal is inactive for an ordinary model, the
    empty-input refusal included."""
    llm = _state_llm(monkeypatch, persistent_state=False, **loaded)

    await _open(llm, _chunks([1], middle, [3]), _params(**params))
    await _until(lambda: len(_sent(llm)) == 4, "three inputs and the close")

    assert [r.resumable for r in _sent(llm)] == [True, True, True, False]


# @spec PSTATE-DECL-013 PSTATE-HIST-006
@pytest.mark.asyncio
async def test_ordinary_stream_reads_inputs_without_credits(monkeypatch):
    llm = _state_llm(monkeypatch, persistent_state=False, max_pending=1)
    reads: list = []

    await _open(llm, _chunks(*([i] for i in range(1, 7)), reads=reads))
    await _until(lambda: len(_sent(llm)) == 7, "six inputs and the close")

    assert len(reads) == 6


# @spec PSTATE-DECL-013
@pytest.mark.asyncio
@pytest.mark.parametrize(("parameter", "params"), UPSTREAM_STREAMING_REFUSALS)
async def test_ordinary_stream_keeps_upstream_streaming_refusals(
    monkeypatch, parameter, params
):
    llm = _state_llm(monkeypatch, persistent_state=False)

    with pytest.raises(
        VLLMValidationError, match="Input streaming not currently supported"
    ):
        await _consume(llm, _chunks([1]), _params(**params))

    assert not _sent(llm)


# @spec PSTATE-HIST-006
@pytest.mark.asyncio
async def test_input_iterator_is_not_advanced_while_no_credit_is_free(monkeypatch):
    llm = _state_llm(monkeypatch, max_pending=2)
    reads: list = []

    await _open(llm, _chunks(*([i] for i in range(1, 7)), reads=reads))
    await _until(lambda: len(_sent(llm)) >= 2, "two inputs")
    await _let_ready_tasks_run()

    assert len(reads) == 2
    assert _sent_prompts(llm) == [[1], [2]]


# @spec PSTATE-HIST-006
@pytest.mark.asyncio
async def test_credit_returns_once_the_consumer_receives_the_turn_end(monkeypatch):
    """Even a turn end that carries no text releases its input's credit."""
    llm = _state_llm(monkeypatch, max_pending=2)
    reads: list = []
    collector = await _open(llm, _chunks([1], [2], [3], [4], reads=reads))
    await _until(lambda: len(_sent(llm)) >= 2, "two inputs")

    _emit(llm, _stream_id(llm), [], FinishReason.STOP)
    await _until(collector.ready.is_set, "the turn end to reach the collector")
    await _let_ready_tasks_run()
    assert len(reads) == 2

    boundary = collector.get_nowait()
    await _until(lambda: len(_sent(llm)) >= 3, "the third input")
    await _let_ready_tasks_run()

    assert boundary is not None and not boundary.finished
    assert boundary.outputs[0].finish_reason == "stop"
    assert len(reads) == 3
    assert _sent_prompts(llm) == [[1], [2], [3]]


# @spec PSTATE-HIST-006 PSTATE-TURN-018
@pytest.mark.asyncio
async def test_stream_closes_and_ends_with_its_only_credit_held(monkeypatch):
    """The close and the terminal output take no credit."""
    llm = _state_llm(monkeypatch, max_pending=1)
    consumer = asyncio.ensure_future(_consume(llm, _chunks([1])))
    await _until(lambda: _sent(llm), "the input")
    await _let_ready_tasks_run()
    assert [r.resumable for r in _sent(llm)] == [True]

    _emit(llm, _stream_id(llm), [5], FinishReason.STOP)
    await _until(lambda: len(_sent(llm)) == 2, "the close")
    assert [r.resumable for r in _sent(llm)] == [True, False]

    _emit(llm, _stream_id(llm), [], FinishReason.STOP, terminal=True)
    outputs = await asyncio.wait_for(consumer, WATCHDOG)

    assert [output.finished for output in outputs] == [False, True]
    await _let_ready_tasks_run()
    assert [r.resumable for r in _sent(llm)] == [True, False]


# @spec PSTATE-HIST-006 PSTATE-ADMIT-015
@pytest.mark.asyncio
async def test_cancelled_input_task_waiting_for_a_credit_aborts_the_stream(
    monkeypatch,
):
    llm = _state_llm(monkeypatch, max_pending=1)
    reads: list = []
    collector = await _open(llm, _chunks([1], [2], [3], [4], reads=reads))
    await _until(lambda: _sent(llm), "the first input")
    await _let_ready_tasks_run()
    assert len(reads) == 1
    input_task = collector._input_stream_task

    input_task.cancel()
    await _until(input_task.done, "the cancelled input task to end")
    await _until(lambda: _stream_id(llm) in _aborted(llm), "the abort")

    assert len(reads) == 1
    assert await _ended(collector, (async_llm_module.InputStreamError,))


# @spec PSTATE-ADMIT-015
@pytest.mark.asyncio
async def test_input_task_error_aborts_the_stream_and_fails_its_collector(
    monkeypatch,
):
    llm = _state_llm(monkeypatch)

    collector = await _open(llm, _chunks([1], then=RuntimeError("source failed")))
    await _until(lambda: _sent(llm), "the input")
    await _until(lambda: _stream_id(llm) in _aborted(llm), "the abort")

    assert await _ended(collector, (async_llm_module.InputStreamError, RuntimeError))


# @spec PSTATE-ADMIT-016
@pytest.mark.asyncio
async def test_engine_core_death_fails_every_open_state_stream(monkeypatch):
    llm = _state_llm(monkeypatch)
    collectors = [await _open(llm, _chunks([1], [2])) for _ in range(2)]

    llm.engine_core.feed.put_nowait(EngineDeadError())

    for collector in collectors:
        with pytest.raises(EngineDeadError):
            await asyncio.wait_for(collector.get(), WATCHDOG)


# @spec PSTATE-ADMIT-017
@pytest.mark.asyncio
async def test_cancelled_admission_aborts_the_stream(monkeypatch):
    llm = _state_llm(monkeypatch)
    admitting = asyncio.Event()

    async def admit(request, *args, **kwargs):
        admitting.set()
        await asyncio.Event().wait()

    llm.engine_core.admit_stream_async.side_effect = admit
    consumer = asyncio.ensure_future(_consume(llm, _chunks([1])))
    await asyncio.wait_for(admitting.wait(), WATCHDOG)

    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert _stream_id(llm) in _aborted(llm)
    assert llm.output_processor.get_num_unfinished_requests() == 0


# @spec PSTATE-ADMIT-017
@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["close", "close_after_terminal", "cancel"])
async def test_output_generator_ending_early_aborts_the_stream(monkeypatch, ending):
    """Before its terminal output is delivered, whether or not the output
    processor still holds the stream."""
    llm = _state_llm(monkeypatch)
    outputs = llm.generate(_chunks([1], [2]), _params(), "ext")
    first = asyncio.ensure_future(outputs.__anext__())
    await _until(lambda: _sent(llm), "the first input")
    stream_id = _stream_id(llm)
    _emit(llm, stream_id, [5], FinishReason.STOP)
    assert not (await asyncio.wait_for(first, WATCHDOG)).finished

    if ending == "cancel":
        second = asyncio.ensure_future(outputs.__anext__())
        await _let_ready_tasks_run()
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
    else:
        if ending == "close_after_terminal":
            _emit(llm, stream_id, [], FinishReason.STOP, terminal=True)
            await _until(
                lambda: not llm.output_processor.has_request(stream_id),
                "the terminal output",
            )
        await outputs.aclose()

    assert stream_id in _aborted(llm)


# Scheduler-side admission


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
    request.client_index = 1
    return request


def _step(scheduler, req_ids: list[str], sampled: list[list[int]]):
    output = scheduler.schedule()
    return scheduler.update_from_output(
        output, ModelRunnerOutput(req_ids=req_ids, sampled_token_ids=sampled)
    )


def _park(scheduler, request_id: str):
    request = _stream_request(request_id)
    scheduler.add_request(request)
    _step(scheduler, [request_id], [[EOS_TOKEN_ID]])
    assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    return request


# @spec PSTATE-ADMIT-001
def test_admission_refuses_beyond_the_active_limit(monkeypatch):
    scheduler = _state_scheduler(monkeypatch, max_num_seqs=1)
    assert scheduler.admit_stream(_stream_request("a", 0)) is None
    assert scheduler.admit_stream(_stream_request("b", 0)) == "active_limit"
    assert not scheduler.has_request("b")


# @spec PSTATE-ADMIT-001
@pytest.mark.parametrize("pause_state", [PauseState.PAUSED_NEW, PauseState.PAUSED_ALL])
def test_admission_refuses_while_paused(monkeypatch, pause_state):
    scheduler = _state_scheduler(monkeypatch)
    scheduler.set_pause_state(pause_state)
    assert scheduler.admit_stream(_stream_request("a", 0)) == "engine_paused"
    assert not scheduler.has_request("a")


# Engine core


def _bare_core(scheduler) -> EngineCoreProc:
    """A bare engine core over `scheduler` whose executor is a mock and which
    publishes into a `_PublishLog`."""
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
    core.scheduler = scheduler
    core.batch_queue = None
    core.engines_running = False
    core.has_coordinator = False
    core.reject_new_requests = False
    core.shutdown_state = EngineShutdownState.RUNNING
    core._idle_state_callbacks = []
    core.output_queue = _PublishLog(core)
    core.input_queue = queue.Queue()
    core.aborts_queue = queue.Queue()
    core._deferred_inputs = deque()
    core.process_input_queue_block = False
    return core


def _core_request(request_id: str, prompt_token_ids: list[int], **kwargs):
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=4, output_kind=DELTA),
        pooling_params=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        **kwargs,
    )


def _free_pages(scheduler) -> int:
    return scheduler.kv_cache_manager.block_pool.get_num_free_blocks()


class _PublishLog(queue.Queue):
    """An output queue that records each output it receives with whether the
    scheduler still held its request, and the free page count, at that moment."""

    def __init__(self, core: EngineCoreProc) -> None:
        super().__init__()
        self.core = core
        self.published: list[tuple[int, EngineCoreOutput, bool, int]] = []

    def put_nowait(self, item) -> None:
        client_index, outputs = item
        scheduler = self.core.scheduler
        for output in outputs.outputs:
            self.published.append(
                (
                    client_index,
                    output,
                    scheduler.has_request(output.request_id),
                    _free_pages(scheduler),
                )
            )
        super().put_nowait(item)


def _live_core(
    monkeypatch: pytest.MonkeyPatch, ordinary: bool = False, **scheduler_kwargs
):
    """A bare engine core over a real scheduler, a state model's unless
    `ordinary`, and the scheduler's free page count before any request."""
    scheduler = (
        create_scheduler(max_model_len=64, block_size=64, **scheduler_kwargs)
        if ordinary
        else _state_scheduler(monkeypatch, **scheduler_kwargs)
    )
    return _bare_core(scheduler), _free_pages(scheduler)


def _add(core: EngineCoreProc, request) -> None:
    request.client_index = 1
    core._handle_client_request(EngineCoreRequestType.ADD, (request, 0))


def _close(request_id: str = "s"):
    return _stream_request(request_id, 1, resumable=False, continuation=True)


def _update(request_id: str = "s", num_tokens: int = 1):
    return _stream_request(request_id, num_tokens, continuation=True)


# @spec PSTATE-TURN-006
@pytest.mark.parametrize("paused", [False, True], ids=["running", "paused"])
@pytest.mark.parametrize("make_input", [_update, _close], ids=["input", "close"])
def test_continuation_for_an_unheld_stream_is_dropped_silently(
    monkeypatch, paused, make_input
):
    core, _ = _live_core(monkeypatch)
    core.reject_new_requests = paused

    _add(core, make_input())

    assert not core.scheduler.has_request("s")
    assert core.output_queue.published == []


# @spec PSTATE-TURN-006
def test_continuation_for_a_held_stream_joins_it(monkeypatch):
    core, _ = _live_core(monkeypatch)
    request = _park(core.scheduler, "s")

    _add(core, _update("s", 2))

    assert request.status == RequestStatus.WAITING
    assert core.output_queue.published == []


def _hold(core: EngineCoreProc, kind: str):
    if kind == "admitted":
        request = _stream_request("s", 0)
        assert core.scheduler.admit_stream(request) is None
        return request
    return _park(core.scheduler, "s")


# @spec PSTATE-TURN-015 PSTATE-TURN-003 PSTATE-PAGE-004
@pytest.mark.parametrize("kind", ["parked", "admitted"])
def test_close_of_a_parked_stream_publishes_its_end_after_freeing_it(monkeypatch, kind):
    core, free_pages = _live_core(monkeypatch)
    request = _hold(core, kind)

    _add(core, _close())

    [(client_index, output, held, free_at_publish)] = core.output_queue.published
    assert request.status == RequestStatus.FINISHED_STOPPED
    assert (client_index, output.finish_reason, output.terminal) == (
        1,
        FinishReason.STOP,
        True,
    )
    assert not held
    assert free_at_publish == free_pages


def _drain_inputs(core: EngineCoreProc) -> None:
    """Handle every queued input, as the busy loop does before a step."""
    core.has_work = lambda: True
    core._process_input_queue()
    del core.has_work


def _reject_in_validation(core: EngineCoreProc, request: EngineCoreRequest) -> None:
    core._handle_request_validation_error(request, VLLMValidationError("bad input"))


def _reject_in_preprocessing(core: EngineCoreProc, request: EngineCoreRequest):
    core._handle_request_preproc_error(request)


def _reject_in_shutdown(core: EngineCoreProc, request: EngineCoreRequest) -> None:
    core.shutdown_state = EngineShutdownState.REQUESTED
    _add(core, _update("s"))


# @spec PSTATE-PAGE-004 PSTATE-TURN-003
@pytest.mark.parametrize(
    "reject",
    [_reject_in_validation, _reject_in_preprocessing, _reject_in_shutdown],
    ids=["validation", "preprocessing", "shutdown"],
)
def test_rejected_input_ends_its_held_stream_after_freeing_it(monkeypatch, reject):
    """The rejection goes through the core loop and finishes the stream there,
    instead of publishing an error beside a held page."""
    core, free_pages = _live_core(monkeypatch)
    _park(core.scheduler, "s")
    rejected = _core_request(
        "s", [1], resumable=True, continuation=True, client_index=1
    )

    reject(core, rejected)
    _drain_inputs(core)

    [(client_index, output, held, free_at_publish)] = core.output_queue.published
    assert client_index == 1
    assert output.finish_reason is not None
    assert output.terminal is True
    assert not held
    assert free_at_publish == free_pages


# @spec PSTATE-ADMIT-014 PSTATE-TURN-008 PSTATE-TURN-006 PSTATE-PAGE-004
def test_input_that_ends_its_stream_is_published_at_its_add_boundary(monkeypatch):
    """A fold past the position limit ends the stream before the next input is
    handled; that input then names a stream core no longer holds."""
    core, free_pages = _live_core(monkeypatch)
    _park(core.scheduler, "s")

    _add(core, _update("s", 64))
    published = list(core.output_queue.published)
    _add(core, _update("s"))

    [(_, output, held, free_at_publish)] = published
    assert (output.finish_reason, output.terminal) == (FinishReason.LENGTH, True)
    assert not held
    assert free_at_publish == free_pages
    assert core.output_queue.published == published
    assert not core.scheduler.has_request("s")


# @spec PSTATE-TURN-003 PSTATE-PRES-006 PSTATE-PAGE-004 PSTATE-DECL-013
@pytest.mark.parametrize("ordinary", [False, True], ids=["state", "ordinary"])
def test_abort_mode_pause_publishes_each_stream_end_after_freeing_it(
    monkeypatch, ordinary
):
    core, free_pages = _live_core(monkeypatch, ordinary=ordinary)
    _park(core.scheduler, "s")

    core.pause_scheduler(mode="abort", clear_cache=False)

    [(_, output, held, free_at_publish)] = core.output_queue.published
    assert output.finish_reason == FinishReason.ABORT
    assert output.terminal is not ordinary
    assert not held
    assert free_at_publish == free_pages


# @spec PSTATE-DECL-013
def test_ordinary_stream_closed_while_parked_publishes_nothing(monkeypatch):
    core, _ = _live_core(monkeypatch, ordinary=True)
    request = _park(core.scheduler, "s")

    _add(core, _close())

    assert request.status == RequestStatus.FINISHED_ABORTED
    assert not core.scheduler.has_request("s")
    assert core.output_queue.published == []


# @spec PSTATE-PAGE-004
def test_no_add_or_utility_overtakes_an_earlier_message_of_its_client(monkeypatch):
    """A utility deferred while a step runs still precedes every later add."""
    core, _ = _live_core(monkeypatch)
    handled: list[str] = []
    core._handle_client_request = lambda kind, request: handled.append(request)
    for message in (
        (EngineCoreRequestType.UTILITY, "admit"),
        (EngineCoreRequestType.ADD, "input"),
        (EngineCoreRequestType.ABORT, "abort"),
        (EngineCoreRequestType.ADD, "next"),
    ):
        core.input_queue.put(message)
    pending = core.input_queue.queue
    step_output = SimpleNamespace(
        poll=lambda: not pending,
        wait_briefly=lambda *args: None,
        result=lambda: None,
    )

    core._wait_for_output(step_output)
    _drain_inputs(core)

    assert [h for h in handled if h != "abort"] == ["admit", "input", "next"]


def _admission() -> EngineCoreRequest:
    return _core_request("s", [], resumable=True, client_index=1)


def _hold_single_request(core: EngineCoreProc) -> None:
    request = _stream_request("held-1", resumable=False)
    core.scheduler.add_request(request)
    core.scheduler.schedule()


def _admit_full(core: EngineCoreProc) -> None:
    assert core.scheduler.admit_stream(_stream_request("held-1", 0)) is None


def _admit_paused(core: EngineCoreProc) -> None:
    core.pause_scheduler(mode="keep", clear_cache=False)


def _admit_draining_pause(core: EngineCoreProc) -> None:
    _hold_single_request(core)
    future = core.pause_scheduler(mode="wait", clear_cache=True)
    assert future is not None and not future.done()


def _admit_draining_sleep(core: EngineCoreProc) -> None:
    _hold_single_request(core)
    future = core.sleep(level=1, mode="wait")
    assert future is not None and not future.done()


# @spec PSTATE-ADMIT-001
@pytest.mark.parametrize(
    ("prepare", "reason"),
    [
        (None, None),
        (_admit_full, "active_limit"),
        (_admit_paused, "engine_paused"),
        (_admit_draining_pause, "engine_paused"),
        (_admit_draining_sleep, "engine_paused"),
    ],
    ids=["admitted", "active_limit", "paused", "draining_pause", "draining_sleep"],
)
def test_admit_stream_admits_or_returns_the_refusal_reason(
    monkeypatch, prepare, reason
):
    """A refusal creates no request and publishes nothing; a wait-mode pause or
    sleep that clears the cache refuses while it drains a single request."""
    core, _ = _live_core(monkeypatch, max_num_seqs=1)
    if prepare is not None:
        prepare(core)

    assert core.admit_stream(_admission()) == reason

    assert core.scheduler.has_request("s") is (reason is None)
    assert core.output_queue.published == []


def _holding_core(monkeypatch: pytest.MonkeyPatch, kind: str) -> EngineCoreProc:
    """A bare engine core whose real scheduler holds `held-1`: a running or a
    parked state stream, or a running single request."""
    core, _ = _live_core(monkeypatch)
    request = _stream_request("held-1", resumable=kind != "single_request")
    core.scheduler.add_request(request)
    if kind == "parked_stream":
        _step(core.scheduler, ["held-1"], [[EOS_TOKEN_ID]])
    else:
        core.scheduler.schedule()
    return core


HELD_STREAMS = ["running_stream", "parked_stream"]
HELD_REQUESTS = [*HELD_STREAMS, "single_request"]

KEEP_MODE_OPERATIONS = {
    "reset_caches": lambda core: core._reset_caches(),
    "release_kv_cache_memory": lambda core: core.release_kv_cache_memory(),
    "pause": lambda core: core.pause_scheduler(mode="keep", clear_cache=True),
    "sleep": lambda core: core.sleep(level=1, mode="keep"),
}
WAIT_MODE_OPERATIONS = {
    "pause": lambda core: core.pause_scheduler(mode="wait", clear_cache=True),
    "sleep": lambda core: core.sleep(level=1, mode="wait"),
}


def _assert_refused(core: EngineCoreProc, operation) -> None:
    with pytest.raises(RuntimeError, match="held-1"):
        operation(core)
    assert core.scheduler.pause_state == PauseState.UNPAUSED
    assert core.scheduler.has_request("held-1")
    core.model_executor.sleep.assert_not_called()
    core.model_executor.discard.assert_not_called()


# @spec PSTATE-PRES-004
@pytest.mark.parametrize("kind", HELD_REQUESTS)
@pytest.mark.parametrize(
    "operation", KEEP_MODE_OPERATIONS.values(), ids=KEEP_MODE_OPERATIONS.keys()
)
def test_cache_clearing_is_refused_on_arrival_while_any_state_request_is_held(
    monkeypatch, kind, operation
):
    _assert_refused(_holding_core(monkeypatch, kind), operation)


# @spec PSTATE-PRES-004
@pytest.mark.parametrize("kind", HELD_STREAMS)
@pytest.mark.parametrize(
    "operation", WAIT_MODE_OPERATIONS.values(), ids=WAIT_MODE_OPERATIONS.keys()
)
def test_draining_cache_clear_is_refused_on_arrival_while_a_stream_is_held(
    monkeypatch, kind, operation
):
    _assert_refused(_holding_core(monkeypatch, kind), operation)


# @spec PSTATE-PRES-004
def test_draining_cache_clear_waits_for_a_single_request(monkeypatch):
    """A wait-mode pause drains a single request rather than refusing it."""
    core = _holding_core(monkeypatch, "single_request")

    future = core.pause_scheduler(mode="wait", clear_cache=True)

    assert future is not None and not future.done()
    assert core.scheduler.pause_state == PauseState.PAUSED_NEW


# @spec PSTATE-PRES-004
@pytest.mark.parametrize("kind", HELD_STREAMS)
@pytest.mark.parametrize("mode", ["wait", "keep"])
def test_checkpoint_prepare_names_the_held_streams_and_changes_nothing(
    monkeypatch, kind, mode
):
    core = _holding_core(monkeypatch, kind)
    reason = core.checkpoint_prepare(mode)
    assert reason is not None and "held-1" in reason
    assert core._checkpoint_state == CheckpointState.RUNNING
    assert core.scheduler.pause_state == PauseState.UNPAUSED


# @spec PSTATE-PRES-006 PSTATE-PAGE-004
@pytest.mark.parametrize("operation", ["pause", "sleep", "checkpoint_prepare"])
def test_abort_mode_frees_every_request_before_the_operation(monkeypatch, operation):
    """Parked and running streams and a single request are all finished, their
    pages back in the pool, when the cache reset, sleep or KV cache discard
    begins."""
    core, free_pages = _live_core(monkeypatch)
    scheduler = core.scheduler
    _park(scheduler, "parked")
    for request_id, resumable in (("running", True), ("single", False)):
        scheduler.add_request(_stream_request(request_id, resumable=resumable))
    scheduler.schedule()
    began: list[tuple[str, set[str], int]] = []

    def record(name: str) -> None:
        began.append((name, set(scheduler.requests), _free_pages(scheduler)))

    reset_prefix_cache = scheduler.reset_prefix_cache

    def recording_reset(*args, **kwargs):
        record("reset")
        return reset_prefix_cache(*args, **kwargs)

    monkeypatch.setattr(scheduler, "reset_prefix_cache", recording_reset)
    core.model_executor.sleep.side_effect = lambda level: record("sleep")
    core.model_executor.discard.side_effect = lambda tags: record("discard")

    if operation == "pause":
        core.pause_scheduler(mode="abort", clear_cache=True)
    elif operation == "sleep":
        core.sleep(level=1, mode="abort")
    else:
        assert core.checkpoint_prepare("abort") is None
        assert core._checkpoint_state == CheckpointState.PREPARED

    expected = {"pause": "reset", "sleep": "sleep", "checkpoint_prepare": "discard"}
    assert expected[operation] in [name for name, _, _ in began]
    assert all(held == set() and free == free_pages for _, held, free in began)
    ended = {
        output.request_id: output for _, output, _, _ in core.output_queue.published
    }
    assert set(ended) == {"parked", "running", "single"}
    assert all(o.finish_reason == FinishReason.ABORT for o in ended.values())


# @spec PSTATE-PRES-007
@pytest.mark.parametrize("kind", HELD_STREAMS)
@pytest.mark.parametrize("mode", ["wait", "keep"])
def test_sleep_level_zero_is_allowed_while_a_stream_is_held(monkeypatch, kind, mode):
    core = _holding_core(monkeypatch, kind)
    free_pages = _free_pages(core.scheduler)

    core.sleep(level=0, mode=mode)

    assert core.scheduler.pause_state != PauseState.UNPAUSED
    assert core.scheduler.has_request("held-1")
    assert _free_pages(core.scheduler) == free_pages
    core.model_executor.sleep.assert_not_called()


# Engine-core guards that hold for every client


def _only_output(core: EngineCoreProc):
    [published] = core.output_queue.published
    return published


def _assert_ended_after_free(core, free_pages, finish_reason, code=None):
    client_index, output, held, free_at_publish = _only_output(core)
    assert (client_index, output.finish_reason, output.terminal) == (
        1,
        finish_reason,
        True,
    )
    assert not held
    assert free_at_publish == free_pages
    if code is not None:
        assert output.error is not None and output.error.code == code
    return output


def _hold_kind(core: EngineCoreProc, kind: str):
    """Hold the state stream `s`: admitted with no input, waiting with its
    first input, or parked after a turn."""
    if kind == "waiting":
        request = _stream_request("s")
        core.scheduler.add_request(request)
        return request
    return _hold(core, kind)


HOLD_KINDS = ["admitted", "waiting", "parked"]


def _abort_from_input_queue(core: EngineCoreProc) -> None:
    core._handle_client_request(EngineCoreRequestType.ABORT, ["s"])


def _abort_from_eager_queue(core: EngineCoreProc) -> None:
    core.aborts_queue.put_nowait(["s"])
    core._process_aborts_queue()


# @spec PSTATE-TURN-003 PSTATE-PAGE-004
def test_abort_of_a_running_stream_publishes_one_end(monkeypatch):
    """The step in flight when the abort lands publishes nothing more."""
    core, free_pages = _live_core(monkeypatch)
    core.scheduler.add_request(_stream_request("s"))
    in_flight = core.scheduler.schedule()

    _abort_from_input_queue(core)
    outputs = core.scheduler.update_from_output(
        in_flight, ModelRunnerOutput(req_ids=["s"], sampled_token_ids=[[7]])
    )

    _assert_ended_after_free(core, free_pages, FinishReason.ABORT)
    assert not any(
        output.request_id == "s"
        for client_outputs in outputs.values()
        for output in client_outputs.outputs
    )


# @spec PSTATE-TURN-019 PSTATE-TURN-015
def test_close_without_tokens_still_ends_a_parked_stream(monkeypatch):
    core, free_pages = _live_core(monkeypatch)
    request = _park(core.scheduler, "s")

    _add(core, _stream_request("s", 0, resumable=False, continuation=True))

    assert request.status == RequestStatus.FINISHED_STOPPED
    _assert_ended_after_free(core, free_pages, FinishReason.STOP)


# @spec PSTATE-ADMIT-022 PSTATE-DECL-013
def test_single_request_add_keeps_the_upstream_paths(monkeypatch):
    """Neither admitted nor refused at the limit, and rejected in shutdown
    with a non-terminal abort output."""
    core, _ = _live_core(monkeypatch, max_num_seqs=1)
    _admit_full(core)

    _add(core, _stream_request("single", 3, resumable=False))
    assert core.scheduler.has_request("single")
    assert core.output_queue.published == []

    core.shutdown_state = EngineShutdownState.REQUESTED
    _add(core, _stream_request("late", 3, resumable=False))
    [(_, output, _, _)] = core.output_queue.published
    assert (output.request_id, output.finish_reason, output.terminal) == (
        "late",
        FinishReason.ABORT,
        False,
    )


# @spec PSTATE-TURN-003 PSTATE-PAGE-004
@pytest.mark.parametrize("kind", HOLD_KINDS)
@pytest.mark.parametrize(
    "abort",
    [_abort_from_input_queue, _abort_from_eager_queue],
    ids=["input_queue", "eager_queue"],
)
def test_explicit_abort_publishes_the_stream_end_after_freeing_it(
    monkeypatch, kind, abort
):
    core, free_pages = _live_core(monkeypatch)
    request = _hold_kind(core, kind)

    abort(core)
    abort(core)

    assert request.status == RequestStatus.FINISHED_ABORTED
    _assert_ended_after_free(core, free_pages, FinishReason.ABORT)


# @spec PSTATE-TURN-003 PSTATE-DECL-013
@pytest.mark.parametrize("ordinary", [False, True], ids=["single_request", "ordinary"])
def test_abort_of_a_single_request_or_an_ordinary_stream_publishes_nothing(
    monkeypatch, ordinary
):
    core, _ = _live_core(monkeypatch, ordinary=ordinary)
    core.scheduler.add_request(_stream_request("s", resumable=ordinary))

    _abort_from_input_queue(core)

    assert not core.scheduler.has_request("s")
    assert core.output_queue.published == []


STRUCTURED = SamplingParams(
    max_tokens=4,
    output_kind=DELTA,
    structured_outputs=StructuredOutputsParams(regex="a+"),
)


def _no_tokens(request_id: str = "s", continuation: bool = False):
    return _stream_request(request_id, 0, continuation=continuation)


def _features_without_tokens(request_id: str = "s", continuation: bool = False):
    request = _no_tokens(request_id, continuation)
    request.mm_features = [_feature(offset=0, length=2)]
    return request


def _with_embeddings(request_id: str = "s", continuation: bool = False):
    request = _stream_request(request_id, 2, continuation=continuation)
    request.prompt_embeds = torch.zeros(2, 4)
    return request


def _with_structured_outputs(request_id: str = "s", continuation: bool = False):
    request = _stream_request(
        request_id, 2, continuation=continuation, sampling_params=STRUCTURED
    )
    assert request.use_structured_output
    return request


def _feature(offset: int, length: int) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=None,
        modality="audio",
        identifier=f"audio-{offset}",
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


INAPPLICABLE = {
    "no_tokens": _no_tokens,
    "features_without_tokens": _features_without_tokens,
    "embeddings": _with_embeddings,
    "structured_outputs": _with_structured_outputs,
}


# @spec PSTATE-TURN-019 PSTATE-PAGE-004
@pytest.mark.parametrize("kind", HOLD_KINDS)
@pytest.mark.parametrize("make", INAPPLICABLE.values(), ids=INAPPLICABLE.keys())
def test_inapplicable_continuation_ends_its_stream_with_a_validation_error(
    monkeypatch, kind, make
):
    core, free_pages = _live_core(monkeypatch)
    request = _hold_kind(core, kind)

    _add(core, make(continuation=True))

    assert request.status == RequestStatus.FINISHED_ERROR
    _assert_ended_after_free(core, free_pages, FinishReason.ERROR, "validation_error")


# @spec PSTATE-TURN-019
@pytest.mark.parametrize("make", INAPPLICABLE.values(), ids=INAPPLICABLE.keys())
def test_inapplicable_opener_creates_no_request(monkeypatch, make):
    core, free_pages = _live_core(monkeypatch)

    _add(core, make())

    assert not core.scheduler.has_request("s")
    _assert_ended_after_free(core, free_pages, FinishReason.ERROR, "validation_error")


# @spec PSTATE-TURN-019
def test_feature_input_with_placeholder_tokens_is_applied(monkeypatch):
    """After preprocessing a feature occupies placeholder token positions, so
    a feature input has tokens and joins the stream."""
    core, _ = _live_core(monkeypatch)
    request = _park(core.scheduler, "s")
    update = _update("s", 3)
    update.mm_features = [_feature(offset=0, length=3)]

    _add(core, update)

    assert request.status == RequestStatus.WAITING
    assert [f.identifier for f in request.mm_features] == ["audio-0"]
    assert core.output_queue.published == []


# @spec PSTATE-TURN-019 PSTATE-ADMIT-022
def test_admit_stream_refuses_structured_outputs(monkeypatch):
    core, _ = _live_core(monkeypatch)
    admission = _admission()
    admission.sampling_params = STRUCTURED

    assert core.admit_stream(admission) == "validation_error"

    assert not core.scheduler.has_request("s")
    assert core.output_queue.published == []


# @spec PSTATE-TURN-006
@pytest.mark.parametrize("make_input", [_update, _close], ids=["input", "close"])
def test_continuation_naming_a_held_single_request_is_dropped(monkeypatch, make_input):
    core, _ = _live_core(monkeypatch)
    single = _stream_request("s", resumable=False)
    core.scheduler.add_request(single)

    _add(core, make_input())

    assert core.scheduler.requests["s"] is single
    assert single.status == RequestStatus.WAITING
    assert core.output_queue.published == []


# @spec PSTATE-ADMIT-021 PSTATE-PAGE-004
@pytest.mark.parametrize("kind", [*HOLD_KINDS, "single_request"])
def test_opener_for_a_held_id_ends_the_held_request(monkeypatch, kind):
    core, free_pages = _live_core(monkeypatch)
    if kind == "single_request":
        held = _stream_request("s", resumable=False)
        core.scheduler.add_request(held)
    else:
        held = _hold_kind(core, kind)

    _add(core, _stream_request("s", 2))

    assert held.status == RequestStatus.FINISHED_ERROR
    assert not core.scheduler.has_request("s")
    _assert_ended_after_free(core, free_pages, FinishReason.ERROR)


# @spec PSTATE-ADMIT-021 PSTATE-ADMIT-022
@pytest.mark.parametrize("paused", [False, True], ids=["running", "paused"])
def test_admit_stream_for_a_held_id_ends_the_held_stream(monkeypatch, paused):
    core, free_pages = _live_core(monkeypatch)
    held = _hold_kind(core, "parked")
    if paused:
        _admit_paused(core)

    assert core.admit_stream(_admission()) == "duplicate_request_id"

    assert held.status == RequestStatus.FINISHED_ERROR
    _assert_ended_after_free(core, free_pages, FinishReason.ERROR)


# @spec PSTATE-ADMIT-018 PSTATE-ADMIT-002
def test_unadmitted_opener_is_admitted_and_runs_its_input(monkeypatch):
    core, _ = _live_core(monkeypatch)
    opener = _stream_request("s", 3)

    _add(core, opener)

    held = core.scheduler.requests["s"]
    assert held.status == RequestStatus.WAITING
    assert held in core.scheduler.resumed_streaming_sessions
    assert len(held.prompt_token_ids) == 3
    assert (held.num_computed_tokens, held.position_offset) == (0, 0)
    assert core.output_queue.published == []
    assert "s" in core.scheduler.schedule().num_scheduled_tokens


# @spec PSTATE-ADMIT-018 PSTATE-ADMIT-004
def test_second_unadmitted_opener_beyond_the_active_limit_is_refused(monkeypatch):
    """The first opener reserves its row on arrival, so a second one that
    arrives before any step is refused."""
    core, _ = _live_core(monkeypatch, max_num_seqs=1)

    _add(core, _stream_request("a", 3))
    _add(core, _stream_request("b", 3))

    assert core.scheduler.has_request("a")
    assert not core.scheduler.has_request("b")
    client_index, output, held, _ = _only_output(core)
    assert (output.request_id, output.finish_reason, output.terminal) == (
        "b",
        FinishReason.ERROR,
        True,
    )
    assert (output.error.code, output.error.retryable) == ("active_limit", True)
    assert not held


# @spec PSTATE-ADMIT-018 PSTATE-ADMIT-022
def test_unadmitted_opener_while_paused_is_refused_terminally(monkeypatch):
    core, _ = _live_core(monkeypatch)
    _admit_paused(core)
    core.reject_new_requests = True

    _add(core, _stream_request("s", 3))

    assert not core.scheduler.has_request("s")
    _, output, _, _ = _only_output(core)
    assert (output.finish_reason, output.terminal) == (FinishReason.ERROR, True)
    assert (output.error.code, output.error.retryable) == ("engine_paused", True)


# @spec PSTATE-ADMIT-022 PSTATE-PAGE-004
@pytest.mark.parametrize(
    ("held_kind", "make_input", "finish_reason"),
    [
        (None, lambda: _stream_request("s", 3), FinishReason.ABORT),
        ("parked", lambda: _stream_request("s", 3), FinishReason.ABORT),
        ("parked", _update, FinishReason.ABORT),
        ("parked", lambda: _no_tokens(continuation=True), FinishReason.ABORT),
    ],
    ids=["opener", "opener_for_held_id", "continuation", "inapplicable_continuation"],
)
def test_shutdown_rejects_each_add_with_one_terminal_output(
    monkeypatch, held_kind, make_input, finish_reason
):
    core, free_pages = _live_core(monkeypatch)
    if held_kind is not None:
        _hold_kind(core, held_kind)
    core.shutdown_state = EngineShutdownState.REQUESTED

    _add(core, make_input())

    assert not core.scheduler.has_request("s")
    _assert_ended_after_free(core, free_pages, finish_reason)


# @spec PSTATE-ADMIT-022
def test_inapplicable_opener_for_a_held_id_publishes_one_output(monkeypatch):
    core, free_pages = _live_core(monkeypatch)
    held = _hold_kind(core, "parked")

    _add(core, _no_tokens())

    assert held.status == RequestStatus.FINISHED_ERROR
    _assert_ended_after_free(core, free_pages, FinishReason.ERROR)


# @spec PSTATE-ADMIT-022
@pytest.mark.parametrize("make_input", [_update, _close], ids=["input", "close"])
@pytest.mark.parametrize("held", [None, "single_request"])
def test_shutdown_still_drops_continuations_core_does_not_hold_as_streams(
    monkeypatch, make_input, held
):
    core, _ = _live_core(monkeypatch)
    if held is not None:
        core.scheduler.add_request(_stream_request("s", resumable=False))
    core.shutdown_state = EngineShutdownState.REQUESTED

    _add(core, make_input())

    assert core.scheduler.has_request("s") is (held is not None)
    assert core.output_queue.published == []


def _paused_and_full(core: EngineCoreProc) -> None:
    _admit_full(core)
    _admit_paused(core)
    core.reject_new_requests = True


# @spec PSTATE-ADMIT-022 PSTATE-ADMIT-021
def test_held_id_outranks_an_inapplicable_input_pause_and_capacity(monkeypatch):
    core, free_pages = _live_core(monkeypatch, max_num_seqs=2)
    held = _hold_kind(core, "parked")
    _paused_and_full(core)

    _add(core, _with_structured_outputs())

    assert held.status == RequestStatus.FINISHED_ERROR
    client_index, output, still_held, _ = _only_output(core)
    assert (output.request_id, output.finish_reason, output.terminal) == (
        "s",
        FinishReason.ERROR,
        True,
    )
    assert output.error is None
    assert not still_held


# @spec PSTATE-ADMIT-022 PSTATE-TURN-019
def test_inapplicable_opener_outranks_pause_and_capacity(monkeypatch):
    core, _ = _live_core(monkeypatch, max_num_seqs=1)
    _paused_and_full(core)

    _add(core, _no_tokens())

    _, output, _, _ = _only_output(core)
    assert output.error is not None and output.error.code == "validation_error"
    assert not core.scheduler.has_request("s")


# @spec PSTATE-ADMIT-022
def test_pause_outranks_capacity_for_an_opener(monkeypatch):
    core, _ = _live_core(monkeypatch, max_num_seqs=1)
    _paused_and_full(core)

    _add(core, _stream_request("s", 3))

    _, output, _, _ = _only_output(core)
    assert output.error is not None and output.error.code == "engine_paused"


# @spec PSTATE-ADMIT-022
@pytest.mark.parametrize(
    ("held", "structured", "reason"),
    [
        (True, True, "duplicate_request_id"),
        (False, True, "validation_error"),
        (False, False, "engine_paused"),
    ],
    ids=["held_id_first", "structured_outputs_next", "pause_before_capacity"],
)
def test_admit_stream_returns_the_first_refusal_in_order(
    monkeypatch, held, structured, reason
):
    core, _ = _live_core(monkeypatch, max_num_seqs=2)
    if held:
        _hold_kind(core, "parked")
    _paused_and_full(core)
    admission = _admission()
    if structured:
        admission.sampling_params = STRUCTURED

    assert core.admit_stream(admission) == reason
