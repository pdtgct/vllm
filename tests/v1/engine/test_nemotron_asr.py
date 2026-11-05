# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-backed, in-process CUDA smoke for core persistent ASR state.

Feed the existing seven-field envelope at EngineCore's processed-MM boundary.
Startup initializes the tokenizer for multimodal profiling. Requests bypass text
tokenization, realtime transport, the segmenter, and the transcript decoder.
"""

import time

import pytest
import torch

from tests.utils import create_new_process_for_each_test


def _engine_args(model="nvidia/nemotron-3.5-asr-streaming-0.6b"):
    from vllm.engine.arg_utils import EngineArgs

    return EngineArgs(
        model=model,
        dtype="float32",
        enforce_eager=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        enable_chunked_prefill=False,
        enable_cumem_allocator=False,
        skip_tokenizer_init=False,
        max_model_len=256,
        max_num_seqs=2,
        max_num_batched_tokens=256,
        num_gpu_blocks_override=3,  # Null block plus exactly two live slots.
        gpu_memory_utilization=0.8,
        distributed_executor_backend="uni",
        additional_config={
            "stream_admission": {
                "never_fed_timeout": 600.0,
                "parked_idle_timeout": 600.0,
                "draining_timeout": 600.0,
            }
        },
    )


pytestmark = [
    pytest.mark.nemotron_gpu,
    pytest.mark.slow_test,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@create_new_process_for_each_test(method="spawn")
def test_nemotron_engine_stream_lifecycle(monkeypatch):
    from transformers import BatchFeature

    from vllm import SamplingParams
    from vllm.models.nemotron_asr.manifests import (
        ADMISSION_EPOCH_MODULUS_MS,
        CADENCES,
        ENVELOPE_HEADER_FIELDS,
        RAW_SAMPLES_PER_CHUNK,
    )
    from vllm.multimodal.inputs import (
        MultiModalFeatureSpec,
        MultiModalFieldConfig,
        MultiModalKwargsItems,
        PlaceholderRange,
    )
    from vllm.v1.engine import (
        EngineCoreRequest,
        StreamAdmissionContext,
        StreamAdmissionId,
    )
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.uniproc_executor import UniProcExecutor
    from vllm.v1.kv_cache_interface import PersistentStateSpec
    from vllm.v1.request import RequestStatus
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    config = _engine_args().create_engine_config()
    assert config.use_v2_model_runner
    core = EngineCore(config, UniProcExecutor, log_stats=False)
    try:
        runner = core.model_executor.driver_worker.worker.model_runner
        assert isinstance(runner, GPUModelRunner)
        assert runner.state_slots is runner.model_state.state_slots
        assert len(runner.state_slots.views) == 1
        view = next(iter(runner.state_slots.views.values()))
        assert isinstance(view.spec, PersistentStateSpec)
        assert view.capacity == 3 and view._tensor.is_cuda
        # Constructor includes determine_available_memory, state-only KV planning,
        # initialize_from_config, and compile_or_warm_up_model (no test stubs).
        assert core.available_gpu_memory_for_kv_cache > 0
        scheduler = core.scheduler
        manager = next(
            iter(scheduler.kv_cache_manager.coordinator.state_managers.values())
        )
        assert manager.num_free_slots == 2
        hf = config.model_config.hf_config
        owner = "cuda-smoke"
        epoch = core.get_stream_admission_epoch(owner)

        def request(request_id):
            params = SamplingParams(temperature=0, max_tokens=142, detokenize=False)
            params.update_from_generation_config({}, hf.eos_token_id)
            return EngineCoreRequest(
                request_id=request_id,
                prompt_token_ids=[],
                mm_features=None,
                sampling_params=params,
                pooling_params=None,
                arrival_time=time.time(),
                lora_request=None,
                cache_salt=None,
                data_parallel_rank=None,
                resumable=True,
            )

        def admit(request_id, sequence):
            return core.admit_stream(
                owner, StreamAdmissionId(epoch, sequence), request(request_id)
            )

        assert admit("audio", 0).accepted
        assert admit("peer", 1).accepted
        original = manager.get("audio")
        peer = manager.get("peer")
        refused = admit("overflow", 2)
        assert not refused.accepted and refused.reason == "capacity"
        assert set(scheduler.requests) == {"audio", "peer"}

        def tick():
            outputs, executed = core.step()
            core.post_step(executed)
            return [item for batch in outputs.values() for item in batch.outputs]

        all_tokens: list[int] = []
        # 560 ms cadence: one regular chunk, then a short final tail.
        geometry = tuple(CADENCES).index("560ms")
        for sequence, final in enumerate((False, True)):
            size = 1600 if final else RAW_SAMPLES_PER_CHUNK["560ms"]
            samples = 0.02 * torch.sin(
                torch.arange(size) * (2 * torch.pi * 440 / 16000)
            )
            samples[: size // 4] = 0
            stamp = (time.time_ns() // 1_000_000) % ADMISSION_EPOCH_MODULUS_MS
            envelope = torch.cat(
                (
                    torch.tensor(
                        [2, size, geometry, int(final), 0, sequence, stamp],
                        dtype=torch.float32,
                    ),
                    samples,
                )
            )
            assert envelope.numel() == len(ENVELOPE_HEADER_FIELDS) + size
            items = MultiModalKwargsItems.from_hf_inputs(
                BatchFeature({"audio": [envelope]}),
                {"audio": MultiModalFieldConfig.batched("audio")},
            )
            payload = request("audio")
            payload.prompt_token_ids = [hf.audio_chunk_token_id]
            payload.mm_features = [
                MultiModalFeatureSpec(
                    data=items["audio"][0],
                    modality="audio",
                    identifier=f"smoke-{sequence}",
                    mm_position=PlaceholderRange(0, 1),
                )
            ]
            payload.stream_admission = StreamAdmissionContext(
                owner, StreamAdmissionId(epoch, 0), final
            )
            core.add_request(*core.preprocess_add_request(payload))
            received = []
            for _ in range(144):
                received.extend(tick())
                assert "audio" in scheduler.requests, (
                    "stream terminated instead of parking"
                )
                if (
                    scheduler.requests["audio"].status
                    == RequestStatus.WAITING_FOR_STREAMING_REQ
                ):
                    break
            else:
                pytest.fail("audio chunk did not park within its emission bound")
            assert received, "no EngineCore output for the audio chunk"
            assert all(item.request_id == "audio" for item in received)
            assert all(
                item.finish_reason is None or str(item.finish_reason) == "stop"
                for item in received
            )
            chunk_tokens = [token for item in received for token in item.new_token_ids]
            assert chunk_tokens and chunk_tokens[-1] == hf.eos_token_id
            all_tokens.extend(chunk_tokens)
            assert manager.get("audio") == original
            assert manager.get("peer") == peer
            assert manager.num_free_slots == 0

        # Silence/tone may produce no lexical labels; control tokens count as output.
        assert all_tokens and all(isinstance(token, int) for token in all_tokens)
        terminal = request("audio")
        terminal.resumable = False
        terminal.stream_admission = StreamAdmissionContext(
            owner, StreamAdmissionId(epoch, 0), True
        )
        core.add_request(*core.preprocess_add_request(terminal))
        assert not admit("replacement", 3).accepted
        for _ in range(4):
            tick()
            if manager.num_free_slots == 1:
                break
        assert manager.num_free_slots == 1
        assert "audio" not in scheduler.requests
        assert admit("replacement", 4).accepted
        replacement = manager.get("replacement")
        assert replacement.slot == original.slot
        assert replacement.generation > original.generation
        assert manager.get("peer") == peer
        core.abort_requests(["peer", "replacement"])
        for _ in range(4):
            tick()
            if manager.num_free_slots == 2:
                break
        assert manager.num_free_slots == 2
        assert not scheduler.requests
    finally:
        core.shutdown()


#: The g0 serving base both arms share; the replay arm adds only the option.
_G0_BASE_OVERRIDES = {
    "decode_dispatch_arm": "dense-graphed",
    "encoder_execution_arm": "eager-graphed",
    "experimental_native_burst": False,
    "supported_num_lookahead_tokens": [0],
}


def _run_g0_stream(hf_overrides, num_chunks):
    """Feed one fresh g0 session ``num_chunks`` CHUNKs, each through its park.

    Returns per-chunk output tokens, the resident state storage after each
    park, and the model's transaction graph counters.
    """
    import gc
    from dataclasses import replace

    from transformers import BatchFeature

    from vllm import SamplingParams
    from vllm.models.nemotron_asr.manifests import RAW_SAMPLES_PER_CHUNK
    from vllm.models.nemotron_asr.transaction_graph import (
        transaction_graph_counters,
    )
    from vllm.multimodal.inputs import (
        MultiModalFeatureSpec,
        MultiModalFieldConfig,
        MultiModalKwargsItems,
        PlaceholderRange,
    )
    from vllm.v1.engine import (
        EngineCoreRequest,
        StreamAdmissionContext,
        StreamAdmissionId,
    )
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.uniproc_executor import UniProcExecutor
    from vllm.v1.request import RequestStatus

    args = replace(
        _engine_args(),
        max_num_seqs=1,
        num_gpu_blocks_override=2,
        gpu_memory_utilization=0.4,
        hf_overrides=hf_overrides,
    )
    config = args.create_engine_config()
    core = EngineCore(config, UniProcExecutor, log_stats=False)
    try:
        runner = core.model_executor.driver_worker.worker.model_runner
        view = next(iter(runner.state_slots.views.values()))
        hf = config.model_config.hf_config
        owner = "transaction-graph"
        epoch = core.get_stream_admission_epoch(owner)

        def request():
            params = SamplingParams(temperature=0, max_tokens=142, detokenize=False)
            params.update_from_generation_config({}, hf.eos_token_id)
            return EngineCoreRequest(
                request_id="audio",
                prompt_token_ids=[],
                mm_features=None,
                sampling_params=params,
                pooling_params=None,
                arrival_time=time.time(),
                lora_request=None,
                cache_salt=None,
                data_parallel_rank=None,
                resumable=True,
            )

        assert core.admit_stream(owner, StreamAdmissionId(epoch, 0), request()).accepted
        size = RAW_SAMPLES_PER_CHUNK["80ms"]
        generator = torch.Generator().manual_seed(0)
        tokens, pages = [], []
        for sequence in range(num_chunks):
            if sequence == 0:
                samples = torch.zeros(size)
            else:
                samples = 0.05 * torch.randn(size, generator=generator)
            envelope = torch.cat(
                (
                    torch.tensor([2, size, 0, 0, 0, sequence, 1], dtype=torch.float32),
                    samples,
                )
            )
            items = MultiModalKwargsItems.from_hf_inputs(
                BatchFeature({"audio": [envelope]}),
                {"audio": MultiModalFieldConfig.batched("audio")},
            )
            payload = request()
            payload.prompt_token_ids = [hf.audio_chunk_token_id]
            payload.mm_features = [
                MultiModalFeatureSpec(
                    data=items["audio"][0],
                    modality="audio",
                    identifier=f"g0-{sequence}",
                    mm_position=PlaceholderRange(0, 1),
                )
            ]
            payload.stream_admission = StreamAdmissionContext(
                owner, StreamAdmissionId(epoch, 0), False
            )
            core.add_request(*core.preprocess_add_request(payload))
            received = []
            for _ in range(144):
                outputs, executed = core.step()
                core.post_step(executed)
                received.extend(
                    token
                    for batch in outputs.values()
                    for item in batch.outputs
                    for token in item.new_token_ids
                )
                status = core.scheduler.requests["audio"].status
                if status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    break
            else:
                pytest.fail("g0 chunk did not park within its emission bound")
            assert received and received[-1] == hf.eos_token_id
            tokens.append(received)
            torch.accelerator.synchronize()
            pages.append(view._tensor.cpu().clone())
        counters = transaction_graph_counters(core)
        core.abort_requests(["audio"])
        return tokens, pages, counters
    finally:
        core.shutdown()
        del core
        gc.collect()
        torch.accelerator.empty_cache()


@create_new_process_for_each_test(method="spawn")
def test_nemotron_engine_transaction_graph_matches_eager(monkeypatch):
    """A fresh g0 B1 session replays its whole transactions bitwise exactly.

    The eager arm runs first, then the replay arm adds only the transaction
    graph option. Tokens and the resident storage after every park must match,
    and every transaction must replay a captured cell.
    """
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    num_chunks = 12
    eager_tokens, eager_pages, eager_counters = _run_g0_stream(
        dict(_G0_BASE_OVERRIDES), num_chunks
    )
    assert eager_counters == {}
    tokens, pages, counters = _run_g0_stream(
        {**_G0_BASE_OVERRIDES, "transaction_graph_populations": [1]}, num_chunks
    )
    assert tokens == eager_tokens
    for actual, expected in zip(pages, eager_pages, strict=True):
        assert torch.equal(actual, expected)
    assert counters["captured_cells"] == 3
    assert counters["fallbacks"] == 0
    assert counters["replays:chunk-fresh:g0:b1"] == 1
    assert counters["replays:chunk-continuing:g0:b1"] == num_chunks - 1
    echo_steps = sum(len(chunk) for chunk in tokens) - num_chunks
    assert counters["replays:replay:b1"] >= echo_steps
