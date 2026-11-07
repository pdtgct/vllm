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


def _run_g0_stream(hf_overrides, num_chunks, sessions=1, inject=None):
    """Feed ``sessions`` fresh g0 sessions ``num_chunks`` CHUNKs in lockstep.

    Every session's CHUNK is queued before the step that carries them, and the
    next CHUNKs wait until every session has parked, so each CHUNK step holds
    the whole population. Returns per-chunk output tokens per session, the
    resident state storage after each park, the model's transaction graph
    counters and the number of executed steps after each CHUNK step.
    ``inject`` receives the model after admission, before the first CHUNK.
    """
    import gc
    from dataclasses import replace

    from transformers import BatchFeature

    from vllm import SamplingParams
    from vllm.models.nemotron_asr.manifests import RAW_SAMPLES_PER_CHUNK
    from vllm.models.nemotron_asr.transaction_graph import (
        _resolve_model,
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
        max_num_seqs=sessions,
        num_gpu_blocks_override=sessions + 1,
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
        names = ["audio"] if sessions == 1 else [f"audio-{i}" for i in range(sessions)]

        def request(name):
            params = SamplingParams(temperature=0, max_tokens=142, detokenize=False)
            params.update_from_generation_config({}, hf.eos_token_id)
            return EngineCoreRequest(
                request_id=name,
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

        for index, name in enumerate(names):
            assert core.admit_stream(
                owner, StreamAdmissionId(epoch, index), request(name)
            ).accepted
        if inject is not None:
            inject(_resolve_model(core))
        size = RAW_SAMPLES_PER_CHUNK["80ms"]
        generators = [torch.Generator().manual_seed(i) for i in range(sessions)]
        tokens, pages, echo_steps = [], [], []
        for sequence in range(num_chunks):
            for index, name in enumerate(names):
                if sequence == 0:
                    samples = torch.zeros(size)
                else:
                    samples = 0.05 * torch.randn(size, generator=generators[index])
                envelope = torch.cat(
                    (
                        torch.tensor(
                            [2, size, 0, 0, 0, sequence, 1], dtype=torch.float32
                        ),
                        samples,
                    )
                )
                items = MultiModalKwargsItems.from_hf_inputs(
                    BatchFeature({"audio": [envelope]}),
                    {"audio": MultiModalFieldConfig.batched("audio")},
                )
                payload = request(name)
                payload.prompt_token_ids = [hf.audio_chunk_token_id]
                payload.mm_features = [
                    MultiModalFeatureSpec(
                        data=items["audio"][0],
                        modality="audio",
                        identifier=f"g0-{name}-{sequence}",
                        mm_position=PlaceholderRange(0, 1),
                    )
                ]
                payload.stream_admission = StreamAdmissionContext(
                    owner, StreamAdmissionId(epoch, index), False
                )
                core.add_request(*core.preprocess_add_request(payload))
            received = {name: [] for name in names}
            executed_steps = 0
            for _ in range(144):
                outputs, executed = core.step()
                core.post_step(executed)
                executed_steps += int(bool(executed))
                for batch in outputs.values():
                    for item in batch.outputs:
                        received[item.request_id].extend(item.new_token_ids)
                if all(
                    core.scheduler.requests[name].status
                    == RequestStatus.WAITING_FOR_STREAMING_REQ
                    for name in names
                ):
                    break
            else:
                pytest.fail("g0 chunk did not park within its emission bound")
            for name in names:
                assert received[name] and received[name][-1] == hf.eos_token_id
            tokens.append([received[name] for name in names])
            echo_steps.append(executed_steps - 1)
            torch.accelerator.synchronize()
            pages.append(view._tensor.cpu().clone())
        counters = transaction_graph_counters(core)
        core.abort_requests(names)
        return tokens, pages, counters, echo_steps
    finally:
        core.shutdown()
        del core
        gc.collect()
        torch.accelerator.empty_cache()


@create_new_process_for_each_test(method="spawn")
def test_nemotron_engine_transaction_graph_matches_eager(monkeypatch):
    """A fresh g0 B1 session replays its whole transactions bitwise exactly.

    The eager arm runs first, then the replay arm adds only the transaction
    graph option. Tokens, echo counts and the resident storage after every
    park must match, and every transaction must replay a captured cell.
    """
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    num_chunks = 12
    eager_tokens, eager_pages, eager_counters, eager_echoes = _run_g0_stream(
        dict(_G0_BASE_OVERRIDES), num_chunks
    )
    assert eager_counters == {}
    tokens, pages, counters, echoes = _run_g0_stream(
        {**_G0_BASE_OVERRIDES, "transaction_graph_populations": [1]}, num_chunks
    )
    assert tokens == eager_tokens
    assert echoes == eager_echoes
    for actual, expected in zip(pages, eager_pages, strict=True):
        assert torch.equal(actual, expected)
    assert counters["captured_cells"] == 3
    assert counters["fallbacks"] == 0
    assert counters["replays:chunk-fresh:g0:b1"] == 1
    assert counters["replays:chunk-continuing:g0:b1"] == num_chunks - 1
    echo_steps = sum(len(chunk[0]) for chunk in tokens) - num_chunks
    assert counters["replays:replay:b1"] >= echo_steps


@create_new_process_for_each_test(method="spawn")
def test_nemotron_engine_transaction_graph_b31_lockstep_matches_eager(monkeypatch):
    """31 lockstep g0 sessions replay every CHUNK step bitwise exactly.

    Every CHUNK step holds all 31 rows (fresh once, then continuing) and must
    replay the B31 cell. Echo steps replay ``replay:b31`` while every row still
    echoes; once rows park, the shrunken population is undeclared and runs the
    eager path, counted. Tokens and the resident storage after every park
    match the eager arm.
    """
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    num_chunks, sessions = 6, 31
    eager_tokens, eager_pages, eager_counters, eager_echoes = _run_g0_stream(
        dict(_G0_BASE_OVERRIDES), num_chunks, sessions
    )
    assert eager_counters == {}
    tokens, pages, counters, echoes = _run_g0_stream(
        {**_G0_BASE_OVERRIDES, "transaction_graph_populations": [sessions]},
        num_chunks,
        sessions,
    )
    assert tokens == eager_tokens
    assert echoes == eager_echoes
    for actual, expected in zip(pages, eager_pages, strict=True):
        assert torch.equal(actual, expected)
    assert counters["captured_cells"] == 3
    assert counters[f"replays:chunk-fresh:g0:b{sessions}"] == 1
    assert counters[f"replays:chunk-continuing:g0:b{sessions}"] == num_chunks - 1
    fallbacks = {
        key: value
        for key, value in counters.items()
        if key.startswith("fallbacks:") and value
    }
    assert all(
        key.startswith("fallbacks:replay:b") and key.endswith(":undeclared")
        for key in fallbacks
    ), fallbacks
    assert counters[f"replays:replay:b{sessions}"] + sum(fallbacks.values()) == sum(
        echoes
    )


def _fail_first_replayed_step(monkeypatch, inject_failure):
    """Run one option-on B1 CHUNK whose replayed step fails as injected.

    Returns the exception ``EngineCore.step`` raised, the committed progress
    the model holds afterwards and its counters.
    """
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    seen = {}

    def inject(model):
        model.consume_committed_progress()
        inject_failure(model)
        seen["model"] = model

    with pytest.raises(RuntimeError, match="injected") as raised:
        _run_g0_stream(
            {**_G0_BASE_OVERRIDES, "transaction_graph_populations": [1]},
            num_chunks=1,
            inject=inject,
        )
    model = seen["model"]
    return raised.value, model.consume_committed_progress(), model


@create_new_process_for_each_test(method="spawn")
def test_nemotron_engine_transaction_graph_replay_failure_propagates_from_step(
    monkeypatch,
):
    """A replay failure leaves ``EngineCore.step`` and publishes no progress."""

    def fail_replay(model):
        def replay():
            raise RuntimeError("injected replay failure")

        for graph in model._transaction_graph._graphs.values():
            monkeypatch.setattr(graph, "replay", replay)

    error, progress, _ = _fail_first_replayed_step(monkeypatch, fail_replay)
    assert str(error) == "injected replay failure"
    assert progress == {}


@create_new_process_for_each_test(method="spawn")
def test_nemotron_engine_transaction_graph_collection_failure_propagates_from_step(
    monkeypatch,
):
    """A commit-sink collection failure after a replay publishes no progress."""
    from vllm.models.nemotron_asr.transaction_graph import (
        transaction_graph_counters,
    )

    def fail_collection(model):
        def collect(*args, **kwargs):
            raise RuntimeError("injected collection failure")

        sink = model._ensure_commit_sink(next(model.parameters()).device)
        monkeypatch.setattr(sink, "collect", collect)

    error, progress, model = _fail_first_replayed_step(monkeypatch, fail_collection)
    assert str(error) == "injected collection failure"
    assert progress == {}
    assert transaction_graph_counters(model)["replays:chunk-fresh:g0:b1"] == 1


def _startup_memory(hf_overrides, gpu_memory_utilization, max_num_seqs=8):
    """Start an engine without a block override and report its memory figures."""
    import gc
    from dataclasses import replace

    from vllm.models.nemotron_asr.transaction_graph import _resolve_model
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.uniproc_executor import UniProcExecutor

    args = replace(
        _engine_args(),
        max_num_seqs=max_num_seqs,
        num_gpu_blocks_override=None,
        gpu_memory_utilization=gpu_memory_utilization,
        hf_overrides=hf_overrides,
    )
    core = EngineCore(args.create_engine_config(), UniProcExecutor, log_stats=False)
    try:
        worker = core.model_executor.driver_worker.worker
        view = next(iter(worker.model_runner.state_slots.views.values()))
        receipt = _resolve_model(core).execution_profile_receipt()
        headroom = worker.peak_activation_memory
        return {
            "total_consumed": worker.total_consumed,
            "headroom": headroom,
            "non_kv": worker.total_consumed + headroom,
            "budget": core.available_gpu_memory_for_kv_cache,
            "page": view.spec.page_size_bytes,
            "count": view.capacity,
            "total": worker.init_snapshot.total_memory,
            "admitted": receipt.get("transaction", {}).get("admission_bytes"),
        }
    finally:
        core.shutdown()
        del core
        gc.collect()
        torch.accelerator.empty_cache()


@create_new_process_for_each_test(method="spawn")
def test_nemotron_engine_transaction_graph_admission_is_charged(monkeypatch):
    """Admitted graph memory is charged before resident capacity is fixed.

    Option-on minus option-off at equal utilization must cover ``A`` in
    ``non_kv_cache_memory`` and in retained consumption, without shrinking the
    transient peak. A third, memory-bound run sized from the option-on figures
    must hold exactly the pages core derives from its reduced budget.
    """
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    # Zero cudagraph estimate: peak_activation_memory is then the peak headroom.
    monkeypatch.setenv("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "0")
    max_num_seqs = 8
    option = {**_G0_BASE_OVERRIDES, "transaction_graph_populations": [1]}
    off = _startup_memory(dict(_G0_BASE_OVERRIDES), 0.4, max_num_seqs)
    on = _startup_memory(option, 0.4, max_num_seqs)
    admitted = on["admitted"]
    assert off["admitted"] is None
    assert admitted is not None and admitted > 0
    assert on["non_kv"] - off["non_kv"] >= admitted
    assert on["total_consumed"] - off["total_consumed"] >= admitted
    assert on["headroom"] >= off["headroom"]

    # Aim the budget at 5.5 pages so memory, not max_num_seqs + 1, binds.
    utilization = (on["non_kv"] + 5.5 * on["page"]) / on["total"]
    bound = _startup_memory(option, utilization, max_num_seqs)
    assert 2 <= bound["count"] < max_num_seqs + 1
    # ``capacity`` counts every allocated page, the null slot 0 included.
    assert bound["count"] == bound["budget"] // bound["page"]
