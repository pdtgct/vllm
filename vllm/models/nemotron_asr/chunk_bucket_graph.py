# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded experimental capture of one already-gathered CHUNK bucket.

The caller constructs this binding before measured execution and keeps its model
immutable. Exact encoder populations are independent of the padded decoder
tier. The optional serving pilot replaces only declared 160-ms B1/B31/B63 cells.
"""

import hashlib
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from vllm.models.nemotron_asr.advance import (
    ENVELOPE_HEADER_SLOTS,
    ChunkBatch,
    ChunkBucketResult,
    SessionStateBatch,
    advance_chunk_bucket,
)
from vllm.models.nemotron_asr.batch_invariance import execution_mode
from vllm.models.nemotron_asr.decode_graph import GraphRuntime, platform_graph_runtime
from vllm.models.nemotron_asr.encoder_execution import _tensor_signature
from vllm.models.nemotron_asr.manifests import CADENCES
from vllm.models.nemotron_asr.rnnt import decode_dense_masked_frames


def _state_tensors(state: SessionStateBatch) -> tuple[torch.Tensor, ...]:
    return (
        state.raw_tail,
        state.mel_tail,
        state.frontend_counters,
        *state.channel,
        *state.window_valid,
        *state.time,
        state.h,
        state.c,
        state.last_label,
    )


def _clone_state(state: SessionStateBatch) -> SessionStateBatch:
    return SessionStateBatch(
        raw_tail=state.raw_tail.clone(),
        mel_tail=state.mel_tail.clone(),
        frontend_counters=state.frontend_counters.clone(),
        channel=[tensor.clone() for tensor in state.channel],
        window_valid=[tensor.clone() for tensor in state.window_valid],
        time=[tensor.clone() for tensor in state.time],
        h=state.h.clone(),
        c=state.c.clone(),
        last_label=state.last_label.clone(),
    )


def _outputs(bucket: ChunkBucketResult) -> tuple[torch.Tensor, ...]:
    result, batch = bucket.result, bucket.batch
    values = (
        result.token_ids,
        result.token_lengths,
        result.row_status,
        result.frame_emission_counts,
        result.frame_final_labels,
        result.frame_valid_lengths,
        batch.valid_samples,
        batch.geometry_id,
        batch.final_tail,
        batch.prompt_index,
        batch.chunk_sequence,
        bucket.counter_invariant_bad,
        bucket.counter_delta_bad,
    )
    if (
        any(not isinstance(tensor, torch.Tensor) for tensor in values)
        or result.captures is not None
    ):
        raise ValueError(
            "bucket graph requires all dense frame outputs and capture-off execution"
        )
    return values  # type: ignore[return-value]


@torch.inference_mode()
def capture_chunk_bucket(
    core: Any,
    env: torch.Tensor,
    state: SessionStateBatch,
    *,
    geometry: int,
    admitted_prompt: torch.Tensor,
    incoming_status: torch.Tensor,
    queue_capacity: int,
    vllm_config: Any,
    runtime: GraphRuntime | None = None,
    capture_decode_fn: Any = None,
    admitted_decode_fn: Any = decode_dense_masked_frames,
    admitted_encoder_transition: Any = None,
    decoder_tier: int | None = None,
) -> Callable[..., ChunkBucketResult]:
    """Capture a single explicit cell, preserving caller state and output lifetime.

    B1/B31/B63 retain exact frontend/encoder rows and use the existing split
    decoder's B1/B32/B64 staging and arithmetic. Unsupported shapes fail before staging.
    Fresh-aware resident gathers and every transaction commit remain outside.
    """
    population = int(env.shape[0])
    if geometry != 1 or population not in (1, 31, 32, 63, 64):
        raise ValueError(
            "experimental bucket graph supports only 160-ms B1/B31/B32/B63/B64"
        )
    expected_tier = 1 if population == 1 else (32 if population <= 32 else 64)
    if decoder_tier is not None and decoder_tier != expected_tier:
        raise ValueError("CHUNK decoder tier differs from the existing split decoder")
    if population in (1, 31, 63) and (
        capture_decode_fn is None or decoder_tier is None
    ):
        raise ValueError("selected CHUNK population requires the sealed split decoder")
    execute_decode = (
        decode_dense_masked_frames if capture_decode_fn is None else capture_decode_fn
    )
    if getattr(core.encoder, "_stream_relative_position_lengths", ()):
        raise ValueError(
            "bucket graph requires unprepared native positional projections"
        )
    runtime = runtime or platform_graph_runtime()
    sample_width = (
        8 * (list(CADENCES.values())[geometry][1] + 1) * int(core.featurizer.hop_length)
    )
    static_env = env.clone()
    static_prompt = admitted_prompt.clone()
    static_status = incoming_status.clone()
    static_state = _clone_state(state)
    sources = (env, admitted_prompt, incoming_status, *_state_tensors(state))
    scratch = (static_env, static_prompt, static_status, *_state_tensors(static_state))
    signature = tuple(_tensor_signature(tensor) for tensor in sources)
    descriptor = runtime.descriptor_factory(population)
    result_type: Any = None
    owned: tuple[torch.Tensor, ...] = ()

    def stage(values: tuple[torch.Tensor, ...]) -> None:
        for destination, source in zip(scratch, values, strict=True):
            destination.copy_(source)

    def body() -> tuple[torch.Tensor, ...]:
        bucket = advance_chunk_bucket(
            core,
            static_env,
            static_state,
            geometry=geometry,
            admitted_prompt=static_prompt,
            incoming_status=static_status,
            queue_capacity=queue_capacity,
            decode_fn=execute_decode,
            capture=False,
        )
        for destination, source in zip(owned, _outputs(bucket), strict=True):
            destination.copy_(source)
        return owned

    # Discover layouts outside capture. Strong output buffers and state scratch
    # outlive the platform wrapper's weak output handles.
    probe = advance_chunk_bucket(
        core,
        static_env,
        static_state,
        geometry=geometry,
        admitted_prompt=static_prompt,
        incoming_status=static_status,
        queue_capacity=queue_capacity,
        decode_fn=execute_decode,
        capture=False,
    )
    result_type = type(probe.result)
    owned = tuple(torch.empty_like(tensor) for tensor in _outputs(probe))
    wrapper = runtime.wrapper_factory(
        body, vllm_config, runtime_mode=runtime.graph_mode
    )

    def call(mode: Any) -> tuple[torch.Tensor, ...]:
        with runtime.forward_context(
            None, vllm_config, cudagraph_runtime_mode=mode, batch_descriptor=descriptor
        ):
            return wrapper()

    for _ in range(3):
        stage(sources)
        call(runtime.eager_mode)
    stage(sources)
    expected_outputs = tuple(tensor.clone() for tensor in call(runtime.eager_mode))
    expected_state = tuple(tensor.clone() for tensor in _state_tensors(static_state))
    runtime.synchronize(env.device)
    runtime.set_capture_enabled(True)
    try:
        with runtime.capture_context(env.device):
            stage(sources)
            call(runtime.graph_mode)
        stage(sources)
        actual = call(runtime.graph_mode)
        for actual_tensor, expected in zip(
            (*actual, *_state_tensors(static_state)),
            (*expected_outputs, *expected_state),
            strict=True,
        ):
            torch.testing.assert_close(actual_tensor, expected, atol=0, rtol=0)
        runtime.synchronize(env.device)
    finally:
        runtime.set_capture_enabled(False)

    @torch.inference_mode()
    def transition(
        actual_core: Any,
        actual_env: torch.Tensor,
        actual_state: SessionStateBatch,
        *,
        geometry: int,
        admitted_prompt: torch.Tensor,
        incoming_status: torch.Tensor,
        queue_capacity: int,
        decode_fn: Any,
        encoder_transition: Any = None,
        capture: bool = False,
        capture_geometry: Any | None = None,
    ) -> ChunkBucketResult:
        values = (
            actual_env,
            admitted_prompt,
            incoming_status,
            *_state_tensors(actual_state),
        )
        if (
            actual_core is not core
            or geometry != 1
            or queue_capacity != frozen_capacity
            or decode_fn is not admitted_decode_fn
            or encoder_transition is not admitted_encoder_transition
            or capture is not False
            or capture_geometry is not None
            or tuple(_tensor_signature(tensor) for tensor in values) != signature
        ):
            raise ValueError("bucket graph invocation differs from its captured cell")
        scratch_storages = {tensor.untyped_storage().data_ptr() for tensor in scratch}
        if any(
            tensor.untyped_storage().data_ptr() in scratch_storages for tensor in values
        ):
            raise ValueError("bucket caller aliases graph-owned scratch")
        stage(values)
        output = call(runtime.graph_mode)
        for destination, source in zip(
            _state_tensors(actual_state), _state_tensors(static_state), strict=True
        ):
            destination.copy_(source)
        escaped = tuple(tensor.clone() for tensor in output)
        result = result_type(
            token_ids=escaped[0],
            token_lengths=escaped[1],
            row_status=escaped[2],
            frame_emission_counts=escaped[3],
            frame_final_labels=escaped[4],
            frame_valid_lengths=escaped[5],
        )
        # Samples retain the caller's input lifetime, as in the eager bucket;
        # every graph-produced result/control tensor is independently owned.
        batch = ChunkBatch(
            samples=actual_env[
                :, ENVELOPE_HEADER_SLOTS : ENVELOPE_HEADER_SLOTS + sample_width
            ],
            valid_samples=escaped[6],
            geometry_id=escaped[7],
            final_tail=escaped[8],
            prompt_index=escaped[9],
            chunk_sequence=escaped[10],
        )
        transition.replay_count += 1
        return ChunkBucketResult(batch, result, escaped[11], escaped[12])

    frozen_capacity = queue_capacity
    transition.replay_count = 0
    return transition


class ExactChunkGraphBinding:
    """Opt-in serving pilot replacing the native B1/B31/B63 encoder captures.

    The encoder execution inventory owns the graph callables. Decoder staging
    borrows the already-sealed tier workspace; all calls use the worker's serial
    execution stream. Fresh gathers and resident commits stay with advance.py.
    """

    def __init__(
        self,
        core: Any,
        config: Any,
        encoder_execution: Any,
        decoder_binding: Any,
        populations: Any,
    ):
        if (
            not isinstance(populations, (tuple, list))
            or not populations
            or any(type(n) is not int or n not in (1, 31, 63) for n in populations)
            or len(set(populations)) != len(populations)
            or decoder_binding is None
        ):
            raise ValueError(
                "CHUNK serving pilot requires distinct populations 1/31/63 and a dense "
                "graph decoder"
            )
        if any(
            decoder_binding.execution_tier(n)
            != (1 if n == 1 else (32 if n == 31 else 64))
            for n in populations
        ):
            raise ValueError(
                "CHUNK pilot requires unchanged physical decoder tiers 1/32/64"
            )
        self._core = core
        self._config = config
        self._encoder = encoder_execution
        self._decoder = decoder_binding
        self._cells = frozenset((1, n) for n in populations)
        self._fallback_counts: dict[tuple[int, int], int] = {}
        self._instance_id = f"{os.getpid()}:{time.monotonic_ns()}"
        self._source_sha256 = {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "advance.py",
                "encoder.py",
                "encoder_execution.py",
                "decode_graph.py",
                "rnnt.py",
                "chunk_bucket_graph.py",
                "nemotron_asr.py",
                "profile_execution.py",
            )
        }
        encoder_execution.reserve_chunk_graph_cells(self._cells)

    @property
    def ready(self) -> bool:
        # @spec PORT-PREC-025, PORT-PREC-026: require the mode-bound inventory.
        return self._encoder.ready and set(self._encoder._chunk_graph_entries) == {
            execution_mode(self._core).graph_key("chunk", cell) for cell in self._cells
        }

    @torch.inference_mode()
    def warmup(self, device: torch.device) -> None:
        """Allocate only replacement scratch, then publish all cells together."""
        from vllm.models.nemotron_asr.manifests import SESSION_LIMITS
        from vllm.models.nemotron_asr.profile_execution import build_profile_invocation

        if self.ready:
            return
        if not self._encoder._sealed or self._encoder._chunk_graph_entries:
            raise ValueError(
                "CHUNK capture requires a sealed, unpublished replacement domain"
            )
        pending: dict[tuple[int, int], Any] = {}
        try:
            for geometry, population in sorted(self._cells):
                invocation = build_profile_invocation(
                    self._config,
                    num_rows=population,
                    device=device,
                    geometry_id=geometry,
                )
                pools = invocation.pools
                # Fresh profile rows deliberately do not read page contents.
                # The final capture owns one cloned cache, replacing the cache
                # the skipped standalone encoder entry would otherwise retain.
                # Singleton slices retain packed page strides under preserve_format;
                # serving gathers use canonical contiguous scratch at every size.
                state = SessionStateBatch(
                    raw_tail=torch.zeros_like(
                        pools.frontend_raw[1:], memory_format=torch.contiguous_format
                    ),
                    mel_tail=torch.zeros_like(
                        pools.frontend_mel[1:], memory_format=torch.contiguous_format
                    ),
                    frontend_counters=torch.zeros_like(
                        pools.frontend_counters[1:],
                        memory_format=torch.contiguous_format,
                    ),
                    channel=[
                        torch.zeros_like(t[1:], memory_format=torch.contiguous_format)
                        for t in pools.channel
                    ],
                    window_valid=[
                        torch.zeros_like(t[1:], memory_format=torch.contiguous_format)
                        for t in pools.valid_length
                    ],
                    time=[
                        torch.zeros_like(t[1:], memory_format=torch.contiguous_format)
                        for t in pools.convolution
                    ],
                    h=torch.zeros_like(
                        pools.predictor_h[1:], memory_format=torch.contiguous_format
                    ),
                    c=torch.zeros_like(
                        pools.predictor_c[1:], memory_format=torch.contiguous_format
                    ),
                    last_label=torch.full(
                        (population,),
                        int(self._core.blank_id),
                        device=device,
                        dtype=torch.long,
                    ),
                )
                tier = self._decoder.execution_tier(population)
                pending[(geometry, population)] = capture_chunk_bucket(
                    self._core,
                    invocation.inputs_embeds,
                    state,
                    geometry=geometry,
                    admitted_prompt=torch.zeros(
                        population, dtype=torch.long, device=device
                    ),
                    incoming_status=torch.zeros(
                        population, dtype=torch.int32, device=device
                    ),
                    queue_capacity=int(SESSION_LIMITS["queue_capacity"]),
                    vllm_config=self._encoder._vllm_config,
                    runtime=self._encoder._graph_runtime,
                    capture_decode_fn=self._decoder.uncaptured_decode_fn(
                        geometry=geometry, tier=tier
                    ),
                    admitted_decode_fn=self._decoder.decode_fn(
                        geometry=geometry, tier=tier
                    ),
                    admitted_encoder_transition=self._encoder.transition,
                    decoder_tier=tier,
                )
                self._encoder._record_memory_diagnostic(
                    device, stage="after-chunk-capture", key=(geometry, population)
                )
                del state, pools, invocation
            self._encoder.publish_chunk_graphs(pending)
            self._encoder._record_memory_diagnostic(
                device, stage="after-complete-mixed-inventory"
            )
        except Exception:
            self._encoder._discard()
            raise

    def resolve(
        self,
        *,
        geometry: int,
        population: int,
        decode_fn: Any,
        encoder_transition: Any,
        capture: bool,
    ) -> Callable[..., ChunkBucketResult]:
        """Resolve from host row authority before any resident state is read."""
        if not self.ready:
            raise ValueError("CHUNK graph inventory is incomplete")
        cell = (geometry, population)
        if cell not in self._cells:
            return self._fallback
        tier = self._decoder.execution_tier(population)
        if (
            capture
            or encoder_transition is not self._encoder.transition
            or decode_fn is not self._decoder.decode_fn(geometry=geometry, tier=tier)
        ):
            raise ValueError("CHUNK graph cell differs from its sealed serving binding")
        return self._encoder._chunk_graph_entries[
            execution_mode(self._core).graph_key("chunk", cell)
        ]

    def _fallback(
        self, core: Any, env: torch.Tensor, state: SessionStateBatch, **kwargs: Any
    ) -> ChunkBucketResult:
        result = advance_chunk_bucket(core, env, state, **kwargs)
        cell = (int(kwargs["geometry"]), int(env.shape[0]))
        self._fallback_counts[cell] = self._fallback_counts.get(cell, 0) + 1
        return result

    def receipt(self) -> dict[str, Any]:
        """Read host-only counters; never synchronize or reset measured state."""
        return execution_mode(self._core).readiness_receipt(
            {
                "ready": self.ready,
                "worker_pid": os.getpid(),
                "instance_id": self._instance_id,
                "source_sha256": dict(self._source_sha256),
                "cells": [
                    {
                        "geometry": g,
                        "encoder_population": n,
                        "decoder_tier": self._decoder.execution_tier(n),
                        "successful_replays": self._encoder._chunk_graph_entries[
                            execution_mode(self._core).graph_key("chunk", (g, n))
                        ].replay_count,
                    }
                    for g, n in sorted(self._cells)
                    if execution_mode(self._core).graph_key("chunk", (g, n))
                    in self._encoder._chunk_graph_entries
                ],
                "fallbacks": [
                    {"geometry": g, "encoder_population": n, "successful_calls": count}
                    for (g, n), count in sorted(self._fallback_counts.items())
                ],
            }
        )
