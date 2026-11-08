# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-resident-state memory profiling for Nemotron streaming execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.models.nemotron_asr.advance import (
    ENV_ADMISSION_MS_MOD,
    ENV_CHUNK_SEQUENCE,
    ENV_FINAL_TAIL,
    ENV_GEOMETRY_ID,
    ENV_PROMPT_INDEX,
    ENV_VALID_SAMPLES,
    ENV_VERSION,
    ENVELOPE_VERSION,
    DecodeRequest,
    DecodeResolver,
    PreparedRowBinding,
    ResolvedDecode,
    RowPlan,
    advance_model_rows,
    consume_batch_stats,
    warmup_advance_model_rows_scatter,
)
from vllm.models.nemotron_asr.manifests import (
    CADENCES,
    ENVELOPE_HEADER_FIELDS,
    RAW_SAMPLES_PER_CHUNK,
)
from vllm.models.nemotron_asr.state_profile import (
    NemotronStatePools,
    StateScratch,
    allocate_state_scratch,
    build_nemotron_persistent_state_spec,
    project_nemotron_state_pools,
)

if TYPE_CHECKING:
    from vllm.models.nemotron_asr.nemotron_asr import NemotronASRForRNNT

logger = init_logger(__name__)


@dataclass(frozen=True)
class PersistentStateProfileInvocation:
    """One maximum-shape transaction backed only by ephemeral storage."""

    geometry_label: str
    geometry_id: int
    num_rows: int
    input_ids: torch.Tensor
    inputs_embeds: torch.Tensor
    plan: RowPlan
    storage: StateScratch
    pools: NemotronStatePools


def _positive_row_count(num_rows: int) -> int:
    if isinstance(num_rows, bool) or not isinstance(num_rows, int):
        raise TypeError("profile row count must be an integer")
    if num_rows <= 0:
        raise ValueError("profile row count must be positive")
    return num_rows


def _required_control(config: Any, name: str) -> int:
    value = getattr(config, name, None)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"profile execution requires integer {name}")
    return value


# @spec PORT-PERF-004
def _profile_decode_resolver(model: NemotronASRForRNNT) -> DecodeResolver:
    """Bind pre-capture dense-graph profiling to dense eager."""
    if getattr(model.config, "decode_dispatch_arm", None) != "dense-graphed":
        return model._decode_resolver

    from vllm.models.nemotron_asr.rnnt import (
        decode_dense_masked_frames,
    )

    def resolve(request: DecodeRequest) -> ResolvedDecode:
        if request.graph_covers_decode:
            raise ValueError("pre-capture memory profile cannot claim graph coverage")
        return ResolvedDecode(
            arm="dense-eager",
            decode_fn=decode_dense_masked_frames,
            override_reason="pre-capture-memory-profile",
        )

    return resolve


# @spec PORT-MIG-005, PORT-STATE-003, PORT-STATE-007
def build_profile_invocation(
    config: Any,
    *,
    num_rows: int,
    device: torch.device,
    geometry_id: int | None = None,
) -> PersistentStateProfileInvocation:
    """Build the maximum live eager transaction without a manager slot."""
    rows = _positive_row_count(num_rows)
    placeholder_id = _required_control(config, "audio_chunk_token_id")
    if geometry_id is None:
        geometry_id = len(CADENCES) - 1
    if (
        isinstance(geometry_id, bool)
        or not isinstance(geometry_id, int)
        or not 0 <= geometry_id < len(CADENCES)
    ):
        raise ValueError("profile geometry id is outside the manifest")
    geometry_label = tuple(CADENCES)[geometry_id]
    valid_samples = RAW_SAMPLES_PER_CHUNK[geometry_label]
    hidden_size = int(config.hidden_size)
    if hidden_size < len(ENVELOPE_HEADER_FIELDS) + valid_samples:
        # The exact carrier guard also runs in the canonical transaction;
        # this message catches malformed profile construction directly.
        required = len(ENVELOPE_HEADER_FIELDS) + valid_samples
        raise ValueError(
            f"profile carrier width {hidden_size} is smaller than {required}"
        )

    spec = build_nemotron_persistent_state_spec(config)
    storage = allocate_state_scratch(
        spec,
        rows + 1,
        device,
    )
    pools = project_nemotron_state_pools(
        storage,
        n_layers=int(config.n_layers),
    )
    block_ids = torch.arange(1, rows + 1, dtype=torch.int64)
    generations = torch.arange(1, rows + 1, dtype=torch.int64)
    deadlines = torch.arange(1, rows + 1, dtype=torch.int64)
    request_ids = tuple(f"profile-{index}" for index in range(rows))
    bindings = tuple(
        PreparedRowBinding(
            request_id=request_id,
            block_id=index + 1,
            admission_generation=index + 1,
            geometry_id=geometry_id,
            prompt_index=0,
            prior_prompt_index=0,
            allow_prompt_transition=False,
            is_chunk=True,
            ready_deadline_ns=index + 1,
        )
        for index, request_id in enumerate(request_ids)
    )
    plan = RowPlan(
        state_indices_d=torch.empty((0, 1), dtype=torch.int64),
        num_decodes=0,
        state_indices_p=block_ids,
        num_prefills=rows,
        has_initial_states_p=torch.zeros(rows, dtype=torch.bool),
        null_block_id=0,
        num_pool_blocks=rows + 1,
        live_block_ids=block_ids.clone(),
        geometry_id=torch.full(
            (rows,),
            geometry_id,
            dtype=torch.int64,
        ),
        prompt_index=torch.zeros(rows, dtype=torch.int64),
        is_chunk=torch.ones(rows, dtype=torch.bool),
        admission_generation=generations,
        ready_deadline_ns=deadlines,
        request_ids=request_ids,
        execution_tier=0,
        bindings=bindings,
        endpoint_mode=torch.zeros(rows, dtype=torch.int64),
        endpoint_threshold_frames=torch.zeros(rows, dtype=torch.int64),
        endpoint_residue_frames=torch.zeros(rows, dtype=torch.int64),
    )
    input_ids = torch.full(
        (rows,),
        placeholder_id,
        dtype=torch.int64,
        device=device,
    )
    inputs_embeds = torch.zeros(
        rows,
        hidden_size,
        dtype=torch.float32,
        device=device,
    )
    inputs_embeds[:, ENV_VERSION] = float(ENVELOPE_VERSION)
    inputs_embeds[:, ENV_VALID_SAMPLES] = float(valid_samples)
    inputs_embeds[:, ENV_GEOMETRY_ID] = float(geometry_id)
    inputs_embeds[:, ENV_FINAL_TAIL] = 0.0
    inputs_embeds[:, ENV_PROMPT_INDEX] = 0.0
    inputs_embeds[:, ENV_CHUNK_SEQUENCE] = 0.0
    inputs_embeds[:, ENV_ADMISSION_MS_MOD] = 1.0
    return PersistentStateProfileInvocation(
        geometry_label=geometry_label,
        geometry_id=geometry_id,
        num_rows=rows,
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        plan=plan,
        storage=storage,
        pools=pools,
    )


# @spec PORT-ADV-001, PORT-MIG-003, PORT-MIG-005, PORT-PERF-004
def run_persistent_state_profile(
    model: NemotronASRForRNNT,
    *,
    num_rows: int,
    device: torch.device,
    geometry_id: int | None = None,
    ready_domain: bool = False,
) -> torch.Tensor:
    """Execute the canonical transition and discard all profile effects."""
    logger.info(
        "Starting Nemotron persistent-state profile execution with %d rows",
        num_rows,
    )
    try:
        execution = model._encoder_execution
        if geometry_id is None and getattr(execution, "arm", None) in {
            "compiled-static",
            "dense-graphed",
            "eager-graphed",
        }:
            geometry_id = max(execution.warmup_geometries)
        invocation_kwargs: dict[str, Any] = {}
        if geometry_id is not None:
            invocation_kwargs["geometry_id"] = geometry_id
        invocation = build_profile_invocation(
            model.config,
            num_rows=num_rows,
            device=device,
            **invocation_kwargs,
        )
        pools = invocation.pools
        native_handoff = getattr(model, "_native_burst_handoff", None)
        if device.type == "cuda":
            warmup_advance_model_rows_scatter(
                channel_pools=list(pools.channel),
                time_pools=list(pools.convolution),
                len_pools=list(pools.valid_length),
                h_pool=pools.predictor_h,
                c_pool=pools.predictor_c,
                queue_pool=pools.replay_queue,
                book_pool=pools.replay_book,
                frontend_raw_pool=pools.frontend_raw,
                frontend_mel_pool=pools.frontend_mel,
                frontend_counter_pool=pools.frontend_counters,
                endpoint_history_pool=pools.endpoint_history,
                endpoint_book_pool=pools.endpoint_book,
            )

        def invoke() -> torch.Tensor:
            native_profile_sink = None
            if native_handoff is not None:
                from vllm.models.nemotron_asr.native_burst import (
                    NativeBurstProfileSink,
                )

                native_profile_sink = NativeBurstProfileSink(
                    max_tokens=native_handoff.max_tokens
                )
            chunk_binding = (
                getattr(model, "_chunk_bucket_binding", None) if ready_domain else None
            )
            profile_plan = invocation.plan
            if chunk_binding is not None:
                assert model._decode_graph_binding is not None
                profile_plan = replace(
                    profile_plan,
                    execution_tier=model._decode_graph_binding.execution_tier(num_rows),
                )
            return advance_model_rows(
                model.core,
                invocation.input_ids,
                invocation.inputs_embeds,
                profile_plan,
                channel_pools=list(pools.channel),
                time_pools=list(pools.convolution),
                len_pools=list(pools.valid_length),
                h_pool=pools.predictor_h,
                c_pool=pools.predictor_c,
                queue_pool=pools.replay_queue,
                book_pool=pools.replay_book,
                frontend_raw_pool=pools.frontend_raw,
                frontend_mel_pool=pools.frontend_mel,
                frontend_counter_pool=pools.frontend_counters,
                endpoint_history_pool=pools.endpoint_history,
                endpoint_book_pool=pools.endpoint_book,
                eou_token_id=_required_control(model.config, "eou_token_id"),
                adapter=model._emission_adapter,
                decode_resolver=model._decode_resolver
                if chunk_binding is not None
                else _profile_decode_resolver(model),
                encoder_transition=model._encoder_execution.transition,
                bucket_transition=chunk_binding,
                placeholder_id=_required_control(
                    model.config,
                    "audio_chunk_token_id",
                ),
                park_id=_required_control(model.config, "eos_token_id"),
                commit_sink=None,
                capture=False,
                # The final coexisting CHUNK inventory must execute its real
                # decoder bindings. The pre-capture profile retains its original
                # dense-eager admission flag and never claims graph coverage.
                memory_profile=chunk_binding is None,
                graph_covers_decode=chunk_binding is not None,
                native_burst_handoff=native_profile_sink,
                staging=None,
            )

        if (
            getattr(execution, "arm", None)
            in {
                "compiled-static",
                "dense-graphed",
                "eager-graphed",
            }
            and not execution.cell_active
        ):
            if geometry_id is None:
                raise ValueError("static encoder profile geometry is missing")
            profile_cell = (
                execution.profile_ready_cell if ready_domain else execution.profile_cell
            )
            return profile_cell(
                geometry=geometry_id,
                population=num_rows,
                invoke=invoke,
            )
        return invoke()
    finally:
        consume_batch_stats()


# @spec PORT-PERF-009, PORT-PERF-011
def warmup_static_encoder_execution(
    model: NemotronASRForRNNT,
    *,
    device: torch.device,
) -> None:
    """Warm and attest every served geometry/population cell."""
    execution = model._encoder_execution
    if execution.arm not in {"compiled-static", "dense-graphed", "eager-graphed"}:
        return
    if getattr(execution, "ready", False):
        return
    cells = tuple(
        (geometry, population)
        for geometry in execution.warmup_geometries
        for population in execution.warmup_populations
    )
    execution.warmup_domain(
        expected_cells=cells,
        invoke=lambda geometry, population: run_persistent_state_profile(
            model,
            num_rows=population,
            device=device,
            geometry_id=geometry,
        ),
    )


__all__ = [
    "PersistentStateProfileInvocation",
    "build_profile_invocation",
    "run_persistent_state_profile",
    "warmup_static_encoder_execution",
]
