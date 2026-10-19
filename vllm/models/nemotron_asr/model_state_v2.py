# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRv2 projection of bounded core state views into Nemotron transactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.models.nemotron_asr.state_profile import (
    project_nemotron_state_pools,
)
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, StateSlotHandle
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.state_slots import StateSlotBindings, StateSlotRows
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


@dataclass(frozen=True)
class RowBinding:
    request_id: str
    handle: StateSlotHandle
    fresh: bool

    @property
    def slot_id(self) -> int:
        return self.handle.slot

    @property
    def generation(self) -> int:
        return self.handle.generation


@dataclass(frozen=True)
class ProjectionSnapshot:
    epoch: int
    req_ids: tuple[str, ...]
    scheduled_encoder_inputs: dict[str, list[int]]
    encoder_features: object
    bindings: tuple[RowBinding, ...]
    block_ids: tuple[tuple[int, ...], ...]
    input_batch: Any
    req_states: Any
    model_kwargs: dict[str, Any]
    request_metadata: dict[str, NewRequestData] | None = None
    token_ids: tuple[int, ...] = ()
    dummy_run: bool = False
    is_profile: bool = False
    no_page_io: bool = False
    state_batch: StateBatch | None = None


class _ResidentCommitSink:
    def __init__(self, sink: Any, access: StateSlotRows):
        self.sink = sink
        self.access = access

    def reserve(self, plan: Any) -> Any:
        self.access.validate_current()
        if tuple(b.request_id for b in plan.bindings) != self.access.request_ids:
            raise ValueError("commit row order differs from resident bindings")
        if tuple(b.block_id for b in plan.bindings) != self.access.slot_indices_cpu:
            raise ValueError("commit slots differ from resident bindings")
        return self.sink.reserve(plan)


class StateBatch:
    """Zero-copy resident pools with a step-scoped pre-commit guard."""

    def __init__(
        self,
        slots: StateSlotBindings,
        layer: str,
        bindings: tuple[RowBinding, ...],
        n_layers: int,
    ):
        self.access = slots.bind_rows(
            layer,
            tuple(b.request_id for b in bindings),
            tuple(b.handle for b in bindings),
        )
        self.pools = project_nemotron_state_pools(self.access, n_layers=n_layers)

    def commit_sink(self, sink: Any) -> _ResidentCommitSink:
        return _ResidentCommitSink(sink, self.access)


class NemotronASRModelState(ModelState):
    """Model-local projection with no slot allocation or retirement authority."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._metadata: dict[str, NewRequestData] = {}
        self._initialized: dict[str, StateSlotHandle] = {}
        self._epoch = 0
        self._snapshot: ProjectionSnapshot | None = None
        self._dummy_run = False
        self._is_profile = False
        self._scheduled_encoder_inputs: dict[str, list[int]] = {}
        self._features: torch.Tensor | None = None
        self._scheduled_tokens: dict[str, tuple[int, ...]] = {}

    def initialize_state_slots(self, bindings: StateSlotBindings) -> None:
        super().initialize_state_slots(bindings)
        layer = self.model._persistent_state_layer.prefix
        if set(bindings.views) != {layer}:
            raise ValueError("Nemotron requires exactly its aggregate state group")
        # All resident writes run on the runner stream. Inner graphs also replay
        # there; no auxiliary state-writing stream is created by this consumer.

    def prepare_step(
        self, scheduler_output: SchedulerOutput, *, dummy_run: bool, is_profile: bool
    ) -> None:
        self._dummy_run = dummy_run
        self._is_profile = is_profile
        self._snapshot = None
        self._scheduled_encoder_inputs = {}
        self._features = None
        self._scheduled_tokens = {}
        self._epoch = getattr(self, "_epoch", 0) + 1
        if is_profile and not dummy_run:
            raise ValueError("profile execution requires a dummy run")
        if dummy_run:
            return
        cached = scheduler_output.scheduled_cached_reqs
        if len(cached.req_ids) != len(cached.new_token_ids):
            raise ValueError("state step omitted cached host token slices")
        self._scheduled_tokens = {
            req_id: tuple(tokens)
            for req_id, tokens in zip(cached.req_ids, cached.new_token_ids)
        }
        for request in scheduler_output.scheduled_new_reqs:
            tokens = request.prefill_token_ids
            if tokens is None:
                raise ValueError("state step omitted new host token slice")
            start = request.num_computed_tokens
            count = scheduler_output.num_scheduled_tokens[request.req_id]
            self._scheduled_tokens[request.req_id] = tuple(
                tokens[start : start + count]
            )
        for req_id in scheduler_output.finished_req_ids:
            self._metadata.pop(req_id, None)
            self._initialized.pop(req_id, None)
        self.model._registry.prune(self._initialized)
        handoff = self.model._native_burst_handoff
        if handoff is not None:
            handoff.clear()

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        del req_index
        handoff = self.model._native_burst_handoff
        if handoff is not None:
            from vllm.models.nemotron_asr.native_burst import (
                validate_native_burst_history,
                validate_native_burst_sampling,
            )

            validate_native_burst_sampling(
                new_req_data.sampling_params,
                park_id=int(self.model.config.eos_token_id),
                capacity=handoff.max_tokens - 1,
            )
            validate_native_burst_history(
                new_req_data,
                max_model_len=self.max_model_len,
                burst_tokens=handoff.max_tokens,
            )
        self._metadata[new_req_data.req_id] = new_req_data

    def remove_request(self, req_id: str) -> None:
        # Core also calls this when replacing a parked turn. prepare_step prunes
        # terminal identities; the same live handle preserves its acoustic state.
        self._metadata.pop(req_id, None)

    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        handoff = getattr(getattr(self, "model", None), "_native_burst_handoff", None)
        if handoff is None:
            return None
        from vllm.models.nemotron_asr.native_burst import NativeBurstSampler

        state = self

        class Sampler(NativeBurstSampler):
            def __call__(self, logits: torch.Tensor, input_batch: Any) -> Any:
                if state._dummy_run:
                    assert handoff is not None
                    with handoff.dummy_sampling(
                        park_id=int(state.model.config.eos_token_id)
                    ):
                        return super().__call__(logits, input_batch)
                return super().__call__(logits, input_batch)

        return Sampler(sampler, handoff), None

    def dummy_inputs_embeds(self, num_tokens: int) -> torch.Tensor:
        return self.encoder_runner.inputs_embeds[:num_tokens]

    def prepare_inputs_embeds(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor | None:
        self._scheduled_encoder_inputs = scheduled_encoder_inputs
        self._features = None
        if self.supports_mm_inputs:
            self.execute_mm_encoder(scheduled_encoder_inputs)
            mm_embeds, is_mm_embed = self.gather_mm_embeddings(input_batch)
            self._features = self.encoder_runner.get_inputs_embeds(
                input_batch.input_ids[: input_batch.num_tokens], mm_embeds, is_mm_embed
            )[: input_batch.num_tokens_after_padding]
        return self._features

    def prepare_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> dict[str, Any]:
        req_ids = tuple(input_batch.req_ids)
        if self._dummy_run:
            snapshot = ProjectionSnapshot(
                self._epoch,
                req_ids,
                {},
                None,
                (),
                (),
                input_batch,
                req_states,
                {},
                dummy_run=True,
                is_profile=self._is_profile,
                no_page_io=True,
            )
        else:
            if self.state_slots is None:
                raise RuntimeError("persistent-state bindings are not initialized")
            if any(int(n) != 1 for n in input_batch.num_scheduled_tokens):
                raise ValueError("every streaming row must be single-token")
            layer = self.model._persistent_state_layer.prefix
            bindings = tuple(
                RowBinding(req_id, handle, self._initialized.get(req_id) != handle)
                for req_id in req_ids
                for handle in (self.state_slots.get_handle(req_id, layer),)
            )
            batch = StateBatch(
                self.state_slots, layer, bindings, int(self.model.config.n_layers)
            )
            token_rows = [self._scheduled_tokens.get(req_id, ()) for req_id in req_ids]
            if any(len(tokens) != 1 for tokens in token_rows):
                raise ValueError("every streaming row needs one scheduled host token")
            snapshot = ProjectionSnapshot(
                self._epoch,
                req_ids,
                self._scheduled_encoder_inputs,
                self._features,
                bindings,
                tuple((b.slot_id,) for b in bindings),
                input_batch,
                req_states,
                {},
                {r: self._metadata[r] for r in req_ids},
                token_ids=tuple(tokens[0] for tokens in token_rows),
                state_batch=batch,
            )
        self._snapshot = snapshot
        return {"persistent_state_projection": snapshot}

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        snapshot = self._snapshot
        if snapshot is None or snapshot.dummy_run:
            return
        _statuses, failed = self.model.collect_commit_status()
        if failed:
            assert self.state_slots is not None
            self.state_slots.fail_requests(failed)
        self._initialized.update(
            (b.request_id, b.handle)
            for b in snapshot.bindings
            if b.request_id not in failed
        )
        self.model.consume_batch_stats()
        self._snapshot = None

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        raise RuntimeError("Nemotron does not support outer CUDA graph capture")

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
    ) -> dict[str, Any]:
        if ubatch_idx or cudagraph_mode == CUDAGraphMode.FULL or for_capture:
            raise RuntimeError("Nemotron does not support DBO or outer graph capture")
        return {}
