# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental owned burst projection after the replay oracle is validated."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

from vllm.models.nemotron_asr.advance import (
    ROLE_CHUNK,
    ROLE_EOU,
    ROLE_REPLAY,
    EmissionContext,
    EmissionProjection,
)
from vllm.models.nemotron_asr.rnnt import (
    BOOK_EXPECTED_LABEL,
    BOOK_PENDING_ECHO,
    QUEUE_HEAD,
    QUEUE_LEN,
)


@dataclass(frozen=True)
class NativeBurstProjection(EmissionProjection):
    sampled_token_ids: torch.Tensor
    num_sampled: torch.Tensor


def finalize_native_burst(
    projection: EmissionProjection,
    context: EmissionContext,
    *,
    park_id: int,
    blank_id: int,
) -> NativeBurstProjection:
    """Drain a validated projection to the state reached by legal replay PARK."""
    del blank_id
    n, cap = projection.queue.shape
    device = projection.queue.device
    ok = projection.row_status == 0
    chunk = (context.roles == ROLE_CHUNK) | (context.roles == ROLE_EOU)
    replay = context.roles == ROLE_REPLAY
    active = ok & (chunk | replay)
    start = torch.where(chunk, 0, context.book[:, QUEUE_HEAD]).long()
    length = projection.book[:, QUEUE_LEN].long()
    count = torch.where(active, (length - start).clamp(min=0, max=cap), 0)
    cols = torch.arange(cap + 1, device=device).unsqueeze(0)
    indices = (start.unsqueeze(1) + cols).clamp(min=0, max=cap - 1)
    tokens = projection.queue.gather(1, indices).long()
    payload = torch.where(cols < count.unsqueeze(1), tokens, -1)
    payload = torch.where(cols == count.unsqueeze(1), park_id, payload)
    book = projection.book.clone()
    book[:, QUEUE_HEAD] = torch.where(active, length, book[:, QUEUE_HEAD])
    book[:, BOOK_PENDING_ECHO] = torch.where(active, 0, book[:, BOOK_PENDING_ECHO])
    last = projection.queue.gather(
        1, (length - 1).clamp(min=0, max=cap - 1).unsqueeze(1)
    ).squeeze(1)
    book[:, BOOK_EXPECTED_LABEL] = torch.where(
        active & (count > 0), last, book[:, BOOK_EXPECTED_LABEL]
    )
    return NativeBurstProjection(
        rows=projection.rows.clone(),
        queue=projection.queue.clone(),
        book=book,
        row_status=projection.row_status.clone(),
        sampled_token_ids=payload,
        num_sampled=(count + 1).to(torch.int32),
    )


class NativeBurstHandoff:
    """One transient execution receipt; never an owner of resumable state."""

    def __init__(self, *, max_tokens: int = 142) -> None:
        self.max_tokens = max_tokens
        self._dummy_park: int | None = None
        self._last_epoch = 0
        self._snapshot: Any = None
        self._identity: tuple[Any, ...] | None = None
        self._payload: NativeBurstProjection | None = None
        self._reserved = False

    @contextmanager
    def dummy_sampling(self, *, park_id: int) -> Iterator[None]:
        if self._snapshot is not None or self._dummy_park is not None:
            raise RuntimeError("native burst dummy scope overlaps a pending execution")
        self._dummy_park = park_id
        try:
            yield
        finally:
            self._dummy_park = None

    @staticmethod
    def _key(snapshot: Any) -> tuple[Any, ...]:
        batch = snapshot.input_batch
        return (
            snapshot.epoch,
            tuple(snapshot.req_ids),
            tuple(batch.req_ids),
            tuple(int(i) for i in batch.idx_mapping_np),
            tuple((b.request_id, b.generation, b.slot_id) for b in snapshot.bindings),
            tuple(snapshot.block_ids),
            tuple(snapshot.req_states.req_id_to_index.get(r) for r in snapshot.req_ids),
        )

    def prepare(self, snapshot: Any) -> None:
        if self._snapshot is not None:
            raise RuntimeError("native burst output is pending")
        if snapshot.epoch <= self._last_epoch:
            raise RuntimeError("native burst epoch is stale")
        identity = self._key(snapshot)
        if (
            identity[1] != identity[2]
            or identity[3] != identity[6]
            or len(snapshot.bindings) != len(snapshot.req_ids)
            or any(
                b.request_id != r for b, r in zip(snapshot.bindings, snapshot.req_ids)
            )
        ):
            raise RuntimeError("native burst snapshot mismatch")
        self._last_epoch = snapshot.epoch
        self._snapshot = snapshot
        self._identity = identity

    def reserve(self, payload: NativeBurstProjection) -> Callable[[], None]:
        if self._snapshot is None:
            raise RuntimeError("native burst receipt is absent")
        if self._reserved:
            raise RuntimeError("native burst receipt is already reserved")
        if self._key(self._snapshot) != self._identity:
            raise RuntimeError("native burst snapshot changed")
        if payload.sampled_token_ids.shape[0] != len(self._snapshot.req_ids):
            raise RuntimeError("native burst payload row count mismatch")
        self._reserved = True

        def stage() -> None:
            self._payload = payload

        return stage

    def cancel_requests(self, req_ids: frozenset[str]) -> None:
        if self._snapshot is not None and req_ids.intersection(self._snapshot.req_ids):
            self.clear()

    def clear(self) -> None:
        self._snapshot = None
        self._identity = None
        self._payload = None
        self._reserved = False

    def consume(self, input_batch: Any) -> NativeBurstProjection:
        if self._snapshot is None or self._payload is None:
            raise RuntimeError("native burst output is absent")
        if (
            input_batch is not self._snapshot.input_batch
            or self._key(self._snapshot) != self._identity
        ):
            raise RuntimeError("native burst snapshot changed")
        payload = self._payload
        self.clear()
        return payload


class NativeBurstSampler:
    """Use the native variable-count result while retaining the base API."""

    def __init__(self, base: Any, handoff: NativeBurstHandoff) -> None:
        self._base = base
        self._handoff = handoff

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def __call__(self, logits: torch.Tensor, input_batch: Any) -> Any:
        from vllm.v1.worker.gpu.sample.output import SamplerOutput

        if self._handoff._dummy_park is not None:
            from vllm.models.nemotron_asr.manifests import SESSION_LIMITS

            rows = len(input_batch.req_ids)
            values = torch.full(
                (rows, SESSION_LIMITS["queue_capacity"] + 1),
                -1,
                dtype=torch.int64,
                device=logits.device,
            )
            values[:, 0] = self._handoff._dummy_park
            _profile_scratch = (
                torch.empty(
                    (rows, SESSION_LIMITS["queue_capacity"]),
                    dtype=torch.int32,
                    device=logits.device,
                ),
                torch.empty((rows, 7), dtype=torch.int32, device=logits.device),
                torch.empty((rows,), dtype=torch.int32, device=logits.device),
            )
            return SamplerOutput(
                sampled_token_ids=values,
                logprobs_tensors=None,
                num_nans=None,
                num_sampled=torch.ones(rows, dtype=torch.int32, device=logits.device),
                num_rejected=torch.zeros(rows, dtype=torch.int32, device=logits.device),
            )
        payload = self._handoff.consume(input_batch)
        return SamplerOutput(
            sampled_token_ids=payload.sampled_token_ids,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=payload.num_sampled,
            num_rejected=torch.zeros_like(payload.num_sampled),
        )


def validate_native_burst_config(config: Any, *, hf_config: Any | None = None) -> None:
    """Reject every native selection outside the initial 160-ms profile."""
    parallel = config.parallel_config
    if (
        config.scheduler_config.async_scheduling is not False
        or config.speculative_config is not None
        or any(
            getattr(parallel, name, None) != 1
            for name in (
                "pipeline_parallel_size",
                "tensor_parallel_size",
                "data_parallel_size",
                "decode_context_parallel_size",
                "prefill_context_parallel_size",
            )
        )
        or bool(getattr(parallel, "enable_expert_parallel", True))
        or bool(config.model_config.logits_processors)
    ):
        raise ValueError(
            "native burst requires synchronous non-speculative single-device execution"
        )
    if hf_config is not None:
        declared = getattr(hf_config, "supported_num_lookahead_tokens", None) or ()
        try:
            lookaheads = tuple(int(value) for value in declared)
        except TypeError:
            lookaheads = ()
        if lookaheads != (1,):
            raise ValueError(
                "native burst requires the declared 160-ms lookahead profile"
            )


def validate_native_burst_sampling(params: Any, *, park_id: int, capacity: int) -> None:
    if (
        params is None
        or params.n != 1
        or params.min_tokens != 0
        or params.ignore_eos
        or getattr(params, "repetition_penalty", None) != 1.0
        or getattr(params, "frequency_penalty", None) != 0.0
        or getattr(params, "presence_penalty", None) != 0.0
        or (params.max_tokens is not None and params.max_tokens < capacity + 1)
        or any(
            getattr(params, name, None) is not None
            for name in (
                "logprobs",
                "prompt_logprobs",
                "logprob_token_ids",
                "structured_outputs",
                "logit_bias",
                "allowed_token_ids",
                "thinking_token_budget",
                "repetition_detection",
                "trace_decode_token_ids",
            )
        )
        or any(
            bool(getattr(params, name, None))
            for name in ("stop", "bad_words", "extra_args")
        )
        or any(token != park_id for token in (params.stop_token_ids or []))
    ):
        raise ValueError(
            "native burst requires unmasked forced-token sampling and room for a "
            "complete burst plus PARK"
        )


def native_burst_invariant_rows(
    source: EmissionProjection,
    context: EmissionContext,
    native: NativeBurstProjection,
    *,
    park_id: int,
    blank_id: int,
    eou_token_id: int | None = None,
) -> torch.Tensor:
    """Validate native payload/count/book agreement before resident scatters."""
    n, cap = source.queue.shape
    device = source.queue.device
    if (
        native.sampled_token_ids.shape != (n, cap + 1)
        or native.sampled_token_ids.dtype != torch.int64
        or native.sampled_token_ids.device != device
        or native.num_sampled.shape != (n,)
        or native.num_sampled.dtype != torch.int32
        or native.num_sampled.device != device
        or native.book.shape != source.book.shape
        or native.queue.shape != source.queue.shape
    ):
        raise ValueError("native burst payload has an invalid tensor contract")
    active = (source.row_status == 0) & (
        (context.roles == ROLE_CHUNK)
        | (context.roles == ROLE_REPLAY)
        | (context.roles == ROLE_EOU)
    )
    start = torch.where(
        context.roles == ROLE_REPLAY, context.book[:, QUEUE_HEAD], 0
    ).long()
    length = source.book[:, QUEUE_LEN].long()
    remaining = torch.where(active, length - start, 0)
    count = native.num_sampled.long()
    cols = torch.arange(cap + 1, device=device).unsqueeze(0)
    emitted = cols < (count - 1).unsqueeze(1)
    park = cols == (count - 1).unsqueeze(1)
    padding = cols >= count.unsqueeze(1)
    values = native.sampled_token_ids
    valid_token = (values >= 0) & (values < blank_id)
    if eou_token_id is not None:
        valid_token |= values == eou_token_id
    expected = source.queue.gather(
        1, (start.unsqueeze(1) + cols).clamp(min=0, max=cap - 1)
    ).long()
    last = source.queue.gather(
        1, (length - 1).clamp(min=0, max=cap - 1).unsqueeze(1)
    ).squeeze(1)
    expected_book = source.book.clone()
    expected_book[:, QUEUE_HEAD] = torch.where(
        active, length, expected_book[:, QUEUE_HEAD]
    )
    expected_book[:, BOOK_PENDING_ECHO] = torch.where(
        active, 0, expected_book[:, BOOK_PENDING_ECHO]
    )
    expected_book[:, BOOK_EXPECTED_LABEL] = torch.where(
        active & (remaining > 0), last, expected_book[:, BOOK_EXPECTED_LABEL]
    )
    return (
        (count != remaining + 1)
        | (count < 1)
        | (count > cap + 1)
        | (emitted & (~valid_token | (values != expected))).any(dim=1)
        | (park & (values != park_id)).any(dim=1)
        | (padding & (values != -1)).any(dim=1)
        | (native.book != expected_book).any(dim=1)
        | (native.queue != source.queue).any(dim=1)
        | (native.row_status != source.row_status)
    )


def native_burst_token_budget(config: Any) -> int:
    from vllm.models.nemotron_asr.manifests import CADENCES, author_emission_manifest
    from vllm.models.nemotron_asr.nemotron_asr import _served_geometry_ids

    manifest = author_emission_manifest(config)["per_geometry"]
    labels = tuple(CADENCES)
    return max(
        int(manifest[labels[index]]["max_emission_tokens"])
        for index in _served_geometry_ids(config)
    )


def validate_native_burst_history(
    request: Any, *, max_model_len: int, burst_tokens: int
) -> None:
    prefill = request.prefill_token_ids
    if (
        prefill is None
        or prefill != request.prompt_token_ids
        or request.num_computed_tokens != 0
        or len(prefill) + burst_tokens > max_model_len
    ):
        raise ValueError(
            "native burst requires fresh per-turn token history with room for the "
            "entire payload"
        )


class NativeBurstProfileSink:
    """Ephemeral profile-only reservation, with no live request authority."""

    def __init__(self, *, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        self.payload: NativeBurstProjection | None = None

    def reserve(self, payload: NativeBurstProjection) -> Callable[[], None]:
        def stage() -> None:
            self.payload = payload

        return stage
