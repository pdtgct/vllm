# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Task-5 plan provider: host row authority for the transaction.

`advance_model_rows` proves that its :class:`RowPlan` columns reproduce
each row's :class:`PreparedRowBinding`; this module is where those
bindings are MINTED truthfully. The model's ``prepare_row_plan_context``
runner hook observes the final host row order (request ids, scheduled
CPU token ids, per-request block ids, and each CHUNK's CPU envelope
header from its request-side multimodal feature) and asks the
:class:`SessionRegistry` to join them
into one immutable, consume-once :class:`PlanContext`; ``forward`` then
combines that context with the attention metadata's host row counts
into the transaction's ``RowPlan`` (design §Phase-6c transaction
seams, PORT-ADV-003 as amended).

Authority boundaries, stated exactly:

- The ENGINE is the authority for row order, decode/prefill counts,
  and block allocation (``CachedRequestState.block_ids`` at the pin).
- The REGISTRY is the authority for session identity across steps:
  admission generation, admitted geometry, the current session-control
  prompt, and block-remap rejection. It is stamped host-side from the
  FIRST minted CHUNK's envelope header — the serving tier's twin
  admission record read on CPU before any device work — and updated
  only by later minted CHUNKs.
- The DEVICE remains the cross-check: every envelope is re-validated
  against the plan authority by the transaction's row-status bits, so
  host/device drift masks rows instead of corrupting sessions.

Registry mutations are atomic per hook call. Prompt transitions remain
provisional until the transaction's status is consumed at the normal
scheduling boundary: a failed row clears the proposal and preserves
the prior admitted prompt, while a clean row publishes it.

Torch + stdlib only (CPU tensors): the whole module runs under the
macOS loader chain and is differentially tested locally.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import torch

from vllm.models.nemotron_asr.advance import (
    ENV_ADMISSION_MS_MOD,
    ENV_CHUNK_SEQUENCE,
    ENV_FINAL_TAIL,
    ENV_GEOMETRY_ID,
    ENV_PROMPT_INDEX,
    ENV_VALID_SAMPLES,
    ENV_VERSION,
    ENVELOPE_HEADER_SLOTS,
    ENVELOPE_VERSION,
    PreparedRowBinding,
    RowPlan,
)
from vllm.models.nemotron_asr.manifests import (
    ADMISSION_EPOCH_MODULUS_MS,
    FRONTEND_CONSTANTS,
    RAW_SAMPLES_PER_CHUNK,
)

#: Per-geometry deadline SLO budget, in ns: the cadence period itself
#: (a row must be serviced before its next chunk would arrive) —
#: derived from the manifest's sample counts and sample rate, never a
#: copied constant. Index = geometry id, in ``RAW_SAMPLES_PER_CHUNK``'s
#: (== ``CADENCES``'s) order.
_SAMPLES_PER_MS = FRONTEND_CONSTANTS["sample_rate"] / 1000
_DEADLINE_BUDGET_NS_BY_GEOMETRY: tuple[int, ...] = tuple(
    round(samples / _SAMPLES_PER_MS * 1_000_000)
    for samples in RAW_SAMPLES_PER_CHUNK.values()
)


def resolve_row_envelope_header(
    *,
    scheduled_token_id: int,
    placeholder_id: int,
    num_computed_tokens: int,
    mm_features: Sequence[object],
    scheduled_encoder_input_ids: Sequence[int],
) -> tuple[float, ...] | None:
    """Resolve one row's host CHUNK header independently of cache state.

    ``scheduled_encoder_input_ids`` names encoder cache misses/new compute,
    not every semantically present multimodal feature. The scheduled CPU
    token and the request feature overlapping that one-token window are the
    row authorities; a present encoder-input id only cross-checks them
    (PORT-ADV-003 / PORT-INT-003).
    """
    start = int(num_computed_tokens)
    end = start + 1
    overlapping: list[tuple[int, object]] = []
    for index, feature in enumerate(mm_features):
        position = getattr(feature, "mm_position", None)
        if position is None:
            raise ValueError(f"multimodal feature {index} has no position metadata")
        offset = int(position.offset)
        length = int(position.length)
        if length <= 0:
            raise ValueError(
                f"multimodal feature {index} has non-positive length {length}"
            )
        if offset < end and offset + length > start:
            overlapping.append((index, feature))

    scheduled = tuple(int(index) for index in scheduled_encoder_input_ids)
    is_chunk = int(scheduled_token_id) == int(placeholder_id)
    if not is_chunk:
        if overlapping:
            raise ValueError("non-CHUNK row has an overlapping multimodal feature")
        if scheduled:
            raise ValueError("non-CHUNK row has a scheduled encoder input")
        return None

    if len(overlapping) != 1:
        raise ValueError(
            "CHUNK row requires exactly one overlapping multimodal feature, found "
            f"{len(overlapping)}"
        )
    feature_index, feature = overlapping[0]
    if len(scheduled) > 1:
        raise ValueError("CHUNK row may have at most one encoder input scheduled")
    if scheduled and scheduled[0] != feature_index:
        raise ValueError(
            "scheduled encoder input must name the same overlapping feature"
        )
    modality = getattr(feature, "modality", None)
    if modality != "audio":
        raise ValueError(f"CHUNK row feature must be audio, got {modality!r}")
    item = getattr(feature, "data", None)
    if item is None:
        raise ValueError("CHUNK row envelope has no kwargs payload")
    try:
        field = item["audio"]
        payload = field.data
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("CHUNK row audio feature has no envelope payload") from exc
    env = payload if isinstance(payload, torch.Tensor) else torch.as_tensor(payload)
    if env.device.type != "cpu":
        raise ValueError("envelope payload must be host-resident at the hook")
    env = env.reshape(-1)
    if int(env.shape[0]) < ENVELOPE_HEADER_SLOTS:
        raise ValueError("CHUNK row envelope is smaller than its header")
    return tuple(float(value) for value in env[:ENVELOPE_HEADER_SLOTS].tolist())


def _reconstruct_deadline_ns(
    admission_ms_mod: int, *, now_ns: int, geometry: int
) -> int:
    """The absolute deadline from the envelope's wraparound admission
    stamp, the worker's own wall-clock ``now_ns``, and the geometry's
    cadence-period budget (design §Ingress-deadline plumbing).

    Wraparound-safe by construction: correct as long as true elapsed
    time between minting and this reconstruction stays under
    ``ADMISSION_EPOCH_MODULUS_MS`` milliseconds (~4.66 hours) — see
    that constant's docstring for the headroom argument.
    """
    now_ms_mod = (now_ns // 1_000_000) % ADMISSION_EPOCH_MODULUS_MS
    elapsed_ms = (now_ms_mod - admission_ms_mod) % ADMISSION_EPOCH_MODULUS_MS
    admission_ns = now_ns - elapsed_ms * 1_000_000
    return admission_ns + _DEADLINE_BUDGET_NS_BY_GEOMETRY[geometry]


@dataclass(frozen=True)
class ObservedRow:
    """One scheduled model row as the runner hook sees it, host-side.

    ``envelope_header``: the first :data:`ENVELOPE_HEADER_SLOTS` values
    of the CPU feature overlapping a CHUNK row's scheduled token,
    irrespective of encoder-cache hit/miss; ``None`` for non-CHUNK rows.
    ``has_prior_state`` mirrors the scheduler's freshness authority
    (``num_computed_tokens > 0``).
    """

    request_id: str
    block_id: int
    scheduled_token_id: int
    has_prior_state: bool
    envelope_header: tuple[float, ...] | None
    endpoint_mode: int = 0
    endpoint_threshold_frames: int = 0
    endpoint_residue_frames: int = 0


@dataclass(frozen=True)
class PlanContext:
    """One step's immutable host row authority (consume-once).

    Parallel CPU columns plus the atomically minted bindings; ``forward``
    splits rows by the metadata's decode/prefill counts and hands the
    result to the transaction, whose preflight re-proves that every
    column reproduces its binding.
    """

    step: int
    request_ids: tuple[str, ...]
    bindings: tuple[PreparedRowBinding, ...]
    block_ids: torch.Tensor
    geometry_id: torch.Tensor
    prompt_index: torch.Tensor
    is_chunk: torch.Tensor
    has_prior_state: torch.Tensor
    admission_generation: torch.Tensor
    ready_deadline_ns: torch.Tensor
    live_block_ids: torch.Tensor
    endpoint_mode: torch.Tensor
    endpoint_threshold_frames: torch.Tensor
    endpoint_residue_frames: torch.Tensor


@dataclass
class _Session:
    block_id: int
    generation: int
    geometry_id: int
    prompt_index: int
    pending_prompt_index: int | None = None
    progress_unit: int = -1
    progress_counter: int = 0
    progress_key: tuple[int, int, int] = (-1, -1, -1)
    progress_flushed: bool = False


def reject_unsupported_outer_graph_mode(compilation_config: object) -> None:
    """Fail startup when the outer runner would capture full decode.

    Phase 6c owns an eager transaction whose row plan and commit sink are
    host-step state. Piecewise graphs may cover lower operators, but the
    runner must not capture/replay the whole model ``forward`` until an
    exact padded-plan and replay-safe handoff exist (PORT-ADV-003).
    """
    mode = getattr(compilation_config, "cudagraph_mode", None)
    if mode is None:
        return
    decode_mode = getattr(mode, "decode_mode", None)
    runtime = decode_mode() if callable(decode_mode) else mode
    name = getattr(runtime, "name", str(runtime)).upper()
    if name == "FULL":
        raise ValueError(
            "Nemotron ASR rejects full CUDA graph decode until the runner "
            "provides exact padded RowPlan and commit-handoff replay semantics; "
            "use cudagraph_mode=NONE or PIECEWISE"
        )


def _header_int(header: tuple[float, ...], slot: int, name: str) -> int:
    value = header[slot]
    if not math.isfinite(value) or value != math.trunc(value):
        raise ValueError(
            f"envelope header {name} is not an exact integer: {value!r} "
            "— the serving tier mints headers, so this is a port defect"
        )
    return int(value)


class SessionRegistry:
    """Worker-side session identity authority (design §Phase-6c seams).

    Registers a session atomically at its first minted CHUNK, retains
    parked sessions, prunes on request completion, and rejects block
    remapping, duplicate block ownership, and re-admission of a live
    request before any state access. ``admission_generation`` is a
    process-monotonic counter — the capture/status ABA guard across
    block reuse.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}
        self._blocks: dict[int, str] = {}
        self._generation = 0

    def __len__(self) -> int:
        return len(self._sessions)

    def prune(self, resident_request_ids: Collection[str]) -> None:
        """Drop sessions whose request is no longer worker-resident."""
        keep = set(resident_request_ids)
        for req_id in [r for r in self._sessions if r not in keep]:
            session = self._sessions.pop(req_id)
            self._blocks.pop(session.block_id, None)

    # @spec PORT-SESS-016
    def publish_progress(
        self,
        request_id: str,
        generation: int,
        raw: tuple[int, int, int],
        *,
        starts_chunk: bool,
    ) -> tuple[int, int, int] | None:
        """Consume one status-clean transaction after registry lease validation.

        The worker's consume-once commit sink is the authority for control
        commits. Logical units include forced controls as well as CHUNKs;
        frontend counters alone cannot identify those ticketless units.
        """
        session = self._sessions.get(request_id)
        if (
            session is None
            or session.generation != generation
            or session.progress_flushed
        ):
            return None
        sequence, cursor, phase = raw
        if starts_chunk:
            if sequence <= session.progress_counter:
                return None
            session.progress_counter = sequence
            session.progress_unit += 1
        elif phase in (4, 5):
            session.progress_unit += 1
            phase -= 3
            if phase == 1:
                # Forced EOU commit alone is not a CHUNK, cursor advance,
                # or legal park. Retain its unit identity without progress.
                return None
        elif phase == 3:
            session.progress_unit += 1
            session.progress_flushed = True
        key = (session.progress_unit, cursor, phase)
        if key <= session.progress_key:
            return None
        session.progress_key = key
        return key

    def live_block_ids(self) -> list[int]:
        """The registry's allocated-block liveness authority, sorted."""
        return sorted(self._blocks)

    def validate_lease(self, bindings: Sequence[PreparedRowBinding]) -> None:
        """Reserve-time lease check: every binding must be CURRENT.

        Raises:
            ValueError: a binding whose request is unknown or whose
                block/generation no longer matches the registry — the
                composite reservation must fail pre-commit rather than
                stage status for a swapped or reused row.

        """
        for binding in bindings:
            session = self._sessions.get(binding.request_id)
            if session is None:
                raise ValueError(
                    f"lease check: request {binding.request_id!r} is not registered"
                )
            if (
                session.block_id != binding.block_id
                or session.generation != binding.admission_generation
            ):
                raise ValueError(
                    f"lease check: request {binding.request_id!r} binding does not "
                    "match the current registry lease"
                )

    def lease_is_current(self, bindings: Sequence[PreparedRowBinding]) -> bool:
        """No-fail stage-time revalidation of the same lease facts."""
        try:
            self.validate_lease(bindings)
        except ValueError:
            return False
        return True

    def resolve_status(
        self,
        *,
        request_id: str,
        admission_generation: int,
        clean: bool,
    ) -> bool:
        """Resolve one staged prompt proposal at status consumption.

        Returns ``False`` for a stale request/generation report instead
        of raising: this runs after resident commit at the scheduling
        boundary, where ABA drift is a recorded fatal fact.
        """
        session = self._sessions.get(request_id)
        if session is None or session.generation != admission_generation:
            return False
        if clean and session.pending_prompt_index is not None:
            session.prompt_index = session.pending_prompt_index
        session.pending_prompt_index = None
        return True

    def _parse_chunk_header(
        self,
        row: ObservedRow,
        *,
        num_prompts: int,
        num_geometries: int,
    ) -> tuple[int, int, int]:
        header = row.envelope_header
        if header is None:
            raise ValueError(
                f"CHUNK row {row.request_id!r} has no overlapping envelope "
                "header — the serving tier mints one per CHUNK"
            )
        if len(header) != ENVELOPE_HEADER_SLOTS:
            raise ValueError(
                f"envelope header for {row.request_id!r} has {len(header)} slots, "
                f"expected {ENVELOPE_HEADER_SLOTS}"
            )
        if _header_int(header, ENV_VERSION, "version") != ENVELOPE_VERSION:
            raise ValueError(
                f"envelope header for {row.request_id!r} carries version "
                f"{header[ENV_VERSION]!r}, expected {ENVELOPE_VERSION}"
            )
        if _header_int(header, ENV_VALID_SAMPLES, "valid_samples") < 0:
            raise ValueError(
                f"envelope header for {row.request_id!r} carries a negative "
                "valid-sample count"
            )
        final = _header_int(header, ENV_FINAL_TAIL, "final_tail")
        if final not in (0, 1):
            raise ValueError(
                f"envelope header for {row.request_id!r} carries a non-boolean final "
                f"flag {final}"
            )
        _header_int(header, ENV_CHUNK_SEQUENCE, "chunk_sequence")
        geometry = _header_int(header, ENV_GEOMETRY_ID, "geometry_id")
        if not 0 <= geometry < num_geometries:
            raise ValueError(
                f"envelope header for {row.request_id!r} carries geometry {geometry} "
                "outside the admitted set"
            )
        prompt = _header_int(header, ENV_PROMPT_INDEX, "prompt_index")
        if not 0 <= prompt < num_prompts:
            raise ValueError(
                f"envelope header for {row.request_id!r} carries prompt {prompt} "
                "outside the prompt dictionary"
            )
        # Host-side mirror of the device-side range check (design
        # §Ingress-deadline plumbing) — the same dual-validation
        # pattern already used for geometry/prompt.
        admission_ms_mod = _header_int(header, ENV_ADMISSION_MS_MOD, "admission_ms_mod")
        if not 0 <= admission_ms_mod < ADMISSION_EPOCH_MODULUS_MS:
            raise ValueError(
                f"envelope header for {row.request_id!r} carries "
                f"admission_ms_mod {admission_ms_mod} outside "
                f"[0, {ADMISSION_EPOCH_MODULUS_MS})"
            )
        return geometry, prompt, admission_ms_mod

    def bind_rows(
        self,
        rows: Sequence[ObservedRow],
        *,
        placeholder_id: int,
        num_prompts: int,
        num_geometries: int,
        now_ns: int,
        resident_request_ids: Sequence[str] | None = None,
    ) -> tuple[PreparedRowBinding, ...]:
        """Mint one binding per observed row, atomically per step.

        Registration, remap rejection, prompt-transition detection, and
        deadline minting happen here — the single writer of registry
        state. A defect at this tier (malformed header from OUR serving
        tier, block remap, duplicate ownership, unregistered
        continuation) raises: this is the host structural tier, before
        any resident read (PORT-STATE-007 posture).

        ``now_ns``: the WORKER's own wall-clock nanoseconds since
        epoch (design §Ingress-deadline plumbing) — NOT monotonic;
        every CHUNK row's true deadline is reconstructed from the
        envelope's wraparound admission stamp against this reference,
        so it must be directly comparable to the frontend process's
        wall clock (the same cross-process basis vLLM's own scheduler
        uses for ``Request.arrival_time``).
        """
        if now_ns <= 0:
            raise ValueError("now_ns must be a positive wall-clock nanosecond stamp")
        # Two-phase delta commit: validate the complete row set against
        # a filtered registry view, staging only newly admitted sessions
        # and prompt proposals. No resident map is cloned on the hot path.
        keep = None if resident_request_ids is None else set(resident_request_ids)
        if keep is not None:
            missing = {row.request_id for row in rows if row.request_id not in keep}
            if missing:
                raise ValueError(
                    "scheduled rows absent from worker-resident authority: "
                    f"{sorted(missing)}"
                )
        new_sessions: dict[str, _Session] = {}
        new_blocks: dict[int, str] = {}
        prompt_updates: dict[str, int | None] = {}
        staged_generation = self._generation

        def visible_session(request_id: str) -> _Session | None:
            if request_id in new_sessions:
                return new_sessions[request_id]
            if keep is not None and request_id not in keep:
                return None
            return self._sessions.get(request_id)

        def visible_block_owner(block_id: int) -> str | None:
            if block_id in new_blocks:
                return new_blocks[block_id]
            owner = self._blocks.get(block_id)
            if owner is not None and (keep is None or owner in keep):
                return owner
            return None

        seen: set[str] = set()
        bindings: list[PreparedRowBinding] = []
        for row in rows:
            if row.request_id in seen:
                raise ValueError(
                    f"request {row.request_id!r} appears twice in one step's row plan"
                )
            seen.add(row.request_id)
            is_chunk = row.scheduled_token_id == placeholder_id
            if not is_chunk and row.envelope_header is not None:
                raise ValueError(
                    f"non-CHUNK row {row.request_id!r} carries a scheduled envelope"
                )
            session = visible_session(row.request_id)
            if is_chunk and not row.has_prior_state:
                if session is not None:
                    raise ValueError(
                        f"request {row.request_id!r} re-admitted while "
                        "still registered — resident sessions are never "
                        "recomputed (PORT-STATE-005)"
                    )
                owner = visible_block_owner(row.block_id)
                if owner is not None:
                    raise ValueError(
                        f"block {row.block_id} is already owned by {owner!r} — "
                        "duplicate ownership"
                    )
                geometry, prompt, admission_ms_mod = self._parse_chunk_header(
                    row,
                    num_prompts=num_prompts,
                    num_geometries=num_geometries,
                )
                staged_generation += 1
                session = _Session(
                    block_id=row.block_id,
                    generation=staged_generation,
                    geometry_id=geometry,
                    prompt_index=prompt,
                )
                new_sessions[row.request_id] = session
                new_blocks[row.block_id] = row.request_id
                bindings.append(
                    PreparedRowBinding(
                        request_id=row.request_id,
                        block_id=row.block_id,
                        admission_generation=session.generation,
                        geometry_id=geometry,
                        prompt_index=prompt,
                        prior_prompt_index=prompt,
                        allow_prompt_transition=False,
                        is_chunk=True,
                        ready_deadline_ns=_reconstruct_deadline_ns(
                            admission_ms_mod,
                            now_ns=now_ns,
                            geometry=geometry,
                        ),
                    )
                )
                continue
            if session is None:
                raise ValueError(
                    f"continuing row {row.request_id!r} is not registered — the worker "
                    "never observed its first CHUNK"
                )
            if session.block_id != row.block_id:
                raise ValueError(
                    f"request {row.request_id!r} block remapped from "
                    f"{session.block_id} to {row.block_id} — resident "
                    "state never migrates outside park serialization"
                )
            if session.pending_prompt_index is not None:
                raise ValueError(
                    f"request {row.request_id!r} has a prompt status pending from the "
                    "previous CHUNK"
                )
            if is_chunk:
                _, prompt, admission_ms_mod = self._parse_chunk_header(
                    row,
                    num_prompts=num_prompts,
                    num_geometries=num_geometries,
                )
                prior = session.prompt_index
                # Provisional only: resolve_status publishes it after a
                # clean transaction status, or clears it on failure.
                prompt_updates[row.request_id] = prompt if prompt != prior else None
                bindings.append(
                    PreparedRowBinding(
                        request_id=row.request_id,
                        block_id=row.block_id,
                        admission_generation=session.generation,
                        geometry_id=session.geometry_id,
                        prompt_index=prompt,
                        prior_prompt_index=prior,
                        allow_prompt_transition=prompt != prior,
                        is_chunk=True,
                        # The SESSION's admitted geometry, not the
                        # envelope's parsed value — the same
                        # authority-over-cross-check pattern used for
                        # the admitted prompt (device-side
                        # ROW_STATUS_GEOMETRY already cross-checks the
                        # envelope against this same host authority).
                        ready_deadline_ns=_reconstruct_deadline_ns(
                            admission_ms_mod,
                            now_ns=now_ns,
                            geometry=session.geometry_id,
                        ),
                    )
                )
                continue
            bindings.append(
                PreparedRowBinding(
                    request_id=row.request_id,
                    block_id=row.block_id,
                    admission_generation=session.generation,
                    geometry_id=session.geometry_id,
                    prompt_index=session.prompt_index,
                    prior_prompt_index=session.prompt_index,
                    allow_prompt_transition=False,
                    is_chunk=False,
                    ready_deadline_ns=0,
                )
            )
        if keep is not None:
            self.prune(keep)
        self._sessions.update(new_sessions)
        self._blocks.update(new_blocks)
        for request_id, pending_prompt in prompt_updates.items():
            self._sessions[request_id].pending_prompt_index = pending_prompt
        self._generation = staged_generation
        return tuple(bindings)


def prepare_plan_context(
    registry: SessionRegistry,
    rows: Sequence[ObservedRow],
    *,
    resident_request_ids: Sequence[str],
    placeholder_id: int,
    num_prompts: int,
    num_geometries: int,
    now_ns: int,
    step: int,
) -> PlanContext:
    """One hook invocation: prune, mint bindings, build the columns."""
    bindings = registry.bind_rows(
        rows,
        placeholder_id=placeholder_id,
        num_prompts=num_prompts,
        num_geometries=num_geometries,
        now_ns=now_ns,
        resident_request_ids=resident_request_ids,
    )
    return PlanContext(
        step=step,
        request_ids=tuple(b.request_id for b in bindings),
        bindings=bindings,
        block_ids=torch.tensor([b.block_id for b in bindings], dtype=torch.int64),
        geometry_id=torch.tensor([b.geometry_id for b in bindings], dtype=torch.int64),
        prompt_index=torch.tensor(
            [b.prompt_index for b in bindings], dtype=torch.int64
        ),
        is_chunk=torch.tensor([b.is_chunk for b in bindings], dtype=torch.bool),
        has_prior_state=torch.tensor(
            [row.has_prior_state for row in rows], dtype=torch.bool
        ),
        admission_generation=torch.tensor(
            [b.admission_generation for b in bindings], dtype=torch.int64
        ),
        ready_deadline_ns=torch.tensor(
            [b.ready_deadline_ns for b in bindings], dtype=torch.int64
        ),
        live_block_ids=torch.tensor(registry.live_block_ids(), dtype=torch.int64),
        endpoint_mode=torch.tensor(
            [row.endpoint_mode for row in rows], dtype=torch.int64
        ),
        endpoint_threshold_frames=torch.tensor(
            [row.endpoint_threshold_frames for row in rows],
            dtype=torch.int64,
        ),
        endpoint_residue_frames=torch.tensor(
            [row.endpoint_residue_frames for row in rows],
            dtype=torch.int64,
        ),
    )


class PlanContextSlot:
    """Consume-once staging between the runner hook and ``forward``.

    ``stage`` replaces any unconsumed context (a context stranded by an
    exception or warm-up call can never leak into a later step);
    ``consume`` pops exactly once and fails loudly when nothing was
    staged this step.
    """

    def __init__(self) -> None:
        self._context: PlanContext | None = None

    def stage(self, context: PlanContext) -> None:
        self._context = context

    def consume(self) -> PlanContext:
        context = self._context
        self._context = None
        if context is None:
            raise ValueError(
                "no staged PlanContext: the prepare_row_plan_context "
                "runner hook did not run for this step (or the context "
                "was already consumed)"
            )
        return context


def build_row_plan(
    context: PlanContext,
    *,
    num_decodes: int,
    num_prefills: int,
    null_block_id: int,
    num_pool_blocks: int,
    execution_tier: int = 0,
) -> RowPlan:
    """Combine the consumed context with the metadata's host counts.

    The engine's decode/prefill split must agree with the context's
    prior-state column (a decode row without prior state, or a fresh
    row classified decode, is host/engine drift and fails the call
    here — before the transaction's own preflight re-proves the
    bindings).
    """
    n_real = len(context.bindings)
    if num_decodes < 0 or num_prefills < 0:
        raise ValueError("negative metadata row counts")
    if num_decodes + num_prefills != n_real:
        raise ValueError(
            f"metadata rows {num_decodes}+{num_prefills} disagree with the staged "
            f"context's {n_real} rows"
        )
    if bool((~context.has_prior_state[:num_decodes]).any()):
        raise ValueError("a decode row has no prior state — engine/context row drift")
    return RowPlan(
        state_indices_d=context.block_ids[:num_decodes].reshape(-1, 1),
        num_decodes=num_decodes,
        state_indices_p=context.block_ids[num_decodes:],
        num_prefills=num_prefills,
        has_initial_states_p=context.has_prior_state[num_decodes:],
        null_block_id=null_block_id,
        num_pool_blocks=num_pool_blocks,
        live_block_ids=context.live_block_ids,
        geometry_id=context.geometry_id,
        prompt_index=context.prompt_index,
        is_chunk=context.is_chunk,
        admission_generation=context.admission_generation,
        ready_deadline_ns=context.ready_deadline_ns,
        request_ids=context.request_ids,
        execution_tier=execution_tier,
        bindings=context.bindings,
        endpoint_mode=context.endpoint_mode,
        endpoint_threshold_frames=context.endpoint_threshold_frames,
        endpoint_residue_frames=context.endpoint_residue_frames,
    )
