# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The advance seam split (ledger P5-1; PORT-ADV-001/003).

The legacy ``run_forward_step`` (``forward_step.py``, deleted at
Task 7) was replaced by two narrower operations that separate the
storage-agnostic checkpoint transition from the outer page-pool
transaction:

- :func:`advance_session` — the ONE storage- and adapter-agnostic
  CHUNK transition (PORT-ADV-001). It owns the incremental frontend:
  the caller hands it raw PCM plus validated controls
  (:class:`ChunkBatch`) and gathered checkpoint state
  (:class:`SessionStateBatch`, including the frontend raw/mel tails and
  counters), and it advances the bounded frontend, encoder,
  conditioning, and RNN-T label loop. It never sees a pool, a
  ``state_indices`` tensor, a queue, or a book.
- :func:`advance_model_rows` — the ONE shared outer transaction
  (PORT-ADV-003) every native/fallback/probe caller passes through. It
  receives the scheduler's row mapping (:class:`RowPlan`) — not a
  pre-joined index tensor — so it itself composes decode-then-prefill
  indices, derives freshness from scheduler metadata (never recycled
  page contents), runs whole-call structural preflight (PORT-STATE-007)
  before any resident read, classifies CHUNK/REPLAY/FLUSH,
  gathers-only-for-CHUNK (PORT-STATE-008), calls ``advance_session``,
  materializes/validates captures, and projects the committed
  adapter-neutral :class:`AdvanceResult` through the selected emission
  adapter in one transaction.

Both operations are implemented as of Phase 6c (the four contract
seams — plan provenance, decode resolver, capture reservation, and
the row-status handoff — are the design's §Phase-6c transaction
seams). The former ``forward_step.py`` / ``run_forward_step`` legacy
path and regression oracle was deleted at Task 7, once the model
``forward`` was rewired through :func:`advance_model_rows` and the
five-cadence pod parity gate passed (ledger P5-1 — a semantic split,
never a second legacy forward path).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import torch

from vllm.models.nemotron_asr.profiling import phase
from vllm.models.nemotron_asr.state_scatter import (
    _execute_masked_page_scatter_,
    validate_masked_page_scatter,
    warmup_masked_page_scatter,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vllm.models.nemotron_asr.encoder_execution import (
        EncoderCaches,
        EncoderTransition,
    )
    from vllm.models.nemotron_asr.nemotron_asr import (
        NemotronASRCore,
    )
    from vllm.models.nemotron_asr.rnnt import (
        DecodeState,
        FrameAlignedDecode,
    )

#: An RNN-T bucket decode: ``(conditioned_frames, enc_lengths,
#: predictor, joint, state) -> (token_ids, token_lengths, next_state)``.
#: Both production candidates (``decode_compact_active``,
#: ``decode_dense_masked``) satisfy it; selection is a startup
#: dispatch-table decision from the measured dense-vs-compact profile
#: (PORT-DEC-008), never a hardcoded default.
RnntDecodeFn = Callable[
    ...,
    "tuple[torch.Tensor, torch.Tensor, DecodeState] | FrameAlignedDecode",
]


# @spec PORT-PERF-010
def prepare_encoder_transition_inputs(
    mel: torch.Tensor,
    caches: EncoderCaches,
    out_offsets: torch.Tensor,
    out_lengths: torch.Tensor,
    out_width: int,
    prompt_index: torch.Tensor,
    mel_lengths: torch.Tensor,
) -> tuple[
    torch.Tensor,
    EncoderCaches,
    torch.Tensor,
    torch.Tensor,
    int,
    torch.Tensor,
    torch.Tensor,
]:
    """Build the one transition tuple shared by profile, warmup, and serve.

    ``mel_lengths`` is each row's valid input mel width (pre-encode prefix
    included); the encoder masks its subsampling past it.
    """
    return (
        mel,
        caches,
        out_offsets,
        out_lengths,
        int(out_width),
        prompt_index,
        mel_lengths,
    )


def _has_packed_decode_state_interior(tensor: torch.Tensor) -> bool:
    """Whether ``(batch, hidden)`` is packed with optional layer padding.

    A tiered graph owns ``(layers, P, hidden)`` output storage and returns the
    exact-live ``[:, :B]`` prefix.  For ``B < P`` that prefix is not globally
    contiguous: the unused tier rows remain between predictor layers.  It is
    nevertheless non-overlapping and dense within every live layer when

    ``stride = (sL, hidden, 1)`` and ``sL >= B * hidden``.

    That is the complete layout needed by the read-only validation and
    batch-major resident-state commit below.  Requiring global contiguity
    would force an uncaptured D2D repack after every non-tier-exact replay;
    accepting anything weaker would admit transposed, stepped, or broadcast
    state interiors.
    """
    if tensor.dim() != 3:
        return False
    # Preserve the canonical eager contract exactly. PyTorch may assign
    # arbitrary strides to size-one dimensions while still correctly
    # classifying the tensor as contiguous (for example ``(H, 2H, 1)`` for
    # shape ``(L, 1, H)``), so the fixed-stride proof below is only the
    # additional graph-prefix case.
    if tensor.is_contiguous():
        return True
    _layers, batch, hidden = tensor.shape
    layer_stride, batch_stride, hidden_stride = tensor.stride()
    return (
        hidden_stride == 1 and batch_stride == hidden and layer_stride >= batch * hidden
    )


#: Chunk-envelope header layout (design §Chunk envelope): the versioned
#: FP32 carrier row is ``[version, valid_samples, geometry_id,
#: final_tail, prompt_index, chunk_sequence, admission_ms_mod,
#: samples...]``. Every header integer must be exactly representable
#: in FP32 (< 2**24). Slot order is the LLD's listing order — and
#: ``manifests.ENVELOPE_HEADER_FIELDS`` is the single canonical source
#: of that order; this tuple is cross-pinned to it by test, the same
#: pattern the module docstring already uses for the book/counter
#: slot layouts. VERSION 2 (bumped from 1, design §Ingress-deadline
#: plumbing): adds ``admission_ms_mod``, the PORT-owned segmenter's
#: acceptance wall-clock stamp (milliseconds since epoch modulo
#: ``manifests.ADMISSION_EPOCH_MODULUS_MS``) — a genuine schema
#: change, not additive-compatible, since the transaction validates
#: the exact header slot count on every row.
ENVELOPE_VERSION = 2
(
    ENV_VERSION,
    ENV_VALID_SAMPLES,
    ENV_GEOMETRY_ID,
    ENV_FINAL_TAIL,
    ENV_PROMPT_INDEX,
    ENV_CHUNK_SEQUENCE,
    ENV_ADMISSION_MS_MOD,
) = range(7)
ENVELOPE_HEADER_SLOTS = 7


@dataclass(frozen=True)
class ChunkBatch:
    """Raw PCM plus validated controls for a batch of CHUNK rows.

    The multimodal processor is stateless: it packs raw samples and
    controls, never mel (PORT-REGIME-001 / design §Chunk envelope). One
    row per CHUNK — every cadence unit ``buffer_realtime_audio`` mints,
    including the single explicit final-tail. Transport packet
    boundaries never mint a CHUNK.

    ``samples``: ``(B, S)`` FP32 mono, padded only inside the transport
    row to the largest admitted raw cadence; padding is never
    semantically appended to the audio.
    ``valid_samples``: ``(B,)`` long, the true sample count per row.
    ``geometry_id``: ``(B,)`` long, the admitted geometry; every
    envelope must match the session's admitted geometry before any
    state mutation.
    ``final_tail``: ``(B,)`` bool, the one explicit final-tail marker.
    ``prompt_index``: ``(B,)`` long, the LID prompt slot.
    ``chunk_sequence``: ``(B,)`` long, the monotonically increasing
    expected chunk sequence, validated against the session book.

    All control fields are validated (integer-valued, in range,
    sequence order) before the row is used.
    """

    samples: torch.Tensor
    valid_samples: torch.Tensor
    geometry_id: torch.Tensor
    final_tail: torch.Tensor
    prompt_index: torch.Tensor
    chunk_sequence: torch.Tensor


@dataclass
class SessionStateBatch:
    """Gathered checkpoint state :func:`advance_session` reads and
    mutates in place into the next state (design §advance_session).

    Storage-agnostic: the caller gathers page-backed (or any other)
    tensor views into these fields; NO pools, ``state_indices``, queue,
    or book live here — those are :func:`advance_model_rows`' concern.

    Frontend continuity (the incremental STFT frontend advances inside
    ``advance_session``):
      ``raw_tail``: ``(B, R)`` FP32 bounded raw-sample tail;
      ``mel_tail``: ``(B, 128, 9)`` FP32 committed-boundary mel tail;
      ``frontend_counters``: ``(B, 8)`` int64 control counters, in
      ``manifests.FRONTEND_COUNTER_FIELDS`` order (the manifest is the
      single source of the slot names and order): total valid samples,
      committed mel frames, encoded mel frames, raw-tail origin,
      raw-tail length, mel-tail length, expected chunk sequence, and
      finalization state.

    Encoder / conv (per layer, list index = encoder layer):
      ``channel``: each ``(B, window, d_model)`` left-context window;
      ``window_valid``: each ``(B, 1)`` int32 valid length;
      ``time``: each ``(B, d_model, conv_tail)`` depthwise-conv tail.

    Predictor:
      ``h`` / ``c``: each ``(B, pred_layers, pred_hidden)`` LSTM state;
      ``last_label``: ``(B,)`` long, the predictor's most recent label
      (the RNN-T loop's continuation across chunks).
    """

    raw_tail: torch.Tensor
    mel_tail: torch.Tensor
    frontend_counters: torch.Tensor
    channel: list[torch.Tensor]
    window_valid: list[torch.Tensor]
    time: list[torch.Tensor]
    h: torch.Tensor
    c: torch.Tensor
    last_label: torch.Tensor


@dataclass(frozen=True)
class PreparedCaptures:
    """GPU-resident named captures: padded storage + exact valid
    lengths, never Python lists (PORT-PERF-001). A finalized
    zero-frame CHUNK stages all three tensors with zero valid length
    — the checkpoint record survives even when no model work ran.

    ``frontend_mel``: ``(B, n_mels, T_mel_pad)``; ``mel_lengths``:
    ``(B,)`` long. ``encoder_raw`` / ``encoder_conditioned``:
    ``(B, T_enc_pad, d_model)``; ``encoder_lengths``: ``(B,)`` long.
    """

    frontend_mel: torch.Tensor
    mel_lengths: torch.Tensor
    encoder_raw: torch.Tensor
    encoder_conditioned: torch.Tensor
    encoder_lengths: torch.Tensor


@dataclass(frozen=True)
class AdvanceResult:
    """The adapter-neutral, GPU-RESIDENT result of one
    :func:`advance_session` call (no host synchronization on the
    result path, PORT-PERF-001).

    ``token_ids``: ``(B, K)`` int32, padded; ``token_lengths``:
    ``(B,)`` int32 — row ``b``'s burst is ``token_ids[b,
    :token_lengths[b]]``, ordered, NONBLANK, bounded by
    ``valid_encoder_frames × max_symbols_per_step`` for the row's
    geometry (PORT-INT-002), NOT by the attention window.
    ``row_status``: ``(B,)`` int32 — the device-resolved per-row
    status bitmask (PORT-ADV-004; bit names are
    ``frontend.ROW_STATUS_*``, 0 == committed clean). A row with any
    bit set mutated no state and its burst is zero-length; the
    transaction stages this tensor through its reserved asynchronous
    output handoff and the scheduler consumes it at its next boundary, mapping
    protocol bits to PORT-STATE-008's row tier and invariant bits to
    port-defect escalation — never a raise or device assertion from
    inside the transition.
    ``captures``: :class:`PreparedCaptures` or ``None`` when capture
    is disabled.

    The result carries no MRV1/runner projection: the selected
    emission adapter turns it into runner rows via an explicit
    :class:`EmissionProjection`; the outer transaction is the sole
    scatter owner.
    """

    token_ids: torch.Tensor
    token_lengths: torch.Tensor
    row_status: torch.Tensor | None = None
    captures: PreparedCaptures | None = None
    frame_emission_counts: torch.Tensor | None = None
    frame_final_labels: torch.Tensor | None = None
    frame_valid_lengths: torch.Tensor | None = None

    @property
    def row_valid(self) -> torch.Tensor | None:
        """``(B,)`` bool view of ``row_status``: True == clean."""
        if self.row_status is None:
            return None
        return self.row_status == 0


@dataclass(frozen=True)
class PreparedRowBinding:
    """One atomically prepared request-to-execution authority row.

    Task 5's runner-hook provider mints this immutable snapshot only after
    joining the request id to the session registry and current scheduler
    controls.  :class:`RowPlan` retains vector columns for efficient CPU and
    device operations; structural preflight requires those columns and the
    attention-metadata block composition to reproduce this binding exactly.
    That catches row swaps, generation-column drift, and control-vector drift
    before any resident read. It does not prove registry currency; Task 5's
    consume-once registry lease supplies that live ABA check.
    """

    request_id: str
    block_id: int
    admission_generation: int
    geometry_id: int
    prompt_index: int
    prior_prompt_index: int
    allow_prompt_transition: bool
    is_chunk: bool
    ready_deadline_ns: int


@dataclass(frozen=True)
class RowPlan:
    """The scheduler's authoritative row mapping (design §Initialization
    and row-to-page mapping).

    :func:`advance_model_rows` receives this — NOT a pre-joined index
    tensor — so it composes decode-then-prefill indices itself and can
    prove the composition (PORT-STATE-007), derives freshness from
    scheduler metadata rather than recycled page contents, and knows
    liveness and per-row geometry without a content sentinel.

    ``state_indices_d``: ``(num_decodes_padded, K)`` decode block
    indices; only ``[:num_decodes, 0]`` are real (extra columns are a
    speculative-decode configuration error, not values to flatten;
    CUDA-graph padding rows are excluded by ``num_decodes``).
    ``num_decodes`` / ``num_prefills``: real row counts.
    ``state_indices_p``: ``(num_prefills,)`` prefill block indices.
    ``has_initial_states_p``: ``(num_prefills,)`` bool; ``~mask`` is the
    fresh-row mask (first-chunk prefills initialize from metadata).
    ``null_block_id``: the reserved block that is never a live session
    page.
    ``num_pool_blocks``: the resident block-pool size (in-range bound).
    ``live_block_ids``: ``(L,)`` long, the scheduler's liveness
    authority — the block ids currently allocated to resident sessions
    (including blocks allocated to this call's fresh prefills). Every
    real row's index must be a member; a valid-but-non-live index is a
    whole-call structural defect. Liveness is never inferred from page
    contents.
    ``geometry_id``: ``(num_decodes + num_prefills,)`` long, the
    admitted geometry per real row, in decode-then-prefill order.
    ``prompt_index``: ``(num_decodes + num_prefills,)`` long, the
    CURRENT admitted session-control prompt per REAL row (the
    registry's authority under PORT-LID-003 as amended) —
    conditioning uses this value; the envelope's stamped prompt is a
    device cross-check (``ROW_STATUS_PROMPT_MISMATCH``).
    ``is_chunk``: ``(num_decodes + num_prefills,)`` bool, the HOST
    role authority — True for a row whose scheduled token is the
    minted placeholder (design §Phase-6c transaction seams: sync-free
    bucketing requires host-side roles; the device token is the
    cross-check via ``ROW_STATUS_ROLE_MISMATCH``, never the bucketing
    authority).
    ``admission_generation``: ``(num_decodes + num_prefills,)`` long,
    the registry's admission generation per row — the capture
    record's block-reuse/ABA guard.
    ``request_ids``: ordered request identities per real row — the
    status handoff's and capture records' provenance authority.
    ``ready_deadline_ns``: ``(num_decodes + num_prefills,)`` long,
    the scheduler's absolute ready deadline for each selected CHUNK;
    non-CHUNK rows carry zero. Buckets execute by their earliest
    selected-row deadline, never enum/geometry order.
    ``execution_tier``: the engine's padded decode execution-tier
    ceiling under a graph-covered profile; 0 for eager profiles, where
    the live bucket size IS the execution size (PORT-DEC-008 as
    amended).
    ``bindings``: one immutable :class:`PreparedRowBinding` per real
    row, minted atomically by the provider from registry + scheduler
    authority. Preflight requires the composed block and every parallel
    control column to reproduce it exactly; neither an ordered request-id
    tuple nor live-set membership alone establishes row identity. Its
    prior-prompt snapshot and explicit transition authorization let the
    first newly minted CHUNK after a valid locale update replace the old
    persisted prompt; REPLAY/FLUSH can never authorize that transition.

    All plan tensors are HOST-side (CPU) scheduler/registry authority:
    structural preflight validates them without any device
    synchronization, and the transaction uploads validated indices
    asynchronously for its device gathers.
    """

    state_indices_d: torch.Tensor
    num_decodes: int
    state_indices_p: torch.Tensor
    num_prefills: int
    has_initial_states_p: torch.Tensor
    null_block_id: int
    num_pool_blocks: int
    live_block_ids: torch.Tensor
    geometry_id: torch.Tensor
    prompt_index: torch.Tensor
    is_chunk: torch.Tensor
    admission_generation: torch.Tensor
    ready_deadline_ns: torch.Tensor
    request_ids: tuple[str, ...]
    execution_tier: int
    bindings: tuple[PreparedRowBinding, ...]
    endpoint_mode: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.int64)
    )
    endpoint_threshold_frames: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.int64)
    )
    endpoint_residue_frames: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.int64)
    )


@dataclass
class EmissionContext:
    """Per-call row context + GATHERED emission scratch the adapter
    consumes. The transaction gathers ``queue``/``book`` copies for
    the batch and later scatters the projection's updated scratch —
    the adapter NEVER mutates resident pools (the transaction is the
    sole scatter owner).

    ``roles``: ``(N,)`` long (``ROLE_CHUNK``/``ROLE_REPLAY``/
    ``ROLE_FLUSH``/``ROLE_EOU``); ``input_ids``: ``(N,)`` long; ``chunk_rows``:
    ``(B,)`` long positions of the CHUNK rows within the N-row batch
    (merged-result row ``j`` is call row ``chunk_rows[j]``); ``queue``
    ``(N, cap)`` / ``book`` ``(N, 7)``: gathered emission scratch;
    ``prompt_index``: ``(N,)`` int64, the row-authoritative prompt
    snapshot to persist on a clean CHUNK. ``row_status``: ``(N,)`` int32 —
    the transaction-composed FINAL
    per-row status (transition bits already folded in for CHUNK
    rows). The adapter trusts it and never recomputes protocol logic:
    any nonzero row emits park and keeps its scratch untouched.
    """

    roles: torch.Tensor
    input_ids: torch.Tensor
    chunk_rows: torch.Tensor
    queue: torch.Tensor
    book: torch.Tensor
    prompt_index: torch.Tensor
    row_status: torch.Tensor


@dataclass(frozen=True)
class EmissionProjection:
    """What the adapter returns: the runner rows plus the UPDATED
    emission scratch for the transaction to scatter at commit.

    ``rows``: ``(N, H)`` runner output (e.g. MRV1 decision carriers);
    ``queue`` / ``book``: the updated scratch, same shapes as the
    context's. ``row_status`` is the adapter's final status view; it
    must preserve every incoming bit and cover any proposed-book
    invariant it detects before choosing a non-park row.
    """

    rows: torch.Tensor
    queue: torch.Tensor
    book: torch.Tensor
    row_status: torch.Tensor


#: A model-local emission adapter: projects a committed
#: :class:`AdvanceResult` under an :class:`EmissionContext` into an
#: :class:`EmissionProjection`. Selection happens once at model init
#: (runner-mode configuration); the transaction owns every resident
#: scatter.
EmissionAdapter = Callable[["AdvanceResult", "EmissionContext"], "EmissionProjection"]

#: Transaction-owned per-row status bits — the PORT-ADV-004 vocabulary
#: extension settled at the Phase-6c preflight (design §Phase-6c
#: transaction seams; PORT-DEC-007 as amended). The frontend owns bits
#: 1..256 (``frontend.ROW_STATUS_*``); these continue the same int32
#: mask for defects the OUTER transaction detects around the CHUNK
#: transition. Any set bit makes the row a safe masked no-op — park
#: emission only, no state scatter — and the session is resolved at
#: the single batched asynchronous status readback, never by a raise
#: or device assertion on the trusted-identity path.
ROW_STATUS_ECHO_MISMATCH = 512
ROW_STATUS_QUEUE_NOT_DRAINED = 1024
ROW_STATUS_ROLE_MISMATCH = 2048
ROW_STATUS_ENVELOPE = 4096
ROW_STATUS_BOOK_IDENTITY = 8192
ROW_STATUS_SESSION_PROTOCOL = 16384
#: The envelope's stamped prompt disagrees with the host admission
#: authority (PORT-LID-003 as amended): conditioning always uses the
#: admitted value, so a valid-but-different carrier prompt masks the
#: row instead of silently changing model output.
ROW_STATUS_PROMPT_MISMATCH = 32768
#: The gathered session book violates its own invariants
#: (0 <= head <= length <= capacity, boolean echo flag, label ranges);
#: values are sanitized to safe substitutes before any use that could
#: fault, and the row masks.
ROW_STATUS_BOOK_INVARIANT = 65536
#: The decode produced more labels than the replay queue can hold —
#: rejected as a port defect, never truncated.
ROW_STATUS_BURST_OVERFLOW = 131072
#: The decode result violates the tensor/result contract (negative or
#: over-width length, invalid active nonblank id, or status mismatch).
#: Its length is sanitized to zero before the adapter sees it.
ROW_STATUS_DECODE_INVARIANT = 262144

#: Model-row roles at a scheduler step (the transaction's own
#: vocabulary; values match the retiring ``forward_ops`` constants for
#: continuity). CHUNK comes from HOST authority (``RowPlan.is_chunk``);
#: REPLAY is an echo-armed non-chunk row; EOU is the ordered forced
#: endpoint barrier on a drained live session; FLUSH is everything else
#: and is validated (drained + finalized) before it may emit park.
ROLE_CHUNK = 0
ROLE_REPLAY = 1
ROLE_FLUSH = 2
ROLE_EOU = 3


@dataclass(frozen=True)
class DecodeRequest:
    """One profile+geometry bucket's dispatch query (design §Phase-6c
    transaction seams).

    ``geometry``: the bucket's admitted geometry id.
    ``execution_batch_size``: the live size of this geometry bucket.
    Eager profiles execute it directly; a regional graph resolver maps
    it to the smallest captured tier not smaller than the live size.
    ``graph_covers_decode``: invocation-specific graph coverage; a
    covering graph structurally forces the capture-eligible arm.
    ``ready_decode_buckets``: decode buckets ready in this engine
    iteration — the multi-bucket compact-override input (the dispatch
    record's step-4 obligation: compact's occupancy-1.000 host lock is
    priced for single buckets only).
    ``execution_tier_limit``: the RowPlan's graph-tier authority for
    this invocation. The binding selects the smallest captured tier
    for the bucket and must not exceed this ceiling. Eager requests
    carry zero.
    ``memory_profile``: this bucket belongs to the disposable
    pre-capture activation-memory profile. The served resolver uses
    the same dense decoder eagerly without claiming graph coverage.
    """

    geometry: int
    execution_batch_size: int
    graph_covers_decode: bool
    ready_decode_buckets: int
    execution_tier_limit: int = 0
    memory_profile: bool = False


@dataclass(frozen=True)
class ResolvedDecode:
    """The resolver's typed answer: the selected arm name (telemetry),
    its bound callable, and the reason when a safety rule — graph
    coverage or the multi-bucket override — displaced the table's raw
    selection."""

    arm: str
    decode_fn: RnntDecodeFn
    override_reason: str | None = None


#: The immutable startup-constructed dispatch seam: the transaction
#: invokes it exactly once per CHUNK bucket. Construction validates
#: table identity (``DispatchTable.validate_runtime``) and compiles an
#: O(1) lookup; the hot path performs no parsing, sorting, logging, or
#: synchronization. A transaction without a resolver fails before any
#: resident-state read (PORT-DEC-008: no hardcoded default).
DecodeResolver = Callable[[DecodeRequest], ResolvedDecode]


def make_table_resolver(
    table: Any,
    *,
    lane: str,
    arms: Mapping[str, RnntDecodeFn],
    max_batch: int,
) -> DecodeResolver:
    """Compile a validated :class:`DispatchTable` into a resolver.

    Startup-only: for every geometry id (``manifests.CADENCES`` order)
    and every execution batch size ``1..max_batch``, the measured-tier
    entries are resolved ONCE — nearest measured tier, sync-free arm
    when the bracketing tiers disagree (the crossover-gap rule) —
    into a dense lookup, with one aggregate log line per geometry
    instead of per-lookup warnings. The returned resolver then:

    - structurally selects the capture-eligible arm whenever the
      invocation's regional graph covers decode;
    - forces the sync-free eager arm whenever more than one decode
      bucket is ready in the iteration (the multi-bucket override),
      recording ``override_reason`` for telemetry;
    - otherwise returns the compiled table arm.

    The caller has already run ``table.validate_runtime`` for its
    deployment profile; this factory trusts the table's content.

    Args:
        table: the validated ``decode_dispatch.DispatchTable``.
        lane: the execution/precision lane key (e.g. ``"fp32"``).
        arms: arm name -> decode callable bindings (``rnnt``'s two
            candidates; ``dense-graphed`` binds the dense callable).
        max_batch: the largest execution batch size to compile
            (resolved ``max_num_seqs``); larger requests clamp.

    Returns:
        The frozen resolver.

    Raises:
        KeyError: a geometry with no measured tiers, or an arm the
            ``arms`` mapping does not bind — startup failures, never
            hot-path ones.

    """
    from vllm.models.nemotron_asr.decode_dispatch import (
        SYNC_FREE_ARMS,
    )
    from vllm.models.nemotron_asr.manifests import (
        CADENCES,
    )

    labels = list(CADENCES)
    for required in ("dense-graphed", "dense-eager"):
        if required not in arms:
            raise KeyError(f"no callable bound for arm {required!r}")
    compiled: list[tuple[str, ...]] = []
    for label in labels:
        tiers = sorted(t for (g, t, la) in table.entries if g == label and la == lane)
        if not tiers:
            raise KeyError(f"no measured tiers for {label!r}/{lane!r}")
        by_batch: list[str] = [""]
        fallbacks = 0
        for batch in range(1, max_batch + 1):
            if batch in tiers:
                by_batch.append(table.entries[(label, batch, lane)])
                continue
            fallbacks += 1
            lower = max((t for t in tiers if t < batch), default=None)
            upper = min((t for t in tiers if t > batch), default=None)
            if lower is None:
                assert upper is not None
                by_batch.append(table.entries[(label, upper, lane)])
            elif upper is None:
                by_batch.append(table.entries[(label, lower, lane)])
            else:
                low_arm = table.entries[(label, lower, lane)]
                high_arm = table.entries[(label, upper, lane)]
                if low_arm != high_arm:
                    by_batch.append(low_arm if low_arm in SYNC_FREE_ARMS else high_arm)
                else:
                    by_batch.append(
                        low_arm if (batch - lower) <= (upper - batch) else high_arm
                    )
        for arm in set(by_batch[1:]):
            if arm not in arms:
                raise KeyError(f"no callable bound for arm {arm!r}")
        logger.info(
            "decode dispatch %s/%s: compiled batches 1..%d (%d unmeasured resolved by "
            "nearest/crossover-gap)",
            label,
            lane,
            max_batch,
            fallbacks,
        )
        compiled.append(tuple(by_batch))
    compiled_t = tuple(compiled)
    sync_free_eager = "dense-eager"

    def resolve(request: DecodeRequest) -> ResolvedDecode:
        if request.graph_covers_decode:
            return ResolvedDecode(
                arm="dense-graphed",
                decode_fn=arms["dense-graphed"],
                override_reason="regional-graph-coverage",
            )
        batch = min(max(request.execution_batch_size, 1), max_batch)
        arm = compiled_t[request.geometry][batch]
        if request.ready_decode_buckets > 1 and arm not in SYNC_FREE_ARMS:
            return ResolvedDecode(
                arm=sync_free_eager,
                decode_fn=arms[sync_free_eager],
                override_reason="multi-bucket-serialization-guard",
            )
        return ResolvedDecode(arm=arm, decode_fn=arms[arm])

    return resolve


@dataclass(frozen=True)
class CapturePlan:
    """The exact bounded footprint of one call's prepared capture
    records: record count AND payload bytes (PORT-HOOK-001 — a sink
    reserves both before any resident-state commit)."""

    rows: int
    payload_bytes: int


@dataclass(frozen=True)
class CaptureRecord:
    """One candidate CHUNK row's named-capture record.

    Host identity: ``row`` (call row position), ``block_id``, the
    registry's ``admission_generation`` (the block-reuse/ABA guard),
    and the admitted ``geometry``. Device identity/tensors stay
    GPU-resident scalar views/rows of the prepared padded captures —
    publication is an in-memory handoff; any D2H or durable I/O is
    the consumer's concern outside the transaction.
    """

    row: int
    request_id: str
    block_id: int
    admission_generation: int
    geometry: int
    chunk_sequence: torch.Tensor
    prompt_index: torch.Tensor
    row_status: torch.Tensor
    frontend_mel: torch.Tensor
    mel_length: torch.Tensor
    encoder_raw: torch.Tensor
    encoder_conditioned: torch.Tensor
    encoder_length: torch.Tensor


@dataclass(frozen=True)
class CommitPlan:
    """Atomic status + optional capture reservation request.

    The complete ordered row bindings bind the eventual device status rows
    to request, block, generation, and controls. Task 5's production sink
    revalidates/pins their registry lease through ``stage``/``cancel``;
    carrying only request ids and generations would not prevent row swaps.
    ``capture`` is ``None`` when capture is disabled, so the measured path
    prepares no capture-only state. Candidate records are frozen into this
    PRE-COMMIT plan; ``stage`` therefore performs no container allocation.
    """

    bindings: tuple[PreparedRowBinding, ...]
    capture: CapturePlan | None
    records: tuple[CaptureRecord, ...] = ()
    progress: torch.Tensor | None = None

    @property
    def request_ids(self) -> tuple[str, ...]:
        """Ordered request identities for bounded-sink indexing."""
        return tuple(binding.request_id for binding in self.bindings)

    @property
    def generations(self) -> tuple[int, ...]:
        """Ordered admission generations for ABA filtering."""
        return tuple(binding.admission_generation for binding in self.bindings)


class CommitReservation(Protocol):
    """One composite ticket spanning status and capture capacity.

    ``stage`` is the sole post-commit operation and is no-fail by
    construction: it stages the device status plus already-prepared
    candidate records into reserved storage. The sink exposes only
    status-clean records after the asynchronous status completes.
    ``cancel`` is idempotent and releases the whole reservation.
    """

    def stage(self, row_status: torch.Tensor) -> None: ...

    def cancel(self) -> None: ...


class CommitSink(Protocol):
    """Model/worker-owned bounded composite handoff (PORT-HOOK-001).

    Serving always provides this sink; GPU-free probes may omit it
    when capture is disabled. One pre-commit ``reserve`` admits both
    status and optional capture capacity atomically.
    """

    def reserve(self, plan: CommitPlan) -> CommitReservation: ...


def make_mrv1_adapter(
    *,
    hidden_size: int,
    park_id: int,
    blank_id: int,
) -> EmissionAdapter:
    """Build the MRV1 emission adapter (the correctness fallback,
    PORT-DEC-010's burst adapter arrives behind the same seam).

    The returned adapter is POOL-FREE and a pure tensor transform: it
    consumes the context's gathered queue/book scratch and the merged
    committed :class:`AdvanceResult`, and returns an
    :class:`EmissionProjection` — the first burst label as each CHUNK
    row's decision carrier (head=1 past it, pending-echo and
    expected-label armed), the next queued label for a validated
    REPLAY row, and ``park_id`` for drained, FLUSH, zero-burst
    (PORT-DEC-004), and status-masked rows. Park never arms an echo
    (PORT-DEC-003) and blank never appears (bursts are nonblank by
    the PORT-DEC-001 decode contract — ``blank_id`` documents the
    pinned factory identity). The adapter issues no host/device
    synchronization and mutates no resident pool; the transaction
    scatters.
    """
    del blank_id  # bursts are nonblank by contract; id pins identity

    def adapter(result: AdvanceResult, context: EmissionContext) -> EmissionProjection:
        book = context.book
        queue = context.queue
        n = book.shape[0]
        device = book.device
        from vllm.models.nemotron_asr.rnnt import (
            BOOK_EXPECTED_LABEL,
            BOOK_PENDING_ECHO,
            QUEUE_HEAD,
            QUEUE_LEN,
            QUEUE_PROMPT,
        )

        ok = context.row_status == 0
        is_replay = context.roles == ROLE_REPLAY
        head = book[:, QUEUE_HEAD].long()
        length = book[:, QUEUE_LEN].long()

        queue_out = queue.clone()
        book_out = book.clone()
        decisions = torch.full((n,), park_id, dtype=torch.long, device=device)

        # ---- CHUNK rows: burst into the queue, first label out ----
        chunk_rows = context.chunk_rows
        b = int(chunk_rows.shape[0])  # host shape, not a sync
        if b:
            burst = result.token_ids.to(queue.dtype)
            blen = result.token_lengths.long()
            k = int(burst.shape[1])
            cok = ok.index_select(0, chunk_rows)
            old_q = queue_out.index_select(0, chunk_rows)
            new_q = torch.zeros_like(old_q)
            if k:
                copied = min(k, int(new_q.shape[1]))
                cols = torch.arange(copied, device=device).unsqueeze(0)
                new_q[:, :copied] = torch.where(
                    cols < blen.unsqueeze(1),
                    burst[:, :copied],
                    torch.zeros_like(burst[:, :copied]),
                )
            queue_out.index_copy_(
                0,
                chunk_rows,
                torch.where(cok.unsqueeze(1), new_q, old_q),
            )
            has = blen > 0
            first = burst[:, 0].long() if k else torch.zeros_like(blen)
            decision_c = torch.where(has, first, torch.full_like(blen, park_id))
            old_b = book_out.index_select(0, chunk_rows)
            new_b = old_b.clone()
            dt = new_b.dtype
            new_b[:, QUEUE_HEAD] = has.to(dt)  # 1 past the emitted
            new_b[:, QUEUE_LEN] = blen.to(dt)
            new_b[:, BOOK_PENDING_ECHO] = has.to(dt)
            new_b[:, BOOK_EXPECTED_LABEL] = torch.where(
                has, first, torch.zeros_like(first)
            ).to(dt)
            new_b[:, QUEUE_PROMPT] = context.prompt_index.index_select(
                0, chunk_rows
            ).to(dt)
            book_out.index_copy_(
                0,
                chunk_rows,
                torch.where(cok.unsqueeze(1), new_b, old_b),
            )
            decisions.index_copy_(
                0,
                chunk_rows,
                torch.where(cok, decision_c, torch.full_like(decision_c, park_id)),
            )

        # ---- REPLAY rows: validated echo → next label or park ----
        rep = ok & is_replay
        drained = head >= length
        cap = queue.shape[1]
        # Clamp BOTH bounds: a corrupt negative head is masked by the
        # transaction's book-invariant bit, but the gather itself must
        # stay in-bounds for every row (a negative index is a device
        # fault, not a maskable value).
        next_label = (
            queue.gather(1, head.clamp(min=0, max=max(cap - 1, 0)).unsqueeze(1))
            .squeeze(1)
            .long()
        )
        emit_rep = rep & ~drained
        decisions = torch.where(emit_rep, next_label, decisions)
        dt = book_out.dtype
        book_out[:, QUEUE_HEAD] = torch.where(
            emit_rep, (head + 1).to(dt), book_out[:, QUEUE_HEAD]
        )
        book_out[:, BOOK_EXPECTED_LABEL] = torch.where(
            emit_rep,
            next_label.to(dt),
            book_out[:, BOOK_EXPECTED_LABEL],
        )
        # An emitting replay keeps its echo armed for the emitted
        # label; a drained replay parks and clears it (park never
        # arms an echo). FLUSH and masked rows keep their book.
        book_out[:, BOOK_PENDING_ECHO] = torch.where(
            emit_rep,
            torch.ones_like(book_out[:, BOOK_PENDING_ECHO]),
            torch.where(
                rep & drained,
                torch.zeros_like(book_out[:, BOOK_PENDING_ECHO]),
                book_out[:, BOOK_PENDING_ECHO],
            ),
        )

        # Decision carrier: slot 0 of each runner row. Written
        # directly, not through the legacy checked helper — its
        # min/max range guard is a host synchronization
        # (PORT-ADV-004); the id range is a STARTUP invariant
        # (num_logits < 2**24 keeps every id fp32-integer-exact).
        rows = torch.zeros(n, hidden_size, dtype=torch.float32, device=device)
        rows[:, 0] = decisions.to(rows.dtype)
        return EmissionProjection(
            rows=rows,
            queue=queue_out,
            book=book_out,
            row_status=context.row_status,
        )

    return adapter


def _mrv1_projection_invariant_rows(
    result: AdvanceResult,
    context: EmissionContext,
    projection: EmissionProjection,
    effective_status: torch.Tensor,
    *,
    park_id: int,
    blank_id: int,
    eou_token_id: int | None = None,
) -> torch.Tensor:
    """Validate MRV1 output and queue/book transitions exactly on device.

    Phase 6c serves only the MRV1 adapter. The future native-burst adapter
    receives its own typed projection validator with PORT-DEC-010; it must not
    silently pass through this one-token decision-carrier contract.
    """
    from vllm.models.nemotron_asr.rnnt import (
        BOOK_EXPECTED_LABEL,
        BOOK_GEOMETRY,
        BOOK_PENDING_ECHO,
        QUEUE_HEAD,
        QUEUE_LAST_LABEL,
        QUEUE_LEN,
        QUEUE_PROMPT,
    )

    rows = projection.rows
    decision = rows[:, 0]
    finite = torch.isfinite(decision)
    integral = finite & (decision == decision.trunc())
    legal = (decision == park_id) | ((decision >= 0) & (decision < blank_id))
    if eou_token_id is not None:
        legal |= decision == eou_token_id
    bad = (~integral) | (~legal)
    if rows.shape[1] > 1:
        bad |= (rows[:, 1:] != 0).any(dim=1)

    queue_changed = (projection.queue != context.queue).any(dim=1)
    book_changed = (projection.book != context.book).any(dim=1)
    clean = effective_status == 0
    failed = ~clean
    bad |= failed & ((decision != park_id) | queue_changed | book_changed)

    is_flush = clean & (context.roles == ROLE_FLUSH)
    bad |= is_flush & ((decision != park_id) | queue_changed | book_changed)

    # REPLAY consumes exactly the prior queue item at old head, updates only
    # head/pending/expected, and never rewrites queue payload or identity.
    head = context.book[:, QUEUE_HEAD].long()
    length = context.book[:, QUEUE_LEN].long()
    cap = int(context.queue.shape[1])
    next_label = (
        context.queue.gather(
            1,
            head.clamp(min=0, max=max(cap - 1, 0)).unsqueeze(1),
        )
        .squeeze(1)
        .long()
    )
    replay = clean & (context.roles == ROLE_REPLAY)
    replay_emits = replay & (head < length)
    replay_drained = replay & ~replay_emits
    expected_replay_decision = torch.where(
        replay_emits,
        next_label,
        torch.full_like(next_label, park_id),
    )
    expected_replay_head = torch.where(replay_emits, head + 1, head)
    expected_replay_pending = torch.where(
        replay_emits,
        torch.ones_like(head),
        torch.zeros_like(head),
    )
    expected_replay_label = torch.where(
        replay_emits,
        next_label,
        context.book[:, BOOK_EXPECTED_LABEL].long(),
    )
    replay_bad = (
        (decision.long() != expected_replay_decision)
        | queue_changed
        | (projection.book[:, QUEUE_HEAD].long() != expected_replay_head)
        | (projection.book[:, QUEUE_LEN] != context.book[:, QUEUE_LEN])
        | (projection.book[:, QUEUE_LAST_LABEL] != context.book[:, QUEUE_LAST_LABEL])
        | (projection.book[:, QUEUE_PROMPT] != context.book[:, QUEUE_PROMPT])
        | (projection.book[:, BOOK_PENDING_ECHO].long() != expected_replay_pending)
        | (projection.book[:, BOOK_EXPECTED_LABEL].long() != expected_replay_label)
        | (projection.book[:, BOOK_GEOMETRY] != context.book[:, BOOK_GEOMETRY])
    )
    bad |= (replay | replay_drained) & replay_bad

    # CHUNK replaces the queue with exactly its active burst, emits the first
    # label, and arms an echo iff the burst is nonempty.
    chunk_rows = context.chunk_rows
    if int(chunk_rows.shape[0]):
        chunk_clean = clean.index_select(0, chunk_rows)
        chunk_queue = projection.queue.index_select(0, chunk_rows)
        chunk_book = projection.book.index_select(0, chunk_rows)
        chunk_decision = decision.index_select(0, chunk_rows)
        lengths = result.token_lengths.long()
        has = lengths > 0
        width = int(result.token_ids.shape[1])
        expected_queue = torch.zeros_like(chunk_queue)
        copied = min(width, cap)
        if copied:
            cols = torch.arange(copied, device=chunk_queue.device).unsqueeze(0)
            expected_queue[:, :copied] = torch.where(
                cols < lengths.unsqueeze(1),
                result.token_ids[:, :copied].to(expected_queue.dtype),
                torch.zeros_like(expected_queue[:, :copied]),
            )
            first = result.token_ids[:, 0].long()
        else:
            first = torch.zeros_like(lengths)
        expected_decision = torch.where(
            has,
            first,
            torch.full_like(first, park_id),
        )
        old_book = context.book.index_select(0, chunk_rows)
        chunk_bad = (
            (chunk_decision.long() != expected_decision)
            | (chunk_queue != expected_queue).any(dim=1)
            | (chunk_book[:, QUEUE_HEAD].long() != has.long())
            | (chunk_book[:, QUEUE_LEN].long() != lengths)
            | (chunk_book[:, QUEUE_LAST_LABEL] != old_book[:, QUEUE_LAST_LABEL])
            | (
                chunk_book[:, QUEUE_PROMPT].long()
                != context.prompt_index.index_select(0, chunk_rows)
            )
            | (chunk_book[:, BOOK_PENDING_ECHO].long() != has.long())
            | (
                chunk_book[:, BOOK_EXPECTED_LABEL].long()
                != torch.where(has, first, torch.zeros_like(first))
            )
            | (chunk_book[:, BOOK_GEOMETRY] != old_book[:, BOOK_GEOMETRY])
        )
        chunk_bad &= chunk_clean
        bad.index_copy_(
            0,
            chunk_rows,
            bad.index_select(0, chunk_rows) | chunk_bad,
        )
    return bad


# @spec PORT-ADV-001, PORT-ADV-004
def advance_session(
    core: NemotronASRCore,
    batch: ChunkBatch,
    state: SessionStateBatch,
    *,
    geometry: int,
    decode_fn: RnntDecodeFn,
    encoder_transition: EncoderTransition | None = None,
    capture: bool = False,
    row_status: torch.Tensor | None = None,
) -> AdvanceResult:
    """The ONE storage- and adapter-agnostic CHUNK transition.

    CHUNK-only: REPLAY and FLUSH rows never enter this function. It owns
    every stateful checkpoint step — the incremental bounded frontend
    (advancing ``state``'s raw/mel tails and counters from ``batch``'s
    raw samples and controls), encoder-cache advancement, language
    conditioning, predictor advancement, and RNN-T emission ordering —
    so no caller reimplements any of them (PORT-ADV-001). It reads and
    mutates ``state`` in place into the next state and returns the
    adapter-neutral :class:`AdvanceResult`.

    Length-aware (PORT-ADV-004): one call serves one profile+geometry
    bucket carrying mixed session-first / continuing / final /
    zero-frame rows. Every shape is host-derived from the bucket
    geometry (padded frontend width ``C``, matching the reference's
    largest single regular/final cadence shift, and encoder
    width from the subsampling formula); decode receives only the
    geometry manifest's maximum-valid encoder prefix, while named
    captures retain the full padded encoder width. Validity is per-row
    length tensors. Every per-row failure — wrong chunk sequence, audio
    after finalization, an envelope geometry different from the
    bucket's, an oversized final residual, or a frontend design
    invariant — is a masked no-op reported as a bit in
    ``AdvanceResult.row_status``; the transaction consumes that
    tensor at its commit readback. The transition never raises on a
    device predicate and never uses a device assertion. The call
    issues no host/device synchronization outside ``decode_fn``.

    Args:
        core: the assembled pipeline (encoder / lid / predictor /
            joint / featurizer).
        batch: raw PCM plus validated controls, one row per CHUNK.
        state: gathered checkpoint state, read and mutated into the
            next state.
        geometry: the bucket's admitted geometry id — a host value,
            per the profile+geometry bucket contract (PORT-PERF-001).
        decode_fn: the RNN-T decode to run — REQUIRED and never
            defaulted here (PORT-DEC-008): ``decode_compact_active``
            (eager, synchronizes on compaction) and
            ``decode_dense_masked`` (fixed-trip, sync-free) stay
            unselected candidates until the measured dense-vs-compact
            profile; startup then builds a dispatch table keyed by
            geometry, padded batch tier, and execution/precision
            profile, and passes the selected callable per bucket.
        encoder_transition: optional startup-bound execution mechanism for
            the exact encoder+conditioning transition. ``None`` executes
            the eager reference directly (unit probes and compatibility).
        capture: the explicit capture policy (PORT-HOOK-001). OFF by
            default: performance runs return ``captures=None`` with no
            capture-only allocations and no extended lifetime for the
            raw/conditioned encoder tensors. ON stages the three named
            tensors at the bucket's fixed padded widths with exact
            logical lengths — including zero length for a finalized
            zero-frame CHUNK.
        row_status: optional ``(B,)`` int32 incoming status carrying
            transaction-owned protocol bits (``ROW_STATUS_QUEUE_NOT_
            DRAINED``, ``ROW_STATUS_ROLE_MISMATCH``, ``ROW_STATUS_
            ENVELOPE``, ``ROW_STATUS_BOOK_IDENTITY``) composed BEFORE
            the transition's own envelope-protocol predicates; a row
            arriving with any bit set is a masked no-op end to end.
            ``None`` means all rows arrive clean. Never mutated.

    Returns:
        The GPU-resident :class:`AdvanceResult` (padded token
        tensors, per-row status, prepared captures).

    Raises:
        ValueError: if ``geometry`` is not an admitted geometry id
            (a host configuration error, not a row condition).

    """
    from vllm.models.nemotron_asr.encoder_execution import (
        encoder_geometry_shape,
        execute_encoder_transition,
    )
    from vllm.models.nemotron_asr.frontend import (
        CTR_COMMITTED_MEL_FRAMES,
        CTR_ENCODED_MEL_FRAMES,
        CTR_EXPECTED_CHUNK_SEQUENCE,
        MEL_TAIL_FRAMES,
        ROW_STATUS_FINAL_OVERSIZE,
        ROW_STATUS_GEOMETRY,
        ROW_STATUS_SEQUENCE,
        advance_frontend,
    )
    from vllm.models.nemotron_asr.manifests import (
        CADENCES,
    )
    from vllm.models.nemotron_asr.rnnt import (
        DecodeState,
    )

    n_rows = batch.samples.shape[0]
    device = batch.samples.device
    lookaheads = [right for (_, right) in CADENCES.values()]
    if not 0 <= geometry < len(lookaheads):
        raise ValueError(f"unknown bucket geometry id {geometry}")
    lookahead = lookaheads[geometry]
    geometry_shape = encoder_geometry_shape(core, geometry)
    cadence = geometry_shape.cadence_frames
    # The bucket's padded frontend width is one reference cadence
    # shift: first rows commit C-7, continuing rows C, session-first
    # final rows at most C-7, and continuing final rows at most C before
    # dropping the sub-eight boundary debt.
    pad_frames = cadence
    mel_width = geometry_shape.mel_width

    counters = state.frontend_counters
    # Envelope-protocol status bits (PORT-ADV-004), all on device: the
    # envelope geometry must match the bucket, the sequence must match
    # the session's expected counter, and a final residual must be
    # strictly smaller than one cadence unit (an oversized final is an
    # ingress defect, never priced into the padded width). The
    # frontend composes in its own FINALIZED and invariant bits.
    hop = core.featurizer.hop_length
    if row_status is None:
        incoming = torch.zeros(n_rows, dtype=torch.int32, device=device)
    else:
        incoming = row_status.to(device=device, dtype=torch.int32).clone()
    incoming |= (batch.geometry_id.to(device) != geometry).to(
        torch.int32
    ) * ROW_STATUS_GEOMETRY
    incoming |= (
        batch.chunk_sequence.to(device) != counters[:, CTR_EXPECTED_CHUNK_SEQUENCE]
    ).to(torch.int32) * ROW_STATUS_SEQUENCE
    incoming |= (
        batch.final_tail.to(device) & (batch.valid_samples.to(device) >= cadence * hop)
    ).to(torch.int32) * ROW_STATUS_FINAL_OVERSIZE
    # Regular-cadence targets, vectorized: B_{seq+1} = (8L+1) + seq·C
    # (final rows ignore targets inside the frontend).
    targets = (8 * lookahead + 1) + batch.chunk_sequence * cadence
    session_first = batch.chunk_sequence == 0

    # Pre-consume snapshot: the fixed-width mel-tail prefix is the
    # encoder's pre-encode cache (NeMo semantics: the FULL nine-slot
    # tail, zeros where not yet valid, with the CONFIGURED two-output
    # post-first-chunk drop — never a dynamic overlap); session-first
    # rows take no prefix and drop nothing.
    prefix = state.mel_tail.clone()

    with phase("port.featurize"):
        new_frames, counts, row_status = advance_frontend(
            core.featurizer,
            batch.samples,
            batch.valid_samples,
            batch.final_tail,
            targets,
            raw_tail=state.raw_tail,
            mel_tail=state.mel_tail,
            counters=state.frontend_counters,
            cadence_frames=cadence,
            pad_frames=pad_frames,
            row_status=incoming,
        )
    row_ok = row_status == 0
    counters[:, CTR_EXPECTED_CHUNK_SEQUENCE] += row_ok.to(torch.int64)

    # Per-row encoder input on the bucket's fixed grid: column j of
    # row b reads [prefix | new][j + 9 - p_b] — p_b is 9 for
    # continuing rows (the full nine-slot pre-encode cache) and 0 for
    # session-first rows. One batched gather, no phase branch.
    p = torch.where(
        session_first,
        torch.zeros_like(counts),
        torch.full_like(counts, MEL_TAIL_FRAMES),
    )
    combined = torch.cat([prefix, new_frames], dim=2)
    col = torch.arange(mel_width, device=device).view(1, 1, -1)
    gidx = (col + (MEL_TAIL_FRAMES - p).view(-1, 1, 1)).clamp(max=combined.shape[2] - 1)
    mel = combined.gather(2, gidx.expand(n_rows, combined.shape[1], mel_width))
    mel_len = p + counts
    mel = torch.where(col < mel_len.view(-1, 1, 1), mel, mel.new_zeros(()))

    # Per-row pre-encode drop and logical encoder lengths; the padded
    # encoder width is the host formula on the fixed mel width.
    drop = torch.where(
        session_first,
        torch.zeros_like(counts),
        torch.full_like(counts, PRE_ENCODE_DROP),
    )
    enc_lengths = torch.where(
        counts > 0,
        torch.clamp(
            core.encoder.pre_encode.output_lengths(mel_len) - drop,
            min=0,
        ),
        torch.zeros_like(counts),
    )
    out_width = geometry_shape.out_width

    caches = _GatheredCaches(state)
    transition_inputs = prepare_encoder_transition_inputs(
        mel,
        caches,
        drop,
        enc_lengths,
        out_width,
        batch.prompt_index,
        mel_len,
    )
    with torch.no_grad():
        with phase("port.encode"):
            if encoder_transition is None:
                enc, conditioned = execute_encoder_transition(
                    core,
                    *transition_inputs,
                )
            else:
                enc, conditioned = encoder_transition(*transition_inputs)
        with phase("port.decode"):
            decode_in = DecodeState(
                h=state.h.transpose(0, 1).contiguous(),
                c=state.c.transpose(0, 1).contiguous(),
                last_label=state.last_label.clone(),
            )
            # Extension callables receive writable tensors. Preserve an
            # independent oracle before handing them over so an in-place decoder
            # cannot rewrite the baseline used for no-emission/state checks.
            decode_baseline = DecodeState(
                h=decode_in.h.clone(),
                c=decode_in.c.clone(),
                last_label=decode_in.last_label.clone(),
            )
            # The padded encoder grid includes PRE_ENCODE_DROP trailing
            # outputs produced from its nine-frame overlap prefix. They are
            # zeroed above and excluded by enc_lengths, but are not part of
            # the geometry's valid decode domain or dense graph key. Keep the
            # full tensor for named captures and pass a zero-copy prefix view
            # to both eager and graphed decoders.
            max_valid_frames = lookahead + 1
            if out_width != max_valid_frames + PRE_ENCODE_DROP:
                raise ValueError(
                    "padded encoder width disagrees with the geometry-valid decode "
                    "prefix"
                )
            decode_frames = conditioned[:, :max_valid_frames]
            decoded = decode_fn(
                decode_frames,
                enc_lengths,
                core.predictor,
                core.joint,
                decode_in,
            )
        from vllm.models.nemotron_asr.rnnt import (
            FrameAlignedDecode,
        )

        if isinstance(decoded, FrameAlignedDecode):
            token_ids = decoded.token_ids
            token_lengths = decoded.token_lengths
            decode_out = decoded.state
            frame_emission_counts = decoded.frame_emission_counts
            frame_final_labels = decoded.frame_final_labels
        else:
            token_ids, token_lengths, decode_out = decoded
            frame_emission_counts = None
            frame_final_labels = None
    if (
        token_ids.device != device
        or token_ids.dtype != torch.int32
        or token_ids.dim() != 2
        or token_ids.shape[0] != n_rows
        or not token_ids.is_contiguous()
        or token_lengths.device != device
        or token_lengths.dtype != torch.int32
        or tuple(token_lengths.shape) != (n_rows,)
        or not token_lengths.is_contiguous()
    ):
        raise ValueError("decode_fn returned malformed token tensors")
    decode_type = type(decode_out)
    if (
        decode_type.__module__ != DecodeState.__module__
        or decode_type.__qualname__ != DecodeState.__qualname__
    ):
        raise ValueError("decode_fn must return DecodeState")
    for name, actual, expected, layout_valid in (
        (
            "h",
            decode_out.h,
            decode_baseline.h,
            _has_packed_decode_state_interior(decode_out.h),
        ),
        (
            "c",
            decode_out.c,
            decode_baseline.c,
            _has_packed_decode_state_interior(decode_out.c),
        ),
        (
            "last_label",
            decode_out.last_label,
            decode_baseline.last_label,
            decode_out.last_label.is_contiguous(),
        ),
    ):
        if (
            tuple(actual.shape) != tuple(expected.shape)
            or actual.dtype != expected.dtype
            or actual.device != expected.device
            or not layout_valid
        ):
            raise ValueError(
                f"decode_fn next-state {name} changed shape/dtype/device/layout"
            )
    lengths = token_lengths.long()
    token_width = int(token_ids.shape[1])
    length_valid = (lengths >= 0) & (lengths <= token_width)
    has_emission = length_valid & (lengths > 0)
    if token_width:
        last_emitted = (
            token_ids.gather(
                1,
                (lengths - 1).clamp(min=0, max=token_width - 1).unsqueeze(1),
            )
            .squeeze(1)
            .long()
        )
    else:
        last_emitted = torch.zeros_like(decode_out.last_label)
    zero_state_changed = (
        (decode_out.h != decode_baseline.h).any(dim=(0, 2))
        | (decode_out.c != decode_baseline.c).any(dim=(0, 2))
        | (decode_out.last_label != decode_baseline.last_label)
    )
    decode_bad = (
        (~torch.isfinite(decode_out.h)).any(dim=(0, 2))
        | (~torch.isfinite(decode_out.c)).any(dim=(0, 2))
        | (decode_out.last_label < 0)
        | (decode_out.last_label > int(core.blank_id))
        | (~length_valid)
        | (has_emission & (decode_out.last_label != last_emitted))
        | ((lengths == 0) & zero_state_changed)
    )
    row_status = row_status | (decode_bad.to(torch.int32) * ROW_STATUS_DECODE_INVARIANT)
    decode_clean = (row_status == 0).view(-1, 1, 1)
    state.h.copy_(torch.where(decode_clean, decode_out.h.transpose(0, 1), state.h))
    state.c.copy_(torch.where(decode_clean, decode_out.c.transpose(0, 1), state.c))
    state.last_label.copy_(
        torch.where(
            row_status == 0,
            decode_out.last_label,
            state.last_label,
        )
    )
    counters[:, CTR_ENCODED_MEL_FRAMES] = torch.where(
        row_ok,
        counters[:, CTR_COMMITTED_MEL_FRAMES],
        counters[:, CTR_ENCODED_MEL_FRAMES],
    )

    if not capture:
        return AdvanceResult(
            token_ids=token_ids,
            token_lengths=token_lengths,
            row_status=row_status,
            frame_emission_counts=frame_emission_counts,
            frame_final_labels=frame_final_labels,
            frame_valid_lengths=enc_lengths,
        )
    # Capture lengths come from logical lengths (PORT-HOOK-001): a
    # zero-commit row stages all three tensors at zero length — its
    # mel capture is fully zero even where the encoder input carried
    # the pre-encode prefix.
    cap_mel_len = torch.where(counts > 0, mel_len, torch.zeros_like(mel_len))
    return AdvanceResult(
        token_ids=token_ids,
        token_lengths=token_lengths,
        row_status=row_status,
        captures=PreparedCaptures(
            frontend_mel=torch.where(
                col < cap_mel_len.view(-1, 1, 1),
                mel,
                mel.new_zeros(()),
            ),
            mel_lengths=cap_mel_len,
            encoder_raw=enc,
            encoder_conditioned=conditioned,
            encoder_lengths=enc_lengths,
        ),
        frame_emission_counts=frame_emission_counts,
        frame_final_labels=frame_final_labels,
        frame_valid_lengths=enc_lengths,
    )


#: The configured pre-encode overlap dropped from every non-first
#: chunk's encoder output (the checkpoint's nine-mel cache -> two
#: subsampled outputs; a value, never derived per row).
PRE_ENCODE_DROP = 2


class _StackedRows:
    """Per-layer gathered views behind stacked-tensor indexing:
    ``stream_step`` reads ``caches.channel.shape[2]`` and does
    ``caches.channel[idx]`` reads / slice-assign writes; reads return
    the (B, ...) view, writes copy through to the gathered tensors."""

    def __init__(self, views: list[torch.Tensor]) -> None:
        self._views = views

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._views[idx]

    def __setitem__(self, idx: int, value: torch.Tensor) -> None:
        self._views[idx].copy_(value)

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self._views), *self._views[0].shape)

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Expose the complete storage for graph-owned adapter cloning."""
        return tuple(self._views)


class _GatheredCaches:
    """``StreamingCaches``' surface over a gathered
    :class:`SessionStateBatch` (channel/time/valid/left_context) —
    ``stream_step`` advances the gathered views directly, so the
    golden-proven advance IS the scratch write."""

    def __init__(self, state: SessionStateBatch) -> None:
        self.channel = _StackedRows(state.channel)
        self.time = _StackedRows(state.time)
        self._window_valid = state.window_valid
        self.left_context = state.channel[0].shape[1]

    @classmethod
    def _from_tensors(
        cls,
        *,
        channel: tuple[torch.Tensor, ...],
        time: tuple[torch.Tensor, ...],
        valid: tuple[torch.Tensor, ...],
        left_context: int,
    ) -> _GatheredCaches:
        adapter = cls.__new__(cls)
        adapter.channel = _StackedRows(list(channel))
        adapter.time = _StackedRows(list(time))
        adapter._window_valid = list(valid)
        adapter.left_context = int(left_context)
        return adapter

    @property
    def valid(self) -> torch.Tensor:
        return self._window_valid[0].reshape(-1).to(torch.long)

    @valid.setter
    def valid(self, value: torch.Tensor) -> None:
        for slot in self._window_valid:
            slot.copy_(value.reshape(slot.shape).to(slot.dtype))

    def graph_storage(self) -> Any:
        """Return all cache tensors, including every valid-length slot."""
        from vllm.models.nemotron_asr.encoder_execution import (
            EncoderCacheStorage,
        )

        return EncoderCacheStorage(
            channel=self.channel.tensors,
            time=self.time.tensors,
            valid=tuple(self._window_valid),
        )

    def empty_like(self) -> _GatheredCaches:
        """Clone this adapter's exact Python layout over fresh storage."""
        storage = self.graph_storage()
        return type(self)._from_tensors(
            channel=tuple(torch.empty_like(tensor) for tensor in storage.channel),
            time=tuple(torch.empty_like(tensor) for tensor in storage.time),
            valid=tuple(torch.empty_like(tensor) for tensor in storage.valid),
            left_context=self.left_context,
        )

    def narrow_rows(self, rows: int) -> _GatheredCaches:
        """Clone this adapter's layout over a leading-row view of its storage."""
        storage = self.graph_storage()
        return type(self)._from_tensors(
            channel=tuple(tensor.narrow(0, 0, rows) for tensor in storage.channel),
            time=tuple(tensor.narrow(0, 0, rows) for tensor in storage.time),
            valid=tuple(tensor.narrow(0, 0, rows) for tensor in storage.valid),
            left_context=self.left_context,
        )


def _h2d(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Host-authority tensor to the compute device without a hot-path
    synchronization: pinned staging + non-blocking copy on CUDA (a
    fresh ``pin_memory()`` allocation every call — the un-pooled
    fallback :class:`HostStaging` exists to avoid), plain move
    elsewhere."""
    if device.type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


@dataclass
class HostStaging:
    """Model-owned reusable pinned H2D staging (PORT-PERF-001).

    Preallocated once per worker at ``capacity`` rows: each named
    purpose below copies host data into its OWN dedicated slice of a
    persistent pinned buffer and issues one non-blocking H2D from
    that same storage every call, eliminating the per-call
    ``pin_memory()`` allocation :func:`_h2d` otherwise performs.
    ``None`` (the default everywhere staging is threaded through)
    keeps the un-pooled ``_h2d`` path for probes/CPU.

    Every named purpose is staged AT MOST ONCE per
    :func:`advance_model_rows` call: the composed decode-then-prefill
    indices, fresh row positions, the fresh-init book rows, the
    ``is_chunk`` role vector, geometry, admitted prompt, prior
    prompt, and the merged CHUNK row positions (``chunk_all``).
    Distinct purposes are distinct storage — two purposes must never
    share a slot within one call, since a slot's non-blocking H2D
    copy can still be in flight (unsynchronized on the host) when the
    next ``stage()`` overwrites its pinned memory. The per-bucket
    CHUNK row-position vector is staged once per RESOLVED BUCKET (a
    call can resolve several buckets), so it cannot share the
    once-per-call arena; instead :meth:`stage_bucket` gives it one
    dedicated buffer PER GEOMETRY (each geometry resolves at most once
    per call), which is both allocation-free and free of the
    overwrite race a single shared bucket slot would carry.
    """

    capacity: int

    #: The single-occurrence-per-call int64 purposes, in arena row
    #: order. ``is_chunk`` (bool) and ``fresh_init_book`` ((n, 7))
    #: live in their own dedicated buffers below, not this arena.
    _INT64_SLOTS: ClassVar[tuple[str, ...]] = (
        "composed_indices",
        "fresh_positions",
        "chunk_all",
        "geometry",
        "admitted_prompt",
        "prior_prompt",
        "cadence_frames",
    )

    _arena: torch.Tensor = field(init=False, repr=False)
    _book_init: torch.Tensor = field(init=False, repr=False)
    _is_chunk_buf: torch.Tensor = field(init=False, repr=False)
    _bucket_arena: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("HostStaging capacity must be positive")
        from vllm.models.nemotron_asr.manifests import (
            CADENCES,
        )

        pin = torch.cuda.is_available()
        self._arena = torch.empty(
            (len(self._INT64_SLOTS), self.capacity),
            dtype=torch.int64,
            pin_memory=pin,
        )
        self._book_init = torch.empty(
            (self.capacity, 7), dtype=torch.int64, pin_memory=pin
        )
        self._is_chunk_buf = torch.empty(
            (self.capacity,), dtype=torch.bool, pin_memory=pin
        )
        # One dedicated bucket-position buffer PER GEOMETRY: each
        # geometry resolves at most once per call, so a per-geometry
        # buffer's in-flight non-blocking copy is never overwritten
        # within a call — which is exactly what a single shared slot
        # could not guarantee, and why the per-bucket vector used to
        # fall back to the un-pooled (pin_memory-per-call) _h2d path.
        self._bucket_arena = torch.empty(
            (len(CADENCES), self.capacity), dtype=torch.int64, pin_memory=pin
        )

    def stage(
        self,
        slot_name: str,
        cpu_tensor: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Copy ``cpu_tensor`` into ``slot_name``'s leading rows and
        return its non-blocking device copy.

        Raises:
            ValueError: ``cpu_tensor`` has more rows than
                ``capacity``, or ``slot_name`` names no known slot.

        """
        n = int(cpu_tensor.shape[0])
        if n > self.capacity:
            raise ValueError(
                f"staged vector of {n} rows exceeds HostStaging capacity "
                f"{self.capacity}"
            )
        if slot_name == "fresh_init_book":
            buf = self._book_init[:n]
        elif slot_name == "is_chunk":
            buf = self._is_chunk_buf[:n]
        else:
            try:
                row = self._INT64_SLOTS.index(slot_name)
            except ValueError:
                raise ValueError(f"unknown HostStaging slot {slot_name!r}") from None
            buf = self._arena[row, :n]
        buf.copy_(cpu_tensor)
        return buf.to(device, non_blocking=True)

    def stage_bucket(
        self,
        geometry: int,
        cpu_tensor: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Copy one resolved bucket's row-position vector into
        ``geometry``'s OWN pinned buffer and return its non-blocking
        device copy. Safe to call once per resolved bucket within a
        call: distinct geometries use distinct buffers, so no bucket's
        in-flight copy is ever overwritten by a later bucket in the
        same call.

        Raises:
            ValueError: ``cpu_tensor`` has more rows than ``capacity``,
                or ``geometry`` is outside the per-geometry arena.

        """
        n = int(cpu_tensor.shape[0])
        if n > self.capacity:
            raise ValueError(
                f"staged bucket of {n} rows exceeds HostStaging capacity "
                f"{self.capacity}"
            )
        if not 0 <= geometry < self._bucket_arena.shape[0]:
            raise ValueError(f"bucket geometry {geometry} outside the staging arena")
        buf = self._bucket_arena[geometry, :n]
        buf.copy_(cpu_tensor)
        return buf.to(device, non_blocking=True)


def _structural_preflight(plan: RowPlan, n_ids: int, n_embeds: int) -> torch.Tensor:
    """PORT-STATE-007: whole-call structural validation, HOST-side
    over the plan's CPU authority tensors, before any page read.

    Returns:
        The composed decode-then-prefill CPU index tensor.

    Raises:
        ValueError: naming the first structural defect.

    """
    if plan.num_decodes < 0 or plan.num_prefills < 0:
        raise ValueError("plan row counts must be nonnegative")
    if plan.num_pool_blocks <= 0:
        raise ValueError("plan.num_pool_blocks must be positive")
    if not 0 <= plan.null_block_id < plan.num_pool_blocks:
        raise ValueError("plan.null_block_id outside the resident pool")
    n_real = plan.num_decodes + plan.num_prefills
    if n_ids != n_real or n_embeds != n_real:
        raise ValueError(
            f"row count mismatch: plan has {n_real} real rows, input_ids {n_ids}, "
            f"inputs_embeds {n_embeds}"
        )
    d = plan.state_indices_d
    if d.device.type != "cpu" or d.dtype != torch.int64:
        raise ValueError("state_indices_d must be CPU int64 authority")
    if d.dim() != 2:
        raise ValueError(f"state_indices_d must be (rows, K); got dim {d.dim()}")
    if d.shape[1] != 1:
        raise ValueError(
            f"{d.shape[1]} decode index columns: extra speculative "
            "columns are a configuration error, never values to "
            "flatten (PORT-STATE-007)"
        )
    if d.shape[0] < plan.num_decodes:
        raise ValueError(
            f"state_indices_d has {d.shape[0]} rows for {plan.num_decodes} real decodes"
        )
    real_d = d[: plan.num_decodes, 0]
    pad_d = d[plan.num_decodes :, 0]
    if pad_d.numel() and bool((pad_d != plan.null_block_id).any()):
        raise ValueError(
            f"graph-padding rows must carry the null block id {plan.null_block_id} "
            "(real/padding mismatch)"
        )
    p = plan.state_indices_p
    if p.device.type != "cpu" or p.dtype != torch.int64:
        raise ValueError("state_indices_p must be CPU int64 authority")
    if p.dim() != 1 or p.shape[0] != plan.num_prefills:
        raise ValueError(
            f"state_indices_p shape {tuple(p.shape)} for {plan.num_prefills} prefills"
        )
    if (
        plan.has_initial_states_p.device.type != "cpu"
        or plan.has_initial_states_p.dtype != torch.bool
        or plan.has_initial_states_p.dim() != 1
        or plan.has_initial_states_p.shape[0] != plan.num_prefills
    ):
        raise ValueError("has_initial_states_p shape mismatch")
    per_row = (
        ("geometry_id", plan.geometry_id, torch.int64),
        ("prompt_index", plan.prompt_index, torch.int64),
        ("is_chunk", plan.is_chunk, torch.bool),
        ("admission_generation", plan.admission_generation, torch.int64),
        ("ready_deadline_ns", plan.ready_deadline_ns, torch.int64),
    )
    for name, tensor, dtype in per_row:
        if (
            tensor.device.type != "cpu"
            or tensor.dtype != dtype
            or tensor.dim() != 1
            or tensor.shape[0] != n_real
        ):
            raise ValueError(f"plan.{name} must be CPU {dtype} shaped ({n_real},)")
    for name, tensor in (
        ("endpoint_mode", plan.endpoint_mode),
        ("endpoint_threshold_frames", plan.endpoint_threshold_frames),
        ("endpoint_residue_frames", plan.endpoint_residue_frames),
    ):
        if tensor.numel() and (
            tensor.device.type != "cpu"
            or tensor.dtype != torch.int64
            or tensor.dim() != 1
            or tensor.shape[0] != n_real
        ):
            raise ValueError(
                f"plan.{name} must be empty or CPU int64 shaped ({n_real},)"
            )
    live = plan.live_block_ids
    if live.device.type != "cpu" or live.dtype != torch.int64 or live.dim() != 1:
        raise ValueError("plan.live_block_ids must be rank-one CPU int64")
    if len(plan.request_ids) != n_real:
        raise ValueError(
            f"plan.request_ids has {len(plan.request_ids)} entries, expected {n_real}"
        )
    if any(not isinstance(req_id, str) or not req_id for req_id in plan.request_ids):
        raise ValueError("plan.request_ids must be nonempty strings")
    if len(set(plan.request_ids)) != len(plan.request_ids):
        raise ValueError("plan.request_ids must be unique within the call")
    if len(plan.bindings) != n_real:
        raise ValueError(
            f"plan.bindings has {len(plan.bindings)} entries, expected {n_real}"
        )
    if bool((plan.admission_generation < 0).any()):
        raise ValueError("plan.admission_generation must be nonnegative")
    chunk_deadlines = plan.ready_deadline_ns[plan.is_chunk]
    if chunk_deadlines.numel() and bool((chunk_deadlines <= 0).any()):
        raise ValueError("selected CHUNK deadlines must be positive")
    if bool((plan.ready_deadline_ns[~plan.is_chunk] != 0).any()):
        raise ValueError("non-CHUNK rows must carry deadline zero")
    if plan.execution_tier < 0:
        raise ValueError("plan.execution_tier must be >= 0")
    idx = torch.cat([real_d, p])
    if idx.numel():
        if bool((idx == plan.null_block_id).any()):
            raise ValueError("null block id claimed by a real row")
        if bool((idx < 0).any()) or bool((idx >= plan.num_pool_blocks).any()):
            raise ValueError(f"state index out of range [0, {plan.num_pool_blocks})")
        if int(torch.unique(idx).numel()) != int(idx.numel()):
            raise ValueError(
                "duplicate state index across the decode/prefill composition"
            )
        if bool((~torch.isin(idx, plan.live_block_ids)).any()):
            raise ValueError(
                "state index outside the live (registry-allocated) block set"
            )
    for row, binding in enumerate(plan.bindings):
        if not isinstance(binding, PreparedRowBinding):
            raise ValueError("plan.bindings must contain PreparedRowBinding")
        if type(binding.request_id) is not str or not binding.request_id:
            raise ValueError("binding request_id must be a nonempty string")
        integer_fields = (
            binding.block_id,
            binding.admission_generation,
            binding.geometry_id,
            binding.prompt_index,
            binding.prior_prompt_index,
            binding.ready_deadline_ns,
        )
        if any(type(value) is not int for value in integer_fields):
            raise ValueError("binding numeric controls must be exact ints")
        if type(binding.is_chunk) is not bool:
            raise ValueError("binding is_chunk must be bool")
        if type(binding.allow_prompt_transition) is not bool:
            raise ValueError("binding allow_prompt_transition must be bool")
        expected = PreparedRowBinding(
            request_id=plan.request_ids[row],
            block_id=int(idx[row]),
            admission_generation=int(plan.admission_generation[row]),
            geometry_id=int(plan.geometry_id[row]),
            prompt_index=int(plan.prompt_index[row]),
            prior_prompt_index=binding.prior_prompt_index,
            allow_prompt_transition=binding.allow_prompt_transition,
            is_chunk=bool(plan.is_chunk[row]),
            ready_deadline_ns=int(plan.ready_deadline_ns[row]),
        )
        if binding != expected:
            raise ValueError(
                f"row {row} does not reproduce its atomically prepared "
                "request/block/control binding"
            )
        prompt_changed = binding.prior_prompt_index != binding.prompt_index
        if binding.allow_prompt_transition != prompt_changed:
            raise ValueError(
                "prompt transition authorization must exactly match a prior-to-current "
                "prompt change"
            )
        if binding.allow_prompt_transition and not binding.is_chunk:
            raise ValueError("only a newly minted CHUNK may change prompt")
    fresh_prefills = (~plan.has_initial_states_p).nonzero(as_tuple=True)[0]
    for local_row in fresh_prefills.tolist():
        binding = plan.bindings[plan.num_decodes + local_row]
        if binding.allow_prompt_transition:
            raise ValueError("a fresh admission has no persisted prompt to transition")
    if live.numel():
        if bool((live < 0).any()) or bool((live >= plan.num_pool_blocks).any()):
            raise ValueError("live block authority contains an out-of-range id")
        if bool((live == plan.null_block_id).any()):
            raise ValueError("live block authority must exclude the null block")
        if int(torch.unique(live).numel()) != int(live.numel()):
            raise ValueError("live block authority contains duplicates")
    return idx


def _gather_initialized_rows(
    pool: torch.Tensor,
    blocks: torch.Tensor,
    fresh_cpu: torch.Tensor,
) -> torch.Tensor:
    """Gather only continuing rows; fresh rows start as exact zero state."""
    rows = int(blocks.shape[0])
    continuing_cpu = (~fresh_cpu).nonzero(as_tuple=True)[0]
    if rows and int(continuing_cpu.numel()) == rows:
        # Keep the legacy zero-scratch layout contract explicit.
        return pool.index_select(0, blocks).contiguous()
    scratch = torch.zeros((rows, *pool.shape[1:]), dtype=pool.dtype, device=pool.device)
    if int(continuing_cpu.numel()):
        continuing = _h2d(continuing_cpu, pool.device)
        continuing_blocks = blocks.index_select(0, continuing)
        scratch.index_copy_(
            0,
            continuing,
            pool.index_select(0, continuing_blocks),
        )
    return scratch


def _counter_invariant_rows(
    counters: torch.Tensor,
    *,
    raw_tail_capacity: int,
    mel_tail_capacity: int,
    hop_length: int,
    n_fft: int,
    cadence_frames: torch.Tensor,
) -> torch.Tensor:
    """Return rows whose persisted frontend-control state is impossible."""
    from vllm.models.nemotron_asr.frontend import (
        CTR_COMMITTED_MEL_FRAMES,
        CTR_ENCODED_MEL_FRAMES,
        CTR_EXPECTED_CHUNK_SEQUENCE,
        CTR_FINALIZED,
        CTR_MEL_TAIL_LENGTH,
        CTR_RAW_TAIL_LENGTH,
        CTR_RAW_TAIL_ORIGIN,
        CTR_TOTAL_VALID_SAMPLES,
    )

    total = counters[:, CTR_TOTAL_VALID_SAMPLES]
    committed = counters[:, CTR_COMMITTED_MEL_FRAMES]
    encoded = counters[:, CTR_ENCODED_MEL_FRAMES]
    raw_origin = counters[:, CTR_RAW_TAIL_ORIGIN]
    raw_length = counters[:, CTR_RAW_TAIL_LENGTH]
    mel_length = counters[:, CTR_MEL_TAIL_LENGTH]
    finalized = counters[:, CTR_FINALIZED]
    expected_sequence = counters[:, CTR_EXPECTED_CHUNK_SEQUENCE]
    chunk_samples = cadence_frames * hop_length
    regular_chunks_from_total = total // chunk_samples
    regular_remainder = total % chunk_samples
    expected_regular_committed = torch.where(
        regular_chunks_from_total == 0,
        torch.zeros_like(committed),
        regular_chunks_from_total * cadence_frames - 7,
    )
    final_base_committed = torch.where(
        regular_chunks_from_total == 0,
        torch.zeros_like(committed),
        regular_chunks_from_total * cadence_frames - 7,
    )
    final_available = total // hop_length
    final_remaining = final_available - final_base_committed
    # A session-first final is the walk's first chunk: at most C-7
    # frames, at least one (mirrors ``advance_frontend``).
    first_final_count = torch.clamp(
        torch.minimum(final_remaining, cadence_frames - 7), min=0
    )
    final_count = torch.where(
        regular_chunks_from_total == 0,
        first_final_count,
        torch.where(
            final_remaining >= cadence_frames,
            cadence_frames,
            torch.where(
                final_remaining >= 8,
                final_remaining,
                torch.zeros_like(final_remaining),
            ),
        ),
    )
    expected_final_committed = final_base_committed + final_count
    finalized_bad = (finalized == 1) & (
        (expected_sequence != regular_chunks_from_total + 1)
        | (committed != expected_final_committed)
    )
    expected_mel_length = torch.minimum(
        committed,
        torch.full_like(committed, mel_tail_capacity),
    )
    bounded_committed = torch.minimum(
        committed.clamp(min=0),
        (total // hop_length).clamp(min=0),
    )
    expected_raw_origin = torch.clamp(
        bounded_committed * hop_length - n_fft // 2 - 1,
        min=0,
    )
    expected_raw_length = torch.clamp(total - expected_raw_origin, min=0)
    raw_retention_bad = torch.where(
        finalized == 1,
        (raw_origin != total) | (raw_length != 0),
        (raw_origin != expected_raw_origin) | (raw_length != expected_raw_length),
    )
    return (
        (counters < 0).any(dim=1)
        | (encoded != committed)
        | (committed > total // hop_length)
        | (
            (finalized == 0)
            & (
                (expected_sequence != regular_chunks_from_total)
                | (regular_remainder != 0)
                | (committed != expected_regular_committed)
            )
        )
        | finalized_bad
        | (raw_origin > total)
        | (raw_length > raw_tail_capacity)
        | (raw_origin + raw_length != total)
        | raw_retention_bad
        | (mel_length > mel_tail_capacity)
        | (mel_length != expected_mel_length)
        | ((finalized != 0) & (finalized != 1))
    )


def _validate_result_structure(
    result: AdvanceResult,
    *,
    rows: int,
    device: torch.device,
    capture: bool,
    expected_capture: (
        tuple[tuple[int, int, int], tuple[int, int, int], torch.dtype] | None
    ) = None,
) -> None:
    """Validate result/capture metadata without reading device values."""
    if (
        result.token_ids.device != device
        or result.token_ids.dtype != torch.int32
        or result.token_ids.dim() != 2
        or result.token_ids.shape[0] != rows
        or not result.token_ids.is_contiguous()
    ):
        raise ValueError(
            "AdvanceResult.token_ids must be device-local int32 shaped (B, K)"
        )
    if (
        result.token_lengths.device != device
        or result.token_lengths.dtype != torch.int32
        or tuple(result.token_lengths.shape) != (rows,)
        or not result.token_lengths.is_contiguous()
    ):
        raise ValueError(
            "AdvanceResult.token_lengths must be device-local int32 shaped (B,)"
        )
    if (
        result.row_status is None
        or result.row_status.device != device
        or result.row_status.dtype != torch.int32
        or tuple(result.row_status.shape) != (rows,)
        or not result.row_status.is_contiguous()
    ):
        raise ValueError(
            "AdvanceResult.row_status must be device-local int32 shaped (B,)"
        )
    captures = result.captures
    if not capture:
        if captures is not None:
            raise ValueError("capture-off transition returned capture tensors")
        return
    if captures is None:
        raise ValueError("capture enabled but AdvanceResult has no captures")
    if expected_capture is None:
        raise ValueError("capture validation requires host-derived geometry")
    expected_mel, expected_encoder, expected_encoder_dtype = expected_capture
    if (
        captures.frontend_mel.device != device
        or tuple(captures.frontend_mel.shape) != expected_mel
        or captures.frontend_mel.dtype != torch.float32
        or not captures.frontend_mel.is_contiguous()
        or captures.mel_lengths.device != device
        or captures.mel_lengths.dtype != torch.int64
        or tuple(captures.mel_lengths.shape) != (rows,)
        or not captures.mel_lengths.is_contiguous()
    ):
        raise ValueError("malformed frontend capture tensors")
    raw = captures.encoder_raw
    conditioned = captures.encoder_conditioned
    if (
        raw.device != device
        or conditioned.device != device
        or tuple(raw.shape) != expected_encoder
        or tuple(conditioned.shape) != expected_encoder
        or raw.dtype != expected_encoder_dtype
        or conditioned.dtype != expected_encoder_dtype
        or not raw.is_contiguous()
        or not conditioned.is_contiguous()
        or raw.shape[0] != rows
        or captures.encoder_lengths.device != device
        or captures.encoder_lengths.dtype != torch.int64
        or tuple(captures.encoder_lengths.shape) != (rows,)
        or not captures.encoder_lengths.is_contiguous()
    ):
        raise ValueError("malformed encoder capture tensors")


def _validated_result(
    result: AdvanceResult,
    *,
    incoming: torch.Tensor,
    blank_id: int,
    queue_capacity: int,
    geometry_bound: int,
    capture: bool,
    expected_capture: (
        tuple[tuple[int, int, int], tuple[int, int, int], torch.dtype] | None
    ) = None,
    expected_capture_lengths: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> AdvanceResult:
    """Compose device result predicates and sanitize every failed burst."""
    rows = int(incoming.shape[0])
    _validate_result_structure(
        result,
        rows=rows,
        device=incoming.device,
        capture=capture,
        expected_capture=expected_capture,
    )
    assert result.row_status is not None
    lengths = result.token_lengths.long()
    width = int(result.token_ids.shape[1])
    safe_for_mask = lengths.clamp(min=0, max=width)
    columns = torch.arange(width, device=incoming.device).unsqueeze(0)
    active = columns < safe_for_mask.unsqueeze(1)
    invalid_token = (
        ((result.token_ids < 0) | (result.token_ids >= blank_id)) & active
    ).any(dim=1)
    lost_incoming = (result.row_status & incoming) != incoming
    unknown = (result.row_status & ~((1 << 19) - 1)) != 0
    invalid = (
        (lengths < 0)
        | (lengths > width)
        | invalid_token
        | lost_incoming
        | unknown
        | ((incoming != 0) & (lengths != 0))
    )
    status = (
        incoming
        | result.row_status
        | (invalid.to(torch.int32) * ROW_STATUS_DECODE_INVARIANT)
    )
    legal_bound = min(queue_capacity, geometry_bound)
    overflow = lengths > legal_bound
    status |= overflow.to(torch.int32) * ROW_STATUS_BURST_OVERFLOW
    captures = result.captures
    if captures is not None:
        if expected_capture_lengths is None:
            raise ValueError(
                "capture validation requires independently derived logical lengths"
            )
        expected_mel_lengths, expected_encoder_lengths = expected_capture_lengths
        capture_bad = (
            (captures.mel_lengths < 0)
            | (captures.mel_lengths > captures.frontend_mel.shape[2])
            | (captures.encoder_lengths < 0)
            | (captures.encoder_lengths > captures.encoder_raw.shape[1])
            | (captures.mel_lengths != expected_mel_lengths)
            | (captures.encoder_lengths != expected_encoder_lengths)
        )
        status |= capture_bad.to(torch.int32) * ROW_STATUS_DECODE_INVARIANT
    sanitized_lengths = torch.where(
        status == 0,
        lengths.clamp(max=legal_bound),
        torch.zeros_like(lengths),
    ).to(torch.int32)
    return AdvanceResult(
        token_ids=result.token_ids,
        token_lengths=sanitized_lengths,
        row_status=status,
        captures=result.captures,
        frame_emission_counts=result.frame_emission_counts,
        frame_final_labels=result.frame_final_labels,
        frame_valid_lengths=result.frame_valid_lengths,
    )


@dataclass(frozen=True)
class ChunkBucketResult:
    """Scratch transition outputs; counter status is merged after endpointing."""

    batch: ChunkBatch
    result: AdvanceResult
    counter_invariant_bad: torch.Tensor
    counter_delta_bad: torch.Tensor


def advance_chunk_bucket(
    core: NemotronASRCore,
    env: torch.Tensor,
    state: SessionStateBatch,
    *,
    geometry: int,
    admitted_prompt: torch.Tensor,
    incoming_status: torch.Tensor,
    queue_capacity: int,
    decode_fn: RnntDecodeFn,
    encoder_transition: EncoderTransition | None = None,
    capture: bool = False,
    capture_geometry: Any | None = None,
) -> ChunkBucketResult:
    """Execute one already-gathered CHUNK bucket without resident writes.

    Endpointing stays with the outer transaction. The unconditional counter
    invariant and unmasked delta predicates are returned separately so the
    latter still use the clean-row mask AFTER endpoint status is composed.
    """
    from vllm.models.nemotron_asr.encoder_execution import encoder_geometry_shape
    from vllm.models.nemotron_asr.frontend import (
        CTR_COMMITTED_MEL_FRAMES,
        CTR_EXPECTED_CHUNK_SEQUENCE,
        CTR_FINALIZED,
        CTR_TOTAL_VALID_SAMPLES,
        MEL_TAIL_FRAMES,
    )
    from vllm.models.nemotron_asr.manifests import ADMISSION_EPOCH_MODULUS_MS, CADENCES
    from vllm.models.nemotron_asr.rnnt import MAX_SYMBOLS_PER_STEP

    g, capture_on, cap = geometry, capture, queue_capacity
    lookaheads = [right for _, right in CADENCES.values()]
    hop = int(core.featurizer.hop_length)
    cadence = 8 * (lookaheads[g] + 1)
    s_g = cadence * hop
    blank = int(core.blank_id)
    adm_prompt_b = admitted_prompt
    if capture_on and capture_geometry is None:
        capture_geometry = encoder_geometry_shape(core, g)
    old_counters = state.frontend_counters.clone()
    hdr = env[:, :ENVELOPE_HEADER_SLOTS]
    final_col = hdr[:, ENV_FINAL_TAIL]
    valid_col = hdr[:, ENV_VALID_SAMPLES]
    env_bad = hdr[:, ENV_VERSION] != ENVELOPE_VERSION
    env_bad |= (hdr != hdr.trunc()).any(dim=1)
    env_bad |= (final_col != 0) & (final_col != 1)
    env_bad |= valid_col > s_g
    env_bad |= (final_col == 0) & (valid_col != s_g)
    admission_col = hdr[:, ENV_ADMISSION_MS_MOD]
    env_bad |= (admission_col < 0) | (admission_col >= ADMISSION_EPOCH_MODULUS_MS)
    tail = env[:, ENVELOPE_HEADER_SLOTS + s_g :]
    if tail.shape[1]:
        env_bad |= (tail != 0).any(dim=1)
    prompt_mm = hdr[:, ENV_PROMPT_INDEX].long() != adm_prompt_b
    incoming = (
        incoming_status
        | env_bad.to(torch.int32) * ROW_STATUS_ENVELOPE
        | prompt_mm.to(torch.int32) * ROW_STATUS_PROMPT_MISMATCH
    )
    batch = ChunkBatch(
        samples=env[:, ENVELOPE_HEADER_SLOTS : ENVELOPE_HEADER_SLOTS + s_g],
        valid_samples=valid_col.long(),
        geometry_id=hdr[:, ENV_GEOMETRY_ID].long(),
        final_tail=final_col != 0,
        # Conditioning uses the ADMITTED authority (PORT-LID-003
        # as amended); the envelope prompt was cross-checked above.
        prompt_index=adm_prompt_b,
        chunk_sequence=hdr[:, ENV_CHUNK_SEQUENCE].long(),
    )
    result = advance_session(
        core,
        batch,
        state,
        geometry=g,
        decode_fn=decode_fn,
        encoder_transition=encoder_transition,
        capture=capture_on,
        row_status=incoming,
    )
    next_counters = state.frontend_counters
    if capture_on:
        committed_delta = (
            next_counters[:, CTR_COMMITTED_MEL_FRAMES]
            - old_counters[:, CTR_COMMITTED_MEL_FRAMES]
        ).clamp(min=0)
        session_first = batch.chunk_sequence == 0
        capture_mel_lengths = torch.where(
            committed_delta > 0,
            committed_delta
            + torch.where(
                session_first,
                torch.zeros_like(committed_delta),
                torch.full_like(committed_delta, MEL_TAIL_FRAMES),
            ),
            torch.zeros_like(committed_delta),
        )
        capture_drop = torch.where(
            session_first,
            torch.zeros_like(committed_delta),
            torch.full_like(committed_delta, PRE_ENCODE_DROP),
        )
        capture_encoder_lengths = torch.where(
            committed_delta > 0,
            torch.clamp(
                core.encoder.pre_encode.output_lengths(capture_mel_lengths)
                - capture_drop,
                min=0,
            ),
            torch.zeros_like(committed_delta),
        )
    result = _validated_result(
        result,
        incoming=incoming,
        blank_id=blank,
        queue_capacity=cap,
        geometry_bound=(lookaheads[g] + 1) * MAX_SYMBOLS_PER_STEP,
        capture=capture_on,
        expected_capture=(
            (
                int(env.shape[0]),
                int(state.mel_tail.shape[1]),
                capture_geometry.mel_width,
            ),
            (
                int(env.shape[0]),
                capture_geometry.out_width,
                int(state.channel[0].shape[2]),
            ),
            state.channel[0].dtype,
        )
        if capture_geometry is not None
        else None,
        expected_capture_lengths=(
            capture_mel_lengths,
            capture_encoder_lengths,
        )
        if capture_on
        else None,
    )
    counter_invariant_bad = _counter_invariant_rows(
        state.frontend_counters,
        raw_tail_capacity=int(state.raw_tail.shape[1]),
        mel_tail_capacity=int(state.mel_tail.shape[2]),
        hop_length=hop,
        n_fft=int(core.featurizer.n_fft),
        cadence_frames=torch.full_like(
            state.frontend_counters[:, CTR_EXPECTED_CHUNK_SEQUENCE], cadence
        ),
    )
    counter_delta_bad = (
        (
            state.frontend_counters[:, CTR_EXPECTED_CHUNK_SEQUENCE]
            != old_counters[:, CTR_EXPECTED_CHUNK_SEQUENCE] + 1
        )
        | (
            state.frontend_counters[:, CTR_TOTAL_VALID_SAMPLES]
            - old_counters[:, CTR_TOTAL_VALID_SAMPLES]
            != batch.valid_samples
        )
        | (
            state.frontend_counters[:, CTR_FINALIZED]
            != batch.final_tail.to(torch.int64)
        )
    )
    return ChunkBucketResult(batch, result, counter_invariant_bad, counter_delta_bad)


@dataclass(frozen=True)
class _ScatterDescriptor:
    pool: torch.Tensor
    scratch: torch.Tensor
    blocks: torch.Tensor
    row_status: torch.Tensor


def _resident_pool_sequence(
    *,
    channel_pools: Sequence[torch.Tensor],
    time_pools: Sequence[torch.Tensor],
    len_pools: Sequence[torch.Tensor],
    h_pool: torch.Tensor,
    c_pool: torch.Tensor,
    queue_pool: torch.Tensor,
    book_pool: torch.Tensor,
    frontend_raw_pool: torch.Tensor,
    frontend_mel_pool: torch.Tensor,
    frontend_counter_pool: torch.Tensor,
    endpoint_history_pool: torch.Tensor | None = None,
    endpoint_book_pool: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Return the complete resident-pool inventory in commit order."""
    pools: tuple[torch.Tensor, ...] = (
        *channel_pools,
        *time_pools,
        *len_pools,
        h_pool,
        c_pool,
        queue_pool,
        book_pool,
        frontend_raw_pool,
        frontend_mel_pool,
        frontend_counter_pool,
    )
    if endpoint_history_pool is not None:
        pools += (endpoint_history_pool,)
    if endpoint_book_pool is not None:
        pools += (endpoint_book_pool,)
    return pools


# @spec PORT-PERF-001, PORT-STATE-008
def warmup_advance_model_rows_scatter(
    *,
    channel_pools: Sequence[torch.Tensor],
    time_pools: Sequence[torch.Tensor],
    len_pools: Sequence[torch.Tensor],
    h_pool: torch.Tensor,
    c_pool: torch.Tensor,
    queue_pool: torch.Tensor,
    book_pool: torch.Tensor,
    frontend_raw_pool: torch.Tensor,
    frontend_mel_pool: torch.Tensor,
    frontend_counter_pool: torch.Tensor,
    endpoint_history_pool: torch.Tensor | None = None,
    endpoint_book_pool: torch.Tensor | None = None,
) -> None:
    """Warm every fixed-shape commit specialization before admission.

    Task 5 calls this once after the actual resident pools are allocated and
    before the first session is admitted. Transaction scratch is contiguous,
    so each one-row scratch below reproduces the runtime source stride while
    preserving the destination pool's actual (possibly padded) block stride.
    Runtime validation then fails closed if a later pool/layout was omitted.
    """
    pools = _resident_pool_sequence(
        channel_pools=channel_pools,
        time_pools=time_pools,
        len_pools=len_pools,
        h_pool=h_pool,
        c_pool=c_pool,
        queue_pool=queue_pool,
        book_pool=book_pool,
        frontend_raw_pool=frontend_raw_pool,
        frontend_mel_pool=frontend_mel_pool,
        frontend_counter_pool=frontend_counter_pool,
        endpoint_history_pool=endpoint_history_pool,
        endpoint_book_pool=endpoint_book_pool,
    )
    for pool in pools:
        if pool.dim() < 1:
            raise ValueError("every resident pool must have a block dimension")
        scratch = torch.empty(
            (1, *pool.shape[1:]),
            dtype=pool.dtype,
            device=pool.device,
        )
        warmup_masked_page_scatter(pool, scratch)


# ---- PORT-OBS-008 batch-size sub-stat: consume-once extraction hook ----
#: Module-level (not per-instance, mirroring ``plan.PlanContextSlot``'s
#: stage/consume shape but never raising on an empty/absent stage):
#: ``advance_model_rows`` overwrites this every call, so a stranded
#: earlier list can never leak into a later transaction's read.
#:
#: Queued EARS amendment (Phase-6 round 2, Q3, lead-authorized
#: spec-tightening): OBS-008 now reads "one (cadence_ms, rows) entry per
#: executed nonempty CHUNK geometry bucket, cadence resolved at
#: recording from the geometry authority" — cadence is resolved HERE
#: (the manifest-table authority is this model package's own), not by
#: any downstream consumer, so the metrics/orchestrator layers stay
#: model-agnostic (never importing a geometry->cadence table of their
#: own).
_batch_stats_slot: list[tuple[str, int]] | None = None


def _stage_batch_stats(stats: list[tuple[str, int]]) -> None:
    """Record one transaction's executed-bucket stats (PORT-OBS-008)."""
    global _batch_stats_slot
    _batch_stats_slot = stats


def consume_batch_stats() -> list[tuple[str, int]] | None:
    """Drain this call's per-executed-bucket ``(cadence_ms, rows)`` list.

    PORT-OBS-008/009 (amended): ``advance_model_rows`` records one entry
    per executed nonempty CHUNK geometry bucket — bounded by the five
    admitted geometries, CPU-only, no device synchronization —
    unconditionally (recording stands even when export is disabled),
    cadence resolved from the geometry authority (``manifests.CADENCES``)
    at recording time, exposed through this consume-once hook, drained
    by the runner once per execution (mirroring ``plan.PlanContextSlot``'s
    stage/consume shape). ``None`` means not collecting (nothing staged
    since the last drain); ``[]`` means a transaction that executed no
    nonempty CHUNK bucket; downstream consumers skip both.
    """
    global _batch_stats_slot
    stats = _batch_stats_slot
    _batch_stats_slot = None
    return stats


# @spec PORT-ADV-003, PORT-ADV-004, PORT-HOOK-001, PORT-LID-003,
# @spec PORT-PERF-001, PORT-PERF-004, PORT-STATE-007, PORT-STATE-008
def advance_model_rows(
    core: NemotronASRCore,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    plan: RowPlan,
    *,
    channel_pools: list[torch.Tensor],
    time_pools: list[torch.Tensor],
    len_pools: list[torch.Tensor],
    h_pool: torch.Tensor,
    c_pool: torch.Tensor,
    queue_pool: torch.Tensor,
    book_pool: torch.Tensor,
    frontend_raw_pool: torch.Tensor,
    frontend_mel_pool: torch.Tensor,
    frontend_counter_pool: torch.Tensor,
    endpoint_history_pool: torch.Tensor | None = None,
    endpoint_book_pool: torch.Tensor | None = None,
    eou_token_id: int | None = None,
    adapter: EmissionAdapter,
    decode_resolver: DecodeResolver,
    encoder_transition: EncoderTransition | None = None,
    bucket_transition: Callable[..., ChunkBucketResult] | None = None,
    placeholder_id: int,
    park_id: int,
    commit_sink: CommitSink | None = None,
    capture: bool = False,
    graph_covers_decode: bool = False,
    memory_profile: bool = False,
    staging: HostStaging | None = None,
    native_burst_handoff: Any | None = None,
) -> torch.Tensor:
    """The ONE shared outer transaction (PORT-ADV-003).

    Fixed order (design §Phase-6c transaction seams):

    1. compose decode-then-prefill indices from ``plan`` and validate
       every structural row/page property HOST-SIDE before any page
       read (PORT-STATE-007): row count vs ``input_ids``/
       ``inputs_embeds``; non-null (``!= null_block_id``), live,
       in-range (``< num_pool_blocks``) indices; uniqueness;
       speculative-column exclusion (extra decode columns);
       real/padding-row agreement — then upload the validated indices
       asynchronously for the device gathers;
    2. reject unqualified graph coverage, form CHUNK buckets from CPU
       plan authority, order them by earliest deadline, and resolve
       every eager decode before the first resident read;
    3. initialize fresh rows from plan authority without reading
       recycled pages, then gather only continuing rows' required
       session state; evaluate replay echoes
       and the host-role cross-check ON DEVICE into the transaction's
       ``ROW_STATUS_*`` bits (echo mismatch is row-local after
       trusted preflight, PORT-DEC-007 as amended); classify each
       real row CHUNK / REPLAY / FLUSH;
    4. group CHUNK rows by execution profile + immutable geometry
       (host authority: ``plan.is_chunk`` + ``plan.geometry_id``) and
       gather their full frontend/encoder/predictor scratch; valid
       length stays a per-row tensor, not a batch-key component;
       and run :func:`advance_session` with its pre-resolved callable;
    5. validate complete decode results, merge and project them,
       prepare optional candidate records, and validate the complete
       all-bucket scatter plan before one composite reservation;
    6. issue fixed-shape predicated page scatters (failed rows issue
       no store), then perform the reservation's one no-fail combined
       status/candidate stage.

    Suppression (PORT-STATE-008): any structural defect fails the
    whole call before a read; an expected per-row defect resolves to
    a device status bit and a masked park-only no-op. An unexpected
    Python/CUDA compute failure aborts the whole call before scatter;
    a catastrophic device failure during the prevalidated commit is
    worker-fatal, never bucket/row suppression or rollback. No row
    scatters unless its emission result can be returned. Transition
    kernels never mutate resident pages directly.

    ``bucket_transition`` is an optional startup-bound experimental executor
    for already-gathered CHUNK scratch. The default uses the same canonical
    tensor helper eagerly. It never owns resident gathers, endpoint processing,
    adapter execution, reservations, or scatter.

    Host staging (PORT-PERF-001): ``staging`` is optional reusable
    pinned H2D buffering (:class:`HostStaging`) for the once-per-call
    composed-index, fresh-row, and control vectors; ``None`` keeps
    the un-pooled per-call :func:`_h2d` path (probes/CPU). The
    per-bucket CHUNK row-position vector, staged once per resolved
    bucket inside step 4's loop, also rides ``staging`` when given —
    through :meth:`HostStaging.stage_bucket`, which keeps one
    dedicated buffer PER GEOMETRY. Since each geometry resolves at
    most once per call, no bucket's in-flight copy is overwritten by a
    later bucket, so this is allocation-free without the overwrite
    race a single shared bucket slot would carry; it falls back to
    ``_h2d`` only when ``staging`` is ``None``.

    Returns:
        The runner's row output as projected by ``adapter`` (e.g. the
        ``(N, H)`` decision-carrier for MRV1).

    Raises:
        ValueError: a structural mapping defect, a missing/invalid
            resolver, or a trusted-identity projection-shape defect —
            always before any resident mutation.

    """
    from vllm.models.nemotron_asr.encoder_execution import (
        encoder_geometry_shape,
    )
    from vllm.models.nemotron_asr.frontend import (
        CTR_EXPECTED_CHUNK_SEQUENCE,
        CTR_FINALIZED,
    )
    from vllm.models.nemotron_asr.manifests import (
        CADENCES,
    )
    from vllm.models.nemotron_asr.rnnt import (
        BOOK_EXPECTED_LABEL,
        BOOK_GEOMETRY,
        BOOK_PENDING_ECHO,
        MAX_SYMBOLS_PER_STEP,
        QUEUE_HEAD,
        QUEUE_LAST_LABEL,
        QUEUE_LEN,
        QUEUE_PROMPT,
    )

    if not callable(decode_resolver):
        raise ValueError(
            "advance_model_rows requires a decode resolver: dispatch is never a "
            "hardcoded default (PORT-DEC-008)"
        )
    if input_ids.dim() != 1 or inputs_embeds.dim() != 2:
        raise ValueError("input_ids/inputs_embeds must be rank one/two")
    if input_ids.dtype != torch.int64 or inputs_embeds.dtype != torch.float32:
        raise ValueError("row ids must be int64 and raw envelopes must be fp32")
    if input_ids.device != inputs_embeds.device:
        raise ValueError("input ids and envelopes must share a device")
    n_real = plan.num_decodes + plan.num_prefills
    hidden = int(inputs_embeds.shape[1])
    device = inputs_embeds.device
    idx_cpu = _structural_preflight(
        plan, int(input_ids.shape[0]), int(inputs_embeds.shape[0])
    )
    has_chunk = bool(plan.is_chunk.any())
    if graph_covers_decode and has_chunk and plan.execution_tier <= 0:
        raise ValueError(
            "regional graph decode requires a positive RowPlan execution tier"
        )
    if not graph_covers_decode and plan.execution_tier != 0:
        raise ValueError("eager execution requires plan.execution_tier == 0")
    if memory_profile and graph_covers_decode:
        raise ValueError("memory profile cannot claim graph coverage")
    if memory_profile and capture:
        raise ValueError("memory profile cannot publish capture records")
    if capture and commit_sink is None:
        raise ValueError("capture requires a composite commit sink")
    lookaheads = [right for (_, right) in CADENCES.values()]
    hop = int(core.featurizer.hop_length)
    num_prompts = int(core.lid.num_prompts)
    if queue_pool.dim() != 2 or book_pool.dim() != 2:
        raise ValueError("queue and book pools must be rank two")
    cap = int(queue_pool.shape[1])
    if cap <= 0:
        raise ValueError("replay queue capacity must be positive")
    if book_pool.shape[1] != 7:
        raise ValueError("session book must match the seven-slot manifest")
    blank = int(core.blank_id)
    if n_real and (
        bool((plan.geometry_id < 0).any())
        or bool((plan.geometry_id >= len(lookaheads)).any())
    ):
        raise ValueError("plan.geometry_id outside the admitted set")
    if n_real and (
        bool((plan.prompt_index < 0).any())
        or bool((plan.prompt_index >= num_prompts).any())
    ):
        raise ValueError(
            "plan.prompt_index outside the prompt dictionary — the admitted authority "
            "is host-validated at admission"
        )
    if any(
        binding.prior_prompt_index < 0 or binding.prior_prompt_index >= num_prompts
        for binding in plan.bindings
    ):
        raise ValueError("binding prior prompt outside the prompt dictionary")
    required_hidden = 1  # every runner row must carry one decision id
    for geometry, lookahead in enumerate(lookaheads):
        if bool((plan.is_chunk & (plan.geometry_id == geometry)).any()):
            required_hidden = max(
                required_hidden,
                ENVELOPE_HEADER_SLOTS + 8 * (lookahead + 1) * hop,
            )
    if hidden < required_hidden:
        raise ValueError(
            f"carrier width {hidden} is smaller than the selected rows require "
            f"({required_hidden})"
        )

    pools = _resident_pool_sequence(
        channel_pools=channel_pools,
        time_pools=time_pools,
        len_pools=len_pools,
        h_pool=h_pool,
        c_pool=c_pool,
        queue_pool=queue_pool,
        book_pool=book_pool,
        frontend_raw_pool=frontend_raw_pool,
        frontend_mel_pool=frontend_mel_pool,
        frontend_counter_pool=frontend_counter_pool,
        endpoint_history_pool=endpoint_history_pool,
        endpoint_book_pool=endpoint_book_pool,
    )
    endpoint_enabled = any(
        value is not None
        for value in (
            endpoint_history_pool,
            endpoint_book_pool,
            eou_token_id,
        )
    )
    if endpoint_enabled:
        if (
            endpoint_history_pool is None
            or endpoint_book_pool is None
            or eou_token_id is None
        ):
            raise ValueError("endpoint execution requires history, book, and EOU id")
        if (
            endpoint_history_pool.dtype != torch.int32
            or endpoint_history_pool.dim() != 2
            or endpoint_book_pool.dtype != torch.int32
            or endpoint_book_pool.dim() != 2
            or endpoint_book_pool.shape[1] != 6
        ):
            raise ValueError("endpoint resident pools disagree with the manifest")
    if any(pool.dim() < 1 for pool in pools):
        raise ValueError("every resident pool must have a block dimension")
    if any(int(pool.shape[0]) != plan.num_pool_blocks for pool in pools):
        raise ValueError("resident pool extent disagrees with RowPlan authority")
    if any(pool.device != device for pool in pools):
        raise ValueError("every resident pool must share the compute device")
    if queue_pool.dtype != torch.int32 or book_pool.dtype != torch.int32:
        raise ValueError("queue and book pools must use int32")
    if frontend_counter_pool.dtype != torch.int64:
        raise ValueError("frontend counters must use int64")
    if (
        any(pool.dtype != torch.float32 for pool in channel_pools)
        or any(pool.dtype != torch.float32 for pool in time_pools)
        or any(pool.dtype != torch.int32 for pool in len_pools)
        or h_pool.dtype != torch.float32
        or c_pool.dtype != torch.float32
        or frontend_raw_pool.dtype != torch.float32
        or frontend_mel_pool.dtype != torch.float32
    ):
        raise ValueError("resident pool dtypes disagree with the state manifest")
    if (
        frontend_raw_pool.dim() != 2
        or frontend_mel_pool.dim() != 3
        or frontend_counter_pool.dim() != 2
        or frontend_counter_pool.shape[1] != 8
        or h_pool.dim() != 3
        or c_pool.dim() != 3
        or tuple(h_pool.shape) != tuple(c_pool.shape)
        or h_pool.dtype != c_pool.dtype
    ):
        raise ValueError("frontend/predictor resident pool geometry is invalid")
    layer_count = len(core.encoder.layers)
    if not (len(channel_pools) == len(time_pools) == len(len_pools) == layer_count):
        raise ValueError("encoder state pool counts disagree with model layers")
    if (
        any(pool.dim() != 3 for pool in channel_pools)
        or any(pool.dim() != 3 for pool in time_pools)
        or any(pool.dim() != 2 or pool.shape[1] != 1 for pool in len_pools)
    ):
        raise ValueError("encoder resident pool geometry is invalid")

    # Resolve ALL buckets from CPU authority before the first resident read.
    unresolved: list[tuple[int, torch.Tensor, int, int]] = []
    for geometry in range(len(lookaheads)):
        positions = (plan.is_chunk & (plan.geometry_id == geometry)).nonzero(
            as_tuple=True
        )[0]
        if int(positions.numel()):
            deadline = int(plan.ready_deadline_ns.index_select(0, positions).min())
            unresolved.append((geometry, positions, deadline, int(positions[0])))
    unresolved.sort(key=lambda item: (item[2], item[3]))
    ready = len(unresolved)
    bucket_pos: list[tuple[int, torch.Tensor, ResolvedDecode]] = []
    bucket_transitions: dict[int, Callable[..., ChunkBucketResult]] = {}
    resolve_bucket = getattr(bucket_transition, "resolve", None)
    for geometry, positions, _, _ in unresolved:
        live = int(positions.numel())
        if (lookaheads[geometry] + 1) * MAX_SYMBOLS_PER_STEP > cap:
            raise ValueError(
                f"geometry {geometry}'s legal burst bound exceeds the replay queue "
                f"capacity {cap}"
            )
        resolved = decode_resolver(
            DecodeRequest(
                geometry=geometry,
                execution_batch_size=live,
                graph_covers_decode=graph_covers_decode,
                ready_decode_buckets=ready,
                execution_tier_limit=plan.execution_tier,
                memory_profile=memory_profile,
            )
        )
        if not isinstance(resolved, ResolvedDecode) or not callable(resolved.decode_fn):
            raise ValueError("decode resolver returned an invalid binding")
        if callable(resolve_bucket):
            selected = resolve_bucket(
                geometry=geometry,
                population=live,
                decode_fn=resolved.decode_fn,
                encoder_transition=encoder_transition,
                capture=capture,
            )
            if not callable(selected):
                raise ValueError("CHUNK resolver returned an invalid transition")
            bucket_transitions[geometry] = selected
        bucket_pos.append((geometry, positions, resolved))

    # PORT-OBS-008 (amended): one (cadence_ms, rows) entry per executed
    # nonempty CHUNK geometry bucket, unconditionally — CPU-only (``live``
    # above is already a plain int, no device sync; ``cadence_labels`` is
    # a python list index, not a tensor read), independent of ``capture``
    # or export being enabled. ``bucket_pos`` is exactly the executed set
    # (the loop below processes every entry; nothing filters it further).
    # Cadence is resolved HERE, at the manifest-table authority, so every
    # downstream consumer (metrics, orchestrator) stays model-agnostic.
    cadence_labels = list(CADENCES)
    _stage_batch_stats(
        [
            (cadence_labels[geometry].removesuffix("ms"), int(positions.numel()))
            for geometry, positions, _ in bucket_pos
        ]
    )

    # ---- small continuing-row gather + metadata-only fresh init ----
    def _stage(slot_name: str, cpu_tensor: torch.Tensor) -> torch.Tensor:
        """Route one once-per-call vector through ``staging`` when
        given, else the un-pooled :func:`_h2d` path."""
        if staging is not None:
            return staging.stage(slot_name, cpu_tensor, device)
        return _h2d(cpu_tensor, device)

    didx = _stage("composed_indices", idx_cpu)
    fresh_local = (~plan.has_initial_states_p).nonzero(as_tuple=True)[0]
    fresh_mask_cpu = torch.zeros(n_real, dtype=torch.bool)
    if int(fresh_local.numel()):
        fresh_pos_cpu = plan.num_decodes + fresh_local
        fresh_mask_cpu[fresh_pos_cpu] = True
    book = _gather_initialized_rows(book_pool, didx, fresh_mask_cpu)
    queue = _gather_initialized_rows(queue_pool, didx, fresh_mask_cpu)
    counters_small = _gather_initialized_rows(
        frontend_counter_pool, didx, fresh_mask_cpu
    )
    endpoint_history = (
        _gather_initialized_rows(endpoint_history_pool, didx, fresh_mask_cpu)
        if endpoint_enabled
        else None
    )
    endpoint_book = (
        _gather_initialized_rows(endpoint_book_pool, didx, fresh_mask_cpu)
        if endpoint_enabled
        else None
    )
    endpoint_mode_cpu = (
        plan.endpoint_mode
        if plan.endpoint_mode.numel()
        else torch.zeros(n_real, dtype=torch.int64)
    )
    endpoint_threshold_cpu = (
        plan.endpoint_threshold_frames
        if plan.endpoint_threshold_frames.numel()
        else torch.zeros(n_real, dtype=torch.int64)
    )
    endpoint_residue_cpu = (
        plan.endpoint_residue_frames
        if plan.endpoint_residue_frames.numel()
        else torch.zeros(n_real, dtype=torch.int64)
    )
    if (
        bool(((endpoint_mode_cpu != 0) & (endpoint_mode_cpu != 1)).any())
        or bool((endpoint_threshold_cpu < 0).any())
        or bool((endpoint_residue_cpu < 0).any())
    ):
        raise ValueError("endpoint policy is outside the admitted vocabulary")
    if int(fresh_local.numel()):
        finit = torch.zeros(int(fresh_local.numel()), book.shape[1], dtype=torch.int64)
        finit[:, QUEUE_LAST_LABEL] = blank
        finit[:, QUEUE_PROMPT] = plan.prompt_index.index_select(0, fresh_pos_cpu)
        finit[:, BOOK_GEOMETRY] = plan.geometry_id.index_select(0, fresh_pos_cpu)
        fresh_dev = _stage("fresh_positions", fresh_pos_cpu)
        book.index_copy_(0, fresh_dev, _stage("fresh_init_book", finit).to(book.dtype))

    # ---- device role/protocol/invariant composition ----
    is_chunk_dev = _stage("is_chunk", plan.is_chunk)
    plan_geom_dev = _stage("geometry", plan.geometry_id)
    admitted_prompt_dev = _stage("admitted_prompt", plan.prompt_index)
    prior_prompt_dev = _stage(
        "prior_prompt",
        torch.tensor(
            [binding.prior_prompt_index for binding in plan.bindings],
            dtype=torch.int64,
        ),
    )
    ids_dev = input_ids.long()
    head_all = book[:, QUEUE_HEAD].long()
    len_all = book[:, QUEUE_LEN].long()
    pend_col = book[:, BOOK_PENDING_ECHO]
    pending = pend_col == 1
    remaining = len_all - head_all
    last = book[:, QUEUE_LAST_LABEL].long()
    expected = book[:, BOOK_EXPECTED_LABEL].long()
    finalized_col = counters_small[:, CTR_FINALIZED]
    finalized = finalized_col == 1
    slot = torch.arange(cap, device=device).unsqueeze(0)
    queued_value_valid = (queue >= 0) & (queue < blank)
    if endpoint_enabled:
        assert eou_token_id is not None
        queued_value_valid |= queue == int(eou_token_id)
    queued_bad = ((~queued_value_valid) & (slot < len_all.unsqueeze(1))).any(dim=1)
    prior_emitted = (
        queue.gather(
            1,
            (head_all - 1).clamp(min=0, max=max(cap - 1, 0)).unsqueeze(1),
        )
        .squeeze(1)
        .long()
    )
    queue_last = (
        queue.gather(
            1,
            (len_all - 1).clamp(min=0, max=max(cap - 1, 0)).unsqueeze(1),
        )
        .squeeze(1)
        .long()
    )
    queue_tail_is_eou = torch.zeros_like(pending)
    expected_is_eou = torch.zeros_like(pending)
    if endpoint_enabled:
        assert eou_token_id is not None
        queue_tail_is_eou = (len_all > 0) & (queue_last == int(eou_token_id))
        expected_is_eou = expected == int(eou_token_id)
    book_bad = (
        (head_all < 0)
        | (head_all > len_all)
        | (len_all > cap)
        | ((pend_col != 0) & (pend_col != 1))
        | (last < 0)
        | (last > blank)
        | (expected < 0)
        | ((expected > blank) & (~expected_is_eou))
        | (pending & (expected >= blank) & (~expected_is_eou))
        | queued_bad
        | (pending & (head_all < 1))
        | (pending & (expected != prior_emitted))
        | ((len_all > 0) & (~queue_tail_is_eou) & (last != queue_last))
        | ((remaining > 0) & (~pending))
    )
    counter_bad = _counter_invariant_rows(
        counters_small,
        raw_tail_capacity=int(frontend_raw_pool.shape[1]),
        mel_tail_capacity=int(frontend_mel_pool.shape[2]),
        hop_length=hop,
        n_fft=int(core.featurizer.n_fft),
        # Build the per-geometry cadence table on the HOST and index it
        # by the CPU RowPlan geometry authority, THEN stage the per-row
        # result to the device — never ``torch.tensor(list,
        # device=cuda)``, which is a synchronizing host→device
        # construction on the per-turn hot path (the full-turn probe's
        # sync tripwire named exactly this call). Mirrors how
        # ``prior_prompt`` etc. are staged: CPU build, pooled
        # non-blocking H2D.
        cadence_frames=_stage(
            "cadence_frames",
            torch.tensor(
                [8 * (lookahead + 1) for lookahead in lookaheads],
                dtype=torch.int64,
            ).index_select(0, plan.geometry_id),
        ),
    )
    status = torch.zeros(n_real, dtype=torch.int32, device=device)
    status |= (is_chunk_dev != (ids_dev == placeholder_id)).to(
        torch.int32
    ) * ROW_STATUS_ROLE_MISMATCH
    status |= ((~is_chunk_dev) & pending & (ids_dev != expected)).to(
        torch.int32
    ) * ROW_STATUS_ECHO_MISMATCH
    status |= (is_chunk_dev & (pending | (remaining > 0))).to(
        torch.int32
    ) * ROW_STATUS_QUEUE_NOT_DRAINED
    # The AR park echo: under async scheduling the engine's in-flight
    # frame legally feeds an emitted park token back as the next input
    # (the label twin of this row is ROLE_REPLAY, armed at commit). A
    # non-chunk park-token row on a drained, unarmed, live session is
    # therefore a distinct asynchronous park echo — validated downstream
    # as emit-park-change-nothing — not FLUSH or a protocol violation.
    park_echo = (
        (~is_chunk_dev)
        & (~pending)
        & (remaining == 0)
        & (~finalized)
        & (ids_dev == park_id)
    )
    forced_eou = torch.zeros_like(park_echo)
    if endpoint_enabled:
        assert eou_token_id is not None
        forced_eou = (
            (~is_chunk_dev)
            & (~pending)
            & (remaining == 0)
            & (~finalized)
            & (ids_dev == int(eou_token_id))
        )
    status |= (
        (~is_chunk_dev)
        & (~pending)
        & ((remaining > 0) | (~finalized))
        & (~park_echo)
        & (~forced_eou)
    ).to(torch.int32) * ROW_STATUS_SESSION_PROTOCOL
    status |= (book[:, BOOK_GEOMETRY].long() != plan_geom_dev).to(
        torch.int32
    ) * ROW_STATUS_BOOK_IDENTITY
    status |= (book[:, QUEUE_PROMPT].long() != prior_prompt_dev).to(
        torch.int32
    ) * ROW_STATUS_BOOK_IDENTITY
    status |= (book_bad | counter_bad).to(torch.int32) * ROW_STATUS_BOOK_INVARIANT
    # Safe substitution (PORT-ADV-004 as amended): a corrupt book's
    # last label must never reach the predictor embedding.
    safe_last = torch.where(
        book_bad | counter_bad,
        torch.full_like(last, blank),
        last.clamp(0, blank),
    )
    roles = torch.full((n_real,), ROLE_FLUSH, dtype=torch.long, device=device)
    roles = torch.where(
        (~is_chunk_dev) & pending,
        torch.full_like(roles, ROLE_REPLAY),
        roles,
    )
    roles = torch.where(
        forced_eou,
        torch.full_like(roles, ROLE_EOU),
        roles,
    )
    roles = torch.where(is_chunk_dev, torch.full_like(roles, ROLE_CHUNK), roles)

    # ---- deadline-ordered buckets → fresh-aware gather → transition ----
    capture_on = capture
    executed: list[dict[str, Any]] = []
    for g, pos_t, resolved in bucket_pos:
        # Routed through ``staging`` by GEOMETRY: each geometry resolves
        # at most once per call, so its dedicated per-geometry buffer's
        # non-blocking copy is never overwritten by a later bucket in
        # this call — allocation-free without the overwrite race a
        # single shared bucket slot would carry. Falls back to the
        # un-pooled _h2d path (probes/CPU) when no staging is given.
        rows_dev = (
            staging.stage_bucket(g, pos_t, device)
            if staging is not None
            else _h2d(pos_t, device)
        )
        blocks_dev = didx.index_select(0, rows_dev)
        fresh_bucket = fresh_mask_cpu.index_select(0, pos_t)
        capture_geometry = encoder_geometry_shape(core, g) if capture_on else None
        cadence = (
            capture_geometry.cadence_frames
            if capture_geometry is not None
            else 8 * (lookaheads[g] + 1)
        )
        s_g = cadence * hop
        if ENVELOPE_HEADER_SLOTS + s_g > hidden:
            raise ValueError(
                f"carrier width {hidden} cannot hold geometry {g}'s {s_g}-sample "
                "cadence"
            )
        env = inputs_embeds.index_select(0, rows_dev)
        adm_prompt_b = admitted_prompt_dev.index_select(0, rows_dev)
        incoming = status.index_select(0, rows_dev)
        state = SessionStateBatch(
            raw_tail=_gather_initialized_rows(
                frontend_raw_pool, blocks_dev, fresh_bucket
            ),
            mel_tail=_gather_initialized_rows(
                frontend_mel_pool, blocks_dev, fresh_bucket
            ),
            frontend_counters=counters_small.index_select(0, rows_dev),
            channel=[
                _gather_initialized_rows(pool, blocks_dev, fresh_bucket)
                for pool in channel_pools
            ],
            window_valid=[
                _gather_initialized_rows(pool, blocks_dev, fresh_bucket)
                for pool in len_pools
            ],
            time=[
                _gather_initialized_rows(pool, blocks_dev, fresh_bucket)
                for pool in time_pools
            ],
            h=_gather_initialized_rows(h_pool, blocks_dev, fresh_bucket),
            c=_gather_initialized_rows(c_pool, blocks_dev, fresh_bucket),
            last_label=safe_last.index_select(0, rows_dev),
        )
        transition = bucket_transitions.get(
            g, advance_chunk_bucket if bucket_transition is None else bucket_transition
        )
        bucket = transition(
            core,
            env,
            state,
            geometry=g,
            admitted_prompt=adm_prompt_b,
            incoming_status=incoming,
            queue_capacity=cap,
            decode_fn=resolved.decode_fn,
            encoder_transition=encoder_transition,
            capture=capture_on,
            capture_geometry=capture_geometry,
        )
        if not isinstance(bucket, ChunkBucketResult):
            raise TypeError("bucket transition must return ChunkBucketResult")
        batch, result = bucket.batch, bucket.result
        # Keep structural rejection on every invocation, including graph
        # replay; capture-time metadata validation alone is insufficient.
        _validate_result_structure(
            result,
            rows=int(rows_dev.shape[0]),
            device=device,
            capture=capture_on,
            expected_capture=(
                (
                    int(rows_dev.shape[0]),
                    int(state.mel_tail.shape[1]),
                    capture_geometry.mel_width,
                ),
                (
                    int(rows_dev.shape[0]),
                    capture_geometry.out_width,
                    int(state.channel[0].shape[2]),
                ),
                state.channel[0].dtype,
            )
            if capture_geometry is not None
            else None,
        )
        for predicate in (bucket.counter_invariant_bad, bucket.counter_delta_bad):
            if (
                predicate.shape != (rows_dev.shape[0],)
                or predicate.dtype != torch.bool
                or predicate.device != device
            ):
                raise ValueError(
                    "bucket counter predicates differ from the row contract"
                )
        endpoint_transition = None
        if endpoint_enabled:
            from vllm.models.nemotron_asr.endpointing import (
                observe_chunk_tensors,
            )

            if (
                result.frame_emission_counts is None
                or result.frame_valid_lengths is None
            ):
                raise ValueError(
                    "selected decode arm does not expose frame-aligned endpoint symbols"
                )
            assert endpoint_history is not None
            assert endpoint_book is not None
            assert eou_token_id is not None
            endpoint_transition = observe_chunk_tensors(
                history=endpoint_history.index_select(0, rows_dev),
                book=endpoint_book.index_select(0, rows_dev),
                frame_emission_counts=result.frame_emission_counts,
                valid_frame_lengths=result.frame_valid_lengths,
                token_ids=result.token_ids,
                token_lengths=result.token_lengths,
                final_tail=batch.final_tail.to(device),
                mode=_h2d(endpoint_mode_cpu.index_select(0, pos_t), device),
                threshold_frames=_h2d(
                    endpoint_threshold_cpu.index_select(0, pos_t), device
                ),
                residue_frames=_h2d(
                    endpoint_residue_cpu.index_select(0, pos_t), device
                ),
                eou_token_id=int(eou_token_id),
                row_clean=result.row_status == 0,
            )
            assert result.row_status is not None
            endpoint_status = result.row_status | (
                endpoint_transition.overflow.to(torch.int32)
                * ROW_STATUS_DECODE_INVARIANT
            )
            result = AdvanceResult(
                token_ids=endpoint_transition.token_ids,
                token_lengths=torch.where(
                    endpoint_status == 0,
                    endpoint_transition.token_lengths,
                    torch.zeros_like(endpoint_transition.token_lengths),
                ),
                row_status=endpoint_status,
                captures=result.captures,
                frame_emission_counts=result.frame_emission_counts,
                frame_final_labels=result.frame_final_labels,
                frame_valid_lengths=result.frame_valid_lengths,
            )
            endpoint_history.index_copy_(
                0,
                rows_dev,
                endpoint_transition.history,
            )
            endpoint_book.index_copy_(
                0,
                rows_dev,
                endpoint_transition.book,
            )
        assert result.row_status is not None
        rs = result.row_status
        # Endpoint overflow changes the clean-row mask. Apply delta failures
        # here, never in the earlier captured bucket computation.
        counter_bad = bucket.counter_invariant_bad | (
            (rs == 0) & bucket.counter_delta_bad
        )
        rs |= counter_bad.to(torch.int32) * ROW_STATUS_BOOK_INVARIANT
        result = AdvanceResult(
            token_ids=result.token_ids,
            token_lengths=torch.where(
                rs == 0,
                result.token_lengths,
                torch.zeros_like(result.token_lengths),
            ),
            row_status=rs,
            captures=result.captures,
        )
        status.index_copy_(0, rows_dev, rs)
        # The transition's advanced last_label goes into the book
        # scratch unconditionally; its scatter predicate suppresses
        # every failed row without reading/restoring the old page.
        brows = book.index_select(0, rows_dev)
        brows[:, QUEUE_LAST_LABEL] = state.last_label.to(brows.dtype)
        book.index_copy_(0, rows_dev, brows)
        executed.append(
            {
                "geometry": g,
                "pos": pos_t,
                "rows_dev": rows_dev,
                "blocks_dev": blocks_dev,
                "state": state,
                "result": result,
                "batch": batch,
                "rs": rs,
                "endpoint": endpoint_transition,
            }
        )

    # ---- merged results + adapter projection (still fallible) ----
    chunk_all_t = (
        torch.cat([positions for _, positions, _ in bucket_pos])
        if bucket_pos
        else torch.zeros(0, dtype=torch.long)
    )
    chunk_all_dev = _stage("chunk_all", chunk_all_t)
    b_total = int(chunk_all_t.numel())
    k_max = max(
        (int(ex["result"].token_ids.shape[1]) for ex in executed),
        default=0,
    )
    merged_ids = torch.zeros(b_total, k_max, dtype=torch.int32, device=device)
    merged_len = torch.zeros(b_total, dtype=torch.int32, device=device)
    row0 = 0
    for ex in executed:
        nb = int(ex["pos"].numel())
        ids_b = ex["result"].token_ids
        merged_ids[row0 : row0 + nb, : ids_b.shape[1]] = ids_b
        merged_len[row0 : row0 + nb] = ex["result"].token_lengths
        row0 += nb
    merged = AdvanceResult(
        token_ids=merged_ids,
        token_lengths=merged_len,
        row_status=status.index_select(0, chunk_all_dev),
    )
    context = EmissionContext(
        roles=roles,
        input_ids=ids_dev,
        chunk_rows=chunk_all_dev,
        queue=queue,
        book=book,
        prompt_index=admitted_prompt_dev,
        row_status=status,
    )
    # The adapter is an extension seam, not transaction authority. Give it
    # independent writable snapshots and retain ``merged``/``context`` as the
    # immutable semantic oracle. Otherwise an in-place adapter could clear a
    # status bit or rewrite replay/FLUSH scratch before validation.
    adapter_result = AdvanceResult(
        token_ids=merged.token_ids.clone(),
        token_lengths=merged.token_lengths.clone(),
        row_status=merged.row_status.clone() if merged.row_status is not None else None,
    )
    adapter_context = EmissionContext(
        roles=context.roles.clone(),
        input_ids=context.input_ids.clone(),
        chunk_rows=context.chunk_rows.clone(),
        queue=context.queue.clone(),
        book=context.book.clone(),
        prompt_index=context.prompt_index.clone(),
        row_status=context.row_status.clone(),
    )
    projection = adapter(adapter_result, adapter_context)

    # Forced endpoint is a model control, not an acoustic CHUNK and not a
    # final FLUSH.  It is admitted only on a drained live session above.
    # Resolve its endpoint book and one-token replay transaction here so
    # both books join the outer transaction's single atomic scatter.
    if endpoint_enabled:
        from vllm.models.nemotron_asr.endpointing import (
            apply_forced_eou_tensors,
        )

        assert endpoint_book is not None
        assert eou_token_id is not None
        forced_clean = (roles == ROLE_EOU) & (projection.row_status == 0)
        forced_transition = apply_forced_eou_tensors(
            book=endpoint_book,
            selected_rows=forced_clean,
        )
        endpoint_book = forced_transition.book
        emit_eou = forced_transition.is_eou
        if cap <= 0:
            raise ValueError("forced endpoint requires a replay queue slot")

        forced_queue = torch.zeros_like(projection.queue)
        forced_queue[:, 0] = torch.where(
            emit_eou,
            torch.full_like(forced_queue[:, 0], int(eou_token_id)),
            forced_queue[:, 0],
        )
        queue_out = torch.where(
            forced_clean.unsqueeze(1),
            forced_queue,
            projection.queue,
        )
        book_out = projection.book.clone()
        book_out[:, QUEUE_HEAD] = torch.where(
            forced_clean,
            emit_eou.to(book_out.dtype),
            book_out[:, QUEUE_HEAD],
        )
        book_out[:, QUEUE_LEN] = torch.where(
            forced_clean,
            emit_eou.to(book_out.dtype),
            book_out[:, QUEUE_LEN],
        )
        book_out[:, BOOK_PENDING_ECHO] = torch.where(
            forced_clean,
            emit_eou.to(book_out.dtype),
            book_out[:, BOOK_PENDING_ECHO],
        )
        book_out[:, BOOK_EXPECTED_LABEL] = torch.where(
            forced_clean,
            torch.where(
                emit_eou,
                torch.full_like(book_out[:, BOOK_EXPECTED_LABEL], int(eou_token_id)),
                torch.zeros_like(book_out[:, BOOK_EXPECTED_LABEL]),
            ),
            book_out[:, BOOK_EXPECTED_LABEL],
        )
        rows_out = projection.rows.clone()
        if hidden:
            rows_out[:, 0] = torch.where(
                forced_clean,
                torch.where(
                    emit_eou,
                    torch.full_like(rows_out[:, 0], int(eou_token_id)),
                    torch.full_like(rows_out[:, 0], park_id),
                ),
                rows_out[:, 0],
            )
        projection = EmissionProjection(
            rows=rows_out,
            queue=queue_out,
            book=book_out,
            row_status=projection.row_status,
        )

    # ---- every conversion + shape/dtype validation, pre-commit ----
    if (
        tuple(projection.rows.shape) != (n_real, hidden)
        or projection.rows.dtype != inputs_embeds.dtype
        or projection.rows.device != device
        or not projection.rows.is_contiguous()
    ):
        raise ValueError(
            "adapter returned rows shaped "
            f"{tuple(projection.rows.shape)}/{projection.rows.dtype}, "
            f"expected {(n_real, hidden)}/{inputs_embeds.dtype}"
        )
    if (
        tuple(projection.queue.shape) != tuple(queue.shape)
        or projection.queue.dtype != queue_pool.dtype
        or projection.queue.device != device
        or tuple(projection.book.shape) != tuple(book.shape)
        or projection.book.dtype != book_pool.dtype
        or projection.book.device != device
    ):
        raise ValueError(
            "adapter returned queue/book scratch with a different shape or dtype than "
            "the resident pools"
        )
    if (
        projection.row_status.device != device
        or projection.row_status.dtype != torch.int32
        or tuple(projection.row_status.shape) != (n_real,)
        or not projection.row_status.is_contiguous()
    ):
        raise ValueError("adapter row_status must be device-local int32 shaped (N,)")

    # Validate the adapter's proposed persistent book before it can
    # become resident. Any defect is a row-tier masked park/no-store.
    pbook = projection.book
    phead = pbook[:, QUEUE_HEAD].long()
    plen = pbook[:, QUEUE_LEN].long()
    ppend_col = pbook[:, BOOK_PENDING_ECHO]
    ppend = ppend_col == 1
    pexpected = pbook[:, BOOK_EXPECTED_LABEL].long()
    plast = pbook[:, QUEUE_LAST_LABEL].long()
    premaining = plen - phead
    proposed_queue_value_valid = (projection.queue >= 0) & (projection.queue < blank)
    if endpoint_enabled:
        assert eou_token_id is not None
        proposed_queue_value_valid |= projection.queue == int(eou_token_id)
    proposed_queued_bad = (
        (~proposed_queue_value_valid) & (slot < plen.unsqueeze(1))
    ).any(dim=1)
    proposed_prior_emitted = (
        projection.queue.gather(
            1,
            (phead - 1).clamp(min=0, max=max(cap - 1, 0)).unsqueeze(1),
        )
        .squeeze(1)
        .long()
    )
    proposed_queue_last = (
        projection.queue.gather(
            1,
            (plen - 1).clamp(min=0, max=max(cap - 1, 0)).unsqueeze(1),
        )
        .squeeze(1)
        .long()
    )
    proposed_tail_is_eou = torch.zeros_like(ppend)
    proposed_expected_is_eou = torch.zeros_like(ppend)
    if endpoint_enabled:
        assert eou_token_id is not None
        proposed_tail_is_eou = (plen > 0) & (proposed_queue_last == int(eou_token_id))
        proposed_expected_is_eou = pexpected == int(eou_token_id)
    proposed_bad = (
        (phead < 0)
        | (phead > plen)
        | (plen > cap)
        | ((ppend_col != 0) & (ppend_col != 1))
        | (plast < 0)
        | (plast > blank)
        | (pexpected < 0)
        | ((pexpected > blank) & (~proposed_expected_is_eou))
        | (ppend & (pexpected >= blank) & (~proposed_expected_is_eou))
        | proposed_queued_bad
        | (ppend & (phead < 1))
        | (ppend & (pexpected != proposed_prior_emitted))
        | ((plen > 0) & (~proposed_tail_is_eou) & (plast != proposed_queue_last))
        | ((premaining > 0) & (~ppend))
        | (pbook[:, BOOK_GEOMETRY].long() != plan_geom_dev)
        | (pbook[:, QUEUE_PROMPT].long() != admitted_prompt_dev)
    )
    incoming_adapter_status = status
    lost_status = (
        projection.row_status & incoming_adapter_status
    ) != incoming_adapter_status
    status = incoming_adapter_status | projection.row_status
    status |= lost_status.to(torch.int32) * ROW_STATUS_DECODE_INVARIANT
    status |= ((projection.row_status & ~((1 << 19) - 1)) != 0).to(
        torch.int32
    ) * ROW_STATUS_DECODE_INVARIANT
    status |= proposed_bad.to(torch.int32) * ROW_STATUS_BOOK_INVARIANT
    projection_bad = _mrv1_projection_invariant_rows(
        merged,
        context,
        projection,
        status,
        park_id=park_id,
        blank_id=blank,
        eou_token_id=eou_token_id if endpoint_enabled else None,
    )
    status |= projection_bad.to(torch.int32) * ROW_STATUS_DECODE_INVARIANT
    native_stage: Callable[[], None] | None = None
    if native_burst_handoff is not None:
        from vllm.models.nemotron_asr.native_burst import (
            finalize_native_burst,
            native_burst_invariant_rows,
        )

        source_projection = EmissionProjection(
            projection.rows,
            projection.queue,
            projection.book,
            status.clone(),
        )
        native_projection = finalize_native_burst(
            source_projection,
            context,
            park_id=park_id,
            blank_id=blank,
        )
        native_bad = native_burst_invariant_rows(
            source_projection,
            context,
            native_projection,
            park_id=park_id,
            blank_id=blank,
            eou_token_id=eou_token_id if endpoint_enabled else None,
        )
        native_bad |= native_projection.num_sampled > native_burst_handoff.max_tokens
        status |= native_bad.to(torch.int32) * ROW_STATUS_DECODE_INVARIANT
        native_projection.sampled_token_ids.masked_fill_(native_bad.unsqueeze(1), -1)
        native_projection.sampled_token_ids[:, 0] = torch.where(
            native_bad,
            park_id,
            native_projection.sampled_token_ids[:, 0],
        )
        native_projection.num_sampled.masked_fill_(native_bad, 1)
        native_projection.row_status.copy_(status)
        projection = native_projection
        native_stage = native_burst_handoff.reserve(native_projection)
    failed = status != 0
    projection_rows = projection.rows
    projection_rows.masked_fill_(failed.reshape(-1, 1), 0)
    if hidden:
        projection_rows[:, 0] = torch.where(
            failed,
            torch.full_like(projection_rows[:, 0], park_id),
            projection_rows[:, 0],
        )

    scatter_ops: list[_ScatterDescriptor] = []
    for ex in executed:
        rows_dev = ex["rows_dev"]
        blocks = ex["blocks_dev"]
        st = ex["state"]
        bucket_status = status.index_select(0, rows_dev)
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = [
            (frontend_raw_pool, st.raw_tail),
            (frontend_mel_pool, st.mel_tail),
            (frontend_counter_pool, st.frontend_counters),
            (h_pool, st.h),
            (c_pool, st.c),
        ]
        pairs += list(zip(channel_pools, st.channel, strict=True))
        pairs += list(zip(time_pools, st.time, strict=True))
        pairs += list(zip(len_pools, st.window_valid, strict=True))
        for pool, scratch in pairs:
            scatter_ops.append(_ScatterDescriptor(pool, scratch, blocks, bucket_status))
    scatter_ops.extend(
        (
            _ScatterDescriptor(queue_pool, projection.queue, didx, status),
            _ScatterDescriptor(book_pool, projection.book, didx, status),
        )
    )
    if endpoint_enabled:
        assert endpoint_history_pool is not None
        assert endpoint_book_pool is not None
        assert endpoint_history is not None
        assert endpoint_book is not None
        scatter_ops.extend(
            (
                _ScatterDescriptor(
                    endpoint_history_pool,
                    endpoint_history,
                    didx,
                    status,
                ),
                _ScatterDescriptor(
                    endpoint_book_pool,
                    endpoint_book,
                    didx,
                    status,
                ),
            )
        )
    # Complete-plan validation is intentionally a separate pass: no
    # descriptor may launch before every later descriptor is known good.
    for op in scatter_ops:
        validate_masked_page_scatter(op.pool, op.scratch, op.blocks, op.row_status)

    # ---- records + ONE composite reservation, still pre-commit ----
    records: list[CaptureRecord] = []
    if capture_on and b_total:
        for ex in executed:
            caps = ex["result"].captures
            if caps is None:
                raise ValueError(
                    "capture enabled but the transition staged no captures "
                    "(PORT-HOOK-001 pre-commit fatal)"
                )
            for i, pos in enumerate(ex["pos"].tolist()):
                records.append(
                    CaptureRecord(
                        row=pos,
                        request_id=plan.request_ids[pos],
                        block_id=int(idx_cpu[pos]),
                        admission_generation=int(plan.admission_generation[pos]),
                        geometry=int(ex["geometry"]),
                        chunk_sequence=ex["batch"].chunk_sequence[i],
                        prompt_index=ex["batch"].prompt_index[i],
                        row_status=status[pos],
                        frontend_mel=caps.frontend_mel[i],
                        mel_length=caps.mel_lengths[i],
                        encoder_raw=caps.encoder_raw[i],
                        encoder_conditioned=caps.encoder_conditioned[i],
                        encoder_length=caps.encoder_lengths[i],
                    )
                )
        records.sort(key=lambda r: r.row)
        payload_bytes = sum(
            tensor.numel() * tensor.element_size()
            for record in records
            for tensor in (
                record.chunk_sequence,
                record.prompt_index,
                record.row_status,
                record.frontend_mel,
                record.mel_length,
                record.encoder_raw,
                record.encoder_conditioned,
                record.encoder_length,
            )
        )
        capture_plan: CapturePlan | None = CapturePlan(
            rows=len(records), payload_bytes=payload_bytes
        )
    else:
        capture_plan = None
    # @spec PORT-SESS-016: publish committed queue state through the mandatory
    # status handoff. Preparation is fallible; staging after scatter is no-fail.
    progress_sequence = counters_small[:, CTR_EXPECTED_CHUNK_SEQUENCE].clone()
    for ex in executed:
        progress_sequence.index_copy_(
            0,
            ex["rows_dev"],
            ex["state"].frontend_counters[:, CTR_EXPECTED_CHUNK_SEQUENCE],
        )
    progress_phase = torch.where(projection.book[:, BOOK_PENDING_ECHO] == 0, 2, 1)
    progress_phase = torch.where((roles == ROLE_FLUSH) & ~park_echo, 3, progress_phase)
    progress_phase = torch.where(roles == ROLE_EOU, progress_phase + 3, progress_phase)
    terminal_park_echo = (roles == ROLE_FLUSH) & (ids_dev == park_id)
    progress_phase = torch.where(park_echo | terminal_park_echo, 0, progress_phase)
    progress = torch.stack(
        (progress_sequence, projection.book[:, QUEUE_HEAD].long(), progress_phase),
        dim=1,
    )
    prepared_records = tuple(records)
    reservation: CommitReservation | None = None
    cancel_reservation: Callable[[], None] | None = None
    stage_reservation: Callable[[torch.Tensor], None] | None = None
    if commit_sink is not None:
        reservation = commit_sink.reserve(
            CommitPlan(
                bindings=plan.bindings,
                capture=capture_plan,
                records=prepared_records,
                progress=progress,
            )
        )
        cancel_candidate = getattr(reservation, "cancel", None)
        if not callable(cancel_candidate):
            raise ValueError("commit sink returned a reservation without cancel()")
        cancel_reservation = cancel_candidate
        try:
            stage_candidate = getattr(reservation, "stage", None)
        except BaseException:
            cancel_reservation()
            raise
        if not callable(stage_candidate):
            cancel_reservation()
            raise ValueError("commit sink returned a reservation without stage()")
        stage_reservation = stage_candidate

    # ---- commit: prevalidated, allocation-free scatters + no-fail stage ----
    # Every descriptor was already validated above (the complete-plan
    # pass); the commit window calls the private prevalidated
    # executor directly so no descriptor is re-validated here.
    with phase("port.scatter"):
        try:
            for op in scatter_ops:
                _execute_masked_page_scatter_(
                    op.pool, op.scratch, op.blocks, op.row_status
                )
        except BaseException:
            if cancel_reservation is not None:
                cancel_reservation()
            raise
        # ---- ONE no-fail combined stage through the reserved ticket ----
        if stage_reservation is not None:
            stage_reservation(status)
        if native_stage is not None:
            native_stage()
    return projection_rows
