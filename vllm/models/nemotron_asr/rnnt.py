# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RNN-T predictor, joint, and greedy label-looping chunk decode.

Semantics match NeMo's greedy label-looping computer (PORT-DEC-001):
``max_symbols_per_step = 10`` with hard-forced frame advance, blank
advances the frame cursor, blank-as-pad (SOS = blank, embedding pads at
the blank index), predictor state and last label carried across chunk
boundaries with no SOS re-injection. The joint follows NeMo's split
form (rnnt.py:1677-1724 @ de242add): per-side projections into the
joint hidden, broadcast sum, activation, final linear over ``V + 1``
(blank last). Greedy argmax needs no softmax at temperature 1.

This module is the single boundary the decode choice lives behind
(D-b chunk-loop + replay is the target; the loop here is the semantic
reference the fused/batched implementation is held to by tests).
"""

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from vllm.models.nemotron_asr.rnnt_cell import (
    ManualLSTM,
)

MAX_SYMBOLS_PER_STEP = 10

#: Bumped whenever either decode candidate's ALGORITHM changes —
#: dispatch tables bind to this, not to a fork commit (an unrelated
#: commit must not invalidate a table; an algorithm change must).
DECODE_ALGO_REVISION = "decode-algo-v1"


@dataclass
class DecodeState:
    """Per-session decode state carried across chunks (page-backed).

    ``h``/``c``: predictor LSTM state ``(layers, batch, hidden)``.
    ``last_label``: ``(batch,)`` int64; blank at session start (SOS).
    """

    h: torch.Tensor
    c: torch.Tensor
    last_label: torch.Tensor


@dataclass
class FrameAlignedDecode:
    """Dense decode output with one endpoint symbol source per frame."""

    token_ids: torch.Tensor
    token_lengths: torch.Tensor
    state: DecodeState
    frame_emission_counts: torch.Tensor
    frame_final_labels: torch.Tensor


class Predictor(nn.Module):
    """Embedding + LSTM prediction network (blank-as-pad)."""

    def __init__(
        self,
        *,
        vocab_size: int,
        pred_hidden: int,
        pred_rnn_layers: int,
    ) -> None:
        super().__init__()
        self.blank_id = vocab_size  # blank is the last index (V)
        self.embed = nn.Embedding(
            vocab_size + 1, pred_hidden, padding_idx=self.blank_id
        )
        self.rnn = ManualLSTM(
            input_size=pred_hidden,
            hidden_size=pred_hidden,
            num_layers=pred_rnn_layers,
        )

    def step(
        self,
        labels: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """One prediction step from the previous label."""
        return self.rnn.step(self.embed(labels), state)


class Joint(nn.Module):
    """NeMo split joint: side projections, sum, activation, final."""

    def __init__(
        self,
        *,
        enc_hidden: int,
        pred_hidden: int,
        joint_hidden: int,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.enc = nn.Linear(enc_hidden, joint_hidden)
        self.pred = nn.Linear(pred_hidden, joint_hidden)
        self.joint_net = nn.Sequential(
            nn.ReLU(), nn.Linear(joint_hidden, vocab_size + 1)
        )

    def logits(self, enc_frame: torch.Tensor, pred_out: torch.Tensor) -> torch.Tensor:
        """Joint logits for one (frame, prediction) pair, ``(B, V+1)``.

        ``pred_out`` arrives in the recurrent-state dtype (fp32 by
        PORT-PREC-005) while the joint computes in the policy's weight
        dtype — both inputs cast to the weights here (PORT-PREC-001).
        """
        dtype = self.enc.weight.dtype
        out: torch.Tensor = self.joint_net(
            self.enc(enc_frame.to(dtype)) + self.pred(pred_out.to(dtype))
        )
        return out


def greedy_decode_chunk(
    enc_frames: torch.Tensor,
    predictor: Predictor,
    joint: Joint,
    state: DecodeState,
    *,
    max_symbols: int = MAX_SYMBOLS_PER_STEP,
) -> tuple[list[int], DecodeState]:
    """Greedy label-looping decode over one chunk's encoder frames.

    Reference (single-stream) semantics of NeMo's batched computer:
    per frame, emit up to ``max_symbols`` non-blank labels — each
    advancing the predictor — then advance the frame on blank or on the
    hard cap. State flows in and out untouched by chunk boundaries
    (no SOS re-injection, PORT-DEC-001).

    Args:
        enc_frames: ``(time, enc_hidden)`` conditioned encoder frames.
        predictor: The prediction network.
        joint: The joint network.
        state: Carried decode state; mutated copy returned.
        max_symbols: Per-frame emission cap (checkpoint value 10).

    Returns:
        Emitted label ids (the replay-queue content for this chunk,
        PORT-DEC-002) and the advanced state.

    """
    h, c = state.h, state.c
    last_label = state.last_label
    emitted: list[int] = []
    pred_out, pred_state = predictor.step(last_label, (h, c))
    for t in range(enc_frames.shape[0]):
        frame = enc_frames[t].unsqueeze(0)
        for _ in range(max_symbols):
            logits = joint.logits(frame, pred_out)
            label = int(logits.argmax(dim=-1))
            if label == predictor.blank_id:
                break
            emitted.append(label)
            last_label = torch.tensor(
                [label], dtype=torch.long, device=enc_frames.device
            )
            h, c = pred_state
            pred_out, pred_state = predictor.step(last_label, (h, c))
    # Blank does not advance the predictor: state reflects the last
    # non-blank emission only.
    return emitted, DecodeState(h=h, c=c, last_label=last_label)


def greedy_decode_batch(
    enc_frames: torch.Tensor,
    predictor: Predictor,
    joint: Joint,
    state: DecodeState,
    *,
    max_symbols: int = MAX_SYMBOLS_PER_STEP,
) -> tuple[list[list[int]], DecodeState]:
    """Batched greedy label-looping decode over stacked sessions.

    Semantically the reference loop (``greedy_decode_chunk``) run per
    stream: each stream emits until its own blank or the per-frame cap,
    committed state advances only on that stream's non-blank emissions,
    and inactive streams are masked out of every update. Streams share
    the frame count (callers batch same-shape chunk-steps).

    Args:
        enc_frames: ``(batch, time, enc_hidden)`` conditioned frames.
        predictor: The prediction network.
        joint: The joint network.
        state: Stacked decode state — ``h``/``c`` ``(layers, batch,
            hidden)``, ``last_label`` ``(batch,)``.
        max_symbols: Per-frame emission cap (checkpoint value 10).

    Returns:
        Per-stream emitted label ids and the advanced stacked state.

    """
    batch = enc_frames.shape[0]
    blank = predictor.blank_id
    h, c = state.h, state.c
    last_label = state.last_label
    emitted: list[list[int]] = [[] for _ in range(batch)]
    pred_out, (pred_h, pred_c) = predictor.step(last_label, (h, c))
    for t in range(enc_frames.shape[1]):
        frame = enc_frames[:, t]
        active = torch.ones(batch, dtype=torch.bool, device=enc_frames.device)
        for _ in range(max_symbols):
            logits = joint.logits(frame, pred_out)
            labels = logits.argmax(dim=-1)
            emit = active & (labels != blank)
            if not bool(emit.any()):
                break
            for i in emit.nonzero(as_tuple=True)[0].tolist():
                emitted[i].append(int(labels[i]))
            gate = emit.view(1, -1, 1)
            last_label = torch.where(emit, labels, last_label)
            # Commit the state that produced this pred_out, emitters only.
            h = torch.where(gate, pred_h, h)
            c = torch.where(gate, pred_c, c)
            new_out, (new_h, new_c) = predictor.step(last_label, (h, c))
            pred_out = torch.where(emit.unsqueeze(-1), new_out, pred_out)
            pred_h = torch.where(gate, new_h, pred_h)
            pred_c = torch.where(gate, new_c, pred_c)
            active = emit
    return emitted, DecodeState(h=h, c=c, last_label=last_label)


def decode_compact_active(
    enc_frames: torch.Tensor,
    enc_lengths: torch.Tensor,
    predictor: Predictor,
    joint: Joint,
    state: DecodeState,
    *,
    max_symbols: int = MAX_SYMBOLS_PER_STEP,
) -> tuple[torch.Tensor, torch.Tensor, DecodeState]:
    """Length-aware compact-active greedy decode (PORT-PERF-001).

    The tensorized production loop ``greedy_decode_batch`` is the
    differential oracle for: fixed outer frame/symbol trips (no
    host-side early exit), GPU masks for semantic activity — a row is
    active at frame ``t`` only while ``t < enc_lengths[row]`` (padded
    frames never decode) and, within a frame, only while its previous
    trip emitted — with tensor-indexed compaction so joint/predictor
    work runs on active rows only, and index_put/index-copy state
    commits. No ``.item()``, ``.tolist()``, Python label lists, or
    host booleans; the one eager-mode concession is ``nonzero``'s
    implicit shape materialization (the compact-vs-dense profile
    decides whether that stays, per the review gate).

    Args:
        enc_frames: ``(B, T_pad, enc_hidden)`` conditioned frames.
        enc_lengths: ``(B,)`` long valid frame counts (rows may be
            padded to ``T_pad``).
        predictor: The prediction network.
        joint: The joint network.
        state: Stacked decode state (NOT mutated; the advanced state
            is returned).
        max_symbols: Per-frame emission cap (checkpoint value 10).

    Returns:
        ``token_ids`` ``(B, T_pad * max_symbols)`` int32 padded,
        ``token_lengths`` ``(B,)`` int32, and the advanced state.

    """
    result = decode_compact_active_frames(
        enc_frames,
        enc_lengths,
        predictor,
        joint,
        state,
        max_symbols=max_symbols,
    )
    return result.token_ids, result.token_lengths, result.state


def decode_compact_active_frames(
    enc_frames: torch.Tensor,
    enc_lengths: torch.Tensor,
    predictor: Predictor,
    joint: Joint,
    state: DecodeState,
    *,
    max_symbols: int = MAX_SYMBOLS_PER_STEP,
) -> FrameAlignedDecode:
    """Compact-active decode retaining the endpoint symbol per frame."""
    batch, t_pad, _ = enc_frames.shape
    device = enc_frames.device
    blank = predictor.blank_id
    if bool((enc_lengths < 0).any()) or bool((enc_lengths > t_pad).any()):
        raise ValueError(
            f"enc_lengths out of range for T_pad={t_pad}: {enc_lengths.tolist()}"
        )
    h = state.h.clone()
    c = state.c.clone()
    last_label = state.last_label.clone()
    token_ids = torch.zeros(
        batch, t_pad * max_symbols, dtype=torch.int32, device=device
    )
    token_lengths = torch.zeros(batch, dtype=torch.long, device=device)
    frame_emission_counts = torch.zeros(batch, t_pad, dtype=torch.int32, device=device)
    frame_final_labels = torch.full(
        (batch, t_pad), blank, dtype=torch.int32, device=device
    )
    pred_out, (pred_h, pred_c) = predictor.step(last_label, (h, c))
    for t in range(t_pad):
        frame = enc_frames[:, t]
        active = t < enc_lengths
        for _ in range(max_symbols):
            rows = active.nonzero(as_tuple=True)[0]
            logits = joint.logits(
                frame.index_select(0, rows),
                pred_out.index_select(0, rows),
            )
            labels = logits.argmax(dim=-1)
            emit = labels != blank
            erows = rows[emit]
            elabels = labels[emit]
            token_ids[erows, token_lengths[erows]] = elabels.to(torch.int32)
            token_lengths[erows] += 1
            frame_emission_counts[erows, t] += 1
            frame_final_labels[erows, t] = elabels.to(torch.int32)
            last_label[erows] = elabels
            # Commit the state that produced this pred_out, emitters
            # only; blank never advances the predictor.
            h[:, erows] = pred_h[:, erows]
            c[:, erows] = pred_c[:, erows]
            new_out, (new_h, new_c) = predictor.step(
                last_label[erows], (h[:, erows], c[:, erows])
            )
            pred_out[erows] = new_out
            pred_h[:, erows] = new_h
            pred_c[:, erows] = new_c
            active = torch.zeros_like(active)
            active[erows] = True
    return FrameAlignedDecode(
        token_ids=token_ids,
        token_lengths=token_lengths.to(torch.int32),
        state=DecodeState(h=h, c=c, last_label=last_label),
        frame_emission_counts=frame_emission_counts,
        frame_final_labels=frame_final_labels,
    )


def decode_dense_masked(
    enc_frames: torch.Tensor,
    enc_lengths: torch.Tensor,
    predictor: Predictor,
    joint: Joint,
    state: DecodeState,
    *,
    max_symbols: int = MAX_SYMBOLS_PER_STEP,
) -> tuple[torch.Tensor, torch.Tensor, DecodeState]:
    """Length-aware fixed-trip dense greedy decode (PORT-ADV-004).

    The synchronization-free candidate the dense-vs-compact profile
    (PORT-DEC-008) selects from at startup: exactly ``T_pad ×
    max_symbols`` full-batch joint/predictor trips with GPU masks for
    semantic activity — a row is active at frame ``t`` only while
    ``t < enc_lengths[row]`` and, within a frame, only while its
    previous trip emitted. Non-emitting rows pass through every
    ``torch.where`` untouched, so the extra trips the compact loop's
    compaction skips are exact no-ops here (the ``decode_chunk_paged``
    masked-trip pattern). Token writes use gather/where/scatter with
    one unique index per row — deterministic and sync-free. No
    ``nonzero``, ``.item()``, ``.tolist()``, host booleans,
    data-dependent shapes, raises on device predicates, or device
    assertions anywhere (a fired CUDA assert corrupts the context —
    PORT-ADV-004): out-of-range lengths clamp to ``[0, T_pad]`` as a
    safe no-op posture; a range defect upstream is ``advance_session``
    row-status territory.

    Args:
        enc_frames: ``(B, T_pad, enc_hidden)`` conditioned frames.
        enc_lengths: ``(B,)`` long valid frame counts (rows may be
            padded to ``T_pad``); values outside ``[0, T_pad]`` are
            clamped.
        predictor: The prediction network.
        joint: The joint network.
        state: Stacked decode state (NOT mutated; the advanced state
            is returned). A zero-length row's state returns
            bit-identical.
        max_symbols: Per-frame emission cap (checkpoint value 10).

    Returns:
        ``token_ids`` ``(B, T_pad * max_symbols)`` int32 padded,
        ``token_lengths`` ``(B,)`` int32, and the advanced state.

    """
    result = decode_dense_masked_frames(
        enc_frames,
        enc_lengths,
        predictor,
        joint,
        state,
        max_symbols=max_symbols,
    )
    return result.token_ids, result.token_lengths, result.state


# @spec PORT-DEC-001, PORT-SEG-002
def decode_dense_masked_frames(
    enc_frames: torch.Tensor,
    enc_lengths: torch.Tensor,
    predictor: Predictor,
    joint: Joint,
    state: DecodeState,
    *,
    max_symbols: int = MAX_SYMBOLS_PER_STEP,
) -> FrameAlignedDecode:
    """Run dense decode and retain one final label per encoder frame."""
    batch, t_pad, _ = enc_frames.shape
    device = enc_frames.device
    blank = predictor.blank_id
    enc_lengths = enc_lengths.clamp(min=0, max=t_pad)
    h = state.h.clone()
    c = state.c.clone()
    last_label = state.last_label.clone()
    capacity = max(t_pad * max_symbols, 1)
    token_ids = torch.zeros(
        batch, t_pad * max_symbols, dtype=torch.int32, device=device
    )
    token_lengths = torch.zeros(batch, dtype=torch.long, device=device)
    frame_emission_counts = torch.zeros(
        batch,
        t_pad,
        dtype=torch.int32,
        device=device,
    )
    frame_final_labels = torch.full(
        (batch, t_pad),
        blank,
        dtype=torch.int32,
        device=device,
    )
    pred_out, (pred_h, pred_c) = predictor.step(last_label, (h, c))
    for t in range(t_pad):
        frame = enc_frames[:, t]
        # The encoder side is invariant across this frame's label attempts.
        # Keep the original per-frame shape/strides and custom joint protocol.
        projected_frame = (
            joint.enc(frame.to(joint.enc.weight.dtype))
            if type(joint) is Joint and max_symbols > 0
            else None
        )
        active = t < enc_lengths
        for _ in range(max_symbols):
            if projected_frame is None:
                logits = joint.logits(frame, pred_out)
            else:
                logits = joint.joint_net(
                    projected_frame + joint.pred(pred_out.to(joint.enc.weight.dtype))
                )
            labels = logits.argmax(dim=-1)
            emit = active & (labels != blank)
            idx = token_lengths.clamp(max=capacity - 1).unsqueeze(1)
            token_ids.scatter_(
                1,
                idx,
                torch.where(
                    emit.unsqueeze(1),
                    labels.unsqueeze(1).to(torch.int32),
                    token_ids.gather(1, idx),
                ),
            )
            token_lengths = token_lengths + emit.long()
            frame_emission_counts[:, t] += emit.to(torch.int32)
            frame_final_labels[:, t] = torch.where(
                emit,
                labels.to(torch.int32),
                frame_final_labels[:, t],
            )
            gate = emit.view(1, -1, 1)
            last_label = torch.where(emit, labels, last_label)
            # Commit the state that produced this pred_out, emitters
            # only; blank never advances the predictor.
            h = torch.where(gate, pred_h, h)
            c = torch.where(gate, pred_c, c)
            new_out, (new_h, new_c) = predictor.step(last_label, (h, c))
            pred_out = torch.where(emit.unsqueeze(-1), new_out, pred_out)
            pred_h = torch.where(gate, new_h, pred_h)
            pred_c = torch.where(gate, new_c, pred_c)
            active = emit
    return FrameAlignedDecode(
        token_ids=token_ids,
        token_lengths=token_lengths.to(torch.int32),
        state=DecodeState(h=h, c=c, last_label=last_label),
        frame_emission_counts=frame_emission_counts,
        frame_final_labels=frame_final_labels,
    )


# ---- engine-tier decode seams (α3 tests-first; PORT-DEC-002/003/007/008) -----

#: Slot indices in the replay-queue page's SEVEN-slot session book
#: (ReplayQueuePage's second tensor) — manifests.BOOK_FIELDS order
#: (the manifest is the single source; test-pinned): queue head,
#: queue length, last emitted label, prompt index, admitted geometry,
#: pending-echo flag, expected label.
QUEUE_HEAD, QUEUE_LEN, QUEUE_LAST_LABEL, QUEUE_PROMPT = 0, 1, 2, 3
BOOK_GEOMETRY, BOOK_PENDING_ECHO, BOOK_EXPECTED_LABEL = 4, 5, 6


def park_token_id(hf_config: Any) -> int:
    """The park token: the checkpoint's ``eos_token_id`` (PORT-DEC-003).

    A config VALUE, never engine structure — published as an HF added
    special token so default ``skip_special_tokens`` strips it.

    Raises:
        ValueError: If the config carries no ``eos_token_id`` — the
            park path cannot exist without it, so fail at load.

    """
    eos_token_id = getattr(hf_config, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError(
            "hf_config has no eos_token_id: the park path (PORT-DEC-003) requires the "
            "checkpoint to publish one"
        )
    return int(eos_token_id)


def realtime_token_budget(
    *, frames_per_chunk: int, max_symbols: int = MAX_SYMBOLS_PER_STEP
) -> int:
    """Constant per-burst budget: queue capacity plus the park token.

    ``realtime_max_tokens = frames_per_chunk × max_symbols + 1``
    (PORT-INT-002; the output counter clears per update, so the budget
    is per-burst, not per-session).
    """
    return frames_per_chunk * max_symbols + 1


def forced_logits_rows(chosen: torch.Tensor, *, num_logits: int) -> torch.Tensor:
    """Forced-emission logits: 0 at the chosen id, −inf elsewhere.

    Never +inf (PORT-DEC-002): with the model-pinned greedy params the
    engine argmaxes these rows, and the single finite entry is the
    emission. One row per session; ``chosen`` is ``(B,)`` long.

    Raises:
        ValueError: If any chosen id falls outside ``num_logits``.

    """
    if chosen.numel() and (int(chosen.min()) < 0 or int(chosen.max()) >= num_logits):
        raise ValueError(
            f"chosen id out of range for num_logits={num_logits}: {chosen.tolist()}"
        )
    rows = torch.full(
        (chosen.shape[0], num_logits),
        float("-inf"),
        dtype=torch.float32,
        device=chosen.device,
    )
    rows.scatter_(1, chosen.view(-1, 1), 0.0)
    return rows


def write_decision_carrier(hidden: torch.Tensor, ids: torch.Tensor) -> None:
    """Write each session's emitted/queued id into its own hidden row.

    The forward's output row is the decision carrier (PORT-DEC-002):
    the engine's ``logits_indices`` gather hands ``compute_logits``
    exactly these rows, row-aligned by construction — no cross-call
    stashing. Rides the ``queue_state`` dtype axis.

    Raises:
        ValueError: If ``hidden.dtype`` cannot represent every id
            exactly (integer-exact range must cover the id space; a
            bf16 carrier corrupts ids > 256).

    """
    dtype = hidden.dtype
    mantissa_bits = round(-math.log2(torch.finfo(dtype).eps))
    bound = 2 ** (mantissa_bits + 1)
    if ids.numel() and int(ids.max()) >= bound:
        raise ValueError(
            f"{dtype} is not integer-exact past {bound}; the decision carrier cannot "
            f"represent id {int(ids.max())}"
        )
    hidden[:, 0] = ids.to(dtype)


def read_decision_carrier(hidden: torch.Tensor) -> torch.Tensor:
    """Recover the per-session ids ``write_decision_carrier`` wrote."""
    return hidden[:, 0].long()


def decode_chunk_paged(
    enc_frames: torch.Tensor,
    predictor: Predictor,
    joint: Joint,
    *,
    h_pool: torch.Tensor,
    c_pool: torch.Tensor,
    queue_pool: torch.Tensor,
    book_pool: torch.Tensor,
    state_indices: torch.Tensor,
    max_symbols: int = MAX_SYMBOLS_PER_STEP,
) -> None:
    """Fixed-trip tensorized D-b chunk decode over page-backed state.

    Exactly ``time × max_symbols`` masked, page-writing joint/predictor
    trips — no data-dependent host branching (PORT-DEC-008); the
    math-tier ``greedy_decode_batch`` is the differential oracle this
    must match bit-for-bit (labels, predictor state, queue contents),
    never the implementation. Reads/writes ``(h, c)`` at
    ``h_pool[state_indices]``/``c_pool``, appends emissions to
    ``queue_pool`` rows, and updates the session book's queue slots in
    ``book_pool`` (QUEUE_* indices).
    """
    batch = enc_frames.shape[0]
    blank = predictor.blank_id
    # Page rows hold (layers, hidden); the predictor takes
    # (layers, batch, hidden).
    h = h_pool[state_indices].transpose(0, 1).contiguous()
    c = c_pool[state_indices].transpose(0, 1).contiguous()
    last_label = book_pool[state_indices, QUEUE_LAST_LABEL].long()
    lens = torch.zeros(batch, dtype=torch.long, device=enc_frames.device)

    pred_out, (pred_h, pred_c) = predictor.step(last_label, (h, c))
    for t in range(enc_frames.shape[1]):
        frame = enc_frames[:, t]
        active = torch.ones(batch, dtype=torch.bool, device=enc_frames.device)
        for _ in range(max_symbols):
            logits = joint.logits(frame, pred_out)
            labels = logits.argmax(dim=-1)
            emit = active & (labels != blank)
            # Masked trip: non-emitting rows pass through every
            # torch.where untouched, so the extra trips the oracle's
            # early break skips are bit-exact no-ops here.
            rows = state_indices[emit]
            queue_pool[rows, lens[emit]] = labels[emit].to(queue_pool.dtype)
            lens = lens + emit.long()
            gate = emit.view(1, -1, 1)
            last_label = torch.where(emit, labels, last_label)
            h = torch.where(gate, pred_h, h)
            c = torch.where(gate, pred_c, c)
            new_out, (new_h, new_c) = predictor.step(last_label, (h, c))
            pred_out = torch.where(emit.unsqueeze(-1), new_out, pred_out)
            pred_h = torch.where(gate, new_h, pred_h)
            pred_c = torch.where(gate, new_c, pred_c)
            active = emit
    h_pool[state_indices] = h.transpose(0, 1)
    c_pool[state_indices] = c.transpose(0, 1)
    # A fresh burst: head rewinds, length is this chunk's emissions.
    book_pool[state_indices, QUEUE_HEAD] = 0.0
    book_pool[state_indices, QUEUE_LEN] = lens.to(book_pool.dtype)
    book_pool[state_indices, QUEUE_LAST_LABEL] = last_label.to(book_pool.dtype)


def replay_step(
    queue_pool: torch.Tensor,
    book_pool: torch.Tensor,
    *,
    state_indices: torch.Tensor,
    park_id: int,
) -> torch.Tensor:
    """One replay step per session: next queued label, or park.

    Returns ``(B,)`` long — the queued label at the head (advancing
    it), or ``park_id`` for a drained/empty queue (PORT-DEC-002/003;
    a blank-only chunk parks immediately, PORT-DEC-004).
    """
    heads = book_pool[state_indices, QUEUE_HEAD].long()
    lens = book_pool[state_indices, QUEUE_LEN].long()
    drained = heads >= lens
    # Clamp so the gather stays in-bounds for drained rows too — the
    # gathered value is discarded by torch.where for those rows.
    clamped_heads = torch.clamp(heads, max=queue_pool.shape[1] - 1)
    queued = queue_pool[state_indices, clamped_heads].long()
    park = torch.full_like(queued, park_id)
    labels = torch.where(drained, park, queued)
    new_heads = torch.where(drained, heads, heads + 1)
    book_pool[state_indices, QUEUE_HEAD] = new_heads.to(book_pool.dtype)
    return labels


def verify_replay_echo(observed: torch.Tensor, forced: torch.Tensor) -> None:
    """The model-owned sampling guard (PORT-DEC-007).

    On every replay step the token id the engine fed back must equal
    the id the decision carrier forced last step. A mismatch means
    something between compute_logits and the next forward corrupted
    the emission (hostile params, an exclusion mask, a processor) —
    the session is unrecoverable and must abort loudly
    (PORT-STATE-005 posture), never continue on a corrupted
    transcript.

    Raises:
        ValueError: On any per-session mismatch, naming both ids.
            (ValueError, not RuntimeError: NotImplementedError is a
            RuntimeError subclass, and the guard's negative test must
            never pass against an unimplemented stub.)

    """
    mismatch = observed != forced
    if bool(mismatch.any()):
        row = int(mismatch.nonzero(as_tuple=True)[0][0])
        raise ValueError(
            f"replay echo mismatch at row {row}: observed={int(observed[row])} "
            f"forced={int(forced[row])}"
        )
