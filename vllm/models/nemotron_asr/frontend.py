# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The exact bounded incremental frontend (design §Exact Bounded Frontend).

An incremental implementation of the checkpoint's whole-utterance
pre-emphasis and centered STFT (``MelFeaturizer``): it retains only a
bounded raw-sample tail and the committed-boundary mel tail, and on
each CHUNK commits exactly the mel frames whose effective STFT window
is stable given the samples received — proven equal to the
whole-prefix featurizer by differential tests (``test_frontend.py``,
local: this module imports only ``torch``).

Frame geometry (hop 160, ``n_fft`` 512, center=True, constant pad):
frame ``k`` of the whole-signal STFT covers preemphasized samples
``[k*160 - 256, k*160 + 256)``; a frame is STABLE once sample
``k*160 + 255`` exists, i.e. ``k < (N - 256)//160 + 1`` for ``N``
received samples. The final tail instead commits all
``output_lengths(N) = N // 160`` frames, with positions at or past
``N`` contributing zeros — exactly the whole-signal mask + constant
pad. Pre-emphasis continuity costs one extra retained sample
(``y[t] = x[t] - p*x[t-1]``); an absent sample (index < 0) is zero,
which also reproduces ``y[0] = x[0]``.

Batching: one call serves one profile+geometry bucket and carries
mixed session-first / continuing / final / zero-frame rows — the
committed frame count is a per-row tensor, never a batch-shape
property. Every shape is host-derived (``pad_frames``, the raw-tail
capacity); no tensor value determines a shape, so the call issues NO
host/device synchronization (PORT-PERF-001/PORT-ADV-004).

Every per-row failure — protocol (caller-set sequence/geometry/
oversize bits, the frontend's own finalization state) and design
invariant (margin, target order, pad/retention overflow, negative
samples) — resolves to one bit of a device-resident int32 status; a
row with any bit set mutates nothing and commits nothing (a safe
masked no-op), and the transaction consumes the status at its commit
readback (PORT-STATE-008 tiers). The transition never raises on a
device predicate and never uses a device-side assertion: a fired CUDA
assert corrupts the context, converting one row's defect into the
loss of every resident session.

The frontend counters live in the ``(B, 8)`` int64 tensor whose slot
names/order are pinned by ``manifests.FRONTEND_COUNTER_FIELDS``; the
``CTR_*`` indices below mirror that tuple (cross-pinned by test).
"""

from __future__ import annotations

from typing import Any

import torch

#: Slot indices into the (B, 8) frontend counter tensor — MUST mirror
#: manifests.FRONTEND_COUNTER_FIELDS order (test-pinned).
(
    CTR_TOTAL_VALID_SAMPLES,
    CTR_COMMITTED_MEL_FRAMES,
    CTR_ENCODED_MEL_FRAMES,
    CTR_RAW_TAIL_ORIGIN,
    CTR_RAW_TAIL_LENGTH,
    CTR_MEL_TAIL_LENGTH,
    CTR_EXPECTED_CHUNK_SEQUENCE,
    CTR_FINALIZED,
) = range(8)

#: Per-row transition status bits (PORT-ADV-004): one int32 bitmask
#: per row, 0 == OK. The caller (``advance_session``) sets the
#: envelope-protocol bits it owns — GEOMETRY (envelope vs bucket),
#: SEQUENCE (vs the session's expected counter), FINAL_OVERSIZE (a
#: final residual of one cadence unit or more, illegal ingress per
#: PORT-SESS-001/003) — and ``advance_frontend`` sets FINALIZED plus
#: the design-invariant bits. Any set bit makes the row a masked
#: no-op; the transaction maps protocol bits to row-tier suppression
#: and invariant bits to port-defect escalation.
ROW_STATUS_GEOMETRY = 1
ROW_STATUS_SEQUENCE = 2
ROW_STATUS_FINALIZED = 4
ROW_STATUS_FINAL_OVERSIZE = 8
ROW_STATUS_NEGATIVE_SAMPLES = 16
ROW_STATUS_TARGET_ORDER = 32
ROW_STATUS_MARGIN = 64
ROW_STATUS_PAD_OVERFLOW = 128
ROW_STATUS_RAW_TAIL_OVERFLOW = 256

#: Mel frames retained before the committed boundary (the encoder's
#: pre-encode cache depth).
MEL_TAIL_FRAMES = 9


def stable_frames(n_samples: int, *, n_fft: int, hop: int) -> int:
    """Frames fully determined by ``n_samples`` received samples."""
    half = n_fft // 2
    if n_samples < half:
        return 0
    return (n_samples - half) // hop + 1


def final_frames(n_samples: int, *, n_fft: int, hop: int) -> int:
    """``MelFeaturizer.output_lengths`` for one element (final tail)."""
    return (n_samples + 2 * (n_fft // 2) - n_fft) // hop


def _gather_rows(
    idx: torch.Tensor,
    *,
    raw_tail: torch.Tensor,
    tail_origin: torch.Tensor,
    tail_length: torch.Tensor,
    samples: torch.Tensor,
    valid_samples: torch.Tensor,
    new_origin: torch.Tensor,
    valid_limit: torch.Tensor,
) -> torch.Tensor:
    """Batched absolute-index sample gather: ``out[b, j]`` is stream
    sample ``idx[b, j]``, sourced from the retained tail or the new
    samples, zero below 0 or at/past ``valid_limit[b]`` (the
    whole-signal mask + constant-pad semantics). The two sources are
    disjoint by construction (tail < new_origin <= new). Pure tensor
    ops — no per-row host work (PORT-PERF-001).
    """
    legal = (idx >= 0) & (idx < valid_limit.unsqueeze(1))
    out = torch.zeros_like(idx, dtype=raw_tail.dtype)
    tail_pos = idx - tail_origin.unsqueeze(1)
    in_tail = legal & (tail_pos >= 0) & (tail_pos < tail_length.unsqueeze(1))
    if raw_tail.shape[1] > 0:
        gathered = raw_tail.gather(1, tail_pos.clamp(0, raw_tail.shape[1] - 1))
        out = torch.where(in_tail, gathered, out)
    new_pos = idx - new_origin.unsqueeze(1)
    in_new = legal & (new_pos >= 0) & (new_pos < valid_samples.unsqueeze(1))
    if samples.shape[1] > 0:
        gathered = samples.gather(1, new_pos.clamp(0, samples.shape[1] - 1))
        out = torch.where(in_new, gathered, out)
    return out


def cadence_boundary(chunk_index: int, *, lookahead: int) -> int:
    """The approved cumulative encoder boundary after ``chunk_index``
    regular cadence units (design boundary formulas): ``B_k = (8L+1) +
    (k-1)·C`` mel frames with ``C = 8(L+1)`` — i.e. ``kC - 7``. Zero
    before the first unit."""
    if chunk_index <= 0:
        return 0
    cadence = 8 * (lookahead + 1)
    return (8 * lookahead + 1) + (chunk_index - 1) * cadence


def advance_frontend(
    featurizer: Any,
    samples: torch.Tensor,
    valid_samples: torch.Tensor,
    final_tail: torch.Tensor,
    target_frames: torch.Tensor,
    *,
    raw_tail: torch.Tensor,
    mel_tail: torch.Tensor,
    counters: torch.Tensor,
    cadence_frames: int,
    pad_frames: int,
    row_status: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance the bounded frontend for one profile+geometry bucket.

    Boundary-capped (the option-(d) resolution, ledger 2026-07-17):
    each row commits mel frames only through its cumulative cadence
    target (``cadence_boundary`` for regular chunks, derived by the
    caller from the admitted geometry and chunk sequence), never
    through every currently stable frame — stability yields a uniform
    6-frame margin over the ``kC-7`` boundary, and the mel tail is
    retained relative to the TARGET so the encoder's pre-encode cache
    always holds the frames before its actual boundary. Frames stable
    beyond the target stay in the raw tail and recompute identically
    on a later update (exactness is preserved). A final tail ignores
    ``target_frames`` and commits through ``final_frames`` under the
    separate residual rules.

    Length-aware (PORT-ADV-004): rows commit PER-ROW counts — mixed
    session-first / continuing / final / zero-frame rows share one
    call. The returned mel tensor has the fixed host-derived width
    ``pad_frames`` (the production bucket bound is one cadence ``C``,
    design §Exact Bounded Frontend); columns at or past a row's count
    are exactly zero. Finalization reproduces the reference buffer
    walk: at most one ``C``-frame cadence shift is committed, and the
    remaining sub-eight boundary debt is dropped; a session-first final
    is the walk's first chunk (one to ``C-7`` frames). Failure handling is
    the ``ROW_STATUS_*``
    bitmask: the caller's incoming protocol bits are AND-composed
    with the frontend-owned FINALIZED predicate and the
    design-invariant predicates, and any row with a set bit mutates
    nothing and commits nothing — a safe masked no-op, never a raise
    or a device assertion. No tensor value determines a shape and no
    host/device synchronization occurs.

    Args:
        featurizer: the checkpoint ``MelFeaturizer`` (supplies
            ``n_fft``/``hop_length``/``preemph``/``fb``/``window``/
            ``log_zero_guard``; the exact op order below matches its
            ``forward``).
        samples: ``(B, S)`` padded raw FP32 rows.
        valid_samples: ``(B,)`` true sample counts.
        final_tail: ``(B,)`` bool final-tail markers.
        target_frames: ``(B,)`` long cumulative cadence targets
            (ignored for final-tail rows).
        raw_tail: ``(B, R)`` retained-tail storage, updated in place.
        mel_tail: ``(B, n_mels, MEL_TAIL_FRAMES)`` committed-boundary
            mel tail, updated in place.
        counters: ``(B, 8)`` int64 counters, updated in place.
        cadence_frames: host-derived admitted cadence ``C`` in mel
            frames. Final rows consume at most one such shift.
        pad_frames: host-derived padded output width; a row whose
            commit would exceed it takes ROW_STATUS_PAD_OVERFLOW and
            masks (defense in depth under a caller-tightened bound).
        row_status: optional ``(B,)`` int32 incoming status carrying
            the caller-owned protocol bits; ``None`` means all rows
            arrive clean. Never mutated; the composed status is
            returned.

    Returns:
        The committed mel frames as ONE zero-padded
        ``(B, n_mels, pad_frames)`` tensor, the ``(B,)`` long per-row
        committed counts, and the ``(B,)`` int32 composed row status
        (0 == committed clean; any set ``ROW_STATUS_*`` bit == masked
        no-op).

    """
    n_fft = featurizer.n_fft
    hop = featurizer.hop_length
    preemph = featurizer.preemph
    half = n_fft // 2
    batch = samples.shape[0]
    capacity = raw_tail.shape[1]
    if cadence_frames <= 0:
        raise ValueError(f"cadence_frames must be positive, got {cadence_frames}")

    device = counters.device
    valid_samples = valid_samples.to(device=device, dtype=torch.long)
    is_final = final_tail.to(device=device, dtype=torch.bool)
    target_frames = target_frames.to(device=device, dtype=torch.long)

    # Snapshot every counter read: the column writes below mutate
    # the underlying storage, and basic-slice reads are VIEWS.
    finalized = counters[:, CTR_FINALIZED].clone()
    tail_origin0 = counters[:, CTR_RAW_TAIL_ORIGIN].clone()
    tail_length0 = counters[:, CTR_RAW_TAIL_LENGTH].clone()
    total_before = counters[:, CTR_TOTAL_VALID_SAMPLES].clone()
    committed = counters[:, CTR_COMMITTED_MEL_FRAMES].clone()
    mel_len0 = counters[:, CTR_MEL_TAIL_LENGTH].clone()

    # Compose the row status (PORT-ADV-004): incoming caller protocol
    # bits, the frontend-owned finalization state, then the design
    # invariants evaluated on still-clean rows against tentative
    # full-consumption values. Pure tensor arithmetic — no raise, no
    # device assertion, no synchronization.
    if row_status is None:
        status = torch.zeros(batch, dtype=torch.int32, device=device)
    else:
        status = row_status.to(device=device, dtype=torch.int32).clone()
    status |= (finalized != 0).to(torch.int32) * ROW_STATUS_FINALIZED
    status |= (valid_samples < 0).to(torch.int32) * ROW_STATUS_NEGATIVE_SAMPLES
    clean = status == 0
    regular = ~is_final

    # Tentative pass: values as if every clean row consumes fully.
    eff0 = torch.where(clean, valid_samples, valid_samples.new_zeros(()))
    total0 = total_before + eff0
    stable = torch.clamp((total0 - half) // hop + 1, min=0)
    # Final residual rule (PORT-FEAT-004): reproduce the reference
    # buffer walk. If a complete cadence shift is available, consume
    # exactly C and drop the resulting sub-eight boundary debt. With
    # less than C available, consume the actual remainder only when it
    # has at least eight frames. Finalization still marks atomically
    # when nothing is committed.
    #
    # A session-first final (nothing committed yet) is the walk's FIRST
    # chunk instead: it needs only one frame and spans at most the first
    # boundary C-7 (= 8L+1). Its remainder past that boundary is under
    # eight frames — a final residual is strictly under one cadence of
    # samples — so the walk drops it, exactly like boundary debt.
    n_final = total0 // hop  # == final_frames(total0)
    final_remaining = n_final - committed
    first_final_count = torch.clamp(
        torch.minimum(
            final_remaining,
            torch.full_like(final_remaining, cadence_frames - 7),
        ),
        min=0,
    )
    final_count = torch.where(
        committed == 0,
        first_final_count,
        torch.where(
            final_remaining >= cadence_frames,
            torch.full_like(final_remaining, cadence_frames),
            torch.where(
                final_remaining >= 8,
                final_remaining,
                torch.zeros_like(final_remaining),
            ),
        ),
    )
    final_target = committed + final_count
    tgt0 = torch.where(is_final, final_target, target_frames)
    chk = clean & regular
    status |= (chk & (target_frames < committed)).to(
        torch.int32
    ) * ROW_STATUS_TARGET_ORDER
    status |= (chk & (stable < target_frames)).to(torch.int32) * ROW_STATUS_MARGIN
    status |= (clean & (tgt0 - committed > pad_frames)).to(
        torch.int32
    ) * ROW_STATUS_PAD_OVERFLOW
    keep_from0 = torch.clamp(tgt0 * hop - half - 1, min=0)
    status |= (chk & (torch.clamp(total0 - keep_from0, min=0) > capacity)).to(
        torch.int32
    ) * ROW_STATUS_RAW_TAIL_OVERFLOW

    # Effective pass: rows with any bit set consume nothing and
    # commit nothing.
    ok = status == 0
    eff_samples = torch.where(ok, valid_samples, valid_samples.new_zeros(()))
    total = total_before + eff_samples
    target = torch.where(ok, tgt0, committed)
    counts = target - committed  # (B,) per-row commit, >= 0
    n_mels = featurizer.fb.shape[0]

    if pad_frames > 0:
        # Segment covering frames [committed, committed + pad_frames):
        # preemphasized samples from ``committed*hop - half``, plus one
        # leading raw sample for pre-emphasis continuity. Width is the
        # host constant; each row's valid columns are its own commit.
        seg_start = committed * hop - half
        seg_len = (pad_frames - 1) * hop + n_fft
        idx = (seg_start - 1).unsqueeze(1) + torch.arange(
            seg_len + 1, device=device
        ).unsqueeze(0)
        x = _gather_rows(
            idx,
            raw_tail=raw_tail,
            tail_origin=tail_origin0,
            tail_length=tail_length0,
            samples=samples,
            valid_samples=eff_samples,
            new_origin=total_before,
            valid_limit=total,
        )
        # Masked pre-emphasis: y[t] = x[t] - p*x[t-1]; the leading
        # extra sample supplies x[t-1] across the segment boundary.
        y = x[:, 1:] - preemph * x[:, :-1]
        # Zero re-mask: y at absolute positions >= the row's valid
        # limit must be exactly 0 (not -p*x[limit-1]).
        abs_pos = seg_start.unsqueeze(1) + torch.arange(
            seg_len, device=device
        ).unsqueeze(0)
        y = torch.where(abs_pos < total.unsqueeze(1), y, y.new_zeros(()))
        spec = torch.stft(
            y,
            n_fft=n_fft,
            hop_length=hop,
            win_length=featurizer.win_length,
            center=False,
            window=featurizer.window,
            return_complex=True,
        )
        magnitude = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1))
        power = magnitude.pow(2.0)
        mel = torch.matmul(featurizer.fb, power)
        new_frames = torch.log(mel + featurizer.log_zero_guard)[:, :, :pad_frames]
        # Padded-column zeroing: columns at or past the row's commit
        # are exactly zero (PORT-ADV-004).
        col = torch.arange(pad_frames, device=device).view(1, 1, -1)
        new_frames = torch.where(
            col < counts.view(-1, 1, 1),
            new_frames,
            new_frames.new_zeros(()),
        )
        # Per-row mel-tail advance: the last MEL_TAIL_FRAMES of
        # [old tail | this row's committed frames] — one batched
        # gather at each row's own offset. Zero-commit rows gather
        # their old tail back bit-identically.
        combined = torch.cat([mel_tail, new_frames], dim=2)
        tidx = (
            counts.view(-1, 1, 1)
            + torch.arange(MEL_TAIL_FRAMES, device=device).view(1, 1, -1)
        ).expand(batch, n_mels, MEL_TAIL_FRAMES)
        mel_tail.copy_(combined.gather(2, tidx))
    else:
        new_frames = torch.zeros(batch, n_mels, 0, device=device)

    counters[:, CTR_TOTAL_VALID_SAMPLES] = total
    counters[:, CTR_COMMITTED_MEL_FRAMES] = target
    counters[:, CTR_MEL_TAIL_LENGTH] = torch.where(
        ok, torch.clamp(target, max=MEL_TAIL_FRAMES), mel_len0
    )

    # Raw-tail retention: nonfinal rows retain from one sample before
    # the next frame's window; final rows clear. Fixed-width gather
    # (the capacity is the host constant); the per-row write mask is
    # the logical retention length. Masked rows recompute their prior
    # retention values exactly (target == committed, total unchanged)
    # and are excluded from the write anyway — a bit-level no-op.
    keep_from = torch.clamp(target * hop - half - 1, min=0)
    keep_len = torch.clamp(total - keep_from, min=0)
    retain_len = torch.where(regular, keep_len, keep_len.new_zeros(()))
    if batch and capacity:
        kidx = keep_from.unsqueeze(1) + torch.arange(capacity, device=device).unsqueeze(
            0
        )
        kept = _gather_rows(
            kidx,
            raw_tail=raw_tail.clone(),
            tail_origin=tail_origin0,
            tail_length=tail_length0,
            samples=samples,
            valid_samples=eff_samples,
            new_origin=total_before,
            valid_limit=total,
        )
        write = (ok & regular).unsqueeze(1) & (
            torch.arange(capacity, device=device).unsqueeze(0) < keep_len.unsqueeze(1)
        )
        raw_tail.copy_(torch.where(write, kept, raw_tail))
    counters[:, CTR_RAW_TAIL_ORIGIN] = torch.where(
        ok,
        torch.where(is_final, total, keep_from),
        tail_origin0,
    )
    counters[:, CTR_RAW_TAIL_LENGTH] = torch.where(ok, retain_len, tail_length0)
    counters[:, CTR_FINALIZED] = torch.where(
        ok & is_final, finalized.new_ones(()), finalized
    )
    return new_frames, counts, status
