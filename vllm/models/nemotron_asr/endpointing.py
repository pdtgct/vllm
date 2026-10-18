# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded cache-aware RNN-T endpointing."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Literal

import torch


class FrameSymbol(IntEnum):
    """One dense endpoint symbol for one valid encoder frame."""

    BLANK = 0
    WORD_START = 1
    NON_WORD_START = 2


@dataclass(frozen=True)
class EndpointPolicy:
    """Immutable endpoint policy resolved at session admission."""

    mode: Literal["disabled", "greedy_blank"]
    stop_history_ms: int | None
    threshold_frames: int
    residue_frames: int
    frame_stride_ms: int
    active_window_frames: int
    history_capacity_frames: int

    # @spec PORT-SEG-001
    @classmethod
    def resolve(
        cls,
        *,
        mode: str,
        stop_history_ms: int | None,
        residue_frames: int,
        frame_stride_ms: int,
        history_capacity_frames: int,
    ) -> EndpointPolicy:
        if frame_stride_ms <= 0:
            raise ValueError("frame stride must be positive")
        if history_capacity_frames <= 0:
            raise ValueError("endpoint history capacity must be positive")
        if mode == "disabled":
            if stop_history_ms is not None or residue_frames != 0:
                raise ValueError("disabled endpoint mode has no history policy")
            return cls(
                mode="disabled",
                stop_history_ms=None,
                threshold_frames=0,
                residue_frames=0,
                frame_stride_ms=frame_stride_ms,
                active_window_frames=0,
                history_capacity_frames=history_capacity_frames,
            )
        if mode != "greedy_blank":
            raise ValueError(f"unsupported endpoint mode: {mode}")
        if stop_history_ms is None or stop_history_ms <= 0:
            raise ValueError("stop_history_ms must be positive")
        if residue_frames < 0:
            raise ValueError("residue_frames must be nonnegative")
        threshold = math.ceil(stop_history_ms / frame_stride_ms)
        active = threshold + residue_frames
        if history_capacity_frames < active:
            raise ValueError("endpoint history capacity is below active window")
        return cls(
            mode="greedy_blank",
            stop_history_ms=stop_history_ms,
            threshold_frames=threshold,
            residue_frames=residue_frames,
            frame_stride_ms=frame_stride_ms,
            active_window_frames=active,
            history_capacity_frames=history_capacity_frames,
        )


@dataclass(frozen=True)
class EndpointBook:
    """Fixed ring plus segment-generation state."""

    history: tuple[FrameSymbol, ...]
    history_length: int
    history_head: int
    endpoint_armed: bool
    segment_generation: int
    segment_has_output: bool
    pending_forced_generation: int

    @classmethod
    def initial(cls, *, history_capacity_frames: int) -> EndpointBook:
        if history_capacity_frames <= 0:
            raise ValueError("endpoint history capacity must be positive")
        return cls(
            history=(FrameSymbol.BLANK,) * history_capacity_frames,
            history_length=0,
            history_head=0,
            endpoint_armed=False,
            segment_generation=0,
            segment_has_output=False,
            pending_forced_generation=0,
        )

    def newest(self, count: int) -> tuple[FrameSymbol, ...]:
        count = min(max(count, 0), self.history_length)
        start = (self.history_head - count) % len(self.history)
        return tuple(
            self.history[(start + offset) % len(self.history)]
            for offset in range(count)
        )

    def append(self, symbol: FrameSymbol) -> EndpointBook:
        history = list(self.history)
        history[self.history_head] = symbol
        return replace(
            self,
            history=tuple(history),
            history_head=(self.history_head + 1) % len(history),
            history_length=min(self.history_length + 1, len(history)),
            endpoint_armed=self.endpoint_armed or symbol != FrameSymbol.BLANK,
            segment_has_output=(self.segment_has_output or symbol != FrameSymbol.BLANK),
        )


@dataclass(frozen=True)
class DetectorTrace:
    """Internal geometry used to audit the reference-equivalent call."""

    active_window_length: int
    pivot: int
    effective_end: int
    silent_frames: int


@dataclass(frozen=True)
class EndpointTransition:
    """Candidate post-CHUNK endpoint state and ordered labels."""

    emitted_labels: tuple[int, ...]
    observed_symbols: tuple[FrameSymbol, ...]
    next_book: EndpointBook
    detector_trace: DetectorTrace
    is_eou: bool


@dataclass(frozen=True)
class EndpointCompletion:
    generation: int
    reason: Literal["forced"]


@dataclass(frozen=True)
class ForcedEndpointTransition:
    output_ids: tuple[int, ...]
    reason: Literal["forced"]
    empty_segment: bool
    coalesced: bool
    completion: EndpointCompletion | None
    next_book: EndpointBook


@dataclass(frozen=True)
class TensorEndpointTransition:
    """GPU-resident candidate state for one CHUNK bucket."""

    history: torch.Tensor
    book: torch.Tensor
    token_ids: torch.Tensor
    token_lengths: torch.Tensor
    is_eou: torch.Tensor
    overflow: torch.Tensor


@dataclass(frozen=True)
class ForcedTensorEndpointTransition:
    """GPU-resident endpoint-book transition for an ordered force."""

    book: torch.Tensor
    is_eou: torch.Tensor


# @spec PORT-SEG-004, PORT-SEG-007
def apply_forced_eou_tensors(
    *,
    book: torch.Tensor,
    selected_rows: torch.Tensor,
) -> ForcedTensorEndpointTransition:
    """Apply a forced boundary after all earlier CHUNK work has parked.

    The accepted-audio authority admits at most one ordered force at a
    time.  A force advances the segment generation only when the current
    segment has output; otherwise it is an empty, park-only no-op.  The
    transition remains a tensor transform so it joins the same atomic
    resident scatter as acoustic CHUNK state.
    """
    if book.dim() != 2 or book.shape[1] != 6:
        raise ValueError("endpoint book must match its six-slot manifest")
    if tuple(selected_rows.shape) != (book.shape[0],):
        raise ValueError("forced endpoint selection must be one bit per row")
    if selected_rows.dtype != torch.bool or selected_rows.device != book.device:
        raise ValueError("forced endpoint selection must be device-local bool")

    next_book = book.clone()
    has_output = next_book[:, 4] != 0
    emit = selected_rows & has_output
    next_book[:, :2] = torch.where(
        emit.unsqueeze(1),
        torch.zeros_like(next_book[:, :2]),
        next_book[:, :2],
    )
    next_book[:, 2] = torch.where(
        selected_rows,
        torch.zeros_like(next_book[:, 2]),
        next_book[:, 2],
    )
    next_book[:, 3] = torch.where(
        emit,
        next_book[:, 3] + 1,
        next_book[:, 3],
    )
    next_book[:, 4] = torch.where(
        selected_rows,
        torch.zeros_like(next_book[:, 4]),
        next_book[:, 4],
    )
    next_book[:, 5] = torch.where(
        selected_rows,
        torch.zeros_like(next_book[:, 5]),
        next_book[:, 5],
    )
    return ForcedTensorEndpointTransition(book=next_book, is_eou=emit)


def _count_silent_before_pivot(
    *,
    history: torch.Tensor,
    length: torch.Tensor,
    head: torch.Tensor,
    enabled: torch.Tensor,
    active_window: torch.Tensor,
) -> torch.Tensor:
    """Count consecutive blanks before the newest pivot without host readback."""
    rows, capacity = history.shape
    silent = torch.zeros(rows, dtype=torch.long, device=history.device)
    still_silent = torch.ones(rows, dtype=torch.bool, device=history.device)
    # Offset one is the pivot itself. The reference starts at pivot-1.
    for offset in range(2, capacity + 1):
        within = offset <= torch.minimum(length, active_window)
        index = (head - offset) % capacity
        symbol = history.gather(1, index.unsqueeze(1)).squeeze(1)
        blank = symbol == int(FrameSymbol.BLANK)
        count = enabled & within & still_silent & blank
        silent += count.long()
        still_silent &= ~within | blank
    return silent


# @spec PORT-SEG-002, PORT-SEG-003, PORT-SEG-007
def observe_chunk_tensors(
    *,
    history: torch.Tensor,
    book: torch.Tensor,
    frame_emission_counts: torch.Tensor,
    valid_frame_lengths: torch.Tensor,
    token_ids: torch.Tensor,
    token_lengths: torch.Tensor,
    final_tail: torch.Tensor,
    mode: torch.Tensor,
    threshold_frames: torch.Tensor,
    residue_frames: torch.Tensor,
    eou_token_id: int,
    row_clean: torch.Tensor,
) -> TensorEndpointTransition:
    """Tensor equivalent of :func:`observe_chunk` without host readback."""
    if history.dim() != 2 or book.dim() != 2 or book.shape[1] != 6:
        raise ValueError("endpoint resident tensors have invalid shape")
    rows, capacity = history.shape
    frame_rows, frame_width = frame_emission_counts.shape
    if frame_rows != rows or tuple(valid_frame_lengths.shape) != (rows,):
        raise ValueError("endpoint frame tensors have invalid shape")
    if tuple(token_lengths.shape) != (rows,) or token_ids.shape[0] != rows:
        raise ValueError("endpoint token tensors have invalid shape")
    if any(
        tuple(value.shape) != (rows,)
        for value in (
            final_tail,
            mode,
            threshold_frames,
            residue_frames,
            row_clean,
        )
    ):
        raise ValueError("endpoint policy tensors have invalid shape")

    next_history = history.clone()
    next_book = book.clone()
    length = next_book[:, 0].long()
    head = next_book[:, 1].long()
    armed = next_book[:, 2] != 0
    generation = next_book[:, 3].long()
    has_output = next_book[:, 4] != 0
    enabled = mode == 1
    regular = ~final_tail
    valid_policy = (
        ((mode == 0) | (mode == 1))
        & (threshold_frames >= 0)
        & (residue_frames >= 0)
        & (threshold_frames + residue_frames <= capacity)
    )
    active_row = row_clean & regular & valid_policy

    last_symbol = torch.zeros(rows, dtype=torch.int32, device=history.device)
    saw_frame = torch.zeros(rows, dtype=torch.bool, device=history.device)
    for frame in range(frame_width):
        valid = active_row & (frame < valid_frame_lengths)
        nonblank = frame_emission_counts[:, frame] > 0
        symbol = torch.where(
            nonblank,
            torch.full_like(last_symbol, int(FrameSymbol.NON_WORD_START)),
            torch.full_like(last_symbol, int(FrameSymbol.BLANK)),
        )
        write = valid & enabled
        current = next_history.gather(
            1, head.clamp(0, capacity - 1).unsqueeze(1)
        ).squeeze(1)
        next_history.scatter_(
            1,
            head.clamp(0, capacity - 1).unsqueeze(1),
            torch.where(write, symbol, current).unsqueeze(1),
        )
        head = torch.where(write, (head + 1) % capacity, head)
        length = torch.where(write, torch.clamp(length + 1, max=capacity), length)
        observed_output = valid & nonblank
        armed |= observed_output
        has_output |= observed_output
        last_symbol = torch.where(valid, symbol, last_symbol)
        saw_frame |= valid

    active_window = threshold_frames + residue_frames
    silent = _count_silent_before_pivot(
        history=next_history,
        length=length,
        head=head,
        enabled=enabled,
        active_window=active_window,
    )

    is_eou = (
        active_row
        & enabled
        & armed
        & has_output
        & saw_frame
        & ((last_symbol == int(FrameSymbol.BLANK)) | (threshold_frames > 0))
        & (silent > threshold_frames)
    )
    label_width = int(token_ids.shape[1])
    # PORT-DEC-005 reserves one control column beyond the checkpoint's
    # complete nonblank-label capacity.  Endpointing must never steal the
    # last legal RNN-T label slot merely because the same CHUNK closes a
    # semantic segment.
    width = label_width + 1
    next_tokens = torch.zeros(
        rows,
        width,
        dtype=token_ids.dtype,
        device=token_ids.device,
    )
    if label_width:
        next_tokens[:, :label_width] = token_ids
    overflow = is_eou & (token_lengths.long() >= width)
    safe_index = token_lengths.long().clamp(min=0, max=width - 1)
    if width:
        previous = next_tokens.gather(1, safe_index.unsqueeze(1)).squeeze(1)
        next_tokens.scatter_(
            1,
            safe_index.unsqueeze(1),
            torch.where(
                is_eou & ~overflow,
                torch.full_like(previous, eou_token_id),
                previous,
            ).unsqueeze(1),
        )
    next_lengths = token_lengths + (is_eou & ~overflow).to(token_lengths.dtype)
    generation += is_eou.long()
    armed &= ~is_eou
    has_output &= ~is_eou
    next_history.masked_fill_(is_eou.unsqueeze(1), int(FrameSymbol.BLANK))
    length = torch.where(is_eou, torch.zeros_like(length), length)
    head = torch.where(is_eou, torch.zeros_like(head), head)
    next_book[:, 0] = length.to(next_book.dtype)
    next_book[:, 1] = head.to(next_book.dtype)
    next_book[:, 2] = armed.to(next_book.dtype)
    next_book[:, 3] = generation.to(next_book.dtype)
    next_book[:, 4] = has_output.to(next_book.dtype)
    return TensorEndpointTransition(
        history=next_history,
        book=next_book,
        token_ids=next_tokens,
        token_lengths=next_lengths,
        is_eou=is_eou,
        overflow=overflow | ~valid_policy,
    )


def frame_symbol(
    labels: Sequence[int],
    *,
    is_word_start: Callable[[int], bool],
) -> FrameSymbol:
    if not labels:
        return FrameSymbol.BLANK
    return (
        FrameSymbol.WORD_START
        if is_word_start(int(labels[-1]))
        else FrameSymbol.NON_WORD_START
    )


def _detect(book: EndpointBook, policy: EndpointPolicy) -> DetectorTrace:
    window = book.newest(policy.active_window_frames)
    if not window:
        return DetectorTrace(0, -1, 0, 0)
    pivot = len(window) - 1
    effective_end = max(0, len(window) - policy.residue_frames)
    silent = 0
    index = pivot - 1
    while index >= 0 and window[index] == FrameSymbol.BLANK:
        silent += 1
        index -= 1
    return DetectorTrace(
        active_window_length=len(window),
        pivot=pivot,
        effective_end=effective_end,
        silent_frames=silent,
    )


# @spec PORT-DEC-001, PORT-SEG-002, PORT-SEG-003, PORT-SEG-007
def observe_chunk(
    book: EndpointBook,
    frames: Sequence[Sequence[int]],
    *,
    is_word_start: Callable[[int], bool],
    policy: EndpointPolicy,
    final_tail: bool = False,
) -> EndpointTransition:
    emitted = tuple(int(label) for labels in frames for label in labels)
    symbols = tuple(
        frame_symbol(labels, is_word_start=is_word_start) for labels in frames
    )
    candidate = book
    if not final_tail:
        for symbol in symbols:
            if policy.mode == "greedy_blank":
                candidate = candidate.append(symbol)
            elif symbol != FrameSymbol.BLANK:
                candidate = replace(
                    candidate,
                    endpoint_armed=True,
                    segment_has_output=True,
                )

    trace = (
        _detect(candidate, policy)
        if policy.mode == "greedy_blank" and not final_tail
        else DetectorTrace(0, -1, 0, 0)
    )
    is_eou = (
        policy.mode == "greedy_blank"
        and not final_tail
        and candidate.endpoint_armed
        and candidate.segment_has_output
        and bool(symbols)
        and (symbols[-1] == FrameSymbol.BLANK or policy.threshold_frames > 0)
        and trace.silent_frames > policy.threshold_frames
    )
    if is_eou:
        candidate = replace(
            candidate,
            history=(FrameSymbol.BLANK,) * len(candidate.history),
            history_length=0,
            history_head=0,
            endpoint_armed=False,
            segment_generation=candidate.segment_generation + 1,
            segment_has_output=False,
        )
    return EndpointTransition(
        emitted_labels=emitted,
        observed_symbols=symbols,
        next_book=candidate,
        detector_trace=trace,
        is_eou=is_eou,
    )


def compose_chunk_output(
    *,
    labels: Sequence[int],
    is_eou: bool,
    eou_token_id: int,
    park_token_id: int,
) -> list[int]:
    output = [int(label) for label in labels]
    if is_eou:
        output.append(eou_token_id)
    output.append(park_token_id)
    return output


def max_emission_tokens(
    *,
    max_valid_frames: int,
    max_symbols_per_step: int,
) -> int:
    return max_valid_frames * max_symbols_per_step + 2


def validate_emission_budget(
    *,
    caller_max_tokens: int,
    max_valid_frames: int,
    max_symbols_per_step: int,
) -> int:
    required = max_emission_tokens(
        max_valid_frames=max_valid_frames,
        max_symbols_per_step=max_symbols_per_step,
    )
    if caller_max_tokens < required:
        raise ValueError(f"emission budget requires {required} tokens")
    return required


def queue_forced_eou(
    book: EndpointBook,
    *,
    request_generation: int,
) -> EndpointBook:
    if request_generation <= 0:
        raise ValueError("request generation must be positive")
    if book.pending_forced_generation == request_generation:
        return book
    return replace(book, pending_forced_generation=request_generation)


# @spec PORT-SEG-004, PORT-SEG-007
def apply_forced_eou(
    book: EndpointBook,
    *,
    request_generation: int,
    expected_request_generation: int,
    eou_token_id: int,
    park_token_id: int,
) -> ForcedEndpointTransition | None:
    if request_generation != expected_request_generation:
        return None
    if (
        book.pending_forced_generation == request_generation
        and not book.segment_has_output
        and book.segment_generation > 0
    ):
        next_book = replace(book, pending_forced_generation=0)
        return ForcedEndpointTransition(
            output_ids=(park_token_id,),
            reason="forced",
            empty_segment=True,
            coalesced=True,
            completion=None,
            next_book=next_book,
        )
    if not book.segment_has_output:
        next_book = replace(book, pending_forced_generation=0)
        return ForcedEndpointTransition(
            output_ids=(park_token_id,),
            reason="forced",
            empty_segment=True,
            coalesced=False,
            completion=None,
            next_book=next_book,
        )
    generation = book.segment_generation + 1
    next_book = replace(
        book,
        history_length=0,
        history_head=0,
        endpoint_armed=False,
        segment_generation=generation,
        segment_has_output=False,
        pending_forced_generation=0,
    )
    return ForcedEndpointTransition(
        output_ids=(eou_token_id, park_token_id),
        reason="forced",
        empty_segment=False,
        coalesced=False,
        completion=EndpointCompletion(generation=generation, reason="forced"),
        next_book=next_book,
    )
