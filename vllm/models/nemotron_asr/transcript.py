# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded exact transcript retention for streaming projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SegmentReason = Literal["model", "forced", "terminal"]


class OutputCapacityExceeded(ValueError):  # noqa: N818
    """A committed result cannot be projected without exceeding capacity."""


@dataclass(frozen=True)
class TranscriptProjection:
    """One result accepted atomically by the output authority."""

    delta: str


@dataclass(frozen=True)
class SegmentCompletion:
    """Model-neutral completion of one transcript segment."""

    generation: int
    text: str
    reason: SegmentReason


@dataclass(frozen=True)
class TerminalResult:
    """Complete terminal text plus the unreported-suffix acknowledgement."""

    complete_text: str
    completion: SegmentCompletion
    emit_segment_event: bool = False


@dataclass(frozen=True)
class TranscriptSnapshot:
    """Immutable retained-output state."""

    fragments: tuple[str, ...]
    retained_bytes: int
    segment_start: int
    segment_generation: int
    finished: bool


# @spec PORT-STATE-021, PORT-SEG-005, PORT-SEG-006, PORT-SEG-007
class BoundedTranscript:
    """Retain exact UTF-8 transcript fragments within one fixed budget."""

    def __init__(
        self,
        *,
        max_retained_bytes: int,
        fragment_overhead_bytes: int,
        terminal_headroom_bytes: int,
    ) -> None:
        if max_retained_bytes <= 0:
            raise ValueError("output capacity must be positive")
        if fragment_overhead_bytes < 0 or terminal_headroom_bytes < 0:
            raise ValueError("output overhead must be nonnegative")
        if max_retained_bytes <= terminal_headroom_bytes:
            raise ValueError("output capacity must cover terminal headroom")
        self.max_retained_bytes = max_retained_bytes
        self.fragment_overhead_bytes = fragment_overhead_bytes
        self.terminal_headroom_bytes = terminal_headroom_bytes
        self._fragments: list[str] = []
        self._retained_bytes = terminal_headroom_bytes
        self._segment_start = 0
        self._segment_generation = 0
        self._finished = False

    @property
    def complete_text(self) -> str:
        return "".join(self._fragments)

    @property
    def retained_bytes(self) -> int:
        return self._retained_bytes

    @property
    def retained_fragment_count(self) -> int:
        return len(self._fragments)

    @property
    def finished(self) -> bool:
        return self._finished

    def bytes_required_for_result(self, text: str) -> int:
        return (
            self.terminal_headroom_bytes
            + self.fragment_overhead_bytes
            + len(text.encode("utf-8"))
        )

    # @spec PORT-STATE-021, PORT-SEG-005
    def commit_result(self, text: str) -> TranscriptProjection:
        if self._finished:
            raise RuntimeError("terminal transcript is already finished")
        if not isinstance(text, str):
            raise ValueError("transcript result must be text")
        if not text:
            return TranscriptProjection(delta="")
        incremental = self.fragment_overhead_bytes + len(text.encode("utf-8"))
        if self._retained_bytes + incremental > self.max_retained_bytes:
            raise OutputCapacityExceeded(
                "output_capacity_exceeded: retained transcript budget exhausted"
            )
        self._fragments.append(text)
        self._retained_bytes += incremental
        return TranscriptProjection(delta=text)

    # @spec PORT-SEG-005, PORT-SEG-006, PORT-SEG-007
    def complete_segment(
        self,
        *,
        generation: int,
        reason: Literal["model", "forced"],
    ) -> SegmentCompletion | None:
        if generation <= 0:
            raise ValueError("segment generation must be positive")
        if reason not in ("model", "forced"):
            raise ValueError("segment reason must be model or forced")
        if self._finished:
            raise RuntimeError("terminal transcript is already finished")
        if generation <= self._segment_generation:
            return None
        if generation != self._segment_generation + 1:
            raise ValueError("segment generation is not contiguous")
        text = "".join(self._fragments[self._segment_start :])
        completion = SegmentCompletion(
            generation=generation,
            text=text,
            reason=reason,
        )
        self._segment_generation = generation
        self._segment_start = len(self._fragments)
        return completion

    # @spec PORT-SESS-003, PORT-SEG-005, PORT-SEG-006
    def finish_terminal(self) -> TerminalResult:
        if self._finished:
            raise RuntimeError("terminal transcript is already finished")
        complete = self.complete_text
        suffix = "".join(self._fragments[self._segment_start :])
        self._finished = True
        return TerminalResult(
            complete_text=complete,
            completion=SegmentCompletion(
                generation=self._segment_generation + 1,
                text=suffix,
                reason="terminal",
            ),
            emit_segment_event=False,
        )

    def snapshot(self) -> TranscriptSnapshot:
        return TranscriptSnapshot(
            fragments=tuple(self._fragments),
            retained_bytes=self._retained_bytes,
            segment_start=self._segment_start,
            segment_generation=self._segment_generation,
            finished=self._finished,
        )
