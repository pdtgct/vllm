# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded accepted-audio ownership for one streaming session."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .manifests import ADMISSION_EPOCH_MODULUS_MS

UnitKind = Literal["regular", "final_tail", "forced_eou"]


@dataclass(frozen=True)
class AcceptedPiece:
    """Acknowledgement for one atomically accepted caller piece."""

    samples_accepted: int


@dataclass(frozen=True)
class ReadyAudioUnit:
    """One ordered scheduler unit owned until legal park or clear."""

    kind: UnitKind
    samples: np.ndarray
    logical_sequence: int
    carrier_sequence: int | None
    ready_at_ns: int
    admission_ms_mod: int

    @property
    def sample_count(self) -> int:
        return int(self.samples.shape[0])


@dataclass(frozen=True)
class OrdinaryReleaseTiming:
    """Canonical operands captured before an ordinary unit is dispatched."""

    true_ready_ns: int
    prior_actual_submission_ns: int
    cadence_ns: int
    eligible_ns: int


@dataclass(frozen=True)
class AcceptedAudioSnapshot:
    """Conservation snapshot for diagnostics and tests."""

    accepted_samples: int
    parked_samples: int
    cleared_samples: int
    residual_samples: int
    ready_samples: int
    in_flight_samples: int
    outstanding_samples: int
    finalizing: bool
    cleared: bool


# @spec PORT-SESS-001, PORT-SESS-003, PORT-SESS-013, PORT-SESS-014
class AcceptedAudioAuthority:
    """Own accepted audio, ordered controls, and sample credit.

    Caller pieces are copied into a segmented FIFO. Complete cadence units
    are materialized once, while incomplete residual remains segmented. A
    unit releases sample credit only through a matching park or terminal
    clear.
    """

    def __init__(
        self,
        *,
        request_id: str,
        engine_epoch: str,
        lease_generation: int,
        chunk_samples: int,
        capacity_samples: int,
        carrier_sequence_modulus: int,
        initial_logical_sequence: int = 0,
        max_session_samples: int | None = None,
        cadence_ns: int | None = None,
    ) -> None:
        if chunk_samples <= 0:
            raise ValueError("chunk_samples must be positive")
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be positive")
        if carrier_sequence_modulus <= 0:
            raise ValueError("carrier_sequence_modulus must be positive")
        if initial_logical_sequence < 0:
            raise ValueError("initial_logical_sequence must be nonnegative")
        if max_session_samples is not None and max_session_samples <= 0:
            raise ValueError("max_session_samples must be positive")
        if cadence_ns is not None and cadence_ns <= 0:
            raise ValueError("cadence_ns must be positive when set")

        self.request_id = request_id
        self.engine_epoch = engine_epoch
        self.lease_generation = lease_generation
        self.chunk_samples = chunk_samples
        self.capacity_samples = capacity_samples
        self.carrier_sequence_modulus = carrier_sequence_modulus
        self.max_session_samples = max_session_samples
        self.cadence_ns = cadence_ns

        self._lock = threading.Lock()
        self._terminal_cause: str | None = None
        self._progress_key = (-1, -1, -1)
        self._last_model_progress: tuple[int, int, int] | None = None
        self._tail_parked_sequence: int | None = None
        self._pieces: deque[np.ndarray] = deque()
        self._head = 0
        self._residual_samples = 0
        self._ready: deque[ReadyAudioUnit] = deque()
        self._in_flight: ReadyAudioUnit | None = None
        self._in_flight_submitted = False
        self._parked = asyncio.Event()
        self._parked.set()
        self.capture_service_timing = False
        self.observed_eligibility_ns: int | None = None
        self._prior_ordinary_submission_ns: int | None = None
        self._next_logical_sequence = initial_logical_sequence
        self._accepted_samples = 0
        self._parked_samples = 0
        self._cleared_samples = 0
        self._finalizing = False
        self._cleared = False
        self._force_pending = False
        self._locale: str | None = None

    # @spec PORT-SESS-018
    def claim_terminal(self, cause: str, *, lease_generation: int) -> bool:
        """Linearize all terminal causes with acceptance and finalize."""
        with self._lock:
            if (
                lease_generation != self.lease_generation
                or self._terminal_cause is not None
            ):
                return False
            self._terminal_cause = cause
            return True

    @property
    def terminal_cause(self) -> str | None:
        with self._lock:
            return self._terminal_cause

    # @spec PORT-SESS-016
    def observe_model_progress(
        self,
        request_id: str,
        generation: int,
        raw: tuple[int, int, int],
    ) -> tuple[int, int, int] | None:
        with self._lock:
            if (
                request_id != self.request_id
                or generation != self.lease_generation
                or self._terminal_cause is not None
                or raw == self._last_model_progress
            ):
                return None
            sequence, cursor, phase = raw
            if cursor < 0 or phase not in (1, 2, 3):
                return None
            unit = self._in_flight
            if phase == 3:
                if (
                    not self._finalizing
                    or self._tail_parked_sequence is None
                    or unit is not None
                ):
                    return None
                logical = self._tail_parked_sequence + 1
                if sequence != logical:
                    return None
            elif unit is None:
                return None
            else:
                if sequence != unit.logical_sequence:
                    return None
                logical = unit.logical_sequence
            key = (logical, cursor, phase)
            if key <= self._progress_key:
                return None
            self._last_model_progress = raw
            self._progress_key = key
            return key

    @property
    def finalization_drained(self) -> bool:
        with self._lock:
            return (
                self._finalizing
                and self._tail_parked_sequence is not None
                and self._outstanding_samples() == 0
                and not self._ready
                and self._in_flight is None
            )

    def finalization_work(self) -> tuple[int, float]:
        """Regular work excludes the residual already minted as final tail."""
        with self._lock:
            units = list(self._ready)
            if self._in_flight is not None:
                units.append(self._in_flight)
            samples = sum(unit.sample_count for unit in units if unit.kind == "regular")
            delay = 0.0
            if (
                self.cadence_ns is not None
                and self._prior_ordinary_submission_ns is not None
            ):
                delay = (
                    max(
                        0,
                        self._prior_ordinary_submission_ns
                        + self.cadence_ns
                        - time.monotonic_ns(),
                    )
                    / 1e9
                )
            return samples, delay

    @property
    def ready_units(self) -> tuple[ReadyAudioUnit, ...]:
        with self._lock:
            return tuple(self._ready)

    @property
    def in_flight_unit(self) -> ReadyAudioUnit | None:
        """The ordered unit awaiting legal park, if one exists."""
        with self._lock:
            return self._in_flight

    async def wait_until_parked(self) -> None:
        """Wait for the stream consumer to acknowledge the current legal park."""
        await self._parked.wait()
        with self._lock:
            if self._cleared or self._terminal_cause is not None:
                raise RuntimeError("accepted audio terminated before legal park")

    @property
    def prior_actual_submission_ns(self) -> int | None:
        with self._lock:
            return self._prior_ordinary_submission_ns

    # @spec PORT-STATE-026
    def release_timing(self, unit: ReadyAudioUnit) -> OrdinaryReleaseTiming:
        """Snapshot readiness and pacing without dispatching a queued CHUNK."""
        with self._lock:
            if (
                unit.kind == "forced_eou"
                or not any(queued is unit for queued in self._ready)
                or self._prior_ordinary_submission_ns is None
                or self.cadence_ns is None
            ):
                raise ValueError(
                    "release timing requires a queued ordinary CHUNK after registration"
                )
            eligible = self._eligibility_ns(unit)
            assert eligible is not None
            return OrdinaryReleaseTiming(
                true_ready_ns=unit.ready_at_ns,
                prior_actual_submission_ns=self._prior_ordinary_submission_ns,
                cadence_ns=self.cadence_ns,
                eligible_ns=eligible,
            )

    def _eligibility_ns(self, unit: ReadyAudioUnit) -> int | None:
        if self.cadence_ns is None or unit.kind == "forced_eou":
            return None
        if self._prior_ordinary_submission_ns is None:
            return unit.ready_at_ns
        return max(
            unit.ready_at_ns,
            self._prior_ordinary_submission_ns + self.cadence_ns,
        )

    @property
    def next_eligibility_ns(self) -> int | None:
        """Earliest release of the FIFO head under the cadence clock."""
        with self._lock:
            if (
                self._cleared
                or self._terminal_cause is not None
                or self._in_flight is not None
                or not self._ready
            ):
                return None
            return self._eligibility_ns(self._ready[0])

    def _outstanding_samples(self) -> int:
        ready = sum(unit.sample_count for unit in self._ready)
        in_flight = 0 if self._in_flight is None else self._in_flight.sample_count
        return self._residual_samples + ready + in_flight

    def _new_unit(
        self,
        kind: UnitKind,
        samples: np.ndarray,
        ready_at_ns: int,
        admission_ms_mod: int,
    ) -> ReadyAudioUnit:
        logical = self._next_logical_sequence
        self._next_logical_sequence += 1
        carrier = None
        if kind != "forced_eou":
            carrier = logical % self.carrier_sequence_modulus
        return ReadyAudioUnit(
            kind=kind,
            samples=samples,
            logical_sequence=logical,
            carrier_sequence=carrier,
            ready_at_ns=ready_at_ns,
            admission_ms_mod=admission_ms_mod,
        )

    def _take_samples(self, count: int) -> np.ndarray:
        output = np.empty(count, dtype=np.float32)
        written = 0
        while written < count:
            piece = self._pieces[0]
            available = piece.shape[0] - self._head
            take = min(count - written, available)
            output[written : written + take] = piece[self._head : self._head + take]
            written += take
            self._head += take
            if self._head == piece.shape[0]:
                self._pieces.popleft()
                self._head = 0
        self._residual_samples -= count
        return output

    # @spec PORT-SESS-001, PORT-SESS-014
    def accept(
        self,
        samples: np.ndarray,
        *,
        accepted_at_ns: int | None = None,
        admission_ms_mod: int | None = None,
    ) -> AcceptedPiece:
        if not isinstance(samples, np.ndarray):
            raise ValueError("audio must be a mono FP32 numpy array")
        if samples.ndim != 1:
            raise ValueError("audio must be mono")
        if samples.dtype != np.float32:
            raise ValueError("audio must be FP32")
        if samples.size == 0:
            raise ValueError("audio piece must be nonempty")
        if not np.isfinite(samples).all():
            raise ValueError("audio samples must be finite")

        owned = np.array(samples, dtype=np.float32, order="C", copy=True)
        ready_at_ns = time.monotonic_ns() if accepted_at_ns is None else accepted_at_ns
        if admission_ms_mod is None:
            admission_ms_mod = int(time.time() * 1000) % ADMISSION_EPOCH_MODULUS_MS
        sample_count = int(owned.shape[0])

        with self._lock:
            if self._cleared or self._terminal_cause is not None:
                raise ValueError("session is cleared")
            if self._finalizing:
                raise ValueError("session is finalizing")
            if self._outstanding_samples() + sample_count > self.capacity_samples:
                raise ValueError("buffer_overflow: accepted-audio capacity exceeded")
            if (
                self.max_session_samples is not None
                and self._accepted_samples + sample_count > self.max_session_samples
            ):
                raise ValueError("session_duration: accepted-audio limit exceeded")

            self._pieces.append(owned)
            self._residual_samples += sample_count
            self._accepted_samples += sample_count
            while self._residual_samples >= self.chunk_samples:
                chunk = self._take_samples(self.chunk_samples)
                self._ready.append(
                    self._new_unit(
                        "regular",
                        chunk,
                        ready_at_ns,
                        admission_ms_mod,
                    )
                )
            return AcceptedPiece(samples_accepted=sample_count)

    # @spec PORT-SEG-004, PORT-SESS-014
    def force_segment(self) -> None:
        with self._lock:
            if self._cleared or self._terminal_cause is not None:
                raise ValueError("session is cleared")
            if self._finalizing:
                raise ValueError("session is finalizing")
            if self._force_pending:
                return
            empty = np.empty(0, dtype=np.float32)
            self._ready.append(
                self._new_unit(
                    "forced_eou",
                    empty,
                    time.monotonic_ns(),
                    int(time.time() * 1000) % ADMISSION_EPOCH_MODULUS_MS,
                )
            )
            self._force_pending = True

    # @spec PORT-LID-001, PORT-SESS-014
    def update_locale(self, locale: str) -> str:
        with self._lock:
            if self._cleared or self._terminal_cause is not None:
                raise ValueError("session is cleared")
            if self._finalizing:
                raise ValueError("session is finalizing")
            self._locale = locale
            return locale

    # @spec PORT-SESS-003, PORT-SESS-014
    def begin_finalize(
        self,
        *,
        finalize_at_ns: int | None = None,
        admission_ms_mod: int | None = None,
    ) -> None:
        ready_at_ns = time.monotonic_ns() if finalize_at_ns is None else finalize_at_ns
        if admission_ms_mod is None:
            admission_ms_mod = int(time.time() * 1000) % ADMISSION_EPOCH_MODULUS_MS
        with self._lock:
            if self._cleared or self._terminal_cause is not None:
                raise ValueError("session is cleared")
            if self._finalizing:
                return
            self._finalizing = True
            tail = self._take_samples(self._residual_samples)
            self._ready.append(
                self._new_unit(
                    "final_tail",
                    tail,
                    ready_at_ns,
                    admission_ms_mod,
                )
            )

    # @spec PORT-SESS-001, PORT-SESS-013
    def dispatch_next(self, *, now_ns: int | None = None) -> ReadyAudioUnit | None:
        with self._lock:
            if (
                self._cleared
                or self._terminal_cause is not None
                or self._in_flight is not None
                or not self._ready
            ):
                return None
            eligibility_ns = self._eligibility_ns(self._ready[0])
            if eligibility_ns is not None:
                current_ns = time.monotonic_ns() if now_ns is None else now_ns
                if current_ns < eligibility_ns:
                    return None
            if self.capture_service_timing:
                self.observed_eligibility_ns = eligibility_ns
            self._parked.clear()
            self._in_flight = self._ready.popleft()
            self._in_flight_submitted = False
            return self._in_flight

    # @spec PORT-SESS-001, PORT-STATE-026
    def record_submission(
        self,
        unit: ReadyAudioUnit,
        *,
        submitted_at_ns: int | None = None,
    ) -> None:
        """Advance the ordinary release clock from actual submission."""
        submitted_ns = (
            time.monotonic_ns() if submitted_at_ns is None else submitted_at_ns
        )
        with self._lock:
            if self._in_flight is not unit:
                raise ValueError("submission does not match in-flight unit")
            if self._in_flight_submitted:
                raise ValueError("in-flight unit was already submitted")
            if unit.kind != "forced_eou":
                self._prior_ordinary_submission_ns = submitted_ns
            self._in_flight_submitted = True

    # @spec PORT-INT-004, PORT-SESS-013
    def park(
        self,
        *,
        request_id: str,
        engine_epoch: str,
        lease_generation: int,
        logical_sequence: int,
        carrier_sequence: int | None,
    ) -> None:
        with self._lock:
            unit = self._in_flight
            if unit is None:
                raise ValueError("park has no in-flight unit")
            expected = (
                self.request_id,
                self.engine_epoch,
                self.lease_generation,
                unit.logical_sequence,
                unit.carrier_sequence,
            )
            actual = (
                request_id,
                engine_epoch,
                lease_generation,
                logical_sequence,
                carrier_sequence,
            )
            if actual != expected:
                raise ValueError("park identity or generation mismatch")
            if self.cadence_ns is not None and not self._in_flight_submitted:
                raise ValueError("paced unit reached park before submission")
            if unit.kind == "final_tail":
                self._tail_parked_sequence = unit.logical_sequence
            self._parked_samples += unit.sample_count
            if unit.kind == "forced_eou":
                self._force_pending = False
            self._in_flight = None
            self._in_flight_submitted = False
            self._parked.set()

    # @spec PORT-SESS-013
    def clear(self, error: BaseException) -> None:
        del error
        with self._lock:
            if self._cleared:
                return
            self._cleared_samples += self._outstanding_samples()
            self._pieces.clear()
            self._head = 0
            self._residual_samples = 0
            self._ready.clear()
            self._in_flight = None
            self._in_flight_submitted = False
            self._parked.set()
            self._force_pending = False
            self._cleared = True

    def snapshot(self) -> AcceptedAudioSnapshot:
        with self._lock:
            ready = sum(unit.sample_count for unit in self._ready)
            in_flight = 0 if self._in_flight is None else self._in_flight.sample_count
            outstanding = self._residual_samples + ready + in_flight
            return AcceptedAudioSnapshot(
                accepted_samples=self._accepted_samples,
                parked_samples=self._parked_samples,
                cleared_samples=self._cleared_samples,
                residual_samples=self._residual_samples,
                ready_samples=ready,
                in_flight_samples=in_flight,
                outstanding_samples=outstanding,
                finalizing=self._finalizing,
                cleared=self._cleared,
            )
