# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for the bounded accepted-audio authority.

These tests deliberately distinguish the serving FIFO from the encoder's
sliding history. Accepted audio is lossless: a full FIFO rejects a whole new
piece before mutation; only legal park or one terminal clear releases sample
credit. The fixed encoder window is tested separately in ``test_encoder.py``.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, NoReturn

import numpy as np
import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]


def _fail(message: str) -> NoReturn:
    pytest.fail(message, pytrace=False)
    raise AssertionError(message)


def _module() -> Any:
    try:
        return importlib.import_module("vllm.models.nemotron_asr.accepted_audio")
    except ModuleNotFoundError:
        _fail("PORT-SESS-001 missing the bounded accepted-audio authority")


def _authority(
    *,
    chunk_samples: int = 4,
    capacity_samples: int = 12,
    initial_logical_sequence: int = 0,
    max_session_samples: int | None = None,
) -> Any:
    cls = _module().AcceptedAudioAuthority
    return cls(
        request_id="request-a",
        engine_epoch="epoch-a",
        lease_generation=7,
        chunk_samples=chunk_samples,
        capacity_samples=capacity_samples,
        carrier_sequence_modulus=2**24,
        initial_logical_sequence=initial_logical_sequence,
        max_session_samples=max_session_samples,
    )


def _paced_authority(
    *,
    cadence_ns: int = 80,
    chunk_samples: int = 4,
    capacity_samples: int = 32,
) -> Any:
    cls = _module().AcceptedAudioAuthority
    try:
        return cls(
            request_id="request-a",
            engine_epoch="epoch-a",
            lease_generation=7,
            chunk_samples=chunk_samples,
            capacity_samples=capacity_samples,
            carrier_sequence_modulus=2**24,
            cadence_ns=cadence_ns,
        )
    except TypeError as error:
        _fail(
            f"PORT-SESS-001 missing the server-regulated cadence release clock: {error}"
        )


def _samples(start: int, count: int) -> np.ndarray:
    return np.arange(start, start + count, dtype=np.float32)


def _snapshot(authority: Any) -> Any:
    snapshot = authority.snapshot()
    assert snapshot.accepted_samples == (
        snapshot.parked_samples
        + snapshot.cleared_samples
        + snapshot.outstanding_samples
    )
    assert snapshot.outstanding_samples == (
        snapshot.residual_samples + snapshot.ready_samples + snapshot.in_flight_samples
    )
    return snapshot


def _park_current(authority: Any, unit: Any) -> None:
    authority.park(
        request_id="request-a",
        engine_epoch="epoch-a",
        lease_generation=7,
        logical_sequence=unit.logical_sequence,
        carrier_sequence=getattr(unit, "carrier_sequence", None),
    )


def _drain_regular(authority: Any) -> list[Any]:
    units: list[Any] = []
    while True:
        unit = authority.dispatch_next()
        if unit is None:
            return units
        units.append(unit)
        _park_current(authority, unit)


def test_authority_uses_segmented_fifo_not_repeated_full_concatenation() -> None:
    # @spec PORT-SESS-001 / PORT-STATE-021
    source = inspect.getsource(_module().AcceptedAudioAuthority)
    assert "deque" in source
    assert "head" in source
    assert "np.concatenate" not in source


def test_packetization_does_not_change_cadence_units_or_final_tail() -> None:
    # @spec PORT-SESS-001 / PORT-SESS-003 / PORT-SESS-013
    waveform = _samples(0, 11)
    left = _authority(capacity_samples=32)
    right = _authority(capacity_samples=32)

    left.accept(waveform, accepted_at_ns=100)
    right.accept(waveform[:2], accepted_at_ns=100)
    right.accept(waveform[2:7], accepted_at_ns=101)
    right.accept(waveform[7:], accepted_at_ns=102)
    left.begin_finalize(finalize_at_ns=200)
    right.begin_finalize(finalize_at_ns=200)

    def collect(authority: Any) -> list[tuple[str, list[float]]]:
        collected: list[tuple[str, list[float]]] = []
        while (unit := authority.dispatch_next()) is not None:
            collected.append((unit.kind, unit.samples.tolist()))
            _park_current(authority, unit)
        return collected

    assert (
        collect(left)
        == collect(right)
        == [
            ("regular", [0.0, 1.0, 2.0, 3.0]),
            ("regular", [4.0, 5.0, 6.0, 7.0]),
            ("final_tail", [8.0, 9.0, 10.0]),
        ]
    )
    assert _snapshot(left).outstanding_samples == 0
    assert _snapshot(right).outstanding_samples == 0


def test_exact_capacity_accepts_and_one_sample_over_rejects_atomically() -> None:
    # @spec PORT-SESS-001 / PORT-SESS-014
    authority = _authority(capacity_samples=8)
    authority.accept(_samples(0, 8), accepted_at_ns=10)
    before = authority.snapshot()

    with pytest.raises(ValueError, match="buffer_overflow"):
        authority.accept(_samples(99, 1), accepted_at_ns=11)

    after = authority.snapshot()
    assert after == before
    assert after.accepted_samples == after.outstanding_samples == 8
    assert [unit.samples.tolist() for unit in authority.ready_units] == [
        [0.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
    ]


def test_paced_stream_has_no_implicit_duration_limit_and_fifo_plateaus() -> None:
    # @spec PORT-SESS-006 / PORT-SESS-013 / PORT-STATE-021
    authority = _authority(capacity_samples=4, max_session_samples=None)

    for sequence in range(100):
        authority.accept(_samples(sequence * 4, 4), accepted_at_ns=sequence)
        unit = authority.dispatch_next()
        assert unit.logical_sequence == sequence
        _park_current(authority, unit)
        snapshot = _snapshot(authority)
        assert snapshot.outstanding_samples == 0
        assert snapshot.residual_samples == 0
        assert snapshot.ready_samples == 0
        assert snapshot.in_flight_samples == 0

    assert snapshot.accepted_samples == snapshot.parked_samples == 400


def test_optional_duration_policy_rejects_whole_piece_after_exact_limit() -> None:
    # @spec PORT-SESS-006 / PORT-SESS-014
    authority = _authority(capacity_samples=4, max_session_samples=8)
    for sequence in range(2):
        authority.accept(_samples(sequence * 4, 4), accepted_at_ns=sequence)
        _park_current(authority, authority.dispatch_next())
    before = authority.snapshot()

    with pytest.raises(ValueError, match="session_duration"):
        authority.accept(_samples(8, 1), accepted_at_ns=3)

    assert authority.snapshot() == before


@pytest.mark.parametrize(
    "bad_piece",
    [
        np.array([], dtype=np.float32),
        np.array([np.nan], dtype=np.float32),
        np.array([np.inf], dtype=np.float32),
        np.array([[1.0]], dtype=np.float32),
        np.array([1.0], dtype=np.float64),
    ],
)
def test_invalid_piece_is_rejected_before_any_state_mutation(
    bad_piece: np.ndarray,
) -> None:
    # @spec PORT-SESS-014
    authority = _authority()
    before = authority.snapshot()

    with pytest.raises(ValueError, match="finite|mono|FP32|nonempty"):
        authority.accept(bad_piece, accepted_at_ns=10)

    assert authority.snapshot() == before


def test_acceptance_owns_a_contiguous_copy_before_caller_mutation() -> None:
    # @spec PORT-SESS-014
    authority = _authority()
    backing = _samples(0, 8)
    noncontiguous = backing[::2]
    assert not noncontiguous.flags.c_contiguous

    authority.accept(noncontiguous, accepted_at_ns=10)
    backing[:] = -99
    unit = authority.dispatch_next()

    assert unit.samples.dtype == np.float32
    assert unit.samples.flags.c_contiguous
    assert unit.samples.tolist() == [0.0, 2.0, 4.0, 6.0]


def test_one_large_piece_queues_every_complete_unit_before_dispatch() -> None:
    # @spec PORT-SESS-001 / PORT-SESS-014
    authority = _authority(capacity_samples=20)
    receipt = authority.accept(_samples(0, 14), accepted_at_ns=777)

    assert receipt.samples_accepted == 14
    assert len(authority.ready_units) == 3
    assert [unit.logical_sequence for unit in authority.ready_units] == [0, 1, 2]
    assert [unit.ready_at_ns for unit in authority.ready_units] == [777, 777, 777]
    snapshot = _snapshot(authority)
    assert snapshot.residual_samples == 2
    assert snapshot.ready_samples == 12
    assert snapshot.in_flight_samples == 0


def test_burst_keeps_true_readiness_but_releases_one_unit_per_cadence() -> None:
    # @spec PORT-SESS-001 / PORT-STATE-026
    authority = _paced_authority(cadence_ns=80)
    authority.accept(_samples(0, 12), accepted_at_ns=100)

    assert [unit.ready_at_ns for unit in authority.ready_units] == [100, 100, 100]
    first = authority.dispatch_next(now_ns=100)
    assert first is not None
    authority.record_submission(first, submitted_at_ns=120)
    _park_current(authority, first)

    assert authority.dispatch_next(now_ns=199) is None
    assert authority.next_eligibility_ns == 200
    second = authority.dispatch_next(now_ns=200)
    assert second is not None
    authority.record_submission(second, submitted_at_ns=230)
    _park_current(authority, second)

    assert authority.dispatch_next(now_ns=309) is None
    third = authority.dispatch_next(now_ns=310)
    assert third is not None
    assert [unit.ready_at_ns for unit in (first, second, third)] == [100, 100, 100]


def test_release_clock_advances_from_actual_not_planned_submission() -> None:
    # @spec PORT-SESS-001 / PORT-STATE-026
    authority = _paced_authority(cadence_ns=80)
    authority.accept(_samples(0, 8), accepted_at_ns=100)
    first = authority.dispatch_next(now_ns=100)

    authority.record_submission(first, submitted_at_ns=1_000)
    _park_current(authority, first)

    assert authority.next_eligibility_ns == 1_080
    assert authority.dispatch_next(now_ns=1_079) is None
    assert authority.dispatch_next(now_ns=1_080) is not None


def test_final_tail_uses_the_same_ordinary_release_clock() -> None:
    # @spec PORT-SESS-003 / PORT-STATE-026
    authority = _paced_authority(cadence_ns=80)
    authority.accept(_samples(0, 4), accepted_at_ns=100)
    first = authority.dispatch_next(now_ns=100)
    authority.record_submission(first, submitted_at_ns=120)
    _park_current(authority, first)
    authority.begin_finalize(finalize_at_ns=130)

    assert authority.dispatch_next(now_ns=199) is None
    tail = authority.dispatch_next(now_ns=200)
    assert tail is not None
    assert tail.kind == "final_tail"
    assert tail.ready_at_ns == 130


def test_forced_eou_does_not_advance_the_ordinary_release_clock() -> None:
    # @spec PORT-SESS-001 / PORT-SEG-004 / PORT-STATE-026
    authority = _paced_authority(cadence_ns=80)
    authority.force_segment()
    barrier = authority.dispatch_next(now_ns=100)

    assert barrier is not None
    assert barrier.kind == "forced_eou"
    authority.record_submission(barrier, submitted_at_ns=120)
    _park_current(authority, barrier)
    assert authority.next_eligibility_ns is None

    authority.accept(_samples(0, 4), accepted_at_ns=121)
    regular = authority.dispatch_next(now_ns=121)
    assert regular is not None
    assert regular.kind == "regular"


def test_dispatch_allows_at_most_one_in_flight_unit() -> None:
    # @spec PORT-SESS-001 / PORT-SESS-013
    authority = _authority()
    authority.accept(_samples(0, 8), accepted_at_ns=1)
    first = authority.dispatch_next()
    assert first.logical_sequence == 0

    assert authority.dispatch_next() is None
    snapshot = _snapshot(authority)
    assert snapshot.in_flight_samples == 4
    assert snapshot.ready_samples == 4

    _park_current(authority, first)
    second = authority.dispatch_next()
    assert second.logical_sequence == 1


def test_only_matching_legal_park_releases_exact_unit_credit() -> None:
    # @spec PORT-SESS-013 / PORT-INT-004
    authority = _authority()
    authority.accept(_samples(0, 5), accepted_at_ns=1)
    unit = authority.dispatch_next()
    before = authority.snapshot()

    for bad in (
        {"request_id": "request-b"},
        {"engine_epoch": "epoch-b"},
        {"lease_generation": 8},
        {"logical_sequence": unit.logical_sequence + 1},
        {"carrier_sequence": unit.carrier_sequence + 1},
    ):
        args = {
            "request_id": "request-a",
            "engine_epoch": "epoch-a",
            "lease_generation": 7,
            "logical_sequence": unit.logical_sequence,
            "carrier_sequence": unit.carrier_sequence,
        }
        args.update(bad)
        with pytest.raises(ValueError, match="park|identity|generation|sequence"):
            authority.park(**args)
        assert authority.snapshot() == before

    _park_current(authority, unit)
    snapshot = _snapshot(authority)
    assert snapshot.parked_samples == 4
    assert snapshot.outstanding_samples == 1
    with pytest.raises(ValueError, match="park|in.flight|duplicate"):
        _park_current(authority, unit)
    assert authority.snapshot() == snapshot


@pytest.mark.parametrize("location", ["residual", "ready", "in_flight"])
def test_terminal_clear_conserves_samples_and_is_idempotent(location: str) -> None:
    # @spec PORT-SESS-013
    authority = _authority()
    if location == "residual":
        authority.accept(_samples(0, 3), accepted_at_ns=1)
    else:
        authority.accept(_samples(0, 8), accepted_at_ns=1)
        if location == "in_flight":
            authority.dispatch_next()
    before = _snapshot(authority)

    authority.clear(RuntimeError("client disconnected"))
    after = _snapshot(authority)
    assert after.cleared_samples == before.outstanding_samples
    assert after.outstanding_samples == 0
    assert after.residual_samples == after.ready_samples == after.in_flight_samples == 0

    authority.clear(RuntimeError("duplicate cleanup"))
    assert authority.snapshot() == after


def test_forced_eou_is_zero_sample_ordered_barrier_and_preserves_residual() -> None:
    # @spec PORT-SEG-004 / PORT-SESS-013 / PORT-SESS-014
    authority = _authority(capacity_samples=20)
    authority.accept(_samples(0, 6), accepted_at_ns=1)
    authority.force_segment()
    authority.accept(_samples(6, 2), accepted_at_ns=2)

    assert [(unit.kind, unit.sample_count) for unit in authority.ready_units] == [
        ("regular", 4),
        ("forced_eou", 0),
        ("regular", 4),
    ]
    assert authority.ready_units[2].samples.tolist() == [4.0, 5.0, 6.0, 7.0]
    assert authority.ready_units[1].carrier_sequence is None
    snapshot = _snapshot(authority)
    assert snapshot.accepted_samples == snapshot.outstanding_samples == 8


def test_duplicate_forces_for_one_generation_coalesce_to_one_barrier() -> None:
    # @spec PORT-SEG-004 / PORT-SEG-007 / PORT-STATE-021
    authority = _authority()

    for _ in range(100):
        authority.force_segment()

    assert [(unit.kind, unit.sample_count) for unit in authority.ready_units] == [
        ("forced_eou", 0)
    ]
    assert _snapshot(authority).outstanding_samples == 0


def test_finalize_first_rejects_later_audio_force_and_locale_update() -> None:
    # @spec PORT-SESS-003 / PORT-SESS-014
    authority = _authority()
    authority.accept(_samples(0, 3), accepted_at_ns=1)
    authority.begin_finalize(finalize_at_ns=2)
    before = authority.snapshot()

    with pytest.raises(ValueError, match="final"):
        authority.accept(_samples(3, 1), accepted_at_ns=3)
    with pytest.raises(ValueError, match="final"):
        authority.force_segment()
    with pytest.raises(ValueError, match="final"):
        authority.update_locale("en-US")

    assert authority.snapshot() == before
    assert [(unit.kind, unit.sample_count) for unit in authority.ready_units] == [
        ("final_tail", 3)
    ]


def test_force_first_before_finalize_remains_before_final_tail() -> None:
    # @spec PORT-SESS-003 / PORT-SESS-014 / PORT-SEG-004
    authority = _authority()
    authority.accept(_samples(0, 3), accepted_at_ns=1)
    authority.force_segment()
    authority.begin_finalize(finalize_at_ns=2)

    assert [(unit.kind, unit.sample_count) for unit in authority.ready_units] == [
        ("forced_eou", 0),
        ("final_tail", 3),
    ]


def test_zero_audio_finalize_still_queues_one_zero_sample_final_tail() -> None:
    # @spec PORT-SESS-003 / PORT-SESS-013
    authority = _authority()
    authority.begin_finalize(finalize_at_ns=2)

    assert len(authority.ready_units) == 1
    tail = authority.dispatch_next()
    assert tail.kind == "final_tail"
    assert tail.sample_count == 0
    assert tail.samples.dtype == np.float32
    _park_current(authority, tail)
    snapshot = _snapshot(authority)
    assert snapshot.accepted_samples == 0
    assert snapshot.parked_samples == 0
    assert snapshot.outstanding_samples == 0


def test_zero_sample_forced_barrier_never_releases_audio_credit() -> None:
    # @spec PORT-SESS-013 / PORT-SEG-004
    authority = _authority()
    authority.accept(_samples(0, 3), accepted_at_ns=1)
    authority.force_segment()
    barrier = authority.dispatch_next()
    before = _snapshot(authority)

    assert barrier.kind == "forced_eou"
    assert barrier.sample_count == 0
    _park_current(authority, barrier)
    after = _snapshot(authority)
    assert after.accepted_samples == before.accepted_samples == 3
    assert after.parked_samples == before.parked_samples == 0
    assert after.outstanding_samples == before.outstanding_samples == 3


def test_carrier_sequence_wraps_but_logical_sequence_never_does() -> None:
    # @spec PORT-INT-004 / PORT-SESS-013
    modulus = 2**24
    authority = _authority(
        capacity_samples=16,
        initial_logical_sequence=modulus - 2,
    )
    authority.accept(_samples(0, 12), accepted_at_ns=1)

    units = _drain_regular(authority)
    assert [unit.logical_sequence for unit in units] == [
        modulus - 2,
        modulus - 1,
        modulus,
    ]
    assert [unit.carrier_sequence for unit in units] == [
        modulus - 2,
        modulus - 1,
        0,
    ]
    assert _snapshot(authority).parked_samples == 12


def test_authority_mutation_api_is_synchronous_and_uses_one_lock() -> None:
    # @spec PORT-SESS-014
    import inspect

    cls = _module().AcceptedAudioAuthority
    for name in ("accept", "force_segment", "update_locale", "begin_finalize"):
        method = getattr(cls, name)
        assert not inspect.iscoroutinefunction(method), name
        source = inspect.getsource(method)
        assert "self._lock" in source, name
        assert "await " not in source, name
