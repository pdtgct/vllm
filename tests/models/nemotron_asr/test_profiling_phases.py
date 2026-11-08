# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The NVTX phase contract: closed names, free when off, safe when on."""

from __future__ import annotations

import pytest

from vllm.models.nemotron_asr import profiling

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]


# @spec PORT-OBS-001
def test_phase_names_are_a_closed_declared_set() -> None:
    """Digests compare across runs only if span names are stable, so the
    set lives in one place and every name is `port.`-prefixed."""
    assert profiling.PHASES == (
        "port.ingest",
        "port.featurize",
        "port.encode",
        "port.decode",
        "port.scatter",
        "port.park",
    )
    assert all(name.startswith("port.") for name in profiling.PHASES)


# @spec PORT-OBS-001
def test_ranges_are_preallocated_and_shared() -> None:
    """The hot path runs ~12,500 transactions/s at target; a phase must
    not allocate per entry. The same object comes back every call."""
    for name in profiling.PHASES:
        assert profiling.phase(name) is profiling.phase(name)


# @spec PORT-OBS-001
def test_an_undeclared_name_is_inert_rather_than_fatal() -> None:
    """A typo must not take down serving; the missing span shows up in
    the digest, which is where a profiling mistake belongs."""
    with profiling.phase("port.not_a_phase"):
        pass


# @spec PORT-OBS-001
def test_a_phase_body_that_raises_still_leaves_the_range() -> None:
    """Unbalanced push/pop corrupts every later range in the capture."""
    with pytest.raises(ValueError), profiling.phase("port.encode"):
        raise ValueError("boom")
    with profiling.phase("port.encode"):
        pass


# @spec PORT-OBS-001
def test_disabled_by_default_so_serving_pays_nothing() -> None:
    """Profiling is a capture-harness decision, never a serving default."""
    assert profiling.enabled() is False
    assert type(profiling.phase("port.encode")).__name__ == "_NullPhase"
