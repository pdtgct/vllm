# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for bounded, exact transcript projection."""

from __future__ import annotations

import importlib
from typing import Any, NoReturn

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]


def _fail(message: str) -> NoReturn:
    pytest.fail(message, pytrace=False)
    raise AssertionError(message)


def _module() -> Any:
    try:
        return importlib.import_module("vllm.models.nemotron_asr.transcript")
    except ModuleNotFoundError:
        _fail("PORT-STATE-021 missing bounded transcript authority")


def _transcript(capacity: int = 128) -> Any:
    return _module().BoundedTranscript(
        max_retained_bytes=capacity,
        fragment_overhead_bytes=4,
        terminal_headroom_bytes=16,
    )


@pytest.mark.parametrize("capacity", [0, -1, 15])
def test_retained_output_capacity_must_be_positive_and_cover_headroom(
    capacity: int,
) -> None:
    # @spec PORT-STATE-021
    with pytest.raises(ValueError, match="capacity|headroom|positive"):
        _transcript(capacity=capacity)


def test_exact_capacity_accepts_one_complete_result_atomically() -> None:
    # @spec PORT-STATE-021 / PORT-SEG-005
    authority = _transcript(capacity=128)
    text = " natural language"
    required = authority.bytes_required_for_result(text)
    exact = _transcript(capacity=required)

    projection = exact.commit_result(text)

    assert projection.delta == text
    assert exact.complete_text == text
    assert exact.retained_bytes == required


def test_one_byte_over_publishes_and_retains_none_of_the_result() -> None:
    # @spec PORT-STATE-021 / PORT-SEG-005 / PORT-SEG-007
    probe = _transcript(capacity=128)
    text = " atomic result"
    required = probe.bytes_required_for_result(text)
    authority = _transcript(capacity=required - 1)
    before = authority.snapshot()

    with pytest.raises(
        _module().OutputCapacityExceeded, match="output_capacity_exceeded"
    ):
        authority.commit_result(text)

    assert authority.snapshot() == before
    assert authority.complete_text == ""


def test_multibyte_utf8_is_charged_by_bytes_not_characters() -> None:
    # @spec PORT-STATE-021
    authority = _transcript(capacity=128)
    ascii_required = authority.bytes_required_for_result("aaaa")
    utf8_required = authority.bytes_required_for_result("éééé")
    assert utf8_required == ascii_required + 4


def test_segment_completion_is_local_and_does_not_duplicate_retained_text() -> None:
    # @spec PORT-STATE-021 / PORT-SEG-005 / PORT-SEG-006
    authority = _transcript(capacity=256)
    authority.commit_result("hello")
    first = authority.complete_segment(generation=1, reason="model")
    bytes_after_first = authority.retained_bytes
    authority.commit_result(" world")
    second = authority.complete_segment(generation=2, reason="forced")

    assert first.text == "hello"
    assert first.reason == "model"
    assert second.text == " world"
    assert second.reason == "forced"
    assert authority.complete_text == "hello world"
    assert authority.retained_bytes > bytes_after_first
    assert authority.retained_fragment_count == 2


def test_terminal_returns_complete_text_and_only_unreported_suffix_ack() -> None:
    # @spec PORT-SESS-003 / PORT-SEG-005 / PORT-SEG-006
    authority = _transcript(capacity=256)
    authority.commit_result("hello")
    authority.complete_segment(generation=1, reason="model")
    authority.commit_result(" again")

    terminal = authority.finish_terminal()

    assert terminal.complete_text == "hello again"
    assert terminal.completion.reason == "terminal"
    assert terminal.completion.text == " again"
    assert authority.finished is True


def test_empty_suffix_still_acknowledges_terminal_once_without_extra_segment() -> None:
    # @spec PORT-SESS-003 / PORT-SEG-005 / PORT-SEG-006
    authority = _transcript(capacity=256)
    authority.commit_result("hello")
    authority.complete_segment(generation=1, reason="model")

    terminal = authority.finish_terminal()

    assert terminal.complete_text == "hello"
    assert terminal.completion.text == ""
    assert terminal.emit_segment_event is False
    with pytest.raises(RuntimeError, match="terminal|finished"):
        authority.finish_terminal()


def test_invalid_generation_rejects_and_stale_generation_is_ignored() -> None:
    # @spec PORT-SEG-007
    authority = _transcript(capacity=256)
    authority.commit_result("hello")
    before = authority.snapshot()

    with pytest.raises(ValueError, match="generation"):
        authority.complete_segment(generation=0, reason="model")
    assert authority.snapshot() == before

    authority.complete_segment(generation=1, reason="model")
    stable = authority.snapshot()
    assert authority.complete_segment(generation=1, reason="forced") is None
    assert authority.snapshot() == stable
