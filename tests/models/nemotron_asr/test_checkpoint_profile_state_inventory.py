# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for Nemotron's complete resumable-state inventory."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any, NoReturn, cast

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]


def _fail(message: str) -> NoReturn:
    pytest.fail(message, pytrace=False)
    raise AssertionError(message)


def _state_manifest() -> dict[str, Any]:
    """Author the public checkpoint-profile state manifest under test."""
    try:
        manifests = importlib.import_module("vllm.models.nemotron_asr.manifests")
    except ModuleNotFoundError:
        _fail("PORT-STATE-001 missing Nemotron checkpoint-profile manifests")

    config = SimpleNamespace(
        n_layers=24,
        att_context_left=56,
        d_model=1024,
        conv_kernel=9,
        pred_rnn_layers=2,
        pred_hidden=640,
        n_mels=128,
        endpoint_history_capacity_frames=12,
    )
    try:
        return cast(dict[str, Any], manifests.author_state_manifest(config))
    except (AttributeError, NotImplementedError) as exc:
        _fail(
            f"PORT-STATE-001 checkpoint-profile state author is missing or inert: {exc}"
        )


def _entry(
    name: str,
    shape: list[int],
    dtype: str,
    init: str = "zeros",
) -> dict[str, Any]:
    return {
        "name": name,
        "shape": shape,
        "dtype": dtype,
        "init": init,
    }


def _expected_inventory() -> list[dict[str, Any]]:
    """Independently spell the reviewed checkpoint-profile inventory."""
    entries: list[dict[str, Any]] = []
    for layer in range(24):
        entries.extend(
            (
                _entry(
                    f"encoder.layers.{layer}.window.channel",
                    [56, 1024],
                    "float32",
                ),
                _entry(
                    f"encoder.layers.{layer}.conv.time",
                    [1024, 8],
                    "float32",
                ),
                _entry(
                    f"encoder.layers.{layer}.window.valid",
                    [1],
                    "int32",
                ),
            )
        )

    entries.extend(
        (
            _entry("frontend.raw_tail", [1953], "float32"),
            _entry("frontend.mel_tail", [128, 9], "float32"),
        )
    )
    entries.append(_entry("decode.layers.0.replay.queue", [141], "int32"))
    for name, init in (
        ("queue_head", "zeros"),
        ("queue_length", "zeros"),
        ("last_label", "blank_label"),
        ("prompt", "admitted_prompt"),
        ("geometry", "admitted_geometry"),
        ("pending_echo", "zeros"),
        ("expected_label", "zeros"),
    ):
        entries.append(
            _entry(f"decode.layers.0.replay.book.{name}", [1], "int32", init)
        )
    entries.append(_entry("padding.frontend_counters_alignment", [1], "int32"))
    for counter in (
        "total_valid_samples",
        "committed_mel_frames",
        "encoded_mel_frames",
        "raw_tail_origin",
        "raw_tail_length",
        "mel_tail_length",
        "expected_chunk_sequence",
        "finalized",
    ):
        entries.append(_entry(f"frontend.counters.{counter}", [1], "int64"))

    entries.extend(
        (
            _entry("predictor.layers.0.lstm_state.h", [2, 640], "float32"),
            _entry("predictor.layers.0.lstm_state.c", [2, 640], "float32"),
        )
    )
    entries.append(_entry("endpoint.history", [12], "int32"))
    for name in (
        "history_length",
        "history_head",
        "endpoint_armed",
        "segment_generation",
        "segment_has_output",
        "pending_forced_generation",
    ):
        entries.append(_entry(f"endpoint.book.{name}", [1], "int32"))
    return entries


def _entry_bytes(entry: dict[str, Any]) -> int:
    dtype_bytes = {"float32": 4, "int32": 4, "int64": 8}
    elements = 1
    for dimension in entry["shape"]:
        elements *= dimension
    return elements * dtype_bytes[entry["dtype"]]


def test_checkpoint_profile_declares_one_complete_resumable_bundle() -> None:
    # @spec PORT-STATE-001 / PORT-ADV-002 / PORT-SEG-001 / PORT-WGT-004
    manifest = _state_manifest()
    expected = _expected_inventory()
    entries = manifest["entries"]

    # Exact equality makes every resumable field manifest-authoritative rather
    # than permitting a runner batch slot, reconstructed request history,
    # model-private pool, or hidden side store to be its sole declaration.
    assert manifest["schema"] == "state-manifest-v1"
    assert manifest["precision_policy"] == "fp32-bringup-v1"
    assert entries == expected
    assert len(entries) == len({entry["name"] for entry in entries}) == 100

    assert sum(entry["name"].endswith(".window.channel") for entry in entries) == 24
    assert sum(entry["name"].endswith(".window.valid") for entry in entries) == 24
    assert sum(entry["name"].endswith(".conv.time") for entry in entries) == 24
    assert sum(entry["name"].startswith("frontend.") for entry in entries) == 10
    assert sum(entry["name"].startswith("predictor.") for entry in entries) == 2
    assert sum(entry["name"].startswith("decode.") for entry in entries) == 8
    assert sum(entry["name"].startswith("endpoint.") for entry in entries) == 7
    assert sum(entry["name"].startswith("padding.") for entry in entries) == 1

    total_bytes = sum(_entry_bytes(entry) for entry in expected)
    assert total_bytes == 6_314_944
    assert manifest["total_page_bytes"] == total_bytes
