# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for publisher-minted endpoint control ids."""

from __future__ import annotations

import inspect

import pytest

from vllm.models.nemotron_asr import publish
from vllm.models.nemotron_asr.configuration_nemotron_asr import (
    NemotronASRConfig,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]


def _config(**overrides: object) -> NemotronASRConfig:
    values: dict[str, object] = {
        "num_asr_labels": 100,
        "vocab_size": 105,
        "eos_token_id": 101,
        "audio_chunk_token_id": 102,
        "eou_token_id": 103,
        "flush_token_id": 104,
        "endpoint_history_capacity_frames": 12,
    }
    values.update(overrides)
    return NemotronASRConfig(**values)


def test_four_controls_are_pairwise_distinct_and_outside_rnnt_space() -> None:
    # @spec PORT-SEG-003 / PORT-WGT-004
    config = _config()
    controls = (
        config.eos_token_id,
        config.audio_chunk_token_id,
        config.eou_token_id,
        config.flush_token_id,
    )

    assert controls == (101, 102, 103, 104)
    assert len(set(controls)) == 4
    assert min(controls) > config.num_asr_labels  # blank == num_asr_labels
    assert config.vocab_size == max(controls) + 1


def test_endpoint_controls_and_capacity_are_explicit_config_fields() -> None:
    # @spec PORT-SEG-001 / PORT-WGT-004
    parameters = inspect.signature(NemotronASRConfig.__init__).parameters
    assert "eou_token_id" in parameters
    assert "flush_token_id" in parameters
    assert "endpoint_history_capacity_frames" in parameters


@pytest.mark.parametrize(
    "overrides",
    [
        {"eou_token_id": 101},
        {"flush_token_id": 102},
        {"eou_token_id": 100},
        {"flush_token_id": 99},
        {"vocab_size": 104},
    ],
)
def test_invalid_control_collision_or_width_fails_at_config_load(
    overrides: dict[str, int],
) -> None:
    # @spec PORT-RTC-001 / PORT-WGT-004
    with pytest.raises(ValueError, match="control|token|vocab|distinct|label"):
        _config(**overrides)


def test_publisher_mints_deterministic_consecutive_control_ids() -> None:
    # @spec PORT-WGT-004
    assert publish._PARK_OFFSET == 1
    assert publish._PLACEHOLDER_OFFSET == 2
    assert publish._EOU_OFFSET == 3
    assert publish._FLUSH_OFFSET == 4


def test_publisher_writes_all_control_and_width_fields_to_artifact() -> None:
    # @spec PORT-INT-005 / PORT-WGT-004
    source = inspect.getsource(publish.publish)
    for field in (
        "num_asr_labels",
        "vocab_size",
        "eos_token_id",
        "audio_chunk_token_id",
        "eou_token_id",
        "flush_token_id",
    ):
        assert field in source
