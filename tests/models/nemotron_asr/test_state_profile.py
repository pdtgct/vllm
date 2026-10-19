# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate-state profile projections for the Nemotron consumer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm.models.nemotron_asr.state_profile import (
    allocate_state_scratch,
    build_nemotron_persistent_state_spec,
    project_nemotron_state_pools,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        n_layers=24,
        att_context_left=56,
        d_model=1024,
        conv_kernel=9,
        pred_rnn_layers=2,
        pred_hidden=640,
        n_mels=128,
        num_asr_labels=13_087,
        endpoint_history_capacity_frames=12,
    )


def test_manifest_builds_one_exact_aggregate_and_zero_copy_pool_views() -> None:
    # @spec PORT-STATE-001 / PORT-STATE-002 / PORT-STATE-009
    spec = build_nemotron_persistent_state_spec(_config())
    storage = allocate_state_scratch(spec, 2, torch.device("cpu"))
    storage.initialize(1)
    pools = project_nemotron_state_pools(storage, n_layers=24)

    assert len(spec.descriptors) == 100
    assert len(pools.channel) == len(pools.convolution) == 24
    assert pools.replay_book.shape == (2, 7)
    assert pools.frontend_counters.shape == (2, 8)
    assert pools.endpoint_book.shape == (2, 6)
    assert pools.replay_book[1, 2].item() == 13_087
    assert storage.views["padding.frontend_counters_alignment"][1].item() == 0

    pools.replay_book[1, 4] = 9
    pools.frontend_counters[1, 7] = 1
    pools.endpoint_book[1, 3] = 6
    assert storage.views["decode.layers.0.replay.book.geometry"][1].item() == 9
    assert storage.views["frontend.counters.finalized"][1].item() == 1
    assert storage.views["endpoint.book.segment_generation"][1].item() == 6


def test_fresh_generation_erases_every_byte_and_reinitializes_blank() -> None:
    # @spec PORT-STATE-003 / PORT-STATE-009
    spec = build_nemotron_persistent_state_spec(_config())
    from vllm.v1.kv_cache_interface import StateSlotHandle
    from vllm.v1.worker.gpu.state_slots import StateSlotViews

    raw = torch.zeros(2 * spec.page_size_bytes, dtype=torch.uint8)
    slots = StateSlotViews(spec, raw)
    old = StateSlotHandle("epoch", 1, 1)
    slots.bind("old", old)
    raw[spec.page_size_bytes :].fill_(0x7F)
    new = StateSlotHandle("epoch", 1, 2)
    slots.bind("replacement", new)
    views = slots.view("replacement", new)
    blank = views["decode.layers.0.replay.book.last_label"]
    assert blank.item() == 13_087
    blank.zero_()
    assert torch.count_nonzero(raw).item() == 0
    with pytest.raises(ValueError, match="Stale"):
        slots.view("old", old)
