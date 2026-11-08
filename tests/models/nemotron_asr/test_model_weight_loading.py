# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Published-checkpoint to runtime tensor-shape contracts."""

from __future__ import annotations

import pytest
import torch

from vllm.models.nemotron_asr.nemotron_asr import (
    NemotronASRForRNNT,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]


def test_published_filterbank_drops_only_its_nemo_batch_axis() -> None:
    # @spec PORT-WGT-001, PORT-WGT-004
    published = torch.zeros(1, 128, 257)

    runtime = NemotronASRForRNNT._normalize_checkpoint_tensor(
        "featurizer.fb",
        published,
    )

    assert runtime.shape == (128, 257)
    assert runtime.data_ptr() == published.data_ptr()


def test_other_checkpoint_tensors_retain_exact_shape_and_identity() -> None:
    # @spec PORT-WGT-001
    published = torch.zeros(2, 3)

    runtime = NemotronASRForRNNT._normalize_checkpoint_tensor(
        "encoder.layers.0.scale",
        published,
    )

    assert runtime is published
