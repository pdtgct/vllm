# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The raw-audio processor binds one carrier placeholder per chunk."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.models.nemotron_asr.processor import (
    NemotronASRMultiModalProcessor,
    NemotronASRProcessingInfo,
)
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing.processor import MultiModalProcessingResult

pytestmark = [pytest.mark.cpu_test]

_CARRIER_ID = 7


class _Info(NemotronASRProcessingInfo):
    def __init__(self) -> None:
        self.ctx = SimpleNamespace(tokenizer=object())

    def get_hf_config(self, *args):
        return SimpleNamespace(audio_chunk_token_id=_CARRIER_ID)

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser()


@pytest.fixture
def processor() -> NemotronASRMultiModalProcessor:
    return NemotronASRMultiModalProcessor(_Info(), dummy_inputs=None)


def _audio_items(processor: NemotronASRMultiModalProcessor):
    audio = np.linspace(-1.0, 1.0, 1600, dtype=np.float64)
    return processor.data_parser.parse_mm_data({"audio": audio}), audio


def test_raw_audio_passes_through_as_float32_tensor(processor) -> None:
    items, audio = _audio_items(processor)
    batch = processor._apply_hf_processor_main(items, {})
    (tensor,) = batch["audio"]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    torch.testing.assert_close(tensor, torch.from_numpy(audio.astype(np.float32)))


@pytest.mark.parametrize("prompt_ids", [[], [_CARRIER_ID]])
def test_single_carrier_placeholder_at_prompt_start(processor, prompt_ids) -> None:
    items, _ = _audio_items(processor)
    result = MultiModalProcessingResult(
        prompt_ids=list(prompt_ids), kwargs={}, hashes={}, prompt_updates={}
    )
    ids, placeholders = processor._maybe_apply_prompt_updates(items, result)
    assert ids == [_CARRIER_ID]
    (info,) = placeholders["audio"]
    assert (info.item_idx, info.start_idx, info.tokens) == (0, 0, [_CARRIER_ID])
