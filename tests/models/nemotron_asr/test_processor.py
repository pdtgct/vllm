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


@pytest.mark.parametrize("enable_cache", [False, True])
def test_core_profiling_constructs_raw_audio_carriers(
    tmp_path, monkeypatch, enable_cache
):
    """Registry/budget profiling must call the processor using core's real API."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from tests.v1.engine.test_nemotron_asr import _engine_args
    from vllm.models.nemotron_asr.configuration_nemotron_asr import NemotronASRConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.encoder_budget import MultiModalBudget
    from vllm.platforms import current_platform

    # Local config and tokenizer only: no checkpoint download or model allocation.
    NemotronASRConfig(
        num_asr_labels=2,
        vocab_size=8,
        eos_token_id=4,
        audio_chunk_token_id=_CARRIER_ID,
        eou_token_id=5,
        flush_token_id=6,
    ).save_pretrained(tmp_path)
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>")),
        unk_token="<unk>",
    ).save_pretrained(tmp_path)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    # Validate the smoke config on CPU; only device/Triton availability is stubbed.
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    monkeypatch.setattr("vllm.config.vllm.HAS_TRITON", True)
    config = _engine_args(str(tmp_path)).create_engine_config()
    assert config.use_v2_model_runner
    assert not config.model_config.skip_tokenizer_init
    processor = MULTIMODAL_REGISTRY.create_processor(config.model_config)
    assert isinstance(processor, NemotronASRMultiModalProcessor)

    budget = MultiModalBudget(config, MULTIMODAL_REGISTRY, enable_cache=enable_cache)
    assert budget.mm_max_toks_per_item == {"audio": 1}
    assert budget.mm_max_items_per_prompt == {"audio": 1}
    assert budget.mm_max_items_per_batch == {"audio": 2}
    inputs = processor.get_dummy_mm_inputs(
        {"audio": 1}, scheduler_config=config.scheduler_config
    )
    assert inputs["prompt_token_ids"] == [_CARRIER_ID] + [0] * 255
    (placeholder,) = inputs["mm_placeholders"]["audio"]
    assert (placeholder.offset, placeholder.length) == (0, 1)
    assert placeholder.get_num_embeds() == 1
    profile_inputs = budget.get_dummy_encoder_profile_inputs()
    assert len(profile_inputs) == 2
    for modality, item in profile_inputs:
        assert modality == "audio"
        audio = item["audio"].data
        assert audio.dtype == torch.float32
        assert audio.shape == (16_000,)
        assert torch.count_nonzero(audio) == 0
