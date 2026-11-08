# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The raw-audio processor binds one carrier placeholder per chunk."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.models.nemotron_asr.advance import ENV_GEOMETRY_ID
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
    # Every audio item is an envelope; this one names a served geometry.
    audio[ENV_GEOMETRY_ID] = 0.0
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


def _registry_processor(tmp_path, monkeypatch, hf_overrides=None):
    """Build the real registry processor from a local config and tokenizer."""
    from dataclasses import replace

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from tests.v1.engine.test_nemotron_asr import _engine_args
    from vllm.models.nemotron_asr.configuration_nemotron_asr import NemotronASRConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
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
    # Validate the smoke config on CPU; only device/Triton availability is stubbed.
    monkeypatch.setattr(current_platform, "device_type", "cpu")
    monkeypatch.setattr(
        current_platform, "uses_standalone_cpu_model_runner", lambda: False
    )
    args = replace(_engine_args(str(tmp_path)), hf_overrides=hf_overrides or {})
    config = args.create_engine_config()
    return config, MULTIMODAL_REGISTRY.create_processor(config.model_config)


@pytest.mark.parametrize("enable_cache", [False, True])
def test_core_profiling_constructs_raw_audio_carriers(
    tmp_path, monkeypatch, enable_cache
):
    """Registry/budget profiling must call the processor using core's real API."""
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.encoder_budget import MultiModalBudget

    config, processor = _registry_processor(tmp_path, monkeypatch)
    assert not config.model_config.skip_tokenizer_init
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


@pytest.mark.parametrize("populations", [None, [1]])
def test_envelope_outside_served_geometries_is_rejected(
    tmp_path, monkeypatch, populations
):
    """An envelope naming an unserved geometry fails in the processor.

    The served set is g0 only; geometry 3 is a published cadence outside it.
    The check holds with and without ``transaction_graph_populations``.
    """
    from vllm.multimodal.processing import ProcessorInputs, TimingContext

    overrides: dict[str, object] = {"supported_num_lookahead_tokens": [0]}
    if populations is not None:
        overrides["transaction_graph_populations"] = populations
    _, processor = _registry_processor(tmp_path, monkeypatch, overrides)

    def apply(geometry: int):
        header = torch.tensor([2, 1_280, geometry, 0, 0, 0, 1], dtype=torch.float32)
        envelope = torch.cat((header, torch.zeros(1_280)))
        items = processor.data_parser.parse_mm_data({"audio": envelope.numpy()})
        return processor.apply(
            ProcessorInputs(prompt=[_CARRIER_ID], mm_data_items=items),
            TimingContext(),
        )

    assert apply(0)["prompt_token_ids"] == [_CARRIER_ID]
    with pytest.raises(ValueError, match="geometry"):
        apply(3)


def _apply_envelope(processor: NemotronASRMultiModalProcessor, geometry: float):
    """Process one CHUNK envelope whose header names ``geometry``."""
    from vllm.multimodal.processing import ProcessorInputs, TimingContext

    header = torch.tensor([2, 1_280, geometry, 0, 0, 0, 1], dtype=torch.float32)
    envelope = torch.cat((header, torch.zeros(1_280)))
    items = processor.data_parser.parse_mm_data({"audio": envelope.numpy()})
    return processor.apply(
        ProcessorInputs(prompt=[_CARRIER_ID], mm_data_items=items), TimingContext()
    )


@pytest.mark.parametrize("geometry", [5, 999], ids=["next-index", "far-index"])
def test_envelope_beyond_published_cadences_is_rejected_without_the_field(
    tmp_path, monkeypatch, geometry
):
    """Without ``supported_num_lookahead_tokens`` every published cadence is
    served: a finite set the processor still enforces."""
    from vllm.models.nemotron_asr.manifests import CADENCES, served_geometry_ids

    _, processor = _registry_processor(tmp_path, monkeypatch)
    hf_config = processor.info.get_hf_config()
    assert getattr(hf_config, "supported_num_lookahead_tokens", None) is None
    published = tuple(range(len(CADENCES)))
    assert served_geometry_ids(hf_config) == published
    assert geometry not in published
    for served in published:
        assert _apply_envelope(processor, served)["prompt_token_ids"] == [_CARRIER_ID]
    with pytest.raises(ValueError, match="geometry"):
        _apply_envelope(processor, geometry)


@pytest.mark.parametrize(
    "overrides",
    [{}, {"supported_num_lookahead_tokens": [0]}],
    ids=["field-absent", "g0"],
)
def test_envelope_with_invalid_geometry_index_is_rejected(
    tmp_path, monkeypatch, overrides
):
    """A geometry slot that is not a served cadence index fails in the processor."""
    _, processor = _registry_processor(tmp_path, monkeypatch, overrides)
    assert _apply_envelope(processor, 0)["prompt_token_ids"] == [_CARRIER_ID]
    for geometry in (-1.0, 0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="geometry"):
            _apply_envelope(processor, geometry)
