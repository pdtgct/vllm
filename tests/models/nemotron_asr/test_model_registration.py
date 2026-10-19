# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registration contracts for the native Nemotron ASR model."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from vllm.models.nemotron_asr import configuration_nemotron_asr, identity

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]

_MODEL_PACKAGE = Path(identity.__file__).resolve().parent
_ARCHITECTURE = "Nemotron3_5AsrForRNNT"
_LAZY_TARGET = ("vllm.models.nemotron_asr.nemotron_asr", "NemotronASRForRNNT")


def test_architecture_has_one_dependency_light_authority() -> None:
    assignments = []
    for path in sorted(_MODEL_PACKAGE.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, ast.AnnAssign)
                else []
            )
            if any(
                isinstance(target, ast.Name) and target.id == "ARCHITECTURE"
                for target in targets
            ):
                assignments.append(path.name)
    assert assignments == ["identity.py"]
    assert identity.ARCHITECTURE == _ARCHITECTURE
    assert configuration_nemotron_asr.ARCHITECTURE == _ARCHITECTURE

    imported: set[str] = set()
    for node in ast.walk(ast.parse(Path(identity.__file__).read_text())):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"torch", "transformers"}


def test_core_registries_declare_exact_targets() -> None:
    from vllm.model_executor.models.registry import _VLLM_MODELS
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    assert _VLLM_MODELS[_ARCHITECTURE] == _LAZY_TARGET
    config_cls = _CONFIG_REGISTRY[configuration_nemotron_asr.MODEL_TYPE]
    assert config_cls is configuration_nemotron_asr.NemotronASRConfig
    card_cls = _CONFIG_REGISTRY[configuration_nemotron_asr.CARD_MODEL_TYPE]
    assert card_cls is configuration_nemotron_asr.NemotronCardServingConfig


def test_registry_inspection_resolves_native_class() -> None:
    from vllm.model_executor.models import ModelRegistry

    info = ModelRegistry._try_inspect_model_cls(_ARCHITECTURE)
    assert info is not None
    assert info.architecture == "NemotronASRForRNNT"
    assert info.supports_multimodal
    assert info.requires_raw_input_tokens
    assert info.is_attention_free


def test_cpu_constructor_declares_one_core_state_layer(monkeypatch):
    """CPU construction with tiny modules still uses the real core declaration."""
    from types import SimpleNamespace

    import torch

    from tests.models.nemotron_asr.test_advance_model_rows_local import _tiny_core
    from vllm.models.nemotron_asr import nemotron_asr
    from vllm.v1.kv_cache_interface import PersistentStateSpec
    from vllm.v1.worker.gpu.attn_utils import get_kv_cache_spec

    def make_core(**kwargs):
        tiny = _tiny_core()
        core = torch.nn.Module()
        for name, value in vars(tiny).items():
            setattr(core, name, value)
        core.policy = kwargs["policy"]
        return core

    monkeypatch.setattr(nemotron_asr, "NemotronASRCore", make_core)
    config = configuration_nemotron_asr.NemotronASRConfig(
        n_layers=2,
        d_model=32,
        pred_hidden=16,
        pred_rnn_layers=2,
        conv_kernel=5,
        att_context_left=8,
        att_context_right=1,
        num_asr_labels=12,
        n_mels=16,
        num_prompts=4,
        vocab_size=20,
        eos_token_id=13,
        audio_chunk_token_id=14,
        eou_token_id=15,
        flush_token_id=16,
        prompt_dictionary={"en-US": 0},
        decode_dispatch_arm="dense-eager",
    )
    runtime = SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(
            hf_config=config, dtype=torch.float32, max_model_len=143
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        compilation_config=SimpleNamespace(static_forward_context={}),
        attention_config=SimpleNamespace(hisparse_config=None),
    )
    model = nemotron_asr.NemotronASRForRNNT(vllm_config=runtime, prefix="asr")
    assert model.is_attention_free
    specs = get_kv_cache_spec(runtime)
    assert list(specs) == ["asr.state"]
    assert isinstance(specs["asr.state"], PersistentStateSpec)
    assert specs["asr.state"].max_num_blocks_per_req(runtime, 100000) == 1


@pytest.mark.parametrize("max_model_len", [1, 141, 142])
def test_model_rejects_history_shorter_than_one_replay_transaction(max_model_len):
    from types import SimpleNamespace

    from vllm.models.nemotron_asr.nemotron_asr import NemotronASRForRNNT

    with pytest.raises(ValueError, match="143 tokens"):
        NemotronASRForRNNT(
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(max_model_len=max_model_len)
            )
        )
