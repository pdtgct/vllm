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
