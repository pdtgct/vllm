# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-free provenance contracts for checkpoint publication."""

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]


def _load_identity_module() -> Any:
    path = (
        Path(__file__).resolve().parents[3]
        / "vllm/models/nemotron_asr/publication_identity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "nemotron_asr_publication_identity_under_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


identity = _load_identity_module()


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _write_metadata(path: Path, source: bytes, dump: bytes) -> None:
    path.write_text(
        json.dumps(
            {
                "source_checkpoint_digest": _digest(source),
                "converted_dump_digest": _digest(dump),
                "prompt_dictionary": {"en-US": 0},
                "num_prompts": 128,
                "vocab_size": 13087,
            }
        ),
        encoding="utf-8",
    )


def test_split_dump_provenance_accepts_the_exact_source_pair(
    tmp_path: Path,
) -> None:
    source_bytes = b"checkpoint-a"
    dump_bytes = b"converted-from-a"
    source = tmp_path / "model.nemo"
    dump = tmp_path / "nemo_state.safetensors"
    metadata = tmp_path / "meta.json"
    source.write_bytes(source_bytes)
    dump.write_bytes(dump_bytes)
    _write_metadata(metadata, source_bytes, dump_bytes)
    _, _, _, expected_source, expected_dump = identity.load_publish_metadata(metadata)
    assert identity.verify_dump_provenance(
        source,
        dump,
        expected_source_digest=expected_source,
        expected_dump_digest=expected_dump,
    ) == (_digest(source_bytes), _digest(dump_bytes))


@pytest.mark.parametrize("mutated", ("source", "dump"))
def test_split_dump_provenance_rejects_mismatched_inputs(
    tmp_path: Path, mutated: str
) -> None:
    source_bytes = b"checkpoint-a"
    dump_bytes = b"converted-from-a"
    source = tmp_path / "model.nemo"
    dump = tmp_path / "nemo_state.safetensors"
    metadata = tmp_path / "meta.json"
    source.write_bytes(source_bytes)
    dump.write_bytes(dump_bytes)
    _write_metadata(metadata, source_bytes, dump_bytes)
    _, _, _, expected_source, expected_dump = identity.load_publish_metadata(metadata)
    if mutated == "source":
        source.write_bytes(b"checkpoint-b")
    else:
        dump.write_bytes(b"converted-from-b")
    with pytest.raises(ValueError, match="disagrees with dump metadata"):
        identity.verify_dump_provenance(
            source,
            dump,
            expected_source_digest=expected_source,
            expected_dump_digest=expected_dump,
        )


@pytest.mark.parametrize(
    "missing", ("source_checkpoint_digest", "converted_dump_digest")
)
def test_publish_metadata_requires_both_provenance_digests(
    tmp_path: Path, missing: str
) -> None:
    metadata = {
        "source_checkpoint_digest": "sha256:" + "a" * 64,
        "converted_dump_digest": "sha256:" + "b" * 64,
        "prompt_dictionary": {"en-US": 0},
        "num_prompts": 128,
        "vocab_size": 13087,
    }
    del metadata[missing]
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match=missing):
        identity.load_publish_metadata(path)
