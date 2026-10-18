# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loader-safe identity checks for split checkpoint publication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import regex as re


def sha256_file(path: Path) -> str:
    """Return the canonical SHA-256 digest string for ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def load_publish_metadata(
    path: Path,
) -> tuple[dict[str, int], int, int, str, str]:
    """Load and validate the dump's prompt, vocabulary, and provenance."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot read valid publish metadata at {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"publish metadata at {path} must be a JSON object")
    prompt_value = value.get("prompt_dictionary")
    num_prompts = value.get("num_prompts")
    vocab_size = value.get("vocab_size")
    source_checkpoint_digest = value.get("source_checkpoint_digest")
    converted_dump_digest = value.get("converted_dump_digest")
    if (
        isinstance(num_prompts, bool)
        or not isinstance(num_prompts, int)
        or num_prompts <= 0
    ):
        raise ValueError("meta.json num_prompts must be a positive integer")
    if (
        isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or vocab_size <= 0
    ):
        raise ValueError("meta.json vocab_size must be a positive integer")
    if not isinstance(prompt_value, dict) or not prompt_value:
        raise ValueError("meta.json must carry a non-empty prompt_dictionary")
    prompt_dictionary: dict[str, int] = {}
    for locale, index in prompt_value.items():
        if (
            not isinstance(locale, str)
            or not locale
            or isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < num_prompts
        ):
            raise ValueError(
                "prompt_dictionary must map non-empty locale strings to integer rows "
                f"in [0, {num_prompts})"
            )
        prompt_dictionary[locale] = index
    for name, digest in (
        ("source_checkpoint_digest", source_checkpoint_digest),
        ("converted_dump_digest", converted_dump_digest),
    ):
        if not isinstance(digest, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", digest
        ):
            raise ValueError(f"meta.json {name} must be a sha256:<64-hex> string")
    assert isinstance(source_checkpoint_digest, str)
    assert isinstance(converted_dump_digest, str)
    return (
        prompt_dictionary,
        num_prompts,
        vocab_size,
        source_checkpoint_digest,
        converted_dump_digest,
    )


def verify_dump_provenance(
    source_model: Path,
    nemo_state: Path,
    *,
    expected_source_digest: str,
    expected_dump_digest: str,
) -> tuple[str, str]:
    """Prove that the declared source and converted dump match metadata."""
    source_checkpoint_digest = sha256_file(source_model)
    if source_checkpoint_digest != expected_source_digest:
        raise ValueError(
            "source .nemo digest disagrees with dump metadata: "
            f"{source_checkpoint_digest} != {expected_source_digest}"
        )
    converted_dump_digest = sha256_file(nemo_state)
    if converted_dump_digest != expected_dump_digest:
        raise ValueError(
            "nemo_state.safetensors digest disagrees with dump metadata: "
            f"{converted_dump_digest} != {expected_dump_digest}"
        )
    return source_checkpoint_digest, converted_dump_digest
