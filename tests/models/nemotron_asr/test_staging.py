# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Atomic publication mechanics (PORT-WGT-004).

``staging.py`` is STDLIB-ONLY, loaded by file path like
``manifests.py`` — these tests run locally on macOS.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]

_STAGING_PATH = (
    Path(__file__).resolve().parents[3] / "vllm/models/nemotron_asr/staging.py"
)


def _load_staging() -> Any:
    spec = importlib.util.spec_from_file_location(
        "nemotron_asr_staging_under_test", _STAGING_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


staging = _load_staging()


def _tree(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): p.read_text()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_success_publishes_the_complete_artifact(tmp_path: Path) -> None:
    # @spec PORT-WGT-004
    dest = tmp_path / "served"

    def build(d: Path) -> None:
        (d / "a.json").write_text("A")
        (d / "sub").mkdir()
        (d / "sub" / "b.bin").write_text("B")

    staging.atomic_publish_dir(dest, build)
    assert _tree(dest) == {"a.json": "A", "sub/b.bin": "B"}
    assert list(tmp_path.iterdir()) == [dest]  # no staging litter


def test_build_failure_leaves_existing_artifact_untouched(
    tmp_path: Path,
) -> None:
    # @spec PORT-WGT-004
    # The finding-7 case: a failure mid-build (e.g. a manifest author
    # raising) must not damage an existing artifact or leave a
    # plausible-looking partial one.
    dest = tmp_path / "served"
    dest.mkdir()
    (dest / "a.json").write_text("ORIGINAL")
    before = _tree(dest)

    def build(d: Path) -> None:
        (d / "a.json").write_text("HALF-WRITTEN")
        raise RuntimeError("author failed")

    with pytest.raises(RuntimeError):
        staging.atomic_publish_dir(dest, build)
    assert _tree(dest) == before
    assert list(tmp_path.iterdir()) == [dest]  # staging removed


def test_build_failure_with_no_existing_artifact_leaves_nothing(
    tmp_path: Path,
) -> None:
    # @spec PORT-WGT-004
    dest = tmp_path / "served"

    def build(d: Path) -> None:
        (d / "a.json").write_text("HALF-WRITTEN")
        raise RuntimeError("author failed")

    with pytest.raises(RuntimeError):
        staging.atomic_publish_dir(dest, build)
    assert list(tmp_path.iterdir()) == []


def test_success_fully_replaces_a_previous_artifact(tmp_path: Path) -> None:
    # @spec PORT-WGT-004
    # Replacement is whole-directory: a file present only in the OLD
    # artifact must not survive into the new one.
    dest = tmp_path / "served"
    dest.mkdir()
    (dest / "stale-only-in-old.json").write_text("STALE")
    (dest / "a.json").write_text("OLD")

    def build(d: Path) -> None:
        (d / "a.json").write_text("NEW")

    staging.atomic_publish_dir(dest, build)
    assert _tree(dest) == {"a.json": "NEW"}
    assert list(tmp_path.iterdir()) == [dest]
