# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Atomic directory publication (PORT-WGT-004).

A published checkpoint directory must never be observable half-written:
``publish.py`` builds the complete artifact in a fresh sibling staging
directory and only then swaps it into place. A failure while building
leaves an existing destination byte-identical and removes the staging
directory; the destination is touched only after every file write has
succeeded.

STDLIB ONLY — testable locally by file-path load (``test_staging.py``),
like ``manifests.py``.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path


def atomic_publish_dir(out_dir: Path, build: Callable[[Path], None]) -> None:
    """Build into a staging sibling of ``out_dir``, then swap it in.

    ``build`` receives the empty staging directory and must write the
    COMPLETE artifact into it. If it raises, the staging directory is
    removed and ``out_dir`` (existing or absent) is untouched. On
    success the previous ``out_dir`` (if any) is renamed aside, the
    staging directory is renamed into place, and the old artifact is
    deleted — same-filesystem renames, so no reader ever sees a
    partial ``out_dir``.

    Args:
        out_dir: the destination directory.
        build: writes the complete artifact into the staging directory
            it is given.

    Raises:
        Whatever ``build`` raises, after staging cleanup.

    """
    parent = out_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{out_dir.name}.staging-{os.getpid()}"
    replaced = parent / f".{out_dir.name}.replaced-{os.getpid()}"
    for leftover in (staging, replaced):
        if leftover.exists():
            shutil.rmtree(leftover)
    staging.mkdir()
    try:
        build(staging)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if out_dir.exists():
        os.rename(out_dir, replaced)
    os.rename(staging, out_dir)
    shutil.rmtree(replaced, ignore_errors=True)
