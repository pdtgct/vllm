# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for the PORT-owned masked page scatter."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]

_MODULE = (
    Path(__file__).resolve().parents[3] / "vllm/models/nemotron_asr/state_scatter.py"
)
_SPEC = importlib.util.spec_from_file_location("nemotron_state_scatter", _MODULE)
assert _SPEC is not None and _SPEC.loader is not None
state_scatter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(state_scatter)


# @spec PORT-ADV-004, PORT-STATE-008
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.int64])
def test_masked_page_scatter_writes_only_clean_rows(dtype: torch.dtype) -> None:
    pool = torch.arange(6 * 2 * 3).reshape(6, 2, 3).to(dtype)
    before = pool.clone()
    scratch = torch.full((3, 2, 3), 77, dtype=dtype)
    blocks = torch.tensor([4, 2, 5], dtype=torch.long)
    status = torch.tensor([0, 16, 0], dtype=torch.int32)

    state_scatter.masked_page_scatter_(pool, scratch, blocks, status)

    torch.testing.assert_close(pool[4], scratch[0], rtol=0, atol=0)
    torch.testing.assert_close(pool[5], scratch[2], rtol=0, atol=0)
    torch.testing.assert_close(pool[2], before[2], rtol=0, atol=0)
    # The reserved null page and every unrelated page are never targets.
    for block in (0, 1, 3):
        torch.testing.assert_close(pool[block], before[block], rtol=0, atol=0)


# @spec PORT-ADV-004
def test_masked_page_scatter_all_failed_is_exact_noop() -> None:
    pool = torch.randn(5, 7)
    before = pool.clone()
    scratch = torch.randn(3, 7)
    blocks = torch.tensor([4, 1, 3], dtype=torch.long)
    status = torch.tensor([1, 2, 4], dtype=torch.int32)

    state_scatter.masked_page_scatter_(pool, scratch, blocks, status)

    torch.testing.assert_close(pool, before, rtol=0, atol=0)


# @spec PORT-ADV-004, PORT-STATE-008
def test_dirty_rows_may_name_null_without_writing_it() -> None:
    pool = torch.randn(5, 4)
    before = pool.clone()
    scratch = torch.full((3, 4), float("nan"))
    blocks = torch.zeros(3, dtype=torch.long)
    status = torch.ones(3, dtype=torch.int32)

    state_scatter.masked_page_scatter_(pool, scratch, blocks, status)

    torch.testing.assert_close(pool, before, rtol=0, atol=0)


# @spec PORT-STATE-008
def test_masked_page_scatter_supports_padded_block_stride() -> None:
    backing = torch.arange(5 * 11, dtype=torch.float32)
    pool = torch.as_strided(backing, (5, 2, 3), (11, 3, 1))
    before = pool.clone()
    scratch = torch.full((2, 2, 3), 31.0)
    blocks = torch.tensor([3, 1], dtype=torch.long)
    status = torch.tensor([0, 8], dtype=torch.int32)

    state_scatter.masked_page_scatter_(pool, scratch, blocks, status)

    torch.testing.assert_close(pool[3], scratch[0], rtol=0, atol=0)
    torch.testing.assert_close(pool[1], before[1], rtol=0, atol=0)
    torch.testing.assert_close(pool[0], before[0], rtol=0, atol=0)


# @spec PORT-STATE-007, PORT-STATE-008
@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p, s, b, r: (p, s[:1], b, r), "row count"),
        (lambda p, s, b, r: (p, s.double(), b, r), "dtype"),
        (lambda p, s, b, r: (p, s[:, :1], b, r), "page shape"),
        (lambda p, s, b, r: (p, s, b.to(torch.int32), r), "int64"),
        (lambda p, s, b, r: (p, s, b, r.to(torch.int64)), "int32"),
        (
            lambda p, s, b, r: (
                p.transpose(1, 2).contiguous().transpose(1, 2),
                s,
                b,
                r,
            ),
            "row-major",
        ),
    ],
)
def test_masked_page_scatter_rejects_bad_descriptor(
    change: Callable[..., tuple[torch.Tensor, ...]], message: str
) -> None:
    pool = torch.zeros(5, 2, 3)
    scratch = torch.ones(2, 2, 3)
    blocks = torch.tensor([1, 3], dtype=torch.long)
    status = torch.zeros(2, dtype=torch.int32)
    before = pool.clone()
    args = change(pool, scratch, blocks, status)
    with pytest.raises(ValueError, match=message):
        state_scatter.masked_page_scatter_(*args)

    torch.testing.assert_close(pool, before, rtol=0, atol=0)
