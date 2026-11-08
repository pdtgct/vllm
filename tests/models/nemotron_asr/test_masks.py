# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chunked-limited attention mask: NeMo-exact construction.

Specs: PORT-POOL-002 (mask construction reproduces NeMo's
chunked-limited mask), PORT-POOL-003 (L1 CPU equivalence for all five
chunk configs before any P3 streaming code merges).

Two layers of assertion:
- Property tests (self-contained): intra-chunk attention is full
  (lookahead is intra-chunk), cross-chunk attention is causal, and the
  left window is ``56 // (L+1)`` whole chunks — NeMo truncates the left
  context to whole chunks (`left_chunks_num = att_context_size[0] //
  chunk_size`, conformer_encoder.py:832 @ de242add), which equals
  exactly 56 frames at every published config.
- Fixture tests: bitwise equality against masks produced by NeMo's own
  ``_create_masks`` at the P3 pin. Fixtures are generated in venv-oracle
  by ``generate_mask_fixtures.py`` (same directory) and committed; the
  test FAILS (never skips) when the fixture is absent.
"""

from pathlib import Path

import pytest
import torch

from vllm.models.nemotron_asr.masks import (
    PUBLISHED_ATT_CONTEXTS,
    chunked_limited_mask,
    left_window_chunks,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "nemo_masks.npz"

_CONFIGS = list(PUBLISHED_ATT_CONTEXTS.items())


@pytest.mark.parametrize(("label", "att_context"), _CONFIGS)
def test_intra_chunk_attention_is_full(label, att_context):
    left, lookahead = att_context
    chunk = lookahead + 1
    mask = chunked_limited_mask(6 * chunk, att_context)
    # Every frame sees every frame of its own chunk, future included.
    for start in range(0, 6 * chunk, chunk):
        block = mask[start : start + chunk, start : start + chunk]
        assert bool(block.all()), f"{label}: intra-chunk not full"


@pytest.mark.parametrize(("label", "att_context"), _CONFIGS)
def test_cross_chunk_attention_is_causal(label, att_context):
    chunk = att_context[1] + 1
    total = 6 * chunk
    mask = chunked_limited_mask(total, att_context)
    frame_chunk = torch.arange(total) // chunk
    future = frame_chunk.unsqueeze(1) < frame_chunk.unsqueeze(0)
    assert not bool(mask[future].any()), f"{label}: sees a future chunk"


@pytest.mark.parametrize(("label", "att_context"), _CONFIGS)
def test_left_window_is_whole_chunks_and_56_frames_when_divisible(label, att_context):
    left, lookahead = att_context
    chunk = lookahead + 1
    window_chunks = left_window_chunks(att_context)
    assert window_chunks == left // chunk
    # All five published configs divide exactly: window == 56 frames.
    assert window_chunks * chunk == 56, f"{label}: window != 56 frames"
    # A frame in a chunk beyond the window must not see chunk 0.
    total = (window_chunks + 3) * chunk
    mask = chunked_limited_mask(total, att_context)
    probe = (window_chunks + 1) * chunk  # first frame past the window
    assert not bool(mask[probe, 0]), f"{label}: window too wide"
    assert bool(mask[probe, chunk]), f"{label}: window too narrow"


def test_non_divisible_context_truncates_to_whole_chunks():
    # Untrained-but-runnable contexts (warning-only in NeMo) truncate:
    # [56, 4] -> chunk 5, 56 // 5 = 11 chunks = 55 frames, not 56.
    att_context = (56, 4)
    assert left_window_chunks(att_context) == 11
    chunk = 5
    total = 14 * chunk
    mask = chunked_limited_mask(total, att_context)
    probe = 12 * chunk  # chunk 12 sees chunks 1..12, not chunk 0
    assert not bool(mask[probe, chunk - 1])
    assert bool(mask[probe, chunk])


@pytest.mark.parametrize(("label", "att_context"), _CONFIGS)
def test_mask_matches_nemo_fixture(label, att_context):
    # PORT-POOL-003: bitwise equality with NeMo _create_masks output.
    # A missing fixture is a FAILURE (generate on the oracle venv with
    # generate_mask_fixtures.py), never a skip.
    assert FIXTURE_PATH.is_file(), (
        f"mask fixture missing: {FIXTURE_PATH} — generate it in venv-oracle via "
        "generate_mask_fixtures.py"
    )
    import numpy as np

    fixtures = np.load(FIXTURE_PATH)
    expected = torch.from_numpy(fixtures[label])
    ours = chunked_limited_mask(expected.shape[-1], att_context)
    assert torch.equal(ours, expected), f"{label}: mask != NeMo"
