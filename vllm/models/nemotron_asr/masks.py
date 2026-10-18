# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chunked-limited attention mask arithmetic (NeMo-exact).

NeMo's chunked-limited style (``_create_masks``,
conformer_encoder.py:823-842 @ de242add) is block-causal at chunk
granularity: attention is full within a chunk (the lookahead is
intra-chunk), causal across chunks, and the left context is
``att_context_size[0] // chunk_size`` **whole chunks** — NeMo truncates
to chunk granularity, so the window is exactly ``left`` frames only
when ``chunk_size`` divides ``left`` (true for every published config
of this checkpoint: 56/(L+1) ∈ {56, 28, 14, 8, 4}).

Under chunk-per-scheduler-token pooling this mask is precisely
"full attention within the pooled block + causal across blocks +
sliding window of ``left_window_chunks`` blocks".
"""

from typing import Final

import torch

PUBLISHED_ATT_CONTEXTS: Final[dict[str, tuple[int, int]]] = {
    "80ms": (56, 0),
    "160ms": (56, 1),
    "320ms": (56, 3),
    "560ms": (56, 6),
    "1120ms": (56, 13),
}
"""Chunk-latency label -> ``att_context_size`` of the shipped checkpoint."""


def left_window_chunks(att_context: tuple[int, int]) -> int:
    """Number of prior whole chunks visible to a chunk (NeMo semantics)."""
    left, lookahead = att_context
    return left // (lookahead + 1)


def chunked_limited_mask(
    total_frames: int,
    att_context: tuple[int, int],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Boolean ``(total_frames, total_frames)`` mask, True = may attend.

    Reproduces NeMo's chunked-limited mask: ``allowed[i, j]`` iff
    ``0 <= chunk(i) - chunk(j) <= left_window_chunks``.
    """
    chunk_size = att_context[1] + 1
    chunk_idx = torch.arange(total_frames, dtype=torch.int64, device=device)
    chunk_idx = torch.div(chunk_idx, chunk_size, rounding_mode="trunc")
    diff = chunk_idx.unsqueeze(1) - chunk_idx.unsqueeze(0)
    window = left_window_chunks(att_context)
    return (diff >= 0) & (diff <= window)
