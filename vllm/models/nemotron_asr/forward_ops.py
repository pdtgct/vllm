# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure forward-orchestration primitives (bring-up sub-slice BU-b).

The model's ``embed_multimodal`` / ``embed_input_ids`` / ``forward``
methods are thin engine-facing wrappers over these stateless functions;
isolating the logic here keeps it loader-testable off the engine and
keeps the invariant that ``embed_multimodal`` is a pure function of the
chunk bytes (PORT-INT-003 — the engine content-hash-caches its output,
so identical silence chunks must collide harmlessly).

The audio carrier (D-BU-2): one chunk's mel is packed into a single
``inputs_embeds`` row of width ``hidden_size`` (the only conduit into
``forward`` for a resumed session), slot 0 holding the mel-frame count
so the row is self-describing (tail chunks are shorter). Row roles at a
decode step (D-BU-4 / PORT-DEC-009) are read from the token id alone —
the only signal that distinguishes single-token chunk/replay/flush rows.
"""

from collections.abc import Sequence

import torch

#: Row roles at a scheduler step.
ROLE_CHUNK = 0  # a placeholder token: ingest audio, run encoder + decode
ROLE_REPLAY = 1  # a queued label to emit
ROLE_FLUSH = 2  # the engine's finish sentinel on a drained queue (PORT-DEC-009)


def pack_audio_carrier(mel: torch.Tensor, *, hidden_size: int) -> torch.Tensor:
    """Pack one chunk's mel into a self-describing carrier row.

    ``mel``: ``(feat, frames)``. Returns ``(hidden_size,)`` — slot 0 is
    the frame count, ``[1 : 1 + feat*frames]`` the flattened mel, the
    rest zero.

    Raises:
        ValueError: If the mel does not fit in ``hidden_size - 1``.

    """
    feat, frames = mel.shape
    payload = feat * frames
    if 1 + payload > hidden_size:
        raise ValueError(
            f"mel ({feat}×{frames} = {payload}) + the frame-count slot exceeds "
            f"hidden_size={hidden_size}"
        )
    row = torch.zeros(hidden_size, dtype=mel.dtype, device=mel.device)
    row[0] = float(frames)
    row[1 : 1 + payload] = mel.reshape(-1)
    return row


def unpack_audio_carrier(row: torch.Tensor, *, feat: int) -> torch.Tensor:
    """Recover ``(feat, frames)`` mel from a carrier row.

    Reads the frame count from slot 0; the inverse of
    ``pack_audio_carrier``.
    """
    frames = int(round(float(row[0])))
    payload = feat * frames
    return row[1 : 1 + payload].reshape(feat, frames)


def merge_mm_embeddings(
    input_ids: torch.Tensor,
    mm_embeds: torch.Tensor | Sequence[torch.Tensor],
    is_multimodal: torch.Tensor,
    *,
    hidden_size: int,
) -> torch.Tensor:
    """``embed_input_ids`` body: carrier rows in, zeros elsewhere.

    This model has no LM embedding table — replay/flush rows carry
    their id in ``input_ids`` and the forward reads it there, so their
    ``inputs_embeds`` rows are zero. The multimodal (placeholder) rows,
    marked by ``is_multimodal``, receive ``mm_embeds`` in order.
    Returns ``(len(input_ids), hidden_size)``.

    ``mm_embeds`` arrives either as one flat ``(n_mm, hidden_size)``
    tensor or, from the runner's ``_gather_mm_embeddings``, as a
    sequence of per-item 2D ``(num_tokens_i, hidden_size)`` tensors;
    both normalize to ``(n_mm, hidden_size)`` carrier rows.

    Raises:
        ValueError: If ``is_multimodal.sum()`` does not equal the total
            number of carrier rows.

    """
    if isinstance(mm_embeds, torch.Tensor):
        mm_embeds = mm_embeds.reshape(-1, hidden_size)
    elif len(mm_embeds):
        mm_embeds = torch.cat([e.reshape(-1, hidden_size) for e in mm_embeds], dim=0)
    else:
        mm_embeds = torch.zeros(
            0, hidden_size, dtype=torch.float32, device=input_ids.device
        )
    n_mm = int(is_multimodal.sum())
    if n_mm != mm_embeds.shape[0]:
        raise ValueError(f"{n_mm} multimodal rows but {mm_embeds.shape[0]} embeddings")
    out = torch.zeros(
        input_ids.shape[0],
        hidden_size,
        dtype=mm_embeds.dtype,
        device=input_ids.device,
    )
    out[is_multimodal] = mm_embeds.to(out.dtype)
    return out


def classify_step_rows(
    input_ids: torch.Tensor,
    *,
    placeholder_id: int,
    queue_lengths: torch.Tensor,
) -> torch.Tensor:
    """Per-row role from the token id and the remaining queue length.

    ``placeholder_id`` → ``ROLE_CHUNK``; any other id with a non-empty
    replay queue → ``ROLE_REPLAY``; any other id with a drained queue →
    ``ROLE_FLUSH`` (the engine's ``[0]`` finish sentinel, PORT-DEC-009 —
    emitted as park, never echo-guarded). ``queue_lengths`` is the
    remaining-labels count per row.
    """
    # Start FLUSH (non-placeholder, drained); non-empty queue → REPLAY;
    # a placeholder id overrides to CHUNK regardless of queue.
    roles = torch.full_like(input_ids, ROLE_FLUSH)
    roles = torch.where(
        queue_lengths > 0, torch.full_like(input_ids, ROLE_REPLAY), roles
    )
    roles = torch.where(
        input_ids == placeholder_id,
        torch.full_like(input_ids, ROLE_CHUNK),
        roles,
    )
    return roles
