# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BU-b: forward-orchestration primitives (loader-runnable, torch-only).

Specs: PORT-INT-003 (the stateless audio carrier — one chunk's mel in
one inputs_embeds row; embed_input_ids merges carrier rows and zeros
elsewhere), PORT-DEC-009 (the flush-row classification), PORT-DEC-002
(chunk-vs-replay discrimination by token id). Consult: D-BU-2/4.

The model methods (embed_multimodal / embed_input_ids / forward) that
wrap these, the mm-processor registration, and forward over bound page
pools are BU-c (engine tier).
"""

import pytest
import torch

from vllm.models.nemotron_asr.forward_ops import (
    ROLE_CHUNK,
    ROLE_FLUSH,
    ROLE_REPLAY,
    classify_step_rows,
    merge_mm_embeddings,
    pack_audio_carrier,
    unpack_audio_carrier,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]

FEAT = 128
HIDDEN = 128 * 121 + 1  # feat × max mel frames + the frame-count slot


# ---- the audio carrier (PORT-INT-003, D-BU-2) ---------------------------------


def test_carrier_roundtrips_and_is_self_describing():
    torch.manual_seed(3)
    mel = torch.randn(FEAT, 40)
    row = pack_audio_carrier(mel, hidden_size=HIDDEN)
    assert row.shape == (HIDDEN,)
    assert int(round(float(row[0]))) == 40  # slot 0 = frame count
    torch.testing.assert_close(
        unpack_audio_carrier(row, feat=FEAT), mel, rtol=0.0, atol=0.0
    )


def test_carrier_zeros_the_unused_tail():
    mel = torch.randn(FEAT, 10)
    row = pack_audio_carrier(mel, hidden_size=HIDDEN)
    used = 1 + FEAT * 10
    assert torch.count_nonzero(row[used:]) == 0


def test_carrier_is_stateless():
    # PORT-INT-003 invariant (load-bearing for correctness: the engine
    # content-hash-caches embed_multimodal, so identical chunks must
    # produce identical carriers). Same bytes → same row, bit-for-bit.
    mel = torch.randn(FEAT, 25)
    a = pack_audio_carrier(mel.clone(), hidden_size=HIDDEN)
    b = pack_audio_carrier(mel.clone(), hidden_size=HIDDEN)
    torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)


def test_carrier_rejects_overflow():
    # A mel too large for the carrier width must fail loudly, never
    # silently truncate.
    mel = torch.randn(FEAT, 200)  # 128*200 + 1 >> HIDDEN
    with pytest.raises(ValueError):
        pack_audio_carrier(mel, hidden_size=HIDDEN)


def test_tail_chunk_shorter_carrier_roundtrips():
    # A short final tail (PORT-SESS-003) packs and unpacks by its own
    # frame count, not a fixed width.
    mel = torch.randn(FEAT, 4)
    row = pack_audio_carrier(mel, hidden_size=HIDDEN)
    torch.testing.assert_close(
        unpack_audio_carrier(row, feat=FEAT), mel, rtol=0.0, atol=0.0
    )


# ---- embed_input_ids merge (PORT-INT-003) -------------------------------------


def test_merge_places_carriers_and_zeros_the_rest():
    # 4 rows: chunk, replay, chunk, flush. The two chunk rows get the
    # two carriers in order; replay/flush rows are zero (their ids
    # travel via input_ids, not embeddings — the model has no LM table).
    input_ids = torch.tensor([500, 7, 501, 0], dtype=torch.long)
    is_mm = torch.tensor([True, False, True, False])
    mm_embeds = torch.stack([torch.full((HIDDEN,), 1.0), torch.full((HIDDEN,), 2.0)])
    out = merge_mm_embeddings(input_ids, mm_embeds, is_mm, hidden_size=HIDDEN)
    assert out.shape == (4, HIDDEN)
    assert torch.all(out[0] == 1.0) and torch.all(out[2] == 2.0)
    assert torch.count_nonzero(out[1]) == 0
    assert torch.count_nonzero(out[3]) == 0


def test_merge_rejects_count_mismatch():
    input_ids = torch.tensor([500, 7], dtype=torch.long)
    is_mm = torch.tensor([True, False])
    two_embeds = torch.zeros(2, HIDDEN)  # 2 embeds, 1 mm row
    with pytest.raises(ValueError):
        merge_mm_embeddings(input_ids, two_embeds, is_mm, hidden_size=HIDDEN)


# ---- row classification + the flush rule (PORT-DEC-009, D-BU-4) ----------------


def test_classify_discriminates_chunk_replay_flush():
    placeholder = 13089
    input_ids = torch.tensor([placeholder, 7, 0, placeholder], dtype=torch.long)
    # row 1 (id 7) has queued labels → REPLAY; row 2 (id 0) drained →
    # FLUSH (the engine's finish sentinel on an empty queue).
    queue_lengths = torch.tensor([3, 2, 0, 3], dtype=torch.long)
    roles = classify_step_rows(
        input_ids, placeholder_id=placeholder, queue_lengths=queue_lengths
    )
    assert roles.tolist() == [ROLE_CHUNK, ROLE_REPLAY, ROLE_FLUSH, ROLE_CHUNK]


def test_flush_id_zero_on_drained_queue_is_flush_not_replay():
    # PORT-DEC-009: token id 0 is a real label, but arriving on a
    # DRAINED queue it is the engine's [0] finish sentinel — flush,
    # never a replay emission (which would echo-guard-fail).
    placeholder = 13089
    roles = classify_step_rows(
        torch.tensor([0], dtype=torch.long),
        placeholder_id=placeholder,
        queue_lengths=torch.tensor([0], dtype=torch.long),
    )
    assert roles.tolist() == [ROLE_FLUSH]
    # The same id 0 WITH a live queue is a genuine replay label.
    roles_live = classify_step_rows(
        torch.tensor([0], dtype=torch.long),
        placeholder_id=placeholder,
        queue_lengths=torch.tensor([2], dtype=torch.long),
    )
    assert roles_live.tolist() == [ROLE_REPLAY]
