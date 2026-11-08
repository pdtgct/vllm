# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact-active decode versus the greedy oracle.

PORT-DEC-008 and PORT-PERF-001 require identical labels, order, per-row
lengths, and advanced predictor state, including masked encoder frames.
CPU tests use the real model package.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]

_PKG = Path(__file__).resolve().parents[3] / "vllm/models/nemotron_asr"


def _load_rnnt() -> Any:
    return importlib.import_module("vllm.models.nemotron_asr.rnnt")


rnnt = _load_rnnt()

VOCAB = 12
ENC = 32
PRED = 16


def _nets(seed: int = 7) -> tuple[Any, Any]:
    torch.manual_seed(seed)
    predictor = rnnt.Predictor(vocab_size=VOCAB, pred_hidden=PRED, pred_rnn_layers=2)
    joint = rnnt.Joint(
        enc_hidden=ENC,
        pred_hidden=PRED,
        joint_hidden=16,
        vocab_size=VOCAB,
    )
    # Damp the random recurrence: untrained weights explode to NaN
    # over cap-saturated 60-step runs, which would poison the exact
    # state comparison.
    with torch.no_grad():
        for module in (predictor, joint):
            for param in module.parameters():
                param.mul_(0.3)
    return predictor, joint


def _state(batch: int) -> Any:
    return rnnt.DecodeState(
        h=torch.zeros(2, batch, PRED),
        c=torch.zeros(2, batch, PRED),
        last_label=torch.full((batch,), VOCAB, dtype=torch.long),
    )


def _oracle_rows(
    predictor: Any,
    joint: Any,
    enc: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[list[list[int]], Any]:
    """Length-aware oracle: run greedy_decode_batch PER ROW on the
    row's valid frames only (the reference loop has no length mask)."""
    bursts: list[list[int]] = []
    hs, cs, lasts = [], [], []
    for b in range(enc.shape[0]):
        state = _state(1)
        labels, out = rnnt.greedy_decode_batch(
            enc[b : b + 1, : int(lengths[b])], predictor, joint, state
        )
        bursts.append(labels[0])
        hs.append(out.h)
        cs.append(out.c)
        lasts.append(out.last_label)
    return bursts, rnnt.DecodeState(
        h=torch.cat(hs, dim=1),
        c=torch.cat(cs, dim=1),
        last_label=torch.cat(lasts),
    )


#: Both length-aware production decodes are held to the SAME oracle
#: bar (PORT-DEC-008): the eager compact loop and the fixed-trip dense
#: masked loop (the sync-free candidate the dense-vs-compact profile
#: selects from at startup).
DECODE_FN_NAMES = ("decode_compact_active", "decode_dense_masked")


def _assert_matches_oracle(
    enc: torch.Tensor,
    lengths: torch.Tensor,
    *,
    seed: int = 7,
    fn_name: str = "decode_compact_active",
) -> None:
    predictor, joint = _nets(seed)
    decode = getattr(rnnt, fn_name)
    with torch.no_grad():
        expected, exp_state = _oracle_rows(predictor, joint, enc, lengths)
        ids, lens, state = decode(enc, lengths, predictor, joint, _state(enc.shape[0]))
    for b, burst in enumerate(expected):
        assert int(lens[b]) == len(burst), f"row {b} length"
        assert ids[b, : len(burst)].tolist() == burst, f"row {b} labels"
        assert not bool((ids[b, len(burst) :] != 0).any())  # padding
    assert not bool(state.h.isnan().any())  # fixture sanity
    # Labels/lengths/padding are EXACT above; the LSTM state compares
    # at a few-ulp bound — compacted-batch matmul reductions differ
    # from the oracle's per-row batch-1 runs by ~2e-9 (the same
    # cross-shape kernel effect as the frontend's FFT-plan bound).
    torch.testing.assert_close(state.h, exp_state.h, rtol=0, atol=1e-6)
    torch.testing.assert_close(state.c, exp_state.c, rtol=0, atol=1e-6)
    assert state.last_label.tolist() == exp_state.last_label.tolist()


@pytest.mark.parametrize("fn_name", DECODE_FN_NAMES)
def test_uniform_lengths_match_the_oracle(fn_name: str) -> None:
    # @spec PORT-DEC-008
    torch.manual_seed(11)
    enc = torch.randn(3, 6, ENC)
    _assert_matches_oracle(enc, torch.tensor([6, 6, 6]), fn_name=fn_name)


@pytest.mark.parametrize("fn_name", DECODE_FN_NAMES)
def test_padded_rows_never_decode_past_their_length(fn_name: str) -> None:
    # @spec PORT-DEC-008 / PORT-PERF-001
    # Mixed lengths incl. a zero-length row: padded frames carry loud
    # garbage that MUST NOT reach the joint.
    torch.manual_seed(12)
    enc = torch.randn(4, 8, ENC)
    lengths = torch.tensor([8, 3, 0, 5])
    for b, n in enumerate(lengths.tolist()):
        enc[b, n:] = 99.0  # poison the padding
    _assert_matches_oracle(enc, lengths, fn_name=fn_name)


@pytest.mark.parametrize("fn_name", DECODE_FN_NAMES)
def test_zero_length_batch_is_a_no_op(fn_name: str) -> None:
    # @spec PORT-PERF-001
    predictor, joint = _nets()
    enc = torch.randn(2, 4, ENC)
    with torch.no_grad():
        ids, lens, state = getattr(rnnt, fn_name)(
            enc, torch.tensor([0, 0]), predictor, joint, _state(2)
        )
    assert lens.tolist() == [0, 0]
    assert not bool((ids != 0).any())
    assert state.last_label.tolist() == [VOCAB, VOCAB]
    assert not bool(state.h.any()) and not bool(state.c.any())


@pytest.mark.parametrize("fn_name", DECODE_FN_NAMES)
def test_carried_state_continues_across_calls(fn_name: str) -> None:
    # @spec PORT-DEC-001
    # Two chunks through the production decode == one concatenated
    # chunk through the oracle (state carries, no SOS re-injection).
    predictor, joint = _nets(9)
    decode = getattr(rnnt, fn_name)
    torch.manual_seed(13)
    a = torch.randn(1, 4, ENC)
    b = torch.randn(1, 3, ENC)
    with torch.no_grad():
        whole, _ = rnnt.greedy_decode_batch(
            torch.cat([a, b], dim=1), predictor, joint, _state(1)
        )
        ids1, lens1, mid = decode(a, torch.tensor([4]), predictor, joint, _state(1))
        ids2, lens2, _ = decode(b, torch.tensor([3]), predictor, joint, mid)
    streamed = ids1[0, : int(lens1[0])].tolist() + ids2[0, : int(lens2[0])].tolist()
    assert streamed == whole[0]


@pytest.mark.parametrize("fn_name", DECODE_FN_NAMES)
def test_cap_saturated_burst_matches(fn_name: str) -> None:
    # @spec PORT-DEC-008
    # A quiet joint bias can drive long per-frame emissions; the
    # ten-symbol cap path must match the oracle exactly.
    predictor, joint = _nets(5)
    with torch.no_grad():
        joint.joint_net[1].bias[:VOCAB] += 3.0  # favor non-blank
    torch.manual_seed(14)
    enc = torch.randn(2, 3, ENC)
    lengths = torch.tensor([3, 2])
    predictor2, joint2 = _nets(5)
    with torch.no_grad():
        joint2.joint_net[1].bias[:VOCAB] += 3.0
        expected, _ = _oracle_rows(predictor2, joint2, enc, lengths)
        ids, lens, _ = getattr(rnnt, fn_name)(enc, lengths, predictor, joint, _state(2))
    for b, burst in enumerate(expected):
        assert len(burst) == int(lengths[b]) * rnnt.MAX_SYMBOLS_PER_STEP
        assert ids[b, : int(lens[b])].tolist() == burst


@pytest.mark.parametrize("fn_name", DECODE_FN_NAMES)
def test_mixed_carried_state_rows_match_the_oracle(fn_name: str) -> None:
    # @spec PORT-DEC-001 / PORT-DEC-008
    # Rows arriving with DISTINCT nonzero h/c/last_label (mid-session
    # resumes) must each evolve exactly as their own single-row run.
    predictor, joint = _nets(15)
    torch.manual_seed(16)
    enc = torch.randn(3, 5, ENC)
    lengths = torch.tensor([5, 2, 4])
    carried = rnnt.DecodeState(
        h=torch.randn(2, 3, PRED) * 0.2,
        c=torch.randn(2, 3, PRED) * 0.2,
        last_label=torch.tensor([1, 7, VOCAB]),
    )
    with torch.no_grad():
        ids, lens, state = getattr(rnnt, fn_name)(
            enc, lengths, predictor, joint, carried
        )
        for b in range(3):
            single = rnnt.DecodeState(
                h=carried.h[:, b : b + 1].clone(),
                c=carried.c[:, b : b + 1].clone(),
                last_label=carried.last_label[b : b + 1].clone(),
            )
            burst, out = rnnt.greedy_decode_batch(
                enc[b : b + 1, : int(lengths[b])],
                predictor,
                joint,
                single,
            )
            assert ids[b, : int(lens[b])].tolist() == burst[0]
            torch.testing.assert_close(state.h[:, b : b + 1], out.h, rtol=0, atol=1e-6)
            assert int(state.last_label[b]) == int(out.last_label[0])


@pytest.mark.parametrize("fn_name", DECODE_FN_NAMES)
def test_zero_length_carried_row_state_is_bit_identical(
    fn_name: str,
) -> None:
    # @spec PORT-PERF-001
    # A zero-length row alongside active peers: its carried state must
    # come back BIT-identical (no masked-arithmetic residue).
    predictor, joint = _nets(17)
    torch.manual_seed(18)
    enc = torch.randn(2, 4, ENC)
    carried = rnnt.DecodeState(
        h=torch.randn(2, 2, PRED) * 0.2,
        c=torch.randn(2, 2, PRED) * 0.2,
        last_label=torch.tensor([3, 9]),
    )
    with torch.no_grad():
        _, lens, state = getattr(rnnt, fn_name)(
            enc, torch.tensor([4, 0]), predictor, joint, carried
        )
    assert int(lens[1]) == 0
    torch.testing.assert_close(state.h[:, 1], carried.h[:, 1], rtol=0, atol=0)
    torch.testing.assert_close(state.c[:, 1], carried.c[:, 1], rtol=0, atol=0)
    assert int(state.last_label[1]) == 9


def test_out_of_range_lengths_are_rejected_or_clamped() -> None:
    # @spec PORT-DEC-008 / PORT-ADV-004
    # The eager compact loop raises ValueError from its host check
    # (it synchronizes anyway). The sync-free dense loop must not
    # raise, assert on device, or synchronize: it clamps lengths to
    # [0, T_pad] — a safe no-op posture; range defects upstream are
    # advance_session row-status territory.
    predictor, joint = _nets()
    torch.manual_seed(21)
    enc = torch.randn(2, 4, ENC)
    for bad, clamped in (([-1, 2], [0, 2]), ([2, 5], [2, 4])):
        with pytest.raises(ValueError, match="enc_lengths"):
            rnnt.decode_compact_active(
                enc, torch.tensor(bad), predictor, joint, _state(2)
            )
        with torch.no_grad():
            ids_b, lens_b, st_b = rnnt.decode_dense_masked(
                enc, torch.tensor(bad), predictor, joint, _state(2)
            )
            ids_c, lens_c, st_c = rnnt.decode_dense_masked(
                enc, torch.tensor(clamped), predictor, joint, _state(2)
            )
        assert ids_b.tolist() == ids_c.tolist()
        assert lens_b.tolist() == lens_c.tolist()
        torch.testing.assert_close(st_b.h, st_c.h, rtol=0, atol=0)
        assert st_b.last_label.tolist() == st_c.last_label.tolist()


def test_dense_and_compact_agree() -> None:
    # @spec PORT-DEC-008
    # The two production variants on identical inputs: labels, lengths,
    # and padding EXACT; LSTM state at the cross-shape ulp bound.
    predictor, joint = _nets(19)
    torch.manual_seed(20)
    enc = torch.randn(4, 6, ENC)
    lengths = torch.tensor([6, 0, 3, 5])
    carried = rnnt.DecodeState(
        h=torch.randn(2, 4, PRED) * 0.2,
        c=torch.randn(2, 4, PRED) * 0.2,
        last_label=torch.tensor([1, 7, VOCAB, 4]),
    )
    with torch.no_grad():
        ids_c, lens_c, st_c = rnnt.decode_compact_active(
            enc,
            lengths,
            predictor,
            joint,
            rnnt.DecodeState(
                h=carried.h.clone(),
                c=carried.c.clone(),
                last_label=carried.last_label.clone(),
            ),
        )
        ids_d, lens_d, st_d = rnnt.decode_dense_masked(
            enc, lengths, predictor, joint, carried
        )
    assert lens_c.tolist() == lens_d.tolist()
    assert ids_c.tolist() == ids_d.tolist()
    assert st_c.last_label.tolist() == st_d.last_label.tolist()
    torch.testing.assert_close(st_c.h, st_d.h, rtol=0, atol=1e-6)
    torch.testing.assert_close(st_c.c, st_d.c, rtol=0, atol=1e-6)


def test_manual_lstm_initializes_its_parameters() -> None:
    # @spec PORT-DEC-001
    # Regression for the uninitialized-parameter landmine (2026-07-18):
    # every parameter must be finite and inside nn.LSTM's uniform
    # bound after construction — removal of reset_parameters() fails
    # here deterministically, not as an allocator-dependent flap.
    cell_mod = sys.modules["vllm.models.nemotron_asr.rnnt_cell"]
    torch.manual_seed(0)
    lstm = cell_mod.ManualLSTM(input_size=8, hidden_size=8, num_layers=2)
    bound = 1.0 / (8**0.5)
    for name, param in lstm.named_parameters():
        assert bool(param.isfinite().all()), name
        assert float(param.abs().max()) <= bound, name
