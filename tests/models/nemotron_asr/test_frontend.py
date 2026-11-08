# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Differential proof for the exact bounded frontend (PORT-FEAT-002).

``frontend.py`` and ``featurizer.py`` import only ``torch`` — loaded
by file path, these tests run locally on macOS. The comparison unit is
``(mel_window, valid_mel_length, next_frontend_state)`` against the
whole-prefix ``MelFeaturizer`` (design §Exact Bounded Frontend), over
arbitrary packet splits, every cadence, exact and partial boundaries,
pre-emphasis discontinuities, silence, and final tails.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]

_PKG = Path(__file__).resolve().parents[3] / "vllm/models/nemotron_asr"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        f"nemotron_asr_{name}_under_test", _PKG / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


frontend = _load("frontend")
featurizer_mod = _load("featurizer")
manifests = _load("manifests")

N_MELS = 16
RAW_TAIL_CAP = 1953


def _featurizer() -> Any:
    torch.manual_seed(3)
    return featurizer_mod.MelFeaturizer(
        filterbank=torch.rand(N_MELS, 257) * 0.01,
        window=torch.hann_window(400),
    )


def _fresh_state(batch: int = 1) -> dict[str, torch.Tensor]:
    return {
        "raw_tail": torch.zeros(batch, RAW_TAIL_CAP),
        "mel_tail": torch.zeros(batch, N_MELS, frontend.MEL_TAIL_FRAMES),
        "counters": torch.zeros(batch, 8, dtype=torch.int64),
    }


def _whole_mel(feat: Any, signal: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        mel, mel_len = feat(signal.unsqueeze(0), torch.tensor([signal.shape[0]]))
    trimmed: torch.Tensor = mel[0, :, : int(mel_len[0])]
    return trimmed


def _stream(
    feat: Any,
    signal: torch.Tensor,
    splits: list[int],
    *,
    final: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Feed ``signal`` through the incremental frontend in ``splits``-
    sized packets (the last one final iff ``final``); concatenate the
    committed frames (the valid prefix of each padded return)."""
    state = _fresh_state()
    committed: list[torch.Tensor] = []
    offset = 0
    for i, n in enumerate(splits):
        chunk = signal[offset : offset + n]
        offset += n
        is_final = final and i == len(splits) - 1
        # Arbitrary-packet differentials cap at stability itself (the
        # loosest legal target); cadence tests pass real boundaries.
        target = frontend.stable_frames(offset, n_fft=512, hop=160)
        # The harness host-knows the exact commit; production passes the
        # admitted cadence C (design §Exact Bounded Frontend). Arbitrary
        # packet tests deliberately use a cadence no smaller than their
        # exact host-known commit so they exercise packetization rather
        # than the ingress cadence walk. Finals may commit past the
        # stable target.
        pad = (
            target
            - int(state["counters"][0, frontend.CTR_COMMITTED_MEL_FRAMES])
            + (frontend.MEL_TAIL_FRAMES if is_final else 0)
        )
        out, counts, status = frontend.advance_frontend(
            feat,
            chunk.unsqueeze(0),
            torch.tensor([n], dtype=torch.long),
            torch.tensor([is_final]),
            torch.tensor([target], dtype=torch.long),
            **state,
            cadence_frames=max(pad, 1),
            pad_frames=max(pad, 0),
        )
        assert status.tolist() == [0]
        assert out.shape[2] == max(pad, 0)  # fixed host-derived width
        if int(counts[0]):
            committed.append(out[0, :, : int(counts[0])])
        # Padded columns are exactly zero (PORT-ADV-004 zeroing).
        torch.testing.assert_close(
            out[0, :, int(counts[0]) :],
            torch.zeros(N_MELS, out.shape[2] - int(counts[0])),
            rtol=0,
            atol=0,
        )
    assert offset == signal.shape[0]
    mel = torch.cat(committed, dim=1) if committed else torch.zeros(N_MELS, 0)
    return mel, state


def _assert_prefix_equal(
    feat: Any, signal: torch.Tensor, streamed: torch.Tensor
) -> None:
    """Streamed frames must equal the same frames of the whole-signal
    featurization — committed frames are FINAL, never recomputed."""
    whole = _whole_mel(feat, signal)
    n = streamed.shape[1]
    assert n <= whole.shape[1]
    torch.testing.assert_close(streamed, whole[:, :n], rtol=0, atol=2e-6)


CADENCE_SAMPLES = [1280, 2560, 5120, 8960, 17920]


@pytest.mark.parametrize("chunk", CADENCE_SAMPLES)
def test_final_run_matches_whole_signal_exactly(chunk: int) -> None:
    # @spec PORT-FEAT-002
    # Every cadence, four chunks, final tail: the committed stream must
    # equal the whole-signal featurization bit-for-bit, full length.
    feat = _featurizer()
    torch.manual_seed(11)
    signal = torch.randn(4 * chunk) * 0.1
    streamed, state = _stream(feat, signal, [chunk] * 4, final=True)
    whole = _whole_mel(feat, signal)
    assert streamed.shape == whole.shape
    torch.testing.assert_close(streamed, whole, rtol=0, atol=2e-6)
    assert int(state["counters"][0, frontend.CTR_FINALIZED]) == 1


def test_nonfinal_commits_are_a_stable_prefix() -> None:
    # @spec PORT-FEAT-002
    # Without a final tail, commits are exactly the stable frames — a
    # strict prefix of the whole-signal result, never recomputed later.
    feat = _featurizer()
    torch.manual_seed(12)
    signal = torch.randn(3 * 17920) * 0.1
    streamed, state = _stream(feat, signal, [17920] * 3, final=False)
    _assert_prefix_equal(feat, signal, streamed)
    n = signal.shape[0]
    expected = frontend.stable_frames(n, n_fft=512, hop=160)
    assert streamed.shape[1] == expected
    ctr = state["counters"][0]
    assert int(ctr[frontend.CTR_COMMITTED_MEL_FRAMES]) == expected
    assert int(ctr[frontend.CTR_TOTAL_VALID_SAMPLES]) == n
    assert int(ctr[frontend.CTR_FINALIZED]) == 0


def test_arbitrary_packet_splits_match_cadence_splits() -> None:
    # @spec PORT-FEAT-002
    # Transport packet boundaries must not affect the committed frames:
    # ragged splits vs one whole-final call, identical output.
    feat = _featurizer()
    torch.manual_seed(13)
    signal = torch.randn(20_000) * 0.1
    ragged = [1, 159, 160, 4096, 2, 7000, 8582]
    assert sum(ragged) == 20_000
    streamed_r, _ = _stream(feat, signal, ragged, final=True)
    streamed_w, _ = _stream(feat, signal, [20_000], final=True)
    # Frames whose STFT ran in different-shaped calls (a 1-frame FFT
    # vs a batched one) differ by FFT-plan ulps that log() surfaces at
    # quiet bins (~2e-7 quiet, ~1.4e-6 loud bins on cu130 torch); the few-ulp
    # bound is the honest bar for cross-shape comparisons.
    torch.testing.assert_close(streamed_r, streamed_w, rtol=0, atol=2e-6)
    torch.testing.assert_close(streamed_r, _whole_mel(feat, signal), rtol=0, atol=2e-6)


def test_preemphasis_continuity_across_a_boundary() -> None:
    # @spec PORT-FEAT-002
    # A hard discontinuity exactly at a packet boundary: the
    # cross-boundary x[t-1] term must come from the retained tail.
    feat = _featurizer()
    signal = torch.cat([torch.full((8960,), 0.5), torch.full((8960,), -0.5)])
    streamed, _ = _stream(feat, signal, [8960, 8960], final=True)
    torch.testing.assert_close(streamed, _whole_mel(feat, signal), rtol=0, atol=2e-6)


def test_silence_matches() -> None:
    # @spec PORT-FEAT-002
    feat = _featurizer()
    signal = torch.zeros(2 * 17920)
    streamed, _ = _stream(feat, signal, [17920, 17920], final=True)
    torch.testing.assert_close(streamed, _whole_mel(feat, signal), rtol=0, atol=2e-6)


def test_partial_final_tail_off_the_hop_grid() -> None:
    # @spec PORT-FEAT-002
    # A final residual that is not a multiple of the hop: committed
    # length must equal output_lengths (floor), matching whole-signal.
    feat = _featurizer()
    torch.manual_seed(14)
    n = 17920 + 4321
    signal = torch.randn(n) * 0.1
    streamed, _ = _stream(feat, signal, [17920, 4321], final=True)
    whole = _whole_mel(feat, signal)
    assert streamed.shape == whole.shape == (N_MELS, n // 160)
    torch.testing.assert_close(streamed, whole, rtol=0, atol=2e-6)


def test_zero_sample_final_tail_finalizes_and_drops_short_residual() -> None:
    # @spec PORT-FEAT-002
    # A final tail carrying zero samples still atomically finalizes;
    # its 1-frame residual past the committed boundary (112 - 111) is
    # BELOW eight and therefore dropped, never committed (design final
    # residual rule).
    feat = _featurizer()
    torch.manual_seed(15)
    signal = torch.randn(17920) * 0.1
    streamed, state = _stream(feat, signal, [17920, 0], final=True)
    assert streamed.shape[1] == frontend.stable_frames(17920, n_fft=512, hop=160)
    _assert_prefix_equal(feat, signal, streamed)
    assert int(state["counters"][0, frontend.CTR_FINALIZED]) == 1


def test_chunk_after_finalization_is_a_masked_no_op() -> None:
    # @spec PORT-ADV-004
    # Audio after finalization is a per-row protocol violation: the row
    # mutates nothing and commits nothing, and the named status bit is
    # set (device-resolved, never a raise — the transaction consumes
    # the status at its commit sync). The frontend derives the
    # finalized predicate itself.
    feat = _featurizer()
    state = _fresh_state()
    args = (
        torch.zeros(1, 160),
        torch.tensor([160], dtype=torch.long),
        torch.tensor([True]),
        torch.zeros(1, dtype=torch.long),  # target ignored on final
    )
    frontend.advance_frontend(feat, *args, **state, cadence_frames=8, pad_frames=8)
    before = {k: v.clone() for k, v in state.items()}
    out, counts, status = frontend.advance_frontend(
        feat, *args, **state, cadence_frames=8, pad_frames=8
    )
    assert int(counts[0]) == 0
    assert status.tolist() == [frontend.ROW_STATUS_FINALIZED]
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
    for key, prev in before.items():
        torch.testing.assert_close(state[key], prev, rtol=0, atol=0)


def test_mel_tail_holds_the_frames_before_the_boundary() -> None:
    # @spec PORT-FEAT-002
    # The mel tail must hold the MEL_TAIL_FRAMES committed frames
    # preceding the boundary — the encoder's pre-encode cache source.
    feat = _featurizer()
    torch.manual_seed(16)
    signal = torch.randn(2 * 17920) * 0.1
    streamed, state = _stream(feat, signal, [17920, 17920], final=False)
    k = frontend.MEL_TAIL_FRAMES
    torch.testing.assert_close(
        state["mel_tail"][0], streamed[:, -k:], rtol=0, atol=2e-6
    )
    assert int(state["counters"][0, frontend.CTR_MEL_TAIL_LENGTH]) == k


LOOKAHEADS = {1280: 0, 2560: 1, 5120: 3, 8960: 6, 17920: 13}


@pytest.mark.parametrize(("chunk", "la"), sorted(LOOKAHEADS.items()))
def test_cadence_boundary_margin_is_six(chunk: int, la: int) -> None:
    # @spec PORT-FEAT-002
    # The option-(d) premise (ledger 2026-07-17): after k regular
    # cadence units, stable frames exceed the approved encoder
    # boundary B_k = (8L+1)+(k-1)C = kC-7 by EXACTLY six, uniformly.
    for k in (1, 2, 3, 7):
        stable = frontend.stable_frames(k * chunk, n_fft=512, hop=160)
        boundary = frontend.cadence_boundary(k, lookahead=la)
        assert boundary == k * 8 * (la + 1) - 7
        assert stable - boundary == 6


def test_cadence_capped_run_commits_exactly_the_boundaries() -> None:
    # @spec PORT-FEAT-002
    # Regular 1120 ms units under real cadence targets: commits land
    # exactly on B_k each update, remain a prefix of the whole-signal
    # featurization, and the final tail flushes the residual so the
    # full stream equals the whole signal.
    feat = _featurizer()
    torch.manual_seed(18)
    chunk, la = 17920, 13
    signal = torch.randn(3 * chunk) * 0.1
    state = _fresh_state()
    committed: list[torch.Tensor] = []
    # The production bucket bound is one reference cadence shift C,
    # uniform for first, continuing, and final rows.
    cadence = 8 * (la + 1)
    pad = cadence
    for k in (1, 2, 3):
        out, counts, status = frontend.advance_frontend(
            feat,
            signal[(k - 1) * chunk : k * chunk].unsqueeze(0),
            torch.tensor([chunk], dtype=torch.long),
            torch.tensor([False]),
            torch.tensor(
                [frontend.cadence_boundary(k, lookahead=la)],
                dtype=torch.long,
            ),
            **state,
            cadence_frames=cadence,
            pad_frames=pad,
        )
        assert status.tolist() == [0]
        committed.append(out[0, :, : int(counts[0])])
        assert int(
            state["counters"][0, frontend.CTR_COMMITTED_MEL_FRAMES]
        ) == frontend.cadence_boundary(k, lookahead=la)
    streamed = torch.cat(committed, dim=1)
    _assert_prefix_equal(feat, signal, streamed)
    # The mel tail sits relative to the CAPPED boundary.
    k9 = frontend.MEL_TAIL_FRAMES
    torch.testing.assert_close(
        state["mel_tail"][0], streamed[:, -k9:], rtol=0, atol=2e-6
    )
    out, counts, status = frontend.advance_frontend(
        feat,
        torch.zeros(1, 0),
        torch.zeros(1, dtype=torch.long),
        torch.tensor([True]),
        torch.zeros(1, dtype=torch.long),
        **state,
        cadence_frames=cadence,
        pad_frames=pad,
    )
    # Exact-cadence finalization: final_frames - B_3 = 7 new frames
    # remain — below eight, so the residual is DROPPED and the
    # committed stream ends exactly at the capped boundary (the design
    # final residual rule; nothing further reaches the encoder).
    assert status.tolist() == [0]
    assert int(counts[0]) == 0
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
    assert int(
        state["counters"][0, frontend.CTR_COMMITTED_MEL_FRAMES]
    ) == frontend.cadence_boundary(3, lookahead=la)
    assert int(state["counters"][0, frontend.CTR_FINALIZED]) == 1
    _assert_prefix_equal(feat, signal, streamed)


@pytest.mark.parametrize(("chunk", "la"), sorted(LOOKAHEADS.items()))
def test_final_tail_consumes_one_reference_cadence_shift(chunk: int, la: int) -> None:
    # @spec PORT-FEAT-004
    # After one regular unit the committed boundary trails the complete
    # mel grid by seven frames. A maximum legal final residual therefore
    # exposes C+6 frames past the boundary. The reference buffer walk
    # consumes exactly one C-frame shift and drops the remaining six;
    # consuming all C+6 would create a cadence-dependent extra encoder
    # output at 80/160/320 ms.
    feat = _featurizer()
    torch.manual_seed(180 + la)
    cadence = 8 * (la + 1)
    signal = torch.randn(2 * chunk - 1) * 0.1
    state = _fresh_state()
    first_boundary = frontend.cadence_boundary(1, lookahead=la)
    frontend.advance_frontend(
        feat,
        signal[:chunk].unsqueeze(0),
        torch.tensor([chunk], dtype=torch.long),
        torch.tensor([False]),
        torch.tensor([first_boundary], dtype=torch.long),
        **state,
        cadence_frames=cadence,
        pad_frames=cadence,
    )
    raw_tail_before_final = state["raw_tail"].clone()
    out, counts, status = frontend.advance_frontend(
        feat,
        signal[chunk:].unsqueeze(0),
        torch.tensor([chunk - 1], dtype=torch.long),
        torch.tensor([True]),
        torch.zeros(1, dtype=torch.long),
        **state,
        cadence_frames=cadence,
        pad_frames=cadence,
    )
    assert status.tolist() == [0]
    assert counts.tolist() == [cadence]
    assert out.shape == (1, N_MELS, cadence)
    whole = _whole_mel(feat, signal)
    expected = whole[:, first_boundary : first_boundary + cadence]
    torch.testing.assert_close(out[0], expected, rtol=0, atol=2e-6)
    torch.testing.assert_close(
        state["mel_tail"][0],
        whole[
            :,
            first_boundary + cadence - frontend.MEL_TAIL_FRAMES : first_boundary
            + cadence,
        ],
        rtol=0,
        atol=2e-6,
    )
    # Finalization clears raw-tail LOGICAL residency without a needless
    # physical write; the complete next state is still deterministic.
    torch.testing.assert_close(state["raw_tail"], raw_tail_before_final, rtol=0, atol=0)
    assert (
        int(state["counters"][0, frontend.CTR_COMMITTED_MEL_FRAMES])
        == first_boundary + cadence
    )
    assert state["counters"][0].tolist() == [
        signal.shape[0],
        first_boundary + cadence,
        0,
        signal.shape[0],
        0,
        frontend.MEL_TAIL_FRAMES,
        0,
        1,
    ]
    assert int(state["counters"][0, frontend.CTR_FINALIZED]) == 1


def test_target_past_stability_is_a_margin_violation() -> None:
    # @spec PORT-ADV-004
    # Design-invariant violations are port defects reported as named
    # status bits — never a raise, never a device assertion (a fired
    # CUDA assert corrupts the context and kills every resident
    # session): the row is a safe masked no-op.
    feat = _featurizer()
    state = _fresh_state()
    before = {k: v.clone() for k, v in state.items()}
    out, counts, status = frontend.advance_frontend(
        feat,
        torch.zeros(1, 1600),
        torch.tensor([1600], dtype=torch.long),
        torch.tensor([False]),
        torch.tensor([99], dtype=torch.long),  # stable is only 9
        **state,
        cadence_frames=99,
        pad_frames=99,
    )
    assert status.tolist() == [frontend.ROW_STATUS_MARGIN]
    assert int(counts[0]) == 0
    assert not bool(out.any())
    for key, prev in before.items():
        torch.testing.assert_close(state[key], prev, rtol=0, atol=0)


def test_target_below_committed_is_rejected() -> None:
    # @spec PORT-ADV-004
    feat = _featurizer()
    torch.manual_seed(19)
    signal = torch.randn(17920) * 0.1
    _, state = _stream(feat, signal, [17920], final=False)
    before = {k: v.clone() for k, v in state.items()}
    out, counts, status = frontend.advance_frontend(
        feat,
        torch.zeros(1, 160),
        torch.tensor([160], dtype=torch.long),
        torch.tensor([False]),
        torch.tensor([1], dtype=torch.long),
        **state,
        cadence_frames=8,
        pad_frames=8,
    )
    assert (
        int(status[0]) & frontend.ROW_STATUS_TARGET_ORDER
    ) == frontend.ROW_STATUS_TARGET_ORDER
    assert int(counts[0]) == 0
    for key, prev in before.items():
        torch.testing.assert_close(state[key], prev, rtol=0, atol=0)


def test_counter_slots_mirror_the_manifest_order() -> None:
    # @spec PORT-STATE-001
    order = [
        ("CTR_TOTAL_VALID_SAMPLES", "total_valid_samples"),
        ("CTR_COMMITTED_MEL_FRAMES", "committed_mel_frames"),
        ("CTR_ENCODED_MEL_FRAMES", "encoded_mel_frames"),
        ("CTR_RAW_TAIL_ORIGIN", "raw_tail_origin"),
        ("CTR_RAW_TAIL_LENGTH", "raw_tail_length"),
        ("CTR_MEL_TAIL_LENGTH", "mel_tail_length"),
        ("CTR_EXPECTED_CHUNK_SEQUENCE", "expected_chunk_sequence"),
        ("CTR_FINALIZED", "finalized"),
    ]
    for const, name in order:
        assert manifests.FRONTEND_COUNTER_FIELDS[getattr(frontend, const)] == name


def test_batched_rows_with_shared_geometry_match_single_rows() -> None:
    # @spec PORT-FEAT-002
    # Two rows advanced together must equal each row advanced alone
    # (the grouped-STFT batching is behavior-invariant).
    feat = _featurizer()
    torch.manual_seed(17)
    a = torch.randn(17920) * 0.1
    b = torch.randn(17920) * 0.2
    stable = frontend.stable_frames(17920, n_fft=512, hop=160)
    state2 = _fresh_state(batch=2)
    out2, counts2, status2 = frontend.advance_frontend(
        feat,
        torch.stack([a, b]),
        torch.tensor([17920, 17920], dtype=torch.long),
        torch.tensor([False, False]),
        torch.tensor([stable] * 2, dtype=torch.long),
        **state2,
        cadence_frames=stable,
        pad_frames=stable,
    )
    assert status2.tolist() == [0, 0]
    for row, signal in ((0, a), (1, b)):
        streamed, state1 = _stream(feat, signal, [17920], final=False)
        assert int(counts2[row]) == streamed.shape[1]
        torch.testing.assert_close(
            out2[row, :, : int(counts2[row])], streamed, rtol=0, atol=2e-6
        )
        for key in ("raw_tail", "mel_tail", "counters"):
            torch.testing.assert_close(
                state2[key][row], state1[key][0], rtol=0, atol=2e-6
            )


def _row_states(batch: dict[str, torch.Tensor], row: int) -> dict[str, torch.Tensor]:
    return {k: v[row : row + 1].clone() for k, v in batch.items()}


def test_mixed_phase_batch_rows_match_single_rows() -> None:
    # @spec PORT-ADV-004
    # THE length-aware acceptance differential: session-first,
    # continuing, committable-final, pad-bound final, and zero-commit
    # final rows advance in ONE call with per-row commit counts, and
    # every row equals its single-row run — states, counters,
    # committed frames, and zeroed padding. Integer counters and pure
    # gathers (raw tail) compare bit-for-bit; mel values carry the
    # cross-batch-shape FFT ulp bound.
    feat = _featurizer()
    torch.manual_seed(41)
    chunk, la = 17920, 13
    b1 = frontend.cadence_boundary(1, lookahead=la)
    b2 = frontend.cadence_boundary(2, lookahead=la)
    cadence = 8 * (la + 1)
    pad = cadence
    n_rows = 5
    signals = [torch.randn(3 * chunk) * s for s in (0.1, 0.2, 0.15, 0.3, 0.25)]

    # Row phases: 0 = session-first regular (commit b1 = 105);
    # 1 = continuing regular (commit C = 112);
    # 2 = continuing final, partial residual (commit 174 - 105 = 69);
    # 3 = continuing final at the MAXIMUM legal residual — one sample
    #     short of a full cadence unit. There are C+6 available frames
    #     past the boundary; the reference consumes C and drops six.
    # 4 = zero-sample final after TWO units (residual 224 - 217 = 7,
    #     below eight: DROPPED, zero commit).
    pre_units = [0, 1, 1, 1, 2]
    stateN = _fresh_state(batch=n_rows)
    pre: list[dict[str, torch.Tensor]] = []
    for row, signal in enumerate(signals):
        s1 = _fresh_state()
        for k in range(1, pre_units[row] + 1):
            frontend.advance_frontend(
                feat,
                signal[(k - 1) * chunk : k * chunk].unsqueeze(0),
                torch.tensor([chunk], dtype=torch.long),
                torch.tensor([False]),
                torch.tensor(
                    [frontend.cadence_boundary(k, lookahead=la)],
                    dtype=torch.long,
                ),
                **s1,
                cadence_frames=cadence,
                pad_frames=pad,
            )
        pre.append(s1)
        for key in stateN:
            stateN[key][row] = s1[key][0]

    second = [
        signals[0][:chunk],  # first regular
        signals[1][chunk : 2 * chunk],  # continuing regular
        signals[2][chunk : chunk + 10_000],  # final, 69-frame residual
        signals[3][chunk : 2 * chunk - 160],  # max legal final: commit C
        signals[4][2 * chunk : 2 * chunk],  # zero-sample final
    ]
    samples = torch.zeros(n_rows, chunk)
    valid = torch.zeros(n_rows, dtype=torch.long)
    for row, sig in enumerate(second):
        samples[row, : sig.shape[0]] = sig
        valid[row] = sig.shape[0]
    finals = torch.tensor([False, False, True, True, True])
    targets = torch.tensor(
        [b1, b2, 0, 0, 0], dtype=torch.long
    )  # targets ignored on final rows
    outN, countsN, statusN = frontend.advance_frontend(
        feat,
        samples,
        valid,
        finals,
        targets,
        **stateN,
        cadence_frames=cadence,
        pad_frames=pad,
    )
    assert statusN.tolist() == [0] * n_rows

    expected_counts = [
        b1,
        b2 - b1,
        (chunk + 10_000) // 160 - b1,
        cadence,
        0,
    ]
    assert expected_counts[3] == pad
    for row in range(n_rows):
        s1 = pre[row]
        out1, counts1, _ = frontend.advance_frontend(
            feat,
            second[row].unsqueeze(0),
            torch.tensor([second[row].shape[0]], dtype=torch.long),
            finals[row : row + 1],
            targets[row : row + 1],
            **s1,
            cadence_frames=cadence,
            pad_frames=pad,
        )
        assert int(countsN[row]) == int(counts1[0]) == expected_counts[row]
        torch.testing.assert_close(outN[row], out1[0], rtol=0, atol=2e-6)
        torch.testing.assert_close(
            stateN["counters"][row : row + 1],
            s1["counters"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            stateN["raw_tail"][row : row + 1],
            s1["raw_tail"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            stateN["mel_tail"][row : row + 1],
            s1["mel_tail"],
            rtol=0,
            atol=2e-6,
        )
        # Padded columns are exactly zero.
        torch.testing.assert_close(
            outN[row, :, int(countsN[row]) :],
            torch.zeros(N_MELS, pad - int(countsN[row])),
            rtol=0,
            atol=0,
        )
    # Committed prefixes stay exact against the whole-signal oracle.
    whole2 = _whole_mel(feat, signals[2][: chunk + 10_000])
    torch.testing.assert_close(
        outN[2, :, : int(countsN[2])],
        whole2[:, b1 : b1 + int(countsN[2])],
        rtol=0,
        atol=2e-6,
    )
    for row, want in ((0, 0), (1, 0), (2, 1), (3, 1), (4, 1)):
        assert int(stateN["counters"][row, frontend.CTR_FINALIZED]) == want


def test_incoming_status_bit_is_a_masked_no_op() -> None:
    # @spec PORT-ADV-004
    # A row arriving with a caller-set protocol bit (advance_session's
    # sequence/geometry/oversize predicates) mutates nothing and
    # commits nothing, and the bit survives in the returned status;
    # sibling clean rows are unaffected and equal their single-row
    # runs.
    feat = _featurizer()
    torch.manual_seed(42)
    a = torch.randn(17920) * 0.1
    b = torch.randn(17920) * 0.2
    stable = frontend.stable_frames(17920, n_fft=512, hop=160)
    state2 = _fresh_state(batch=2)
    before_row1 = {k: v[1].clone() for k, v in state2.items()}
    incoming = torch.tensor([0, frontend.ROW_STATUS_SEQUENCE], dtype=torch.int32)
    out2, counts2, status2 = frontend.advance_frontend(
        feat,
        torch.stack([a, b]),
        torch.tensor([17920, 17920], dtype=torch.long),
        torch.tensor([False, False]),
        torch.tensor([stable] * 2, dtype=torch.long),
        **state2,
        cadence_frames=stable,
        pad_frames=stable,
        row_status=incoming,
    )
    assert status2.tolist() == [0, frontend.ROW_STATUS_SEQUENCE]
    assert int(counts2[1]) == 0
    torch.testing.assert_close(out2[1], torch.zeros_like(out2[1]), rtol=0, atol=0)
    for key, prev in before_row1.items():
        torch.testing.assert_close(state2[key][1], prev, rtol=0, atol=0)
    streamed, state1 = _stream(feat, a, [17920], final=False)
    assert int(counts2[0]) == streamed.shape[1]
    torch.testing.assert_close(
        out2[0, :, : int(counts2[0])], streamed, rtol=0, atol=2e-6
    )
    for key, bound in (
        ("raw_tail", 0.0),
        ("mel_tail", 2e-6),
        ("counters", 0.0),
    ):
        torch.testing.assert_close(state2[key][0:1], state1[key], rtol=0, atol=bound)


def test_exact_boundary_endpoint_has_one_legal_decomposition() -> None:
    # @spec PORT-ADV-004
    # A stream ending exactly on a cadence boundary: PORT-SESS-001/003
    # force the decomposition [regular CHUNK, zero-sample final] — the
    # residual past B_k is seven frames and DROPS, deterministically.
    # The coalesced full-unit-final form is illegal ingress: the caller
    # flags it (advance_session's oversize predicate) and the frontend
    # masks it to a no-op, so no packetization/finalize coalescing can
    # produce different committed output for the same audio.
    feat = _featurizer()
    torch.manual_seed(43)
    chunk, la = 17920, 13
    b1 = frontend.cadence_boundary(1, lookahead=la)
    b2 = frontend.cadence_boundary(2, lookahead=la)
    cadence = 8 * (la + 1)
    pad = cadence
    signal = torch.randn(2 * chunk) * 0.1

    # Legal decomposition: unit 2 as a regular CHUNK, then the
    # zero-sample final (7-frame residual dropped).
    legal = _fresh_state()
    committed: list[torch.Tensor] = []
    for k in (1, 2):
        out, counts, status = frontend.advance_frontend(
            feat,
            signal[(k - 1) * chunk : k * chunk].unsqueeze(0),
            torch.tensor([chunk], dtype=torch.long),
            torch.tensor([False]),
            torch.tensor(
                [frontend.cadence_boundary(k, lookahead=la)],
                dtype=torch.long,
            ),
            **legal,
            cadence_frames=cadence,
            pad_frames=pad,
        )
        assert status.tolist() == [0]
        committed.append(out[0, :, : int(counts[0])])
    _, counts, status = frontend.advance_frontend(
        feat,
        torch.zeros(1, 0),
        torch.zeros(1, dtype=torch.long),
        torch.tensor([True]),
        torch.zeros(1, dtype=torch.long),
        **legal,
        cadence_frames=cadence,
        pad_frames=pad,
    )
    assert status.tolist() == [0]
    assert int(counts[0]) == 0  # the 224 - 217 = 7 residual drops
    assert int(legal["counters"][0, frontend.CTR_COMMITTED_MEL_FRAMES]) == b2
    assert int(legal["counters"][0, frontend.CTR_FINALIZED]) == 1
    _assert_prefix_equal(feat, signal, torch.cat(committed, dim=1))

    # Illegal coalesced form: a final tail carrying the whole second
    # unit. The caller's oversize predicate marks it; the frontend
    # masks it — state untouched, nothing committed, bit reported.
    coalesced = _fresh_state()
    frontend.advance_frontend(
        feat,
        signal[:chunk].unsqueeze(0),
        torch.tensor([chunk], dtype=torch.long),
        torch.tensor([False]),
        torch.tensor([b1], dtype=torch.long),
        **coalesced,
        cadence_frames=cadence,
        pad_frames=pad,
    )
    before = {k: v.clone() for k, v in coalesced.items()}
    out, counts, status = frontend.advance_frontend(
        feat,
        signal[chunk:].unsqueeze(0),
        torch.tensor([chunk], dtype=torch.long),
        torch.tensor([True]),
        torch.zeros(1, dtype=torch.long),
        **coalesced,
        cadence_frames=cadence,
        pad_frames=pad,
        row_status=torch.tensor(
            [frontend.ROW_STATUS_FINAL_OVERSIZE], dtype=torch.int32
        ),
    )
    assert status.tolist() == [frontend.ROW_STATUS_FINAL_OVERSIZE]
    assert int(counts[0]) == 0
    assert not bool(out.any())
    for key, prev in before.items():
        torch.testing.assert_close(coalesced[key], prev, rtol=0, atol=0)
