# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mel featurizer: NeMo-exact (PORT-FEAT-001).

Fixture tests assert tensor parity against NeMo's own
``FilterbankFeatures`` (restored from the shipped checkpoint's
preprocessor config) on fixed synthetic waveforms; the fixture also
carries the mel filterbank, which production code receives from the
offline conversion tool rather than recomputing (no librosa at
runtime, no reimplementation risk). Generated in venv-oracle by
``generate_featurizer_fixtures.py``; a missing fixture FAILS.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from vllm.models.nemotron_asr.featurizer import (
    MelFeaturizer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "nemo_mel.npz"


def _fixture():
    assert FIXTURE_PATH.is_file(), (
        f"featurizer fixture missing: {FIXTURE_PATH} — generate in venv-oracle via "
        "generate_featurizer_fixtures.py"
    )
    return np.load(FIXTURE_PATH)


def _featurizer(fixture) -> MelFeaturizer:
    return MelFeaturizer(
        filterbank=torch.from_numpy(fixture["filterbank"]),
        window=torch.from_numpy(fixture["window"]),
        n_fft=int(fixture["n_fft"]),
        win_length=int(fixture["win_length"]),
        hop_length=int(fixture["hop_length"]),
        preemph=float(fixture["preemph"]),
        log_zero_guard=float(fixture["log_zero_guard"]),
    )


def test_mel_matches_nemo_on_synthetic_waveform():
    fixture = _fixture()
    featurizer = _featurizer(fixture)
    waveform = torch.from_numpy(fixture["waveform"])  # (1, S) fp32
    seq_len = torch.tensor([waveform.shape[1]])
    mel, mel_len = featurizer(waveform, seq_len)
    expected = torch.from_numpy(fixture["mel"])
    expected_len = torch.from_numpy(fixture["mel_len"])
    assert mel_len.tolist() == expected_len.tolist()
    torch.testing.assert_close(
        mel[:, :, : int(mel_len[0])],
        expected[:, :, : int(expected_len[0])],
        atol=1e-5,
        # FFT plans can differ by a few FP32 ulps before log scaling. The
        # pinned Python-3.10 / torch-2.11 CPU referee differs in two of 53,120
        # values (maximum relative delta 1.49e-5); this remains far tighter
        # than the model-level parity gate and does not alter production math.
        rtol=2e-5,
    )


def test_mel_matches_nemo_on_silence():
    # Silence exercises the log-zero guard exactly.
    fixture = _fixture()
    featurizer = _featurizer(fixture)
    waveform = torch.from_numpy(fixture["silence_waveform"])
    seq_len = torch.tensor([waveform.shape[1]])
    mel, mel_len = featurizer(waveform, seq_len)
    expected = torch.from_numpy(fixture["silence_mel"])
    torch.testing.assert_close(
        mel[:, :, : int(mel_len[0])],
        expected[:, :, : int(mel_len[0])],
        atol=1e-5,
        rtol=1e-5,
    )


def test_length_formula_matches_nemo():
    # NeMo get_seq_len (features.py:405-409, center=True):
    # frames = (samples + 2*(n_fft//2) - n_fft) // hop = samples // hop.
    fixture = _fixture()
    featurizer = _featurizer(fixture)
    for samples in (400, 401, 8000, 16000, 66432):
        got = featurizer.output_lengths(torch.tensor([samples]))
        assert int(got[0]) == samples // int(fixture["hop_length"])
