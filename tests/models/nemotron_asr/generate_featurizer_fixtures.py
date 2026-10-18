# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate NeMo mel-featurizer fixtures (venv-oracle ONLY).

Restores the shipped checkpoint so the preprocessor carries the
checkpoint's own parameters (config.json/model config authoritative —
no hand-copied values), runs it in eval mode on fixed synthetic
waveforms, and saves waveform/mel pairs plus the resolved parameters
and the librosa-built mel filterbank to ``fixtures/nemo_mel.npz``.

Usage (dev pod):
    /opt/venv-oracle/bin/python generate_featurizer_fixtures.py \
        --model /path/to/nemotron-3.5-asr-streaming-0.6b.nemo
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from nemo.collections.asr.models import ASRModel

SAMPLE_RATE = 16000
SECONDS = 4.152  # matches the golden clip's duration class


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    model = ASRModel.restore_from(
        restore_path=str(args.model), map_location=torch.device("cpu")
    )
    model.eval()
    pre = model.preprocessor.featurizer

    torch.manual_seed(20260711)
    samples = int(SAMPLE_RATE * SECONDS)
    waveform = (0.1 * torch.randn(1, samples)).clamp(-1.0, 1.0)
    silence = torch.zeros(1, SAMPLE_RATE)

    with torch.no_grad():
        mel, mel_len = pre(waveform, torch.tensor([samples]))
        s_mel, s_len = pre(silence, torch.tensor([SAMPLE_RATE]))

    out = Path(__file__).parent / "fixtures" / "nemo_mel.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        waveform=waveform.numpy(),
        mel=mel.numpy(),
        mel_len=mel_len.numpy(),
        silence_waveform=silence.numpy(),
        silence_mel=s_mel.numpy(),
        filterbank=pre.fb[0].numpy(),
        window=pre.window.numpy(),
        n_fft=np.int64(pre.n_fft),
        win_length=np.int64(pre.win_length),
        hop_length=np.int64(pre.hop_length),
        preemph=np.float64(pre.preemph),
        log_zero_guard=np.float64(pre.log_zero_guard_value),
    )
    print(
        f"wrote {out} (n_mels={pre.nfilt}, n_fft={pre.n_fft}, "
        f"win={pre.win_length}, hop={pre.hop_length}, "
        f"normalize={pre.normalize!r}, dither(train-only)={pre.dither})"
    )


if __name__ == "__main__":
    main()
