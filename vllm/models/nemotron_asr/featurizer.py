# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NeMo-exact mel featurizer (PORT-FEAT-001).

Reproduces NeMo ``FilterbankFeatures`` inference behavior at the pinned
commit (features.py:378-473 @ de242add): masked preemphasis, constant-pad
centered STFT, power-2 magnitude, mel projection, ``log(x + guard)``.
Normalization is "NA" for this checkpoint and dither is train-only, so
neither appears here. The mel filterbank arrives as a tensor from the
offline conversion tool (NeMo computes it with librosa at graph build;
shipping the computed bank removes the runtime dependency and the
reimplementation risk — parity is asserted against a NeMo-produced
fixture).
"""

import math

import torch
from torch import nn


def synthesize_card_featurizer_buffers(
    *,
    n_mels: int,
    n_fft: int = 512,
    win_length: int = 400,
    sample_rate: int = 16000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the featurizer buffers the public HF card does not persist.

    The published checkpoint carries no ``featurizer.fb``/``window``
    tensors; its authoritative computation is the transformers feature
    extractor: ``librosa.filters.mel(sr, n_fft, n_mels, fmin=0,
    fmax=sr/2, norm="slaney")`` (slaney scale, slaney norm) and
    ``torch.hann_window(win_length, periodic=False)``. The filterbank
    below is that same slaney construction in float64, cast to float32
    at the end like the extractor; the cross-check test asserts
    equality against librosa wherever it is installed. The parity note
    on :class:`MelFeaturizer` about persisted-buffer ulps applies: a
    served artifact that ships buffers still wins over synthesis.
    """
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0

    def hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
        mel = hz / f_sp
        log_region = hz >= min_log_hz
        return torch.where(
            log_region,
            min_log_mel + torch.log(hz.clamp_min(min_log_hz) / min_log_hz) / logstep,
            mel,
        )

    def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
        hz = mel * f_sp
        log_region = mel >= min_log_mel
        return torch.where(
            log_region,
            min_log_hz * torch.exp(logstep * (mel - min_log_mel)),
            hz,
        )

    n_freqs = n_fft // 2 + 1
    fft_hz = torch.linspace(0.0, sample_rate / 2.0, n_freqs, dtype=torch.float64)
    mel_edges = torch.linspace(
        hz_to_mel(torch.tensor(0.0, dtype=torch.float64)),
        hz_to_mel(torch.tensor(sample_rate / 2.0, dtype=torch.float64)),
        n_mels + 2,
        dtype=torch.float64,
    )
    hz_edges = mel_to_hz(mel_edges)
    fdiff = hz_edges[1:] - hz_edges[:-1]
    ramps = hz_edges.unsqueeze(1) - fft_hz.unsqueeze(0)
    lower = -ramps[:-2] / fdiff[:-1].unsqueeze(1)
    upper = ramps[2:] / fdiff[1:].unsqueeze(1)
    weights = torch.maximum(
        torch.zeros(1, dtype=torch.float64),
        torch.minimum(lower, upper),
    )
    enorm = 2.0 / (hz_edges[2 : n_mels + 2] - hz_edges[:n_mels])
    fb = (weights * enorm.unsqueeze(1)).to(torch.float32)
    window = torch.hann_window(win_length, periodic=False).to(torch.float32)
    return fb, window


class MelFeaturizer(nn.Module):
    """Log-mel features over a whole buffer, one call per utterance/chunk.

    Streaming continuity (raw-sample tail, pre-encode overlap,
    ``drop_extra_pre_encoded``) is the serving layer's job
    (PORT-FEAT-002); this module is stateless.
    """

    def __init__(
        self,
        *,
        filterbank: torch.Tensor,
        window: torch.Tensor,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        preemph: float = 0.97,
        log_zero_guard: float = 2.0**-24,
    ) -> None:
        super().__init__()
        if filterbank.dim() != 2 or filterbank.shape[1] != n_fft // 2 + 1:
            raise ValueError(
                f"filterbank must be (n_mels, {n_fft // 2 + 1}), got "
                f"{tuple(filterbank.shape)}"
            )
        if window.shape != (win_length,):
            raise ValueError(
                f"window must be ({win_length},), got {tuple(window.shape)}"
            )
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.preemph = preemph
        self.log_zero_guard = log_zero_guard
        self.register_buffer("fb", filterbank.to(torch.float32))
        # The checkpoint's persisted window, like the filterbank — a
        # freshly built hann differs by one ulp (trained under a
        # different torch rounding), which log() amplifies to ~1e-4 at
        # quiet mel bins. Checkpoint buffers are inputs, not recomputed.
        self.register_buffer("window", window.to(torch.float32))

    def output_lengths(self, sample_lengths: torch.Tensor) -> torch.Tensor:
        """Mel frame count per NeMo ``get_seq_len`` (center=True)."""
        pad = 2 * (self.n_fft // 2)
        return torch.div(
            sample_lengths + pad - self.n_fft,
            self.hop_length,
            rounding_mode="floor",
        ).to(torch.long)

    def forward(
        self, waveform: torch.Tensor, sample_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log-mel features.

        Args:
            waveform: ``(batch, samples)`` fp32 PCM in [-1, 1].
            sample_lengths: valid samples per batch element.

        Returns:
            ``(batch, n_mels, frames)`` log-mel and per-element frame
            counts.

        """
        mel_len = self.output_lengths(sample_lengths)

        # Masked preemphasis: first sample kept, x[t] - p*x[t-1], zeros
        # past each element's valid length (features.py:431-434).
        time_mask = torch.arange(waveform.shape[1], device=waveform.device).unsqueeze(
            0
        ) < sample_lengths.unsqueeze(1)
        x = torch.cat(
            (
                waveform[:, :1],
                waveform[:, 1:] - self.preemph * waveform[:, :-1],
            ),
            dim=1,
        )
        x = x.masked_fill(~time_mask, 0.0)

        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            window=self.window,
            return_complex=True,
            pad_mode="constant",
        )
        # NeMo computes magnitude (sqrt) then squares it back to power
        # (features.py:443-453). Algebraically an identity, numerically
        # an fp32 round-trip that log() then amplifies at low-energy
        # bins — parity requires the same op order.
        magnitude = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1))
        power = magnitude.pow(2.0)
        mel = torch.matmul(self.fb, power)
        return torch.log(mel + self.log_zero_guard), mel_len
