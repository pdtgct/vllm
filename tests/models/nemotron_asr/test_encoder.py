# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FastConformer encoder structure tests (P2 full-context regime).

Numerical parity against the golden sixth cell lands with weight
conversion on-pod; these GPU-free tests pin the structural contracts:
shape/length arithmetic, strict causality at zero lookahead (through
subsampling, attention, AND the causal conv), intra-chunk lookahead at
nonzero right context, and padding invariance.
"""

import pytest
import torch

from vllm.models.nemotron_asr.encoder import (
    FastConformerEncoder,
    RelPositionalEncoding,
)
from vllm.models.nemotron_asr.frontend import (
    MEL_TAIL_FRAMES,
)
from vllm.models.nemotron_asr.manifests import (
    CADENCES,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]


def _tiny(att_context=(8, 0)) -> FastConformerEncoder:
    torch.manual_seed(31)
    enc = FastConformerEncoder(
        feat_in=16,
        d_model=32,
        d_ff=64,
        n_layers=2,
        n_heads=4,
        conv_kernel=5,
        subsampling_channels=16,
        att_context=att_context,
    )
    enc.eval()
    return enc


def _mel(frames: int, feat: int = 16, seed: int = 7) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(1, feat, frames)


def test_output_shapes_and_lengths():
    enc = _tiny()
    mel = _mel(160)
    out, lens = enc(mel, torch.tensor([160]))
    assert out.shape[0] == 1 and out.shape[2] == 32
    assert (
        out.shape[1]
        == int(lens[0])
        == enc.pre_encode.output_lengths(torch.tensor([160]))
    )


def test_materialized_positional_capacity_serves_every_geometry_bit_exactly() -> None:
    # @spec PORT-PERF-010
    enc = _tiny(att_context=(56, 13))
    history = enc.att_context[0]
    attention_lengths = []
    for _, right in CADENCES.values():
        mel_width = MEL_TAIL_FRAMES + 8 * (right + 1)
        out_width = int(enc.pre_encode.output_lengths(torch.tensor([mel_width]))[0])
        attention_lengths.append(out_width + history)
    t_cap = max(attention_lengths)
    capacity_ref = torch.zeros(1, t_cap, 32)
    enc.pos_enc._extend(t_cap, capacity_ref)
    positional_id = id(enc.pos_enc.pe)

    for attention_length in attention_lengths:
        ref = torch.zeros(1, attention_length, 32)
        actual = enc.pos_enc(ref)
        exact = RelPositionalEncoding(32)
        exact._extend(attention_length, ref)
        expected = exact(ref)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert id(enc.pos_enc.pe) == positional_id


def test_strictly_causal_at_zero_lookahead():
    # att_context (8, 0): chunk = 1 frame; with causal subsampling and
    # the causal conv, encoder frame t must be invariant to any change
    # in audio strictly after its receptive window.
    enc = _tiny(att_context=(8, 0))
    mel_a = _mel(160)
    mel_b = mel_a.clone()
    mel_b[:, :, 120:] += 3.0  # perturb the tail only
    with torch.no_grad():
        out_a, _ = enc(mel_a, torch.tensor([160]))
        out_b, _ = enc(mel_b, torch.tensor([160]))
    # 120 mel frames -> at least 120//8 - 1 clean encoder frames; use a
    # conservative margin for subsampling edge effects.
    safe = 120 // 8 - 2
    torch.testing.assert_close(out_a[:, :safe], out_b[:, :safe])


def test_lookahead_is_intra_chunk_only():
    # att_context (8, 3): chunk = 4 encoder frames. Frames in chunk 0
    # may see chunk 0 only; perturbing chunk 2's audio must not change
    # chunk 0's output, but should change chunk 1's neighbors' -- no:
    # chunk 1 cannot see chunk 2 either (lookahead is INTRA-chunk).
    enc = _tiny(att_context=(8, 3))
    mel_a = _mel(160)
    mel_b = mel_a.clone()
    mel_b[:, :, 96:] += 3.0  # perturb from encoder frame 12 == chunk 3
    with torch.no_grad():
        out_a, _ = enc(mel_a, torch.tensor([160]))
        out_b, _ = enc(mel_b, torch.tensor([160]))
    # chunks 0-1 (frames 0..7) sit >1 full chunk before the
    # perturbation; margin for subsampling receptive field.
    torch.testing.assert_close(out_a[:, :6], out_b[:, :6])


def test_padding_frames_do_not_change_valid_output():
    enc = _tiny()
    mel = _mel(160)
    padded = torch.cat([mel, torch.zeros(1, 16, 40)], dim=2)
    with torch.no_grad():
        out_a, lens_a = enc(mel, torch.tensor([160]))
        out_b, lens_b = enc(padded, torch.tensor([160]))
    n = int(lens_a[0])
    assert int(lens_b[0]) == n
    torch.testing.assert_close(out_a[:, :n], out_b[:, :n])


def test_weight_names_follow_nemo_layout():
    enc = _tiny()
    names = {name for name, _ in enc.named_parameters()}
    for expected in (
        "pre_encode.conv.0.weight",
        "pre_encode.out.weight",
        "layers.0.self_attn.linear_q.weight",
        "layers.0.self_attn.pos_bias_u",
        "layers.0.conv.depthwise_conv.weight",
        "layers.1.norm_out.weight",
        "layers.0.feed_forward1.linear1.weight",
    ):
        assert expected in names, expected
    # Bias-free per the checkpoint config (attention_bias /
    # convolution_bias false).
    assert "layers.0.self_attn.linear_q.bias" not in names
    assert "layers.0.conv.pointwise_conv1.bias" not in names


# ---- length-aware streaming step (PORT-ADV-004 / PORT-PERF-001) ----

from vllm.models.nemotron_asr.encoder import (  # noqa: E402
    StreamingCaches,
    _stream_attention,
    stream_step,
)


def _caches(batch: int, enc: FastConformerEncoder) -> StreamingCaches:
    return StreamingCaches(
        n_layers=len(enc.layers),
        batch=batch,
        d_model=32,
        left_context=enc.att_context[0],
        conv_kernel=5,
        device=torch.device("cpu"),
    )


def _clone_caches(c: StreamingCaches) -> tuple:
    return (c.channel.clone(), c.time.clone(), c.valid.clone())


def test_stream_step_mixed_lengths_match_single_rows() -> None:
    # @spec PORT-ADV-004
    # THE length-aware encoder differential: rows with different valid
    # mel widths and different pre-encode drops advance in ONE padded
    # call; each row's valid output, advanced window, conv tail, and
    # window_valid must match its own exact-width single-row run.
    enc = _tiny(att_context=(8, 1))
    torch.manual_seed(51)
    n_mels, width = 16, 41
    mel = torch.randn(3, n_mels, width) * 0.5
    # Per-row valid mel widths and drops: row 0 full-width continuing
    # (drop 2), row 1 shorter continuing (drop 2), row 2 session-first
    # (drop 0, width 24).
    mel_lens = torch.tensor([41, 33, 24])
    offsets = torch.tensor([2, 2, 0])
    for b, n in enumerate(mel_lens.tolist()):
        # Padded mel columns must be ZERO (the stream_step input
        # contract): the subsampling's last valid output legitimately
        # covers input position n — the exact computation's causal
        # right-pad, which zero padding reproduces. advance_session
        # zeroes its gathered mel grid for exactly this reason.
        mel[b, :, n:] = 0.0
    out_lens = enc.pre_encode.output_lengths(mel_lens) - offsets
    out_width = int(enc.pre_encode.output_lengths(torch.tensor([width]))[0])

    caches3 = _caches(3, enc)
    # Distinct carried windows/tails per row (mid-session resumes).
    torch.manual_seed(52)
    caches3.channel.normal_(0.0, 0.2)
    caches3.time.normal_(0.0, 0.2)
    caches3.valid = torch.tensor([8, 3, 0])
    singles = []
    for b in range(3):
        c1 = _caches(1, enc)
        c1.channel.copy_(caches3.channel[:, b : b + 1])
        c1.time.copy_(caches3.time[:, b : b + 1])
        c1.valid = caches3.valid[b : b + 1].clone()
        singles.append(c1)

    with torch.no_grad():
        out3 = stream_step(
            enc,
            mel,
            caches3,
            out_offsets=offsets,
            out_lengths=out_lens,
            out_width=out_width,
        )
    assert out3.shape[1] == out_width
    for b in range(3):
        c1 = singles[b]
        with torch.no_grad():
            out1 = stream_step(
                enc,
                mel[b : b + 1, :, : int(mel_lens[b])],
                c1,
                out_offsets=offsets[b : b + 1],
                out_lengths=out_lens[b : b + 1],
                out_width=int(out_lens[b]),
            )
        f = int(out_lens[b])
        torch.testing.assert_close(out3[b, :f], out1[0, :f], rtol=0, atol=1e-5)
        # Padded output frames are exactly zero.
        torch.testing.assert_close(
            out3[b, f:],
            torch.zeros(out_width - f, 32),
            rtol=0,
            atol=0,
        )
        # Caches advanced by the row's LOGICAL length only.
        torch.testing.assert_close(
            caches3.channel[:, b], c1.channel[:, 0], rtol=0, atol=1e-5
        )
        torch.testing.assert_close(caches3.time[:, b], c1.time[:, 0], rtol=0, atol=1e-5)
        assert int(caches3.valid[b]) == int(c1.valid[0])


def test_stream_step_zero_length_row_is_a_masked_no_op() -> None:
    # @spec PORT-ADV-004
    # A zero-valid row beside an active one: output all-zero, window,
    # conv tail, and window_valid BIT-identical (no advancement).
    enc = _tiny(att_context=(8, 1))
    torch.manual_seed(53)
    mel = torch.randn(2, 16, 25) * 0.5
    caches = _caches(2, enc)
    caches.channel.normal_(0.0, 0.2)
    caches.time.normal_(0.0, 0.2)
    caches.valid = torch.tensor([5, 5])
    before = _clone_caches(caches)
    out_width = int(enc.pre_encode.output_lengths(torch.tensor([25]))[0]) - 2
    with torch.no_grad():
        out = stream_step(
            enc,
            mel,
            caches,
            out_offsets=torch.tensor([2, 2]),
            out_lengths=torch.tensor([out_width, 0]),
            out_width=out_width,
        )
    torch.testing.assert_close(out[1], torch.zeros_like(out[1]), rtol=0, atol=0)
    torch.testing.assert_close(caches.channel[:, 1], before[0][:, 1], rtol=0, atol=0)
    torch.testing.assert_close(caches.time[:, 1], before[1][:, 1], rtol=0, atol=0)
    assert int(caches.valid[1]) == int(before[2][1])
    # The active row DID advance.
    assert int(caches.valid[0]) == min(5 + out_width, 8)


def test_stream_step_window_valid_saturates_per_row() -> None:
    # @spec PORT-ADV-004
    # window_valid accumulates each row's logical length and clamps at
    # the window capacity independently per row.
    enc = _tiny(att_context=(8, 1))
    torch.manual_seed(54)
    mel = torch.randn(2, 16, 41) * 0.5
    caches = _caches(2, enc)
    caches.valid = torch.tensor([7, 2])
    out_width = int(enc.pre_encode.output_lengths(torch.tensor([41]))[0])
    lens = torch.tensor([out_width, 3])
    with torch.no_grad():
        stream_step(
            enc,
            mel,
            caches,
            out_offsets=torch.tensor([0, 0]),
            out_lengths=lens,
            out_width=out_width,
        )
    assert caches.valid.tolist() == [8, 5]


def test_streaming_attention_evicts_oldest_frames_after_many_chunks() -> None:
    # @spec PORT-STATE-006 / PORT-STATE-020 / PORT-STATE-021
    enc = _tiny(att_context=(8, 1))
    layer = enc.layers[0]
    cache = torch.zeros(1, 8, 32)
    valid = torch.zeros(1, dtype=torch.long)
    expected = torch.empty(1, 0, 32)

    # One hundred two-frame chunks are twenty-five full context turnovers.
    # The resident cache must stay fixed-width and retain exactly the newest
    # eight frames.
    with torch.no_grad():
        for sequence in range(100):
            x = torch.full((1, 2, 32), float(sequence + 1))
            new_valid = torch.ones(1, 2, dtype=torch.bool)
            pos_emb = enc.pos_enc(torch.zeros(1, cache.shape[1] + 2, 32))
            _, cache = _stream_attention(
                layer,
                x,
                cache=cache,
                valid=valid,
                pos_emb=pos_emb,
                new_valid=new_valid,
                new_lengths=torch.tensor([2]),
            )
            expected = torch.cat([expected, x], dim=1)[:, -8:]
            valid = torch.clamp(valid + 2, max=8)

            assert cache.shape == (1, 8, 32)
            torch.testing.assert_close(
                cache[:, -int(valid.item()) :],
                expected,
                rtol=0,
                atol=0,
            )


def test_streaming_attention_eviction_is_row_local_for_mixed_lengths() -> None:
    # @spec PORT-STATE-006 / PORT-STATE-021
    enc = _tiny(att_context=(8, 1))
    layer = enc.layers[0]
    cache = torch.stack(
        (
            torch.arange(8, dtype=torch.float32).view(8, 1).expand(8, 32),
            (100 + torch.arange(8, dtype=torch.float32)).view(8, 1).expand(8, 32),
        )
    )
    valid = torch.tensor([8, 8])
    x = torch.stack(
        (
            torch.tensor([[8.0], [9.0]]).expand(2, 32),
            torch.tensor([[108.0], [999.0]]).expand(2, 32),
        )
    )
    new_lengths = torch.tensor([2, 1])
    new_valid = torch.tensor([[True, True], [True, False]])
    pos_emb = enc.pos_enc(torch.zeros(1, 10, 32))

    with torch.no_grad():
        _, advanced = _stream_attention(
            layer,
            x,
            cache=cache,
            valid=valid,
            pos_emb=pos_emb,
            new_valid=new_valid,
            new_lengths=new_lengths,
        )

    assert advanced.shape == (2, 8, 32)
    torch.testing.assert_close(
        advanced[0, :, 0], torch.arange(2, 10, dtype=torch.float32)
    )
    torch.testing.assert_close(
        advanced[1, :, 0], torch.arange(101, 109, dtype=torch.float32)
    )
    assert 999.0 not in advanced[1, :, 0].tolist()
