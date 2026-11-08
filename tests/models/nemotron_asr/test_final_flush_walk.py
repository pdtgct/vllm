# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Final-flush parity against the cache-aware streaming buffer walk.

The oracle is an independent whole-utterance simulation of NeMo's public
cache-aware streaming walk (``CacheAwareStreamingAudioBuffer.__iter__`` in
``nemo/collections/asr/parts/utils/streaming_utils.py`` with the chunk
geometry from ``ConformerEncoder.setup_streaming_params``): featurize the
whole signal once, take a first chunk of ``8L+1`` mel frames (at least one
frame), then ``8(L+1)``-frame chunks behind a nine-frame pre-encode cache
whose two leading encoder outputs are dropped, and stop when a non-first
chunk holds fewer than eight frames. Every chunk is encoded at its exact
width, with the same tiny random-weight encoder and the same RNN-T decode.

The engine side drives ``advance_session`` one cadence unit at a time, the
final residual included, exactly as the serving segmenter packages it. At a
fixed lookahead the two must consume the same mel frames, produce the same
encoder frames, and emit the same labels — for audio shorter than one
cadence and for lengths off the chunk grid.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]

_BASE = "vllm.models.nemotron_asr"
mods = {
    name: importlib.import_module(f"{_BASE}.{name}")
    for name in (
        "precision",
        "masks",
        "featurizer",
        "encoder",
        "encoder_execution",
        "lid",
        "manifests",
        "frontend",
        "profiling",
        "rnnt_cell",
        "rnnt",
        "state_scatter",
        "advance",
    )
}
advance = mods["advance"]
frontend = mods["frontend"]
rnnt = mods["rnnt"]
encoder_mod = mods["encoder"]
LOOKAHEADS = [right for _, right in mods["manifests"].CADENCES.values()]

FEAT = 16
D_MODEL = 32
N_LAYERS = 2
KERNEL = 5
WINDOW = 8
VOCAB = 12
PRED = 16
HOP = 160
RAW_TAIL = 1_953


def _core(seed: int) -> Any:
    torch.manual_seed(seed)
    encoder = encoder_mod.FastConformerEncoder(
        feat_in=FEAT,
        d_model=D_MODEL,
        d_ff=64,
        n_layers=N_LAYERS,
        n_heads=4,
        conv_kernel=KERNEL,
        subsampling_channels=16,
        att_context=(WINDOW, 1),
    )
    encoder.eval()
    # A filterbank large enough that the log-mel is signal-dominated
    # rather than log-guard-dominated, so labels discriminate.
    featurizer = mods["featurizer"].MelFeaturizer(
        filterbank=torch.rand(FEAT, 257),
        window=torch.hann_window(400),
    )
    return SimpleNamespace(
        encoder=encoder,
        lid=mods["lid"].PromptConditioner(enc_hidden=D_MODEL, num_prompts=4),
        predictor=rnnt.Predictor(vocab_size=VOCAB, pred_hidden=PRED, pred_rnn_layers=2),
        joint=rnnt.Joint(
            enc_hidden=D_MODEL, pred_hidden=PRED, joint_hidden=16, vocab_size=VOCAB
        ),
        featurizer=featurizer,
        blank_id=VOCAB,
    )


def _fresh_state() -> Any:
    return advance.SessionStateBatch(
        raw_tail=torch.zeros(1, RAW_TAIL),
        mel_tail=torch.zeros(1, FEAT, frontend.MEL_TAIL_FRAMES),
        frontend_counters=torch.zeros(1, 8, dtype=torch.int64),
        channel=[torch.zeros(1, WINDOW, D_MODEL) for _ in range(N_LAYERS)],
        window_valid=[torch.zeros(1, 1, dtype=torch.int32) for _ in range(N_LAYERS)],
        time=[torch.zeros(1, D_MODEL, KERNEL - 1) for _ in range(N_LAYERS)],
        h=torch.zeros(1, 2, PRED),
        c=torch.zeros(1, 2, PRED),
        last_label=torch.full((1,), VOCAB, dtype=torch.long),
    )


def _walk_oracle(
    core: Any, signal: torch.Tensor, lookahead: int
) -> tuple[int, list[torch.Tensor], list[int]]:
    """Consumed mel frames, per-step encoder outputs, and labels."""
    cadence = 8 * (lookahead + 1)
    with torch.no_grad():
        mel, mel_len = core.featurizer(
            signal.unsqueeze(0), torch.tensor([signal.shape[0]])
        )
        total = int(mel_len[0])
        mel = mel[:, :, :total]
        caches = encoder_mod.StreamingCaches(
            n_layers=N_LAYERS,
            batch=1,
            d_model=D_MODEL,
            left_context=WINDOW,
            conv_kernel=KERNEL,
            device=torch.device("cpu"),
        )
        state = rnnt.DecodeState(
            h=torch.zeros(2, 1, PRED),
            c=torch.zeros(2, 1, PRED),
            last_label=torch.full((1,), core.blank_id),
        )
        consumed = 0
        encoded: list[torch.Tensor] = []
        labels: list[int] = []
        while consumed < total:
            first = consumed == 0
            chunk = mel[
                :, :, consumed : consumed + (8 * lookahead + 1 if first else cadence)
            ]
            if chunk.shape[2] < (1 if first else 8):
                break
            if first:
                step_input, drop = chunk, 0
            else:
                cache = mel[
                    :, :, max(0, consumed - frontend.MEL_TAIL_FRAMES) : consumed
                ]
                missing = frontend.MEL_TAIL_FRAMES - cache.shape[2]
                cache = torch.cat([cache.new_zeros(1, FEAT, missing), cache], dim=2)
                step_input, drop = (
                    torch.cat([cache, chunk], dim=2),
                    advance.PRE_ENCODE_DROP,
                )
            enc = encoder_mod.stream_step(
                core.encoder, step_input, caches, drop_extra=drop
            )
            encoded.append(enc[0])
            emitted, state = rnnt.greedy_decode_batch(
                core.lid(enc, prompt_index=0), core.predictor, core.joint, state
            )
            labels.extend(emitted[0])
            consumed += chunk.shape[2]
    return consumed, encoded, labels


def _engine(
    core: Any, signal: torch.Tensor, geometry: int
) -> tuple[int, list[torch.Tensor], list[int]]:
    """Drive ``advance_session`` over the segmenter's cadence units."""
    lookahead = LOOKAHEADS[geometry]
    unit = 8 * (lookahead + 1) * HOP
    pieces = [
        signal[start : start + unit]
        for start in range(0, signal.shape[0] - unit + 1, unit)
    ]
    residual = signal[len(pieces) * unit :]
    state = _fresh_state()
    encoded: list[torch.Tensor] = []
    labels: list[int] = []
    for seq, (samples, final) in enumerate(
        [*[(p, False) for p in pieces], (residual, True)]
    ):
        padded = torch.zeros(1, unit)
        padded[0, : samples.shape[0]] = samples
        batch = advance.ChunkBatch(
            samples=padded,
            valid_samples=torch.tensor([samples.shape[0]], dtype=torch.long),
            geometry_id=torch.tensor([geometry], dtype=torch.long),
            final_tail=torch.tensor([final]),
            prompt_index=torch.tensor([0], dtype=torch.long),
            chunk_sequence=torch.tensor([seq], dtype=torch.long),
        )
        with torch.no_grad():
            result = advance.advance_session(
                core,
                batch,
                state,
                geometry=geometry,
                decode_fn=rnnt.decode_compact_active,
                capture=True,
            )
        assert result.row_status.tolist() == [0]
        frames = int(result.captures.encoder_lengths[0])
        if frames:
            encoded.append(result.captures.encoder_raw[0, :frames])
        labels.extend(result.token_ids[0, : int(result.token_lengths[0])].tolist())
    assert int(state.frontend_counters[0, frontend.CTR_FINALIZED]) == 1
    return (
        int(state.frontend_counters[0, frontend.CTR_COMMITTED_MEL_FRAMES]),
        encoded,
        labels,
    )


def _assert_walk_parity(geometry: int, n_samples: int, seed: int) -> None:
    core = _core(seed)
    torch.manual_seed(1000 + n_samples)
    signal = torch.randn(n_samples) * 0.3
    oracle_frames, oracle_encoded, oracle_labels = _walk_oracle(
        core, signal, LOOKAHEADS[geometry]
    )
    engine_frames, engine_encoded, engine_labels = _engine(core, signal, geometry)
    assert engine_frames == oracle_frames
    assert len(engine_encoded) == len(oracle_encoded)
    for got, want in zip(engine_encoded, oracle_encoded, strict=True):
        assert got.shape == want.shape
        torch.testing.assert_close(got, want, rtol=0, atol=1e-5)
    assert engine_labels == oracle_labels


# (geometry, samples): audio shorter than one cadence unit — a
# session-first final tail — spanning one mel frame, under eight frames,
# exactly the first-chunk boundary 8L+1, and past it (the residual beyond
# the first chunk is under eight frames and the walk drops it).
_SESSION_FIRST_FINALS = [
    (0, 200),  # 80 ms: one frame
    (0, 1_100),  # 80 ms: six frames, first chunk takes one
    (1, 300),  # 160 ms: one frame
    (1, 1_300),  # 160 ms: eight frames
    (1, 1_500),  # 160 ms: nine frames == 8L+1
    (1, 2_000),  # 160 ms: twelve frames, first chunk takes nine
    (1, 2_500),  # 160 ms: fifteen frames, first chunk takes nine
    (2, 4_500),  # 320 ms: 28 frames, first chunk takes 25
    (4, 300),  # 1120 ms: one frame
    (4, 2_800),  # 1120 ms: seventeen frames
    (4, 17_000),  # 1120 ms: 106 frames, first chunk takes 105
    (4, 17_900),  # 1120 ms: 111 frames, first chunk takes 105
]


# @spec PORT-FEAT-004, PORT-SESS-003
@pytest.mark.parametrize("seed", [1, 3])
@pytest.mark.parametrize(("geometry", "n_samples"), _SESSION_FIRST_FINALS)
def test_session_first_final_matches_streaming_walk(
    geometry: int, n_samples: int, seed: int
) -> None:
    _assert_walk_parity(geometry, n_samples, seed)


# Multi-chunk sessions off the chunk grid, including the ~1.25 s length of
# a short utterance at every lookahead whose final chunk is full width or
# empty. Partial final chunks whose padded width reaches the subsampling
# right pad are covered separately below.
_MULTI_CHUNK = [
    (0, 20_000),
    (0, 19_300),
    (1, 20_000),
    (1, 19_300),
    (2, 20_000),
    (3, 19_300),
    (4, 19_250),
]


# @spec PORT-FEAT-004, PORT-SESS-003
@pytest.mark.parametrize("seed", [1, 3])
@pytest.mark.parametrize(("geometry", "n_samples"), _MULTI_CHUNK)
def test_multi_chunk_final_flush_matches_streaming_walk(
    geometry: int, n_samples: int, seed: int
) -> None:
    _assert_walk_parity(geometry, n_samples, seed)


# Partial final chunks whose valid mel width is not 0 or 1 mod 8. The
# causal subsampling convs pad one step on the right, so an intermediate
# layer whose valid length is even reads one position past it. The walk
# encodes exactly the valid width (and NeMo's masked subsampling zeroes
# padded steps before every conv), so the engine must mask the same steps
# of its fixed-width input; otherwise the conv bias of the zero-padded mel
# tail reaches the last valid frame and intra-chunk attention spreads it
# to every frame of the final chunk.
_PADDED_PARTIAL_FINALS = [
    (1, 1_000),  # session-first, six frames
    (2, 2_000),  # session-first, twelve frames
    (2, 17_000),  # continuing final, 17 frames
    (3, 20_000),  # continuing final, 20 frames
    (4, 20_000),  # continuing final, 20 frames
    (4, 22_000),  # continuing final, 32 frames
]


# @spec PORT-FEAT-004, PORT-ADV-004
@pytest.mark.parametrize("seed", [1, 3])
@pytest.mark.parametrize(("geometry", "n_samples"), _PADDED_PARTIAL_FINALS)
def test_padded_partial_final_matches_streaming_walk(
    geometry: int, n_samples: int, seed: int
) -> None:
    _assert_walk_parity(geometry, n_samples, seed)


def _unmasked_stream_step(
    encoder: Any, mel: torch.Tensor, caches: Any, **kwargs: Any
) -> torch.Tensor:
    """The pre-mask computation: the subsampling sees the full input width."""
    return encoder_mod.stream_step(encoder, mel, caches, **kwargs)


def _frozen_subsampling(pre_encode: Any, mel: torch.Tensor) -> torch.Tensor:
    """The subsampling exactly as it ran before the length mask existed."""
    x = mel.to(next(pre_encode.parameters()).dtype).transpose(1, 2).unsqueeze(1)
    x = pre_encode.conv(x)
    b, c, t, f = x.size()
    return pre_encode.out(x.transpose(1, 2).reshape(b, t, c * f))


# @spec PORT-ADV-004
@pytest.mark.parametrize("geometry", range(len(LOOKAHEADS)))
def test_masked_subsampling_keeps_every_valid_frame_of_full_width_rows(
    geometry: int,
) -> None:
    core = _core(seed=5)
    pre_encode = core.encoder.pre_encode
    cadence = 8 * (LOOKAHEADS[geometry] + 1)
    width = frontend.MEL_TAIL_FRAMES + cadence
    torch.manual_seed(100 + geometry)
    mel = torch.randn(2, FEAT, width)
    mel[1, :, cadence - 7 :] = 0
    lengths = torch.tensor([width, cadence - 7])
    with torch.no_grad():
        expected = _frozen_subsampling(pre_encode, mel)
        actual, _ = pre_encode(mel, lengths, mask_padding=True)
        unmasked, _ = pre_encode(mel, lengths)
    valid = pre_encode.output_lengths(lengths).tolist()
    assert valid[0] == expected.shape[1]
    assert torch.equal(unmasked, expected)
    for row, frames in enumerate(valid):
        assert torch.equal(actual[row, :frames], expected[row, :frames])


def _fresh_caches(batch: int) -> Any:
    return encoder_mod.StreamingCaches(
        n_layers=N_LAYERS,
        batch=batch,
        d_model=D_MODEL,
        left_context=WINDOW,
        conv_kernel=KERNEL,
        device=torch.device("cpu"),
    )


# @spec PORT-ADV-004, PORT-PERF-009
@pytest.mark.parametrize("geometry", range(len(LOOKAHEADS)))
def test_full_width_rows_are_bitwise_unchanged_by_the_mask(geometry: int) -> None:
    # Regular chunks: a continuing row spans the whole 9+C input, and a
    # session-first row's 8L+1 frames keep every subsampling layer's valid
    # length odd, so no valid output reads a masked step. Both must equal
    # the unmasked computation bit for bit, cache writes included.
    core = _core(seed=5)
    cadence = 8 * (LOOKAHEADS[geometry] + 1)
    width = frontend.MEL_TAIL_FRAMES + cadence
    out_width = int(core.encoder.pre_encode.output_lengths(torch.tensor([width]))[0])
    torch.manual_seed(geometry)
    mel = torch.randn(2, FEAT, width)
    mel[1, :, cadence - 7 :] = 0  # session-first row: 8L+1 valid frames
    mel_lengths = torch.tensor([width, cadence - 7])
    offsets = torch.tensor([advance.PRE_ENCODE_DROP, 0])
    lengths = torch.full((2,), LOOKAHEADS[geometry] + 1)
    reference_caches, masked_caches = _fresh_caches(2), _fresh_caches(2)
    for caches in (reference_caches, masked_caches):
        caches.channel.normal_()
        caches.time.normal_()
        caches.valid.fill_(WINDOW)
    masked_caches.channel.copy_(reference_caches.channel)
    masked_caches.time.copy_(reference_caches.time)
    common = {"out_offsets": offsets, "out_lengths": lengths, "out_width": out_width}
    with torch.no_grad():
        expected = _unmasked_stream_step(core.encoder, mel, reference_caches, **common)
        actual = encoder_mod.stream_step(
            core.encoder, mel, masked_caches, mel_lengths=mel_lengths, **common
        )
    assert torch.equal(actual, expected)
    assert torch.equal(masked_caches.channel, reference_caches.channel)
    assert torch.equal(masked_caches.time, reference_caches.time)
    assert torch.equal(masked_caches.valid, reference_caches.valid)


# @spec PORT-PERF-010
def test_transition_tuple_carries_the_valid_mel_width() -> None:
    # The serving transition receives the per-row valid mel width as a
    # tensor (a staged graph input, never a Python constant).
    core = _core(seed=5)
    received: list[tuple[Any, ...]] = []
    execute = mods["encoder_execution"].execute_encoder_transition

    def observe(*args: Any) -> Any:
        received.append(args)
        return execute(core, *args)

    state = _fresh_state()
    unit = 8 * (LOOKAHEADS[1] + 1) * HOP
    signal = torch.randn(1, unit)
    for seq, (samples, final) in enumerate(
        ((signal, False), (signal[:, :1_000], True))
    ):
        padded = torch.zeros(1, unit)
        padded[:, : samples.shape[1]] = samples
        batch = advance.ChunkBatch(
            samples=padded,
            valid_samples=torch.tensor([samples.shape[1]]),
            geometry_id=torch.tensor([1]),
            final_tail=torch.tensor([final]),
            prompt_index=torch.tensor([0]),
            chunk_sequence=torch.tensor([seq]),
        )
        with torch.no_grad():
            advance.advance_session(
                core,
                batch,
                state,
                geometry=1,
                decode_fn=rnnt.decode_compact_active,
                encoder_transition=observe,
            )
    # First chunk: B_1 = 9 frames, no prefix. Final: 3,560 samples hold
    # 22 frames, 13 past B_1, behind the nine-frame prefix: 9 + 13 = 22.
    assert [args[6].tolist() for args in received] == [[9], [22]]
    assert all(args[6].dtype == torch.int64 for args in received)
