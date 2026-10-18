# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage for the canonical single-session transaction.

Real package imports preserve shared class identity across the unit suite.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]

_PKG = Path(__file__).resolve().parents[3] / "vllm/models/nemotron_asr"
_BASE = "vllm.models.nemotron_asr"


def _load_chain() -> dict[str, Any]:
    """Import real packages so collection preserves shared module identity."""
    loaded = {
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
    return loaded


mods = _load_chain()
advance = mods["advance"]
frontend = mods["frontend"]
rnnt = mods["rnnt"]
manifests = mods["manifests"]

FEAT = 16
D_MODEL = 32
N_LAYERS = 2
KERNEL = 5
WINDOW = 8
VOCAB = 12
CHUNK = 17_920  # 1120 ms cadence (geometry id 4, lookahead 13)
GEOMETRY_1120 = 4
LOOKAHEAD = 13
CADENCE = 8 * (LOOKAHEAD + 1)  # 112
PAD_FRAMES = CADENCE  # one reference regular/final cadence shift
# Frontend: a legal final residual is strictly under one cadence per
# PORT-SESS-001/003); mel width = MEL_TAIL_FRAMES + PAD_FRAMES
MEL_WIDTH = frontend.MEL_TAIL_FRAMES + PAD_FRAMES
RAW_TAIL = 1_953


def _core() -> Any:
    torch.manual_seed(7)
    encoder = mods["encoder"].FastConformerEncoder(
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
    return SimpleNamespace(
        encoder=encoder,
        lid=mods["lid"].PromptConditioner(enc_hidden=D_MODEL, num_prompts=4),
        predictor=rnnt.Predictor(vocab_size=VOCAB, pred_hidden=16, pred_rnn_layers=2),
        joint=rnnt.Joint(
            enc_hidden=D_MODEL,
            pred_hidden=16,
            joint_hidden=16,
            vocab_size=VOCAB,
        ),
        featurizer=mods["featurizer"].MelFeaturizer(
            filterbank=torch.rand(FEAT, 257) * 0.01,
            window=torch.hann_window(400),
        ),
        blank_id=VOCAB,
    )


def _fresh_state(batch: int) -> Any:
    return advance.SessionStateBatch(
        raw_tail=torch.zeros(batch, RAW_TAIL),
        mel_tail=torch.zeros(batch, FEAT, frontend.MEL_TAIL_FRAMES),
        frontend_counters=torch.zeros(batch, 8, dtype=torch.int64),
        channel=[torch.zeros(batch, WINDOW, D_MODEL) for _ in range(N_LAYERS)],
        window_valid=[
            torch.zeros(batch, 1, dtype=torch.int32) for _ in range(N_LAYERS)
        ],
        time=[torch.zeros(batch, D_MODEL, KERNEL - 1) for _ in range(N_LAYERS)],
        h=torch.zeros(batch, 2, 16),
        c=torch.zeros(batch, 2, 16),
        last_label=torch.full((batch,), VOCAB, dtype=torch.long),
    )


def _stack_states(states: list[Any]) -> Any:
    return advance.SessionStateBatch(
        raw_tail=torch.cat([s.raw_tail for s in states]),
        mel_tail=torch.cat([s.mel_tail for s in states]),
        frontend_counters=torch.cat([s.frontend_counters for s in states]),
        channel=[torch.cat([s.channel[i] for s in states]) for i in range(N_LAYERS)],
        window_valid=[
            torch.cat([s.window_valid[i] for s in states]) for i in range(N_LAYERS)
        ],
        time=[torch.cat([s.time[i] for s in states]) for i in range(N_LAYERS)],
        h=torch.cat([s.h for s in states]),
        c=torch.cat([s.c for s in states]),
        last_label=torch.cat([s.last_label for s in states]),
    )


def _row_clone(state: Any, row: int) -> dict[str, Any]:
    """Clone EVERY state family for one row (PORT-ADV-004's
    no-mutation promise covers the whole snapshot)."""
    r = slice(row, row + 1)
    return {
        "raw_tail": state.raw_tail[r].clone(),
        "mel_tail": state.mel_tail[r].clone(),
        "frontend_counters": state.frontend_counters[r].clone(),
        "channel": [t[r].clone() for t in state.channel],
        "window_valid": [t[r].clone() for t in state.window_valid],
        "time": [t[r].clone() for t in state.time],
        "h": state.h[r].clone(),
        "c": state.c[r].clone(),
        "last_label": state.last_label[r].clone(),
    }


def _assert_row_unchanged(state: Any, row: int, before: dict[str, Any]) -> None:
    """Bit-identical comparison over the full row snapshot."""
    r = slice(row, row + 1)
    for key in (
        "raw_tail",
        "mel_tail",
        "frontend_counters",
        "h",
        "c",
        "last_label",
    ):
        torch.testing.assert_close(getattr(state, key)[r], before[key], rtol=0, atol=0)
    for fam in ("channel", "window_valid", "time"):
        for layer in range(N_LAYERS):
            torch.testing.assert_close(
                getattr(state, fam)[layer][r],
                before[fam][layer],
                rtol=0,
                atol=0,
            )


def _chunk(
    samples: torch.Tensor,
    *,
    seq: int,
    final: bool = False,
    prompts: list[int] | None = None,
) -> Any:
    batch = samples.shape[0]
    return advance.ChunkBatch(
        samples=samples,
        valid_samples=torch.full((batch,), samples.shape[1], dtype=torch.long),
        geometry_id=torch.full((batch,), GEOMETRY_1120, dtype=torch.long),
        final_tail=torch.full((batch,), final, dtype=torch.bool),
        prompt_index=torch.tensor(
            prompts if prompts is not None else [0] * batch,
            dtype=torch.long,
        ),
        chunk_sequence=torch.full((batch,), seq, dtype=torch.long),
    )


def _advance(core: Any, batch: Any, state: Any, **kw: Any) -> Any:
    # Decode selection is NEVER a hardcoded default (PORT-DEC-008):
    # the harness passes the eager compact loop explicitly; the
    # mixed-phase differential runs both candidates.
    kw.setdefault("decode_fn", rnnt.decode_compact_active)
    return advance.advance_session(core, batch, state, geometry=GEOMETRY_1120, **kw)


def test_encode_phase_exits_before_decode_phase_enters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-ADV-001, PORT-PERF-007
    events: list[str] = []

    class _RecordedPhase:
        def __init__(self, name: str) -> None:
            self._name = name

        def __enter__(self) -> None:
            events.append(f"{self._name}:enter")

        def __exit__(self, *_args: Any) -> None:
            events.append(f"{self._name}:exit")

    monkeypatch.setattr(advance, "phase", _RecordedPhase)
    core = _core()
    state = _fresh_state(1)
    torch.manual_seed(20)
    _advance(core, _chunk(torch.randn(1, CHUNK) * 0.1, seq=0), state)

    assert events.index("port.encode:exit") < events.index("port.decode:enter")


@pytest.mark.parametrize(
    ("geometry", "label", "lookahead"),
    [
        (geometry, label, right)
        for geometry, (label, (_, right)) in enumerate(manifests.CADENCES.items())
    ],
)
def test_decode_receives_only_geometry_valid_encoder_frames(
    geometry: int,
    label: str,
    lookahead: int,
) -> None:
    # @spec PORT-ADV-004, PORT-PERF-004
    # The encoder retains its fixed padded width for capture, while decode's
    # fixed-trip width is the geometry manifest's maximum valid prefix. The
    # pre-encode cache produces two trailing padded outputs which must not
    # become part of the dense graph key or its repeated label loop.
    core = _core()
    state = _fresh_state(1)
    valid_width = lookahead + 1
    raw_width = manifests.RAW_SAMPLES_PER_CHUNK[label]
    batch = advance.ChunkBatch(
        samples=torch.zeros(1, raw_width),
        valid_samples=torch.tensor([raw_width]),
        geometry_id=torch.tensor([geometry]),
        final_tail=torch.tensor([False]),
        prompt_index=torch.tensor([0]),
        chunk_sequence=torch.tensor([0]),
    )

    def decode_probe(
        enc_frames: torch.Tensor,
        enc_lengths: torch.Tensor,
        _predictor: Any,
        _joint: Any,
        decode_state: Any,
    ) -> Any:
        assert enc_frames.shape[1] == valid_width
        assert enc_lengths.tolist() == [valid_width]
        return (
            torch.zeros(
                1,
                valid_width * rnnt.MAX_SYMBOLS_PER_STEP,
                dtype=torch.int32,
            ),
            torch.zeros(1, dtype=torch.int32),
            decode_state,
        )

    result = advance.advance_session(
        core,
        batch,
        state,
        geometry=geometry,
        decode_fn=decode_probe,
        capture=True,
    )

    assert result.captures is not None
    assert result.captures.encoder_conditioned.shape[1] == valid_width + 2


# @spec PORT-ADV-004, PORT-PERF-004
def test_decode_accepts_exact_live_state_with_only_outer_tier_padding() -> None:
    core = _core()
    state = _fresh_state(3)
    samples = torch.zeros(3, CHUNK)

    def tier_padded_decode(
        enc_frames: torch.Tensor,
        enc_lengths: torch.Tensor,
        _predictor: Any,
        _joint: Any,
        decode_state: Any,
    ) -> Any:
        del enc_frames, enc_lengths
        layers, live, hidden = decode_state.h.shape
        tier = 4
        assert live == 3
        h_storage = torch.zeros(layers, tier, hidden)
        c_storage = torch.zeros_like(h_storage)
        h_storage[:, :live].copy_(decode_state.h + 1)
        c_storage[:, :live].copy_(decode_state.c + 2)
        next_h = h_storage[:, :live]
        next_c = c_storage[:, :live]
        assert next_h.stride() == (tier * hidden, hidden, 1)
        assert not next_h.is_contiguous()
        return (
            torch.ones(live, 1, dtype=torch.int32),
            torch.ones(live, dtype=torch.int32),
            rnnt.DecodeState(
                h=next_h,
                c=next_c,
                last_label=torch.ones(live, dtype=torch.long),
            ),
        )

    _advance(
        core,
        _chunk(samples, seq=0),
        state,
        decode_fn=tier_padded_decode,
    )

    torch.testing.assert_close(state.h, torch.ones_like(state.h))
    torch.testing.assert_close(state.c, torch.full_like(state.c, 2))
    assert state.last_label.tolist() == [1, 1, 1]


# @spec PORT-ADV-004
def test_decode_rejects_noncanonical_inner_state_layout() -> None:
    core = _core()
    state = _fresh_state(1)

    def inner_strided_decode(
        enc_frames: torch.Tensor,
        enc_lengths: torch.Tensor,
        _predictor: Any,
        _joint: Any,
        decode_state: Any,
    ) -> Any:
        del enc_frames, enc_lengths
        layers, live, hidden = decode_state.h.shape
        h = torch.zeros(layers, live, hidden * 2)[:, :, ::2]
        assert h.shape == decode_state.h.shape
        assert h.stride(-1) == 2
        return (
            torch.zeros(live, 1, dtype=torch.int32),
            torch.zeros(live, dtype=torch.int32),
            rnnt.DecodeState(
                h=h,
                c=decode_state.c,
                last_label=decode_state.last_label,
            ),
        )

    with pytest.raises(ValueError, match="next-state h"):
        _advance(
            core,
            _chunk(torch.zeros(1, CHUNK), seq=0),
            state,
            decode_fn=inner_strided_decode,
        )


def test_advance_session_reuses_geometry_and_transition_input_authorities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-010
    advance = mods["advance"]
    encoder_execution = mods["encoder_execution"]
    geometry_shape = getattr(
        encoder_execution,
        "encoder_geometry_shape",
        None,
    )
    prepare_inputs = getattr(
        advance,
        "prepare_encoder_transition_inputs",
        None,
    )
    assert callable(geometry_shape), "PORT-PERF-010 missing encoder_geometry_shape"
    assert callable(prepare_inputs), (
        "PORT-PERF-010 missing prepare_encoder_transition_inputs"
    )
    geometry_results: list[Any] = []
    prepared: list[tuple[Any, ...]] = []
    received: list[tuple[Any, ...]] = []

    def observe_shape(*args: Any, **kwargs: Any) -> Any:
        result = geometry_shape(*args, **kwargs)
        geometry_results.append(result)
        return result

    def observe_inputs(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        result = prepare_inputs(*args, **kwargs)
        assert isinstance(result, tuple)
        prepared.append(result)
        return result

    def observe_transition(*args: Any) -> Any:
        received.append(args)
        return encoder_execution.execute_encoder_transition(core, *args)

    monkeypatch.setattr(
        encoder_execution,
        "encoder_geometry_shape",
        observe_shape,
    )
    monkeypatch.setattr(
        advance,
        "prepare_encoder_transition_inputs",
        observe_inputs,
    )
    core = _core()
    _advance(
        core,
        _chunk(torch.zeros(1, CHUNK), seq=0),
        _fresh_state(1),
        encoder_transition=observe_transition,
    )

    assert len(geometry_results) == 1
    assert len(prepared) == len(received) == 1
    assert len(prepared[0]) == len(received[0])
    assert all(
        prepared_arg is received_arg
        for prepared_arg, received_arg in zip(
            prepared[0],
            received[0],
            strict=True,
        )
    )
    assert geometry_results[0].mel_width == int(received[0][0].shape[2])
    assert geometry_results[0].out_width == received[0][4]


def test_static_fullgraph_encoder_transition_matches_eager_exactly() -> None:
    # @spec PORT-PERF-009
    # The production candidate uses Inductor, but this contract is
    # backend-independent: Dynamo must capture the real cache-mutating
    # transition as one full static graph, and that graph must preserve every
    # observable tensor and resident-state write exactly. ``backend="eager"``
    # keeps this CPU gate independent of a platform compiler toolchain.
    core = _core()
    torch.manual_seed(20)
    batch = _chunk(torch.randn(1, CHUNK) * 0.1, seq=0)
    eager_state = _fresh_state(1)
    graph_state = _fresh_state(1)

    eager = _advance(core, batch, eager_state, capture=True)
    encoder_execution = mods["encoder_execution"]

    def transition(*args: Any) -> Any:
        return encoder_execution.execute_encoder_transition(core, *args)

    fullgraph = torch.compile(
        transition,
        backend="eager",
        fullgraph=True,
        dynamic=False,
    )
    graphed = _advance(
        core,
        batch,
        graph_state,
        encoder_transition=fullgraph,
        capture=True,
    )

    for field in (
        "token_ids",
        "token_lengths",
        "row_valid",
        "row_status",
        "frame_emission_counts",
        "frame_final_labels",
        "frame_valid_lengths",
    ):
        torch.testing.assert_close(
            getattr(graphed, field),
            getattr(eager, field),
            rtol=0,
            atol=0,
        )
    assert eager.captures is not None and graphed.captures is not None
    for field in (
        "frontend_mel",
        "mel_lengths",
        "encoder_raw",
        "encoder_conditioned",
        "encoder_lengths",
    ):
        torch.testing.assert_close(
            getattr(graphed.captures, field),
            getattr(eager.captures, field),
            rtol=0,
            atol=0,
        )
    _assert_row_unchanged(graph_state, 0, _row_clone(eager_state, 0))


def test_session_first_chunk_advances_and_captures() -> None:
    # @spec PORT-ADV-001
    # The canonical transition, exercised: session-first 1120 ms chunk
    # commits B_1 = 105 mel frames, runs encoder/LID/decode, advances
    # every counter, and (capture ON) stages the three named tensors
    # at the bucket's fixed padded widths with exact logical lengths.
    core = _core()
    state = _fresh_state(1)
    torch.manual_seed(21)
    samples = torch.randn(1, CHUNK) * 0.1
    result = _advance(core, _chunk(samples, seq=0), state, capture=True)
    ctr = state.frontend_counters[0]
    b1 = frontend.cadence_boundary(1, lookahead=13)
    assert int(ctr[frontend.CTR_COMMITTED_MEL_FRAMES]) == b1 == 105
    assert int(ctr[frontend.CTR_ENCODED_MEL_FRAMES]) == b1
    assert int(ctr[frontend.CTR_EXPECTED_CHUNK_SEQUENCE]) == 1
    assert result.row_valid.tolist() == [True]
    caps = result.captures
    assert caps is not None
    # Fixed bucket width; logical length = 105 (no prefix, first).
    assert caps.frontend_mel.shape == (1, FEAT, MEL_WIDTH)
    assert int(caps.mel_lengths[0]) == b1
    expected_f = int(core.encoder.pre_encode.output_lengths(torch.tensor([b1]))[0])
    assert int(caps.encoder_lengths[0]) == expected_f
    assert caps.encoder_raw.shape[1] >= expected_f
    # Padded capture columns are exactly zero.
    assert not bool(caps.frontend_mel[0, :, b1:].any())
    assert not bool(caps.encoder_raw[0, expected_f:].any())
    assert result.token_ids.shape[0] == 1
    # The encoder window advanced: valid length grew and is mirrored
    # across every layer's slot.
    assert int(state.window_valid[0][0, 0]) > 0
    assert int(state.window_valid[1][0, 0]) == int(state.window_valid[0][0, 0])


def test_capture_off_returns_none() -> None:
    # @spec PORT-HOOK-001
    core = _core()
    state = _fresh_state(1)
    torch.manual_seed(22)
    samples = torch.randn(1, CHUNK) * 0.1
    result = _advance(core, _chunk(samples, seq=0), state)
    assert result.captures is None
    assert result.token_ids.shape[0] == 1


def test_continuing_chunk_prepends_the_prefix_and_drops() -> None:
    # @spec PORT-ADV-001
    # Chunk 2 consumes [9-slot prefix + C frames] with the configured
    # drop; commits land exactly on B_2.
    core = _core()
    state = _fresh_state(1)
    torch.manual_seed(23)
    samples = torch.randn(1, 2 * CHUNK) * 0.1
    _advance(core, _chunk(samples[:, :CHUNK], seq=0), state)
    result = _advance(core, _chunk(samples[:, CHUNK:], seq=1), state, capture=True)
    b2 = frontend.cadence_boundary(2, lookahead=13)
    assert int(state.frontend_counters[0, frontend.CTR_COMMITTED_MEL_FRAMES]) == b2
    caps = result.captures
    assert caps is not None
    # Fixed width; logical length = prefix (9) + C = 121.
    assert caps.frontend_mel.shape[2] == MEL_WIDTH
    assert int(caps.mel_lengths[0]) == frontend.MEL_TAIL_FRAMES + CADENCE


def test_wrong_sequence_row_is_masked_and_reported() -> None:
    # @spec PORT-ADV-004
    # A protocol-invalid row (wrong chunk sequence) in a two-row
    # bucket: that row mutates nothing and returns a zero-length
    # burst; row_valid reports it; the sibling valid row advances
    # exactly as its single-row run.
    core = _core()
    torch.manual_seed(24)
    samples = torch.randn(2, CHUNK) * 0.1
    state = _fresh_state(2)
    before = _row_clone(state, 1)
    batch = _chunk(samples, seq=0)
    batch.chunk_sequence[1] = 3  # wrong: expected 0
    result = _advance(core, batch, state, capture=True)
    assert result.row_valid.tolist() == [True, False]
    assert result.row_status.tolist() == [
        0,
        frontend.ROW_STATUS_SEQUENCE,
    ]
    assert int(result.token_lengths[1]) == 0
    _assert_row_unchanged(state, 1, before)
    # The valid row matches its single-row run.
    state1 = _fresh_state(1)
    r1 = _advance(core, _chunk(samples[:1], seq=0), state1, capture=True)
    assert int(result.token_lengths[0]) == int(r1.token_lengths[0])
    torch.testing.assert_close(
        state.frontend_counters[0:1],
        state1.frontend_counters,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(state.h[0:1], state1.h, rtol=0, atol=1e-6)


def test_geometry_mismatch_row_is_masked() -> None:
    # @spec PORT-ADV-004
    # An envelope geometry different from the bucket geometry is a
    # per-row protocol violation: masked no-op, reported.
    core = _core()
    torch.manual_seed(27)
    samples = torch.randn(1, CHUNK) * 0.1
    state = _fresh_state(1)
    before = _row_clone(state, 0)
    batch = _chunk(samples, seq=0)
    batch.geometry_id[0] = 2  # bucket is GEOMETRY_1120 = 4
    result = _advance(core, batch, state)
    assert result.row_valid.tolist() == [False]
    assert result.row_status.tolist() == [frontend.ROW_STATUS_GEOMETRY]
    assert int(result.token_lengths[0]) == 0
    _assert_row_unchanged(state, 0, before)


def test_zero_frame_final_leaves_state_and_stages_captures() -> None:
    # @spec PORT-ADV-001 / PORT-HOOK-001
    # A final tail whose short residual is dropped rides the uniform
    # masked path: encoder/predictor state untouched, zero-length
    # burst, finalization marked, and (capture ON) all three named
    # tensors present at zero logical length.
    core = _core()
    state = _fresh_state(1)
    torch.manual_seed(25)
    samples = torch.randn(1, CHUNK) * 0.1
    _advance(core, _chunk(samples, seq=0), state)
    h_before = state.h.clone()
    window_before = state.channel[0].clone()
    valid_before = state.window_valid[0].clone()
    result = _advance(
        core,
        _chunk(torch.zeros(1, 0), seq=1, final=True),
        state,
        capture=True,
    )
    assert int(state.frontend_counters[0, frontend.CTR_FINALIZED]) == 1
    assert result.row_valid.tolist() == [True]
    torch.testing.assert_close(state.h, h_before, rtol=0, atol=0)
    torch.testing.assert_close(state.channel[0], window_before, rtol=0, atol=0)
    torch.testing.assert_close(state.window_valid[0], valid_before, rtol=0, atol=0)
    assert int(result.token_lengths[0]) == 0
    caps = result.captures
    assert caps is not None
    assert int(caps.mel_lengths[0]) == 0
    assert int(caps.encoder_lengths[0]) == 0
    assert not bool(caps.frontend_mel.any())
    assert not bool(caps.encoder_raw.any())


def test_mixed_prompts_condition_row_wise() -> None:
    # @spec PORT-ADV-001
    # Two rows, identical audio, different prompts: conditioned
    # captures must differ (no shared multi-hot prompt), and each row
    # must equal its single-row run exactly.
    core = _core()
    torch.manual_seed(26)
    samples = torch.randn(1, CHUNK) * 0.1
    two = samples.repeat(2, 1)
    state2 = _fresh_state(2)
    result2 = _advance(core, _chunk(two, seq=0, prompts=[0, 2]), state2, capture=True)
    caps2 = result2.captures
    assert caps2 is not None
    assert bool((caps2.encoder_conditioned[0] != caps2.encoder_conditioned[1]).any())
    for row, prompt in ((0, 0), (1, 2)):
        state1 = _fresh_state(1)
        r1 = _advance(
            core,
            _chunk(samples, seq=0, prompts=[prompt]),
            state1,
            capture=True,
        )
        c1 = r1.captures
        assert c1 is not None
        torch.testing.assert_close(
            caps2.encoder_conditioned[row],
            c1.encoder_conditioned[0],
            rtol=0,
            atol=1e-6,  # provisional cross-batch-shape bound
        )
        assert int(result2.token_lengths[row]) == int(r1.token_lengths[0])


@pytest.mark.parametrize(
    "decode_name", ["decode_compact_active", "decode_dense_masked"]
)
def test_mixed_phase_bucket_rows_match_single_rows(
    decode_name: str,
) -> None:
    # @spec PORT-ADV-004
    # THE length-aware transition differential, run under BOTH
    # unselected decode candidates (PORT-DEC-008 keeps them unselected
    # until the profile): session-first, continuing, committable-final,
    # and zero-frame-final rows advance in ONE bucket call, each equal
    # to its own single-row run — counters exact, states and captures
    # at the provisional cross-shape bound, token bursts identical.
    decode_fn = getattr(rnnt, decode_name)
    core = _core()
    torch.manual_seed(31)
    signals = [torch.randn(2 * CHUNK) * s for s in (0.1, 0.2, 0.15, 0.25)]

    # Phases: 0 session-first; 1 continuing; 2 final with a
    # committable 69-frame residual; 3 zero-sample final (7-frame
    # residual DROPPED — zero model work inside the mixed bucket).
    second_widths = [CHUNK, CHUNK, 10_000, 0]
    finals = [False, False, True, True]
    seqs = [0, 1, 1, 1]

    singles: list[Any] = []
    results1: list[Any] = []
    for row in range(4):
        s1 = _fresh_state(1)
        if row != 0:
            _advance(
                core,
                _chunk(signals[row][:CHUNK].unsqueeze(0), seq=0),
                s1,
                decode_fn=decode_fn,
            )
        singles.append(s1)
    stateN = _stack_states(singles)

    samples = torch.zeros(4, CHUNK)
    valid = torch.zeros(4, dtype=torch.long)
    for row in range(4):
        w = second_widths[row]
        samples[row, :w] = signals[row][CHUNK : CHUNK + w]
        valid[row] = w
    batchN = advance.ChunkBatch(
        samples=samples,
        valid_samples=valid,
        geometry_id=torch.full((4,), GEOMETRY_1120, dtype=torch.long),
        final_tail=torch.tensor(finals),
        prompt_index=torch.tensor([0, 2, 1, 0], dtype=torch.long),
        chunk_sequence=torch.tensor(seqs, dtype=torch.long),
    )
    resultN = _advance(core, batchN, stateN, capture=True, decode_fn=decode_fn)
    assert resultN.row_valid.tolist() == [True] * 4
    assert resultN.row_status.tolist() == [0] * 4

    for row in range(4):
        s1 = singles[row]
        w = second_widths[row]
        batch1 = advance.ChunkBatch(
            samples=samples[row : row + 1],
            valid_samples=valid[row : row + 1],
            geometry_id=torch.tensor([GEOMETRY_1120]),
            final_tail=torch.tensor([finals[row]]),
            prompt_index=batchN.prompt_index[row : row + 1],
            chunk_sequence=torch.tensor([seqs[row]], dtype=torch.long),
        )
        r1 = _advance(core, batch1, s1, capture=True, decode_fn=decode_fn)
        results1.append(r1)
        # Token bursts identical.
        nb = int(resultN.token_lengths[row])
        assert nb == int(r1.token_lengths[0]), f"row {row}"
        assert resultN.token_ids[row, :nb].tolist() == r1.token_ids[0, :nb].tolist(), (
            f"row {row}"
        )
        # Frontend counters exact.
        torch.testing.assert_close(
            stateN.frontend_counters[row : row + 1],
            s1.frontend_counters,
            rtol=0,
            atol=0,
        )
        # Encoder caches, conv tails, window_valid, predictor state.
        for layer in range(N_LAYERS):
            torch.testing.assert_close(
                stateN.channel[layer][row : row + 1],
                s1.channel[layer],
                rtol=0,
                atol=1e-5,
            )
            torch.testing.assert_close(
                stateN.time[layer][row : row + 1],
                s1.time[layer],
                rtol=0,
                atol=1e-5,
            )
            torch.testing.assert_close(
                stateN.window_valid[layer][row : row + 1],
                s1.window_valid[layer],
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(stateN.h[row : row + 1], s1.h, rtol=0, atol=1e-6)
        torch.testing.assert_close(stateN.c[row : row + 1], s1.c, rtol=0, atol=1e-6)
        # Captures: logical lengths equal; valid regions at the bound.
        capsN, caps1 = resultN.captures, r1.captures
        assert capsN is not None and caps1 is not None
        assert int(capsN.mel_lengths[row]) == int(caps1.mel_lengths[0])
        assert int(capsN.encoder_lengths[row]) == int(caps1.encoder_lengths[0])
        ml = int(capsN.mel_lengths[row])
        el = int(capsN.encoder_lengths[row])
        torch.testing.assert_close(
            capsN.frontend_mel[row, :, :ml],
            caps1.frontend_mel[0, :, :ml],
            rtol=0,
            atol=2e-6,
        )
        torch.testing.assert_close(
            capsN.encoder_conditioned[row, :el],
            caps1.encoder_conditioned[0, :el],
            rtol=0,
            atol=1e-5,
        )
    # Zero-frame final row: untouched model state, finalized.
    assert int(stateN.frontend_counters[3, frontend.CTR_FINALIZED]) == 1
    assert int(resultN.token_lengths[3]) == 0


def test_oversize_final_row_is_masked_and_reported() -> None:
    # @spec PORT-ADV-004
    # A final-tail carrier holding one full cadence unit or more is
    # illegal ingress (PORT-SESS-001/003 drain complete cadences as
    # regular units before minting the actual residual): the row is a
    # masked no-op with the named status bit; nothing commits.
    core = _core()
    state = _fresh_state(1)
    torch.manual_seed(28)
    samples = torch.randn(1, 2 * CHUNK) * 0.1
    _advance(core, _chunk(samples[:, :CHUNK], seq=0), state)
    before = _row_clone(state, 0)
    result = _advance(
        core,
        _chunk(samples[:, CHUNK:], seq=1, final=True),
        state,
    )
    assert result.row_valid.tolist() == [False]
    assert result.row_status.tolist() == [frontend.ROW_STATUS_FINAL_OVERSIZE]
    assert int(result.token_lengths[0]) == 0
    _assert_row_unchanged(state, 0, before)
    # The session is NOT finalized — recoverable, per the row tier.
    assert int(state.frontend_counters[0, frontend.CTR_FINALIZED]) == 0
