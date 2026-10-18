# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PORT-OBS-008 batch-size sub-stat contracts.

Each executed nonempty CHUNK geometry bucket records its geometry and row
count through the consume-once batch-stat hook. CPU fixtures exercise the
canonical transaction with real package imports.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

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
            "rnnt_cell",
            "rnnt",
            "decode_dispatch",
            "state_scatter",
            "advance",
        )
    }
    return loaded


mods = _load_chain()
advance = mods["advance"]
rnnt = mods["rnnt"]
manifests = mods["manifests"]

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]

FEAT = 16
D_MODEL = 32
N_LAYERS = 2
KERNEL = 5
WINDOW = 8
PARK_ID = 9000
PLACEHOLDER_ID = 9001
VOCAB = 12
CAP = 48
CARRIER_HIDDEN = 5_127
RAW_TAIL = 1_953
NULL_INDEX = 0

GEOM_REG = 0
REG_SAMPLES = 1_280
GEOM_FINAL = 2
FINAL_SAMPLES = 3_840
# PORT-OBS-008 amended (Phase-6 round 2, Q3, lead-authorized): recorded
# stats now carry the cadence label, resolved from the geometry
# authority (manifests.CADENCES) at recording time — never the bare
# geometry id.
CADENCE_REG = list(manifests.CADENCES)[GEOM_REG].removesuffix("ms")
CADENCE_FINAL = list(manifests.CADENCES)[GEOM_FINAL].removesuffix("ms")

_BOOK = {name: i for i, (name, _) in enumerate(manifests.BOOK_FIELDS)}
_CTR = {name: i for i, name in enumerate(manifests.FRONTEND_COUNTER_FIELDS)}
BOOK_WIDTH = len(manifests.BOOK_FIELDS)
CTR_WIDTH = len(manifests.FRONTEND_COUNTER_FIELDS)

Pools = dict[str, Any]


def _tiny_core(seed: int = 7) -> Any:
    torch.manual_seed(seed)
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
            enc_hidden=D_MODEL, pred_hidden=16, joint_hidden=16, vocab_size=VOCAB
        ),
        featurizer=mods["featurizer"].MelFeaturizer(
            filterbank=torch.rand(FEAT, 257) * 0.01,
            window=torch.hann_window(400),
        ),
        blank_id=VOCAB,
    )


def _fresh_pools(num_blocks: int = 4) -> Pools:
    return {
        "channel_pools": [
            torch.zeros(num_blocks, WINDOW, D_MODEL) for _ in range(N_LAYERS)
        ],
        "time_pools": [
            torch.zeros(num_blocks, D_MODEL, KERNEL - 1) for _ in range(N_LAYERS)
        ],
        "len_pools": [
            torch.zeros(num_blocks, 1, dtype=torch.int32) for _ in range(N_LAYERS)
        ],
        "h_pool": torch.zeros(num_blocks, 2, 16),
        "c_pool": torch.zeros(num_blocks, 2, 16),
        "queue_pool": torch.zeros(num_blocks, CAP, dtype=torch.int32),
        "book_pool": torch.zeros(num_blocks, BOOK_WIDTH, dtype=torch.int32),
        "frontend_raw_pool": torch.zeros(num_blocks, RAW_TAIL),
        "frontend_mel_pool": torch.zeros(num_blocks, FEAT, 9),
        "frontend_counter_pool": torch.zeros(num_blocks, CTR_WIDTH, dtype=torch.int64),
    }


def _plan(
    *,
    prefills: list[int] | None = None,
    decodes: list[int] | None = None,
    num_pool_blocks: int,
    geometries: list[int] | None = None,
    deadlines: list[int] | None = None,
    chunk: list[bool] | None = None,
) -> Any:
    prefills = prefills or []
    decodes = decodes or []
    num_decodes = len(decodes)
    n_real = num_decodes + len(prefills)
    has_initial = [False] * len(prefills)
    prompts = [0] * n_real
    generations = [0] * n_real
    if chunk is None:
        chunk = [False] * num_decodes + [True] * len(prefills)
    if geometries is None:
        geometries = [GEOM_REG] * n_real
    if deadlines is None:
        deadlines = [i + 1 if chunk[i] else 0 for i in range(n_real)]
    request_ids = tuple(f"req-{i}" for i in range(n_real))
    blocks = [*decodes, *prefills]
    bindings = tuple(
        advance.PreparedRowBinding(
            request_id=request_ids[row],
            block_id=blocks[row],
            admission_generation=generations[row],
            geometry_id=geometries[row],
            prompt_index=prompts[row],
            prior_prompt_index=prompts[row],
            allow_prompt_transition=False,
            is_chunk=chunk[row],
            ready_deadline_ns=deadlines[row],
        )
        for row in range(n_real)
    )
    d = torch.tensor([[i] for i in decodes], dtype=torch.long).reshape(num_decodes, 1)
    return advance.RowPlan(
        state_indices_d=d,
        num_decodes=num_decodes,
        state_indices_p=torch.tensor(prefills, dtype=torch.long),
        num_prefills=len(prefills),
        has_initial_states_p=torch.tensor(has_initial, dtype=torch.bool),
        null_block_id=NULL_INDEX,
        num_pool_blocks=num_pool_blocks,
        live_block_ids=torch.tensor(list(range(1, num_pool_blocks)), dtype=torch.long),
        geometry_id=torch.tensor(geometries, dtype=torch.long),
        prompt_index=torch.tensor(prompts, dtype=torch.long),
        is_chunk=torch.tensor(chunk, dtype=torch.bool),
        admission_generation=torch.tensor(generations, dtype=torch.long),
        ready_deadline_ns=torch.tensor(deadlines, dtype=torch.long),
        request_ids=request_ids,
        execution_tier=0,
        bindings=bindings,
    )


def _set_drained_book(pools: Pools, block: int, *, blank: int) -> None:
    """A FLUSH/async-park-echo-ready book: empty queue, no pending echo."""
    book = pools["book_pool"]
    book[block, _BOOK["queue_head"]] = 0
    book[block, _BOOK["queue_length"]] = 0
    book[block, _BOOK["last_label"]] = blank
    book[block, _BOOK["pending_echo"]] = 0
    book[block, _BOOK["expected_label"]] = 0


def _set_replay_book(
    pools: Pools, block: int, *, queue: list[int], head: int, expected: int
) -> None:
    """A REPLAY-ready book: an armed pending echo over a nonempty queue."""
    book = pools["book_pool"]
    qp = pools["queue_pool"]
    for i, label in enumerate(queue):
        qp[block, i] = label
    book[block, _BOOK["queue_head"]] = head
    book[block, _BOOK["queue_length"]] = len(queue)
    book[block, _BOOK["last_label"]] = queue[-1]
    book[block, _BOOK["pending_echo"]] = 1
    book[block, _BOOK["expected_label"]] = expected


def _envelope(
    samples: torch.Tensor,
    *,
    final: bool,
    seq: int,
    geometry: int,
    hidden: int = CARRIER_HIDDEN,
) -> Any:
    n = samples.shape[0]
    assert advance.ENVELOPE_HEADER_SLOTS + n <= hidden
    row = torch.zeros(hidden)
    row[advance.ENV_VERSION] = advance.ENVELOPE_VERSION
    row[advance.ENV_VALID_SAMPLES] = n
    row[advance.ENV_GEOMETRY_ID] = geometry
    row[advance.ENV_FINAL_TAIL] = 1.0 if final else 0.0
    row[advance.ENV_PROMPT_INDEX] = 0
    row[advance.ENV_CHUNK_SEQUENCE] = seq
    row[advance.ENVELOPE_HEADER_SLOTS : advance.ENVELOPE_HEADER_SLOTS + n] = samples
    return row


class _CommitTicket:
    def __init__(self, sink: _CommitRecorder) -> None:
        self._sink = sink

    def stage(self, row_status: torch.Tensor) -> None:
        self._sink.staged.append(row_status.clone())

    def cancel(self) -> None:
        self._sink.cancels += 1


class _CommitRecorder:
    """Minimal composite commit sink: ``capture=True`` requires one."""

    def __init__(self) -> None:
        self.staged: list[torch.Tensor] = []
        self.cancels = 0

    def reserve(self, plan: Any) -> _CommitTicket:
        return _CommitTicket(self)


def _fixed_resolver() -> Any:
    def resolve(request: Any) -> Any:
        return advance.ResolvedDecode(
            arm="dense-eager", decode_fn=rnnt.decode_dense_masked
        )

    return resolve


def _two_geometry_call(*, capture: bool = False) -> None:
    """One executed nonempty CHUNK bucket per geometry (GEOM_REG, GEOM_FINAL)."""
    core = _tiny_core()
    pools = _fresh_pools(num_blocks=4)
    torch.manual_seed(5)
    reg = _envelope(
        torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0, geometry=GEOM_REG
    )
    fin = _envelope(
        torch.randn(FINAL_SAMPLES) * 0.01, final=True, seq=0, geometry=GEOM_FINAL
    )
    plan = _plan(
        prefills=[1, 2],
        num_pool_blocks=4,
        geometries=[GEOM_REG, GEOM_FINAL],
        deadlines=[200, 100],
    )
    adapter = advance.make_mrv1_adapter(
        hidden_size=CARRIER_HIDDEN, park_id=PARK_ID, blank_id=core.blank_id
    )
    advance.advance_model_rows(
        core,
        torch.tensor([PLACEHOLDER_ID, PLACEHOLDER_ID]),
        torch.stack([reg, fin]),
        plan,
        adapter=adapter,
        decode_resolver=_fixed_resolver(),
        placeholder_id=PLACEHOLDER_ID,
        park_id=PARK_ID,
        capture=capture,
        commit_sink=_CommitRecorder() if capture else None,
        **pools,
    )


def _single_geometry_call(*, num_rows: int, geometry: int = GEOM_REG) -> None:
    """``num_rows`` CHUNK prefills, all in ONE geometry bucket."""
    core = _tiny_core()
    pools = _fresh_pools(num_blocks=num_rows + 1)
    torch.manual_seed(5)
    samples = REG_SAMPLES if geometry == GEOM_REG else FINAL_SAMPLES
    envelopes = [
        _envelope(
            torch.randn(samples) * 0.01,
            final=(geometry == GEOM_FINAL),
            seq=0,
            geometry=geometry,
        )
        for _ in range(num_rows)
    ]
    plan = _plan(
        prefills=list(range(1, num_rows + 1)),
        num_pool_blocks=num_rows + 1,
        geometries=[geometry] * num_rows,
        deadlines=[i + 1 for i in range(num_rows)],
    )
    adapter = advance.make_mrv1_adapter(
        hidden_size=CARRIER_HIDDEN, park_id=PARK_ID, blank_id=core.blank_id
    )
    advance.advance_model_rows(
        core,
        torch.tensor([PLACEHOLDER_ID] * num_rows),
        torch.stack(envelopes),
        plan,
        adapter=adapter,
        decode_resolver=_fixed_resolver(),
        placeholder_id=PLACEHOLDER_ID,
        park_id=PARK_ID,
        capture=False,
        **pools,
    )


def _no_chunk_call() -> None:
    """A transaction whose only real rows are FLUSH (drained, non-
    finalized -> async park echo) and REPLAY (armed pending echo) —
    zero CHUNK rows, so the executed-bucket list must be exactly []."""
    core = _tiny_core()
    pools = _fresh_pools(num_blocks=3)
    _set_drained_book(pools, 1, blank=core.blank_id)  # async park echo
    _set_replay_book(pools, 2, queue=[3, 5], head=1, expected=3)  # REPLAY
    plan = _plan(decodes=[1, 2], num_pool_blocks=3)
    adapter = advance.make_mrv1_adapter(
        hidden_size=CARRIER_HIDDEN, park_id=PARK_ID, blank_id=core.blank_id
    )
    advance.advance_model_rows(
        core,
        torch.tensor([PARK_ID, 3], dtype=torch.long),
        torch.zeros(2, CARRIER_HIDDEN),
        plan,
        adapter=adapter,
        decode_resolver=_fixed_resolver(),
        placeholder_id=PLACEHOLDER_ID,
        park_id=PARK_ID,
        capture=False,
        **pools,
    )


def _mixed_replay_flush_and_chunk_call() -> None:
    """One FLUSH row + one REPLAY row + one CHUNK row (GEOM_REG) — the
    executed-bucket list must reflect ONLY the CHUNK row."""
    core = _tiny_core()
    pools = _fresh_pools(num_blocks=4)
    _set_drained_book(pools, 1, blank=core.blank_id)
    _set_replay_book(pools, 2, queue=[3, 5], head=1, expected=3)
    torch.manual_seed(5)
    reg = _envelope(
        torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0, geometry=GEOM_REG
    )
    plan = _plan(
        decodes=[1, 2],
        prefills=[3],
        num_pool_blocks=4,
        geometries=[GEOM_REG, GEOM_REG, GEOM_REG],
        deadlines=[0, 0, 1],
    )
    adapter = advance.make_mrv1_adapter(
        hidden_size=CARRIER_HIDDEN, park_id=PARK_ID, blank_id=core.blank_id
    )
    ids = torch.tensor([PARK_ID, 3, PLACEHOLDER_ID], dtype=torch.long)
    embeds = torch.cat([torch.zeros(2, CARRIER_HIDDEN), reg.unsqueeze(0)], dim=0)
    advance.advance_model_rows(
        core,
        ids,
        embeds,
        plan,
        adapter=adapter,
        decode_resolver=_fixed_resolver(),
        placeholder_id=PLACEHOLDER_ID,
        park_id=PARK_ID,
        capture=False,
        **pools,
    )


# ---- structural pin: no flag exists to disable recording (real, GREEN) -------


# @spec PORT-OBS-008
def test_advance_model_rows_exposes_no_stats_disable_flag() -> None:
    """Recording is unconditional per PORT-OBS-008 — there is, and must
    remain, no parameter that lets a caller opt out of it."""
    params = set(inspect.signature(advance.advance_model_rows).parameters)
    assert not (
        params & {"log_stats", "collect_stats", "stats_enabled", "disable_stats"}
    )


# ---- behavioral: one entry per executed nonempty CHUNK bucket — RED ----------


# @spec PORT-OBS-008
def test_records_one_entry_per_executed_nonempty_chunk_geometry_bucket() -> None:
    _two_geometry_call()
    stats = advance.consume_batch_stats()
    assert sorted(stats) == sorted([(CADENCE_REG, 1), (CADENCE_FINAL, 1)])


# @spec PORT-OBS-008
def test_consume_once_hook_returns_the_list_then_none() -> None:
    _two_geometry_call()
    first = advance.consume_batch_stats()
    assert first is not None and len(first) == 2
    second = advance.consume_batch_stats()
    assert second is None


# @spec PORT-OBS-008
@pytest.mark.parametrize("capture", [True, False])
def test_recording_is_unconditional_regardless_of_capture_flag(capture: bool) -> None:
    """``capture`` (parity tensor capture) is an orthogonal switch; batch
    stats must be recorded whether or not it is enabled — exercised at
    BOTH capture=True and capture=False."""
    _two_geometry_call(capture=capture)
    stats = advance.consume_batch_stats()
    assert stats is not None and len(stats) == 2


# ---- additional batch cases (review correction) -------------------------------


# @spec PORT-OBS-008
def test_no_chunk_transaction_produces_an_empty_list_not_none() -> None:
    """A transaction with zero CHUNK rows (only FLUSH/REPLAY) executes no
    bucket at all — the list must be [] (a transaction that collected but
    found nothing), never None (not collecting)."""
    _no_chunk_call()
    stats = advance.consume_batch_stats()
    assert stats == []


# @spec PORT-OBS-008
def test_multiple_rows_in_one_geometry_aggregate_under_a_single_entry() -> None:
    """Three CHUNK rows sharing ONE geometry execute as a single bucket —
    one (cadence_ms, rows) entry with rows=3, never three entries."""
    _single_geometry_call(num_rows=3, geometry=GEOM_REG)
    stats = advance.consume_batch_stats()
    assert stats == [(CADENCE_REG, 3)]


# @spec PORT-OBS-008
def test_replay_and_flush_rows_are_excluded_from_batch_stats() -> None:
    """A FLUSH row and a REPLAY row alongside one CHUNK row: the executed-
    bucket list must reflect ONLY the CHUNK row's cadence, at rows=1 —
    REPLAY/FLUSH never contribute to any bucket's row count."""
    _mixed_replay_flush_and_chunk_call()
    stats = advance.consume_batch_stats()
    assert stats == [(CADENCE_REG, 1)]


# @spec PORT-OBS-008
def test_successive_transactions_produce_independent_stats() -> None:
    """A second, unrelated transaction's list must be its OWN — never
    carrying over or accumulating entries from an earlier transaction."""
    _two_geometry_call()
    first = advance.consume_batch_stats()
    assert first is not None and sorted(first) == sorted(
        [(CADENCE_REG, 1), (CADENCE_FINAL, 1)]
    )

    _single_geometry_call(num_rows=1, geometry=GEOM_REG)
    second = advance.consume_batch_stats()
    assert second == [(CADENCE_REG, 1)]
