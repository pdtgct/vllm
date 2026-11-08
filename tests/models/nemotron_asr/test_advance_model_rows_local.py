# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU transaction and MRv1 adapter coverage against the real model package.

Tests structural preflight, metadata initialization, geometry dispatch,
capture ownership, status handoff, and pool-free adapter emissions.
"""

from __future__ import annotations

import importlib
from dataclasses import replace
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
            "decode_dispatch",
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
decode_dispatch = mods["decode_dispatch"]

FEAT = 16
D_MODEL = 32
N_LAYERS = 2
KERNEL = 5
WINDOW = 8
PARK_ID = 9000
PLACEHOLDER_ID = 9001
VOCAB = 12
CAP = 48
#: Header (7 slots, design §Ingress-deadline plumbing) + the largest
#: cadence THESE fixtures admit (320 ms = 5,120 raw samples) — the
#: tiny-core analogue of the production 17,927.
CARRIER_HIDDEN = 5_127
RAW_TAIL = 1_953
NULL_INDEX = 0  # NULL_BLOCK_ID at the pin (utils.py:46 @ ee0da84)

GEOM_REG = 0
REG_SAMPLES = 1_280
GEOM_FINAL = 2
FINAL_SAMPLES = 3_840

_BOOK = {name: i for i, (name, _) in enumerate(manifests.BOOK_FIELDS)}
_CTR = {name: i for i, name in enumerate(manifests.FRONTEND_COUNTER_FIELDS)}
BOOK_WIDTH = len(manifests.BOOK_FIELDS)
CTR_WIDTH = len(manifests.FRONTEND_COUNTER_FIELDS)

Pools = dict[str, Any]


def _tiny_core(seed: int = 7) -> Any:
    # Seed 7 is the shared fixture default; the burst-arc test builds
    # a seed-1 core instead — through the REAL featurizer path the
    # seed-7 tiny joint is single-label on every probed signal family
    # (the log-guard-dominated mel is near-constant), while seed 1
    # yields a deterministic cap-saturated burst with two distinct
    # labels, the drain-order discriminator.
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


def _reference_burst(core: Any, samples: torch.Tensor, prompt_index: int) -> list[int]:
    """The labels a session-first final-tail chunk should emit, via
    the golden whole-signal path."""
    with torch.no_grad():
        mel, mel_len = core.featurizer(
            samples.unsqueeze(0), torch.tensor([samples.shape[0]])
        )
        mel = mel[:, :, : int(mel_len[0])]
        caches = mods["encoder"].StreamingCaches(
            n_layers=N_LAYERS,
            batch=1,
            d_model=D_MODEL,
            left_context=WINDOW,
            conv_kernel=KERNEL,
            device=torch.device("cpu"),
        )
        enc = mods["encoder"].stream_step(core.encoder, mel, caches, drop_extra=0)
        conditioned = core.lid(enc, prompt_index=prompt_index)
        state = rnnt.DecodeState(
            h=torch.zeros(2, 1, 16),
            c=torch.zeros(2, 1, 16),
            last_label=torch.full((1,), core.blank_id),
        )
        labels, _ = rnnt.greedy_decode_batch(
            conditioned, core.predictor, core.joint, state
        )
    return list(labels[0])


def _fresh_pools(num_blocks: int = 3) -> Pools:
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


def _sentinel_pools(num_blocks: int = 3, value: float = 12345.0) -> Pools:
    pools = _fresh_pools(num_blocks)
    for val in pools.values():
        if isinstance(val, list):
            for t in val:
                t.fill_(value)
        else:
            val.fill_(int(value) if not val.dtype.is_floating_point else value)
    return pools


def _clone_pools(pools: Pools) -> Pools:
    return {
        k: [t.clone() for t in v] if isinstance(v, list) else v.clone()
        for k, v in pools.items()
    }


def _assert_pools_equal(a: Pools, b: Pools) -> None:
    for key in a:
        av, bv = a[key], b[key]
        if isinstance(av, list):
            for at, bt in zip(av, bv, strict=True):
                torch.testing.assert_close(at, bt, rtol=0, atol=0)
        else:
            torch.testing.assert_close(av, bv, rtol=0, atol=0)


def _plan(
    decodes: list[int] | None = None,
    prefills: list[int] | None = None,
    *,
    num_pool_blocks: int = 3,
    pad_decodes_to: int | None = None,
    padding_value: int = NULL_INDEX,
    decode_columns: int = 1,
    has_initial: list[bool] | None = None,
    live: list[int] | None = None,
    prompts: list[int] | None = None,
    chunk: list[bool] | None = None,
    geometries: list[int] | None = None,
    generations: list[int] | None = None,
    deadlines: list[int] | None = None,
    request_ids: tuple[str, ...] | None = None,
    prior_prompts: list[int] | None = None,
    allow_prompt_transitions: list[bool] | None = None,
    execution_tier: int = 0,
) -> Any:
    decodes = decodes or []
    prefills = prefills or []
    num_decodes = len(decodes)
    padded = decodes + [padding_value] * ((pad_decodes_to or num_decodes) - num_decodes)
    d = torch.tensor([[i] * decode_columns for i in padded], dtype=torch.long).reshape(
        len(padded), decode_columns
    )
    p = torch.tensor(prefills, dtype=torch.long)
    if has_initial is None:
        has_initial = [False] * len(prefills)
    if live is None:
        live = list(range(1, num_pool_blocks))
    n_real = num_decodes + len(prefills)
    if prompts is None:
        prompts = [0] * n_real  # ADMITTED prompt authority, every row
    if chunk is None:
        chunk = [False] * num_decodes + [True] * len(prefills)
    if geometries is None:
        geometries = [GEOM_REG] * n_real
    if generations is None:
        generations = [0] * n_real
    if deadlines is None:
        deadlines = [i + 1 if chunk[i] else 0 for i in range(n_real)]
    if request_ids is None:
        request_ids = tuple(f"req-{i}" for i in range(n_real))
    if prior_prompts is None:
        prior_prompts = list(prompts)
    if allow_prompt_transitions is None:
        allow_prompt_transitions = [False] * n_real
    blocks = [*decodes, *prefills]
    bindings = tuple(
        advance.PreparedRowBinding(
            request_id=request_ids[row],
            block_id=blocks[row],
            admission_generation=generations[row],
            geometry_id=geometries[row],
            prompt_index=prompts[row],
            prior_prompt_index=prior_prompts[row],
            allow_prompt_transition=allow_prompt_transitions[row],
            is_chunk=chunk[row],
            ready_deadline_ns=deadlines[row],
        )
        for row in range(n_real)
    )
    return advance.RowPlan(
        state_indices_d=d,
        num_decodes=num_decodes,
        state_indices_p=p,
        num_prefills=len(prefills),
        has_initial_states_p=torch.tensor(has_initial, dtype=torch.bool),
        null_block_id=NULL_INDEX,
        num_pool_blocks=num_pool_blocks,
        live_block_ids=torch.tensor(live, dtype=torch.long),
        geometry_id=torch.tensor(geometries, dtype=torch.long),
        prompt_index=torch.tensor(prompts, dtype=torch.long),
        is_chunk=torch.tensor(chunk, dtype=torch.bool),
        admission_generation=torch.tensor(generations, dtype=torch.long),
        ready_deadline_ns=torch.tensor(deadlines, dtype=torch.long),
        request_ids=request_ids,
        execution_tier=execution_tier,
        bindings=bindings,
    )


def _envelope(
    samples: torch.Tensor,
    *,
    final: bool,
    seq: int,
    geometry: int = GEOM_REG,
    prompt: int = 0,
    hidden: int = CARRIER_HIDDEN,
) -> torch.Tensor:
    n = samples.shape[0]
    assert advance.ENVELOPE_HEADER_SLOTS + n <= hidden
    row = torch.zeros(hidden)
    row[advance.ENV_VERSION] = advance.ENVELOPE_VERSION
    row[advance.ENV_VALID_SAMPLES] = n
    row[advance.ENV_GEOMETRY_ID] = geometry
    row[advance.ENV_FINAL_TAIL] = 1.0 if final else 0.0
    row[advance.ENV_PROMPT_INDEX] = prompt
    row[advance.ENV_CHUNK_SEQUENCE] = seq
    row[advance.ENVELOPE_HEADER_SLOTS : advance.ENVELOPE_HEADER_SLOTS + n] = samples
    return row


def _refuse_adapter(*_args: Any) -> Any:
    raise AssertionError("adapter must not be reached in this test")


def _fixed_resolver(decode_fn: Any = None) -> Any:
    fn = decode_fn or rnnt.decode_dense_masked

    def resolve(request: Any) -> Any:
        return advance.ResolvedDecode(arm="dense-eager", decode_fn=fn)

    return resolve


class _RecordingResolver:
    def __init__(self, decode_fn: Any = None) -> None:
        self.requests: list[Any] = []
        self._fn = decode_fn or rnnt.decode_dense_masked

    def __call__(self, request: Any) -> Any:
        self.requests.append(request)
        return advance.ResolvedDecode(arm="dense-eager", decode_fn=self._fn)


class _CommitTicket:
    def __init__(self, sink: _CommitRecorder) -> None:
        self._sink = sink

    def stage(self, row_status: torch.Tensor) -> None:
        candidates = list(self._sink.plans[-1].records)
        self._sink.staged.append(row_status.clone())
        self._sink.records.append(candidates)
        self._sink.published.append(
            [r for r in candidates if int(row_status[r.row]) == 0]
        )
        self._sink.log.append("stage")
        if self._sink.on_stage is not None:
            self._sink.on_stage()

    def cancel(self) -> None:
        self._sink.cancels += 1
        self._sink.log.append("cancel")


class _CommitRecorder:
    def __init__(self, *, fail_reserve: bool = False) -> None:
        self.plans: list[Any] = []
        self.staged: list[torch.Tensor] = []
        self.records: list[list[Any]] = []
        self.published: list[list[Any]] = []
        self.cancels = 0
        self.log: list[str] = []
        self.on_stage: Any = None
        self._fail = fail_reserve

    def reserve(self, plan: Any) -> _CommitTicket:
        self.plans.append(plan)
        self.log.append("reserve")
        if self._fail:
            raise RuntimeError("commit sink at capacity")
        return _CommitTicket(self)


def _call(
    core: Any,
    pools: Pools,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    plan: Any,
    adapter: Any = None,
    *,
    resolver: Any = None,
    commit_sink: Any = None,
    capture: bool = False,
    graph_covers_decode: bool = False,
    memory_profile: bool = False,
    staging: Any = None,
) -> torch.Tensor:
    if adapter is None:
        adapter = advance.make_mrv1_adapter(
            hidden_size=CARRIER_HIDDEN,
            park_id=PARK_ID,
            blank_id=core.blank_id,
        )
    out: torch.Tensor = advance.advance_model_rows(
        core,
        input_ids,
        inputs_embeds,
        plan,
        adapter=adapter,
        decode_resolver=(resolver if resolver is not None else _fixed_resolver()),
        placeholder_id=PLACEHOLDER_ID,
        park_id=PARK_ID,
        commit_sink=commit_sink,
        capture=capture,
        graph_covers_decode=graph_covers_decode,
        memory_profile=memory_profile,
        staging=staging,
        **pools,
    )
    return out


def _decision(out: torch.Tensor) -> list[int]:
    return out[:, 0].long().tolist()


def _set_replay_book(
    pools: Pools,
    block: int,
    *,
    queue: list[int],
    head: int,
    expected: int,
) -> None:
    book = pools["book_pool"]
    qp = pools["queue_pool"]
    for i, label in enumerate(queue):
        qp[block, i] = label
    book[block, _BOOK["queue_head"]] = head
    book[block, _BOOK["queue_length"]] = len(queue)
    book[block, _BOOK["last_label"]] = queue[-1]
    book[block, _BOOK["pending_echo"]] = 1
    book[block, _BOOK["expected_label"]] = expected


def _set_drained_book(pools: Pools, block: int, *, blank: int) -> None:
    book = pools["book_pool"]
    book[block, _BOOK["queue_head"]] = 0
    book[block, _BOOK["queue_length"]] = 0
    book[block, _BOOK["last_label"]] = blank
    book[block, _BOOK["pending_echo"]] = 0
    book[block, _BOOK["expected_label"]] = 0


# ---- PORT-STATE-007: host structural preflight ---------------------------


@pytest.mark.parametrize(
    "name",
    [
        "wrong_row_count",
        "null_among_real",
        "out_of_range",
        "non_live",
        "duplicate_composition",
        "speculative_column",
        "live_padding",
    ],
)
def test_structural_defects_fail_whole_call_before_any_read(
    name: str,
) -> None:
    core = _tiny_core()
    pools = _sentinel_pools(num_blocks=5)
    before = _clone_pools(pools)
    if name == "wrong_row_count":
        plan = _plan(decodes=[1, 2], num_pool_blocks=5)
        ids, embeds = [PARK_ID, PARK_ID], torch.zeros(3, CARRIER_HIDDEN)
    elif name == "null_among_real":
        plan = _plan(decodes=[1, NULL_INDEX], num_pool_blocks=5)
        ids, embeds = [PARK_ID, PARK_ID], torch.zeros(2, CARRIER_HIDDEN)
    elif name == "out_of_range":
        plan = _plan(decodes=[1, 99], num_pool_blocks=5, live=[1, 2, 99])
        ids, embeds = [PARK_ID, PARK_ID], torch.zeros(2, CARRIER_HIDDEN)
    elif name == "non_live":
        plan = _plan(decodes=[2], num_pool_blocks=5, live=[1, 3])
        ids, embeds = [PARK_ID], torch.zeros(1, CARRIER_HIDDEN)
    elif name == "duplicate_composition":
        plan = _plan(decodes=[1], prefills=[1], num_pool_blocks=5)
        ids = [PARK_ID, PLACEHOLDER_ID]
        embeds = torch.zeros(2, CARRIER_HIDDEN)
    elif name == "speculative_column":
        plan = _plan(decodes=[1, 2], num_pool_blocks=5, decode_columns=2)
        ids, embeds = [PARK_ID, PARK_ID], torch.zeros(2, CARRIER_HIDDEN)
    else:  # live_padding
        plan = _plan(
            decodes=[1],
            num_pool_blocks=5,
            pad_decodes_to=2,
            padding_value=2,
        )
        ids, embeds = [PARK_ID], torch.zeros(1, CARRIER_HIDDEN)
    with pytest.raises(ValueError):
        _call(
            core,
            pools,
            torch.tensor(ids, dtype=torch.long),
            embeds,
            plan,
            _refuse_adapter,
        )
    _assert_pools_equal(pools, before)


def test_missing_resolver_fails_before_state_reads() -> None:
    core = _tiny_core()
    pools = _sentinel_pools()
    before = _clone_pools(pools)
    plan = _plan(decodes=[1])
    with pytest.raises(ValueError):
        _call(
            core,
            pools,
            torch.tensor([PARK_ID], dtype=torch.long),
            torch.zeros(1, CARRIER_HIDDEN),
            plan,
            _refuse_adapter,
            resolver=False,
        )
    _assert_pools_equal(pools, before)


# @spec PORT-ADV-003, PORT-STATE-007
@pytest.mark.parametrize(
    ("plan", "input_ids", "hidden"),
    [
        (
            _plan(prefills=[1]),
            [PLACEHOLDER_ID],
            advance.ENVELOPE_HEADER_SLOTS + REG_SAMPLES - 1,
        ),
        (_plan(decodes=[1]), [PARK_ID], 0),
    ],
)
def test_carrier_width_fails_before_resolver_or_resident_read(
    monkeypatch: pytest.MonkeyPatch,
    plan: Any,
    input_ids: list[int],
    hidden: int,
) -> None:
    core = _tiny_core()
    pools = _sentinel_pools()
    before = _clone_pools(pools)
    events: list[str] = []

    def forbid_gather(*_args: Any, **_kwargs: Any) -> Any:
        events.append("read")
        raise AssertionError("resident gather must not run")

    def resolver(_request: Any) -> Any:
        events.append("resolve")
        raise AssertionError("resolver must not run")

    monkeypatch.setattr(advance, "_gather_initialized_rows", forbid_gather)
    with pytest.raises(ValueError, match="carrier width"):
        _call(
            core,
            pools,
            torch.tensor(input_ids),
            torch.zeros(len(input_ids), hidden),
            plan,
            _refuse_adapter,
            resolver=resolver,
        )
    assert events == []
    _assert_pools_equal(pools, before)


# @spec PORT-STATE-007
@pytest.mark.parametrize(
    "field,value",
    [
        ("geometry_id", torch.zeros(1, dtype=torch.int32)),
        ("geometry_id", torch.empty(1, dtype=torch.long, device="meta")),
        ("prompt_index", torch.zeros(1, 1, dtype=torch.long)),
        ("is_chunk", torch.ones(1, dtype=torch.long)),
        ("admission_generation", torch.tensor([-1], dtype=torch.long)),
        ("ready_deadline_ns", torch.tensor([0], dtype=torch.long)),
    ],
)
def test_row_plan_authority_fails_closed_before_reads(
    field: str, value: torch.Tensor
) -> None:
    core = _tiny_core()
    pools = _sentinel_pools()
    before = _clone_pools(pools)
    plan = replace(_plan(prefills=[1]), **{field: value})
    with pytest.raises(ValueError):
        _call(
            core,
            pools,
            torch.tensor([PLACEHOLDER_ID]),
            torch.zeros(1, CARRIER_HIDDEN),
            plan,
            _refuse_adapter,
        )
    _assert_pools_equal(pools, before)


# @spec PORT-STATE-007, PORT-HOOK-001
@pytest.mark.parametrize("request_ids", [("",), ("same", "same")])
def test_request_identity_is_nonempty_and_unique(request_ids: tuple[str, ...]) -> None:
    rows = len(request_ids)
    plan = _plan(
        prefills=list(range(1, rows + 1)),
        num_pool_blocks=rows + 2,
        request_ids=request_ids,
    )
    pools = _sentinel_pools(num_blocks=rows + 2)
    before = _clone_pools(pools)
    with pytest.raises(ValueError):
        _call(
            _tiny_core(),
            pools,
            torch.full((rows,), PLACEHOLDER_ID),
            torch.zeros(rows, CARRIER_HIDDEN),
            plan,
            _refuse_adapter,
        )
    _assert_pools_equal(pools, before)


# @spec PORT-STATE-007, PORT-HOOK-001
@pytest.mark.parametrize("drift", ["request_swap", "generation_aba", "bool_block"])
def test_atomic_row_binding_rejects_identity_or_control_drift(
    drift: str,
) -> None:
    plan = _plan(decodes=[1, 2], num_pool_blocks=4)
    if drift == "request_swap":
        plan = replace(plan, request_ids=("req-1", "req-0"))
    elif drift == "generation_aba":
        plan = replace(
            plan,
            admission_generation=torch.tensor([1, 0], dtype=torch.long),
        )
    else:
        plan = replace(
            plan,
            bindings=(
                replace(plan.bindings[0], block_id=True),
                plan.bindings[1],
            ),
        )
    pools = _sentinel_pools(num_blocks=4)
    before = _clone_pools(pools)
    with pytest.raises(ValueError, match="binding"):
        _call(
            _tiny_core(),
            pools,
            torch.full((2,), PARK_ID),
            torch.zeros(2, CARRIER_HIDDEN),
            plan,
            _refuse_adapter,
        )
    _assert_pools_equal(pools, before)


# @spec PORT-LID-003, PORT-STATE-007
def test_fresh_binding_cannot_authorize_prompt_transition() -> None:
    plan = _plan(
        prefills=[1],
        prompts=[1],
        prior_prompts=[0],
        allow_prompt_transitions=[True],
    )
    pools = _sentinel_pools()
    before = _clone_pools(pools)
    with pytest.raises(ValueError, match="fresh admission"):
        _call(
            _tiny_core(),
            pools,
            torch.tensor([PLACEHOLDER_ID]),
            torch.zeros(1, CARRIER_HIDDEN),
            plan,
            _refuse_adapter,
        )
    _assert_pools_equal(pools, before)


# @spec PORT-ADV-003, PORT-DEC-008, PORT-PERF-004, PORT-STATE-007
def test_regional_graph_resolves_before_resident_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _tiny_core()
    pools = _sentinel_pools()
    before = _clone_pools(pools)
    events: list[str] = []
    original = torch.Tensor.index_select
    resident_tensors = {
        id(tensor)
        for value in pools.values()
        for tensor in (value if isinstance(value, list) else [value])
        if isinstance(tensor, torch.Tensor)
    }

    def observed(tensor: torch.Tensor, dim: int, index: torch.Tensor) -> Any:
        if id(tensor) in resident_tensors:
            events.append("read")
        return original(tensor, dim, index)

    monkeypatch.setattr(torch.Tensor, "index_select", observed)

    def resolver(request: Any) -> Any:
        events.append("resolve")
        assert request.graph_covers_decode is True
        assert request.execution_batch_size == 1
        raise RuntimeError("graph resolved")

    with pytest.raises(RuntimeError, match="graph resolved"):
        _call(
            core,
            pools,
            torch.tensor([PLACEHOLDER_ID]),
            torch.zeros(1, CARRIER_HIDDEN),
            _plan(prefills=[1], execution_tier=1),
            _refuse_adapter,
            resolver=resolver,
            graph_covers_decode=True,
        )
    assert events == ["resolve"]
    _assert_pools_equal(pools, before)


# @spec PORT-PERF-004
def test_memory_profile_purpose_reaches_resolver_before_resident_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _tiny_core()
    pools = _sentinel_pools()
    before = _clone_pools(pools)
    events: list[str] = []
    original = torch.Tensor.index_select
    resident_tensors = {
        id(tensor)
        for value in pools.values()
        for tensor in (value if isinstance(value, list) else [value])
        if isinstance(tensor, torch.Tensor)
    }

    def observed(tensor: torch.Tensor, dim: int, index: torch.Tensor) -> Any:
        if id(tensor) in resident_tensors:
            events.append("read")
        return original(tensor, dim, index)

    monkeypatch.setattr(torch.Tensor, "index_select", observed)

    from vllm.models.nemotron_asr.nemotron_asr import (
        build_decode_resolver,
    )

    class Binding:
        def execution_tier(self, live_rows: int) -> int:
            raise AssertionError("profile must not consult captured tiers")

        def decode_fn(self, *, geometry: int, tier: int) -> object:
            raise AssertionError("profile must not bind a graph key")

    canonical = build_decode_resolver(
        SimpleNamespace(
            decode_dispatch_arm="dense-graphed",
            decode_dispatch_table=None,
        ),
        graph_binding=Binding(),
    )

    def resolver(request: Any) -> Any:
        events.append("resolve")
        assert request.memory_profile is True
        assert request.graph_covers_decode is False
        assert request.execution_tier_limit == 0
        resolved = canonical(request)
        assert resolved.arm == "dense-eager"
        assert resolved.override_reason == "pre-capture-memory-profile"
        raise RuntimeError("profile resolved")

    with pytest.raises(RuntimeError, match="profile resolved"):
        _call(
            core,
            pools,
            torch.tensor([PLACEHOLDER_ID]),
            torch.zeros(1, CARRIER_HIDDEN),
            _plan(prefills=[1]),
            _refuse_adapter,
            resolver=resolver,
            memory_profile=True,
        )
    assert events == ["resolve"]
    _assert_pools_equal(pools, before)


def test_legal_graph_padding_replays_normally() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    _set_replay_book(pools, 1, queue=[3, 5], head=1, expected=3)
    plan = _plan(decodes=[1], pad_decodes_to=2, padding_value=NULL_INDEX)
    out = _call(
        core,
        pools,
        torch.tensor([3], dtype=torch.long),
        torch.zeros(1, CARRIER_HIDDEN),
        plan,
    )
    assert out.shape == (1, CARRIER_HIDDEN)
    assert _decision(out) == [5]
    assert int(pools["book_pool"][1, _BOOK["queue_head"]]) == 2
    assert int(pools["book_pool"][1, _BOOK["expected_label"]]) == 5


# ---- fresh init + the echo row tier --------------------------------------


def test_fresh_rows_ignore_recycled_poison() -> None:
    core = _tiny_core()
    torch.manual_seed(5)
    samples = torch.randn(FINAL_SAMPLES) * 0.01
    carrier = _envelope(samples, final=True, seq=0, geometry=GEOM_FINAL).unsqueeze(0)
    ids = torch.tensor([PLACEHOLDER_ID], dtype=torch.long)
    plan = _plan(prefills=[1], geometries=[GEOM_FINAL])
    clean = _fresh_pools()
    out_clean = _call(core, clean, ids, carrier, plan)
    dirty = _sentinel_pools(value=3.14)
    out_dirty = _call(core, dirty, ids, carrier, plan)
    assert _decision(out_dirty) == _decision(out_clean)
    assert _decision(out_clean)[0] != PARK_ID  # a real burst ran


def test_corrupt_echo_is_row_local_and_reported() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    _set_replay_book(pools, 1, queue=[3, 5], head=1, expected=3)
    _set_replay_book(pools, 2, queue=[4, 6], head=1, expected=4)
    before = _clone_pools(pools)
    plan = _plan(decodes=[1, 2])
    status = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([3, 8], dtype=torch.long),
        torch.zeros(2, CARRIER_HIDDEN),
        plan,
        commit_sink=status,
    )
    assert _decision(out) == [5, PARK_ID]
    # The healthy peer advanced...
    assert int(pools["book_pool"][1, _BOOK["queue_head"]]) == 2
    # ...the corrupted row is bit-identical.
    torch.testing.assert_close(
        pools["book_pool"][2], before["book_pool"][2], rtol=0, atol=0
    )
    torch.testing.assert_close(
        pools["queue_pool"][2], before["queue_pool"][2], rtol=0, atol=0
    )
    assert len(status.staged) == 1
    assert int(status.staged[0][0]) == 0
    assert int(status.staged[0][1]) & advance.ROW_STATUS_ECHO_MISMATCH


def test_absurd_echo_id_is_safe_and_masked() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    _set_replay_book(pools, 1, queue=[3, 5], head=1, expected=3)
    before = _clone_pools(pools)
    plan = _plan(decodes=[1])
    out = _call(
        core,
        pools,
        torch.tensor([10**6], dtype=torch.long),
        torch.zeros(1, CARRIER_HIDDEN),
        plan,
    )
    assert _decision(out) == [PARK_ID]
    _assert_pools_equal(pools, before)


def test_chunk_on_undrained_queue_masks_and_reports() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    _set_replay_book(pools, 1, queue=[3, 5], head=1, expected=3)
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0).unsqueeze(
        0
    )
    plan = _plan(decodes=[1], chunk=[True])
    status = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
        carrier,
        plan,
        commit_sink=status,
    )
    assert _decision(out) == [PARK_ID]
    _assert_pools_equal(pools, before)
    assert int(status.staged[0][0]) & advance.ROW_STATUS_QUEUE_NOT_DRAINED


def test_flush_and_async_park_echo_are_clean_noops() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    # Row 1 is the explicit post-final-tail FLUSH; row 2 is the distinct
    # async-scheduler echo of an earlier park on a live session.
    _set_drained_book(pools, 1, blank=core.blank_id)
    pools["frontend_counter_pool"][1, _CTR["finalized"]] = 1
    pools["frontend_counter_pool"][1, _CTR["expected_chunk_sequence"]] = 1
    _set_drained_book(pools, 2, blank=core.blank_id)  # NOT finalized
    before = _clone_pools(pools)
    plan = _plan(decodes=[1, 2])
    status = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([PARK_ID, PARK_ID], dtype=torch.long),
        torch.zeros(2, CARRIER_HIDDEN),
        plan,
        commit_sink=status,
    )
    assert _decision(out) == [PARK_ID, PARK_ID]
    assert int(status.staged[0][0]) == 0
    assert int(status.staged[0][1]) == 0
    _assert_pools_equal(pools, before)


# ---- the full MRV1 arc ----------------------------------------------------


def test_burst_then_drain_then_park_matches_reference() -> None:
    core = _tiny_core(seed=1)  # two-distinct-label burst by design
    torch.manual_seed(5)
    samples = torch.randn(FINAL_SAMPLES) * 0.01
    expected = _reference_burst(core, samples, prompt_index=0)
    assert len(expected) >= 2 and len(set(expected)) >= 2

    pools = _fresh_pools()
    plan = _plan(prefills=[1], geometries=[GEOM_FINAL])
    carrier = _envelope(samples, final=True, seq=0, geometry=GEOM_FINAL).unsqueeze(0)
    out = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
        carrier,
        plan,
    )
    emitted = [_decision(out)[0]]
    book = pools["book_pool"]
    assert int(book[1, _BOOK["queue_length"]]) == len(expected)
    assert int(book[1, _BOOK["queue_head"]]) == 1
    assert int(book[1, _BOOK["pending_echo"]]) == 1
    assert int(book[1, _BOOK["expected_label"]]) == emitted[0]
    assert int(pools["frontend_counter_pool"][1, _CTR["finalized"]]) == 1

    decode_plan = _plan(decodes=[1], geometries=[GEOM_FINAL])
    for _ in range(len(expected) + 1):
        if emitted[-1] == PARK_ID:
            break
        out = _call(
            core,
            pools,
            torch.tensor([emitted[-1]], dtype=torch.long),
            torch.zeros(1, CARRIER_HIDDEN),
            decode_plan,
        )
        emitted.append(_decision(out)[0])

    assert emitted[:-1] == expected
    assert emitted[-1] == PARK_ID
    assert int(book[1, _BOOK["pending_echo"]]) == 0

    # An INVALID CONTINUING chunk (wrong sequence) on the now-parked
    # session leaves every page of its block bit-identical.
    parked = _clone_pools(pools)
    wrong_seq = _envelope(samples, final=True, seq=5, geometry=GEOM_FINAL).unsqueeze(0)
    status = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
        wrong_seq,
        _plan(decodes=[1], chunk=[True], geometries=[GEOM_FINAL]),
        commit_sink=status,
    )
    assert _decision(out) == [PARK_ID]
    _assert_pools_equal(pools, parked)
    assert int(status.staged[0][0]) != 0


def test_mixed_geometries_bucket_resolver_and_row_order() -> None:
    # Two CHUNK rows at different geometries plus one replay row in a
    # single call: per-geometry buckets, one resolver query per bucket
    # with ready_decode_buckets == 2, and the returned rows stay
    # row-aligned with the call order.
    core = _tiny_core()
    pools = _fresh_pools(num_blocks=5)
    _set_replay_book(pools, 3, queue=[7, 9], head=1, expected=7)
    resolver = _RecordingResolver()
    plan = _plan(
        decodes=[3],
        prefills=[1, 2],
        num_pool_blocks=5,
        geometries=[GEOM_REG, GEOM_REG, GEOM_FINAL],
    )
    torch.manual_seed(5)
    reg = _envelope(
        torch.randn(REG_SAMPLES) * 0.01,
        final=False,
        seq=0,
        geometry=GEOM_REG,
    )
    fin = _envelope(
        torch.randn(FINAL_SAMPLES) * 0.01,
        final=True,
        seq=0,
        geometry=GEOM_FINAL,
    )
    input_ids = torch.tensor([7, PLACEHOLDER_ID, PLACEHOLDER_ID], dtype=torch.long)
    embeds = torch.stack([torch.zeros(CARRIER_HIDDEN), reg, fin])
    out = _call(core, pools, input_ids, embeds, plan, resolver=resolver)
    assert len(resolver.requests) == 2
    assert {r.geometry for r in resolver.requests} == {
        GEOM_REG,
        GEOM_FINAL,
    }
    assert all(r.ready_decode_buckets == 2 for r in resolver.requests)
    assert all(r.execution_batch_size == 1 for r in resolver.requests)
    decisions = _decision(out)
    assert decisions[0] == 9  # the replay row advanced its queue
    assert decisions[2] != PARK_ID  # the final chunk burst emitted
    assert int(pools["book_pool"][3, _BOOK["queue_head"]]) == 2


# @spec PORT-ADV-003, PORT-PERF-001
def test_deadline_orders_all_resolutions_before_resident_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _tiny_core()
    pools = _fresh_pools(num_blocks=4)
    events: list[str] = []
    original = torch.Tensor.index_select
    pool_ptrs = {
        tensor.data_ptr()
        for value in pools.values()
        for tensor in (value if isinstance(value, list) else [value])
    }

    def observed(tensor: torch.Tensor, dim: int, index: torch.Tensor) -> Any:
        if tensor.data_ptr() in pool_ptrs:
            events.append("read")
        return original(tensor, dim, index)

    monkeypatch.setattr(torch.Tensor, "index_select", observed)

    class Resolver:
        def __call__(self, request: Any) -> Any:
            events.append(f"resolve-{request.geometry}")
            return advance.ResolvedDecode("dense-eager", rnnt.decode_dense_masked)

    torch.manual_seed(5)
    reg = _envelope(
        torch.randn(REG_SAMPLES) * 0.01,
        final=False,
        seq=0,
        geometry=GEOM_REG,
    )
    fin = _envelope(
        torch.randn(FINAL_SAMPLES) * 0.01,
        final=True,
        seq=0,
        geometry=GEOM_FINAL,
    )
    _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID, PLACEHOLDER_ID]),
        torch.stack([reg, fin]),
        _plan(
            prefills=[1, 2],
            num_pool_blocks=4,
            geometries=[GEOM_REG, GEOM_FINAL],
            deadlines=[200, 100],
        ),
        resolver=Resolver(),
    )
    assert events[:2] == [f"resolve-{GEOM_FINAL}", f"resolve-{GEOM_REG}"]
    assert "read" not in events[:2]


# @spec PORT-ADV-003, PORT-DEC-008
def test_later_resolver_failure_precedes_every_resident_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _tiny_core()
    pools = _sentinel_pools(num_blocks=4)
    before = _clone_pools(pools)
    reads = {"count": 0}
    original = torch.Tensor.index_select
    pool_ptrs = {
        tensor.data_ptr()
        for value in pools.values()
        for tensor in (value if isinstance(value, list) else [value])
    }

    def observed(tensor: torch.Tensor, dim: int, index: torch.Tensor) -> Any:
        if tensor.data_ptr() in pool_ptrs:
            reads["count"] += 1
        return original(tensor, dim, index)

    monkeypatch.setattr(torch.Tensor, "index_select", observed)
    calls = {"count": 0}

    def resolver(_request: Any) -> Any:
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("second resolver failed")
        return advance.ResolvedDecode("dense-eager", rnnt.decode_dense_masked)

    plan = _plan(
        prefills=[1, 2],
        num_pool_blocks=4,
        geometries=[GEOM_REG, GEOM_FINAL],
        deadlines=[200, 100],
    )
    with pytest.raises(RuntimeError, match="second resolver"):
        _call(
            core,
            pools,
            torch.tensor([PLACEHOLDER_ID, PLACEHOLDER_ID]),
            torch.zeros(2, CARRIER_HIDDEN),
            plan,
            _refuse_adapter,
            resolver=resolver,
        )
    assert calls["count"] == 2
    assert reads["count"] == 0
    _assert_pools_equal(pools, before)


# @spec PORT-STATE-003
def test_fresh_rows_are_absent_from_every_resident_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _tiny_core()
    pools = _sentinel_pools(num_blocks=4, value=3.14)
    fresh_block = 2
    read_blocks: list[int] = []
    original = torch.Tensor.index_select
    pool_ptrs = {
        tensor.data_ptr()
        for value in pools.values()
        for tensor in (value if isinstance(value, list) else [value])
    }

    def observed(tensor: torch.Tensor, dim: int, index: torch.Tensor) -> Any:
        if tensor.data_ptr() in pool_ptrs and dim == 0:
            read_blocks.extend(int(x) for x in index.tolist())
        return original(tensor, dim, index)

    monkeypatch.setattr(torch.Tensor, "index_select", observed)
    torch.manual_seed(5)
    carrier = _envelope(torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0).unsqueeze(
        0
    )
    _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID]),
        carrier,
        _plan(prefills=[fresh_block], num_pool_blocks=4),
    )
    assert fresh_block not in read_blocks


def test_bucket_compute_failure_leaves_pools_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("simulated mid-bucket compute failure")

    monkeypatch.setattr(advance, "advance_session", _boom)
    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0).unsqueeze(
        0
    )
    plan = _plan(prefills=[1])
    with pytest.raises(RuntimeError):
        _call(
            core,
            pools,
            torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
            carrier,
            plan,
        )
    _assert_pools_equal(pools, before)


# ---- PORT-HOOK-001: the reservation-owned capture sink -------------------


def test_capture_reservation_failure_is_pre_commit_fatal() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(
        torch.randn(FINAL_SAMPLES) * 0.01,
        final=True,
        seq=0,
        geometry=GEOM_FINAL,
    ).unsqueeze(0)
    plan = _plan(prefills=[1], geometries=[GEOM_FINAL])
    sink = _CommitRecorder(fail_reserve=True)
    with pytest.raises(RuntimeError):
        _call(
            core,
            pools,
            torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
            carrier,
            plan,
            commit_sink=sink,
            capture=True,
        )
    _assert_pools_equal(pools, before)
    assert sink.published == []


# @spec PORT-HOOK-001, PORT-STATE-008
@pytest.mark.parametrize("ticket", [None, object()])
def test_malformed_commit_ticket_is_precommit_fatal(ticket: object) -> None:
    class BadSink:
        def reserve(self, plan: Any) -> object:
            del plan
            return ticket

    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0).unsqueeze(
        0
    )
    with pytest.raises(ValueError, match="reservation"):
        _call(
            core,
            pools,
            torch.tensor([PLACEHOLDER_ID]),
            carrier,
            _plan(prefills=[1]),
            commit_sink=BadSink(),
        )
    _assert_pools_equal(pools, before)


# @spec PORT-HOOK-001, PORT-STATE-008
def test_ticket_without_stage_is_cancelled_precommit() -> None:
    class CancelOnly:
        def __init__(self) -> None:
            self.cancels = 0

        def cancel(self) -> None:
            self.cancels += 1

    ticket = CancelOnly()

    class BadSink:
        def reserve(self, plan: Any) -> CancelOnly:
            del plan
            return ticket

    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0).unsqueeze(
        0
    )
    with pytest.raises(ValueError, match="without stage"):
        _call(
            core,
            pools,
            torch.tensor([PLACEHOLDER_ID]),
            carrier,
            _plan(prefills=[1]),
            commit_sink=BadSink(),
        )
    assert ticket.cancels == 1
    _assert_pools_equal(pools, before)


# @spec PORT-ADV-003, PORT-HOOK-001, PORT-STATE-008
def test_ticket_methods_are_bound_exactly_once_before_commit() -> None:
    class DynamicTicket:
        def __init__(self) -> None:
            self.stage_lookups = 0
            self.cancel_lookups = 0
            self.stages = 0

        @property
        def stage(self) -> Any:
            self.stage_lookups += 1

            def bound(_status: torch.Tensor) -> None:
                self.stages += 1

            return bound

        @property
        def cancel(self) -> Any:
            self.cancel_lookups += 1

            def bound() -> None:
                return None

            return bound

    ticket = DynamicTicket()

    class Sink:
        def reserve(self, plan: Any) -> DynamicTicket:
            del plan
            return ticket

    core = _tiny_core()
    pools = _fresh_pools()
    torch.manual_seed(5)
    carrier = _envelope(torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0).unsqueeze(
        0
    )
    _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID]),
        carrier,
        _plan(prefills=[1]),
        commit_sink=Sink(),
    )
    assert ticket.cancel_lookups == 1
    assert ticket.stage_lookups == 1
    assert ticket.stages == 1


# @spec PORT-ADV-003, PORT-HOOK-001, PORT-STATE-008
def test_ticket_stage_binding_failure_cancels_reservation() -> None:
    class RaisingStageTicket:
        def __init__(self) -> None:
            self.cancel_lookups = 0
            self.stage_lookups = 0
            self.cancels = 0

        @property
        def cancel(self) -> Any:
            self.cancel_lookups += 1

            def bound() -> None:
                self.cancels += 1

            return bound

        @property
        def stage(self) -> Any:
            self.stage_lookups += 1
            raise RuntimeError("stage binding failed")

    ticket = RaisingStageTicket()

    class Sink:
        def reserve(self, plan: Any) -> RaisingStageTicket:
            del plan
            return ticket

    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0).unsqueeze(
        0
    )
    with pytest.raises(RuntimeError, match="stage binding failed"):
        _call(
            core,
            pools,
            torch.tensor([PLACEHOLDER_ID]),
            carrier,
            _plan(prefills=[1]),
            commit_sink=Sink(),
        )
    assert ticket.cancel_lookups == 1
    assert ticket.stage_lookups == 1
    assert ticket.cancels == 1
    _assert_pools_equal(pools, before)


# @spec PORT-HOOK-001, PORT-STATE-008
def test_capture_shape_mismatch_is_precommit_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = advance.advance_session

    def malformed(*args: Any, **kwargs: Any) -> Any:
        result = real(*args, **kwargs)
        assert result.captures is not None
        captures = replace(
            result.captures,
            frontend_mel=result.captures.frontend_mel[:, :, :-1],
        )
        return replace(result, captures=captures)

    monkeypatch.setattr(advance, "advance_session", malformed)
    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0).unsqueeze(
        0
    )
    sink = _CommitRecorder()
    with pytest.raises(ValueError, match="frontend capture"):
        _call(
            core,
            pools,
            torch.tensor([PLACEHOLDER_ID]),
            carrier,
            _plan(prefills=[1]),
            commit_sink=sink,
            capture=True,
        )
    assert sink.plans == []
    _assert_pools_equal(pools, before)


# @spec PORT-ADV-004, PORT-HOOK-001, PORT-STATE-008
def test_capture_length_predicate_is_row_tier_without_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = advance.advance_session

    def malformed(*args: Any, **kwargs: Any) -> Any:
        result = real(*args, **kwargs)
        assert result.captures is not None
        captures = replace(
            result.captures,
            mel_lengths=torch.full_like(
                result.captures.mel_lengths,
                result.captures.frontend_mel.shape[2] + 1,
            ),
        )
        return replace(result, captures=captures)

    monkeypatch.setattr(advance, "advance_session", malformed)
    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0).unsqueeze(
        0
    )
    sink = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID]),
        carrier,
        _plan(prefills=[1]),
        commit_sink=sink,
        capture=True,
    )
    assert _decision(out) == [PARK_ID]
    assert int(sink.staged[0][0]) & advance.ROW_STATUS_DECODE_INVARIANT
    assert sink.published == [[]]
    _assert_pools_equal(pools, before)


# @spec PORT-ADV-004, PORT-HOOK-001, PORT-STATE-008
def test_capture_in_range_wrong_length_is_row_tier_without_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = advance.advance_session

    def malformed(*args: Any, **kwargs: Any) -> Any:
        result = real(*args, **kwargs)
        assert result.captures is not None
        captures = replace(
            result.captures,
            mel_lengths=torch.zeros_like(result.captures.mel_lengths),
            encoder_lengths=torch.zeros_like(result.captures.encoder_lengths),
        )
        return replace(result, captures=captures)

    monkeypatch.setattr(advance, "advance_session", malformed)
    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0).unsqueeze(
        0
    )
    sink = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID]),
        carrier,
        _plan(prefills=[1]),
        commit_sink=sink,
        capture=True,
    )
    assert _decision(out) == [PARK_ID]
    assert int(sink.staged[0][0]) & advance.ROW_STATUS_DECODE_INVARIANT
    assert sink.published == [[]]
    _assert_pools_equal(pools, before)


def test_capture_validation_reuses_geometry_shape_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-010
    encoder_execution = mods["encoder_execution"]
    geometry_shape = getattr(
        encoder_execution,
        "encoder_geometry_shape",
        None,
    )
    assert callable(geometry_shape), "PORT-PERF-010 missing encoder_geometry_shape"
    observed: list[tuple[int, int, int]] = []

    def observe_shape(*args: Any, **kwargs: Any) -> Any:
        result = geometry_shape(*args, **kwargs)
        observed.append(
            (
                result.cadence_frames,
                result.mel_width,
                result.out_width,
            )
        )
        return result

    monkeypatch.setattr(
        encoder_execution,
        "encoder_geometry_shape",
        observe_shape,
    )
    core = _tiny_core()
    pools = _fresh_pools()
    carrier = _envelope(
        torch.randn(FINAL_SAMPLES) * 0.01,
        final=True,
        seq=0,
        geometry=GEOM_FINAL,
    ).unsqueeze(0)
    _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
        carrier,
        _plan(prefills=[1], geometries=[GEOM_FINAL]),
        commit_sink=_CommitRecorder(),
        capture=True,
    )

    assert len(observed) == 2
    assert observed[0] == observed[1]


def test_capture_records_publish_after_commit_with_identity() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    torch.manual_seed(5)
    carrier = _envelope(
        torch.randn(FINAL_SAMPLES) * 0.01,
        final=True,
        seq=0,
        geometry=GEOM_FINAL,
    ).unsqueeze(0)
    plan = _plan(prefills=[1], geometries=[GEOM_FINAL], generations=[41])
    sink = _CommitRecorder()
    _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
        carrier,
        plan,
        commit_sink=sink,
        capture=True,
    )
    assert len(sink.plans) == 1
    assert sink.plans[0].capture.rows == 1
    assert sink.plans[0].capture.payload_bytes > 0
    [records] = sink.published
    [record] = records
    assert record.row == 0
    assert record.block_id == 1
    assert record.admission_generation == 41
    assert record.geometry == GEOM_FINAL
    assert int(record.row_status) == 0
    assert int(record.mel_length) == FINAL_SAMPLES // 160
    assert record.frontend_mel.shape[0] == FEAT
    assert record.encoder_raw.shape == record.encoder_conditioned.shape
    assert sink.cancels == 0


# @spec PORT-HOOK-001, PORT-STATE-008
def test_composite_stage_exposes_only_status_clean_capture_records() -> None:
    core = _tiny_core()
    pools = _fresh_pools(num_blocks=4)
    torch.manual_seed(5)
    samples = torch.randn(FINAL_SAMPLES) * 0.01
    rows = torch.stack(
        [
            _envelope(
                samples,
                final=True,
                seq=0,
                geometry=GEOM_FINAL,
                prompt=0,
            ),
            _envelope(
                samples,
                final=True,
                seq=0,
                geometry=GEOM_FINAL,
                prompt=2,
            ),
        ]
    )
    sink = _CommitRecorder()
    _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID, PLACEHOLDER_ID]),
        rows,
        _plan(
            prefills=[1, 2],
            num_pool_blocks=4,
            geometries=[GEOM_FINAL, GEOM_FINAL],
            prompts=[0, 1],
        ),
        commit_sink=sink,
        capture=True,
    )
    assert len(sink.records[0]) == 2
    assert [record.row for record in sink.published[0]] == [0]
    assert int(sink.staged[0][1]) & advance.ROW_STATUS_PROMPT_MISMATCH


def test_adapter_failure_precedes_reservation_entirely() -> None:
    # The corrected state machine (PORT-ADV-003 as amended) runs the
    # adapter and every other fallible operation BEFORE reservation:
    # an adapter failure therefore never creates a reservation to
    # leak — nothing reserved, nothing published, pools untouched.
    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(
        torch.randn(FINAL_SAMPLES) * 0.01,
        final=True,
        seq=0,
        geometry=GEOM_FINAL,
    ).unsqueeze(0)
    plan = _plan(prefills=[1], geometries=[GEOM_FINAL])
    sink = _CommitRecorder()

    def _bad_adapter(*_args: Any) -> Any:
        raise RuntimeError("adapter blew up")

    with pytest.raises(RuntimeError):
        _call(
            core,
            pools,
            torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
            carrier,
            plan,
            _bad_adapter,
            commit_sink=sink,
            capture=True,
        )
    _assert_pools_equal(pools, before)
    assert sink.plans == []
    assert sink.cancels == 0
    assert sink.published == []


def test_no_capture_sink_stages_nothing() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    torch.manual_seed(5)
    carrier = _envelope(
        torch.randn(FINAL_SAMPLES) * 0.01,
        final=True,
        seq=0,
        geometry=GEOM_FINAL,
    ).unsqueeze(0)
    plan = _plan(prefills=[1], geometries=[GEOM_FINAL])
    out = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
        carrier,
        plan,
    )
    assert out.shape == (1, CARRIER_HIDDEN)


# ---- envelope validation --------------------------------------------------


def test_bad_envelope_on_a_fresh_row_scatters_nothing() -> None:
    # PORT-ADV-004 as amended: a failed row — fresh initialization
    # included — leaves EVERY resident page bit-identical. Sentinel
    # pools prove the fresh init did not leak.
    core = _tiny_core()
    pools = _sentinel_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0)
    carrier[advance.ENV_VERSION] = 99.0
    plan = _plan(prefills=[1])
    status = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
        carrier.unsqueeze(0),
        plan,
        commit_sink=status,
    )
    assert _decision(out) == [PARK_ID]
    assert int(status.staged[0][0]) & advance.ROW_STATUS_ENVELOPE
    _assert_pools_equal(pools, before)


def test_regular_chunk_with_wrong_cadence_length_masks() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    torch.manual_seed(5)
    carrier = _envelope(torch.randn(REG_SAMPLES - 160) * 0.01, final=False, seq=0)
    plan = _plan(prefills=[1])
    status = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
        carrier.unsqueeze(0),
        plan,
        commit_sink=status,
    )
    assert _decision(out) == [PARK_ID]
    assert int(status.staged[0][0]) & advance.ROW_STATUS_ENVELOPE


def test_wrong_but_valid_prompt_masks_and_admission_conditions() -> None:
    # PORT-LID-003 as amended: the envelope prompt is only a device
    # cross-check. Row 0's carrier agrees with its admitted prompt (2)
    # and computes; row 1's carrier stamps a DIFFERENT valid prompt
    # (2) against admitted 1 — masked with PROMPT_MISMATCH, never a
    # silently accepted substitute.
    core = _tiny_core()  # num_prompts == 4
    pools = _fresh_pools(num_blocks=4)
    torch.manual_seed(5)
    samples = torch.randn(FINAL_SAMPLES) * 0.01
    ok_row = _envelope(samples, final=True, seq=0, geometry=GEOM_FINAL, prompt=2)
    bad_row = _envelope(samples, final=True, seq=0, geometry=GEOM_FINAL, prompt=2)
    plan = _plan(
        prefills=[1, 2],
        num_pool_blocks=4,
        geometries=[GEOM_FINAL, GEOM_FINAL],
        prompts=[2, 1],  # admitted authority per row
    )
    status = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID, PLACEHOLDER_ID], dtype=torch.long),
        torch.stack([ok_row, bad_row]),
        plan,
        commit_sink=status,
    )
    decisions = _decision(out)
    assert decisions[0] != PARK_ID  # admitted==stamped: computed
    assert decisions[1] == PARK_ID  # valid-but-different: masked
    assert int(status.staged[0][0]) == 0
    assert int(status.staged[0][1]) & advance.ROW_STATUS_PROMPT_MISMATCH


def test_out_of_dictionary_carrier_prompt_masks() -> None:
    # A carrier prompt outside the dictionary disagrees with the
    # (host-validated) admitted authority by construction.
    core = _tiny_core()
    pools = _fresh_pools()
    torch.manual_seed(5)
    carrier = _envelope(torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0, prompt=7)
    plan = _plan(prefills=[1])
    status = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
        carrier.unsqueeze(0),
        plan,
        commit_sink=status,
    )
    assert _decision(out) == [PARK_ID]
    assert int(status.staged[0][0]) & advance.ROW_STATUS_PROMPT_MISMATCH


# ---- correction-pass hardening pins ---------------------------------------


def test_second_bucket_failure_leaves_pools_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A failure in bucket 2 AFTER bucket 1 computed is still a
    # whole-call failure before any scatter: scratch-first means the
    # first bucket's results never reached the pools.
    real = advance.advance_session
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("second bucket exploded")
        return real(*args, **kwargs)

    monkeypatch.setattr(advance, "advance_session", flaky)
    core = _tiny_core()
    pools = _fresh_pools(num_blocks=4)
    before = _clone_pools(pools)
    torch.manual_seed(5)
    reg = _envelope(
        torch.randn(REG_SAMPLES) * 0.01,
        final=False,
        seq=0,
        geometry=GEOM_REG,
    )
    fin = _envelope(
        torch.randn(FINAL_SAMPLES) * 0.01,
        final=True,
        seq=0,
        geometry=GEOM_FINAL,
    )
    plan = _plan(
        prefills=[1, 2],
        num_pool_blocks=4,
        geometries=[GEOM_REG, GEOM_FINAL],
    )
    with pytest.raises(RuntimeError):
        _call(
            core,
            pools,
            torch.tensor([PLACEHOLDER_ID, PLACEHOLDER_ID], dtype=torch.long),
            torch.stack([reg, fin]),
            plan,
        )
    assert calls["n"] == 2
    _assert_pools_equal(pools, before)


def test_composite_reservation_failure_is_pre_commit_fatal() -> None:
    # Status and capture capacity are admitted by one atomic reserve;
    # a failure creates no ticket and leaves every resident page exact.
    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(
        torch.randn(FINAL_SAMPLES) * 0.01,
        final=True,
        seq=0,
        geometry=GEOM_FINAL,
    ).unsqueeze(0)
    plan = _plan(prefills=[1], geometries=[GEOM_FINAL])
    sink = _CommitRecorder(fail_reserve=True)
    with pytest.raises(RuntimeError):
        _call(
            core,
            pools,
            torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
            carrier,
            plan,
            commit_sink=sink,
            capture=True,
        )
    _assert_pools_equal(pools, before)
    assert sink.cancels == 0
    assert sink.published == []


def test_status_ticket_is_request_aware_and_staged_once() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    _set_replay_book(pools, 1, queue=[3, 5], head=1, expected=3)
    plan = _plan(decodes=[1], generations=[9])
    status = _CommitRecorder()
    _call(
        core,
        pools,
        torch.tensor([3], dtype=torch.long),
        torch.zeros(1, CARRIER_HIDDEN),
        plan,
        commit_sink=status,
    )
    assert status.plans[0].request_ids == ("req-0",)
    assert status.plans[0].generations == (9,)
    assert len(status.staged) == 1


@pytest.mark.parametrize(
    "corruption", ["negative_head", "oversized_length", "bad_last_label"]
)
def test_corrupt_book_masks_safely(corruption: str) -> None:
    # PORT-ADV-004 as amended: book-invariant violations are sanitized
    # to safe substitutes (no gather fault, no embedding fault), the
    # row masks with BOOK_INVARIANT, and its pages stay bit-identical.
    core = _tiny_core()
    pools = _fresh_pools()
    book = pools["book_pool"]
    if corruption == "negative_head":
        _set_replay_book(pools, 1, queue=[3, 5], head=1, expected=3)
        book[1, _BOOK["queue_head"]] = -3
        ids = [3]
        chunk = [False]
        embeds = torch.zeros(1, CARRIER_HIDDEN)
    elif corruption == "oversized_length":
        _set_replay_book(pools, 1, queue=[3, 5], head=1, expected=3)
        book[1, _BOOK["queue_length"]] = CAP + 10
        ids = [3]
        chunk = [False]
        embeds = torch.zeros(1, CARRIER_HIDDEN)
    else:  # bad_last_label on a CHUNK row: must not reach the embed
        book[1, _BOOK["last_label"]] = 500  # >> vocab
        ids = [PLACEHOLDER_ID]
        chunk = [True]
        torch.manual_seed(5)
        embeds = _envelope(
            torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0
        ).unsqueeze(0)
    before = _clone_pools(pools)
    plan = _plan(decodes=[1], chunk=chunk)
    status = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor(ids, dtype=torch.long),
        embeds,
        plan,
        commit_sink=status,
    )
    assert _decision(out) == [PARK_ID]
    _assert_pools_equal(pools, before)
    assert int(status.staged[0][0]) & advance.ROW_STATUS_BOOK_INVARIANT


# @spec PORT-ADV-004, PORT-STATE-008
def test_nonnegative_corrupt_replay_counters_mask_without_store() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    _set_replay_book(pools, 1, queue=[3, 5], head=1, expected=3)
    counters = pools["frontend_counter_pool"]
    counters[1, _CTR["total_valid_samples"]] = 100
    # origin + retained length must equal total for every persisted row.
    counters[1, _CTR["raw_tail_origin"]] = 0
    counters[1, _CTR["raw_tail_length"]] = 0
    before = _clone_pools(pools)
    sink = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([3]),
        torch.zeros(1, CARRIER_HIDDEN),
        _plan(decodes=[1]),
        commit_sink=sink,
    )
    assert _decision(out) == [PARK_ID]
    assert int(sink.staged[0][0]) & advance.ROW_STATUS_BOOK_INVARIANT
    _assert_pools_equal(pools, before)


# @spec PORT-ADV-004, PORT-STATE-008
def test_replay_counter_sequence_must_match_geometry_cadence() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    _set_replay_book(pools, 1, queue=[3, 5], head=1, expected=3)
    pools["frontend_counter_pool"][1, _CTR["expected_chunk_sequence"]] = 7
    before = _clone_pools(pools)
    sink = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([3]),
        torch.zeros(1, CARRIER_HIDDEN),
        _plan(decodes=[1]),
        commit_sink=sink,
    )
    assert _decision(out) == [PARK_ID]
    assert int(sink.staged[0][0]) & advance.ROW_STATUS_BOOK_INVARIANT
    _assert_pools_equal(pools, before)


# @spec PORT-ADV-004, PORT-FEAT-002, PORT-STATE-008
def test_nonfinal_raw_retention_must_match_committed_boundary() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    _set_replay_book(pools, 1, queue=[3, 5], head=1, expected=3)
    counters = pools["frontend_counter_pool"]
    counters[1, _CTR["total_valid_samples"]] = REG_SAMPLES
    counters[1, _CTR["committed_mel_frames"]] = 1
    counters[1, _CTR["encoded_mel_frames"]] = 1
    counters[1, _CTR["mel_tail_length"]] = 1
    counters[1, _CTR["expected_chunk_sequence"]] = 1
    # Still sums to total, but does not begin at the exact next-window
    # retention origin (zero for the first 80-ms cadence).
    counters[1, _CTR["raw_tail_origin"]] = 100
    counters[1, _CTR["raw_tail_length"]] = REG_SAMPLES - 100
    before = _clone_pools(pools)
    sink = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([3]),
        torch.zeros(1, CARRIER_HIDDEN),
        _plan(decodes=[1]),
        commit_sink=sink,
    )
    assert _decision(out) == [PARK_ID]
    assert int(sink.staged[0][0]) & advance.ROW_STATUS_BOOK_INVARIANT
    _assert_pools_equal(pools, before)


# @spec PORT-ADV-004, PORT-FEAT-002, PORT-STATE-008
def test_final_tail_must_account_for_exact_valid_sample_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = advance.advance_session

    def undercount(*args: Any, **kwargs: Any) -> Any:
        result = real(*args, **kwargs)
        state = args[2]
        counters = state.frontend_counters
        plausible_total = 3_200
        plausible_frames = plausible_total // 160
        counters[:, _CTR["total_valid_samples"]] = plausible_total
        counters[:, _CTR["committed_mel_frames"]] = plausible_frames
        counters[:, _CTR["encoded_mel_frames"]] = plausible_frames
        counters[:, _CTR["raw_tail_origin"]] = plausible_total
        counters[:, _CTR["raw_tail_length"]] = 0
        counters[:, _CTR["mel_tail_length"]] = 9
        counters[:, _CTR["expected_chunk_sequence"]] = 1
        counters[:, _CTR["finalized"]] = 1
        return result

    monkeypatch.setattr(advance, "advance_session", undercount)
    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(
        torch.randn(FINAL_SAMPLES) * 0.01,
        final=True,
        seq=0,
        geometry=GEOM_FINAL,
    ).unsqueeze(0)
    sink = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID]),
        carrier,
        _plan(prefills=[1], geometries=[GEOM_FINAL]),
        commit_sink=sink,
    )
    assert _decision(out) == [PARK_ID]
    assert int(sink.staged[0][0]) & advance.ROW_STATUS_BOOK_INVARIANT
    _assert_pools_equal(pools, before)


# @spec PORT-ADV-004, PORT-STATE-008
def test_finalized_counter_cannot_commit_frames_beyond_audio() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    _set_drained_book(pools, 1, blank=core.blank_id)
    counters = pools["frontend_counter_pool"]
    counters[1, _CTR["committed_mel_frames"]] = 100
    counters[1, _CTR["encoded_mel_frames"]] = 100
    counters[1, _CTR["mel_tail_length"]] = 9
    counters[1, _CTR["finalized"]] = 1
    before = _clone_pools(pools)
    sink = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([PARK_ID]),
        torch.zeros(1, CARRIER_HIDDEN),
        _plan(decodes=[1]),
        commit_sink=sink,
    )
    assert _decision(out) == [PARK_ID]
    assert int(sink.staged[0][0]) & advance.ROW_STATUS_BOOK_INVARIANT
    _assert_pools_equal(pools, before)


# @spec PORT-ADV-004, PORT-STATE-008
def test_finalized_counter_sequence_must_bound_final_residual() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    _set_drained_book(pools, 1, blank=core.blank_id)
    counters = pools["frontend_counter_pool"]
    counters[1, _CTR["expected_chunk_sequence"]] = 7
    counters[1, _CTR["finalized"]] = 1
    before = _clone_pools(pools)
    sink = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([PARK_ID]),
        torch.zeros(1, CARRIER_HIDDEN),
        _plan(decodes=[1]),
        commit_sink=sink,
    )
    assert _decision(out) == [PARK_ID]
    assert int(sink.staged[0][0]) & advance.ROW_STATUS_BOOK_INVARIANT
    _assert_pools_equal(pools, before)


def _no_label_decode(
    frames: torch.Tensor,
    lengths: torch.Tensor,
    predictor: Any,
    joint: Any,
    state: Any,
) -> Any:
    del lengths, predictor, joint
    rows = frames.shape[0]
    return (
        torch.zeros(rows, 1, dtype=torch.int32, device=frames.device),
        torch.zeros(rows, dtype=torch.int32, device=frames.device),
        state,
    )


# @spec PORT-FEAT-004, PORT-ADV-004, PORT-STATE-008
def test_maximum_legal_continuing_final_commits_one_cadence_transactionally() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    chunk_samples = 5_120  # geometry 2 / 320 ms
    cadence = 32
    first_boundary = 25
    torch.manual_seed(194)
    signal = torch.randn(2 * chunk_samples - 160) * 0.01
    sink = _CommitRecorder()
    first = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID]),
        _envelope(
            signal[:chunk_samples],
            final=False,
            seq=0,
            geometry=GEOM_FINAL,
        ).unsqueeze(0),
        _plan(prefills=[1], geometries=[GEOM_FINAL]),
        resolver=_fixed_resolver(_no_label_decode),
        commit_sink=sink,
    )
    assert _decision(first) == [PARK_ID]
    final = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID]),
        _envelope(
            signal[chunk_samples:],
            final=True,
            seq=1,
            geometry=GEOM_FINAL,
        ).unsqueeze(0),
        _plan(
            decodes=[1],
            chunk=[True],
            geometries=[GEOM_FINAL],
        ),
        resolver=_fixed_resolver(_no_label_decode),
        commit_sink=sink,
    )
    assert _decision(final) == [PARK_ID]
    assert all(int(status[0]) == 0 for status in sink.staged)
    counters = pools["frontend_counter_pool"][1]
    assert int(counters[_CTR["total_valid_samples"]]) == signal.shape[0]
    assert int(counters[_CTR["committed_mel_frames"]]) == (first_boundary + cadence)
    assert int(counters[_CTR["encoded_mel_frames"]]) == (first_boundary + cadence)
    assert int(counters[_CTR["expected_chunk_sequence"]]) == 2
    assert int(counters[_CTR["finalized"]]) == 1


@pytest.mark.parametrize("old_extra", range(1, 7))
# @spec PORT-FEAT-004, PORT-ADV-004, PORT-STATE-008
def test_old_cadence_plus_debt_finalized_tuple_is_rejected_transactionally(
    old_extra: int,
) -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    _set_drained_book(pools, 1, blank=core.blank_id)
    pools["book_pool"][1, _BOOK["geometry"]] = GEOM_FINAL
    cadence = 32
    first_boundary = 25
    total = 2 * 5_120 - 160
    counters = pools["frontend_counter_pool"]
    counters[1, _CTR["total_valid_samples"]] = total
    counters[1, _CTR["committed_mel_frames"]] = first_boundary + cadence + old_extra
    counters[1, _CTR["encoded_mel_frames"]] = first_boundary + cadence + old_extra
    counters[1, _CTR["raw_tail_origin"]] = total
    counters[1, _CTR["raw_tail_length"]] = 0
    counters[1, _CTR["mel_tail_length"]] = 9
    counters[1, _CTR["expected_chunk_sequence"]] = 2
    counters[1, _CTR["finalized"]] = 1
    before = _clone_pools(pools)
    sink = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([PARK_ID]),
        torch.zeros(1, CARRIER_HIDDEN),
        _plan(decodes=[1], geometries=[GEOM_FINAL]),
        commit_sink=sink,
    )
    assert _decision(out) == [PARK_ID]
    assert int(sink.staged[0][0]) & advance.ROW_STATUS_BOOK_INVARIANT
    _assert_pools_equal(pools, before)


# @spec PORT-LID-001, PORT-LID-003, PORT-STATE-008
def test_authorized_prompt_transition_commits_on_new_chunk() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    book = pools["book_pool"]
    book[1, _BOOK["last_label"]] = core.blank_id
    book[1, _BOOK["geometry"]] = GEOM_REG
    book[1, _BOOK["prompt"]] = 2
    torch.manual_seed(5)
    samples = torch.randn(REG_SAMPLES) * 0.01
    sink = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID]),
        _envelope(samples, final=False, seq=0, prompt=1).unsqueeze(0),
        _plan(
            decodes=[1],
            chunk=[True],
            prompts=[1],
            prior_prompts=[2],
            allow_prompt_transitions=[True],
        ),
        commit_sink=sink,
    )
    assert int(sink.staged[0][0]) == 0
    assert int(pools["book_pool"][1, _BOOK["prompt"]]) == 1
    assert _decision(out)[0] != PARK_ID


# @spec PORT-LID-003, PORT-STATE-008
def test_replay_persisted_prompt_drift_masks_only_that_row() -> None:
    core = _tiny_core()
    pools = _fresh_pools(num_blocks=4)
    _set_replay_book(pools, 1, queue=[3, 5], head=1, expected=3)
    _set_replay_book(pools, 2, queue=[3, 5], head=1, expected=3)
    pools["book_pool"][1, _BOOK["prompt"]] = 2
    pools["book_pool"][2, _BOOK["prompt"]] = 1
    before = _clone_pools(pools)
    sink = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([3, 3]),
        torch.zeros(2, CARRIER_HIDDEN),
        _plan(
            decodes=[1, 2],
            chunk=[False, False],
            prompts=[1, 1],
            num_pool_blocks=4,
        ),
        commit_sink=sink,
    )
    assert _decision(out) == [PARK_ID, 5]
    assert int(sink.staged[0][0]) & advance.ROW_STATUS_BOOK_IDENTITY
    for value, old in zip(pools.values(), before.values(), strict=True):
        values = value if isinstance(value, list) else [value]
        olds = old if isinstance(old, list) else [old]
        for tensor, original in zip(values, olds, strict=True):
            torch.testing.assert_close(tensor[1], original[1], rtol=0, atol=0)


# @spec PORT-ADV-004
@pytest.mark.parametrize(
    ("lengths", "tokens"),
    [
        ([-1, 1], [[1], [2]]),
        ([2, 1], [[1], [2]]),
        ([1, 1], [[-1], [2]]),
        ([1, 1], [[VOCAB], [2]]),
    ],
)
def test_malformed_decode_content_masks_only_affected_row(
    lengths: list[int], tokens: list[list[int]]
) -> None:
    def malformed(
        frames: Any, enc_lengths: Any, predictor: Any, joint: Any, state: Any
    ) -> Any:
        del frames, enc_lengths, predictor, joint
        token_tensor = torch.tensor(tokens, dtype=torch.int32)
        length_tensor = torch.tensor(lengths, dtype=torch.int32)
        next_label = state.last_label.clone()
        for row, length in enumerate(lengths):
            if 0 < length <= len(tokens[row]):
                next_label[row] = tokens[row][length - 1]
        return (
            token_tensor,
            length_tensor,
            replace(state, last_label=next_label.contiguous()),
        )

    core = _tiny_core()
    pools = _fresh_pools(num_blocks=4)
    before = _clone_pools(pools)
    torch.manual_seed(5)
    samples = torch.randn(REG_SAMPLES) * 0.01
    rows = torch.stack(
        [
            _envelope(samples, final=False, seq=0),
            _envelope(samples, final=False, seq=0),
        ]
    )
    sink = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID, PLACEHOLDER_ID]),
        rows,
        _plan(prefills=[1, 2], num_pool_blocks=4),
        resolver=_fixed_resolver(malformed),
        commit_sink=sink,
    )
    assert _decision(out) == [PARK_ID, 2]
    assert int(sink.staged[0][0]) & advance.ROW_STATUS_DECODE_INVARIANT
    for value, old in zip(pools.values(), before.values(), strict=True):
        values = value if isinstance(value, list) else [value]
        olds = old if isinstance(old, list) else [old]
        for tensor, original in zip(values, olds, strict=True):
            torch.testing.assert_close(tensor[1], original[1], rtol=0, atol=0)


# @spec PORT-ADV-004, PORT-STATE-008
def test_malformed_decode_next_state_structure_is_precommit_fatal() -> None:
    def malformed(
        frames: Any, enc_lengths: Any, predictor: Any, joint: Any, state: Any
    ) -> Any:
        del frames, enc_lengths, predictor, joint
        bad = rnnt.DecodeState(
            h=state.h[:, :, :-1],
            c=state.c,
            last_label=state.last_label,
        )
        return (
            torch.zeros(1, 1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            bad,
        )

    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0).unsqueeze(
        0
    )
    with pytest.raises(ValueError, match="next-state h"):
        _call(
            core,
            pools,
            torch.tensor([PLACEHOLDER_ID]),
            carrier,
            _plan(prefills=[1]),
            resolver=_fixed_resolver(malformed),
        )
    _assert_pools_equal(pools, before)


# @spec PORT-ADV-004, PORT-STATE-008
def test_invalid_decode_next_label_masks_only_affected_row() -> None:
    def malformed(
        frames: Any, enc_lengths: Any, predictor: Any, joint: Any, state: Any
    ) -> Any:
        del frames, enc_lengths, predictor, joint
        bad = rnnt.DecodeState(
            h=state.h,
            c=state.c,
            last_label=torch.tensor([VOCAB + 1, 2], dtype=torch.long),
        )
        return (
            torch.tensor([[1], [2]], dtype=torch.int32),
            torch.ones(2, dtype=torch.int32),
            bad,
        )

    core = _tiny_core()
    pools = _fresh_pools(num_blocks=4)
    before = _clone_pools(pools)
    torch.manual_seed(5)
    samples = torch.randn(REG_SAMPLES) * 0.01
    sink = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID, PLACEHOLDER_ID]),
        torch.stack(
            [
                _envelope(samples, final=False, seq=0),
                _envelope(samples, final=False, seq=0),
            ]
        ),
        _plan(prefills=[1, 2], num_pool_blocks=4),
        resolver=_fixed_resolver(malformed),
        commit_sink=sink,
    )
    assert _decision(out) == [PARK_ID, 2]
    assert int(sink.staged[0][0]) & advance.ROW_STATUS_DECODE_INVARIANT
    assert int(sink.staged[0][1]) == 0
    for value, old in zip(pools.values(), before.values(), strict=True):
        values = value if isinstance(value, list) else [value]
        olds = old if isinstance(old, list) else [old]
        for tensor, original in zip(values, olds, strict=True):
            torch.testing.assert_close(tensor[1], original[1], rtol=0, atol=0)


# @spec PORT-ADV-004
def test_transition_validation_preserves_incoming_status_bits() -> None:
    incoming = torch.tensor([advance.ROW_STATUS_ENVELOPE], dtype=torch.int32)
    result = advance.AdvanceResult(
        token_ids=torch.zeros(1, 1, dtype=torch.int32),
        token_lengths=torch.zeros(1, dtype=torch.int32),
        row_status=torch.zeros(1, dtype=torch.int32),
    )
    checked = advance._validated_result(
        result,
        incoming=incoming,
        blank_id=VOCAB,
        queue_capacity=CAP,
        geometry_bound=CAP,
        capture=False,
    )
    assert int(checked.row_status[0]) & advance.ROW_STATUS_ENVELOPE
    assert int(checked.row_status[0]) & advance.ROW_STATUS_DECODE_INVARIANT


# @spec PORT-ADV-004, PORT-HOOK-001
def test_adapter_validation_preserves_incoming_status_bits() -> None:
    real = advance.make_mrv1_adapter(
        hidden_size=CARRIER_HIDDEN,
        park_id=PARK_ID,
        blank_id=VOCAB,
    )

    def dropping(result: Any, context: Any) -> Any:
        projection = real(result, context)
        return replace(
            projection,
            row_status=torch.zeros_like(projection.row_status),
        )

    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0).unsqueeze(
        0
    )
    carrier[0, advance.ENV_VERSION] = 99
    sink = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID]),
        carrier,
        _plan(prefills=[1]),
        adapter=dropping,
        commit_sink=sink,
    )
    assert _decision(out) == [PARK_ID]
    assert int(sink.staged[0][0]) & advance.ROW_STATUS_ENVELOPE
    assert int(sink.staged[0][0]) & advance.ROW_STATUS_DECODE_INVARIANT
    _assert_pools_equal(pools, before)


# @spec PORT-ADV-003, PORT-ADV-004, PORT-STATE-008
@pytest.mark.parametrize(
    "corruption",
    [
        "nan",
        "fractional",
        "out_of_range",
        "tail_garbage",
        "echo_disagreement",
        "coherent_substitution",
    ],
)
def test_malformed_adapter_decision_masks_without_store(corruption: str) -> None:
    real = advance.make_mrv1_adapter(
        hidden_size=CARRIER_HIDDEN,
        park_id=PARK_ID,
        blank_id=VOCAB,
    )

    def malformed(result: Any, context: Any) -> Any:
        projection = real(result, context)
        rows = projection.rows.clone()
        book = projection.book.clone()
        if corruption == "nan":
            rows[0, 0] = torch.nan
        elif corruption == "fractional":
            rows[0, 0] = 1.5
        elif corruption == "out_of_range":
            rows[0, 0] = VOCAB
        elif corruption == "tail_garbage":
            rows[0, 1] = 1
        elif corruption == "echo_disagreement":
            rows[0, 0] = 1
            book[0, _BOOK["pending_echo"]] = 1
            book[0, _BOOK["expected_label"]] = 2
        else:
            rows[0, 0] = 1
            queue = projection.queue.clone()
            queue[0].zero_()
            queue[0, 0] = 1
            book[0, _BOOK["queue_head"]] = 1
            book[0, _BOOK["queue_length"]] = 1
            book[0, _BOOK["pending_echo"]] = 1
            book[0, _BOOK["expected_label"]] = 1
            return replace(projection, rows=rows, queue=queue, book=book)
        return replace(projection, rows=rows, book=book)

    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0).unsqueeze(
        0
    )
    sink = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID]),
        carrier,
        _plan(prefills=[1]),
        adapter=malformed,
        commit_sink=sink,
    )
    assert _decision(out) == [PARK_ID]
    assert int(sink.staged[0][0]) & advance.ROW_STATUS_DECODE_INVARIANT
    _assert_pools_equal(pools, before)


# @spec PORT-ADV-003, PORT-ADV-004, PORT-STATE-008
@pytest.mark.parametrize("corruption", ["clear_status", "flush_book", "replay_queue"])
def test_adapter_cannot_mutate_transaction_authority_in_place(
    corruption: str,
) -> None:
    real = advance.make_mrv1_adapter(
        hidden_size=CARRIER_HIDDEN,
        park_id=PARK_ID,
        blank_id=VOCAB,
    )

    def mutating(result: Any, context: Any) -> Any:
        if corruption == "clear_status":
            context.row_status.zero_()
        elif corruption == "flush_book":
            context.book[0, _BOOK["last_label"]] = 1
        else:
            context.queue[0, 1] = 6
        return real(result, context)

    core = _tiny_core()
    pools = _fresh_pools()
    if corruption == "clear_status":
        torch.manual_seed(5)
        embeds = _envelope(
            torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0
        ).unsqueeze(0)
        embeds[0, advance.ENV_VERSION] = 99
        ids = torch.tensor([PLACEHOLDER_ID])
        plan = _plan(prefills=[1])
    elif corruption == "flush_book":
        _set_drained_book(pools, 1, blank=core.blank_id)
        counters = pools["frontend_counter_pool"]
        counters[1, _CTR["finalized"]] = 1
        counters[1, _CTR["expected_chunk_sequence"]] = 1
        embeds = torch.zeros(1, CARRIER_HIDDEN)
        ids = torch.tensor([PARK_ID])
        plan = _plan(decodes=[1])
    else:
        _set_replay_book(pools, 1, queue=[3, 5], head=1, expected=3)
        embeds = torch.zeros(1, CARRIER_HIDDEN)
        ids = torch.tensor([3])
        plan = _plan(decodes=[1])
    before = _clone_pools(pools)
    sink = _CommitRecorder()
    out = _call(
        core,
        pools,
        ids,
        embeds,
        plan,
        adapter=mutating,
        commit_sink=sink,
    )
    assert _decision(out) == [PARK_ID]
    assert int(sink.staged[0][0]) != 0
    _assert_pools_equal(pools, before)


# @spec PORT-ADV-004
def test_decode_validation_ignores_padding_beyond_active_length() -> None:
    result = advance.AdvanceResult(
        token_ids=torch.tensor([[3, VOCAB, -1]], dtype=torch.int32),
        token_lengths=torch.tensor([1], dtype=torch.int32),
        row_status=torch.zeros(1, dtype=torch.int32),
    )
    checked = advance._validated_result(
        result,
        incoming=torch.zeros(1, dtype=torch.int32),
        blank_id=VOCAB,
        queue_capacity=CAP,
        geometry_bound=CAP,
        capture=False,
    )
    assert int(checked.row_status[0]) == 0
    assert int(checked.token_lengths[0]) == 1


# @spec PORT-ADV-004
def test_malformed_advance_result_structure_is_precommit_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = advance.advance_session

    def malformed(*args: Any, **kwargs: Any) -> Any:
        result = real(*args, **kwargs)
        return replace(result, token_ids=result.token_ids.long())

    monkeypatch.setattr(advance, "advance_session", malformed)
    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0).unsqueeze(
        0
    )
    with pytest.raises(ValueError, match="token_ids"):
        _call(
            core,
            pools,
            torch.tensor([PLACEHOLDER_ID]),
            carrier,
            _plan(prefills=[1]),
            _refuse_adapter,
        )
    _assert_pools_equal(pools, before)


# @spec PORT-ADV-001, PORT-ADV-004, PORT-STATE-008
@pytest.mark.parametrize(
    "corruption",
    ["last_label", "zero_emission_state", "zero_emission_inplace"],
)
def test_decode_state_must_match_emission_semantics(corruption: str) -> None:
    def malformed_decode(
        frames: torch.Tensor,
        lengths: torch.Tensor,
        predictor: Any,
        joint: Any,
        state: Any,
    ) -> Any:
        token_ids, token_lengths, next_state = rnnt.decode_dense_masked(
            frames,
            lengths,
            predictor,
            joint,
            state,
        )
        if corruption == "last_label":
            assert bool((token_lengths > 0).all())
            wrong = (next_state.last_label + 1) % VOCAB
            next_state = replace(next_state, last_label=wrong.contiguous())
        elif corruption == "zero_emission_state":
            token_lengths = torch.zeros_like(token_lengths)
            next_state = replace(
                next_state,
                h=(state.h + 1).contiguous(),
                c=(state.c + 1).contiguous(),
                last_label=state.last_label.clone(),
            )
        else:
            token_lengths = torch.zeros_like(token_lengths)
            state.h.add_(1)
            state.c.add_(1)
            next_state = state
        return token_ids, token_lengths, next_state

    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0).unsqueeze(
        0
    )
    sink = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID]),
        carrier,
        _plan(prefills=[1]),
        resolver=_fixed_resolver(malformed_decode),
        commit_sink=sink,
    )
    assert _decision(out) == [PARK_ID]
    assert int(sink.staged[0][0]) & advance.ROW_STATUS_DECODE_INVARIANT
    _assert_pools_equal(pools, before)


# @spec PORT-ADV-004, PORT-STATE-008
def test_armed_blank_expected_label_is_a_book_invariant() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    _set_replay_book(
        pools,
        1,
        queue=[3, 5],
        head=1,
        expected=core.blank_id,
    )
    before = _clone_pools(pools)
    sink = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([core.blank_id]),
        torch.zeros(1, CARRIER_HIDDEN),
        _plan(decodes=[1]),
        commit_sink=sink,
    )
    assert _decision(out) == [PARK_ID]
    assert int(sink.staged[0][0]) & advance.ROW_STATUS_BOOK_INVARIANT
    _assert_pools_equal(pools, before)


# @spec PORT-ADV-004, PORT-DEC-007, PORT-STATE-008
@pytest.mark.parametrize(
    "corruption",
    ["pending_at_zero", "wrong_expected", "wrong_last", "undrained_unarmed"],
)
def test_replay_book_cross_relations_are_reachable(corruption: str) -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    _set_replay_book(pools, 1, queue=[3, 5], head=1, expected=3)
    book = pools["book_pool"]
    if corruption == "pending_at_zero":
        book[1, _BOOK["queue_head"]] = 0
    elif corruption == "wrong_expected":
        book[1, _BOOK["expected_label"]] = 4
    elif corruption == "wrong_last":
        book[1, _BOOK["last_label"]] = 4
    else:
        book[1, _BOOK["pending_echo"]] = 0
    before = _clone_pools(pools)
    sink = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([3]),
        torch.zeros(1, CARRIER_HIDDEN),
        _plan(decodes=[1]),
        commit_sink=sink,
    )
    assert _decision(out) == [PARK_ID]
    assert int(sink.staged[0][0]) & advance.ROW_STATUS_BOOK_INVARIANT
    _assert_pools_equal(pools, before)


# @spec PORT-ADV-004, PORT-STATE-007, PORT-STATE-008
@pytest.mark.parametrize("corruption", ["negative_head", "armed_blank"])
def test_adapter_proposed_book_defect_masks_without_store(
    corruption: str,
) -> None:
    real = advance.make_mrv1_adapter(
        hidden_size=CARRIER_HIDDEN,
        park_id=PARK_ID,
        blank_id=VOCAB,
    )

    def malformed(result: Any, context: Any) -> Any:
        projection = real(result, context)
        book = projection.book.clone()
        if corruption == "negative_head":
            book[0, _BOOK["queue_head"]] = -1
        else:
            book[0, _BOOK["pending_echo"]] = 1
            book[0, _BOOK["expected_label"]] = VOCAB
        return replace(projection, book=book)

    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0).unsqueeze(
        0
    )
    sink = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID]),
        carrier,
        _plan(prefills=[1]),
        adapter=malformed,
        commit_sink=sink,
    )
    assert _decision(out) == [PARK_ID]
    assert int(sink.staged[0][0]) & advance.ROW_STATUS_BOOK_INVARIANT
    _assert_pools_equal(pools, before)


# @spec PORT-ADV-004, PORT-STATE-008
def test_all_scatter_descriptors_validate_before_first_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    validations = {"count": 0}
    writes = {"count": 0}
    real_validate = advance.validate_masked_page_scatter

    def fail_late(*args: Any) -> None:
        validations["count"] += 1
        if validations["count"] == 2:
            raise ValueError("malformed later scatter descriptor")
        real_validate(*args)

    def observe_write(*_args: Any) -> None:
        writes["count"] += 1

    monkeypatch.setattr(advance, "validate_masked_page_scatter", fail_late)
    monkeypatch.setattr(advance, "_execute_masked_page_scatter_", observe_write)
    torch.manual_seed(5)
    carrier = _envelope(torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0).unsqueeze(
        0
    )
    with pytest.raises(ValueError, match="later scatter"):
        _call(
            core,
            pools,
            torch.tensor([PLACEHOLDER_ID]),
            carrier,
            _plan(prefills=[1]),
        )
    assert validations["count"] == 2
    assert writes["count"] == 0
    _assert_pools_equal(pools, before)


# @spec PORT-ADV-003, PORT-HOOK-001
def test_commit_failure_cancels_composite_ticket_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    sink = _CommitRecorder()

    def fail_commit(*_args: Any) -> None:
        raise RuntimeError("scatter launch failed")

    monkeypatch.setattr(advance, "_execute_masked_page_scatter_", fail_commit)
    torch.manual_seed(5)
    carrier = _envelope(torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0).unsqueeze(
        0
    )
    with pytest.raises(RuntimeError, match="scatter launch"):
        _call(
            core,
            pools,
            torch.tensor([PLACEHOLDER_ID]),
            carrier,
            _plan(prefills=[1]),
            commit_sink=sink,
        )
    assert sink.log == ["reserve", "cancel"]
    assert sink.cancels == 1
    assert sink.staged == []


# @spec PORT-ADV-003, PORT-STATE-008
def test_commit_loop_never_revalidates_a_prevalidated_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The complete-plan pass validates every scatter descriptor once,
    # BEFORE the no-fail commit window; the commit loop then calls the
    # private prevalidated executor directly (state_scatter.py's own
    # module docstring), so a successful two-bucket transaction must
    # validate each descriptor EXACTLY once total — zero times inside
    # the commit loop.
    core = _tiny_core()
    pools = _fresh_pools(num_blocks=4)
    counts = {"n": 0}
    real_validate = advance.validate_masked_page_scatter

    def counting_validate(*args: Any) -> None:
        counts["n"] += 1
        real_validate(*args)

    monkeypatch.setattr(advance, "validate_masked_page_scatter", counting_validate)
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
    )
    out = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID, PLACEHOLDER_ID], dtype=torch.long),
        torch.stack([reg, fin]),
        plan,
    )
    assert out.shape == (2, CARRIER_HIDDEN)
    per_bucket = (
        len(pools["channel_pools"])
        + len(pools["time_pools"])
        + len(pools["len_pools"])
        + 5
    )
    expected_descriptors = 2 * per_bucket + 2  # two buckets + queue/book
    assert counts["n"] == expected_descriptors


def test_reservation_stage_occurs_before_scatter_phase_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-ADV-003, PORT-HOOK-001, PORT-PERF-007
    events: list[str] = []

    class _RecordedPhase:
        def __init__(self, name: str) -> None:
            self._name = name

        def __enter__(self) -> None:
            events.append(f"{self._name}:enter")

        def __exit__(self, *_args: Any) -> None:
            events.append(f"{self._name}:exit")

    monkeypatch.setattr(advance, "phase", _RecordedPhase)
    core = _tiny_core()
    pools = _fresh_pools()
    sink = _CommitRecorder()
    sink.on_stage = lambda: events.append("reservation:stage")
    torch.manual_seed(5)
    carrier = _envelope(
        torch.randn(REG_SAMPLES) * 0.01,
        final=False,
        seq=0,
    ).unsqueeze(0)
    _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID]),
        carrier,
        _plan(prefills=[1]),
        commit_sink=sink,
    )

    assert events.index("port.scatter:enter") < events.index("reservation:stage")
    assert events.index("reservation:stage") < events.index("port.scatter:exit")


def test_burst_overflow_is_rejected_not_truncated() -> None:
    # A decode returning more labels than the queue holds is a port
    # defect: the row masks with BURST_OVERFLOW and nothing scatters —
    # the queue never sees a truncated burst.
    def _overflowing_decode(
        frames: Any, lengths: Any, predictor: Any, joint: Any, state: Any
    ) -> Any:
        b = frames.shape[0]
        ids = torch.ones(b, CAP + 8, dtype=torch.int32)
        lens = torch.full((b,), CAP + 1, dtype=torch.int32)
        return ids, lens, state

    core = _tiny_core()
    pools = _fresh_pools()
    before = _clone_pools(pools)
    torch.manual_seed(5)
    carrier = _envelope(
        torch.randn(FINAL_SAMPLES) * 0.01,
        final=True,
        seq=0,
        geometry=GEOM_FINAL,
    ).unsqueeze(0)
    plan = _plan(prefills=[1], geometries=[GEOM_FINAL])
    status = _CommitRecorder()
    out = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
        carrier,
        plan,
        resolver=_fixed_resolver(_overflowing_decode),
        commit_sink=status,
    )
    assert _decision(out) == [PARK_ID]
    assert int(status.staged[0][0]) & advance.ROW_STATUS_BURST_OVERFLOW
    torch.testing.assert_close(
        pools["queue_pool"], before["queue_pool"], rtol=0, atol=0
    )
    torch.testing.assert_close(pools["book_pool"], before["book_pool"], rtol=0, atol=0)


def test_composite_stage_observes_commit() -> None:
    # Ordering proof: one reserve → scatters → one combined stage; at
    # stage time the resident book already holds the committed value.
    core = _tiny_core()
    pools = _fresh_pools()
    torch.manual_seed(5)
    carrier = _envelope(
        torch.randn(FINAL_SAMPLES) * 0.01,
        final=True,
        seq=0,
        geometry=GEOM_FINAL,
    ).unsqueeze(0)
    plan = _plan(prefills=[1], geometries=[GEOM_FINAL])
    sink = _CommitRecorder()
    seen_at_stage: dict[str, int] = {}

    def snapshot() -> None:
        seen_at_stage["head"] = int(pools["book_pool"][1, _BOOK["queue_head"]])

    sink.on_stage = snapshot
    _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
        carrier,
        plan,
        commit_sink=sink,
        capture=True,
    )
    assert sink.log == ["reserve", "stage"]
    # The CHUNK committed head=1 before the combined no-fail stage.
    assert seen_at_stage["head"] == 1


# ---- the MRV1 adapter as a pure tensor transform --------------------------


def _context(
    *,
    roles: list[int],
    input_ids: list[int],
    chunk_rows: list[int],
    queue: torch.Tensor,
    book: torch.Tensor,
    row_status: list[int] | None = None,
    prompt_index: list[int] | None = None,
) -> Any:
    n = len(roles)
    return advance.EmissionContext(
        roles=torch.tensor(roles, dtype=torch.long),
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        chunk_rows=torch.tensor(chunk_rows, dtype=torch.long),
        queue=queue,
        book=book,
        prompt_index=torch.tensor(prompt_index or [0] * n, dtype=torch.long),
        row_status=torch.tensor(row_status or [0] * n, dtype=torch.int32),
    )


def _adapter() -> Any:
    return advance.make_mrv1_adapter(hidden_size=64, park_id=PARK_ID, blank_id=VOCAB)


def test_adapter_chunk_burst_first_label_and_armed_echo() -> None:
    adapter = _adapter()
    result = advance.AdvanceResult(
        token_ids=torch.tensor([[4, 6, 0]], dtype=torch.int32),
        token_lengths=torch.tensor([2], dtype=torch.int32),
    )
    ctx = _context(
        roles=[advance.ROLE_CHUNK],
        input_ids=[PLACEHOLDER_ID],
        chunk_rows=[0],
        queue=torch.zeros(1, CAP, dtype=torch.int32),
        book=torch.zeros(1, BOOK_WIDTH, dtype=torch.int32),
    )
    proj = adapter(result, ctx)
    assert proj.rows[0, 0] == 4
    assert proj.queue[0, :2].tolist() == [4, 6]
    assert int(proj.book[0, _BOOK["queue_head"]]) == 1
    assert int(proj.book[0, _BOOK["queue_length"]]) == 2
    assert int(proj.book[0, _BOOK["pending_echo"]]) == 1
    assert int(proj.book[0, _BOOK["expected_label"]]) == 4


def test_adapter_zero_burst_parks_without_echo() -> None:
    # PORT-DEC-004: a blank-only chunk parks immediately; park never
    # arms an echo (PORT-DEC-003).
    adapter = _adapter()
    result = advance.AdvanceResult(
        token_ids=torch.zeros(1, 3, dtype=torch.int32),
        token_lengths=torch.zeros(1, dtype=torch.int32),
    )
    ctx = _context(
        roles=[advance.ROLE_CHUNK],
        input_ids=[PLACEHOLDER_ID],
        chunk_rows=[0],
        queue=torch.zeros(1, CAP, dtype=torch.int32),
        book=torch.zeros(1, BOOK_WIDTH, dtype=torch.int32),
    )
    proj = adapter(result, ctx)
    assert proj.rows[0, 0] == PARK_ID
    assert int(proj.book[0, _BOOK["pending_echo"]]) == 0
    assert int(proj.book[0, _BOOK["queue_length"]]) == 0


def test_adapter_masked_chunk_keeps_scratch_and_parks() -> None:
    adapter = _adapter()
    result = advance.AdvanceResult(
        token_ids=torch.tensor([[4, 6]], dtype=torch.int32),
        token_lengths=torch.tensor([2], dtype=torch.int32),
    )
    book = torch.zeros(1, BOOK_WIDTH, dtype=torch.int32)
    book[0, _BOOK["last_label"]] = 3
    queue = torch.zeros(1, CAP, dtype=torch.int32)
    ctx = _context(
        roles=[advance.ROLE_CHUNK],
        input_ids=[PLACEHOLDER_ID],
        chunk_rows=[0],
        queue=queue,
        book=book,
        row_status=[advance.ROW_STATUS_ENVELOPE],
    )
    proj = adapter(result, ctx)
    assert proj.rows[0, 0] == PARK_ID
    torch.testing.assert_close(proj.book, book, rtol=0, atol=0)
    torch.testing.assert_close(proj.queue, queue, rtol=0, atol=0)


def test_adapter_replay_emits_next_and_advances_head() -> None:
    adapter = _adapter()
    queue = torch.zeros(1, CAP, dtype=torch.int32)
    queue[0, 0], queue[0, 1] = 4, 6
    book = torch.zeros(1, BOOK_WIDTH, dtype=torch.int32)
    book[0, _BOOK["queue_head"]] = 1
    book[0, _BOOK["queue_length"]] = 2
    book[0, _BOOK["pending_echo"]] = 1
    book[0, _BOOK["expected_label"]] = 4
    ctx = _context(
        roles=[advance.ROLE_REPLAY],
        input_ids=[4],
        chunk_rows=[],
        queue=queue,
        book=book,
    )
    proj = adapter(
        advance.AdvanceResult(
            token_ids=torch.zeros(0, 0, dtype=torch.int32),
            token_lengths=torch.zeros(0, dtype=torch.int32),
        ),
        ctx,
    )
    assert proj.rows[0, 0] == 6
    assert int(proj.book[0, _BOOK["queue_head"]]) == 2
    assert int(proj.book[0, _BOOK["expected_label"]]) == 6
    assert int(proj.book[0, _BOOK["pending_echo"]]) == 1


def test_adapter_drained_replay_parks_and_clears_echo() -> None:
    adapter = _adapter()
    queue = torch.zeros(1, CAP, dtype=torch.int32)
    queue[0, 0], queue[0, 1] = 4, 6
    book = torch.zeros(1, BOOK_WIDTH, dtype=torch.int32)
    book[0, _BOOK["queue_head"]] = 2
    book[0, _BOOK["queue_length"]] = 2
    book[0, _BOOK["pending_echo"]] = 1
    book[0, _BOOK["expected_label"]] = 6
    ctx = _context(
        roles=[advance.ROLE_REPLAY],
        input_ids=[6],
        chunk_rows=[],
        queue=queue,
        book=book,
    )
    proj = adapter(
        advance.AdvanceResult(
            token_ids=torch.zeros(0, 0, dtype=torch.int32),
            token_lengths=torch.zeros(0, dtype=torch.int32),
        ),
        ctx,
    )
    assert proj.rows[0, 0] == PARK_ID
    assert int(proj.book[0, _BOOK["pending_echo"]]) == 0
    assert int(proj.book[0, _BOOK["queue_head"]]) == 2


def test_endpoint_control_is_a_legal_replay_decision_only_when_installed() -> None:
    # @spec PORT-DEC-005 / PORT-SEG-003 / PORT-ADV-004
    eou_token_id = VOCAB + 3
    queue = torch.zeros(1, CAP, dtype=torch.int32)
    queue[0, 0] = eou_token_id
    book = torch.zeros(1, BOOK_WIDTH, dtype=torch.int32)
    book[0, _BOOK["queue_length"]] = 1
    ctx = _context(
        roles=[advance.ROLE_REPLAY],
        input_ids=[4],
        chunk_rows=[],
        queue=queue,
        book=book,
    )
    result = advance.AdvanceResult(
        token_ids=torch.zeros(0, 0, dtype=torch.int32),
        token_lengths=torch.zeros(0, dtype=torch.int32),
    )
    projection = _adapter()(result, ctx)
    assert int(projection.rows[0, 0]) == eou_token_id

    installed_bad = advance._mrv1_projection_invariant_rows(
        result,
        ctx,
        projection,
        torch.zeros(1, dtype=torch.int32),
        park_id=PARK_ID,
        blank_id=VOCAB,
        eou_token_id=eou_token_id,
    )
    disabled_bad = advance._mrv1_projection_invariant_rows(
        result,
        ctx,
        projection,
        torch.zeros(1, dtype=torch.int32),
        park_id=PARK_ID,
        blank_id=VOCAB,
        eou_token_id=None,
    )
    assert installed_bad.tolist() == [False]
    assert disabled_bad.tolist() == [True]


def test_adapter_queue_saturation_full_capacity_burst() -> None:
    adapter = _adapter()
    burst = torch.arange(1, CAP + 1, dtype=torch.int32).unsqueeze(0)
    result = advance.AdvanceResult(
        token_ids=burst,
        token_lengths=torch.tensor([CAP], dtype=torch.int32),
    )
    ctx = _context(
        roles=[advance.ROLE_CHUNK],
        input_ids=[PLACEHOLDER_ID],
        chunk_rows=[0],
        queue=torch.zeros(1, CAP, dtype=torch.int32),
        book=torch.zeros(1, BOOK_WIDTH, dtype=torch.int32),
    )
    proj = adapter(result, ctx)
    assert proj.queue[0].tolist() == list(range(1, CAP + 1))
    assert int(proj.book[0, _BOOK["queue_length"]]) == CAP
    assert proj.rows[0, 0] == 1


def test_adapter_mixed_roles_keep_row_order() -> None:
    adapter = _adapter()
    n = 4
    queue = torch.zeros(n, CAP, dtype=torch.int32)
    book = torch.zeros(n, BOOK_WIDTH, dtype=torch.int32)
    # row 1: replay mid-queue; row 3: drained replay.
    queue[1, 0], queue[1, 1] = 7, 9
    book[1, _BOOK["queue_head"]] = 1
    book[1, _BOOK["queue_length"]] = 2
    book[1, _BOOK["pending_echo"]] = 1
    book[3, _BOOK["queue_head"]] = 1
    book[3, _BOOK["queue_length"]] = 1
    book[3, _BOOK["pending_echo"]] = 1
    result = advance.AdvanceResult(
        token_ids=torch.tensor([[5, 0]], dtype=torch.int32),
        token_lengths=torch.tensor([1], dtype=torch.int32),
    )
    ctx = _context(
        roles=[
            advance.ROLE_CHUNK,
            advance.ROLE_REPLAY,
            advance.ROLE_FLUSH,
            advance.ROLE_REPLAY,
        ],
        input_ids=[PLACEHOLDER_ID, 7, PARK_ID, 11],
        chunk_rows=[0],
        queue=queue,
        book=book,
    )
    proj = adapter(result, ctx)
    assert proj.rows[:, 0].long().tolist() == [5, 9, PARK_ID, PARK_ID]
    assert int(proj.book[3, _BOOK["pending_echo"]]) == 0


# ---- the compiled table resolver -----------------------------------------


def _table() -> Any:
    entries = {}
    for g in manifests.CADENCES:
        entries[(g, 8, "fp32")] = "dense-eager"
        entries[(g, 64, "fp32")] = "dense-eager"
        entries[(g, 512, "fp32")] = "compact-eager"
    return decode_dispatch.DispatchTable(
        entries=entries,
        fingerprint={},
        hysteresis_pct=25.0,
        policy_version="dcp-v1",
        lanes=("fp32",),
    )


def _arms() -> dict[str, Any]:
    return {
        "dense-eager": rnnt.decode_dense_masked,
        "dense-graphed": rnnt.decode_dense_masked,
        "compact-eager": rnnt.decode_compact_active,
    }


def test_table_resolver_matches_select_semantics() -> None:
    table = _table()
    resolver = advance.make_table_resolver(
        table, lane="fp32", arms=_arms(), max_batch=1024
    )
    for batch in (1, 8, 33, 64, 100, 512, 700, 1024):
        got = resolver(
            advance.DecodeRequest(
                geometry=0,
                execution_batch_size=batch,
                graph_covers_decode=False,
                ready_decode_buckets=1,
            )
        )
        want = table.select("80ms", batch, lane="fp32", graph_covers_decode=False)
        assert got.arm == want, f"batch {batch}: {got.arm} != {want}"


def test_table_resolver_bracket_disagreement_takes_sync_free() -> None:
    resolver = advance.make_table_resolver(
        _table(), lane="fp32", arms=_arms(), max_batch=1024
    )
    # 64 (dense) and 512 (compact) bracket 200: sync-free wins.
    got = resolver(
        advance.DecodeRequest(
            geometry=0,
            execution_batch_size=200,
            graph_covers_decode=False,
            ready_decode_buckets=1,
        )
    )
    assert got.arm == "dense-eager"


def test_table_resolver_graph_coverage_structurally_selects_dense_graph() -> None:
    resolver = advance.make_table_resolver(
        _table(), lane="fp32", arms=_arms(), max_batch=1024
    )
    resolved = resolver(
        advance.DecodeRequest(
            geometry=0,
            execution_batch_size=512,
            graph_covers_decode=True,
            ready_decode_buckets=1,
        )
    )
    assert resolved.arm == "dense-graphed"
    assert resolved.decode_fn is _arms()["dense-graphed"]


def test_table_resolver_multi_bucket_forces_sync_free() -> None:
    resolver = advance.make_table_resolver(
        _table(), lane="fp32", arms=_arms(), max_batch=1024
    )
    single = resolver(
        advance.DecodeRequest(
            geometry=0,
            execution_batch_size=512,
            graph_covers_decode=False,
            ready_decode_buckets=1,
        )
    )
    assert single.arm == "compact-eager"
    multi = resolver(
        advance.DecodeRequest(
            geometry=0,
            execution_batch_size=512,
            graph_covers_decode=False,
            ready_decode_buckets=2,
        )
    )
    assert multi.arm == "dense-eager"
    assert multi.override_reason == "multi-bucket-serialization-guard"


def test_table_resolver_missing_geometry_fails_at_startup() -> None:
    table = _table()
    entries = {k: v for k, v in table.entries.items() if k[0] != "320ms"}
    broken = decode_dispatch.DispatchTable(
        entries=entries,
        fingerprint={},
        hysteresis_pct=25.0,
        policy_version="dcp-v1",
        lanes=("fp32",),
    )
    with pytest.raises(KeyError):
        advance.make_table_resolver(broken, lane="fp32", arms=_arms(), max_batch=64)


def test_table_resolver_unbound_arm_fails_at_startup() -> None:
    arms = _arms()
    del arms["compact-eager"]
    with pytest.raises(KeyError):
        advance.make_table_resolver(_table(), lane="fp32", arms=arms, max_batch=1024)


# ---- PORT-PERF-001: reusable host staging (HostStaging) -------------------


def test_host_staging_rejects_oversized_vector() -> None:
    staging = advance.HostStaging(4)
    with pytest.raises(ValueError):
        staging.stage(
            "composed_indices",
            torch.arange(5, dtype=torch.int64),
            torch.device("cpu"),
        )


def test_host_staging_rejects_unknown_slot() -> None:
    staging = advance.HostStaging(4)
    with pytest.raises(ValueError):
        staging.stage(
            "not_a_slot",
            torch.arange(2, dtype=torch.int64),
            torch.device("cpu"),
        )


def _run_burst_then_drain(staging: Any) -> tuple[Pools, list[int]]:
    """The ``test_burst_then_drain_then_park_matches_reference`` arc,
    parameterized by staging, so it can be run once unstaged and once
    through a :class:`advance.HostStaging` and diffed bit-for-bit."""
    core = _tiny_core(seed=1)  # two-distinct-label burst by design
    torch.manual_seed(5)
    samples = torch.randn(FINAL_SAMPLES) * 0.01
    pools = _fresh_pools()
    plan = _plan(prefills=[1], geometries=[GEOM_FINAL])
    carrier = _envelope(samples, final=True, seq=0, geometry=GEOM_FINAL).unsqueeze(0)
    out = _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
        carrier,
        plan,
        staging=staging,
    )
    emitted = [_decision(out)[0]]
    decode_plan = _plan(decodes=[1], geometries=[GEOM_FINAL])
    for _ in range(CAP + 1):  # bounded: a drain never exceeds queue capacity
        if emitted[-1] == PARK_ID:
            break
        out = _call(
            core,
            pools,
            torch.tensor([emitted[-1]], dtype=torch.long),
            torch.zeros(1, CARRIER_HIDDEN),
            decode_plan,
            staging=staging,
        )
        emitted.append(_decision(out)[0])
    return pools, emitted


def test_host_staging_matches_unstaged_transaction_bit_identical() -> None:
    pools_unstaged, emitted_unstaged = _run_burst_then_drain(None)
    pools_staged, emitted_staged = _run_burst_then_drain(advance.HostStaging(8))
    assert len(emitted_unstaged) >= 3 and len(set(emitted_unstaged)) >= 2
    assert emitted_staged == emitted_unstaged
    _assert_pools_equal(pools_staged, pools_unstaged)


def _run_multi_bucket(staging: Any) -> tuple[Pools, list[int]]:
    """One transaction spanning TWO chunk geometries (GEOM_REG +
    GEOM_FINAL) plus a replay row, so ``HostStaging.stage_bucket`` is
    invoked once per geometry WITHIN a single call — the multi-bucket
    case a single shared bucket slot could not serve safely. Diffed
    staged-vs-unstaged bit-for-bit."""
    core = _tiny_core()
    pools = _fresh_pools(num_blocks=5)
    _set_replay_book(pools, 3, queue=[7, 9], head=1, expected=7)
    plan = _plan(
        decodes=[3],
        prefills=[1, 2],
        num_pool_blocks=5,
        geometries=[GEOM_REG, GEOM_REG, GEOM_FINAL],
    )
    torch.manual_seed(5)
    reg = _envelope(
        torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0, geometry=GEOM_REG
    )
    fin = _envelope(
        torch.randn(FINAL_SAMPLES) * 0.01, final=True, seq=0, geometry=GEOM_FINAL
    )
    input_ids = torch.tensor([7, PLACEHOLDER_ID, PLACEHOLDER_ID], dtype=torch.long)
    embeds = torch.stack([torch.zeros(CARRIER_HIDDEN), reg, fin])
    out = _call(core, pools, input_ids, embeds, plan, staging=staging)
    return pools, _decision(out)


def test_host_staging_multi_bucket_matches_unstaged_bit_identical() -> None:
    # Two chunk geometries in ONE call invoke stage_bucket twice (once
    # per geometry); each geometry's dedicated buffer must not clobber
    # the other's in-flight copy. Bit-identical to the un-pooled path
    # proves the per-geometry bucket routing is correct.
    pools_unstaged, dec_unstaged = _run_multi_bucket(None)
    pools_staged, dec_staged = _run_multi_bucket(advance.HostStaging(8))
    assert dec_staged[0] == 9  # replay row advanced its queue
    assert dec_staged[2] != PARK_ID  # the final-chunk burst emitted
    assert dec_staged == dec_unstaged
    _assert_pools_equal(pools_staged, pools_unstaged)


def test_host_staging_stage_bucket_rejects_oversized_and_bad_geometry() -> None:
    staging = advance.HostStaging(4)
    with pytest.raises(ValueError, match="exceeds HostStaging capacity"):
        staging.stage_bucket(0, torch.arange(5, dtype=torch.int64), torch.device("cpu"))
    with pytest.raises(ValueError, match="outside the staging arena"):
        staging.stage_bucket(
            999, torch.arange(2, dtype=torch.int64), torch.device("cpu")
        )


def test_host_staging_slot_reuse_across_calls_is_safe() -> None:
    # ONE HostStaging instance drives two structurally different
    # transactions back to back; each call's own slots are fully
    # overwritten by stage(), so a later call must not observe an
    # earlier call's staged content.
    staging = advance.HostStaging(8)
    core = _tiny_core()
    replay_pools = _fresh_pools()
    _set_replay_book(replay_pools, 1, queue=[3, 5], head=1, expected=3)
    replay_plan = _plan(decodes=[1], pad_decodes_to=2, padding_value=NULL_INDEX)
    out1 = _call(
        core,
        replay_pools,
        torch.tensor([3], dtype=torch.long),
        torch.zeros(1, CARRIER_HIDDEN),
        replay_plan,
        staging=staging,
    )
    assert _decision(out1) == [5]
    assert int(replay_pools["book_pool"][1, _BOOK["queue_head"]]) == 2

    torch.manual_seed(5)
    carrier = _envelope(torch.randn(REG_SAMPLES) * 0.01, final=False, seq=0).unsqueeze(
        0
    )
    chunk_pools = _fresh_pools()
    chunk_plan = _plan(prefills=[1])
    out2 = _call(
        core,
        chunk_pools,
        torch.tensor([PLACEHOLDER_ID], dtype=torch.long),
        carrier,
        chunk_plan,
        staging=staging,
    )
    assert out2.shape == (1, CARRIER_HIDDEN)
    assert _decision(out2)[0] != PARK_ID  # a real chunk computed, unaffected by call 1


# @spec PORT-STATE-003
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_gather_continuing_rows_preserves_order_layout_and_isolation(
    noncontiguous: bool,
) -> None:
    pool = torch.arange(60, dtype=torch.float32).reshape(5, 3, 4)
    if noncontiguous:
        pool = pool.transpose(1, 2)
        assert not pool.is_contiguous()
    before = pool.clone()
    blocks = torch.tensor([4, 1, 3])
    gathered = advance._gather_initialized_rows(
        pool, blocks, torch.zeros(3, dtype=torch.bool)
    )
    torch.testing.assert_close(gathered, torch.stack([before[4], before[1], before[3]]))
    assert gathered.is_contiguous()
    assert gathered.dtype == pool.dtype
    assert gathered.device == pool.device
    assert gathered.data_ptr() != pool.data_ptr()
    gathered.fill_(-1)
    torch.testing.assert_close(pool, before)
    pool.fill_(99)
    assert torch.all(gathered == -1)


# @spec PORT-STATE-003
@pytest.mark.parametrize("all_fresh", [False, True])
def test_gather_fresh_poison_is_never_read(
    all_fresh: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = torch.full((5, 3, 4), float("nan"))
    pool[1].fill_(7)
    blocks = torch.tensor([4, 1, 3])
    fresh = torch.tensor([True, all_fresh, True])
    reads: list[int] = []
    original = torch.Tensor.index_select

    def observed(tensor: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
        if tensor is pool:
            assert dim == 0
            reads.extend(index.tolist())
        return original(tensor, dim, index)

    monkeypatch.setattr(torch.Tensor, "index_select", observed)
    gathered = advance._gather_initialized_rows(pool, blocks, fresh)
    expected = torch.zeros((3, 3, 4))
    if not all_fresh:
        expected[1].fill_(7)
    torch.testing.assert_close(gathered, expected)
    assert reads == ([] if all_fresh else [1])
    assert gathered.is_contiguous()
    gathered.fill_(-1)
    assert torch.isnan(pool[4]).all()
    assert torch.isnan(pool[3]).all()
    assert torch.all(pool[1] == 7)


def test_gather_zero_rows_preserves_empty_layout() -> None:
    pool = torch.full((5, 3, 4), float("nan"), dtype=torch.float64).transpose(1, 2)
    gathered = advance._gather_initialized_rows(
        pool, torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.bool)
    )
    assert gathered.shape == (0, 4, 3)
    assert gathered.dtype == pool.dtype
    assert gathered.device == pool.device
    assert gathered.is_contiguous()


# @spec PORT-SESS-016
def test_post_final_tail_park_echo_cannot_publish_flush_progress() -> None:
    core = _tiny_core()
    pools = _fresh_pools()
    pools["frontend_counter_pool"][1, _CTR["finalized"]] = 1
    pools["frontend_counter_pool"][1, _CTR["expected_chunk_sequence"]] = 1
    _set_drained_book(pools, 1, blank=core.blank_id)
    status = _CommitRecorder()
    _call(
        core,
        pools,
        torch.tensor([PARK_ID], dtype=torch.long),
        torch.zeros(1, CARRIER_HIDDEN),
        _plan(decodes=[1]),
        commit_sink=status,
    )
    assert int(status.staged[0][0]) == 0
    assert status.plans[0].progress[0, 2].item() == 0


# @spec PORT-SESS-016
def test_forced_eou_replay_and_park_echo_keep_unit_identity():
    """The installed forced EOU emits one label; its echo is REPLAY, not EOU."""
    from tests.models.nemotron_asr.test_plan_local import _sink_and_plan

    core = _tiny_core()
    pools = _fresh_pools()
    _call(
        core,
        pools,
        torch.tensor([PLACEHOLDER_ID]),
        _envelope(torch.zeros(REG_SAMPLES), final=False, seq=0).unsqueeze(0),
        _plan(prefills=[1]),
    )
    _set_drained_book(pools, 1, blank=core.blank_id)
    pools["endpoint_history_pool"] = torch.zeros(3, 8, dtype=torch.int32)
    pools["endpoint_book_pool"] = torch.zeros(3, 6, dtype=torch.int32)
    pools["endpoint_book_pool"][1, 4] = 1
    eou = VOCAB + 3
    adapter = advance.make_mrv1_adapter(
        hidden_size=CARRIER_HIDDEN, park_id=PARK_ID, blank_id=core.blank_id
    )
    roles = []

    def record_adapter(result, ctx):
        roles.append(int(ctx.roles[0]))
        return adapter(result, ctx)

    registry, _, registry_plan = _sink_and_plan()
    generation = registry_plan.bindings[0].admission_generation
    assert registry.publish_progress("a", generation, (1, 0, 2), starts_chunk=True) == (
        0,
        0,
        2,
    )
    keys, phases = [], []
    for token in (eou, eou, PARK_ID):
        status = _CommitRecorder()
        advance.advance_model_rows(
            core,
            torch.tensor([token]),
            torch.zeros(1, CARRIER_HIDDEN),
            _plan(decodes=[1]),
            adapter=record_adapter,
            decode_resolver=_fixed_resolver(),
            placeholder_id=PLACEHOLDER_ID,
            park_id=PARK_ID,
            eou_token_id=eou,
            commit_sink=status,
            **pools,
        )
        assert int(status.staged[0][0]) == 0
        raw = tuple(status.plans[0].progress[0].tolist())
        phases.append(raw[2])
        keys.append(registry.publish_progress("a", generation, raw, starts_chunk=False))
    assert roles == [advance.ROLE_EOU, advance.ROLE_REPLAY, advance.ROLE_FLUSH]
    assert phases == [4, 2, 0]
    assert keys == [None, (1, 1, 2), None]
    assert registry.publish_progress("a", generation, (2, 0, 2), starts_chunk=True) == (
        2,
        0,
        2,
    )
