# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU plan-provider and commit-sink contracts.

Registry lifecycle, binding minting, consume-once slots, RowPlan assembly,
and reserve/stage/collect tickets run against the real package. A registry-
minted plan must pass the canonical transaction structural preflight.
"""

from __future__ import annotations

import importlib
import types
from pathlib import Path
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
            "rnnt_cell",
            "rnnt",
            "decode_dispatch",
            "state_scatter",
            "advance",
            "plan",
            "commit_sink",
        )
    }
    return loaded


mods = _load_chain()
advance = mods["advance"]
plan_mod = mods["plan"]
commit_sink_mod = mods["commit_sink"]

PLACEHOLDER_ID = 9001
NUM_PROMPTS = 4
NUM_GEOMETRIES = 5
#: Round numbers chosen so the deadline-reconstruction math in the
#: tests below is easy to verify by hand: NOW_NS is exactly 10,000 ms
#: since epoch, and _DEFAULT_ADMISSION_MS matches it exactly — the
#: "zero elapsed time" baseline every other test's default assumes.
NOW_NS = 10_000_000_000
_DEFAULT_ADMISSION_MS = 10_000


def _header(
    *,
    valid: int = 1_280,
    geometry: int = 0,
    final: int = 0,
    prompt: int = 0,
    seq: int = 0,
    version: float = 2.0,
    admission_ms_mod: int = _DEFAULT_ADMISSION_MS,
) -> tuple[float, ...]:
    return (
        float(version),
        float(valid),
        float(geometry),
        float(final),
        float(prompt),
        float(seq),
        float(admission_ms_mod),
    )


def _row(
    req: str,
    block: int,
    *,
    chunk: bool = False,
    prior: bool = True,
    header: tuple[float, ...] | None = None,
) -> Any:
    if chunk and header is None:
        header = _header()
    return plan_mod.ObservedRow(
        request_id=req,
        block_id=block,
        scheduled_token_id=PLACEHOLDER_ID if chunk else 7,
        has_prior_state=prior,
        envelope_header=header,
    )


def _bind(registry: Any, rows: list[Any], *, now_ns: int = NOW_NS) -> Any:
    return registry.bind_rows(
        rows,
        placeholder_id=PLACEHOLDER_ID,
        num_prompts=NUM_PROMPTS,
        num_geometries=NUM_GEOMETRIES,
        now_ns=now_ns,
    )


# ---- CPU feature/header authority (PORT-ADV-003 / PORT-INT-003) ----------


def _feature(
    *,
    header: tuple[float, ...] | None = None,
    offset: int = 0,
    length: int = 1,
    modality: str = "audio",
    payload: torch.Tensor | None | object = ...,
) -> Any:
    if header is None:
        header = _header()
    if payload is ...:
        payload = torch.tensor(header, dtype=torch.float32)
    data = None if payload is None else {"audio": types.SimpleNamespace(data=payload)}
    return types.SimpleNamespace(
        data=data,
        modality=modality,
        identifier="same-complete-envelope",
        mm_position=types.SimpleNamespace(offset=offset, length=length),
    )


def _resolve_header(
    *,
    token: int = PLACEHOLDER_ID,
    computed: int = 0,
    features: list[Any] | None = None,
    scheduled: list[int] | None = None,
) -> tuple[float, ...] | None:
    return plan_mod.resolve_row_envelope_header(
        scheduled_token_id=token,
        placeholder_id=PLACEHOLDER_ID,
        num_computed_tokens=computed,
        mm_features=[] if features is None else features,
        scheduled_encoder_input_ids=[] if scheduled is None else scheduled,
    )


def test_cache_hit_and_cache_miss_resolve_the_same_chunk_header() -> None:
    # @spec PORT-ADV-003 / PORT-INT-003
    # A missing scheduled_encoder_inputs entry is a legal encoder-cache hit,
    # not evidence that the CHUNK feature is absent.
    feature = _feature()
    cache_hit = _resolve_header(features=[feature], scheduled=[])
    cache_miss = _resolve_header(features=[feature], scheduled=[0])
    assert cache_hit == cache_miss == _header()


def test_non_chunk_without_current_feature_has_no_header() -> None:
    # Carrier-free REPLAY/FLUSH remains legal.
    assert (
        _resolve_header(
            token=7,
            computed=1,
            features=[_feature(offset=0)],
            scheduled=[],
        )
        is None
    )


@pytest.mark.parametrize(
    ("features", "scheduled", "match"),
    [
        ([_feature(offset=1)], [], "exactly one overlapping"),
        ([_feature(), _feature()], [], "exactly one overlapping"),
        (
            [_feature(offset=0), _feature(offset=2)],
            [1],
            "same overlapping feature",
        ),
        ([_feature(), _feature(offset=2)], [0, 1], "at most one encoder input"),
        ([_feature(modality="video")], [], "audio"),
        ([_feature(payload=None)], [], "no kwargs payload"),
        (
            [_feature(payload=torch.empty(7, device="meta"))],
            [],
            "host-resident",
        ),
    ],
)
def test_chunk_feature_authority_rejects_structural_drift(
    features: list[Any],
    scheduled: list[int],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _resolve_header(features=features, scheduled=scheduled)


def test_non_chunk_rejects_overlapping_multimodal_feature() -> None:
    with pytest.raises(ValueError, match="non-CHUNK.*overlapping"):
        _resolve_header(token=7, features=[_feature()], scheduled=[])


def test_non_chunk_rejects_scheduled_encoder_input() -> None:
    with pytest.raises(ValueError, match="non-CHUNK.*encoder input"):
        _resolve_header(
            token=7,
            computed=1,
            features=[_feature(offset=0)],
            scheduled=[0],
        )


# ---- SessionRegistry lifecycle -------------------------------------------


def test_fresh_chunk_registers_with_monotonic_generation() -> None:
    registry = plan_mod.SessionRegistry()
    [b1] = _bind(registry, [_row("a", 3, chunk=True, prior=False)])
    [b2] = _bind(
        registry,
        [_row("b", 4, chunk=True, prior=False, header=_header(prompt=2))],
    )
    assert b1.admission_generation == 1
    assert b2.admission_generation == 2
    # Zero elapsed time (the default admission stamp matches NOW_NS
    # exactly) -> the reconstructed deadline is NOW_NS plus geometry
    # 0's cadence-period budget, never a raw now_ns passthrough.
    assert b1.is_chunk and b1.ready_deadline_ns == (
        NOW_NS + plan_mod._DEADLINE_BUDGET_NS_BY_GEOMETRY[0]
    )
    assert not b1.allow_prompt_transition
    assert b2.prompt_index == 2 and b2.prior_prompt_index == 2
    assert registry.live_block_ids() == [3, 4]


def test_duplicate_block_ownership_rejected() -> None:
    registry = plan_mod.SessionRegistry()
    _bind(registry, [_row("a", 3, chunk=True, prior=False)])
    with pytest.raises(ValueError, match="already owned"):
        _bind(registry, [_row("b", 3, chunk=True, prior=False)])


def test_readmission_of_registered_request_rejected() -> None:
    registry = plan_mod.SessionRegistry()
    _bind(registry, [_row("a", 3, chunk=True, prior=False)])
    with pytest.raises(ValueError, match="re-admitted"):
        _bind(registry, [_row("a", 3, chunk=True, prior=False)])


def test_block_remap_rejected_before_state_access() -> None:
    registry = plan_mod.SessionRegistry()
    _bind(registry, [_row("a", 3, chunk=True, prior=False)])
    with pytest.raises(ValueError, match="remapped"):
        _bind(registry, [_row("a", 5)])


def test_unregistered_continuation_rejected() -> None:
    registry = plan_mod.SessionRegistry()
    with pytest.raises(ValueError, match="not.*registered"):
        _bind(registry, [_row("ghost", 3)])


def test_non_chunk_row_with_envelope_rejected() -> None:
    registry = plan_mod.SessionRegistry()
    _bind(registry, [_row("a", 3, chunk=True, prior=False)])
    with pytest.raises(ValueError, match="carries a scheduled envelope"):
        _bind(registry, [_row("a", 3, header=_header(seq=1))])


def test_duplicate_request_in_one_step_rejected() -> None:
    registry = plan_mod.SessionRegistry()
    _bind(registry, [_row("a", 3, chunk=True, prior=False)])
    with pytest.raises(ValueError, match="twice"):
        _bind(registry, [_row("a", 3), _row("a", 3)])


def test_bind_rows_is_atomic_when_a_later_row_is_invalid() -> None:
    registry = plan_mod.SessionRegistry()
    with pytest.raises(ValueError, match="already owned"):
        _bind(
            registry,
            [
                _row("a", 3, chunk=True, prior=False),
                _row("b", 3, chunk=True, prior=False),
            ],
        )
    assert len(registry) == 0
    assert registry.live_block_ids() == []
    [binding] = _bind(registry, [_row("c", 4, chunk=True, prior=False)])
    assert binding.admission_generation == 1


def test_prepare_context_rolls_back_prune_when_binding_fails() -> None:
    registry = plan_mod.SessionRegistry()
    _bind(registry, [_row("a", 3, chunk=True, prior=False)])
    _bind(registry, [_row("b", 4, chunk=True, prior=False)])
    with pytest.raises(ValueError, match="already owned"):
        plan_mod.prepare_plan_context(
            registry,
            [
                _row("c", 5, chunk=True, prior=False),
                _row("d", 5, chunk=True, prior=False),
            ],
            resident_request_ids=["a", "c", "d"],
            placeholder_id=PLACEHOLDER_ID,
            num_prompts=NUM_PROMPTS,
            num_geometries=NUM_GEOMETRIES,
            now_ns=NOW_NS,
            step=2,
        )
    assert registry.live_block_ids() == [3, 4]


def test_prepare_context_rejects_rows_missing_from_resident_authority() -> None:
    registry = plan_mod.SessionRegistry()
    with pytest.raises(ValueError, match="absent from worker-resident"):
        plan_mod.prepare_plan_context(
            registry,
            [_row("a", 3, chunk=True, prior=False)],
            resident_request_ids=[],
            placeholder_id=PLACEHOLDER_ID,
            num_prompts=NUM_PROMPTS,
            num_geometries=NUM_GEOMETRIES,
            now_ns=NOW_NS,
            step=1,
        )
    assert len(registry) == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"version": 9.0}, "version"),
        ({"valid": -1}, "negative valid-sample"),
        ({"geometry": 7}, "outside the admitted set"),
        ({"prompt": 9}, "outside the prompt dictionary"),
        ({"final": 2}, "non-boolean final"),
        ({"seq": 1.5}, "not an exact integer"),
        ({"admission_ms_mod": -1}, "admission_ms_mod"),
        (
            {"admission_ms_mod": plan_mod.ADMISSION_EPOCH_MODULUS_MS},
            "admission_ms_mod",
        ),
    ],
)
def test_malformed_minted_header_is_a_loud_port_defect(
    mutation: dict[str, Any], message: str
) -> None:
    registry = plan_mod.SessionRegistry()
    with pytest.raises(ValueError, match=message):
        _bind(
            registry,
            [
                _row(
                    "a",
                    3,
                    chunk=True,
                    prior=False,
                    header=_header(**mutation),
                )
            ],
        )


def test_prompt_transition_flags_exactly_on_change() -> None:
    registry = plan_mod.SessionRegistry()
    _bind(registry, [_row("a", 3, chunk=True, prior=False)])
    # Same prompt on the next minted CHUNK: no transition authorized.
    [same] = _bind(registry, [_row("a", 3, chunk=True, header=_header(seq=1))])
    assert not same.allow_prompt_transition
    # A locale update surfaces on the next minted CHUNK's header.
    [changed] = _bind(
        registry,
        [_row("a", 3, chunk=True, header=_header(prompt=2, seq=2))],
    )
    assert changed.allow_prompt_transition
    assert changed.prior_prompt_index == 0
    assert changed.prompt_index == 2
    with pytest.raises(ValueError, match="status.*pending"):
        _bind(registry, [_row("a", 3)])
    registry.resolve_status(
        request_id="a",
        admission_generation=changed.admission_generation,
        clean=True,
    )
    # REPLAY continuation carries the persisted (new) prompt, never a
    # retroactive selection, and can never authorize a transition.
    [replay] = _bind(registry, [_row("a", 3)])
    assert not replay.is_chunk
    assert not replay.allow_prompt_transition
    assert replay.prompt_index == 2 and replay.prior_prompt_index == 2
    assert replay.ready_deadline_ns == 0


def test_failed_prompt_transition_preserves_prior_prompt() -> None:
    registry = plan_mod.SessionRegistry()
    _bind(registry, [_row("a", 3, chunk=True, prior=False)])
    [changed] = _bind(
        registry,
        [_row("a", 3, chunk=True, header=_header(prompt=2, seq=1))],
    )
    registry.resolve_status(
        request_id="a",
        admission_generation=changed.admission_generation,
        clean=False,
    )
    [replay] = _bind(registry, [_row("a", 3)])
    assert replay.prompt_index == 0
    assert replay.prior_prompt_index == 0


def test_outer_full_decode_graph_mode_is_rejected() -> None:
    class _RuntimeMode:
        name = "FULL"

    class _ConfiguredMode:
        def decode_mode(self) -> _RuntimeMode:
            return _RuntimeMode()

    with pytest.raises(ValueError, match="full CUDA graph"):
        plan_mod.reject_unsupported_outer_graph_mode(
            type("Compilation", (), {"cudagraph_mode": _ConfiguredMode()})()
        )


def test_outer_none_and_piecewise_graph_modes_are_accepted() -> None:
    for name in ("NONE", "PIECEWISE"):
        runtime = type("Runtime", (), {"name": name})()
        configured = type(
            "Configured",
            (),
            {"decode_mode": lambda self, value=runtime: value},
        )()
        plan_mod.reject_unsupported_outer_graph_mode(
            type("Compilation", (), {"cudagraph_mode": configured})()
        )


def test_prune_releases_sessions_and_blocks() -> None:
    registry = plan_mod.SessionRegistry()
    _bind(registry, [_row("a", 3, chunk=True, prior=False)])
    _bind(registry, [_row("b", 4, chunk=True, prior=False)])
    registry.prune(["b"])
    assert registry.live_block_ids() == [4]
    # Block 3 is reusable by a NEW session after release.
    [b3] = _bind(registry, [_row("c", 3, chunk=True, prior=False)])
    assert b3.admission_generation == 3  # ABA guard advanced


def test_lease_validation_tracks_registry_currency() -> None:
    registry = plan_mod.SessionRegistry()
    [binding] = _bind(registry, [_row("a", 3, chunk=True, prior=False)])
    registry.validate_lease([binding])
    assert registry.lease_is_current([binding])
    registry.prune([])
    with pytest.raises(ValueError, match="lease"):
        registry.validate_lease([binding])
    assert not registry.lease_is_current([binding])


# ---- ingress-deadline reconstruction (design §Ingress-deadline plumbing) --


def test_deadline_reconstruction_is_now_plus_geometry_budget_at_zero_elapsed() -> None:
    registry = plan_mod.SessionRegistry()
    for geometry in range(NUM_GEOMETRIES):
        [binding] = _bind(
            registry,
            [
                _row(
                    f"g{geometry}",
                    geometry + 1,
                    chunk=True,
                    prior=False,
                    header=_header(geometry=geometry),
                )
            ],
        )
        expected = NOW_NS + plan_mod._DEADLINE_BUDGET_NS_BY_GEOMETRY[geometry]
        assert binding.ready_deadline_ns == expected
    # Budgets strictly increase with cadence — a longer-cadence row's
    # deadline is always later than a shorter one admitted at the
    # same instant (PORT-PERF-001 cross-geometry ordering).
    assert list(plan_mod._DEADLINE_BUDGET_NS_BY_GEOMETRY) == sorted(
        plan_mod._DEADLINE_BUDGET_NS_BY_GEOMETRY
    )


def test_deadline_reconstruction_accounts_for_elapsed_time() -> None:
    # Admitted 5ms before "now": the reconstructed admission instant
    # is now_ns - 5ms, shifting the deadline back from the
    # zero-elapsed baseline by exactly that much.
    registry = plan_mod.SessionRegistry()
    elapsed_ms = 5
    header = _header(admission_ms_mod=_DEFAULT_ADMISSION_MS - elapsed_ms)
    [binding] = _bind(registry, [_row("a", 3, chunk=True, prior=False, header=header)])
    expected = (
        NOW_NS - elapsed_ms * 1_000_000 + plan_mod._DEADLINE_BUDGET_NS_BY_GEOMETRY[0]
    )
    assert binding.ready_deadline_ns == expected


def test_deadline_reconstruction_handles_wraparound() -> None:
    # now_ms_mod effectively 2 (just past a wrap); admission was
    # minted 3ms earlier at ms_mod == modulus - 1 (just before the
    # wrap) -> true elapsed is 3ms, not a huge negative-then-wrong
    # value from a naive unmodded subtraction.
    registry = plan_mod.SessionRegistry()
    modulus = plan_mod.ADMISSION_EPOCH_MODULUS_MS
    now_ns = (modulus + 2) * 1_000_000
    header = _header(admission_ms_mod=modulus - 1)
    [binding] = _bind(
        registry,
        [_row("a", 3, chunk=True, prior=False, header=header)],
        now_ns=now_ns,
    )
    expected = now_ns - 3 * 1_000_000 + plan_mod._DEADLINE_BUDGET_NS_BY_GEOMETRY[0]
    assert binding.ready_deadline_ns == expected


def test_deadline_reconstruction_continuing_chunk_uses_session_geometry() -> None:
    # A continuing CHUNK's budget comes from the SESSION's admitted
    # geometry (host authority), not the envelope's parsed value —
    # mirrors the admitted-prompt authority pattern; the device layer
    # cross-checks the envelope separately (ROW_STATUS_GEOMETRY). The
    # envelope here deliberately claims a DIFFERENT (but still
    # in-bounds) geometry than the session's, so this test actually
    # discriminates "used session.geometry_id" from "used the
    # freshly-parsed envelope value" — a prior version of this test
    # used the same geometry in both headers and would have passed
    # under either implementation.
    registry = plan_mod.SessionRegistry()
    _bind(
        registry,
        [_row("a", 3, chunk=True, prior=False, header=_header(geometry=2))],
    )
    [binding] = _bind(
        registry,
        [_row("a", 3, chunk=True, header=_header(geometry=4, seq=1))],
    )
    expected = NOW_NS + plan_mod._DEADLINE_BUDGET_NS_BY_GEOMETRY[2]
    assert binding.ready_deadline_ns == expected
    assert (
        plan_mod._DEADLINE_BUDGET_NS_BY_GEOMETRY[2]
        != plan_mod._DEADLINE_BUDGET_NS_BY_GEOMETRY[4]
    )


# ---- consume-once slot and context assembly -------------------------------


def _context(registry: Any, rows: list[Any], *, step: int = 1) -> Any:
    return plan_mod.prepare_plan_context(
        registry,
        rows,
        resident_request_ids=[row.request_id for row in rows],
        placeholder_id=PLACEHOLDER_ID,
        num_prompts=NUM_PROMPTS,
        num_geometries=NUM_GEOMETRIES,
        now_ns=NOW_NS,
        step=step,
    )


def test_slot_is_consume_once_and_replaces_stale_context() -> None:
    registry = plan_mod.SessionRegistry()
    slot = plan_mod.PlanContextSlot()
    first = _context(registry, [_row("a", 3, chunk=True, prior=False)], step=1)
    slot.stage(first)
    second = _context(registry, [_row("a", 3)], step=2)
    slot.stage(second)  # a stranded context is replaced, never leaked
    consumed = slot.consume()
    assert consumed.step == 2
    with pytest.raises(ValueError, match="no staged PlanContext"):
        slot.consume()


def test_context_columns_mirror_bindings_exactly() -> None:
    registry = plan_mod.SessionRegistry()
    context = _context(
        registry,
        [
            _row("a", 3, chunk=True, prior=False, header=_header(prompt=1, geometry=2)),
            _row("b", 4, chunk=True, prior=False),
        ],
    )
    assert context.request_ids == ("a", "b")
    assert context.block_ids.tolist() == [3, 4]
    assert context.geometry_id.tolist() == [2, 0]
    assert context.prompt_index.tolist() == [1, 0]
    assert context.is_chunk.tolist() == [True, True]
    assert context.admission_generation.tolist() == [1, 2]
    assert context.live_block_ids.tolist() == [3, 4]
    for row, binding in enumerate(context.bindings):
        assert binding.request_id == context.request_ids[row]
        assert binding.block_id == int(context.block_ids[row])


# ---- RowPlan assembly + the transaction's own preflight -------------------


def test_registry_minted_plan_passes_transaction_preflight() -> None:
    # The decisive integration pin: the plan the provider builds must
    # satisfy the transaction's structural preflight, which re-proves
    # every PreparedRowBinding from the plan columns (PORT-ADV-003).
    registry = plan_mod.SessionRegistry()
    # Establish an ongoing session (decode row) + a fresh prefill.
    _bind(registry, [_row("a", 1, chunk=True, prior=False)])
    context = _context(
        registry,
        [
            _row("a", 1, chunk=True, header=_header(seq=1)),
            _row("b", 2, chunk=True, prior=False),
        ],
    )
    plan = plan_mod.build_row_plan(
        context,
        num_decodes=1,
        num_prefills=1,
        null_block_id=0,
        num_pool_blocks=4,
    )
    idx = advance._structural_preflight(plan, 2, 2)
    assert idx.tolist() == [1, 2]
    assert plan.has_initial_states_p.tolist() == [False]
    assert plan.request_ids == ("a", "b")


def test_build_row_plan_rejects_count_and_freshness_drift() -> None:
    registry = plan_mod.SessionRegistry()
    context = _context(registry, [_row("a", 1, chunk=True, prior=False)])
    with pytest.raises(ValueError, match="disagree"):
        plan_mod.build_row_plan(
            context,
            num_decodes=1,
            num_prefills=1,
            null_block_id=0,
            num_pool_blocks=4,
        )
    with pytest.raises(ValueError, match="decode row has no prior"):
        plan_mod.build_row_plan(
            context,
            num_decodes=1,
            num_prefills=0,
            null_block_id=0,
            num_pool_blocks=4,
        )


# ---- the production composite sink ----------------------------------------


def _sink_and_plan() -> tuple[Any, Any, Any]:
    registry = plan_mod.SessionRegistry()
    [binding] = _bind(registry, [_row("a", 3, chunk=True, prior=False)])
    sink = commit_sink_mod.BoundedCommitSink(registry, max_rows=4)
    commit_plan = advance.CommitPlan(bindings=(binding,), capture=None)
    return registry, sink, commit_plan


def test_sink_reserve_stage_collect_roundtrip() -> None:
    registry, sink, commit_plan = _sink_and_plan()
    ticket = sink.reserve(commit_plan)
    status = torch.tensor([0], dtype=torch.int32)
    ticket.stage(status)
    reports, committed, lease_ok = sink.collect()
    assert [r.request_id for r in reports] == ["a"]
    assert reports[0].row_status == 0
    assert reports[0].admission_generation == 1
    assert committed == [] and lease_ok


def test_sink_exposes_only_status_clean_records() -> None:
    registry = plan_mod.SessionRegistry()
    bindings = _bind(
        registry,
        [
            _row("a", 3, chunk=True, prior=False),
            _row("b", 4, chunk=True, prior=False),
        ],
    )
    sink = commit_sink_mod.BoundedCommitSink(
        registry,
        max_rows=4,
        max_capture_rows=4,
        max_capture_bytes=1 << 20,
    )
    records = [
        _record("a", 3, 1, row=0),
        _record("b", 4, 2, row=1),
    ]
    commit_plan = advance.CommitPlan(
        bindings=tuple(bindings),
        capture=advance.CapturePlan(rows=2, payload_bytes=_payload_bytes(records)),
        records=tuple(records),
    )
    ticket = sink.reserve(commit_plan)
    ticket.stage(torch.tensor([512, 0], dtype=torch.int32))
    reports, committed, lease_ok = sink.collect()
    assert [r.row_status for r in reports] == [512, 0]
    assert [r.request_id for r in committed] == ["b"]
    assert lease_ok


def _record(req: str, block: int, generation: int, *, row: int = 0) -> Any:
    scalar = torch.zeros((), dtype=torch.int64)
    return advance.CaptureRecord(
        row=row,
        request_id=req,
        block_id=block,
        admission_generation=generation,
        geometry=0,
        chunk_sequence=scalar,
        prompt_index=scalar,
        row_status=torch.zeros((), dtype=torch.int32),
        frontend_mel=torch.zeros(2, 3),
        mel_length=scalar,
        encoder_raw=torch.zeros(1, 2),
        encoder_conditioned=torch.zeros(1, 2),
        encoder_length=scalar,
    )


def _payload_bytes(records: list[Any]) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for record in records
        for tensor in (
            record.chunk_sequence,
            record.prompt_index,
            record.row_status,
            record.frontend_mel,
            record.mel_length,
            record.encoder_raw,
            record.encoder_conditioned,
            record.encoder_length,
        )
    )


def test_sink_rejects_second_reservation_until_collected() -> None:
    registry, sink, commit_plan = _sink_and_plan()
    ticket = sink.reserve(commit_plan)
    with pytest.raises(ValueError, match="unconsumed reservation"):
        sink.reserve(commit_plan)
    ticket.cancel()
    ticket.cancel()  # idempotent
    sink.reserve(commit_plan)  # released by cancel


def test_sink_reserve_validates_lease_and_bounds() -> None:
    registry, sink, commit_plan = _sink_and_plan()
    registry.prune([])
    with pytest.raises(ValueError, match="lease"):
        sink.reserve(commit_plan)
    registry2 = plan_mod.SessionRegistry()
    bindings = _bind(
        registry2,
        [_row(f"r{i}", i + 1, chunk=True, prior=False) for i in range(3)],
    )
    small = commit_sink_mod.BoundedCommitSink(registry2, max_rows=2)
    with pytest.raises(ValueError, match="exceed the sink bound"):
        small.reserve(advance.CommitPlan(bindings=tuple(bindings), capture=None))
    capped = commit_sink_mod.BoundedCommitSink(
        registry2, max_rows=4, max_capture_rows=0, max_capture_bytes=0
    )
    with pytest.raises(ValueError, match="capture rows exceed"):
        capped.reserve(
            advance.CommitPlan(
                bindings=tuple(bindings),
                capture=advance.CapturePlan(rows=1, payload_bytes=8),
            )
        )


def test_sink_reserve_rejects_capture_identity_drift() -> None:
    registry = plan_mod.SessionRegistry()
    [binding] = _bind(registry, [_row("a", 3, chunk=True, prior=False)])
    sink = commit_sink_mod.BoundedCommitSink(
        registry,
        max_rows=2,
        max_capture_rows=2,
        max_capture_bytes=1 << 20,
    )
    bad = _record("other", 3, binding.admission_generation)
    with pytest.raises(ValueError, match="capture record identity"):
        sink.reserve(
            advance.CommitPlan(
                bindings=(binding,),
                capture=advance.CapturePlan(
                    rows=1, payload_bytes=_payload_bytes([bad])
                ),
                records=(bad,),
            )
        )


def test_sink_stage_records_lease_break_without_raising() -> None:
    # Nothing may fail after resident state committed: a lease broken
    # between reserve and stage is RECORDED, never raised.
    registry, sink, commit_plan = _sink_and_plan()
    ticket = sink.reserve(commit_plan)
    registry.prune([])  # simulate a lease break inside the window
    ticket.stage(torch.tensor([0], dtype=torch.int32))
    _, _, lease_ok = sink.collect()
    assert not lease_ok


def test_sink_collect_without_stage_raises() -> None:
    registry, sink, commit_plan = _sink_and_plan()
    sink.reserve(commit_plan)
    with pytest.raises(ValueError, match="no staged commit"):
        sink.collect()


def test_geometry_only_status_preserves_session_and_prompt() -> None:
    registry = plan_mod.SessionRegistry()
    _bind(registry, [_row("a", 3, chunk=True, prior=False)])
    [changed] = _bind(
        registry,
        [_row("a", 3, chunk=True, header=_header(prompt=2, seq=1))],
    )
    sink = commit_sink_mod.BoundedCommitSink(registry, max_rows=2)
    ticket = sink.reserve(advance.CommitPlan(bindings=(changed,), capture=None))
    ticket.stage(torch.tensor([1], dtype=torch.int32))
    reports, _, lease_ok = sink.collect()

    statuses, failed = commit_sink_mod.resolve_status_reports(
        registry, reports, lease_ok=lease_ok
    )

    assert statuses == {"a": 1}
    assert failed == set()
    [replay] = _bind(registry, [_row("a", 3)])
    assert replay.prompt_index == 0


def test_nonrecoverable_status_is_terminal() -> None:
    registry, sink, commit_plan = _sink_and_plan()
    ticket = sink.reserve(commit_plan)
    ticket.stage(torch.tensor([512], dtype=torch.int32))
    reports, _, lease_ok = sink.collect()

    statuses, failed = commit_sink_mod.resolve_status_reports(
        registry, reports, lease_ok=lease_ok
    )

    assert statuses == {"a": 512}
    assert failed == {"a"}


# @spec PORT-SESS-016
def test_commit_sink_publishes_progress_only_after_clean_commit() -> None:
    registry, sink, plan = _sink_and_plan()
    progress = torch.tensor([[1, 0, 2]], dtype=torch.int64)
    ticket = sink.reserve(
        advance.CommitPlan(bindings=plan.bindings, capture=None, progress=progress)
    )
    ticket.stage(torch.zeros(1, dtype=torch.int32))
    reports, _, lease_ok = sink.collect()
    assert lease_ok
    assert reports[0].progress_key == (1, 0, 2)


# @spec PORT-SESS-016
def test_model_progress_orders_chunk_forced_control_replay_and_flush() -> None:
    registry, sink, plan = _sink_and_plan()
    generation = plan.bindings[0].admission_generation
    publish = registry.publish_progress
    assert publish("a", generation, (1, 0, 2), starts_chunk=True) == (0, 0, 2)
    # An empty forced control still owns its own logical sequence.
    assert publish("a", generation, (1, 0, 5), starts_chunk=False) == (1, 0, 2)
    assert publish("a", generation, (2, 1, 1), starts_chunk=True) == (2, 1, 1)
    assert publish("a", generation, (2, 3, 1), starts_chunk=False) == (2, 3, 1)
    assert publish("a", generation, (2, 2, 1), starts_chunk=False) is None
    assert publish("a", generation, (2, 3, 2), starts_chunk=False) == (2, 3, 2)
    assert publish("a", generation, (2, 3, 3), starts_chunk=False) == (3, 3, 3)
    assert publish("a", generation, (2, 3, 3), starts_chunk=False) is None
    assert publish("a", generation + 1, (3, 0, 2), starts_chunk=True) is None


# @spec PORT-SESS-016
def test_forced_control_commit_waits_for_legal_park_to_raise_progress() -> None:
    registry, _, plan = _sink_and_plan()
    generation = plan.bindings[0].admission_generation
    assert registry.publish_progress("a", generation, (1, 3, 2), starts_chunk=True) == (
        0,
        3,
        2,
    )
    assert (
        registry.publish_progress("a", generation, (1, 1, 4), starts_chunk=False)
        is None
    )
    assert registry.publish_progress(
        "a", generation, (1, 1, 2), starts_chunk=False
    ) == (1, 1, 2)


# @spec PORT-SESS-016
def test_forced_eou_across_two_replay_waves_counts_one_unit():
    """A forced-control unit retains its identity across multiple cursor advances."""
    registry, _, plan = _sink_and_plan()
    generation = plan.bindings[0].admission_generation
    publish = registry.publish_progress
    assert publish("a", generation, (1, 3, 2), starts_chunk=True) == (0, 3, 2)
    assert publish("a", generation, (1, 1, 4), starts_chunk=False) is None
    assert publish("a", generation, (1, 2, 1), starts_chunk=False) == (1, 2, 1)
    assert publish("a", generation, (1, 3, 1), starts_chunk=False) == (1, 3, 1)
    assert publish("a", generation, (1, 3, 2), starts_chunk=False) == (1, 3, 2)
    assert publish("a", generation, (2, 0, 2), starts_chunk=True) == (2, 0, 2)
