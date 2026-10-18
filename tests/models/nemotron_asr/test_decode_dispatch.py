# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch-table generator contracts (PORT-DEC-008, dispatch-table
form; decisions/decode-dispatch-regime.md r3.2).

The generator consumes a p6b_decode_profile artifact and COMPUTES
selections — never eyeballed brackets: lanes come from
``generator_eligible_lanes``; compact may be selected only from
SELECTIVE cells and only when it beats the sync-free arm by the
recorded hysteresis at EVERY dcp activity level; nonselective cells
force the sync-free arm; unmeasured tiers resolve to the nearest
measured tier, sync-free on bracketing disagreement; a profile whose
engine graph covers decode never dispatches the capture-ineligible
candidate; fingerprint mismatch fails closed for performance-gated
deployments. Pure host logic — no torch, runs anywhere.
"""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]

_PKG = Path(__file__).resolve().parents[3] / "vllm/models/nemotron_asr"
_BASE = "vllm.models.nemotron_asr"
_mod = importlib.import_module(f"{_BASE}.decode_dispatch")

DispatchTable = _mod.DispatchTable
FingerprintMismatchError = _mod.FingerprintMismatchError
generate_dispatch_table = _mod.generate_dispatch_table


def _cell(
    geometry: str,
    batch: int,
    lane: str,
    arm: str,
    activity: str,
    latency_us: float,
    *,
    selective: bool | None = None,
) -> dict[str, Any]:
    if selective is None:
        selective = activity in ("silence", "speech")
    return {
        "geometry": geometry,
        "batch": batch,
        "precision": lane,
        "arm": arm,
        "activity": activity,
        "latency_us": latency_us,
        "policy_realized": selective,
        "selective": selective,
        "tokens_match": True,
    }


def _report(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "probe": "p6b_decode_profile",
        "execution_ok": True,
        "fingerprint": {
            "probe_schema": "p6b-decode-profile-v3",
            "decode_algo_revision": "decode-algo-v1",
            "lane_definitions_digest": "sha256:lanes",
            "device_name": "NVIDIA L4",
            "driver": "580.126.20",
            "torch": "2.11.0+cu130",
            "tf32_matmul": False,
            "model_shape_digest": "sha256:abc",
            "fork_commit": "deadbeef",
            "dirty_tree": False,
            "config_asserted": True,
            "untracked_files": 1,
            "untracked_paths": ["abc123model-download.lock"],
        },
        "dcp": {
            "version": "dcp-v1",
            "targets": {"silence": 0.3, "speech": 3.0},
            "tolerance": {
                "silence": (0.0, 1.0),
                "speech": (1.5, 4.5),
            },
            "selective_activities": ("silence", "speech"),
            "fallback_rule": "nonselective -> sync-free",
        },
        "lane_results": {
            "fp32": {
                "mismatch_cells": 0,
                "performance_qualified": True,
                "dispatch_eligible": True,
            },
        },
        "generator_eligible_lanes": ["fp32"],
        "cells": cells,
    }


def _tier_cells(
    geometry: str,
    batch: int,
    *,
    dense_eager: float,
    dense_graphed: float,
    compact_silence: float,
    compact_speech: float,
    lane: str = "fp32",
    selective: bool = True,
) -> list[dict[str, Any]]:
    return [
        _cell(geometry, batch, lane, "dense-eager", "speech", dense_eager),
        _cell(geometry, batch, lane, "dense-graphed", "speech", dense_graphed),
        _cell(
            geometry,
            batch,
            lane,
            "compact-eager",
            "silence",
            compact_silence,
            selective=selective,
        ),
        _cell(
            geometry,
            batch,
            lane,
            "compact-eager",
            "speech",
            compact_speech,
            selective=selective,
        ),
    ]


# A two-tier fixture: small tier where dense dominates, large tier
# where compact beats dense-eager by far more than the hysteresis at
# both dcp levels.
def _two_tier_report() -> dict[str, Any]:
    cells = _tier_cells(
        "1120ms",
        8,
        dense_eager=140_000,
        dense_graphed=65_000,
        compact_silence=100_000,
        compact_speech=115_000,
    ) + _tier_cells(
        "1120ms",
        1024,
        dense_eager=890_000,
        dense_graphed=880_000,
        compact_silence=120_000,
        compact_speech=230_000,
    )
    return _report(cells)


def test_lanes_come_from_generator_eligible_lanes() -> None:
    table = generate_dispatch_table(_two_tier_report())
    assert table.lanes == ("fp32",)
    with pytest.raises(KeyError):
        table.select(
            "1120ms",
            8,
            lane="bf16-joint",
            graph_covers_decode=False,
        )


def test_graph_covered_profile_never_dispatches_compact() -> None:
    # Sanity invariant, not a measured outcome: capture-ineligible
    # compact never enters a profile whose graph covers decode —
    # even at the tier where compact wins eager.
    table = generate_dispatch_table(_two_tier_report())
    assert (
        table.select(
            "1120ms",
            1024,
            lane="fp32",
            graph_covers_decode=True,
        )
        == "dense-graphed"
    )


def test_compact_needs_every_dcp_level_beyond_hysteresis() -> None:
    table = generate_dispatch_table(_two_tier_report())
    # Large tier: compact beats dense-eager by >25% at BOTH levels.
    assert (
        table.select(
            "1120ms",
            1024,
            lane="fp32",
            graph_covers_decode=False,
        )
        == "compact-eager"
    )
    # Small tier: compact loses to the best sync-free arm.
    assert (
        table.select(
            "1120ms",
            8,
            lane="fp32",
            graph_covers_decode=False,
        )
        == "dense-eager"
    )


def test_hysteresis_margin_blocks_thin_wins() -> None:
    report = _two_tier_report()
    # Make compact's speech win thinner than 25% at the large tier.
    for cell in report["cells"]:
        if (
            cell["batch"] == 1024
            and cell["arm"] == "compact-eager"
            and cell["activity"] == "speech"
        ):
            cell["latency_us"] = 700_000  # wins, but only ~21%
    table = generate_dispatch_table(report)
    assert (
        table.select(
            "1120ms",
            1024,
            lane="fp32",
            graph_covers_decode=False,
        )
        == "dense-eager"
    )


def test_nonselective_cell_forces_sync_free() -> None:
    report = _two_tier_report()
    for cell in report["cells"]:
        if (
            cell["batch"] == 1024
            and cell["arm"] == "compact-eager"
            and cell["activity"] == "silence"
        ):
            cell["policy_realized"] = False
            cell["selective"] = False
    table = generate_dispatch_table(report)
    assert (
        table.select(
            "1120ms",
            1024,
            lane="fp32",
            graph_covers_decode=False,
        )
        == "dense-eager"
    )


def test_unmeasured_tier_resolves_nearest_measured() -> None:
    # Nearest-tier resolution applies only when the bracketing tiers
    # AGREE (PORT-DEC-008; disagreement is the crossover-gap rule,
    # tested separately) — so give 900 agreeing compact brackets.
    report = _two_tier_report()
    report["cells"] += _tier_cells(
        "1120ms",
        512,
        dense_eager=430_000,
        dense_graphed=420_000,
        compact_silence=110_000,
        compact_speech=210_000,
    )
    table = generate_dispatch_table(report)
    # Below the lowest tier: only one side exists — dense tier 8.
    assert (
        table.select(
            "1120ms",
            2,
            lane="fp32",
            graph_covers_decode=False,
        )
        == "dense-eager"
    )
    # 900 sits between 512 and 1024, BOTH compact — nearest (1024).
    assert (
        table.select(
            "1120ms",
            900,
            lane="fp32",
            graph_covers_decode=False,
        )
        == "compact-eager"
    )


def test_bracketing_disagreement_selects_sync_free() -> None:
    table = generate_dispatch_table(_two_tier_report())
    # 516 sits between tier 8 (dense) and tier 1024 (compact):
    # nearest is 1024, but the bracketing tiers disagree — the
    # crossover-gap rule selects the sync-free arm.
    assert (
        table.select(
            "1120ms",
            516,
            lane="fp32",
            graph_covers_decode=False,
        )
        == "dense-eager"
    )


def test_unknown_geometry_is_an_error() -> None:
    table = generate_dispatch_table(_two_tier_report())
    with pytest.raises(KeyError):
        table.select(
            "999ms",
            8,
            lane="fp32",
            graph_covers_decode=False,
        )


def test_fingerprint_mismatch_fails_closed_when_gated() -> None:
    table = generate_dispatch_table(_two_tier_report())
    runtime = dict(table.fingerprint)
    runtime["device_name"] = "NVIDIA A100 80GB PCIe"
    with pytest.raises(FingerprintMismatchError):
        table.validate_runtime(runtime, performance_gated=True)
    # Dev profile: warns (returns the mismatched keys), no raise.
    assert "device_name" in table.validate_runtime(
        runtime,
        performance_gated=False,
    )


def test_table_records_selection_constants() -> None:
    table = generate_dispatch_table(_two_tier_report())
    assert table.hysteresis_pct == 25.0
    assert table.policy_version == "dcp-v1"
    assert table.fingerprint["model_shape_digest"] == "sha256:abc"


def test_dirty_tree_artifact_is_rejected() -> None:
    report = _two_tier_report()
    report["fingerprint"]["dirty_tree"] = True
    with pytest.raises(ValueError, match="dirty"):
        generate_dispatch_table(report)


def test_old_schema_without_lane_results_is_rejected() -> None:
    report = _two_tier_report()
    del report["lane_results"]
    del report["generator_eligible_lanes"]
    with pytest.raises(ValueError, match="lane"):
        generate_dispatch_table(report)


def test_ineligible_lane_cells_are_ignored_entirely() -> None:
    report = _two_tier_report()
    # Add bf16 cells where compact "wins" hugely — they must not
    # produce entries (the lane is not generator-eligible).
    report["cells"] += _tier_cells(
        "1120ms",
        1024,
        lane="bf16-joint",
        dense_eager=900_000,
        dense_graphed=880_000,
        compact_silence=50_000,
        compact_speech=60_000,
    )
    table = generate_dispatch_table(report)
    assert table.lanes == ("fp32",)


def test_missing_activity_level_forces_sync_free() -> None:
    report = _two_tier_report()
    report["cells"] = [
        c
        for c in report["cells"]
        if not (
            c["batch"] == 1024
            and c["arm"] == "compact-eager"
            and c["activity"] == "silence"
        )
    ]
    table = generate_dispatch_table(report)
    # Compact cannot prove the worst case without BOTH dcp levels.
    assert (
        table.select(
            "1120ms",
            1024,
            lane="fp32",
            graph_covers_decode=False,
        )
        == "dense-eager"
    )


def test_serialization_round_trip() -> None:
    table = generate_dispatch_table(_two_tier_report())
    blob = table.to_json()
    loaded = DispatchTable.from_json(blob)
    assert loaded.entries == table.entries
    assert loaded.fingerprint == table.fingerprint
    assert (
        loaded.select(
            "1120ms",
            1024,
            lane="fp32",
            graph_covers_decode=False,
        )
        == "compact-eager"
    )


def test_generation_is_deterministic() -> None:
    a = generate_dispatch_table(_two_tier_report())
    b = generate_dispatch_table(copy.deepcopy(_two_tier_report()))
    assert a.entries == b.entries


# ---- fail-closed admission (PR #96 review round) -----------------------------


def test_execution_not_ok_is_rejected() -> None:
    report = _two_tier_report()
    report["execution_ok"] = False
    with pytest.raises(ValueError, match="execution_ok"):
        generate_dispatch_table(report)
    del report["execution_ok"]
    with pytest.raises(ValueError, match="execution_ok"):
        generate_dispatch_table(report)


def test_declared_lane_contradicting_lane_results_is_rejected() -> None:
    report = _two_tier_report()
    report["lane_results"]["fp32"]["dispatch_eligible"] = False
    with pytest.raises(ValueError, match="disagrees"):
        generate_dispatch_table(report)


def test_dirty_tree_none_is_rejected() -> None:
    report = _two_tier_report()
    report["fingerprint"]["dirty_tree"] = None
    with pytest.raises(ValueError, match="dirty"):
        generate_dispatch_table(report)


def test_unknown_dcp_version_is_rejected() -> None:
    report = _two_tier_report()
    report["dcp"]["version"] = "dcp-v999"
    with pytest.raises(ValueError, match="dcp version"):
        generate_dispatch_table(report)


def test_unasserted_config_is_rejected() -> None:
    report = _two_tier_report()
    report["fingerprint"]["config_asserted"] = False
    with pytest.raises(ValueError, match="config"):
        generate_dispatch_table(report)


def test_unknown_probe_schema_is_rejected() -> None:
    report = _two_tier_report()
    report["fingerprint"]["probe_schema"] = "p6b-decode-profile-v99"
    with pytest.raises(ValueError, match="schema"):
        generate_dispatch_table(report)


def test_untracked_outside_lock_class_is_rejected() -> None:
    report = _two_tier_report()
    report["fingerprint"]["untracked_paths"] = [
        "abc.lock",
        "sitecustomize.py",
    ]
    with pytest.raises(ValueError, match="untracked"):
        generate_dispatch_table(report)
    report["fingerprint"]["untracked_paths"] = None
    with pytest.raises(ValueError, match="untracked"):
        generate_dispatch_table(report)


def test_unknown_arm_is_rejected() -> None:
    report = _two_tier_report()
    report["cells"][0]["arm"] = "dense-fused"
    with pytest.raises(ValueError, match="unknown arm"):
        generate_dispatch_table(report)


def test_duplicate_cell_is_rejected() -> None:
    report = _two_tier_report()
    report["cells"].append(dict(report["cells"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        generate_dispatch_table(report)


def test_missing_dense_row_is_rejected_strict() -> None:
    report = _two_tier_report()
    report["cells"] = [
        c
        for c in report["cells"]
        if not (c["batch"] == 8 and c["arm"] == "dense-eager")
    ]
    with pytest.raises(ValueError, match="dense-eager"):
        generate_dispatch_table(report)


def test_non_finite_latency_is_rejected() -> None:
    report = _two_tier_report()
    report["cells"][0]["latency_us"] = float("nan")
    with pytest.raises(ValueError, match="latency"):
        generate_dispatch_table(report)


def test_lane_results_mismatch_count_is_cross_checked() -> None:
    report = _two_tier_report()
    report["lane_results"]["fp32"]["mismatch_cells"] = 2
    # Declared 2 mismatches but cells carry none — and eligibility
    # derivation already refuses a lane with mismatches.
    with pytest.raises(ValueError):
        generate_dispatch_table(report)


def test_analysis_mode_is_labeled_and_never_deployable() -> None:
    report = _two_tier_report()
    report["fingerprint"]["dirty_tree"] = True  # hygiene defect
    table = generate_dispatch_table(report, admission="analysis")
    assert table.analysis_only is True
    with pytest.raises(FingerprintMismatchError, match="analysis"):
        table.validate_runtime(
            dict(table.fingerprint),
            performance_gated=True,
        )
    # Round trip preserves the label.
    from_json = DispatchTable.from_json(table.to_json())
    assert from_json.analysis_only is True


def test_identity_covers_math_mode_lane_and_algorithm() -> None:
    table = generate_dispatch_table(_two_tier_report())
    for key, changed in (
        ("tf32_matmul", True),
        ("lane_definitions_digest", "sha256:other"),
        ("decode_algo_revision", "decode-algo-v2"),
        ("probe_schema", "p6b-decode-profile-v4"),
    ):
        runtime = dict(table.fingerprint)
        runtime[key] = changed
        assert key in table.validate_runtime(
            runtime,
            performance_gated=False,
        )
        with pytest.raises(FingerprintMismatchError):
            table.validate_runtime(runtime, performance_gated=True)


def test_cross_run_disagreement_forces_sync_free() -> None:
    report = _two_tier_report()
    other = copy.deepcopy(report)
    # The independent run flips the large tier: compact's speech win
    # evaporates there.
    for cell in other["cells"]:
        if cell["batch"] == 1024 and cell["arm"] == "compact-eager":
            cell["latency_us"] = 900_000
    table = generate_dispatch_table(report, validation_reports=[other])
    assert table.validation_runs == 1
    assert table.entries[("1120ms", 1024, "fp32")] == "dense-eager"
    assert "1120ms|1024|fp32" in table.cross_run_forced
    # Serialization preserves the validation record.
    loaded = DispatchTable.from_json(table.to_json())
    assert loaded.validation_runs == 1
    assert loaded.cross_run_forced == table.cross_run_forced


def _validation_copy(report: dict[str, Any]) -> dict[str, Any]:
    """A comparable-but-distinct independent run: identical identity
    and selections, one latency perturbed inside its margin."""
    other = copy.deepcopy(report)
    other["cells"][0]["latency_us"] += 1.0
    return other


def test_cross_run_agreement_changes_nothing() -> None:
    report = _two_tier_report()
    table = generate_dispatch_table(
        report,
        validation_reports=[_validation_copy(report)],
    )
    unvalidated = generate_dispatch_table(_two_tier_report())
    assert table.entries == unvalidated.entries
    assert table.validation_runs == 1
    assert table.cross_run_forced == ()
    assert len(table.validation_report_digests) == 1
    assert table.source_report_digest not in table.validation_report_digests


# ---- cross-run gate + loader hardening (PR #96 round 2) ----------------------


def test_performance_gate_requires_validation() -> None:
    # A strict-but-unvalidated table never serves a performance-gated
    # runtime; a validated one does.
    table = generate_dispatch_table(_two_tier_report())
    with pytest.raises(FingerprintMismatchError, match="validate"):
        table.validate_runtime(
            dict(table.fingerprint),
            performance_gated=True,
        )
    report = _two_tier_report()
    validated = generate_dispatch_table(
        report,
        validation_reports=[_validation_copy(report)],
    )
    assert (
        validated.validate_runtime(
            dict(validated.fingerprint),
            performance_gated=True,
        )
        == []
    )


def test_duplicate_validation_report_is_rejected() -> None:
    report = _two_tier_report()
    with pytest.raises(ValueError, match="UNIQUE"):
        generate_dispatch_table(
            report,
            validation_reports=[copy.deepcopy(report)],
        )
    other = _validation_copy(report)
    with pytest.raises(ValueError, match="UNIQUE"):
        generate_dispatch_table(
            report,
            validation_reports=[other, copy.deepcopy(other)],
        )


def test_incomparable_validation_report_is_rejected() -> None:
    report = _two_tier_report()
    empty: dict[str, Any] = {}
    with pytest.raises(ValueError):
        generate_dispatch_table(
            report,
            validation_reports=[empty],
        )
    relabeled = _validation_copy(report)
    relabeled["fingerprint"]["device_name"] = "NVIDIA A100 80GB PCIe"
    with pytest.raises(ValueError, match="not comparable"):
        generate_dispatch_table(
            report,
            validation_reports=[relabeled],
        )
    other_algo = _validation_copy(report)
    other_algo["fingerprint"]["decode_algo_revision"] = "decode-algo-v2"
    with pytest.raises(ValueError, match="not comparable"):
        generate_dispatch_table(
            report,
            validation_reports=[other_algo],
        )


def test_validation_accepts_different_driver_minor() -> None:
    # An independent host IS the point of the validation run: the
    # comparability projection holds algorithm/model/lane/policy/
    # math/device fixed but not the driver minor (rounds 8/9 ran on
    # .09 vs .20). The table's runtime validity still binds the
    # SOURCE run's driver.
    report = _two_tier_report()
    other = _validation_copy(report)
    other["fingerprint"]["driver"] = "580.126.09"
    table = generate_dispatch_table(
        report,
        validation_reports=[other],
    )
    assert table.validation_runs == 1
    assert table.fingerprint["driver"] == "580.126.20"


def test_validation_key_coverage_must_match() -> None:
    report = _two_tier_report()
    partial = _validation_copy(report)
    partial["cells"] = [c for c in partial["cells"] if c["batch"] != 1024]
    with pytest.raises(ValueError, match="key set"):
        generate_dispatch_table(
            report,
            validation_reports=[partial],
        )


def test_mutated_dcp_contents_are_rejected() -> None:
    report = _two_tier_report()
    report["dcp"]["selective_activities"] = ()
    with pytest.raises(ValueError, match="canonical"):
        generate_dispatch_table(report)


def test_unqualified_hysteresis_is_rejected_strict() -> None:
    with pytest.raises(ValueError, match="qualified"):
        generate_dispatch_table(
            _two_tier_report(),
            hysteresis_pct=5.0,
        )
    # Analysis mode may explore other margins.
    table = generate_dispatch_table(
        _two_tier_report(),
        hysteresis_pct=5.0,
        admission="analysis",
    )
    assert table.analysis_only


def test_from_json_fails_closed() -> None:
    import json as _json

    table = generate_dispatch_table(_two_tier_report())
    raw = _json.loads(table.to_json())
    # Missing qualification field.
    broken = dict(raw)
    del broken["analysis_only"]
    with pytest.raises(ValueError, match="missing"):
        DispatchTable.from_json(_json.dumps(broken))
    # Unsupported schema.
    broken = dict(raw)
    broken["schema"] = "nemotron-dispatch-table-v1"
    with pytest.raises(ValueError, match="schema"):
        DispatchTable.from_json(_json.dumps(broken))
    # Unsupported policy version.
    broken = dict(raw)
    broken["policy_version"] = "dcp-v999"
    with pytest.raises(ValueError, match="policy_version"):
        DispatchTable.from_json(_json.dumps(broken))
    # Unknown arm.
    broken = _json.loads(table.to_json())
    key = next(iter(broken["entries"]))
    broken["entries"][key] = "dense-fused"
    with pytest.raises(ValueError, match="unknown arm"):
        DispatchTable.from_json(_json.dumps(broken))
    # Inconsistent validation metadata.
    broken = _json.loads(table.to_json())
    broken["validation_runs"] = 2
    with pytest.raises(ValueError, match="disagrees"):
        DispatchTable.from_json(_json.dumps(broken))
    # Fingerprint missing an identity key.
    broken = _json.loads(table.to_json())
    del broken["fingerprint"]["tf32_matmul"]
    with pytest.raises(ValueError, match="identity"):
        DispatchTable.from_json(_json.dumps(broken))


def test_validation_reports_admitted_at_primary_level() -> None:
    # A strict table may not launder its validation through
    # analysis-mode admission: an execution_ok=False or dirty-tree
    # "validation run" must reject, not count.
    report = _two_tier_report()
    bad = _validation_copy(report)
    bad["execution_ok"] = False
    with pytest.raises(ValueError, match="execution_ok"):
        generate_dispatch_table(
            report,
            validation_reports=[bad],
        )
    bad = _validation_copy(report)
    bad["fingerprint"]["dirty_tree"] = True
    with pytest.raises(ValueError, match="dirty"):
        generate_dispatch_table(
            report,
            validation_reports=[bad],
        )


def test_loader_enforces_qualified_hysteresis() -> None:
    import json as _json

    report = _two_tier_report()
    table = generate_dispatch_table(
        report,
        validation_reports=[_validation_copy(report)],
    )
    edited = _json.loads(table.to_json())
    edited["hysteresis_pct"] = 5.0
    with pytest.raises(ValueError, match="qualified"):
        DispatchTable.from_json(_json.dumps(edited))
    # Analysis tables may carry exploratory margins and still load —
    # they can never deploy anyway.
    analysis = generate_dispatch_table(
        _two_tier_report(),
        hysteresis_pct=5.0,
        admission="analysis",
    )
    loaded = DispatchTable.from_json(analysis.to_json())
    assert loaded.analysis_only


def test_performance_gate_enforces_qualified_hysteresis() -> None:
    # Defense in depth: a programmatically constructed table with an
    # unqualified margin never passes the gate, even without a
    # from_json round trip.
    report = _two_tier_report()
    table = generate_dispatch_table(
        report,
        validation_reports=[_validation_copy(report)],
    )
    tampered = DispatchTable(
        entries=table.entries,
        fingerprint=table.fingerprint,
        hysteresis_pct=5.0,
        policy_version=table.policy_version,
        lanes=table.lanes,
        analysis_only=False,
        validation_runs=table.validation_runs,
        cross_run_forced=table.cross_run_forced,
        source_report_digest=table.source_report_digest,
        validation_report_digests=table.validation_report_digests,
    )
    with pytest.raises(FingerprintMismatchError, match="qualified"):
        tampered.validate_runtime(
            dict(table.fingerprint),
            performance_gated=True,
        )


def test_untracked_count_mismatch_is_rejected() -> None:
    report = _two_tier_report()
    report["fingerprint"]["untracked_files"] = 3
    with pytest.raises(ValueError, match="count disagrees"):
        generate_dispatch_table(report)


def test_unmeasured_tier_fallback_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    table = generate_dispatch_table(_two_tier_report())
    with caplog.at_level("WARNING"):
        table.select(
            "1120ms",
            516,
            lane="fp32",
            graph_covers_decode=False,
        )
    assert any("unmeasured tier" in rec.message for rec in caplog.records)
