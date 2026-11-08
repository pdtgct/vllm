# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup decode dispatch-table generation (PORT-DEC-008).

Consumes a ``p6b_decode_profile`` artifact and COMPUTES per-bucket
decode selections under the versioned dispatch-calibration policy —
never eyeballed brackets (decisions/decode-dispatch-regime.md r3.2):

- lanes come from the artifact's ``generator_eligible_lanes`` — a
  correctness-unqualified lane never produces entries;
- compact may be selected only from SELECTIVE cells (policy realized
  within tolerance) and only when it beats the sync-free arm by the
  recorded hysteresis at EVERY dcp activity level (the worst-case
  objective); a nonselective or missing level forces the sync-free
  arm;
- an unmeasured tier resolves to the nearest measured tier, and to
  the sync-free arm when the bracketing tiers disagree (the
  crossover-gap rule — dense's worst case is bounded overcompute,
  compact's adds syncs, host stalls, and capture-ineligibility);
- an execution profile whose engine graph covers decode never
  dispatches the capture-ineligible candidate (sanity invariant, not
  a measured outcome);
- the table carries the measurement fingerprint (device, driver,
  torch, model-shape digest — deliberately CHECKPOINT-independent)
  and a runtime mismatch fails closed for performance-gated
  deployments.

Pure host logic: no torch, importable anywhere. The consumer is
``advance_model_rows``, which resolves the callable per
profile+geometry bucket at batch-composition time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import regex as re

logger = logging.getLogger(__name__)

#: The recorded hysteresis margin (decisions/decode-dispatch-regime.md
#: r3: chosen 2026-07-18, validated by independent profile runs before
#: any performance-gated deployment).
HYSTERESIS_PCT = 25.0

#: Arms whose valid path issues no host/device synchronization.
SYNC_FREE_ARMS = ("dense-eager", "dense-graphed")
_KNOWN_ARMS = frozenset(("dense-eager", "dense-graphed", "compact-eager"))

#: Admission allowlists — strict generation fails closed on anything
#: outside them.
SUPPORTED_DCP_VERSIONS = ("dcp-v1",)
SUPPORTED_PROBE_SCHEMAS = ("p6b-decode-profile-v3",)
#: Serialized-table schema; loading fails closed on anything else.
TABLE_SCHEMA = "nemotron-dispatch-table-v2"
#: The canonical CONTENTS each policy version binds — a mutated
#: declaration under a known version name is rejected (the version
#: string alone binds nothing).
CANONICAL_DCP: dict[str, dict[str, Any]] = {
    "dcp-v1": {
        "version": "dcp-v1",
        "targets": {"silence": 0.3, "speech": 3.0},
        "tolerance": {
            "silence": [0.0, 1.0],
            "speech": [1.5, 4.5],
        },
        "selective_activities": ["silence", "speech"],
    },
}
#: The qualified hysteresis per policy version
#: (decisions/decode-dispatch-regime.md r3) — strict generation
#: accepts no other value.
QUALIFIED_HYSTERESIS: dict[str, float] = {"dcp-v1": 25.0}
#: Minimum comparable independent runs a table must carry to serve a
#: performance-gated runtime (the record's "validated by an
#: independent profile run").
MIN_VALIDATION_RUNS = 1
#: The one benign untracked class on a measurement checkout:
#: root-level model-download lock files (round 7's audit).
_UNTRACKED_ALLOWED = re.compile(r"^[^/]+\.lock$")

#: Fingerprint keys a runtime must match for a table to be valid —
#: the full normative identity: device, driver, torch, model shape,
#: decode-algorithm revision (never the fork commit — an unrelated
#: commit must not invalidate a table), probe schema, lane
#: definitions, math mode, and the calibration-policy version.
_FINGERPRINT_KEYS = (
    "device_name",
    "driver",
    "torch",
    "model_shape_digest",
    "decode_algo_revision",
    "probe_schema",
    "lane_definitions_digest",
    "tf32_matmul",
)
#: Comparability projection for cross-run VALIDATION: the
#: algorithm/model/lane/policy/math/device identity must match, but
#: the ENTIRE driver field is excluded (any component may differ —
#: rounds 8/9 differed only in the maintenance component, but this
#: projection would also accept a different driver branch): an
#: independent host is the point of the validation run, and the
#: table's runtime validity still binds the SOURCE run's exact
#: driver via _FINGERPRINT_KEYS.
_VALIDATION_COMPARABILITY_KEYS = tuple(k for k in _FINGERPRINT_KEYS if k != "driver")


class FingerprintMismatchError(RuntimeError):
    """A dispatch table presented to a runtime it was not measured
    on (performance-gated deployments fail closed)."""


@dataclass(frozen=True)
class DispatchTable:
    """Computed decode selections plus the identity that scopes them.

    ``entries`` maps ``(geometry, tier, lane)`` to the EAGER-profile
    arm; graph-covered profiles are resolved structurally in
    :meth:`select`, never stored (compact is capture-ineligible by
    construction there).
    """

    entries: dict[tuple[str, int, str], str]
    fingerprint: dict[str, Any]
    hysteresis_pct: float
    policy_version: str
    lanes: tuple[str, ...]
    #: True when generated under admission="analysis" (comparison and
    #: review use only): such a table NEVER validates for a
    #: performance-gated runtime.
    analysis_only: bool = field(default=False)
    #: Independent runs whose tables were compared during generation
    #: (0 = unvalidated — such a table never serves a
    #: performance-gated runtime) and the entries cross-run
    #: disagreement forced to the sync-free arm.
    validation_runs: int = field(default=0)
    cross_run_forced: tuple[str, ...] = field(default=())
    #: Canonical digests binding the table to its exact inputs.
    source_report_digest: str = field(default="")
    validation_report_digests: tuple[str, ...] = field(default=())

    def select(
        self,
        geometry: str,
        batch: int,
        *,
        lane: str,
        graph_covers_decode: bool,
    ) -> str:
        """The decode arm for one profile+geometry bucket.

        Raises:
            KeyError: unknown ``lane`` (not generator-eligible) or a
                geometry with no measured tiers in this table.

        """
        if lane not in self.lanes:
            raise KeyError(
                f"lane {lane!r} is not generator-eligible (eligible: {self.lanes})"
            )
        tiers = sorted(t for (g, t, la) in self.entries if g == geometry and la == lane)
        if not tiers:
            raise KeyError(f"no measured tiers for {geometry!r}")
        if graph_covers_decode:
            return "dense-graphed"
        if batch in tiers:
            return self.entries[(geometry, batch, lane)]
        lower = max((t for t in tiers if t < batch), default=None)
        upper = min((t for t in tiers if t > batch), default=None)
        if lower is None:
            assert upper is not None
            return self.entries[(geometry, upper, lane)]
        if upper is None:
            return self.entries[(geometry, lower, lane)]
        low_arm = self.entries[(geometry, lower, lane)]
        high_arm = self.entries[(geometry, upper, lane)]
        if low_arm != high_arm:
            # Crossover-gap rule: bracketing tiers disagree — take
            # the sync-free arm of the pair.
            logger.warning(
                "decode dispatch: unmeasured tier %d for %s/%s with "
                "disagreeing brackets (%d=%s, %d=%s) — selecting the "
                "sync-free arm",
                batch,
                geometry,
                lane,
                lower,
                low_arm,
                upper,
                high_arm,
            )
            return low_arm if low_arm in SYNC_FREE_ARMS else high_arm
        nearest = lower if (batch - lower) <= (upper - batch) else upper
        logger.warning(
            "decode dispatch: unmeasured tier %d for %s/%s — resolving to nearest "
            "measured tier %d (%s)",
            batch,
            geometry,
            lane,
            nearest,
            self.entries[(geometry, nearest, lane)],
        )
        return self.entries[(geometry, nearest, lane)]

    def validate_runtime(
        self,
        runtime_fingerprint: dict[str, Any],
        *,
        performance_gated: bool,
    ) -> list[str]:
        """Check this table's identity against the runtime.

        Returns:
            The mismatched fingerprint keys (empty when valid); a
            dev-profile caller warns on them and regenerates.

        Raises:
            FingerprintMismatchError: any mismatch while
                ``performance_gated`` — the table fails closed.

        """
        if self.analysis_only and performance_gated:
            raise FingerprintMismatchError(
                "analysis-only table (admission='analysis') can never serve a "
                "performance-gated runtime"
            )
        if performance_gated and (
            self.validation_runs < MIN_VALIDATION_RUNS
            or len(self.validation_report_digests) != self.validation_runs
            or not self.source_report_digest
        ):
            raise FingerprintMismatchError(
                "performance-gated deployment requires a table "
                f"cross-validated by >= {MIN_VALIDATION_RUNS} "
                "comparable independent run(s) with bound report "
                f"digests (this table: {self.validation_runs})"
            )
        if performance_gated and (
            self.hysteresis_pct != QUALIFIED_HYSTERESIS.get(self.policy_version)
        ):
            # Defense in depth for programmatically constructed
            # tables that never round-tripped through from_json.
            raise FingerprintMismatchError(
                f"hysteresis {self.hysteresis_pct}% is not the qualified margin for "
                f"{self.policy_version}"
            )
        mismatched = [
            key
            for key in _FINGERPRINT_KEYS
            if runtime_fingerprint.get(key) != self.fingerprint.get(key)
        ]
        if mismatched and performance_gated:
            raise FingerprintMismatchError(
                "dispatch table measured on a different runtime: "
                + ", ".join(
                    f"{k}: table={self.fingerprint.get(k)!r} "
                    f"runtime={runtime_fingerprint.get(k)!r}"
                    for k in mismatched
                )
            )
        return mismatched

    def to_json(self) -> str:
        """Serialize (entry keys flatten to 'geometry|tier|lane')."""
        return json.dumps(
            {
                "schema": TABLE_SCHEMA,
                "entries": {
                    f"{g}|{t}|{lane}": arm
                    for (g, t, lane), arm in sorted(self.entries.items())
                },
                "fingerprint": self.fingerprint,
                "hysteresis_pct": self.hysteresis_pct,
                "policy_version": self.policy_version,
                "lanes": list(self.lanes),
                "analysis_only": self.analysis_only,
                "validation_runs": self.validation_runs,
                "cross_run_forced": list(self.cross_run_forced),
                "source_report_digest": self.source_report_digest,
                "validation_report_digests": list(self.validation_report_digests),
            }
        )

    @classmethod
    def from_json(cls, blob: str) -> DispatchTable:
        """Load a serialized table, FAIL-CLOSED: every qualification
        field must be explicitly present and internally consistent —
        a legacy or hand-edited table never becomes deployable by
        defaulting.

        Raises:
            ValueError: unsupported schema, missing fields, unknown
                arms/policy, or inconsistent validation metadata.

        """
        raw = json.loads(blob)
        if raw.get("schema") != TABLE_SCHEMA:
            raise ValueError(
                f"unsupported table schema {raw.get('schema')!r} (expected "
                f"{TABLE_SCHEMA!r})"
            )
        required = (
            "entries",
            "fingerprint",
            "hysteresis_pct",
            "policy_version",
            "lanes",
            "analysis_only",
            "validation_runs",
            "cross_run_forced",
            "source_report_digest",
            "validation_report_digests",
        )
        missing = [k for k in required if k not in raw]
        if missing:
            raise ValueError(f"table is missing required fields: {missing}")
        if raw["policy_version"] not in SUPPORTED_DCP_VERSIONS:
            raise ValueError(f"unsupported policy_version {raw['policy_version']!r}")
        if (
            not raw["analysis_only"]
            and float(raw["hysteresis_pct"])
            != QUALIFIED_HYSTERESIS[raw["policy_version"]]
        ):
            raise ValueError(
                f"deployable table carries hysteresis "
                f"{raw['hysteresis_pct']}%, not the qualified "
                f"margin for {raw['policy_version']} — an edited "
                "table cannot load"
            )
        for key_id in _FINGERPRINT_KEYS:
            if key_id not in raw["fingerprint"]:
                raise ValueError(f"table fingerprint missing identity key {key_id!r}")
        if len(raw["validation_report_digests"]) != int(raw["validation_runs"]):
            raise ValueError(
                "validation_report_digests count disagrees with validation_runs"
            )
        entries: dict[tuple[str, int, str], str] = {}
        for key, arm in raw["entries"].items():
            if arm not in _KNOWN_ARMS:
                raise ValueError(f"unknown arm {arm!r} in table")
            geometry, tier, lane = key.split("|")
            entries[(geometry, int(tier), lane)] = arm
        if not raw["lanes"]:
            raise ValueError("table declares no lanes")
        return cls(
            entries=entries,
            fingerprint=raw["fingerprint"],
            hysteresis_pct=float(raw["hysteresis_pct"]),
            policy_version=raw["policy_version"],
            lanes=tuple(raw["lanes"]),
            analysis_only=bool(raw["analysis_only"]),
            validation_runs=int(raw["validation_runs"]),
            cross_run_forced=tuple(raw["cross_run_forced"]),
            source_report_digest=str(raw["source_report_digest"]),
            validation_report_digests=tuple(raw["validation_report_digests"]),
        )


def _latency(cell: dict[str, Any]) -> float:
    return float(cell["latency_us"])


def _report_digest(report: dict[str, Any]) -> str:
    """Canonical sha256 of a whole report (tuples serialize as
    lists, so in-memory and JSON-loaded forms digest identically)."""
    blob = json.dumps(report, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _dcp_canonical(dcp: dict[str, Any]) -> str:
    """The policy declaration minus prose, canonically serialized."""
    return json.dumps(
        {k: v for k, v in dcp.items() if k != "fallback_rule"},
        sort_keys=True,
        separators=(",", ":"),
    )


def _admit_dcp(dcp: dict[str, Any]) -> str:
    """The version string binds its canonical CONTENTS, not just its
    name — a mutated declaration under a known version rejects."""
    version = dcp.get("version")
    if version not in SUPPORTED_DCP_VERSIONS:
        raise ValueError(
            f"unsupported dcp version {version!r} (supported: {SUPPORTED_DCP_VERSIONS})"
        )
    if _dcp_canonical(dcp) != _dcp_canonical(CANONICAL_DCP[version]):
        raise ValueError(
            f"dcp declaration does not match the canonical "
            f"{version} contents — a mutated policy under a known "
            "version name is rejected"
        )
    return str(version)


def _admit_strict(report: dict[str, Any], fingerprint: dict[str, Any]) -> None:
    """Fail-closed artifact admission: every gate must POSITIVELY
    hold (absent or None never passes)."""
    if report.get("execution_ok") is not True:
        raise ValueError(
            "artifact does not record execution_ok=True — the sweep may not have "
            "completed"
        )
    if fingerprint.get("dirty_tree") is not False:
        raise ValueError(
            "artifact does not record a clean tree (dirty_tree must "
            "be exactly False) — evidence identity is not "
            "reproducible"
        )
    if fingerprint.get("config_asserted") is not True:
        raise ValueError(
            "artifact dims were not asserted against the model config (config_asserted "
            "must be True)"
        )
    schema = fingerprint.get("probe_schema")
    if schema not in SUPPORTED_PROBE_SCHEMAS:
        raise ValueError(
            f"unsupported probe schema {schema!r} (supported: "
            f"{SUPPORTED_PROBE_SCHEMAS})"
        )
    untracked = fingerprint.get("untracked_paths")
    if untracked is None:
        raise ValueError(
            "artifact records no untracked_paths — untracked state is unauditable"
        )
    if fingerprint.get("untracked_files") != len(untracked):
        raise ValueError(
            "untracked_files count disagrees with untracked_paths — "
            "the inventory may be truncated and cannot be audited"
        )
    offending = [p for p in untracked if not _UNTRACKED_ALLOWED.match(p)]
    if offending:
        raise ValueError(
            "untracked files outside the allowlisted root lock-file class on the "
            f"measurement checkout: {offending[:5]}"
        )


def _eligible_lanes(report: dict[str, Any], *, strict: bool) -> tuple[str, ...]:
    """Lanes are DERIVED from lane_results, never trusted from the
    declared list; strict admission requires the two to agree."""
    lane_results = report["lane_results"]
    derived = tuple(
        sorted(
            lane
            for lane, res in lane_results.items()
            if res.get("dispatch_eligible") is True
            and res.get("performance_qualified") is True
            and res.get("mismatch_cells") == 0
        )
    )
    declared = tuple(sorted(report["generator_eligible_lanes"]))
    if strict and derived != declared:
        raise ValueError(
            f"generator_eligible_lanes {declared} disagrees with lane_results "
            f"derivation {derived}"
        )
    return derived


def generate_dispatch_table(
    report: dict[str, Any],
    *,
    hysteresis_pct: float = HYSTERESIS_PCT,
    admission: str = "strict",
    validation_reports: list[dict[str, Any]] | None = None,
) -> DispatchTable:
    """Compute the dispatch table from a profile artifact.

    Args:
        report: a ``p6b_decode_profile`` report (probe r8+ schema).
        hysteresis_pct: the recorded sync-free preference margin.
        admission: ``"strict"`` (default) applies the fail-closed
            gates; ``"analysis"`` bypasses ONLY the artifact-hygiene
            gates for comparison/review work and marks the table
            ``analysis_only`` — such a table never validates for a
            performance-gated runtime.
        validation_reports: independent-run artifacts (admitted in
            analysis mode) whose computed selections CROSS-VALIDATE
            this one: any entry that disagrees across runs is forced
            to the sync-free arm and recorded — selection stability
            under re-measurement is enforced, not hoped (the round-8
            arc caught a discrete calibration-realization flip at a
            B=1 edge cell this way).

    Raises:
        ValueError: schema, hygiene, consistency, or structural
            defects — generation fails closed on all of them.

    """
    if admission not in ("strict", "analysis"):
        raise ValueError(f"unknown admission mode {admission!r}")
    strict = admission == "strict"
    for required in ("fingerprint", "cells", "dcp"):
        if required not in report:
            raise ValueError(f"artifact missing {required!r}")
    if "lane_results" not in report or "generator_eligible_lanes" not in report:
        raise ValueError(
            "artifact predates per-lane qualification "
            "(lane_results/generator_eligible_lanes missing) — "
            "not generator input"
        )
    fingerprint = dict(report["fingerprint"])
    dcp = report["dcp"]
    dcp_version = _admit_dcp(dcp)
    if strict:
        _admit_strict(report, fingerprint)
        qualified = QUALIFIED_HYSTERESIS[dcp_version]
        if hysteresis_pct != qualified:
            raise ValueError(
                f"hysteresis {hysteresis_pct}% is not the qualified margin for "
                f"{dcp_version} ({qualified}%)"
            )
    lanes = _eligible_lanes(report, strict=strict)
    selective_activities = tuple(dcp["selective_activities"])

    by_key: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = {}
    mismatch_recount: dict[str, int] = {}
    seen: set[tuple[str, int, str, str, str]] = set()
    for cell in report["cells"]:
        lane = str(cell["precision"])
        arm = str(cell["arm"])
        if arm not in _KNOWN_ARMS:
            raise ValueError(f"unknown arm {arm!r} in artifact")
        latency = float(cell["latency_us"])
        if not math.isfinite(latency) or latency <= 0.0:
            raise ValueError(
                "non-finite/non-positive latency in cell "
                f"{cell['geometry']}/{cell['batch']}/{lane}/{arm}"
            )
        if not cell.get("tokens_match", True):
            mismatch_recount[lane] = mismatch_recount.get(lane, 0) + 1
        ident = (
            str(cell["geometry"]),
            int(cell["batch"]),
            lane,
            arm,
            str(cell["activity"]),
        )
        if ident in seen:
            raise ValueError(f"duplicate cell {ident}")
        seen.add(ident)
        if lane not in lanes:
            continue
        key = (str(cell["geometry"]), int(cell["batch"]), lane)
        arms = by_key.setdefault(key, {})
        if arm == "compact-eager":
            # Keep every activity row; selection reads dcp levels.
            arms[f"compact:{cell['activity']}"] = cell
        else:
            arms.setdefault(arm, cell)
    if strict:
        for lane, res in report["lane_results"].items():
            if mismatch_recount.get(lane, 0) != res.get("mismatch_cells"):
                raise ValueError(
                    f"lane_results mismatch_cells for {lane!r} "
                    f"({res.get('mismatch_cells')}) disagrees with "
                    f"the cells ({mismatch_recount.get(lane, 0)})"
                )

    entries: dict[tuple[str, int, str], str] = {}
    factor = 1.0 - hysteresis_pct / 100.0
    for key, arms in by_key.items():
        dense = arms.get("dense-eager")
        if dense is None:
            if strict:
                raise ValueError(
                    f"measured group {key} has no dense-eager row — artifact is "
                    "structurally incomplete"
                )
            continue
        dense_lat = _latency(dense)
        compact_wins = True
        for activity in selective_activities:
            cell = arms.get(f"compact:{activity}")
            if cell is None or not cell.get("selective"):
                # Missing or nonselective dcp level: the worst case
                # is unproven — sync-free forced.
                compact_wins = False
                break
            if _latency(cell) > dense_lat * factor:
                compact_wins = False
                break
        entries[key] = "compact-eager" if compact_wins else "dense-eager"

    source_digest = _report_digest(report)
    forced: list[str] = []
    validation_digests: list[str] = []
    if validation_reports:
        for other_report in validation_reports:
            digest = _report_digest(other_report)
            if digest == source_digest or digest in (validation_digests):
                raise ValueError(
                    "validation reports must be UNIQUE independent runs (duplicate "
                    "digest)"
                )
            # Validation reports are admitted at the SAME level as
            # the primary: a strict table may not launder its
            # validation through analysis-mode admission (an
            # execution_ok=False or dirty-tree "validation run"
            # would count otherwise).
            other = generate_dispatch_table(
                other_report,
                hysteresis_pct=hysteresis_pct,
                admission=admission,
            )
            # Comparability: the full identity projection AND the
            # policy contents must match — counting an incomparable
            # run would launder the validation requirement.
            other_fp = other_report["fingerprint"]
            incomparable = [
                k
                for k in _VALIDATION_COMPARABILITY_KEYS
                if other_fp.get(k) != fingerprint.get(k)
            ]
            if incomparable:
                raise ValueError(
                    "validation report is not comparable — identity differs on "
                    f"{incomparable}"
                )
            if _dcp_canonical(other_report["dcp"]) != _dcp_canonical(dcp):
                raise ValueError(
                    "validation report carries a different dcp declaration"
                )
            if set(other.entries) != set(entries):
                raise ValueError(
                    "validation report does not cover the same "
                    "(geometry, tier, lane) key set — coverage "
                    "gaps cannot be silently skipped"
                )
            validation_digests.append(digest)
            for key, arm in list(entries.items()):
                other_arm = other.entries[key]
                if other_arm != arm:
                    if arm not in SYNC_FREE_ARMS:
                        entries[key] = "dense-eager"
                    forced.append("|".join(map(str, key)))
                    logger.warning(
                        "decode dispatch: cross-run disagreement at %s (%s vs %s) — "
                        "forcing the sync-free arm",
                        key,
                        arm,
                        other_arm,
                    )
    return DispatchTable(
        entries=entries,
        fingerprint=fingerprint,
        hysteresis_pct=hysteresis_pct,
        policy_version=dcp_version,
        lanes=lanes,
        analysis_only=not strict,
        validation_runs=len(validation_digests),
        cross_run_forced=tuple(sorted(set(forced))),
        source_report_digest=source_digest,
        validation_report_digests=tuple(validation_digests),
    )
