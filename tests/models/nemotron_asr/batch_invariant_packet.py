# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure, stdlib-only validators for supplied v0.30 qualification packets.

No runtime installation, audio acquisition, experiment or qualification occurs here.
Fingerprints are complete JSON objects; receipt envelopes bind their original
payloads to SHA256 of canonical JSON (sorted keys, compact separators, UTF-8,
ensure_ascii=True). Identity fields describe the matched experimental controls.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import statistics
from pathlib import Path

CLIPS = tuple(f"dev-clean/1272/135031/1272-135031-{i:04}" for i in range(4))
CADENCES = (80, 160, 320, 560, 1120)
CELLS = frozenset((clip, cadence) for clip in CLIPS for cadence in CADENCES)
MATCHED_FIELDS = (
    "gpu",
    "checkpoint",
    "software",
    "precision",
    "precision_policy",
    "optimizations",
    "audio_sha256",
    "cadence",
    "population",
    "concurrency",
    "language_prompt",
    "decoding",
    "postprocessing",
    "execution_path",
)
# PORT schema compatibility layer, additional to upstream's source-derived inventory.
# Pinned to SCHEMA_ADAPTER_SIDE_EFFECTS in nemotron_asr/batch_invariance.py.
PORT_SIDE_EFFECTS = {
    "nemotron_bi.aten_schema_adapters",
    "aten::mm.out",
    "aten::mm.dtype",
    "aten::mm.dtype_out",
    "aten::addmm.out",
    "aten::addmm.dtype",
    "aten::addmm.dtype_out",
    "aten::bmm.out",
    "aten::bmm.dtype",
    "aten::bmm.dtype_out",
    "aten::matmul.out",
    "aten::linear.out",
    "aten::softmax.int",
    "aten::softmax.int_out",
    "aten::_softmax.out",
    "aten::_log_softmax.out",
    "aten::mean.out",
}
PUBLICATION_ONLY = (
    "PORT-PREC-022 blocks publication only; readiness, PREC-024 evidence and "
    "verdict are unchanged"
)


def fingerprint_sha256(fingerprint):
    return hashlib.sha256(
        json.dumps(
            fingerprint, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _require(condition, message):
    assert condition, message


def _matched_arms(packet, fields):
    arms = packet["arms"]
    _require(set(arms) == {"off", "on"}, "both arms required")
    for key in ("identity", "fingerprint"):
        off, on = arms["off"][key], dict(arms["on"][key])
        _require(isinstance(off, dict) and bool(off), f"nonempty {key} required")
        _require("batch_invariant_mode" not in off, f"mode-off {key} must omit mode")
        _require(
            on.pop("batch_invariant_mode", None) == "on",
            f"mode-on {key} must record installed mode",
        )
        _require(
            off == on, f"{key}: arms must differ only in installed batch-invariant mode"
        )
    identity = arms["off"]["identity"]
    for field in fields:
        _require(
            field in identity and identity[field] is not None and identity[field] != "",
            f"matched {field} required",
        )
    gpu = identity["gpu"]
    _require(
        isinstance(gpu, dict) and bool(gpu.get("model")) and bool(gpu.get("partition")),
        "GPU partition identity and model required; model alone does not identify "
        "matched hardware",
    )
    for arm in arms.values():
        digest = fingerprint_sha256(arm["fingerprint"])
        for kind in ("readiness", "service"):
            envelope = arm["receipts"][kind]
            _require(
                envelope["fingerprint_sha256"] == digest,
                f"{kind} receipt fingerprint binding",
            )
            _require(
                isinstance(envelope["receipt"], dict) and bool(envelope["receipt"]),
                f"original {kind} receipt required",
            )
    return arms, identity


def validate_transcript_identity(packet):
    """@spec PORT-PREC-024: exact 20-cell FP32 finals, no WER/tensor tolerance."""
    try:
        _require(
            packet["schema"] == "nemotron-bi-transcript-identity-v1",
            "transcript schema",
        )
        arms, identity = _matched_arms(packet, MATCHED_FIELDS)
        _require(identity["precision"] == "fp32", "FP32 policy required")
        _require(
            identity["decoding"] == {"strategy": "greedy", "deterministic": True},
            "deterministic greedy decoding required",
        )
        _require(
            set(identity["audio_sha256"]) == set(CLIPS),
            "exact four dev-clean audio identities required",
        )
        _require(
            all(_sha256(v) for v in identity["audio_sha256"].values()),
            "audio SHA256 required",
        )
        _require(
            sorted(identity["cadence"]) == list(CADENCES),
            "exact five cadences required",
        )
        parity = packet["mode_off_nemo_parity"]
        _require(
            parity["fingerprint_sha256"]
            == fingerprint_sha256(arms["off"]["fingerprint"]),
            "mode-off NeMo parity must bind mode-off fingerprint",
        )
        _require(
            parity["passed"] is True
            and parity["unchanged"] is True
            and parity["oracle"] == "NeMo",
            "passing unchanged mode-off NeMo parity required",
        )
        _require(
            bool(parity["golden_revision"])
            and _sha256(parity["golden_sha256"])
            and isinstance(parity["receipt"], dict)
            and bool(parity["receipt"]),
            "unchanged golden identity and original parity receipt required",
        )
        cells = packet["cells"]
        _require(
            len(cells) == 20
            and {(c["clip_id"], c["cadence_ms"]) for c in cells} == CELLS,
            "exact 20 cells required, with no omissions, duplicates or extras",
        )
        for cell in cells:
            _require(
                isinstance(cell["off"], str) and isinstance(cell["on"], str),
                "final transcript strings required",
            )
            _require(
                cell["off"].encode("utf-8") == cell["on"].encode("utf-8"),
                f"PORT-PREC-024 transcript mismatch: {cell['clip_id']} at "
                f"{cell['cadence_ms']}ms",
            )
        _require(
            type(packet["differing_transcripts"]) is int
            and packet["differing_transcripts"] == 0,
            "zero differing transcripts required",
        )
    except (KeyError, TypeError, ValueError, UnicodeError) as error:
        raise AssertionError(f"invalid PORT-PREC-024 packet: {error}") from error


def _sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _number(value, *, positive=False):
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def cost_metrics(runs):
    """Medians of pooled replay/chunk samples and per-window audio/wall ratios."""
    replay: list[float] = []
    service: list[float] = []
    throughput: list[float] = []
    for run in runs:
        _require(
            run["purpose"] == "PORT-PREC-022",
            "exploratory and PORT-PREC-029 timings are not disclosure",
        )
        _require(run["profiled"] is False, "profiled timings are not disclosure")
        _require(
            bool(run["encoder_replay_ns"]) and bool(run["chunks"]),
            "raw replay and chunk samples required",
        )
        _require(
            all(_number(v) for v in run["encoder_replay_ns"]),
            "finite nonnegative replay durations required",
        )
        replay.extend(v / 1e6 for v in run["encoder_replay_ns"])
        start, end = run["window_start_ns"], run["window_end_ns"]
        _require(
            _number(start) and _number(end) and end > start,
            "positive measurement window required",
        )
        audio = 0.0
        for chunk in run["chunks"]:
            ready, park = chunk["ready_ns"], chunk["park_ns"]
            _require(
                _number(ready) and _number(park) and start <= ready <= park <= end,
                "ready/park clocks must lie within the same measurement window",
            )
            _require(
                _number(chunk["audio_seconds"], positive=True),
                "positive audio seconds required",
            )
            service.append((park - ready) / 1e6)
            audio += chunk["audio_seconds"]
        throughput.append(audio / ((end - start) / 1e9))
    return {
        "replay_ms": statistics.median(replay),
        "service_ms": statistics.median(service),
        "throughput": statistics.median(throughput),
    }


def validate_cost_disclosure(packet, expected_side_effects, source_sha256):
    """@spec PORT-PREC-022: failure blocks publication only, never readiness/verdict.

    Validate a supplied packet without modifying it or calling qualification_verdict.
    No performance threshold is imposed, including for arbitrarily slower arms.
    """
    try:
        _require(
            packet["schema"] == "nemotron-bi-cost-v1"
            and packet["purpose"] == "PORT-PREC-022",
            "cost schema/purpose",
        )
        inventory = packet["side_effects"]
        _require(
            set(inventory) == set(expected_side_effects)
            and len(inventory) == len(set(inventory)),
            "exact aten/process-wide side-effect inventory by symbol required",
        )
        _require(
            packet["side_effects_source_sha256"] == source_sha256,
            "inventory must bind installed vLLM source",
        )
        _require(packet["threshold"] is None, "no performance threshold")
        _require(
            packet["default_changed"] is False and packet["supported_changed"] is False,
            "selection cannot change default or supported status",
        )
        arms, _ = _matched_arms(
            packet,
            (
                "gpu",
                "audio_sha256",
                "cadence",
                "concurrency",
                "software",
                "precision",
                "precision_policy",
                "optimizations",
            ),
        )
        pairs = packet["pairs"]
        _require(
            len(pairs) >= 4, "at least four alternating paired repetitions required"
        )
        first = pairs[0]["order"]
        _require(first in (["off", "on"], ["on", "off"]), "both arms in each pair")
        for index, pair in enumerate(pairs):
            _require(
                pair["order"] == (first if index % 2 == 0 else first[::-1]),
                "arm order must alternate",
            )
        for mode, arm in arms.items():
            _require(
                len(arm["warmup"]) == 1 and arm["warmup"][0]["discarded"] is True,
                "exactly one discarded warmup per arm required",
            )
            runs = [p[mode] for p in pairs]
            for run in [*arm["warmup"], *runs]:
                _require(
                    run["fingerprint_sha256"] == fingerprint_sha256(arm["fingerprint"]),
                    "run fingerprint binding",
                )
            cost_metrics(arm["warmup"])
            _require(
                all(r["discarded"] is False for r in runs),
                "measured runs must exclude warmup",
            )
            for metric, value in cost_metrics(runs).items():
                actual = arm["reported"][metric]
                _require(
                    _number(actual)
                    and math.isclose(actual, value, rel_tol=1e-9, abs_tol=1e-12),
                    f"reported {metric} must equal raw-sample median",
                )
    except (AssertionError, KeyError, TypeError, ValueError, OverflowError) as error:
        raise AssertionError(f"{PUBLICATION_ONLY}: {error}") from error


def inventory_from_source(source):
    """Inspect init_batch_invariance and its local helpers, never execute them.

    Inventory includes conditional registrations/mutations on every branch, so
    it describes the full initializer's side effects, not only one dispatch lane.
    Unknown nonliteral registrations fail closed instead of guessing a version.
    """
    functions = {
        n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
    }
    _require(
        "init_batch_invariance" in functions, "installed initializer source required"
    )
    pending, visited, symbols = ["init_batch_invariance"], set(), set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in functions:
                    pending.append(node.func.id)
                symbol = ast.unparse(node.func)
                if isinstance(node.func, ast.Attribute) and node.func.attr == "impl":
                    first = node.args[0] if node.args else None
                    value = first.value if isinstance(first, ast.Constant) else None
                    _require(
                        isinstance(value, str),
                        "nonliteral aten registration needs inventory review",
                    )
                    assert isinstance(value, str)
                    symbols.add(value)
                elif symbol.startswith("torch.backends."):
                    symbols.add(symbol)
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                ):
                    symbol = ast.unparse(target)
                    if symbol.startswith("torch."):
                        symbols.add(symbol)
                    elif (
                        isinstance(target, ast.Subscript)
                        and ast.unparse(target.value) == "os.environ"
                    ):
                        symbols.add(ast.literal_eval(target.slice))
    _require(
        any(s.startswith("aten::") for s in symbols), "no aten registrations found"
    )
    return symbols


def installed_inventory(source_path=None):
    """Read the installed version's source, independent of line numbers or imports."""
    if source_path is None:
        spec = importlib.util.find_spec("vllm")
        origin = spec.origin if spec is not None else None
        _require(origin is not None, "installed vLLM source required")
        assert origin is not None
        source_path = (
            Path(origin).parent / "model_executor/determinism/batch_invariant.py"
        )
    source = Path(source_path).read_bytes()
    return inventory_from_source(
        source.decode("utf-8")
    ) | PORT_SIDE_EFFECTS, hashlib.sha256(source).hexdigest()
