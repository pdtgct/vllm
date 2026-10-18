# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel-profile record: per-operator accumulation evidence for the encoder.

The profile is evidence, not configuration: it selects no kernel and
qualifies nothing by being present. The engine owns the schema, operator
vocabulary, expected-set derivation, validation, qualification predicate and
the PORT-PREC-021 adapter; capture belongs to the harness, which builds the
record through :func:`build_kernel_profile`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from vllm.models.nemotron_asr.batch_invariance import accumulation_evidence_complete
from vllm.models.nemotron_asr.manifests import canonical_json, manifest_hash

__all__ = [
    "KERNEL_PROFILE_VERSION",
    "PROFILE_OPERATORS",
    "NON_REDUCTION",
    "KernelObservation",
    "expected_operators",
    "execution_fingerprint_digest",
    "build_kernel_profile",
    "validate_kernel_profile",
    "kernel_profile_qualified",
    "profile_identity",
    "prec021_evidence_complete",
]

KERNEL_PROFILE_VERSION = "nemotron-kernel-profile-v1"
_ACCUMULATION_TABLE = "operator-accumulation-v1"
NON_REDUCTION = "non_reduction"

PROFILE_OPERATORS: Mapping[str, str] = MappingProxyType(
    {
        "gemm": "gemm",
        "bmm": "gemm",
        "depthwise_conv1d": "convolution",
        "strided_conv2d": "convolution",
        "pointwise_conv": "convolution",
        "layer_norm": "normalization",
        "mean": "normalization",
        "sum": "normalization",
        "softmax": "attention",
        "log_softmax": "attention",
        "sdpa": "attention",
        "positional_projection": "positional",
    }
)

_SENSITIVE = {
    "gemm": "tf32_operand_rounding",
    "bmm": "tf32_operand_rounding",
    "positional_projection": "tf32_operand_rounding",
    "pointwise_conv": "bias_before_cast",
    "layer_norm": "statistics_accumulator",
    "mean": "sum_accumulator",
    "sum": "sum_accumulator",
    "softmax": "row_sum_accumulator",
    "log_softmax": "row_sum_accumulator",
    "sdpa": "fused_score_reduction",
}
_ACCUMULATORS = frozenset({"float32", "float16", "bfloat16"})
_KINDS = {
    "compiled": frozenset({"generated_kernel"}),
    "eager": frozenset({"dispatch", "pinned_backend"}),
}
_ALWAYS_EXPECTED = frozenset(
    {
        "gemm",
        "bmm",
        "layer_norm",
        "depthwise_conv1d",
        "strided_conv2d",
        "pointwise_conv",
    }
)
_CAPTURE_BOOLS = (
    "compiled",
    "graph_replay",
    "startup_scope",
    "eager_fallback",
    "synthetic_weights",
    "cudnn_benchmark",
    "cudnn_deterministic",
)
_CAPTURE_STRINGS = ("tool", "tool_version", "kernel_map_version")
_DECLARATIONS = ("positional_table_compute_dtype", "positional_table_storage_dtype")
_RECORD_KEYS = frozenset(
    {
        "version",
        "fingerprint_digest",
        "accumulation_table",
        "capture",
        "device",
        "software",
        "reduced_precision",
        "declarations",
        "rows",
        "unattributed",
    }
)
_ROW_KEYS = frozenset(
    {
        "operator",
        "class",
        "executed",
        "kernels",
        "accumulator",
        "resolved",
        "evidence",
        "sensitive",
    }
)
_ENTRY_KEYS = frozenset(
    {
        "operator",
        "backend",
        "execution",
        "accumulator",
        "kind",
        "symbol",
        "source_sha256",
    }
)
_PROFILE_FIELD = "kernel_profile"


@dataclass(frozen=True)
class KernelObservation:
    """@spec PORT-PREC-030: one captured device kernel claimed by a row.

    ``evidence`` is the kernel-map (or generated-kernel) evidence for the
    symbol; ``None`` means no map entry. ``compiled`` is per kernel, from
    Inductor provenance: kernels outside a compiled region stay eager.
    """

    operator: str
    symbol: str
    evidence: Mapping[str, Any] | None
    compiled: bool = False


def expected_operators(
    *, fp16_sdpa_lane: bool, startup_scope: bool, eager_fallback: bool = False
) -> frozenset[str]:
    """@spec PORT-PREC-032: executed operators the encoder must show.

    ``mean``, ``sum`` and ``log_softmax`` are never expected: the encoder's
    sums live inside the pinned compositions and the attention rows.
    """
    expected = set(_ALWAYS_EXPECTED)
    expected.add("sdpa" if fp16_sdpa_lane else "softmax")
    if startup_scope or eager_fallback:
        expected.add("positional_projection")
    return frozenset(expected)


def _profile_fields(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            where = f"{path}.{key}" if path else str(key)
            if isinstance(key, str) and key.startswith(_PROFILE_FIELD):
                found.append(where)
            found.extend(_profile_fields(item, where))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_profile_fields(item, f"{path}[{index}]"))
    return found


def execution_fingerprint_digest(fingerprint: Mapping[str, Any]) -> str:
    """@spec PORT-PREC-030: digest of a fingerprint computed without profile fields."""
    fields = _profile_fields(fingerprint)
    if fields:
        raise ValueError(
            f"execution fingerprint must not contain kernel-profile fields: {fields}"
        )
    return manifest_hash(fingerprint)


def _contract_operator(operator: str) -> str:
    """The PORT-PREC-021 adapter checks the positional projection as ``gemm``."""
    return "gemm" if operator == "positional_projection" else operator


def _sensitive(operator: str, reduced_precision: Mapping[str, Any]) -> dict[str, Any]:
    behaviour = _SENSITIVE.get(operator)
    sensitive: dict[str, Any] = {"behaviour": behaviour}
    if behaviour == "tf32_operand_rounding":
        sensitive["tf32_matmul_enabled"] = reduced_precision.get("tf32_matmul_enabled")
    return sensitive


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_manifest_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and _is_digest(value[len("sha256:") :])
    )


def _entry_valid(operator: str, entry: Mapping[str, Any]) -> bool:
    """@spec PORT-PREC-031: evidence that may establish an accumulator.

    Operand dtype, output dtype and numerical agreement are not evidence
    kinds; an entry naming none of the three kinds never resolves a row.
    """
    execution = entry.get("execution")
    backend = entry.get("backend")
    return (
        entry.get("operator") == _contract_operator(operator)
        and execution in _KINDS
        and entry.get("kind") in _KINDS[execution]
        and entry.get("accumulator") in _ACCUMULATORS
        and isinstance(backend, str)
        and bool(backend)
        and _is_digest(entry.get("source_sha256"))
        and isinstance(entry.get("symbol"), str)
        and bool(entry.get("symbol"))
        and (
            operator != "pointwise_conv"
            or isinstance(entry.get("bias_before_cast"), bool)
        )
    )


def _resolve(
    operator: str, kernels: Sequence[str], evidence: Sequence[Mapping[str, Any]]
) -> tuple[bool, str | None]:
    """@spec PORT-PREC-031: resolved only when every claimed kernel has valid
    evidence and all of it names one accumulator; never derived from dtype."""
    if not kernels:
        return False, None
    covered = {entry.get("symbol") for entry in evidence}
    if not set(kernels) <= covered or not all(
        _entry_valid(operator, entry) for entry in evidence
    ):
        return False, None
    accumulators = {entry["accumulator"] for entry in evidence}
    if len(accumulators) != 1:
        return False, None
    return True, accumulators.pop()


def _plain(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    try:
        plain = json.loads(canonical_json(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not JSON-serializable: {error}") from error
    return plain


def _entry(
    observation: KernelObservation, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    entry = {
        "operator": _contract_operator(observation.operator),
        "backend": evidence.get("backend"),
        "execution": "compiled" if observation.compiled else "eager",
        "accumulator": evidence.get("accumulator"),
        "kind": evidence.get("kind"),
        "symbol": observation.symbol,
        "source_sha256": evidence.get("source_sha256"),
    }
    if "pinned_build" in evidence:
        entry["pinned_build"] = evidence["pinned_build"]
    if observation.operator == "pointwise_conv":
        entry["bias_before_cast"] = evidence.get("bias_before_cast")
    return entry


def build_kernel_profile(
    *,
    execution_fingerprint: Mapping[str, Any],
    capture: Mapping[str, Any],
    device: Mapping[str, Any],
    software: Mapping[str, Any],
    reduced_precision: Mapping[str, Any],
    declarations: Mapping[str, Any],
    observations: Iterable[KernelObservation],
    unattributed: Iterable[str],
) -> dict[str, Any]:
    """@spec PORT-PREC-014, PORT-PREC-030, PORT-PREC-031: assemble one record.

    Raises ``ValueError`` on an unknown operator, a malformed observation or
    a record that does not validate. Output is deterministic in observation
    and unattributed order.
    """
    observations = tuple(observations)
    unattributed = tuple(unattributed)
    digest = execution_fingerprint_digest(execution_fingerprint)
    capture_plain = _plain(capture, "capture")
    reduced_plain = _plain(reduced_precision, "reduced_precision")
    kernels: dict[str, set[str]] = {name: set() for name in PROFILE_OPERATORS}
    entries: dict[str, dict[str, dict[str, Any]]] = {
        name: {} for name in PROFILE_OPERATORS
    }
    missing: set[tuple[str, str]] = set()
    claimed: set[tuple[str, str]] = set()
    for observation in observations:
        if observation.operator not in PROFILE_OPERATORS:
            raise ValueError(
                f"unknown kernel-profile operator {observation.operator!r}"
            )
        if not isinstance(observation.symbol, str) or not observation.symbol:
            raise ValueError(
                f"{observation.operator}: kernel symbol must be a non-empty string"
            )
        if not isinstance(observation.compiled, bool):
            raise ValueError(f"{observation.operator}: compiled must be a bool")
        if observation.compiled and capture_plain.get("compiled") is not True:
            raise ValueError(
                f"{observation.operator}: compiled kernel {observation.symbol!r} in a "
                "non-compiled capture"
            )
        key = (observation.operator, observation.symbol)
        kernels[observation.operator].add(observation.symbol)
        if observation.evidence is None:
            missing.add(key)
            continue
        if not isinstance(observation.evidence, Mapping):
            raise ValueError(
                f"{observation.operator}: evidence must be a mapping or None"
            )
        claimed.add(key)
        entry = _plain(_entry(observation, observation.evidence), "evidence")
        entries[observation.operator][canonical_json(entry)] = entry
    if missing & claimed:
        raise ValueError(
            f"kernels both with and without evidence: {sorted(missing & claimed)}"
        )

    rows: dict[str, Any] = {}
    for name, cls in PROFILE_OPERATORS.items():
        symbols = sorted(kernels[name])
        evidence = [
            entries[name][key]
            for key in sorted(
                entries[name], key=lambda k: (entries[name][k]["symbol"], k)
            )
        ]
        resolved, accumulator = _resolve(name, symbols, evidence)
        rows[name] = {
            "operator": name,
            "class": cls,
            "executed": bool(symbols),
            "kernels": symbols,
            "accumulator": accumulator,
            "resolved": resolved,
            "evidence": evidence,
            "sensitive": _sensitive(name, reduced_plain),
        }
    for symbol in unattributed:
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("unattributed kernel symbols must be non-empty strings")
    profile = {
        "version": KERNEL_PROFILE_VERSION,
        "fingerprint_digest": digest,
        "accumulation_table": _ACCUMULATION_TABLE,
        "capture": capture_plain,
        "device": _plain(device, "device"),
        "software": _plain(software, "software"),
        "reduced_precision": reduced_plain,
        "declarations": _plain(declarations, "declarations"),
        "rows": rows,
        "unattributed": sorted(set(unattributed)),
    }
    errors = validate_kernel_profile(profile)
    if errors:
        raise ValueError("invalid kernel profile: " + "; ".join(errors))
    return profile


def _validate_capture(capture: Any) -> list[str]:
    if not isinstance(capture, Mapping):
        return ["capture must be a mapping"]
    errors = [
        f"capture.{key} must be a non-empty string"
        for key in _CAPTURE_STRINGS
        if not isinstance(capture.get(key), str) or not capture.get(key)
    ]
    if not _is_manifest_hash(capture.get("kernel_map_sha256")):
        errors.append(
            "capture.kernel_map_sha256 must be a manifest hash 'sha256:<64 lowercase "
            "hex>'"
        )
    errors += [
        f"capture.{key} must be a bool"
        for key in _CAPTURE_BOOLS
        if not isinstance(capture.get(key), bool)
    ]
    if not isinstance(capture.get("sdpa_backends"), Mapping):
        errors.append("capture.sdpa_backends must be a mapping")
    if capture.get("batch_invariant_mode") not in ("on", "off"):
        errors.append("capture.batch_invariant_mode must be 'on' or 'off'")
    purpose = capture.get("purpose")
    packet = capture.get("qualification_packet")
    if purpose == "qualification":
        if not isinstance(packet, str) or not packet:
            errors.append(
                "capture.qualification_packet must be a non-empty string for "
                "qualification"
            )
    elif purpose == "smoke":
        if packet is not None:
            errors.append(
                "capture.qualification_packet must be absent or null for smoke"
            )
    else:
        errors.append("capture.purpose must be 'smoke' or 'qualification'")
    return errors


def _declarations_present(declarations: Any) -> bool:
    return isinstance(declarations, Mapping) and all(
        isinstance(declarations.get(key), str) and bool(declarations.get(key))
        for key in _DECLARATIONS
    )


def _valid_pin(pin: Any) -> bool:
    return (
        isinstance(pin, Mapping)
        and bool(pin)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in pin.items())
    )


def _validate_row(
    name: str, row: Any, capture: Mapping[str, Any], reduced: Mapping[str, Any]
) -> list[str]:
    if not isinstance(row, Mapping) or set(row) != _ROW_KEYS:
        return [f"rows.{name}: fields must be {sorted(_ROW_KEYS)}"]
    errors = []
    if row["operator"] != name or row["class"] != PROFILE_OPERATORS[name]:
        errors.append(f"rows.{name}: operator/class mismatch")
    kernels, evidence = row["kernels"], row["evidence"]
    if not isinstance(kernels, list) or not all(
        isinstance(k, str) and k for k in kernels
    ):
        return [*errors, f"rows.{name}: kernels must be a list of symbols"]
    if kernels != sorted(set(kernels)):
        errors.append(f"rows.{name}: kernels must be sorted and unique")
    if row["executed"] is not bool(kernels):
        errors.append(
            f"rows.{name}: executed must equal whether any kernel was observed"
        )
    if not isinstance(evidence, list) or not all(
        isinstance(e, Mapping) for e in evidence
    ):
        return [*errors, f"rows.{name}: evidence must be a list of entries"]
    entry_keys = _ENTRY_KEYS | (
        {"bias_before_cast"} if name == "pointwise_conv" else set()
    )
    for entry in evidence:
        allowed_keys = entry_keys | (
            {"pinned_build"} if "pinned_build" in entry else set()
        )
        if set(entry) != allowed_keys:
            errors.append(
                f"rows.{name}: evidence entry fields must be {sorted(entry_keys)} "
                "(plus 'pinned_build' only on kind 'pinned_backend' evidence)"
            )
        elif "pinned_build" in entry and entry["kind"] != "pinned_backend":
            errors.append(
                f"rows.{name}: pinned_build is allowed only on 'pinned_backend' "
                "evidence"
            )
        elif "pinned_build" in entry and not _valid_pin(entry["pinned_build"]):
            errors.append(
                f"rows.{name}: pinned_build must be a non-empty mapping of str to str"
            )
        elif entry["operator"] != _contract_operator(name):
            errors.append(
                f"rows.{name}: evidence operator must be {_contract_operator(name)!r}"
            )
        elif entry["execution"] not in _KINDS:
            errors.append(
                f"rows.{name}: evidence execution must be 'compiled' or 'eager'"
            )
        elif entry["execution"] == "compiled" and capture.get("compiled") is not True:
            errors.append(f"rows.{name}: compiled evidence in a non-compiled capture")
        elif entry["symbol"] not in kernels:
            errors.append(
                f"rows.{name}: evidence symbol {entry['symbol']!r} is not a claimed "
                "kernel"
            )
    resolved, accumulator = _resolve(name, kernels, evidence)
    if row["resolved"] is not resolved:
        errors.append(
            f"rows.{name}: resolved={row['resolved']!r} but evidence derives {resolved}"
        )
    if row["accumulator"] != accumulator:
        errors.append(
            f"rows.{name}: accumulator={row['accumulator']!r} but evidence derives "
            f"{accumulator!r}"
        )
    if row["sensitive"] != _sensitive(name, reduced):
        errors.append(f"rows.{name}: sensitive declaration mismatch")
    return errors


def validate_kernel_profile(profile: Mapping[str, Any]) -> list[str]:
    """@spec PORT-PREC-030, PORT-PREC-031: schema errors; empty means well-formed.

    Re-derives each row's resolution and accumulator from its evidence, so a
    row claiming resolution without valid evidence is reported.
    """
    if not isinstance(profile, Mapping):
        return ["profile must be a mapping"]
    errors = []
    if set(profile) != _RECORD_KEYS:
        errors.append(f"record fields must be {sorted(_RECORD_KEYS)}")
    if profile.get("version") != KERNEL_PROFILE_VERSION:
        errors.append(f"version must be {KERNEL_PROFILE_VERSION!r}")
    if profile.get("accumulation_table") != _ACCUMULATION_TABLE:
        errors.append(f"accumulation_table must be {_ACCUMULATION_TABLE!r}")
    digest = profile.get("fingerprint_digest")
    if not _is_manifest_hash(digest):
        errors.append("fingerprint_digest must be a sha256 manifest hash")
    capture = profile.get("capture")
    errors += _validate_capture(capture)
    for key in ("device", "software", "reduced_precision"):
        if not isinstance(profile.get(key), Mapping):
            errors.append(f"{key} must be a mapping")
    for section, required in (
        ("device", ("name", "compute_capability")),
        ("software", ("torch", "cuda", "cudnn", "triton")),
    ):
        identity = profile.get(section)
        if isinstance(identity, Mapping):
            errors += [
                f"{section}.{key} is missing" for key in required if key not in identity
            ]
    if not _declarations_present(profile.get("declarations")):
        errors.append(f"declarations must name {list(_DECLARATIONS)}")
    rows = profile.get("rows")
    if not isinstance(rows, Mapping) or set(rows) != set(PROFILE_OPERATORS):
        errors.append("rows must contain exactly one row per profile operator")
    else:
        reduced = profile.get("reduced_precision")
        for name in PROFILE_OPERATORS:
            errors += _validate_row(
                name,
                rows[name],
                capture if isinstance(capture, Mapping) else {},
                reduced if isinstance(reduced, Mapping) else {},
            )
    unattributed = profile.get("unattributed")
    if (
        not isinstance(unattributed, list)
        or not all(isinstance(s, str) and s for s in unattributed)
        or unattributed != sorted(set(unattributed))
    ):
        errors.append("unattributed must be a sorted unique list of kernel symbols")
    return errors


def _executed_rows(profile: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = profile.get("rows")
    if not isinstance(rows, Mapping):
        return {}
    return {
        name: row
        for name, row in rows.items()
        if name in PROFILE_OPERATORS
        and isinstance(row, Mapping)
        and row.get("executed") is True
    }


def kernel_profile_qualified(
    profile: Mapping[str, Any],
    *,
    verdict_fingerprint: Mapping[str, Any],
    qualification_packet: str,
) -> tuple[bool, list[str]]:
    """@spec PORT-PREC-014, PORT-PREC-025, PORT-PREC-032, PORT-PREC-033.

    Every condition must hold.

    Collects every failing reason. A resolved float16 accumulator is a
    declaration here, not a failure; PORT-PREC-021 is a separate adapter.
    """
    reasons = [
        f"not well-formed: {error}" for error in validate_kernel_profile(profile)
    ]
    capture = profile.get("capture")
    capture = capture if isinstance(capture, Mapping) else {}
    if capture.get("purpose") != "qualification":
        reasons.append(f"purpose is {capture.get('purpose')!r}, not 'qualification'")
    if (
        not qualification_packet
        or capture.get("qualification_packet") != qualification_packet
    ):
        reasons.append(
            f"capture.qualification_packet {capture.get('qualification_packet')!r} "
            f"does not match {qualification_packet!r}"
        )
    try:
        if profile.get("fingerprint_digest") != execution_fingerprint_digest(
            verdict_fingerprint
        ):
            reasons.append(
                "fingerprint digest does not match the verdict's execution fingerprint"
            )
    except ValueError as error:
        reasons.append(f"verdict fingerprint digest unavailable: {error}")
    executed = _executed_rows(profile)
    selection = verdict_fingerprint.get("optimization_selection")
    software = profile.get("software")
    software = software if isinstance(software, Mapping) else {}
    verdict_software = verdict_fingerprint.get("software")
    if not isinstance(verdict_software, Mapping) or not verdict_software:
        reasons.append("verdict fingerprint software is missing or empty")
    else:
        reasons += [
            f"verdict fingerprint software is missing {component!r}"
            for component in ("torch", "cuda", "cudnn", "triton")
            if component not in verdict_software
        ]
        for component, version in verdict_software.items():
            if component not in software or software[component] != version:
                reasons.append(
                    f"software.{component} does not match the verdict fingerprint"
                )
    for name, row in executed.items():
        evidence = row.get("evidence")
        for entry in evidence if isinstance(evidence, list) else ():
            if not isinstance(entry, Mapping) or entry.get("kind") != "pinned_backend":
                continue
            pin = entry.get("pinned_build")
            if not isinstance(pin, Mapping) or not pin:
                reasons.append(f"{name}: pinned_build is missing or empty")
            elif any(
                component not in software or software[component] != version
                for component, version in pin.items()
            ):
                reasons.append(f"{name}: pinned_build does not match profile software")
    for key in ("compiled", "graph_replay"):
        selected = selection.get(key) if isinstance(selection, Mapping) else None
        if not isinstance(selected, bool):
            reasons.append(
                f"verdict fingerprint optimization_selection.{key} is missing or not a "
                "bool"
            )
        elif capture.get(key) is not selected:
            reasons.append(f"capture.{key} does not match the verdict fingerprint")
    mode = verdict_fingerprint.get("batch_invariant_mode", "off")
    if mode not in ("on", "off") or capture.get("batch_invariant_mode") != mode:
        reasons.append(
            "capture.batch_invariant_mode does not match the verdict fingerprint"
        )
    lane = selection.get("fp16_sdpa_lane") if isinstance(selection, Mapping) else None
    if not isinstance(lane, bool):
        reasons.append(
            "verdict fingerprint optimization_selection.fp16_sdpa_lane is missing"
        )
    else:
        expected = expected_operators(
            fp16_sdpa_lane=lane,
            startup_scope=capture.get("startup_scope") is True,
            eager_fallback=capture.get("eager_fallback") is True,
        )
        if set(executed) != expected:
            reasons.append(
                f"executed operators {sorted(executed)} != expected {sorted(expected)}"
            )
            reasons += [
                f"unexpected executed operator: {name}"
                for name in sorted(set(executed) - expected)
            ]
            reasons += [
                f"expected operator not executed: {name}"
                for name in sorted(expected - set(executed))
            ]
    reasons += [
        f"executed row unresolved: {name}"
        for name, row in sorted(executed.items())
        if not row.get("resolved")
    ]
    if profile.get("unattributed"):
        reasons.append(f"unattributed kernels: {profile.get('unattributed')}")
    if not _declarations_present(profile.get("declarations")):
        reasons.append("positional declarations missing")
    return not reasons, reasons


def profile_identity(profile: Mapping[str, Any]) -> tuple[str, str]:
    """@spec PORT-PREC-032: (fingerprint digest, profile manifest hash)."""
    return profile["fingerprint_digest"], manifest_hash(profile)


def prec021_evidence_complete(
    profile: Mapping[str, Any],
    *,
    verdict_fingerprint: Mapping[str, Any],
    qualification_packet: str,
) -> tuple[bool, list[str]]:
    """@spec PORT-PREC-021: float32 accumulation evidence for every executed row.

    ``positional_projection`` is checked as ``gemm``; its entries carry the
    contract operator name.
    """
    qualified, reasons = kernel_profile_qualified(
        profile,
        verdict_fingerprint=verdict_fingerprint,
        qualification_packet=qualification_packet,
    )
    if not qualified:
        return False, reasons
    for name, row in sorted(_executed_rows(profile).items()):
        if not row.get("resolved"):
            reasons.append(f"{name}: executed row unresolved")
        evidence = row.get("evidence")
        for entry in evidence if isinstance(evidence, list) else ():
            backend, execution = entry.get("backend"), entry.get("execution")
            if not (
                isinstance(backend, str)
                and isinstance(execution, str)
                and accumulation_evidence_complete(
                    _contract_operator(name),
                    backend=backend,
                    execution=execution,
                    evidence=entry,
                )
            ):
                reasons.append(
                    f"{name}: evidence for {entry.get('symbol')!r} is not float32 "
                    "PORT-PREC-021 evidence"
                )
    return not reasons, reasons
