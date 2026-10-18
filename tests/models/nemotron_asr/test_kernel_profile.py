# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PORT-PREC-014/021/030/031/032 kernel-profile record contract (CPU)."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from vllm.models.nemotron_asr import kernel_profile as kp
from vllm.models.nemotron_asr.batch_invariance import operator_accumulation_contract
from vllm.models.nemotron_asr.manifests import canonical_json, manifest_hash

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]

DIGEST = "ab" * 32
PACKET = "v030-requal-a100-encoder-fp16"
CORE = (
    "gemm",
    "bmm",
    "layer_norm",
    "depthwise_conv1d",
    "strided_conv2d",
    "pointwise_conv",
)


def _fingerprint(
    lane: bool = True, *, compiled: bool = False, graph_replay: bool = False
) -> dict[str, Any]:
    return {
        "precision_policy": "encoder-fp16-v1",
        "batch_invariant_mode": "on",
        "optimization_selection": {
            "fp16_sdpa_lane": lane,
            "compiled": compiled,
            "graph_replay": graph_replay,
        },
        "software": {
            "torch": "2.9.0",
            "cuda": "12.8",
            "cudnn": "9.10",
            "triton": "3.4.0",
        },
    }


def _capture(**overrides: Any) -> dict[str, Any]:
    capture = {
        "tool": "nemotron-kernel-capture",
        "tool_version": "1.0.0",
        "kernel_map_version": "kernel-map-v1",
        "kernel_map_sha256": "sha256:" + DIGEST,
        "compiled": False,
        "graph_replay": False,
        "batch_invariant_mode": "on",
        "startup_scope": False,
        "eager_fallback": False,
        "purpose": "qualification",
        "qualification_packet": PACKET,
        "synthetic_weights": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "sdpa_backends": {
            "flash": False,
            "efficient": True,
            "math": True,
            "cudnn": False,
        },
    }
    capture.update(overrides)
    return capture


def _evidence(
    operator: str, *, kind: str = "dispatch", accumulator: str = "float32", **extra: Any
) -> dict:
    evidence: dict[str, Any] = {
        "backend": "triton",
        "kind": kind,
        "accumulator": accumulator,
        "source_sha256": DIGEST,
    }
    if operator == "pointwise_conv":
        evidence["bias_before_cast"] = True
    if kind == "pinned_backend":
        evidence["pinned_build"] = {"torch": "2.9.0"}
    evidence.update(extra)
    return evidence


def _obs(
    operator: str, symbol: str | None = None, *, compiled: bool = False, **evidence: Any
) -> kp.KernelObservation:
    return kp.KernelObservation(
        operator=operator,
        symbol=symbol or f"{operator}_kernel",
        evidence=_evidence(operator, **evidence),
        compiled=compiled,
    )


def _valid_observations(
    lane: bool = True, *, compiled: bool = False
) -> list[kp.KernelObservation]:
    kind = "generated_kernel" if compiled else "dispatch"
    names = (*CORE, "sdpa" if lane else "softmax")
    return [_obs(name, compiled=compiled, kind=kind) for name in names]


def _build(
    observations=None,
    *,
    fingerprint=None,
    capture=None,
    unattributed=(),
    reduced_precision=None,
    declarations=None,
) -> dict[str, Any]:
    capture = capture if capture is not None else _capture()
    fingerprint = (
        fingerprint
        if fingerprint is not None
        else _fingerprint(
            compiled=capture["compiled"], graph_replay=capture["graph_replay"]
        )
    )
    return kp.build_kernel_profile(
        execution_fingerprint=fingerprint
        if fingerprint is not None
        else _fingerprint(),
        capture=capture if capture is not None else _capture(),
        device={"name": "NVIDIA A100-SXM4-80GB", "compute_capability": [8, 0]},
        software={"torch": "2.9.0", "cuda": "12.8", "cudnn": "9.10", "triton": "3.4.0"},
        reduced_precision=reduced_precision
        if reduced_precision is not None
        else {"tf32_matmul_enabled": False, "tf32_cudnn_conv_enabled": False},
        declarations=declarations
        if declarations is not None
        else {
            "positional_table_compute_dtype": "float32",
            "positional_table_storage_dtype": "float16",
        },
        observations=_valid_observations() if observations is None else observations,
        unattributed=unattributed,
    )


def _qualified(profile, fingerprint=None, packet: str = PACKET):
    return kp.kernel_profile_qualified(
        profile,
        verdict_fingerprint=fingerprint
        if fingerprint is not None
        else _fingerprint(
            compiled=profile["capture"]["compiled"],
            graph_replay=profile["capture"]["graph_replay"],
        ),
        qualification_packet=packet,
    )


def _prec021(profile, fingerprint=None, packet: str = PACKET):
    return kp.prec021_evidence_complete(
        profile,
        verdict_fingerprint=fingerprint
        if fingerprint is not None
        else _fingerprint(
            compiled=profile["capture"]["compiled"],
            graph_replay=profile["capture"]["graph_replay"],
        ),
        qualification_packet=packet,
    )


# -- vocabulary and expected set -------------------------------------------


def test_vocabulary_equals_contract_rows_plus_positional_projection():
    """@spec PORT-PREC-030:
    one row per operator-accumulation-v1 operator + projection.
    """
    rows = operator_accumulation_contract(device_capability=(8, 0))["rows"]
    assert set(kp.PROFILE_OPERATORS) == set(rows) | {"positional_projection"}
    assert kp.KERNEL_PROFILE_VERSION == "nemotron-kernel-profile-v1"
    assert kp.NON_REDUCTION == "non_reduction"
    classes = {
        "gemm": {"gemm", "bmm"},
        "convolution": {"depthwise_conv1d", "strided_conv2d", "pointwise_conv"},
        "normalization": {"layer_norm", "mean", "sum"},
        "attention": {"softmax", "log_softmax", "sdpa"},
        "positional": {"positional_projection"},
    }
    for cls, ops in classes.items():
        assert {op for op, c in kp.PROFILE_OPERATORS.items() if c == cls} == ops


def test_public_surface_is_exactly_the_contract():
    assert set(kp.__all__) == {
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
    }


@pytest.mark.parametrize(
    ("lane", "startup", "fallback", "extra"),
    [
        (True, False, False, {"sdpa"}),
        (False, False, False, {"softmax"}),
        (True, True, False, {"sdpa", "positional_projection"}),
        (False, False, True, {"softmax", "positional_projection"}),
    ],
)
def test_expected_operators(lane, startup, fallback, extra):
    """@spec PORT-PREC-032: derived from optimization selection and capture scope."""
    expected = kp.expected_operators(
        fp16_sdpa_lane=lane, startup_scope=startup, eager_fallback=fallback
    )
    assert expected == frozenset(CORE) | extra
    assert not expected & {"mean", "sum", "log_softmax"}


# -- identity ----------------------------------------------------------------


@pytest.mark.parametrize("key", ["kernel_profile", "kernel_profile_digest"])
def test_fingerprint_with_profile_field_rejected(key):
    """@spec PORT-PREC-030: fingerprint computed without any profile field."""
    with pytest.raises(ValueError):
        kp.execution_fingerprint_digest({**_fingerprint(), key: "x"})
    with pytest.raises(ValueError):
        kp.execution_fingerprint_digest({**_fingerprint(), "nested": {key: "x"}})
    with pytest.raises(ValueError):
        _build(fingerprint={**_fingerprint(), key: "x"})


def test_profile_identity_pair():
    """@spec PORT-PREC-032: identity = (fingerprint digest, profile manifest hash)."""
    profile = _build()
    assert profile["fingerprint_digest"] == manifest_hash(_fingerprint())
    assert "execution_fingerprint" not in profile
    assert kp.profile_identity(profile) == (
        manifest_hash(_fingerprint()),
        manifest_hash(profile),
    )


def test_record_shape_and_json():
    """@spec PORT-PREC-030: record fields, every row present, JSON-serializable."""
    profile = _build(unattributed=["z_kernel", "a_kernel", "z_kernel"])
    assert set(profile) == {
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
    assert profile["version"] == "nemotron-kernel-profile-v1"
    assert profile["accumulation_table"] == "operator-accumulation-v1"
    assert set(profile["rows"]) == set(kp.PROFILE_OPERATORS)
    assert profile["unattributed"] == ["a_kernel", "z_kernel"]
    assert json.loads(canonical_json(profile)) == profile
    row = profile["rows"]["gemm"]
    assert set(row) == {
        "operator",
        "class",
        "executed",
        "kernels",
        "accumulator",
        "resolved",
        "evidence",
        "sensitive",
    }
    assert row["evidence"] == [
        {
            "operator": "gemm",
            "backend": "triton",
            "execution": "eager",
            "accumulator": "float32",
            "kind": "dispatch",
            "symbol": "gemm_kernel",
            "source_sha256": DIGEST,
        }
    ]
    assert profile["rows"]["pointwise_conv"]["evidence"][0]["bias_before_cast"] is True
    assert kp.validate_kernel_profile(profile) == []


def test_sensitive_declarations():
    """@spec PORT-PREC-014: declared sensitive behaviour per row."""
    rows = _build(reduced_precision={"tf32_matmul_enabled": True})["rows"]
    for name in ("gemm", "bmm", "positional_projection"):
        assert rows[name]["sensitive"] == {
            "behaviour": "tf32_operand_rounding",
            "tf32_matmul_enabled": True,
        }
    assert rows["pointwise_conv"]["sensitive"] == {"behaviour": "bias_before_cast"}
    assert rows["layer_norm"]["sensitive"] == {"behaviour": "statistics_accumulator"}
    assert (
        rows["mean"]["sensitive"]
        == rows["sum"]["sensitive"]
        == {"behaviour": "sum_accumulator"}
    )
    assert rows["softmax"]["sensitive"] == {"behaviour": "row_sum_accumulator"}
    assert rows["log_softmax"]["sensitive"] == {"behaviour": "row_sum_accumulator"}
    assert rows["sdpa"]["sensitive"] == {"behaviour": "fused_score_reduction"}
    assert rows["depthwise_conv1d"]["sensitive"] == {"behaviour": None}


def test_unknown_operator_rejected():
    with pytest.raises(ValueError):
        _build([kp.KernelObservation(operator="conv3d", symbol="k", evidence=None)])


def test_compiled_kernel_outside_compiled_capture_rejected():
    with pytest.raises(ValueError):
        _build([_obs("gemm", compiled=True, kind="generated_kernel")])


def test_missing_capture_key_rejected():
    capture = _capture()
    del capture["sdpa_backends"]
    with pytest.raises(ValueError):
        _build(capture=capture)


@pytest.mark.parametrize(
    "value",
    [DIGEST, "sha256:" + DIGEST.upper(), "sha256:abc", "md5:" + DIGEST, "", None],
)
def test_kernel_map_digest_must_be_manifest_hash(value):
    """kernel_map_sha256 uses the engine's manifest_hash form."""
    with pytest.raises(ValueError):
        _build(capture=_capture(kernel_map_sha256=value))
    profile = _build()
    profile["capture"]["kernel_map_sha256"] = value
    assert any("kernel_map_sha256" in e for e in kp.validate_kernel_profile(profile))


def test_packet_binding_by_purpose():
    with pytest.raises(ValueError):
        _build(capture=_capture(qualification_packet=""))
    with pytest.raises(ValueError):
        _build(capture=_capture(purpose="smoke"))  # smoke must not name a packet
    smoke = _build(capture=_capture(purpose="smoke", qualification_packet=None))
    assert kp.validate_kernel_profile(smoke) == []


# -- resolution (PORT-PREC-031) ----------------------------------------------


def test_fully_valid_profile_qualifies():
    """@spec PORT-PREC-032: all conditions hold."""
    profile = _build()
    assert all(profile["rows"][name]["resolved"] for name in (*CORE, "sdpa"))
    assert _qualified(profile) == (True, [])
    assert _prec021(profile) == (True, [])


def test_operand_dtype_only_claim_unresolved():
    """@spec PORT-PREC-014, PORT-PREC-031: no map entry or dtype-only evidence."""
    observations = _valid_observations()
    observations[0] = kp.KernelObservation(
        operator="gemm", symbol="gemm_kernel", evidence=None
    )
    profile = _build(observations)
    row = profile["rows"]["gemm"]
    assert row["executed"] and not row["resolved"] and row["accumulator"] is None
    assert row["evidence"] == []
    ok, reasons = _qualified(profile)
    assert not ok and any("gemm" in r and "unresolved" in r for r in reasons)

    dtype_only = {
        "backend": "triton",
        "accumulator": "float32",
        "operand_dtype": "float16",
    }
    observations[0] = kp.KernelObservation(
        operator="gemm", symbol="gemm_kernel", evidence=dtype_only
    )
    profile = _build(observations)
    assert not profile["rows"]["gemm"]["resolved"]
    assert profile["rows"]["gemm"]["accumulator"] is None
    assert not _qualified(profile)[0]


def test_one_kernel_without_evidence_leaves_row_unresolved():
    observations = [
        *_valid_observations(),
        kp.KernelObservation("gemm", "gemm_other", None),
    ]
    profile = _build(observations)
    assert profile["rows"]["gemm"]["kernels"] == ["gemm_kernel", "gemm_other"]
    assert not profile["rows"]["gemm"]["resolved"]


def test_mixed_accumulators_unresolved():
    observations = [
        *_valid_observations(),
        _obs("bmm", "bmm_fp16", accumulator="float16"),
    ]
    profile = _build(observations)
    assert not profile["rows"]["bmm"]["resolved"]
    assert profile["rows"]["bmm"]["accumulator"] is None
    assert not _qualified(profile)[0]


@pytest.mark.parametrize("kind", ["dispatch", "pinned_backend"])
def test_compiled_kernel_rejects_dispatch_and_pinned(kind):
    """@spec PORT-PREC-031: compiled execution accepts only generated_kernel."""
    observations = _valid_observations(compiled=True)
    observations[0] = _obs("gemm", compiled=True, kind=kind)
    profile = _build(observations, capture=_capture(compiled=True, graph_replay=True))
    assert profile["rows"]["gemm"]["evidence"][0]["execution"] == "compiled"
    assert not profile["rows"]["gemm"]["resolved"]
    assert all(profile["rows"][n]["resolved"] for n in (*CORE[1:], "sdpa"))
    assert not _qualified(profile)[0]


def test_eager_kernel_rejects_generated_kernel():
    observations = _valid_observations()
    observations[0] = _obs("gemm", kind="generated_kernel")
    profile = _build(observations)
    assert not profile["rows"]["gemm"]["resolved"]


def test_compiled_capture_with_eager_startup_projection_qualifies():
    """Per-kernel execution: startup projection stays eager inside a compiled capture.

    Opaque/extern kernels in the compiled graph carry generated_kernel evidence.
    """
    observations = [
        *_valid_observations(compiled=True),
        _obs("gemm", "nemotron_bi_mm_extern", compiled=True, kind="generated_kernel"),
        _obs("positional_projection", "proj_kernel", kind="pinned_backend"),
    ]
    profile = _build(
        observations,
        capture=_capture(compiled=True, graph_replay=True, startup_scope=True),
    )
    entry = profile["rows"]["positional_projection"]["evidence"][0]
    assert entry["execution"] == "eager" and entry["operator"] == "gemm"
    assert _qualified(profile) == (True, [])
    assert _prec021(profile) == (True, [])


def test_symbol_with_compiled_and_eager_entries_allowed():
    """One symbol may carry both a compiled and an eager evidence entry in a row."""
    observations = [
        *_valid_observations(compiled=True),
        _obs("gemm", "gemm_kernel", kind="dispatch"),
    ]
    profile = _build(observations, capture=_capture(compiled=True, graph_replay=True))
    row = profile["rows"]["gemm"]
    assert row["kernels"] == ["gemm_kernel"]
    assert sorted(e["execution"] for e in row["evidence"]) == ["compiled", "eager"]
    assert row["resolved"] and row["accumulator"] == "float32"
    assert kp.validate_kernel_profile(profile) == []
    assert _qualified(profile) == (True, [])
    assert _prec021(profile) == (True, [])


# -- qualification (PORT-PREC-032) -------------------------------------------


def test_unattributed_unqualified():
    ok, reasons = _qualified(_build(unattributed=["mystery_reduce_kernel"]))
    assert not ok and any("unattributed" in r for r in reasons)


def test_softmax_executed_while_lane_selected_unqualified():
    observations = [*_valid_observations()[:-1], _obs("softmax")]
    ok, reasons = _qualified(_build(observations))
    assert not ok
    assert any("softmax" in r for r in reasons) and any("sdpa" in r for r in reasons)


@pytest.mark.parametrize("extra", ["mean", "sum", "log_softmax"])
def test_unexpected_reduction_unqualified(extra):
    """mean/sum/log_softmax are never expected for the encoder."""
    ok, reasons = _qualified(_build([*_valid_observations(), _obs(extra)]))
    assert not ok and any("unexpected" in r and extra in r for r in reasons)


def test_missing_lane_selection_unqualified():
    fingerprint = _fingerprint()
    del fingerprint["optimization_selection"]["fp16_sdpa_lane"]
    profile = _build(fingerprint=fingerprint)
    ok, reasons = _qualified(profile, fingerprint)
    assert not ok and any("fp16_sdpa_lane" in r for r in reasons)


def test_smoke_purpose_unqualified():
    profile = _build(capture=_capture(purpose="smoke", qualification_packet=None))
    ok, reasons = _qualified(profile)
    assert not ok and any("purpose" in r for r in reasons)


def test_packet_mismatch_unqualified():
    ok, reasons = _qualified(_build(), packet="another-packet")
    assert not ok and any("qualification_packet" in r for r in reasons)


def test_digest_mismatch_unqualified():
    ok, reasons = _qualified(
        _build(), _fingerprint(lane=False) | {"batch_invariant_mode": "off"}
    )
    assert not ok and any("digest" in r for r in reasons)


def test_missing_declarations_unqualified():
    profile = _build()
    profile["declarations"] = {}
    ok, reasons = _qualified(profile)
    assert not ok and any("declarations" in r for r in reasons)


def test_float16_accumulator_prec014_qualified_but_not_prec021():
    """@spec PORT-PREC-014, PORT-PREC-021: fp16 is a declaration, not BI evidence."""
    observations = _valid_observations()
    observations[-1] = _obs("sdpa", accumulator="float16")
    profile = _build(observations)
    assert (
        profile["rows"]["sdpa"]["resolved"]
        and profile["rows"]["sdpa"]["accumulator"] == "float16"
    )
    assert _qualified(profile) == (True, [])
    ok, reasons = _prec021(profile)
    assert not ok and any("sdpa" in r for r in reasons)


def test_prec021_positional_projection_checked_as_gemm():
    observations = [
        *_valid_observations(),
        _obs("positional_projection", kind="pinned_backend"),
    ]
    profile = _build(observations, capture=_capture(startup_scope=True))
    assert profile["rows"]["positional_projection"]["evidence"][0]["operator"] == "gemm"
    assert _prec021(profile) == (True, [])


def test_prec021_pointwise_requires_bias_before_cast():
    observations = _valid_observations()
    observations[CORE.index("pointwise_conv")] = _obs(
        "pointwise_conv", bias_before_cast=False
    )
    profile = _build(observations)
    assert profile["rows"]["pointwise_conv"]["resolved"]
    ok, reasons = _prec021(profile)
    assert not ok and any("pointwise_conv" in r for r in reasons)


def test_prec021_requires_resolved_rows():
    observations = _valid_observations()
    observations[1] = kp.KernelObservation("bmm", "bmm_kernel", None)
    ok, reasons = _prec021(_build(observations))
    assert not ok and any("bmm" in r for r in reasons)


# -- validation and determinism ----------------------------------------------


def test_tampered_row_flagged():
    profile = _build([kp.KernelObservation("gemm", "gemm_kernel", None)])
    tampered = copy.deepcopy(profile)
    tampered["rows"]["gemm"]["resolved"] = True
    tampered["rows"]["gemm"]["accumulator"] = "float32"
    errors = kp.validate_kernel_profile(tampered)
    assert any("gemm" in e and "resolved" in e for e in errors)
    assert not _qualified(tampered)[0]


def test_tampered_evidence_kind_flagged():
    profile = _build()
    profile["rows"]["gemm"]["evidence"][0]["kind"] = "numerical_agreement"
    assert any("gemm" in e for e in kp.validate_kernel_profile(profile))


def test_deterministic_under_observation_permutation():
    observations = [
        *_valid_observations(),
        _obs("gemm", "gemm_b"),
        _obs("gemm", "gemm_a"),
    ]
    first = _build(observations, unattributed=["b", "a"])
    second = _build(list(reversed(observations)), unattributed=["a", "b", "a"])
    assert canonical_json(first) == canonical_json(second)
    assert manifest_hash(first) == manifest_hash(second)


def test_cpu_smoke_zero_observations():
    """On a CPU host no device kernels: well-formed, nothing executed, unqualified."""
    profile = _build(
        [],
        capture=_capture(
            purpose="smoke", qualification_packet=None, synthetic_weights=True
        ),
    )
    assert kp.validate_kernel_profile(profile) == []
    assert not any(row["executed"] for row in profile["rows"].values())
    assert not any(row["resolved"] for row in profile["rows"].values())
    ok, reasons = _qualified(profile)
    assert not ok
    assert any("purpose" in r for r in reasons)
    assert any("executed" in r and "expected" in r for r in reasons)


@pytest.mark.parametrize("symbols", [[], ["unknown_kernel"]])
def test_generator_inputs_equal_lists(symbols):
    """@spec PORT-PREC-030: single-pass inputs preserve all observations."""
    observations = _valid_observations()
    profile = _build((o for o in observations), unattributed=(s for s in symbols))
    assert profile == _build(observations, unattributed=symbols)
    assert _qualified(profile)[0] is (not symbols)


@pytest.mark.parametrize("component", ["torch", "cuda", "cudnn", "triton"])
@pytest.mark.parametrize("missing", [False, True])
def test_verdict_software_must_match_profile(component, missing):
    """@spec PORT-PREC-031, PORT-PREC-032: digest alone cannot bind the build."""
    fingerprint = _fingerprint()
    if missing:
        del fingerprint["software"]
    else:
        fingerprint["software"][component] = "another-build"
    profile = _build(fingerprint=fingerprint)
    ok, reasons = _qualified(profile, fingerprint)
    assert not ok and any("software" in r for r in reasons)


@pytest.mark.parametrize(
    "pin",
    [
        {"torch": "2.9."},
        {"torch": "2.9.0+cu128"},
        {"torch": "2.9.0", "unknown_component": "x"},
    ],
)
def test_pinned_backend_requires_exact_build(pin):
    """@spec PORT-PREC-031: every pin component must exist and match."""
    observations = _valid_observations()
    observations[0] = _obs("gemm", kind="pinned_backend", pinned_build=pin)
    profile = _build(observations)
    ok, reasons = _qualified(profile)
    assert not ok and any("pinned_build" in r for r in reasons)


@pytest.mark.parametrize("pin", [None, {}, "torch==2.9.0", ["torch"], {"torch": 2.9}])
def test_malformed_pinned_build_rejected_at_validation(pin):
    """@spec PORT-PREC-031: a pin is a non-empty str-to-str mapping."""
    observations = _valid_observations()
    observations[0] = _obs("gemm", kind="pinned_backend", pinned_build=pin)
    with pytest.raises(ValueError, match="pinned_build"):
        _build(observations)
    profile = _build()
    profile["rows"]["gemm"]["evidence"][0]["kind"] = "pinned_backend"
    profile["rows"]["gemm"]["evidence"][0]["pinned_build"] = pin
    assert any("pinned_build" in e for e in kp.validate_kernel_profile(profile))


def test_pinned_build_on_dispatch_evidence_is_a_validation_error():
    """@spec PORT-PREC-031: only pinned_backend evidence carries a pin."""
    observations = _valid_observations()
    observations[0] = _obs("gemm", kind="dispatch", pinned_build={"torch": "2.9.0"})
    with pytest.raises(ValueError, match="pinned_build"):
        _build(observations)


@pytest.mark.parametrize("present", ["torch", "cuda", "cudnn", "triton"])
def test_verdict_software_must_contain_every_component(present):
    """@spec PORT-PREC-032: a partial verdict software block cannot qualify."""
    fingerprint = _fingerprint()
    fingerprint["software"] = {present: fingerprint["software"][present]}
    profile = _build()
    ok, reasons = _qualified(profile, fingerprint)
    assert not ok and any("missing" in r and "software" in r for r in reasons)


def test_pinned_backend_preserves_matching_build():
    """@spec PORT-PREC-031: backend evidence records its exact pin."""
    observations = _valid_observations()
    observations[0] = _obs("gemm", kind="pinned_backend")
    profile = _build(observations)
    assert profile["rows"]["gemm"]["evidence"][0]["pinned_build"] == {"torch": "2.9.0"}
    assert _qualified(profile) == (True, [])


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("device", "name"),
        ("device", "compute_capability"),
        *(("software", k) for k in ("torch", "cuda", "cudnn", "triton")),
    ],
)
def test_required_identity_keys(section, key):
    """@spec PORT-PREC-030: identity mappings must carry the required keys."""
    profile = _build()
    del profile[section][key]
    assert any(f"{section}.{key}" in r for r in kp.validate_kernel_profile(profile))


@pytest.mark.parametrize("key", ["compiled", "graph_replay", "batch_invariant_mode"])
def test_capture_configuration_must_match_verdict(key):
    """@spec PORT-PREC-025, PORT-PREC-032: a matching digest is insufficient."""
    fingerprint = _fingerprint()
    overrides = {key: "off" if key == "batch_invariant_mode" else True}
    profile = _build(fingerprint=fingerprint, capture=_capture(**overrides))
    ok, reasons = _qualified(profile, fingerprint)
    assert not ok and any(key in r for r in reasons)


@pytest.mark.parametrize("key", ["compiled", "graph_replay"])
def test_missing_verdict_configuration_is_unqualified(key):
    """@spec PORT-PREC-032: qualification requires an explicit selection."""
    fingerprint = _fingerprint()
    del fingerprint["optimization_selection"][key]
    ok, reasons = _qualified(_build(fingerprint=fingerprint), fingerprint)
    assert not ok and any(key in r for r in reasons)


def test_missing_capture_mode_is_unqualified():
    """@spec PORT-PREC-025: capture must record the mode actually run."""
    profile = _build()
    del profile["capture"]["batch_invariant_mode"]
    ok, reasons = _qualified(profile)
    assert not ok and any("batch_invariant_mode" in r for r in reasons)


def test_mode_off_profile_rejected_for_mode_on_verdict():
    """@spec PORT-PREC-025: no mode-off artifact reuse for mode-on."""
    fingerprint = _fingerprint()
    del fingerprint["batch_invariant_mode"]
    profile = _build(
        fingerprint=fingerprint, capture=_capture(batch_invariant_mode="off")
    )
    assert _qualified(profile, fingerprint) == (True, [])
    assert not _qualified(profile, _fingerprint())[0]
    # Even relabelling the digest cannot turn a mode-off capture into mode-on.
    profile["fingerprint_digest"] = kp.execution_fingerprint_digest(_fingerprint())
    ok, reasons = _qualified(profile, _fingerprint())
    assert not ok and any("batch_invariant_mode" in r for r in reasons)


@pytest.mark.parametrize("empty", [True, False])
def test_prec021_requires_qualified_profile(empty):
    """@spec PORT-PREC-021, PORT-PREC-032: no vacuous FP32 evidence pass."""
    profile = _build([]) if empty else _build(unattributed=["mystery"])
    ok, reasons = _prec021(profile)
    assert not ok and any("expected" in r or "unattributed" in r for r in reasons)


def test_malformed_backend_evidence_is_unqualified():
    """@spec PORT-PREC-031, PORT-PREC-032: malformed evidence fails closed."""
    profile = _build()
    profile["rows"]["gemm"]["evidence"] = None
    ok, reasons = _qualified(profile)
    assert not ok and any("evidence" in r for r in reasons)
