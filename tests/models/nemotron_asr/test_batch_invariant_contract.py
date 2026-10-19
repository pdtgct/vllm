# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 5 contracts for PORT-PREC-018..028; no production implementation.

Missing imports occur inside tests, so absent Phase-6 seams fail individually.
Proposed seam: nemotron_asr.batch_invariance supplies an installed-state adapter
(initializer dependency injection witnesses successful return), an immutable
BatchInvariantExecution record, operator helpers and qualification_verdict.
Record methods stamp copies of receipts/fingerprints and namespace existing keys;
they must be wired into model construction and the existing capture owners.
GPU execution tests live in test_batch_invariant_cuda.py.
"""

from __future__ import annotations

import copy
import importlib
import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm.models.nemotron_asr.encoder_execution import ResolvedEncoderExecution
from vllm.models.nemotron_asr.manifests import canonical_json
from vllm.models.nemotron_asr.precision import FP32_BRINGUP

pytestmark = [pytest.mark.cpu_test, pytest.mark.core_model]


def test_cuda_qualification_compile_domain_guard():
    """@spec PORT-PREC-020, PORT-PREC-023: bounded tiers retain all geometries."""
    from test_batch_invariant_cuda import (
        CAPTURE_TIERS,
        GEOMETRIES,
        _qualification_cells,
    )

    from vllm.models.nemotron_asr.encoder_execution import _population_capture_tiers

    execution = SimpleNamespace(
        warmup_geometries=GEOMETRIES,
        warmup_populations=_population_capture_tiers(128),
    )
    assert _qualification_cells(execution) == tuple(
        (g, p) for g in GEOMETRIES for p in CAPTURE_TIERS
    )
    execution.warmup_populations = tuple(range(1, 129))
    with pytest.raises(AssertionError, match="compile domain exceeds 40 cells"):
        _qualification_cells(execution)
    execution.warmup_populations = CAPTURE_TIERS
    execution.warmup_geometries = GEOMETRIES[:-1]
    with pytest.raises(AssertionError):
        _qualification_cells(execution)


def api() -> Any:
    return importlib.import_module("vllm.models.nemotron_asr.batch_invariance")


def installed(enabled: bool = True) -> Any:
    module = SimpleNamespace(
        _batch_invariant_MODE=enabled,
        _batch_invariant_LIB=object() if enabled else None,
    )
    return api().installed_batch_invariant_mode(module=module, initialize=lambda: None)


@pytest.mark.parametrize("enabled", [False, True])
def test_prec018_installed_mode_snapshot(monkeypatch, enabled):
    """@spec PORT-PREC-018: initialization completes before reading installed state."""
    subject = api()
    module = SimpleNamespace(_batch_invariant_MODE=False, _batch_invariant_LIB=None)
    events = []

    def initialize():
        events.append("returned")
        module._batch_invariant_MODE = enabled
        module._batch_invariant_LIB = object() if enabled else None

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0" if enabled else "1")
    record = subject.installed_batch_invariant_mode(
        module=module, initialize=initialize
    )
    assert record.enabled is enabled
    assert events == ["returned"]
    module._batch_invariant_MODE = not enabled
    assert record.enabled is enabled  # snapshot, not a live proxy


@pytest.mark.parametrize(
    "broken",
    ["missing_flag", "missing_library", "null_library", "raised_after_library"],
)
def test_prec018_installed_mode_snapshot_incomplete(broken):
    """@spec PORT-PREC-018: a library object alone does not attest completion."""
    subject = api()
    module = SimpleNamespace(_batch_invariant_MODE=True, _batch_invariant_LIB=object())
    error = RuntimeError("registration failed after library creation")

    def initialize():
        if broken == "raised_after_library":
            raise error
        if broken == "missing_flag":
            del module._batch_invariant_MODE
        elif broken == "missing_library":
            del module._batch_invariant_LIB
        else:
            module._batch_invariant_LIB = None

    with pytest.raises(RuntimeError) as caught:
        subject.installed_batch_invariant_mode(module=module, initialize=initialize)
    if broken == "raised_after_library":
        assert caught.value is error


@pytest.mark.parametrize("enabled", [False, True])
def test_prec028_mode_immutability(monkeypatch, enabled):
    """@spec PORT-PREC-028: environment/config/session writes cannot change a record."""
    record = installed(enabled)
    before = record.fingerprint({"policy": FP32_BRINGUP.identifier})
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0" if enabled else "1")
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        record.enabled = not enabled
    # Unknown model/session overrides must never become a second mode selector.
    for overrides in (
        {"batch_invariant_mode": not enabled},
        {"VLLM_BATCH_INVARIANT": not enabled},
    ):
        assert (
            api().resolve_execution_mode(
                record, hf_overrides=overrides, session_controls=overrides
            )
            is record
        )
    assert record.fingerprint({"policy": FP32_BRINGUP.identifier}) == before


_READY_BYTES = (
    b'{"arm":"eager","ready":true,"warmup_cells":[],'
    b'"warmup_geometries":[],"warmup_populations":[]}'
)


def test_prec019_legacy_serialization():
    """@spec PORT-PREC-019: fixed observations retain baseline bytes and policy id."""
    execution = ResolvedEncoderExecution(arm="eager", transition=lambda *args: None)
    assert canonical_json(execution.ready_receipt()).encode() == _READY_BYTES
    assert FP32_BRINGUP.identifier == "pp-d479361445b4"
    assert (
        FP32_BRINGUP.content_hash
        == "sha256:d479361445b45b54ce1b0df4cd11b7ee1a06deb235ceeb8cf97c565bed2f5ccd"
    )
    assert "batch_invariant_mode" not in execution.ready_receipt()


def test_prec019_legacy_serialization_adapter():
    """@spec PORT-PREC-019:
    off omits the new field; it does not append False to keys.
    """
    record = installed(False)
    fixtures = [
        b'{"policy":"pp-d479361445b4","source":"fixed-source","version":1}',
        _READY_BYTES,
        b'{"bounds":[100,200],"capacity":2,"durations":[90,190],"execution_environment_key":"fixed-env"}',
    ]
    for raw, stamp in zip(
        fixtures,
        (record.fingerprint, record.readiness_receipt, record.service_receipt),
        strict=True,
    ):
        baseline = json.loads(raw)
        assert canonical_json(stamp(baseline)).encode() == raw
        assert canonical_json(baseline).encode() == raw
    for kind, key in (
        ("decode", (1, 31, "fp32")),
        ("decode_fn", (1, 31)),
        ("cuda", (1, 31)),
        ("chunk", (1, 31)),
    ):
        assert record.graph_key(kind, key) == key


@pytest.mark.parametrize("kind", ["decode", "decode_fn", "cuda", "chunk"])
def test_prec020_mode_identity_isolation(kind):
    """@spec PORT-PREC-020, PORT-PREC-025: cross-mode/cross-kernel artifacts never
    match."""
    off, on = installed(False), installed(True)
    base = {
        "policy": "fp32",
        "kernel": "operator-accumulation-v1",
        "optimization": "prefix-bank",
    }
    original = copy.deepcopy(base)
    old_key = (1, 31, "fp32") if kind == "decode" else (1, 31)
    assert off.graph_key(kind, old_key) == old_key
    assert on.graph_key(kind, old_key) != old_key
    cache = {off.graph_key(kind, old_key): "off", on.graph_key(kind, old_key): "on"}
    assert len(cache) == 2
    for stamp in (on.fingerprint, on.readiness_receipt, on.service_receipt):
        assert stamp(base) == {**base, "batch_invariant_mode": "on"}
    assert base == original
    fp = on.fingerprint(base)
    assert api().artifact_matches(fp, copy.deepcopy(fp))
    assert not api().artifact_matches(fp, off.fingerprint(base))
    for field in base:
        assert not api().artifact_matches(fp, {**fp, field: "different"})


# Table rows are obligations, NOT claims that a tensor dtype proves a kernel.
OPERATOR_ROWS = {
    "gemm": "vllm",
    "bmm": "vllm",
    "softmax": "port",
    "log_softmax": "port",
    "layer_norm": "port",
    "depthwise_conv1d": "port",
    "strided_conv2d": "port",
    "pointwise_conv": "port",
    "mean": "vllm",
    "sum": "port",
    "sdpa": "port",
}


def test_prec021_operator_accumulation_contract():
    """@spec PORT-PREC-021: pin all table rows and require actual-backend evidence."""
    contract = api().operator_accumulation_contract(device_capability=(8, 0))
    assert contract["version"] == "operator-accumulation-v1"
    assert set(contract["rows"]) == set(OPERATOR_ROWS)
    for name, owner in OPERATOR_ROWS.items():
        row = contract["rows"][name]
        assert row["owner"] == owner
        assert row["required_accumulator"] == "float32"
        assert row["fp16_accumulation"] == "unspecified"
    assert contract["rows"]["pointwise_conv"]["bias_order"] == "before_result_cast"
    assert contract["rows"]["softmax"]["reduction"] == "summation"
    assert api().operator_accumulation_contract(device_capability=(8, 6)) == contract
    other = api().operator_accumulation_contract(device_capability=(9, 0))
    assert not other["rows"]["gemm"]["accumulator_established"]


class DispatchTrace(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.calls.append((str(func), args, dict(kwargs or {})))
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("ndim", [1, 2])
@pytest.mark.parametrize("biased", [False, True])
def test_prec021_operator_accumulation_contract_pointwise(ndim, biased):
    """@spec PORT-PREC-021: k=1 lowering preserves layout and adds bias before cast."""
    lower = api().pointwise_conv_as_linear
    cls = torch.nn.Conv1d if ndim == 1 else torch.nn.Conv2d
    conv = cls(2, 3, 1, bias=biased).half()
    with torch.no_grad():
        conv.weight.fill_(1.0)
        if biased:
            conv.bias.fill_(-1.0)
    # 1 + 2^-11 rounds to 1 in fp16; subtracting 1 BEFORE that cast
    # gives 2^-11. Post-store linear bias addition instead gives zero.
    shape = (2, 2, 5) if ndim == 1 else (2, 2, 3, 5)
    x = torch.empty(shape, dtype=torch.float16)
    x[:, 0] = 1.0
    x[:, 1] = 2**-11
    trace = DispatchTrace()
    with trace:
        actual = lower(x, conv.weight, conv.bias)
    expected = x.float().movedim(1, -1) @ conv.weight.float().reshape(3, 2).T
    if biased:
        expected = expected + conv.bias.float()
    expected = expected.movedim(-1, 1).half()
    assert torch.equal(actual, expected)
    assert actual.shape == (shape[0], 3, *shape[2:])
    assert actual.dtype == x.dtype
    names = [name for name, _, _ in trace.calls]
    assert not any("convolution" in name for name in names)
    assert any(("addmm" if biased else "mm") in name for name in names), names
    if biased:
        post_cast = (
            x.movedim(1, -1) @ conv.weight.reshape(3, 2).T + conv.bias
        ).movedim(-1, 1)
        assert not torch.equal(actual, post_cast)


@pytest.mark.parametrize(
    "operation", ["softmax", "sum", "layer_norm", "depthwise_conv1d", "strided_conv2d"]
)
def test_prec021_operator_accumulation_contract_dispatch(operation):
    """@spec PORT-PREC-021: CPU-visible PORT paths must promote before reduction."""
    subject = api()
    x = torch.tensor(
        [[[1.0, 2**-11, -1.0, 0.25], [0.5, -0.5, 0.125, 1.0]]], dtype=torch.float16
    )
    trace = DispatchTrace()
    with trace:
        if operation == "softmax":
            result = subject.softmax_fp32_sum(x, dim=-1)
        elif operation == "sum":
            result = subject.sum_fp32(x, dim=-1)
        elif operation == "layer_norm":
            result = subject.layer_norm_fp32(x, (4,), eps=1e-5)
        elif operation == "depthwise_conv1d":
            result = subject.depthwise_conv1d_fp32(
                x, torch.ones(2, 1, 3, dtype=x.dtype), padding=1
            )
        else:
            result = subject.strided_conv2d_fp32(
                x.unsqueeze(1), torch.ones(2, 1, 2, 2, dtype=x.dtype), stride=2
            )
    assert result.dtype == x.dtype
    relevant = [
        call
        for call in trace.calls
        if any(s in call[0] for s in ("sum.", "native_layer_norm", "convolution"))
    ]
    assert relevant, trace.calls
    for name, args, kwargs in relevant:
        assert args[0].dtype == torch.float32 or kwargs.get("dtype") == torch.float32, (
            name
        )
    if operation == "sum":
        assert torch.equal(result, x.float().sum(-1).half())
    elif operation == "softmax":
        assert torch.isfinite(result).all()
        # Explicitly inspect masked and empty reductions as well.
        for special in (torch.full_like(x, -torch.inf), x[..., :0]):
            masked_trace = DispatchTrace()
            with masked_trace:
                special_result = subject.softmax_fp32_sum(special, dim=-1)
            assert special_result.shape == special.shape
            for name, args, kwargs in masked_trace.calls:
                if "sum." in name:
                    assert (
                        args[0].dtype == torch.float32
                        or kwargs.get("dtype") == torch.float32
                    )


def _fingerprint(**changes):
    result = {
        "gpu": "NVIDIA A100",
        "precision": "fp32",
        "software": "98dff2a81+292e48605",
        "kernel": "operator-accumulation-v1",
        "backend": "triton",
        "batch_invariant_mode": "on",
        "optimizations": {
            "prefix_bank": True,
            "inplace_cache": True,
            "head_major": False,
            "sdpa": False,
        },
    }
    return {**result, **changes}


def _evidence(fp):
    return {
        "fingerprint": copy.deepcopy(fp),
        "executable": True,
        "populations": [1, 2, 31, 63, 64, 128],
        "geometries": [0, 1, 2, 3, 4],
        "compositions": ["repeated", "mixed"],
        "independent_b128_runs": 2,
        "gates": {f"PORT-PREC-{n:03}": True for n in (14, 16, 20, 21, 22, 24, 26, 27)},
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("gpu", "NVIDIA A40"),
        ("precision", "fp16-encoder"),
        ("software", "other"),
        ("kernel", "other"),
        ("backend", "other"),
        ("batch_invariant_mode", "off"),
        (
            "optimizations",
            {
                "prefix_bank": True,
                "inplace_cache": True,
                "head_major": False,
                "sdpa": True,
            },
        ),
    ],
)
def test_prec023_qualification_identity(field, value):
    """@spec PORT-PREC-023, PORT-PREC-025:
    evidence binds the full execution identity.
    """
    qualify = api().qualification_verdict
    fp = _fingerprint()
    evidence = _evidence(fp)
    assert qualify(
        fp, evidence, served_geometries=(0, 1, 2, 3, 4), deployment_cap=20
    ).qualified
    assert not qualify(
        {**fp, field: value},
        evidence,
        served_geometries=(0, 1, 2, 3, 4),
        deployment_cap=20,
    ).qualified


@pytest.mark.parametrize(
    "missing",
    [
        "128",
        "geometry",
        "reset",
        "mixed",
        "014",
        "016",
        "020",
        "021",
        "024",
        "026",
        "027",
        "transcript",
    ],
)
def test_prec023_qualification_failure_isolation(missing):
    """@spec PORT-PREC-023, PORT-PREC-024: functional and numerical misses are local,
    not startup errors."""
    qualify = api().qualification_verdict
    fp = _fingerprint()
    good = _evidence(fp)
    bad = copy.deepcopy(good)
    if missing == "transcript":
        from batch_invariant_packet import validate_transcript_identity

        packet = _transcript_packet()
        packet["cells"][0]["on"] = "Different final transcript."
        with pytest.raises(AssertionError, match="transcript mismatch"):
            validate_transcript_identity(packet)
        # The verdict surface consumes verified gate attestations, not packets.
        bad["gates"]["PORT-PREC-024"] = False
    elif missing == "128":
        bad["populations"].remove(128)
    elif missing == "geometry":
        bad["geometries"].pop()
    elif missing == "reset":
        bad["independent_b128_runs"] = 1
    elif missing == "mixed":
        bad["compositions"].remove("mixed")
    else:
        bad["gates"][f"PORT-PREC-{missing}"] = False
    result = qualify(fp, bad, served_geometries=(0, 1, 2, 3, 4), deployment_cap=20)
    assert (
        result.ready
        and not result.qualified
        and not result.default
        and not result.supported
    )
    assert qualify(
        fp, good, served_geometries=(0, 1, 2, 3, 4), deployment_cap=20
    ).qualified
    independent = _fingerprint(precision="fp16-encoder")
    assert qualify(
        independent,
        _evidence(independent),
        served_geometries=(0, 1, 2, 3, 4),
        deployment_cap=20,
    ).qualified
    # Unspecified hardware is permitted to execute, but receives no inherited
    # qualification.
    unknown = _fingerprint(gpu="NVIDIA H100")
    result = qualify(
        unknown,
        _evidence(unknown),
        served_geometries=(0, 1, 2, 3, 4),
        deployment_cap=20,
    )
    assert result.ready and not result.qualified


@pytest.mark.parametrize("disclosure", ["missing", "false"])
def test_prec022_disclosure_does_not_gate_verdict_or_readiness(disclosure):
    """@spec PORT-PREC-022, PORT-PREC-023: disclosure gates publication only."""
    fp = _fingerprint()
    evidence = _evidence(fp)
    if disclosure == "missing":
        del evidence["gates"]["PORT-PREC-022"]
    else:
        evidence["gates"]["PORT-PREC-022"] = False
    result = api().qualification_verdict(
        fp, evidence, served_geometries=(0, 1, 2, 3, 4), deployment_cap=20
    )
    assert (
        result.ready
        and result.qualified
        and not result.default
        and not result.supported
    )


@pytest.mark.parametrize("gate", [14, 16, 20, 21, 24, 26, 27])
@pytest.mark.parametrize("attestation", ["missing", "false"])
def test_prec023_each_qualification_gate_is_required(gate, attestation):
    """@spec PORT-PREC-023: each required gate independently controls qualification."""
    fp = _fingerprint()
    evidence = _evidence(fp)
    name = f"PORT-PREC-{gate:03}"
    if attestation == "missing":
        del evidence["gates"][name]
    else:
        evidence["gates"][name] = False
    result = api().qualification_verdict(
        fp, evidence, served_geometries=(0, 1, 2, 3, 4), deployment_cap=20
    )
    assert (
        result.ready
        and not result.qualified
        and not result.default
        and not result.supported
    )


@pytest.mark.parametrize("failure", ["compile", "capture", "replay"])
def test_mode_on_capture_failure_closes_readiness(failure):
    """@spec PORT-PREC-023, PORT-PREC-026:
    execution failure is not a numerical waiver.
    """
    fp = _fingerprint()
    evidence = _evidence(fp)
    evidence.update(executable=False, execution_failure=failure)
    result = api().qualification_verdict(
        fp, evidence, served_geometries=(0, 1, 2, 3, 4), deployment_cap=20
    )
    assert not result.ready and not result.qualified


@pytest.mark.parametrize(
    "option", ["prefix_bank", "inplace_cache", "head_major", "sdpa"]
)
def test_prec027_optimization_equivalence(option):
    """@spec PORT-PREC-027: mode binding preserves selections and gates independent
    evidence."""
    subject = api()
    for enabled in (False, True):
        selected = dict(_fingerprint()["optimizations"], **{option: enabled})
        assert subject.resolve_optimizations(installed(True), selected) == selected
    fp = _fingerprint(optimizations={**_fingerprint()["optimizations"], option: True})
    evidence = _evidence(fp)
    evidence["gates"]["PORT-PREC-027"] = False
    assert not subject.qualification_verdict(
        fp, evidence, served_geometries=(0, 1, 2, 3, 4), deployment_cap=128
    ).qualified


@pytest.mark.parametrize("fails", [False, True])
def test_prec018_installed_mode_snapshot_construction(monkeypatch, fails):
    """@spec PORT-PREC-018, PORT-PREC-028: construction resolves before model work."""
    from vllm.models.nemotron_asr import nemotron_asr as model

    record = installed(True)
    events = []
    error = RuntimeError("incomplete installation")

    def adapter():
        events.append("mode")
        if fails:
            raise error
        return record

    def component(**kwargs):
        assert events == ["mode"], "model work preceded installation witness"
        return torch.nn.Identity()

    monkeypatch.setattr(model, "installed_batch_invariant_mode", adapter)
    for name in (
        "MelFeaturizer",
        "FastConformerEncoder",
        "PromptConditioner",
        "Predictor",
        "Joint",
    ):
        monkeypatch.setattr(model, name, component)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    kwargs = dict(vocab_size=4, filterbank=torch.ones(1), window=torch.ones(1))
    if fails:
        with pytest.raises(RuntimeError) as caught:
            model.NemotronASRCore(**kwargs)
        assert caught.value is error and events == ["mode"]
    else:
        core = model.NemotronASRCore(**kwargs)
        assert core.batch_invariant_mode is record
        assert events == ["mode"]
        with pytest.raises((AttributeError, RuntimeError, TypeError)):
            api().bind_batch_invariant_mode(core, installed(False))
        assert core.batch_invariant_mode is record


@pytest.mark.parametrize("operator", list(OPERATOR_ROWS))
@pytest.mark.parametrize("execution", ["eager", "compiled"])
def test_prec021_operator_accumulation_contract_evidence(operator, execution):
    """@spec PORT-PREC-021: numerical equality, dtype and a wrong backend are
    insufficient."""
    check = api().accumulation_evidence_complete
    evidence = {
        "operator": operator,
        "backend": "actual-backend",
        "execution": execution,
        "accumulator": "float32",
        "kind": "generated_kernel" if execution == "compiled" else "pinned_backend",
        "source_sha256": "a" * 64,
        "symbol": "actual_kernel.sum",
        "bias_before_cast": True,
    }
    assert check(
        operator, backend="actual-backend", execution=execution, evidence=evidence
    )
    for patch in (
        {"kind": "numerical_equality"},
        {"kind": "operand_dtype"},
        {"backend": "other"},
        {"source_sha256": ""},
        {"accumulator": "float16"},
    ):
        assert not check(
            operator,
            backend="actual-backend",
            execution=execution,
            evidence={**evidence, **patch},
        )
    if execution == "compiled":
        assert not check(
            operator,
            backend="actual-backend",
            execution=execution,
            evidence={**evidence, "execution": "eager", "kind": "dispatch"},
        )
    if operator == "pointwise_conv":
        assert not check(
            operator,
            backend="actual-backend",
            execution=execution,
            evidence={**evidence, "bias_before_cast": False},
        )


def test_prec020_bit_patterns_include_signed_zero_and_nan_payloads():
    """@spec PORT-PREC-020: discriminator compares storage, not IEEE value equality."""
    from test_batch_invariant_cuda import _bits

    # qNaNs with distinct payloads, positive/negative zero, positive/negative infinity.
    values = torch.tensor(
        [0, -2147483648, 2143289345, 2143289346, 2139095040, -8388608],
        dtype=torch.int32,
    ).view(torch.float32)
    assert torch.equal(_bits(values), _bits(values.clone()))
    assert not torch.equal(_bits(values[0:1]), _bits(values[1:2]))
    assert not torch.equal(_bits(values[2:3]), _bits(values[3:4]))
    assert not torch.equal(_bits(values[4:5]), _bits(values[5:6]))


@pytest.mark.parametrize("biased", [False, True])
def test_prec021_compiled_pointwise_keeps_dispatch(monkeypatch, biased):
    """@spec PORT-PREC-021: compilation must retain the installed GEMM boundary."""
    subject = api()
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    x = torch.randn(2, 3, 5)
    weight = torch.randn(4, 3, 1)
    bias = torch.randn(4) if biased else None
    compiled = torch.compile(
        subject.pointwise_conv_as_linear, backend=backend, fullgraph=True
    )
    with torch.inference_mode():
        expected = subject.pointwise_conv_as_linear(x, weight, bias)
        actual = compiled(x, weight, bias)
    assert torch.equal(actual, expected)
    assert any(
        "nemotron_bi" in str(node.target)
        for graph in graphs
        for node in graph.graph.nodes
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_prec018_probe_binds_engine_record(monkeypatch, enabled, explicit):
    """@spec PORT-PREC-018: standalone probes use the engine's module routing."""
    from tests.models.nemotron_asr import test_encoder_batch_invariance as probe

    record = installed(enabled)
    monkeypatch.setattr(
        probe, "_EXECUTION_MODE", api().MODE_OFF if explicit else record
    )
    core = probe.build_core(
        "fp32", torch.device("cpu"), probe._TINY, mode=record if explicit else None
    )
    assert api().execution_mode(core) is record
    assert all(
        api().execution_mode(module) is record for module in core.encoder.modules()
    )


def test_prec028_test_installation_restored_after_failure():
    """@spec PORT-PREC-028: test teardown cannot contaminate the next runtime."""
    import os

    from tests.models.nemotron_asr.conftest import preserve_batch_invariance

    class Library:
        destroyed = False

        def _destroy(self):
            self.destroyed = True

    library = Library()
    module = SimpleNamespace(
        _batch_invariant_MODE=False,
        _batch_invariant_LIB=None,
        _fp16_block_size_n=256,
        _fp32_block_size_n=128,
        _fp32_num_stages=3,
    )
    configs = SimpleNamespace(
        _TUNED_MATMUL_CONFIGS_FOR_DEVICE=None, _TUNED_MATMUL_CONFIGS_RESOLVED=False
    )
    before_env = os.environ.get("VLLM_BATCH_INVARIANT")
    before_bmm = torch.bmm
    with (
        pytest.raises(RuntimeError, match="partial installation"),
        preserve_batch_invariance(module, configs),
    ):
        module._batch_invariant_MODE = True
        module._batch_invariant_LIB = library
        configs._TUNED_MATMUL_CONFIGS_RESOLVED = True
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        torch.bmm = lambda *args: None
        raise RuntimeError("partial installation")
    assert library.destroyed
    assert module._batch_invariant_MODE is False
    assert module._batch_invariant_LIB is None
    assert configs._TUNED_MATMUL_CONFIGS_RESOLVED is False
    assert os.environ.get("VLLM_BATCH_INVARIANT") == before_env
    assert torch.bmm is before_bmm


def test_prec028_core_initializer_side_effects_restored(monkeypatch):
    """@spec PORT-PREC-028: core's compiler/NCCL pins cannot leak between tests."""
    import os

    from torch._inductor import config as inductor_config

    from tests.models.nemotron_asr.conftest import preserve_batch_invariance
    from vllm.model_executor.determinism import batch_invariant, batch_invariant_configs
    from vllm.utils import nccl

    monkeypatch.setattr(nccl, "_nccl_has_no_cache", lambda: True)
    monkeypatch.setattr(nccl, "_unpinned_env", None)
    monkeypatch.setattr(inductor_config, "batch_invariant", False)
    env_keys = (
        "VLLM_CUSTOM_ALLREDUCE_ALGO",
        "VLLM_ROCM_USE_SKINNY_GEMM",
        "NCCL_NO_CACHE",
    )
    before_env = {key: os.environ.get(key) for key in env_keys}
    before_sizes = batch_invariant._BLOCK_SIZES_RESOLVED
    before_sms = batch_invariant._NUM_SMS
    with (
        pytest.raises(RuntimeError, match="partial initialization"),
        preserve_batch_invariance(batch_invariant, batch_invariant_configs),
    ):
        batch_invariant.override_envs_for_invariance()
        assert inductor_config.batch_invariant
        batch_invariant._BLOCK_SIZES_RESOLVED = not before_sizes
        batch_invariant._NUM_SMS = before_sms + 1
        raise RuntimeError("partial initialization")
    assert not inductor_config.batch_invariant
    assert nccl._unpinned_env is None
    assert before_sizes == batch_invariant._BLOCK_SIZES_RESOLVED
    assert before_sms == batch_invariant._NUM_SMS
    assert {key: os.environ.get(key) for key in env_keys} == before_env


def _packet_arms():
    from batch_invariant_packet import fingerprint_sha256

    identity = {
        "gpu": {"model": "NVIDIA A100", "partition": "GPU-test"},
        "checkpoint": "sha256:checkpoint",
        "software": {"vllm": "test-build", "engine": "test-build"},
        "precision": "fp32",
        "precision_policy": "pp-d479361445b4",
        "optimizations": {"prefix_bank": True, "sdpa": False},
        "audio_sha256": {
            f"dev-clean/1272/135031/1272-135031-{i:04}": "a" * 64 for i in range(4)
        },
        "cadence": [80, 160, 320, 560, 1120],
        "population": 1,
        "concurrency": 1,
        "language_prompt": "en",
        "decoding": {"strategy": "greedy", "deterministic": True},
        "postprocessing": "verbatim",
        "execution_path": "optimized-graph-replay",
    }
    arms = {}
    for mode in ("off", "on"):
        matched = copy.deepcopy(identity)
        if mode == "on":
            matched["batch_invariant_mode"] = "on"
        fp = copy.deepcopy(matched)
        arms[mode] = {
            "identity": matched,
            "fingerprint": fp,
            "receipts": {
                kind: {
                    "fingerprint_sha256": fingerprint_sha256(fp),
                    "receipt": {"id": f"{mode}-{kind}"},
                }
                for kind in ("readiness", "service")
            },
        }
    return arms


def _transcript_packet():
    from batch_invariant_packet import CELLS, fingerprint_sha256

    arms = _packet_arms()
    return {
        "schema": "nemotron-bi-transcript-identity-v1",
        "arms": arms,
        "mode_off_nemo_parity": {
            "fingerprint_sha256": fingerprint_sha256(arms["off"]["fingerprint"]),
            "passed": True,
            "unchanged": True,
            "oracle": "NeMo",
            "golden_revision": "immutable-golden",
            "golden_sha256": "a" * 64,
            "receipt": {"id": "unchanged-mode-off-parity"},
        },
        "cells": [
            {"clip_id": clip, "cadence_ms": cadence, "off": "Final. ", "on": "Final. "}
            for clip, cadence in sorted(CELLS)
        ],
        "differing_transcripts": 0,
    }


def _cost_packet():
    arms = _packet_arms()
    run = {
        "purpose": "PORT-PREC-022",
        "profiled": False,
        "discarded": False,
        "encoder_replay_ns": [2_000_000, 4_000_000],
        "chunks": [{"ready_ns": 10, "park_ns": 3_000_010, "audio_seconds": 0.16}],
        "window_start_ns": 0,
        "window_end_ns": 1_000_000_000,
    }
    for mode, arm in arms.items():
        run_for_arm = {
            **run,
            "fingerprint_sha256": arm["receipts"]["service"]["fingerprint_sha256"],
        }
        arm["warmup"] = [{**copy.deepcopy(run_for_arm), "discarded": True}]
        arm["reported"] = {"replay_ms": 3.0, "service_ms": 3.0, "throughput": 0.16}
    return {
        "schema": "nemotron-bi-cost-v1",
        "purpose": "PORT-PREC-022",
        "side_effects": ["aten::mm"],
        "side_effects_source_sha256": "a" * 64,
        "threshold": None,
        "default_changed": False,
        "supported_changed": False,
        "arms": arms,
        "pairs": [
            {
                "order": ["off", "on"] if i % 2 == 0 else ["on", "off"],
                **{
                    mode: {
                        **copy.deepcopy(run),
                        "fingerprint_sha256": arm["receipts"]["service"][
                            "fingerprint_sha256"
                        ],
                    }
                    for mode, arm in arms.items()
                },
            }
            for i in range(4)
        ],
    }


def test_prec024_packet_accepts_exact_finals():
    """@spec PORT-PREC-024: supplied packet validates on CPU without installing mode."""
    from batch_invariant_packet import validate_transcript_identity

    validate_transcript_identity(_transcript_packet())


@pytest.mark.parametrize(
    "fault",
    [
        "final",
        "unicode",
        "missing_cell",
        "duplicate",
        "extra_cell",
        "count",
        "identity",
        "fingerprint",
        "receipt",
        "parity_binding",
        "parity_fail",
        "parity_changed",
        "fp16",
        "decoding",
        "audio",
    ],
)
def test_prec024_packet_rejects_invalid(fault):
    """@spec PORT-PREC-024: no normalization, WER tolerance or incomplete evidence."""
    from batch_invariant_packet import validate_transcript_identity

    packet = _transcript_packet()
    if fault == "final":
        packet["cells"][0]["on"] = "Final."
    elif fault == "unicode":
        packet["cells"][0].update(off="é", on="e\u0301")
    elif fault == "missing_cell":
        packet["cells"].pop()
    elif fault == "duplicate":
        packet["cells"][-1] = packet["cells"][0]
    elif fault == "extra_cell":
        packet["cells"].append({**packet["cells"][0], "cadence_ms": 42})
    elif fault == "count":
        packet["differing_transcripts"] = 1
    elif fault == "identity":
        packet["arms"]["on"]["identity"]["concurrency"] = 2
    elif fault == "fingerprint":
        packet["arms"]["on"]["fingerprint"]["software"] = "different"
    elif fault == "receipt":
        packet["arms"]["on"]["receipts"]["service"] = {}
    elif fault == "parity_binding":
        packet["mode_off_nemo_parity"]["fingerprint_sha256"] = "b" * 64
    elif fault == "parity_fail":
        packet["mode_off_nemo_parity"]["passed"] = False
    elif fault == "parity_changed":
        packet["mode_off_nemo_parity"]["unchanged"] = False
    elif fault in ("fp16", "decoding", "audio"):
        key, value = {
            "fp16": ("precision", "fp16"),
            "decoding": ("decoding", {"strategy": "beam"}),
            "audio": ("audio_sha256", {}),
        }[fault]
        for arm in packet["arms"].values():
            arm["identity"][key] = value
    with pytest.raises(AssertionError):
        validate_transcript_identity(packet)


def test_prec022_packet_accepts_cost_without_threshold():
    """@spec PORT-PREC-022:
    recomputed medians, even a slower arm, permit publication.
    """
    from batch_invariant_packet import validate_cost_disclosure

    packet = _cost_packet()
    for pair in packet["pairs"]:
        pair["on"]["encoder_replay_ns"] = [200_000_000]
    packet["arms"]["on"]["reported"]["replay_ms"] = 200
    validate_cost_disclosure(packet, {"aten::mm"}, "a" * 64)


@pytest.mark.parametrize(
    "fault",
    [
        "profiled",
        "exploratory",
        "prec029",
        "warmup",
        "repetitions",
        "order",
        "inventory",
        "source",
        "receipt",
        "run_binding",
        "median",
        "negative",
        "nan",
        "clock",
        "threshold",
        "identity",
    ],
)
def test_prec022_packet_rejects_invalid_publication_only(fault):
    """@spec PORT-PREC-022: invalid disclosure blocks publication only, never
    readiness/verdict."""
    from batch_invariant_packet import validate_cost_disclosure

    packet = _cost_packet()
    run = packet["pairs"][0]["on"]
    if fault == "profiled":
        run["profiled"] = True
    elif fault in ("exploratory", "prec029"):
        run["purpose"] = "exploratory" if fault == "exploratory" else "PORT-PREC-029"
    elif fault == "warmup":
        packet["arms"]["on"]["warmup"] = []
    elif fault == "repetitions":
        packet["pairs"].pop()
    elif fault == "order":
        packet["pairs"][1]["order"] = ["off", "on"]
    elif fault == "inventory":
        packet["side_effects"].append("unknown")
    elif fault == "source":
        packet["side_effects_source_sha256"] = "b" * 64
    elif fault == "receipt":
        packet["arms"]["off"]["receipts"] = {}
    elif fault == "run_binding":
        run["fingerprint_sha256"] = "b" * 64
    elif fault == "median":
        packet["arms"]["on"]["reported"]["service_ms"] = 100
    elif fault == "negative":
        run["encoder_replay_ns"] = [-1]
    elif fault == "nan":
        run["encoder_replay_ns"] = [float("nan")]
    elif fault == "clock":
        run["window_end_ns"] = 1
    elif fault == "threshold":
        packet["threshold"] = 1.05
    elif fault == "identity":
        packet["arms"]["on"]["identity"]["software"] = "different"
    before = copy.deepcopy(packet)
    with pytest.raises(AssertionError, match="blocks publication only"):
        validate_cost_disclosure(packet, {"aten::mm"}, "a" * 64)
    # Validator has no engine-state or verdict writes.
    assert repr(packet) == repr(before)


def test_prec022_inventory_follows_registration_source():
    """@spec PORT-PREC-022:
    symbol discovery follows helpers, not version line numbers.
    """
    from batch_invariant_packet import inventory_from_source

    source = """
def enable():
    lib.impl("aten::mm", kernel, key)
    torch.bmm = kernel
    torch.backends.cuda.preferred_blas_library(backend="cublaslt")
    os.environ["NEW_BACKEND_SETTING"] = "1"
def init_batch_invariance():
    enable()
    torch.backends.cuda.matmul.fp32_precision = "ieee"
"""
    assert inventory_from_source(source) == {
        "aten::mm",
        "torch.bmm",
        "torch.backends.cuda.preferred_blas_library",
        "NEW_BACKEND_SETTING",
        "torch.backends.cuda.matmul.fp32_precision",
    }
    assert "aten::new" in inventory_from_source(source.replace("aten::mm", "aten::new"))


@pytest.mark.parametrize(
    "field",
    [
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
    ],
)
def test_prec024_packet_requires_each_matched_control(field):
    """@spec PORT-PREC-024: agreement cannot hide a missing experimental control."""
    from batch_invariant_packet import validate_transcript_identity

    packet = _transcript_packet()
    for arm in packet["arms"].values():
        del arm["identity"][field]
    with pytest.raises(AssertionError, match="required"):
        validate_transcript_identity(packet)


@pytest.mark.parametrize("arm", ["off", "on"])
@pytest.mark.parametrize("kind", ["readiness", "service"])
def test_packet_receipts_bind_their_own_fingerprint(arm, kind):
    """@spec PORT-PREC-022, PORT-PREC-024:
    a present but unrelated receipt is invalid.
    """
    from batch_invariant_packet import (
        validate_cost_disclosure,
        validate_transcript_identity,
    )

    for packet, check in [
        (_transcript_packet(), validate_transcript_identity),
        (_cost_packet(), lambda p: validate_cost_disclosure(p, {"aten::mm"}, "a" * 64)),
    ]:
        packet["arms"][arm]["receipts"][kind]["fingerprint_sha256"] = "f" * 64
        with pytest.raises(AssertionError, match="receipt fingerprint binding"):
            check(packet)


def test_prec022_inventory_contains_port_adapters():
    """@spec PORT-PREC-022: pinned test inventory stays aligned with engine adapters."""
    from batch_invariant_packet import PORT_SIDE_EFFECTS

    assert set(api().SCHEMA_ADAPTER_SIDE_EFFECTS) == PORT_SIDE_EFFECTS


def test_packet_requires_gpu_partition_not_just_model():
    """@spec PORT-PREC-022, PORT-PREC-024: model name alone is not matched hardware."""
    from batch_invariant_packet import (
        validate_cost_disclosure,
        validate_transcript_identity,
    )

    for packet, check in [
        (_transcript_packet(), validate_transcript_identity),
        (_cost_packet(), lambda p: validate_cost_disclosure(p, {"aten::mm"}, "a" * 64)),
    ]:
        for arm in packet["arms"].values():
            arm["identity"]["gpu"] = {"model": "NVIDIA A100"}
        with pytest.raises(AssertionError, match="GPU partition"):
            check(packet)
