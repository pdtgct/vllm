# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM batch-invariant installation witness and Nemotron accumulation contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

import vllm.envs as envs
from vllm.models.nemotron_asr import bi_ops  # noqa: F401 -- register opaque operators


@dataclass(frozen=True)
class BatchInvariantExecution:
    """@spec PORT-PREC-019, PORT-PREC-025, PORT-PREC-028: frozen identity axis."""

    enabled: bool
    schema_adapters: tuple[str, ...] = ()

    def fingerprint(self, baseline: Mapping[str, Any]) -> dict[str, Any]:
        return {**baseline, **({"batch_invariant_mode": "on"} if self.enabled else {})}

    readiness_receipt = fingerprint
    service_receipt = fingerprint

    def graph_key(self, kind: str, key: tuple[Any, ...]) -> tuple[Any, ...]:
        return ("batch_invariant_mode=on", kind, key) if self.enabled else key


MODE_OFF = BatchInvariantExecution(False)


def installed_batch_invariant_mode(
    *, module: Any = None, initialize: Callable[[], None] | None = None
) -> BatchInvariantExecution:
    """@spec PORT-PREC-018: require successful initialization, not just a library.

    vLLM's worker initializer owns installation, including the schema
    adapters; the model only snapshots its completed, sticky state. A request
    without that state fails closed instead of serving with the mode off.
    Explicit dependencies let probes run the initializer themselves;
    opaque test witnesses only snapshot the installed flag.
    """
    if module is not None or initialize is not None:
        if module is None or initialize is None:
            raise ValueError("module and initialize must be supplied together")
        initialize()
        return _snapshot(module)
    from vllm.model_executor.determinism import batch_invariant

    record = _snapshot(batch_invariant)
    if envs.VLLM_BATCH_INVARIANT and not record.enabled:
        raise RuntimeError(
            "VLLM_BATCH_INVARIANT is set but vLLM did not initialize batch invariance"
        )
    return record


def _snapshot(module: Any) -> BatchInvariantExecution:
    # Read under vLLM's initializer lock so an install in progress is awaited
    # rather than observed half-done; opaque probes have no lock.
    lock = getattr(module, "_init_lock", None)
    with lock if lock is not None else nullcontext():
        return _read_installed(module)


def _read_installed(module: Any) -> BatchInvariantExecution:
    failure = getattr(module, "_init_failure", None)
    if failure is not None:
        raise failure
    flag = getattr(module, "_batch_invariant_MODE", None)
    if type(flag) is not bool or not hasattr(module, "_batch_invariant_LIB"):
        raise RuntimeError("vLLM batch-invariant installed state is absent")
    if flag and module._batch_invariant_LIB is None:
        raise RuntimeError("vLLM batch-invariant installation is incomplete")
    # Dependency-injected record-only probes may supply an opaque witness.
    # A real Library (including CUDA probes) must have its adapters installed.
    if flag and isinstance(module._batch_invariant_LIB, torch.library.Library):
        from vllm.model_executor.determinism import schema_adapters

        return BatchInvariantExecution(
            flag, schema_adapters.installed_schema_adapters(module)
        )
    return BatchInvariantExecution(flag)


# @spec PORT-PREC-022: additional process-wide effects for cost disclosure.
# Existing vLLM entries remain required; these describe the schema adapters
# vLLM installs (vllm/model_executor/determinism/schema_adapters.py).
SCHEMA_ADAPTER_SIDE_EFFECTS = (
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
)


def execution_mode(owner: Any) -> BatchInvariantExecution:
    return getattr(owner, "batch_invariant_mode", MODE_OFF)


def batch_invariant_enabled(owner: Any) -> bool:
    """Use the construction-bound flag in traced code, retaining unbound probes."""
    enabled = getattr(owner, "_batch_invariant_enabled", None)
    return execution_mode(owner).enabled if enabled is None else enabled


def bind_batch_invariant_mode(core: Any, record: BatchInvariantExecution) -> None:
    """@spec PORT-PREC-018, PORT-PREC-028: bind once, before any model work."""
    owners = [core]
    encoder = getattr(core, "encoder", None)
    if encoder is not None:
        owners.extend(encoder.modules())
    lid = getattr(core, "lid", None)
    if lid is not None:
        owners.extend(lid.modules())
    for owner in owners:
        previous = getattr(owner, "batch_invariant_mode", None)
        if previous is not None and previous != record:
            raise RuntimeError("batch-invariant mode cannot change on a loaded model")
    for owner in owners:
        owner._batch_invariant_enabled = record.enabled
        if isinstance(getattr(type(owner), "batch_invariant_mode", None), property):
            owner._batch_invariant_mode = record
        else:
            owner.batch_invariant_mode = record


def resolve_execution_mode(
    record: BatchInvariantExecution,
    *,
    hf_overrides: Any = None,
    session_controls: Any = None,
) -> BatchInvariantExecution:
    """@spec PORT-PREC-028: configuration is not an installed-mode selector."""
    return record


def resolve_optimizations(
    record: BatchInvariantExecution, selected: Mapping[str, Any]
) -> dict[str, Any]:
    """@spec PORT-PREC-027: mode selection does not change optimizations."""
    return dict(selected)


def artifact_matches(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    """@spec PORT-PREC-025: match the entire execution/kernel identity."""
    return dict(expected) == dict(actual)


def operator_accumulation_contract(
    *, device_capability: tuple[int, int]
) -> dict[str, Any]:
    """@spec PORT-PREC-021: obligations, separate from actual-backend evidence."""
    vllm_rows = {"gemm", "bmm", "mean"}
    names = (
        *sorted(vllm_rows),
        "softmax",
        "log_softmax",
        "layer_norm",
        "depthwise_conv1d",
        "strided_conv2d",
        "pointwise_conv",
        "sum",
        "sdpa",
    )
    rows = {
        name: {
            "owner": "vllm" if name in vllm_rows else "port",
            "required_accumulator": "float32",
            "fp16_accumulation": "unspecified",
            "accumulator_established": name in {"bmm", "mean"}
            or (name == "gemm" and device_capability[0] == 8),
        }
        for name in names
    }
    rows["pointwise_conv"]["bias_order"] = "before_result_cast"
    rows["softmax"]["reduction"] = "summation"
    return {"version": "operator-accumulation-v1", "rows": rows}


def accumulation_evidence_complete(
    operator: str, *, backend: str, execution: str, evidence: Mapping[str, Any]
) -> bool:
    """@spec PORT-PREC-021: a value comparison is not accumulator evidence."""
    digest = evidence.get("source_sha256", "")
    return (
        evidence.get("operator") == operator
        and evidence.get("backend") == backend
        and evidence.get("execution") == execution
        and evidence.get("accumulator") == "float32"
        and evidence.get("kind")
        in (
            {"generated_kernel"}
            if execution == "compiled"
            else {"dispatch", "pinned_backend"}
        )
        and isinstance(digest, str)
        and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest)
        and bool(evidence.get("symbol"))
        and (operator != "pointwise_conv" or evidence.get("bias_before_cast") is True)
    )


def invariant_gemm(
    operand: torch.Tensor, matrix: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    """@spec PORT-PREC-021: opaque pinned GEMM, bias before result cast."""
    return (
        torch.ops.nemotron_bi.mm.default(operand, matrix)
        if bias is None
        else torch.ops.nemotron_bi.addmm.default(bias, operand, matrix)
    )


def invariant_linear(module: torch.nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Route only the installed mode; preserve the original mode-off call."""
    if not batch_invariant_enabled(module):
        return module(x)
    result = invariant_gemm(x.reshape(-1, x.shape[-1]), module.weight.t(), module.bias)
    return result.reshape(*x.shape[:-1], module.weight.shape[0])


def invariant_matmul(
    owner: torch.nn.Module, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    return (
        torch.ops.nemotron_bi.matmul.default(a, b)
        if batch_invariant_enabled(owner)
        else torch.matmul(a, b)
    )


def pointwise_conv_as_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """@spec PORT-PREC-021: k=1 GEMM with bias inside the accumulator."""
    if weight.ndim != x.ndim or any(size != 1 for size in weight.shape[2:]):
        raise ValueError("pointwise lowering requires a k=1 Conv1d or Conv2d weight")
    channels_last = x.movedim(1, -1)
    operand = channels_last.reshape(-1, x.shape[1])
    matrix = weight.reshape(weight.shape[0], weight.shape[1]).t()
    # CUDA calls the pinned biased persistent GEMM directly. Keep the CPU
    # oracle operands FP32 before the biased reduction.
    if x.device.type == "cpu":
        operand, matrix = operand.float(), matrix.float()
        bias = None if bias is None else bias.float()
    result = invariant_gemm(operand, matrix, bias)
    return (
        result.to(x.dtype)
        .reshape(*channels_last.shape[:-1], weight.shape[0])
        .movedim(-1, 1)
    )


def sum_fp32(
    x: torch.Tensor, dim: int | tuple[int, ...], keepdim: bool = False
) -> torch.Tensor:
    """@spec PORT-PREC-021: widen before summation, restore the activation dtype."""
    return x.float().sum(dim=dim, keepdim=keepdim).to(x.dtype)


def softmax_fp32_sum(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """@spec PORT-PREC-021: opaque fixed FP32 reduction, activation result."""
    return torch.ops.nemotron_bi.softmax_sum.default(x.float(), dim).to(x.dtype)


def log_softmax_fp32_sum(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """@spec PORT-PREC-021: accumulate exp in FP32 before log normalization."""
    if x.shape[dim] == 0:
        return x.clone()
    shifted = x - x.amax(dim=dim, keepdim=True)
    return (
        shifted.float() - shifted.exp().float().sum(dim=dim, keepdim=True).log()
    ).to(x.dtype)


def layer_norm_fp32(
    x: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """@spec PORT-PREC-021: normalization and affine precede the result cast."""
    return torch.ops.nemotron_bi.layer_norm_sum.default(
        x.float(),
        list(normalized_shape),
        None if weight is None else weight.float(),
        None if bias is None else bias.float(),
        eps,
    ).to(x.dtype)


def depthwise_conv1d_fp32(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> torch.Tensor:
    """@spec PORT-PREC-021: depthwise taps reduce in FP32, independently per row."""
    return torch.ops.nemotron_bi.depthwise_conv1d_sum.default(
        x.float(),
        weight.float(),
        None if bias is None else bias.float(),
        stride,
        padding,
        dilation,
    ).to(x.dtype)


def strided_conv2d_fp32(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> torch.Tensor:
    """@spec PORT-PREC-020, PORT-PREC-021: opaque ordered FP32 tap sum."""
    return torch.ops.nemotron_bi.strided_conv2d_sum.default(
        x.float(),
        weight.float(),
        None if bias is None else bias.float(),
        [stride, stride] if isinstance(stride, int) else list(stride),
        [padding, padding] if isinstance(padding, int) else list(padding),
        [dilation, dilation] if isinstance(dilation, int) else list(dilation),
        groups,
    ).to(x.dtype)


@dataclass(frozen=True)
class QualificationVerdict:
    ready: bool
    qualified: bool
    default: bool = False
    supported: bool = False


def qualification_verdict(
    fingerprint: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    served_geometries: Sequence[int],
    deployment_cap: int,
) -> QualificationVerdict:
    """@spec PORT-PREC-014, PORT-PREC-016, PORT-PREC-020, PORT-PREC-021,
    PORT-PREC-023, PORT-PREC-024, PORT-PREC-026, PORT-PREC-027:
    accept only a complete, identity-bound qualification envelope.

    Gates are the lab's verified evidence attestations, not inferred from mode
    selection. The deployment cap never reduces the B128 qualification envelope.
    PORT-PREC-022 disclosure gates publication only, not qualification or readiness.
    """
    ready = evidence.get("executable") is True and not evidence.get("execution_failure")
    gates = evidence.get("gates", {})
    qualified = (
        ready
        and artifact_matches(fingerprint, evidence.get("fingerprint", {}))
        and fingerprint.get("batch_invariant_mode") == "on"
        and str(fingerprint.get("gpu", "")).split("-")[0]
        in {"NVIDIA A100", "NVIDIA A40"}
        and {1, 2, 31, 63, 64, 128} <= set(evidence.get("populations", ()))
        and bool(served_geometries)
        and set(served_geometries) <= set(evidence.get("geometries", ()))
        and {"repeated", "mixed"} <= set(evidence.get("compositions", ()))
        and evidence.get("independent_b128_runs", 0) >= 2
        and all(
            gates.get(f"PORT-PREC-{number:03}") is True
            for number in (14, 16, 20, 21, 24, 26, 27)
        )
    )
    return QualificationVerdict(ready=ready, qualified=qualified)
