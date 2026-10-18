# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned vLLM installation adapter and Nemotron accumulation contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from threading import RLock
from typing import Any

import torch

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
_install_lock = RLock()
_installed: BatchInvariantExecution | None = None
_install_failure: Exception | None = None
_schema_library: Any = None
_schema_source: Any = None
_schema_installed: tuple[str, ...] = ()


def installed_batch_invariant_mode(
    *, module: Any = None, initialize: Callable[[], None] | None = None
) -> BatchInvariantExecution:
    """@spec PORT-PREC-018: require successful initialization, not just a library.

    The production witness is process-local and sticky, including failure:
    vLLM sets its flag before registering operators, so retrying a failed
    install could otherwise mistake its early return for successful completion.
    Explicit dependencies give probes the same validation and adapters;
    opaque test witnesses only snapshot the installed flag.
    """
    global _installed, _install_failure
    if module is not None or initialize is not None:
        if module is None or initialize is None:
            raise ValueError("module and initialize must be supplied together")
        with _install_lock:
            return _snapshot_install(module, initialize)
    with _install_lock:
        if _install_failure is not None:
            raise _install_failure
        if _installed is None:
            from vllm.model_executor.determinism import batch_invariant

            try:
                _installed = _snapshot_install(
                    batch_invariant, batch_invariant.init_batch_invariance
                )
            except Exception as error:
                _install_failure = error
                raise
        return _installed


def _snapshot_install(
    module: Any, initialize: Callable[[], None]
) -> BatchInvariantExecution:
    initialize()
    flag = getattr(module, "_batch_invariant_MODE", None)
    if type(flag) is not bool or not hasattr(module, "_batch_invariant_LIB"):
        raise RuntimeError("vLLM batch-invariant installed state is absent")
    if flag and module._batch_invariant_LIB is None:
        raise RuntimeError("vLLM batch-invariant installation is incomplete")
    # Dependency-injected record-only probes may supply an opaque witness.
    # A real Library (including CUDA probes) always receives the adapters.
    adapters = (
        _install_schema_adapters(module)
        if flag and isinstance(module._batch_invariant_LIB, torch.library.Library)
        else ()
    )
    return BatchInvariantExecution(flag, adapters)


def _schema_wrappers(module: Any) -> dict[str, Callable]:
    """@spec PORT-PREC-021, PORT-PREC-027: adapt schemas, retain vLLM kernels.

    The optional third GEMM argument is out_dtype in newer aten schemas,
    not the output buffer. All out overloads copy only after computation.
    Unsupported accumulator/output types fail explicitly instead of narrowing.
    Retained for non-encoder ATen callers; opaque encoder ops bypass these.
    """

    def operands(a, b, out_dtype):
        if a.dtype != b.dtype:
            raise ValueError("batch-invariant GEMM operands must have matching dtypes")
        if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("batch-invariant GEMM requires fp16, bf16, or fp32")
        if out_dtype is None or out_dtype == a.dtype:
            return a, b
        if out_dtype != torch.float32:
            raise ValueError(
                "batch-invariant out_dtype must be float32 or the input dtype"
            )
        return a.float(), b.float()

    def store(result, out):
        if out is None:
            return result
        if out.dtype != result.dtype or out.device != result.device:
            raise ValueError("batch-invariant out tensor dtype/device mismatch")
        if out.shape != result.shape:
            out.resize_(result.shape)
        out.copy_(result)
        return out

    def mm(input, mat2, out_dtype=None, *, out=None):
        a, b = operands(input, mat2, out_dtype)
        return store(module.mm_batch_invariant(a, b), out)

    def bmm(input, mat2, out_dtype=None, *, out=None):
        a, b = operands(input, mat2, out_dtype)
        return store(module.bmm_batch_invariant(a, b), out)

    def addmm(input, mat1, mat2, out_dtype=None, *, beta=1, alpha=1, out=None):
        a, b = operands(mat1, mat2, out_dtype)
        if input.dtype != mat1.dtype:
            raise ValueError("batch-invariant addmm bias must match input dtype")
        if isinstance(beta, complex) or isinstance(alpha, complex):
            raise ValueError("batch-invariant addmm requires real alpha/beta")
        # The pinned fused kernel accepts a 1-D bias only. Preserve its
        # bias-before-store path for the usual linear case, including fp16.
        if (
            beta == 1
            and alpha == 1
            and input.ndim == 1
            and input.shape[0] == b.shape[1]
        ):
            result = module.addmm_batch_invariant(input.to(a.dtype), a, b)
        else:
            # Widen BEFORE GEMM: scaling a stored fp16 result would lose the
            # accumulator. beta=0 must ignore NaNs/Infs in the bias entirely.
            bias = torch.broadcast_to(input, (a.shape[0], b.shape[1]))
            product = module.mm_batch_invariant(a.float(), b.float())
            result = product * alpha
            if beta != 0:
                result = result + bias.float() * beta
            result = result.to(a.dtype)
        return store(result, out)

    def matmul(input, other, *, out=None):
        return store(module.matmul_batch_invariant(input, other), out)

    def linear(input, weight, bias=None, *, out=None):
        return store(module.linear_batch_invariant(input, weight, bias), out)

    def softmax(input, dim, half_to_float, *, out=None):
        return store(
            module.softmax_batch_invariant(
                input.float() if half_to_float else input, dim
            ),
            out,
        )

    def softmax_int(input, dim, dtype=None, *, out=None):
        x = input if dtype is None else input.to(dtype)
        return store(module.softmax_batch_invariant(x, dim), out)

    def log_softmax(input, dim, half_to_float, *, out=None):
        return store(
            module._log_softmax_batch_invariant(input, dim, half_to_float), out
        )

    def mean(input, dim, keepdim=False, *, dtype=None, out=None):
        target = input.dtype if dtype is None else dtype
        if target not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("batch-invariant mean requires fp16, bf16, or fp32 result")
        dims = list(range(input.ndim)) if dim is None or len(dim) == 0 else dim
        result = module.mean_batch_invariant(
            input.float(), dims, keepdim, dtype=torch.float32
        )
        return store(result.to(target), out)

    return {
        "mm": mm,
        "addmm": addmm,
        "bmm": bmm,
        "matmul": matmul,
        "linear": linear,
        "_softmax": softmax,
        "softmax.int": softmax_int,
        "_log_softmax": log_softmax,
        "mean.dim": mean,
    }


# @spec PORT-PREC-022: additional process-wide effects for cost disclosure.
# Existing vLLM entries remain required; these describe our signature layer.
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


def _schema_dispatch(
    wrapper: Callable, keyset: torch._C.DispatchKeySet, *args: Any, **kwargs: Any
) -> Any:
    """Keep the dispatcher calling convention separate from tensor schemas.

    Native conditional overrides (including CUDA bmm) receive a keyset before
    the schema arguments. Own that convention explicitly when replacing them;
    the public torch.bmm wrapper must continue to accept tensors only.
    """
    return wrapper(*args, **kwargs)


def _install_schema_adapters(module: Any) -> tuple[str, ...]:
    """@spec PORT-PREC-018, PORT-PREC-022, PORT-PREC-028: install after vLLM.

    Only adapt kernels actually registered by vLLM on this platform. In
    particular, do not install Ampere GEMM kernels on Hopper/Blackwell.
    Called under the installed-mode lock, before construction/compilation.
    """
    global _schema_library, _schema_source, _schema_installed
    if not module._batch_invariant_MODE:
        return ()
    source = module._batch_invariant_LIB
    if _schema_library is not None:
        if _schema_source is not source:
            raise RuntimeError("batch-invariant schema adapter source changed")
        return _schema_installed
    wrappers = _schema_wrappers(module)
    library = torch.library.Library("aten", "IMPL")
    installed = []
    try:
        for entry in sorted(source._op_impls):
            namespace, name, key = entry.split("/")
            if namespace != "aten":
                continue
            name = "softmax.int" if name == "softmax" else name
            if name not in wrappers:
                continue
            base = name.split(".")[0]
            overloads = {name: wrappers[name]}
            schemas = getattr(torch.ops.aten, base)._schemas
            candidates = (
                ("out", "dtype", "dtype_out")
                if base in ("mm", "addmm", "bmm")
                else ("int_out" if name == "softmax.int" else "out",)
            )
            for overload in candidates:
                if overload in schemas:
                    overloads[f"{base}.{overload}"] = wrappers[name]
            for overload, wrapper in overloads.items():
                library.impl(
                    overload,
                    partial(_schema_dispatch, wrapper),
                    key,
                    with_keyset=True,
                    allow_override=True,
                )
                installed.append(f"aten::{overload}/{key}")
        # vLLM also monkeypatches the public Python entry point. Its original
        # keyword-only out signature cannot accept the dtype positional form.
        if any(item.startswith("aten::bmm/") for item in installed):
            torch.bmm = wrappers["bmm"]
    except Exception:
        library._destroy()
        raise
    _schema_library, _schema_source = library, source
    _schema_installed = tuple(installed)
    return _schema_installed


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
