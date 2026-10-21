# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ATen schema adapters for the batch-invariant kernel overrides.

``enable_batch_invariant_mode`` registers kernels for the default overloads of
a platform-dependent set of operators. Callers also reach those operators
through their ``out``, ``dtype`` and ``dtype_out`` overloads (Inductor lowers to
``out=``; newer ATen schemas take a positional ``out_dtype``), which would
otherwise run the stock, batch-variant kernels. The adapters here route those
overloads to the registered kernels, and only for operators the platform
actually overrode.
"""

import threading
from collections.abc import Callable
from functools import partial
from typing import Any

import torch

_library: Any = None
_source: Any = None
_installed: tuple[str, ...] = ()
_failure: Exception | None = None
# Taken after batch_invariant._init_lock when both are held, never before it.
_lock = threading.RLock()


def _schema_wrappers(module: Any) -> dict[str, Callable]:
    """Adapt ATen schemas to the kernels in ``module``.

    The optional third GEMM argument is out_dtype in newer aten schemas,
    not the output buffer. All out overloads copy only after computation.
    Unsupported accumulator/output types fail explicitly instead of narrowing.
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
        # The fused kernel accepts a 1-D bias only. Preserve its
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


def _schema_dispatch(
    wrapper: Callable, keyset: torch._C.DispatchKeySet, *args: Any, **kwargs: Any
) -> Any:
    """Keep the dispatcher calling convention separate from tensor schemas.

    Native conditional overrides (including CUDA bmm) receive a keyset before
    the schema arguments. Own that convention explicitly when replacing them;
    the public torch.bmm entry point must continue to accept tensors only.
    """
    return wrapper(*args, **kwargs)


def install_schema_adapters(module: Any) -> tuple[str, ...]:
    """Adapt the overloads of operators that ``module`` registered.

    Only adapt kernels actually registered on this platform; in particular,
    do not install Ampere GEMM kernels on Hopper/Blackwell. Idempotent for
    the same source library. A failure removes the partial registration and
    is raised again on every later call. Concurrent calls serialize.
    """
    with _lock:
        return _install(module)


def _install(module: Any) -> tuple[str, ...]:
    global _library, _source, _installed, _failure
    if _failure is not None:
        raise _failure
    if not module._batch_invariant_MODE:
        return ()
    source = module._batch_invariant_LIB
    if _library is not None:
        if _source is not source:
            raise RuntimeError("batch-invariant schema adapter source changed")
        return _installed
    wrappers = _schema_wrappers(module)
    library = torch.library.Library("aten", "IMPL")
    installed = []
    try:
        for entry in sorted(source._op_impls):
            namespace, name, key = entry.split("/")
            if namespace != "aten":
                continue
            # The default softmax registration is the int overload.
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
    except Exception as error:
        library._destroy()
        _failure = error
        raise
    _library, _source = library, source
    _installed = tuple(installed)
    return _installed


def installed_schema_adapters(module: Any) -> tuple[str, ...]:
    """Return the adapters installed for ``module``'s library; never install.

    An install in progress is awaited; it is never reported as installed.
    """
    with _lock:
        if _failure is not None:
            raise _failure
        if _library is None or _source is not module._batch_invariant_LIB:
            raise RuntimeError("batch-invariant installation is incomplete")
        return _installed
