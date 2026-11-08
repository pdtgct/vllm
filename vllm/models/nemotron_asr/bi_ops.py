# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opaque inference-only boundaries for PORT-PREC-021.

CUDA GEMMs call pinned vLLM launchers directly, never an ATen override.
Reductions use an explicit FP32 binary tree (convolution uses ordered taps).
Their eager elementwise launches cannot be fused/reassociated by Inductor.
Warm these operators before CUDA capture; replay executes captured launches,
not Python dispatch. CPU GEMM is an FP32 arithmetic oracle, not CUDA evidence.
"""

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch.nn import functional as F

_DISPATCH: ContextVar[Counter | None] = ContextVar("nemotron_bi_dispatch", default=None)


@contextmanager
def dispatch_counts():
    """Count completed Python dispatches; CUDA replay does not increment these."""
    counts: Counter = Counter()
    token = _DISPATCH.set(counts)
    try:
        yield counts
    finally:
        _DISPATCH.reset(token)


def _record(op: str, x: torch.Tensor, kernel: str) -> None:
    counts = _DISPATCH.get()
    if counts is not None:
        counts[f"{op}:{x.device.type}:{kernel}"] += 1


def _gemm(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    if a.is_cuda:
        from vllm.model_executor.determinism.batch_invariant import matmul_persistent

        result = matmul_persistent(a, b, bias=bias)
        kernel = "vllm.matmul_kernel_persistent"
    else:
        result = a.float() @ b.float()
        if bias is not None:
            result = result + bias.float()
        result = result.to(a.dtype)
        kernel = "cpu.fp32_gemm_oracle"
    _record("mm" if bias is None else "addmm", a, kernel)
    return result


@torch.library.custom_op("nemotron_bi::mm", mutates_args=())
def mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _gemm(a, b, None)


@mm.register_fake
def _mm_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.new_empty((a.shape[0], b.shape[1]))


@torch.library.custom_op("nemotron_bi::addmm", mutates_args=())
def addmm(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _gemm(a, b, bias)


@addmm.register_fake
def _addmm_fake(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _mm_fake(a, b)


@torch.library.custom_op("nemotron_bi::matmul", mutates_args=())
def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Includes the head-major, batch-folded relative position: [H,B*Q,D] @ [H,D,P].
    if a.is_cuda:
        from vllm.model_executor.determinism.batch_invariant import (
            matmul_batch_invariant,
        )

        result = matmul_batch_invariant(a, b)
        kernel = "vllm.matmul_kernel_persistent" if b.ndim == 2 else "vllm.bmm_kernel"
    else:
        result = torch.matmul(a.float(), b.float()).to(a.dtype)
        kernel = "cpu.fp32_matmul_oracle"
    _record("matmul", a, kernel)
    return result


@matmul.register_fake
def _matmul_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    return a.new_empty((*shape, a.shape[-2], b.shape[-1]))


def _tree_sum(x: torch.Tensor) -> torch.Tensor:
    """FP32 adjacent-pair tree; topology depends only on reduction width."""
    assert x.dtype == torch.float32
    if x.shape[-1] == 0:
        return x.new_zeros(x.shape[:-1])
    while x.shape[-1] > 1:
        pairs = x.shape[-1] // 2
        partial = x[..., : 2 * pairs : 2] + x[..., 1 : 2 * pairs : 2]
        x = torch.cat((partial, x[..., -1:]), -1) if x.shape[-1] % 2 else partial
    return x[..., 0]


def _tree_max(x: torch.Tensor) -> torch.Tensor:
    while x.shape[-1] > 1:
        pairs = x.shape[-1] // 2
        partial = torch.maximum(x[..., : 2 * pairs : 2], x[..., 1 : 2 * pairs : 2])
        x = torch.cat((partial, x[..., -1:]), -1) if x.shape[-1] % 2 else partial
    return x[..., 0]


@torch.library.custom_op("nemotron_bi::softmax_sum", mutates_args=())
def softmax_sum(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.shape[dim] == 0:
        result = x.clone(memory_format=torch.contiguous_format)
    else:
        rows = x.movedim(dim, -1)
        exp = (rows - _tree_max(rows).unsqueeze(-1)).exp()
        result = (exp / _tree_sum(exp).unsqueeze(-1)).movedim(-1, dim).contiguous()
    _record("softmax_sum", x, "port.fp32_tree")
    return result


@softmax_sum.register_fake
def _softmax_fake(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op("nemotron_bi::layer_norm_sum", mutates_args=())
def layer_norm_sum(
    x: torch.Tensor,
    shape: list[int],
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    rows = x.flatten(x.ndim - len(shape))
    mean = _tree_sum(rows).unsqueeze(-1) / rows.shape[-1]
    centered = rows - mean
    var = _tree_sum(centered * centered).unsqueeze(-1) / rows.shape[-1]
    result = (centered * torch.rsqrt(var + eps)).reshape(x.shape)
    if weight is not None:
        result = result * weight
    if bias is not None:
        result = result + bias
    _record("layer_norm_sum", x, "port.fp32_tree")
    return result.contiguous()


@layer_norm_sum.register_fake
def _norm_fake(
    x: torch.Tensor,
    shape: list[int],
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op("nemotron_bi::depthwise_conv1d_sum", mutates_args=())
def depthwise_conv1d_sum(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    taps = weight.shape[-1]
    windows = F.pad(x, (padding, padding)).unfold(
        -1, dilation * (taps - 1) + 1, stride
    )[..., ::dilation]
    result = _tree_sum(windows * weight[:, 0, :][None, :, None, :])
    if bias is not None:
        result = result + bias[None, :, None]
    _record("depthwise_conv1d_sum", x, "port.fp32_tree")
    return result


@depthwise_conv1d_sum.register_fake
def _conv1_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    length = (
        x.shape[-1] + 2 * padding - dilation * (weight.shape[-1] - 1) - 1
    ) // stride + 1
    return x.new_empty((x.shape[0], weight.shape[0], length))


@torch.library.custom_op("nemotron_bi::strided_conv2d_sum", mutates_args=())
def strided_conv2d_sum(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    groups: int,
) -> torch.Tensor:
    # Ordered FP32 channel/tap accumulation avoids cuDNN selection and large
    # broadcasted im2col products. Only pointwise convolutions use GEMM.
    batch, channels, height, width = x.shape
    outputs, per_group, kh, kw = weight.shape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    oh = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    if channels != per_group * groups or outputs % groups:
        raise ValueError("invalid grouped convolution geometry")
    padded = F.pad(x, (pw, pw, ph, ph)).reshape(
        batch, groups, per_group, height + 2 * ph, width + 2 * pw
    )
    weights = weight.reshape(groups, outputs // groups, per_group, kh, kw)
    result = x.new_zeros((batch, groups, outputs // groups, oh, ow))
    for channel in range(per_group):
        for h in range(kh):
            for w in range(kw):
                values = padded[
                    :,
                    :,
                    channel,
                    h * dh : h * dh + oh * sh : sh,
                    w * dw : w * dw + ow * sw : sw,
                ]
                result = (
                    result
                    + values.unsqueeze(2)
                    * weights[:, :, channel, h, w][None, :, :, None, None]
                )
    result = result.reshape(batch, outputs, oh, ow)
    if bias is not None:
        result = result + bias[None, :, None, None]
    _record("strided_conv2d_sum", x, "port.fp32_ordered_taps")
    return result


@strided_conv2d_sum.register_fake
def _conv2_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    groups: int,
) -> torch.Tensor:
    spatial = [
        (x.shape[i + 2] + 2 * padding[i] - dilation[i] * (weight.shape[i + 2] - 1) - 1)
        // stride[i]
        + 1
        for i in range(2)
    ]
    return x.new_empty((x.shape[0], weight.shape[0], *spatial))
