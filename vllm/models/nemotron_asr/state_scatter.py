# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Predicated resident-page commits for cache-aware streaming state.

The transaction validates every descriptor before its first resident write.
This module then performs one fixed-shape operation per state tensor: clean
rows write their unique live page and failed rows issue no store at all.  In
particular, failed rows are never redirected to vLLM's null block; block zero
is only a metadata sentinel unless a consuming kernel explicitly masks it.
"""

from __future__ import annotations

import math
import threading
from typing import Final

import torch

try:
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover - optional on CPU development
    triton = None
    tl = None
    _TRITON_LOAD_ERROR: Exception | None = exc
else:
    _TRITON_LOAD_ERROR = None

_MAX_BLOCK_SIZE: Final = 1024
_CUDA_DTYPES: Final = frozenset((torch.float32, torch.int32, torch.int64))
#: Compute capabilities the CUDA masked scatter is qualified on — the
#: complete SM8x (Ampere/Ada) set plus Turing SM75, each validated by a
#: live serving round (SM75 qualified 2026-08-05 on a T4: warmup
#: compile/execute at pool allocation, a full streaming session, and a
#: four-way concurrent cell, all green). Admission is keyed off this
#: declared set, never a code default: a capability joins only with a
#: green qualification round on real hardware (Hopper SM90 is a pending
#: lane), so an unqualified device fails closed at warmup with a named
#: error instead of surfacing a raw kernel fault mid-stream.
_QUALIFIED_CAPABILITIES: Final = frozenset(((7, 5), (8, 0), (8, 6), (8, 7), (8, 9)))
_OP_NAME: Final = "nemotron_asr_masked_page_scatter_"
_OP_REGISTERED = False
_REGISTRATION_LOCK = threading.Lock()
_WARMUP_LOCK = threading.Lock()
_WARMED_SPECIALIZATIONS: set[
    tuple[int, int, int, torch.dtype, int, int, int, int, int]
] = set()


if triton is not None and tl is not None:

    @triton.jit  # type: ignore[untyped-decorator]
    def _masked_page_scatter_kernel(  # type: ignore[no-untyped-def]
        pool,
        scratch,
        block_ids,
        row_status,
        page_numel: tl.constexpr,
        pool_stride_0: tl.constexpr,
        scratch_stride_0: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        tile = tl.program_id(1).to(tl.int64)
        status = tl.load(row_status + row)
        if status != 0:
            return
        block = tl.load(block_ids + row).to(tl.int64)
        offsets = tile * block_size + tl.arange(0, block_size).to(tl.int64)
        mask = offsets < page_numel
        source = tl.load(
            scratch + row * scratch_stride_0 + offsets,
            mask=mask,
        )
        tl.store(
            pool + block * pool_stride_0 + offsets,
            source,
            mask=mask,
        )


def _canonical_inner_layout(tensor: torch.Tensor) -> bool:
    """Whether every page is dense row-major despite block padding."""
    expected = 1
    for size, stride in zip(
        reversed(tensor.shape[1:]), reversed(tensor.stride()[1:]), strict=True
    ):
        if size > 1 and stride != expected:
            return False
        expected *= size
    return tensor.stride(0) >= expected


def _page_numel(tensor: torch.Tensor) -> int:
    return math.prod(tensor.shape[1:]) if tensor.dim() > 1 else 1


def _launch_config(page_numel: int) -> tuple[int, int]:
    """Choose a bounded power-of-two tile without over-warping tiny pages."""
    block_size = max(
        32,
        1 << (min(page_numel, _MAX_BLOCK_SIZE) - 1).bit_length(),
    )
    if block_size <= 256:
        return block_size, 1
    if block_size <= 512:
        return block_size, 4
    return block_size, 8


def _cuda_specialization(
    pool: torch.Tensor,
    scratch: torch.Tensor,
) -> tuple[int, int, int, torch.dtype, int, int, int, int, int]:
    """Return every value that can select a distinct Triton binary."""
    device_index = pool.get_device()
    major, minor = torch.cuda.get_device_capability(pool.device)
    page_numel = _page_numel(pool)
    block_size, num_warps = _launch_config(page_numel)
    return (
        device_index,
        major,
        minor,
        pool.dtype,
        page_numel,
        pool.stride(0),
        scratch.stride(0),
        block_size,
        num_warps,
    )


def _require_cuda_specialization_warmed(
    pool: torch.Tensor,
    scratch: torch.Tensor,
) -> None:
    signature = _cuda_specialization(pool, scratch)
    # Warmup is a pre-admission phase. CPython set membership is atomic under
    # the GIL; taking the compile lock for every state descriptor on every
    # streaming step would add needless host serialization.
    if signature not in _WARMED_SPECIALIZATIONS:
        raise ValueError(
            "masked scatter specialization was not warmed before serving; "
            "warm every resident page layout after pool allocation"
        )


def _validate_masked_page_scatter(
    pool: torch.Tensor,
    scratch: torch.Tensor,
    block_ids: torch.Tensor,
    row_status: torch.Tensor,
    *,
    require_cuda_warmup: bool,
) -> None:
    """Validate descriptor metadata, optionally requiring startup warmup."""
    if pool.dim() < 1 or scratch.dim() < 1:
        raise ValueError("pool and scratch must have rank >= 1")
    if tuple(pool.shape[1:]) != tuple(scratch.shape[1:]):
        raise ValueError("scratch page shape differs from the pool page shape")
    if pool.dtype != scratch.dtype:
        raise ValueError("scratch dtype differs from the pool dtype")
    rows = int(scratch.shape[0])
    if block_ids.dim() != 1 or row_status.dim() != 1:
        raise ValueError("block_ids and row_status must be rank-one")
    if int(block_ids.shape[0]) != rows or int(row_status.shape[0]) != rows:
        raise ValueError("scatter descriptor row count disagreement")
    if block_ids.dtype != torch.int64:
        raise ValueError("block_ids must have int64 dtype")
    if row_status.dtype != torch.int32:
        raise ValueError("row_status must have int32 dtype")
    devices = {pool.device, scratch.device, block_ids.device, row_status.device}
    if len(devices) != 1:
        raise ValueError("all scatter tensors must be on the same device")
    if _page_numel(pool) <= 0:
        raise ValueError("resident pages must not be empty")
    if not _canonical_inner_layout(pool) or not _canonical_inner_layout(scratch):
        raise ValueError("pool and scratch require contiguous row-major page interiors")
    if not block_ids.is_contiguous() or not row_status.is_contiguous():
        raise ValueError("block_ids and row_status must be contiguous")
    if pool.requires_grad or scratch.requires_grad:
        raise ValueError("resident scatter is inference-only")
    if pool.untyped_storage().data_ptr() == scratch.untyped_storage().data_ptr():
        raise ValueError("pool and scratch must not alias storage")
    if pool.device.type == "cuda":
        if pool.dtype not in _CUDA_DTYPES:
            raise ValueError(f"CUDA masked scatter dtype {pool.dtype} is not qualified")
        capability = torch.cuda.get_device_capability(pool.device)
        if capability not in _QUALIFIED_CAPABILITIES:
            qualified = ", ".join(
                f"SM{major}{minor}" for major, minor in sorted(_QUALIFIED_CAPABILITIES)
            )
            raise ValueError(
                "CUDA masked scatter is not qualified on "
                f"SM{capability[0]}{capability[1]}; "
                f"qualified capabilities: {qualified}. A capability joins the declared "
                "set only with a green qualification round on real hardware."
            )
        if _TRITON_LOAD_ERROR is not None:
            raise ValueError(
                "CUDA masked scatter requires Triton"
            ) from _TRITON_LOAD_ERROR
        _ensure_cuda_op_registered()
        if require_cuda_warmup:
            _require_cuda_specialization_warmed(pool, scratch)
    elif pool.device.type != "cpu":
        raise ValueError(
            f"masked page scatter does not support device {pool.device.type!r}"
        )


# @spec PORT-STATE-007, PORT-STATE-008
def validate_masked_page_scatter(
    pool: torch.Tensor,
    scratch: torch.Tensor,
    block_ids: torch.Tensor,
    row_status: torch.Tensor,
) -> None:
    """Validate one scatter descriptor without reading tensor values.

    Index range, liveness, non-nullness, and uniqueness remain the CPU
    :class:`RowPlan` preflight's responsibility.  This validator checks only
    shape/layout/device metadata and is safe on the synchronization-free path.
    """
    _validate_masked_page_scatter(
        pool,
        scratch,
        block_ids,
        row_status,
        require_cuda_warmup=True,
    )


def _cuda_scatter(
    pool: torch.Tensor,
    scratch: torch.Tensor,
    block_ids: torch.Tensor,
    row_status: torch.Tensor,
) -> None:
    assert triton is not None and tl is not None
    rows = int(scratch.shape[0])
    page_numel = _page_numel(pool)
    if rows == 0:
        return
    block_size, num_warps = _launch_config(page_numel)
    _masked_page_scatter_kernel[(rows, triton.cdiv(page_numel, block_size))](
        pool,
        scratch,
        block_ids,
        row_status,
        page_numel=page_numel,
        pool_stride_0=pool.stride(0),
        scratch_stride_0=scratch.stride(0),
        block_size=block_size,
        num_warps=num_warps,
    )


# @spec PORT-PERF-001, PORT-STATE-008
def warmup_masked_page_scatter(
    pool: torch.Tensor,
    scratch: torch.Tensor,
) -> None:
    """Compile one CUDA specialization before any session is admitted.

    The warmup launches a one-row all-dirty descriptor and synchronizes,
    proving both compilation and execution while the kernel's leading status
    predicate guarantees that no resident address is loaded or stored. Call
    this once for every distinct resident pool/scratch layout after cache
    allocation. Runtime validation fails closed on a specialization that was
    not warmed, preventing lazy JIT compilation inside the commit window.
    """
    if pool.device.type != "cuda" or scratch.device.type != "cuda":
        raise ValueError("masked scatter warmup requires CUDA tensors")
    if int(scratch.shape[0]) < 1:
        raise ValueError("masked scatter warmup requires one scratch row")
    one_scratch = scratch[:1]
    block_ids = torch.zeros(1, dtype=torch.int64, device=pool.device)
    row_status = torch.ones(1, dtype=torch.int32, device=pool.device)
    _validate_masked_page_scatter(
        pool,
        one_scratch,
        block_ids,
        row_status,
        require_cuda_warmup=False,
    )
    signature = _cuda_specialization(pool, one_scratch)
    with _WARMUP_LOCK:
        if signature in _WARMED_SPECIALIZATIONS:
            return
        _cuda_scatter(pool, one_scratch, block_ids, row_status)
        torch.accelerator.synchronize(pool.device)
        _WARMED_SPECIALIZATIONS.add(signature)


def _fake_scatter(
    pool: torch.Tensor,
    scratch: torch.Tensor,
    block_ids: torch.Tensor,
    row_status: torch.Tensor,
) -> None:
    del pool, scratch, block_ids, row_status


def _cuda_op_is_registered() -> bool:
    """Read process-global registration state without stale narrowing."""
    return _OP_REGISTERED


def _ensure_cuda_op_registered() -> None:
    global _OP_REGISTERED
    if _cuda_op_is_registered():
        return
    # Pool allocation/warmup can be parallel across workers.  Registration is
    # process-global, so protect the first definition independently from the
    # specialization compile lock.
    with _REGISTRATION_LOCK:
        if _cuda_op_is_registered():
            return
        if not hasattr(torch.ops.vllm, _OP_NAME):
            from vllm.utils.torch_utils import direct_register_custom_op

            direct_register_custom_op(
                op_name=_OP_NAME,
                op_func=_cuda_scatter,
                mutates_args=["pool"],
                fake_impl=_fake_scatter,
            )
        _OP_REGISTERED = True


def _execute_masked_page_scatter_(
    pool: torch.Tensor,
    scratch: torch.Tensor,
    block_ids: torch.Tensor,
    row_status: torch.Tensor,
) -> None:
    """Execute an already-validated descriptor without allocation."""
    rows = int(scratch.shape[0])
    if rows == 0:
        return
    if pool.device.type == "cpu":
        for row in range(rows):
            if int(row_status[row]) == 0:
                pool[int(block_ids[row])].copy_(scratch[row])
        return
    torch.ops.vllm.nemotron_asr_masked_page_scatter_(
        pool, scratch, block_ids, row_status
    )


# @spec PORT-ADV-004, PORT-STATE-008
def masked_page_scatter_(
    pool: torch.Tensor,
    scratch: torch.Tensor,
    block_ids: torch.Tensor,
    row_status: torch.Tensor,
) -> None:
    """Validate and execute one clean-row-only resident scatter.

    Transactions with several descriptors validate all of them first, then
    call the private prevalidated executor in their no-fail commit window.
    """
    validate_masked_page_scatter(pool, scratch, block_ids, row_status)
    _execute_masked_page_scatter_(pool, scratch, block_ids, row_status)
