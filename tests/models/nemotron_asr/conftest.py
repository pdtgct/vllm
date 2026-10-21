# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test-only isolation for vLLM's process-wide batch-invariance installation."""

import os
import sys
from contextlib import contextmanager
from typing import Any

import pytest
import torch

from vllm.model_executor.determinism import schema_adapters

_ENV_KEYS = (
    "VLLM_BATCH_INVARIANT",
    "VLLM_ALLREDUCE_USE_SYMM_MEM",
    "VLLM_CUSTOM_ALLREDUCE_ALGO",
    "VLLM_ROCM_USE_SKINNY_GEMM",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUBLASLT_WORKSPACE_SIZE",
    "VLLM_USE_AOT_COMPILE",
    "NCCL_LAUNCH_MODE",
    "NCCL_NO_CACHE",
    "NCCL_COLLNET_ENABLE",
    "NCCL_NVLS_ENABLE",
    "NCCL_P2P_NET_DISABLE",
    "NCCL_MIN_NCHANNELS",
    "NCCL_MAX_NCHANNELS",
    "NCCL_PROTO",
    "NCCL_ALGO",
    "NCCL_NTHREADS",
    "NCCL_SOCKET_NTHREADS",
)


@contextmanager
def preserve_batch_invariance(batch_invariant, configs):
    """Undo registrations and all pinned initializer side effects, even on failure."""
    from torch._inductor import config as inductor_config

    from vllm.utils import nccl

    env = {key: os.environ.get(key) for key in _ENV_KEYS}
    saved: list[tuple[Any, str, Any]] = []
    for owner, names in (
        (
            batch_invariant,
            (
                "_batch_invariant_MODE",
                "_batch_invariant_LIB",
                "_fp16_block_size_n",
                "_fp32_block_size_n",
                "_fp32_num_stages",
            ),
        ),
        (
            configs,
            ("_TUNED_MATMUL_CONFIGS_FOR_DEVICE", "_TUNED_MATMUL_CONFIGS_RESOLVED"),
        ),
        (
            schema_adapters,
            ("_library", "_source", "_installed", "_failure"),
        ),
        (torch, ("bmm",)),
        (nccl, ("_unpinned_env",)),
        (
            torch.backends.cuda.matmul,
            (
                "allow_fp16_reduced_precision_reduction",
                "allow_bf16_reduced_precision_reduction",
                "fp32_precision",
            ),
        ),
        (torch.backends.cudnn.conv, ("fp32_precision",)),
        (torch.backends.cudnn.rnn, ("fp32_precision",)),
    ):
        saved.extend((owner, name, getattr(owner, name)) for name in names)
    for owner, names in (
        (batch_invariant, ("_BLOCK_SIZES_RESOLVED", "_NUM_SMS", "_init_failure")),
        (inductor_config, ("batch_invariant",)),
    ):
        saved.extend(
            (owner, name, getattr(owner, name))
            for name in names
            if hasattr(owner, name)
        )
    for name, module in tuple(sys.modules.items()):
        if name.endswith("test_encoder_batch_invariance"):
            saved.extend(
                (module, attr, getattr(module, attr))
                for attr in (
                    "_APPLIED_MODE",
                    "_EXECUTION_MODE",
                    "_CUDA_INITIALIZED_BEFORE_MODE",
                    "_REDUCED_PRECISION_NOTE",
                )
            )
    adapter_library = schema_adapters._library
    library = batch_invariant._batch_invariant_LIB
    blas = torch.backends.cuda.preferred_blas_library()
    try:
        yield
    finally:
        # Drop compiled artifacts that could retain a mode-on specialization.
        torch._dynamo.reset()
        if (
            schema_adapters._library is not None
            and schema_adapters._library is not adapter_library
        ):
            schema_adapters._library._destroy()
        installed = batch_invariant._batch_invariant_LIB
        if installed is not None and installed is not library:
            installed._destroy()
        for owner, name, value in saved:
            setattr(owner, name, value)
        torch.backends.cuda.preferred_blas_library(blas)
        for key, value in env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


@pytest.fixture
def isolated_batch_invariance():
    from vllm.model_executor.determinism import batch_invariant, batch_invariant_configs

    with preserve_batch_invariance(batch_invariant, batch_invariant_configs):
        yield
