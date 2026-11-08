# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVTX phase ranges for the streaming transaction (PORT-OBS).

A profile can only be compared to another profile if the spans carry the
same names, so the phase names live here as a closed set rather than at
each call site. They are the names the profiling-record schema and the
evidence card expect (``port.<phase>``); renaming one breaks comparison
with every prior digest and is a reviewed change, like renaming a metric.

Instrumentation on a hot path has to be free when it is off. At the
target operating point the transaction runs ~12,500 times a second, so a
disabled phase must cost one attribute load and one predictable branch:
no generator, no context-manager allocation, no string work. Both the
enabled and disabled objects are preallocated once per name at import.
"""

from __future__ import annotations

import os
from types import TracebackType
from typing import Literal

import torch

#: Enabled by environment only: profiling is a deployment-time decision
#: made by the capture harness (``nsys profile``), never a request-time
#: one, and the flag is read once so the hot path never touches os.environ.
_ENV_FLAG = "VLLM_NEMOTRON_ASR_NVTX"
_ENABLED = os.getenv(_ENV_FLAG, "0").strip().lower() in {"1", "true", "yes", "on"}

#: The closed set of transaction phases. Ordered as the transaction runs.
PHASES: tuple[str, ...] = (
    "port.ingest",
    "port.featurize",
    "port.encode",
    "port.decode",
    "port.scatter",
    "port.park",
)


#: Nested attribution preserves the comparable transaction-level PHASES.
ENCODER_SUBPHASES: tuple[str, ...] = (
    "port.encode.stage_in",
    "port.encode.replay",
    "port.encode.stage_out",
)


class _NullPhase:
    """Shared no-op range; entering and leaving it allocates nothing."""

    __slots__ = ()

    def __enter__(self) -> _NullPhase:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        return False


class _NvtxPhase:
    """One named NVTX range, shared across calls (it holds only a name).

    NVTX ranges are a per-thread stack, so a shared instance stays
    correct under interleaved use: every ``__enter__`` pushes and every
    ``__exit__`` pops, including when the body raises.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __enter__(self) -> _NvtxPhase:
        torch.cuda.nvtx.range_push(self._name)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        torch.cuda.nvtx.range_pop()
        return False


_NULL = _NullPhase()
_RANGES: dict[str, _NullPhase | _NvtxPhase] = {
    name: (_NvtxPhase(name) if _ENABLED else _NULL)
    for name in (*PHASES, *ENCODER_SUBPHASES)
}


def enabled() -> bool:
    """Whether NVTX phase ranges are active in this process."""
    return _ENABLED


def phase(name: str) -> _NullPhase | _NvtxPhase:
    """Return the preallocated range for one declared phase name.

    An undeclared name returns the no-op range rather than raising: a
    typo must not take down serving, and the missing span is visible in
    the digest, which is where a profiling mistake belongs.
    """
    return _RANGES.get(name, _NULL)
