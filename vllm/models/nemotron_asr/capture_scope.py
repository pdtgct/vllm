# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scope marker for an enclosing whole-transaction graph capture.

While the scope is active, nested encoder, decoder and CHUNK graph bindings run
their bodies eagerly, so the enclosing capture records their kernels directly
instead of a nested graph launch. Host-to-device staging is forbidden inside the
scope: a captured copy would freeze its pinned source and replay stale values.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_OUTER_TRANSACTION_CAPTURE: ContextVar[bool] = ContextVar(
    "nemotron_outer_transaction_capture", default=False
)


@contextmanager
def outer_transaction_capture() -> Iterator[None]:
    """Mark the enclosed calls as the body of a whole-transaction capture."""
    token = _OUTER_TRANSACTION_CAPTURE.set(True)
    try:
        yield
    finally:
        _OUTER_TRANSACTION_CAPTURE.reset(token)


def outer_transaction_capture_active() -> bool:
    return _OUTER_TRANSACTION_CAPTURE.get()


def nested_graph_mode(runtime: Any) -> Any:
    """Return the mode a nested binding launches with in the current scope."""
    if _OUTER_TRANSACTION_CAPTURE.get():
        return runtime.eager_mode
    return runtime.graph_mode


def reject_host_staging_in_capture() -> None:
    if _OUTER_TRANSACTION_CAPTURE.get():
        raise RuntimeError(
            "host-to-device staging inside a transaction graph body would replay "
            "a frozen host source"
        )
