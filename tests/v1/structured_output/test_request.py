# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for StructuredOutputRequest grammar-future polling."""

from concurrent.futures import Future

from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.structured_output.request import (
    StructuredOutputRequest,
    same_structured_outputs,
)


def _request_with_future(fut: Future) -> StructuredOutputRequest:
    req = StructuredOutputRequest(params=StructuredOutputsParams(regex="a+"))
    req.grammar = fut
    return req


def test_pending_grammar_future_is_not_ready():
    req = _request_with_future(Future())
    assert not req.is_grammar_ready
    assert req.grammar is None


def test_grammar_compile_raising_timeout_error_is_a_failure():
    """A compile that raised TimeoutError must be treated as a failed compile,
    not as still-compiling (#53130)."""
    fut: Future = Future()
    fut.set_exception(TimeoutError("compiler timed out"))
    req = _request_with_future(fut)
    assert req.is_grammar_ready
    assert isinstance(req.grammar, TimeoutError)


# @spec PSTATE-DECL-007
def test_same_structured_outputs_ignores_the_backend_validation_records():
    a = StructuredOutputsParams(regex="a+")
    b = StructuredOutputsParams(regex="a+")
    b._backend = "xgrammar"
    b._backend_was_auto = True
    assert a != b
    assert same_structured_outputs(a, b)


# @spec PSTATE-DECL-007
def test_same_structured_outputs_compares_the_constraint():
    assert not same_structured_outputs(
        StructuredOutputsParams(json='{"type": "object"}'),
        StructuredOutputsParams(json='{"type": "string"}'),
    )
    assert not same_structured_outputs(
        StructuredOutputsParams(regex="a+"), StructuredOutputsParams(choice=["a"])
    )


# @spec PSTATE-DECL-007
def test_same_structured_outputs_with_none():
    params = StructuredOutputsParams(regex="a+")
    assert same_structured_outputs(None, None)
    assert not same_structured_outputs(None, params)
    assert not same_structured_outputs(params, None)
