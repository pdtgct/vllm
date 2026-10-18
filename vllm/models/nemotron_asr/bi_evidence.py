# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in PORT-PREC-021 Inductor callsite -> dispatch attribution.

Pass a bound encoder-transition callable and its tensor arguments. Retain the
returned post-grad graph, generated code and dispatch counts with GPU profiler
receipts. CPU results establish coverage only, not CUDA accumulation evidence.
This diagnostic disables Inductor's internal CUDA graphs to count dispatches;
production outer CUDA graphs capture the same custom-op launches on warmup.
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm.models.nemotron_asr.bi_ops import dispatch_counts

# Any of these outside the opaque namespace is a coverage failure, including
# future paths introduced through decompositions or optional optimizations.
_FORBIDDEN = (
    "mm",
    "bmm",
    "addmm",
    "matmul",
    "linear",
    "einsum",
    "convolution",
    "conv1d",
    "conv2d",
    "softmax",
    "layer_norm",
    "batch_norm",
    "scaled_dot_product",
    "sum",
    "mean",
    "amax",
    "max",
    "var",
    "std",
    "prod",
)


@dataclass(frozen=True)
class CompiledEvidence:
    post_grad_graph: str
    generated_code: str
    op_counts: dict[str, int]
    dispatch_counts: dict[str, int]


def assert_opaque_graph(graph: torch.fx.Graph) -> Counter:
    counts: Counter = Counter()
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        if target.startswith("nemotron_bi."):
            counts[target.split(".")[1]] += 1
        elif target.startswith(("aten.", "prims.")):
            base = target.split(".")[1]
            if (
                base.lstrip("_") in _FORBIDDEN
                or any(
                    part in base
                    for part in (
                        "softmax",
                        "layer_norm",
                        "convolution",
                        "scaled_dot_product",
                        "batch_norm",
                    )
                )
                or base.endswith("mm")
            ):
                raise AssertionError(f"unprotected reduction/GEMM: {target}")
    if not counts:
        raise AssertionError("no nemotron_bi callsites in post-grad graph")
    return counts


def assert_opaque_code(code: str) -> None:
    forbidden = (
        "extern_kernels.mm(",
        "extern_kernels.bmm(",
        "extern_kernels.addmm(",
        "extern_kernels.convolution(",
        "tl.dot(",
        "tl.sum(",
        "tl.max(",
        "triton_helpers.sum(",
        "triton_helpers.max(",
    )
    # Triton pointwise kernels are expected on CUDA; matmul templates carry
    # tl.dot or a matmul name, not merely async_compile.triton.
    for marker in forbidden:
        if marker in code:
            raise AssertionError(f"unprotected generated kernel: {marker}")
    if "torch.ops.nemotron_bi." not in code:
        raise AssertionError("generated code has no opaque callsites")


def compile_with_evidence(fn: Callable, args: tuple) -> tuple[Any, CompiledEvidence]:
    """Fullgraph compile/run; raises on unprotected GEMM/reduction lowering."""
    from torch._inductor import config
    from torch._inductor.utils import run_and_get_code

    graphs: list[str] = []
    counts: Counter = Counter()

    def inspect(graph):
        counts.update(assert_opaque_graph(graph))
        graphs.append(str(graph))

    with config.patch(
        post_grad_custom_post_pass=inspect,
        fx_graph_cache=False,
        **{"triton.cudagraphs": False},
    ):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=False)
        with dispatch_counts() as dispatched:
            result, codes = run_and_get_code(compiled, *args)
    code = "\n".join(codes)
    assert_opaque_code(code)
    if not graphs:
        raise AssertionError("post-grad coverage hook did not execute")
    for op, count in counts.items():
        executed = sum(
            value for key, value in dispatched.items() if key.split(":")[0] == op
        )
        if executed < count:
            raise AssertionError(f"missing dispatch for {op}: {executed} < {count}")
    return result, CompiledEvidence(
        "\n".join(graphs), code, dict(counts), dict(dispatched)
    )
