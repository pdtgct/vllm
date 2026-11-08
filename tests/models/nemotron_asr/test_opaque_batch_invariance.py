# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PORT-PREC-021 compiled dispatch custody and CPU arithmetic oracles."""

import pytest
import torch

from vllm.models.nemotron_asr import batch_invariance as bi

pytestmark = [pytest.mark.cpu_test, pytest.mark.core_model]


def test_opaque_attention_codegen():
    """@spec PORT-PREC-021: no GEMM or reduction escapes into Inductor."""
    from vllm.models.nemotron_asr.bi_evidence import compile_with_evidence
    from vllm.models.nemotron_asr.encoder import RelPositionMHA

    attention = RelPositionMHA(d_model=8, n_heads=2).eval()
    for module in attention.modules():
        module.batch_invariant_mode = bi.BatchInvariantExecution(True)
    x, pos = torch.randn(2, 3, 8), torch.randn(1, 5, 8)
    mask = torch.zeros(2, 3, 3, dtype=torch.bool)

    def forward(x, pos, mask):
        return attention(x, pos_emb=pos, masked=mask)

    with torch.inference_mode():
        actual, evidence = compile_with_evidence(forward, (x, pos, mask))
        torch.testing.assert_close(actual, forward(x, pos, mask), rtol=0, atol=0)
    assert evidence.op_counts == {"mm": 5, "matmul": 3, "softmax_sum": 1}
    assert sum(evidence.dispatch_counts.values()) == 9


def op_cases(dtype=torch.float32):
    from vllm.models.nemotron_asr import bi_ops as ops

    a, b = torch.randn(3, 7).to(dtype), torch.randn(7, 5).to(dtype)
    bias = torch.randn(5).to(dtype)
    x = torch.randn(2, 3, 7)
    w1, b1 = torch.randn(3, 1, 3), torch.randn(3)
    image, w2, b2 = torch.randn(2, 4, 7, 9), torch.randn(6, 2, 3, 2), torch.randn(6)
    return [
        (ops.mm, (a, b), (a.float() @ b.float()).to(dtype)),
        (ops.addmm, (bias, a, b), (a.float() @ b.float() + bias.float()).to(dtype)),
        (
            ops.matmul,
            (a.reshape(1, 1, 3, 7).expand(2, 3, 3, 7), b.reshape(1, 1, 7, 5)),
            torch.matmul(
                a.float().reshape(1, 1, 3, 7).expand(2, 3, 3, 7),
                b.float().reshape(1, 1, 7, 5),
            ).to(dtype),
        ),
        (ops.softmax_sum, (x.transpose(1, 2), 1), x.softmax(-1).transpose(1, 2)),
        (
            ops.layer_norm_sum,
            (x, [3, 7], None, None, 1e-5),
            torch.nn.functional.layer_norm(x, (3, 7)),
        ),
        (
            ops.depthwise_conv1d_sum,
            (x, w1, b1, 2, 2, 2),
            torch.nn.functional.conv1d(
                x, w1, b1, stride=2, padding=2, dilation=2, groups=3
            ),
        ),
        (
            ops.strided_conv2d_sum,
            (image, w2, b2, [2, 1], [1, 2], [2, 1], 2),
            torch.nn.functional.conv2d(
                image, w2, b2, stride=(2, 1), padding=(1, 2), dilation=(2, 1), groups=2
            ),
        ),
    ]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fake_metadata_and_fp32_arithmetic(dtype):
    """@spec PORT-PREC-021: all op fakes match real shape, stride and dtype."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    torch.manual_seed(42)
    for op, args, expected in op_cases(dtype):
        actual = op(*args)
        if op in (bi.bi_ops.mm, bi.bi_ops.addmm, bi.bi_ops.matmul):
            assert torch.equal(actual, expected)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        with FakeTensorMode() as mode:
            fake_args = tuple(
                mode.from_tensor(arg) if isinstance(arg, torch.Tensor) else arg
                for arg in args
            )
            fake = op(*fake_args)
        assert (fake.shape, fake.dtype, fake.device, fake.stride()) == (
            actual.shape,
            actual.dtype,
            actual.device,
            actual.stride(),
        )


def test_bias_precedes_half_cast():
    """@spec PORT-PREC-021: rounding cannot discard bias cancellation."""
    a = torch.tensor([[1, 2**-11]], dtype=torch.float16)
    b = torch.ones(2, 1, dtype=a.dtype)
    bias = torch.tensor([-1], dtype=a.dtype)
    assert bi.invariant_gemm(a, b, bias).item() == 2**-11


def test_reduction_population_and_fp32_affine():
    """@spec PORT-PREC-020, PORT-PREC-021: fixed reduction trees ignore rows."""
    for op, args, _ in op_cases():
        if "sum" not in str(op):
            continue
        row = args[0][:1].clone()
        one = op(row, *args[1:])
        many = op(row.repeat(31, *([1] * (row.ndim - 1))), *args[1:])
        assert torch.equal(one[0], many[-1])
    x = torch.randn(2, 5, 7).half()
    weight, bias = torch.randn(7).half(), torch.randn(7).half()
    actual = bi.layer_norm_fp32(x, (7,), weight, bias)
    expected = torch.nn.functional.layer_norm(
        x.float(), (7,), weight.float(), bias.float()
    ).half()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    assert bi.softmax_fp32_sum(x[..., :0]).shape == (2, 5, 0)
    assert bi.softmax_fp32_sum(torch.full_like(x, -torch.inf)).isnan().all()


def test_mode_off_identity():
    """@spec PORT-PREC-019: bound off follows original modules byte for byte."""
    from types import SimpleNamespace

    from vllm.models.nemotron_asr.bi_ops import dispatch_counts
    from vllm.models.nemotron_asr.encoder import FeedForward
    from vllm.models.nemotron_asr.lid import PromptConditioner

    ff, lid = (
        FeedForward(d_model=8, d_ff=16),
        PromptConditioner(enc_hidden=8, num_prompts=3),
    )
    x = torch.randn(2, 4, 8)
    prompts = torch.tensor([0, 2])
    expected_ff = ff.linear2(torch.nn.functional.silu(ff.linear1(x)))
    hot = (prompts[:, None] == torch.arange(3)).float()[:, None, :].expand(2, 4, 3)
    expected_lid = lid.prompt_kernel(torch.cat((x, hot), -1))
    bi.bind_batch_invariant_mode(SimpleNamespace(encoder=ff, lid=lid), bi.MODE_OFF)
    with dispatch_counts() as counts:
        assert torch.equal(ff(x), expected_ff)
        assert torch.equal(lid(x, prompt_index=prompts), expected_lid)
    assert not counts


def test_full_transition_codegen():
    """@spec PORT-PREC-021: subsampling, every layer, caches and LID compile."""
    from tests.models.nemotron_asr import test_encoder_batch_invariance as probe
    from vllm.models.nemotron_asr.bi_evidence import compile_with_evidence

    device = torch.device("cpu")
    core = probe.build_core(
        "fp32", device, probe._TINY, mode=bi.BatchInvariantExecution(True)
    )
    schedule = probe.schedule_for(core)
    state = probe.new_state(
        schedule, population=2, composition="mixed", probe_row=1, device=device
    )
    args = probe.chunk_inputs(
        schedule, population=2, chunk=0, composition="mixed", probe_row=1, device=device
    )
    caches = probe.gathered(state)

    def transition(mel, offsets, lengths, prompt):
        return probe.execute_encoder_transition(
            core, mel, caches, offsets, lengths, schedule.out_width, prompt
        )

    with torch.inference_mode():
        _, evidence = compile_with_evidence(transition, args)
    # 3 layers: q/k/v/pos/out + 4 FF + 2 pointwise GEMMs each;
    # subsampling: 2 pointwise + out; LID: 2 biased linears.
    assert evidence.op_counts == {
        "mm": 33,
        "addmm": 5,
        "matmul": 9,
        "softmax_sum": 3,
        "layer_norm_sum": 18,
        "depthwise_conv1d_sum": 3,
        "strided_conv2d_sum": 3,
    }


def test_evidence_rejects_unprotected_matmul():
    """@spec PORT-PREC-021: the former raw-matmul route is a hard failure."""
    from vllm.models.nemotron_asr.bi_evidence import (
        assert_opaque_code,
        assert_opaque_graph,
    )

    graph = torch.fx.Graph()
    a, b = graph.placeholder("a"), graph.placeholder("b")
    graph.call_function(torch.ops.aten.bmm.default, (a, b))
    with pytest.raises(AssertionError, match="unprotected"):
        assert_opaque_graph(graph)
    with pytest.raises(AssertionError, match="unprotected"):
        assert_opaque_code(
            "torch.ops.nemotron_bi.mm.default(a,b)\nextern_kernels.mm(a,b)"
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_bound_transition_avoids_mode_record_lookup(monkeypatch, enabled):
    """@spec PORT-PREC-019, PORT-PREC-027: routing is fixed before tracing."""
    from tests.models.nemotron_asr import test_encoder_batch_invariance as probe

    device = torch.device("cpu")
    core = probe.build_core(
        "fp32", device, probe._TINY, mode=bi.BatchInvariantExecution(enabled)
    )
    schedule = probe.schedule_for(core)
    state = probe.new_state(
        schedule, population=2, composition="mixed", probe_row=1, device=device
    )
    args = probe.chunk_inputs(
        schedule, population=2, chunk=0, composition="mixed", probe_row=1, device=device
    )
    caches = probe.gathered(state)
    graphs = []

    def reject_lookup(owner):
        raise AssertionError("bound transition must not read the execution record")

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    def transition(mel, offsets, lengths, prompt):
        return probe.execute_encoder_transition(
            core, mel, caches, offsets, lengths, schedule.out_width, prompt
        )

    with torch.inference_mode():
        core.encoder.prepare_stream_relative_position_projections(
            out_widths=(schedule.out_width,),
            cache_len=56,
            reference=next(core.encoder.parameters()),
        )
        monkeypatch.setattr(bi, "execution_mode", reject_lookup)
        compiled = torch.compile(
            transition, backend=backend, fullgraph=True, dynamic=False
        )
        compiled(*args)
        compiled(*args)
    assert len(graphs) == 1
