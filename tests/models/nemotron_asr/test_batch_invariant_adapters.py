# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU schema and arithmetic oracles; these do not qualify CUDA kernels."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.nemotron_asr import batch_invariance as port

pytestmark = [pytest.mark.cpu_test, pytest.mark.core_model]


def kernels():
    # Independent CPU references, not replacement production kernels.
    return SimpleNamespace(
        mm_batch_invariant=lambda a, b: torch.mm(a, b),
        addmm_batch_invariant=lambda bias, a, b: torch.addmm(bias, a, b),
        bmm_batch_invariant=lambda a, b: torch.bmm(a, b),
        matmul_batch_invariant=lambda a, b: torch.matmul(a, b),
        linear_batch_invariant=lambda x, w, bias: torch.nn.functional.linear(
            x, w, bias
        ),
        softmax_batch_invariant=lambda x, dim: torch.softmax(x, dim),
        _log_softmax_batch_invariant=lambda x, dim, half: torch.log_softmax(
            x.float() if half else x, dim
        ),
        mean_batch_invariant=lambda x, dim, keepdim, dtype: x.mean(
            tuple(dim), keepdim, dtype=dtype
        ),
    )


@pytest.mark.parametrize("beta,alpha", [(1, 1), (0, 1), (2, -0.5), (-1, 0), (0, 0)])
@pytest.mark.parametrize("bias_shape", [(), (3,), (2, 3)])
def test_prec021_addmm_scalars(beta, alpha, bias_shape):
    """@spec PORT-PREC-021: scaling and broadcasting happen in FP32."""
    op = port._schema_wrappers(kernels())["addmm"]
    a = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 4
    b = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 2
    bias = torch.full(bias_shape, float("nan") if beta == 0 else 0.5)
    expected = torch.addmm(bias, a, b, beta=beta, alpha=alpha)
    out = torch.empty_like(expected)
    assert torch.equal(op(bias, a, b, beta=beta, alpha=alpha), expected)
    assert op(bias, a, b, beta=beta, alpha=alpha, out=out) is out
    assert torch.equal(out, expected)


def test_prec021_addmm_bias_before_cast():
    """@spec PORT-PREC-021: preserve vLLM's fused path for unit scaling."""
    source = kernels()
    calls = []

    def fused(bias, a, b):
        calls.append(True)
        return torch.addmm(bias.float(), a.float(), b.float()).to(a.dtype)

    source.addmm_batch_invariant = fused
    op = port._schema_wrappers(source)["addmm"]
    a = torch.tensor([[1, 2**-11]], dtype=torch.float16)
    b = torch.ones(2, 1, dtype=torch.float16)
    bias = torch.tensor([-1], dtype=torch.float16)
    assert op(bias, a, b).item() == 2**-11
    assert calls == [True]
    assert op(bias, a, b, beta=2, alpha=2).item() == 2**-10


@pytest.mark.parametrize("name", ["mm", "bmm"])
def test_prec021_positional_dtype_and_out(name):
    """@spec PORT-PREC-021, PORT-PREC-027: accept dtype and out call shapes."""
    op = port._schema_wrappers(kernels())[name]
    a = torch.ones((2, 3) if name == "mm" else (1, 2, 3), dtype=torch.float16)
    b = torch.ones((3, 4) if name == "mm" else (1, 3, 4), dtype=torch.float16)
    expected = torch.full((2, 4) if name == "mm" else (1, 2, 4), 3.0)
    out = torch.empty_like(expected)
    assert torch.equal(op(a, b, torch.float32), expected)
    assert op(a, b, torch.float32, out=out) is out
    assert torch.equal(out, expected)
    with pytest.raises((ValueError, TypeError)):
        op(a, b, torch.float64)


def test_prec021_reduction_schema_arguments():
    """@spec PORT-PREC-021: half_to_float, dtype, and optional dim are honored."""
    ops = port._schema_wrappers(kernels())
    x = torch.tensor([[1, 2]], dtype=torch.float16)
    for name in ("_softmax", "_log_softmax"):
        result = ops[name](x, dim=-1, half_to_float=True)
        assert result.dtype == torch.float32
        out = torch.empty_like(result)
        assert ops[name](x, -1, True, out=out) is out
        assert torch.equal(result, out)
    assert ops["softmax.int"](x, -1, dtype=torch.float32).dtype == torch.float32
    assert ops["mean.dim"](x, None).dtype == torch.float16
    assert ops["mean.dim"](x, [], dtype=torch.float32).item() == 1.5


def test_prec018_schema_installation_lifetime(monkeypatch):
    """@spec PORT-PREC-018, PORT-PREC-022, PORT-PREC-028: ordered and reversible."""
    from tests.models.nemotron_asr.conftest import preserve_batch_invariance

    source = kernels()
    source.mm_batch_invariant = lambda a, b: (a.unsqueeze(-1) * b.unsqueeze(0)).sum(1)
    source.addmm_batch_invariant = (
        lambda bias, a, b: source.mm_batch_invariant(a, b) + bias
    )
    source.bmm_batch_invariant = lambda a, b: (a.unsqueeze(-1) * b.unsqueeze(1)).sum(2)
    source._batch_invariant_MODE = False
    source._batch_invariant_LIB = None
    source._fp16_block_size_n, source._fp32_block_size_n, source._fp32_num_stages = (
        256,
        128,
        3,
    )
    configs = SimpleNamespace(
        _TUNED_MATMUL_CONFIGS_FOR_DEVICE=None, _TUNED_MATMUL_CONFIGS_RESOLVED=False
    )
    before = port._schema_library
    original_bmm = torch.bmm
    monkeypatch.setattr(port, "_schema_library", None)
    monkeypatch.setattr(port, "_schema_source", None)
    monkeypatch.setattr(port, "_schema_installed", ())
    with preserve_batch_invariance(source, configs):
        events = []

        def initialize():
            events.append("vllm initialized")
            source._batch_invariant_MODE = True
            source._batch_invariant_LIB = torch.library.Library("aten", "IMPL")
            source._batch_invariant_LIB.impl(
                "mm", source.mm_batch_invariant, "CPU", allow_override=True
            )
            source._batch_invariant_LIB.impl(
                "addmm", source.addmm_batch_invariant, "CPU", allow_override=True
            )
            source._batch_invariant_LIB.impl(
                "bmm", source.bmm_batch_invariant, "CPU", allow_override=True
            )

        off = port.installed_batch_invariant_mode(
            module=source, initialize=lambda: None
        )
        assert off.schema_adapters == () and port._schema_library is None
        record = port.installed_batch_invariant_mode(
            module=source, initialize=initialize
        )
        assert events == ["vllm initialized"]
        assert record.enabled and "aten::addmm/CPU" in record.schema_adapters
        assert "aten::bmm.out/CPU" in record.schema_adapters
        library = port._schema_library
        assert port._install_schema_adapters(source) == record.schema_adapters
        assert port._schema_library is library
        a, b, bias = torch.ones(2, 3), torch.ones(3, 4), torch.ones(4)
        assert torch.equal(
            torch.ops.aten.addmm(a.new_ones(4), a, b, beta=2, alpha=2),
            a.new_full((2, 4), 8),
        )
        out = torch.empty(2, 4)
        assert torch.ops.aten.addmm.out(bias, a, b, out=out) is out
        assert torch.equal(out, a.new_full((2, 4), 4))
        a3, b3 = a[None], b[None]
        assert torch.equal(torch.ops.aten.bmm(a3, b3), a.new_full((1, 2, 4), 3))
        # Exercise actual dtype overload dispatch where this Torch exposes it.
        if "dtype" in torch.ops.aten.bmm._schemas:
            assert (
                torch.ops.aten.bmm.dtype(a3.half(), b3.half(), torch.float32).dtype
                == torch.float32
            )
        assert torch.bmm(a3.half(), b3.half(), torch.float32).dtype == torch.float32
    assert port._schema_library is None
    assert torch.bmm is original_bmm
    assert torch.equal(
        torch.mm(torch.ones(1, 2), torch.ones(2, 1)), torch.tensor([[2.0]])
    )
    monkeypatch.setattr(port, "_schema_library", before)


def test_prec022_schema_disclosure_is_additive():
    """@spec PORT-PREC-022: freeze the added effects independently of the code."""
    assert set(port.SCHEMA_ADAPTER_SIDE_EFFECTS) == {
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
    }


def test_prec027_cpu_inductor_encoder_transition(tmp_path, monkeypatch):
    """@spec PORT-PREC-021, PORT-PREC-027: fullgraph CPU transition smoke only."""
    from tests.models.nemotron_asr import test_encoder_batch_invariance as probe

    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    torch._dynamo.reset()
    core = probe.build_core(
        "fp32",
        torch.device("cpu"),
        probe._TINY,
        mode=port.BatchInvariantExecution(True),
    )
    sched = probe.schedule_for(core)
    state = probe.new_state(
        sched,
        population=1,
        composition="repeated",
        probe_row=0,
        device=torch.device("cpu"),
    )
    reference = SimpleNamespace(
        channel=[x.clone() for x in state.channel],
        time=[x.clone() for x in state.time],
        window_valid=[x.clone() for x in state.window_valid],
    )
    inputs = probe.chunk_inputs(
        sched,
        population=1,
        chunk=0,
        composition="repeated",
        probe_row=0,
        device=torch.device("cpu"),
    )
    caches = probe.gathered(state)

    def transition(mel, offsets, lengths, prompt):
        return probe.execute_encoder_transition(
            core, mel, caches, offsets, lengths, sched.out_width, prompt
        )

    compiled = torch.compile(transition, backend="inductor", fullgraph=True)
    try:
        with torch.inference_mode():
            expected = probe.EagerRunner(core, reference, sched.out_width).step(*inputs)
            actual = compiled(*inputs)
        for got, want in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, want)
        for got, want in zip(
            probe._state_tensors(state), probe._state_tensors(reference), strict=True
        ):
            torch.testing.assert_close(got, want)
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("enabled", [False, True])
def test_prec019_probe_enforces_only_mode_on(monkeypatch, capsys, enabled):
    """@spec PORT-PREC-019, PORT-PREC-020: mode-off divergence is diagnostic."""
    from tests.models.nemotron_asr import test_encoder_batch_invariance as probe

    monkeypatch.setattr(probe, "_EXECUTION_MODE", port.BatchInvariantExecution(enabled))
    monkeypatch.setattr(probe, "_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(probe, "build_core", lambda *args: object())
    monkeypatch.setattr(
        probe,
        "invariance_cases",
        lambda *args, **kwargs: [SimpleNamespace(equal=False)],
    )
    monkeypatch.setattr(probe, "format_cases", lambda cases: "recorded divergence")
    if enabled:
        with pytest.raises(AssertionError, match="recorded divergence"):
            probe.test_probe_row_is_bitwise_batch_invariant(
                "vllm", None, "fp32", "graph", "mixed"
            )
    else:
        probe.test_probe_row_is_bitwise_batch_invariant(
            "none", None, "fp32", "graph", "mixed"
        )
    assert "recorded divergence" in capsys.readouterr().out
