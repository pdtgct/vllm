# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU installation-witness oracles; these do not qualify CUDA kernels.

Schema and arithmetic tests for the adapters themselves live with vLLM's
determinism tests (tests/v1/determinism/test_schema_adapters.py).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.determinism import schema_adapters
from vllm.models.nemotron_asr import batch_invariance as port

pytestmark = [pytest.mark.cpu_test, pytest.mark.core_model]


def test_prec018_schema_installation_witness():
    """@spec PORT-PREC-018, PORT-PREC-022, PORT-PREC-028: vLLM installs, model reads."""
    from tests.models.nemotron_asr.conftest import preserve_batch_invariance

    source = SimpleNamespace(
        mm_batch_invariant=lambda a, b: (a.unsqueeze(-1) * b.unsqueeze(0)).sum(1),
        bmm_batch_invariant=lambda a, b: (a.unsqueeze(-1) * b.unsqueeze(1)).sum(2),
    )
    source.addmm_batch_invariant = (
        lambda bias, a, b: source.mm_batch_invariant(a, b) + bias
    )
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
    original_bmm, before = torch.bmm, schema_adapters._library
    with preserve_batch_invariance(source, configs):
        schema_adapters._library, schema_adapters._source = None, None
        schema_adapters._installed, schema_adapters._failure = (), None
        events = []

        def initialize():
            # Mirrors vLLM's initializer: register kernels, then adapters.
            events.append("vllm initialized")
            source._batch_invariant_MODE = True
            source._batch_invariant_LIB = torch.library.Library("aten", "IMPL")
            for op in ("mm", "addmm", "bmm"):
                source._batch_invariant_LIB.impl(
                    op,
                    getattr(source, f"{op}_batch_invariant"),
                    "CPU",
                    allow_override=True,
                )
            schema_adapters.install_schema_adapters(source)

        off = port.installed_batch_invariant_mode(
            module=source, initialize=lambda: None
        )
        assert off.schema_adapters == () and schema_adapters._library is None
        # A real library without completed adapters is not an installed mode.
        source._batch_invariant_MODE = True
        source._batch_invariant_LIB = torch.library.Library("aten", "IMPL")
        with pytest.raises(RuntimeError, match="incomplete"):
            port.installed_batch_invariant_mode(module=source, initialize=lambda: None)
        source._batch_invariant_LIB._destroy()
        source._batch_invariant_MODE = False
        record = port.installed_batch_invariant_mode(
            module=source, initialize=initialize
        )
        assert events == ["vllm initialized"]
        assert record.enabled and record.schema_adapters == schema_adapters._installed
        assert "aten::addmm/CPU" in record.schema_adapters
        assert "aten::bmm.out/CPU" in record.schema_adapters
        # Overloads beyond vLLM's own registrations are all disclosed.
        registered = {
            "aten::" + entry.split("/")[1]
            for entry in source._batch_invariant_LIB._op_impls
        }
        overloads: tuple[str, ...] = record.schema_adapters
        adapted = {item.split("/")[0] for item in overloads}
        added = adapted - registered
        assert added and added <= set(port.SCHEMA_ADAPTER_SIDE_EFFECTS)
        # Snapshotting again reads the same installation; it never reinstalls.
        library = schema_adapters._library
        again = port.installed_batch_invariant_mode(
            module=source, initialize=lambda: None
        )
        assert again == record and schema_adapters._library is library
        a, b = torch.ones(2, 3), torch.ones(3, 4)
        a3, b3 = a[None], b[None]
        if "dtype" in torch.ops.aten.bmm._schemas:
            assert torch.bmm(a3.half(), b3.half(), torch.float32).dtype == torch.float32
        assert torch.bmm is original_bmm
    assert schema_adapters._library is before
    assert torch.bmm is original_bmm
    assert torch.equal(
        torch.mm(torch.ones(1, 2), torch.ones(2, 1)), torch.tensor([[2.0]])
    )


def set_requested(monkeypatch, enabled):
    # vllm.envs may hold a concrete attribute left by an earlier monkeypatch,
    # which would mask the environment; set both, as core determinism tests do.
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", enabled)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1" if enabled else "0")


@pytest.mark.parametrize("enabled", [False, True])
def test_prec018_production_witness_fails_closed(monkeypatch, enabled):
    """@spec PORT-PREC-018: a requested mode vLLM did not install is an error."""
    from vllm.model_executor.determinism import batch_invariant

    monkeypatch.setattr(batch_invariant, "_batch_invariant_MODE", False)
    monkeypatch.setattr(batch_invariant, "_batch_invariant_LIB", None)
    monkeypatch.setattr(batch_invariant, "_init_failure", None)
    set_requested(monkeypatch, enabled)
    if enabled:
        with pytest.raises(RuntimeError, match="did not initialize"):
            port.installed_batch_invariant_mode()
    else:
        assert port.installed_batch_invariant_mode() == port.MODE_OFF


def test_prec018_production_witness_raises_initializer_failure(monkeypatch):
    """@spec PORT-PREC-018: a failed native install is never a usable mode."""
    from vllm.model_executor.determinism import batch_invariant

    error = RuntimeError("registration failed")
    monkeypatch.setattr(batch_invariant, "_batch_invariant_MODE", True)
    monkeypatch.setattr(batch_invariant, "_batch_invariant_LIB", object())
    monkeypatch.setattr(batch_invariant, "_init_failure", error)
    set_requested(monkeypatch, True)
    with pytest.raises(RuntimeError) as caught:
        port.installed_batch_invariant_mode()
    assert caught.value is error


@pytest.mark.parametrize("outcome", ["complete", "failure"])
def test_prec018_observer_waits_for_native_install(monkeypatch, outcome):
    """@spec PORT-PREC-018: an in-progress install is awaited, never attested."""
    import threading

    from vllm.model_executor.determinism import batch_invariant

    for owner, names in (
        (batch_invariant, ("_batch_invariant_MODE", "_batch_invariant_LIB")),
        (batch_invariant, ("_init_failure",)),
        (schema_adapters, ("_library", "_source", "_installed", "_failure")),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, getattr(owner, name))
    batch_invariant._init_failure = None
    schema_adapters._library, schema_adapters._source = None, None
    schema_adapters._installed, schema_adapters._failure = (), None
    set_requested(monkeypatch, True)
    error = RuntimeError("registration failed")
    source = torch.library.Library("aten", "IMPL")
    adapters = torch.library.Library("aten", "IMPL")
    holding, finish = threading.Event(), threading.Event()
    results: dict = {}

    def initializer():
        # The native initializer's state while registration is in progress.
        with batch_invariant._init_lock:
            batch_invariant._batch_invariant_MODE = True
            batch_invariant._batch_invariant_LIB = source
            holding.set()
            assert finish.wait(10)
            if outcome == "failure":
                batch_invariant._init_failure = error
            else:
                schema_adapters._library, schema_adapters._source = adapters, source
                schema_adapters._installed = ("aten::bmm.out/CPU",)

    def observe():
        try:
            results["record"] = port.installed_batch_invariant_mode()
        except RuntimeError as caught:
            results["error"] = caught

    native = threading.Thread(target=initializer, daemon=True)
    native.start()
    try:
        assert holding.wait(10)
        observer = threading.Thread(target=observe, daemon=True)
        observer.start()
        observer.join(0.2)
        assert observer.is_alive(), "observer attested an in-progress install"
        finish.set()
        native.join(10)
        observer.join(10)
        assert not observer.is_alive()
        if outcome == "failure":
            assert results == {"error": error}
        else:
            assert results == {
                "record": port.BatchInvariantExecution(True, ("aten::bmm.out/CPU",))
            }
    finally:
        finish.set()
        native.join(10)
        adapters._destroy()
        source._destroy()


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
