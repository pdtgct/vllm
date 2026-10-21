# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the ATen schema adapters of batch-invariant mode.

The kernels below are independent CPU references registered on the CPU key,
so these tests exercise schema handling and installation, not the Triton
kernels themselves.
"""

import threading
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.determinism import batch_invariant as bi
from vllm.model_executor.determinism import schema_adapters as adapters

pytestmark = pytest.mark.cpu_test


def kernels():
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


@pytest.fixture
def clean_adapters(monkeypatch):
    """Give each test fresh adapter state and remove its registrations."""
    for name in ("_library", "_source", "_installed", "_failure"):
        monkeypatch.setattr(adapters, name, getattr(adapters, name))
    adapters._library, adapters._source = None, None
    adapters._installed, adapters._failure = (), None
    libraries: list[torch.library.Library] = []
    yield libraries
    if adapters._library is not None:
        adapters._library._destroy()
    for library in libraries:
        library._destroy()


def cpu_source(libraries, ops=("mm", "addmm", "bmm")):
    """A module-shaped source that registered order-explicit CPU kernels."""
    source = kernels()
    source.mm_batch_invariant = lambda a, b: (a.unsqueeze(-1) * b.unsqueeze(0)).sum(1)
    source.addmm_batch_invariant = (
        lambda bias, a, b: source.mm_batch_invariant(a, b) + bias
    )
    source.bmm_batch_invariant = lambda a, b: (a.unsqueeze(-1) * b.unsqueeze(1)).sum(2)
    source._batch_invariant_MODE = True
    source._batch_invariant_LIB = torch.library.Library("aten", "IMPL")
    libraries.append(source._batch_invariant_LIB)
    for op in ops:
        source._batch_invariant_LIB.impl(
            op, getattr(source, f"{op}_batch_invariant"), "CPU", allow_override=True
        )
    return source


@pytest.mark.parametrize("beta,alpha", [(1, 1), (0, 1), (2, -0.5), (-1, 0), (0, 0)])
@pytest.mark.parametrize("bias_shape", [(), (3,), (2, 3)])
def test_addmm_scalars(beta, alpha, bias_shape):
    """Scaling and broadcasting happen in FP32; beta=0 ignores the bias."""
    op = adapters._schema_wrappers(kernels())["addmm"]
    a = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 4
    b = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 2
    bias = torch.full(bias_shape, float("nan") if beta == 0 else 0.5)
    expected = torch.addmm(bias, a, b, beta=beta, alpha=alpha)
    out = torch.empty_like(expected)
    assert torch.equal(op(bias, a, b, beta=beta, alpha=alpha), expected)
    assert op(bias, a, b, beta=beta, alpha=alpha, out=out) is out
    assert torch.equal(out, expected)


def test_addmm_bias_before_cast():
    """Unit scaling keeps the fused kernel; other scaling widens first."""
    source = kernels()
    calls = []

    def fused(bias, a, b):
        calls.append(True)
        return torch.addmm(bias.float(), a.float(), b.float()).to(a.dtype)

    source.addmm_batch_invariant = fused
    op = adapters._schema_wrappers(source)["addmm"]
    a = torch.tensor([[1, 2**-11]], dtype=torch.float16)
    b = torch.ones(2, 1, dtype=torch.float16)
    bias = torch.tensor([-1], dtype=torch.float16)
    assert op(bias, a, b).item() == 2**-11
    assert calls == [True]
    assert op(bias, a, b, beta=2, alpha=2).item() == 2**-10


@pytest.mark.parametrize("name", ["mm", "bmm"])
def test_positional_dtype_and_out(name):
    """Accept the out_dtype and out call shapes; reject narrowing."""
    op = adapters._schema_wrappers(kernels())[name]
    a = torch.ones((2, 3) if name == "mm" else (1, 2, 3), dtype=torch.float16)
    b = torch.ones((3, 4) if name == "mm" else (1, 3, 4), dtype=torch.float16)
    expected = torch.full((2, 4) if name == "mm" else (1, 2, 4), 3.0)
    out = torch.empty_like(expected)
    assert torch.equal(op(a, b, torch.float32), expected)
    assert op(a, b, torch.float32, out=out) is out
    assert torch.equal(out, expected)
    with pytest.raises((ValueError, TypeError)):
        op(a, b, torch.float64)


def test_reduction_schema_arguments():
    """half_to_float, dtype, and an optional dim are honored."""
    ops = adapters._schema_wrappers(kernels())
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


def test_mode_off_installs_nothing(clean_adapters):
    source = kernels()
    source._batch_invariant_MODE, source._batch_invariant_LIB = False, None
    assert adapters.install_schema_adapters(source) == ()
    assert adapters._library is None


def test_install_dispatch_and_lifetime(clean_adapters):
    """Overloads dispatch through the keyset shim; install is idempotent."""
    original_bmm = torch.bmm
    source = cpu_source(clean_adapters)
    installed = adapters.install_schema_adapters(source)
    assert {"aten::addmm/CPU", "aten::bmm.out/CPU", "aten::mm.out/CPU"} <= set(
        installed
    )
    assert adapters.installed_schema_adapters(source) == installed
    library = adapters._library
    assert adapters.install_schema_adapters(source) == installed
    assert adapters._library is library

    a, b, bias = torch.ones(2, 3), torch.ones(3, 4), torch.ones(4)
    assert torch.equal(
        torch.ops.aten.addmm(a.new_ones(4), a, b, beta=2, alpha=2),
        a.new_full((2, 4), 8),
    )
    out = torch.empty(2, 4)
    assert torch.ops.aten.addmm.out(bias, a, b, out=out) is out
    assert torch.equal(out, a.new_full((2, 4), 4))
    a3, b3 = a[None], b[None]
    assert torch.equal(torch.bmm(a3, b3), a.new_full((1, 2, 4), 3))
    if "dtype" in torch.ops.aten.bmm._schemas:
        assert torch.bmm(a3.half(), b3.half(), torch.float32).dtype == torch.float32
    # The public entry point is never replaced.
    assert torch.bmm is original_bmm


def test_only_registered_kernels_are_adapted(clean_adapters):
    """A platform that did not override a GEMM keeps the stock kernel."""
    source = cpu_source(clean_adapters, ops=("bmm",))
    installed = adapters.install_schema_adapters(source)
    assert installed and all(item.startswith("aten::bmm") for item in installed)


def test_source_change_is_rejected(clean_adapters):
    adapters.install_schema_adapters(cpu_source(clean_adapters))
    other = cpu_source(clean_adapters)
    with pytest.raises(RuntimeError, match="source changed"):
        adapters.install_schema_adapters(other)
    with pytest.raises(RuntimeError, match="incomplete"):
        adapters.installed_schema_adapters(other)


def test_partial_install_failure_is_sticky(clean_adapters, monkeypatch):
    source = cpu_source(clean_adapters)
    error = RuntimeError("registration failed")
    impl = torch.library.Library.impl
    calls = []

    def failing(self, *args, **kwargs):
        calls.append(args[0])
        if len(calls) == 3:
            raise error
        return impl(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.library.Library, "impl", failing)
        with pytest.raises(RuntimeError) as caught:
            adapters.install_schema_adapters(source)
    assert caught.value is error
    assert adapters._library is None
    for call in (adapters.install_schema_adapters, adapters.installed_schema_adapters):
        with pytest.raises(RuntimeError) as caught:
            call(source)
        assert caught.value is error
    # The partial adapter library was removed; only the source kernel remains.
    assert torch.equal(
        torch.ops.aten.addmm(torch.ones(4), torch.ones(2, 3), torch.ones(3, 4)),
        torch.full((2, 4), 4.0),
    )


def test_native_initializer_failure_is_sticky(monkeypatch):
    """A failed enable cannot later return as if the mode were installed."""
    for name in ("_batch_invariant_MODE", "_batch_invariant_LIB", "_init_failure"):
        monkeypatch.setattr(bi, name, getattr(bi, name))
    matmul = torch.backends.cuda.matmul
    for name in (
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
    ):
        monkeypatch.setattr(matmul, name, getattr(matmul, name))
    bi._batch_invariant_MODE, bi._batch_invariant_LIB = False, None
    bi._init_failure = None
    error = RuntimeError("adapter install failed")

    def fail(module):
        raise error

    monkeypatch.setattr(adapters, "install_schema_adapters", fail)
    created = []
    real_library = torch.library.Library

    def library(*args):
        created.append(real_library(*args))
        return created[-1]

    monkeypatch.setattr(torch.library, "Library", library)
    try:
        for _ in range(2):
            with pytest.raises(RuntimeError) as caught:
                bi.enable_batch_invariant_mode()
            assert caught.value is error
        assert len(created) == 1
    finally:
        for lib in created:
            lib._destroy()


_WAIT = 10  # seconds; bounds every wait so a regression fails instead of hanging


@pytest.fixture
def cpu_native(monkeypatch, clean_adapters):
    """Run the real native initializer against the CPU key, then undo it."""
    monkeypatch.setattr(
        bi,
        "current_platform",
        SimpleNamespace(
            dispatch_key="CPU",
            is_cuda_alike=lambda: False,
            is_xpu=lambda: False,
            is_cuda=lambda: False,
        ),
    )
    for name in ("_batch_invariant_MODE", "_batch_invariant_LIB", "_init_failure"):
        monkeypatch.setattr(bi, name, getattr(bi, name))
    matmul = torch.backends.cuda.matmul
    for name in (
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
    ):
        monkeypatch.setattr(matmul, name, getattr(matmul, name))
    bi._batch_invariant_MODE, bi._batch_invariant_LIB = False, None
    bi._init_failure = None
    yield
    if bi._batch_invariant_LIB is not None:
        bi._batch_invariant_LIB._destroy()


def pause_first_impl(monkeypatch, thread_name, error=None):
    """Block ``thread_name`` at its first Library.impl; optionally fail there."""
    entered, release = threading.Event(), threading.Event()
    impl = torch.library.Library.impl
    paused: list[bool] = []

    def gated(self, *args, **kwargs):
        if not paused and threading.current_thread().name == thread_name:
            paused.append(True)
            entered.set()
            assert release.wait(_WAIT)
            if error is not None:
                raise error
        return impl(self, *args, **kwargs)

    monkeypatch.setattr(torch.library.Library, "impl", gated)
    return entered, release


def run(name, target, results):
    def body():
        try:
            results[name] = ("ok", target())
        except BaseException as error:  # noqa: BLE001 - recorded for the assertion
            results[name] = ("error", error)

    thread = threading.Thread(target=body, name=name, daemon=True)
    thread.start()
    return thread


def test_concurrent_enable_waits_for_complete_install(monkeypatch, cpu_native):
    """No caller reports success, or sees evidence, before installation completes."""
    entered, release = pause_first_impl(monkeypatch, "first")
    results: dict = {}
    first = run("first", bi.enable_batch_invariant_mode, results)
    assert entered.wait(_WAIT)
    second = run("second", bi.enable_batch_invariant_mode, results)
    second.join(0.2)
    assert second.is_alive(), "second caller returned during registration"
    with pytest.raises(RuntimeError, match="incomplete"):
        adapters.installed_schema_adapters(bi)
    release.set()
    first.join(_WAIT)
    second.join(_WAIT)
    assert not first.is_alive() and not second.is_alive()
    assert results == {"first": ("ok", None), "second": ("ok", None)}
    library = bi._batch_invariant_LIB
    assert library is not None
    registered = {entry.split("/")[1] for entry in library._op_impls}
    assert {"bmm", "bmm.out", "_softmax", "mean.dim"} <= registered
    installed = adapters.installed_schema_adapters(bi)
    assert {"aten::bmm.out/CPU", "aten::_softmax.out/CPU", "aten::mean.out/CPU"} <= (
        set(installed)
    )


def test_concurrent_enable_failure_reaches_every_waiter(monkeypatch, cpu_native):
    """A waiter receives the sticky failure, never an early success."""
    error = RuntimeError("registration failed")
    entered, release = pause_first_impl(monkeypatch, "first", error)
    results: dict = {}
    first = run("first", bi.enable_batch_invariant_mode, results)
    assert entered.wait(_WAIT)
    second = run("second", bi.enable_batch_invariant_mode, results)
    second.join(0.2)
    assert second.is_alive(), "second caller returned during registration"
    release.set()
    first.join(_WAIT)
    second.join(_WAIT)
    assert results == {"first": ("error", error), "second": ("error", error)}
    with pytest.raises(RuntimeError) as caught:
        bi.enable_batch_invariant_mode()
    assert caught.value is error
    with pytest.raises(RuntimeError, match="incomplete"):
        adapters.installed_schema_adapters(bi)


def test_concurrent_adapter_install_is_single(monkeypatch, clean_adapters):
    """Direct adapter installs serialize; one library, one published tuple."""
    source = cpu_source(clean_adapters)
    entered, release = pause_first_impl(monkeypatch, "first")
    created = []
    real_library = torch.library.Library

    def library(*args):
        created.append(real_library(*args))
        return created[-1]

    monkeypatch.setattr(torch.library, "Library", library)
    results: dict = {}

    def install():
        return adapters.install_schema_adapters(source)

    first = run("first", install, results)
    assert entered.wait(_WAIT)
    second = run("second", install, results)
    observer = run(
        "observer", lambda: adapters.installed_schema_adapters(source), results
    )
    second.join(0.2)
    assert second.is_alive(), "second install returned during the first"
    assert observer.is_alive(), "evidence was attested during the install"
    release.set()
    for thread in (first, second, observer):
        thread.join(_WAIT)
    assert results["first"][0] == "ok" and results["first"] == results["second"]
    assert results["observer"] == results["first"]
    assert results["first"][1] == adapters.installed_schema_adapters(source)
    assert len(created) == 1
