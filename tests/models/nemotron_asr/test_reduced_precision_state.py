# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``reduced_precision_state`` never raises, whichever torch API set the flags."""

import pytest
import torch

from vllm.models.nemotron_asr.precision import reduced_precision_state

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]


@pytest.fixture(autouse=True)
def _restore_torch_precision():
    matmul = torch.backends.cuda.matmul
    cudnn = torch.backends.cudnn
    owners = (matmul, cudnn.conv, cudnn.rnn)
    saved = [(o, o.fp32_precision) for o in owners]
    legacy = (
        matmul.allow_fp16_reduced_precision_reduction,
        matmul.allow_bf16_reduced_precision_reduction,
    )
    mode = torch.get_float32_matmul_precision()
    try:
        yield
    finally:
        for owner, value in saved:
            owner.fp32_precision = value
        (
            matmul.allow_fp16_reduced_precision_reduction,
            matmul.allow_bf16_reduced_precision_reduction,
        ) = legacy
        torch.set_float32_matmul_precision(mode)


def test_new_api_ieee_reports_tf32_off() -> None:
    for owner in (
        torch.backends.cuda.matmul,
        torch.backends.cudnn.conv,
        torch.backends.cudnn.rnn,
    ):
        owner.fp32_precision = "ieee"
    state = reduced_precision_state()
    assert state["tf32_matmul_enabled"] is False
    assert state["tf32_cudnn_enabled"] is False
    assert state["matmul_fp32_precision"] == "ieee"


def test_mixed_conv_rnn_does_not_raise() -> None:
    """The legacy aggregate cuDNN read raises here; the helper must not."""
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.rnn.fp32_precision = "tf32"
    with pytest.raises(RuntimeError):
        _ = torch.backends.cudnn.allow_tf32
    state = reduced_precision_state()
    assert state["tf32_cudnn_conv_enabled"] is False
    assert state["tf32_cudnn_rnn_enabled"] is True
    assert state["tf32_cudnn_enabled"] is None


def test_new_api_tf32_reports_tf32_on() -> None:
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    state = reduced_precision_state()
    assert state["tf32_matmul_enabled"] is True


def test_legacy_api_is_reflected() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state = reduced_precision_state()
    assert state["tf32_matmul_enabled"] is False
    assert state["tf32_cudnn_enabled"] is False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    state = reduced_precision_state()
    assert state["tf32_matmul_enabled"] is True
    assert state["tf32_cudnn_enabled"] is True


def test_reduced_precision_reduction_and_matmul_mode() -> None:
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("high")
    state = reduced_precision_state()
    assert state["allow_bf16_reduced_precision_reduction"] is False
    assert state["float32_matmul_precision"] == "high"


def test_read_failure_yields_none_not_exception(monkeypatch) -> None:
    def boom(self: object) -> object:
        raise RuntimeError("mixed legacy and new APIs")

    class Boom:
        fp32_precision = property(boom)
        allow_tf32 = property(boom)

    monkeypatch.setattr(torch.backends.cuda, "matmul", Boom())
    state = reduced_precision_state()
    assert state["tf32_matmul_enabled"] is None
    assert state["matmul_fp32_precision"] is None
