# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Predictor LSTM: manual cell (default) vs fused torch.nn.LSTM variant.

Specs: PORT-PREC-004 (manual cell from primitives is the default and
keeps ``(h, c)`` under the PrecisionPolicy; a weight-shared fused
``torch.nn.LSTM`` is the optional fp32-CUDA variant; an L1 test holds
the two to tensor-tolerance equivalence), PORT-PREC-005 (state dtype is
policy-controlled, defaults fp32).
"""

import pytest
import torch

from vllm.models.nemotron_asr.rnnt_cell import (
    ManualLSTM,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]

_LAYERS = 2
_HIDDEN = 640
_INPUT = 640


def _fused_reference(manual: ManualLSTM) -> torch.nn.LSTM:
    """A torch.nn.LSTM sharing the manual cell's weights."""
    fused = torch.nn.LSTM(
        input_size=_INPUT,
        hidden_size=_HIDDEN,
        num_layers=_LAYERS,
        batch_first=True,
    )
    with torch.no_grad():
        for name, param in fused.named_parameters():
            param.copy_(manual.get_parameter(name))
    return fused


@pytest.fixture
def manual() -> ManualLSTM:
    torch.manual_seed(1234)
    cell = ManualLSTM(input_size=_INPUT, hidden_size=_HIDDEN, num_layers=_LAYERS)
    for param in cell.parameters():
        torch.nn.init.uniform_(param, -0.1, 0.1)
    return cell


def test_weight_names_match_torch_lstm(manual):
    # Weight layout mirrors torch.nn.LSTM so NeMo/HF predictor weights
    # load into either implementation unchanged.
    expected = {
        f"{kind}_{gate}_l{layer}"
        for kind in ("weight", "bias")
        for gate in ("ih", "hh")
        for layer in range(_LAYERS)
    }
    assert {n for n, _ in manual.named_parameters()} == expected


def test_manual_equals_fused_over_sequence(manual):
    # PORT-PREC-004's equivalence bar: stepwise manual == fused within
    # tensor tolerance over a multi-step sequence, states included.
    fused = _fused_reference(manual)
    torch.manual_seed(7)
    seq = torch.randn(3, 17, _INPUT)

    h = torch.zeros(_LAYERS, 3, _HIDDEN)
    c = torch.zeros(_LAYERS, 3, _HIDDEN)
    outs = []
    for t in range(seq.shape[1]):
        y, (h, c) = manual.step(seq[:, t], (h, c))
        outs.append(y)
    manual_out = torch.stack(outs, dim=1)

    fused_out, (fh, fc) = fused(seq)
    torch.testing.assert_close(manual_out, fused_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(h, fh, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(c, fc, atol=1e-5, rtol=1e-5)


def test_state_dtype_is_policy_controlled(manual):
    # (h, c) may be held fp32 while inputs arrive in a lower compute
    # dtype — the recurrent-state axis is independent (PORT-PREC-005).
    x = torch.randn(2, _INPUT, dtype=torch.bfloat16)
    h = torch.zeros(_LAYERS, 2, _HIDDEN, dtype=torch.float32)
    c = torch.zeros(_LAYERS, 2, _HIDDEN, dtype=torch.float32)
    y, (h2, c2) = manual.step(x, (h, c))
    assert h2.dtype == torch.float32
    assert c2.dtype == torch.float32
