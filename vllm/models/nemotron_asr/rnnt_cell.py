# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RNN-T predictor LSTM: manual cell built from primitives.

The manual cell is the default implementation — it keeps the ``(h, c)``
recurrent state under the model's ``PrecisionPolicy`` (state dtype
independent of compute dtype, defaulting fp32) and is backend-portable.
Its parameters use ``torch.nn.LSTM`` names (``weight_ih_l{k}``, …) so
checkpoint weights load into either implementation unchanged and a
weight-shared fused ``torch.nn.LSTM`` remains selectable as the
fp32-CUDA performance variant.
"""

import torch
from torch import nn


class ManualLSTM(nn.Module):
    """Multi-layer LSTM advanced one step at a time.

    Gate order matches ``torch.nn.LSTM``: input, forget, cell, output.
    ``step`` computes gates in the state dtype (fp32 by default) while
    accepting inputs in any compute dtype, so recurrent accumulation
    never happens below the policy's state precision.
    """

    def __init__(self, *, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        for layer in range(num_layers):
            in_features = input_size if layer == 0 else hidden_size
            self.register_parameter(
                f"weight_ih_l{layer}",
                nn.Parameter(torch.empty(4 * hidden_size, in_features)),
            )
            self.register_parameter(
                f"weight_hh_l{layer}",
                nn.Parameter(torch.empty(4 * hidden_size, hidden_size)),
            )
            self.register_parameter(
                f"bias_ih_l{layer}",
                nn.Parameter(torch.empty(4 * hidden_size)),
            )
            self.register_parameter(
                f"bias_hh_l{layer}",
                nn.Parameter(torch.empty(4 * hidden_size)),
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """``nn.LSTM``'s init: U(-1/sqrt(hidden), 1/sqrt(hidden)).

        ``torch.empty`` alone left UNINITIALIZED memory: the loaded
        checkpoint overwrites every parameter in production, but
        random-fixture tests decoded through allocator garbage —
        run-to-run NaN/tie flapping (the 2026-07-17 pod flappers).
        """
        stdv = 1.0 / (self.hidden_size**0.5)
        for param in self.parameters():
            nn.init.uniform_(param, -stdv, stdv)

    def step(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Advance one timestep.

        Args:
            x: ``(batch, input_size)`` input, any compute dtype.
            state: ``(h, c)`` each ``(num_layers, batch, hidden)``; their
                dtype is the recurrent-state dtype and is preserved.

        Returns:
            Top-layer output ``(batch, hidden)`` in the state dtype, and
            the advanced ``(h, c)``.

        """
        h, c = state
        state_dtype = h.dtype
        layer_input = x.to(state_dtype)
        next_h = []
        next_c = []
        hidden = self.hidden_size
        for layer in range(self.num_layers):
            w_ih = self.get_parameter(f"weight_ih_l{layer}").to(state_dtype)
            w_hh = self.get_parameter(f"weight_hh_l{layer}").to(state_dtype)
            b_ih = self.get_parameter(f"bias_ih_l{layer}").to(state_dtype)
            b_hh = self.get_parameter(f"bias_hh_l{layer}").to(state_dtype)
            gates = layer_input @ w_ih.t() + b_ih + h[layer] @ w_hh.t() + b_hh
            i_gate = torch.sigmoid(gates[:, 0 * hidden : 1 * hidden])
            f_gate = torch.sigmoid(gates[:, 1 * hidden : 2 * hidden])
            g_gate = torch.tanh(gates[:, 2 * hidden : 3 * hidden])
            o_gate = torch.sigmoid(gates[:, 3 * hidden : 4 * hidden])
            c_next = f_gate * c[layer] + i_gate * g_gate
            h_next = o_gate * torch.tanh(c_next)
            next_h.append(h_next)
            next_c.append(c_next)
            layer_input = h_next
        return layer_input, (torch.stack(next_h), torch.stack(next_c))
