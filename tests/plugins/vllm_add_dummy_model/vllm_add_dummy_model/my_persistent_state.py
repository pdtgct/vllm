# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A deterministic model whose only cache is fixed-size per-request state.

`StubStateLayer` keeps the sum of the tokens a request has fed it in a
`MambaSpec` page allocated by `MambaManager`, and emits a function of that sum.
A slot reused without starting from zero, or a state lost between streaming
inputs, changes the output.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.models.interfaces import IsAttentionFree
from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import KVCacheLayerConfig, MambaSpec


def emit(total, vocab_size: int):
    return 3 + total * 7919 % (vocab_size - 3)


def expected_turns(
    inputs: Sequence[Sequence[int]], max_tokens: int, vocab_size: int
) -> list[list[int]]:
    """The tokens each streaming input produces. A turn's last sampled token
    is never fed back: the next input replaces it."""
    total = 0
    turns = []
    for prompt in inputs:
        total += sum(prompt)
        turn = [emit(total, vocab_size)]
        for _ in range(max_tokens - 1):
            total += turn[-1]
            turn.append(emit(total, vocab_size))
        turns.append(turn)
    return turns


class StubStateLayer(MambaBase, nn.Module):
    mamba_type = MambaAttentionBackendEnum.SHORT_CONV

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix
        self.vocab_size = vllm_config.model_config.get_vocab_size()
        vllm_config.compilation_config.static_forward_context[prefix] = self

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return ((1,),)

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return (torch.int64,)

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig:
        spec = MambaSpec(
            shapes=self.get_state_shape(),
            dtypes=self.get_state_dtype(),
            block_size=vllm_config.model_config.max_model_len,
            mamba_type=self.mamba_type,
        )
        return MambaManager.layer_config(spec, recomputable=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Add each request's tokens to its sum, and return the token to emit
        at each request's last position."""
        out = torch.zeros(input_ids.shape[0], 1, device=input_ids.device)
        attn_metadata = get_forward_context().attn_metadata
        # Warm-up runs carry no metadata for this layer.
        if not isinstance(attn_metadata, dict) or self.prefix not in attn_metadata:
            return out
        metadata = attn_metadata[self.prefix]
        assert isinstance(metadata, BaseMambaAttentionMetadata)
        totals = self.kv_cache[0][:, 0]

        if num_decodes := metadata.num_decodes:
            assert metadata.state_indices_tensor_d is not None
            slots = metadata.state_indices_tensor_d[:num_decodes, 0]
            totals[slots] += input_ids[:num_decodes]
            out[:num_decodes, 0] = emit(totals[slots], self.vocab_size)

        if num_prefills := metadata.num_prefills:
            assert metadata.state_indices_tensor_p is not None
            assert metadata.query_start_loc_p is not None
            assert metadata.has_initial_states_p is not None
            slots = metadata.state_indices_tensor_p[:num_prefills]
            start = metadata.num_decode_tokens
            bounds = start + metadata.query_start_loc_p
            cumsum = nn.functional.pad(input_ids.cumsum(0), (1, 0))
            totals[slots] = (
                totals[slots] * metadata.has_initial_states_p
                + cumsum[bounds[1:]]
                - cumsum[bounds[:-1]]
            )
            out[bounds[1:] - 1, 0] = emit(totals[slots], self.vocab_size).to(out.dtype)
        return out


class StubPersistentStateForCausalLM(nn.Module, IsAttentionFree):
    """See the module docstring."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vocab_size = vllm_config.model_config.get_vocab_size()
        self.state = StubStateLayer(vllm_config, prefix=f"{prefix}state")

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(input_ids.shape[0], 1, device=input_ids.device)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return self.state(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(
            hidden_states.shape[0], self.vocab_size, device=hidden_states.device
        )
        return logits.scatter_(1, hidden_states.long(), 1.0)

    def load_weights(self, weights) -> set[str]:
        return set()
