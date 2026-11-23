# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata for per-request state that a request's tokens cannot rebuild.

The state lives in a `MambaSpec` page of a non-recomputable group, which the
worker initializes on allocation, so every row holds valid initial state.
"""

from dataclasses import replace
from typing import Any

import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.short_conv_attn import (
    ShortConvAttentionBackend,
    ShortConvAttentionMetadata,
    ShortConvAttentionMetadataBuilder,
)


class PersistentStateAttentionBackend(ShortConvAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "PERSISTENT_STATE"

    @staticmethod
    def get_builder_cls() -> type["PersistentStateAttentionMetadataBuilder"]:
        return PersistentStateAttentionMetadataBuilder


class PersistentStateAttentionMetadataBuilder(ShortConvAttentionMetadataBuilder):
    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs: Any,
    ) -> ShortConvAttentionMetadata:
        metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build, **kwargs
        )
        # @spec PSTATE-PAGE-003
        if metadata.num_prefills == 0:
            return metadata
        return replace(
            metadata,
            has_initial_states_p=torch.ones(
                metadata.num_prefills,
                dtype=torch.bool,
                device=common_attn_metadata.query_start_loc.device,
            ),
        )
