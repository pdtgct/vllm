# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-manifest projection onto the generic persistent-state page."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from vllm.models.nemotron_asr.manifests import (
    author_state_manifest,
)
from vllm.v1.kv_cache_interface import (
    PersistentStateDescriptor,
    PersistentStateSpec,
)

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "int32": torch.int32,
    "int64": torch.int64,
}


def _zeros(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.zeros(shape, dtype=dtype, device=device)


def _filled(value: int) -> Callable[..., torch.Tensor]:
    def initialize(
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.full(shape, value, dtype=dtype, device=device)

    return initialize


def build_nemotron_persistent_state_spec(config: Any) -> PersistentStateSpec:
    """Build the one aggregate page directly from the published manifest.

    ``admitted_prompt`` and ``admitted_geometry`` start at zero in raw
    storage; the first transaction stamps both fields from the scheduler's
    admitted authority before any gather can observe them.  The checkpoint
    blank id is stable at construction and can therefore use a concrete
    initializer.
    """
    manifest = author_state_manifest(config)
    descriptors: list[PersistentStateDescriptor] = []
    offset = 0
    blank_label = int(config.num_asr_labels)
    for entry in manifest["entries"]:
        dtype = _DTYPES[entry["dtype"]]
        alignment = torch.empty((), dtype=dtype).element_size()
        aligned = (offset + alignment - 1) // alignment * alignment
        if aligned != offset:
            raise ValueError(
                f"state manifest requires implicit padding before {entry['name']!r}; "
                "publish the padding explicitly"
            )
        init_name = entry["init"]
        initializer = _filled(blank_label) if init_name == "blank_label" else _zeros
        descriptor = PersistentStateDescriptor(
            name=entry["name"],
            shape=tuple(int(value) for value in entry["shape"]),
            dtype=dtype,
            offset_bytes=offset,
            alignment_bytes=alignment,
            initializer=initializer,
        )
        descriptors.append(descriptor)
        offset += descriptor.size_bytes

    if offset != manifest["total_page_bytes"]:
        raise ValueError("state manifest byte total disagrees with its entries")
    return PersistentStateSpec(
        descriptors=descriptors,
        page_size_bytes=offset,
        state_name="nemotron.cache_aware_streaming",
    )


class ComponentPools(Protocol):
    @property
    def spec(self) -> PersistentStateSpec: ...

    @property
    def views(self) -> Mapping[str, torch.Tensor]: ...


def _adjacent_matrix(
    storage: ComponentPools,
    names: tuple[str, ...],
) -> torch.Tensor:
    """Project adjacent scalar descriptors as one write-through matrix."""
    if not names:
        raise ValueError("adjacent view requires at least one descriptor")
    descriptors = {item.name: item for item in storage.spec.descriptors}
    selected = [descriptors[name] for name in names]
    dtype = selected[0].dtype
    if any(item.dtype != dtype or item.shape != (1,) for item in selected):
        raise ValueError("adjacent matrix fields must be same-dtype scalars")
    offset = selected[0].offset_bytes
    expected = offset
    for item in selected:
        if item.offset_bytes != expected:
            raise ValueError("adjacent matrix fields are not contiguous")
        expected += item.size_bytes
    element_size = selected[0].element_size_bytes
    first = storage.views[names[0]]
    return torch.as_strided(
        first,
        size=(first.shape[0], len(selected)),
        stride=(storage.spec.page_size_bytes // element_size, 1),
    )


@dataclass(frozen=True)
class NemotronStatePools:
    """Zero-copy tensor vocabulary consumed by ``advance_model_rows``."""

    channel: tuple[torch.Tensor, ...]
    convolution: tuple[torch.Tensor, ...]
    valid_length: tuple[torch.Tensor, ...]
    predictor_h: torch.Tensor
    predictor_c: torch.Tensor
    replay_queue: torch.Tensor
    replay_book: torch.Tensor
    frontend_raw: torch.Tensor
    frontend_mel: torch.Tensor
    frontend_counters: torch.Tensor
    endpoint_history: torch.Tensor
    endpoint_book: torch.Tensor


def project_nemotron_state_pools(
    storage: ComponentPools,
    *,
    n_layers: int,
) -> NemotronStatePools:
    """Return named zero-copy views over the one aggregate allocation."""
    views = storage.views
    channels = tuple(
        views[f"encoder.layers.{index}.window.channel"] for index in range(n_layers)
    )
    convolutions = tuple(
        views[f"encoder.layers.{index}.conv.time"] for index in range(n_layers)
    )
    valid_lengths = tuple(
        views[f"encoder.layers.{index}.window.valid"] for index in range(n_layers)
    )
    replay_book = _adjacent_matrix(
        storage,
        (
            "decode.layers.0.replay.book.queue_head",
            "decode.layers.0.replay.book.queue_length",
            "decode.layers.0.replay.book.last_label",
            "decode.layers.0.replay.book.prompt",
            "decode.layers.0.replay.book.geometry",
            "decode.layers.0.replay.book.pending_echo",
            "decode.layers.0.replay.book.expected_label",
        ),
    )
    frontend_counters = _adjacent_matrix(
        storage,
        (
            "frontend.counters.total_valid_samples",
            "frontend.counters.committed_mel_frames",
            "frontend.counters.encoded_mel_frames",
            "frontend.counters.raw_tail_origin",
            "frontend.counters.raw_tail_length",
            "frontend.counters.mel_tail_length",
            "frontend.counters.expected_chunk_sequence",
            "frontend.counters.finalized",
        ),
    )
    endpoint_book = _adjacent_matrix(
        storage,
        (
            "endpoint.book.history_length",
            "endpoint.book.history_head",
            "endpoint.book.endpoint_armed",
            "endpoint.book.segment_generation",
            "endpoint.book.segment_has_output",
            "endpoint.book.pending_forced_generation",
        ),
    )
    return NemotronStatePools(
        channel=channels,
        convolution=convolutions,
        valid_length=valid_lengths,
        predictor_h=views["predictor.layers.0.lstm_state.h"],
        predictor_c=views["predictor.layers.0.lstm_state.c"],
        replay_queue=views["decode.layers.0.replay.queue"],
        replay_book=replay_book,
        frontend_raw=views["frontend.raw_tail"],
        frontend_mel=views["frontend.mel_tail"],
        frontend_counters=frontend_counters,
        endpoint_history=views["endpoint.history"],
        endpoint_book=endpoint_book,
    )


@dataclass
class StateScratch:
    """Ephemeral batch workspace, never resident slot ownership."""

    spec: PersistentStateSpec
    raw: torch.Tensor
    views: dict[str, torch.Tensor]

    def initialize(self, index: int) -> None:
        start = index * self.spec.page_size_bytes
        self.raw[start : start + self.spec.page_size_bytes].zero_()
        for descriptor in self.spec.descriptors:
            self.views[descriptor.name][index].copy_(
                descriptor.initializer(
                    descriptor.shape, dtype=descriptor.dtype, device=self.raw.device
                )
            )


def allocate_state_scratch(
    spec: PersistentStateSpec, rows: int, device: torch.device
) -> StateScratch:
    """Allocate the transaction's bounded scratch, including its null row."""
    raw = torch.zeros(rows * spec.page_size_bytes, dtype=torch.uint8, device=device)
    views = {}
    for descriptor in spec.descriptors:
        stride = 1
        strides = []
        for dimension in reversed(descriptor.shape):
            strides.append(stride)
            stride *= dimension
        views[descriptor.name] = torch.as_strided(
            raw[descriptor.offset_bytes :].view(descriptor.dtype),
            size=(rows, *descriptor.shape),
            stride=(
                spec.page_size_bytes // descriptor.element_size_bytes,
                *reversed(strides),
            ),
        )
    scratch = StateScratch(spec, raw, views)
    for index in range(rows):
        scratch.initialize(index)
    return scratch
