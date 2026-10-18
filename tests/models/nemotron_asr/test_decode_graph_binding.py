# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm.models.nemotron_asr.decode_graph import (
    DenseGraphBinding,
    GraphRuntime,
    execution_tiers,
)
from vllm.models.nemotron_asr.rnnt import (
    DecodeState,
    FrameAlignedDecode,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]


class _FakeWrapper:
    def __init__(
        self,
        runnable: Callable[[], tuple[torch.Tensor, ...]],
        *_args: object,
        **_kwargs: object,
    ) -> None:
        self.runnable = runnable
        self.calls = 0

    def __call__(self) -> tuple[torch.Tensor, ...]:
        self.calls += 1
        return self.runnable()


def _runtime(*, fail_capture: bool = False) -> GraphRuntime:
    mode = {"value": "none"}

    def context(
        _metadata: object,
        _config: object,
        *,
        cudagraph_runtime_mode: str,
        batch_descriptor: object,
    ) -> nullcontext[None]:
        del batch_descriptor
        mode["value"] = cudagraph_runtime_mode
        if fail_capture and cudagraph_runtime_mode == "graph":
            raise RuntimeError("capture refused")
        return nullcontext()

    return GraphRuntime(
        wrapper_factory=_FakeWrapper,
        forward_context=context,
        capture_context=lambda _device: nullcontext(),
        descriptor_factory=lambda tier: (tier, tier),
        eager_mode="none",
        graph_mode="graph",
        synchronize=lambda _device: None,
        set_capture_enabled=lambda _enabled: None,
    )


def _decode_calls() -> tuple[list[torch.Tensor], Callable[..., FrameAlignedDecode]]:
    calls: list[torch.Tensor] = []

    def decode(
        enc_frames: torch.Tensor,
        enc_lengths: torch.Tensor,
        _predictor: object,
        _joint: object,
        state: DecodeState,
    ) -> FrameAlignedDecode:
        calls.append(enc_lengths.clone())
        batch, frames, _ = enc_frames.shape
        active = enc_lengths > 0
        gate = active.view(1, batch, 1)
        h = torch.where(gate, state.h + 1, state.h)
        c = torch.where(gate, state.c + 2, state.c)
        labels = torch.where(active, state.last_label + 3, state.last_label)
        return FrameAlignedDecode(
            token_ids=torch.zeros(batch, frames * 10, dtype=torch.int32),
            token_lengths=active.to(torch.int32),
            state=DecodeState(h=h, c=c, last_label=labels),
            frame_emission_counts=torch.zeros(batch, frames, dtype=torch.int32),
            frame_final_labels=torch.zeros(batch, frames, dtype=torch.int32),
        )

    return calls, decode


# @spec PORT-PERF-004
def test_execution_tiers_cover_engine_population_without_duplicate_maximum() -> None:
    assert execution_tiers(1) == (1,)
    assert execution_tiers(4) == (1, 2, 4)
    assert execution_tiers(6) == (1, 2, 4, 6)


# @spec PORT-PERF-004
def test_warmup_captures_every_geometry_tier_and_runtime_rejects_unknown_key() -> None:
    calls, decode = _decode_calls()
    binding = DenseGraphBinding(
        decode_fn=decode,
        predictor=object(),
        joint=object(),
        vllm_config=object(),
        frame_widths=(1, 2),
        tiers=(1, 2, 4),
        encoder_hidden=3,
        predictor_layers=2,
        predictor_hidden=5,
        blank_id=7,
        runtime=_runtime(),
    )

    binding.warmup(torch.device("cpu"), torch.float32)

    assert binding.captured_keys == (
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 1),
        (1, 2),
        (1, 4),
    )
    assert calls
    assert binding.decode_fn(geometry=1, tier=2) is binding.decode_fn(
        geometry=1,
        tier=2,
    )
    with pytest.raises(ValueError, match="uncaptured dense graph"):
        binding.decode_fn(geometry=2, tier=4)


# @spec PORT-PERF-004
def test_warmup_captures_only_served_geometries_with_stable_ids() -> None:
    _, decode = _decode_calls()
    binding = DenseGraphBinding(
        decode_fn=decode,
        predictor=object(),
        joint=object(),
        vllm_config=object(),
        frame_widths=(1, None, 4),
        tiers=(1, 2),
        encoder_hidden=3,
        predictor_layers=2,
        predictor_hidden=5,
        blank_id=7,
        runtime=_runtime(),
    )

    binding.warmup(torch.device("cpu"), torch.float32)

    assert binding.captured_keys == ((0, 1), (0, 2), (2, 1), (2, 2))
    with pytest.raises(ValueError, match="uncaptured dense graph"):
        binding.decode_fn(geometry=1, tier=1)


# @spec PORT-ADV-003, PORT-PERF-004
def test_runtime_pads_with_zero_lengths_and_returns_only_live_rows() -> None:
    calls, decode = _decode_calls()
    binding = DenseGraphBinding(
        decode_fn=decode,
        predictor=object(),
        joint=object(),
        vllm_config=object(),
        frame_widths=(2,),
        tiers=(1, 2, 4),
        encoder_hidden=3,
        predictor_layers=2,
        predictor_hidden=5,
        blank_id=7,
        runtime=_runtime(),
    )
    binding.warmup(torch.device("cpu"), torch.float32)
    graph_decode = binding.decode_fn(geometry=0, tier=4)
    state = DecodeState(
        h=torch.ones(2, 3, 5),
        c=torch.ones(2, 3, 5),
        last_label=torch.tensor([4, 5, 6]),
    )

    result = graph_decode(
        torch.ones(3, 2, 3),
        torch.tensor([2, 1, 2]),
        object(),
        object(),
        state,
    )

    assert calls[-1].tolist() == [2, 1, 2, 0]
    assert result.token_ids.shape[0] == 3
    assert result.state.h.shape == (2, 3, 5)
    # The graph owns fixed tier-four storage. Selecting three live rows keeps
    # each predictor layer densely packed while retaining one padded row
    # between layers; materializing a contiguous copy here would add an
    # uncaptured D2D repack to every non-tier-exact replay.
    assert result.state.h.stride() == (20, 5, 1)
    assert result.state.c.stride() == (20, 5, 1)
    assert not result.state.h.is_contiguous()
    assert not result.state.c.is_contiguous()
    assert result.state.last_label.tolist() == [7, 8, 9]


# @spec PORT-PERF-004
def test_capture_failure_fails_startup_without_eager_fallback() -> None:
    _, decode = _decode_calls()
    binding = DenseGraphBinding(
        decode_fn=decode,
        predictor=object(),
        joint=object(),
        vllm_config=object(),
        frame_widths=(1,),
        tiers=(1,),
        encoder_hidden=3,
        predictor_layers=2,
        predictor_hidden=5,
        blank_id=7,
        runtime=_runtime(fail_capture=True),
    )

    with pytest.raises(RuntimeError, match="capture refused"):
        binding.warmup(torch.device("cpu"), torch.float32)
    assert binding.captured_keys == ()


# @spec PORT-PERF-004
def test_served_dense_graph_arm_requires_binding_and_resolves_exact_key() -> None:
    from vllm.models.nemotron_asr.advance import (
        DecodeRequest,
    )
    from vllm.models.nemotron_asr.nemotron_asr import (
        build_decode_resolver,
    )

    config = SimpleNamespace(
        decode_dispatch_arm="dense-graphed",
        decode_dispatch_table=None,
    )
    with pytest.raises(ValueError, match="requires a captured graph binding"):
        build_decode_resolver(config)

    seen: list[tuple[int, int]] = []
    sentinel = object()

    class Binding:
        def execution_tier(self, live_rows: int) -> int:
            return live_rows

        def decode_fn(self, *, geometry: int, tier: int) -> object:
            seen.append((geometry, tier))
            return sentinel

    resolver = build_decode_resolver(config, graph_binding=Binding())
    resolved = resolver(
        DecodeRequest(
            geometry=3,
            execution_batch_size=4,
            graph_covers_decode=True,
            ready_decode_buckets=2,
            execution_tier_limit=4,
        )
    )
    assert resolved.arm == "dense-graphed"
    assert resolved.decode_fn is sentinel
    assert seen == [(3, 4)]


# @spec PORT-PERF-004
def test_dense_graph_memory_profile_uses_canonical_dense_eager_resolver() -> None:
    from vllm.models.nemotron_asr.advance import (
        DecodeRequest,
    )
    from vllm.models.nemotron_asr.nemotron_asr import (
        build_decode_resolver,
    )
    from vllm.models.nemotron_asr.rnnt import (
        decode_dense_masked_frames,
    )

    config = SimpleNamespace(
        decode_dispatch_arm="dense-graphed",
        decode_dispatch_table=None,
    )

    class Binding:
        def execution_tier(self, live_rows: int) -> int:
            raise AssertionError("profile must not consult captured tiers")

        def decode_fn(self, *, geometry: int, tier: int) -> object:
            raise AssertionError("profile must not bind a graph key")

    resolver = build_decode_resolver(config, graph_binding=Binding())
    resolved = resolver(
        DecodeRequest(
            geometry=4,
            execution_batch_size=4,
            graph_covers_decode=False,
            ready_decode_buckets=1,
            execution_tier_limit=0,
            memory_profile=True,
        )
    )

    assert resolved.arm == "dense-eager"
    assert resolved.decode_fn is decode_dense_masked_frames
    assert resolved.override_reason == "pre-capture-memory-profile"


# @spec PORT-PERF-004
def test_served_dense_graph_arm_selects_smallest_tier_within_plan_ceiling() -> None:
    from vllm.models.nemotron_asr.advance import (
        DecodeRequest,
    )
    from vllm.models.nemotron_asr.nemotron_asr import (
        build_decode_resolver,
    )

    config = SimpleNamespace(
        decode_dispatch_arm="dense-graphed",
        decode_dispatch_table=None,
    )
    seen: list[tuple[int, int]] = []

    class Binding:
        def execution_tier(self, live_rows: int) -> int:
            return 1 if live_rows == 1 else 4

        def decode_fn(self, *, geometry: int, tier: int) -> object:
            seen.append((geometry, tier))
            return lambda *args: args

    resolver = build_decode_resolver(config, graph_binding=Binding())
    resolved = resolver(
        DecodeRequest(
            geometry=2,
            execution_batch_size=3,
            graph_covers_decode=True,
            ready_decode_buckets=1,
            execution_tier_limit=4,
        )
    )
    assert resolved.arm == "dense-graphed"
    assert seen == [(2, 4)]

    with pytest.raises(ValueError, match="exceeds RowPlan"):
        resolver(
            DecodeRequest(
                geometry=2,
                execution_batch_size=3,
                graph_covers_decode=True,
                ready_decode_buckets=1,
                execution_tier_limit=2,
            )
        )


def test_execution_profile_receipt_reports_resolved_arms_and_ready_keys() -> None:
    # @spec PORT-PERF-004, PORT-PERF-009
    from vllm.models.nemotron_asr.nemotron_asr import (
        NemotronASRForRNNT,
    )

    model = object.__new__(NemotronASRForRNNT)
    object.__setattr__(
        model,
        "config",
        SimpleNamespace(decode_dispatch_arm="dense-graphed"),
    )
    object.__setattr__(
        model,
        "_encoder_execution",
        SimpleNamespace(
            ready_receipt=lambda: {
                "arm": "compiled-static",
                "ready": True,
                "warmup_cells": [[0, 1], [0, 2]],
                "warmup_geometries": [0],
                "warmup_populations": [1, 2],
            }
        ),
    )
    object.__setattr__(
        model,
        "_decode_graph_binding",
        SimpleNamespace(captured_keys=((0, 1), (0, 2))),
    )

    assert model.execution_profile_receipt() == {
        "schema": "nemotron-execution-profile/1",
        "decode": {
            "arm": "dense-graphed",
            "captured_keys": [[0, 1], [0, 2]],
            "ready": True,
        },
        "encoder": {
            "arm": "compiled-static",
            "ready": True,
            "warmup_cells": [[0, 1], [0, 2]],
            "warmup_geometries": [0],
            "warmup_populations": [1, 2],
        },
    }
