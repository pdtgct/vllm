# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regional graph binding for the fixed-trip dense RNN-T decoder."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import torch

from vllm.models.nemotron_asr.batch_invariance import MODE_OFF, BatchInvariantExecution
from vllm.models.nemotron_asr.rnnt import (
    MAX_SYMBOLS_PER_STEP,
    DecodeState,
    FrameAlignedDecode,
)


def execution_tiers(maximum: int) -> tuple[int, ...]:
    """Return power-of-two tiers plus an exact non-power maximum."""
    if maximum <= 0:
        raise ValueError("decode graph maximum population must be positive")
    tiers: list[int] = []
    value = 1
    while value < maximum:
        tiers.append(value)
        value *= 2
    tiers.append(maximum)
    return tuple(tiers)


@dataclass(frozen=True)
class GraphRuntime:
    """Backend graph operations used by :class:`DenseGraphBinding`."""

    wrapper_factory: Callable[..., Any]
    forward_context: Callable[..., AbstractContextManager]
    capture_context: Callable[[torch.device], AbstractContextManager]
    descriptor_factory: Callable[[int], Any]
    eager_mode: Any
    graph_mode: Any
    synchronize: Callable[[torch.device], None]
    set_capture_enabled: Callable[[bool], None]


@dataclass
class _GraphEntry:
    geometry: int
    tier: int
    frames: int
    enc_frames: torch.Tensor
    enc_lengths: torch.Tensor
    h: torch.Tensor
    c: torch.Tensor
    last_label: torch.Tensor
    token_ids: torch.Tensor
    token_lengths: torch.Tensor
    out_h: torch.Tensor
    out_c: torch.Tensor
    out_last_label: torch.Tensor
    frame_emission_counts: torch.Tensor
    frame_final_labels: torch.Tensor
    descriptor: Any
    wrapper: Any

    def output_tuple(self) -> tuple[torch.Tensor, ...]:
        return (
            self.token_ids,
            self.token_lengths,
            self.out_h,
            self.out_c,
            self.out_last_label,
            self.frame_emission_counts,
            self.frame_final_labels,
        )


def platform_graph_runtime() -> GraphRuntime:
    """Build the installed accelerator's vLLM graph runtime."""
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled
    from vllm.config import CUDAGraphMode
    from vllm.distributed.parallel_state import graph_capture
    from vllm.forward_context import BatchDescriptor, set_forward_context
    from vllm.platforms import current_platform
    from vllm.utils.import_utils import resolve_obj_by_qualname

    return GraphRuntime(
        wrapper_factory=resolve_obj_by_qualname(
            current_platform.get_static_graph_wrapper_cls()
        ),
        forward_context=set_forward_context,
        capture_context=lambda device: graph_capture(device=device),
        descriptor_factory=lambda tier: BatchDescriptor(
            num_tokens=tier,
            num_reqs=tier,
            uniform=True,
        ),
        eager_mode=CUDAGraphMode.NONE,
        graph_mode=CUDAGraphMode.PIECEWISE,
        synchronize=lambda device: torch.accelerator.synchronize(device),
        set_capture_enabled=set_cudagraph_capturing_enabled,
    )


class DenseGraphBinding:
    """Own stable decoder buffers and platform graph wrappers.

    Capture is transactional: no key becomes callable until every declared
    geometry/tier pair has completed eager warmup, capture, and replay
    equivalence.
    """

    def __init__(
        self,
        *,
        decode_fn: Callable[..., FrameAlignedDecode],
        predictor: Any,
        joint: Any,
        vllm_config: Any,
        frame_widths: tuple[int | None, ...],
        tiers: tuple[int, ...],
        encoder_hidden: int,
        predictor_layers: int,
        predictor_hidden: int,
        blank_id: int,
        runtime: GraphRuntime | None = None,
        batch_invariant_mode: BatchInvariantExecution = MODE_OFF,
    ) -> None:
        if (
            not frame_widths
            or not any(value is not None for value in frame_widths)
            or any(value is not None and value <= 0 for value in frame_widths)
        ):
            raise ValueError("decode graph requires at least one positive frame width")
        if not tiers or tuple(sorted(set(tiers))) != tiers or tiers[0] <= 0:
            raise ValueError("decode graph tiers must be positive and increasing")
        # @spec PORT-PREC-018, PORT-PREC-025: namespace before capture.
        self._batch_invariant_mode = batch_invariant_mode
        self._decode_fn = decode_fn
        self._predictor = predictor
        self._joint = joint
        self._vllm_config = vllm_config
        self._frame_widths = frame_widths
        self._tiers = tiers
        self._encoder_hidden = encoder_hidden
        self._predictor_layers = predictor_layers
        self._predictor_hidden = predictor_hidden
        self._blank_id = blank_id
        self._runtime = runtime
        self._entries: dict[tuple[Any, ...], _GraphEntry] = {}
        self._decode_fns: dict[tuple[Any, ...], Callable[..., Any]] = {}

    @property
    def captured_keys(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(sorted(self._entries))

    def execution_tier(self, live_rows: int) -> int:
        for tier in self._tiers:
            if live_rows <= tier:
                return tier
        raise ValueError(
            f"decode population {live_rows} exceeds captured maximum {self._tiers[-1]}"
        )

    def _new_entry(
        self,
        geometry: int,
        tier: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        runtime: GraphRuntime,
    ) -> _GraphEntry:
        frames = self._frame_widths[geometry]
        if frames is None:
            raise ValueError(f"decode graph geometry {geometry} is not served")
        enc_frames = torch.zeros(
            tier,
            frames,
            self._encoder_hidden,
            dtype=dtype,
            device=device,
        )
        enc_lengths = torch.full(
            (tier,),
            frames,
            dtype=torch.int64,
            device=device,
        )
        h = torch.zeros(
            self._predictor_layers,
            tier,
            self._predictor_hidden,
            dtype=torch.float32,
            device=device,
        )
        c = torch.zeros_like(h)
        last_label = torch.full(
            (tier,),
            self._blank_id,
            dtype=torch.int64,
            device=device,
        )
        token_ids = torch.zeros(
            tier,
            frames * MAX_SYMBOLS_PER_STEP,
            dtype=torch.int32,
            device=device,
        )
        token_lengths = torch.zeros(
            tier,
            dtype=torch.int32,
            device=device,
        )
        out_h = torch.zeros_like(h)
        out_c = torch.zeros_like(c)
        out_last_label = torch.zeros_like(last_label)
        frame_emission_counts = torch.zeros(
            tier,
            frames,
            dtype=torch.int32,
            device=device,
        )
        frame_final_labels = torch.zeros_like(frame_emission_counts)
        entry: _GraphEntry

        def run() -> tuple[torch.Tensor, ...]:
            decoded = self._decode_fn(
                enc_frames,
                enc_lengths,
                self._predictor,
                self._joint,
                DecodeState(h=h, c=c, last_label=last_label),
            )
            if not isinstance(decoded, FrameAlignedDecode):
                raise TypeError("dense graph decoder must return FrameAlignedDecode")
            token_ids.copy_(decoded.token_ids)
            token_lengths.copy_(decoded.token_lengths)
            out_h.copy_(decoded.state.h)
            out_c.copy_(decoded.state.c)
            out_last_label.copy_(decoded.state.last_label)
            frame_emission_counts.copy_(decoded.frame_emission_counts)
            frame_final_labels.copy_(decoded.frame_final_labels)
            return entry.output_tuple()

        wrapper = runtime.wrapper_factory(
            run,
            self._vllm_config,
            runtime_mode=runtime.graph_mode,
        )
        entry = _GraphEntry(
            geometry=geometry,
            tier=tier,
            frames=frames,
            enc_frames=enc_frames,
            enc_lengths=enc_lengths,
            h=h,
            c=c,
            last_label=last_label,
            token_ids=token_ids,
            token_lengths=token_lengths,
            out_h=out_h,
            out_c=out_c,
            out_last_label=out_last_label,
            frame_emission_counts=frame_emission_counts,
            frame_final_labels=frame_final_labels,
            descriptor=runtime.descriptor_factory(tier),
            wrapper=wrapper,
        )
        return entry

    def _call(
        self,
        entry: _GraphEntry,
        runtime: GraphRuntime,
        mode: Any,
    ) -> tuple[torch.Tensor, ...]:
        with runtime.forward_context(
            None,
            self._vllm_config,
            cudagraph_runtime_mode=mode,
            batch_descriptor=entry.descriptor,
        ):
            output = entry.wrapper()
        if not isinstance(output, tuple) or len(output) != 7:
            raise TypeError("dense graph wrapper returned malformed output")
        return output

    @staticmethod
    def _snapshot(output: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(tensor.clone() for tensor in output)

    @staticmethod
    def _assert_equal(
        expected: tuple[torch.Tensor, ...],
        actual: tuple[torch.Tensor, ...],
    ) -> None:
        if any(not torch.equal(left, right) for left, right in zip(expected, actual)):
            raise RuntimeError("dense graph capture/replay differs from eager decode")

    # @spec PORT-PERF-004
    def warmup(self, device: torch.device, dtype: torch.dtype) -> None:
        if self._entries:
            return
        runtime = self._runtime or platform_graph_runtime()
        pending = {
            self._batch_invariant_mode.graph_key(
                "decode_fn", (geometry, tier)
            ): self._new_entry(
                geometry,
                tier,
                device=device,
                dtype=dtype,
                runtime=runtime,
            )
            for geometry, frames in enumerate(self._frame_widths)
            if frames is not None
            for tier in self._tiers
        }
        runtime.set_capture_enabled(True)
        try:
            with torch.inference_mode(), runtime.capture_context(device):
                for entry in pending.values():
                    eager = self._snapshot(
                        self._call(entry, runtime, runtime.eager_mode)
                    )
                    captured = self._call(
                        entry,
                        runtime,
                        runtime.graph_mode,
                    )
                    self._assert_equal(eager, captured)
                    replayed = self._call(
                        entry,
                        runtime,
                        runtime.graph_mode,
                    )
                    self._assert_equal(eager, replayed)
                runtime.synchronize(device)
        finally:
            runtime.set_capture_enabled(False)
        decode_fns = {
            key: self._bind_decode(entry, runtime) for key, entry in pending.items()
        }
        self._runtime = runtime
        self._entries = pending
        self._decode_fns = decode_fns

    def _bind_decode(
        self,
        entry: _GraphEntry,
        runtime: GraphRuntime,
        *,
        captured: bool = True,
    ) -> Callable[..., Any]:
        def decode(
            enc_frames: torch.Tensor,
            enc_lengths: torch.Tensor,
            predictor: Any,
            joint: Any,
            state: DecodeState,
        ) -> FrameAlignedDecode:
            del predictor, joint
            live = int(enc_frames.shape[0])
            if live <= 0 or live > entry.tier:
                raise ValueError(
                    f"dense graph live population {live} outside 1..{entry.tier}"
                )
            if tuple(enc_frames.shape[1:]) != (
                entry.frames,
                self._encoder_hidden,
            ):
                raise ValueError("dense graph encoder shape differs from captured key")
            if enc_lengths.shape != (live,):
                raise ValueError("dense graph lengths disagree with live population")
            if (
                state.h.shape
                != (
                    self._predictor_layers,
                    live,
                    self._predictor_hidden,
                )
                or state.c.shape != state.h.shape
            ):
                raise ValueError(
                    "dense graph predictor state differs from captured key"
                )
            if state.last_label.shape != (live,):
                raise ValueError("dense graph labels disagree with live population")
            if (
                enc_frames.device != entry.enc_frames.device
                or enc_frames.dtype != entry.enc_frames.dtype
                or enc_lengths.device != entry.enc_lengths.device
                or enc_lengths.dtype != entry.enc_lengths.dtype
                or state.h.device != entry.h.device
                or state.h.dtype != entry.h.dtype
                or state.c.device != entry.c.device
                or state.c.dtype != entry.c.dtype
                or state.last_label.device != entry.last_label.device
                or state.last_label.dtype != entry.last_label.dtype
            ):
                raise ValueError(
                    "dense graph runtime tensors differ from the captured device/dtype"
                )

            entry.enc_frames.zero_()
            entry.enc_frames[:live].copy_(enc_frames)
            entry.enc_lengths.zero_()
            entry.enc_lengths[:live].copy_(enc_lengths)
            entry.h.zero_()
            entry.h[:, :live].copy_(state.h)
            entry.c.zero_()
            entry.c[:, :live].copy_(state.c)
            entry.last_label.fill_(self._blank_id)
            entry.last_label[:live].copy_(state.last_label)
            output = self._call(
                entry, runtime, runtime.graph_mode if captured else runtime.eager_mode
            )
            (
                token_ids,
                token_lengths,
                out_h,
                out_c,
                out_last_label,
                frame_emission_counts,
                frame_final_labels,
            ) = output
            return FrameAlignedDecode(
                token_ids=token_ids[:live],
                token_lengths=token_lengths[:live],
                state=DecodeState(
                    h=out_h[:, :live],
                    c=out_c[:, :live],
                    last_label=out_last_label[:live],
                ),
                frame_emission_counts=frame_emission_counts[:live],
                frame_final_labels=frame_final_labels[:live],
            )

        return decode

    def uncaptured_decode_fn(self, *, geometry: int, tier: int) -> Callable[..., Any]:
        """Reuse sealed tier storage while an enclosing CHUNK graph captures it.

        Input padding, output layouts and numerical arithmetic are identical to
        the split decoder. The nested wrapper runs eagerly, so the enclosing
        capture records the decoder operations instead of another graph replay.
        Callers must serialize this storage with the ordinary decoder binding.
        """
        key = self._batch_invariant_mode.graph_key("decode_fn", (geometry, tier))
        if key not in self._entries or self._runtime is None:
            raise ValueError(
                f"uncaptured dense graph key geometry={geometry} tier={tier}"
            )
        return self._bind_decode(self._entries[key], self._runtime, captured=False)

    def decode_fn(self, *, geometry: int, tier: int) -> Callable[..., Any]:
        """Return the startup-bound callable for one exact graph key."""
        key = self._batch_invariant_mode.graph_key("decode_fn", (geometry, tier))
        try:
            return self._decode_fns[key]
        except KeyError:
            raise ValueError(
                f"uncaptured dense graph key geometry={geometry} tier={tier}"
            ) from None
