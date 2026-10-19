# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded session views and the state-write completion contract."""

from copy import copy, deepcopy
from typing import TYPE_CHECKING

import torch

from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    PersistentStateSpec,
    StateCompletion,
    StateSlotHandle,
)
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class StateSlotViews:
    """Expose only the current generation's bounded slot to a model."""

    def __init__(self, spec: PersistentStateSpec, tensor: torch.Tensor):
        if (
            tensor.dtype not in (torch.int8, torch.uint8)
            or tensor.ndim != 1
            or not tensor.is_contiguous()
            or tensor.numel() % spec.page_size_bytes
        ):
            raise ValueError("Invalid persistent state backing storage")
        self.spec = spec
        self.capacity = tensor.numel() // spec.page_size_bytes
        self._tensor = tensor
        self._views: dict[str, torch.Tensor] = {}
        for descriptor in spec.descriptors:
            if (
                tensor.data_ptr() + descriptor.offset_bytes
            ) % descriptor.alignment_bytes:
                raise ValueError("Persistent state backing storage is misaligned")
            stride = 1
            strides = []
            for dimension in reversed(descriptor.shape):
                strides.append(stride)
                stride *= dimension
            self._views[descriptor.name] = torch.as_strided(
                tensor[descriptor.offset_bytes :].view(descriptor.dtype),
                size=(self.capacity, *descriptor.shape),
                stride=(
                    spec.page_size_bytes // descriptor.element_size_bytes,
                    *reversed(strides),
                ),
            )
        self._epoch: str | None = None
        self._owners: dict[int, tuple[str, StateSlotHandle]] = {}

    def bind(self, request_id: str, handle: StateSlotHandle) -> None:
        if not 0 < handle.slot < self.capacity or handle.generation <= 0:
            raise ValueError("Invalid state slot handle")
        if self._epoch is not None and handle.engine_epoch != self._epoch:
            raise ValueError("Stale state engine epoch")
        previous = self._owners.get(handle.slot)
        if previous is not None:
            if previous == (request_id, handle):
                return
            if handle.generation <= previous[1].generation:
                raise ValueError("Stale state slot generation")
        start = handle.slot * self.spec.page_size_bytes
        self._tensor[start : start + self.spec.page_size_bytes].zero_()
        for descriptor in self.spec.descriptors:
            target = self._views[descriptor.name][handle.slot]
            initialized = descriptor.initializer(
                descriptor.shape, dtype=descriptor.dtype, device=target.device
            )
            if initialized.shape != descriptor.shape:
                raise ValueError("Persistent state initializer returned wrong shape")
            target.copy_(initialized)
        self._epoch = handle.engine_epoch
        self._owners[handle.slot] = (request_id, handle)

    def view(self, request_id: str, handle: StateSlotHandle) -> dict[str, torch.Tensor]:
        if self._owners.get(handle.slot) != (request_id, handle):
            raise ValueError("Stale state slot handle")
        return {name: tensor[handle.slot] for name, tensor in self._views.items()}


class StateSlotBindings:
    def __init__(
        self, groups: list[KVCacheGroupSpec], tensors: dict[str, torch.Tensor]
    ):
        self.views = {
            group.layer_names[0]: StateSlotViews(
                group.kv_cache_spec, tensors[group.layer_names[0]]
            )
            for group in groups
            if isinstance(group.kv_cache_spec, PersistentStateSpec)
        }
        self._streams: list[torch.cuda.Stream] = []
        self._completion: StateCompletion | None = None
        self._main_stream: torch.cuda.Stream | None = None
        self._cleaning_up = False

    def register_writing_stream(self, stream: torch.cuda.Stream) -> None:
        """Register a stream before enqueueing writes outside the runner stream."""
        if stream not in self._streams:
            self._streams.append(stream)
            if self._main_stream is not None:
                stream.wait_stream(self._main_stream)

    def begin_step(
        self, scheduled: "SchedulerOutput", main_stream: torch.cuda.Stream
    ) -> None:
        for stream in self._streams:
            main_stream.wait_stream(stream)
        cleanup = scheduled.state_cleanup_handles or {}
        live = scheduled.state_slot_handles or {}
        for request_id, handles in cleanup.items():
            if handles.keys() != self.views.keys():
                raise ValueError("Cleanup bindings do not match the state groups")
            for name, handle in handles.items():
                self.views[name].view(request_id, handle)
        for request_id, handles in live.items():
            if handles.keys() != self.views.keys():
                raise ValueError(
                    "State bindings do not match the declared state groups"
                )
            for name, handle in handles.items():
                self.views[name].bind(request_id, handle)
        for stream in self._streams:
            stream.wait_stream(main_stream)
        self._main_stream = main_stream
        self._completion = StateCompletion(
            scheduled.state_step_seq,
            deepcopy(live),
            deepcopy(cleanup),
        )
        self._cleaning_up = True

    def activate_requests(self) -> None:
        self._cleaning_up = False

    def get_handle(self, request_id: str, layer_name: str) -> StateSlotHandle:
        """Resolve a handle only while its request is scheduled on this worker."""
        if self._completion is None:
            raise ValueError("No state step is active")
        if self._cleaning_up:
            return self._completion.cleanup_handles[request_id][layer_name]
        return self._completion.handles[request_id][layer_name]

    def finish_step(
        self,
        output: AsyncModelRunnerOutput | ModelRunnerOutput,
        main_stream: torch.cuda.Stream,
    ) -> "StateCompletionOutput":
        if self._completion is None:
            raise RuntimeError("State completion requires a scheduled step")
        for stream in self._streams:
            main_stream.wait_stream(stream)
        event = torch.cuda.Event(blocking=True)
        event.record(main_stream)
        result = StateCompletionOutput(output, event, self._completion)
        self._completion = None
        self._main_stream = None
        return result


class StateCompletionOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        output: AsyncModelRunnerOutput | ModelRunnerOutput,
        event: torch.cuda.Event,
        completion: StateCompletion,
    ):
        self.output = output
        self.event = event
        self.completion = completion

    def get_output(self) -> ModelRunnerOutput:
        self.event.synchronize()
        output = (
            self.output.get_output()
            if isinstance(self.output, AsyncModelRunnerOutput)
            else self.output
        )
        output = copy(output)
        output.state_completion = self.completion
        return output
