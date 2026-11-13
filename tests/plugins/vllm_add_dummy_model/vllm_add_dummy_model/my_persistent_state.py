# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A deterministic persistent-state model for end-to-end tests.

The model keeps one counter per state slot. Each scheduled step of a request
advances its counter by one and emits a token that is a function of the
counter and the request's last input token; every third step emits EOS, so a
resumable request parks after two tokens and its next input continues from
the counter the slot kept. A slot that is reused without its release fence or
without the initializer, or that is cleared or re-created mid-stream, changes
the output, which is what the regression, pause and cache-clear tests rely on.

`hf_overrides` fields select what a test needs:

- `stub_page_mib`: the state page size in MiB (default 1).
- `stub_capture_graphs`: the model state captures one CUDA graph over its
  count pool in `capture_cudagraphs`, replays it once on a stream of its own,
  and `needs_cudagraph_capture` returns `True`.
- `stub_capture_mib`: scratch memory, in MiB, that each capture keeps beside
  its graph (default 0).
- `stub_execution_mib`: memory, in MiB, that `prepare_execution_memory`
  allocates and keeps (default 0).
- `stub_memory_contributors`: the spec's `memory_contributors`.
- `stub_fail`: the hooks that raise, from `prepare_execution_memory`,
  `profile_run` (the profiling forward pass), `capture_profile`,
  `capture_serving`, `clear_cudagraphs`, `post_warmup` and `shutdown`. A hook
  raises `StubHookError` naming itself, after it has done its work.
- `stub_fail_oom`: the two capture failures raise `torch.OutOfMemoryError`
  instead.

`EVENTS` records, in order, the hook calls and forward passes of every stub
built in this process, so an in-process engine test can tell how far startup
got even when it fails. `BINDINGS` records each `initialize_state_slots` call
and `RAISED` each exception a hook raised.
"""

import gc
import random
import weakref
from typing import Any

import torch
import torch.nn as nn

from vllm.compilation.cudagraph_pool import capture_pool
from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.model_executor.layers.state import PersistentStateLayerBase
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils import jit_monitor
from vllm.v1.kv_cache_interface import (
    PersistentStateDescriptor,
    PersistentStateSpec,
)
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.workspace import current_workspace_manager

STATE_LAYER = "state"
TOKENS_PER_TURN = 2
MiB = 1 << 20

# ("initialize_state_slots",), ("prepare_step", dummy_run, is_profile,
# slots_bound) and ("forward", dummy_run) records, and one (name, info) record
# per call of "needs_cudagraph_capture", "prepare_execution_memory",
# "profile_run", "capture_cudagraphs", "clear_cudagraphs", "post_warmup" and
# "shutdown", where info is a dict of what the hook observed.
EVENTS: list[tuple[Any, ...]] = []

# One dict per `initialize_state_slots` call: a weak reference to the bindings
# ("ref"), their page count ("pages"), whether the bindings of the previous
# call were still alive ("previous_alive") and whether the model state's
# `state_slots` had been cleared before the call ("slots_cleared").
BINDINGS: list[dict[str, Any]] = []

# The exceptions that hooks raised, in order.
RAISED: list[BaseException] = []


class StubHookError(RuntimeError):
    """What a stub hook raises when `stub_fail` names it."""


def allocator_settings() -> str:
    """The device allocator's settings string, as the worker reads it; empty
    when the allocator has no such settings."""
    if (
        not current_platform.is_cuda_alike()
        or torch.cuda.memory.get_allocator_backend() != "native"
    ):
        return ""
    get_settings = getattr(torch._C, "_accelerator_getAllocatorSettings", None)
    if get_settings is not None:
        return get_settings()
    return torch.cuda.memory._snapshot()["allocator_settings"][
        "PYTORCH_CUDA_ALLOC_CONF"
    ]


def _record(name: str, **info: Any) -> dict[str, Any]:
    EVENTS.append((name, info))
    return info


def expected_tokens(
    counter: int, last_input_token: int, vocab_size: int, eos_token_id: int
) -> list[int]:
    """The tokens one turn emits from `counter` (steps already taken) after an
    input ending in `last_input_token`; the EOS that parks the turn is last."""
    tokens = []
    last = last_input_token
    for _ in range(TOKENS_PER_TURN + 1):
        counter += 1
        last = next_token(counter, last, vocab_size, eos_token_id)
        tokens.append(last)
    return tokens


def next_token(counter: int, last_token: int, vocab_size: int, eos_token_id: int):
    if counter % (TOKENS_PER_TURN + 1) == 0:
        return eos_token_id
    return 3 + (counter * 7919 + last_token) % (vocab_size - 3)


class StubPersistentStateModelState(ModelState):
    """Records the hook calls and binds the step's slot rows for the model."""

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        self._handles: dict[str, Any] = {}
        hf = self.model_config.hf_config
        self._captures = bool(getattr(hf, "stub_capture_graphs", False))
        self._capture_bytes = int(getattr(hf, "stub_capture_mib", 0) * MiB)
        self._execution_bytes = int(getattr(hf, "stub_execution_mib", 0) * MiB)
        self._fail = frozenset(getattr(hf, "stub_fail", ()))
        self._fail_oom = bool(getattr(hf, "stub_fail_oom", False))
        self._execution_memory: torch.Tensor | None = None
        # The captured graph, its output and its scratch memory.
        self._graph: tuple[Any, ...] | None = None
        self._access: Any = None
        self._layout: Any = None

    def _maybe_fail(self, hook: str) -> None:
        if hook not in self._fail:
            return
        error: BaseException = StubHookError(f"stub {hook} failed")
        if self._fail_oom and hook.startswith("capture_"):
            error = torch.OutOfMemoryError(f"stub {hook} failed")
        RAISED.append(error)
        raise error

    def _bindings_index(self) -> int | None:
        """Which `initialize_state_slots` call gave the current bindings."""
        for index, binding in enumerate(BINDINGS):
            if self.state_slots is not None and binding["ref"]() is self.state_slots:
                return index
        return None

    def _stale_access_refused(self) -> bool | None:
        """Whether the access object of the last capture is refused now that
        the capture call has ended; `None` when no capture took one."""
        access, self._access = self._access, None
        if access is None:
            return None
        try:
            access.component_pools  # noqa: B018
        except ValueError:
            return True
        return False

    def _on_main_stream(self) -> bool:
        return torch.cuda.current_stream(self.device) == torch.cuda.default_stream(
            self.device
        )

    def initialize_state_slots(self, bindings) -> None:
        previous = BINDINGS[-1]["ref"]() if BINDINGS else None
        BINDINGS.append(
            {
                "ref": weakref.ref(bindings),
                "pages": bindings.views[STATE_LAYER].capacity,
                "previous_alive": previous is not None,
                "slots_cleared": self.state_slots is None,
            }
        )
        super().initialize_state_slots(bindings)
        EVENTS.append(("initialize_state_slots",))

    def prepare_step(self, scheduler_output, *, dummy_run: bool, is_profile: bool):
        EVENTS.append(
            ("prepare_step", dummy_run, is_profile, self.state_slots is not None)
        )
        self._handles = {} if dummy_run else (scheduler_output.state_slot_handles or {})
        if is_profile:
            _record("profile_run", allocator_settings=allocator_settings())
            self._maybe_fail("profile_run")

    def prepare_execution_memory(self) -> None:
        _record(
            "prepare_execution_memory",
            slots_bound=self.state_slots is not None,
            config_set=get_current_vllm_config_or_none() is not None,
            allocator_settings=allocator_settings(),
            allocated_bytes=self._execution_bytes,
        )
        self._execution_memory = torch.empty(
            self._execution_bytes, dtype=torch.uint8, device=self.device
        )
        self._maybe_fail("prepare_execution_memory")

    def needs_cudagraph_capture(self) -> bool:
        _record("needs_cudagraph_capture", slots_bound=self.state_slots is not None)
        return self._captures and current_platform.is_cuda()

    def capture_cudagraphs(self, *, graph_pool, stream, profile_only: bool) -> None:
        bindings = self.state_slots
        assert bindings is not None
        info = _record(
            "capture_cudagraphs",
            profile_only=profile_only,
            graph_pool=graph_pool,
            global_graph_pool=current_platform.get_global_graph_pool(),
            stream_is_current=torch.cuda.current_stream(self.device) == stream,
            stream_is_main=stream == torch.cuda.default_stream(self.device),
            bindings=self._bindings_index(),
            workspace_locked=current_workspace_manager().is_locked(),
        )
        # The validation replay runs on a stream of the model's own.
        own_stream = torch.cuda.Stream(self.device)
        try:
            self._access = bindings.startup_pool_access(STATE_LAYER)
            counts = self._access.component_pools["count"]
            info["pool_pages"] = counts.shape[0]
            # Model-owned memory: the capture reads the pool and stores here.
            scratch = torch.zeros(
                self._capture_bytes, dtype=torch.uint8, device=self.device
            )
            info["scratch_bytes"] = scratch.numel()
            output = torch.zeros(counts.shape[0], dtype=torch.int64, device=self.device)
            graph = torch.cuda.CUDAGraph()
            with (
                capture_pool(graph_pool, self.vllm_config) as pool,
                torch.cuda.graph(graph, pool=pool, stream=stream),
            ):
                output.copy_(counts[:, 0] + 1)
            self._graph = (graph, output, scratch)
            own_stream.wait_stream(stream)
            with torch.cuda.stream(own_stream):
                graph.replay()
            if not profile_only:
                self._layout = bindings.pool_layout(STATE_LAYER)
            self._maybe_fail("capture_profile" if profile_only else "capture_serving")
        finally:
            # Whether the call returns or raises, the device work it started
            # on either stream has finished.
            own_stream.synchronize()
            stream.synchronize()
            info["work_completed"] = own_stream.query() and stream.query()

    def clear_cudagraphs(self) -> None:
        _record(
            "clear_cudagraphs",
            bindings=self._bindings_index(),
            held_graph=self._graph is not None,
            stale_access_refused=self._stale_access_refused(),
            workspace_locked=current_workspace_manager().is_locked(),
            on_main_stream=self._on_main_stream(),
        )
        self._graph = None
        self._layout = None
        self._maybe_fail("clear_cudagraphs")

    def post_warmup(self) -> None:
        info = _record(
            "post_warmup",
            bindings=self._bindings_index(),
            stale_access_refused=self._stale_access_refused(),
            workspace_locked=current_workspace_manager().is_locked(),
            on_main_stream=self._on_main_stream(),
            jit_monitor_active=jit_monitor.is_active(),
            gc_frozen_objects=gc.get_freeze_count(),
        )
        if self._layout is not None and self.state_slots is not None:
            # Nothing the captured graph reads moved during warmup.
            layout = self.state_slots.pool_layout(STATE_LAYER)
            info["layout_unchanged"] = dict(layout) == dict(self._layout)
        # The worker resets the seed after this hook, so drawing is allowed.
        torch.rand(8)
        random.random()
        self._maybe_fail("post_warmup")

    def shutdown(self) -> None:
        bindings = self.state_slots
        _record(
            "shutdown",
            slots_bound=bindings is not None,
            pool_pages=None
            if bindings is None
            else bindings.views[STATE_LAYER].capacity,
            stale_access_refused=self._stale_access_refused(),
            device_idle=torch.cuda.current_stream(self.device).query(),
        )
        self._graph = None
        self._layout = None
        self._execution_memory = None
        self._maybe_fail("shutdown")

    def prepare_inputs(self, input_batch, req_states) -> dict[str, Any]:
        if self.state_slots is None or not self._handles:
            return {}
        num_reqs = input_batch.num_reqs
        request_ids = tuple(input_batch.req_ids[:num_reqs])
        handles = tuple(self._handles[rid][STATE_LAYER] for rid in request_ids)
        rows = self.state_slots.bind_rows(STATE_LAYER, request_ids, handles)
        return {
            # Pinned and non-blocking, so that the copy does not wait on the
            # device.
            "slot_indices": torch.tensor(rows.slot_indices_cpu)
            .pin_memory()
            .to(self.device, non_blocking=True),
            "count_pool": rows.component_pools["count"],
            "query_start_loc": input_batch.query_start_loc,
            "num_reqs": num_reqs,
        }

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        return {"dummy_run": True}

    def prepare_inputs_embeds(self, scheduled_encoder_inputs, input_batch, req_states):
        return None

    def prepare_attn(self, *args, **kwargs) -> dict[str, Any]:
        return {}


class StubPersistentStateForCausalLM(nn.Module):
    """One counter per slot; see the module docstring."""

    supports_persistent_state = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        model_config = vllm_config.model_config
        hf = model_config.hf_config
        self.vocab_size = model_config.get_vocab_size()
        self.eos_token_id = int(hf.eos_token_id)
        page_bytes = int(getattr(hf, "stub_page_mib", 1) * MiB)
        spec = PersistentStateSpec(
            descriptors=(
                PersistentStateDescriptor(
                    "count", (1,), torch.int64, 0, 8, torch.zeros
                ),
                PersistentStateDescriptor(
                    "filler", (page_bytes - 8,), torch.uint8, 8, 8, torch.zeros
                ),
            ),
            page_size_bytes=page_bytes,
            memory_contributors=tuple(getattr(hf, "stub_memory_contributors", ())),
        )
        self.state = PersistentStateLayerBase(
            spec, prefix=f"{prefix}{STATE_LAYER}", vllm_config=vllm_config
        )
        self.embed = nn.Embedding(self.vocab_size, 8)

    @classmethod
    def get_model_state_cls(cls) -> type[ModelState]:
        return StubPersistentStateModelState

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        slot_indices: torch.Tensor | None = None,
        count_pool: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        num_reqs: int = 0,
        dummy_run: bool = False,
    ) -> torch.Tensor:
        assert input_ids is not None
        EVENTS.append(("forward", dummy_run))
        hidden = torch.zeros(
            input_ids.shape[0], 8, dtype=torch.float32, device=input_ids.device
        )
        if slot_indices is None or count_pool is None or query_start_loc is None:
            return hidden
        last = query_start_loc[1 : num_reqs + 1] - 1
        counters = count_pool[slot_indices, 0] + 1
        count_pool[slot_indices, 0] = counters
        last_tokens = input_ids[last].long()
        tokens = torch.where(
            counters % (TOKENS_PER_TURN + 1) == 0,
            torch.full_like(counters, self.eos_token_id),
            3 + (counters * 7919 + last_tokens) % (self.vocab_size - 3),
        )
        hidden[last, 0] = tokens.to(torch.float32)
        return hidden

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(
            hidden_states.shape[0],
            self.vocab_size,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        logits.scatter_(1, hidden_states[:, :1].long(), 1.0)
        return logits

    def load_weights(self, weights) -> set[str]:
        loaded = set()
        params = dict(self.named_parameters())
        for name, weight in weights:
            if name in params:
                params[name].data.copy_(weight)
                loaded.add(name)
        return loaded


class StubUndeclaredPersistentStateForCausalLM(StubPersistentStateForCausalLM):
    """The same model without the persistent-state declaration."""

    supports_persistent_state = False
