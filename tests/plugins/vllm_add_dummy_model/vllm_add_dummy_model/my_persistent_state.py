# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A deterministic model whose only cache is fixed-size per-request state.

`StubStateLayer` keeps two components in a `MambaSpec` page allocated by
`MambaManager`: the sum of the values a request has fed it (starting from
`stub_initial_total`) and the number of decode steps in the current turn,
which starts at `TURN_LEN - 1`, no turn in progress. A turn ends with EOS
after `TURN_LEN - 1` sampled tokens. Core schedules a single-token input as a
decode; on a row with no turn in progress, that token starts the next turn. A
page that is not initialized on allocation, a state lost between streaming
inputs, or a state write for an input the request never commits changes the
output.

The model reads these `hf_overrides` entries:

- `stub_initial_total`: the first component's initial value (default 0).
- `stub_reads_positions`: add the token position to what each emitted token
  depends on, so a wrong position changes the output.
- `stub_page_elems`: elements per state component in a page (default 1).
  A test that lets core size the KV cache from device memory raises it, so a
  page is not a few bytes and the block count stays small.

`StubChunkedPrefillPersistentStateForCausalLM` keeps no turns: every row,
prefill or decode, adds its tokens to the sum and emits from it, and a
request ends at `max_tokens`. A one-token chunk on the decode path then does
what finishing its prefill does, so the model declares
`persistent_state_chunked_prefill`.

`StubModelState` records its hook calls in `EVENTS` (readable only where the
engine runs in the test's process), and reports a request whose prompt holds
`FAIL_TOKEN` as failed after its first step.

The state update breaks a breakable CUDA graph capture and runs eagerly, as a
model's state kernel does, so a piecewise graph reads each replayed batch's
metadata.

`StubModelState.capture_cudagraphs` also captures one CUDA graph, over block
id 0, that reads and updates the state pages a device block-id tensor names,
as a model's transaction graph does. It never names another page at capture.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.models.interfaces import (
    IsAttentionFree,
    SupportsPersistentState,
)
from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import KVCacheLayerConfig, MambaSpec
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

TURN_LEN = 3
"""Tokens per turn, the closing EOS included."""

FAIL_TOKEN = 999
"""A prompt token that makes `StubModelState` report its request as failed."""

EVENTS: list[tuple] = []
"""Model-state hook calls: ("init",), ("add", req_id), ("remove", req_id),
("prepare_execution_memory",), ("capture", profile_only), ("clear",)."""

PREFILL_STARTS: list[int] = []
"""The state sum each prefill row starts from, recorded in eager forwards."""

PAGES: dict[str, list[tuple[int, tuple[int, ...], int]]] = {}
"""Per request id, each time the model state takes the request (once per
turn): the state block id, the addresses of its components' pages read
through `state_group_layouts`, and the turn's `position_offset`, which
compaction raises."""

PROMPTS: dict[str, tuple[int, ...]] = {}
"""Per request id, the prompt of its first turn."""

LAYOUTS_AT_CAPTURE: list[tuple] = []
"""(profile_only, `describe_layouts(state_group_layouts)`) at each
`capture_cudagraphs` call."""

CAPTURES: list[tuple[bool, tuple[int, ...], tuple[int, ...]]] = []
"""(profile_only, the block ids the captured graph addressed, the page count of
each component it addressed) at each `capture_cudagraphs` call on a CUDA
device."""

FORWARDS: list[tuple[int, int]] = []
"""(decode rows, prefill rows) of each model forward that ran eagerly with
metadata for the state layer; a replayed graph runs no Python forward."""

EOS_TOKEN_ID = 2


def emit(total, vocab_size: int):
    return 3 + total * 7919 % (vocab_size - 3)


def expected_turns(
    inputs: Sequence[Sequence[int]],
    vocab_size: int,
    *,
    initial_total: int = 0,
    reads_positions: bool = False,
) -> list[list[int]]:
    """The tokens each streaming input produces, EOS excluded.

    A turn's EOS is never fed back, and the next input replaces it. With
    `reads_positions`, each token depends on the absolute position of the
    token that produced it, which no compaction may change.
    """
    total = initial_total
    fed = 0
    turns = []
    for prompt in inputs:
        total += sum(prompt)
        fed += len(prompt)
        turn = [emit(total + (fed - 1) * reads_positions, vocab_size)]
        for _ in range(TURN_LEN - 2):
            total += turn[-1]
            turn.append(emit(total + fed * reads_positions, vocab_size))
            fed += 1
        # The step that samples EOS feeds the turn's last token.
        total += turn[-1]
        fed += 1
        turns.append(turn)
    return turns


def expected_sums(
    prompt: Sequence[int],
    vocab_size: int,
    num_tokens: int,
    *,
    initial_total: int = 0,
    reads_positions: bool = False,
) -> tuple[list[int], int]:
    """The tokens a request of `StubChunkedPrefillPersistentStateForCausalLM`
    produces from `prompt` until it stops at `num_tokens`, and the sum its
    page then holds: each token feeds the sum, except the last, which no step
    feeds."""
    total = initial_total + sum(prompt)
    position = len(prompt) - 1
    tokens = [emit(total + position * reads_positions, vocab_size)]
    for _ in range(num_tokens - 1):
        total += tokens[-1]
        position += 1
        tokens.append(emit(total + position * reads_positions, vocab_size))
    return tokens, total


def without_eos(token_ids: Sequence[int]) -> list[int]:
    token_ids = list(token_ids)
    if token_ids and token_ids[-1] == EOS_TOKEN_ID:
        token_ids.pop()
    return token_ids


def describe_layouts(layouts) -> tuple:
    """The state-group layouts as plain values, which hold no tensor."""
    return tuple(
        (
            layout.group_id,
            tuple(
                (
                    name,
                    tuple(
                        (
                            c.tensor.data_ptr(),
                            c.dtype,
                            c.tensor.shape[0],
                            c.page_shape,
                            c.page_stride,
                        )
                        for c in components
                    ),
                )
                for name, components in layout.layers.items()
            ),
        )
        for layout in layouts
    )


class StubStateLayer(MambaBase, nn.Module):
    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.PERSISTENT_STATE

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix
        self.vocab_size = vllm_config.model_config.get_vocab_size()
        hf_config = vllm_config.model_config.hf_config
        self.initial_total = int(getattr(hf_config, "stub_initial_total", 0))
        self.page_elems = int(getattr(hf_config, "stub_page_elems", 1))
        self.reads_positions = bool(getattr(hf_config, "stub_reads_positions", False))
        self.counts_turns = True
        vllm_config.compilation_config.static_forward_context[prefix] = self

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return ((self.page_elems,), (self.page_elems,))

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return (torch.int64, torch.int64)

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig:
        spec = MambaSpec(
            shapes=self.get_state_shape(),
            dtypes=self.get_state_dtype(),
            block_size=vllm_config.model_config.max_model_len,
            mamba_type=self.mamba_type,
            initial_values=(self.initial_total, TURN_LEN - 1),
            tp_replicated=True,
        )
        return MambaManager.layer_config(spec, recomputable=False)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Add each request's tokens to its sum, and return the token to emit
        at each request's last position."""
        out = torch.zeros(input_ids.shape[0], 1, device=input_ids.device)
        _update_stub_state(input_ids, positions, out, self.prefix)
        return out

    def update(
        self, input_ids: torch.Tensor, positions: torch.Tensor | None, out: torch.Tensor
    ) -> None:
        attn_metadata = get_forward_context().attn_metadata
        # Warm-up runs carry no metadata for this layer.
        if not isinstance(attn_metadata, dict) or self.prefix not in attn_metadata:
            return
        metadata = attn_metadata[self.prefix]
        assert isinstance(metadata, BaseMambaAttentionMetadata)
        totals = self.kv_cache[0][:, 0]
        counts = self.kv_cache[1][:, 0]
        if self.reads_positions:
            assert positions is not None
            offsets = positions.to(torch.int64)
        else:
            offsets = torch.zeros_like(input_ids, dtype=torch.int64)

        if num_decodes := metadata.num_decodes:
            assert metadata.state_indices_tensor_d is not None
            slots = metadata.state_indices_tensor_d[:num_decodes, 0]
            if self.counts_turns:
                # A single-token input to a row whose turn ended starts a turn.
                new_turn = counts[slots] >= TURN_LEN - 1
                counts[slots] = torch.where(new_turn, 0, counts[slots] + 1)
            totals[slots] += input_ids[:num_decodes]
            token = emit(totals[slots] + offsets[:num_decodes], self.vocab_size)
            if self.counts_turns:
                token = torch.where(counts[slots] >= TURN_LEN - 1, EOS_TOKEN_ID, token)
            out[:num_decodes, 0] = token.to(out.dtype)

        if num_prefills := metadata.num_prefills:
            assert metadata.state_indices_tensor_p is not None
            assert metadata.query_start_loc_p is not None
            assert metadata.has_initial_states_p is not None
            slots = metadata.state_indices_tensor_p[:num_prefills]
            start = metadata.num_decode_tokens
            bounds = start + metadata.query_start_loc_p
            if not torch.cuda.is_current_stream_capturing():
                starts = totals[slots] * metadata.has_initial_states_p
                PREFILL_STARTS.extend(starts.tolist())
            cumsum = nn.functional.pad(input_ids.cumsum(0), (1, 0))
            totals[slots] = (
                totals[slots] * metadata.has_initial_states_p
                + cumsum[bounds[1:]]
                - cumsum[bounds[:-1]]
            )
            if self.counts_turns:
                counts[slots] = 0
            last = bounds[1:] - 1
            out[last, 0] = emit(totals[slots] + offsets[last], self.vocab_size).to(
                out.dtype
            )


@eager_break_during_capture
def _update_stub_state(
    input_ids: torch.Tensor,
    positions: torch.Tensor | None,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    layer.update(input_ids, positions, out)


class StateTransactionGraph:
    """A CUDA graph over the state pages that `block_ids` names, one row per
    request: it reads each row's page into `out` (the first element of every
    component) and adds the row's `deltas` entry to the page.

    It is captured with every row on block id 0, with zero deltas, and takes
    the pages it runs on from `block_ids` at replay, so it needs no recapture
    when the KV cache grows. `run` is the same update, eagerly.
    """

    def __init__(self, layouts, num_rows: int, device: torch.device) -> None:
        self.tensors = [
            component.tensor
            for layout in layouts
            for components in layout.layers.values()
            for component in components
        ]
        self.num_pages = tuple(tensor.shape[0] for tensor in self.tensors)
        self.block_ids = torch.zeros(num_rows, dtype=torch.int64, device=device)
        self.deltas = torch.zeros(num_rows, dtype=torch.int64, device=device)
        self.out = torch.zeros(
            len(self.tensors), num_rows, dtype=torch.int64, device=device
        )
        self.graph: torch.cuda.CUDAGraph | None = None

    def run(self) -> None:
        with torch.inference_mode():
            for k, tensor in enumerate(self.tensors):
                pages = tensor.index_select(0, self.block_ids)
                self.out[k] = pages[:, 0]
                tensor.index_copy_(
                    0, self.block_ids, pages + self.deltas.unsqueeze(1).to(pages.dtype)
                )

    def capture(self, graph_pool, stream: torch.cuda.Stream) -> None:
        # Warm up on the capture stream. Block id 0 only, and a zero delta,
        # leave the null block unchanged.
        with torch.cuda.stream(stream):
            self.run()
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=graph_pool, stream=stream):
            self.run()

    def set_inputs(self, block_ids: Sequence[int], deltas: Sequence[int]) -> None:
        with torch.inference_mode():
            self.block_ids.copy_(torch.tensor(block_ids, dtype=torch.int64))
            self.deltas.copy_(torch.tensor(deltas, dtype=torch.int64))

    def replay(self) -> None:
        assert self.graph is not None
        self.graph.replay()


class StubModelState(MambaHybridModelState):
    """Records its hook calls and fails the requests that ask for it."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._failed: set[str] = set()
        self.state_graph: StateTransactionGraph | None = None
        EVENTS.append(("init",))

    def add_request(self, req_index, new_req_data) -> None:
        super().add_request(req_index, new_req_data)
        EVENTS.append(("add", new_req_data.req_id))
        for layout in self.state_group_layouts:
            # The state row reads the first block of its group.
            block_id = new_req_data.block_ids[layout.group_id][0]
            addresses = tuple(
                component.page(block_id).data_ptr()
                for components in layout.layers.values()
                for component in components
            )
            PAGES.setdefault(new_req_data.req_id, []).append(
                (block_id, addresses, new_req_data.position_offset)
            )
        PROMPTS.setdefault(
            new_req_data.req_id, tuple(new_req_data.prompt_token_ids or ())
        )
        if FAIL_TOKEN in (new_req_data.prompt_token_ids or ()):
            self._failed.add(new_req_data.req_id)

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        EVENTS.append(("remove", req_id))
        self._failed.discard(req_id)

    def take_failed_requests(self) -> set[str]:
        failed, self._failed = self._failed, set()
        return failed

    def prepare_execution_memory(self) -> None:
        super().prepare_execution_memory()
        self.workspace = torch.empty(1 << 20, dtype=torch.uint8, device=self.device)
        EVENTS.append(("prepare_execution_memory",))

    def needs_cudagraph_capture(self) -> bool:
        return True

    def capture_cudagraphs(self, *, graph_pool, stream, profile_only) -> None:
        EVENTS.append(("capture", profile_only))
        layouts = self.state_group_layouts
        LAYOUTS_AT_CAPTURE.append((profile_only, describe_layouts(layouts)))
        if self.device.type != "cuda" or not layouts:
            return
        graph = StateTransactionGraph(layouts, self.max_num_reqs, self.device)
        CAPTURES.append(
            (profile_only, tuple(graph.block_ids.tolist()), graph.num_pages)
        )
        graph.capture(graph_pool, stream)
        self.state_graph = graph

    def clear_cudagraphs(self) -> None:
        EVENTS.append(("clear",))
        self.state_graph = None

    def shutdown(self) -> None:
        # The graph holds views of the KV cache, which shutdown releases.
        self.state_graph = None
        super().shutdown()


class StubPersistentStateForCausalLM(
    nn.Module, IsAttentionFree, SupportsPersistentState
):
    """See the module docstring."""

    persistent_state_reads_positions = True
    persistent_state_accepts_synthetic_warmup = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vocab_size = vllm_config.model_config.get_vocab_size()
        self.state = StubStateLayer(vllm_config, prefix=f"{prefix}state")

    @staticmethod
    def get_model_state_cls() -> type[StubModelState]:
        return StubModelState

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(input_ids.shape[0], 1, device=input_ids.device)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if (
            not torch.cuda.is_current_stream_capturing()
            and isinstance(attn_metadata, dict)
            and self.state.prefix in attn_metadata
        ):
            metadata = attn_metadata[self.state.prefix]
            FORWARDS.append((metadata.num_decodes, metadata.num_prefills))
        return self.state(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(
            hidden_states.shape[0], self.vocab_size, device=hidden_states.device
        )
        return logits.scatter_(1, hidden_states.long(), 1.0)

    def load_weights(self, weights) -> set[str]:
        return set()


class StubNoWarmupPersistentStateForCausalLM(StubPersistentStateForCausalLM):
    """The same model declaring that it ignores positions and refuses synthetic
    warmup, for the tests that run a model with those facts."""

    persistent_state_reads_positions = False
    persistent_state_accepts_synthetic_warmup = False


class StubChunkedPrefillPersistentStateForCausalLM(StubPersistentStateForCausalLM):
    """The same model without turns, declaring chunked prefill: every row adds
    its tokens to the sum, so any split of an input into prefill chunks,
    including a one-token chunk that runs on the decode path, gives the
    unsplit output and page (PSTATE-DECL-017)."""

    persistent_state_chunked_prefill = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.state.counts_turns = False
