# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in whole-transaction graph capture for declared row populations.

One graph per declared cell covers the device body of ``advance_model_rows``
and its predicated page scatters. Nested encoder, decoder and CHUNK bindings run
eagerly inside the capture, so the graph records their kernels directly. Every
host vector the body reads comes from a persistent arena uploaded with one
host-to-device copy before replay. The reservation happens before replay and the
status stage after it. Calls outside the captured cells return ``None`` and the
caller runs the eager path; such calls are counted per cell and reason.

The declared inventory is captured twice per worker through one capture path:
over stand-in storage inside the memory-profiling window, which admits its
retained memory before resident capacity is fixed, then once over the resident
pools for serving. A failure from the replay onward, and every worker-fatal
check, latches the binding.
"""

from __future__ import annotations

import gc
import os
import time
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.models.nemotron_asr.advance import (
    CommitPlan,
    CommitSink,
    LiveTransactionStager,
    PreparedRowBinding,
    RowPlan,
    TransactionInputs,
    TransactionPools,
    bucket_vector_name,
    consume_batch_stats,
    execute_transaction_scatters,
    prepare_transaction,
    reserve_transaction_commit,
    run_transaction_body,
)
from vllm.models.nemotron_asr.capture_scope import outer_transaction_capture
from vllm.models.nemotron_asr.decode_graph import GraphRuntime, platform_graph_runtime

CELL_FRESH_CHUNK = "chunk-fresh"
CELL_CONTINUING_CHUNK = "chunk-continuing"
CELL_REPLAY = "replay"

#: The option every configuration and startup error names.
OPTION = "transaction_graph_populations"
#: The one served geometry: the 80 ms cadence, ``CADENCES`` index 0 (g0).
SERVED_GEOMETRY = 0


@dataclass(frozen=True, order=True)
class TransactionCell:
    """A declared transaction shape: row kind, CHUNK geometry and population.

    ``chunk-fresh`` and ``chunk-continuing`` are one geometry bucket whose rows
    are all fresh or all continuing. ``replay`` has no CHUNK row; it covers
    REPLAY, park-echo, FLUSH and forced-endpoint rows of any geometry.
    """

    kind: str
    geometry: int
    population: int

    @property
    def label(self) -> str:
        if self.kind == CELL_REPLAY:
            return f"{self.kind}:b{self.population}"
        return f"{self.kind}:g{self.geometry}:b{self.population}"


def classify_transaction(
    inputs: TransactionInputs,
) -> tuple[TransactionCell | None, str | None]:
    """Return the call's cell, or the reason it has none."""
    n_real = inputs.n_real
    if n_real <= 0:
        return None, "empty"
    if not inputs.buckets:
        if inputs.n_fresh:
            return None, "fresh-non-chunk"
        return TransactionCell(CELL_REPLAY, -1, n_real), None
    if len(inputs.buckets) != 1:
        return None, "multi-geometry"
    bucket = inputs.buckets[0]
    if int(bucket.positions.numel()) != n_real:
        return None, "mixed-roles"
    if inputs.n_fresh == n_real:
        return TransactionCell(CELL_FRESH_CHUNK, bucket.geometry, n_real), None
    if inputs.n_fresh == 0:
        return TransactionCell(CELL_CONTINUING_CHUNK, bucket.geometry, n_real), None
    return None, "mixed-fresh"


def _callable_identity(fn: Any) -> tuple[int, int]:
    """Identity stable across repeated bound-method attribute reads."""
    return id(getattr(fn, "__self__", None)), id(getattr(fn, "__func__", fn))


def _tensor_layout(tensor: torch.Tensor) -> tuple[Any, ...]:
    return (
        int(tensor.data_ptr()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _lazy_buffer_signature(core: Any) -> tuple[tuple[str, Any], ...]:
    """Named layouts of the encoder buffers eager calls may rebind lazily."""
    encoder = getattr(core, "encoder", None)
    pe = getattr(getattr(encoder, "pos_enc", None), "pe", None)
    return (
        ("encoder.pos_enc.pe", None if pe is None else _tensor_layout(pe)),
        (
            "encoder._stream_relative_position_lengths",
            tuple(getattr(encoder, "_stream_relative_position_lengths", ())),
        ),
    )


def _arena_layout(
    inputs: TransactionInputs,
) -> tuple[tuple[str, torch.dtype, tuple[int, ...]], ...]:
    return tuple(
        (name, tensor.dtype, tuple(tensor.shape))
        for name, tensor in inputs.staged_vectors()
    )


def transaction_signature(
    core: Any,
    inputs: TransactionInputs,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    *,
    graph_covers_decode: bool,
) -> tuple[Any, ...]:
    """Every structural input a captured body freezes, arena layout included."""
    return (
        inputs.n_real,
        inputs.hidden,
        inputs.queue_capacity,
        inputs.blank_id,
        inputs.endpoint_enabled,
        inputs.eou_token_id,
        inputs.placeholder_id,
        inputs.park_id,
        graph_covers_decode,
        inputs.n_fresh,
        tuple(
            (
                bucket.geometry,
                int(bucket.positions.numel()),
                bucket.n_fresh,
                _callable_identity(bucket.resolved.decode_fn),
                _callable_identity(bucket.transition),
            )
            for bucket in inputs.buckets
        ),
        _callable_identity(inputs.adapter),
        _callable_identity(inputs.encoder_transition),
        tuple(_tensor_layout(pool) for pool in inputs.pools.sequence()),
        _lazy_buffer_signature(core),
        (tuple(input_ids.shape), input_ids.dtype, input_ids.device),
        (tuple(inputs_embeds.shape), inputs_embeds.dtype, inputs_embeds.device),
        _arena_layout(inputs),
    )


#: The names of :func:`transaction_signature`'s entries, in order.
_SIGNATURE_FIELDS = (
    "population",
    "carrier width",
    "replay queue capacity",
    "blank id",
    "endpoint_enabled",
    "EOU token id",
    "placeholder id",
    "park id",
    "decode coverage",
    "fresh row count",
    "CHUNK buckets and their decode and transition callables",
    "emission adapter",
    "encoder transition",
    "resident pools",
    "lazily bound encoder buffers",
    "input ids",
    "input embeddings",
    "arena layout",
)

#: The names of a startup key's call-binding entries, in order.
_CALL_FIELDS = (
    "adapter",
    "decode_resolver",
    "encoder_transition",
    "bucket_transition",
    "execution_tier",
    "placeholder_id",
    "park_id",
    "eou_token_id",
    "graph_covers_decode",
    "hidden_size",
    "device",
    "null_block_id",
)


def _pool_names(pools: TransactionPools) -> tuple[str, ...]:
    """Names of ``pools.sequence()``'s entries, in order."""
    names = [
        f"{family}.{index}"
        for family, members in (
            ("channel", pools.channel),
            ("time", pools.time),
            ("valid_length", pools.valid_length),
        )
        for index in range(len(members))
    ]
    names += ("h", "c", "queue", "book")
    names += ("frontend_raw", "frontend_mel", "frontend_counter")
    names += [
        name
        for name in ("endpoint_history", "endpoint_book")
        if getattr(pools, name) is not None
    ]
    return tuple(names)


def _first_change(
    names: Sequence[str], captured: Sequence[Any], live: Sequence[Any]
) -> str | None:
    """The name of the first entry that differs, or ``None`` if none does."""
    if len(captured) != len(live):
        return "inventory"
    for name, old, new in zip(names, captured, live, strict=True):
        if old != new:
            return name
    return None


def _signature_change(
    captured: tuple[Any, ...], live: tuple[Any, ...], pools: TransactionPools
) -> str | None:
    """Describe the first signature entry that differs, naming its buffer."""
    field = _first_change(_SIGNATURE_FIELDS, captured, live)
    if field == "resident pools":
        index = _SIGNATURE_FIELDS.index(field)
        pool = _first_change(_pool_names(pools), captured[index], live[index])
        return f"resident pool {pool}"
    if field == "lazily bound encoder buffers":
        index = _SIGNATURE_FIELDS.index(field)
        names = [name for name, _ in live[index]]
        buffer = _first_change(names, captured[index], live[index])
        return f"lazily bound buffer {buffer}"
    return field


class TransactionArena:
    """Persistent staging for one cell's host vectors.

    Two pinned host buffers alternate between calls, and each records an event
    after its upload, so a buffer is rewritten only after its previous copy
    completed. One device buffer holds the views the captured body reads.
    """

    _DTYPES = (torch.int64, torch.bool)

    def __init__(
        self,
        layout: Sequence[tuple[str, torch.dtype, tuple[int, ...]]],
        device: torch.device,
    ) -> None:
        if any(dtype not in self._DTYPES for _, dtype, _ in layout):
            raise ValueError("transaction arena holds only int64 and bool vectors")
        # int64 segments first keep every segment naturally aligned.
        ordered = sorted(layout, key=lambda item: item[1] != torch.int64)
        offsets: list[tuple[str, torch.dtype, tuple[int, ...], int, int]] = []
        cursor = 0
        for name, dtype, shape in ordered:
            count = 1
            for dimension in shape:
                count *= int(dimension)
            nbytes = count * (8 if dtype == torch.int64 else 1)
            offsets.append((name, dtype, shape, cursor, nbytes))
            cursor += nbytes
        self.layout = tuple(layout)
        self.nbytes = cursor
        self.device = device
        cuda = device.type == "cuda"
        self._host = [
            torch.zeros(cursor, dtype=torch.uint8, pin_memory=cuda) for _ in range(2)
        ]
        self._events: list[Any] = [
            torch.cuda.Event() if cuda else None for _ in range(2)
        ]
        self._device = torch.zeros(cursor, dtype=torch.uint8, device=device)
        self._next = 0

        def views(buffer: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                name: buffer[start : start + nbytes].view(dtype).view(shape)
                for name, dtype, shape, start, nbytes in offsets
            }

        self.views: Mapping[str, torch.Tensor] = views(self._device)
        self._host_views = [views(buffer) for buffer in self._host]

    def upload(self, inputs: TransactionInputs) -> None:
        """Write the call's vectors into the next host buffer and copy once."""
        vectors = inputs.staged_vectors()
        if len(vectors) != len(self.layout):
            raise ValueError("transaction vectors differ from the arena layout")
        index = self._next
        self._next ^= 1
        event = self._events[index]
        if event is not None:
            event.synchronize()
        host_views = self._host_views[index]
        for (name, tensor), (expected, dtype, shape) in zip(
            vectors, self.layout, strict=True
        ):
            if (
                name != expected
                or tensor.dtype != dtype
                or tuple(tensor.shape) != shape
            ):
                raise ValueError(f"transaction vector {name!r} differs from the arena")
            host_views[name].copy_(tensor)
        self._device.copy_(self._host[index], non_blocking=True)
        if event is not None:
            event.record()


class ArenaTransactionStager:
    """Read every transaction vector from a persistent device arena."""

    def __init__(self, arena: TransactionArena) -> None:
        self._views = arena.views

    def vector(self, name: str) -> torch.Tensor:
        return self._views[name]

    def bucket_vector(self, index: int, field: str) -> torch.Tensor:
        return self._views[bucket_vector_name(index, field)]


# ---------------------------------------------------------------- device seams
def normalized_free_memory(device: torch.device) -> int | None:
    """Free device memory after a synchronize, ``gc.collect()`` and cache release.

    Every admission figure is a difference of two such readings, so it covers
    memory the allocator does not track. Returns ``None`` off CUDA.
    """
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info(device)
    return int(free)


def device_memory_info(device: torch.device) -> tuple[int | None, int | None]:
    """The device's free and total memory; ``(None, None)`` off CUDA."""
    if device.type != "cuda":
        return None, None
    free, total = torch.cuda.mem_get_info(device)
    return int(free), int(total)


def allocate_escaped_rows(rows: torch.Tensor) -> torch.Tensor:
    """Caller-owned storage for one replay's projection rows."""
    return torch.empty_like(rows)


def _process_constants() -> dict[str, Any]:
    """Host-only reads of the settings every captured graph was built under.

    Only the per-operation ``fp32_precision`` settings are read: they also
    reflect the legacy matmul-precision setter, whose getters raise once a
    per-operation value is set.
    """
    from vllm.model_executor.determinism import batch_invariant

    return {
        "batch-invariant mode": batch_invariant._batch_invariant_MODE,
        "torch.backends.cuda.matmul.fp32_precision": (
            torch.backends.cuda.matmul.fp32_precision
        ),
        "torch.backends.cudnn.conv.fp32_precision": (
            torch.backends.cudnn.conv.fp32_precision
        ),
        "torch.backends.cudnn.rnn.fp32_precision": (
            torch.backends.cudnn.rnn.fp32_precision
        ),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _external_buffers(core: Any, nested: Sequence[Any] = ()) -> dict[str, Any]:
    """Layouts of the binding-external tensors the captured graphs read.

    These are the core's parameters and buffers and the scratch each nested
    binding in ``nested`` reports through ``external_tensors()``.
    """
    if isinstance(core, torch.nn.Module):
        modules = [("", core)]
    else:
        modules = [
            (name, value)
            for name, value in getattr(core, "__dict__", {}).items()
            if isinstance(value, torch.nn.Module)
        ]
    layouts: dict[str, Any] = {}
    for prefix, module in modules:
        for name, tensor in (
            *module.named_parameters(prefix),
            *module.named_buffers(prefix),
        ):
            layouts[name] = _tensor_layout(tensor)
    encoder = getattr(core, "encoder", None)
    layouts["encoder._stream_relative_position_lengths"] = tuple(
        getattr(encoder, "_stream_relative_position_lengths", ())
    )
    for owner in nested:
        for name, tensor in owner.external_tensors().items():
            layouts[name] = _tensor_layout(tensor)
    return layouts


def _failure_figures(
    device: torch.device, baseline: int | None
) -> tuple[int | None, int | None, int | None]:
    """Bytes admitted since ``baseline`` and the device's free and total memory."""
    try:
        current = normalized_free_memory(device)
        free, total = device_memory_info(device)
    except Exception:
        return None, None, None
    admitted = None if baseline is None or current is None else baseline - current
    return admitted, free, total


@contextmanager
def _automatic_gc_paused() -> Iterator[None]:
    """Pause automatic garbage collection for one capture.

    A finalizer running inside a capture, such as a pinned host buffer's,
    could issue a CUDA call that is illegal while a stream captures and
    invalidate the capture.
    """
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


@contextmanager
def _on_stream(stream: Any) -> Iterator[None]:
    """Run the scope on ``stream``, ordered after and before the current one."""
    if stream is None:
        yield
        return
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        yield
    current.wait_stream(stream)


# ------------------------------------------------------------------ cell state
@dataclass
class _CaptureContext:
    """Everything one cell's capture and self-check read; cleared when verified.

    ``inputs`` is also bound for the duration of each replay, for runtimes
    that re-execute the body instead of launching a graph.
    """

    inputs: TransactionInputs | None
    blocks: torch.Tensor | None = None
    pages: tuple[torch.Tensor, ...] | None = None
    expected: tuple[torch.Tensor, ...] | None = None

    def require_inputs(self) -> TransactionInputs:
        if self.inputs is None:
            raise RuntimeError("transaction graph body has no bound inputs")
        return self.inputs

    def clear(self) -> None:
        self.inputs = None
        self.blocks = None
        self.pages = None
        self.expected = None


@dataclass
class _TransactionGraph:
    cell: TransactionCell
    signature: tuple[Any, ...]
    arena: TransactionArena
    static_ids: torch.Tensor
    static_embeds: torch.Tensor
    rows_out: torch.Tensor
    status_out: torch.Tensor
    progress_out: torch.Tensor
    wrapper: Any
    descriptor: Any
    context: _CaptureContext
    replay: Callable[[], None]
    retained_bytes: int | None = None
    warmup_seconds: float | None = None
    capture_seconds: float | None = None
    capture_index: int | None = None
    capture_graph: Callable[[], None] | None = None
    verify: Callable[[], None] | None = None

    def holds_graph(self) -> bool:
        """Whether the wrapper entry still holds this cell's graph."""
        entry = self.wrapper.concrete_cudagraph_entries.get(self.descriptor)
        return entry is not None and entry.cudagraph is not None

    def clear_capture(self) -> None:
        """Drop the capture and self-check callbacks and the capture context."""
        self.capture_graph = None
        self.verify = None
        self.context.clear()


def _free_blocks(num_pool_blocks: int, null_block_id: int) -> list[int]:
    """The pool's usable block ids; capture rows occupy a leading prefix."""
    return [block for block in range(num_pool_blocks) if block != null_block_id]


def _capture_plan(
    cell: TransactionCell,
    *,
    block_ids: Sequence[int],
    num_pool_blocks: int,
    null_block_id: int,
    geometry: int,
    execution_tier: int,
) -> RowPlan:
    rows = cell.population
    is_chunk = cell.kind != CELL_REPLAY
    fresh = cell.kind == CELL_FRESH_CHUNK
    request_ids = tuple(f"transaction-capture-{index}" for index in range(rows))
    deadlines = [index + 1 if is_chunk else 0 for index in range(rows)]
    bindings = tuple(
        PreparedRowBinding(
            request_id=request_ids[index],
            block_id=int(block_ids[index]),
            admission_generation=index + 1,
            geometry_id=geometry,
            prompt_index=0,
            prior_prompt_index=0,
            allow_prompt_transition=False,
            is_chunk=is_chunk,
            ready_deadline_ns=deadlines[index],
        )
        for index in range(rows)
    )
    blocks = torch.tensor(list(block_ids), dtype=torch.int64)
    return RowPlan(
        state_indices_d=torch.empty((0, 1), dtype=torch.int64),
        num_decodes=0,
        state_indices_p=blocks,
        num_prefills=rows,
        has_initial_states_p=torch.full((rows,), not fresh, dtype=torch.bool),
        null_block_id=null_block_id,
        num_pool_blocks=num_pool_blocks,
        live_block_ids=blocks.clone(),
        geometry_id=torch.full((rows,), geometry, dtype=torch.int64),
        prompt_index=torch.zeros(rows, dtype=torch.int64),
        is_chunk=torch.full((rows,), is_chunk, dtype=torch.bool),
        admission_generation=torch.arange(1, rows + 1, dtype=torch.int64),
        ready_deadline_ns=torch.tensor(deadlines, dtype=torch.int64),
        request_ids=request_ids,
        execution_tier=execution_tier if is_chunk else 0,
        bindings=bindings,
        endpoint_mode=torch.zeros(rows, dtype=torch.int64),
        endpoint_threshold_frames=torch.zeros(rows, dtype=torch.int64),
        endpoint_residue_frames=torch.zeros(rows, dtype=torch.int64),
    )


@dataclass
class TransactionCallBindings:
    """The startup-bound callables and controls every served call uses."""

    adapter: Any
    decode_resolver: Any
    encoder_transition: Any
    bucket_transition: Any
    placeholder_id: int
    park_id: int
    eou_token_id: int | None
    graph_covers_decode: bool
    execution_tier: Callable[[int], int] = field(default=lambda population: 0)


def _startup_key(
    pools: TransactionPools,
    call: TransactionCallBindings,
    *,
    hidden_size: int,
    device: torch.device,
    null_block_id: int,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """The pool layout and the call bindings one serving capture is valid for."""
    return (
        tuple(_tensor_layout(pool) for pool in pools.sequence()),
        (
            *(
                _callable_identity(fn)
                for fn in (
                    call.adapter,
                    call.decode_resolver,
                    call.encoder_transition,
                    call.bucket_transition,
                    call.execution_tier,
                )
            ),
            call.placeholder_id,
            call.park_id,
            call.eou_token_id,
            call.graph_covers_decode,
            hidden_size,
            device,
            null_block_id,
        ),
    )


@dataclass
class _CaptureProgress:
    """The cell being prepared or captured, and the cells captured so far."""

    cell: TransactionCell | None = None
    captured: list[TransactionCell] = field(default_factory=list)


@dataclass
class _Inventory:
    """Every declared cell, verified, with its capture cost."""

    graphs: dict[TransactionCell, _TransactionGraph]
    retained_bytes: int | None
    seconds: float

    @property
    def order(self) -> tuple[str, ...]:
        ranked = sorted(
            self.graphs.values(), key=lambda graph: graph.capture_index or 0
        )
        return tuple(graph.cell.label for graph in ranked)


@dataclass
class _AdmissionHolder:
    """The unpublished admission inventory and the stand-in storage it reached."""

    storage: torch.Tensor | None = None
    inventory: _Inventory | None = None


@dataclass(frozen=True)
class _Startup:
    """The startup key and binding-external buffers recorded at publication.

    Each cell's own signature is recorded on its graph.
    """

    key: tuple[tuple[Any, ...], tuple[Any, ...]]
    external: Mapping[str, Any]


class TransactionGraphBinding:
    """Declared whole-transaction cells, their graphs and replay accounting."""

    def __init__(
        self,
        *,
        populations: Sequence[int],
        geometries: Sequence[int],
        maximum_population: int,
        vllm_config: Any,
        runtime: GraphRuntime | None = None,
    ) -> None:
        if (
            not isinstance(populations, (tuple, list))
            or not populations
            or any(
                type(n) is not int or not 0 < n <= maximum_population
                for n in populations
            )
            or len(set(populations)) != len(populations)
        ):
            raise ValueError(
                f"{OPTION} must be distinct integers in 1..{maximum_population}"
            )
        if tuple(geometries) != (SERVED_GEOMETRY,):
            raise ValueError(
                f"{OPTION} serves only geometry g{SERVED_GEOMETRY}, "
                f"not {tuple(geometries)}"
            )
        cells = [
            TransactionCell(kind, SERVED_GEOMETRY, int(population))
            for population in populations
            for kind in (CELL_FRESH_CHUNK, CELL_CONTINUING_CHUNK)
        ]
        cells += [
            TransactionCell(CELL_REPLAY, -1, int(population))
            for population in populations
        ]
        self._cells = tuple(sorted(cells))
        self._replay_geometry = SERVED_GEOMETRY
        self._vllm_config = vllm_config
        self._runtime = runtime
        self._capture_stream: Any = None
        self._graphs: dict[TransactionCell, _TransactionGraph] = {}
        self._pool_signature: tuple[Any, ...] | None = None
        self._startup: _Startup | None = None
        self._admission: _AdmissionHolder | None = None
        self._constants: dict[str, Any] | None = None
        self._failed: str | None = None
        self._retained_bytes_total: int | None = None
        self._admission_bytes: int | None = None
        self._serving_bytes: int | None = None
        self._admission_release_verified: bool | None = None
        self._admission_seconds: float | None = None
        self._serving_seconds: float | None = None
        self._admission_order: tuple[str, ...] = ()
        self._replays: dict[TransactionCell, int] = {}
        self._fallbacks: dict[tuple[str, str], int] = {}
        self._prologue_ns_last = 0
        self._prologue_ns_total = 0
        self._instance_id = f"{os.getpid()}:{time.monotonic_ns()}"

    @property
    def cells(self) -> tuple[TransactionCell, ...]:
        return self._cells

    @property
    def ready(self) -> bool:
        return self._failed is None and set(self._graphs) == set(self._cells)

    @property
    def serving_captured(self) -> bool:
        return self._startup is not None

    def discard(self) -> None:
        """Drop every graph, arena and owned buffer, published or admitted."""
        self._graphs = {}
        self._admission = None
        self._pool_signature = None
        self._startup = None
        self._retained_bytes_total = None

    def require_admission(self) -> None:
        """Raise, latching the binding, unless admission measured its inventory.

        Serving capture compares its retained memory with the admitted bytes;
        without them that comparison cannot run.
        """
        self.check_latch()
        if self._admission_bytes is None:
            raise self._fatal(
                f"{OPTION}: serving capture requires the admission capture of "
                "prepare_execution_memory, which did not measure this inventory"
            )

    # ------------------------------------------------------------ the latch
    def check_latch(self) -> None:
        """Raise if a worker-fatal failure has latched the binding."""
        if self._failed is not None:
            raise RuntimeError(
                f"{OPTION}: the transaction graph binding failed earlier "
                f"({self._failed}); a fresh worker is required"
            )

    def _latch(self, reason: str) -> None:
        if self._failed is None:
            self._failed = reason

    def _fatal(self, message: str) -> RuntimeError:
        self._latch(message)
        return RuntimeError(f"{message}; a fresh worker is required")

    def _verify_process_constants(self) -> None:
        if self._constants is None:
            return
        for name, value in _process_constants().items():
            captured = self._constants[name]
            if value != captured:
                raise self._fatal(
                    f"{OPTION}: process constant {name} changed from "
                    f"{captured!r} to {value!r} since capture"
                )

    # ------------------------------------------------------------- capture
    @torch.inference_mode()
    def admit(
        self,
        core: Any,
        *,
        allocate: Callable[[int], tuple[torch.Tensor, TransactionPools]],
        call: TransactionCallBindings,
        hidden_size: int,
        device: torch.device,
        null_block_id: int = 0,
    ) -> None:
        """Capture the declared inventory over stand-in storage to admit its memory.

        Runs inside the memory-profiling window, before the eager profile
        transactions. ``allocate(slots)`` returns raw stand-in storage of
        ``slots`` pages with the real layout and its pools. The unpublished
        inventory stays alive until serving capture releases it. The stand-in
        storage is released after ``Na``; release is verified when ``N1 - Na``
        covers its bytes and no owning reference survives. ``A = N0 - N1``.
        """
        self.check_latch()
        if self._startup is not None or self._admission is not None:
            raise RuntimeError(f"{OPTION} admits its inventory once, before serving")
        slots = max(cell.population for cell in self._cells) + 1
        progress = _CaptureProgress()
        holder = _AdmissionHolder()
        baseline: int | None = None
        try:
            baseline = normalized_free_memory(device)
            storage, pools = allocate(slots)
            holder.storage = storage
            stand_in = weakref.ref(storage)
            stand_in_bytes = int(storage.untyped_storage().nbytes())
            del storage
            holder.inventory = self._capture_inventory(
                core,
                pools=pools,
                call=call,
                hidden_size=hidden_size,
                device=device,
                null_block_id=null_block_id,
                progress=progress,
            )
            del pools
            retained = normalized_free_memory(device)
            holder.storage = None
            released = normalized_free_memory(device)
        except Exception as error:
            failure = self._inventory_error(
                "admission", error, progress, device, baseline
            )
            holder.storage = None
            holder.inventory = None
            self._latch(str(failure))
            raise failure from error
        inventory = holder.inventory
        assert inventory is not None
        release = None if retained is None or released is None else released - retained
        owned = stand_in() is not None
        verified = not owned and (release is None or release >= stand_in_bytes)
        self._admission_release_verified = verified
        if not verified:
            holder.inventory = None
            message = (
                f"{OPTION}: admission release is not verified: N1 - Na is "
                f"{release} bytes for {stand_in_bytes} stand-in bytes"
            )
            if owned:
                message += ", and an owning reference to the stand-in storage survived"
            self._latch(message)
            raise RuntimeError(message)
        self._admission = holder
        self._admission_bytes = (
            None if baseline is None or released is None else baseline - released
        )
        self._admission_seconds = inventory.seconds
        self._admission_order = inventory.order

    @torch.inference_mode()
    def capture(
        self,
        core: Any,
        *,
        pools: TransactionPools,
        call: TransactionCallBindings,
        hidden_size: int,
        device: torch.device,
        null_block_id: int = 0,
        nested: Sequence[Any] = (),
    ) -> None:
        """Capture every declared cell over the allocated resident pools.

        Runs once per worker. The admission inventory is released before
        ``Ns0``; the serving inventory is published only when every cell
        verifies and ``S = Ns0 - Ns1`` does not exceed the admitted ``A``. A
        repeated call with identical pool layout and call bindings is a no-op;
        any difference, and any capture failure, discards every graph and
        latches the binding. Publication records the layout of every
        binding-external buffer: the core's, and the scratch of each nested
        binding in ``nested`` the outer body runs eagerly.

        Capture inputs use real free block ids and rows the transaction rejects
        (a non-envelope CHUNK payload, or a placeholder id on a non-CHUNK row),
        so warm-up and self-check runs store no page. Each graph must reproduce
        the eager transaction bitwise on that input before it is published.
        """
        self.check_latch()
        key = _startup_key(
            pools,
            call,
            hidden_size=hidden_size,
            device=device,
            null_block_id=null_block_id,
        )
        if self._startup is not None:
            if key == self._startup.key:
                return
            changed = (
                "pool layout" if key[0] != self._startup.key[0] else "call bindings"
            )
            self.discard()
            raise self._fatal(
                f"{OPTION}: initialize_state_slots repeated with a different "
                f"{changed}; nothing is recaptured after startup"
            )
        self._admission = None
        progress = _CaptureProgress()
        baseline: int | None = None
        try:
            baseline = normalized_free_memory(device)
            inventory = self._capture_inventory(
                core,
                pools=pools,
                call=call,
                hidden_size=hidden_size,
                device=device,
                null_block_id=null_block_id,
                progress=progress,
            )
            retained = normalized_free_memory(device)
        except Exception as error:
            failure = self._inventory_error(
                "serving", error, progress, device, baseline
            )
            self.discard()
            self._latch(str(failure))
            raise failure from error
        serving = None if baseline is None or retained is None else baseline - retained
        admitted = self._admission_bytes
        if serving is not None and admitted is not None and serving > admitted:
            del inventory
            self.discard()
            raise self._fatal(
                f"{OPTION}: serving capture retained {serving} bytes, more than the "
                f"{admitted} admitted bytes"
            )
        self._graphs = inventory.graphs
        self._pool_signature = key[0]
        self._startup = _Startup(key=key, external=_external_buffers(core, nested))
        self._retained_bytes_total = inventory.retained_bytes
        self._serving_bytes = serving
        self._serving_seconds = inventory.seconds

    def _inventory_error(
        self,
        phase: str,
        error: Exception,
        progress: _CaptureProgress,
        device: torch.device,
        baseline: int | None = None,
    ) -> RuntimeError:
        failing = "none" if progress.cell is None else progress.cell.label
        captured = [cell.label for cell in progress.captured]
        message = (
            f"{OPTION}: {phase} capture failed at cell {failing} with "
            f"{len(captured)} cells captured ({', '.join(captured) or 'none'})"
        )
        if isinstance(error, torch.OutOfMemoryError):
            admitted, free, total = _failure_figures(device, baseline)
            message += (
                f"; out of memory with {admitted} normalized bytes admitted so far "
                f"and {free} of {total} device bytes free"
            )
        return RuntimeError(f"{message}: {error}")

    def _capture_inventory(
        self,
        core: Any,
        *,
        pools: TransactionPools,
        call: TransactionCallBindings,
        hidden_size: int,
        device: torch.device,
        null_block_id: int,
        progress: _CaptureProgress,
    ) -> _Inventory:
        """Prepare, capture and verify every declared cell into one fresh pool.

        Every cell's eager reference and warm-ups precede the first capture, so
        lazily extended module buffers are not rebound under a graph that has
        already recorded their addresses. Cells are captured largest population
        first on the binding's one capture stream. Each cell's capture context
        is cleared before this returns, whether or not every cell verified.
        """
        started = time.perf_counter()
        runtime = self._runtime or platform_graph_runtime()
        self._runtime = runtime
        if self._constants is None:
            self._constants = _process_constants()
        else:
            self._verify_process_constants()
        num_pool_blocks = int(pools.queue.shape[0])
        free = _free_blocks(num_pool_blocks, null_block_id)
        largest = max(cell.population for cell in self._cells)
        if len(free) < largest:
            raise ValueError(
                f"{OPTION} declares population {largest}, but the state pools hold "
                f"{len(free)} usable slots"
            )
        graph_pool = None
        if device.type == "cuda":
            from vllm.platforms import current_platform

            graph_pool = current_platform.graph_pool_handle()
            if self._capture_stream is None:
                self._capture_stream = torch.cuda.Stream(device=device)
        graphs: dict[TransactionCell, _TransactionGraph] = {}
        try:
            for cell in self._cells:
                progress.cell = cell
                graphs[cell] = self._capture_cell(
                    core,
                    cell,
                    pools=pools,
                    call=call,
                    runtime=runtime,
                    block_ids=free[: cell.population],
                    num_pool_blocks=num_pool_blocks,
                    null_block_id=null_block_id,
                    hidden_size=hidden_size,
                    device=device,
                    graph_pool=graph_pool,
                    capture_stream=self._capture_stream,
                )
            reserved = (
                torch.cuda.memory_reserved(device) if device.type == "cuda" else None
            )
            order = sorted(graphs.values(), key=lambda graph: -graph.cell.population)
            for index, graph in enumerate(order):
                progress.cell = graph.cell
                graph.capture_index = index
                assert graph.capture_graph is not None
                graph.capture_graph()
                progress.captured.append(graph.cell)
            retained = (
                None
                if reserved is None
                else torch.cuda.memory_reserved(device) - reserved
            )
            for graph in graphs.values():
                progress.cell = graph.cell
                assert graph.verify is not None
                graph.verify()
        finally:
            for graph in graphs.values():
                graph.clear_capture()
            consume_batch_stats()
        return _Inventory(graphs, retained, time.perf_counter() - started)

    def _cell_inputs(
        self,
        core: Any,
        cell: TransactionCell,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        *,
        pools: TransactionPools,
        call: TransactionCallBindings,
        block_ids: Sequence[int],
        num_pool_blocks: int,
        null_block_id: int,
    ) -> TransactionInputs:
        """The host prologue of ``cell``'s capture rows over ``pools``.

        Capture and post-warmup verification both build a cell's inputs here,
        so a cell's live signature is computed exactly as at capture.
        """
        geometry = self._replay_geometry if cell.kind == CELL_REPLAY else cell.geometry
        plan = _capture_plan(
            cell,
            block_ids=block_ids,
            num_pool_blocks=num_pool_blocks,
            null_block_id=null_block_id,
            geometry=geometry,
            execution_tier=(
                call.execution_tier(cell.population) if call.graph_covers_decode else 0
            ),
        )
        return prepare_transaction(
            core,
            input_ids,
            inputs_embeds,
            plan,
            pools=pools,
            eou_token_id=call.eou_token_id,
            adapter=call.adapter,
            decode_resolver=call.decode_resolver,
            encoder_transition=call.encoder_transition,
            bucket_transition=call.bucket_transition,
            placeholder_id=call.placeholder_id,
            park_id=call.park_id,
            has_commit_sink=False,
            capture=False,
            graph_covers_decode=call.graph_covers_decode,
            memory_profile=False,
            native_burst_handoff=None,
        )

    def _capture_cell(
        self,
        core: Any,
        cell: TransactionCell,
        *,
        pools: TransactionPools,
        call: TransactionCallBindings,
        runtime: GraphRuntime,
        block_ids: Sequence[int],
        num_pool_blocks: int,
        null_block_id: int,
        hidden_size: int,
        device: torch.device,
        graph_pool: Any = None,
        capture_stream: Any = None,
    ) -> _TransactionGraph:
        """Prepare one cell: its eager reference, warm-ups and capture callbacks.

        Every capture-only value lives in the returned cell's capture context,
        which the wrapper's body and the callbacks read through.
        """
        started = time.perf_counter()
        input_ids = torch.full(
            (cell.population,), call.placeholder_id, dtype=torch.int64, device=device
        )
        inputs_embeds = torch.zeros(
            cell.population, hidden_size, dtype=torch.float32, device=device
        )
        inputs = self._cell_inputs(
            core,
            cell,
            input_ids,
            inputs_embeds,
            pools=pools,
            call=call,
            block_ids=block_ids,
            num_pool_blocks=num_pool_blocks,
            null_block_id=null_block_id,
        )
        classified, reason = classify_transaction(inputs)
        if classified != cell:
            raise RuntimeError(
                f"transaction capture input for {cell.label} classified as "
                f"{classified} ({reason})"
            )
        blocks = torch.tensor(list(block_ids), dtype=torch.int64, device=device)
        pages = tuple(pool.index_select(0, blocks).clone() for pool in pools.sequence())

        reference = run_transaction_body(
            core, input_ids, inputs_embeds, inputs, LiveTransactionStager(inputs, None)
        )
        execute_transaction_scatters(reference.scatter_ops)
        expected = (
            reference.projection_rows.clone(),
            reference.status.clone(),
            reference.progress.clone(),
        )
        if not bool((expected[1] != 0).all()):
            raise RuntimeError(
                f"transaction capture input for {cell.label} was not rejected"
            )
        del reference

        # Persistent buffers outlive this inference-mode scope; callers may
        # update them in place from either mode.
        with torch.inference_mode(False):
            arena = TransactionArena(_arena_layout(inputs), device)
            static_ids = torch.empty_like(input_ids).copy_(input_ids)
            static_embeds = torch.empty_like(inputs_embeds).copy_(inputs_embeds)
            rows_out = torch.empty_like(expected[0])
            status_out = torch.empty_like(expected[1])
            progress_out = torch.empty_like(expected[2])
        context = _CaptureContext(inputs, blocks, pages, expected)
        stager = ArenaTransactionStager(arena)
        owned = (rows_out, status_out, progress_out)
        covers = call.graph_covers_decode
        vllm_config = self._vllm_config

        def body() -> tuple[torch.Tensor, ...]:
            with outer_transaction_capture():
                result = run_transaction_body(
                    core, static_ids, static_embeds, context.require_inputs(), stager
                )
                if result.native_stage is not None or result.records:
                    raise RuntimeError("transaction graph body staged host work")
                execute_transaction_scatters(result.scatter_ops)
            for destination, source in zip(
                owned,
                (result.projection_rows, result.status, result.progress),
                strict=True,
            ):
                destination.copy_(source)
            return owned

        wrapper = runtime.wrapper_factory(
            body, vllm_config, runtime_mode=runtime.graph_mode
        )
        if graph_pool is not None:
            wrapper.graph_pool = graph_pool
        descriptor = runtime.descriptor_factory(cell.population)

        def launch(mode: Any) -> None:
            with runtime.forward_context(
                None,
                vllm_config,
                cudagraph_runtime_mode=mode,
                batch_descriptor=descriptor,
            ):
                wrapper()

        def matches() -> bool:
            blocks, pages, expected = context.blocks, context.pages, context.expected
            assert blocks is not None and pages is not None and expected is not None
            if any(
                not torch.equal(actual, wanted)
                for actual, wanted in zip(owned, expected, strict=True)
            ):
                return False
            resident = context.require_inputs().pools.sequence()
            return all(
                torch.equal(pool.index_select(0, blocks), page)
                for pool, page in zip(resident, pages, strict=True)
            )

        # Warm up on the capture stream, so state created lazily per stream
        # exists before the capture instead of being recorded into it.
        with _on_stream(capture_stream):
            for _ in range(3):
                arena.upload(inputs)
                launch(runtime.eager_mode)
        if not matches():
            raise RuntimeError(f"transaction eager twin differs for {cell.label}")
        graph = _TransactionGraph(
            cell=cell,
            signature=transaction_signature(
                core, inputs, static_ids, static_embeds, graph_covers_decode=covers
            ),
            arena=arena,
            static_ids=static_ids,
            static_embeds=static_embeds,
            rows_out=rows_out,
            status_out=status_out,
            progress_out=progress_out,
            wrapper=wrapper,
            descriptor=descriptor,
            context=context,
            replay=lambda: launch(runtime.graph_mode),
            warmup_seconds=time.perf_counter() - started,
        )

        def capture_graph() -> None:
            started = time.perf_counter()
            reserved = (
                torch.cuda.memory_reserved(device) if device.type == "cuda" else None
            )
            captured = context.require_inputs()
            graph.signature = transaction_signature(
                core, captured, static_ids, static_embeds, graph_covers_decode=covers
            )
            arena.upload(captured)
            runtime.synchronize(device)
            with _automatic_gc_paused():
                runtime.set_capture_enabled(True)
                try:
                    with runtime.capture_context(device), _on_stream(capture_stream):
                        launch(runtime.graph_mode)
                finally:
                    runtime.set_capture_enabled(False)
            if reserved is not None:
                graph.retained_bytes = torch.cuda.memory_reserved(device) - reserved
            graph.capture_seconds = time.perf_counter() - started

        def verify() -> None:
            for value in owned:
                value.fill_(-1)
            arena.upload(context.require_inputs())
            launch(runtime.graph_mode)
            runtime.synchronize(device)
            if not matches():
                raise RuntimeError(f"transaction graph replay differs for {cell.label}")

        graph.capture_graph = capture_graph
        graph.verify = verify
        return graph

    # ------------------------------------------------- post-warmup verification
    def verify_after_warmup(
        self,
        core: Any,
        *,
        resident_pools: Callable[[], TransactionPools] | None,
        call: TransactionCallBindings,
        hidden_size: int,
        null_block_id: int = 0,
        nested: Sequence[Any] = (),
    ) -> None:
        """Check every published cell against live state.

        ``resident_pools`` projects the live resident pools; ``call`` and
        ``nested`` are the live call bindings and nested bindings. Each cell's
        signature is rebuilt from them as at capture and compared with the one
        recorded at publication; so are the startup key and every
        binding-external buffer's address and size. Nothing built here is
        retained. A mismatch is worker-fatal and latches the binding.
        """
        self.check_latch()
        startup = self._startup
        if startup is None or not self.ready or resident_pools is None:
            raise self._fatal(f"{OPTION}: no serving inventory was published")
        labels = ", ".join(cell.label for cell in self._cells)

        def failed(cells: str, change: str) -> RuntimeError:
            return self._fatal(
                f"{OPTION}: post-warmup verification failed for {cells}: {change}"
            )

        try:
            pools = resident_pools()
        except Exception as error:
            raise failed(
                f"cells {labels}", f"the resident pools no longer project: {error}"
            ) from error
        device = pools.queue.device
        num_pool_blocks = int(pools.queue.shape[0])
        free = _free_blocks(num_pool_blocks, null_block_id)
        try:
            for cell in self._cells:
                # Only the shapes, dtypes and device of these inputs are read.
                input_ids = torch.empty(
                    cell.population, dtype=torch.int64, device=device
                )
                inputs_embeds = torch.empty(
                    cell.population, hidden_size, dtype=torch.float32, device=device
                )
                try:
                    inputs = self._cell_inputs(
                        core,
                        cell,
                        input_ids,
                        inputs_embeds,
                        pools=pools,
                        call=call,
                        block_ids=free[: cell.population],
                        num_pool_blocks=num_pool_blocks,
                        null_block_id=null_block_id,
                    )
                except Exception as error:
                    raise failed(
                        f"cell {cell.label}",
                        f"its capture rows no longer prepare over live state: {error}",
                    ) from error
                live = transaction_signature(
                    core,
                    inputs,
                    input_ids,
                    inputs_embeds,
                    graph_covers_decode=call.graph_covers_decode,
                )
                del inputs
                change = _signature_change(self._graphs[cell].signature, live, pools)
                if change is not None:
                    raise failed(
                        f"cell {cell.label}",
                        f"its signature's {change} changed since capture",
                    )
        finally:
            consume_batch_stats()
        key = _startup_key(
            pools,
            call,
            hidden_size=hidden_size,
            device=device,
            null_block_id=null_block_id,
        )
        pool = _first_change(_pool_names(pools), startup.key[0], key[0])
        if pool is not None:
            raise failed(f"cells {labels}", f"resident pool {pool} changed")
        binding = _first_change(_CALL_FIELDS, startup.key[1], key[1])
        if binding is not None:
            raise failed(f"cells {labels}", f"call binding {binding} changed")
        external = _external_buffers(core, nested)
        for name, layout in startup.external.items():
            if external.get(name) != layout:
                raise failed(
                    f"cells {labels}",
                    f"buffer {name} changed from {layout} to {external.get(name)}",
                )

    # ----------------------------------------------------------- execution
    def _fallback(self, cell: TransactionCell | None, reason: str) -> None:
        key = (cell.label if cell is not None else "uncovered", reason)
        self._fallbacks[key] = self._fallbacks.get(key, 0) + 1

    def execute(
        self,
        core: Any,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        inputs: TransactionInputs,
        *,
        commit_sink: CommitSink | None,
        graph_covers_decode: bool,
        memory_profile: bool,
        started_ns: int,
    ) -> torch.Tensor | None:
        """Replay the call's captured cell, or return ``None`` to run eagerly.

        Checks run in a fixed order: the failed latch, the process constants,
        classification with its mode reasons, the signature and the wrapper
        entry. Arena upload, static-input copies, the escaped-row allocation and
        the commit reservation precede ``graph.replay()``. Any failure from the
        replay onward latches the binding.
        """
        self.check_latch()
        self._verify_process_constants()
        cell, reason = classify_transaction(inputs)
        if inputs.capture:
            reason = "capture"
        elif memory_profile:
            reason = "memory-profile"
        elif inputs.native_burst_handoff is not None:
            reason = "native-burst"
        graph = self._graphs.get(cell) if cell is not None else None
        if reason is None and graph is None:
            reason = "undeclared" if cell not in self._cells else "uncaptured"
        if reason is None:
            assert graph is not None
            signature = transaction_signature(
                core,
                inputs,
                input_ids,
                inputs_embeds,
                graph_covers_decode=graph_covers_decode,
            )
            if signature != graph.signature:
                reason = "signature"
        if reason is not None:
            self._fallback(cell, reason)
            return None
        assert graph is not None
        if not graph.holds_graph():
            raise self._fatal(
                f"{OPTION}: the wrapper entry of cell {graph.cell.label} no longer "
                "holds its graph"
            )
        graph.arena.upload(inputs)
        graph.static_ids.copy_(input_ids)
        graph.static_embeds.copy_(inputs_embeds)
        rows = allocate_escaped_rows(graph.rows_out)
        cancel: Callable[[], None] | None = None
        stage: Callable[[torch.Tensor], None] | None = None
        if commit_sink is not None:
            cancel, stage = reserve_transaction_commit(
                commit_sink,
                CommitPlan(
                    bindings=inputs.plan.bindings,
                    capture=None,
                    records=(),
                    progress=graph.progress_out,
                ),
            )
        prologue_ns = time.perf_counter_ns() - started_ns
        self._prologue_ns_last = prologue_ns
        self._prologue_ns_total += prologue_ns
        # A runtime that re-executes the body at replay reads these inputs; they
        # match the captured signature, so the host control flow is the same.
        graph.context.inputs = inputs
        try:
            graph.replay()
            rows.copy_(graph.rows_out)
            if stage is not None:
                stage(graph.status_out)
        except BaseException as error:
            self._latch(
                f"{graph.cell.label} failed from its replay on: "
                f"{type(error).__name__}: {error}"
            )
            if cancel is not None:
                with suppress(Exception):
                    cancel()
            raise
        finally:
            graph.context.inputs = None
        self._replays[graph.cell] = self._replays.get(graph.cell, 0) + 1
        return rows

    # ------------------------------------------------------------ reporting
    def counters(self) -> dict[str, int]:
        """Host-only replay and fallback counts; never synchronizes."""
        counts: dict[str, int] = {
            "captured_cells": len(self._graphs),
            "replays": sum(self._replays.values()),
            "fallbacks": sum(self._fallbacks.values()),
            "prologue_ns_last": self._prologue_ns_last,
            "prologue_ns_total": self._prologue_ns_total,
        }
        for cell in self._cells:
            counts[f"replays:{cell.label}"] = self._replays.get(cell, 0)
        for (label, reason), count in sorted(self._fallbacks.items()):
            counts[f"fallbacks:{label}:{reason}"] = count
        return counts

    def receipt(self) -> dict[str, Any]:
        """Startup coverage, admission and per-cell capture cost.

        ``retained_bytes`` is each serving capture's reserved-memory growth and
        ``retained_bytes_total`` the serving inventory's; no reuse across cells
        is assumed. ``warmup_seconds`` covers a cell's eager reference and
        warm-ups, ``capture_seconds`` its capture and instantiation.
        ``admission_bytes`` (``A``) and ``serving_bytes`` (``S``) are
        normalized free-memory differences, ``None`` off CUDA.
        """
        fields = (
            "retained_bytes",
            "warmup_seconds",
            "capture_seconds",
            "capture_index",
        )

        def cell_entry(cell: TransactionCell) -> dict[str, Any]:
            graph = self._graphs.get(cell)
            entry: dict[str, Any] = {
                "cell": cell.label,
                "captured": graph is not None,
                "arena_bytes": None if graph is None else graph.arena.nbytes,
            }
            for name in fields:
                entry[name] = None if graph is None else getattr(graph, name)
            return entry

        return {
            "ready": self.ready,
            "worker_pid": os.getpid(),
            "instance_id": self._instance_id,
            "retained_bytes_total": self._retained_bytes_total,
            "admission_bytes": self._admission_bytes,
            "serving_bytes": self._serving_bytes,
            "admission_release_verified": self._admission_release_verified,
            "admission_capture_seconds": self._admission_seconds,
            "serving_capture_seconds": self._serving_seconds,
            "admission_capture_order": list(self._admission_order),
            "cells": [cell_entry(cell) for cell in self._cells],
            "counters": self.counters(),
        }


def _resolve_model(target: Any) -> Any:
    """Reach the Nemotron model from a model, runner, worker or EngineCore."""
    current = target
    for _ in range(8):
        if hasattr(current, "_transaction_graph"):
            return current
        for attribute in ("model_executor", "driver_worker", "worker", "model_runner"):
            nxt = getattr(current, attribute, None)
            if nxt is not None:
                current = nxt
                break
        else:
            model = getattr(current, "model", None)
            if model is None:
                return None
            current = model
    return None


def transaction_graph_counters(target: Any) -> dict[str, int]:
    """Return per-cell replay and fallback counts for an in-process engine.

    ``target`` is the Nemotron model or any in-process owner of it (model
    runner, worker, executor or EngineCore). Returns an empty mapping when the
    transaction graph option is off.

    Raises:
        RuntimeError: the model is not reachable in this process, for example
            under a multiprocess executor.

    """
    model = _resolve_model(target)
    if model is None:
        raise RuntimeError(
            "transaction graph counters cannot reach the Nemotron model in this "
            "process; read them where the model runs"
        )
    binding = getattr(model, "_transaction_graph", None)
    if binding is None:
        return {}
    return binding.counters()
