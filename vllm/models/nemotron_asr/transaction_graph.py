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
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
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


def transaction_signature(
    inputs: TransactionInputs,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    *,
    graph_covers_decode: bool,
) -> tuple[Any, ...]:
    """Every structural input a captured body freezes."""
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
        (tuple(input_ids.shape), input_ids.dtype, input_ids.device),
        (tuple(inputs_embeds.shape), inputs_embeds.dtype, inputs_embeds.device),
    )


def _arena_layout(
    inputs: TransactionInputs,
) -> tuple[tuple[str, torch.dtype, tuple[int, ...]], ...]:
    return tuple(
        (name, tensor.dtype, tuple(tensor.shape))
        for name, tensor in inputs.staged_vectors()
    )


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
    replay: Callable[[], None]
    retained_bytes: int | None = None


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
                "transaction_graph_populations must be distinct integers in "
                f"1..{maximum_population}"
            )
        if not geometries:
            raise ValueError("transaction graphs require a served geometry")
        cells = [
            TransactionCell(kind, int(geometry), int(population))
            for population in populations
            for geometry in geometries
            for kind in (CELL_FRESH_CHUNK, CELL_CONTINUING_CHUNK)
        ]
        cells += [
            TransactionCell(CELL_REPLAY, -1, int(population))
            for population in populations
        ]
        self._cells = tuple(sorted(cells))
        self._replay_geometry = int(min(geometries))
        self._vllm_config = vllm_config
        self._runtime = runtime
        self._graphs: dict[TransactionCell, _TransactionGraph] = {}
        self._pool_signature: tuple[Any, ...] | None = None
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
        return set(self._graphs) == set(self._cells)

    def discard(self) -> None:
        self._graphs = {}
        self._pool_signature = None

    # ------------------------------------------------------------- capture
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
    ) -> None:
        """Capture every declared cell over the allocated resident pools.

        Capture inputs use real free block ids and rows the transaction rejects
        (a non-envelope CHUNK payload, or a placeholder id on a non-CHUNK row),
        so warm-up and self-check runs store no page. Each graph must reproduce
        the eager transaction bitwise on that input before it is published.
        """
        signature = tuple(_tensor_layout(pool) for pool in pools.sequence())
        if self._graphs and signature == self._pool_signature:
            return
        self.discard()
        runtime = self._runtime or platform_graph_runtime()
        num_pool_blocks = int(pools.queue.shape[0])
        free = [block for block in range(num_pool_blocks) if block != null_block_id]
        pending: dict[TransactionCell, _TransactionGraph] = {}
        try:
            for cell in self._cells:
                if len(free) < cell.population:
                    raise ValueError(
                        f"transaction cell {cell.label} needs {cell.population} free "
                        f"state slots; {len(free)} exist"
                    )
                pending[cell] = self._capture_cell(
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
                )
        finally:
            consume_batch_stats()
        self._graphs = pending
        self._pool_signature = signature
        self._runtime = runtime

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
    ) -> _TransactionGraph:
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
        input_ids = torch.full(
            (cell.population,), call.placeholder_id, dtype=torch.int64, device=device
        )
        inputs_embeds = torch.zeros(
            cell.population, hidden_size, dtype=torch.float32, device=device
        )
        inputs = prepare_transaction(
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
        stager = ArenaTransactionStager(arena)
        owned = (rows_out, status_out, progress_out)

        def body() -> tuple[torch.Tensor, ...]:
            with outer_transaction_capture():
                result = run_transaction_body(
                    core, static_ids, static_embeds, inputs, stager
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
            body, self._vllm_config, runtime_mode=runtime.graph_mode
        )
        descriptor = runtime.descriptor_factory(cell.population)

        def launch(mode: Any) -> None:
            with runtime.forward_context(
                None,
                self._vllm_config,
                cudagraph_runtime_mode=mode,
                batch_descriptor=descriptor,
            ):
                wrapper()

        def matches() -> bool:
            if any(
                not torch.equal(actual, wanted)
                for actual, wanted in zip(owned, expected, strict=True)
            ):
                return False
            return all(
                torch.equal(pool.index_select(0, blocks), page)
                for pool, page in zip(pools.sequence(), pages, strict=True)
            )

        reserved_before = (
            torch.cuda.memory_reserved(device) if device.type == "cuda" else None
        )
        for _ in range(3):
            arena.upload(inputs)
            launch(runtime.eager_mode)
        if not matches():
            raise RuntimeError(f"transaction eager twin differs for {cell.label}")
        arena.upload(inputs)
        runtime.synchronize(device)
        runtime.set_capture_enabled(True)
        try:
            with runtime.capture_context(device):
                launch(runtime.graph_mode)
        finally:
            runtime.set_capture_enabled(False)
        for value in owned:
            value.fill_(-1)
        arena.upload(inputs)
        launch(runtime.graph_mode)
        runtime.synchronize(device)
        if not matches():
            raise RuntimeError(f"transaction graph replay differs for {cell.label}")
        retained = (
            torch.cuda.memory_reserved(device) - reserved_before
            if reserved_before is not None
            else None
        )
        return _TransactionGraph(
            cell=cell,
            signature=transaction_signature(
                inputs,
                static_ids,
                static_embeds,
                graph_covers_decode=call.graph_covers_decode,
            ),
            arena=arena,
            static_ids=static_ids,
            static_embeds=static_embeds,
            rows_out=rows_out,
            status_out=status_out,
            progress_out=progress_out,
            replay=lambda: launch(runtime.graph_mode),
            retained_bytes=retained,
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
        """Replay the call's captured cell, or return ``None`` to run eagerly."""
        del core
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
        graph.arena.upload(inputs)
        graph.static_ids.copy_(input_ids)
        graph.static_embeds.copy_(inputs_embeds)
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
        try:
            graph.replay()
        except BaseException:
            if cancel is not None:
                cancel()
            raise
        if stage is not None:
            stage(graph.status_out)
        self._replays[graph.cell] = self._replays.get(graph.cell, 0) + 1
        return graph.rows_out.clone()

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
        return {
            "ready": self.ready,
            "worker_pid": os.getpid(),
            "instance_id": self._instance_id,
            "cells": [
                {
                    "cell": cell.label,
                    "captured": cell in self._graphs,
                    "arena_bytes": (
                        self._graphs[cell].arena.nbytes
                        if cell in self._graphs
                        else None
                    ),
                    "retained_bytes": (
                        self._graphs[cell].retained_bytes
                        if cell in self._graphs
                        else None
                    ),
                }
                for cell in self._cells
            ],
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
    """
    model = _resolve_model(target)
    binding = getattr(model, "_transaction_graph", None)
    if binding is None:
        return {}
    return binding.counters()
