# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whole-transaction graph replay against the eager transaction.

CPU tests drive the binding through a runtime whose "replay" re-executes the
captured body over the static arena, which checks the arena plumbing, the
staging guard, key classification and fallback accounting. CUDA tests use real
``torch.cuda.CUDAGraph`` replay, where every host value outside the arena would
be frozen at capture.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.models.nemotron_asr.test_advance_model_rows_local import (
    _BOOK,
    CAP,
    CARRIER_HIDDEN,
    FINAL_SAMPLES,
    GEOM_FINAL,
    GEOM_REG,
    PARK_ID,
    PLACEHOLDER_ID,
    REG_SAMPLES,
    VOCAB,
    _assert_pools_equal,
    _clone_pools,
    _CommitRecorder,
    _envelope,
    _fixed_resolver,
    _fresh_pools,
    _plan,
    _tiny_core,
    advance,
    rnnt,
)
from vllm.models.nemotron_asr import transaction_graph as tg
from vllm.models.nemotron_asr.capture_scope import (
    nested_graph_mode,
    outer_transaction_capture,
)
from vllm.models.nemotron_asr.decode_graph import GraphRuntime

pytestmark = [pytest.mark.core_model]

EOU = VOCAB + 3
CPU = torch.device("cpu")
Pools = dict[str, Any]


# ---- runtimes --------------------------------------------------------------


def _runtime(wrapper_factory: Callable[..., Any], mode: dict[str, str]) -> GraphRuntime:
    def context(
        _metadata: object,
        _config: object,
        *,
        cudagraph_runtime_mode: str,
        batch_descriptor: object,
    ) -> Any:
        del batch_descriptor

        @contextmanager
        def scope() -> Iterator[None]:
            previous = mode["value"]
            mode["value"] = cudagraph_runtime_mode
            try:
                yield
            finally:
                mode["value"] = previous

        return scope()

    return GraphRuntime(
        wrapper_factory=wrapper_factory,
        forward_context=context,
        capture_context=lambda _device: nullcontext(),
        descriptor_factory=lambda population: population,
        eager_mode="none",
        graph_mode="graph",
        synchronize=lambda device: (
            torch.cuda.synchronize(device) if device.type == "cuda" else None
        ),
        set_capture_enabled=lambda _enabled: None,
    )


def _rerun_runtime() -> GraphRuntime:
    """Replay re-executes the body; used to test plumbing on CPU."""

    class Wrapper:
        def __init__(self, runnable: Callable[[], Any], *_: Any, **__: Any):
            self.runnable = runnable

        def __call__(self) -> Any:
            return self.runnable()

    return _runtime(Wrapper, {"value": "none"})


def _cuda_runtime() -> GraphRuntime:
    """Capture on the first graph-mode call, then replay the CUDA graph."""
    mode = {"value": "none"}

    class Wrapper:
        def __init__(self, runnable: Callable[[], Any], *_: Any, **__: Any):
            self.runnable = runnable
            self.graph: torch.cuda.CUDAGraph | None = None
            self.output: Any = None

        def __call__(self) -> Any:
            if mode["value"] != "graph":
                return self.runnable()
            if self.graph is None:
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self.output = self.runnable()
                return self.output
            self.graph.replay()
            return self.output

    return _runtime(Wrapper, mode)


# ---- fixtures --------------------------------------------------------------


def _endpoint_pools(num_blocks: int, device: torch.device = CPU) -> Pools:
    pools = _fresh_pools(num_blocks)
    pools["endpoint_history_pool"] = torch.zeros(num_blocks, 8, dtype=torch.int32)
    pools["endpoint_book_pool"] = torch.zeros(num_blocks, 6, dtype=torch.int32)
    return {
        key: [tensor.to(device) for tensor in value]
        if isinstance(value, list)
        else value.to(device)
        for key, value in pools.items()
    }


def _transaction_pools(pools: Pools) -> advance.TransactionPools:
    return advance.TransactionPools(
        channel=pools["channel_pools"],
        time=pools["time_pools"],
        valid_length=pools["len_pools"],
        h=pools["h_pool"],
        c=pools["c_pool"],
        queue=pools["queue_pool"],
        book=pools["book_pool"],
        frontend_raw=pools["frontend_raw_pool"],
        frontend_mel=pools["frontend_mel_pool"],
        frontend_counter=pools["frontend_counter_pool"],
        endpoint_history=pools["endpoint_history_pool"],
        endpoint_book=pools["endpoint_book_pool"],
    )


def _core(device: torch.device = CPU) -> Any:
    core = _tiny_core(seed=1)
    for name in ("encoder", "lid", "predictor", "joint", "featurizer"):
        getattr(core, name).to(device)
    return core


def _call_bindings(core: Any) -> tg.TransactionCallBindings:
    return tg.TransactionCallBindings(
        adapter=advance.make_mrv1_adapter(
            hidden_size=CARRIER_HIDDEN, park_id=PARK_ID, blank_id=core.blank_id
        ),
        decode_resolver=_fixed_resolver(rnnt.decode_dense_masked_frames),
        encoder_transition=None,
        bucket_transition=None,
        placeholder_id=PLACEHOLDER_ID,
        park_id=PARK_ID,
        eou_token_id=EOU,
        graph_covers_decode=False,
    )


def _binding(
    core: Any,
    pools: Pools,
    call: tg.TransactionCallBindings,
    runtime: GraphRuntime,
    *,
    populations: tuple[int, ...] = (1, 2),
    device: torch.device = CPU,
) -> tg.TransactionGraphBinding:
    binding = tg.TransactionGraphBinding(
        populations=list(populations),
        geometries=(GEOM_REG, GEOM_FINAL),
        maximum_population=4,
        vllm_config=object(),
        runtime=runtime,
    )
    if device.type == "cuda":
        advance.warmup_advance_model_rows_scatter(**pools)
    binding.capture(
        core,
        pools=_transaction_pools(pools),
        call=call,
        hidden_size=CARRIER_HIDDEN,
        device=device,
    )
    return binding


def _policy(plan: Any, mode: int, threshold: int, residue: int) -> Any:
    rows = plan.num_decodes + plan.num_prefills
    return replace(
        plan,
        endpoint_mode=torch.full((rows,), mode, dtype=torch.int64),
        endpoint_threshold_frames=torch.full((rows,), threshold, dtype=torch.int64),
        endpoint_residue_frames=torch.full((rows,), residue, dtype=torch.int64),
    )


def _step(
    core: Any,
    pools: Pools,
    call: tg.TransactionCallBindings,
    ids: list[int],
    embeds: torch.Tensor,
    plan: Any,
    graph: Any,
    *,
    capture: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = embeds.device
    sink = _CommitRecorder()
    out = advance.advance_model_rows(
        core,
        torch.tensor(ids, dtype=torch.long, device=device),
        embeds,
        plan,
        adapter=call.adapter,
        decode_resolver=call.decode_resolver,
        placeholder_id=PLACEHOLDER_ID,
        park_id=PARK_ID,
        eou_token_id=EOU,
        commit_sink=sink,
        capture=capture,
        transaction_graph=graph,
        **pools,
    )
    assert sink.log == ["reserve", "stage"]
    return (
        out.clone(),
        sink.staged[0].clone(),
        sink.plans[0].progress.clone(),
    )


def _chunk(
    seed: int, *, seq: int, geometry: int, final: bool, prompt: int
) -> torch.Tensor:
    torch.manual_seed(seed)
    samples = FINAL_SAMPLES if geometry == GEOM_FINAL else REG_SAMPLES
    return _envelope(
        torch.randn(samples) * 0.01,
        final=final,
        seq=seq,
        geometry=geometry,
        prompt=prompt,
    )


def _drive(
    core: Any,
    pools: Pools,
    call: tg.TransactionCallBindings,
    graph: Any,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Two sessions through fresh, continuing and final CHUNKs plus echoes.

    Rows use block ids, prompts, row orders and endpoint policies that differ
    from the capture inputs, and the populations change between steps.
    """
    device = pools["book_pool"].device
    records: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    sessions = {"a": (2, GEOM_REG, 1), "c": (3, GEOM_REG, 2)}
    policies = {"a": (1, 2, 1), "c": (0, 0, 0)}

    def echo(tokens: dict[str, int]) -> None:
        for _ in range(CAP + 2):
            active = [name for name, token in tokens.items() if token != PARK_ID]
            if not active:
                return
            plan = _plan(
                decodes=[sessions[name][0] for name in active],
                num_pool_blocks=4,
                geometries=[sessions[name][1] for name in active],
                prompts=[sessions[name][2] for name in active],
                request_ids=tuple(active),
            )
            mode, threshold, residue = policies[active[0]]
            record = _step(
                core,
                pools,
                call,
                [tokens[name] for name in active],
                torch.zeros(len(active), CARRIER_HIDDEN, device=device),
                _policy(plan, mode, threshold, residue),
                graph,
            )
            records.append(record)
            for index, name in enumerate(active):
                tokens[name] = int(record[0][index, 0])
        raise AssertionError("replay queue did not drain")

    def chunk(names: list[str], seq: int, *, fresh: bool, final: bool) -> None:
        embeds = torch.stack(
            [
                _chunk(
                    11 * seq + index,
                    seq=seq,
                    geometry=sessions[name][1],
                    final=final,
                    prompt=sessions[name][2],
                )
                for index, name in enumerate(names)
            ]
        ).to(device)
        plan = _plan(
            prefills=[sessions[name][0] for name in names],
            num_pool_blocks=4,
            has_initial=[not fresh] * len(names),
            geometries=[sessions[name][1] for name in names],
            prompts=[sessions[name][2] for name in names],
            request_ids=tuple(names),
        )
        mode, threshold, residue = policies[names[0]]
        record = _step(
            core,
            pools,
            call,
            [PLACEHOLDER_ID] * len(names),
            embeds,
            _policy(plan, mode, threshold, residue),
            graph,
        )
        records.append(record)
        echo({name: int(record[0][index, 0]) for index, name in enumerate(names)})

    chunk(["a", "c"], 0, fresh=True, final=False)
    chunk(["c", "a"], 1, fresh=False, final=False)
    chunk(["a"], 2, fresh=False, final=False)
    sessions["b"] = (1, GEOM_FINAL, 3)
    policies["b"] = (1, 1, 0)
    chunk(["b"], 0, fresh=True, final=True)
    return records


def _assert_records_equal(actual: list[Any], expected: list[Any]) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        for left, right in zip(got, want, strict=True):
            torch.testing.assert_close(left.cpu(), right.cpu(), atol=0, rtol=0)


def _replay_parity(device: torch.device, runtime: GraphRuntime) -> None:
    core = _core(device)
    call = _call_bindings(core)
    eager_pools = _endpoint_pools(4, device)
    graph_pools = _clone_pools(eager_pools)
    binding = _binding(core, graph_pools, call, runtime, device=device)
    _assert_pools_equal(graph_pools, eager_pools)
    assert binding.ready

    expected = _drive(core, eager_pools, call, None)
    actual = _drive(core, graph_pools, call, binding)

    _assert_records_equal(actual, expected)
    _assert_pools_equal(graph_pools, eager_pools)
    emitted = {
        int(token)
        for rows, _status, _progress in expected
        for token in rows[:, 0].tolist()
    }
    assert emitted - {PARK_ID}, "the trajectory must emit labels"
    assert all(not bool(status.any()) for _rows, status, _progress in expected)
    counts = binding.counters()
    assert counts["fallbacks"] == 0
    assert counts["replays"] == len(expected)
    for label in (
        "chunk-fresh:g0:b2",
        "chunk-continuing:g0:b2",
        "chunk-continuing:g0:b1",
        "chunk-fresh:g2:b1",
        "replay:b1",
    ):
        assert counts[f"replays:{label}"] >= 1, label


# ---- CPU: arena, guard, classification and replay plumbing -----------------


@pytest.mark.cpu_test
def test_nested_bindings_run_eagerly_only_inside_the_capture_scope() -> None:
    runtime = _rerun_runtime()
    assert nested_graph_mode(runtime) == "graph"
    with outer_transaction_capture():
        assert nested_graph_mode(runtime) == "none"
    assert nested_graph_mode(runtime) == "graph"


@pytest.mark.cpu_test
def test_host_staging_is_rejected_inside_the_capture_scope() -> None:
    staging = advance.HostStaging(4)
    vector = torch.arange(2, dtype=torch.int64)
    with outer_transaction_capture():
        with pytest.raises(RuntimeError, match="frozen host source"):
            advance._h2d(vector, CPU)
        with pytest.raises(RuntimeError, match="frozen host source"):
            staging.stage("geometry", vector, CPU)
        with pytest.raises(RuntimeError, match="frozen host source"):
            staging.stage_bucket(0, vector, CPU)
    assert torch.equal(advance._h2d(vector, CPU), vector)


def _fresh_chunk_inputs(
    core: Any, pools: Pools, call: tg.TransactionCallBindings, blocks: list[int]
) -> Any:
    plan = _policy(
        _plan(prefills=blocks, num_pool_blocks=4, prompts=[1] * len(blocks)),
        1,
        3,
        2,
    )
    embeds = torch.stack(
        [_chunk(3, seq=0, geometry=GEOM_REG, final=False, prompt=1)] * len(blocks)
    )
    return advance.prepare_transaction(
        core,
        torch.full((len(blocks),), PLACEHOLDER_ID, dtype=torch.long),
        embeds,
        plan,
        pools=_transaction_pools(pools),
        eou_token_id=EOU,
        adapter=call.adapter,
        decode_resolver=call.decode_resolver,
        encoder_transition=None,
        bucket_transition=None,
        placeholder_id=PLACEHOLDER_ID,
        park_id=PARK_ID,
        has_commit_sink=True,
        capture=False,
        graph_covers_decode=False,
        memory_profile=False,
        native_burst_handoff=None,
    )


@pytest.mark.cpu_test
def test_arena_stager_matches_live_staging_bit_identical() -> None:
    core = _core()
    call = _call_bindings(core)
    live_pools = _endpoint_pools(4)
    arena_pools = _clone_pools(live_pools)
    for pools in (live_pools, arena_pools):
        pools["book_pool"][1:, _BOOK["last_label"]] = core.blank_id
    inputs = _fresh_chunk_inputs(core, live_pools, call, [3, 1])
    arena_inputs = replace(inputs, pools=_transaction_pools(arena_pools))
    arena = tg.TransactionArena(tg._arena_layout(inputs), CPU)
    arena.upload(arena_inputs)
    for name, vector in inputs.staged_vectors():
        assert torch.equal(arena.views[name], vector), name

    ids = torch.full((2,), PLACEHOLDER_ID, dtype=torch.long)
    embeds = torch.stack(
        [_chunk(3, seq=0, geometry=GEOM_REG, final=False, prompt=1)] * 2
    )
    live = advance.run_transaction_body(
        core,
        ids,
        embeds,
        inputs,
        advance.LiveTransactionStager(inputs, advance.HostStaging(4)),
    )
    advance.execute_transaction_scatters(live.scatter_ops)
    with outer_transaction_capture():
        staged = advance.run_transaction_body(
            core, ids, embeds, arena_inputs, tg.ArenaTransactionStager(arena)
        )
        advance.execute_transaction_scatters(staged.scatter_ops)
    for left, right in (
        (staged.projection_rows, live.projection_rows),
        (staged.status, live.status),
        (staged.progress, live.progress),
    ):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    _assert_pools_equal(arena_pools, live_pools)


@pytest.mark.cpu_test
def test_arena_rejects_vectors_outside_its_layout() -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    arena = tg.TransactionArena(
        tg._arena_layout(_fresh_chunk_inputs(core, pools, call, [1])), CPU
    )
    with pytest.raises(ValueError, match="differs from the arena"):
        arena.upload(_fresh_chunk_inputs(core, pools, call, [1, 2]))
    with pytest.raises(ValueError, match="only int64 and bool"):
        tg.TransactionArena((("x", torch.float32, (1,)),), CPU)


@pytest.mark.cpu_test
def test_arena_alternates_host_buffers_across_calls() -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    first = _fresh_chunk_inputs(core, pools, call, [1])
    second = _fresh_chunk_inputs(core, pools, call, [3])
    arena = tg.TransactionArena(tg._arena_layout(first), CPU)
    arena.upload(first)
    arena.upload(second)
    assert int(arena._host_views[0]["composed_indices"][0]) == 1
    assert int(arena._host_views[1]["composed_indices"][0]) == 3
    assert int(arena.views["composed_indices"][0]) == 3


@pytest.mark.cpu_test
def test_classification_covers_declared_shapes_only() -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    cell, reason = tg.classify_transaction(
        _fresh_chunk_inputs(core, pools, call, [1, 2])
    )
    assert cell == tg.TransactionCell(tg.CELL_FRESH_CHUNK, GEOM_REG, 2)
    assert reason is None
    mixed = replace(
        _fresh_chunk_inputs(core, pools, call, [1, 2]),
        fresh_positions=torch.tensor([0]),
    )
    assert tg.classify_transaction(mixed) == (None, "mixed-fresh")


@pytest.mark.cpu_test
def test_replay_matches_eager_across_varied_plans_on_cpu() -> None:
    _replay_parity(CPU, _rerun_runtime())


@pytest.mark.cpu_test
def test_capture_input_stores_no_page_and_publishes_every_cell() -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    for tensor in (pools["book_pool"], pools["queue_pool"]):
        tensor.fill_(3)
    before = _clone_pools(pools)
    binding = _binding(core, pools, call, _rerun_runtime(), populations=(1,))
    _assert_pools_equal(pools, before)
    assert binding.ready
    assert [cell.label for cell in binding.cells] == [
        "chunk-continuing:g0:b1",
        "chunk-continuing:g2:b1",
        "chunk-fresh:g0:b1",
        "chunk-fresh:g2:b1",
        "replay:b1",
    ]
    assert advance.consume_batch_stats() is None


def _first_step(
    graph: Any,
    pools: Pools,
    *,
    core: Any,
    call: tg.TransactionCallBindings,
    blocks: list[int],
    fresh: list[bool],
    capture: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embeds = torch.stack(
        [
            _chunk(5 + i, seq=0, geometry=GEOM_REG, final=False, prompt=0)
            for i in range(len(blocks))
        ]
    ).to(pools["book_pool"].device)
    plan = _plan(
        prefills=blocks,
        num_pool_blocks=4,
        has_initial=[not value for value in fresh],
    )
    return _step(
        core,
        pools,
        call,
        [PLACEHOLDER_ID] * len(blocks),
        embeds,
        plan,
        graph,
        capture=capture,
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("blocks", "fresh", "capture", "key"),
    [
        ([1, 2], [True, False], False, "fallbacks:uncovered:mixed-fresh"),
        ([1, 2, 3], [True] * 3, False, "fallbacks:chunk-fresh:g0:b3:undeclared"),
        ([1], [True], True, "fallbacks:chunk-fresh:g0:b1:capture"),
    ],
)
def test_uncovered_calls_run_eager_and_are_counted(
    blocks: list[int], fresh: list[bool], capture: bool, key: str
) -> None:
    core = _core()
    call = _call_bindings(core)
    graph_pools = _endpoint_pools(4)
    eager_pools = _clone_pools(graph_pools)
    binding = _binding(core, graph_pools, call, _rerun_runtime())
    for pools in (graph_pools, eager_pools):
        pools["book_pool"][1:, _BOOK["last_label"]] = core.blank_id
    kwargs = dict(core=core, call=call, blocks=blocks, fresh=fresh, capture=capture)
    expected = _first_step(None, eager_pools, **kwargs)
    actual = _first_step(binding, graph_pools, **kwargs)
    _assert_records_equal([actual], [expected])
    _assert_pools_equal(graph_pools, eager_pools)
    counts = binding.counters()
    assert counts[key] == 1
    assert counts["fallbacks"] == 1 and counts["replays"] == 0


@pytest.mark.cpu_test
def test_relocated_pools_fall_back_instead_of_replaying() -> None:
    core = _core()
    call = _call_bindings(core)
    captured_pools = _endpoint_pools(4)
    binding = _binding(core, captured_pools, call, _rerun_runtime())
    moved = _clone_pools(captured_pools)
    reference = _clone_pools(captured_pools)
    kwargs = dict(core=core, call=call, blocks=[1], fresh=[True])
    actual = _first_step(binding, moved, **kwargs)
    expected = _first_step(None, reference, **kwargs)
    _assert_records_equal([actual], [expected])
    _assert_pools_equal(captured_pools, _endpoint_pools(4))
    assert binding.counters()["fallbacks:chunk-fresh:g0:b1:signature"] == 1


@pytest.mark.cpu_test
def test_replay_reserves_before_any_page_write_and_stages_after() -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    binding = _binding(core, pools, call, _rerun_runtime())
    untouched = _clone_pools(pools)
    seen: list[str] = []

    class Sink(_CommitRecorder):
        def reserve(self, plan: Any) -> Any:
            _assert_pools_equal(pools, untouched)
            seen.append("reserve")
            return super().reserve(plan)

    sink = Sink()
    advance.advance_model_rows(
        core,
        torch.tensor([PLACEHOLDER_ID]),
        _chunk(5, seq=0, geometry=GEOM_REG, final=False, prompt=0).unsqueeze(0),
        _plan(prefills=[1], num_pool_blocks=4),
        adapter=call.adapter,
        decode_resolver=call.decode_resolver,
        placeholder_id=PLACEHOLDER_ID,
        park_id=PARK_ID,
        eou_token_id=EOU,
        commit_sink=sink,
        transaction_graph=binding,
        **pools,
    )
    assert seen == ["reserve"] and sink.log == ["reserve", "stage"]
    assert int(sink.staged[0][0]) == 0
    assert int(pools["frontend_counter_pool"][1].abs().sum()) > 0
    assert binding.counters()["replays:chunk-fresh:g0:b1"] == 1


@pytest.mark.cpu_test
def test_replay_failure_cancels_the_reservation_once() -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    binding = _binding(core, pools, call, _rerun_runtime())
    graph = binding._graphs[tg.TransactionCell(tg.CELL_FRESH_CHUNK, GEOM_REG, 1)]

    def fail() -> None:
        raise RuntimeError("replay failed")

    graph.replay = fail
    sink = _CommitRecorder()
    with pytest.raises(RuntimeError, match="replay failed"):
        advance.advance_model_rows(
            core,
            torch.tensor([PLACEHOLDER_ID]),
            _chunk(5, seq=0, geometry=GEOM_REG, final=False, prompt=0).unsqueeze(0),
            _plan(prefills=[1], num_pool_blocks=4),
            adapter=call.adapter,
            decode_resolver=call.decode_resolver,
            placeholder_id=PLACEHOLDER_ID,
            park_id=PARK_ID,
            eou_token_id=EOU,
            commit_sink=sink,
            transaction_graph=binding,
            **pools,
        )
    assert sink.log == ["reserve", "cancel"]
    assert binding.counters()["replays"] == 0


@pytest.mark.cpu_test
def test_stale_lease_fails_before_replay_writes_a_page() -> None:
    from vllm.models.nemotron_asr.commit_sink import BoundedCommitSink
    from vllm.models.nemotron_asr.plan import SessionRegistry

    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    binding = _binding(core, pools, call, _rerun_runtime())
    before = _clone_pools(pools)
    with pytest.raises(ValueError, match="not registered"):
        advance.advance_model_rows(
            core,
            torch.tensor([PLACEHOLDER_ID]),
            _chunk(5, seq=0, geometry=GEOM_REG, final=False, prompt=0).unsqueeze(0),
            _plan(prefills=[1], num_pool_blocks=4),
            adapter=call.adapter,
            decode_resolver=call.decode_resolver,
            placeholder_id=PLACEHOLDER_ID,
            park_id=PARK_ID,
            eou_token_id=EOU,
            commit_sink=BoundedCommitSink(SessionRegistry(), max_rows=4),
            transaction_graph=binding,
            **pools,
        )
    _assert_pools_equal(pools, before)
    assert binding.counters()["replays"] == 0


@pytest.mark.cpu_test
def test_failed_capture_publishes_nothing_and_every_call_runs_eager() -> None:
    core = _core()
    call = _call_bindings(core)

    def broken(*_args: Any) -> Any:
        raise RuntimeError("adapter blew up")

    broken_call = replace(call, adapter=broken)
    pools = _endpoint_pools(4)
    binding = tg.TransactionGraphBinding(
        populations=[1],
        geometries=(GEOM_REG,),
        maximum_population=4,
        vllm_config=object(),
        runtime=_rerun_runtime(),
    )
    with pytest.raises(RuntimeError, match="adapter blew up"):
        binding.capture(
            core,
            pools=_transaction_pools(pools),
            call=broken_call,
            hidden_size=CARRIER_HIDDEN,
            device=CPU,
        )
    assert not binding.ready and binding.counters()["captured_cells"] == 0
    _first_step(binding, pools, core=core, call=call, blocks=[1], fresh=[True])
    assert binding.counters()["fallbacks:chunk-fresh:g0:b1:uncaptured"] == 1


@pytest.mark.cpu_test
def test_replayed_rows_are_owned_by_the_caller() -> None:
    core = _core()
    call = _call_bindings(core)
    pools = _endpoint_pools(4)
    binding = _binding(core, pools, call, _rerun_runtime())
    for tensor in (pools["book_pool"],):
        tensor[1:, _BOOK["last_label"]] = core.blank_id
    first = _first_step(binding, pools, core=core, call=call, blocks=[1], fresh=[True])
    kept = first[0].clone()
    _first_step(binding, pools, core=core, call=call, blocks=[2, 3], fresh=[True] * 2)
    graph = binding._graphs[tg.TransactionCell(tg.CELL_FRESH_CHUNK, GEOM_REG, 1)]
    assert first[0].data_ptr() != graph.rows_out.data_ptr()
    torch.testing.assert_close(first[0], kept, atol=0, rtol=0)


@pytest.mark.cpu_test
def test_population_declaration_is_validated() -> None:
    for populations in ([], [0], [5], [1, 1], ["1"]):
        with pytest.raises(ValueError, match="transaction_graph_populations"):
            tg.TransactionGraphBinding(
                populations=populations,
                geometries=(0,),
                maximum_population=4,
                vllm_config=object(),
            )


@pytest.mark.cpu_test
def test_counters_reach_the_model_from_an_engine_core() -> None:
    binding = tg.TransactionGraphBinding(
        populations=[1],
        geometries=(0,),
        maximum_population=1,
        vllm_config=object(),
    )
    model = SimpleNamespace(_transaction_graph=binding)
    engine = SimpleNamespace(
        model_executor=SimpleNamespace(
            driver_worker=SimpleNamespace(
                worker=SimpleNamespace(model_runner=SimpleNamespace(model=model))
            )
        )
    )
    counts = tg.transaction_graph_counters(engine)
    assert counts["replays:replay:b1"] == 0
    assert counts["replays:chunk-fresh:g0:b1"] == 0
    assert counts["captured_cells"] == 0
    off = SimpleNamespace(_transaction_graph=None)
    assert tg.transaction_graph_counters(off) == {}


# ---- CUDA: real graph replay -----------------------------------------------


@pytest.fixture
def exact_fp32() -> Iterator[None]:
    """Disable TF32 so eager and replay both run IEEE FP32 kernels."""
    targets = (
        torch.backends.cuda.matmul,
        torch.backends.cudnn.conv,
        torch.backends.cudnn.rnn,
    )
    saved = [target.fp32_precision for target in targets]
    for target in targets:
        target.fp32_precision = "ieee"
    try:
        yield
    finally:
        for target, value in zip(targets, saved, strict=True):
            target.fp32_precision = value


@pytest.mark.nemotron_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.usefixtures("exact_fp32")
@torch.inference_mode()
def test_cuda_graph_replay_matches_eager_across_varied_plans() -> None:
    _replay_parity(torch.device("cuda"), _cuda_runtime())


@pytest.mark.nemotron_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.usefixtures("exact_fp32")
@torch.inference_mode()
@pytest.mark.parametrize("corruption", ["sequence", "echo", "book"])
def test_cuda_graph_replay_rejects_corrupt_rows_like_eager(corruption: str) -> None:
    device = torch.device("cuda")
    core = _core(device)
    call = _call_bindings(core)
    eager_pools = _endpoint_pools(4, device)
    graph_pools = _clone_pools(eager_pools)
    binding = _binding(core, graph_pools, call, _cuda_runtime(), device=device)

    def run(pools: Pools, graph: Any) -> list[Any]:
        records = [
            _first_step(graph, pools, core=core, call=call, blocks=[2], fresh=[True])
        ]
        token = int(records[0][0][0, 0])
        if corruption == "sequence":
            embeds = _chunk(9, seq=5, geometry=GEOM_REG, final=False, prompt=0)
            embeds = embeds.to(device)
            plan = _plan(prefills=[2], num_pool_blocks=4, has_initial=[True])
            ids, rows = [PLACEHOLDER_ID], embeds.unsqueeze(0).to(device)
        else:
            if corruption == "book":
                pools["book_pool"][2, _BOOK["queue_head"]] = CAP + 5
            plan = _plan(decodes=[2], num_pool_blocks=4)
            wrong = token + 1 if token + 1 < VOCAB else 0
            ids = [wrong if corruption == "echo" else token]
            rows = torch.zeros(1, CARRIER_HIDDEN, device=device)
        before = _clone_pools(pools)
        records.append(_step(core, pools, call, ids, rows, plan, graph))
        _assert_pools_equal(pools, before)
        return records

    expected = run(eager_pools, None)
    actual = run(graph_pools, binding)
    _assert_records_equal(actual, expected)
    assert int(expected[1][1][0]) != 0
    _assert_pools_equal(graph_pools, eager_pools)
    assert binding.counters()["fallbacks"] == 0


@pytest.mark.nemotron_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.usefixtures("exact_fp32")
@torch.inference_mode()
def test_cuda_replayed_rows_survive_another_cells_replay() -> None:
    device = torch.device("cuda")
    core = _core(device)
    call = _call_bindings(core)
    pools = _endpoint_pools(4, device)
    binding = _binding(core, pools, call, _cuda_runtime(), device=device)
    first = _first_step(binding, pools, core=core, call=call, blocks=[1], fresh=[True])
    kept = first[0].clone()
    _first_step(binding, pools, core=core, call=call, blocks=[2, 3], fresh=[True] * 2)
    torch.cuda.synchronize()
    torch.testing.assert_close(first[0], kept, atol=0, rtol=0)
    assert binding.counters()["replays"] == 2
