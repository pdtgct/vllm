# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for bounded cache-aware RNN-T endpointing."""

from __future__ import annotations

import copy
import importlib
import inspect
from dataclasses import replace
from typing import Any, NoReturn

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]


def _fail(message: str) -> NoReturn:
    pytest.fail(message, pytrace=False)
    raise AssertionError(message)


def _module() -> Any:
    try:
        return importlib.import_module("vllm.models.nemotron_asr.endpointing")
    except ModuleNotFoundError:
        _fail("PORT-SEG-001 missing bounded cache-aware endpointing")


def _policy(
    *,
    stop_history_ms: int = 240,
    residue_frames: int = 2,
    capacity: int = 16,
) -> Any:
    return _module().EndpointPolicy.resolve(
        mode="greedy_blank",
        stop_history_ms=stop_history_ms,
        residue_frames=residue_frames,
        frame_stride_ms=80,
        history_capacity_frames=capacity,
    )


def _book(capacity: int = 16) -> Any:
    return _module().EndpointBook.initial(history_capacity_frames=capacity)


def _word_start(label: int) -> bool:
    return label in {10, 20, 30}


def _observe(
    book: Any,
    frames: list[list[int]],
    *,
    policy: Any | None = None,
    final_tail: bool = False,
) -> Any:
    return _module().observe_chunk(
        book,
        frames,
        is_word_start=_word_start,
        policy=policy or _policy(),
        final_tail=final_tail,
    )


def test_reference_policy_derives_threshold_and_active_window() -> None:
    # @spec PORT-SEG-001
    policy = _module().EndpointPolicy.resolve(
        mode="greedy_blank",
        stop_history_ms=800,
        residue_frames=2,
        frame_stride_ms=80,
        history_capacity_frames=12,
    )

    assert policy.mode == "greedy_blank"
    assert policy.stop_history_ms == 800
    assert policy.threshold_frames == 10
    assert policy.residue_frames == 2
    assert policy.active_window_frames == 12
    assert policy.history_capacity_frames == 12

    rounded = _module().EndpointPolicy.resolve(
        mode="greedy_blank",
        stop_history_ms=100,
        residue_frames=2,
        frame_stride_ms=80,
        history_capacity_frames=8,
    )
    assert rounded.threshold_frames == 2
    assert rounded.active_window_frames == 4


@pytest.mark.parametrize(
    ("kwargs", "pattern"),
    [
        ({"mode": "vad"}, "mode"),
        ({"stop_history_ms": 0}, "stop_history"),
        ({"stop_history_ms": -1}, "stop_history"),
        ({"residue_frames": -1}, "residue"),
        ({"frame_stride_ms": 0}, "stride"),
        ({"history_capacity_frames": 4}, "capacity"),
    ],
)
def test_invalid_or_uninstalled_policy_fails_closed(
    kwargs: dict[str, Any], pattern: str
) -> None:
    # @spec PORT-SEG-001 / PORT-RTC-001
    values: dict[str, Any] = {
        "mode": "greedy_blank",
        "stop_history_ms": 240,
        "residue_frames": 2,
        "frame_stride_ms": 80,
        "history_capacity_frames": 16,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=pattern):
        _module().EndpointPolicy.resolve(**values)


def test_disabled_policy_is_explicit_and_has_no_active_window() -> None:
    # @spec PORT-SEG-001
    policy = _module().EndpointPolicy.resolve(
        mode="disabled",
        stop_history_ms=None,
        residue_frames=0,
        frame_stride_ms=80,
        history_capacity_frames=16,
    )
    assert policy.mode == "disabled"
    assert policy.threshold_frames == 0
    assert policy.active_window_frames == 0


@pytest.mark.parametrize(
    ("labels", "expected_name"),
    [
        ([], "BLANK"),
        ([10], "WORD_START"),
        ([11], "NON_WORD_START"),
        ([10, 11], "NON_WORD_START"),
        ([11, 20], "WORD_START"),
    ],
)
def test_one_dense_endpoint_symbol_uses_the_final_label_of_each_frame(
    labels: list[int], expected_name: str
) -> None:
    # @spec PORT-DEC-001 / PORT-SEG-002
    symbol = _module().frame_symbol(labels, is_word_start=_word_start)
    assert symbol.name == expected_name


def test_multiple_labels_remain_in_transcript_order_but_use_one_frame_slot() -> None:
    # @spec PORT-DEC-001 / PORT-SEG-002
    transition = _observe(_book(), [[10, 11, 20], [], [21]])

    assert transition.emitted_labels == (10, 11, 20, 21)
    assert tuple(symbol.name for symbol in transition.observed_symbols) == (
        "WORD_START",
        "BLANK",
        "NON_WORD_START",
    )
    assert transition.next_book.history_length == 3


def test_endpoint_ring_is_fixed_width_and_keeps_newest_frame_symbols() -> None:
    # @spec PORT-SEG-002 / PORT-STATE-021
    book = _book(capacity=5)
    policy = _policy(capacity=5)
    expected: list[str] = []

    assert len(book.history) == 5
    assert all(symbol.name == "BLANK" for symbol in book.history)
    assert book.history_length == 0

    for index in range(40):
        labels = [10] if index % 3 == 0 else []
        transition = _observe(book, [labels], policy=policy)
        book = transition.next_book
        expected.append("WORD_START" if labels else "BLANK")
        expected = expected[-5:]

        assert len(book.history) == 5
        assert book.history_length <= 5
        assert [symbol.name for symbol in book.newest(book.history_length)] == expected


@pytest.mark.parametrize(
    ("silence_frames", "detected"),
    [(3, False), (4, False), (5, True)],
)
def test_greedy_blank_threshold_is_strict(silence_frames: int, detected: bool) -> None:
    # @spec PORT-SEG-002
    policy = _policy(stop_history_ms=240, residue_frames=2, capacity=8)
    book = _book(capacity=8)
    book = _observe(book, [[10]], policy=policy).next_book
    transition = _observe(book, [[] for _ in range(silence_frames)], policy=policy)

    assert transition.detector_trace.silent_frames == silence_frames - 1
    assert transition.is_eou is detected


def test_residue_shortens_effective_end_without_moving_newest_pivot() -> None:
    # @spec PORT-SEG-001 / PORT-SEG-002
    policy = _policy(stop_history_ms=240, residue_frames=2, capacity=8)
    transition = _observe(
        _observe(_book(8), [[10]], policy=policy).next_book,
        [[], [], [], [], []],
        policy=policy,
    )

    assert transition.detector_trace.active_window_length == 5
    assert transition.detector_trace.pivot == 4
    assert transition.detector_trace.effective_end == 3
    assert transition.detector_trace.silent_frames == 4
    assert transition.is_eou is True


def test_disabled_and_final_tail_never_make_model_eou_decisions() -> None:
    # @spec PORT-SESS-003 / PORT-SEG-002 / PORT-SEG-003
    enabled = _policy()
    disabled = _module().EndpointPolicy.resolve(
        mode="disabled",
        stop_history_ms=None,
        residue_frames=0,
        frame_stride_ms=80,
        history_capacity_frames=16,
    )
    book = _observe(_book(), [[10]], policy=enabled).next_book
    silence: list[list[int]] = [[] for _ in range(8)]

    assert _observe(book, silence, policy=disabled).is_eou is False
    assert _observe(book, silence, policy=enabled, final_tail=True).is_eou is False


def test_regular_unit_eligibility_is_independent_of_session_phase() -> None:
    # @spec PORT-SESS-003 / PORT-SEG-003
    signature = inspect.signature(_module().observe_chunk)
    assert "final_tail" in signature.parameters
    assert "session_phase" not in signature.parameters

    policy = _policy()
    book = _observe(_book(), [[10]], policy=policy).next_book
    assert (
        _observe(
            book,
            [[], [], [], [], []],
            policy=policy,
            final_tail=False,
        ).is_eou
        is True
    )


def test_eou_disarms_and_continuing_silence_cannot_emit_empty_segments() -> None:
    # @spec PORT-SEG-002 / PORT-SEG-007
    policy = _policy()
    book = _observe(_book(), [[10]], policy=policy).next_book
    detected = _observe(book, [[], [], [], [], []], policy=policy)
    assert detected.is_eou is True
    assert detected.next_book.endpoint_armed is False
    assert detected.next_book.segment_has_output is False
    assert detected.next_book.segment_generation == 1

    repeated = _observe(
        detected.next_book,
        [[], [], [], [], []],
        policy=policy,
    )
    assert repeated.is_eou is False
    assert repeated.next_book.segment_generation == 1

    # NEW silence after the reset belongs to the next segment. Its first
    # nonblank can rearm and close in the same regular CHUNK.
    closed = _observe(repeated.next_book, [[20]], policy=policy)
    assert closed.is_eou is True
    assert closed.emitted_labels == (20,)
    assert closed.next_book.endpoint_armed is False
    assert closed.next_book.segment_has_output is False
    assert closed.next_book.segment_generation == 2

    # An immediate new label must not reuse silence from the closed segment.
    immediate = _observe(detected.next_book, [[20]], policy=policy)
    assert immediate.is_eou is False
    assert immediate.next_book.endpoint_armed is True
    assert immediate.next_book.segment_has_output is True
    assert immediate.next_book.segment_generation == 1
    later = _observe(immediate.next_book, [[], [], [], [], []], policy=policy)
    assert later.is_eou is True
    assert later.next_book.segment_generation == 2


def test_one_chunk_produces_at_most_one_boolean_eou_decision() -> None:
    # @spec PORT-SEG-002
    transition = _observe(
        _observe(_book(), [[10]], policy=_policy()).next_book,
        [[], [], [], [], [], [], [], []],
    )

    assert isinstance(transition.is_eou, bool)
    assert not hasattr(transition, "eou_detected_at")
    assert not hasattr(transition, "tail_output")
    assert not hasattr(transition, "clipped_output")


def test_detected_chunk_orders_all_labels_then_eou_then_park() -> None:
    # @spec PORT-SEG-003 / PORT-INT-002
    compose = _module().compose_chunk_output

    assert compose(
        labels=[7, 8, 9],
        is_eou=True,
        eou_token_id=101,
        park_token_id=103,
    ) == [7, 8, 9, 101, 103]
    assert compose(
        labels=[],
        is_eou=False,
        eou_token_id=101,
        park_token_id=103,
    ) == [103]


def test_output_budget_adds_eou_without_reducing_label_capacity() -> None:
    # @spec PORT-DEC-005 / PORT-INT-002
    emission_bound = _module().max_emission_tokens

    assert emission_bound(max_valid_frames=14, max_symbols_per_step=10) == 142
    with pytest.raises(ValueError, match="142|emission"):
        _module().validate_emission_budget(
            caller_max_tokens=141,
            max_valid_frames=14,
            max_symbols_per_step=10,
        )
    assert (
        _module().validate_emission_budget(
            caller_max_tokens=1_000,
            max_valid_frames=14,
            max_symbols_per_step=10,
        )
        == 142
    )


def test_failed_candidate_does_not_mutate_prior_endpoint_book() -> None:
    # @spec PORT-SEG-003 / PORT-SEG-007
    policy = _policy()
    book = _observe(_book(), [[10]], policy=policy).next_book
    before = copy.deepcopy(book)

    candidate = _observe(book, [[], [], [], [], []], policy=policy)
    assert candidate.is_eou is True
    assert book == before
    # The transition becomes authoritative only when its next_book is
    # committed by the surrounding state transaction.
    assert candidate.next_book != before


def test_forced_eou_with_output_emits_control_and_increments_generation() -> None:
    # @spec PORT-SEG-004 / PORT-SEG-007
    module = _module()
    book = _observe(_book(), [[10]], policy=_policy()).next_book

    forced = module.apply_forced_eou(
        book,
        request_generation=7,
        expected_request_generation=7,
        eou_token_id=101,
        park_token_id=103,
    )

    assert forced.output_ids == (101, 103)
    assert forced.reason == "forced"
    assert forced.empty_segment is False
    assert forced.next_book.segment_generation == 1
    assert forced.next_book.endpoint_armed is False
    assert forced.next_book.segment_has_output is False


def test_forced_eou_without_output_is_ordered_park_only_noop() -> None:
    # @spec PORT-SEG-004
    forced = _module().apply_forced_eou(
        _book(),
        request_generation=7,
        expected_request_generation=7,
        eou_token_id=101,
        park_token_id=103,
    )

    assert forced.output_ids == (103,)
    assert forced.empty_segment is True
    assert forced.completion is None
    assert forced.next_book.segment_generation == 0


def test_model_eou_winning_same_generation_coalesces_queued_force() -> None:
    # @spec PORT-SEG-004 / PORT-SEG-007
    module = _module()
    policy = _policy()
    armed = _observe(_book(), [[10]], policy=policy).next_book
    queued = module.queue_forced_eou(armed, request_generation=7)
    model = _observe(
        queued,
        [[], [], [], [], []],
        policy=policy,
    )
    assert model.is_eou is True
    assert model.next_book.segment_generation == 1

    forced = module.apply_forced_eou(
        model.next_book,
        request_generation=7,
        expected_request_generation=7,
        eou_token_id=101,
        park_token_id=103,
    )
    assert forced.coalesced is True
    assert forced.output_ids == (103,)
    assert forced.completion is None
    assert forced.next_book.segment_generation == 1


def test_tensor_endpoint_transition_matches_strict_reference_boundary() -> None:
    # @spec PORT-SEG-002 / PORT-SEG-003 / PORT-SEG-007
    module = _module()
    history = torch.zeros(1, 8, dtype=torch.int32)
    book = torch.zeros(1, 6, dtype=torch.int32)
    counts = torch.tensor([[1, 0, 0, 0, 0, 0]], dtype=torch.int32)
    tokens = torch.zeros(1, 10, dtype=torch.int32)
    tokens[0, 0] = 7

    transition = module.observe_chunk_tensors(
        history=history,
        book=book,
        frame_emission_counts=counts,
        valid_frame_lengths=torch.tensor([6]),
        token_ids=tokens,
        token_lengths=torch.tensor([1], dtype=torch.int32),
        final_tail=torch.tensor([False]),
        mode=torch.tensor([1]),
        threshold_frames=torch.tensor([3]),
        residue_frames=torch.tensor([2]),
        eou_token_id=19,
        row_clean=torch.tensor([True]),
    )

    assert transition.is_eou.tolist() == [True]
    assert transition.token_ids[0, :2].tolist() == [7, 19]
    assert transition.token_lengths.tolist() == [2]
    assert transition.book[0, 2:5].tolist() == [0, 1, 0]


def test_tensor_endpoint_eou_has_headroom_after_a_full_label_burst() -> None:
    # @spec PORT-DEC-001 / PORT-DEC-005 / PORT-SEG-003
    module = _module()
    tokens = torch.tensor([[7, 8]], dtype=torch.int32)

    transition = module.observe_chunk_tensors(
        history=torch.zeros(1, 8, dtype=torch.int32),
        book=torch.zeros(1, 6, dtype=torch.int32),
        frame_emission_counts=torch.tensor([[1, 0, 0, 0, 0, 0]], dtype=torch.int32),
        valid_frame_lengths=torch.tensor([6]),
        token_ids=tokens,
        token_lengths=torch.tensor([2], dtype=torch.int32),
        final_tail=torch.tensor([False]),
        mode=torch.tensor([1]),
        threshold_frames=torch.tensor([3]),
        residue_frames=torch.tensor([2]),
        eou_token_id=19,
        row_clean=torch.tensor([True]),
    )

    assert transition.is_eou.tolist() == [True]
    assert transition.overflow.tolist() == [False]
    assert transition.token_ids.tolist() == [[7, 8, 19]]
    assert transition.token_lengths.tolist() == [3]


def test_tensor_endpoint_equality_does_not_fire_and_final_tail_is_inert() -> None:
    # @spec PORT-SEG-002 / PORT-SEG-003
    module = _module()
    common = dict(
        history=torch.zeros(2, 8, dtype=torch.int32),
        book=torch.zeros(2, 6, dtype=torch.int32),
        frame_emission_counts=torch.tensor(
            [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.int32
        ),
        valid_frame_lengths=torch.tensor([4, 5]),
        token_ids=torch.zeros(2, 10, dtype=torch.int32),
        token_lengths=torch.tensor([1, 1], dtype=torch.int32),
        final_tail=torch.tensor([False, True]),
        mode=torch.tensor([1, 1]),
        threshold_frames=torch.tensor([3, 3]),
        residue_frames=torch.tensor([2, 2]),
        eou_token_id=19,
        row_clean=torch.tensor([True, True]),
    )
    common["token_ids"][:, 0] = 7

    transition = module.observe_chunk_tensors(**common)

    assert transition.is_eou.tolist() == [False, False]
    assert transition.token_lengths.tolist() == [1, 1]
    assert transition.book[1].tolist() == [0, 0, 0, 0, 0, 0]


def test_tensor_forced_endpoint_advances_only_a_nonempty_segment() -> None:
    # @spec PORT-SEG-004 / PORT-SEG-007
    module = _module()
    book = torch.zeros(2, 6, dtype=torch.int32)
    book[:, 2] = 1
    book[:, 3] = torch.tensor([4, 7])
    book[:, 4] = torch.tensor([1, 0])
    book[:, 5] = 9

    transition = module.apply_forced_eou_tensors(
        book=book,
        selected_rows=torch.tensor([True, True]),
    )

    assert transition.is_eou.tolist() == [True, False]
    assert transition.book[:, 2].tolist() == [0, 0]
    assert transition.book[:, 3].tolist() == [5, 7]
    assert transition.book[:, 4].tolist() == [0, 0]
    assert transition.book[:, 5].tolist() == [0, 0]
    assert book[:, 3].tolist() == [4, 7]


def test_stale_forced_generation_is_ignored_and_preserves_book() -> None:
    # @spec PORT-SEG-007
    book = _observe(_book(), [[10]], policy=_policy()).next_book
    before = copy.deepcopy(book)

    ignored = _module().apply_forced_eou(
        book,
        request_generation=6,
        expected_request_generation=7,
        eou_token_id=101,
        park_token_id=103,
    )

    assert ignored is None
    assert book == before


# Public NeMo PIN: de242add77945a110568c7c44bdae4891451851e.
# Goldens below execute that PIN's actual detect_eou_near_pivot function,
# not a local reconstruction. Runtime tests require neither NeMo nor Git.
# Reproduce with Python >=3.10 from the repository root
# (NEMO_CHECKOUT is a public NeMo checkout):
# python - "$NEMO_CHECKOUT" <<'PY'
# import ast, pathlib, subprocess, sys, types
# pin = 'de242add77945a110568c7c44bdae4891451851e'
# base = 'nemo/collections/asr/inference/'
# def source(path):
#     return ast.parse(subprocess.check_output(
#         ['git', '-C', sys.argv[1], 'show', pin + ':' + base + path], text=True))
# utilities = source('utils/endpointing_utils.py')
# detector = source('streaming/endpointing/greedy/greedy_endpointing.py')
# cls = next(n for n in detector.body
#            if isinstance(n, ast.ClassDef) and n.name == 'GreedyEndpointing')
# method = next(n for n in cls.body
#               if isinstance(n, ast.FunctionDef)
#               and n.name == 'detect_eou_near_pivot')
# nodes = [n for n in utilities.body if isinstance(n, ast.FunctionDef)] + [method]
# namespace = {}
# module = ast.Module(body=nodes, type_ignores=[])
# exec(compile(module, '<public-NeMo-PIN>', 'exec'), namespace)
# path = pathlib.Path('tests/models/nemotron_asr/test_endpointing.py')
# test = ast.parse(path.read_text())
# cases = next(n.value for n in test.body if isinstance(n, ast.Assign)
#              and any(isinstance(t, ast.Name) and t.id == '_PUBLIC_ENDPOINT_CASES'
#                      for t in n.targets))
# for threshold, residue, symbols, _ in ast.literal_eval(cases):
#     oracle = types.SimpleNamespace(residue_tokens_at_end=residue,
#         stop_history_eou=threshold, ms_per_timestep=80,
#         is_token_silent=lambda token: token == 0,
#         is_token_start_of_word=lambda token: token == 1)
#     window = (1, *symbols)[-(threshold + residue):]
#     detect = namespace['detect_eou_near_pivot']
#     result = detect(oracle, list(window), len(window) - 1)
#     print((threshold, residue, symbols, result[0]))
# PY
# Columns: threshold, residue, frame symbols (0 blank, 1 word, 2 nonword), EOU.
# A preceding nonblank seeds arming/output; it is included in the public window.
_PUBLIC_ENDPOINT_CASES = [
    (3, 2, (0, 0, 0, 0, 1), True),
    (3, 2, (0, 0, 0, 0, 2), True),
    (3, 2, (0, 0, 0, 0, 0), True),
    (10, 2, (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1), True),
    (10, 2, (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), True),
    (3, 2, (0, 0, 0, 1), False),
    (3, 2, (0, 0, 0, 0), False),
    (3, 0, (0, 0, 0, 0, 1), False),
    (3, 0, (0, 0, 0, 0, 0), False),
    (3, 1, (0, 0, 0, 0, 1), False),
    (3, 1, (0, 0, 0, 0, 0), False),
    (1, 0, (0,), False),
    (1, 1, (0, 1), False),
    (1, 2, (0, 0, 1), True),
    (1, 3, (2,), False),
    (1, 3, (0, 0, 2), True),
    (3, 2, (0, 2, 0, 0, 1), False),
    (3, 2, (0, 1, 0, 0, 2), False),
]


_ENDPOINT_EXECUTIONS = [
    pytest.param("scalar", id="scalar"),
    pytest.param("cpu", id="tensor-cpu"),
    pytest.param(
        "cuda",
        id="tensor-cuda",
        marks=[
            pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires real CUDA"
            )
        ],
    ),
]


def _endpoint_tensors(book: Any, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    history = torch.tensor(
        [[int(value) for value in book.history]], dtype=torch.int32, device=device
    )
    fields = [
        book.history_length,
        book.history_head,
        int(book.endpoint_armed),
        book.segment_generation,
        int(book.segment_has_output),
        book.pending_forced_generation,
    ]
    return history, torch.tensor([fields], dtype=torch.int32, device=device)


def _tensor_book(history: torch.Tensor, book: torch.Tensor) -> Any:
    module = _module()
    values = book[0].tolist()
    return module.EndpointBook(
        history=tuple(module.FrameSymbol(value) for value in history[0].tolist()),
        history_length=values[0],
        history_head=values[1],
        endpoint_armed=bool(values[2]),
        segment_generation=values[3],
        segment_has_output=bool(values[4]),
        pending_forced_generation=values[5],
    )


def _endpoint_step(
    execution: str, book: Any, frames: list[list[int]], policy: Any
) -> tuple[Any, bool, tuple[int, ...]]:
    module = _module()
    if execution == "scalar":
        result = _observe(book, frames, policy=policy)
        labels = module.compose_chunk_output(
            labels=result.emitted_labels,
            is_eou=result.is_eou,
            eou_token_id=101,
            park_token_id=103,
        )
        return result.next_book, result.is_eou, tuple(labels)
    history, fields = _endpoint_tensors(book, execution)
    labels = [label for frame in frames for label in frame]
    result = module.observe_chunk_tensors(
        history=history,
        book=fields,
        frame_emission_counts=torch.tensor(
            [[len(frame) for frame in frames]], dtype=torch.int32, device=execution
        ),
        valid_frame_lengths=torch.tensor(
            [len(frames)], dtype=torch.int64, device=execution
        ),
        token_ids=torch.tensor([labels], dtype=torch.int32, device=execution),
        token_lengths=torch.tensor([len(labels)], dtype=torch.int32, device=execution),
        final_tail=torch.tensor([False], device=execution),
        mode=torch.tensor([1], dtype=torch.int64, device=execution),
        threshold_frames=torch.tensor(
            [policy.threshold_frames], dtype=torch.int64, device=execution
        ),
        residue_frames=torch.tensor(
            [policy.residue_frames], dtype=torch.int64, device=execution
        ),
        eou_token_id=101,
        row_clean=torch.tensor([True], device=execution),
    )
    assert not result.overflow.item()
    assert torch.equal(history, _endpoint_tensors(book, execution)[0]), (
        "observer mutated input history"
    )
    assert torch.equal(fields, _endpoint_tensors(book, execution)[1]), (
        "observer mutated input book"
    )
    output = tuple(result.token_ids[0, : result.token_lengths.item()].tolist()) + (103,)
    return _tensor_book(result.history, result.book), bool(result.is_eou.item()), output


@pytest.mark.parametrize("execution", _ENDPOINT_EXECUTIONS)
@pytest.mark.parametrize("threshold,residue,symbols,expected", _PUBLIC_ENDPOINT_CASES)
def test_endpoint_matches_pinned_public_boolean(
    execution, threshold, residue, symbols, expected
):
    # @spec PORT-SEG-002, PORT-SEG-003
    capacity = max(8, threshold + residue)
    policy = _policy(
        stop_history_ms=80 * threshold, residue_frames=residue, capacity=capacity
    )
    frames = [[10]] + [
        ([] if value == 0 else [20 if value == 1 else 11]) for value in symbols
    ]
    book, detected, output = _endpoint_step(execution, _book(capacity), frames, policy)
    assert detected is expected, "Boolean differs from actual pinned public NeMo golden"
    labels = tuple(label for frame in frames for label in frame)
    assert output == labels + ((101,) if expected else ()) + (103,)
    assert book.segment_generation == int(expected)
    if expected:
        assert not book.endpoint_armed and not book.segment_has_output


@pytest.mark.parametrize("execution", _ENDPOINT_EXECUTIONS)
def test_model_eou_physically_resets_history(execution):
    # @spec PORT-SEG-002, PORT-SEG-003, PORT-SEG-007
    policy = _policy(capacity=8)
    book, detected, output = _endpoint_step(
        execution, _book(8), [[10], [], [], [], [], []], policy
    )
    assert detected and output == (10, 101, 103)
    assert (book.history_length, book.history_head) == (0, 0), (
        "EOU retained valid old history"
    )
    assert all(int(symbol) == 0 for symbol in book.history), (
        "model EOU did not clear backing history"
    )
    assert (book.segment_generation, book.endpoint_armed, book.segment_has_output) == (
        1,
        False,
        False,
    )


@pytest.mark.parametrize("execution", _ENDPOINT_EXECUTIONS)
def test_model_reset_separates_old_silence_from_new_silence(execution):
    # @spec PORT-SEG-002, PORT-SEG-003, PORT-SEG-007
    policy = _policy(capacity=8)
    reset, detected, _ = _endpoint_step(
        execution, _book(8), [[10], [], [], [], [], []], policy
    )
    assert detected
    immediate, detected, output = _endpoint_step(execution, reset, [[20]], policy)
    assert not detected and output == (20, 103), "new label reused pre-EOU silence"
    assert (
        immediate.segment_generation == 1
        and immediate.endpoint_armed
        and immediate.segment_has_output
    )
    quiet, detected, _ = _endpoint_step(execution, reset, [[], [], [], [], []], policy)
    assert not detected and quiet.segment_generation == 1
    closed, detected, output = _endpoint_step(execution, quiet, [[20]], policy)
    assert detected and output == (20, 101, 103), (
        "new silence plus nonblank pivot must close"
    )
    assert closed.segment_generation == 2 and not closed.endpoint_armed


def _force_step(execution: str, book: Any) -> tuple[Any, bool]:
    module = _module()
    if execution == "scalar":
        result = module.apply_forced_eou(
            book,
            request_generation=7,
            expected_request_generation=7,
            eou_token_id=101,
            park_token_id=103,
        )
        assert result is not None and result.output_ids == (101, 103)
        assert not result.empty_segment and not result.coalesced
        return result.next_book, True
    history, fields = _endpoint_tensors(book, execution)
    result = module.apply_forced_eou_tensors(
        book=fields, selected_rows=torch.tensor([True], device=execution)
    )
    assert torch.equal(fields, _endpoint_tensors(book, execution)[1]), (
        "force mutated input book"
    )
    return _tensor_book(history, result.book), bool(result.is_eou.item())


@pytest.mark.parametrize("execution", _ENDPOINT_EXECUTIONS)
def test_successful_force_logically_resets_history(execution):
    # @spec PORT-SEG-004, PORT-SEG-007
    policy = _policy(capacity=8)
    old, detected, _ = _endpoint_step(execution, _book(8), [[10], [], [], []], policy)
    assert not detected
    reset, emitted = _force_step(execution, old)
    assert emitted
    assert (reset.history_length, reset.history_head) == (0, 0), (
        "force retained valid old history"
    )
    assert reset.history == old.history, (
        "forced logical reset must retain backing slots"
    )
    assert (
        reset.segment_generation,
        reset.endpoint_armed,
        reset.segment_has_output,
    ) == (1, False, False)
    assert reset.pending_forced_generation == 0


@pytest.mark.parametrize("execution", _ENDPOINT_EXECUTIONS)
def test_successful_force_blocks_old_silence_but_allows_new_silence(execution):
    # @spec PORT-SEG-002, PORT-SEG-004, PORT-SEG-007
    policy = _policy(capacity=8)
    old, detected, _ = _endpoint_step(execution, _book(8), [[10], [], [], []], policy)
    assert not detected
    reset, emitted = _force_step(execution, old)
    assert emitted
    resumed, detected, output = _endpoint_step(execution, reset, [[], [20]], policy)
    assert not detected and output == (20, 103), "force boundary leaked old silence"
    assert resumed.history_length == 2 and resumed.segment_generation == 1
    quiet, detected, _ = _endpoint_step(execution, reset, [[], [], [], [], []], policy)
    assert not detected
    closed, detected, output = _endpoint_step(execution, quiet, [[20]], policy)
    assert detected and output == (20, 101, 103)
    assert closed.segment_generation == 2


@pytest.mark.parametrize("execution", _ENDPOINT_EXECUTIONS)
def test_model_wins_queued_force_preserves_new_history(execution):
    # @spec PORT-SEG-002, PORT-SEG-004, PORT-SEG-007
    module = _module()
    policy = _policy(capacity=8)
    queued = module.queue_forced_eou(_book(8), request_generation=7)
    reset, detected, _ = _endpoint_step(
        execution, queued, [[10], [], [], [], [], []], policy
    )
    assert detected
    quiet, detected, _ = _endpoint_step(execution, reset, [[], []], policy)
    assert not detected and quiet.history_length == 2
    if execution == "scalar":
        result = module.apply_forced_eou(
            quiet,
            request_generation=7,
            expected_request_generation=7,
            eou_token_id=101,
            park_token_id=103,
        )
        assert result is not None and result.coalesced and result.output_ids == (103,)
        after = result.next_book
    else:
        history, fields = _endpoint_tensors(quiet, execution)
        result = module.apply_forced_eou_tensors(
            book=fields, selected_rows=torch.tensor([True], device=execution)
        )
        assert result.is_eou.tolist() == [False]
        after = _tensor_book(history, result.book)
    assert after == replace(quiet, pending_forced_generation=0), (
        "coalesced force erased new history"
    )


@pytest.mark.parametrize("execution", _ENDPOINT_EXECUTIONS)
@pytest.mark.parametrize("last_frame,expected", [([20], False), ([], True)])
def test_zero_threshold_legacy_probe_is_preserved(execution, last_frame, expected):
    # @spec PORT-SEG-002
    # Deliberately outside enabled admission; pin the existing helper behavior.
    policy = replace(
        _policy(capacity=8),
        threshold_frames=0,
        residue_frames=2,
        active_window_frames=2,
    )
    book = replace(
        _book(8),
        history_length=2,
        history_head=2,
        endpoint_armed=True,
        segment_has_output=True,
    )
    _, detected, _ = _endpoint_step(execution, book, [last_frame], policy)
    assert detected is expected


@pytest.mark.parametrize("execution", _ENDPOINT_EXECUTIONS)
@pytest.mark.parametrize(
    "capacity,threshold,residue", [(1, 1, 0), (2, 1, 1), (8, 3, 2), (12, 10, 2)]
)
def test_empty_observation_preserves_admitted_book(
    execution, capacity, threshold, residue
):
    # @spec PORT-SEG-001, PORT-SEG-002
    policy = _policy(
        capacity=capacity, stop_history_ms=threshold * 80, residue_frames=residue
    )
    original = _book(capacity)
    after, detected, output = _endpoint_step(execution, original, [], policy)
    assert after == original and not detected and output == (103,)


@pytest.mark.parametrize("device", _ENDPOINT_EXECUTIONS[1:])
def test_tensor_force_mixed_rows_preserves_noop_metadata(device):
    # @spec PORT-SEG-004, PORT-SEG-007, PORT-STATE-008
    book = torch.tensor(
        [
            [4, 4, 1, 2, 1, 9],
            [5, 5, 1, 3, 0, 9],
            [6, 6, 1, 4, 0, 9],
            [7, 7, 1, 5, 1, 9],
        ],
        dtype=torch.int32,
        device=device,
    )
    before = book.clone()
    selected = torch.tensor([True, True, False, False], device=device)
    result = _module().apply_forced_eou_tensors(book=book, selected_rows=selected)
    assert result.is_eou.tolist() == [True, False, False, False]
    expected = before.clone()
    expected[0] = torch.tensor([0, 0, 0, 3, 0, 0], device=device)
    # Existing selected-empty behavior clears armed/pending, but NOT history.
    expected[1] = torch.tensor([5, 5, 0, 3, 0, 0], device=device)
    assert torch.equal(result.book, expected)
    assert torch.equal(book, before)
    result.book.zero_()
    assert torch.equal(book, before), "force output aliases its input"


def test_scalar_malformed_empty_force_retains_existing_armed_bit():
    # @spec PORT-SEG-004, PORT-SEG-007
    # Scalar and tensor legacy behavior differs outside armed/has_output invariant.
    book = replace(
        _book(8),
        history_length=5,
        history_head=5,
        endpoint_armed=True,
        segment_has_output=False,
        pending_forced_generation=9,
    )
    result = _module().apply_forced_eou(
        book,
        request_generation=7,
        expected_request_generation=7,
        eou_token_id=101,
        park_token_id=103,
    )
    assert result is not None and result.empty_segment and not result.coalesced
    assert result.next_book == replace(book, pending_forced_generation=0)


def _mixed_tensor_observation(device):
    history = torch.zeros(5, 8, dtype=torch.int32, device=device)
    history[:, 0] = 2
    book = torch.tensor([[4, 4, 1, 3, 1, 7]] * 5, dtype=torch.int32, device=device)
    book[3, 2] = book[3, 4] = 0
    inputs = dict(
        history=history,
        book=book,
        frame_emission_counts=torch.tensor(
            [[0, 1]] * 5, dtype=torch.int32, device=device
        ),
        valid_frame_lengths=torch.tensor([2] * 5, dtype=torch.int64, device=device),
        token_ids=torch.tensor([[20]] * 5, dtype=torch.int32, device=device),
        token_lengths=torch.ones(5, dtype=torch.int32, device=device),
        final_tail=torch.tensor([False, False, True, False, False], device=device),
        mode=torch.tensor([1, 1, 1, 0, 1], dtype=torch.int64, device=device),
        threshold_frames=torch.tensor(
            [3, 3, 3, 0, -1], dtype=torch.int64, device=device
        ),
        residue_frames=torch.tensor([2, 2, 2, 0, 2], dtype=torch.int64, device=device),
        eou_token_id=101,
        row_clean=torch.tensor([True, False, True, True, True], device=device),
    )
    return inputs


@pytest.mark.parametrize("device", _ENDPOINT_EXECUTIONS[1:])
def test_tensor_endpoint_mixed_eligibility_and_all_outputs(device):
    # @spec PORT-SEG-002, PORT-SEG-003, PORT-SEG-007, PORT-STATE-008
    inputs = _mixed_tensor_observation(device)
    history_before, book_before = inputs["history"].clone(), inputs["book"].clone()
    result = _module().observe_chunk_tensors(**inputs)
    assert result.is_eou.tolist() == [True, False, False, False, False]
    assert result.overflow.tolist() == [False, False, False, False, True]
    assert result.token_ids.tolist() == [[20, 101], [20, 0], [20, 0], [20, 0], [20, 0]]
    assert result.token_lengths.tolist() == [2, 1, 1, 1, 1]
    expected_history = history_before.clone()
    expected_history[0].zero_()
    assert torch.equal(result.history, expected_history)
    expected_book = book_before.clone()
    expected_book[0] = torch.tensor([0, 0, 0, 4, 0, 7], device=device)
    expected_book[3] = torch.tensor([4, 4, 1, 3, 1, 7], device=device)
    assert torch.equal(result.book, expected_book)
    assert torch.equal(inputs["history"], history_before) and torch.equal(
        inputs["book"], book_before
    )
    # Dirty-row candidate labels are scratch; the existing transaction owns
    # zero publication and no resident scatter, not this standalone helper.


@pytest.mark.parametrize("device", _ENDPOINT_EXECUTIONS[1:])
def test_tensor_endpoint_retained_outputs_have_independent_storage(device):
    # @spec PORT-SEG-007, PORT-STATE-008
    inputs = _mixed_tensor_observation(device)
    before = {
        name: value.clone()
        for name, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }
    first = _module().observe_chunk_tensors(**inputs)
    fields = ("history", "book", "token_ids", "token_lengths", "is_eou", "overflow")
    retained = {name: getattr(first, name).clone() for name in fields}
    second = _module().observe_chunk_tensors(**inputs)
    for name in fields:
        getattr(second, name).fill_(1)
    for name in fields:
        assert torch.equal(getattr(first, name), retained[name]), (
            f"later result aliases retained {name}"
        )
        getattr(first, name).fill_(1)
    for name, value in before.items():
        assert torch.equal(inputs[name], value), f"output mutation changed input {name}"
