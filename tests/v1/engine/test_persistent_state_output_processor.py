# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent state in the output processor, detokenizer and output collector.

A state request's history is replaced, not appended, on each update; a state
stream ends only on a core output marked terminal; and the output processor sets
no bound of its own on pending inputs or undelivered turns.
"""

from collections import deque

import numpy as np
import pytest
from tokenizers import (
    AddedToken,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
)
from transformers import TokenizersBackend

from vllm.outputs import STREAM_FINISHED
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tokenizers import get_tokenizer
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine import detokenizer as detokenizer_module
from vllm.v1.engine.detokenizer import (
    INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET,
    FastIncrementalDetokenizer,
    IncrementalDetokenizer,
    SlowIncrementalDetokenizer,
)
from vllm.v1.engine.output_processor import (
    OutputProcessor,
    StreamingInputOutputCollector,
)
from vllm.v1.outputs import LogprobsLists

MAX_PENDING = 8
"""The default `max_pending_stream_inputs`."""


def _processor(tokenizer=None, persistent_state: bool = True) -> OutputProcessor:
    return OutputProcessor(
        tokenizer, log_stats=False, persistent_state=persistent_state
    )


def _input(
    request_id: str,
    prompt_token_ids: list[int],
    *,
    resumable: bool = True,
    **sampling_kwargs,
) -> EngineCoreRequest:
    sampling_kwargs.setdefault("detokenize", False)
    return EngineCoreRequest(
        request_id=request_id,
        external_req_id=f"{request_id}-ext",
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=SamplingParams(
            output_kind=RequestOutputKind.DELTA, **sampling_kwargs
        ),
        pooling_params=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        resumable=resumable,
    )


def _open(
    processor: OutputProcessor,
    request_id: str = "s",
    prompt_token_ids: list[int] | None = None,
    prompt: str | None = None,
    **sampling_kwargs,
) -> StreamingInputOutputCollector:
    queue = StreamingInputOutputCollector(RequestOutputKind.DELTA, request_id)
    request = _input(request_id, prompt_token_ids or [1, 2, 3], **sampling_kwargs)
    processor.add_request(request, prompt, queue=queue)
    return queue


def _queue_input(processor: OutputProcessor, token_ids: list[int], prompt=None):
    processor.add_request(_input("s", token_ids), prompt)


def _turn_end(
    processor: OutputProcessor,
    token: int | None = 4,
    finish_reason: FinishReason = FinishReason.STOP,
    **fields,
):
    return processor.process_outputs(
        [
            EngineCoreOutput(
                request_id="s",
                new_token_ids=[] if token is None else [token],
                finish_reason=finish_reason,
                **fields,
            )
        ]
    )


# @spec PSTATE-TURN-016
def test_terminal_finish_ends_the_stream_instead_of_starting_the_next_turn():
    processor = _processor()
    queue = _open(processor)
    _queue_input(processor, [7, 8])

    _turn_end(processor, terminal=True)

    assert processor.get_num_unfinished_requests() == 0
    output = queue.get_nowait()
    assert output is not None and output.finished is True


def _close(processor: OutputProcessor) -> None:
    processor.add_request(_input("s", [0], resumable=False), None)


def _admit(processor: OutputProcessor) -> StreamingInputOutputCollector:
    """An admitted stream: its opening request carries no prompt."""
    queue = StreamingInputOutputCollector(RequestOutputKind.DELTA, "s")
    processor.add_request(_input("s", []), None, queue=queue)
    return queue


# @spec PSTATE-TURN-016 PSTATE-TURN-015
@pytest.mark.parametrize("opening", ["parked", "admitted"])
def test_close_of_a_parked_stream_waits_for_the_terminal_output(opening):
    """The output processor synthesizes no completion for a state stream; the
    core's terminal output ends it."""
    processor = _processor()
    if opening == "admitted":
        queue = _admit(processor)
    else:
        queue = _open(processor)
        _turn_end(processor)
        assert queue.get_nowait().finished is False

    _close(processor)

    assert processor.has_request("s")
    assert queue.get_nowait() is None

    _turn_end(processor, token=None, terminal=True)

    assert not processor.has_request("s")
    output = queue.get_nowait()
    assert output is not STREAM_FINISHED and output.finished is True


# @spec PSTATE-TURN-016
def test_eos_boundary_in_flight_when_the_close_is_sent_stays_a_boundary():
    """The close can reach the frontend before core applies it: the turn end
    that core marks resumable keeps the stream, and core's terminal output
    after it ends the stream."""
    processor = _processor()
    queue = _open(processor)
    _close(processor)

    _turn_end(processor, terminal=False)

    assert processor.has_request("s")
    assert queue.get_nowait().finished is False

    _turn_end(processor, token=None, terminal=True)

    assert not processor.has_request("s")
    assert queue.get_nowait().finished is True


# @spec PSTATE-DECL-013
def test_ordinary_stream_closed_while_parked_gets_a_synthesized_completion():
    processor = _processor(persistent_state=False)
    queue = _open(processor)
    _turn_end(processor)
    assert queue.get_nowait().finished is False

    _close(processor)

    assert not processor.has_request("s")
    assert queue.get_nowait() is STREAM_FINISHED


# @spec PSTATE-ADMIT-002
def test_admitted_streams_first_input_becomes_its_first_turn():
    processor = _processor()
    _admit(processor)

    processor.add_request(_input("s", [7, 8]), "ab")

    state = processor.request_states["s"]
    assert (state.prompt, state.prompt_token_ids, state.prompt_len) == ("ab", [7, 8], 2)
    assert state.streaming_input and state.input_chunk_queue == deque()


# @spec PSTATE-HIST-005
def test_update_replaces_the_prompt_text_and_ids():
    processor = _processor()
    _open(processor, prompt_token_ids=[1, 2, 3], prompt="abc")
    _queue_input(processor, [7, 8], prompt="de")

    _turn_end(processor)

    state = processor.request_states["s"]
    assert state.prompt == "de"
    assert state.prompt_token_ids == [7, 8]
    assert state.prompt_len == 2


_TURN_TEXT = (
    "The quick brown fox jumps over the lazy dog, and then it runs into the "
    "forest where the trees are tall and the river is cold and clear."
)


def _turn_texts(tokenizer, persistent_state: bool) -> tuple[list[str], list[int]]:
    """Stream six turns, each input sent once the previous turn ends, and
    return each turn's text and the tokens the detokenizer retains at the
    turn's end, before the next input starts the next turn."""
    processor = _processor(tokenizer, persistent_state=persistent_state)
    queue = StreamingInputOutputCollector(RequestOutputKind.DELTA, "s")
    prompts = [
        tokenizer.encode(text, add_special_tokens=False)
        for text in ("Hello there.", "Next part:", "And then")
    ]
    generated = tokenizer.encode(_TURN_TEXT, add_special_tokens=False) * 2
    assert len(generated) >= 48
    turns = [generated[i : i + 8] for i in range(0, 48, 8)]
    processor.add_request(_input("s", prompts[0], detokenize=True), None, queue=queue)
    texts, retained = [], []
    for i, turn in enumerate(turns):
        processor.process_outputs(
            [EngineCoreOutput("s", turn, finish_reason=FinishReason.STOP)]
        )
        output = queue.get_nowait()
        assert output is not None
        texts.append(output.outputs[0].text)
        retained.append(len(processor.request_states["s"].detokenizer.token_ids))
        if i + 1 < len(turns):
            next_prompt = prompts[(i + 1) % len(prompts)]
            processor.add_request(_input("s", next_prompt, detokenize=True), None)
    return texts, retained


# @spec PSTATE-HIST-005
def test_detokenizer_tail_decodes_each_turn_as_full_history_does():
    tokenizer = get_tokenizer("facebook/opt-125m")
    expected, _ = _turn_texts(tokenizer, persistent_state=False)

    texts, retained = _turn_texts(tokenizer, persistent_state=True)

    assert texts == expected
    assert all(text for text in texts)
    assert max(retained[2:]) <= max(retained[:2])


SENTENCEPIECE_TOKENIZER = "hf-internal-testing/llama-tokenizer"
"""Its Metaspace decoder drops the space before a text's first piece, so a turn
decoded from empty context loses its first word-start space; byte fallback
splits an emoji or an accented character across tokens."""

_SPEECH = "He tried to think how it could be, naïve café 😀 ok, and then some more."


@pytest.fixture(scope="module")
def sentencepiece_tokenizer():
    return get_tokenizer(SENTENCEPIECE_TOKENIZER)


@pytest.fixture(params=["fast", "slow"])
def detokenizer_kind(request, monkeypatch) -> str:
    """Run the test on each incremental detokenizer."""
    if request.param == "slow":
        monkeypatch.setattr(
            detokenizer_module, "uses_fast_detokenizer", lambda tokenizer: False
        )
    return request.param


Turn = tuple[list[int], bool]
"""A turn's output ids and whether it finished on a stop token (its last id)."""


def _turns_by_word(tokenizer, ids: list[int]) -> list[Turn]:
    """One turn per word, each ending on EOS, with EOS-only turns among them."""
    groups: list[list[int]] = []
    for token_id in ids:
        piece = tokenizer.convert_ids_to_tokens(token_id)
        if piece.startswith("\u2581") or not groups:
            groups.append([token_id])
        else:
            groups[-1].append(token_id)
    eos = tokenizer.eos_token_id
    turns = [(group + [eos], True) for group in groups]
    turns[2:2] = [([eos], True), ([eos], True)]
    return turns + [([eos], True)]


def _turns_by_token(tokenizer, ids: list[int]) -> list[Turn]:
    """One turn per token, so boundaries fall inside the emoji's bytes."""
    return [([token_id, tokenizer.eos_token_id], True) for token_id in ids]


def _turns_without_stop(tokenizer, ids: list[int]) -> list[Turn]:
    """Turns of three ids that end on length, not a stop token, so a boundary
    leaves bytes pending."""
    turns = [(ids[i : i + 3], False) for i in range(0, len(ids), 3)]
    return turns + [([tokenizer.eos_token_id], True)]


def _turns_with_adjacent_specials(tokenizer, ids: list[int]) -> list[Turn]:
    """Turns that end and begin with special tokens that are not stops."""
    unk, bos = tokenizer.unk_token_id, tokenizer.bos_token_id
    turns: list[Turn] = []
    for i in range(0, len(ids), 4):
        turns.append(([unk, bos, *ids[i : i + 4], unk], False))
        turns.append(([unk, tokenizer.eos_token_id], True))
    return turns


_TURN_BUILDERS = [
    _turns_by_word,
    _turns_by_token,
    _turns_without_stop,
    _turns_with_adjacent_specials,
]


def _stream(
    tokenizer,
    turns: list[Turn],
    opener: list[int],
    prompts: list[list[int]],
    first_input: list[int] | None = None,
    after_turn=None,
    **sampling_kwargs,
) -> tuple[str, list[str]]:
    """Run the turns through a state stream, each input sent once the previous
    turn ends, and return the joined deltas and each turn's delta. With
    `first_input`, the stream opens as an admitted one does, with the `opener`
    prompt (empty), and the first turn's input follows."""
    processor = _processor(tokenizer)
    queue = StreamingInputOutputCollector(RequestOutputKind.DELTA, "s")
    processor.add_request(
        _input("s", opener, detokenize=True, **sampling_kwargs), None, queue=queue
    )
    if first_input is not None:
        processor.add_request(
            _input("s", first_input, detokenize=True, **sampling_kwargs), None
        )
    deltas = []
    for i, (ids, stop) in enumerate(turns):
        processor.process_outputs(
            [
                EngineCoreOutput(
                    "s",
                    ids,
                    finish_reason=FinishReason.STOP if stop else FinishReason.LENGTH,
                )
            ]
        )
        output = queue.get_nowait()
        assert output is not None
        deltas.append(output.outputs[0].text)
        if after_turn is not None:
            after_turn(i, processor)
        if i + 1 < len(turns):
            next_prompt = prompts[i % len(prompts)]
            processor.add_request(
                _input("s", next_prompt, detokenize=True, **sampling_kwargs), None
            )
    return "".join(deltas), deltas


def _reference(
    tokenizer, turns: list[Turn], opener: list[int], **sampling_kwargs
) -> tuple[str, list[int]]:
    """The PSTATE-HIST-010 reference: one request with the opener prompt that
    decodes the turns' output ids in order, without each stop token, with the
    same detokenizer. Returns its text and the ids it decoded."""
    request = _input("ref", opener, detokenize=True, **sampling_kwargs)
    detokenizer = IncrementalDetokenizer.from_new_request(tokenizer, request)
    decoded = [ids[:-1] if stop else ids for ids, stop in turns]
    for ids in decoded:
        detokenizer.update(ids, stop_terminated=False)
    text = detokenizer.get_next_output_text(finished=True, delta=False)
    return text, [token_id for ids in decoded for token_id in ids]


# The turns' prompts hold only special tokens: BOS stands for an audio chunk
# carrier that decoding skips; the last prompt is empty.
_SPECIAL_ONLY_PROMPTS = [[1], [1, 1], [2, 1]]


# @spec PSTATE-HIST-005 PSTATE-HIST-010
@pytest.mark.parametrize("make_turns", _TURN_BUILDERS)
@pytest.mark.parametrize(
    "sampling_kwargs",
    [
        {"skip_special_tokens": True},
        {"skip_special_tokens": False, "spaces_between_special_tokens": False},
    ],
    ids=["skip_special", "keep_special_no_spaces"],
)
def test_stream_text_equals_one_request_decoding_the_turns_output(
    sentencepiece_tokenizer, detokenizer_kind, make_turns, sampling_kwargs
):
    tokenizer = sentencepiece_tokenizer
    ids = tokenizer.encode(_SPEECH, add_special_tokens=False)
    turns = make_turns(tokenizer, ids)
    opener = [tokenizer.bos_token_id]

    text, deltas = _stream(
        tokenizer, turns, opener, _SPECIAL_ONLY_PROMPTS, **sampling_kwargs
    )

    expected, decoded_ids = _reference(tokenizer, turns, opener, **sampling_kwargs)
    assert text == expected
    assert text.strip()
    if sampling_kwargs["skip_special_tokens"]:
        assert text == tokenizer.decode(decoded_ids, skip_special_tokens=True)
        if make_turns is _turns_by_token:
            # Every piece keeps its word-start space, including each turn's first.
            assert text == _SPEECH
            assert deltas[1].startswith(" ")


# @spec PSTATE-HIST-005 PSTATE-HIST-010
def test_turn_that_ends_inside_a_character_completes_it_in_the_next(
    sentencepiece_tokenizer, detokenizer_kind
):
    tokenizer = sentencepiece_tokenizer
    word = tokenizer.encode("ok", add_special_tokens=False)
    emoji = tokenizer.encode(" \U0001f600", add_special_tokens=False)
    assert len(emoji) == 5  # a space piece, then the emoji's four byte tokens
    eos = tokenizer.eos_token_id
    turns: list[Turn] = [
        (word + [eos], True),
        (emoji[:3], False),  # ends after the emoji's first two bytes
        ([eos], True),
        (emoji[3:], False),
        ([eos], True),
    ]

    text, deltas = _stream(
        tokenizer, turns, [tokenizer.bos_token_id], _SPECIAL_ONLY_PROMPTS
    )

    assert text == "ok \U0001f600"
    assert deltas[2] == ""
    assert deltas[3] == "\U0001f600"


# @spec PSTATE-HIST-005 PSTATE-HIST-010
@pytest.mark.parametrize("num_prompt_ids", [3, 10], ids=["after_word", "mid_word"])
def test_admitted_streams_first_input_seeds_the_detokenizer(
    sentencepiece_tokenizer, detokenizer_kind, num_prompt_ids
):
    """An admitted stream opens with an empty prompt; its first input seeds the
    decoding state once, and later special-only prompts never do."""
    tokenizer = sentencepiece_tokenizer
    ids = tokenizer.encode(_SPEECH, add_special_tokens=False)
    first_prompt, rest = ids[:num_prompt_ids], ids[num_prompt_ids:]
    turns = _turns_by_word(tokenizer, rest)

    text, deltas = _stream(
        tokenizer, turns, [], _SPECIAL_ONLY_PROMPTS, first_input=first_prompt
    )

    expected, _ = _reference(tokenizer, turns, first_prompt)
    assert text == expected
    # The first turn decodes as one request with that prompt does.
    assert deltas[0] == _reference(tokenizer, turns[:1], first_prompt)[0]
    assert deltas[0].startswith(" ") == (num_prompt_ids == 3)
    if num_prompt_ids == 3:
        # Unseeded, the first piece would lose its word-start space.
        assert text != _reference(tokenizer, turns, [])[0]


# @spec PSTATE-HIST-005
def test_a_later_inputs_prompt_never_seeds_the_detokenizer(
    sentencepiece_tokenizer, detokenizer_kind
):
    tokenizer = sentencepiece_tokenizer
    ids = tokenizer.encode(_SPEECH, add_special_tokens=False)
    turns = _turns_by_word(tokenizer, ids)
    opener = [tokenizer.bos_token_id]
    # Text prompts that would shift a seeded decoder's context.
    text_prompts = [ids[:6], ids[3:9]]

    seeded, _ = _stream(tokenizer, turns, opener, text_prompts)
    unseeded, _ = _stream(tokenizer, turns, opener, _SPECIAL_ONLY_PROMPTS)

    assert seeded == unseeded == _reference(tokenizer, turns, opener)[0]


@pytest.fixture
def slow_window(monkeypatch) -> dict[str, int]:
    """The most tokens and ids the slow detokenizer holds at any step."""
    monkeypatch.setattr(
        detokenizer_module, "uses_fast_detokenizer", lambda tokenizer: False
    )
    window = {"tokens": 0, "token_ids": 0}
    decode_next = SlowIncrementalDetokenizer.decode_next

    def logged_decode_next(self, next_token_id):
        text = decode_next(self, next_token_id)
        window["tokens"] = max(window["tokens"], len(self.tokens))
        window["token_ids"] = max(window["token_ids"], len(self.token_ids))
        return text

    monkeypatch.setattr(SlowIncrementalDetokenizer, "decode_next", logged_decode_next)
    return window


_RUN_LENGTH = 5000
_TURN_LENGTH = 100
# The stream's text costs time quadratic in a run that returns no text.
_MALFORMED_RUN_LENGTH = 1500
_TEXT_TURNS = 1000


def _run_in_turns(ids: list[int], stop: bool = False) -> list[Turn]:
    """The ids in turns of `_TURN_LENGTH`, none ending on a stop token."""
    return [(ids[i : i + _TURN_LENGTH], stop) for i in range(0, len(ids), _TURN_LENGTH)]


def _byte_ids(tokenizer, names: list[str]) -> list[int]:
    return [tokenizer.convert_tokens_to_ids(name) for name in names]


# Byte-fallback ids that no following byte can make valid: a continuation byte
# with no lead, and bytes UTF-8 never uses.
_MALFORMED_BYTES = {
    "one_byte": ["<0x80>"],
    "mixed_bytes": ["<0x80>", "<0xBF>", "<0xC0>", "<0xF8>", "<0xFF>", "<0x9F>"],
}


# @spec PSTATE-HIST-010
@pytest.mark.parametrize("bytes_kind", sorted(_MALFORMED_BYTES))
def test_run_of_malformed_bytes_decodes_as_one_request_does(
    sentencepiece_tokenizer, detokenizer_kind, bytes_kind
):
    """The ids return no text until a later one ends the run, across turns."""
    tokenizer = sentencepiece_tokenizer
    cycle = _byte_ids(tokenizer, _MALFORMED_BYTES[bytes_kind])
    run = [cycle[i % len(cycle)] for i in range(_MALFORMED_RUN_LENGTH)]
    word = tokenizer.encode("He tried to think", add_special_tokens=False)
    opener = [tokenizer.bos_token_id]

    text, _ = _stream(
        tokenizer, _run_in_turns(run + word), opener, _SPECIAL_ONLY_PROMPTS
    )

    assert text == "\ufffd" * _MALFORMED_RUN_LENGTH + " He tried to think"
    # The reference holds the whole run, which costs the slow detokenizer
    # time quadratic in its length, so it decodes a shorter one.
    short = run[: _TURN_LENGTH * 3]
    turns = _run_in_turns(short + word)
    expected, _ = _reference(tokenizer, turns, opener)
    assert _stream(tokenizer, turns, opener, _SPECIAL_ONLY_PROMPTS)[0] == expected


# @spec PSTATE-HIST-009
def test_slow_detokenizers_window_stays_bounded_over_many_turns(
    sentencepiece_tokenizer, slow_window
):
    """The tokens and ids the slow detokenizer holds are the turn's output and
    a kept tail, however many turns have run."""
    tokenizer = sentencepiece_tokenizer
    eos, bos = tokenizer.eos_token_id, tokenizer.bos_token_id
    cycle = tokenizer.encode(_SPEECH, add_special_tokens=False)
    position = 0
    turns: list[Turn] = []
    for turn in range(_RUN_LENGTH):
        count = 1 + turn % 3
        ids = [cycle[(position + j) % len(cycle)] for j in range(count)]
        turns.append((ids + [eos], True))
        position += count

    text, _ = _stream(tokenizer, turns, [bos], _SPECIAL_ONLY_PROMPTS)

    assert text
    # At most 3 output ids and the stop id, plus the tail: the prefix window,
    # and the bytes of a character a turn ends inside.
    tail = 2 * INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET
    assert slow_window["tokens"] <= 3 + tail
    assert slow_window["token_ids"] <= 4 + tail


def _text_turns(tokenizer, count: int) -> list[Turn]:
    """Turns of 1 to 3 text ids, each ending on a stop token."""
    cycle = tokenizer.encode(_SPEECH, add_special_tokens=False)
    turns: list[Turn] = []
    for turn in range(count):
        ids = [cycle[(turn * 3 + j) % len(cycle)] for j in range(1 + turn % 3)]
        turns.append((ids + [tokenizer.eos_token_id], True))
    return turns


# @spec PSTATE-HIST-009 PSTATE-HIST-010
def test_slow_detokenizers_window_grows_in_a_malformed_byte_run_and_drains_after(
    sentencepiece_tokenizer, slow_window
):
    """The offsets stop while a run of malformed bytes leaves the text ending
    in an incomplete character, so the window grows with the run; the next
    step that returns text moves them, and the next turn trims the window."""
    tokenizer = sentencepiece_tokenizer
    eos, bos = tokenizer.eos_token_id, tokenizer.bos_token_id
    byte = tokenizer.convert_tokens_to_ids("<0x80>")
    word = tokenizer.encode("He tried", add_special_tokens=False)

    before = _text_turns(tokenizer, _TEXT_TURNS)
    after = _text_turns(tokenizer, _TEXT_TURNS)
    run = _run_in_turns([byte] * _MALFORMED_RUN_LENGTH)
    ending = [(word + [eos], True)]
    turns = before + run + ending + after
    first_run, last_run = len(before), len(before) + len(run) - 1
    seen: dict[str, int] = {}

    def after_turn(i: int, processor: OutputProcessor) -> None:
        detokenizer = processor.request_states["s"].detokenizer
        if i == first_run - 1:
            seen["ordinary"] = slow_window["tokens"]
        elif i == last_run:
            seen["in_run"] = slow_window["tokens"]
            seen["held_in_run"] = len(detokenizer.tokens)
        elif i == last_run + 1:
            # The step that ends the run returns its text; the next turn
            # trims the window to a tail.
            seen["after_ending_turn"] = len(detokenizer.tokens)
            slow_window["tokens"] = 0
        elif i == len(turns) - 1:
            seen["later"] = slow_window["tokens"]

    opener = [bos]
    text, _ = _stream(
        tokenizer, turns, opener, _SPECIAL_ONLY_PROMPTS, after_turn=after_turn
    )

    tail = 2 * INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET
    ordinary_bound = 3 + tail
    assert seen["ordinary"] <= ordinary_bound
    assert seen["in_run"] >= _MALFORMED_RUN_LENGTH
    assert seen["held_in_run"] >= _MALFORMED_RUN_LENGTH
    # The turn that ends the run still holds it; the window is back to the
    # ordinary bound once the next turn starts and over the turns after it.
    assert seen["after_ending_turn"] >= _MALFORMED_RUN_LENGTH
    assert seen["later"] <= ordinary_bound
    assert text == _reference(tokenizer, turns, opener)[0]


# @spec PSTATE-HIST-009 PSTATE-HIST-010
def test_slow_detokenizers_window_after_a_one_token_recovery_turn(
    sentencepiece_tokenizer, slow_window
):
    """The first step that returns text after a malformed byte run moves the
    prefix offset only to the old read offset, so the turn trim after a single
    text token still holds the run; the trim after a second text step does
    not."""
    tokenizer = sentencepiece_tokenizer
    eos, bos = tokenizer.eos_token_id, tokenizer.bos_token_id
    byte = tokenizer.convert_tokens_to_ids("<0x80>")
    pieces = tokenizer.encode("He tried to think", add_special_tokens=False)
    before = _text_turns(tokenizer, 20)
    run = _run_in_turns([byte] * _TURN_LENGTH * 3)
    recovery = [([piece, eos], True) for piece in pieces]
    turns = before + run + recovery + _text_turns(tokenizer, 20)
    first = len(before) + len(run)
    held: dict[int, int] = {}

    def after_turn(i: int, processor: OutputProcessor) -> None:
        held[i] = len(processor.request_states["s"].detokenizer.tokens)

    opener = [bos]
    text, _ = _stream(
        tokenizer, turns, opener, _SPECIAL_ONLY_PROMPTS, after_turn=after_turn
    )

    run_length = _TURN_LENGTH * 3
    ordinary_bound = 3 + 2 * INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET
    # The single-token turn, and the turn after it, still hold the run.
    assert held[first] >= run_length
    assert held[first + 1] >= run_length
    # The trim before the third text turn releases it.
    assert all(held[i] <= ordinary_bound for i in range(first + 2, len(turns)))
    assert text == _reference(tokenizer, turns, opener)[0]


# @spec PSTATE-HIST-010
@pytest.mark.parametrize("special_ids", ["eos", "bos_eos_unk"])
def test_long_runs_of_skipped_specials_across_many_turns_decode_as_one_request_does(
    sentencepiece_tokenizer, detokenizer_kind, special_ids
):
    """Each turn ends on a special token that decoding skips and the
    detokenizer decodes (the stop token is in the output ids), and a text
    follows thousands of them: its first piece keeps its word-start space."""
    tokenizer = sentencepiece_tokenizer
    specials = (
        [tokenizer.eos_token_id]
        if special_ids == "eos"
        else [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id]
    )
    first = tokenizer.encode("He", add_special_tokens=False)
    second = tokenizer.encode(" ok", add_special_tokens=False)
    turns: list[Turn] = [(first + [specials[0]], False)]
    turns += [([specials[i % len(specials)]], False) for i in range(_RUN_LENGTH)]
    turns += [(second + [specials[0]], False)]
    opener = [tokenizer.bos_token_id]

    text, _ = _stream(
        tokenizer, turns, opener, _SPECIAL_ONLY_PROMPTS, include_stop_str_in_output=True
    )

    assert text == "He ok" == tokenizer.decode(first + second, skip_special_tokens=True)


def _normalized_special_tokenizer():
    """A tokenizer whose special token "SPECIAL" is normalized, so the decoder
    sees its lowercase text "special" and, on tokenizers that filter on that
    text, decodes it though `skip_special_tokens` is set."""
    tokenizer = Tokenizer(
        models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]")
    )
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.Fuse()
    tokenizer.add_special_tokens([AddedToken("SPECIAL", special=True, normalized=True)])
    return TokenizersBackend(tokenizer_object=tokenizer)


# @spec PSTATE-HIST-010
def test_a_special_token_that_decodes_to_text_is_not_dropped(detokenizer_kind):
    tokenizer = _normalized_special_tokenizer()
    hello, world = 1, 2
    special = tokenizer.convert_tokens_to_ids("SPECIAL")
    turns: list[Turn] = [([hello], False)]
    turns += [([special] * 40, False)] * 3
    turns += [([world, special], False)]
    opener = [0]

    text, _ = _stream(tokenizer, turns, opener, [[0]])

    expected, decoded = _reference(tokenizer, turns, opener)
    assert text == expected
    if detokenizer_kind == "fast":
        # The slow detokenizer skips the tokens the transformers tokenizer
        # knows as special, which does not include this one.
        assert text == tokenizer.decode(decoded, skip_special_tokens=True)


# @spec PSTATE-HIST-005
def test_special_token_adjacency_flag_carries_across_turns(sentencepiece_tokenizer):
    tokenizer = sentencepiece_tokenizer
    processor = _processor(tokenizer)
    _open(
        processor,
        prompt_token_ids=[tokenizer.bos_token_id],
        detokenize=True,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
    )
    detokenizer = processor.request_states["s"].detokenizer
    assert isinstance(detokenizer, FastIncrementalDetokenizer)
    unk = tokenizer.unk_token_id

    processor.process_outputs(
        [EngineCoreOutput("s", [unk], finish_reason=FinishReason.LENGTH)]
    )
    assert detokenizer.last_special
    processor.add_request(_input("s", [tokenizer.bos_token_id], detokenize=True), None)

    assert detokenizer.last_special


# @spec PSTATE-HIST-005
def test_logprobs_reset_each_turn():
    processor = _processor()
    _open(processor, logprobs=1)
    _queue_input(processor, [7, 8])
    logprobs = LogprobsLists(
        logprob_token_ids=np.array([[4, 5]]),
        logprobs=np.array([[-0.5, -1.0]]),
        sampled_token_ranks=np.array([1]),
    )

    _turn_end(processor, new_logprobs=logprobs)

    processor_state = processor.request_states["s"].logprobs_processor
    assert len(processor_state.logprobs) == 0
    assert processor_state.cumulative_logprob == 0.0


# @spec PSTATE-HIST-005
def test_per_request_accumulators_reset_each_turn():
    processor = _processor()
    _open(processor)
    _queue_input(processor, [7, 8])
    state = processor.request_states["s"]
    state.routed_experts_chunks.append(np.zeros((1, 1, 1), dtype=np.int16))
    state.weight_version_starts.append(("v1", 0))

    _turn_end(processor)

    assert state.routed_experts_chunks == []
    assert state.weight_version_starts == []


def _run_turns(processor: OutputProcessor, num_turns: int) -> None:
    for token in range(num_turns):
        _turn_end(processor, token)


# @spec PSTATE-HIST-006
def test_output_processor_sets_no_bound_of_its_own():
    """`AsyncLLM` bounds pending inputs and undelivered turns by credits; past
    that bound the output processor still queues inputs and delivers turns."""
    processor = _processor()
    queue = _open(processor)
    for i in range(MAX_PENDING + 2):
        _queue_input(processor, [10 + i])
    _close(processor)
    assert len(processor.request_states["s"].input_chunk_queue) == MAX_PENDING + 2

    _run_turns(processor, MAX_PENDING + 2)

    assert processor.has_request("s")
    outputs = [queue.get_nowait() for _ in range(MAX_PENDING + 2)]
    assert [output.outputs[0].token_ids for output in outputs] == [
        [token] for token in range(MAX_PENDING + 2)
    ]


# @spec PSTATE-TURN-016 PSTATE-HIST-006
def test_terminal_output_is_delivered_behind_undelivered_turns_and_drops_inputs():
    processor = _processor()
    queue = _open(processor)
    for i in range(MAX_PENDING):
        _queue_input(processor, [10 + i])
    _run_turns(processor, MAX_PENDING)
    _queue_input(processor, [50])

    _turn_end(processor, terminal=True)

    assert not processor.has_request("s")
    outputs = [queue.get_nowait() for _ in range(MAX_PENDING + 1)]
    assert [output.finished for output in outputs] == [False] * MAX_PENDING + [True]
