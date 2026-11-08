# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-5 tests-first: checkpoint-profile manifests.

``manifests.py`` is STDLIB-ONLY (no ``torch``, no ``vllm`` package
import), so — unlike the rest of the port — this file loads it by
direct file location (``importlib.util.spec_from_file_location``),
bypassing the ``vllm`` package ``__init__`` chain (which pulls
``vllm``, uninstallable on macOS). That is what lets this file run
locally; every other nemotron_asr test file is pod-tier.

``canonical_json``/``manifest_hash`` and the canonical name/order/
arithmetic pins are real, PASSING tests. Every ``author_*`` /
``verify_state_manifest`` test below is BEHAVIORAL: it calls the
function and asserts the exact intended output (or the specific
``ValueError``). Those tests are EXPECTED TO FAIL today — the stubs
raise ``NotImplementedError`` — and turn green in Phase 6 without
being rewritten (the LID tests-first rule).

The canonical layout pins the DECIDED design inventory
(PORT-STATE-001/009, ``port-design.md``'s state-page table): total
6,314,944 bytes (~6.0223 MiB) FP32 bring-up, including the explicit
4-byte alignment descriptor required after the 141-slot replay queue.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]

_MANIFESTS_PATH = (
    Path(__file__).resolve().parents[3] / "vllm/models/nemotron_asr/manifests.py"
)


def _load_manifests() -> Any:
    """Load ``manifests.py`` by file path — no ``vllm`` import."""
    spec = importlib.util.spec_from_file_location(
        "nemotron_asr_manifests_under_test", _MANIFESTS_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


manifests = _load_manifests()


def _checkpoint_config() -> Any:
    """The published checkpoint's geometry, as the authors consume it."""
    return types.SimpleNamespace(
        n_layers=24,
        att_context_left=56,
        d_model=1024,
        conv_kernel=9,
        pred_rnn_layers=2,
        pred_hidden=640,
        n_mels=128,
        hidden_size=17_927,  # >= 7 header slots + 17,920 raw samples
        endpoint_history_capacity_frames=12,
    )


def _tiny_config() -> Any:
    """A deliberately different geometry — proves the authors derive
    from ``config`` rather than hardcoding the checkpoint."""
    return types.SimpleNamespace(
        n_layers=2,
        att_context_left=8,
        d_model=32,
        conv_kernel=5,
        pred_rnn_layers=2,
        pred_hidden=16,
        n_mels=16,
        hidden_size=17_927,
        endpoint_history_capacity_frames=5,
    )


def _expected_entries(cfg: Any) -> list[dict[str, Any]]:
    """The exact ``author_state_manifest`` entry list for ``cfg`` —
    written out independently here so the author test is behavioral,
    not a shared-helper tautology."""
    entries: list[dict[str, Any]] = []
    for i in range(cfg.n_layers):
        entries.append(
            {
                "name": f"encoder.layers.{i}.window.channel",
                "shape": [cfg.att_context_left, cfg.d_model],
                "dtype": "float32",
                "init": "zeros",
            }
        )
        entries.append(
            {
                "name": f"encoder.layers.{i}.conv.time",
                "shape": [cfg.d_model, cfg.conv_kernel - 1],
                "dtype": "float32",
                "init": "zeros",
            }
        )
        entries.append(
            {
                "name": f"encoder.layers.{i}.window.valid",
                "shape": [1],
                "dtype": "int32",
                "init": "zeros",
            }
        )
    entries.append(
        {
            "name": "frontend.raw_tail",
            "shape": [1953],
            "dtype": "float32",
            "init": "zeros",
        }
    )
    entries.append(
        {
            "name": "frontend.mel_tail",
            "shape": [cfg.n_mels, 9],
            "dtype": "float32",
            "init": "zeros",
        }
    )
    entries.append(
        {
            "name": "decode.layers.0.replay.queue",
            "shape": [141],
            "dtype": "int32",
            "init": "zeros",
        }
    )
    for name, init in (
        ("queue_head", "zeros"),
        ("queue_length", "zeros"),
        ("last_label", "blank_label"),
        ("prompt", "admitted_prompt"),
        ("geometry", "admitted_geometry"),
        ("pending_echo", "zeros"),
        ("expected_label", "zeros"),
    ):
        entries.append(
            {
                "name": f"decode.layers.0.replay.book.{name}",
                "shape": [1],
                "dtype": "int32",
                "init": init,
            }
        )
    entries.append(
        {
            "name": "padding.frontend_counters_alignment",
            "shape": [1],
            "dtype": "int32",
            "init": "zeros",
        }
    )
    for counter in (
        "total_valid_samples",
        "committed_mel_frames",
        "encoded_mel_frames",
        "raw_tail_origin",
        "raw_tail_length",
        "mel_tail_length",
        "expected_chunk_sequence",
        "finalized",
    ):
        entries.append(
            {
                "name": f"frontend.counters.{counter}",
                "shape": [1],
                "dtype": "int64",
                "init": "zeros",
            }
        )
    for tensor in ("h", "c"):
        entries.append(
            {
                "name": f"predictor.layers.0.lstm_state.{tensor}",
                "shape": [cfg.pred_rnn_layers, cfg.pred_hidden],
                "dtype": "float32",
                "init": "zeros",
            }
        )
    entries.append(
        {
            "name": "endpoint.history",
            "shape": [cfg.endpoint_history_capacity_frames],
            "dtype": "int32",
            "init": "zeros",
        }
    )
    for name in (
        "history_length",
        "history_head",
        "endpoint_armed",
        "segment_generation",
        "segment_has_output",
        "pending_forced_generation",
    ):
        entries.append(
            {
                "name": f"endpoint.book.{name}",
                "shape": [1],
                "dtype": "int32",
                "init": "zeros",
            }
        )
    return entries


_DTYPE_BYTES = {"float32": 4, "int32": 4, "int64": 8}


def _entry_bytes(entry: dict[str, Any]) -> int:
    size = _DTYPE_BYTES[entry["dtype"]]
    n = 1
    for dim in entry["shape"]:
        n *= dim
    return n * size


def _expected_state_manifest(cfg: Any) -> dict[str, Any]:
    entries = _expected_entries(cfg)
    return {
        "schema": "state-manifest-v1",
        "precision_policy": "fp32-bringup-v1",
        "entries": entries,
        "total_page_bytes": sum(_entry_bytes(e) for e in entries),
    }


def _specs_from_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The live registered-page inventory matching ``entries`` — the
    ``verify_state_manifest`` comparison target (no ``init``: the
    engine's specs carry name/shape/dtype only)."""
    return [
        {"name": e["name"], "shape": list(e["shape"]), "dtype": e["dtype"]}
        for e in entries
    ]


# ---- canonical name/order/arithmetic pins: real, PASSING ------------------


def test_book_fields_are_the_seven_decided_slots_in_order() -> None:
    # @spec PORT-STATE-001
    # First four slots must keep rnnt.py's pinned queue-book order.
    assert manifests.BOOK_FIELDS == (
        ("queue_head", "zeros"),
        ("queue_length", "zeros"),
        ("last_label", "blank_label"),
        ("prompt", "admitted_prompt"),
        ("geometry", "admitted_geometry"),
        ("pending_echo", "zeros"),
        ("expected_label", "zeros"),
    )


def test_frontend_counters_are_the_eight_decided_slots_in_order() -> None:
    # @spec PORT-STATE-001
    assert manifests.FRONTEND_COUNTER_FIELDS == (
        "total_valid_samples",
        "committed_mel_frames",
        "encoded_mel_frames",
        "raw_tail_origin",
        "raw_tail_length",
        "mel_tail_length",
        "expected_chunk_sequence",
        "finalized",
    )


def test_envelope_header_fields_are_the_seven_decided_slots_in_order() -> None:
    # @spec PORT-REGIME-001
    assert manifests.ENVELOPE_HEADER_FIELDS == (
        "version",
        "valid_samples",
        "geometry_id",
        "final_tail",
        "prompt_index",
        "chunk_sequence",
        "admission_ms_mod",
    )


def test_frontend_constants_pin_the_checkpoint_featurizer() -> None:
    # @spec PORT-STATE-009
    fc = manifests.FRONTEND_CONSTANTS
    assert fc["sample_rate"] == 16_000
    assert fc["n_fft"] == 512
    assert fc["win_length"] == 400
    assert fc["hop_length"] == 160
    assert fc["n_mels"] == 128
    assert fc["pre_encode_cache_frames"] == 9
    assert fc["subsampling_factor"] == 8
    # The raw-tail bound honors its own published formula.
    assert fc["raw_tail_capacity"] == 9 * 160 + 512 + 1 == 1953
    assert fc["final_tail_min_new_mel_frames"] == 8


def test_session_limits_are_fp32_representable() -> None:
    # @spec PORT-INT-004
    # Every envelope header integer must be exact in FP32.
    limits = manifests.SESSION_LIMITS
    assert limits["carrier_sequence_modulus"] == 2**24
    assert "max_session_chunks" not in limits
    assert limits["queue_capacity"] == (
        limits["max_symbols_per_step"] * limits["max_frames_per_chunk"] + 1
    )


def test_canonical_fp32_layout_arithmetic_is_auditable() -> None:
    # @spec PORT-STATE-009 / PORT-STATE-021 / PORT-SEG-001
    # Per-layer: channel (56*1024*4=229376) + time (1024*8*4=32768) +
    # valid (1*4=4) = 262148 bytes/layer; * 24 layers = 6,291,552.
    per_layer = 56 * 1024 * 4 + 1024 * 8 * 4 + 4
    assert per_layer == 262_148
    assert per_layer * 24 == 6_291_552
    # Frontend: 1953*4 + 128*9*4 + 8 counters * 8 bytes = 12,484.
    assert 1953 * 4 + 128 * 9 * 4 + 8 * 8 == 12_484
    # Predictor h/c: 2 * (2*640*4) = 10,240; queue 141*4 = 564;
    # 7-slot int32 book = 28; explicit int64-alignment padding = 4.
    total = 6_291_552 + 12_484 + 10_240 + 564 + 28 + 4
    # Installed endpoint capacity 12: 12 int32 ring slots + six int32
    # book fields. The pre-endpoint A8 subtotal remains independently
    # auditable and no endpoint state can hide in a side store.
    endpoint = 12 * 4 + 6 * 4
    assert total == 6_314_872
    assert endpoint == 72
    entries = _expected_entries(_checkpoint_config())
    assert sum(_entry_bytes(e) for e in entries) == 6_314_944


def test_carrier_width_covers_header_plus_largest_raw_cadence() -> None:
    # @spec PORT-REGIME-001
    needed = len(manifests.ENVELOPE_HEADER_FIELDS) + max(
        manifests.RAW_SAMPLES_PER_CHUNK.values()
    )
    assert needed == 17_927
    assert _checkpoint_config().hidden_size >= needed


# ---- canonical_json / manifest_hash: real, PASSING helpers ----------------


def test_canonical_json_is_deterministic_regardless_of_key_order() -> None:
    a = {"b": 2, "a": 1}
    b = {"a": 1, "b": 2}
    assert manifests.canonical_json(a) == manifests.canonical_json(b)


def test_canonical_json_is_compact_and_ascii() -> None:
    s = manifests.canonical_json({"x": [1, 2], "y": "café"})
    assert " " not in s  # compact separators
    assert "\\u00e9" in s  # ensure_ascii


def test_manifest_hash_shape_is_sha256_prefixed_hex() -> None:
    h = manifests.manifest_hash({"a": 1})
    assert h.startswith("sha256:")
    assert len(h) == len("sha256:") + 64
    assert all(c in "0123456789abcdef" for c in h[len("sha256:") :])


def test_manifest_hash_is_deterministic_and_content_sensitive() -> None:
    h1 = manifests.manifest_hash({"a": 1, "b": 2})
    h2 = manifests.manifest_hash({"b": 2, "a": 1})
    h3 = manifests.manifest_hash({"a": 1, "b": 3})
    assert h1 == h2
    assert h1 != h3


def test_cadences_are_the_five_published_pairs_exactly() -> None:
    assert manifests.CADENCES == {
        "80ms": (56, 0),
        "160ms": (56, 1),
        "320ms": (56, 3),
        "560ms": (56, 6),
        "1120ms": (56, 13),
    }


# ---- PORT-WGT-004: authors produce their exact manifests ------------------
# BEHAVIORAL, red until Phase 6: the stubs raise NotImplementedError.


def test_author_state_manifest_produces_the_exact_canonical_layout() -> None:
    # @spec PORT-WGT-004
    authored = manifests.author_state_manifest(_checkpoint_config())
    expected = _expected_state_manifest(_checkpoint_config())
    assert authored == expected
    assert authored["total_page_bytes"] == 6_314_944


def test_author_state_manifest_derives_from_config_not_the_checkpoint() -> None:
    # @spec PORT-WGT-004
    authored = manifests.author_state_manifest(_tiny_config())
    assert authored == _expected_state_manifest(_tiny_config())


def test_author_geometry_manifest_produces_the_exact_manifest() -> None:
    # @spec PORT-WGT-004
    authored = manifests.author_geometry_manifest(_checkpoint_config())
    assert authored == {
        "schema": "geometry-manifest-v1",
        "cadences": {
            "80ms": {
                "att_context": [56, 0],
                "frames_per_chunk": 1,
                "raw_samples_per_chunk": 1_280,
            },
            "160ms": {
                "att_context": [56, 1],
                "frames_per_chunk": 2,
                "raw_samples_per_chunk": 2_560,
            },
            "320ms": {
                "att_context": [56, 3],
                "frames_per_chunk": 4,
                "raw_samples_per_chunk": 5_120,
            },
            "560ms": {
                "att_context": [56, 6],
                "frames_per_chunk": 7,
                "raw_samples_per_chunk": 8_960,
            },
            "1120ms": {
                "att_context": [56, 13],
                "frames_per_chunk": 14,
                "raw_samples_per_chunk": 17_920,
            },
        },
        "envelope": {
            "version": 2,
            "header_fields": [
                "version",
                "valid_samples",
                "geometry_id",
                "final_tail",
                "prompt_index",
                "chunk_sequence",
                "admission_ms_mod",
            ],
        },
        "carrier_width": 17_927,
        "frontend": manifests.FRONTEND_CONSTANTS,
        "endpoint": {
            "algorithm": "cache-aware-greedy-blank-v1",
            "installed_modes": ["disabled", "greedy_blank"],
            "frame_stride_ms": 80,
            "history_capacity_frames": 12,
        },
    }


def test_author_geometry_manifest_rejects_a_narrow_carrier() -> None:
    # @spec PORT-WGT-004
    cfg = _checkpoint_config()
    cfg.hidden_size = 17_926  # one short of header + largest cadence
    with pytest.raises(ValueError, match="carrier_width|hidden_size"):
        manifests.author_geometry_manifest(cfg)


def test_author_transition_manifest_produces_the_exact_manifest() -> None:
    # @spec PORT-WGT-004
    authored = manifests.author_transition_manifest(_checkpoint_config())
    assert authored == {
        "schema": "transition-manifest-v1",
        "transition": "advance_session-v1",
        "roles": ["CHUNK", "REPLAY", "EOU", "FLUSH"],
        "chunk_roles_entering_transition": ["CHUNK"],
        "session_first_init": "metadata-books-zero-scratch",
        "encode_overlap": "pre-encode-cache-on-non-first-chunks",
        "final_tail": "actual-residual-final-stft",
        "echo_policy": "mrv1-echo-verify",
    }


def test_author_emission_manifest_produces_the_exact_manifest() -> None:
    # @spec PORT-WGT-004 / PORT-INT-002
    authored = manifests.author_emission_manifest(_checkpoint_config())
    per_geometry = {
        label: {
            "max_valid_encoder_frames": right + 1,
            "max_symbols_per_step": 10,
            "max_emission_tokens": (right + 1) * 10 + 2,
        }
        for label, (_, right) in manifests.CADENCES.items()
    }
    assert authored == {
        "schema": "emission-manifest-v1",
        "limits": manifests.SESSION_LIMITS,
        "per_geometry": per_geometry,
    }
    # One optional EOU and one park are independent of the full label budget.
    assert authored["per_geometry"]["80ms"]["max_emission_tokens"] == 12
    assert authored["per_geometry"]["1120ms"]["max_emission_tokens"] == 142


# ---- PORT-INT-005: checkpoint profile -------------------------------------


def _four_hashes() -> dict[str, str]:
    return {
        "state": "sha256:" + "1" * 64,
        "geometry": "sha256:" + "2" * 64,
        "transition": "sha256:" + "3" * 64,
        "emission": "sha256:" + "4" * 64,
    }


def _profile_identity() -> dict[str, str]:
    return {
        "source_checkpoint_digest": "sha256:" + "5" * 64,
        "converted_dump_digest": "sha256:" + "8" * 64,
        "derived_model_digest": "sha256:" + "6" * 64,
        "prompt_dictionary_hash": "sha256:" + "7" * 64,
    }


def test_author_checkpoint_profile_produces_the_exact_profile() -> None:
    # @spec PORT-INT-005
    profile = manifests.author_checkpoint_profile(_four_hashes(), **_profile_identity())
    body = {
        "schema": "checkpoint-profile-v2",
        "id": "cp-nemotron-3.5-asr-streaming-0.6b",
        "precision_policy": "fp32-bringup-v1",
        "limits": manifests.SESSION_LIMITS,
        "state_manifest_hash": "sha256:" + "1" * 64,
        "geometry_manifest_hash": "sha256:" + "2" * 64,
        "transition_manifest_hash": "sha256:" + "3" * 64,
        "emission_manifest_hash": "sha256:" + "4" * 64,
        **_profile_identity(),
    }
    assert profile == dict(body, content_hash=manifests.manifest_hash(body))


def test_author_checkpoint_profile_content_hash_recomputes() -> None:
    # @spec PORT-INT-005
    # The recomputation rule: strip content_hash, rehash, get it back.
    profile = manifests.author_checkpoint_profile(_four_hashes(), **_profile_identity())
    body = {k: v for k, v in profile.items() if k != "content_hash"}
    assert manifests.manifest_hash(body) == profile["content_hash"]


def test_author_checkpoint_profile_rejects_a_missing_manifest_key() -> None:
    # @spec PORT-INT-005
    hashes = _four_hashes()
    del hashes["transition"]
    with pytest.raises(ValueError, match="transition"):
        manifests.author_checkpoint_profile(hashes, **_profile_identity())


def test_author_checkpoint_profile_rejects_a_malformed_hash() -> None:
    # @spec PORT-INT-005
    hashes = _four_hashes()
    hashes["state"] = "md5:" + "1" * 32
    with pytest.raises(ValueError, match="state"):
        manifests.author_checkpoint_profile(hashes, **_profile_identity())


@pytest.mark.parametrize(
    "field",
    (
        "source_checkpoint_digest",
        "converted_dump_digest",
        "derived_model_digest",
        "prompt_dictionary_hash",
    ),
)
def test_author_checkpoint_profile_rejects_a_malformed_identity_hash(
    field: str,
) -> None:
    # @spec PORT-INT-005, PORT-LID-001
    identity = _profile_identity()
    identity[field] = "unknown"
    with pytest.raises(ValueError, match=field):
        manifests.author_checkpoint_profile(_four_hashes(), **identity)


# ---- PORT-STATE-009: verify_state_manifest --------------------------------
# BEHAVIORAL, red until Phase 6.


def test_verify_state_manifest_accepts_matching_registered_specs() -> None:
    # @spec PORT-STATE-009
    manifest = _expected_state_manifest(_checkpoint_config())
    specs = _specs_from_entries(manifest["entries"])
    assert manifests.verify_state_manifest(manifest, specs) is None


@pytest.mark.parametrize(
    ("mutate", "names_the_defect"),
    [
        (
            lambda m, s: m["entries"].pop(),
            "endpoint.book.pending_forced_generation",
        ),
        (
            lambda m, s: m["entries"][0].__setitem__("shape", [1, 1]),
            "encoder.layers.0.window.channel",
        ),
        (
            lambda m, s: m["entries"][0].__setitem__("dtype", "float16"),
            "encoder.layers.0.window.channel",
        ),
        (
            lambda m, s: m["entries"][-1].__setitem__("init", "banana"),
            "init",
        ),
        (
            lambda m, s: m.__setitem__("total_page_bytes", 1),
            "total_page_bytes",
        ),
        (
            lambda m, s: s[3].__setitem__("shape", [2, 2]),
            "encoder.layers.1.window.channel",
        ),
    ],
    ids=[
        "missing_entry",
        "wrong_shape",
        "wrong_dtype",
        "unknown_init",
        "wrong_total_bytes",
        "live_spec_disagrees",
    ],
)
def test_verify_state_manifest_names_each_defect(
    mutate: Any, names_the_defect: str
) -> None:
    # @spec PORT-STATE-009
    # Each defect raises ValueError NAMING the first offending
    # entry/field — canonical and corrupted inputs must never receive
    # indistinguishable treatment.
    manifest = _expected_state_manifest(_checkpoint_config())
    specs = _specs_from_entries(manifest["entries"])
    mutate(manifest, specs)
    with pytest.raises(ValueError, match=names_the_defect):
        manifests.verify_state_manifest(manifest, specs)
