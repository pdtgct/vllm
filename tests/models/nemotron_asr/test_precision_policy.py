# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PrecisionPolicy: every dtype declaration delegates to one policy object.

Specs: PORT-PREC-001 (delegation, no hardcoded dtypes), PORT-PREC-002
(engine-dtype conflict fails fast naming the policy), PORT-PREC-003
(fp32 bring-up value), PORT-PREC-005 (recurrent-state dtype is an
independent axis, defaults fp32, never silently reduced).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.nemotron_asr import precision
from vllm.models.nemotron_asr.precision import (
    FP32_BRINGUP,
    PrecisionPolicy,
    RecurrentStatePrecisionError,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]


def test_bringup_policy_is_fp32_everywhere():
    for tensor_class in (
        "weights",
        "activations",
        "attention_cache",
        "conv_state",
        "lstm_state",
        "queue_state",
    ):
        assert FP32_BRINGUP.dtype_for(tensor_class) == torch.float32


def test_identifier_matches_harness_scheme():
    # The uniform fp32 policy canonicalizes to {"*": "fp32"} and must hash
    # to the same identifier the eval harness records in golden/run
    # provenance (EVAL-GOLD-004) — provenance keys are shared across repos.
    assert FP32_BRINGUP.identifier == "pp-d479361445b4"
    assert FP32_BRINGUP.content_hash == (
        "sha256:d479361445b45b54ce1b0df4cd11b7ee1a06deb235ceeb8cf97c565bed2f5ccd"
    )


def test_identifier_is_order_independent_and_distinct():
    a = PrecisionPolicy({"weights": "bf16", "lstm_state": "fp32"})
    b = PrecisionPolicy({"lstm_state": "fp32", "weights": "bf16"})
    assert a.identifier == b.identifier
    assert a.identifier != FP32_BRINGUP.identifier


def test_unknown_tensor_class_is_an_error():
    with pytest.raises(KeyError):
        FP32_BRINGUP.dtype_for("no_such_class")


def test_wildcard_covers_unlisted_classes():
    # Both recurrent-state classes pinned fp32 — a bf16 wildcard alone
    # would (correctly) trip the PORT-PREC-005 guard via conv_state.
    policy = PrecisionPolicy({"*": "bf16", "conv_state": "fp32", "lstm_state": "fp32"})
    assert policy.dtype_for("weights") == torch.bfloat16
    assert policy.dtype_for("lstm_state") == torch.float32


def test_recurrent_state_below_fp32_requires_explicit_override():
    # PORT-PREC-005: lstm_state/conv_state dtype never silently drops
    # below fp32 — an explicit reviewed-policy flag is the only path.
    with pytest.raises(RecurrentStatePrecisionError):
        PrecisionPolicy({"*": "fp32", "lstm_state": "bf16"})
    with pytest.raises(RecurrentStatePrecisionError):
        PrecisionPolicy({"*": "bf16"})  # wildcard would cover conv_state
    ok = PrecisionPolicy({"*": "bf16", "conv_state": "fp32", "lstm_state": "fp32"})
    assert ok.dtype_for("activations") == torch.bfloat16


def test_engine_dtype_conflict_fails_fast_naming_policy():
    # PORT-PREC-002: --dtype bfloat16 during fp32 bring-up must fail at
    # startup with the policy named, never silently allocate pages.
    with pytest.raises(ValueError, match="PrecisionPolicy"):
        FP32_BRINGUP.assert_engine_dtype(torch.bfloat16)
    FP32_BRINGUP.assert_engine_dtype(torch.float32)


@pytest.mark.parametrize("selection", [None, "float32"])
def test_encoder_compute_default_preserves_existing_policy_identifiers(selection):
    """@spec PORT-PREC-009, PORT-PREC-013:
    defaults retain the exact bring-up identity.
    """
    assert precision.resolve_precision_policy(SimpleNamespace()) is FP32_BRINGUP
    resolved = precision.resolve_precision_policy(
        SimpleNamespace(experimental_encoder_compute_dtype=selection)
    )
    assert resolved is FP32_BRINGUP
    assert resolved.identifier == "pp-d479361445b4"
    assert FP32_BRINGUP.dtype_for("encoder_compute") == torch.float32
    assert precision.BF16_COMPUTE.dtype_for("encoder_compute") == torch.bfloat16
    assert precision.BF16_COMPUTE.content_hash == (
        "sha256:023cadacb8c7c9aaae64acb544be77c991da83b7591c9457f2aeac39deb9daa9"
    )


def test_encoder_fp16_policy_keeps_all_other_classes_fp32():
    """@spec PORT-PREC-010: only the encoder axis changes; engine stays FP32."""
    policy = precision.resolve_precision_policy(
        SimpleNamespace(experimental_encoder_compute_dtype="float16")
    )
    assert policy.dtype_for("encoder_compute") == torch.float16
    for tensor_class in precision.TENSOR_CLASSES:
        if tensor_class != "encoder_compute":
            assert policy.dtype_for(tensor_class) == torch.float32
    assert policy.identifier != FP32_BRINGUP.identifier
    policy.assert_engine_dtype(torch.float32)
    for engine_dtype in (torch.float16, torch.bfloat16):
        with pytest.raises(ValueError, match="PrecisionPolicy"):
            policy.assert_engine_dtype(engine_dtype)


@pytest.mark.parametrize(
    "selection",
    ["fp32", "fp16", "bf16", "bfloat16", "FLOAT16", " float16", "", True, 16],
)
def test_encoder_compute_rejects_invalid_selection(selection):
    """@spec PORT-PREC-009: invalid values fail without aliases or coercion."""
    with pytest.raises(ValueError, match="experimental_encoder_compute_dtype") as error:
        precision.resolve_precision_policy(
            SimpleNamespace(experimental_encoder_compute_dtype=selection)
        )
    message = str(error.value)
    assert "float32" in message and "float16" in message
    assert "None" in message or "null" in message


def test_encoder_selector_is_read_once():
    """@spec PORT-PREC-009: resolution reads the effective selector once."""

    class EffectiveConfig:
        reads = 0

        @property
        def experimental_encoder_compute_dtype(self):
            self.reads += 1
            return "float16"

    config = EffectiveConfig()
    assert (
        precision.resolve_precision_policy(config).dtype_for("encoder_compute")
        == torch.float16
    )
    assert config.reads == 1
