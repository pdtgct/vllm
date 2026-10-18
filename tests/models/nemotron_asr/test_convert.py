# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-conversion contract (PORT-WGT-002/003)."""

import pytest
import torch

from vllm.models.nemotron_asr.convert import (
    ConversionError,
    TensorRule,
    convert_state_dict,
    diff_against_referee,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]


def _checkpoint() -> dict[str, torch.Tensor]:
    return {
        "encoder.layer0.w": torch.randn(4, 2),
        "prompt_kernel.0.weight": torch.randn(6, 3),
    }


_RULES = (
    TensorRule(
        source=r"encoder\.layer(\d+)\.w",
        target=r"encoder.layers.\1.weight",
        transform="transpose01",
        expect_shape=(2, 4),
    ),
    TensorRule(
        source=r"prompt_kernel\.0\.weight",
        target="lid.prompt_kernel.0.weight",
    ),
)


def test_convert_consumes_exactly_once_with_shapes():
    out, report = convert_state_dict(_checkpoint(), _RULES)
    assert set(out) == {
        "encoder.layers.0.weight",
        "lid.prompt_kernel.0.weight",
    }
    assert report.produced["encoder.layers.0.weight"] == (2, 4)


def test_unconsumed_tensor_is_fatal_and_named():
    checkpoint = _checkpoint()
    checkpoint["stray.tensor"] = torch.randn(1)
    with pytest.raises(ConversionError, match="stray.tensor"):
        convert_state_dict(checkpoint, _RULES)


def test_rule_matching_nothing_is_fatal():
    rules = (*_RULES, TensorRule(source=r"ghost\..*", target="x"))
    with pytest.raises(ConversionError, match="ghost"):
        convert_state_dict(_checkpoint(), rules)


def test_shape_mismatch_names_tensor_and_transform():
    rules = (
        TensorRule(
            source=r"encoder\.layer(\d+)\.w",
            target=r"encoder.layers.\1.weight",
            expect_shape=(9, 9),
        ),
        _RULES[1],
    )
    with pytest.raises(ConversionError, match="encoder.layer0.w"):
        convert_state_dict(_checkpoint(), rules)


def test_missing_prompt_kernel_is_fatal():
    # PORT-WGT-003: no silent degradation to unconditioned ASR.
    checkpoint = {"encoder.layer0.w": torch.randn(4, 2)}
    with pytest.raises(ConversionError, match="prompt_kernel"):
        convert_state_dict(checkpoint, (_RULES[0],))


def test_referee_diff_names_offenders():
    out, _ = convert_state_dict(_checkpoint(), _RULES)
    referee = {"nemo.enc.w": out["encoder.layers.0.weight"] + 1.0}
    with pytest.raises(ConversionError, match="max abs diff"):
        diff_against_referee(out, referee, {"encoder.layers.0.weight": "nemo.enc.w"})
    # And equality passes silently.
    referee_ok = {"nemo.enc.w": out["encoder.layers.0.weight"]}
    diff_against_referee(out, referee_ok, {"encoder.layers.0.weight": "nemo.enc.w"})
