# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Language-ID prompt conditioning (PORT-LID-001/002/004).

The conditioner reproduces NeMo's ``_apply_prompt_to_encoded``
(mixins.py:971-1001 @ de242add): one-hot prompt of size ``num_prompts``
scattered at the session's prompt index, concatenated with encoder
frames, projected through ``prompt_kernel`` (Linear -> ReLU -> Linear),
cast back to the input dtype. NeMo transposes (B, D, T) <-> (B, T, D)
around the math; the port is (B, T, D)-native — the reference test
below proves layout equivalence op-for-op.
"""

import pytest
import torch

from vllm.models.nemotron_asr.lid import (
    PromptConditioner,
    resolve_prompt_index,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]

_ENC = 32  # small enc_hidden for tests; real model uses 1024
_PROMPTS = 8  # real model uses 128


def _nemo_reference(
    conditioner: PromptConditioner,
    encoded_bdt: torch.Tensor,
    index: int,
) -> torch.Tensor:
    """NeMo's exact op sequence, (B, D, T) layout, shared weights."""
    encoded = encoded_bdt.transpose(1, 2)
    batch, time, _ = encoded.shape
    prompt = torch.zeros(
        batch, time, _PROMPTS, dtype=encoded.dtype, device=encoded.device
    )
    idx = torch.full((batch,), index, dtype=torch.long)
    prompt.scatter_(2, idx.view(batch, 1, 1).expand(-1, time, -1), 1.0)
    out_dtype = encoded.dtype
    out = conditioner.prompt_kernel(torch.cat([encoded, prompt], dim=-1)).to(out_dtype)
    return out.transpose(1, 2)


@pytest.fixture
def conditioner() -> PromptConditioner:
    torch.manual_seed(99)
    module = PromptConditioner(enc_hidden=_ENC, num_prompts=_PROMPTS)
    for param in module.parameters():
        torch.nn.init.uniform_(param, -0.2, 0.2)
    return module


def test_matches_nemo_reference_layout_for_layout(conditioner):
    torch.manual_seed(3)
    encoded_btd = torch.randn(2, 5, _ENC)
    ours = conditioner(encoded_btd, prompt_index=3)
    reference = _nemo_reference(
        conditioner, encoded_btd.transpose(1, 2), index=3
    ).transpose(1, 2)
    torch.testing.assert_close(ours, reference, atol=0.0, rtol=0.0)


def test_output_shape_and_dtype_preserved(conditioner):
    encoded = torch.randn(1, 7, _ENC)
    out = conditioner(encoded, prompt_index=0)
    assert out.shape == encoded.shape
    assert out.dtype == encoded.dtype


def test_prompt_index_changes_output(conditioner):
    # Distinct locales condition distinctly (the kernel sees the one-hot).
    encoded = torch.randn(1, 4, _ENC)
    a = conditioner(encoded, prompt_index=1)
    b = conditioner(encoded, prompt_index=2)
    assert not torch.allclose(a, b)


def test_resolve_prompt_index_validates_locale():
    # PORT-LID-001: unknown locale rejected naming availability;
    # PORT-LID-004: "auto" is a dictionary entry, not an algorithm.
    prompt_dictionary = {"en-US": 0, "de-DE": 7, "auto": 101}
    assert resolve_prompt_index(prompt_dictionary, "en-US") == 0
    assert resolve_prompt_index(prompt_dictionary, "auto") == 101
    with pytest.raises(ValueError, match="de-DE"):
        resolve_prompt_index(prompt_dictionary, "xx-XX")
