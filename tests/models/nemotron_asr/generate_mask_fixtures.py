# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate NeMo chunked-limited mask fixtures (venv-oracle ONLY).

Runs NeMo's own ``_create_masks`` at the pinned NeMo Speech commit for
every published ``att_context_size`` of the nemotron-3.5-asr-streaming
checkpoint and saves the boolean masks to ``fixtures/nemo_masks.pt``.
The committed fixture is what ``test_masks.py`` asserts bitwise
equality against (PORT-POOL-003); regenerate on any NeMo pin bump.

Usage (dev pod):
    /opt/venv-oracle/bin/python generate_mask_fixtures.py
"""

from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder

ATT_CONTEXTS = {
    "80ms": [56, 0],
    "160ms": [56, 1],
    "320ms": [56, 3],
    "560ms": [56, 6],
    "1120ms": [56, 13],
}

# Frames long enough to exercise several full windows at every config.
TOTAL_FRAMES = 6 * 56


def main() -> None:
    encoder = ConformerEncoder(
        feat_in=128,
        n_layers=1,
        d_model=64,
        feat_out=-1,
        subsampling_factor=8,
        self_attention_model="rel_pos",
        att_context_size=[56, 13],
        att_context_style="chunked_limited",
        n_heads=4,
    )
    fixtures: dict[str, np.ndarray] = {}
    for label, att_context in ATT_CONTEXTS.items():
        _, att_mask = encoder._create_masks(
            att_context_size=att_context,
            padding_length=torch.tensor([TOTAL_FRAMES]),
            max_audio_length=TOTAL_FRAMES,
            offset=None,
            device=torch.device("cpu"),
        )
        # NeMo returns (1, T, T) INVERTED (True = ignore,
        # conformer_encoder.py:892); store (T, T) True = may attend.
        # Saved as compressed npz: loadable without torch, not covered
        # by the repo's *.pt ignore rule, and a few KB instead of hundreds.
        fixtures[label] = (~att_mask[0]).numpy()
    out = Path(__file__).parent / "fixtures" / "nemo_masks.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Fixture labels are data keys, never NumPy's optional control keywords.
    np.savez_compressed(out, **cast(dict[str, Any], fixtures))
    print(f"wrote {out} ({sorted(fixtures)})")


if __name__ == "__main__":
    main()
