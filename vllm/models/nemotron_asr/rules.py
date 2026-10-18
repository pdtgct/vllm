# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Concrete weight-mapping rules for nemotron-3.5-asr-streaming-0.6b.

Written against the restored checkpoint's state dict (657 tensors,
dumped 2026-07-11 on-pod). Encoder-layer and prompt-kernel names map
identity; the predictor and joint re-root; the featurizer's persisted
``fb``/``window`` buffers map to the port featurizer (checkpoint
buffers are inputs, never recomputed). Enforced by
:func:`convert.convert_state_dict`'s consume-exactly-once contract.
"""

from vllm.models.nemotron_asr.convert import TensorRule

NEMO_RULES: tuple[TensorRule, ...] = (
    # Featurizer buffers (persisted in the checkpoint). NeMo wraps these
    # in a leading singleton dim; squeeze to the port featurizer's bare
    # (n_mels, n_freq) / (win_length,) and assert the canonical shape.
    TensorRule(
        source=r"preprocessor\.featurizer\.fb",
        target="featurizer.fb",
        transform="squeeze",
        expect_shape=(128, 257),
    ),
    TensorRule(
        source=r"preprocessor\.featurizer\.window",
        target="featurizer.window",
        transform="squeeze",
        expect_shape=(400,),
    ),
    # Subsampling + conformer layers: identity under the encoder root.
    TensorRule(
        source=r"encoder\.pre_encode\.(conv\.\d+\.(?:weight|bias)"
        r"|out\.(?:weight|bias))",
        target=r"encoder.pre_encode.\1",
    ),
    TensorRule(
        source=r"encoder\.layers\.(\d+)\.(.+)",
        target=r"encoder.layers.\1.\2",
    ),
    # Predictor: embed + LSTM re-root (torch LSTM param names shared).
    TensorRule(
        source=r"decoder\.prediction\.embed\.weight",
        target="predictor.embed.weight",
    ),
    TensorRule(
        source=r"decoder\.prediction\.dec_rnn\.lstm\."
        r"((?:weight|bias)_(?:ih|hh)_l\d+)",
        target=r"predictor.rnn.\1",
    ),
    # Joint: side projections identity; the final linear sits at
    # Sequential index 2 in NeMo (activation+dropout precede) and
    # index 1 here (ReLU, Linear).
    TensorRule(
        source=r"joint\.(enc|pred)\.(weight|bias)",
        target=r"joint.\1.\2",
    ),
    TensorRule(
        source=r"joint\.joint_net\.2\.(weight|bias)",
        target=r"joint.joint_net.1.\1",
    ),
    # LID prompt kernel: identity indices (0, 2), with biases.
    TensorRule(
        source=r"prompt_kernel\.(\d+)\.(weight|bias)",
        target=r"lid.prompt_kernel.\1.\2",
    ),
)


import regex as _re  # noqa: E402  (kept local to the card map below)

#: Renames from the public HF card's tensor names (transformers'
#: ``nemotron3_5_asr`` modeling tree) to this port's module tree.
#: Written against the published ``model.safetensors`` header (655
#: tensors) and the upstream modeling source. Names the card and the
#: port already share (conformer feed-forward/norm/conv leaves) are
#: deliberately absent — they load as-is; only genuine renames appear.
_CARD_RENAMES: tuple[tuple[_re.Pattern[str], object], ...] = (
    # Subsampling: named separable stages -> the port's Sequential
    # indices (0 = conv_in; each later stage is [dw, pw, act] at
    # 2+3n / 3+3n).
    (
        _re.compile(r"^encoder\.subsampling\.conv_in\.(weight|bias)$"),
        r"encoder.pre_encode.conv.0.\1",
    ),
    (
        _re.compile(
            r"^encoder\.subsampling\.layers\.(\d+)\.depthwise_conv\.(weight|bias)$"
        ),
        lambda m: f"encoder.pre_encode.conv.{2 + 3 * int(m.group(1))}.{m.group(2)}",
    ),
    (
        _re.compile(
            r"^encoder\.subsampling\.layers\.(\d+)\.pointwise_conv\.(weight|bias)$"
        ),
        lambda m: f"encoder.pre_encode.conv.{3 + 3 * int(m.group(1))}.{m.group(2)}",
    ),
    (
        _re.compile(r"^encoder\.subsampling\.linear\.(weight|bias)$"),
        r"encoder.pre_encode.out.\1",
    ),
    # Conformer attention: projection and relative-position renames.
    (
        _re.compile(r"^(encoder\.layers\.\d+\.self_attn\.)q_proj\.(weight)$"),
        r"\1linear_q.\2",
    ),
    (
        _re.compile(r"^(encoder\.layers\.\d+\.self_attn\.)k_proj\.(weight)$"),
        r"\1linear_k.\2",
    ),
    (
        _re.compile(r"^(encoder\.layers\.\d+\.self_attn\.)v_proj\.(weight)$"),
        r"\1linear_v.\2",
    ),
    (
        _re.compile(r"^(encoder\.layers\.\d+\.self_attn\.)o_proj\.(weight)$"),
        r"\1linear_out.\2",
    ),
    (
        _re.compile(r"^(encoder\.layers\.\d+\.self_attn\.)relative_k_proj\.(weight)$"),
        r"\1linear_pos.\2",
    ),
    (
        _re.compile(r"^(encoder\.layers\.\d+\.self_attn\.)bias_(u|v)$"),
        r"\1pos_bias_\2",
    ),
    # Conformer conv-module norm: the card names it ``norm``; the port
    # keeps NeMo's ``batch_norm`` attribute (a LayerNorm in this model).
    (
        _re.compile(r"^(encoder\.layers\.\d+\.conv\.)norm\.(weight|bias)$"),
        r"\1batch_norm.\2",
    ),
    # Predictor re-root.
    (
        _re.compile(r"^decoder\.embedding\.weight$"),
        "predictor.embed.weight",
    ),
    (
        _re.compile(r"^decoder\.lstm\.((?:weight|bias)_(?:ih|hh)_l\d+)$"),
        r"predictor.rnn.\1",
    ),
    # Joint re-root: the card's encoder/decoder projectors are the
    # joint side projections; ``head`` is the final linear.
    (
        _re.compile(r"^encoder_projector\.(weight|bias)$"),
        r"joint.enc.\1",
    ),
    (
        _re.compile(r"^decoder\.decoder_projector\.(weight|bias)$"),
        r"joint.pred.\1",
    ),
    (
        _re.compile(r"^joint\.head\.(weight|bias)$"),
        r"joint.joint_net.1.\1",
    ),
    # LID prompt fusion MLP.
    (
        _re.compile(r"^prompt_projector\.linear_1\.(weight|bias)$"),
        r"lid.prompt_kernel.0.\1",
    ),
    (
        _re.compile(r"^prompt_projector\.linear_2\.(weight|bias)$"),
        r"lid.prompt_kernel.2.\1",
    ),
)


def remap_card_name(name: str) -> str | None:
    """Map one public-card tensor name to the port tree, or ``None``.

    ``None`` means the name is not a card-specific rename — either it is
    already a port name (shared leaves, or the served artifact) or it is
    genuinely unknown; the strict loader distinguishes those.
    """
    for pattern, replacement in _CARD_RENAMES:
        match = pattern.match(name)
        if match is None:
            continue
        if callable(replacement):
            return str(replacement(match))
        return pattern.sub(replacement, name)
    return None
