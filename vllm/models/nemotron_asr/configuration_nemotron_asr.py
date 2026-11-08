# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HF config for the streaming ASR model (PORT-WGT-004).

The served checkpoint has no upstream HF modeling code, so the port
authors its own ``config.json`` at conversion time and registers this
class with ``AutoConfig`` (see ``transformers_utils/configs``) so vLLM
loads it without ``trust_remote_code``. Only the fields vLLM's config
machinery and this model's ``__init__`` actually read live here; the
values are produced by the conversion publisher
(``convert.author_config``), never hand-authored.
"""

import json
from pathlib import Path
from typing import Any

from transformers import PreTrainedConfig

from vllm.models.nemotron_asr.identity import ARCHITECTURE
from vllm.models.nemotron_asr.manifests import (
    ENVELOPE_HEADER_FIELDS,
    RAW_SAMPLES_PER_CHUNK,
)

MODEL_TYPE = "nemotron_asr"

#: The public HuggingFace card's model type (owned by transformers'
#: ``nemotron3_5_asr`` implementation; this port serves that checkpoint
#: directly through the translating config below).
CARD_MODEL_TYPE = "nemotron3_5_asr"

#: The port's pinned right attention context (lookahead, in subsampled
#: encoder frames) — the 1120 ms arm. The card's
#: ``default_num_lookahead_tokens`` is the *default* arm, not a bound;
#: translation validates this pin against the card's supported set.
_PINNED_LOOKAHEAD = 13

#: The raw chunk-envelope carrier width: header slots plus the largest
#: admitted raw cadence (1120 ms at 16 kHz) — DERIVED from the
#: manifests, never a copied constant (PORT-INT-004; the retired mel
#: default 15,617 is superseded by this envelope value, 17,926).
_CARRIER_WIDTH = len(ENVELOPE_HEADER_FIELDS) + max(RAW_SAMPLES_PER_CHUNK.values())


def validate_prompt_dictionary(
    prompt_dictionary: object,
    num_prompts: object,
) -> dict[str, int]:
    """Validate the published locale-to-conditioning-row authority.

    Config construction remains permissive so Transformers can create an
    empty default config for registry/introspection work. The serving model
    calls this guard before allocating model state; a served artifact may
    never start without a complete, in-range prompt binding.

    Args:
        prompt_dictionary: Candidate locale-to-row mapping.
        num_prompts: Number of conditioning rows in the checkpoint.

    Returns:
        A defensive copy with normalized static types.

    Raises:
        ValueError: If the row count or any mapping entry is invalid.

    """
    if (
        isinstance(num_prompts, bool)
        or not isinstance(num_prompts, int)
        or num_prompts <= 0
    ):
        raise ValueError("num_prompts must be a positive integer")
    if not isinstance(prompt_dictionary, dict) or not prompt_dictionary:
        raise ValueError("prompt_dictionary must be a non-empty object")
    validated: dict[str, int] = {}
    for locale, index in prompt_dictionary.items():
        if (
            not isinstance(locale, str)
            or not locale
            or isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < num_prompts
        ):
            raise ValueError(
                "prompt_dictionary must map non-empty locale strings to integer rows "
                f"in [0, {num_prompts})"
            )
        validated[locale] = index
    return validated


class NemotronASRConfig(PreTrainedConfig):
    """Config for the FastConformer RNN-T streaming ASR model.

    ``hidden_size`` is the multimodal-carrier row width (D-BU-2), not
    an LM width — this model has no LM stack; the encoder's true width
    is ``d_model``. ``vocab_size`` covers the checkpoint's label set
    plus the minted specials (park = ``eos_token_id``, the audio-chunk
    placeholder).
    """

    model_type = MODEL_TYPE

    def __init__(
        self,
        *,
        vocab_size: int = 13090,
        num_asr_labels: int = 13087,
        hidden_size: int = _CARRIER_WIDTH,
        eos_token_id: int | None = None,
        audio_chunk_token_id: int | None = None,
        eou_token_id: int | None = None,
        flush_token_id: int | None = None,
        endpoint_history_capacity_frames: int = 12,
        encoder_execution_arm: str | None = None,
        decode_dispatch_arm: str | None = None,
        decode_dispatch_table: str | None = None,
        performance_gated: bool = False,
        prompt_dictionary: dict[str, int] | None = None,
        d_model: int = 1024,
        n_layers: int = 24,
        conv_kernel: int = 9,
        att_context_left: int = 56,
        att_context_right: int = 13,
        pred_hidden: int = 640,
        pred_rnn_layers: int = 2,
        joint_hidden: int = 640,
        num_prompts: int = 128,
        n_mels: int = 128,
        max_position_embeddings: int = 32768,
        torch_dtype: str = "float32",
        **kwargs: object,
    ) -> None:
        self.vocab_size = vocab_size
        #: The decoded label-set size V (blank = V, joint emits V+1);
        #: distinct from ``vocab_size``, the engine logit width that
        #: also covers the minted park/placeholder specials. NOT named
        #: ``num_labels`` — that is a reserved ``PreTrainedConfig``
        #: field (it drives ``id2label`` and is reset to a 2-label
        #: default by ``super().__init__``, silently clobbering ours).
        self.num_asr_labels = num_asr_labels
        self.hidden_size = hidden_size
        self.audio_chunk_token_id = audio_chunk_token_id
        self.eou_token_id = eou_token_id
        self.flush_token_id = flush_token_id
        self.endpoint_history_capacity_frames = endpoint_history_capacity_frames
        controls = (
            eos_token_id,
            audio_chunk_token_id,
            eou_token_id,
            flush_token_id,
        )
        if any(value is not None for value in controls):
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in controls
            ):
                raise ValueError("all four control token ids must be integers")
            typed_controls = tuple(
                int(value) for value in controls if value is not None
            )
            if len(set(typed_controls)) != len(typed_controls):
                raise ValueError("control token ids must be pairwise distinct")
            if min(typed_controls) <= num_asr_labels:
                raise ValueError(
                    "control token ids must be outside the label and blank space"
                )
            if vocab_size <= max(typed_controls):
                raise ValueError("vocab_size must cover every control token id")
        if endpoint_history_capacity_frames <= 0:
            raise ValueError("endpoint history capacity must be positive")
        # The encoder execution gate is an analysis-only startup choice.
        # Absence retains the compatibility eager baseline; an experiment
        # stamps the resolved value into its evidence fingerprint.
        self.encoder_execution_arm = encoder_execution_arm
        # Startup decode policy is part of the served artifact, not a
        # process-local default. Preserve all three fields through HF
        # serialization so build_decode_resolver sees the declaration
        # authored by the publisher (PORT-WGT-004 / PORT-DEC-008).
        self.decode_dispatch_arm = decode_dispatch_arm
        self.decode_dispatch_table = decode_dispatch_table
        self.performance_gated = performance_gated
        # Complete locale -> prompt-row authority, authored from the
        # source checkpoint metadata. Serving must never rely on a
        # sidecar path or reconstruct this binding from labels.
        self.prompt_dictionary = dict(prompt_dictionary or {})
        self.d_model = d_model
        self.n_layers = n_layers
        self.conv_kernel = conv_kernel
        self.att_context_left = att_context_left
        self.att_context_right = att_context_right
        self.pred_hidden = pred_hidden
        self.pred_rnn_layers = pred_rnn_layers
        self.joint_hidden = joint_hidden
        self.num_prompts = num_prompts
        self.n_mels = n_mels
        self.max_position_embeddings = max_position_embeddings
        # ``architectures`` rides kwargs, never an explicit arg: a
        # config.json (from author_config or to_dict) already carries
        # it, and ``from_dict`` re-passes every key — an explicit
        # ``architectures=`` would then collide with **kwargs and raise
        # on every reload.
        kwargs.setdefault("architectures", [ARCHITECTURE])
        super().__init__(
            eos_token_id=eos_token_id,
            torch_dtype=torch_dtype,
            **kwargs,
        )


def _require_int(mapping: dict, key: str, source: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{source} requires an integer {key!r}")
    return value


def translate_card_config(card: dict) -> dict:
    """Translate the public HF card schema into this port's schema.

    The card (transformers' ``nemotron3_5_asr``) declares the joint
    width as ``vocab_size`` with blank last; this port's ``vocab_size``
    is the engine logit width, which additionally covers four minted
    control ids. The controls are minted deterministically as the four
    ids immediately past blank, matching the offline publisher's
    arithmetic (``convert.author_config``): a control must be a genuinely
    new id (> V) and the logit width must cover every id.

    Raises:
        ValueError: If the card's identity arithmetic is inconsistent,
            a required encoder field is absent, or the port's pinned
            lookahead arm is outside the card's supported set.

    """
    card = dict(card)
    encoder = card.pop("encoder_config", None)
    if not isinstance(encoder, dict):
        encoder = (
            dict(getattr(encoder, "__dict__", None) or {})
            if encoder is not None
            else None
        )
    if not encoder:
        raise ValueError("public card requires encoder_config")
    blank = _require_int(card, "blank_token_id", "public card")
    joint_width = _require_int(card, "vocab_size", "public card")
    if joint_width != blank + 1:
        raise ValueError(
            "public card identity mismatch: vocab_size must be "
            f"blank_token_id + 1 (blank last), got vocab_size={joint_width} "
            f"blank_token_id={blank}"
        )
    card.pop("vocab_size")
    supported = encoder.get("supported_num_lookahead_tokens")
    if supported is not None and _PINNED_LOOKAHEAD not in supported:
        raise ValueError(
            f"the pinned lookahead arm ({_PINNED_LOOKAHEAD}) is not in the card's "
            f"supported set {supported}"
        )
    if supported is not None:
        # Retained verbatim: session admission validates each requested
        # cadence's implied lookahead against this declared set
        # (PORT-SESS-015) — the card, not the code, decides which arms
        # the publisher stands behind.
        card["supported_num_lookahead_tokens"] = [int(v) for v in supported]
    for key in (
        "hidden_size",
        "num_hidden_layers",
        "conv_kernel_size",
        "sliding_window",
        "num_mel_bins",
    ):
        _require_int(encoder, key, "public card encoder_config")
    d_model = int(encoder["hidden_size"])
    prompt_intermediate = card.get("prompt_intermediate_size")
    if prompt_intermediate is not None and int(prompt_intermediate) != 2 * d_model:
        raise ValueError(
            "prompt fusion width mismatch: the port's prompt kernel is "
            f"Linear(hidden+prompts -> 2*hidden); card declares "
            f"{prompt_intermediate} for hidden {d_model}"
        )
    translated = {
        "num_asr_labels": blank,
        "eos_token_id": blank + 1,
        "audio_chunk_token_id": blank + 2,
        "eou_token_id": blank + 3,
        "flush_token_id": blank + 4,
        "vocab_size": blank + 5,
        # Decode dispatch is a publisher declaration, never a runtime
        # default (PORT-DEC-008). On the card path this translation IS
        # the publisher, and it declares the qualified eager arm — the
        # same value the offline publisher stamps into the served
        # artifact.
        "decode_dispatch_arm": "dense-eager",
        "d_model": d_model,
        "n_layers": int(encoder["num_hidden_layers"]),
        "conv_kernel": int(encoder["conv_kernel_size"]),
        "att_context_left": int(encoder["sliding_window"]) - 1,
        "att_context_right": _PINNED_LOOKAHEAD,
        "n_mels": int(encoder["num_mel_bins"]),
        "torch_dtype": str(card.pop("dtype", card.pop("torch_dtype", "float32"))),
    }
    if "decoder_hidden_size" in card:
        translated["pred_hidden"] = int(card.pop("decoder_hidden_size"))
    if "num_decoder_layers" in card:
        translated["pred_rnn_layers"] = int(card.pop("num_decoder_layers"))
    # Retained card facts ride through as plain attributes; keys this
    # port reinterprets were popped above so they cannot collide.
    card.pop("model_type", None)
    # The card describes the architecture as encoder-decoder; this
    # port's serving schema deliberately does not. The model runs its
    # encoder inside its own forward with model-local attention, so
    # vLLM's encoder-decoder input path (which requires a decoder start
    # token the tokenizer does not define) must stay off — GPU launch
    # 2026-08-04 failed on exactly this rider ("Cannot find decoder
    # start token id or <BOS>").
    card.pop("is_encoder_decoder", None)
    translated.update(card)
    return translated


class NemotronCardServingConfig(NemotronASRConfig):
    """Serve the public HF card directly (PORT-WGT-004, card path).

    Registered with vLLM's config registry for ``nemotron3_5_asr`` so
    every process that builds a ``ModelConfig`` from the public
    checkpoint sees this port's canonical fields. A dict carrying
    ``encoder_config`` is the card shape and is translated; a dict
    already in the served schema (a round-trip of this class's own
    ``to_dict``) passes through unchanged.
    """

    model_type = CARD_MODEL_TYPE

    def __init__(self, **kwargs: Any) -> None:
        if "encoder_config" in kwargs:
            kwargs = translate_card_config(kwargs)
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):  # type: ignore[override]
        loaded = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        config = loaded[0] if isinstance(loaded, tuple) else loaded
        ensure_prompt_dictionary(config, pretrained_model_name_or_path)
        return loaded

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        # vLLM registers this class with AutoConfig and loads checkpoints
        # through ``AutoConfig.from_pretrained``, which dispatches here —
        # never through ``from_pretrained`` above. AutoConfig forwards the
        # checkpoint path as ``name_or_path``, so the sidecar hydration
        # must also fire on this path.
        loaded = super().from_dict(config_dict, **kwargs)
        config = loaded[0] if isinstance(loaded, tuple) else loaded
        if config.name_or_path:
            ensure_prompt_dictionary(config, config.name_or_path)
        return loaded


def ensure_prompt_dictionary(config: PreTrainedConfig, model_path: object) -> None:
    """Fill ``prompt_dictionary`` from the card's processor sidecar.

    The public card ships the complete locale-to-row authority in
    ``processor_config.json``, not ``config.json``. Idempotent: a config
    that already carries a non-empty mapping (the served artifact) is
    left untouched, and an absent sidecar leaves the config unchanged —
    the model's existing ``validate_prompt_dictionary`` gate then fails
    admission with its own error.
    """
    if getattr(config, "prompt_dictionary", None):
        return
    sidecar = None
    local = Path(str(model_path)) / "processor_config.json"
    if local.is_file():
        sidecar = json.loads(local.read_text())
    else:
        try:
            from transformers.utils import cached_file

            resolved = cached_file(
                str(model_path),
                "processor_config.json",
                _raise_exceptions_for_missing_entries=False,
            )
            if resolved:
                sidecar = json.loads(Path(resolved).read_text())
        except Exception:
            sidecar = None
    if not sidecar:
        return
    mapping = sidecar.get("prompt_dictionary")
    if mapping:
        config.prompt_dictionary = validate_prompt_dictionary(
            mapping,
            getattr(config, "num_prompts", None),
        )
