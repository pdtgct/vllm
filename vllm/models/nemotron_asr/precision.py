# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-tensor-class precision policy.

Precision is a value, never structure: every dtype or quantization
declaration in this model (spec pages, weights, compute) delegates to
one ``PrecisionPolicy``; no call site hardcodes a dtype. The policy's
identifier is a content hash over the per-tensor-class mapping and keys
golden/run provenance and parity tolerances in the evaluation harness,
so the hash scheme here must stay byte-compatible with it.

Recurrent per-session state (predictor LSTM ``(h, c)``, depthwise-conv
page) is an independent precision axis that defaults fp32 and is never
silently reduced — recurrent accumulation compounds rounding error
(the same convention as vLLM's ``mamba_ssm_cache_dtype``).
"""

import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Final

import torch

TENSOR_CLASSES: Final = (
    "weights",
    "activations",
    "encoder_compute",
    "attention_cache",
    "conv_state",
    "lstm_state",
    "queue_state",
    # The frontend raw/mel tails (Phase 6b): golden-boundary audio
    # state, fp32 at bring-up; sub-fp32 policies must scope it
    # explicitly (BF16_COMPUTE pins it fp32 below).
    "frontend_state",
)

RECURRENT_STATE_CLASSES: Final = ("conv_state", "lstm_state")

_DTYPES: Final = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

_SUB_FP32: Final = frozenset({"bf16", "fp16"})


class RecurrentStatePrecisionError(ValueError):
    """A recurrent-state dtype below fp32 without an explicit override."""


class PrecisionPolicy:
    """Immutable mapping of tensor classes to dtype names.

    A ``"*"`` key provides the default for unlisted classes. Recurrent
    state classes below fp32 raise unless ``allow_sub_fp32_recurrent``
    is set — the constructor flag is the reviewed-policy-change hook.
    """

    def __init__(
        self,
        mapping: Mapping[str, str],
        *,
        allow_sub_fp32_recurrent: bool = False,
    ) -> None:
        for tensor_class, dtype_name in mapping.items():
            if tensor_class != "*" and tensor_class not in TENSOR_CLASSES:
                raise KeyError(f"unknown tensor class: {tensor_class!r}")
            if dtype_name not in _DTYPES:
                raise KeyError(f"unknown dtype name: {dtype_name!r}")
        self._mapping = dict(mapping)
        if not allow_sub_fp32_recurrent:
            for state_class in RECURRENT_STATE_CLASSES:
                try:
                    resolved = self._name_for(state_class)
                except KeyError:
                    # Partial policy: unresolved classes fail at use time.
                    continue
                if resolved in _SUB_FP32:
                    raise RecurrentStatePrecisionError(
                        f"{state_class} resolves below fp32; recurrent "
                        "state precision is never reduced implicitly "
                        "(pass allow_sub_fp32_recurrent=True only as an "
                        "explicit reviewed policy change)"
                    )

    def _name_for(self, tensor_class: str) -> str:
        if tensor_class in self._mapping:
            return self._mapping[tensor_class]
        # A missing encoder override inherits the original compute policy;
        # do not add a key to existing mappings or restamp their provenance.
        if tensor_class == "encoder_compute":
            return self._name_for("weights")
        if "*" in self._mapping:
            return self._mapping["*"]
        raise KeyError(f"policy declares no dtype for {tensor_class!r} and no '*'")

    def dtype_for(self, tensor_class: str) -> torch.dtype:
        """Resolve one tensor class to its torch dtype."""
        if tensor_class != "*" and tensor_class not in TENSOR_CLASSES:
            raise KeyError(f"unknown tensor class: {tensor_class!r}")
        return _DTYPES[self._name_for(tensor_class)]

    @property
    def encoder_compute_override(self) -> str | None:
        """The explicit experimental axis, absent in existing policies."""
        return self._mapping.get("encoder_compute")

    @property
    def content_hash(self) -> str:
        """Full canonical mapping digest used in run provenance."""
        canonical = json.dumps(
            {key: self._mapping[key] for key in sorted(self._mapping)},
            separators=(",", ":"),
            ensure_ascii=True,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    @property
    def identifier(self) -> str:
        """Short content identifier, byte-compatible with the harness."""
        return f"pp-{self.content_hash.removeprefix('sha256:')[:12]}"

    def assert_engine_dtype(self, engine_dtype: torch.dtype) -> None:
        """Fail fast when the launch dtype conflicts with the policy."""
        expected = self.dtype_for("weights")
        if engine_dtype != expected:
            raise ValueError(
                f"engine dtype {engine_dtype} conflicts with "
                f"PrecisionPolicy {self.identifier} (weights: {expected}); "
                "precision changes are reviewed policy values, not launch "
                "flags"
            )


FP32_BRINGUP: Final = PrecisionPolicy({"*": "fp32"})
"""The bring-up value: fp32 everywhere, matching the fp32-locked oracle."""

FP16_ENCODER_EXPERIMENT: Final = PrecisionPolicy(
    {"*": "fp32", "encoder_compute": "fp16"}
)
"""Unqualified encoder-only experiment; frontend, decoder and state stay FP32."""


def resolve_precision_policy(hf_config: object) -> PrecisionPolicy:
    """@spec PORT-PREC-009, PORT-PREC-013: resolve the startup-only selector.

    The effective HF configuration already includes stage-local overrides.
    Keep omission, null and explicit float32 byte-identical to the bring-up
    policy; public selector spellings do not alias the internal dtype names.
    """
    selection = getattr(hf_config, "experimental_encoder_compute_dtype", None)
    if selection is None:
        return FP32_BRINGUP
    if isinstance(selection, str):
        if selection == "float32":
            return FP32_BRINGUP
        if selection == "float16":
            return FP16_ENCODER_EXPERIMENT
    raise ValueError(
        "experimental_encoder_compute_dtype must be None (null), 'float32' or "
        f"'float16'; got {selection!r}"
    )


BF16_COMPUTE: Final = PrecisionPolicy(
    {
        "weights": "bf16",
        "activations": "bf16",
        "attention_cache": "fp32",
        "conv_state": "fp32",
        "lstm_state": "fp32",
        "queue_state": "fp32",
        # Frontend tails stay fp32 under reduced compute: they are the
        # golden input boundary (adding this key restamps this
        # policy's content-hash identifier; only FP32_BRINGUP's id is
        # pinned by value, and its mapping is unchanged).
        "frontend_state": "fp32",
    }
)
"""The first reviewed sub-fp32 policy (PORT-PREC-003): compute classes
bf16, every state class fp32 — the vLLM mamba mixed-precision idiom
(compute dtype below the recurrent/cache dtype, never the reverse).
Transcript parity vs the fp32-locked oracle stays BLOCKING; tensor
deltas are advisory under this identifier (EVAL-PAR-006)."""


def _safe_read(read: Callable[[], object]) -> object:
    """Evaluate ``read()``; any failure (e.g. torch's mixed legacy/new TF32
    API ``RuntimeError``) yields ``None`` instead of propagating."""
    try:
        return read()
    except Exception:
        return None


def _tf32_from(precision: object, legacy_read: Callable[[], object]) -> bool | None:
    """Effective TF32 state: the per-operation ``fp32_precision`` string when
    it is decisive, else the legacy flag, else ``None`` (unknown)."""
    if precision == "tf32":
        return True
    if precision == "ieee":
        return False
    legacy = _safe_read(legacy_read)
    return legacy if isinstance(legacy, bool) else None


def reduced_precision_state() -> dict[str, object]:
    """Report torch's effective reduced-precision flags without ever raising.

    Read-only. Newer PyTorch exposes per-operation ``fp32_precision``
    settings; once those are set (vLLM batch-invariant mode does so),
    reading the aggregate legacy ``allow_tf32`` attributes can raise
    ``RuntimeError`` when the cuBLAS or cuDNN conv/RNN settings disagree.
    This prefers the new API, falls back to the legacy attributes, and
    reports ``None`` for any value that cannot be determined.
    """
    matmul = torch.backends.cuda.matmul
    cudnn = torch.backends.cudnn
    matmul_precision = _safe_read(lambda: matmul.fp32_precision)
    conv_precision = _safe_read(lambda: cudnn.conv.fp32_precision)
    rnn_precision = _safe_read(lambda: cudnn.rnn.fp32_precision)
    conv_tf32 = _tf32_from(conv_precision, lambda: cudnn.allow_tf32)
    rnn_tf32 = _tf32_from(rnn_precision, lambda: cudnn.allow_tf32)
    return {
        "matmul_fp32_precision": matmul_precision,
        "cudnn_conv_fp32_precision": conv_precision,
        "cudnn_rnn_fp32_precision": rnn_precision,
        "float32_matmul_precision": _safe_read(torch.get_float32_matmul_precision),
        "tf32_matmul_enabled": _tf32_from(matmul_precision, lambda: matmul.allow_tf32),
        "tf32_cudnn_conv_enabled": conv_tf32,
        "tf32_cudnn_rnn_enabled": rnn_tf32,
        # Aggregate cuDNN flag: defined only when conv and RNN agree.
        "tf32_cudnn_enabled": conv_tf32 if conv_tf32 == rnn_tf32 else None,
        "allow_fp16_reduced_precision_reduction": _safe_read(
            lambda: matmul.allow_fp16_reduced_precision_reduction
        ),
        "allow_bf16_reduced_precision_reduction": _safe_read(
            lambda: matmul.allow_bf16_reduced_precision_reduction
        ),
    }
