# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline weight-conversion verification (PORT-WGT-001/002/003).

The runtime loads standard HF safetensors; this module is the offline
tool's core: a declarative mapping from checkpoint tensor names to the
port's module tree, enforced by a consume-exactly-once ledger with
post-transform shape checks. Any mismatch is a hard failure carrying a
named tensor-level diff — never warn-and-proceed. Missing LID weights
(``prompt_kernel``) are fatal for this model id: language conditioning
is part of the contract, the model never degrades to unconditioned
transcription (PORT-WGT-003).

The concrete rule table for the shipped checkpoint is data, filled in
against the real safetensors index at P2 bring-up; the enforcement
machinery here is what the GPU-free tests pin.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import regex as re
import torch

from vllm.models.nemotron_asr.configuration_nemotron_asr import MODEL_TYPE
from vllm.models.nemotron_asr.identity import ARCHITECTURE

Transform = Callable[[torch.Tensor], torch.Tensor]

TRANSFORMS: dict[str, Transform] = {
    "identity": lambda t: t,
    "transpose01": lambda t: t.transpose(0, 1).contiguous(),
    # NeMo persists the featurizer's fb/window buffers wrapped in leading
    # singleton dim(s); the port featurizer wants the bare (n_mels,
    # n_freq) / (win_length,) tensors. Squeeze normalizes at conversion
    # so the served checkpoint is canonical (paired with expect_shape).
    "squeeze": lambda t: t.squeeze().contiguous(),
}

LID_REQUIRED_PATTERN = re.compile(r"prompt_kernel")


class ConversionError(ValueError):
    """A conversion contract violation, carrying named tensor detail."""


@dataclass(frozen=True)
class TensorRule:
    """One source→target mapping.

    ``source`` is a regex fully matching checkpoint tensor names; each
    matched tensor maps to ``target`` (with regex group expansion) after
    ``transform``; ``expect_shape``, when given, is asserted on the
    transformed tensor.
    """

    source: str
    target: str
    transform: str = "identity"
    expect_shape: tuple[int, ...] | None = None


@dataclass
class ConversionReport:
    """What the conversion consumed and produced."""

    consumed: dict[str, str] = field(default_factory=dict)
    produced: dict[str, tuple[int, ...]] = field(default_factory=dict)


def convert_state_dict(
    checkpoint: Mapping[str, torch.Tensor],
    rules: Sequence[TensorRule],
) -> tuple[dict[str, torch.Tensor], ConversionReport]:
    """Apply the rule table under the consume-exactly-once contract.

    Raises:
        ConversionError: If any checkpoint tensor is unconsumed, any
            tensor is matched by more than one rule, any rule matches
            nothing, a transformed shape misses its expectation, or a
            required LID tensor is absent (PORT-WGT-002/003). The
            message names every offending tensor.

    """
    compiled = [(rule, re.compile(rule.source)) for rule in rules]
    report = ConversionReport()
    out: dict[str, torch.Tensor] = {}
    unmatched_rules = {rule.source for rule in rules}

    for name, tensor in checkpoint.items():
        hits = [(rule, regex) for rule, regex in compiled if regex.fullmatch(name)]
        if len(hits) > 1:
            sources = [rule.source for rule, _ in hits]
            raise ConversionError(
                f"tensor {name!r} matched by multiple rules: {sources}"
            )
        if not hits:
            continue  # collected below as unconsumed
        rule, regex = hits[0]
        unmatched_rules.discard(rule.source)
        target = regex.sub(rule.target, name)
        transformed = TRANSFORMS[rule.transform](tensor)
        if (
            rule.expect_shape is not None
            and tuple(transformed.shape) != rule.expect_shape
        ):
            raise ConversionError(
                f"shape mismatch for {name!r} -> {target!r}: expected "
                f"{rule.expect_shape}, got {tuple(transformed.shape)} "
                f"(transform {rule.transform!r})"
            )
        if target in out:
            raise ConversionError(f"two tensors map to the same target {target!r}")
        out[target] = transformed
        report.consumed[name] = target
        report.produced[target] = tuple(transformed.shape)

    unconsumed = sorted(set(checkpoint) - set(report.consumed))
    problems: list[str] = []
    if unconsumed:
        problems.append(f"unconsumed checkpoint tensors: {unconsumed}")
    if unmatched_rules:
        problems.append(f"rules matching no tensor: {sorted(unmatched_rules)}")
    lid_present = any(LID_REQUIRED_PATTERN.search(name) for name in checkpoint)
    if not lid_present:
        problems.append(
            "prompt_kernel (LID) weights absent — fatal for this model "
            "id; the model never degrades to unconditioned transcription"
        )
    if problems:
        raise ConversionError("; ".join(problems))
    return out, report


def diff_against_referee(
    converted: Mapping[str, torch.Tensor],
    referee: Mapping[str, torch.Tensor],
    pairs: Mapping[str, str],
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    """Numerically compare converted tensors against the referee.

    ``pairs`` maps converted-tree names to referee (``.nemo``) names —
    the referee settles layout ambiguities (PORT-WGT-002).

    Raises:
        ConversionError: Naming every differing tensor with max-abs
            difference.

    """
    diffs: list[str] = []
    for target, referee_name in sorted(pairs.items()):
        if target not in converted:
            diffs.append(f"{target}: absent from converted tree")
            continue
        if referee_name not in referee:
            diffs.append(f"{referee_name}: absent from referee")
            continue
        a, b = converted[target], referee[referee_name]
        if a.shape != b.shape:
            diffs.append(
                f"{target}: shape {tuple(a.shape)} != referee {tuple(b.shape)}"
            )
        elif not torch.allclose(a, b, atol=atol, rtol=rtol):
            diffs.append(f"{target}: max abs diff {(a - b).abs().max().item():.3e}")
    if diffs:
        raise ConversionError("referee diff: " + "; ".join(diffs))


# ---- config authoring (PORT-WGT-004, bring-up sub-slice BU-a) -----------------

_ENCODER_LAYER_RE = re.compile(r"^encoder\.layers\.(\d+)\.")


def derive_n_layers(state_dict: Mapping[str, torch.Tensor]) -> int:
    """The encoder's layer count, derived from checkpoint tensor names.

    Reality wins over a hand-authored default: the converted state
    dict's keys are ``encoder.layers.{i}.*`` (``rules.py``'s own
    target pattern), so the layer count is one past the highest
    observed index — never a copied constant, and never silently
    wrong for a checkpoint whose depth changes.

    Fails closed on a non-contiguous index set (a gap like ``{0, 1, 3}``
    or a set that does not start at 0): the runtime builds a dense
    ``nn.ModuleList`` of ``range(n_layers)`` and loads by
    ``encoder.layers.{i}.*``, so a hole means the published depth would
    describe a layer stack the weights cannot fill — a corrupt/partial
    dump, never a smaller model. ``max(indices) + 1`` alone would
    silently paper over that gap.

    Raises:
        ConversionError: If no ``encoder.layers.{i}.*`` tensor is
            present (the state dict does not look converted), or if the
            observed indices are not exactly ``0..max`` contiguous.

    """
    indices = {
        int(m.group(1))
        for name in state_dict
        if (m := _ENCODER_LAYER_RE.match(name)) is not None
    }
    if not indices:
        raise ConversionError(
            "cannot derive n_layers: no 'encoder.layers.{i}.*' tensor in the converted "
            "state dict"
        )
    expected = set(range(max(indices) + 1))
    if indices != expected:
        missing = sorted(expected - indices)
        raise ConversionError(
            "encoder layer indices are not contiguous 0.."
            f"{max(indices)}: missing {missing} — the converted state "
            "dict is partial or corrupt, not a shorter model"
        )
    return max(indices) + 1


def derive_vocab_size(state_dict: Mapping[str, torch.Tensor]) -> int:
    """The label-set size V, derived from checkpoint tensor shapes.

    Reality wins over metadata (the .nemo meta.json and the HF model
    card disagree on 13087 vs 13088): the joint final linear has
    ``out_features = V + 1`` (blank last) and the predictor embedding
    has ``V + 1`` rows. Both must agree with each other; this returns V.

    Raises:
        ConversionError: If the two corroborating tensors imply
            different V, or either is absent.

    """

    def rows(name: str) -> int:
        if name not in state_dict:
            raise ConversionError(
                f"cannot derive vocab size: {name!r} absent from the converted state "
                "dict"
            )
        return int(state_dict[name].shape[0])

    joint_rows = rows("joint.joint_net.1.weight")  # V + 1 (blank last)
    embed_rows = rows("predictor.embed.weight")  # V + 1 (blank pad row)
    if joint_rows != embed_rows:
        raise ConversionError(
            "vocab-size tensors disagree: joint final linear implies "
            f"V={joint_rows - 1}, predictor embedding implies "
            f"V={embed_rows - 1}"
        )
    return joint_rows - 1


def author_config(
    state_dict: Mapping[str, torch.Tensor],
    *,
    eos_token_id: int,
    audio_chunk_token_id: int,
    eou_token_id: int,
    flush_token_id: int,
    hidden_size: int,
    reference_vocab_size: int | None = None,
    max_symbols_per_step: int = 10,
) -> dict:
    """Assemble the served ``config.json`` dict (PORT-WGT-004).

    ``vocab_size`` = derived V + the minted specials (park =
    ``eos_token_id`` and the audio-chunk placeholder must be distinct
    and both >= V, i.e. genuinely new ids); ``architectures`` = the
    shared ``ARCHITECTURE`` constant; ``hidden_size`` = the mm-carrier
    width; ``eos_token_id`` = the park token; ``torch_dtype`` = float32;
    ``n_layers`` = the derived encoder depth (:func:`derive_n_layers`) —
    ``author_state_manifest``'s per-layer state-page enumeration reads
    this field and previously had no source for it at all.

    Raises:
        ConversionError: If ``reference_vocab_size`` (a metadata
            cross-check, when supplied) disagrees with the derived V,
            or if the two minted special ids are not distinct new ids.

    """
    v = derive_vocab_size(state_dict)
    n_layers = derive_n_layers(state_dict)
    if reference_vocab_size is not None and reference_vocab_size != v:
        raise ConversionError(
            f"reference vocab size {reference_vocab_size} disagrees with "
            f"the derived V={v} (checkpoint tensors win — settle the "
            "metadata, do not proceed)"
        )
    controls = (
        ("eos_token_id/park", eos_token_id),
        ("audio_chunk_token_id/placeholder", audio_chunk_token_id),
        ("eou_token_id/eou", eou_token_id),
        ("flush_token_id/flush", flush_token_id),
    )
    if len({sid for _, sid in controls}) != len(controls):
        raise ConversionError("park, placeholder, EOU, and FLUSH ids must be distinct")
    for label, sid in controls:
        # ids 0..V-1 are labels and V is blank, so a minted special
        # must be strictly past blank (> V), not merely >= V.
        if sid <= v:
            raise ConversionError(
                f"{label}={sid} collides with a label (0..{v - 1}) or "
                f"blank ({v}); a minted special must be a new id (> V={v})"
            )
    # vocab_size is the logit width: it must cover every id, specials
    # included (contiguous V, V+1 gives V+2).
    vocab_size = max(v, *(sid + 1 for _, sid in controls))
    return {
        "architectures": [ARCHITECTURE],
        "model_type": MODEL_TYPE,
        "vocab_size": vocab_size,
        "num_asr_labels": v,
        "hidden_size": hidden_size,
        "eos_token_id": eos_token_id,
        "audio_chunk_token_id": audio_chunk_token_id,
        "eou_token_id": eou_token_id,
        "flush_token_id": flush_token_id,
        "torch_dtype": "float32",
        "n_layers": n_layers,
        # The publisher's decode-cap declaration (PORT-DEC-005): serving
        # validates its budgets against this value and fails closed on
        # disagreement rather than silently preferring either source.
        "max_symbols_per_step": max_symbols_per_step,
    }
