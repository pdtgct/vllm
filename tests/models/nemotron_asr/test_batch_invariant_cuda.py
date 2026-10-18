# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA qualification discriminators for the opt-in batch-invariant mode.

Run in a fresh process: installing vLLM overrides is process-wide. The synthetic
matrix uses the production optimized graph owner and real encoder dimensions.
It is a necessary numerical falsifier, never a complete qualification verdict.

Transcript and cost tests validate supplied lab packets, not synthetic substitutes.
NEMOTRON_BI_TRANSCRIPT_IDENTITY_MANIFEST names the matched FP32 arms' exact
20-cell final transcripts, fingerprints, receipts and mode-off NeMo parity.
NEMOTRON_BI_COST_MANIFEST names matched unprofiled raw observations and medians.
Missing transcript evidence leaves the verdict unqualified; missing/invalid cost
disclosure blocks publication only, without changing readiness or the verdict.
"""

from __future__ import annotations

import copy
import importlib
import json
import os
from contextlib import contextmanager
from dataclasses import replace
from itertools import combinations, product
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_encoder_batch_invariance import HISTORY, REAL_DIMENSIONS, build_core

from vllm.models.nemotron_asr.advance import _GatheredCaches
from vllm.models.nemotron_asr.decode_graph import platform_graph_runtime
from vllm.models.nemotron_asr.encoder_execution import (
    build_encoder_execution,
    encoder_geometry_shape,
    execute_encoder_transition,
)

pytestmark = pytest.mark.core_model
cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA qualification only"
)
POPULATIONS = (1, 2, 31, 63, 64, 128)
GEOMETRIES = (0, 1, 2, 3, 4)
CAPTURE_TIERS = (1, 2, 4, 8, 16, 32, 64, 128)
MAX_COMPILED_CELLS = 40


def _covering_axes(padded):
    # Tail and row are semantic positions: resolve against geometry/population
    # only when executing. B1/B2 position aliases still cover their categories.
    populations = (
        tuple(sorted(set(CAPTURE_TIERS) | set(POPULATIONS)))
        if padded
        else CAPTURE_TIERS
    )
    return (
        GEOMETRIES,
        (0, 37),
        ("zero", "one", "partial"),
        populations,
        ("repeated", "mixed"),
        ("first", "middle", "last"),
    )


def _covering_cases(padded):
    """Greedy 2-wise cover; stable product order breaks ties without randomness."""
    axes = _covering_axes(padded)
    axis_pairs = tuple(combinations(range(len(axes)), 2))
    candidates = tuple(product(*axes))
    pairs = tuple(
        {(i, case[i], j, case[j]) for i, j in axis_pairs} for case in candidates
    )
    uncovered = set().union(*pairs)
    cases, reference_keys = [], set()
    while uncovered:
        # Prefer reusing a B1 reference only when pair-coverage scores tie.
        best = max(
            range(len(candidates)),
            key=lambda k: (
                len(pairs[k] & uncovered),
                candidates[k][:3] in reference_keys,
            ),
        )
        case = candidates[best]
        cases.append(case)
        reference_keys.add(case[:3])
        uncovered.difference_update(pairs[best])
    return tuple(cases)


def _exhaustive():
    return os.environ.get("NEMOTRON_BI_EXHAUSTIVE") == "1"


def _falsifier_cases(padded):
    """Pairwise cover by default; NEMOTRON_BI_EXHAUSTIVE=1 runs the full product."""
    if _exhaustive():
        return tuple(product(*_covering_axes(padded)))
    return _covering_cases(padded)


@pytest.mark.cpu_test
@pytest.mark.parametrize("padded", [False, True], ids=["exact", "padded"])
def test_covering_cases_pairwise_complete(padded):
    axes = _covering_axes(padded)
    cases = _covering_cases(padded)
    assert cases == _covering_cases(padded)
    assert len(cases) == len(set(cases))
    for i, values in enumerate(axes):
        assert {case[i] for case in cases} == set(values)
        for j in range(i + 1, len(axes)):
            assert {(case[i], case[j]) for case in cases} == set(
                product(values, axes[j])
            )
    references = len({case[:3] for case in cases})
    resets = sum(case[3] == 128 for case in cases)
    sequences = len(cases) + references + resets
    print(
        f"{padded=}: {len(cases)} covering cases + {references} B1 references + "
        f"{resets} B128 resets = {sequences} sequences"
    )
    assert 40 <= sequences <= 80


@pytest.mark.cpu_test
def test_covering_cases_mandated_falsifiers():
    # Exact exercises all capture tiers; padded adds the non-tier 31/63 probes.
    for padded in (False, True):
        cases = _covering_cases(padded)
        assert set(CAPTURE_TIERS) <= {case[3] for case in cases}
        assert {case[4] for case in cases} == {"repeated", "mixed"}
        assert {case[0] for case in cases if case[3] == 128} == set(GEOMETRIES)
    assert set(POPULATIONS) <= {case[3] for case in _covering_cases(True)}


def _qualification_cells(execution):
    cells = tuple(
        (g, p)
        for g in execution.warmup_geometries
        for p in execution.warmup_populations
    )
    # Fail before compilation if bucketing is accidentally disabled. Keep the
    # bound independent of the domain so expanding it cannot relax the guard.
    assert len(cells) <= MAX_COMPILED_CELLS, (
        "CUDA qualification compile domain exceeds 40 cells"
    )
    assert execution.warmup_geometries == GEOMETRIES
    assert execution.warmup_populations == CAPTURE_TIERS
    return cells


def _api():
    return importlib.import_module("vllm.models.nemotron_asr.batch_invariance")


@pytest.fixture
def mode_on(isolated_batch_invariance):
    # Must precede weights, capture, or reservations. The adapter owns the
    # successful-return witness; the environment alone is not the record.
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    from vllm.model_executor.determinism import batch_invariant

    record = _api().installed_batch_invariant_mode(
        module=batch_invariant, initialize=batch_invariant.init_batch_invariance
    )
    assert record.enabled
    return record


def _bits(tensor):
    return tensor.detach().contiguous().view(torch.uint8).cpu().clone()


def _snapshot(raw, state, row, length):
    # Conditioned frames, predictor state, labels and joint logits are outside
    # the cross-batch promise. Valid raw frames and FULL floating caches are in.
    return {
        "encoder_raw": _bits(raw[row, :length]),
        **{f"encoder_window/{i}": _bits(t[row]) for i, t in enumerate(state.channel)},
        **{f"conv_tail/{i}": _bits(t[row]) for i, t in enumerate(state.time)},
    }


def _assert_snapshot(actual, expected, context):
    assert actual.keys() == expected.keys(), context
    for name in expected:
        assert torch.equal(actual[name], expected[name]), f"{context}: {name}"


def _state(population, device, history, composition, row):
    gen = torch.Generator(device=device).manual_seed(751)
    dims = REAL_DIMENSIONS
    probe_c = torch.randn(HISTORY, dims["d_model"], generator=gen, device=device) * 0.1
    probe_t = torch.randn(dims["d_model"], 8, generator=gen, device=device) * 0.1
    channels, times, valids = [], [], []
    for _ in range(dims["n_layers"]):
        c = probe_c.expand(population, -1, -1).clone()
        t = probe_t.expand(population, -1, -1).clone()
        v = torch.full((population, 1), history, dtype=torch.int32, device=device)
        if composition == "mixed":
            c.normal_(generator=gen)
            t.normal_(generator=gen)
            v[:, 0] = torch.arange(population, device=device) % (HISTORY + 1)
        c[row], t[row], v[row] = probe_c, probe_t, history
        channels.append(c)
        times.append(t)
        valids.append(v)
    return SimpleNamespace(channel=channels, time=times, window_valid=valids)


def _inputs(core, geometry, population, row, composition, chunk, final_tail, device):
    shape = encoder_geometry_shape(core, geometry)
    gen = torch.Generator(device=device).manual_seed(993 + chunk)
    probe = torch.randn(128, shape.mel_width, generator=gen, device=device)
    mel = probe.expand(population, -1, -1).clone()
    offsets = torch.full((population,), 2, dtype=torch.long, device=device)
    lengths = torch.full(
        (population,), shape.out_width, dtype=torch.long, device=device
    )
    prompts = torch.full((population,), 3, dtype=torch.long, device=device)
    if composition == "mixed":
        mel.normal_(generator=gen)
        rows = torch.arange(population, device=device)
        offsets = rows % 3
        lengths = rows % (shape.out_width + 1)  # includes idle co-rows
        prompts = rows % 128
    mel[row], offsets[row], prompts[row] = probe, 2, 3
    valid = min(final_tail, shape.out_width) if chunk == 2 else shape.out_width
    lengths[row] = valid
    if composition == "repeated":
        lengths.fill_(valid)
    # Zero-work and partial final tails exercise clipping, not padded outputs.
    if chunk == 2:
        mel[row, :, min(shape.mel_width, 9 + 8 * valid) :] = 0
        if composition == "repeated":
            mel[:] = mel[row].clone()
    return mel, offsets, lengths, prompts, shape.out_width, valid


@contextmanager
def _capture_stream(device):
    current = torch.cuda.current_stream(device)
    side = torch.cuda.Stream(device=device)
    side.wait_stream(current)
    with torch.cuda.stream(side):
        yield
    current.wait_stream(side)


@pytest.mark.parametrize("policy", ["fp32", "fp16-encoder"])
@pytest.mark.parametrize("padded", [False, True], ids=["exact", "padded"])
@torch.inference_mode()
@cuda_only
def test_prec020_023_graph_row_invariance(mode_on, policy, padded):
    """@spec PORT-PREC-020, PORT-PREC-023, PORT-PREC-026, PORT-PREC-027.

    Every geometry, first/middle/last position, repeated/mixed rows,
    empty/nonempty histories and final tails in a deterministic pairwise cover,
    with independent B128 resets for every geometry. The
    deployment cap is deliberately irrelevant to the B128 qualification envelope.

    Both arms use production population bucketing. Exact covers every capture
    tier; padded additionally covers every required falsifier population (31->32,
    63->64). This preserves the exact/padded axis without compiling all 128 sizes.
    The cover has 41 exact / 50 padded cases; with 15 B1 references and five
    independent resets this runs 61 / 70 three-chunk sequences per parameter.
    This is a 2-wise falsifier, not exhaustive higher-order interaction coverage.
    Five geometries x eight tiers = 40 static cells per parameter case, versus
    640 for the former exact arm; both policies/arms total 160 versus 1360 cells.
    At the r6 planning assumption of >~2 s/cell, compilation alone is >~80 s/case
    (>~320 s total), versus >~1280 s/exact case (>~2720 s total). These are paper
    estimates, not measured cell timings or wall-time caps: capture, replay and
    eager comparisons add cost, and cold compiler/autotuning costs can dominate.
    """
    from vllm.config import VllmConfig

    device = torch.device("cuda", torch.accelerator.current_device_index())
    gpu = torch.cuda.get_device_name(device)
    assert "A100" in gpu or "A40" in gpu, "qualification partition must be A100 or A40"
    core = build_core(policy, device, mode=mode_on)
    torch._dynamo.reset()
    execution = build_encoder_execution(
        core,
        SimpleNamespace(
            encoder_execution_arm="dense-graphed",
            encoder_population_bucketing=True,
            att_context_left=56,
        ),
        maximum_population=128,
        warmup_geometries=GEOMETRIES,
        vllm_config=VllmConfig(),
        graph_runtime=replace(
            platform_graph_runtime(), capture_context=_capture_stream
        ),
    )

    def warmup(geometry, population):
        state = _state(population, device, 37, "mixed", 0)
        mel, offsets, lengths, prompts, width, _ = _inputs(
            core, geometry, population, 0, "mixed", 0, 1, device
        )
        return execution.transition(
            mel, _GatheredCaches(state), offsets, lengths, width, prompts
        )

    cells = _qualification_cells(execution)
    execution.warmup_domain(expected_cells=cells, invoke=warmup)
    assert execution.ready
    assert execution.ready_receipt()["batch_invariant_mode"] == "on"

    def sequence(geometry, population, row, composition, history, tail):
        # Each call allocates a fresh independent state, including the second B128 run.
        state = _state(population, device, history, composition, row)
        eager_state = copy.deepcopy(state)
        snapshots = []
        for chunk in range(3):
            mel, offsets, lengths, prompts, width, valid = _inputs(
                core, geometry, population, row, composition, chunk, tail, device
            )
            raw, conditioned = execution.transition(
                mel, _GatheredCaches(state), offsets, lengths, width, prompts
            )
            eager_raw, eager_conditioned = execute_encoder_transition(
                core,
                mel,
                _GatheredCaches(eager_state),
                offsets,
                lengths,
                width,
                prompts,
            )
            # Same-batch graph/eager obligations remain broader than raw-only BI.
            assert torch.equal(_bits(raw), _bits(eager_raw))
            assert torch.equal(_bits(conditioned), _bits(eager_conditioned))
            for name in ("channel", "time", "window_valid"):
                for actual, expected in zip(
                    getattr(state, name), getattr(eager_state, name), strict=True
                ):
                    assert torch.equal(_bits(actual), _bits(expected))
            selected = range(population) if composition == "repeated" else (row,)
            snapshots.append([_snapshot(raw, state, r, valid) for r in selected])
        return snapshots

    cases = _falsifier_cases(padded)
    references = {}
    sequences = (
        len(cases)
        + len({case[:3] for case in cases})
        + sum(case[3] == 128 for case in cases)
    )
    print(
        f"{policy=} {padded=}: {len(cases)} pairwise cases, {sequences} sequences "
        "including B1/reset runs"
    )
    if not _exhaustive():
        # The bound holds for the pairwise cover only; the exhaustive
        # product (NEMOTRON_BI_EXHAUSTIVE=1) is far larger.
        assert 40 <= sequences <= 80
    for geometry, history, tail_kind, population, composition, position in cases:
        width = encoder_geometry_shape(core, geometry).out_width
        tail = {"zero": 0, "one": 1, "partial": max(1, width - 1)}[tail_kind]
        row = {"first": 0, "middle": population // 2, "last": population - 1}[position]
        key = (geometry, history, tail)
        if key not in references:
            references[key] = sequence(geometry, 1, 0, "repeated", history, tail)
        reference = references[key]
        actual = sequence(geometry, population, row, composition, history, tail)
        context = (
            policy,
            padded,
            geometry,
            population,
            composition,
            row,
            history,
            tail,
        )
        for chunk, (outputs, baseline) in enumerate(
            zip(actual, reference, strict=True)
        ):
            for output in outputs:
                _assert_snapshot(output, baseline[0], (context, chunk))
        if population == 128:
            repeat = sequence(geometry, population, row, composition, history, tail)
            for chunk_a, chunk_b in zip(actual, repeat, strict=True):
                for a, b in zip(chunk_a, chunk_b, strict=True):
                    _assert_snapshot(a, b, (context, "independent reset"))


def _manifest(variable):
    value = os.environ.get(variable)
    assert value, (
        f"provide pre-existing lab evidence via {variable}; no goldens are generated "
        "here"
    )
    path = Path(value)
    assert path.name != ".env"
    return json.loads(path.read_text()), path.parent


@cuda_only
def test_prec024_fp32_mode_transcript_identity():
    """@spec PORT-PREC-024: validate supplied 20-cell, byte-identical FP32 finals.

    No mode-on NeMo tensor gate, WER tolerance, experiment or mode installation.
    """
    from batch_invariant_packet import validate_transcript_identity

    manifest, _ = _manifest("NEMOTRON_BI_TRANSCRIPT_IDENTITY_MANIFEST")
    validate_transcript_identity(manifest)


@cuda_only
def test_prec022_disclosure_manifest():
    """@spec PORT-PREC-022: invalid disclosure blocks publication only.

    Failure does not invalidate PREC-024 evidence, change readiness or change
    the verdict. Supplied unprofiled paired cost has no performance threshold.
    """
    from batch_invariant_packet import (
        PUBLICATION_ONLY,
        installed_inventory,
        validate_cost_disclosure,
    )

    try:
        manifest, _ = _manifest("NEMOTRON_BI_COST_MANIFEST")
        inventory, source_sha256 = installed_inventory()
        validate_cost_disclosure(manifest, inventory, source_sha256)
    except (AssertionError, OSError, ValueError) as error:
        raise AssertionError(f"{PUBLICATION_ONLY}: {error}") from error


@pytest.mark.parametrize("ndim", [1, 2])
@torch.inference_mode()
@cuda_only
def test_prec021_compiled_biased_pointwise(mode_on, ndim):
    """@spec PORT-PREC-021: Inductor retains addmm and its pre-cast bias."""
    device = torch.device("cuda", torch.accelerator.current_device_index())
    shape = (2, 2, 5) if ndim == 1 else (2, 2, 3, 5)
    x = torch.empty(shape, device=device, dtype=torch.float16)
    x[:, 0], x[:, 1] = 1.0, 2**-11
    weight = torch.ones((3, 2) + (1,) * ndim, device=device, dtype=x.dtype)
    bias = torch.full((3,), -1.0, device=device, dtype=x.dtype)
    compiled = torch.compile(
        _api().pointwise_conv_as_linear, backend="inductor", fullgraph=True
    )
    actual = compiled(x, weight, bias)
    # A post-store bias addition would lose the half-ULP and return zero.
    assert torch.equal(
        actual, torch.full((2, 3) + shape[2:], 2**-11, device=device, dtype=x.dtype)
    )
    assert torch.equal(actual, _api().pointwise_conv_as_linear(x, weight, bias))


@pytest.mark.cpu_test
def test_exhaustive_switch_selects_full_product(monkeypatch):
    """NEMOTRON_BI_EXHAUSTIVE=1 selects every axis combination for long runs."""
    monkeypatch.setenv("NEMOTRON_BI_EXHAUSTIVE", "1")
    full = _falsifier_cases(False)
    assert len(full) == len(tuple(product(*_covering_axes(False))))
    assert set(_covering_cases(False)) <= set(full)
    monkeypatch.delenv("NEMOTRON_BI_EXHAUSTIVE")
    assert _falsifier_cases(False) == _covering_cases(False)
