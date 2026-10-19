# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests-first contract for honest no-resident-state startup profiling."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from vllm.models.nemotron_asr.precision import FP32_BRINGUP

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]


def _profile_module() -> Any:
    try:
        return importlib.import_module("vllm.models.nemotron_asr.profile_execution")
    except ModuleNotFoundError:
        pytest.fail(
            "PORT-MIG-005 missing persistent-state profile execution",
            pytrace=False,
        )


def _model_module() -> Any:
    return importlib.import_module("vllm.models.nemotron_asr.nemotron_asr")


def _model_class() -> type[Any]:
    return cast(type[Any], _model_module().NemotronASRForRNNT)


def _config(*, decode_dispatch_arm: str = "dense-eager") -> Any:
    config_module = importlib.import_module(
        "vllm.models.nemotron_asr.configuration_nemotron_asr"
    )
    return config_module.NemotronASRConfig(
        vocab_size=13_092,
        num_asr_labels=13_087,
        eos_token_id=13_088,
        audio_chunk_token_id=13_089,
        eou_token_id=13_090,
        flush_token_id=13_091,
        prompt_dictionary={"en-US": 0},
        decode_dispatch_arm=decode_dispatch_arm,
    )


def _projection(*, is_profile: bool, num_rows: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        dummy_run=True,
        is_profile=is_profile,
        no_page_io=True,
        req_ids=tuple(f"opaque-{index}" for index in range(num_rows)),
        bindings=(),
        request_metadata=None,
    )


def _no_state_model() -> Any:
    model = object.__new__(_model_class())
    object.__setattr__(model, "config", _config())
    object.__setattr__(model, "core", object())
    object.__setattr__(model, "_emission_adapter", object())
    object.__setattr__(model, "_decode_resolver", object())
    object.__setattr__(model, "_decode_graph_binding", None)
    object.__setattr__(
        model,
        "_encoder_execution",
        SimpleNamespace(transition=object()),
    )
    return model


def test_profile_invocation_uses_largest_geometry_and_maximum_live_rows() -> None:
    # @spec PORT-MIG-005 / PORT-STATE-003 / PORT-STATE-007
    profile = _profile_module()
    config = _config()

    invocation = profile.build_profile_invocation(
        config,
        num_rows=3,
        device=torch.device("cpu"),
    )

    assert invocation.geometry_label == "1120ms"
    assert invocation.geometry_id == 4
    assert invocation.num_rows == 3
    assert invocation.plan.num_decodes == 0
    assert invocation.plan.num_prefills == 3
    assert invocation.plan.num_pool_blocks == 4
    assert invocation.plan.null_block_id == 0
    assert invocation.plan.state_indices_p.tolist() == [1, 2, 3]
    assert invocation.plan.live_block_ids.tolist() == [1, 2, 3]
    assert invocation.plan.has_initial_states_p.tolist() == [False] * 3
    assert invocation.plan.is_chunk.tolist() == [True] * 3
    assert invocation.plan.geometry_id.tolist() == [4] * 3


def test_profile_invocation_uses_ephemeral_manifest_storage_and_valid_carriers() -> (
    None
):
    # @spec PORT-MIG-005 / PORT-INT-004 / PORT-STATE-009
    profile = _profile_module()
    manifests = importlib.import_module("vllm.models.nemotron_asr.manifests")
    config = _config()

    invocation = profile.build_profile_invocation(
        config,
        num_rows=2,
        device=torch.device("cpu"),
    )

    assert invocation.storage.raw.numel() == (
        invocation.storage.spec.page_size_bytes * 3
    )
    assert invocation.input_ids.tolist() == [config.audio_chunk_token_id] * 2
    assert invocation.inputs_embeds.shape == (2, config.hidden_size)
    valid_index = manifests.ENVELOPE_HEADER_FIELDS.index("valid_samples")
    geometry_index = manifests.ENVELOPE_HEADER_FIELDS.index("geometry_id")
    assert invocation.inputs_embeds[:, valid_index].tolist() == [17_920.0] * 2
    assert invocation.inputs_embeds[:, geometry_index].tolist() == [4.0] * 2


def test_profile_invocation_accepts_one_exact_geometry() -> None:
    # @spec PORT-PERF-009
    profile = _profile_module()
    config = _config()

    invocation = profile.build_profile_invocation(
        config,
        num_rows=2,
        device=torch.device("cpu"),
        geometry_id=1,
    )

    assert invocation.geometry_label == "160ms"
    assert invocation.geometry_id == 1
    assert invocation.plan.geometry_id.tolist() == [1, 1]
    valid_index = profile.ENVELOPE_HEADER_FIELDS.index("valid_samples")
    assert invocation.inputs_embeds[:, valid_index].tolist() == [2_560.0, 2_560.0]


def test_profile_execution_invokes_the_canonical_transaction_and_drains_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-ADV-001 / PORT-MIG-003 / PORT-MIG-005
    profile = _profile_module()
    invocation = SimpleNamespace(
        input_ids=torch.zeros(2, dtype=torch.long),
        inputs_embeds=torch.zeros(2, 7),
        plan=object(),
        pools=SimpleNamespace(
            channel=(),
            convolution=(),
            valid_length=(),
            predictor_h=object(),
            predictor_c=object(),
            replay_queue=object(),
            replay_book=object(),
            frontend_raw=object(),
            frontend_mel=object(),
            frontend_counters=object(),
            endpoint_history=object(),
            endpoint_book=object(),
        ),
    )
    events: list[tuple[str, object]] = []
    monkeypatch.setattr(
        profile.logger,
        "info",
        lambda *args: events.append(("log", args)),
    )

    def build_invocation(config, *, num_rows, device):
        events.append(("build", (config, num_rows, device)))
        return invocation

    monkeypatch.setattr(profile, "build_profile_invocation", build_invocation)
    monkeypatch.setattr(
        profile,
        "advance_model_rows",
        lambda *args, **kwargs: events.append(("advance", (args, kwargs))),
    )
    monkeypatch.setattr(
        profile,
        "warmup_advance_model_rows_scatter",
        lambda **kwargs: events.append(("warmup", kwargs)),
    )
    monkeypatch.setattr(
        profile,
        "consume_batch_stats",
        lambda: events.append(("drain", None)),
    )
    model = _no_state_model()

    profile.run_persistent_state_profile(
        model,
        num_rows=2,
        device=torch.device("cuda"),
    )

    assert [name for name, _ in events] == [
        "log",
        "build",
        "warmup",
        "advance",
        "drain",
    ]
    assert (
        "persistent-state profile execution" in cast(tuple[str, ...], events[0][1])[0]
    )
    warmup_kwargs = cast(dict[str, object], events[2][1])
    assert warmup_kwargs["channel_pools"] == []
    assert warmup_kwargs["h_pool"] is invocation.pools.predictor_h
    assert warmup_kwargs["endpoint_book_pool"] is invocation.pools.endpoint_book
    # The patched transaction records its positional tuple and keyword dict.
    advance_args, advance_kwargs = cast(
        tuple[tuple[object, ...], dict[str, object]], events[3][1]
    )
    assert advance_args[:4] == (
        model.core,
        invocation.input_ids,
        invocation.inputs_embeds,
        invocation.plan,
    )
    assert advance_kwargs["capture"] is False
    assert advance_kwargs["memory_profile"] is True
    assert advance_kwargs["commit_sink"] is None
    assert advance_kwargs["decode_resolver"] is model._decode_resolver
    assert advance_kwargs["encoder_transition"] is model._encoder_execution.transition


def test_profile_execution_bypasses_strict_served_graph_resolver_before_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-PERF-004
    profile = _profile_module()
    invocation = SimpleNamespace(
        input_ids=torch.zeros(1, dtype=torch.long),
        inputs_embeds=torch.zeros(1, 7),
        plan=object(),
        pools=SimpleNamespace(
            channel=(),
            convolution=(),
            valid_length=(),
            predictor_h=object(),
            predictor_c=object(),
            replay_queue=object(),
            replay_book=object(),
            frontend_raw=object(),
            frontend_mel=object(),
            frontend_counters=object(),
            endpoint_history=object(),
            endpoint_book=object(),
        ),
    )
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        profile,
        "build_profile_invocation",
        lambda *args, **kwargs: invocation,
    )
    monkeypatch.setattr(
        profile,
        "advance_model_rows",
        lambda *args, **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(profile, "consume_batch_stats", lambda: None)
    model = _no_state_model()
    object.__setattr__(
        model,
        "config",
        _config(decode_dispatch_arm="dense-graphed"),
    )

    def served_resolver(request: Any) -> Any:
        raise ValueError("dense-graphed dispatch requires regional graph coverage")

    object.__setattr__(model, "_decode_resolver", served_resolver)
    object.__setattr__(model, "_decode_graph_binding", object())

    profile.run_persistent_state_profile(
        model,
        num_rows=1,
        device=torch.device("cpu"),
    )

    from vllm.models.nemotron_asr.advance import (
        DecodeRequest,
    )
    from vllm.models.nemotron_asr.rnnt import (
        decode_dense_masked_frames,
    )

    resolved = captured["decode_resolver"](
        DecodeRequest(
            geometry=4,
            execution_batch_size=1,
            graph_covers_decode=False,
            ready_decode_buckets=1,
            memory_profile=True,
        )
    )
    assert resolved.arm == "dense-eager"
    assert resolved.decode_fn is decode_dense_masked_frames
    assert resolved.override_reason == "pre-capture-memory-profile"
    assert captured["memory_profile"] is True


@pytest.mark.parametrize(
    "encoder_arm", ["compiled-static", "dense-graphed", "eager-graphed"]
)
def test_compiled_profile_uses_largest_admitted_geometry(
    monkeypatch: pytest.MonkeyPatch,
    encoder_arm: str,
) -> None:
    # @spec PORT-PERF-009, PORT-PERF-011
    profile = _profile_module()
    captured: dict[str, Any] = {}

    def stop_after_build(
        _config: Any,
        *,
        num_rows: int,
        device: torch.device,
        geometry_id: int,
    ) -> Any:
        captured.update(
            num_rows=num_rows,
            device=device,
            geometry_id=geometry_id,
        )
        raise RuntimeError("stop after profile geometry selection")

    monkeypatch.setattr(profile, "build_profile_invocation", stop_after_build)
    monkeypatch.setattr(profile, "consume_batch_stats", lambda: None)
    model = _no_state_model()
    object.__setattr__(
        model,
        "_encoder_execution",
        SimpleNamespace(
            arm=encoder_arm,
            transition=object(),
            warmup_geometries=(2, 0),
        ),
    )

    with pytest.raises(RuntimeError, match="stop after profile geometry selection"):
        profile.run_persistent_state_profile(
            model,
            num_rows=4,
            device=torch.device("cpu"),
        )

    assert captured == {
        "num_rows": 4,
        "device": torch.device("cpu"),
        "geometry_id": 2,
    }


def test_profile_execution_drains_stats_when_the_transition_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-MIG-003 / PORT-MIG-005
    profile = _profile_module()
    invocation = SimpleNamespace(
        input_ids=torch.zeros(1, dtype=torch.long),
        inputs_embeds=torch.zeros(1, 7),
        plan=object(),
        pools=SimpleNamespace(
            channel=(),
            convolution=(),
            valid_length=(),
            predictor_h=object(),
            predictor_c=object(),
            replay_queue=object(),
            replay_book=object(),
            frontend_raw=object(),
            frontend_mel=object(),
            frontend_counters=object(),
            endpoint_history=object(),
            endpoint_book=object(),
        ),
    )
    drained: list[bool] = []
    monkeypatch.setattr(
        profile,
        "build_profile_invocation",
        lambda *args, **kwargs: invocation,
    )
    monkeypatch.setattr(
        profile,
        "advance_model_rows",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    monkeypatch.setattr(
        profile,
        "consume_batch_stats",
        lambda: drained.append(True),
    )

    with pytest.raises(RuntimeError, match="failed"):
        profile.run_persistent_state_profile(
            _no_state_model(),
            num_rows=1,
            device=torch.device("cpu"),
        )

    assert drained == [True]


def test_profile_execution_drains_stats_on_empty_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-MIG-003 / PORT-MIG-005
    profile = _profile_module()
    drained: list[bool] = []
    monkeypatch.setattr(
        profile,
        "consume_batch_stats",
        lambda: drained.append(True),
    )

    with pytest.raises(ValueError, match="row count must be positive"):
        profile.run_persistent_state_profile(
            _no_state_model(),
            num_rows=0,
            device=torch.device("cpu"),
        )

    assert drained == [True]


@pytest.mark.parametrize(
    "encoder_arm", ["compiled-static", "dense-graphed", "eager-graphed"]
)
def test_static_encoder_warmup_is_product_owned_cartesian_execution(
    monkeypatch: pytest.MonkeyPatch,
    encoder_arm: str,
) -> None:
    # @spec PORT-PERF-009, PORT-PERF-011
    profile = _profile_module()
    invoked: list[tuple[int, int]] = []
    warmed: list[tuple[int, int]] = []
    sealed: list[tuple[tuple[int, int], ...]] = []

    class Execution:
        arm = encoder_arm
        warmup_geometries = (0, 2, 4)
        warmup_populations = (1, 3)

        def warmup_domain(
            self, *, expected_cells: tuple[tuple[int, int], ...], invoke: Any
        ) -> None:
            for geometry, population in expected_cells:
                warmed.append((geometry, population))
                invoke(geometry, population)
            sealed.append(expected_cells)

    model = _no_state_model()
    object.__setattr__(
        model,
        "config",
        _config(),
    )
    object.__setattr__(
        model.config,
        "supported_num_lookahead_tokens",
        [0, 3, 13],
    )
    object.__setattr__(model, "_encoder_execution", Execution())
    monkeypatch.setattr(
        profile,
        "run_persistent_state_profile",
        lambda _model, *, num_rows, device, geometry_id=None: invoked.append(
            (int(geometry_id), num_rows)
        ),
    )

    profile.warmup_static_encoder_execution(
        model,
        device=torch.device("cpu"),
    )

    expected = (
        (0, 1),
        (0, 3),
        (2, 1),
        (2, 3),
        (4, 1),
        (4, 3),
    )
    assert invoked == list(expected)
    assert warmed == list(expected)
    assert sealed == [expected]


@pytest.mark.parametrize("is_profile", [True, False])
def test_forward_consumes_no_state_projection_before_resident_authorities(
    monkeypatch: pytest.MonkeyPatch,
    is_profile: bool,
) -> None:
    # @spec PORT-MIG-005 / PORT-STATE-003 / PORT-STATE-007
    profile = _profile_module()
    model_cls = _model_class()
    model = _no_state_model()
    calls: list[tuple[int, torch.device]] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail(
            "PORT-MIG-005 dummy touched resident projection authority",
            pytrace=False,
        )

    monkeypatch.setattr(model_cls, "_stage_v2_projection", forbidden)
    monkeypatch.setattr(model_cls, "_state_pools", forbidden)
    monkeypatch.setattr(
        profile,
        "run_persistent_state_profile",
        lambda value, *, num_rows, device: calls.append((num_rows, device)),
    )
    input_ids = torch.zeros(7, dtype=torch.long)
    inputs_embeds = torch.zeros(7, 11, dtype=torch.float32)

    output = model.forward(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        persistent_state_projection=_projection(is_profile=is_profile),
    )

    assert output.shape == inputs_embeds.shape
    assert output.dtype == inputs_embeds.dtype
    assert output.device == inputs_embeds.device
    assert calls == ([(2, torch.device("cpu"))] if is_profile else [])


def test_profile_forward_accepts_mrv2_embedding_only_dummy_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-MIG-005 / PORT-STATE-007
    profile = _profile_module()
    model_cls = _model_class()
    model = _no_state_model()
    calls: list[tuple[int, torch.device]] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail(
            "PORT-MIG-005 MRv2 profile touched resident authority",
            pytrace=False,
        )

    monkeypatch.setattr(model_cls, "_stage_v2_projection", forbidden)
    monkeypatch.setattr(model_cls, "_state_pools", forbidden)
    monkeypatch.setattr(
        profile,
        "run_persistent_state_profile",
        lambda value, *, num_rows, device: calls.append((num_rows, device)),
    )
    inputs_embeds = torch.zeros(7, 11, dtype=torch.float32)

    output = model.forward(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        persistent_state_projection=_projection(is_profile=True),
    )

    assert output.shape == inputs_embeds.shape
    assert output.dtype == inputs_embeds.dtype
    assert output.device == inputs_embeds.device
    assert calls == [(2, torch.device("cpu"))]


def test_prepare_inputs_snapshot_flows_directly_into_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # @spec PORT-MIG-005 / PORT-STATE-003 / PORT-STATE-007
    profile = _profile_module()
    state_module = importlib.import_module("vllm.models.nemotron_asr.model_state_v2")
    state = object.__new__(state_module.NemotronASRModelState)
    state.prepare_step(
        SimpleNamespace(finished_req_ids=set()), dummy_run=True, is_profile=True
    )

    model_cls = _model_class()
    model = _no_state_model()
    calls: list[tuple[int, torch.device]] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail(
            "PORT-MIG-005 composed dummy touched resident projection authority",
            pytrace=False,
        )

    monkeypatch.setattr(model_cls, "_stage_v2_projection", forbidden)
    monkeypatch.setattr(model_cls, "_state_pools", forbidden)
    monkeypatch.setattr(
        profile,
        "run_persistent_state_profile",
        lambda value, *, num_rows, device: calls.append((num_rows, device)),
    )
    input_ids = torch.zeros(7, dtype=torch.long)
    inputs_embeds = torch.zeros(7, 11, dtype=torch.float32)

    try:
        prepared = state.prepare_inputs(
            SimpleNamespace(req_ids=["opaque-dummy"]),
            object(),
        )
        output = model.forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            persistent_state_projection=prepared["persistent_state_projection"],
        )
    finally:
        state._snapshot = None

    assert output.shape == inputs_embeds.shape
    assert output.dtype == inputs_embeds.dtype
    assert output.device == inputs_embeds.device
    assert calls == [(1, torch.device("cpu"))]


@pytest.mark.parametrize("fail_selected_profile", [False, True])
def test_chunk_inventory_profiles_cells_and_maximum_before_state_allocation(
    monkeypatch, fail_selected_profile
):
    profile = _profile_module()
    model = _no_state_model()
    events = []
    parameter = torch.zeros(1)
    execution = SimpleNamespace(
        ready=False, _chunk_graph_cells=frozenset({(1, 31), (1, 63)})
    )
    decoder = SimpleNamespace(captured_keys=())
    chunk = SimpleNamespace(ready=False)

    def encoder_warmup(value, *, device):
        assert value is model and device == parameter.device
        events.append("encoder")

    def decoder_warmup(device, dtype):
        assert not execution.ready
        events.append("decoder")
        decoder.captured_keys = tuple((1, n) for n in (1, 2, 4, 8, 16, 32, 64, 128))

    def chunk_warmup(device):
        assert len(decoder.captured_keys) == 8
        events.append("chunk")
        execution.ready = chunk.ready = True

    def discard():
        events.append("discard")
        execution.ready = chunk.ready = False

    def final_profile(value, *, num_rows, device, geometry_id, ready_domain):
        assert value is model and ready_domain and execution.ready and chunk.ready
        assert geometry_id == 1 and device == parameter.device
        events.append(num_rows)
        if fail_selected_profile:
            raise RuntimeError("selected profile failed")

    execution._discard = discard
    decoder.warmup = decoder_warmup
    chunk.warmup = chunk_warmup
    for name, value in {
        "config": SimpleNamespace(
            supported_num_lookahead_tokens=[1], decode_dispatch_arm="dense-graphed"
        ),
        "core": SimpleNamespace(
            encoder=SimpleNamespace(parameters=lambda: iter([parameter])),
            policy=FP32_BRINGUP,
        ),
        "_max_num_seqs": 128,
        "_encoder_execution": execution,
        "_decode_graph_binding": decoder,
        "_chunk_bucket_binding": chunk,
        "_execution_memory_prepared": False,
        "_execution_memory_failed": False,
        "_state_pools": lambda: pytest.fail("startup touched resident state"),
    }.items():
        object.__setattr__(model, name, value)
    monkeypatch.setattr(profile, "warmup_static_encoder_execution", encoder_warmup)
    monkeypatch.setattr(profile, "run_persistent_state_profile", final_profile)
    if fail_selected_profile:
        with pytest.raises(RuntimeError, match="selected profile"):
            model.prepare_execution_memory()
        assert events == ["encoder", "decoder", "chunk", 31, "discard"]
        assert model._execution_memory_failed and not model._execution_memory_prepared
        assert not execution.ready and not chunk.ready
    else:
        model.prepare_execution_memory()
        assert events == ["encoder", "decoder", "chunk", 31, 63, 128]
        assert model._execution_memory_prepared


def test_chunk_final_profile_uses_sealed_decoder_tier_and_live_graph_route(monkeypatch):
    profile = _profile_module()
    model = _no_state_model()
    config = _config(decode_dispatch_arm="dense-graphed")
    for name, value in {
        "d_model": 8,
        "n_layers": 2,
        "conv_kernel": 3,
        "att_context_left": 4,
        "pred_hidden": 4,
        "pred_rnn_layers": 1,
        "n_mels": 2,
    }.items():
        setattr(config, name, value)
    execution = SimpleNamespace(
        arm="eager-graphed",
        cell_active=False,
        warmup_geometries=(1,),
        transition=object(),
        profile_ready_cell=lambda *, geometry, population, invoke: invoke(),
    )
    chunk_binding, decode_resolver = object(), object()
    for name, value in {
        "config": config,
        "_encoder_execution": execution,
        "_decode_graph_binding": SimpleNamespace(execution_tier=lambda _n: 32),
        "_chunk_bucket_binding": chunk_binding,
        "_decode_resolver": decode_resolver,
    }.items():
        object.__setattr__(model, name, value)
    calls = []

    def advance(_core, _ids, _env, plan, **kwargs):
        assert plan.execution_tier == 32 and plan.num_prefills == 31
        assert (
            kwargs["graph_covers_decode"] is True and kwargs["memory_profile"] is False
        )
        assert (
            kwargs["bucket_transition"] is chunk_binding
            and kwargs["decode_resolver"] is decode_resolver
        )
        calls.append(True)
        return torch.zeros(31)

    monkeypatch.setattr(profile, "advance_model_rows", advance)
    profile.run_persistent_state_profile(
        model, num_rows=31, device=torch.device("cpu"), geometry_id=1, ready_domain=True
    )
    assert calls == [True]
