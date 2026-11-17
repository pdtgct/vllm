# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the `vllm.endpoint_plugins` framework (RFC #46565).

Uses the worked in repo example plugin (`vllm_add_dummy_endpoint_plugin`,
installed via `tests/plugins/vllm_add_dummy_endpoint_plugin`) exercising both
`EndpointPlugin` hooks against a fake `EngineClient`, unit tests for the
`load_endpoint_plugins` gating matrix and an e2e test that drives a real HTTP
request through the plugin's route.
"""

from argparse import Namespace
from typing import Any

import httpx
import pytest
from fastapi import APIRouter, FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDenialResponse
from vllm_add_dummy_endpoint_plugin import DummyAdminEndpointPlugin

from vllm.entrypoints.common.cli_args import make_arg_parser
from vllm.entrypoints.launchers.app import build_app
from vllm.plugins import load_endpoint_plugins
from vllm.plugins.endpoint_plugins.interface import (
    EndpointPlugin,
    attach_endpoint_plugins,
    init_endpoint_plugins_state,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser


class _RaisingEndpointPlugin:
    """Factory that raises to exercise the "instantiation fails" path."""

    name = "raising_endpoint_plugin"
    required_tasks = None

    def __init__(self):
        raise RuntimeError("boom")


class _ShadowEndpointPlugin:
    name = "shadow_endpoint_plugin"
    required_tasks = None

    def __init__(self, path: str, include_in_schema: bool):
        self.path = path
        self.include_in_schema = include_in_schema

    def attach_router(self, app: FastAPI) -> None:
        assert str(app.url_path_for("show_version")) == "/version"
        app.openapi()

        @app.get(self.path, status_code=209, include_in_schema=self.include_in_schema)
        async def plugin_version(plugin_arg: str) -> dict[str, str]:
            return {"served_by": plugin_arg}

    async def init_state(self, engine_client, state, args) -> None:
        pass


class _RouterEndpointPlugin(DummyAdminEndpointPlugin):
    def __init__(self, router: APIRouter):
        self.router = router

    def attach_router(self, app: FastAPI) -> None:
        app.include_router(self.router)


class _AuthenticatedRouterPlugin(_RouterEndpointPlugin):
    name = "authenticated_endpoint_plugin"

    def __init__(self, router: APIRouter, prefixes: Any):
        super().__init__(router)
        self.authenticated_path_prefixes = prefixes


class _FakeEngineClient:
    """Minimal stand in exercising `collective_rpc`. Not a real engine."""

    def __init__(self, rpc_result: Any = None):
        self.rpc_result = rpc_result
        self.rpc_calls: list[tuple[str, tuple, dict]] = []

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.rpc_calls.append((method, args, kwargs or {}))
        return self.rpc_result


def _build_args(*argv: str) -> Namespace:
    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers()
    serve_parser = subparsers.add_parser("serve")
    make_arg_parser(serve_parser)
    return serve_parser.parse_args(list(argv))


def _fake_loader(factories: dict[str, Any]):
    def _load_plugins_by_group(group: str) -> dict[str, Any]:
        assert group == "vllm.endpoint_plugins"
        return factories

    return _load_plugins_by_group


def test_dummy_plugin_satisfies_protocol():
    assert isinstance(DummyAdminEndpointPlugin(), EndpointPlugin)


def test_no_plugins_loaded_when_allowlist_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)

    assert load_endpoint_plugins(("generate",)) == []


def test_no_plugins_loaded_when_allowlist_is_empty_string(
    monkeypatch: pytest.MonkeyPatch,
):
    """`VLLM_PLUGINS=""` parses to `[""]`, not `None` (see `vllm.envs`), so it
    must be treated as a (non strict) allowlist matching no plugin name, not
    as "unset"."""
    monkeypatch.setenv("VLLM_PLUGINS", "")

    assert load_endpoint_plugins(("generate",)) == []


def test_plugin_loaded_when_allowlisted_and_task_matches(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_PLUGINS", "dummy_admin_endpoint_plugin")

    plugins = load_endpoint_plugins(("generate",))

    assert len(plugins) == 1
    assert isinstance(plugins[0], DummyAdminEndpointPlugin)


def test_plugin_skipped_when_required_tasks_miss(monkeypatch: pytest.MonkeyPatch):
    class _GenerateOnlyPlugin(DummyAdminEndpointPlugin):
        required_tasks = ("generate",)

    monkeypatch.setenv("VLLM_PLUGINS", "dummy_admin_endpoint_plugin")
    monkeypatch.setattr(
        "vllm.plugins.load_plugins_by_group",
        _fake_loader({"dummy_admin_endpoint_plugin": _GenerateOnlyPlugin}),
    )

    assert load_endpoint_plugins(("embed",)) == []
    assert len(load_endpoint_plugins(("generate",))) == 1


def test_plugin_loaded_when_required_tasks_is_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_PLUGINS", "dummy_admin_endpoint_plugin")

    assert len(load_endpoint_plugins(supported_tasks=None)) == 1


def test_plugin_skipped_when_required_tasks_set_but_supported_tasks_none(
    monkeypatch: pytest.MonkeyPatch,
):
    class _GenerateOnlyPlugin(DummyAdminEndpointPlugin):
        required_tasks = ("generate",)

    monkeypatch.setenv("VLLM_PLUGINS", "dummy_admin_endpoint_plugin")
    monkeypatch.setattr(
        "vllm.plugins.load_plugins_by_group",
        _fake_loader({"dummy_admin_endpoint_plugin": _GenerateOnlyPlugin}),
    )

    assert load_endpoint_plugins(supported_tasks=None) == []


def test_factory_raising_is_logged_and_skipped(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(
        "VLLM_PLUGINS", "raising_endpoint_plugin,dummy_admin_endpoint_plugin"
    )
    monkeypatch.setattr(
        "vllm.plugins.load_plugins_by_group",
        _fake_loader(
            {
                "raising_endpoint_plugin": _RaisingEndpointPlugin,
                "dummy_admin_endpoint_plugin": DummyAdminEndpointPlugin,
            }
        ),
    )

    plugins = load_endpoint_plugins(("generate",))

    assert len(plugins) == 1
    assert isinstance(plugins[0], DummyAdminEndpointPlugin)


def test_attach_is_noop_when_nothing_discovered(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)

    app = FastAPI()
    attach_endpoint_plugins(app, ("generate",))

    assert app.state.endpoint_plugins == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/version", "/metrics"])
@pytest.mark.parametrize("include_in_schema", [True, False])
async def test_endpoint_plugin_override_preserves_hook_and_schema(
    monkeypatch: pytest.MonkeyPatch, path: str, include_in_schema: bool
):
    """Hooks see core routes; overrides win over both HTTP routes and mounts."""
    args = _build_args()
    monkeypatch.setenv("VLLM_PLUGINS", "shadow_endpoint_plugin")
    monkeypatch.setattr(
        "vllm.plugins.load_plugins_by_group",
        _fake_loader(
            {
                "shadow_endpoint_plugin": lambda: _ShadowEndpointPlugin(
                    path, include_in_schema
                )
            }
        ),
    )
    app = build_app(args, supported_tasks=())

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(path, params={"plugin_arg": "plugin"})
        schema = (await client.get("/openapi.json")).json()

    assert response.status_code == 209
    assert response.json() == {"served_by": "plugin"}
    if include_in_schema:
        operation = schema["paths"][path]["get"]
        assert "209" in operation["responses"]
        assert "200" not in operation["responses"]
        assert operation["parameters"][0]["name"] == "plugin_arg"
        assert operation["parameters"][0]["required"] is True
        response_schema = operation["responses"]["209"]["content"]["application/json"][
            "schema"
        ]
        assert response_schema["additionalProperties"] == {"type": "string"}
    else:
        assert path not in schema["paths"]


@pytest.mark.asyncio
async def test_endpoint_plugin_override_preserves_other_methods(
    monkeypatch: pytest.MonkeyPatch,
):
    """Overriding GET preserves POST's validation and the shared core route."""
    core = APIRouter()

    @core.api_route("/shared", methods=["GET", "POST"], status_code=201)
    async def original(value: int) -> dict[str, int]:
        return {"value": value}

    plugin = APIRouter()

    @plugin.get("/shared", status_code=202)
    async def replacement():
        return {"served_by": "plugin"}

    app = FastAPI()
    app.include_router(core)
    original_route = app.routes[-1]
    monkeypatch.setattr(
        "vllm.plugins.load_endpoint_plugins",
        lambda supported_tasks: [_RouterEndpointPlugin(plugin)],
    )
    attach_endpoint_plugins(app, ())

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/shared")
        post_response = await client.post("/shared", params={"value": 7})
        invalid_response = await client.post("/shared", params={"value": "invalid"})

    assert response.status_code == 202
    assert response.json() == {"served_by": "plugin"}
    assert post_response.status_code == 201
    assert post_response.json() == {"value": 7}
    assert invalid_response.status_code == 422
    assert original_route.methods == {"GET", "POST"}
    assert core.routes[0].methods == {"GET", "POST"}
    operations = app.openapi()["paths"]["/shared"]
    assert "202" in operations["get"]["responses"]
    assert "201" in operations["post"]["responses"]


@pytest.mark.asyncio
async def test_later_endpoint_plugin_overrides_same_operation(
    monkeypatch: pytest.MonkeyPatch,
):
    """The last registered override wins while unrelated plugin routes survive."""
    first = APIRouter()

    @first.get("/shared", status_code=201)
    @first.get("/plugins/first")
    async def first_handler():
        return {"served_by": "first"}

    second = APIRouter()

    @second.get("/shared", status_code=202)
    async def second_handler():
        return {"served_by": "second"}

    monkeypatch.setattr(
        "vllm.plugins.load_endpoint_plugins",
        lambda supported_tasks: [
            _RouterEndpointPlugin(first),
            _RouterEndpointPlugin(second),
        ],
    )
    app = FastAPI()
    attach_endpoint_plugins(app, ())

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/shared")
        unrelated = await client.get("/plugins/first")
        unsupported = await client.delete("/shared")

    assert response.status_code == 202
    assert response.json() == {"served_by": "second"}
    assert unrelated.status_code == 200
    assert unrelated.json() == {"served_by": "first"}
    assert unsupported.status_code == 405
    responses = app.openapi()["paths"]["/shared"]["get"]["responses"]
    assert "202" in responses
    assert "201" not in responses


@pytest.mark.asyncio
async def test_init_state_is_noop_without_phase_a(monkeypatch: pytest.MonkeyPatch):
    """`init_app_state` callers that never ran `build_app` (e.g.
    `run_batch.py`, which builds a bare `State()`) must not crash just
    because `state.endpoint_plugins` was never set."""
    from starlette.datastructures import State

    monkeypatch.setenv("VLLM_PLUGINS", "dummy_admin_endpoint_plugin")

    state = State()
    await init_endpoint_plugins_state(_FakeEngineClient(), state, _build_args())

    assert not hasattr(state, "dummy_engine_client")


@pytest.mark.asyncio
async def test_render_server_attaches_endpoint_plugins_with_no_engine_client(
    monkeypatch: pytest.MonkeyPatch,
):
    """The CPU only render server has no `EngineClient` but a plugin eligible
    for the `render` task (`required_tasks` is `None` or includes `"render"`)
    still gets its routes attached at Phase A. Phase B passes `None` for
    `engine_client` and it's up to the plugin to handle that."""
    monkeypatch.setenv("VLLM_PLUGINS", "dummy_admin_endpoint_plugin")

    args = _build_args()
    app = build_app(args, ("render",))

    assert len(app.state.endpoint_plugins) == 1
    assert any(
        getattr(route, "path", None) == "/v1/admin/scheduler_config"
        for route in app.routes
    )

    await init_endpoint_plugins_state(None, app.state, args)

    assert app.state.dummy_engine_client is None

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/admin/scheduler_config")

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_endpoint_plugin_end_to_end(monkeypatch: pytest.MonkeyPatch):
    """Phase A (attach) + Phase B (init) wired through `build_app` then
    exercised with a real HTTP request against the worked example plugin."""
    monkeypatch.setenv("VLLM_PLUGINS", "dummy_admin_endpoint_plugin")

    args = _build_args()
    app = build_app(args, supported_tasks=())

    assert len(app.state.endpoint_plugins) == 1
    assert any(
        getattr(route, "path", None) == "/v1/admin/scheduler_config"
        for route in app.routes
    )

    fake_engine_client = _FakeEngineClient(rpc_result=["cfg-a", "cfg-b"])
    await init_endpoint_plugins_state(fake_engine_client, app.state, args)

    assert app.state.dummy_engine_client is fake_engine_client

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/admin/scheduler_config")

    assert response.status_code == 200
    assert response.json() == {"scheduler_config": ["cfg-a", "cfg-b"]}
    assert fake_engine_client.rpc_calls == [("get_scheduler_config", (), {})]


def _build_authenticated_app(monkeypatch: pytest.MonkeyPatch, *argv: str) -> FastAPI:
    """App with one plugin guarding `/plugins/auth/api/` and one declaring
    nothing, both served under `/plugins/` outside core's guarded prefixes."""
    guarded = APIRouter()

    @guarded.get("/plugins/auth/api/data")
    @guarded.get("/plugins/auth/health")
    async def guarded_handler():
        return {"served_by": "authenticated"}

    @guarded.websocket("/plugins/auth/api/ws")
    async def guarded_ws(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_text("accepted")
        await websocket.close()

    legacy = APIRouter()

    @legacy.get("/plugins/legacy/data")
    async def legacy_handler():
        return {"served_by": "legacy"}

    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.setenv(
        "VLLM_PLUGINS", "authenticated_endpoint_plugin,dummy_admin_endpoint_plugin"
    )
    monkeypatch.setattr(
        "vllm.plugins.load_plugins_by_group",
        _fake_loader(
            {
                "authenticated_endpoint_plugin": lambda: _AuthenticatedRouterPlugin(
                    guarded, ("/plugins/auth/api/",)
                ),
                "dummy_admin_endpoint_plugin": lambda: _RouterEndpointPlugin(legacy),
            }
        ),
    )
    return build_app(_build_args(*argv), supported_tasks=())


_AUTH_HEADERS = {"Authorization": "Bearer secret"}


@pytest.mark.parametrize(
    ("method", "headers", "status_code"),
    [("GET", {}, 401), ("GET", _AUTH_HEADERS, 200), ("OPTIONS", {}, 405)],
)
def test_authenticated_path_prefix_requires_api_key(
    monkeypatch: pytest.MonkeyPatch, method: str, headers: dict, status_code: int
):
    """`--api-key` guards plugin declared prefixes like core's, OPTIONS exempt."""
    app = _build_authenticated_app(monkeypatch, "--api-key", "secret")

    response = TestClient(app).request(
        method, "/plugins/auth/api/data", headers=headers
    )

    assert response.status_code == status_code
    if status_code == 401:
        assert response.json() == {"error": "Unauthorized"}


def test_authenticated_path_prefix_guards_websocket(monkeypatch: pytest.MonkeyPatch):
    app = _build_authenticated_app(monkeypatch, "--api-key", "secret")
    client = TestClient(app)

    with (
        pytest.raises(WebSocketDenialResponse) as denied,
        client.websocket_connect("/plugins/auth/api/ws"),
    ):
        pass
    with client.websocket_connect(
        "/plugins/auth/api/ws", headers=_AUTH_HEADERS
    ) as websocket:
        accepted = websocket.receive_text()

    assert denied.value.status_code == 401
    assert accepted == "accepted"


@pytest.mark.parametrize("path", ["/plugins/auth/health", "/plugins/legacy/data"])
def test_plugin_routes_outside_authenticated_path_prefixes_stay_open(
    monkeypatch: pytest.MonkeyPatch, path: str
):
    """Undeclared paths, e.g. probes or plugins without the attribute, stay
    open even with an API key configured."""
    app = _build_authenticated_app(monkeypatch, "--api-key", "secret")

    assert TestClient(app).get(path).status_code == 200


def test_authenticated_path_prefixes_open_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
):
    app = _build_authenticated_app(monkeypatch)

    assert TestClient(app).get("/plugins/auth/api/data").status_code == 200


@pytest.mark.parametrize(
    "prefixes", [("/",), ("plugins/x/",), ["/plugins/x/"], (b"/plugins/x/",)]
)
def test_loader_refuses_invalid_authenticated_path_prefixes(
    monkeypatch: pytest.MonkeyPatch, prefixes: Any
):
    """A prefix of `/` would put `/health` and other probes behind the key."""
    monkeypatch.setenv("VLLM_PLUGINS", "authenticated_endpoint_plugin")
    monkeypatch.setattr(
        "vllm.plugins.load_plugins_by_group",
        _fake_loader(
            {
                "authenticated_endpoint_plugin": lambda: _AuthenticatedRouterPlugin(
                    APIRouter(), prefixes
                )
            }
        ),
    )

    with pytest.raises(
        ValueError, match="authenticated_endpoint_plugin.*not supported"
    ):
        load_endpoint_plugins(("generate",))
