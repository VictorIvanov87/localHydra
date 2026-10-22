"""Health/models endpoints, error envelopes, privacy of errors, config validation, CLI refusal."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from localhydra import __main__ as cli
from localhydra.app import create_app
from localhydra.contracts import registry
from localhydra.contracts.api import HealthResponse
from localhydra.contracts.config import AppConfig
from localhydra.contracts.errors import ERROR_HTTP_STATUS, ErrorCode, ErrorEnvelope
from localhydra.contracts.models import (
    ModelProfile,
    ModelsResponse,
    resolve_model_statuses,
)
from localhydra.contracts.runs import CreateRunRequest
from localhydra.settings import ConfigError, load_settings
from tests.conftest import EXAMPLE_CONFIG, ROOT, SCHEMAS


# ---- endpoints ----------------------------------------------------------------------------


def test_health(app_client):
    r = app_client.get("/api/health")
    assert r.status_code == 200
    h = HealthResponse.model_validate(r.json())
    assert h.local_only is True and h.bind_host == "127.0.0.1" and h.models_configured == 0
    assert h.ollama_target == "http://127.0.0.1:11434"
    assert r.headers["x-request-id"] and r.headers["cache-control"] == "no-store"


def test_models_unconfigured_is_empty_not_guessed(app_client):
    m = ModelsResponse.model_validate(app_client.get("/api/models").json())
    assert m.configured is False and m.models == [] and m.auto_download is False


def test_models_configured_but_not_probed_are_unknown():
    client = TestClient(create_app(load_settings(EXAMPLE_CONFIG)), base_url="http://127.0.0.1")
    m = ModelsResponse.model_validate(client.get("/api/models").json())
    assert m.configured and len(m.models) == 4 and m.provider_probe.value == "not_probed"
    assert {s.availability.value for s in m.models} == {"unknown"}
    assert {s.loaded_state.value for s in m.models} == {"unknown"}
    assert client.get("/api/health").json()["models_configured"] == 4


def test_unknown_route_uses_error_envelope(app_client):
    r = app_client.get("/api/nope")
    assert r.status_code == 404
    env = ErrorEnvelope.model_validate(r.json())
    assert env.error.code is ErrorCode.not_found and env.error.request_id


def test_wrong_method_uses_error_envelope(app_client):
    r = app_client.post("/api/health")
    assert r.status_code == 405
    assert ErrorEnvelope.model_validate(r.json()).error.code is ErrorCode.bad_request


def test_no_cdn_backed_docs_pages(app_client):
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert app_client.get(path).status_code == 404


@pytest.mark.parametrize(
    "host,ok",
    [("127.0.0.1:8765", True), ("localhost:8765", True), ("[::1]:8765", True),
     ("evil.example.com", False), ("127.0.0.1.evil.com:8765", False), ("", False)],
)
def test_host_header_must_be_loopback(app_client, host, ok):
    r = app_client.get("/api/health", headers={"Host": host})
    if ok:
        assert r.status_code == 200
    else:
        assert r.status_code == 400
        assert ErrorEnvelope.model_validate(r.json()).error.code is ErrorCode.host_not_allowed


def test_validation_errors_never_echo_input():
    app = create_app()

    @app.post("/t")
    async def t(body: CreateRunRequest) -> dict[str, str]:  # pragma: no cover - never reached
        return {}

    client = TestClient(app, base_url="http://127.0.0.1")
    secret = "TOP-SECRET-PROMPT-TEXT"
    r = client.post("/t", json={"messages": [{"role": "user", "content": secret}], "routing_mode": "manual",
                                "unexpected": secret})
    assert r.status_code == 422
    assert secret not in r.text
    env = ErrorEnvelope.model_validate(r.json())
    assert env.error.code is ErrorCode.validation_error
    assert env.error.details and env.error.details["errors"]
    assert set(env.error.details["errors"][0]) == {"loc", "type"}


def test_unhandled_error_is_generic_and_leaks_nothing():
    app = create_app()

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("secret-internal-detail")

    r = TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False).get("/boom")
    assert r.status_code == 500 and "secret-internal-detail" not in r.text
    assert ErrorEnvelope.model_validate(r.json()).error.code is ErrorCode.internal_error


def test_every_error_code_has_an_http_status():
    assert set(ERROR_HTTP_STATUS) == set(ErrorCode)


# ---- model availability: missing is unavailable, never downloaded -----------------------------


def _profiles() -> list[ModelProfile]:
    cfg = AppConfig.model_validate(json.loads(EXAMPLE_CONFIG.read_text()))
    return list(cfg.models)


def test_missing_models_are_unavailable_not_pulled():
    ps = _profiles()
    installed = [ps[0].tag, ps[1].tag]
    st = {s.profile.id: s for s in resolve_model_statuses(ps, installed, loaded_tags=[ps[0].tag])}
    assert st[ps[0].id].availability.value == "available" and st[ps[0].id].loaded_state.value == "loaded"
    assert st[ps[1].id].availability.value == "available" and st[ps[1].id].loaded_state.value == "not_loaded"
    for p in ps[2:]:
        assert st[p.id].availability.value == "unavailable"
        assert st[p.id].availability_reason.value == "not_installed"  # type: ignore[union-attr]


def test_matching_is_exact_not_prefix_or_latest():
    ps = _profiles()
    name = ps[0].tag.split(":")[0]
    for near in (name, name + ":latest", ps[0].tag + "-x", ps[0].tag.upper()):
        st = resolve_model_statuses(ps[:1], [near])[0]
        assert st.availability.value == "unavailable", near


def test_probe_failure_is_unknown_never_optimistic():
    st = resolve_model_statuses(_profiles(), None)
    assert {s.availability.value for s in st} == {"unknown"}


def test_disabled_profile_is_unavailable_even_if_installed():
    p = _profiles()[0].model_copy(update={"enabled": False})
    st = resolve_model_statuses([p], [p.tag])[0]
    assert st.availability.value == "unavailable" and st.availability_reason.value == "disabled"  # type: ignore[union-attr]


def test_config_has_no_download_switch_anywhere():
    def walk(model):
        for name, f in model.model_fields.items():
            assert not any(w in name.lower() for w in ("pull", "download", "auto_install")), name
            ann = f.annotation
            sub = getattr(ann, "model_fields", None)
            if sub:
                walk(ann)

    walk(AppConfig)


# ---- config loading -----------------------------------------------------------------------


def test_example_config_loads_with_4_exact_tags():
    s = load_settings(EXAMPLE_CONFIG)
    assert s.configured and len(s.models) == 4 and all(":" in m.tag for m in s.models)


def test_no_config_is_unconfigured_not_defaulted():
    s = load_settings(None)
    assert not s.configured and s.models == ()


@pytest.mark.parametrize("n,ok", [(1, False), (3, False), (4, True), (5, True), (6, False)])
def test_model_count_bounds(tmp_path, n, ok):
    cfg = json.loads(EXAMPLE_CONFIG.read_text())
    base = cfg["models"][0]
    cfg["models"] = [{**base, "id": f"m{i}", "tag": f"example/m{i}:synthetic"} for i in range(n)]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg))
    if ok:
        assert len(load_settings(p).models) == n
    else:
        with pytest.raises(ConfigError):
            load_settings(p)


def test_duplicate_tags_and_ids_rejected(tmp_path):
    cfg = json.loads(EXAMPLE_CONFIG.read_text())
    cfg["models"][1]["tag"] = cfg["models"][0]["tag"].replace("synthetic", "SYNTHETIC")
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(ConfigError, match="unique"):
        load_settings(p)


def test_missing_and_malformed_config_files(tmp_path):
    with pytest.raises(ConfigError):
        load_settings(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ConfigError):
        load_settings(bad)


# ---- CLI never starts a non-loopback server -----------------------------------------------------


def test_cli_refuses_non_loopback_host(monkeypatch, capsys):
    called = []
    monkeypatch.setattr(cli.uvicorn, "run", lambda *a, **k: called.append((a, k)))
    assert cli.main(["--host", "0.0.0.0"]) == 2
    assert not called and "loopback" in capsys.readouterr().err


def test_cli_refuses_invalid_config(monkeypatch, tmp_path, capsys):
    called = []
    monkeypatch.setattr(cli.uvicorn, "run", lambda *a, **k: called.append(1))
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"models": []}))
    assert cli.main(["--config", str(p)]) == 2
    assert not called and "configuration error" in capsys.readouterr().err


def test_cli_binds_loopback_by_default(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **k: seen.update(k))
    assert cli.main([]) == 0
    assert seen["host"] == "127.0.0.1" and seen["port"] == 8765


# ---- drift guards -------------------------------------------------------------------------


def test_committed_schemas_match_code():
    assert registry.main(["--check", "--out", str(SCHEMAS)]) == 0


def test_manifest_hashes_match_files():
    import hashlib

    manifest = json.loads((SCHEMAS / "MANIFEST.json").read_text())
    from localhydra.contracts.common import CONTRACT_VERSION

    assert manifest["contract_version"] == CONTRACT_VERSION
    for name, meta in manifest["schemas"].items():
        assert hashlib.sha256((SCHEMAS / meta["file"]).read_bytes()).hexdigest() == meta["sha256"], name


def test_committed_fixtures_match_builder():
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_fixtures", ROOT / "contracts" / "tools" / "build_fixtures.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    assert mod.main(["--check"]) == 0


def test_every_endpoint_references_registered_schemas():
    for e in registry.api.ENDPOINTS:
        for name in (e.request_schema, e.response_schema):
            assert name is None or name in registry.REGISTRY, (e.path, name)
    paths = {(e.method, e.path) for e in registry.api.ENDPOINTS}
    for required in [("GET", "/api/health"), ("GET", "/api/models"), ("POST", "/api/runs"), ("GET", "/api/runs"),
                     ("GET", "/api/runs/{run_id}"), ("GET", "/api/runs/{run_id}/events"),
                     ("POST", "/api/runs/{run_id}/cancel"), ("GET", "/api/conversations/{conversation_id}"),
                     ("POST", "/api/conversations/{conversation_id}/reset")]:
        assert required in paths


def test_only_health_and_models_are_implemented_here():
    impl = {(e.method, e.path) for e in registry.api.ENDPOINTS if e.implemented_by == "LH-001"}
    assert impl == {("GET", "/api/health"), ("GET", "/api/models")}
    routes = {(m, r.path) for r in create_app().routes for m in getattr(r, "methods", set()) if r.path.startswith("/api")}
    assert routes == impl


def test_source_has_no_cloud_or_pull_calls():
    src = Path(registry.__file__).resolve().parents[1]
    banned = ("api/pull", "ollama pull", "api.openai.com", "api.anthropic.com", "googleapis.com", "cdn.jsdelivr")
    for f in src.rglob("*.py"):
        text = f.read_text()
        for b in banned:
            assert b not in text, f"{f}: {b}"
