"""LH-002: discovery and loaded-state inspection against SYNTHETIC fake servers."""

from __future__ import annotations

import json
import socket

import pytest

from localhydra.adapters import (
    OllamaProvider,
    ProviderError,
    ProviderErrorKind,
    ProviderTimeouts,
    probe_models,
)
from localhydra.contracts.config import AppConfig
from localhydra.contracts.models import Availability, LoadedState, ProbeStatus, UnavailableReason

from .conftest import EXAMPLE_CONFIG
from .provider_fakes import Conn, FakeOllama, anyio_backend  # noqa: F401

pytestmark = pytest.mark.anyio

PROFILES = AppConfig.model_validate(json.loads(EXAMPLE_CONFIG.read_text())).models  # synthetic tags
TAGS = [p.tag for p in PROFILES]


def tags_body(*names: str) -> dict:
    return {
        "models": [
            {"name": n, "model": n, "modified_at": "2026-01-01T00:00:00Z", "size": 1234, "digest": f"sha256:synthetic-{i}"}
            for i, n in enumerate(names)
        ]
    }


@pytest.fixture
async def fake():
    async with FakeOllama() as f:
        yield f


async def test_list_models_preserves_exact_tags_and_metadata(fake: FakeOllama) -> None:
    fake.json_route("GET", "/api/tags", tags_body(TAGS[0], "other/thing:Q4_K_M"))
    async with OllamaProvider(fake.base_url) as p:
        inv = await p.list_models()
    assert [m.tag for m in inv.models] == [TAGS[0], "other/thing:Q4_K_M"]
    assert inv.models[0].digest == "sha256:synthetic-0" and inv.models[0].size_bytes == 1234
    assert inv.models[0].modified_at == "2026-01-01T00:00:00Z"
    assert inv.observed_at.tzinfo is not None


@pytest.mark.parametrize("body", [{"models": []}, {"models": None}, {}])
async def test_empty_inventories(fake: FakeOllama, body: dict) -> None:
    fake.json_route("GET", "/api/tags", body)
    async with OllamaProvider(fake.base_url) as p:
        assert (await p.list_models()).models == ()


@pytest.mark.parametrize(
    "body",
    [{"models": "nope"}, {"models": [1]}, {"models": [{"size": 1}]}, {"models": [{"name": ""}]}, {"models": [{"name": 5}]}],
)
async def test_malformed_inventory_is_a_typed_error_not_a_partial_list(fake: FakeOllama, body: dict) -> None:
    fake.json_route("GET", "/api/tags", body)
    async with OllamaProvider(fake.base_url) as p:
        with pytest.raises(ProviderError) as ei:
            await p.list_models()
    assert ei.value.kind is ProviderErrorKind.malformed_response


@pytest.mark.parametrize("raw", [b"not json", b"[]", b"\xff"])
async def test_non_object_json_is_malformed(fake: FakeOllama, raw: bytes) -> None:
    async def h(c: Conn) -> None:
        await c.respond_bytes(200, raw)

    fake.on("GET", "/api/tags", h)
    async with OllamaProvider(fake.base_url) as p:
        with pytest.raises(ProviderError) as ei:
            await p.list_models()
    assert ei.value.kind is ProviderErrorKind.malformed_response


async def test_tags_http_error_is_typed_and_a_404_is_not_a_missing_model(fake: FakeOllama) -> None:
    fake.json_route("GET", "/api/tags", {"error": "x"}, status=404)
    async with OllamaProvider(fake.base_url) as p:
        with pytest.raises(ProviderError) as ei:
            await p.list_models()
    assert ei.value.kind is ProviderErrorKind.http_error and ei.value.details["status"] == 404


async def test_loaded_state_listing(fake: FakeOllama) -> None:
    fake.json_route(
        "GET", "/api/ps", {"models": [{"name": TAGS[1], "digest": "sha256:d", "expires_at": "2026-01-01T00:05:00Z", "size_vram": 1}]}
    )
    async with OllamaProvider(fake.base_url) as p:
        loaded = await p.list_loaded()
    assert loaded.supported and loaded.tags == {TAGS[1]}
    assert loaded.models[0].expires_at == "2026-01-01T00:05:00Z"
    assert not hasattr(loaded.models[0], "size_vram")  # GPU memory is deliberately not surfaced


async def test_server_without_loaded_state_endpoint_is_unknown_not_empty(fake: FakeOllama) -> None:
    fake.json_route("GET", "/api/ps", {"error": "not found"}, status=404)
    async with OllamaProvider(fake.base_url) as p:
        loaded = await p.list_loaded()
    assert loaded.supported is False and loaded.tags is None


# ---- composition with the frozen contract ----------------------------------------------------------------


async def test_probe_maps_installed_and_loaded_state_by_exact_tag(fake: FakeOllama) -> None:
    # model-2 is installed in a different quantization tag: NOT an exact match, so unavailable
    fake.json_route("GET", "/api/tags", tags_body(TAGS[0], TAGS[2], "example/model-2:other"))
    fake.json_route("GET", "/api/ps", {"models": [{"name": TAGS[2]}]})
    async with OllamaProvider(fake.base_url) as p:
        probe = await probe_models(p, PROFILES)
    assert probe.probe_status is ProbeStatus.ok and probe.error is None
    by_tag = {s.profile.tag: s for s in probe.statuses}
    assert by_tag[TAGS[0]].availability is Availability.available
    assert by_tag[TAGS[0]].loaded_state is LoadedState.not_loaded
    assert by_tag[TAGS[2]].loaded_state is LoadedState.loaded
    for t in (TAGS[1], TAGS[3]):
        assert by_tag[t].availability is Availability.unavailable
        assert by_tag[t].availability_reason is UnavailableReason.not_installed
    assert all(s.observed_at is not None for s in probe.statuses)
    assert {r.path for r in fake.requests} == {"/api/tags", "/api/ps"}  # no pull of the missing ones


async def test_probe_without_ps_support_leaves_loaded_state_unknown(fake: FakeOllama) -> None:
    fake.json_route("GET", "/api/tags", tags_body(*TAGS))
    fake.json_route("GET", "/api/ps", {"error": "nf"}, status=404)
    async with OllamaProvider(fake.base_url) as p:
        probe = await probe_models(p, PROFILES)
    assert probe.probe_status is ProbeStatus.ok
    assert {s.loaded_state for s in probe.statuses} == {LoadedState.unknown}
    assert {s.availability for s in probe.statuses} == {Availability.available}


async def test_probe_survives_a_failing_loaded_state_call(fake: FakeOllama) -> None:
    fake.json_route("GET", "/api/tags", tags_body(*TAGS))
    fake.json_route("GET", "/api/ps", {"models": "garbage"})
    async with OllamaProvider(fake.base_url) as p:
        probe = await probe_models(p, PROFILES)
    assert probe.probe_status is ProbeStatus.ok
    assert probe.loaded_state_error is not None and {s.loaded_state for s in probe.statuses} == {LoadedState.unknown}


async def test_probe_of_an_offline_server_is_unreachable_and_never_claims_available() -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    async with OllamaProvider(f"http://127.0.0.1:{port}") as p:
        probe = await probe_models(p, PROFILES)
    assert probe.probe_status is ProbeStatus.unreachable
    assert probe.error is not None and probe.error.kind is ProviderErrorKind.offline
    assert {s.availability for s in probe.statuses} == {Availability.unknown}
    assert {s.loaded_state for s in probe.statuses} == {LoadedState.unknown}
    assert all(s.observed_at is None for s in probe.statuses)


async def test_probe_does_not_swallow_a_rejected_target() -> None:
    async def external(host: str, port: int) -> list[str]:
        return ["203.0.113.7"]

    async with OllamaProvider("http://localhost:11434", resolver=external) as p:
        with pytest.raises(ProviderError) as ei:
            await probe_models(p, PROFILES)
    assert ei.value.kind is ProviderErrorKind.target_not_loopback


async def test_discovery_timeout_is_typed(fake: FakeOllama) -> None:
    async def hang(c: Conn) -> None:
        await c.client_disconnected.wait()

    fake.on("GET", "/api/tags", hang)
    async with OllamaProvider(fake.base_url, timeouts=ProviderTimeouts(request_s=0.15)) as p:
        with pytest.raises(ProviderError) as ei:
            await p.list_models()
    assert ei.value.kind is ProviderErrorKind.request_timeout and ei.value.retryable
    await fake.wait_connections_closed()
