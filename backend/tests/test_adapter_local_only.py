"""LH-002: local-only inference. An Ollama server can list cloud-proxied models (a real Ollama
did: entries carry `remote_host` / `remote_model`); sending a prompt to one would leave the
machine. The adapter must never do that. Inventories below are SYNTHETIC (`.invalid` hosts)."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from localhydra.adapters import (
    ChatMessage,
    ChatRequest,
    OllamaProvider,
    ProviderError,
    ProviderErrorKind,
    probe_models,
)
from localhydra.contracts.config import AppConfig
from localhydra.contracts.errors import ErrorCode
from localhydra.contracts.models import Availability

from .conftest import EXAMPLE_CONFIG
from .provider_fakes import SYNTH_TAG, FakeOllama, anyio_backend, final  # noqa: F401

pytestmark = pytest.mark.anyio

PROFILES = AppConfig.model_validate(json.loads(EXAMPLE_CONFIG.read_text())).models
REMOTE_TAG = PROFILES[1].tag  # synthetic tag that the fake inventory flags as remote


class Out(BaseModel):
    ok: bool


def req(tag: str) -> ChatRequest:
    return ChatRequest(model=tag, messages=(ChatMessage(role="user", content="x"),))


def inventory(*, remote: dict[str, dict] | None = None, local: tuple[str, ...] = (SYNTH_TAG,)) -> dict:
    models = [{"name": t, "digest": "sha256:synthetic", "size": 1} for t in local]
    for tag, extra in (remote or {}).items():
        models.append({"name": tag, "size": 362, **extra})
    return {"models": models}


REMOTE_FIELDS = {"remote_model": "synthetic-upstream", "remote_host": "https://remote.invalid:443"}


@pytest.fixture
async def fake():
    async with FakeOllama() as f:

        async def ok(c) -> None:  # noqa: ANN001
            await c.play([final()])

        f.on("POST", "/api/chat", ok)
        yield f


@pytest.mark.parametrize("tag", ["minimax-m3:cloud", "gpt-oss:120b-cloud", "x/y:Cloud", "a:Big-CLOUD"])
async def test_cloud_tag_names_are_refused_before_any_request(fake: FakeOllama, tag: str) -> None:
    async with OllamaProvider(fake.base_url) as p:
        with pytest.raises(ProviderError) as ei:
            [x async for x in p.chat_stream(req(tag))]
        with pytest.raises(ProviderError) as ei2:
            await p.chat_structured(req(tag), Out)
    for err in (ei.value, ei2.value):
        assert err.kind is ProviderErrorKind.remote_model_refused
        assert err.code is ErrorCode.model_unavailable and not err.retryable
        assert err.details["model"] == tag
    assert fake.total_connections == 0


async def test_inventory_marked_remote_is_refused_and_no_chat_is_sent(fake: FakeOllama) -> None:
    fake.json_route("GET", "/api/tags", inventory(remote={REMOTE_TAG: REMOTE_FIELDS}))
    async with OllamaProvider(fake.base_url) as p:
        with pytest.raises(ProviderError) as ei:
            [x async for x in p.chat_stream(req(REMOTE_TAG))]
        with pytest.raises(ProviderError) as ei2:
            await p.chat_structured(req(REMOTE_TAG), Out)
    assert ei.value.kind is ei2.value.kind is ProviderErrorKind.remote_model_refused
    assert ei.value.details["detected_by"] == "inventory"
    assert fake.chat_requests == []


@pytest.mark.parametrize("fields", [{"remote_host": "https://remote.invalid"}, {"remote_model": "up"}])
async def test_either_remote_field_is_enough(fake: FakeOllama, fields: dict) -> None:
    fake.json_route("GET", "/api/tags", inventory(remote={REMOTE_TAG: fields}))
    async with OllamaProvider(fake.base_url) as p:
        with pytest.raises(ProviderError):
            [x async for x in p.chat_stream(req(REMOTE_TAG))]
    assert fake.chat_requests == []


async def test_local_model_passes_and_inventory_is_cached_per_provider(fake: FakeOllama) -> None:
    async with OllamaProvider(fake.base_url) as p:
        for _ in range(3):
            assert [x async for x in p.chat_stream(req(SYNTH_TAG))]
    assert fake.paths == ["/api/tags", "/api/chat", "/api/chat", "/api/chat"]


async def test_a_refreshed_inventory_that_turns_remote_stops_further_chats(fake: FakeOllama) -> None:
    async with OllamaProvider(fake.base_url) as p:
        assert [x async for x in p.chat_stream(req(SYNTH_TAG))]
        fake.json_route("GET", "/api/tags", inventory(remote={SYNTH_TAG: REMOTE_FIELDS}, local=()))
        await p.list_models()  # discovery refresh updates the verified-local cache
        with pytest.raises(ProviderError) as ei:
            [x async for x in p.chat_stream(req(SYNTH_TAG))]
    assert ei.value.kind is ProviderErrorKind.remote_model_refused
    assert len(fake.chat_requests) == 1


async def test_discovery_reports_remote_models_verbatim_but_separately(fake: FakeOllama) -> None:
    fake.json_route("GET", "/api/tags", inventory(remote={REMOTE_TAG: REMOTE_FIELDS}))
    async with OllamaProvider(fake.base_url) as p:
        inv = await p.list_models()
    remote = next(m for m in inv.models if m.tag == REMOTE_TAG)
    assert remote.is_remote and remote.remote_host == "https://remote.invalid:443"
    assert inv.remote_tags == {REMOTE_TAG} and inv.local_tags == {SYNTH_TAG} and REMOTE_TAG in inv.tags


async def test_probe_never_reports_a_remote_model_as_available(fake: FakeOllama) -> None:
    fake.json_route("GET", "/api/tags", inventory(remote={REMOTE_TAG: REMOTE_FIELDS}, local=(PROFILES[0].tag,)))
    fake.json_route("GET", "/api/ps", {"models": []})
    async with OllamaProvider(fake.base_url) as p:
        probe = await probe_models(p, PROFILES)
    by_tag = {s.profile.tag: s.availability for s in probe.statuses}
    assert by_tag[PROFILES[0].tag] is Availability.available
    assert by_tag[REMOTE_TAG] is Availability.unavailable
    assert probe.remote_tags == {REMOTE_TAG}  # surfaced so a consumer can explain why
