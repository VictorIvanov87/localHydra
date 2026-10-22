"""LH-002: the provider is injectable. Downstream code written against `ModelProvider` runs
unchanged on the real adapter (against a SYNTHETIC fake server) and on an in-memory synthetic one."""

from __future__ import annotations

import asyncio

import pytest

from localhydra.adapters import (
    ChatCancelled,
    ChatDelta,
    ChatFinished,
    ChatMessage,
    ChatRequest,
    ModelProvider,
    OllamaProvider,
    probe_models,
)
from localhydra.contracts.config import AppConfig
from localhydra.contracts.models import Availability

from .conftest import EXAMPLE_CONFIG
from .provider_fakes import SYNTH_FINAL, SYNTH_TAG, Conn, FakeOllama, SyntheticProvider, anyio_backend, chunk, final  # noqa: F401

pytestmark = pytest.mark.anyio

REQ = ChatRequest(model=SYNTH_TAG, messages=(ChatMessage(role="user", content="x"),))


async def consumer(provider: ModelProvider) -> tuple[str, type]:
    """Stand-in for a downstream user (e.g. the runtime): only sees the Protocol."""
    text = ""
    last: object = None
    async for item in provider.chat_stream(REQ):
        if isinstance(item, ChatDelta):
            text += item.content
        last = item
    return text, type(last)


async def test_same_consumer_runs_on_the_real_adapter_and_on_a_synthetic_provider() -> None:
    async with FakeOllama() as fake:

        async def h(c: Conn) -> None:
            await c.play([chunk("hello "), chunk("world"), final(**SYNTH_FINAL)])

        fake.on("POST", "/api/chat", h)
        async with OllamaProvider(fake.base_url) as real:
            assert isinstance(real, ModelProvider)
            via_http = await consumer(real)
    synthetic = SyntheticProvider(reply=("hello ", "world"))
    assert isinstance(synthetic, ModelProvider)
    assert await consumer(synthetic) == via_http == ("hello world", ChatFinished)


async def test_synthetic_provider_drives_discovery_and_cancellation_without_a_network() -> None:
    profiles = AppConfig.model_validate_json(EXAMPLE_CONFIG.read_text()).models
    provider = SyntheticProvider(installed_tags=(profiles[0].tag,), loaded_tags=(profiles[0].tag,))
    probe = await probe_models(provider, profiles)
    assert [s.availability for s in probe.statuses][:2] == [Availability.available, Availability.unavailable]

    cancel = asyncio.Event()
    stream = provider.chat_stream(REQ, cancel=cancel)
    first = await stream.__anext__()
    cancel.set()
    assert isinstance(first, ChatDelta) and isinstance(await stream.__anext__(), ChatCancelled)
    await provider.aclose()
    assert provider.closed
