"""LH-002: schema-constrained (structured) output against a SYNTHETIC fake server."""

from __future__ import annotations

import json
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from localhydra.adapters import (
    ChatMessage,
    ChatRequest,
    GenerationOptions,
    OllamaProvider,
    ProviderError,
    ProviderErrorKind,
)
from localhydra.contracts.errors import ErrorCode
from localhydra.contracts.metrics import IncompleteReason

from .provider_fakes import SYNTH_FINAL, SYNTH_TAG, Conn, FakeOllama, anyio_backend  # noqa: F401

pytestmark = pytest.mark.anyio

SECRET = "SECRET-PROMPT-ECHO-12345"


class Verdict(BaseModel):
    label: Literal["alpha", "beta"]
    confidence: float = Field(ge=0, le=1)


REQ = ChatRequest(
    model=SYNTH_TAG,
    messages=(ChatMessage(role="user", content=SECRET),),
    options=GenerationOptions(temperature=0.0, num_predict=32),
)


@pytest.fixture
async def fake():
    async with FakeOllama() as f:
        yield f


def reply(content: str, **extra: object) -> dict:
    return {
        "model": SYNTH_TAG,
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
        **extra,
    }


def serve(fake: FakeOllama, body: object, status: int = 200) -> None:
    fake.json_route("POST", "/api/chat", body, status=status)


async def call(fake: FakeOllama):
    async with OllamaProvider(fake.base_url) as p:
        return await p.chat_structured(REQ, Verdict)


async def error_of(fake: FakeOllama) -> ProviderError:
    with pytest.raises(ProviderError) as ei:
        await call(fake)
    return ei.value


async def test_valid_structured_reply_is_parsed_and_the_schema_is_sent_as_format(fake: FakeOllama) -> None:
    serve(fake, reply('{"label": "alpha", "confidence": 0.75}', **SYNTH_FINAL))
    result = await call(fake)
    assert result.parsed == Verdict(label="alpha", confidence=0.75)
    assert result.done_reason == "stop" and result.served_model == SYNTH_TAG
    body = fake.chat_requests[0].json
    assert body == {
        "model": SYNTH_TAG,
        "messages": [{"role": "user", "content": SECRET}],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 32},
        "format": Verdict.model_json_schema(),
    }
    assert body["format"]["properties"]["label"]["enum"] == ["alpha", "beta"]


async def test_structured_telemetry_is_raw_preserving_and_has_no_ttft(fake: FakeOllama) -> None:
    serve(fake, reply('{"label": "beta", "confidence": 0}', **SYNTH_FINAL))
    t = (await call(fake)).telemetry
    assert t.raw_ollama.eval_count == 100 and t.raw_ollama.eval_duration == 2_000_000_000
    assert t.ttft_ms is None and t.visible_chunks == 0 and t.wall_ms is not None
    m = t.normalized(routing_ms=1.0)
    assert m.generation_tokens_per_second == 50.0
    # nothing was streamed, so the only gap is the first-visible-token time
    assert m.completeness.missing == ["ttft_ms"] and m.completeness.reason is IncompleteReason.no_visible_token


@pytest.mark.parametrize(
    "content,first_error",
    [
        (f"{SECRET} not json", ("json_invalid", [])),
        ('{"label": "gamma", "confidence": 0.5}', ("literal_error", ["label"])),
        ('{"label": "alpha"}', ("missing", ["confidence"])),
        ('{"label": "alpha", "confidence": 7}', ("less_than_equal", ["confidence"])),
        ("", ("json_invalid", [])),
        ("[1]", ("model_type", [])),
    ],
)
async def test_invalid_output_is_a_typed_error_listing_paths_only(fake: FakeOllama, content: str, first_error) -> None:
    serve(fake, reply(content))
    err = await error_of(fake)
    assert err.kind is ProviderErrorKind.structured_output_invalid and err.code is ErrorCode.provider_error
    assert not err.retryable
    assert (err.details["errors"][0]["type"], err.details["errors"][0]["loc"]) == first_error
    leaked = json.dumps(err.details) + err.message + repr(err)
    assert SECRET not in leaked and "gamma" not in leaked  # no echo of model output or input


async def test_truncated_output_reports_the_done_reason(fake: FakeOllama) -> None:
    serve(fake, {**reply('{"label": "alp'), "done_reason": "length"})
    err = await error_of(fake)
    assert err.kind is ProviderErrorKind.structured_output_invalid and err.details["done_reason"] == "length"


async def test_missing_tag_is_model_unavailable_and_nothing_is_pulled(fake: FakeOllama) -> None:
    fake.json_route("GET", "/api/tags", {"models": []})
    serve(fake, {"error": f"model '{SYNTH_TAG}' not found"}, status=404)
    err = await error_of(fake)
    assert err.kind is ProviderErrorKind.model_not_found and err.code is ErrorCode.model_unavailable
    assert fake.paths == ["/api/tags", "/api/chat"]


@pytest.mark.parametrize("raw", [b"not json", b"[]", b"\xff\xfe"])
async def test_malformed_envelope_is_typed(fake: FakeOllama, raw: bytes) -> None:
    async def h(c: Conn) -> None:
        await c.respond_bytes(200, raw)

    fake.on("POST", "/api/chat", h)
    err = await error_of(fake)
    assert err.kind is ProviderErrorKind.malformed_response


async def test_error_object_in_a_200_reply_is_reported_without_its_text(fake: FakeOllama) -> None:
    serve(fake, {"error": f"{SECRET} runner crashed"})
    err = await error_of(fake)
    assert err.kind is ProviderErrorKind.midstream_error and SECRET not in repr(err) + str(err.details)


@pytest.mark.parametrize("status", [500, 503])
async def test_http_errors(fake: FakeOllama, status: int) -> None:
    serve(fake, {"error": "x"}, status=status)
    err = await error_of(fake)
    assert err.kind is ProviderErrorKind.http_error and err.retryable is (status == 503)
