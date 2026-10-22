"""LH-002: streaming chat against SYNTHETIC fake servers (no real model is involved)."""

from __future__ import annotations

import contextlib
import socket
from typing import Any

import pytest
from pydantic import ValidationError

import localhydra.adapters.ollama as ollama_module
from localhydra.adapters import (
    ChatDelta,
    ChatFinished,
    ChatMessage,
    ChatRequest,
    GenerationOptions,
    OllamaProvider,
    ProviderError,
    ProviderErrorKind,
)
from localhydra.contracts.errors import ERROR_HTTP_STATUS, ErrorCode, ErrorEnvelope
from localhydra.contracts.metrics import IncompleteReason, MetricsStatus, RawOllamaMetrics

from .provider_fakes import (
    SYNTH_FINAL,
    SYNTH_TAG,
    Conn,
    FakeOllama,
    anyio_backend,  # noqa: F401
    chunk,
    final,
    line,
)

pytestmark = pytest.mark.anyio

SECRET = "SECRET-PROMPT-ECHO-12345"
REQ = ChatRequest(model=SYNTH_TAG, messages=(ChatMessage(role="user", content=SECRET),))


@pytest.fixture
async def fake():
    async with FakeOllama() as f:
        yield f


def script(fake: FakeOllama, steps: list[bytes | float], end: str = "terminate") -> None:
    async def h(c: Conn) -> None:
        await c.play(steps, end=end)

    fake.on("POST", "/api/chat", h)


async def run(fake: FakeOllama, request: ChatRequest = REQ, **kw: Any) -> list[Any]:
    items: list[Any] = []
    async with OllamaProvider(fake.base_url, **kw) as p:
        async for item in p.chat_stream(request):
            items.append(item)
    return items


def text(items: list[Any]) -> str:
    return "".join(i.content for i in items if isinstance(i, ChatDelta))


async def error_of(fake: FakeOllama, request: ChatRequest = REQ) -> tuple[ProviderError, list[Any]]:
    items: list[Any] = []
    async with OllamaProvider(fake.base_url) as p:
        with pytest.raises(ProviderError) as ei:
            async for item in p.chat_stream(request):
                items.append(item)
    return ei.value, items


# ---- happy path and exact settings -------------------------------------------------------------


async def test_streams_deltas_then_one_terminal_item_with_raw_metrics(fake: FakeOllama) -> None:
    script(fake, [chunk("Hel"), chunk("lo"), final(**SYNTH_FINAL)])
    items = await run(fake)
    assert [type(i) for i in items] == [ChatDelta, ChatDelta, ChatFinished]
    assert text(items) == "Hello"
    done = items[-1]
    assert done.done_reason == "stop" and done.served_model == SYNTH_TAG
    assert done.telemetry.raw_ollama == RawOllamaMetrics(**SYNTH_FINAL)  # verbatim ns / counts
    assert done.telemetry.chunk_count == 3 and done.telemetry.visible_chunks == 2


async def test_model_tag_and_settings_are_forwarded_exactly(fake: FakeOllama) -> None:
    script(fake, [final()])
    req = ChatRequest(
        model="example/model-1:synthetic",
        messages=(
            ChatMessage(role="system", content="s"),
            ChatMessage(role="user", content="u"),
            ChatMessage(role="assistant", content="a"),
            ChatMessage(role="user", content="ünïcode 日本"),
        ),
        options=GenerationOptions(temperature=0.0, seed=7, num_ctx=2048, num_predict=64, stop=("END",)),
        keep_alive="5m",
        think=False,
    )
    await run(fake, req)
    assert fake.chat_requests[0].json == {
        "model": "example/model-1:synthetic",  # not rewritten, ":latest" never inferred
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "ünïcode 日本"},
        ],
        "stream": True,
        "options": {"temperature": 0.0, "seed": 7, "num_ctx": 2048, "num_predict": 64, "stop": ["END"]},
        "keep_alive": "5m",
        "think": False,
    }


async def test_unset_options_are_omitted_not_defaulted(fake: FakeOllama) -> None:
    script(fake, [final()])
    await run(fake)
    assert fake.chat_requests[0].json == {
        "model": SYNTH_TAG,
        "messages": [{"role": "user", "content": SECRET}],
        "stream": True,
    }
    h = fake.chat_requests[0].headers
    assert h["accept-encoding"] == "identity"
    assert not {"authorization", "cookie"} & set(h)


@pytest.mark.parametrize("bare", ["llama3", "example/model-1", "Example:Tag", "", "a:b:c", " a:b"])
def test_bare_or_invalid_tags_are_rejected_before_any_request(bare: str) -> None:
    with pytest.raises(ValidationError):
        ChatRequest(model=bare, messages=(ChatMessage(role="user", content="x"),))


def test_requests_are_frozen_and_strict() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(model=SYNTH_TAG, messages=())
    with pytest.raises(ValidationError):
        ChatRequest.model_validate({"model": SYNTH_TAG, "messages": [{"role": "user", "content": "x"}], "pull": True})


# ---- chunk-split NDJSON --------------------------------------------------------------------------


async def test_ndjson_split_at_every_byte_boundary(fake: FakeOllama) -> None:
    payload = chunk("héllo 日本 ") + chunk("wörld") + final(**SYNTH_FINAL)
    # one HTTP chunk per byte: lines AND multi-byte UTF-8 characters are split mid-sequence
    script(fake, [bytes([b]) for b in payload])
    items = await run(fake)
    assert text(items) == "héllo 日本 wörld"
    assert isinstance(items[-1], ChatFinished)
    assert items[-1].telemetry.raw_ollama == RawOllamaMetrics(**SYNTH_FINAL)


async def test_many_lines_in_one_chunk_and_crlf_and_blank_lines(fake: FakeOllama) -> None:
    blob = chunk("a") + b"\n" + chunk("b").replace(b"\n", b"\r\n") + chunk("c") + final()
    script(fake, [blob])
    items = await run(fake)
    assert text(items) == "abc" and isinstance(items[-1], ChatFinished)


async def test_unterminated_final_line_is_still_parsed(fake: FakeOllama) -> None:
    script(fake, [chunk("x"), final(**SYNTH_FINAL).rstrip(b"\n")])
    items = await run(fake)
    assert isinstance(items[-1], ChatFinished) and text(items) == "x"


async def test_bytes_after_the_final_chunk_are_ignored(fake: FakeOllama) -> None:
    script(fake, [chunk("x"), final(), chunk("late"), b"garbage"])
    items = await run(fake)
    assert text(items) == "x" and isinstance(items[-1], ChatFinished)


async def test_thinking_text_is_not_visible_content(fake: FakeOllama) -> None:
    script(fake, [chunk("", thinking="pondering"), chunk("answer"), final()])
    items = await run(fake)
    assert items[0] == ChatDelta("", "pondering")
    assert items[-1].telemetry.visible_chunks == 1


# ---- missing final usage -----------------------------------------------------------------------


async def test_final_chunk_without_usage_is_flagged_incomplete_not_invented(fake: FakeOllama) -> None:
    script(fake, [chunk("hi"), final()])
    done = (await run(fake))[-1]
    assert isinstance(done, ChatFinished)
    assert done.telemetry.raw_ollama == RawOllamaMetrics()
    m = done.telemetry.normalized(routing_ms=1.0)
    assert m.completeness.status is MetricsStatus.incomplete
    assert m.completeness.reason is IncompleteReason.partial_final_chunk
    assert {"load_ms", "prompt_eval_ms", "generation_ms", "ollama_total_ms", "prompt_tokens", "generated_tokens"} <= set(
        m.completeness.missing
    )
    assert m.generation_tokens_per_second is None and m.prompt_tokens_per_second is None


async def test_partial_usage_keeps_what_was_sent(fake: FakeOllama) -> None:
    partial = {k: v for k, v in SYNTH_FINAL.items() if k != "prompt_eval_count"}  # e.g. cached prompt
    script(fake, [chunk("hi"), final(**partial)])
    done = (await run(fake))[-1]
    m = done.telemetry.normalized(routing_ms=1.0)
    assert m.completeness.missing == ["prompt_tokens"]
    assert m.generated_tokens == 100 and m.generation_tokens_per_second == 50.0
    assert m.prompt_tokens is None and m.prompt_tokens_per_second is None


async def test_unusable_usage_values_are_dropped_and_reported(fake: FakeOllama) -> None:
    script(fake, [final(**{**SYNTH_FINAL, "eval_count": "100", "eval_duration": -5, "load_duration": 1.5})])
    done = (await run(fake))[-1]
    assert done.telemetry.raw_ollama.eval_count is None
    assert set(done.telemetry.discarded_metric_fields) == {"eval_count", "eval_duration", "load_duration"}


# ---- offline, missing tag, HTTP errors ----------------------------------------------------------


def closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_offline_server_is_a_typed_retryable_error() -> None:
    async with OllamaProvider(f"http://127.0.0.1:{closed_port()}") as p:
        with pytest.raises(ProviderError) as ei:
            [x async for x in p.chat_stream(REQ)]
        with pytest.raises(ProviderError) as ei2:
            await p.list_models()
    for err in (ei.value, ei2.value):
        assert err.kind is ProviderErrorKind.offline
        assert err.code is ErrorCode.provider_unreachable and err.retryable


async def test_missing_tag_is_model_unavailable_and_not_retryable(fake: FakeOllama) -> None:
    async def missing(c: Conn) -> None:
        await c.respond_json(404, {"error": f"model '{SYNTH_TAG}' not found"})

    fake.json_route("GET", "/api/tags", {"models": []})  # the tag is genuinely not installed
    fake.on("POST", "/api/chat", missing)
    err, _ = await error_of(fake)
    assert err.kind is ProviderErrorKind.model_not_found
    assert err.code is ErrorCode.model_unavailable and not err.retryable
    assert err.details["model"] == SYNTH_TAG
    assert fake.paths == ["/api/tags", "/api/chat"]  # inventory check, one chat; nothing pulled or retried


@pytest.mark.parametrize(
    "status,retryable", [(400, False), (401, False), (500, False), (429, True), (502, True), (503, True), (504, True)]
)
async def test_http_errors_are_typed(fake: FakeOllama, status: int, retryable: bool) -> None:
    async def boom(c: Conn) -> None:
        await c.respond_json(status, {"error": f"{SECRET} details"})

    fake.on("POST", "/api/chat", boom)
    err, _ = await error_of(fake)
    assert err.kind is ProviderErrorKind.http_error and err.retryable is retryable
    assert err.details["status"] == status
    assert SECRET not in repr(err) + str(err.details) + err.message


# ---- malformed data and mid-stream failure ------------------------------------------------------


MALFORMED = [
    pytest.param(f"{SECRET} not json\n".encode(), "invalid_json", id="invalid-json"),
    pytest.param(b"\xff\xfe" + SECRET.encode() + b"\n", "invalid_utf8", id="invalid-utf8"),
    pytest.param(f'["{SECRET}"]\n'.encode(), "not_an_object", id="array"),
    pytest.param(line({"message": {"content": SECRET}}), "missing_done_flag", id="no-done"),
    pytest.param(line({"message": {"content": SECRET}, "done": "yes"}), "missing_done_flag", id="done-not-bool"),
    pytest.param(line({"message": {"content": 5, "x": SECRET}, "done": False}), "non_string_content", id="content-int"),
    pytest.param(line({"message": SECRET, "done": False}), "message_not_an_object", id="message-str"),
    pytest.param(line({"done": False, "x": SECRET}), "missing_message", id="no-message"),
]


@pytest.mark.parametrize("bad,reason", MALFORMED)
async def test_malformed_chunks_are_typed_errors_after_earlier_deltas(fake: FakeOllama, bad: bytes, reason: str) -> None:
    script(fake, [chunk("ok "), bad, chunk("never")])
    err, items = await error_of(fake)
    assert err.kind is ProviderErrorKind.malformed_response and err.code is ErrorCode.provider_error
    assert err.details["reason"] == reason
    assert text(items) == "ok "  # what was received before the bad line was delivered
    assert err.details["chunks_received"] == 1 and err.details["visible_chunks"] == 1
    assert SECRET not in repr(err) + str(err.details) + err.message  # no prompt/response echo


async def test_error_object_inside_the_stream_is_reported_without_its_text(fake: FakeOllama) -> None:
    script(fake, [chunk("a"), line({"error": f"{SECRET} runner crashed"})])
    err, items = await error_of(fake)
    assert err.kind is ProviderErrorKind.midstream_error and err.code is ErrorCode.provider_error
    assert text(items) == "a" and SECRET not in repr(err) + str(err.details) + err.message


async def test_disconnect_midstream_is_stream_interrupted(fake: FakeOllama) -> None:
    script(fake, [chunk("one "), chunk("two "), 0.02], end="abort")
    err, items = await error_of(fake)
    assert err.kind is ProviderErrorKind.stream_interrupted
    assert err.code is ErrorCode.stream_interrupted and err.retryable
    assert text(items) == "one two " and err.details["chunks_received"] == 2


async def test_clean_eof_without_final_chunk_is_stream_interrupted(fake: FakeOllama) -> None:
    script(fake, [chunk("one "), chunk("two ")], end="terminate")
    err, items = await error_of(fake)
    assert err.kind is ProviderErrorKind.stream_interrupted and err.retryable
    assert "final chunk" in err.message and text(items) == "one two "


async def test_truncated_last_line_is_stream_interrupted(fake: FakeOllama) -> None:
    script(fake, [chunk("one "), chunk("two")[:20]], end="terminate")
    err, _ = await error_of(fake)
    assert err.kind is ProviderErrorKind.stream_interrupted and "middle of a chunk" in err.message


async def test_overlong_line_is_rejected(fake: FakeOllama, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama_module, "MAX_LINE_BYTES", 64)
    script(fake, [b"x" * 500], end="hang")
    err, _ = await error_of(fake)
    assert err.kind is ProviderErrorKind.malformed_response and err.details["reason"] == "line_too_long"


@pytest.mark.parametrize("payload", [chunk("x" * 200), b"x" * 200 + b"\n", b" " * 200 + b"\n"])
@pytest.mark.parametrize("split", [False, True])
async def test_complete_overlong_line_is_rejected_before_parsing(
    fake: FakeOllama, monkeypatch: pytest.MonkeyPatch, payload: bytes, split: bool
) -> None:
    monkeypatch.setattr(ollama_module, "MAX_LINE_BYTES", 128)
    steps = [payload[:100], payload[100:]] if split else [payload]
    script(fake, [*steps, line({"done": True})])
    err, items = await error_of(fake)
    assert err.kind is ProviderErrorKind.malformed_response
    assert err.details["reason"] == "line_too_long"
    assert not items  # not parsed/yielded, even for valid JSON or whitespace


async def test_line_limit_is_per_line_and_inclusive(fake: FakeOllama, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = line({"done": False, "message": {"content": "a"}})
    monkeypatch.setattr(ollama_module, "MAX_LINE_BYTES", len(payload) - 1)
    script(fake, [payload * 4 + line({"done": True})])
    items = await run(fake)
    assert text(items) == "aaaa"
    assert isinstance(items[-1], ChatFinished)


# ---- error envelope conformance -----------------------------------------------------------------


def test_every_error_kind_maps_to_a_frozen_envelope_code() -> None:
    for kind in ProviderErrorKind:
        err = ProviderError(kind, "m")
        body = err.to_error_body(request_id="req_1")
        assert body.code in ERROR_HTTP_STATUS
        env = ErrorEnvelope.model_validate(ErrorEnvelope(error=body).model_dump(mode="json"))
        assert env.error.details["kind"] == kind.value


async def test_stream_can_be_closed_early_with_aclosing(fake: FakeOllama) -> None:
    script(fake, [chunk("a"), chunk("b"), chunk("c")], end="hang")
    async with OllamaProvider(fake.base_url) as p:
        stream = p.chat_stream(REQ)
        async with contextlib.aclosing(stream):
            first = await stream.__anext__()
            assert first == ChatDelta("a")
        # leaving aclosing() closed the HTTP connection although the server was still streaming
        await fake.wait_connections_closed()
        assert fake.last_conn is not None and fake.last_conn.client_disconnected.is_set()
