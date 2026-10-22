"""LH-002: timeouts, cancellation and resource release against SYNTHETIC fake servers.

What these tests can prove: the adapter stops promptly, the TCP connection is closed (the fake
server observes the disconnect) and no connection stays open. They CANNOT prove that a real
Ollama stops computing; that is documented as best-effort and `upstream_stop_confirmed` is
always False.
"""

from __future__ import annotations

import asyncio
import contextlib
import time

import httpx
import pytest
from pydantic import BaseModel

from localhydra.adapters import (
    ChatCancelled,
    ChatDelta,
    ChatFinished,
    ChatMessage,
    ChatRequest,
    OllamaProvider,
    ProviderCancelled,
    ProviderError,
    ProviderErrorKind,
    ProviderTimeouts,
)
from localhydra.adapters.ollama import _map_httpx_error
from localhydra.contracts.errors import ErrorCode
from localhydra.contracts.metrics import IncompleteReason

from .provider_fakes import SYNTH_TAG, Conn, FakeOllama, anyio_backend, chunk, final  # noqa: F401

pytestmark = pytest.mark.anyio

REQ = ChatRequest(model=SYNTH_TAG, messages=(ChatMessage(role="user", content="x"),))
PROMPT_S = 0.75  # generous bound for "promptly" on a loaded CI machine


class Verdict(BaseModel):
    label: str


@pytest.fixture
async def fake():
    async with FakeOllama() as f:
        yield f


def on_chat(fake: FakeOllama, handler) -> None:
    fake.on("POST", "/api/chat", handler)


async def hang(c: Conn) -> None:
    await c.client_disconnected.wait()


async def headers_then_silence(c: Conn) -> None:
    await c.start_stream()
    await c.client_disconnected.wait()


# ---- timeouts ---------------------------------------------------------------------------------------


async def test_no_response_headers_in_time_is_first_chunk_timeout(fake: FakeOllama) -> None:
    on_chat(fake, hang)
    async with OllamaProvider(fake.base_url, timeouts=ProviderTimeouts(first_chunk_s=0.15)) as p:
        with pytest.raises(ProviderError) as ei:
            [x async for x in p.chat_stream(REQ)]
    assert ei.value.kind is ProviderErrorKind.first_chunk_timeout
    assert ei.value.code is ErrorCode.provider_unreachable and ei.value.retryable
    await fake.wait_connections_closed()


async def test_headers_but_no_first_chunk_in_time_is_first_chunk_timeout(fake: FakeOllama) -> None:
    on_chat(fake, headers_then_silence)
    async with OllamaProvider(fake.base_url, timeouts=ProviderTimeouts(first_chunk_s=0.15)) as p:
        with pytest.raises(ProviderError) as ei:
            [x async for x in p.chat_stream(REQ)]
    assert ei.value.kind is ProviderErrorKind.first_chunk_timeout
    await fake.wait_connections_closed()


async def test_first_chunk_budget_is_shared_by_headers_and_first_body(fake: FakeOllama) -> None:
    async def delayed(c: Conn) -> None:
        await asyncio.sleep(0.08)
        await c.start_stream()
        await asyncio.sleep(0.08)
        await c.send(chunk("too late"))
        await c.send(final())
        await c.end_stream()

    on_chat(fake, delayed)
    async with OllamaProvider(fake.base_url) as p:
        with pytest.raises(ProviderError) as ei:
            [item async for item in p.chat_stream(REQ, timeouts=ProviderTimeouts(first_chunk_s=0.12))]
        assert ei.value.kind is ProviderErrorKind.first_chunk_timeout
        await fake.wait_connections_closed()


async def test_gap_between_chunks_is_idle_timeout_and_keeps_earlier_output(fake: FakeOllama) -> None:
    async def stall(c: Conn) -> None:
        await c.start_stream()
        await c.send(chunk("early "))
        await c.client_disconnected.wait()

    on_chat(fake, stall)
    got: list[object] = []
    async with OllamaProvider(fake.base_url, timeouts=ProviderTimeouts(first_chunk_s=5, idle_s=0.15)) as p:
        stream = p.chat_stream(REQ)
        with pytest.raises(ProviderError) as ei:
            async for item in stream:
                got.append(item)
    assert got == [ChatDelta("early ")]
    assert ei.value.kind is ProviderErrorKind.idle_timeout and ei.value.code is ErrorCode.stream_interrupted
    assert ei.value.details["chunks_received"] == 1
    m = stream.telemetry.snapshot().normalized(routing_ms=1.0)  # failure keeps honest partial telemetry
    assert m.wall_ms is not None and m.ttft_ms is not None
    assert m.completeness.reason is IncompleteReason.no_final_chunk
    await fake.wait_connections_closed()


async def test_total_deadline_ends_a_stream_that_never_stalls(fake: FakeOllama) -> None:
    async def endless(c: Conn) -> None:
        await c.stream_forever(0.01)

    on_chat(fake, endless)
    started = time.monotonic()
    async with OllamaProvider(fake.base_url, timeouts=ProviderTimeouts(idle_s=5, total_s=0.25)) as p:
        with pytest.raises(ProviderError) as ei:
            [x async for x in p.chat_stream(REQ)]
    assert ei.value.kind is ProviderErrorKind.total_timeout and ei.value.code is ErrorCode.stream_interrupted
    assert time.monotonic() - started < PROMPT_S + 0.25
    await fake.wait_connections_closed()


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("inventory_s,total_s,reply_s", [(0.2, 0.05, 0.08), (0.15, 0.4, 0.3)])
async def test_chat_total_budget_includes_inventory_and_is_not_restarted(
    fake: FakeOllama, structured: bool, inventory_s: float, total_s: float, reply_s: float
) -> None:
    async def inventory(c: Conn) -> None:
        await asyncio.sleep(inventory_s)
        await c.respond_json(200, {"models": [{"name": SYNTH_TAG}]})

    async def reply(c: Conn) -> None:
        await asyncio.sleep(reply_s)
        if structured:
            await c.respond_json(200, {"message": {"content": '{"label":"ok"}'}, "done": True})
        else:
            await c.play([chunk("ok"), final()])

    fake.on("GET", "/api/tags", inventory)
    on_chat(fake, reply)
    async with OllamaProvider(fake.base_url) as p:
        with pytest.raises(ProviderError) as ei:
            to = ProviderTimeouts(total_s=total_s)
            if structured:
                await p.chat_structured(REQ, Verdict, timeouts=to)
            else:
                [item async for item in p.chat_stream(REQ, timeouts=to)]
        assert ei.value.kind is ProviderErrorKind.total_timeout
        assert len(fake.chat_requests) == (0 if inventory_s > total_s else 1)
        await fake.wait_connections_closed()


async def test_total_deadline_before_any_response_is_reported_as_unreachable(fake: FakeOllama) -> None:
    on_chat(fake, hang)
    async with OllamaProvider(fake.base_url, timeouts=ProviderTimeouts(first_chunk_s=30, total_s=0.15)) as p:
        with pytest.raises(ProviderError) as ei:
            [x async for x in p.chat_stream(REQ)]
    assert ei.value.kind is ProviderErrorKind.total_timeout and ei.value.code is ErrorCode.provider_unreachable


def test_transport_timeouts_and_disconnects_map_to_typed_errors() -> None:
    assert _map_httpx_error(httpx.ConnectTimeout(""), midstream=False).kind is ProviderErrorKind.connect_timeout
    assert _map_httpx_error(httpx.ConnectError(""), midstream=False).kind is ProviderErrorKind.offline
    assert _map_httpx_error(httpx.ReadTimeout(""), midstream=True).kind is ProviderErrorKind.idle_timeout
    assert _map_httpx_error(httpx.ReadTimeout(""), midstream=False).kind is ProviderErrorKind.request_timeout
    assert _map_httpx_error(httpx.RemoteProtocolError(""), midstream=True).kind is ProviderErrorKind.stream_interrupted
    assert _map_httpx_error(httpx.DecodingError(""), midstream=True).kind is ProviderErrorKind.malformed_response
    assert _map_httpx_error(httpx.TooManyRedirects(""), midstream=False).kind is ProviderErrorKind.http_error


# ---- cancellation: streaming ---------------------------------------------------------------------------


async def test_cancel_closes_the_stream_promptly_and_releases_the_connection(fake: FakeOllama) -> None:
    async def endless(c: Conn) -> None:
        await c.stream_forever(0.01)

    on_chat(fake, endless)
    async with OllamaProvider(fake.base_url) as p:
        stream = p.chat_stream(REQ)
        seen: list[object] = []
        async for item in stream:
            seen.append(item)
            if len(seen) == 3:
                stream.cancel()
                t_cancel = time.monotonic()
                break
        terminal = await stream.__anext__()
        latency = time.monotonic() - t_cancel
        with pytest.raises(StopAsyncIteration):  # exactly one terminal item
            await stream.__anext__()
    assert isinstance(terminal, ChatCancelled) and latency < PROMPT_S
    assert terminal.upstream_stop_confirmed is False  # closing the socket is all we can claim
    assert all(isinstance(i, ChatDelta) for i in seen)
    await fake.wait_connections_closed()
    assert fake.last_conn is not None and fake.last_conn.client_disconnected.is_set()
    m = terminal.telemetry.normalized(routing_ms=1.0)  # cancelled -> no final chunk, flagged as such
    assert m.raw_ollama is None and m.completeness.reason is IncompleteReason.no_final_chunk
    assert m.wall_ms is not None and m.ttft_ms is not None


@pytest.mark.parametrize("external_event", [False, True])
async def test_cancel_releases_response_without_advancing_suspended_consumer(
    fake: FakeOllama, external_event: bool
) -> None:
    async def stall(c: Conn) -> None:
        await c.play([chunk("first")], end="hang")

    on_chat(fake, stall)
    cancel = asyncio.Event()
    async with OllamaProvider(fake.base_url) as p:
        stream = p.chat_stream(REQ, cancel=cancel)
        assert await stream.__anext__() == ChatDelta("first")
        conn = fake.last_conn
        assert conn is not None
        try:
            if external_event:
                cancel.set()
            else:
                stream.cancel()
                stream.cancel()  # repeated cancellation must not spawn extra work
            await asyncio.sleep(0.15)  # deliberately do not advance or close the iterator
            assert conn.client_disconnected.is_set()
            assert fake.open_connections == 0
            assert stream.telemetry.finished
            assert not [
                task for task in asyncio.all_tasks()
                if task.get_coro().cr_code.co_filename.endswith("adapters/ollama.py")
            ]
            assert isinstance(await stream.__anext__(), ChatCancelled)
            with pytest.raises(StopAsyncIteration):
                await stream.__anext__()
            stream.cancel()
        finally:
            await stream.aclose()


async def test_external_cancel_event_wakes_a_stream_that_is_blocked_on_the_network(fake: FakeOllama) -> None:
    on_chat(fake, headers_then_silence)
    cancel = asyncio.Event()
    async with OllamaProvider(fake.base_url) as p:
        stream = p.chat_stream(REQ, cancel=cancel)
        asyncio.get_running_loop().call_later(0.1, cancel.set)
        t0 = time.monotonic()
        items = [i async for i in stream]
    assert len(items) == 1 and isinstance(items[0], ChatCancelled)
    assert time.monotonic() - t0 < PROMPT_S
    await fake.wait_connections_closed()


async def test_cancel_while_waiting_for_response_headers(fake: FakeOllama) -> None:
    on_chat(fake, hang)
    cancel = asyncio.Event()
    async with OllamaProvider(fake.base_url) as p:
        asyncio.get_running_loop().call_later(0.1, cancel.set)
        t0 = time.monotonic()
        items = [i async for i in p.chat_stream(REQ, cancel=cancel)]
    assert isinstance(items[-1], ChatCancelled) and time.monotonic() - t0 < PROMPT_S
    await fake.wait_connections_closed()


async def test_cancel_set_before_start_sends_no_request(fake: FakeOllama) -> None:
    cancel = asyncio.Event()
    cancel.set()
    async with OllamaProvider(fake.base_url) as p:
        items = [i async for i in p.chat_stream(REQ, cancel=cancel)]
    assert len(items) == 1 and isinstance(items[0], ChatCancelled)
    assert fake.total_connections == 0


async def test_cancelling_the_consuming_task_closes_the_connection(fake: FakeOllama) -> None:
    async def endless(c: Conn) -> None:
        await c.stream_forever(0.01)

    on_chat(fake, endless)
    got_first = asyncio.Event()

    async with OllamaProvider(fake.base_url) as p:

        async def consume() -> None:
            async for _ in p.chat_stream(REQ):
                got_first.set()

        task = asyncio.ensure_future(consume())
        await asyncio.wait_for(got_first.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await fake.wait_connections_closed()
        assert fake.last_conn is not None and fake.last_conn.client_disconnected.is_set()


async def test_aclose_from_another_task_while_blocked_ends_the_stream(fake: FakeOllama) -> None:
    on_chat(fake, headers_then_silence)
    async with OllamaProvider(fake.base_url) as p:
        stream = p.chat_stream(REQ)

        async def consume() -> list[object]:
            return [i async for i in stream]

        task = asyncio.ensure_future(consume())
        await asyncio.sleep(0.1)
        await stream.aclose()  # consumer is parked inside __anext__: must not raise or hang
        items = await asyncio.wait_for(task, PROMPT_S)
    assert len(items) == 1 and isinstance(items[0], ChatCancelled)
    await fake.wait_connections_closed()


@pytest.mark.parametrize("action", ["none", "cancel", "external_event"])
@pytest.mark.parametrize("final_content", [False, True])
async def test_finished_releases_response_without_advancing_terminal_consumer(
    fake: FakeOllama, action: str, final_content: bool
) -> None:
    async def finish_and_hold_open(c: Conn) -> None:
        metrics = {"message": {"role": "assistant", "content": "last"}} if final_content else {}
        await c.play([final(**metrics)], end="hang")

    on_chat(fake, finish_and_hold_open)
    cancel = asyncio.Event()
    async with OllamaProvider(fake.base_url) as p:
        stream = p.chat_stream(REQ, cancel=cancel)
        try:
            if final_content:
                assert await stream.__anext__() == ChatDelta("last")
            terminal = await stream.__anext__()
            assert isinstance(terminal, ChatFinished)
            conn = fake.last_conn
            assert conn is not None
            if action == "cancel":
                stream.cancel()
                stream.cancel()
            elif action == "external_event":
                cancel.set()
            await asyncio.sleep(0.15)  # no next/aclose, including the successful no-cancel case
            assert conn.client_disconnected.is_set()
            assert fake.open_connections == 0
            assert stream.telemetry.finished
            assert stream.telemetry.snapshot() == terminal.telemetry
            assert not [
                task for task in asyncio.all_tasks()
                if task.get_coro().cr_code.co_filename.endswith("adapters/ollama.py")
            ]
            with pytest.raises(StopAsyncIteration):
                await stream.__anext__()
        finally:
            await stream.aclose()


async def test_cancel_after_completion_is_harmless(fake: FakeOllama) -> None:
    async def ok(c: Conn) -> None:
        await c.play([chunk("a"), final()])

    on_chat(fake, ok)
    async with OllamaProvider(fake.base_url) as p:
        stream = p.chat_stream(REQ)
        items = [i async for i in stream]
        stream.cancel()
        await stream.aclose()
    assert isinstance(items[-1], ChatFinished)


async def test_provider_close_releases_the_http_client_and_all_connections(fake: FakeOllama) -> None:
    async def endless(c: Conn) -> None:
        await c.stream_forever(0.01)

    on_chat(fake, endless)
    p = OllamaProvider(fake.base_url)
    async with contextlib.aclosing(p.chat_stream(REQ)) as stream:
        await stream.__anext__()
    await p.aclose()
    assert p._client.is_closed
    await fake.wait_connections_closed()


# ---- cancellation: non-streaming calls -----------------------------------------------------------------


async def test_structured_call_cancel_raises_and_closes_connection(fake: FakeOllama) -> None:
    on_chat(fake, hang)
    cancel = asyncio.Event()
    async with OllamaProvider(fake.base_url) as p:
        asyncio.get_running_loop().call_later(0.1, cancel.set)
        t0 = time.monotonic()
        with pytest.raises(ProviderCancelled) as ei:
            await p.chat_structured(REQ, Verdict, cancel=cancel)
    assert time.monotonic() - t0 < PROMPT_S and ei.value.upstream_stop_confirmed is False
    await fake.wait_connections_closed()


async def test_discovery_cancel(fake: FakeOllama) -> None:
    fake.on("GET", "/api/tags", hang)
    cancel = asyncio.Event()
    async with OllamaProvider(fake.base_url) as p:
        asyncio.get_running_loop().call_later(0.1, cancel.set)
        with pytest.raises(ProviderCancelled):
            await p.list_models(cancel=cancel)
    await fake.wait_connections_closed()


async def test_structured_total_timeout(fake: FakeOllama) -> None:
    on_chat(fake, hang)
    async with OllamaProvider(fake.base_url, timeouts=ProviderTimeouts(total_s=0.15)) as p:
        with pytest.raises(ProviderError) as ei:
            await p.chat_structured(REQ, Verdict)
    assert ei.value.kind is ProviderErrorKind.total_timeout
    await fake.wait_connections_closed()
