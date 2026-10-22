"""SYNTHETIC test doubles for the provider adapter (LH-002). Nothing here is real Ollama output.

- `FakeOllama`: a real TCP server on 127.0.0.1 (ephemeral port) speaking just enough HTTP/1.1 to
  exercise the adapter byte-for-byte: chunk-split NDJSON, stalls, mid-stream disconnects,
  redirects, error statuses. It records every request so tests can prove which endpoints and
  bodies were (and were not) used. Model tags/values are placeholders, never measurements.
- `SyntheticProvider`: in-memory `ModelProvider` for interface-injection tests.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from localhydra.adapters.provider import (
    ChatCancelled,
    ChatDelta,
    ChatFinished,
    ChatRequest,
    InstalledModel,
    InstalledModels,
    LoadedModel,
    LoadedModels,
    ProviderTimeouts,
    StructuredResult,
)
from localhydra.adapters.telemetry import StreamTelemetry

SYNTH_TAG = "example/model-1:synthetic"

# Invented numbers (nanoseconds / counts) chosen so expected conversions are exact by hand.
SYNTH_FINAL = {
    "total_duration": 5_000_000_000,
    "load_duration": 1_000_000_000,
    "prompt_eval_count": 26,
    "prompt_eval_duration": 500_000_000,
    "eval_count": 100,
    "eval_duration": 2_000_000_000,
}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def line(obj: dict[str, Any]) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n"


def chunk(text: str, *, model: str = SYNTH_TAG, thinking: str | None = None) -> bytes:
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if thinking is not None:
        message["thinking"] = thinking
    return line({"model": model, "created_at": "2026-01-01T00:00:00Z", "message": message, "done": False})


def final(*, model: str = SYNTH_TAG, reason: str = "stop", **metrics: Any) -> bytes:
    body = {
        "model": model,
        "created_at": "2026-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": reason,
        **metrics,
    }
    return line(body)


@dataclass
class Recorded:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes

    @property
    def json(self) -> Any:
        return json.loads(self.body)


class Conn:
    """One accepted connection plus helpers for writing scripted responses."""

    def __init__(self, request: Recorded, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.request = request
        self.reader = reader
        self.writer = writer
        self.client_disconnected = asyncio.Event()
        self._watcher = asyncio.ensure_future(self._watch())

    async def _watch(self) -> None:
        try:
            while await self.reader.read(1024):
                pass
        except (ConnectionError, OSError):
            pass
        self.client_disconnected.set()

    async def raw(self, data: bytes) -> None:
        self.writer.write(data)
        await self.writer.drain()

    async def respond_json(self, status: int, obj: Any, headers: dict[str, str] | None = None) -> None:
        body = json.dumps(obj).encode()
        await self.respond_bytes(status, body, "application/json", headers)

    async def respond_bytes(
        self, status: int, body: bytes, content_type: str = "application/json", headers: dict[str, str] | None = None
    ) -> None:
        head = [f"HTTP/1.1 {status} X", f"Content-Type: {content_type}", f"Content-Length: {len(body)}", "Connection: close"]
        head += [f"{k}: {v}" for k, v in (headers or {}).items()]
        await self.raw(("\r\n".join(head) + "\r\n\r\n").encode() + body)

    async def start_stream(self, status: int = 200) -> None:
        await self.raw(
            f"HTTP/1.1 {status} X\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n".encode()
        )

    async def send(self, data: bytes) -> None:
        """One HTTP chunk. Bytes are flushed immediately so the client sees this boundary."""
        await self.raw(f"{len(data):x}\r\n".encode() + data + b"\r\n")

    async def end_stream(self) -> None:
        await self.raw(b"0\r\n\r\n")

    def abort(self) -> None:
        """Drop the TCP connection without terminating the chunked body."""
        self.writer.transport.abort()

    async def play(self, script: list[bytes | float], *, end: str = "terminate") -> None:
        """Send a stream script: bytes = one HTTP chunk, float = pause seconds.
        end: 'terminate' (clean end), 'abort' (disconnect mid-body), 'hang' (wait for client)."""
        await self.start_stream()
        for step in script:
            if isinstance(step, float):
                await asyncio.sleep(step)
            else:
                await self.send(step)
        if end == "terminate":
            await self.end_stream()
        elif end == "abort":
            self.abort()
        elif end == "hang":
            await self.client_disconnected.wait()

    async def stream_forever(self, interval: float = 0.01) -> None:
        """Endless synthetic tokens until the client goes away (records that it noticed)."""
        await self.start_stream()
        try:
            while not self.client_disconnected.is_set():
                await self.send(chunk("tok "))
                await asyncio.sleep(interval)
        except (ConnectionError, OSError):
            self.client_disconnected.set()

    def close(self) -> None:
        self._watcher.cancel()


Handler = Callable[[Conn], Awaitable[None]]


class FakeOllama:
    def __init__(self) -> None:
        self.requests: list[Recorded] = []
        self.routes: dict[tuple[str, str], Handler] = {}
        self.total_connections = 0
        self.open_connections = 0
        self.last_conn: Conn | None = None
        self._server: asyncio.AbstractServer | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self.port = 0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def paths(self) -> list[str]:
        return [r.path for r in self.requests]

    @property
    def chat_requests(self) -> list[Recorded]:
        return [r for r in self.requests if r.path == "/api/chat"]

    def on(self, method: str, path: str, handler: Handler) -> None:
        self.routes[(method, path)] = handler

    def json_route(self, method: str, path: str, obj: Any, status: int = 200) -> None:
        async def h(c: Conn) -> None:
            await c.respond_json(status, obj)

        self.on(method, path, h)

    async def __aenter__(self) -> FakeOllama:
        self._server = await asyncio.start_server(self._accept, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        # Chat calls preflight /api/tags (local-only guard); by default the synthetic tag is local.
        self.json_route("GET", "/api/tags", {"models": [{"name": SYNTH_TAG, "digest": "sha256:synthetic"}]})
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._server is not None
        self._server.close()
        for t in list(self._tasks):
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._server.wait_closed()

    def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.ensure_future(self._serve(reader, writer))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.total_connections += 1
        self.open_connections += 1
        conn: Conn | None = None
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head.decode("latin-1").split("\r\n")
            method, path, _ = lines[0].split(" ", 2)
            headers = {k.lower(): v.strip() for k, _, v in (ln.partition(":") for ln in lines[1:] if ln)}
            body = await reader.readexactly(int(headers.get("content-length", "0")))
            rec = Recorded(method, path, headers, body)
            self.requests.append(rec)
            conn = Conn(rec, reader, writer)
            self.last_conn = conn
            handler = self.routes.get((method, path))
            if handler is None:
                await conn.respond_json(404, {"error": "synthetic: no such route"})
            else:
                await handler(conn)
        except (asyncio.IncompleteReadError, ConnectionError, OSError, asyncio.CancelledError):
            pass
        finally:
            if conn is not None:
                conn.close()
            writer.close()
            self.open_connections -= 1

    async def wait_connections_closed(self, timeout: float = 2.0) -> None:
        async def poll() -> None:
            while self.open_connections:
                await asyncio.sleep(0.005)

        await asyncio.wait_for(poll(), timeout)


class Canary(FakeOllama):
    """A second loopback listener that must receive NO connections (redirect / proxy targets)."""


# ------------------------------------------------------------------------------------------------


@dataclass
class SyntheticProvider:
    """In-memory ModelProvider: proves downstream code can be written against the Protocol."""

    installed_tags: tuple[str, ...] = (SYNTH_TAG,)
    loaded_tags: tuple[str, ...] = ()
    reply: tuple[str, ...] = ("hello ", "world")
    closed: bool = False
    calls: list[str] = field(default_factory=list)

    async def list_models(self, *, cancel: asyncio.Event | None = None) -> InstalledModels:
        from datetime import datetime, timezone

        self.calls.append("list_models")
        return InstalledModels(tuple(InstalledModel(t) for t in self.installed_tags), datetime.now(timezone.utc))

    async def list_loaded(self, *, cancel: asyncio.Event | None = None) -> LoadedModels:
        from datetime import datetime, timezone

        self.calls.append("list_loaded")
        return LoadedModels(True, tuple(LoadedModel(t) for t in self.loaded_tags), datetime.now(timezone.utc))

    def chat_stream(
        self, request: ChatRequest, *, timeouts: ProviderTimeouts | None = None, cancel: asyncio.Event | None = None
    ) -> SyntheticStream:
        self.calls.append("chat_stream")
        return SyntheticStream(self.reply, cancel)

    async def chat_structured(
        self, request: ChatRequest, schema: type, *, timeouts: ProviderTimeouts | None = None, cancel: asyncio.Event | None = None
    ) -> StructuredResult:
        self.calls.append("chat_structured")
        tel = StreamTelemetry(lambda: 0.0)
        tel.start()
        tel.finish(None)
        return StructuredResult(schema.model_validate({}), "stop", request.model, tel.snapshot())

    async def aclose(self) -> None:
        self.closed = True


class SyntheticStream:
    def __init__(self, reply: tuple[str, ...], cancel: asyncio.Event | None) -> None:
        self.telemetry = StreamTelemetry(lambda: 0.0)
        self.telemetry.start()
        self._cancel = cancel or asyncio.Event()
        self._items = [*(ChatDelta(t) for t in reply), "final"]

    def __aiter__(self) -> SyntheticStream:
        return self

    async def __anext__(self) -> Any:
        if self._cancel.is_set():
            self.telemetry.finish(None)
            if not self._items:
                raise StopAsyncIteration
            self._items = []
            return ChatCancelled(self.telemetry.snapshot())
        if not self._items:
            raise StopAsyncIteration
        item = self._items.pop(0)
        if item == "final":
            self.telemetry.finish(None)
            return ChatFinished("stop", None, self.telemetry.snapshot())
        self.telemetry.on_chunk(item.content)
        return item

    def cancel(self) -> None:
        self._cancel.set()

    def progress(self) -> Any:
        return self.telemetry.progress()

    async def aclose(self) -> None:
        self._cancel.set()
