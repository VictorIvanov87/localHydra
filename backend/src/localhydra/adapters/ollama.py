"""Async Ollama adapter: discovery, loaded state, streaming chat, structured output.

Guarantees (each is covered by tests against synthetic local fake servers):
- Only loopback targets. The host is resolved by us per request, every resolved address must be
  loopback, and the connection goes to the validated IP literal (see `loopback.py`).
- Redirects are never followed (a 3xx is a typed error) and environment proxies are ignored
  (`trust_env=False`), so a request can only reach the validated address.
- The only endpoints ever requested are GET /api/tags, GET /api/ps and POST /api/chat. There is
  no pull/create/delete/embedding code path: a missing tag is a typed error, never a download.
- Model tags and generation options are forwarded exactly; nothing is defaulted or rewritten.
- Errors are typed (`ProviderError`) and never contain prompt or response text.

Cancellation: setting the cancel event (or `ChatStream.cancel()` / cancelling the consuming
task) closes the HTTP connection promptly and releases client resources. Closing the connection
is the only signal Ollama receives; that it stops computing is expected but NOT confirmed here
(`upstream_stop_confirmed` is always False).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import functools
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from datetime import datetime, timezone
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from ..contracts.config import DEFAULT_OLLAMA_URL
from ..contracts.errors import ErrorCode
from ..contracts.metrics import ProgressMetrics
from .errors import ProviderCancelled, ProviderError, ProviderErrorKind
from .loopback import LoopbackTarget, Resolver, system_resolver
from .provider import (
    ChatCancelled,
    ChatDelta,
    ChatFinished,
    ChatItem,
    ChatRequest,
    InstalledModel,
    InstalledModels,
    LoadedModel,
    LoadedModels,
    ProviderTimeouts,
    StructuredResult,
)
from .telemetry import StreamTelemetry, parse_raw_metrics

T = TypeVar("T", bound=BaseModel)

MAX_LINE_BYTES = 8 * 1024 * 1024
MAX_BODY_BYTES = 16 * 1024 * 1024
_RETRYABLE_STATUS = {408, 429, 502, 503, 504}


class _CancelRequested(Exception):
    pass


async def _guarded(aw: Awaitable[Any], *, timeout: float | None, cancel: asyncio.Event | None) -> Any:
    """Await `aw`, but stop early on timeout or when `cancel` is set. A result that is already
    available wins over a simultaneous cancel (callers re-check the event)."""
    task = asyncio.ensure_future(aw)
    cancel_task = asyncio.ensure_future(cancel.wait()) if cancel is not None else None
    waiters = {task} if cancel_task is None else {task, cancel_task}
    try:
        done, _ = await asyncio.wait(waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            return task.result()
        if cancel_task is not None and cancel_task in done:
            raise _CancelRequested
        raise TimeoutError
    finally:
        for t in waiters:
            if not t.done():
                t.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)


def _malformed(reason: str, **details: Any) -> ProviderError:
    return ProviderError(
        ProviderErrorKind.malformed_response,
        "provider sent a malformed response",
        details={"reason": reason, **details},
    )


def _map_httpx_error(exc: httpx.HTTPError, *, midstream: bool) -> ProviderError:
    name = type(exc).__name__
    if isinstance(exc, httpx.ConnectTimeout):
        return ProviderError(ProviderErrorKind.connect_timeout, "timed out connecting to Ollama")
    if isinstance(exc, httpx.ConnectError):
        return ProviderError(
            ProviderErrorKind.offline, "could not connect to the local Ollama server (is it running?)"
        )
    if isinstance(exc, httpx.TimeoutException):
        kind = ProviderErrorKind.idle_timeout if midstream else ProviderErrorKind.request_timeout
        return ProviderError(kind, "timed out waiting for Ollama", details={"exception": name})
    if isinstance(exc, httpx.DecodingError):
        return _malformed("undecodable_body")
    if isinstance(exc, httpx.TransportError):
        return ProviderError(
            ProviderErrorKind.stream_interrupted,
            "connection to Ollama was lost",
            details={"exception": name},
        )
    return ProviderError(
        ProviderErrorKind.http_error, "unexpected HTTP client error", details={"exception": name}
    )


def _status_error(status: int, *, model_tag: str | None) -> ProviderError | None:
    if status == 200:
        return None
    if 300 <= status < 400:
        return ProviderError(
            ProviderErrorKind.redirect_refused,
            "provider answered with a redirect; redirects are never followed",
            details={"status": status},
        )
    if status == 404 and model_tag is not None:
        return ProviderError(
            ProviderErrorKind.model_not_found,
            "model tag is not installed in Ollama; localHydra never pulls models",
            details={"status": status, "model": model_tag},
        )
    return ProviderError(
        ProviderErrorKind.http_error,
        f"provider returned HTTP {status}",
        retryable=status in _RETRYABLE_STATUS,
        details={"status": status},
    )


def _decode_object(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise _malformed("invalid_utf8", bytes=len(raw)) from None
    try:
        obj = json.loads(text)
    except ValueError:
        raise _malformed("invalid_json", bytes=len(raw)) from None
    if not isinstance(obj, dict):
        raise _malformed("not_an_object", bytes=len(raw))
    return obj


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _interpret_chunk(obj: dict[str, Any]) -> tuple[str, str, bool, str | None, str | None]:
    """-> (content, thinking, done, done_reason, served_model). Never echoes content in errors."""
    if "error" in obj:
        raise ProviderError(
            ProviderErrorKind.midstream_error,
            "Ollama reported an error while generating",
        )
    done = obj.get("done")
    if not isinstance(done, bool):
        raise _malformed("missing_done_flag")
    message = obj.get("message")
    if message is None:
        if not done:
            raise _malformed("missing_message")
        content = thinking = ""
    elif isinstance(message, dict):
        content = message.get("content", "")
        thinking = message.get("thinking", "")
        if not isinstance(content, str) or not isinstance(thinking, str):
            raise _malformed("non_string_content")
    else:
        raise _malformed("message_not_an_object")
    return content, thinking, done, _optional_str(obj.get("done_reason")), _optional_str(obj.get("model"))


def _looks_remote_tag(model: str) -> bool:
    tag = model.rsplit(":", 1)[-1].lower()
    return tag == "cloud" or tag.endswith("-cloud")


def _remote_refused(model: str, how: str) -> ProviderError:
    return ProviderError(
        ProviderErrorKind.remote_model_refused,
        "model is served by a remote host (Ollama cloud); localHydra runs local inference only",
        details={"model": model, "detected_by": how},
    )


class OllamaProvider:
    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        *,
        timeouts: ProviderTimeouts | None = None,
        resolver: Resolver | None = None,
        clock: Callable[[], float] | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._target = LoopbackTarget.parse(base_url)  # raises ProviderError if not loopback
        self._timeouts = timeouts or ProviderTimeouts()
        self._resolver = resolver or system_resolver
        self._clock = clock or time.perf_counter
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._closed = False
        self._verified_local: set[str] = set()  # tags the inventory reported as non-remote
        # Read timeouts are enforced by `_guarded`; httpx only bounds connect/write/pool.
        c = self._timeouts.connect_s
        self._client = httpx.AsyncClient(
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(connect=c, read=None, write=c, pool=c),
            limits=httpx.Limits(max_keepalive_connections=0),
        )

    @property
    def target(self) -> LoopbackTarget:
        return self._target

    async def __aenter__(self) -> OllamaProvider:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            await self._client.aclose()

    def _check_open(self) -> None:
        if self._closed:
            raise ProviderError(ProviderErrorKind.closed, "provider is closed")

    # ---- transport -------------------------------------------------------------------------

    async def _guard(
        self,
        factory: Callable[[], Awaitable[Any]],
        *,
        stage_s: float | None,
        stage_kind: ProviderErrorKind,
        deadline: float | None,
        cancel: asyncio.Event | None,
        midstream: bool,
        total_kind: ProviderErrorKind = ProviderErrorKind.total_timeout,
    ) -> Any:
        remaining = None if deadline is None else deadline - asyncio.get_running_loop().time()
        total_binds = remaining is not None and (stage_s is None or remaining <= stage_s)
        if remaining is not None and remaining <= 0:
            raise self._timeout_error(total_kind, midstream)
        if stage_s is not None and stage_s <= 0:
            raise self._timeout_error(stage_kind, midstream)
        timeout = remaining if total_binds else stage_s
        try:
            return await _guarded(factory(), timeout=timeout, cancel=cancel)
        except TimeoutError:
            raise self._timeout_error(total_kind if total_binds else stage_kind, midstream) from None

    @staticmethod
    def _timeout_error(kind: ProviderErrorKind, midstream: bool) -> ProviderError:
        code = None
        if kind is ProviderErrorKind.total_timeout and not midstream:
            code = ErrorCode.provider_unreachable
        return ProviderError(kind, f"provider call exceeded its {kind.value.replace('_', ' ')} budget", code=code)

    async def _open(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
        *,
        accept: str,
        stage_s: float | None,
        stage_kind: ProviderErrorKind,
        deadline: float | None,
        cancel: asyncio.Event | None,
        total_kind: ProviderErrorKind = ProviderErrorKind.total_timeout,
    ) -> httpx.Response:
        """Resolve + validate (every request), then send to the pinned loopback IP."""
        self._check_open()
        if cancel is not None and cancel.is_set():
            raise _CancelRequested
        addrs = await self._guard(
            lambda: self._target.resolve(self._resolver),
            stage_s=self._timeouts.connect_s,
            stage_kind=ProviderErrorKind.connect_timeout,
            deadline=deadline,
            cancel=cancel,
            midstream=False,
            total_kind=total_kind,
        )
        headers = {
            "Host": self._target.authority,
            "Accept": accept,
            "Accept-Encoding": "identity",
        }
        content: bytes | None = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            content = json.dumps(body, ensure_ascii=False).encode("utf-8")
        extensions: dict[str, Any] = {}
        if self._target.scheme == "https" and not self._target.host_is_ip:
            extensions["sni_hostname"] = self._target.host
        last: ProviderError | None = None
        for ip in addrs:
            request = self._client.build_request(
                method,
                self._target.url_for(ip, path),
                headers=headers,
                content=content,
                extensions=extensions,
            )
            try:
                return await self._guard(
                    functools.partial(self._client.send, request, stream=True),
                    stage_s=stage_s,
                    stage_kind=stage_kind,
                    deadline=deadline,
                    cancel=cancel,
                    midstream=False,
                    total_kind=total_kind,
                )
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last = _map_httpx_error(exc, midstream=False)  # try the next loopback address
            except httpx.HTTPError as exc:
                raise _map_httpx_error(exc, midstream=False) from exc
        assert last is not None
        raise last

    async def _exchange(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
        *,
        stage_s: float | None,
        stage_kind: ProviderErrorKind,
        total_s: float | None,
        cancel: asyncio.Event | None,
        deadline: float | None = None,
        total_kind: ProviderErrorKind = ProviderErrorKind.total_timeout,
        model_tag: str | None = None,
        tolerate_404: bool = False,
    ) -> bytes | None:
        """Whole-body request. Returns the body, or None for a tolerated 404."""
        if deadline is None and total_s is not None:
            deadline = asyncio.get_running_loop().time() + total_s
        try:
            response = await self._open(
                method,
                path,
                body,
                accept="application/json",
                stage_s=stage_s,
                stage_kind=stage_kind,
                deadline=deadline,
                cancel=cancel,
                total_kind=total_kind,
            )
        except _CancelRequested:
            raise ProviderCancelled from None
        try:
            if tolerate_404 and response.status_code == 404:
                return None
            if (err := _status_error(response.status_code, model_tag=model_tag)) is not None:
                raise err
            data = bytearray()
            chunks = response.aiter_bytes().__aiter__()
            while True:
                try:
                    part = await self._guard(
                        chunks.__anext__,
                        stage_s=stage_s,
                        stage_kind=stage_kind,
                        deadline=deadline,
                        cancel=cancel,
                        midstream=True,
                        total_kind=total_kind,
                    )
                except StopAsyncIteration:
                    return bytes(data)
                except _CancelRequested:
                    raise ProviderCancelled from None
                except httpx.HTTPError as exc:
                    raise _map_httpx_error(exc, midstream=True) from exc
                data.extend(part)
                if len(data) > MAX_BODY_BYTES:
                    raise _malformed("body_too_large")
        finally:
            with contextlib.suppress(Exception):
                await response.aclose()

    # ---- discovery -------------------------------------------------------------------------

    async def list_models(self, *, cancel: asyncio.Event | None = None) -> InstalledModels:
        body = await self._exchange(
            "GET",
            "/api/tags",
            None,
            stage_s=self._timeouts.request_s,
            stage_kind=ProviderErrorKind.request_timeout,
            total_s=self._timeouts.request_s,
            total_kind=ProviderErrorKind.request_timeout,
            cancel=cancel,
        )
        assert body is not None
        out: list[InstalledModel] = []
        for entry in self._model_entries(_decode_object(body)):
            size = entry.get("size")
            out.append(
                InstalledModel(
                    tag=self._entry_name(entry),
                    digest=_optional_str(entry.get("digest")),
                    size_bytes=size if isinstance(size, int) and not isinstance(size, bool) and size >= 0 else None,
                    modified_at=_optional_str(entry.get("modified_at")),
                    remote_host=_optional_str(entry.get("remote_host")),
                    remote_model=_optional_str(entry.get("remote_model")),
                )
            )
        inventory = InstalledModels(models=tuple(out), observed_at=self._now())
        self._verified_local = set(inventory.local_tags)
        return inventory

    async def _ensure_local(
        self, model: str, cancel: asyncio.Event | None, *, deadline: float | None
    ) -> None:
        """Refuse Ollama-cloud tags: those would send the prompt to a remote host.

        Detection: the tag name (conservative: any `cloud` tag), and `remote_host` /
        `remote_model` in /api/tags (observed on a real Ollama; older servers may not expose
        them, in which case only the tag-name check applies). A tag that is not installed is
        left to /api/chat, which answers 404 (never pulled)."""
        if _looks_remote_tag(model):
            raise _remote_refused(model, "tag_name")
        if model in self._verified_local:
            return
        try:
            inventory = await self._guard(
                lambda: self.list_models(cancel=cancel),
                stage_s=None,
                stage_kind=ProviderErrorKind.total_timeout,
                deadline=deadline,
                cancel=cancel,
                midstream=False,
            )
        except _CancelRequested:
            raise ProviderCancelled from None
        for entry in inventory.models:
            if entry.tag == model and entry.is_remote:
                raise _remote_refused(model, "inventory")

    async def list_loaded(self, *, cancel: asyncio.Event | None = None) -> LoadedModels:
        body = await self._exchange(
            "GET",
            "/api/ps",
            None,
            stage_s=self._timeouts.request_s,
            stage_kind=ProviderErrorKind.request_timeout,
            total_s=self._timeouts.request_s,
            total_kind=ProviderErrorKind.request_timeout,
            cancel=cancel,
            tolerate_404=True,
        )
        if body is None:
            return LoadedModels(supported=False, models=(), observed_at=self._now())
        out = [
            LoadedModel(
                tag=self._entry_name(entry),
                digest=_optional_str(entry.get("digest")),
                expires_at=_optional_str(entry.get("expires_at")),
            )
            for entry in self._model_entries(_decode_object(body))
        ]
        return LoadedModels(supported=True, models=tuple(out), observed_at=self._now())

    @staticmethod
    def _model_entries(obj: dict[str, Any]) -> list[dict[str, Any]]:
        entries = obj.get("models")
        if entries is None:
            return []
        if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
            raise _malformed("models_not_a_list_of_objects")
        return entries

    @staticmethod
    def _entry_name(entry: dict[str, Any]) -> str:
        name = entry.get("name") or entry.get("model")
        if not isinstance(name, str) or not name:
            raise _malformed("model_entry_without_name")
        return name

    # ---- generation ------------------------------------------------------------------------

    @staticmethod
    def _chat_body(request: ChatRequest, *, stream: bool, fmt: dict[str, Any] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": stream,
        }
        if options := request.options.to_wire():
            body["options"] = options
        if request.keep_alive is not None:
            body["keep_alive"] = request.keep_alive
        if request.think is not None:
            body["think"] = request.think
        if fmt is not None:
            body["format"] = fmt
        return body

    def chat_stream(
        self,
        request: ChatRequest,
        *,
        timeouts: ProviderTimeouts | None = None,
        cancel: asyncio.Event | None = None,
    ) -> OllamaChatStream:
        return OllamaChatStream(self, request, timeouts or self._timeouts, cancel)

    async def chat_structured(
        self,
        request: ChatRequest,
        schema: type[T],
        *,
        timeouts: ProviderTimeouts | None = None,
        cancel: asyncio.Event | None = None,
    ) -> StructuredResult[T]:
        """Schema-constrained, non-streamed chat. The pydantic model's JSON schema is sent as
        Ollama's `format`, and the reply is validated against the same model. The result carries
        no ttft (nothing is streamed) and no response text."""
        to = timeouts or self._timeouts
        deadline = None if to.total_s is None else asyncio.get_running_loop().time() + to.total_s
        tel = StreamTelemetry(self._clock)
        await self._ensure_local(request.model, cancel, deadline=deadline)
        tel.start()
        try:
            body = await self._exchange(
                "POST",
                "/api/chat",
                self._chat_body(request, stream=False, fmt=schema.model_json_schema()),
                stage_s=None,
                stage_kind=ProviderErrorKind.total_timeout,
                total_s=to.total_s,
                deadline=deadline,
                cancel=cancel,
                model_tag=request.model,
            )
            assert body is not None
            obj = _decode_object(body)
            content, _thinking, _done, done_reason, served = _interpret_chunk({"done": True, **obj})
            raw, discarded = parse_raw_metrics(obj)
            try:
                parsed = schema.model_validate_json(content)
            except ValidationError as exc:
                errors = [
                    {"loc": [str(p) for p in e["loc"]], "type": e["type"]} for e in exc.errors()
                ][:20]
                raise ProviderError(
                    ProviderErrorKind.structured_output_invalid,
                    "model output did not satisfy the requested schema",
                    details={"done_reason": done_reason, "errors": errors},
                ) from None
            tel.finish(raw, discarded)
            return StructuredResult(
                parsed=parsed, done_reason=done_reason, served_model=served, telemetry=tel.snapshot()
            )
        finally:
            tel.finish(None)


class OllamaChatStream:
    """Async iterator of `ChatItem`; yields deltas, then exactly one terminal item
    (`ChatFinished` or `ChatCancelled`), or raises `ProviderError`."""

    def __init__(
        self,
        provider: OllamaProvider,
        request: ChatRequest,
        timeouts: ProviderTimeouts,
        cancel: asyncio.Event | None,
    ) -> None:
        self._provider = provider
        self._request = request
        self._timeouts = timeouts
        self._cancel = cancel if cancel is not None else asyncio.Event()
        self.telemetry = StreamTelemetry(provider._clock)
        self._gen = self._run()

    def __aiter__(self) -> AsyncIterator[ChatItem]:
        return self

    async def __anext__(self) -> ChatItem:
        item = await self._gen.__anext__()
        if isinstance(item, ChatFinished):
            # Run the generator's response cleanup before exposing its terminal yield.
            await self._gen.aclose()
        return item

    def cancel(self) -> None:
        self._cancel.set()

    def progress(self) -> ProgressMetrics:
        return self.telemetry.progress()

    async def aclose(self) -> None:
        self._cancel.set()
        with contextlib.suppress(RuntimeError):  # RuntimeError: another task is inside __anext__;
            await self._gen.aclose()  # it observes the cancel event and finishes by itself.

    @contextlib.asynccontextmanager
    async def _watch_suspended_cancel(self, response: httpx.Response) -> AsyncIterator[None]:
        """Release the response even if the consumer never resumes after a delta."""
        async def close_on_cancel() -> None:
            await self._cancel.wait()
            try:
                await response.aclose()
            finally:
                self.telemetry.finish(None)

        watcher = asyncio.create_task(close_on_cancel())
        try:
            yield
        finally:
            if not self._cancel.is_set():
                watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)

    def _error_details(self) -> dict[str, Any]:
        snap = self.telemetry.snapshot()
        return {"chunks_received": snap.chunk_count, "visible_chunks": snap.visible_chunks}

    def _handle_line(self, line: bytes) -> tuple[ChatDelta | None, ChatFinished | None]:
        obj = _decode_object(line)
        content, thinking, done, reason, served = _interpret_chunk(obj)
        self.telemetry.on_chunk(content)
        delta = ChatDelta(content, thinking) if (content or thinking) else None
        finished = None
        if done:
            raw, discarded = parse_raw_metrics(obj)
            self.telemetry.finish(raw, discarded)
            finished = ChatFinished(
                done_reason=reason, served_model=served, telemetry=self.telemetry.snapshot()
            )
        return delta, finished

    async def _run(self) -> AsyncGenerator[ChatItem, None]:
        p, tel, cancel, to = self._provider, self.telemetry, self._cancel, self._timeouts
        response: httpx.Response | None = None
        cancelled = False
        deadline = None if to.total_s is None else asyncio.get_running_loop().time() + to.total_s
        try:
            p._check_open()
            try:
                await p._ensure_local(self._request.model, cancel, deadline=deadline)
            except ProviderCancelled:
                raise _CancelRequested from None
            body = p._chat_body(self._request, stream=True)
            tel.start()
            first_deadline = asyncio.get_running_loop().time() + to.first_chunk_s
            first_binds = deadline is None or first_deadline < deadline
            response = await p._open(
                "POST",
                "/api/chat",
                body,
                accept="application/x-ndjson",
                stage_s=to.first_chunk_s,
                stage_kind=ProviderErrorKind.first_chunk_timeout,
                deadline=first_deadline if first_binds else deadline,
                total_kind=(
                    ProviderErrorKind.first_chunk_timeout if first_binds else ProviderErrorKind.total_timeout
                ),
                cancel=cancel,
            )
            if cancel.is_set():
                raise _CancelRequested
            if (err := _status_error(response.status_code, model_tag=self._request.model)) is not None:
                raise err
            buf = bytearray()
            received_bytes = False
            chunks = response.aiter_bytes().__aiter__()
            while True:
                if cancel.is_set():
                    raise _CancelRequested
                try:
                    data = await p._guard(
                        chunks.__anext__,
                        stage_s=(
                            to.idle_s if received_bytes else first_deadline - asyncio.get_running_loop().time()
                        ),
                        stage_kind=(
                            ProviderErrorKind.idle_timeout
                            if received_bytes
                            else ProviderErrorKind.first_chunk_timeout
                        ),
                        deadline=deadline,
                        cancel=cancel,
                        midstream=True,
                    )
                except StopAsyncIteration:
                    break
                except httpx.HTTPError as exc:
                    raise _map_httpx_error(exc, midstream=True) from exc
                received_bytes = True
                buf.extend(data)
                while (nl := buf.find(b"\n")) >= 0:
                    if nl > MAX_LINE_BYTES:
                        raise _malformed("line_too_long", **self._error_details())
                    line = bytes(buf[:nl])
                    del buf[: nl + 1]
                    if not line.strip():
                        continue
                    if cancel.is_set():
                        raise _CancelRequested
                    delta, finished = self._handle_line(line)
                    if delta is not None:
                        async with self._watch_suspended_cancel(response):
                            yield delta
                        if cancel.is_set():
                            raise _CancelRequested
                    if finished is not None:
                        yield finished
                        return
                if len(buf) > MAX_LINE_BYTES:
                    raise _malformed("line_too_long", **self._error_details())
            if buf.strip():
                try:
                    json.loads(bytes(buf))
                except ValueError:
                    raise ProviderError(
                        ProviderErrorKind.stream_interrupted,
                        "stream ended in the middle of a chunk",
                        details=self._error_details(),
                    ) from None
                delta, finished = self._handle_line(bytes(buf))
                if delta is not None:
                    async with self._watch_suspended_cancel(response):
                        yield delta
                    if cancel.is_set():
                        raise _CancelRequested
                if finished is not None:
                    yield finished
                    return
            raise ProviderError(
                ProviderErrorKind.stream_interrupted,
                "stream ended before Ollama's final chunk",
                details=self._error_details(),
            )
        except _CancelRequested:
            cancelled = True
        except ProviderError as err:
            for key, value in self._error_details().items():
                err.details.setdefault(key, value)
            raise
        finally:
            if response is not None:
                with contextlib.suppress(Exception):
                    await response.aclose()
            if not tel.finished:
                tel.finish(None)
        if cancelled:
            yield ChatCancelled(telemetry=tel.snapshot())
