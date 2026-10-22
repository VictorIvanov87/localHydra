"""LH-002: loopback-only enforcement. All servers are SYNTHETIC listeners on 127.0.0.1.

Non-loopback addresses used below (203.0.113.0/24, 198.51.100.0/24) are RFC 5737 documentation
ranges; the tests assert they are never connected to (the adapter refuses before connecting).
"""

from __future__ import annotations

import asyncio
import re
import socket
from pathlib import Path

import pytest

from localhydra.adapters import (
    ChatMessage,
    ChatRequest,
    LoopbackTarget,
    OllamaProvider,
    ProviderError,
    ProviderErrorKind,
)
from localhydra.contracts.errors import ErrorCode

from .provider_fakes import SYNTH_TAG, Canary, Conn, FakeOllama, anyio_backend, chunk, final  # noqa: F401

pytestmark = pytest.mark.anyio

REQ = ChatRequest(model=SYNTH_TAG, messages=(ChatMessage(role="user", content="hi"),))
TAGS = {"models": [{"name": SYNTH_TAG, "digest": "sha256:synthetic", "size": 1}]}
EXTERNAL = "203.0.113.7"


class Resolve:
    """Scripted resolver; records the calls."""

    def __init__(self, *answers: list[str] | Exception) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, int]] = []

    async def __call__(self, host: str, port: int) -> list[str]:
        self.calls.append((host, port))
        answer = self.answers[min(len(self.calls), len(self.answers)) - 1]
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
async def fake():
    async with FakeOllama() as f:
        f.json_route("GET", "/api/tags", TAGS)
        yield f


@pytest.fixture
async def canary():
    async with Canary() as c:
        yield c


# ---- URL validation (contract helper + the gaps it leaves) ------------------------------------


@pytest.mark.parametrize(
    "url,host,port",
    [
        ("http://127.0.0.1:11434", "127.0.0.1", 11434),
        ("http://localhost:11434", "localhost", 11434),
        ("http://LOCALHOST:11434", "localhost", 11434),
        ("http://[::1]:11434", "::1", 11434),
        ("http://127.0.0.2:5", "127.0.0.2", 5),
        ("https://localhost", "localhost", 443),
        ("http://127.0.0.1", "127.0.0.1", 80),
    ],
)
def test_accepts_loopback_origins(url: str, host: str, port: int) -> None:
    t = LoopbackTarget.parse(url)
    assert (t.host, t.port) == (host, port)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "http://192.168.1.5:11434",
        "http://10.0.0.1:11434",
        "http://0.0.0.0:11434",
        "http://127.0.0.1.evil.example:11434",
        "http://localhost.evil.example:11434",
        "http://evil.example#@127.0.0.1",
        "http://user:pw@127.0.0.1:11434",
        "http://localhost@evil.example",
        "ftp://127.0.0.1",
        "file:///etc/passwd",
        "http://127.0.0.1:99999",  # port range is not checked by the contract helper
        "http://127.0.0.1:abc",  # nor is a non-numeric port
        "http://127.0.0.1:0",
        "http://127.0.0.1/api",
        "http://127.0.0.1?x=1",
        "http://127.1:11434",  # shorthand IPv4 is not a canonical literal: rejected, not guessed
        "http://0x7f.0.0.1:11434",
        "http://localhost.:11434",
        "http://[::ffff:203.0.113.7]:11434",
        "http://",
        "",
    ],
)
def test_rejects_non_loopback_or_malformed_origins(url: str) -> None:
    with pytest.raises(ProviderError) as ei:
        LoopbackTarget.parse(url)
    assert ei.value.kind is ProviderErrorKind.target_not_loopback
    assert ei.value.code is ErrorCode.target_not_loopback


def test_provider_constructor_refuses_external_target_without_any_io() -> None:
    with pytest.raises(ProviderError) as ei:
        OllamaProvider("http://example.com:11434")
    assert ei.value.code is ErrorCode.target_not_loopback


# ---- resolution, pinning, rebinding ------------------------------------------------------------


async def test_ip_literal_never_calls_the_resolver(fake: FakeOllama) -> None:
    resolver = Resolve(["203.0.113.7"])
    async with OllamaProvider(fake.base_url, resolver=resolver) as p:
        assert (await p.list_models()).tags == {SYNTH_TAG}
    assert resolver.calls == []


async def test_localhost_is_resolved_validated_and_pinned(fake: FakeOllama, monkeypatch: pytest.MonkeyPatch) -> None:
    looked_up: list[object] = []
    loop = asyncio.get_running_loop()
    real = loop.getaddrinfo

    async def spy(host, *a, **kw):  # noqa: ANN001
        # the stack passes bytes hosts (verified by mutating the adapter to connect by name)
        looked_up.append(host.decode() if isinstance(host, bytes) else host)
        return await real(host, *a, **kw)

    monkeypatch.setattr(loop, "getaddrinfo", spy)
    resolver = Resolve(["127.0.0.1"])
    async with OllamaProvider(f"http://localhost:{fake.port}", resolver=resolver) as p:
        await p.list_models()
        await p.list_models()
    assert resolver.calls == [("localhost", fake.port)] * 2  # re-validated on every request
    assert "localhost" not in looked_up  # the HTTP stack connected to the pinned IP, no 2nd lookup
    assert fake.requests[0].headers["host"] == f"localhost:{fake.port}"  # original authority kept


@pytest.mark.parametrize(
    "answer",
    [
        [EXTERNAL],
        ["127.0.0.1", EXTERNAL],  # one bad address poisons the whole answer
        [EXTERNAL, "127.0.0.1"],
        ["::ffff:203.0.113.7"],
        ["fe80::1%lo0"],
        ["10.0.0.1"],
    ],
)
async def test_external_resolution_is_rejected_before_any_connection(fake: FakeOllama, answer: list[str]) -> None:
    resolver = Resolve(answer)
    async with OllamaProvider(f"http://localhost:{fake.port}", resolver=resolver) as p:
        with pytest.raises(ProviderError) as ei:
            await p.list_models()
        with pytest.raises(ProviderError):
            [x async for x in p.chat_stream(REQ)]
        with pytest.raises(ProviderError):
            await p.list_loaded()
    assert ei.value.kind is ProviderErrorKind.target_not_loopback
    assert ei.value.code is ErrorCode.target_not_loopback
    assert not ei.value.retryable
    assert fake.total_connections == 0 and fake.requests == []


async def test_rebinding_after_first_request_is_caught_on_the_next(fake: FakeOllama) -> None:
    resolver = Resolve(["127.0.0.1"], [EXTERNAL])
    async with OllamaProvider(f"http://localhost:{fake.port}", resolver=resolver) as p:
        await p.list_models()
        with pytest.raises(ProviderError) as ei:
            await p.list_models()
    assert ei.value.kind is ProviderErrorKind.target_not_loopback
    assert len(fake.requests) == 1


@pytest.mark.parametrize("answer", [[], OSError(8, "synthetic gaierror")])
async def test_unresolvable_host_is_a_dns_failure(fake: FakeOllama, answer: object) -> None:
    resolver = Resolve(answer)  # type: ignore[arg-type]
    async with OllamaProvider(f"http://localhost:{fake.port}", resolver=resolver) as p:
        with pytest.raises(ProviderError) as ei:
            await p.list_models()
    assert ei.value.kind is ProviderErrorKind.dns_failure
    assert ei.value.code is ErrorCode.provider_unreachable and ei.value.retryable


async def test_tries_next_loopback_address_when_first_refuses(fake: FakeOllama) -> None:
    # fake listens on IPv4 only; ::1 (or an unsupported v6 stack) must fall through to 127.0.0.1
    resolver = Resolve(["::1", "127.0.0.1"])
    async with OllamaProvider(f"http://localhost:{fake.port}", resolver=resolver) as p:
        assert (await p.list_models()).tags == {SYNTH_TAG}


# ---- redirects and proxies ---------------------------------------------------------------------


@pytest.mark.parametrize("status", [301, 302, 307, 308])
async def test_redirects_are_refused_not_followed(fake: FakeOllama, canary: Canary, status: int) -> None:
    async def redirect(c: Conn) -> None:
        await c.respond_bytes(status, b"", headers={"Location": f"{canary.base_url}/api/tags"})

    fake.on("GET", "/api/tags", redirect)
    fake.on("POST", "/api/chat", redirect)
    async with OllamaProvider(fake.base_url) as p:
        with pytest.raises(ProviderError) as ei:
            await p.list_models()
        assert ei.value.kind is ProviderErrorKind.redirect_refused
        assert ei.value.details["status"] == status
        # chat's local-only inventory preflight hits the redirecting /api/tags first: also refused
        with pytest.raises(ProviderError) as ei2:
            [x async for x in p.chat_stream(REQ)]
        assert ei2.value.kind is ProviderErrorKind.redirect_refused
        fake.json_route("GET", "/api/tags", TAGS)  # now only the chat endpoint redirects
        with pytest.raises(ProviderError) as ei3:
            [x async for x in p.chat_stream(REQ)]
        assert ei3.value.kind is ProviderErrorKind.redirect_refused
    assert canary.total_connections == 0 and canary.requests == []
    assert fake.paths == ["/api/tags", "/api/tags", "/api/tags", "/api/chat"]


async def test_redirect_to_an_external_address_is_not_followed(fake: FakeOllama) -> None:
    async def redirect(c: Conn) -> None:
        await c.respond_bytes(302, b"", headers={"Location": f"http://{EXTERNAL}/api/tags"})

    fake.on("GET", "/api/tags", redirect)
    async with OllamaProvider(fake.base_url, timeouts=None) as p:
        with pytest.raises(ProviderError) as ei:
            await p.list_models()
    assert ei.value.kind is ProviderErrorKind.redirect_refused
    assert EXTERNAL not in str(ei.value.details) and EXTERNAL not in ei.value.message


async def test_environment_proxies_are_ignored(fake: FakeOllama, canary: Canary, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.setenv(var, canary.base_url)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    async with OllamaProvider(fake.base_url) as p:
        assert (await p.list_models()).tags == {SYNTH_TAG}
    assert canary.total_connections == 0
    assert len(fake.requests) == 1


# ---- no unexpected requests, ever ---------------------------------------------------------------


async def test_only_tags_ps_and_chat_are_ever_requested_and_a_missing_tag_is_never_pulled(fake: FakeOllama) -> None:
    fake.json_route("GET", "/api/ps", {"models": []})

    async def missing(c: Conn) -> None:
        await c.respond_json(404, {"error": "model 'example/model-1:synthetic' not found"})

    fake.on("POST", "/api/chat", missing)
    async with OllamaProvider(fake.base_url) as p:
        await p.list_models()
        await p.list_loaded()
        with pytest.raises(ProviderError) as ei:
            [x async for x in p.chat_stream(REQ)]
        assert ei.value.kind is ProviderErrorKind.model_not_found
    assert [(r.method, r.path) for r in fake.requests] == [
        ("GET", "/api/tags"),
        ("GET", "/api/ps"),
        ("POST", "/api/chat"),
    ]


def test_adapter_source_has_no_pull_or_external_url_paths() -> None:
    src = Path(__file__).resolve().parents[1] / "src" / "localhydra" / "adapters"
    text = "\n".join(p.read_text() for p in src.glob("*.py"))
    for forbidden in ("/api/pull", "/api/create", "/api/push", "/api/delete", "/api/copy", "/api/embed", "/api/blobs"):
        assert forbidden not in text, forbidden
    assert re.findall(r"https?://", text) == [], "adapters must not contain literal URLs"
    assert "trust_env=False" in text and "follow_redirects=False" in text


async def test_closed_provider_refuses_work(fake: FakeOllama) -> None:
    p = OllamaProvider(fake.base_url)
    await p.aclose()
    await p.aclose()  # idempotent
    with pytest.raises(ProviderError) as ei:
        await p.list_models()
    assert ei.value.kind is ProviderErrorKind.closed
    assert fake.total_connections == 0
