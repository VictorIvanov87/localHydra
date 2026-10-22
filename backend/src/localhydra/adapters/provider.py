"""Provider-neutral interface, request/result types and timeouts.

Nothing here knows about routing, run lifecycle or the UI. Downstream code depends on the
`ModelProvider` protocol so tests (and LH-004) can inject any implementation.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generic, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ..contracts.common import OllamaTag
from ..contracts.metrics import ProgressMetrics
from .telemetry import StreamTelemetry, TelemetrySnapshot

T = TypeVar("T", bound=BaseModel)


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChatMessage(_Frozen):
    role: Literal["system", "user", "assistant"]
    content: str


class GenerationOptions(_Frozen):
    """Passed to Ollama as `options` exactly as set; unset fields are omitted, never defaulted."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    num_ctx: int | None = Field(default=None, ge=1)
    num_predict: int | None = Field(default=None, ge=-2)
    stop: tuple[str, ...] | None = None

    def to_wire(self) -> dict[str, Any]:
        wire = self.model_dump(exclude_none=True)
        if "stop" in wire:
            wire["stop"] = list(wire["stop"])
        return wire


class ChatRequest(_Frozen):
    """`model` must be an exact `name:tag` (bare names are rejected, `:latest` is never added)."""

    model: OllamaTag
    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    options: GenerationOptions = Field(default_factory=GenerationOptions)
    keep_alive: str | int | None = None
    think: bool | None = None


@dataclass(frozen=True)
class ProviderTimeouts:
    """Seconds. Operational defaults, not capability claims; override per provider or call."""

    connect_s: float = 3.0
    request_s: float = 10.0  # discovery / loaded-state requests
    first_chunk_s: float = 120.0  # response headers + first chunk (covers model load and prefill)
    idle_s: float = 30.0  # max gap between stream chunks after the first
    total_s: float | None = 600.0  # whole call; None disables the overall deadline


@dataclass(frozen=True)
class InstalledModel:
    tag: str  # exactly as Ollama reports it
    digest: str | None = None
    size_bytes: int | None = None
    modified_at: str | None = None
    # Set by Ollama for cloud-proxied tags (e.g. "name:cloud"): the weights are NOT local and
    # generation would be sent to that remote host. Reported verbatim; never used for inference.
    remote_host: str | None = None
    remote_model: str | None = None

    @property
    def is_remote(self) -> bool:
        return self.remote_host is not None or self.remote_model is not None


@dataclass(frozen=True)
class InstalledModels:
    models: tuple[InstalledModel, ...]
    observed_at: datetime

    @property
    def tags(self) -> frozenset[str]:
        """Every listed tag, including remote ones (verbatim inventory)."""
        return frozenset(m.tag for m in self.models)

    @property
    def local_tags(self) -> frozenset[str]:
        return frozenset(m.tag for m in self.models if not m.is_remote)

    @property
    def remote_tags(self) -> frozenset[str]:
        return frozenset(m.tag for m in self.models if m.is_remote)


@dataclass(frozen=True)
class LoadedModel:
    tag: str
    digest: str | None = None
    expires_at: str | None = None


@dataclass(frozen=True)
class LoadedModels:
    """`supported=False` means the server has no loaded-state endpoint: state is unknown,
    which is different from 'nothing is loaded'."""

    supported: bool
    models: tuple[LoadedModel, ...]
    observed_at: datetime

    @property
    def tags(self) -> frozenset[str] | None:
        return frozenset(m.tag for m in self.models) if self.supported else None


@dataclass(frozen=True)
class ChatDelta:
    content: str  # may be empty when only `thinking` is present
    thinking: str = ""


@dataclass(frozen=True)
class ChatFinished:
    done_reason: str | None
    served_model: str | None
    telemetry: TelemetrySnapshot


@dataclass(frozen=True)
class ChatCancelled:
    """The local stream was closed on request. Whether Ollama stopped computing is not known:
    closing the connection is the only signal sent, so `upstream_stop_confirmed` is always False."""

    telemetry: TelemetrySnapshot
    upstream_stop_confirmed: bool = False


ChatItem = ChatDelta | ChatFinished | ChatCancelled


@dataclass(frozen=True)
class StructuredResult(Generic[T]):
    parsed: T
    done_reason: str | None
    served_model: str | None
    telemetry: TelemetrySnapshot  # ttft_ms is None: the response is not streamed


@runtime_checkable
class ChatStream(Protocol):
    """Async iterator of `ChatItem`; ends after exactly one terminal item or raises ProviderError."""

    telemetry: StreamTelemetry

    def __aiter__(self) -> AsyncIterator[ChatItem]: ...
    async def __anext__(self) -> ChatItem: ...
    def cancel(self) -> None: ...
    async def aclose(self) -> None: ...
    def progress(self) -> ProgressMetrics: ...


@runtime_checkable
class ModelProvider(Protocol):
    async def list_models(self, *, cancel: asyncio.Event | None = None) -> InstalledModels: ...

    async def list_loaded(self, *, cancel: asyncio.Event | None = None) -> LoadedModels: ...

    def chat_stream(
        self,
        request: ChatRequest,
        *,
        timeouts: ProviderTimeouts | None = None,
        cancel: asyncio.Event | None = None,
    ) -> ChatStream: ...

    async def chat_structured(
        self,
        request: ChatRequest,
        schema: type[T],
        *,
        timeouts: ProviderTimeouts | None = None,
        cancel: asyncio.Event | None = None,
    ) -> StructuredResult[T]: ...

    async def aclose(self) -> None: ...
