"""Local Ollama provider adapter (LH-002). Transport only: no routing, orchestration or UI."""

from .discovery import ModelProbe, probe_models
from .errors import ProviderCancelled, ProviderError, ProviderErrorKind
from .loopback import LoopbackTarget, Resolver, system_resolver
from .ollama import OllamaChatStream, OllamaProvider
from .provider import (
    ChatCancelled,
    ChatDelta,
    ChatFinished,
    ChatItem,
    ChatMessage,
    ChatRequest,
    ChatStream,
    GenerationOptions,
    InstalledModel,
    InstalledModels,
    LoadedModel,
    LoadedModels,
    ModelProvider,
    ProviderTimeouts,
    StructuredResult,
)
from .telemetry import StreamTelemetry, TelemetrySnapshot, parse_raw_metrics

__all__ = [
    "ChatCancelled",
    "ChatDelta",
    "ChatFinished",
    "ChatItem",
    "ChatMessage",
    "ChatRequest",
    "ChatStream",
    "GenerationOptions",
    "InstalledModel",
    "InstalledModels",
    "LoadedModel",
    "LoadedModels",
    "LoopbackTarget",
    "ModelProbe",
    "ModelProvider",
    "OllamaChatStream",
    "OllamaProvider",
    "ProviderCancelled",
    "ProviderError",
    "ProviderErrorKind",
    "ProviderTimeouts",
    "Resolver",
    "StreamTelemetry",
    "StructuredResult",
    "TelemetrySnapshot",
    "parse_raw_metrics",
    "probe_models",
    "system_resolver",
]
