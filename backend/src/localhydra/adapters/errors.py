"""Typed provider errors. They map onto the frozen `ErrorCode`s; no new codes are introduced.

Errors never carry prompt or response text (privacy defaults): only kinds, statuses, counts and
field paths. Upstream error messages are deliberately not forwarded.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from ..contracts.errors import ErrorBody, ErrorCode


class ProviderErrorKind(str, Enum):
    target_not_loopback = "target_not_loopback"  # URL or any resolved address is not loopback
    dns_failure = "dns_failure"  # name did not resolve at all
    offline = "offline"  # connection refused / unreachable
    connect_timeout = "connect_timeout"
    request_timeout = "request_timeout"  # non-streaming request exceeded its budget
    first_chunk_timeout = "first_chunk_timeout"  # nothing arrived (load + prefill) in time
    idle_timeout = "idle_timeout"  # gap between stream chunks too long
    total_timeout = "total_timeout"  # overall deadline exceeded
    redirect_refused = "redirect_refused"  # any 3xx; redirects are never followed
    model_not_found = "model_not_found"  # tag not installed; the adapter never pulls
    remote_model_refused = "remote_model_refused"  # Ollama-cloud tag: inference would leave the machine
    http_error = "http_error"
    malformed_response = "malformed_response"
    midstream_error = "midstream_error"  # Ollama reported an error inside the stream
    stream_interrupted = "stream_interrupted"  # disconnect or EOF before the final chunk
    structured_output_invalid = "structured_output_invalid"
    closed = "closed"  # provider used after aclose()


_DEFAULT_CODE: dict[ProviderErrorKind, ErrorCode] = {
    ProviderErrorKind.target_not_loopback: ErrorCode.target_not_loopback,
    ProviderErrorKind.dns_failure: ErrorCode.provider_unreachable,
    ProviderErrorKind.offline: ErrorCode.provider_unreachable,
    ProviderErrorKind.connect_timeout: ErrorCode.provider_unreachable,
    ProviderErrorKind.request_timeout: ErrorCode.provider_unreachable,
    ProviderErrorKind.first_chunk_timeout: ErrorCode.provider_unreachable,
    ProviderErrorKind.idle_timeout: ErrorCode.stream_interrupted,
    ProviderErrorKind.total_timeout: ErrorCode.stream_interrupted,
    ProviderErrorKind.redirect_refused: ErrorCode.provider_error,
    ProviderErrorKind.model_not_found: ErrorCode.model_unavailable,
    ProviderErrorKind.remote_model_refused: ErrorCode.model_unavailable,
    ProviderErrorKind.http_error: ErrorCode.provider_error,
    ProviderErrorKind.malformed_response: ErrorCode.provider_error,
    ProviderErrorKind.midstream_error: ErrorCode.provider_error,
    ProviderErrorKind.stream_interrupted: ErrorCode.stream_interrupted,
    ProviderErrorKind.structured_output_invalid: ErrorCode.provider_error,
    ProviderErrorKind.closed: ErrorCode.internal_error,
}

_RETRYABLE = {
    ProviderErrorKind.dns_failure,
    ProviderErrorKind.offline,
    ProviderErrorKind.connect_timeout,
    ProviderErrorKind.request_timeout,
    ProviderErrorKind.first_chunk_timeout,
    ProviderErrorKind.idle_timeout,
    ProviderErrorKind.total_timeout,
    ProviderErrorKind.stream_interrupted,
}


class ProviderError(Exception):
    def __init__(
        self,
        kind: ProviderErrorKind,
        message: str,
        *,
        code: ErrorCode | None = None,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.code = code or _DEFAULT_CODE[kind]
        self.retryable = (kind in _RETRYABLE) if retryable is None else retryable
        self.details: dict[str, Any] = {"kind": kind.value, **(details or {})}

    def to_error_body(self, request_id: str | None = None) -> ErrorBody:
        return ErrorBody(
            code=self.code,
            message=self.message,
            retryable=self.retryable,
            details=self.details,
            request_id=request_id,
        )

    def __repr__(self) -> str:
        return f"ProviderError(kind={self.kind.value}, code={self.code.value}, message={self.message!r})"


class ProviderCancelled(Exception):
    """A non-streaming provider call was cancelled through its cancel event.

    Streaming calls report cancellation as a terminal `ChatCancelled` item instead.
    """

    upstream_stop_confirmed = False
