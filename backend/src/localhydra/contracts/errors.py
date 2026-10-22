"""Consistent error envelope used by every non-2xx response and by failed runs."""

from __future__ import annotations

from enum import Enum
from typing import Any

from .common import Contract


class ErrorCode(str, Enum):
    validation_error = "validation_error"
    bad_request = "bad_request"
    not_found = "not_found"
    run_not_found = "run_not_found"
    conversation_not_found = "conversation_not_found"
    conflict = "conflict"
    conversation_busy = "conversation_busy"
    invalid_cursor = "invalid_cursor"
    invalid_last_event_id = "invalid_last_event_id"
    host_not_allowed = "host_not_allowed"
    no_eligible_model = "no_eligible_model"
    model_unavailable = "model_unavailable"
    model_override_ineligible = "model_override_ineligible"
    context_too_large = "context_too_large"
    target_not_loopback = "target_not_loopback"
    provider_unreachable = "provider_unreachable"
    provider_error = "provider_error"
    stream_interrupted = "stream_interrupted"
    internal_error = "internal_error"


# HTTP status when the error is returned as a response. When the same code is carried
# inside a failed run (RunDetail.error / run.failed event) there is no HTTP status.
ERROR_HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.validation_error: 422,
    ErrorCode.bad_request: 400,
    ErrorCode.not_found: 404,
    ErrorCode.run_not_found: 404,
    ErrorCode.conversation_not_found: 404,
    ErrorCode.conflict: 409,
    ErrorCode.conversation_busy: 409,
    ErrorCode.invalid_cursor: 400,
    ErrorCode.invalid_last_event_id: 400,
    ErrorCode.host_not_allowed: 400,
    ErrorCode.no_eligible_model: 422,
    ErrorCode.model_unavailable: 422,
    ErrorCode.model_override_ineligible: 422,
    ErrorCode.context_too_large: 422,
    ErrorCode.target_not_loopback: 500,
    ErrorCode.provider_unreachable: 502,
    ErrorCode.provider_error: 502,
    ErrorCode.stream_interrupted: 502,
    ErrorCode.internal_error: 500,
}


class ErrorBody(Contract):
    code: ErrorCode
    message: str
    retryable: bool = False
    # Must never echo prompt/response content (privacy defaults); field paths and codes only.
    details: dict[str, Any] | None = None
    request_id: str | None = None


class ErrorEnvelope(Contract):
    error: ErrorBody
