"""Frozen HTTP surface: health response and the endpoint registry exported to contracts/schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .common import Contract, UtcDatetime
from .errors import ErrorCode


class HealthResponse(Contract):
    status: Literal["ok"] = "ok"
    service: Literal["localhydra"] = "localhydra"
    version: str
    contract_version: str
    event_schema_version: int
    local_only: Literal[True] = True
    bind_host: str
    ollama_target: str
    models_configured: int = Field(ge=0)
    time: UtcDatetime


class Endpoint(Contract):
    method: Literal["GET", "POST"]
    path: str
    summary: str
    request_schema: str | None = Field(default=None, description="Body (POST) or query (GET) schema name.")
    response_schema: str | None = None
    content_type: str = "application/json"
    success_status: int = 200
    error_codes: list[ErrorCode] = Field(default_factory=list)
    implemented_by: str = Field(description="Ticket that implements the handler.")


_E = ErrorCode

ENDPOINTS: list[Endpoint] = [
    Endpoint(method="GET", path="/api/health", summary="Liveness and local-only boundary",
             response_schema="HealthResponse", implemented_by="LH-001"),
    Endpoint(method="GET", path="/api/models", summary="Configured model profiles with observed state",
             response_schema="ModelsResponse", implemented_by="LH-001"),
    Endpoint(method="POST", path="/api/runs", summary="Create a run (one solver per turn)",
             request_schema="CreateRunRequest", response_schema="CreateRunResponse", success_status=202,
             error_codes=[_E.validation_error, _E.conversation_not_found, _E.conversation_busy,
                          _E.model_unavailable, _E.model_override_ineligible],
             implemented_by="LH-004"),
    Endpoint(method="GET", path="/api/runs", summary="Paginated run history",
             request_schema="ListRunsQuery", response_schema="RunPage",
             error_codes=[_E.validation_error, _E.invalid_cursor], implemented_by="LH-004"),
    Endpoint(method="GET", path="/api/runs/{run_id}", summary="Run detail",
             response_schema="RunDetail", error_codes=[_E.run_not_found], implemented_by="LH-004"),
    Endpoint(method="GET", path="/api/runs/{run_id}/events",
             summary="SSE event stream with Last-Event-ID replay (see contracts/sse.py)",
             response_schema="Event", content_type="text/event-stream",
             error_codes=[_E.run_not_found, _E.invalid_last_event_id], implemented_by="LH-004"),
    Endpoint(method="POST", path="/api/runs/{run_id}/cancel", summary="Idempotent cancellation",
             response_schema="CancelResponse", error_codes=[_E.run_not_found], implemented_by="LH-004"),
    Endpoint(method="GET", path="/api/conversations/{conversation_id}",
             summary="Read in-memory conversation context", response_schema="ConversationDetail",
             error_codes=[_E.conversation_not_found], implemented_by="LH-004"),
    Endpoint(method="POST", path="/api/conversations/{conversation_id}/reset",
             summary="Idempotently clear conversation context and sticky model",
             response_schema="ConversationResetResponse",
             error_codes=[_E.conversation_not_found, _E.conversation_busy], implemented_by="LH-004"),
]
