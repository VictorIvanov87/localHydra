"""FastAPI application factory. LH-001 serves only /api/health and /api/models.

Run, conversation and event routes are frozen as contracts (contracts/api.py) and implemented
by LH-004; until then they are simply absent and answer with the standard 404 envelope.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .contracts.api import HealthResponse
from .contracts.common import CONTRACT_VERSION, EVENT_SCHEMA_VERSION
from .contracts.errors import ERROR_HTTP_STATUS, ErrorBody, ErrorCode, ErrorEnvelope
from .contracts.models import ModelsResponse, ProbeStatus, resolve_model_statuses
from .settings import Settings

log = logging.getLogger("localhydra")

DEFAULT_ALLOWED_HOSTS = frozenset({"localhost", "127.0.0.1", "[::1]", "::1"})


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or uuid.uuid4().hex


def error_response(
    code: ErrorCode,
    message: str,
    *,
    request_id: str | None = None,
    details: dict[str, object] | None = None,
    retryable: bool = False,
) -> JSONResponse:
    body = ErrorEnvelope(
        error=ErrorBody(
            code=code, message=message, retryable=retryable, details=details, request_id=request_id
        )
    )
    return JSONResponse(body.model_dump(mode="json"), status_code=ERROR_HTTP_STATUS[code])


def _host_only(host_header: str) -> str:
    if host_header.startswith("["):  # [::1]:8765
        return host_header.split("]")[0] + "]"
    return host_header.split(":")[0]


def create_app(
    settings: Settings | None = None, *, allowed_hosts: frozenset[str] = DEFAULT_ALLOWED_HOSTS
) -> FastAPI:
    settings = settings or Settings()
    # No OpenAPI/Swagger UI: those pages load assets from a CDN, which the local-only rule forbids.
    app = FastAPI(
        title="localHydra", version=__version__, docs_url=None, redoc_url=None, openapi_url=None
    )

    @app.middleware("http")
    async def guard(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        request.state.request_id = uuid.uuid4().hex
        # Rejects DNS-rebinding style requests whose Host is not a loopback name.
        if _host_only(request.headers.get("host", "")) not in allowed_hosts:
            return error_response(
                ErrorCode.host_not_allowed,
                "Host header is not a loopback name",
                request_id=request.state.request_id,
            )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RequestValidationError)
    async def on_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        # loc + type only: never echo `input`, which could contain prompt text.
        errs = [{"loc": [str(p) for p in e["loc"]], "type": e["type"]} for e in exc.errors()]
        return error_response(
            ErrorCode.validation_error,
            "Request validation failed",
            request_id=_request_id(request),
            details={"errors": errs},
        )

    @app.exception_handler(StarletteHTTPException)
    async def on_http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = ErrorCode.not_found if exc.status_code == 404 else ErrorCode.bad_request
        resp = error_response(code, "Not found" if code is ErrorCode.not_found else "Bad request",
                              request_id=_request_id(request))
        resp.status_code = exc.status_code
        return resp

    @app.exception_handler(Exception)
    async def on_unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled error on %s %s: %s", request.method, request.url.path, type(exc).__name__)
        return error_response(
            ErrorCode.internal_error, "Internal error", request_id=_request_id(request)
        )

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            version=__version__,
            contract_version=CONTRACT_VERSION,
            event_schema_version=EVENT_SCHEMA_VERSION,
            bind_host=settings.server.host,
            ollama_target=settings.provider.base_url,
            models_configured=len(settings.models),
            time=datetime.now(timezone.utc),
        )

    @app.get("/api/models", response_model=ModelsResponse)
    async def list_models() -> ModelsResponse:
        # No provider adapter yet (LH-002): availability is 'unknown', never assumed present.
        return ModelsResponse(
            configured=settings.configured,
            provider_probe=ProbeStatus.not_probed,
            models=resolve_model_statuses(settings.models, installed_tags=None),
        )

    return app
