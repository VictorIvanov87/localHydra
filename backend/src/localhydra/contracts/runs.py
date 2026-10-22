"""Run, conversation and pagination contracts."""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from .common import (
    Contract,
    ConversationId,
    ModelProfileId,
    Page,
    PageQuery,
    RunId,
    UtcDatetime,
)
from .errors import ErrorBody
from .lifecycle import TERMINAL_STATUSES, RunStatus
from .metrics import MetricsStatus, NormalizedMetrics
from .privacy import Retention
from .routing import RoutingDecision, RoutingMode

MAX_MESSAGES_PER_REQUEST = 200
MAX_MESSAGE_CHARS = 200_000
DEFAULT_MAX_CONVERSATION_TURNS_IN_MEMORY = 50


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class Message(Contract):
    """v1 messages are text-only; image input needs a contract version bump."""

    role: Role
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class CreateRunRequest(Contract):
    """`messages` are the new messages for this turn; when conversation_id refers to an existing
    conversation the server prepends its bounded in-memory history. The last message must be
    from the user. model_override is a model profile id (never a raw tag or endpoint) and is
    required exactly when routing_mode is `manual`; it remains subject to eligibility."""

    conversation_id: ConversationId | None = None
    messages: list[Message] = Field(min_length=1, max_length=MAX_MESSAGES_PER_REQUEST)
    routing_mode: RoutingMode = RoutingMode.auto
    model_override: ModelProfileId | None = None
    max_output_tokens: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _consistent(self) -> CreateRunRequest:
        if self.messages[-1].role is not Role.user:
            raise ValueError("the last message must have role 'user'")
        if (self.routing_mode is RoutingMode.manual) != (self.model_override is not None):
            raise ValueError("model_override is required if and only if routing_mode is 'manual'")
        return self


class CreateRunResponse(Contract):
    run_id: RunId
    conversation_id: ConversationId
    status: RunStatus = RunStatus.queued
    events_url: str
    created_at: UtcDatetime

    @model_validator(mode="after")
    def _queued(self) -> CreateRunResponse:
        if self.status is not RunStatus.queued:
            raise ValueError("a newly created run is always queued")
        return self


class RunSummary(Contract):
    run_id: RunId
    conversation_id: ConversationId
    status: RunStatus
    routing_mode: RoutingMode
    selected_model_id: ModelProfileId | None = None
    created_at: UtcDatetime
    updated_at: UtcDatetime
    finished_at: UtcDatetime | None = None
    last_sequence: int = Field(ge=1)
    retention: Retention
    metrics_status: MetricsStatus | None = None
    error: ErrorBody | None = None

    @model_validator(mode="after")
    def _consistent(self) -> RunSummary:
        terminal = self.status in TERMINAL_STATUSES
        if terminal != (self.finished_at is not None):
            raise ValueError("finished_at must be set exactly when the status is terminal")
        if (self.status is RunStatus.failed) != (self.error is not None):
            raise ValueError("error must be set exactly when the status is failed")
        return self


class RunDetail(RunSummary):
    model_override: ModelProfileId | None = None
    routing: RoutingDecision | None = None
    metrics: NormalizedMetrics | None = None
    cancel_requested: bool = False
    # Content fields are non-null only if the matching Retention flag is true.
    messages: list[Message] | None = None
    response_text: str | None = None
    # True when persisted deltas carry text (retention.responses). Metadata replay is always possible.
    transcript_replay_available: bool = False

    @model_validator(mode="after")
    def _privacy(self) -> RunDetail:
        if self.messages is not None and not self.retention.prompts:
            raise ValueError("messages present although prompt retention is off")
        if self.response_text is not None and not self.retention.responses:
            raise ValueError("response_text present although response retention is off")
        if self.transcript_replay_available != self.retention.responses:
            raise ValueError("transcript_replay_available must equal retention.responses")
        return self


class ListRunsQuery(PageQuery):
    status: RunStatus | None = None
    conversation_id: ConversationId | None = None


class RunPage(Page[RunSummary]):
    pass


class ConversationHistoryInfo(Contract):
    max_turns: int = Field(ge=1)
    turns: int = Field(ge=0)
    dropped_turns: int = Field(ge=0, description="Oldest turns evicted to honor max_turns.")

    @model_validator(mode="after")
    def _bounded(self) -> ConversationHistoryInfo:
        if self.turns > self.max_turns:
            raise ValueError("turns must not exceed max_turns")
        return self


class ConversationDetail(Contract):
    conversation_id: ConversationId
    created_at: UtcDatetime
    updated_at: UtcDatetime
    sticky_model_id: ModelProfileId | None = None
    active_run_id: RunId | None = None
    run_ids: list[RunId] = Field(default_factory=list, max_length=1000)
    history: ConversationHistoryInfo
    # In-memory context only (empty after a restart). Not DB/event/log content.
    messages: list[Message]
    persistence: str = Field(default="memory_only")
    lost_on_restart: bool = True

    @model_validator(mode="after")
    def _memory_only(self) -> ConversationDetail:
        if self.persistence != "memory_only" or not self.lost_on_restart:
            raise ValueError("conversation context is memory-only and lost on restart in v1")
        return self


class ConversationResetResponse(Contract):
    conversation_id: ConversationId
    reset: bool = True
    cleared_turns: int = Field(ge=0)
    # Reset clears in-memory history and sticky model only; run metadata/history is untouched.
    runs_retained: bool = True
