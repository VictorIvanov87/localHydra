"""Event envelope, typed payloads and the stream-level invariants every run must satisfy.

Envelope: schema_version, run_id, conversation_id, phase_id, phase, sequence, timestamp, type,
payload. `sequence` starts at 1 per run, is gapless, and is the SSE `id`. Events are persisted
(committed) before they are delivered to any client, so replay never lags live delivery.

Phases: a new phase_id begins at run.queued, routing.started and generation.started. Every other
event carries the phase_id/phase currently in effect (terminal events carry the phase that was
in progress when the run ended).
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

from .common import (
    EVENT_SCHEMA_VERSION,
    Contract,
    ConversationId,
    ModelProfileId,
    OllamaTag,
    PhaseId,
    RunId,
    UtcDatetime,
)
from .errors import ErrorBody, ErrorCode
from .lifecycle import RunStatus, transition
from .metrics import NormalizedMetrics, ProgressMetrics
from .privacy import Retention
from .routing import RoutingDecision, RoutingMode, RoutingOutcome


class Phase(str, Enum):
    queued = "queued"
    routing = "routing"
    generating = "generating"


class EventType(str, Enum):
    run_queued = "run.queued"
    routing_started = "routing.started"
    routing_decided = "routing.decided"
    generation_started = "generation.started"
    generation_delta = "generation.delta"
    generation_progress = "generation.progress"
    generation_metrics = "generation.metrics"
    run_cancel_requested = "run.cancel_requested"
    run_completed = "run.completed"
    run_failed = "run.failed"
    run_cancelled = "run.cancelled"
    run_interrupted = "run.interrupted"


TERMINAL_EVENT_TYPES = frozenset(
    {
        EventType.run_completed,
        EventType.run_failed,
        EventType.run_cancelled,
        EventType.run_interrupted,
    }
)


# --- payloads -------------------------------------------------------------------------------


class RunQueuedPayload(Contract):
    routing_mode: RoutingMode
    model_override: ModelProfileId | None = None
    message_count: int = Field(ge=1)
    retention: Retention


class RoutingStartedPayload(Contract):
    pass


class RoutingDecidedPayload(Contract):
    decision: RoutingDecision


class GenerationStartedPayload(Contract):
    model_id: ModelProfileId
    tag: OllamaTag
    reserved_output_tokens: int = Field(ge=1)


class GenerationDeltaPayload(Contract):
    index: int = Field(ge=0, description="0-based, gapless within the run.")
    text: str | None
    redacted: bool = False

    @model_validator(mode="after")
    def _redaction(self) -> GenerationDeltaPayload:
        if self.redacted != (self.text is None):
            raise ValueError("text must be null exactly when redacted is true")
        if self.text is not None and not self.text:
            raise ValueError("a delta must carry non-empty text (or be redacted)")
        return self


class GenerationProgressPayload(Contract):
    progress: ProgressMetrics


class GenerationMetricsPayload(Contract):
    """Final metrics for the generation phase, emitted at most once, before the terminal event.
    Incomplete metrics (e.g. cancelled mid-stream) are still emitted and flagged incomplete."""

    metrics: NormalizedMetrics


class CancelRequestedPayload(Contract):
    source: Literal["api"] = "api"


class RunCompletedPayload(Contract):
    finish_reason: str | None = Field(default=None, max_length=64)
    visible_chunks: int = Field(ge=0)


class RunFailedPayload(Contract):
    error: ErrorBody


class RunCancelledPayload(Contract):
    partial_output: bool = Field(description="True if visible content was already emitted.")


class RunInterruptedPayload(Contract):
    previous_status: Phase


# --- envelope + discriminated union --------------------------------------------------------


class _EventBase(Contract):
    schema_version: Literal[1] = EVENT_SCHEMA_VERSION  # type: ignore[assignment]
    run_id: RunId
    conversation_id: ConversationId
    phase_id: PhaseId
    phase: Phase
    sequence: int = Field(ge=1)
    timestamp: UtcDatetime


class RunQueuedEvent(_EventBase):
    type: Literal["run.queued"]
    payload: RunQueuedPayload


class RoutingStartedEvent(_EventBase):
    type: Literal["routing.started"]
    payload: RoutingStartedPayload


class RoutingDecidedEvent(_EventBase):
    type: Literal["routing.decided"]
    payload: RoutingDecidedPayload


class GenerationStartedEvent(_EventBase):
    type: Literal["generation.started"]
    payload: GenerationStartedPayload


class GenerationDeltaEvent(_EventBase):
    type: Literal["generation.delta"]
    payload: GenerationDeltaPayload


class GenerationProgressEvent(_EventBase):
    type: Literal["generation.progress"]
    payload: GenerationProgressPayload


class GenerationMetricsEvent(_EventBase):
    type: Literal["generation.metrics"]
    payload: GenerationMetricsPayload


class RunCancelRequestedEvent(_EventBase):
    type: Literal["run.cancel_requested"]
    payload: CancelRequestedPayload


class RunCompletedEvent(_EventBase):
    type: Literal["run.completed"]
    payload: RunCompletedPayload


class RunFailedEvent(_EventBase):
    type: Literal["run.failed"]
    payload: RunFailedPayload


class RunCancelledEvent(_EventBase):
    type: Literal["run.cancelled"]
    payload: RunCancelledPayload


class RunInterruptedEvent(_EventBase):
    type: Literal["run.interrupted"]
    payload: RunInterruptedPayload


Event = Annotated[
    RunQueuedEvent
    | RoutingStartedEvent
    | RoutingDecidedEvent
    | GenerationStartedEvent
    | GenerationDeltaEvent
    | GenerationProgressEvent
    | GenerationMetricsEvent
    | RunCancelRequestedEvent
    | RunCompletedEvent
    | RunFailedEvent
    | RunCancelledEvent
    | RunInterruptedEvent,
    Field(discriminator="type"),
]
EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(Event)


def parse_event(data: object) -> Event:
    return EVENT_ADAPTER.validate_python(data)


def is_terminal_event(event: Event) -> bool:
    return EventType(event.type) in TERMINAL_EVENT_TYPES


# --- privacy -------------------------------------------------------------------------------


def redact_for_persistence(event: Event, retention: Retention) -> Event:
    """Return the copy of `event` that may be written to the DB / persisted event log."""
    if isinstance(event, GenerationDeltaEvent) and not retention.responses:
        return event.model_copy(
            update={"payload": GenerationDeltaPayload(index=event.payload.index, text=None, redacted=True)}
        )
    return event


# --- stream invariants ---------------------------------------------------------------------

def validate_event_stream(events: Sequence[Event], *, require_terminal: bool = False) -> list[str]:
    """Return a list of violations (empty means valid). Checks sequence, identity, phase,
    lifecycle order, delta indexes, metrics-before-completion and exactly-one-terminal."""
    v: list[str] = []
    if not events:
        return ["stream is empty"] if require_terminal else v

    first = events[0]
    status = RunStatus.queued
    phase_id: str | None = None
    seen_phase_ids: set[str] = set()
    decision_outcome: RoutingOutcome | None = None
    selected_model_id: str | None = None
    next_delta = 0
    metrics_seen = False
    cancel_requested = False
    terminal_at: int | None = None
    last_ts = first.timestamp

    for i, ev in enumerate(events):
        where = f"event[{i}] seq={ev.sequence} {ev.type}"
        if ev.sequence != i + 1:
            v.append(f"{where}: sequence must be {i + 1} (gapless, starting at 1)")
        if ev.run_id != first.run_id or ev.conversation_id != first.conversation_id:
            v.append(f"{where}: run_id/conversation_id changed within the stream")
        if ev.timestamp < last_ts:
            v.append(f"{where}: timestamp went backwards")
        last_ts = max(last_ts, ev.timestamp)

        if terminal_at is not None:
            v.append(f"{where}: event after terminal event (seq {terminal_at}); exactly one terminal allowed")
            continue

        t = ev.type
        before = status
        is_terminal = EventType(t) in TERMINAL_EVENT_TYPES
        if (i == 0) != (t == "run.queued"):
            v.append(f"{where}: run.queued must be first and appear only once")

        # phase bookkeeping
        starts_phase = t in ("run.queued", "routing.started", "generation.started")
        if starts_phase:
            if ev.phase_id in seen_phase_ids:
                v.append(f"{where}: phase_id reused for a new phase")
            seen_phase_ids.add(ev.phase_id)
            phase_id = ev.phase_id

        def need(target: RunStatus) -> None:
            nonlocal status
            res = transition(status, target, recovery=target is RunStatus.interrupted)
            if not res.accepted:
                v.append(f"{where}: illegal lifecycle step {status.value} -> {target.value} ({res.rejection.value})")  # type: ignore[union-attr]
            else:
                status = target

        if t == "routing.started":
            need(RunStatus.routing)
        elif t == "generation.started":
            if decision_outcome is not RoutingOutcome.selected:
                v.append(f"{where}: generation requires a prior routing.decided with a selected model")
            elif ev.payload.model_id != selected_model_id:  # type: ignore[union-attr]
                v.append(f"{where}: generation model differs from routed model")
            need(RunStatus.generating)
        elif t == "routing.decided":
            if status is not RunStatus.routing or decision_outcome is not None:
                v.append(f"{where}: routing.decided must occur exactly once while routing")
            decision_outcome = ev.payload.decision.outcome  # type: ignore[union-attr]
            selected_model_id = ev.payload.decision.selected_model_id  # type: ignore[union-attr]
        elif t in ("generation.delta", "generation.progress", "generation.metrics"):
            if status is not RunStatus.generating:
                v.append(f"{where}: only valid while generating")
            if t == "generation.delta":
                if ev.payload.index != next_delta:  # type: ignore[union-attr]
                    v.append(f"{where}: delta index must be {next_delta}")
                next_delta += 1
            if t == "generation.metrics":
                if metrics_seen:
                    v.append(f"{where}: final metrics emitted twice")
                metrics_seen = True
        elif t == "run.cancel_requested":
            if cancel_requested:
                v.append(f"{where}: duplicate cancel_requested (cancel is idempotent: one event)")
            cancel_requested = True
        elif t == "run.completed":
            if not metrics_seen:
                v.append(f"{where}: completion requires final generation.metrics first")
            need(RunStatus.completed)
        elif t == "run.failed":
            err = ev.payload.error  # type: ignore[union-attr]
            if err.code is ErrorCode.no_eligible_model and decision_outcome is not RoutingOutcome.no_eligible_model:
                v.append(f"{where}: no_eligible_model failure requires a no_eligible_model routing decision")
            if decision_outcome is RoutingOutcome.no_eligible_model and err.code is not ErrorCode.no_eligible_model:
                v.append(f"{where}: a no_eligible_model decision must fail with no_eligible_model")
            need(RunStatus.failed)
        elif t == "run.cancelled":
            if not cancel_requested:
                v.append(f"{where}: cancelled requires a prior run.cancel_requested")
            need(RunStatus.cancelled)
        elif t == "run.interrupted":
            if ev.payload.previous_status != Phase(status.value):  # type: ignore[union-attr]
                v.append(f"{where}: previous_status does not match the stream state ({status.value})")
            need(RunStatus.interrupted)

        if not starts_phase and ev.phase_id != phase_id:
            v.append(f"{where}: phase_id does not match the phase in effect")
        # Transition events carry the phase they enter; terminal events the phase they end.
        expected_phase = before.value if is_terminal else status.value
        if ev.phase.value != expected_phase:
            v.append(f"{where}: phase '{ev.phase.value}' does not match lifecycle state '{expected_phase}'")

        if is_terminal:
            terminal_at = ev.sequence

    if require_terminal and terminal_at is None:
        v.append("stream has no terminal event")
    return v
