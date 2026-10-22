"""Run lifecycle and race semantics (executable reference).

States: queued -> routing -> generating -> {completed | failed | cancelled}, plus
`interrupted`, which is produced only by startup recovery of persisted unfinished runs.

Exactly-one-terminal rule: the first terminal transition that is *persisted* wins. The runtime
must make "check not terminal + write terminal status + append terminal event" one atomic
step (single SQLite transaction / compare-and-set on status). Any later terminal attempt is
rejected with reason `already_terminal`, appends no event, and changes nothing.

Cancellation:
- POST /cancel on a terminal run: 200, effect `already_terminal`, no event (idempotent).
- On `queued`: the run becomes `cancelled` immediately (effect `cancelled`); events
  `run.cancel_requested` then `run.cancelled` are appended in one transaction.
- On `routing`/`generating`: the first request appends `run.cancel_requested` and returns
  effect `cancel_requested`; repeats return the same effect with no new event. The worker must
  stop and finalize `cancelled`. If the worker persisted `completed`/`failed` first, that
  outcome stands (a `run.cancel_requested` followed by `run.completed` is a valid stream).
  A worker that observes the cancel flag before persisting completion MUST finalize
  `cancelled` rather than `completed`.
"""

from __future__ import annotations

from enum import Enum

from .common import Contract, RunId


class RunStatus(str, Enum):
    queued = "queued"
    routing = "routing"
    generating = "generating"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"
    interrupted = "interrupted"


TERMINAL_STATUSES = frozenset(
    {RunStatus.completed, RunStatus.failed, RunStatus.cancelled, RunStatus.interrupted}
)

ALLOWED_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.queued: frozenset(
        {RunStatus.routing, RunStatus.failed, RunStatus.cancelled, RunStatus.interrupted}
    ),
    RunStatus.routing: frozenset(
        {RunStatus.generating, RunStatus.failed, RunStatus.cancelled, RunStatus.interrupted}
    ),
    RunStatus.generating: frozenset(
        {RunStatus.completed, RunStatus.failed, RunStatus.cancelled, RunStatus.interrupted}
    ),
}


class TransitionRejection(str, Enum):
    already_terminal = "already_terminal"
    illegal_transition = "illegal_transition"
    interrupted_requires_recovery = "interrupted_requires_recovery"


class TransitionResult(Contract):
    accepted: bool
    status: RunStatus
    rejection: TransitionRejection | None = None


def transition(current: RunStatus, target: RunStatus, *, recovery: bool = False) -> TransitionResult:
    """Pure compare-and-set. `status` is the resulting status either way."""
    if current in TERMINAL_STATUSES:
        return TransitionResult(
            accepted=False, status=current, rejection=TransitionRejection.already_terminal
        )
    if target is RunStatus.interrupted and not recovery:
        return TransitionResult(
            accepted=False,
            status=current,
            rejection=TransitionRejection.interrupted_requires_recovery,
        )
    if target not in ALLOWED_TRANSITIONS[current]:
        return TransitionResult(
            accepted=False, status=current, rejection=TransitionRejection.illegal_transition
        )
    return TransitionResult(accepted=True, status=target)


class CancelEffect(str, Enum):
    cancelled = "cancelled"
    cancel_requested = "cancel_requested"
    already_terminal = "already_terminal"


class CancelDecision(Contract):
    effect: CancelEffect
    append_cancel_requested_event: bool
    finalize_cancelled_now: bool


def decide_cancel(status: RunStatus, cancel_already_requested: bool) -> CancelDecision:
    if status in TERMINAL_STATUSES:
        return CancelDecision(
            effect=CancelEffect.already_terminal,
            append_cancel_requested_event=False,
            finalize_cancelled_now=False,
        )
    if status is RunStatus.queued:
        return CancelDecision(
            effect=CancelEffect.cancelled,
            append_cancel_requested_event=not cancel_already_requested,
            finalize_cancelled_now=True,
        )
    return CancelDecision(
        effect=CancelEffect.cancel_requested,
        append_cancel_requested_event=not cancel_already_requested,
        finalize_cancelled_now=False,
    )


def recover_status(persisted: RunStatus) -> RunStatus:
    """Startup recovery: unfinished persisted runs become `interrupted`; terminal ones stay."""
    return persisted if persisted in TERMINAL_STATUSES else RunStatus.interrupted


class CancelResponse(Contract):
    run_id: RunId
    status: RunStatus
    cancel_effect: CancelEffect
