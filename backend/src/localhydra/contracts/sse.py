"""SSE framing, Last-Event-ID replay and client de-duplication (executable reference).

GET /api/runs/{id}/events  (Content-Type: text/event-stream)

Frame:      id: <sequence>\\nevent: <type>\\ndata: <compact event JSON>\\n\\n
Preamble:   `retry: 2000\\n\\n` once at connection start.
Keepalive:  `: keepalive\\n\\n` comment every 15 s while the run is non-terminal.
Replay:     the server first sends every persisted event with sequence > Last-Event-ID, in
            order, then streams live events. The `Last-Event-ID` header wins over a
            `?last_event_id=` query parameter (for clients that cannot set headers).
            Absent/empty means 0 (everything). A value that is not a non-negative decimal integer,
            or that exceeds the last persisted sequence, is rejected with HTTP 400 and
            ErrorCode.invalid_last_event_id (events are persisted before delivery, so no client
            can legitimately hold a larger id).
Closing:    after the terminal event has been sent, the server closes the stream (an already
            terminal run replays then closes). Clients must not reconnect after a terminal event.
Client:     SequenceTracker drops duplicates and treats a gap as a reason to reconnect with
            Last-Event-ID = last applied sequence.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import Enum

from .common import Contract
from .events import Event

RETRY_MS = 2000
KEEPALIVE_SECONDS = 15
KEEPALIVE_FRAME = ": keepalive\n\n"
RETRY_FRAME = f"retry: {RETRY_MS}\n\n"
_MAX_ID = 2**53 - 1
_DIGITS = re.compile(r"^[0-9]{1,16}$")


class InvalidLastEventId(ValueError):
    """Maps to HTTP 400 / ErrorCode.invalid_last_event_id."""


def format_sse_event(event: Event) -> str:
    data = event.model_dump_json()
    assert "\n" not in data and "\r" not in data
    return f"id: {event.sequence}\nevent: {event.type}\ndata: {data}\n\n"


def parse_last_event_id(header: str | None, query: str | None = None) -> int:
    raw = header if header not in (None, "") else query
    if raw in (None, ""):
        return 0
    assert raw is not None
    if not _DIGITS.fullmatch(raw) or int(raw) > _MAX_ID:
        raise InvalidLastEventId(f"Last-Event-ID must be a non-negative integer, got {raw!r}"[:80])
    return int(raw)


def select_replay(persisted: Sequence[Event], last_event_id: int) -> list[Event]:
    last_persisted = persisted[-1].sequence if persisted else 0
    if last_event_id > last_persisted:
        raise InvalidLastEventId("Last-Event-ID is beyond the last persisted event")
    return [e for e in persisted if e.sequence > last_event_id]


class Accept(str, Enum):
    apply = "apply"
    duplicate = "duplicate"
    gap = "gap"


class SequenceTracker(Contract):
    """Client-side de-duplication. `last` is the highest sequence applied so far (0 = none)."""

    last: int = 0

    def accept(self, sequence: int) -> Accept:
        if sequence <= self.last:
            return Accept.duplicate
        if sequence == self.last + 1:
            self.last = sequence
            return Accept.apply
        return Accept.gap
