"""Honest stream telemetry built on the frozen metrics contract.

Only values that were actually observed are recorded. Load/prefill/decode durations and token
counts exist solely in Ollama's final chunk and are preserved verbatim (nanoseconds / counts);
they are never estimated, and nothing about GPU memory, energy or cost is derived.

App-observed values (clock injectable, `time.perf_counter` by default):
- wall_ms: request about to be sent -> stream ended / failed / cancelled.
- ttft_ms: request about to be sent -> the first chunk whose text content is non-empty and not
  whitespace-only. Measured when the adapter hands the delta to its consumer, so it is a lower
  bound of the time the client sees it. `thinking` text is not visible content.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..contracts.metrics import NormalizedMetrics, ProgressMetrics, RawOllamaMetrics, normalize_ollama_metrics

RAW_METRIC_FIELDS = (
    "total_duration",
    "load_duration",
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
)


def parse_raw_metrics(chunk: Mapping[str, Any]) -> tuple[RawOllamaMetrics, tuple[str, ...]]:
    """Extract raw Ollama measurements from a final chunk.

    Absent fields stay None. A present field that is not a non-negative JSON integer (string,
    float, bool, negative) is NOT coerced: it is dropped to None and its name is returned so the
    caller can report that the provider sent unusable data.
    """
    values: dict[str, int | None] = {}
    discarded: list[str] = []
    for name in RAW_METRIC_FIELDS:
        if name not in chunk or chunk[name] is None:
            values[name] = None
            continue
        v = chunk[name]
        if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
            values[name] = v
        else:
            values[name] = None
            discarded.append(name)
    return RawOllamaMetrics(**values), tuple(discarded)


@dataclass(frozen=True)
class TelemetrySnapshot:
    wall_ms: float | None  # None while the call is still running
    ttft_ms: float | None
    chunk_count: int
    visible_chunks: int
    raw_ollama: RawOllamaMetrics | None  # None unless a final chunk was received
    discarded_metric_fields: tuple[str, ...] = ()

    def normalized(self, routing_ms: float | None = None) -> NormalizedMetrics:
        """Contract normalization. `routing_ms` belongs to the router/runtime; when it is not
        supplied the result is flagged incomplete by the contract (it is listed as missing)."""
        return normalize_ollama_metrics(
            self.raw_ollama, routing_ms=routing_ms, wall_ms=self.wall_ms, ttft_ms=self.ttft_ms
        )


class StreamTelemetry:
    """Mutable recorder. Calls `clock` once in `start`, once per `on_chunk`, once in `finish`."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._t0: float | None = None
        self._wall_ms: float | None = None
        self._ttft_ms: float | None = None
        self._chunks = 0
        self._visible = 0
        self._raw: RawOllamaMetrics | None = None
        self._discarded: tuple[str, ...] = ()
        self._last_ms = 0.0

    @property
    def finished(self) -> bool:
        return self._wall_ms is not None

    def start(self) -> None:
        self._t0 = self._clock()

    def _elapsed_ms(self) -> float:
        assert self._t0 is not None, "start() not called"
        self._last_ms = max(0.0, (self._clock() - self._t0) * 1000.0)
        return self._last_ms

    def on_chunk(self, text: str) -> bool:
        """Record a received chunk; returns whether its text counts as visible."""
        elapsed = self._elapsed_ms()
        self._chunks += 1
        visible = bool(text.strip())
        if visible:
            self._visible += 1
            if self._ttft_ms is None:
                self._ttft_ms = elapsed
        return visible

    def finish(
        self, raw: RawOllamaMetrics | None = None, discarded: tuple[str, ...] = ()
    ) -> None:
        if self.finished:
            return
        if self._t0 is None:
            self._wall_ms = 0.0
        else:
            self._wall_ms = self._elapsed_ms()
        self._raw = raw
        self._discarded = discarded

    def snapshot(self) -> TelemetrySnapshot:
        return TelemetrySnapshot(
            wall_ms=self._wall_ms,
            ttft_ms=self._ttft_ms,
            chunk_count=self._chunks,
            visible_chunks=self._visible,
            raw_ollama=self._raw,
            discarded_metric_fields=self._discarded,
        )

    def progress(self) -> ProgressMetrics:
        """Live, app-observed progress only (no load/prefill/decode signals exist live)."""
        if self._wall_ms is not None:
            elapsed = self._wall_ms
        elif self._t0 is not None:
            elapsed = self._elapsed_ms()
        else:
            elapsed = 0.0
        return ProgressMetrics(
            elapsed_ms=elapsed, visible_chunks=self._visible, ttft_ms=self._ttft_ms
        )
