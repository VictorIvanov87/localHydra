"""Measurement contract: raw Ollama values are preserved; normalized values never invent data.

Units: durations in the normalized form are milliseconds (float), counts are tokens (int),
throughput is tokens/second. Raw Ollama durations are nanoseconds and kept as received.

Definitions (frozen):
- routing_ms        app-measured: run leaves `queued` until routing outcome is decided.
- wall_ms           app-measured: generation request sent -> stream finished/terminated.
- ttft_ms           app-measured: generation request sent -> first *visible* content delta emitted
                    to the client (non-empty text; empty/whitespace-only chunks do not count).
- load_ms           Ollama load_duration (model load). Available only in the final chunk.
- prompt_eval_ms    Ollama prompt_eval_duration (prefill). Available only in the final chunk.
- generation_ms     Ollama eval_duration (decode). Available only in the final chunk.
- ollama_total_ms   Ollama total_duration. Available only in the final chunk.
- prompt_tokens     Ollama prompt_eval_count. Generated tokens: Ollama eval_count.
Throughput is tokens / (duration in seconds) from those measured values, never from words,
characters or estimates. GPU memory, energy, cache hit rate and cost are intentionally absent.
"""

from __future__ import annotations

import math
from enum import Enum

from pydantic import Field, model_validator

from .common import Contract

NS_PER_MS = 1_000_000.0


class RawOllamaMetrics(Contract):
    """Fields of Ollama's final stream chunk, verbatim (nanoseconds / counts). All optional."""

    total_duration: int | None = Field(default=None, ge=0)
    load_duration: int | None = Field(default=None, ge=0)
    prompt_eval_count: int | None = Field(default=None, ge=0)
    prompt_eval_duration: int | None = Field(default=None, ge=0)
    eval_count: int | None = Field(default=None, ge=0)
    eval_duration: int | None = Field(default=None, ge=0)


class MetricsStatus(str, Enum):
    complete = "complete"
    incomplete = "incomplete"


class IncompleteReason(str, Enum):
    no_final_chunk = "no_final_chunk"  # stream ended/cancelled/failed before Ollama's final chunk
    partial_final_chunk = "partial_final_chunk"  # final chunk lacked some fields (e.g. cached prompt)
    no_visible_token = "no_visible_token"  # only ttft_ms is missing: no visible content emitted


class MetricsCompleteness(Contract):
    status: MetricsStatus
    missing: list[str] = Field(default_factory=list)
    reason: IncompleteReason | None = None

    @model_validator(mode="after")
    def _consistent(self) -> MetricsCompleteness:
        if self.status is MetricsStatus.complete and (self.missing or self.reason):
            raise ValueError("complete metrics cannot list missing fields or a reason")
        if self.status is MetricsStatus.incomplete and (not self.missing or self.reason is None):
            raise ValueError("incomplete metrics must list missing fields and a reason")
        return self


class ProgressMetrics(Contract):
    """Live, app-observed values only. Load/prefill/decode durations are NOT available live."""

    elapsed_ms: float = Field(ge=0)
    visible_chunks: int = Field(ge=0)
    ttft_ms: float | None = Field(default=None, ge=0)


_REQUIRED = (
    "routing_ms",
    "wall_ms",
    "ttft_ms",
    "load_ms",
    "prompt_eval_ms",
    "generation_ms",
    "ollama_total_ms",
    "prompt_tokens",
    "generated_tokens",
)


class NormalizedMetrics(Contract):
    routing_ms: float | None = Field(default=None, ge=0)
    wall_ms: float | None = Field(default=None, ge=0)
    ttft_ms: float | None = Field(default=None, ge=0)
    load_ms: float | None = Field(default=None, ge=0)
    prompt_eval_ms: float | None = Field(default=None, ge=0)
    generation_ms: float | None = Field(default=None, ge=0)
    ollama_total_ms: float | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    generated_tokens: int | None = Field(default=None, ge=0)
    prompt_tokens_per_second: float | None = Field(default=None, ge=0)
    generation_tokens_per_second: float | None = Field(default=None, ge=0)
    raw_ollama: RawOllamaMetrics | None = None
    completeness: MetricsCompleteness

    @model_validator(mode="after")
    def _consistent(self) -> NormalizedMetrics:
        missing = [name for name in _REQUIRED if getattr(self, name) is None]
        if set(missing) != set(self.completeness.missing):
            raise ValueError(
                f"completeness.missing {sorted(self.completeness.missing)} does not match "
                f"fields that are null {sorted(missing)}"
            )
        _check_rate(
            self.generation_tokens_per_second,
            self.generated_tokens,
            self.generation_ms,
            "generation_tokens_per_second",
        )
        _check_rate(
            self.prompt_tokens_per_second,
            self.prompt_tokens,
            self.prompt_eval_ms,
            "prompt_tokens_per_second",
        )
        return self


def _rate(tokens: int | None, ms: float | None) -> float | None:
    if tokens is None or ms is None or ms <= 0:
        return None
    return tokens / (ms / 1000.0)


def _check_rate(value: float | None, tokens: int | None, ms: float | None, name: str) -> None:
    expected = _rate(tokens, ms)
    if expected is None:
        if value is not None:
            raise ValueError(f"{name} must be null when tokens or duration are unavailable")
    elif value is None or not math.isclose(value, expected, rel_tol=1e-6):
        raise ValueError(f"{name} must equal tokens / duration (expected {expected})")


def _ms(ns: int | None) -> float | None:
    return None if ns is None else ns / NS_PER_MS


def normalize_ollama_metrics(
    raw: RawOllamaMetrics | None,
    *,
    routing_ms: float | None,
    wall_ms: float | None,
    ttft_ms: float | None,
) -> NormalizedMetrics:
    """Reference normalization. Missing inputs stay null and mark the result incomplete."""
    raw_present = raw is not None
    r = raw or RawOllamaMetrics()
    generation_ms = _ms(r.eval_duration)
    prompt_eval_ms = _ms(r.prompt_eval_duration)
    fields = {
        "routing_ms": routing_ms,
        "wall_ms": wall_ms,
        "ttft_ms": ttft_ms,
        "load_ms": _ms(r.load_duration),
        "prompt_eval_ms": prompt_eval_ms,
        "generation_ms": generation_ms,
        "ollama_total_ms": _ms(r.total_duration),
        "prompt_tokens": r.prompt_eval_count,
        "generated_tokens": r.eval_count,
    }
    missing = [name for name in _REQUIRED if fields[name] is None]
    if not missing:
        completeness = MetricsCompleteness(status=MetricsStatus.complete)
    else:
        raw_fields = {n for n in missing if n not in ("routing_ms", "wall_ms", "ttft_ms")}
        if not raw_present:
            reason = IncompleteReason.no_final_chunk
        elif raw_fields:
            reason = IncompleteReason.partial_final_chunk
        else:
            reason = IncompleteReason.no_visible_token if "ttft_ms" in missing else (
                IncompleteReason.partial_final_chunk
            )
        completeness = MetricsCompleteness(
            status=MetricsStatus.incomplete, missing=missing, reason=reason
        )
    return NormalizedMetrics(
        **fields,
        prompt_tokens_per_second=_rate(r.prompt_eval_count, prompt_eval_ms),
        generation_tokens_per_second=_rate(r.eval_count, generation_ms),
        raw_ollama=raw,
        completeness=completeness,
    )
