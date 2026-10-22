"""LH-002: telemetry honesty. Inputs are SYNTHETIC numbers picked so expected values are exact by hand."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from localhydra.adapters import (
    ChatFinished,
    ChatMessage,
    ChatRequest,
    OllamaProvider,
    StreamTelemetry,
    parse_raw_metrics,
)
from localhydra.contracts.metrics import IncompleteReason, MetricsStatus, RawOllamaMetrics

from .provider_fakes import SYNTH_FINAL, SYNTH_TAG, Conn, FakeOllama, anyio_backend, chunk, final  # noqa: F401

REQ = ChatRequest(model=SYNTH_TAG, messages=(ChatMessage(role="user", content="x"),))


def scripted_clock(*readings: float) -> Callable[[], float]:
    it = iter(readings)

    def clock() -> float:
        try:
            return next(it)
        except StopIteration:
            raise AssertionError("clock read more often than the documented contract") from None

    return clock


# ---- raw parsing ---------------------------------------------------------------------------------


def test_raw_metrics_preserved_verbatim() -> None:
    raw, discarded = parse_raw_metrics({**SYNTH_FINAL, "model": "ignored", "done": True})
    assert raw == RawOllamaMetrics(**SYNTH_FINAL) and discarded == ()


def test_absent_fields_stay_none() -> None:
    raw, discarded = parse_raw_metrics({"done": True})
    assert raw == RawOllamaMetrics() and discarded == ()


@pytest.mark.parametrize("bad", ["5", 1.5, -1, True, [1], {"a": 1}])
def test_unusable_values_are_discarded_and_named_never_coerced(bad: object) -> None:
    raw, discarded = parse_raw_metrics({"eval_count": bad, "eval_duration": 7})
    assert raw.eval_count is None and raw.eval_duration == 7
    assert discarded == ("eval_count",)


def test_explicit_zero_is_a_value_not_missing() -> None:
    raw, discarded = parse_raw_metrics({"eval_count": 0, "eval_duration": 0})
    assert (raw.eval_count, raw.eval_duration) == (0, 0) and discarded == ()


# ---- unit normalization and throughput (contract reference function) -------------------------------


def norm(raw: dict, **kw: float | None):
    from localhydra.adapters.telemetry import TelemetrySnapshot

    base = {"routing_ms": 1.0, "wall_ms": 5200.0, "ttft_ms": 1700.0}
    base.update(kw)
    snap = TelemetrySnapshot(
        wall_ms=base["wall_ms"], ttft_ms=base["ttft_ms"], chunk_count=1, visible_chunks=1,
        raw_ollama=RawOllamaMetrics(**raw),
    )
    return snap.normalized(routing_ms=base["routing_ms"])


def test_nanoseconds_become_milliseconds_and_throughput_uses_measured_tokens() -> None:
    m = norm(SYNTH_FINAL)
    assert m.completeness.status is MetricsStatus.complete
    assert m.load_ms == 1000.0 and m.prompt_eval_ms == 500.0
    assert m.generation_ms == 2000.0 and m.ollama_total_ms == 5000.0
    assert (m.prompt_tokens, m.generated_tokens) == (26, 100)
    assert m.generation_tokens_per_second == 50.0  # 100 tokens / 2.0 s
    assert m.prompt_tokens_per_second == 52.0  # 26 tokens / 0.5 s
    assert m.raw_ollama == RawOllamaMetrics(**SYNTH_FINAL)  # raw ns kept alongside
    assert m.wall_ms == 5200.0 and m.ttft_ms == 1700.0 and m.routing_ms == 1.0


def test_sub_millisecond_durations_are_not_rounded() -> None:
    m = norm({**SYNTH_FINAL, "eval_duration": 1_500, "eval_count": 3})
    assert m.generation_ms == pytest.approx(0.0015)
    assert m.generation_tokens_per_second == pytest.approx(3 / 0.0000015)


def test_fractional_throughput() -> None:
    m = norm({**SYNTH_FINAL, "eval_count": 7, "eval_duration": 3_500_000_000})
    assert m.generation_tokens_per_second == 2.0


def test_zero_duration_gives_no_throughput_instead_of_infinity() -> None:
    m = norm({**SYNTH_FINAL, "eval_duration": 0, "prompt_eval_duration": 0})
    assert m.generation_ms == 0.0 and m.prompt_eval_ms == 0.0
    assert m.generation_tokens_per_second is None and m.prompt_tokens_per_second is None
    # a measured zero is not "missing": the contract still calls the metrics complete
    assert m.completeness.status is MetricsStatus.complete


def test_zero_tokens_with_positive_duration_is_zero_throughput() -> None:
    m = norm({**SYNTH_FINAL, "eval_count": 0})
    assert m.generated_tokens == 0 and m.generation_tokens_per_second == 0.0


def test_unavailable_counts_give_null_throughput_and_are_listed_missing() -> None:
    raw = {k: v for k, v in SYNTH_FINAL.items() if k not in ("eval_count", "prompt_eval_count")}
    m = norm(raw)
    assert m.generated_tokens is None and m.prompt_tokens is None
    assert m.generation_tokens_per_second is None and m.prompt_tokens_per_second is None
    assert set(m.completeness.missing) == {"generated_tokens", "prompt_tokens"}
    assert m.completeness.reason is IncompleteReason.partial_final_chunk


def test_missing_routing_ms_is_reported_not_defaulted() -> None:
    from localhydra.adapters.telemetry import TelemetrySnapshot

    snap = TelemetrySnapshot(100.0, 10.0, 1, 1, RawOllamaMetrics(**SYNTH_FINAL))
    m = snap.normalized()
    assert m.routing_ms is None and m.completeness.missing == ["routing_ms"]


def test_no_final_chunk_means_no_raw_and_no_invented_values() -> None:
    from localhydra.adapters.telemetry import TelemetrySnapshot

    m = TelemetrySnapshot(250.0, 40.0, 3, 2, None).normalized(routing_ms=1.0)
    assert m.raw_ollama is None and m.completeness.reason is IncompleteReason.no_final_chunk
    assert m.wall_ms == 250.0 and m.ttft_ms == 40.0  # app-observed values survive
    assert m.generation_tokens_per_second is None and m.load_ms is None


def test_no_gpu_energy_cache_or_cost_fields_exist() -> None:
    fields = set(norm(SYNTH_FINAL).model_dump())
    assert not [f for f in fields if any(w in f for w in ("gpu", "vram", "energy", "cache", "cost", "dollar"))]


# ---- app-observed timing with a scripted clock -------------------------------------------------------


def test_ttft_ignores_empty_and_whitespace_chunks_and_wall_covers_the_whole_call() -> None:
    tel = StreamTelemetry(scripted_clock(10.0, 10.1, 10.2, 10.25, 10.5))
    tel.start()
    assert tel.on_chunk("") is False
    assert tel.on_chunk(" \n\t") is False
    assert tel.on_chunk("Hi") is True
    tel.finish(RawOllamaMetrics(**SYNTH_FINAL))
    snap = tel.snapshot()
    assert snap.ttft_ms == pytest.approx(250.0)
    assert snap.wall_ms == pytest.approx(500.0)
    assert (snap.chunk_count, snap.visible_chunks) == (3, 1)


def test_progress_exposes_only_app_observed_values() -> None:
    tel = StreamTelemetry(scripted_clock(0.0, 0.3, 0.8))
    tel.start()
    tel.on_chunk("x")
    p = tel.progress()
    assert p.model_dump() == {"elapsed_ms": pytest.approx(800.0), "visible_chunks": 1, "ttft_ms": pytest.approx(300.0)}


def test_finish_is_idempotent_and_first_outcome_wins() -> None:
    tel = StreamTelemetry(scripted_clock(0.0, 1.0))
    tel.start()
    tel.finish(None)
    tel.finish(RawOllamaMetrics(**SYNTH_FINAL))
    assert tel.snapshot().raw_ollama is None and tel.snapshot().wall_ms == 1000.0


@pytest.fixture
async def fake():
    async with FakeOllama() as f:
        yield f


@pytest.mark.anyio
async def test_stream_timing_end_to_end_with_scripted_clock(fake: FakeOllama) -> None:
    async def h(c: Conn) -> None:
        await c.play([chunk("  "), chunk("Hi"), final(**SYNTH_FINAL)])

    fake.on("POST", "/api/chat", h)
    # start, 3 chunks (whitespace, visible, final), finish
    clock = scripted_clock(10.0, 10.1, 10.25, 10.4, 10.5)
    async with OllamaProvider(fake.base_url, clock=clock) as p:
        items = [i async for i in p.chat_stream(REQ)]
    done = items[-1]
    assert isinstance(done, ChatFinished)
    m = done.telemetry.normalized(routing_ms=2.0)
    assert m.ttft_ms == pytest.approx(250.0) and m.wall_ms == pytest.approx(500.0)
    assert m.completeness.status is MetricsStatus.complete
    assert math.isclose(m.generation_tokens_per_second, 50.0)


@pytest.mark.anyio
async def test_reply_without_any_visible_token_reports_no_ttft(fake: FakeOllama) -> None:
    async def h(c: Conn) -> None:
        await c.play([chunk(""), chunk("   "), final(**SYNTH_FINAL)])

    fake.on("POST", "/api/chat", h)
    async with OllamaProvider(fake.base_url) as p:
        done = [i async for i in p.chat_stream(REQ)][-1]
    m = done.telemetry.normalized(routing_ms=2.0)
    assert m.ttft_ms is None and m.completeness.missing == ["ttft_ms"]
    assert m.completeness.reason is IncompleteReason.no_visible_token
