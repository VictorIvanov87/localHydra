"""Executable semantics: streams, lifecycle, SSE, context, metrics, privacy, loopback."""

from __future__ import annotations

import json

import pytest

from localhydra.contracts.config import all_loopback, validate_loopback_url
from localhydra.contracts.config import ServerConfig
from localhydra.contracts.context import ContextEstimate, check_context_fit, estimate_context
from localhydra.contracts.events import (
    GenerationDeltaEvent,
    parse_event,
    redact_for_persistence,
    validate_event_stream,
)
from localhydra.contracts.lifecycle import RunStatus, decide_cancel, recover_status, transition
from localhydra.contracts.metrics import RawOllamaMetrics, normalize_ollama_metrics
from localhydra.contracts.models import ModelProfile
from localhydra.contracts.privacy import Retention
from localhydra.contracts.sse import (
    InvalidLastEventId,
    SequenceTracker,
    format_sse_event,
    parse_last_event_id,
    select_replay,
)
from pydantic import ValidationError
from tests.conftest import EXAMPLE_CONFIG, approx_equal, load_fixture


# ---- event streams ------------------------------------------------------------------------


@pytest.mark.parametrize("case", load_fixture("streams.json")["cases"], ids=lambda c: c["name"])
def test_event_stream(case):
    events = [parse_event(e) for e in case["events"]]
    violations = validate_event_stream(events, require_terminal=case["require_terminal"])
    if case["valid"]:
        assert violations == []
    else:
        assert violations, "expected violations"
        for needle in case["expect_violation_contains"]:
            assert any(needle in v for v in violations), (needle, violations)


def test_exactly_one_terminal_in_every_valid_stream():
    from localhydra.contracts.events import is_terminal_event

    for case in load_fixture("streams.json")["cases"]:
        if case["valid"] and case["require_terminal"]:
            events = [parse_event(e) for e in case["events"]]
            assert sum(is_terminal_event(e) for e in events) == 1
            assert is_terminal_event(events[-1])


# ---- lifecycle ----------------------------------------------------------------------------


def test_lifecycle_transition_matrix():
    doc = load_fixture("lifecycle.json")
    assert len(doc["transitions"]) == 7 * 7 * 2
    for c in doc["transitions"]:
        got = transition(RunStatus(c["current"]), RunStatus(c["target"]), recovery=c["recovery"])
        assert json.loads(got.model_dump_json()) == c["expected"], c


def test_terminal_state_is_absorbing_under_every_attempt():
    for terminal in ("completed", "failed", "cancelled", "interrupted"):
        for target in RunStatus:
            for recovery in (False, True):
                r = transition(RunStatus(terminal), target, recovery=recovery)
                assert not r.accepted and r.status.value == terminal


def test_cancel_decisions_are_idempotent():
    for c in load_fixture("lifecycle.json")["cancels"]:
        got = decide_cancel(RunStatus(c["status"]), c["cancel_already_requested"])
        assert json.loads(got.model_dump_json()) == c["expected"], c
    # A repeated request never appends a second cancel_requested event.
    for st in ("queued", "routing", "generating"):
        assert decide_cancel(RunStatus(st), True).append_cancel_requested_event is False


def test_recovery():
    for c in load_fixture("lifecycle.json")["recovery"]:
        assert recover_status(RunStatus(c["persisted"])).value == c["recovered"]


# ---- SSE ----------------------------------------------------------------------------------


def test_sse_frame_and_constants():
    doc = load_fixture("sse.json")
    ev = parse_event(doc["frame"]["event"])
    frame = format_sse_event(ev)
    lines = frame.split("\n")
    assert lines[0] == f"id: {doc['frame']['expected_id']}"
    assert lines[1] == f"event: {doc['frame']['expected_event']}"
    assert lines[2].startswith("data: ") and lines[3] == "" and lines[4] == "" and len(lines) == 5
    assert parse_event(json.loads(lines[2][6:])) == ev
    from localhydra.contracts import sse

    assert sse.RETRY_FRAME == doc["preamble"] and sse.KEEPALIVE_FRAME == doc["keepalive"]


def test_sse_frame_survives_newlines_in_content():
    ev = parse_event(
        {
            **load_fixture("sse.json")["frame"]["event"],
            "type": "generation.delta",
            "phase": "generating",
            "payload": {"index": 0, "text": "line1\nline2\r\n ", "redacted": False},
        }
    )
    frame = format_sse_event(ev)
    assert frame.count("\n\n") == 1 and frame.endswith("\n\n")
    assert parse_event(json.loads(frame.split("data: ", 1)[1].strip())) == ev


@pytest.mark.parametrize("c", load_fixture("sse.json")["last_event_id"], ids=lambda c: repr(c["header"]))
def test_last_event_id_parsing(c):
    if "error" in c:
        with pytest.raises(InvalidLastEventId):
            parse_last_event_id(c["header"], c["query"])
    else:
        assert parse_last_event_id(c["header"], c["query"]) == c["expected"]


def test_last_event_id_rejects_trailing_newline():
    with pytest.raises(InvalidLastEventId):
        parse_last_event_id("3\n")


def test_replay():
    doc = load_fixture("sse.json")["replay"]
    persisted = [parse_event(e) for e in doc["persisted"]]
    for c in doc["cases"]:
        if "error" in c:
            with pytest.raises(InvalidLastEventId):
                select_replay(persisted, c["last_event_id"])
        else:
            assert [e.sequence for e in select_replay(persisted, c["last_event_id"])] == c["expected_sequences"]


def test_tracker_dedup_and_gap():
    for c in load_fixture("sse.json")["tracker"]:
        t = SequenceTracker()
        got = [t.accept(s).value for s in c["sequences"]]
        assert got == c["expected"], c["name"]
        if "last_after" in c:
            assert t.last == c["last_after"]


def test_reconnect_with_replay_yields_each_event_once():
    persisted = [parse_event(e) for e in load_fixture("sse.json")["replay"]["persisted"]]
    t = SequenceTracker()
    applied: list[int] = []
    for ev in persisted[:4]:  # connection drops after event 4
        if t.accept(ev.sequence).value == "apply":
            applied.append(ev.sequence)
    # Browser reconnects with Last-Event-ID: 4 but the server also re-sends 3-4 (at-least-once).
    for ev in persisted[2:]:
        if t.accept(ev.sequence).value == "apply":
            applied.append(ev.sequence)
    assert applied == [1, 2, 3, 4, 5, 6, 7]


# ---- context ------------------------------------------------------------------------------


@pytest.mark.parametrize("c", load_fixture("context.json")["estimates"], ids=lambda c: c["name"][:50])
def test_context_estimate(c):
    est = estimate_context(c["contents"])
    assert (est.point_tokens, est.upper_bound_tokens) == (c["point_tokens"], c["upper_bound_tokens"])
    assert est.method == "ascii3_nonascii1_v1" and est.margin == 1.25 and "not a tokenizer" in est.uncertainty
    assert est.upper_bound_tokens >= est.point_tokens


@pytest.mark.parametrize("c", load_fixture("context.json")["fits"], ids=lambda c: c["name"][:50])
def test_context_fit(c):
    profile = _profile(c["context_window"], c["max_output_tokens"])
    est = ContextEstimate(message_count=1, point_tokens=0, upper_bound_tokens=c["upper_bound_tokens"])
    got = check_context_fit(est, profile, c["requested_max_output_tokens"])
    exp = c["expected"]
    assert (got.fits, got.output_reserve, got.required_tokens) == (
        exp["fits"], exp["output_reserve"], exp["required_tokens"])
    assert (got.failure.value if got.failure else None) == exp["failure"]


def _profile(context_window: int, max_output_tokens: int) -> ModelProfile:
    return ModelProfile(
        id="m1", tag="example/m:t", context_window=context_window, max_output_tokens=max_output_tokens
    )


# ---- metrics ------------------------------------------------------------------------------


@pytest.mark.parametrize("c", load_fixture("metrics_normalization.json")["cases"], ids=lambda c: c["name"][:60])
def test_metrics_normalization(c):
    raw = RawOllamaMetrics.model_validate(c["raw"]) if c["raw"] is not None else None
    got = normalize_ollama_metrics(raw, routing_ms=c["routing_ms"], wall_ms=c["wall_ms"], ttft_ms=c["ttft_ms"])
    assert approx_equal(got.model_dump(mode="json"), c["expected"]), got.model_dump(mode="json")


def test_raw_values_are_preserved_verbatim():
    raw = {"total_duration": 7, "load_duration": 3, "prompt_eval_count": 1, "prompt_eval_duration": 2,
           "eval_count": 5, "eval_duration": 11}
    got = normalize_ollama_metrics(RawOllamaMetrics(**raw), routing_ms=1.0, wall_ms=2.0, ttft_ms=0.5)
    assert got.raw_ollama is not None and got.raw_ollama.model_dump() == raw


def test_no_throughput_without_measured_tokens_even_if_wall_time_exists():
    got = normalize_ollama_metrics(None, routing_ms=1.0, wall_ms=5000.0, ttft_ms=10.0)
    assert got.generation_tokens_per_second is None and got.prompt_tokens_per_second is None


# ---- privacy ------------------------------------------------------------------------------


@pytest.mark.parametrize("c", load_fixture("privacy.json")["redaction"], ids=lambda c: c["name"])
def test_redaction(c):
    ev = parse_event(c["event"])
    out = redact_for_persistence(ev, Retention(**c["retention"]))
    if c.get("unchanged"):
        assert out == ev
    else:
        assert isinstance(out, GenerationDeltaEvent)
        assert out.payload.text == c["expected_text"] and out.payload.redacted == c["expected_redacted"]
        assert out.payload.index == ev.payload.index  # type: ignore[union-attr]
        if c["expected_redacted"]:
            assert "Rain" not in out.model_dump_json()


def test_redaction_does_not_mutate_the_live_event():
    ev = parse_event(load_fixture("privacy.json")["redaction"][0]["event"])
    redact_for_persistence(ev, Retention())
    assert ev.payload.text == "Rain"  # type: ignore[union-attr]


def test_privacy_defaults_are_all_off():
    from localhydra.contracts.privacy import PrivacyConfig
    from localhydra.contracts.config import AppConfig

    assert PrivacyConfig().model_dump() == {
        "retain_prompts": False, "retain_responses": False, "log_content": False, "export_content": False}
    assert PrivacyConfig().retention() == Retention(prompts=False, responses=False)
    models_only = {"models": json.loads(EXAMPLE_CONFIG.read_text())["models"]}
    assert AppConfig.model_validate(models_only).privacy == PrivacyConfig()


# ---- loopback -----------------------------------------------------------------------------


@pytest.mark.parametrize("c", load_fixture("loopback.json")["urls"], ids=lambda c: c["url"])
def test_provider_urls(c):
    if c["valid"]:
        assert validate_loopback_url(c["url"]) == c["url"].rstrip("/")
    else:
        with pytest.raises(ValueError):
            validate_loopback_url(c["url"])


@pytest.mark.parametrize("c", load_fixture("loopback.json")["bind_hosts"], ids=lambda c: c["host"])
def test_bind_hosts(c):
    if c["valid"]:
        assert ServerConfig(host=c["host"]).host == c["host"]
    else:
        with pytest.raises(ValidationError):
            ServerConfig(host=c["host"])


@pytest.mark.parametrize("c", load_fixture("loopback.json")["resolved"], ids=lambda c: ",".join(c["addresses"]) or "none")
def test_resolved_addresses(c):
    assert all_loopback(c["addresses"]) is c["all_loopback"]
