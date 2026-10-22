"""Authoring tool for contracts/fixtures/*.json and config/localhydra.example.json.

Stdlib only and deliberately independent of the pydantic models: fixtures are expectations that
the models (and the TypeScript side) are tested against, not output copied from them.

ALL DATA HERE IS SYNTHETIC. Model tags, scores, latencies and metrics are invented placeholders
for contract tests; they are not benchmarks, recommendations or real measurements.

    python contracts/tools/build_fixtures.py          # rewrite files
    python contracts/tools/build_fixtures.py --check  # fail if committed files differ
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "contracts" / "fixtures"
EXAMPLE_CONFIG = ROOT / "config" / "localhydra.example.json"

SYNTHETIC = "SYNTHETIC contract fixture: invented placeholder data, not a measurement or recommendation."


def rid(n: int) -> str:
    return f"run_{n:032x}"


def cid(n: int) -> str:
    return f"conv_{n:032x}"


def ph(n: int) -> str:
    return f"ph_{n:032x}"


CONV = cid(1)


def ts(ms: int = 0) -> str:
    s, milli = divmod(ms, 1000)
    return f"2026-10-22T10:00:{s:02d}.{milli:03d}Z"


# ---------------------------------------------------------------- model profiles / config


def profile(i: int, **over):
    p = {
        "id": f"model-{i}",
        "tag": f"example/model-{i}:synthetic",
        "enabled": True,
        "display_name": f"Synthetic model {i}",
        "context_window": 8192,
        "max_output_tokens": 1024,
        "input_modalities": ["text"],
        "supports_tools": False,
        "task_suitability": {
            "chat": {"score": 0.5, "source": "operator_configured", "note": None},
            "code": {"score": 0.25, "source": "operator_configured", "note": None},
        },
        "latency": {"ttft_ms": 800.0, "tokens_per_second": 20.0, "source": "configured"},
    }
    p.update(over)
    return p


def app_config(n: int = 4):
    return {
        "server": {"host": "127.0.0.1", "port": 8765},
        "provider": {"kind": "ollama", "base_url": "http://127.0.0.1:11434", "follow_redirects": False},
        "privacy": {
            "retain_prompts": False,
            "retain_responses": False,
            "log_content": False,
            "export_content": False,
        },
        "conversation": {"max_turns_in_memory": 50},
        "models": [profile(i) for i in range(1, n + 1)],
    }


def status(i: int, availability: str, reason, loaded: str, observed):
    return {
        "profile": profile(i),
        "availability": availability,
        "availability_reason": reason,
        "loaded_state": loaded,
        "observed_at": observed,
    }


def case(name, schema, data, valid=True, schema_valid=None, note=None):
    c = {"name": name, "schema": schema, "valid": valid}
    if schema_valid is not None and schema_valid != valid:
        c["schema_valid"] = schema_valid
    if note:
        c["note"] = note
    c["data"] = data
    return c


def cases_file(description, cases):
    return {
        "synthetic": True,
        "label": SYNTHETIC,
        "description": description,
        "cases": cases,
    }


def models_fixture():
    cfg = app_config()

    def mut(fn):
        c = copy.deepcopy(cfg)
        fn(c)
        return c

    stat_ok = lambda: [  # noqa: E731
        status(1, "available", None, "loaded", ts(0)),
        status(2, "available", None, "not_loaded", ts(0)),
        status(3, "unavailable", "not_installed", "not_loaded", ts(0)),
    ]
    disabled = status(4, "unavailable", "disabled", "not_loaded", ts(0))
    disabled["profile"]["enabled"] = False

    cases = [
        case("profile: valid text model", "ModelProfile", profile(1)),
        case(
            "profile: valid vision + tools, nullable latency",
            "ModelProfile",
            profile(
                2,
                input_modalities=["text", "image"],
                supports_tools=True,
                latency={"ttft_ms": None, "tokens_per_second": None, "source": "unknown"},
                task_suitability={},
            ),
        ),
        case("profile: tag without explicit :tag is not exact", "ModelProfile", profile(1, tag="example/model-1"), False),
        case("profile: uppercase id rejected", "ModelProfile", profile(1, id="Model-1"), False),
        case("profile: score above 1 rejected", "ModelProfile", profile(
            1, task_suitability={"chat": {"score": 1.5, "source": "operator_configured", "note": None}}), False),
        case("profile: max_output_tokens must be < context_window", "ModelProfile",
             profile(1, context_window=1024, max_output_tokens=1024), False, True),
        case("profile: modalities must include text", "ModelProfile",
             profile(1, input_modalities=["image"]), False, True),
        case("profile: latency values need configured/measured source", "ModelProfile",
             profile(1, latency={"ttft_ms": 5.0, "tokens_per_second": None, "source": "unknown"}), False, True),
        case("profile: no download/pull field exists", "ModelProfile", {**profile(1), "pull_if_missing": True}, False),
        case("config: valid 4 models (also config/localhydra.example.json)", "AppConfig", cfg),
        case("config: valid 5 models", "AppConfig", app_config(5)),
        case("config: 3 models rejected", "AppConfig", app_config(3), False),
        case("config: 6 models rejected", "AppConfig", app_config(6), False),
        case("config: duplicate tag (case-insensitive) rejected", "AppConfig",
             mut(lambda c: c["models"][1].update(tag="example/model-1:SYNTHETIC")), False, True),
        case("config: duplicate id rejected", "AppConfig",
             mut(lambda c: c["models"][1].update(id="model-1")), False, True),
        case("config: non-loopback provider rejected", "AppConfig",
             mut(lambda c: c["provider"].update(base_url="http://example.com:11434")), False, True),
        case("config: provider URL with path rejected", "AppConfig",
             mut(lambda c: c["provider"].update(base_url="http://127.0.0.1:11434/v1")), False, True),
        case("config: redirects can never be enabled", "AppConfig",
             mut(lambda c: c["provider"].update(follow_redirects=True)), False),
        case("config: non-loopback bind host rejected", "AppConfig",
             mut(lambda c: c["server"].update(host="0.0.0.0")), False, True),
        case("config: unknown key rejected (no auto-pull switch)", "AppConfig",
             mut(lambda c: c.update(pull_missing_models=True)), False),
        case("config: export_content needs retention", "AppConfig",
             mut(lambda c: c["privacy"].update(export_content=True)), False, True),
        case("privacy: all defaults off", "PrivacyConfig",
             {"retain_prompts": False, "retain_responses": False, "log_content": False, "export_content": False}),
        case("privacy: opt in to response retention", "PrivacyConfig",
             {"retain_prompts": False, "retain_responses": True, "log_content": False, "export_content": False}),
        case("privacy: export without retention rejected", "PrivacyConfig",
             {"retain_prompts": False, "retain_responses": False, "log_content": False, "export_content": True},
             False, True),
        case("models response: unconfigured (no config file)", "ModelsResponse",
             {"configured": False, "provider_probe": "not_probed", "auto_download": False, "models": []}),
        case("models response: configured but provider not probed -> availability unknown", "ModelsResponse",
             {"configured": True, "provider_probe": "not_probed", "auto_download": False, "models": [
                 status(i, "unknown", "provider_not_probed", "unknown", None) for i in range(1, 5)]}),
        case("models response: probed, missing model is unavailable (never downloaded)", "ModelsResponse",
             {"configured": True, "provider_probe": "ok", "auto_download": False,
              "models": stat_ok() + [disabled]}),
        case("models response: auto_download true is a contract violation", "ModelsResponse",
             {"configured": True, "provider_probe": "ok", "auto_download": True, "models": stat_ok()},
             False, True),
    ]
    return cases_file("Model profiles, validated config, privacy defaults, model availability.", cases)


# ---------------------------------------------------------------- metrics

RAW_COMPLETE = {
    "total_duration": 2_500_000_000,
    "load_duration": 500_000_000,
    "prompt_eval_count": 26,
    "prompt_eval_duration": 100_000_000,
    "eval_count": 100,
    "eval_duration": 1_600_000_000,
}


def metrics_complete():
    return {
        "routing_ms": 3.5,
        "wall_ms": 2600.0,
        "ttft_ms": 700.0,
        "load_ms": 500.0,
        "prompt_eval_ms": 100.0,
        "generation_ms": 1600.0,
        "ollama_total_ms": 2500.0,
        "prompt_tokens": 26,
        "generated_tokens": 100,
        "prompt_tokens_per_second": 260.0,
        "generation_tokens_per_second": 62.5,
        "raw_ollama": dict(RAW_COMPLETE),
        "completeness": {"status": "complete", "missing": [], "reason": None},
    }


def metrics_no_final_chunk():
    return {
        "routing_ms": 2.0,
        "wall_ms": 350.0,
        "ttft_ms": 120.0,
        "load_ms": None,
        "prompt_eval_ms": None,
        "generation_ms": None,
        "ollama_total_ms": None,
        "prompt_tokens": None,
        "generated_tokens": None,
        "prompt_tokens_per_second": None,
        "generation_tokens_per_second": None,
        "raw_ollama": None,
        "completeness": {
            "status": "incomplete",
            "missing": [
                "load_ms", "prompt_eval_ms", "generation_ms", "ollama_total_ms",
                "prompt_tokens", "generated_tokens",
            ],
            "reason": "no_final_chunk",
        },
    }


def raw_nulls(**kw):
    r = {k: None for k in RAW_COMPLETE}
    r.update(kw)
    return r


def metrics_fixture():
    inconsistent = metrics_complete()
    inconsistent["generation_tokens_per_second"] = 99.0
    wrong_missing = metrics_no_final_chunk()
    wrong_missing["completeness"]["missing"] = ["load_ms"]
    complete_but_null = metrics_complete()
    complete_but_null["load_ms"] = None
    invented_rate = metrics_no_final_chunk()
    invented_rate["generation_tokens_per_second"] = 10.0

    cases = [
        case("normalized: complete", "NormalizedMetrics", metrics_complete()),
        case("normalized: incomplete, stream ended before final chunk", "NormalizedMetrics", metrics_no_final_chunk()),
        case("normalized: throughput must equal tokens/duration", "NormalizedMetrics", inconsistent, False, True),
        case("normalized: missing list must match null fields", "NormalizedMetrics", wrong_missing, False, True),
        case("normalized: 'complete' with a null required field", "NormalizedMetrics", complete_but_null, False, True),
        case("normalized: throughput is never invented without measured tokens", "NormalizedMetrics",
             invented_rate, False, True),
        case("raw ollama: all fields optional, preserved verbatim", "RawOllamaMetrics", dict(RAW_COMPLETE)),
        case("raw ollama: negative duration rejected", "RawOllamaMetrics",
             {**RAW_COMPLETE, "load_duration": -1}, False),
        case("progress: only app-observed values", "ProgressMetrics",
             {"elapsed_ms": 400.0, "visible_chunks": 2, "ttft_ms": 120.0}),
        case("progress: no Ollama stage durations allowed live", "ProgressMetrics",
             {"elapsed_ms": 400.0, "visible_chunks": 2, "ttft_ms": 120.0, "load_ms": 5.0}, False),
    ]
    return cases_file("Metrics schema cases.", cases)


def normalization_fixture():
    cases = [
        {
            "name": "complete final chunk",
            "raw": dict(RAW_COMPLETE),
            "routing_ms": 3.5, "wall_ms": 2600.0, "ttft_ms": 700.0,
            "expected": metrics_complete(),
        },
        {
            "name": "cancelled/failed before final chunk: app timings kept, rest null, flagged incomplete",
            "raw": None,
            "routing_ms": 2.0, "wall_ms": 350.0, "ttft_ms": 120.0,
            "expected": metrics_no_final_chunk(),
        },
        {
            "name": "final chunk without prompt fields (e.g. cached prompt): flagged partial, no invented prompt tokens",
            "raw": raw_nulls(total_duration=1_200_000_000, load_duration=10_000_000,
                             eval_count=50, eval_duration=800_000_000),
            "routing_ms": 1.0, "wall_ms": 1250.0, "ttft_ms": 90.0,
            "expected": {
                "routing_ms": 1.0, "wall_ms": 1250.0, "ttft_ms": 90.0,
                "load_ms": 10.0, "prompt_eval_ms": None, "generation_ms": 800.0, "ollama_total_ms": 1200.0,
                "prompt_tokens": None, "generated_tokens": 50,
                "prompt_tokens_per_second": None, "generation_tokens_per_second": 62.5,
                "raw_ollama": raw_nulls(total_duration=1_200_000_000, load_duration=10_000_000,
                                        eval_count=50, eval_duration=800_000_000),
                "completeness": {"status": "incomplete", "missing": ["prompt_eval_ms", "prompt_tokens"],
                                 "reason": "partial_final_chunk"},
            },
        },
        {
            "name": "no visible token: only ttft is missing",
            "raw": {"total_duration": 60_000_000, "load_duration": 1_000_000, "prompt_eval_count": 10,
                    "prompt_eval_duration": 50_000_000, "eval_count": 0, "eval_duration": 1_000_000},
            "routing_ms": 1.0, "wall_ms": 70.0, "ttft_ms": None,
            "expected": {
                "routing_ms": 1.0, "wall_ms": 70.0, "ttft_ms": None,
                "load_ms": 1.0, "prompt_eval_ms": 50.0, "generation_ms": 1.0, "ollama_total_ms": 60.0,
                "prompt_tokens": 10, "generated_tokens": 0,
                "prompt_tokens_per_second": 200.0, "generation_tokens_per_second": 0.0,
                "raw_ollama": {"total_duration": 60_000_000, "load_duration": 1_000_000, "prompt_eval_count": 10,
                               "prompt_eval_duration": 50_000_000, "eval_count": 0, "eval_duration": 1_000_000},
                "completeness": {"status": "incomplete", "missing": ["ttft_ms"], "reason": "no_visible_token"},
            },
        },
        {
            "name": "zero measured duration: no division by zero, throughput stays null",
            "raw": {"total_duration": 5_000_000, "load_duration": 0, "prompt_eval_count": 3,
                    "prompt_eval_duration": 1_000_000, "eval_count": 5, "eval_duration": 0},
            "routing_ms": 0.5, "wall_ms": 6.0, "ttft_ms": 2.0,
            "expected": {
                "routing_ms": 0.5, "wall_ms": 6.0, "ttft_ms": 2.0,
                "load_ms": 0.0, "prompt_eval_ms": 1.0, "generation_ms": 0.0, "ollama_total_ms": 5.0,
                "prompt_tokens": 3, "generated_tokens": 5,
                "prompt_tokens_per_second": 3000.0, "generation_tokens_per_second": None,
                "raw_ollama": {"total_duration": 5_000_000, "load_duration": 0, "prompt_eval_count": 3,
                               "prompt_eval_duration": 1_000_000, "eval_count": 5, "eval_duration": 0},
                "completeness": {"status": "complete", "missing": [], "reason": None},
            },
        },
    ]
    return {"synthetic": True, "label": SYNTHETIC,
            "description": "Inputs/outputs of normalize_ollama_metrics (executed by the Python reference).",
            "cases": cases}


# ---------------------------------------------------------------- routing


def ctx_estimate(upper=42):
    return {"method": "ascii3_nonascii1_v1", "message_count": 1, "point_tokens": 33,
            "upper_bound_tokens": upper, "margin": 1.25, "uncertainty": "Synthetic fixture value."}


def fit(ok=True, failure=None, window=8192, upper=42, reserve=1024):
    return {"fits": ok, "context_window": window, "upper_bound_tokens": upper,
            "output_reserve": reserve, "required_tokens": upper + reserve, "failure": failure}


def cand(i, eligible=True, rejections=None, score=None, components=None, context_fit=None):
    return {"model_id": f"model-{i}", "tag": f"example/model-{i}:synthetic", "eligible": eligible,
            "rejections": rejections or [], "context_fit": context_fit,
            "components": components or [], "score": score}


def comps(a, b):
    return [
        {"name": "task_suitability", "value": a, "weight": 0.5, "contribution": a * 0.5},
        {"name": "latency", "value": b, "weight": 0.5, "contribution": b * 0.5},
    ]


def requirements(**over):
    r = {"task_type": "chat", "required_modalities": ["text"], "requires_tools": False,
         "source": "rules", "confidence": None, "classifier_id": None}
    r.update(over)
    return r


def decision(**over):
    d = {
        "policy_version": "synthetic-policy-0",
        "mode": "auto",
        "requirements": requirements(),
        "context_estimate": ctx_estimate(),
        "candidates": [
            cand(1, True, [], 0.625, comps(0.75, 0.5), fit()),
            cand(2, True, [], 0.375, comps(0.5, 0.25), fit()),
            cand(3, False, ["not_installed"]),
            cand(4, False, ["context_exceeded"], context_fit=fit(False, "context_exceeded", 512, 42, 1024)),
        ],
        "outcome": "selected",
        "selected_model_id": "model-1",
        "reasons": ["highest_score"],
        "override_applied": False,
        "sticky_applied": False,
        "decided_at": ts(5),
    }
    d.update(over)
    return d


def decision_none():
    return decision(
        candidates=[cand(1, False, ["not_installed"]), cand(2, False, ["disabled"]),
                    cand(3, False, ["modality_unsupported"]), cand(4, False, ["tools_unsupported"])],
        outcome="no_eligible_model", selected_model_id=None, reasons=[])


def routing_fixture():
    sel_inelig = decision(selected_model_id="model-3")
    not_in = decision(selected_model_id="model-9")
    ovr_no_reason = decision(override_applied=True)
    manual_no_ovr = decision(mode="manual")
    none_but_eligible = decision(outcome="no_eligible_model", selected_model_id=None, reasons=[])
    cases = [
        case("requirements: rules", "RoutingRequirements", requirements()),
        case("requirements: classifier with confidence", "RoutingRequirements",
             requirements(source="classifier", confidence=0.75, classifier_id="synthetic-classifier")),
        case("requirements: confidence only valid for classifier source", "RoutingRequirements",
             requirements(confidence=0.5), False, True),
        case("requirements: classifier cannot name a model", "RoutingRequirements",
             {**requirements(source="classifier"), "model": "model-1"}, False),
        case("requirements: classifier cannot name an endpoint", "RoutingRequirements",
             {**requirements(source="classifier"), "endpoint": "http://example.com"}, False),
        case("decision: selected by highest score, rejected candidates recorded", "RoutingDecision", decision()),
        case("decision: manual override", "RoutingDecision",
             decision(mode="manual", override_applied=True, reasons=["manual_override"])),
        case("decision: sticky conversation", "RoutingDecision",
             decision(sticky_applied=True, reasons=["sticky_conversation"])),
        case("decision: no eligible model", "RoutingDecision", decision_none()),
        case("decision: ineligible candidate can never be selected", "RoutingDecision", sel_inelig, False, True),
        case("decision: selected model must be a candidate", "RoutingDecision", not_in, False, True),
        case("decision: override_applied needs manual_override reason", "RoutingDecision", ovr_no_reason, False, True),
        case("decision: manual selection must record override", "RoutingDecision", manual_no_ovr, False, True),
        case("decision: no_eligible_model while a candidate is eligible", "RoutingDecision",
             none_but_eligible, False, True),
    ]
    return cases_file("Routing requirement/decision invariants.", cases)


# ---------------------------------------------------------------- events


QUEUED_PAYLOAD = {"routing_mode": "auto", "model_override": None, "message_count": 1,
                  "retention": {"prompts": False, "responses": False}}
STARTED_PAYLOAD = {"model_id": "model-1", "tag": "example/model-1:synthetic", "reserved_output_tokens": 1024}


def ev(seq, typ, phase, phase_id, payload, run=1):
    return {"schema_version": 1, "run_id": rid(run), "conversation_id": CONV, "phase_id": phase_id,
            "phase": phase, "sequence": seq, "timestamp": ts(seq * 10), "type": typ, "payload": payload}


def delta(i, text, redacted=False):
    return {"index": i, "text": None if redacted else text, "redacted": redacted}


def stream(*steps):
    """steps: (type, phase, phase_id, payload). Auto-numbers sequence/timestamps."""
    return [ev(i + 1, *s) for i, s in enumerate(steps)]


Q = ("run.queued", "queued", ph(1), QUEUED_PAYLOAD)
RS = ("routing.started", "routing", ph(2), {})
RD = ("routing.decided", "routing", ph(2), {"decision": decision()})
GS = ("generation.started", "generating", ph(3), STARTED_PAYLOAD)


def D(i, text="x", redacted=False):
    return ("generation.delta", "generating", ph(3), delta(i, text, redacted))


PR = ("generation.progress", "generating", ph(3),
      {"progress": {"elapsed_ms": 400.0, "visible_chunks": 2, "ttft_ms": 120.0}})
GM = ("generation.metrics", "generating", ph(3), {"metrics": metrics_complete()})
GM_INC = ("generation.metrics", "generating", ph(3), {"metrics": metrics_no_final_chunk()})
CR_GEN = ("run.cancel_requested", "generating", ph(3), {"source": "api"})
DONE = ("run.completed", "generating", ph(3), {"finish_reason": "stop", "visible_chunks": 2})


def FAIL(phase, phase_id, code="provider_unreachable", msg="Provider failed"):
    return ("run.failed", phase, phase_id,
            {"error": {"code": code, "message": msg, "retryable": False, "details": None, "request_id": None}})


def events_fixture():
    base = stream(Q, RS, RD, GS, D(0, "Rain"), D(1, " falls"), PR, GM, DONE)
    cases = [
        case("event: run.queued", "Event", base[0]),
        case("event: routing.started (empty payload)", "Event", base[1]),
        case("event: routing.decided carries decision with rejected candidates", "Event", base[2]),
        case("event: generation.started", "Event", base[3]),
        case("event: generation.delta live text", "Event", base[4]),
        case("event: generation.delta redacted copy", "Event", ev(5, "generation.delta", "generating", ph(3), delta(0, "", True))),
        case("event: generation.progress (app-observed only)", "Event", base[6]),
        case("event: generation.metrics final", "Event", base[7]),
        case("event: run.cancel_requested", "Event", ev(5, *CR_GEN)),
        case("event: run.completed", "Event", base[8]),
        case("event: run.failed", "Event", ev(5, *FAIL("generating", ph(3)))),
        case("event: run.cancelled", "Event", ev(6, "run.cancelled", "generating", ph(3), {"partial_output": True})),
        case("event: run.interrupted (recovery)", "Event", ev(3, "run.interrupted", "routing", ph(2), {"previous_status": "routing"})),
        case("event: unknown type rejected", "Event", ev(1, "run.exploded", "queued", ph(1), {}), False),
        case("event: schema_version 2 rejected", "Event", {**base[0], "schema_version": 2}, False),
        case("event: extra envelope field rejected", "Event", {**base[0], "debug": True}, False),
        case("event: sequence 0 rejected", "Event", {**base[0], "sequence": 0}, False),
        case("event: naive timestamp rejected", "Event", {**base[0], "timestamp": "2026-10-22T10:00:00"}, False),
        case("event: payload must match type", "Event", {**base[8], "payload": FAIL("generating", ph(3))[3]}, False),
        case("event: bad run id rejected", "Event", {**base[0], "run_id": "run_123"}, False),
        case("event: delta text and redacted must agree", "Event",
             ev(5, "generation.delta", "generating", ph(3), {"index": 0, "text": "x", "redacted": True}), False, True),
        case("event: delta with redacted=false requires text", "Event",
             ev(5, "generation.delta", "generating", ph(3), {"index": 0, "text": None, "redacted": False}), False, True),
        case("event: empty delta text rejected", "Event",
             ev(5, "generation.delta", "generating", ph(3), {"index": 0, "text": "", "redacted": False}), False, True),
    ]
    return cases_file("Single-event schema cases.", cases)


def mutate(events, fn):
    e = copy.deepcopy(events)
    fn(e)
    return e


def renumber(events):
    events = copy.deepcopy(events)
    for i, e in enumerate(events):
        e["sequence"] = i + 1
        e["timestamp"] = ts((i + 1) * 10)
    return events


def stream_case(name, events, valid=True, contains=None, require_terminal=False):
    return {"name": name, "valid": valid, "require_terminal": require_terminal,
            "expect_violation_contains": contains or [], "events": events}


def streams_fixture():
    ok = stream(Q, RS, RD, GS, D(0, "Rain"), D(1, " falls"), PR, GM, DONE)
    noel = stream(Q, RS, ("routing.decided", "routing", ph(2), {"decision": decision_none()}),
                  FAIL("routing", ph(2), "no_eligible_model", "No eligible model"))
    cancel_gen = stream(Q, RS, RD, GS, D(0, "Rain"), CR_GEN, GM_INC,
                        ("run.cancelled", "generating", ph(3), {"partial_output": True}))
    race = stream(Q, RS, RD, GS, D(0, "Rain"), D(1, " falls"), CR_GEN, GM, DONE)
    cancel_q = stream(Q, ("run.cancel_requested", "queued", ph(1), {"source": "api"}),
                      ("run.cancelled", "queued", ph(1), {"partial_output": False}))
    interrupted = stream(Q, RS, ("run.interrupted", "routing", ph(2), {"previous_status": "routing"}))
    failed_gen = stream(Q, RS, RD, GS, D(0, "Rain"), FAIL("generating", ph(3)))
    redacted = stream(Q, RS, RD, GS, D(0, redacted=True), D(1, redacted=True), GM, DONE)

    cases = [
        stream_case("valid: full completed run", ok, require_terminal=True),
        stream_case("valid: no eligible model fails in routing phase", noel, require_terminal=True),
        stream_case("valid: cancel while generating, incomplete metrics", cancel_gen, require_terminal=True),
        stream_case("valid RACE: cancel requested but completion persisted first -> completed wins",
                    race, require_terminal=True),
        stream_case("valid: cancel while queued", cancel_q, require_terminal=True),
        stream_case("valid: recovery interrupts unfinished run", interrupted, require_terminal=True),
        stream_case("valid: provider failure mid-generation without metrics", failed_gen, require_terminal=True),
        stream_case("valid: persisted copy with redacted deltas", redacted, require_terminal=True),
        stream_case("valid: in-progress prefix (no terminal yet)", ok[:5]),
        stream_case("invalid: prefix required to be terminal", ok[:5], False, ["no terminal event"], True),
        stream_case("invalid: two terminal events", ok + [ev(10, *FAIL("generating", ph(3)))], False,
                    ["event after terminal"]),
        stream_case("invalid RACE: cancelled persisted after completed", ok + [
            ev(10, *CR_GEN), ev(11, "run.cancelled", "generating", ph(3), {"partial_output": True})],
                    False, ["event after terminal"]),
        stream_case("invalid: sequence gap", renumber_gap(ok), False, ["sequence must be"]),
        stream_case("invalid: first event is not run.queued", ok[1:], False, ["run.queued must be first"]),
        stream_case("invalid: delta index skips", mutate(ok, lambda e: e[5]["payload"].update(index=5)), False,
                    ["delta index must be"]),
        stream_case("invalid: completion without final metrics", renumber([x for x in ok if x["type"] != "generation.metrics"]),
                    False, ["requires final generation.metrics"]),
        stream_case("invalid: generation before routing decision", renumber([x for x in ok if x["type"] != "routing.decided"]),
                    False, ["requires a prior routing.decided"]),
        stream_case("invalid: generation model differs from routed model",
                    mutate(ok, lambda e: e[3]["payload"].update(model_id="model-2", tag="example/model-2:synthetic")),
                    False, ["differs from routed model"]),
        stream_case("invalid: cancelled without cancel_requested",
                    renumber(mutate(cancel_gen, lambda e: e.pop(5))),
                    False, ["requires a prior run.cancel_requested"]),
        stream_case("invalid: duplicate cancel_requested (cancel is idempotent)",
                    renumber(cancel_gen[:6] + [copy.deepcopy(cancel_gen[5])] + cancel_gen[6:]),
                    False, ["duplicate cancel_requested"]),
        stream_case("invalid: delta carries wrong phase_id",
                    mutate(ok, lambda e: e[4].update(phase_id=ph(9))), False, ["phase_id does not match"]),
        stream_case("invalid: phase name disagrees with lifecycle",
                    mutate(ok, lambda e: e[4].update(phase="routing")), False, ["does not match lifecycle state"]),
        stream_case("invalid: no_eligible decision must fail with no_eligible_model",
                    mutate(noel, lambda e: e[3]["payload"]["error"].update(code="internal_error")), False,
                    ["must fail with no_eligible_model"]),
        stream_case("invalid: no_eligible_model failure without such a decision",
                    mutate(failed_gen, lambda e: e[5]["payload"]["error"].update(code="no_eligible_model")), False,
                    ["requires a no_eligible_model routing decision"]),
        stream_case("invalid: events from two runs mixed",
                    mutate(ok, lambda e: e[4].update(run_id=rid(2))), False, ["run_id/conversation_id changed"]),
        stream_case("invalid: interrupted previous_status mismatch",
                    mutate(interrupted, lambda e: e[2]["payload"].update(previous_status="generating")), False,
                    ["previous_status does not match"]),
        stream_case("invalid: timestamps go backwards",
                    mutate(ok, lambda e: e[4].update(timestamp=ts(1))), False, ["timestamp went backwards"]),
    ]
    return {"synthetic": True, "label": SYNTHETIC,
            "description": "Whole-run event streams checked by validate_event_stream (Python reference); "
                           "each event is also schema-validated in both languages.",
            "cases": cases}


def renumber_gap(events):
    e = copy.deepcopy(events)
    e.pop(5)  # remove seq 6 -> 5 followed by 7
    return e


# ---------------------------------------------------------------- lifecycle / sse / context / privacy / loopback

STATES = ["queued", "routing", "generating", "completed", "failed", "cancelled", "interrupted"]
NONTERMINAL = ["queued", "routing", "generating"]
ALLOWED = {
    "queued": {"routing", "failed", "cancelled", "interrupted"},
    "routing": {"generating", "failed", "cancelled", "interrupted"},
    "generating": {"completed", "failed", "cancelled", "interrupted"},
}


def lifecycle_fixture():
    transitions = []
    for cur in STATES:
        for tgt in STATES:
            for recovery in (False, True):
                if cur in NONTERMINAL and tgt in ALLOWED[cur] and (tgt != "interrupted" or recovery):
                    exp = {"accepted": True, "status": tgt, "rejection": None}
                elif cur not in NONTERMINAL:
                    exp = {"accepted": False, "status": cur, "rejection": "already_terminal"}
                elif tgt == "interrupted" and not recovery:
                    exp = {"accepted": False, "status": cur, "rejection": "interrupted_requires_recovery"}
                else:
                    exp = {"accepted": False, "status": cur, "rejection": "illegal_transition"}
                transitions.append({"current": cur, "target": tgt, "recovery": recovery, "expected": exp})
    cancels = []
    for st in STATES:
        for already in (False, True):
            if st not in NONTERMINAL:
                exp = {"effect": "already_terminal", "append_cancel_requested_event": False,
                       "finalize_cancelled_now": False}
            elif st == "queued":
                exp = {"effect": "cancelled", "append_cancel_requested_event": not already,
                       "finalize_cancelled_now": True}
            else:
                exp = {"effect": "cancel_requested", "append_cancel_requested_event": not already,
                       "finalize_cancelled_now": False}
            cancels.append({"status": st, "cancel_already_requested": already, "expected": exp})
    recovery = [{"persisted": s, "recovered": s if s not in NONTERMINAL else "interrupted"} for s in STATES]
    return {"synthetic": True, "label": SYNTHETIC,
            "description": "Lifecycle compare-and-set, idempotent cancel decisions and restart recovery.",
            "transitions": transitions, "cancels": cancels, "recovery": recovery}


def sse_fixture():
    persisted = stream(Q, RS, RD, GS, D(0, "Rain"), GM, DONE)
    return {
        "synthetic": True, "label": SYNTHETIC,
        "description": "SSE framing, Last-Event-ID parsing/replay, client de-duplication.",
        "frame": {"event": persisted[0], "expected_id": "1", "expected_event": "run.queued"},
        "preamble": "retry: 2000\n\n",
        "keepalive": ": keepalive\n\n",
        "last_event_id": [
            {"header": None, "query": None, "expected": 0},
            {"header": "", "query": None, "expected": 0},
            {"header": "0", "query": None, "expected": 0},
            {"header": "42", "query": None, "expected": 42},
            {"header": None, "query": "7", "expected": 7},
            {"header": "3", "query": "7", "expected": 3, "note": "header wins over query"},
            {"header": "-1", "query": None, "error": "invalid_last_event_id"},
            {"header": "abc", "query": None, "error": "invalid_last_event_id"},
            {"header": "1.5", "query": None, "error": "invalid_last_event_id"},
            {"header": "0x10", "query": None, "error": "invalid_last_event_id"},
            {"header": " 3", "query": None, "error": "invalid_last_event_id"},
            {"header": "99999999999999999", "query": None, "error": "invalid_last_event_id"},
        ],
        "replay": {
            "persisted": persisted,
            "cases": [
                {"last_event_id": 0, "expected_sequences": [1, 2, 3, 4, 5, 6, 7]},
                {"last_event_id": 4, "expected_sequences": [5, 6, 7]},
                {"last_event_id": 7, "expected_sequences": []},
                {"last_event_id": 8, "error": "invalid_last_event_id",
                 "note": "beyond last persisted event: events are persisted before delivery"},
            ],
        },
        "tracker": [
            {"name": "in order", "sequences": [1, 2, 3], "expected": ["apply", "apply", "apply"]},
            {"name": "duplicates after reconnect are dropped", "sequences": [1, 2, 3, 2, 3, 4],
             "expected": ["apply", "apply", "apply", "duplicate", "duplicate", "apply"]},
            {"name": "gap is reported, not applied", "sequences": [1, 2, 4],
             "expected": ["apply", "apply", "gap"], "last_after": 2},
            {"name": "gap then replay fills it", "sequences": [1, 3, 2, 3],
             "expected": ["apply", "gap", "apply", "apply"], "last_after": 3},
            {"name": "sequence 0 / stale is duplicate", "sequences": [0], "expected": ["duplicate"]},
        ],
    }


def context_fixture():
    return {
        "synthetic": True, "label": SYNTHETIC,
        "description": "Context estimation method ascii3_nonascii1_v1 (hand-computed expectations).",
        "estimates": [
            {"name": "one ascii message (25 chars): ceil(25/3)=9 +8 +16 = 33; ceil(33*1.25)=42",
             "contents": ["Write a haiku about rain."], "point_tokens": 33, "upper_bound_tokens": 42},
            {"name": "no messages: only fixed overhead 16 -> 20", "contents": [],
             "point_tokens": 16, "upper_bound_tokens": 20},
            {"name": "non-ASCII counts one token per code point: 3 +8 +16 = 27 -> 34",
             "contents": ["日本語"], "point_tokens": 27, "upper_bound_tokens": 34},
            {"name": "mixed: 3 ascii (1) + 2 non-ascii (2) = 3; +16 +16 = 35 -> 44",
             "contents": ["abc", "日本"], "point_tokens": 35, "upper_bound_tokens": 44},
            {"name": "300 ascii chars: 100 +8 +16 = 124 -> 155",
             "contents": ["a" * 300], "point_tokens": 124, "upper_bound_tokens": 155},
        ],
        "fits": [
            {"name": "fits with default reserve", "context_window": 8192, "max_output_tokens": 1024,
             "upper_bound_tokens": 42, "requested_max_output_tokens": None,
             "expected": {"fits": True, "output_reserve": 1024, "required_tokens": 1066, "failure": None}},
            {"name": "exactly at the window fits", "context_window": 1024, "max_output_tokens": 512,
             "upper_bound_tokens": 512, "requested_max_output_tokens": None,
             "expected": {"fits": True, "output_reserve": 512, "required_tokens": 1024, "failure": None}},
            {"name": "one token over does not fit", "context_window": 1024, "max_output_tokens": 512,
             "upper_bound_tokens": 513, "requested_max_output_tokens": None,
             "expected": {"fits": False, "output_reserve": 512, "required_tokens": 1025,
                          "failure": "context_exceeded"}},
            {"name": "smaller requested output frees room", "context_window": 1024, "max_output_tokens": 512,
             "upper_bound_tokens": 600, "requested_max_output_tokens": 256,
             "expected": {"fits": True, "output_reserve": 256, "required_tokens": 856, "failure": None}},
            {"name": "requested output above profile budget is rejected, not clamped", "context_window": 8192,
             "max_output_tokens": 1024, "upper_bound_tokens": 42, "requested_max_output_tokens": 2048,
             "expected": {"fits": False, "output_reserve": 2048, "required_tokens": 2090,
                          "failure": "output_budget_exceeded"}},
        ],
    }


def privacy_fixture():
    d = ev(5, "generation.delta", "generating", ph(3), delta(0, "Rain"))
    done = ev(9, *DONE)
    return {
        "synthetic": True, "label": SYNTHETIC,
        "description": "Persisted-copy redaction (redact_for_persistence). Live SSE delivery is never redacted.",
        "redaction": [
            {"name": "responses not retained -> text removed", "event": d,
             "retention": {"prompts": False, "responses": False}, "expected_text": None, "expected_redacted": True},
            {"name": "prompt retention alone does not retain responses", "event": d,
             "retention": {"prompts": True, "responses": False}, "expected_text": None, "expected_redacted": True},
            {"name": "responses retained -> text kept", "event": d,
             "retention": {"prompts": False, "responses": True}, "expected_text": "Rain", "expected_redacted": False},
            {"name": "non-delta events are untouched", "event": done,
             "retention": {"prompts": False, "responses": False}, "unchanged": True},
        ],
    }


def loopback_fixture():
    return {
        "synthetic": True, "label": SYNTHETIC,
        "description": "Loopback-only rules for provider URLs, bind hosts and resolved addresses.",
        "urls": [
            {"url": "http://127.0.0.1:11434", "valid": True},
            {"url": "http://localhost:11434", "valid": True},
            {"url": "http://[::1]:11434", "valid": True},
            {"url": "http://127.5.5.5:11434", "valid": True},
            {"url": "https://127.0.0.1:11434", "valid": True},
            {"url": "http://127.0.0.1:11434/", "valid": True},
            {"url": "http://0.0.0.0:11434", "valid": False},
            {"url": "http://192.168.1.10:11434", "valid": False},
            {"url": "http://example.com:11434", "valid": False},
            {"url": "http://127.0.0.1.example.com:11434", "valid": False},
            {"url": "http://user:pw@127.0.0.1:11434", "valid": False},
            {"url": "ftp://127.0.0.1", "valid": False},
            {"url": "http://127.0.0.1:11434/api", "valid": False},
            {"url": "http://127.0.0.1:11434?x=1", "valid": False},
            {"url": "127.0.0.1:11434", "valid": False},
        ],
        "bind_hosts": [
            {"host": "127.0.0.1", "valid": True}, {"host": "::1", "valid": True},
            {"host": "0.0.0.0", "valid": False}, {"host": "::", "valid": False},
            {"host": "192.168.0.2", "valid": False}, {"host": "localhost", "valid": False},
        ],
        "resolved": [
            {"addresses": ["127.0.0.1"], "all_loopback": True},
            {"addresses": ["127.0.0.1", "::1"], "all_loopback": True},
            {"addresses": ["127.0.0.1", "93.184.216.34"], "all_loopback": False},
            {"addresses": [], "all_loopback": False},
            {"addresses": ["not-an-ip"], "all_loopback": False},
        ],
    }


# ---------------------------------------------------------------- api


def run_summary(**over):
    s = {
        "run_id": rid(1), "conversation_id": CONV, "status": "completed", "routing_mode": "auto",
        "selected_model_id": "model-1", "created_at": ts(0), "updated_at": ts(90), "finished_at": ts(90),
        "last_sequence": 9, "retention": {"prompts": False, "responses": False},
        "metrics_status": "complete", "error": None,
    }
    s.update(over)
    return s


def run_detail(**over):
    d = run_summary()
    d.update({
        "model_override": None, "routing": decision(), "metrics": metrics_complete(),
        "cancel_requested": False, "messages": None, "response_text": None,
        "transcript_replay_available": False,
    })
    d.update(over)
    return d


def err(code, message, details=None):
    return {"error": {"code": code, "message": message, "retryable": False, "details": details,
                      "request_id": "0123456789abcdef0123456789abcdef"}}


def api_fixture():
    health = {"status": "ok", "service": "localhydra", "version": "0.1.0", "contract_version": "1.0.0",
              "event_schema_version": 1, "local_only": True, "bind_host": "127.0.0.1",
              "ollama_target": "http://127.0.0.1:11434", "models_configured": 4, "time": ts(0)}
    msgs = [{"role": "user", "content": "Write a haiku about rain."}]
    req = {"conversation_id": None, "messages": msgs, "routing_mode": "auto", "model_override": None,
           "max_output_tokens": None}
    failed_summary = run_summary(
        status="failed", metrics_status=None, selected_model_id=None,
        error=err("no_eligible_model", "No eligible model")["error"])
    page_item = run_summary()
    convo = {
        "conversation_id": CONV, "created_at": ts(0), "updated_at": ts(90), "sticky_model_id": "model-1",
        "active_run_id": None, "run_ids": [rid(1)],
        "history": {"max_turns": 50, "turns": 1, "dropped_turns": 0},
        "messages": msgs + [{"role": "assistant", "content": "Rain falls softly."}],
        "persistence": "memory_only", "lost_on_restart": True,
    }
    cases = [
        case("health", "HealthResponse", health),
        case("health: local_only cannot be false", "HealthResponse", {**health, "local_only": False}, False),
        case("error: validation (no input echo)", "ErrorEnvelope",
             err("validation_error", "Request validation failed",
                 {"errors": [{"loc": ["body", "messages"], "type": "too_short"}]})),
        case("error: no eligible model", "ErrorEnvelope", err("no_eligible_model", "No eligible model")),
        case("error: unknown code rejected", "ErrorEnvelope", err("boom", "x"), False),
        case("error: missing message rejected", "ErrorEnvelope", {"error": {"code": "not_found"}}, False),
        case("create run: auto", "CreateRunRequest", req),
        case("create run: manual override with output cap and conversation", "CreateRunRequest",
             {**req, "conversation_id": CONV, "routing_mode": "manual", "model_override": "model-2",
              "max_output_tokens": 256}),
        case("create run: omitted optional fields use defaults", "CreateRunRequest", {"messages": msgs}),
        case("create run: manual without override rejected", "CreateRunRequest",
             {**req, "routing_mode": "manual"}, False, True),
        case("create run: override without manual rejected", "CreateRunRequest",
             {**req, "model_override": "model-1"}, False, True),
        case("create run: last message must be from user", "CreateRunRequest",
             {**req, "messages": [{"role": "assistant", "content": "hi"}]}, False, True),
        case("create run: empty messages rejected", "CreateRunRequest", {**req, "messages": []}, False),
        case("create run: override is a profile id, not an Ollama tag", "CreateRunRequest",
             {**req, "routing_mode": "manual", "model_override": "example/model-1:synthetic"}, False),
        case("create run: unknown field (endpoint) rejected", "CreateRunRequest",
             {**req, "endpoint": "http://example.com"}, False),
        case("create run response", "CreateRunResponse",
             {"run_id": rid(1), "conversation_id": CONV, "status": "queued",
              "events_url": f"/api/runs/{rid(1)}/events", "created_at": ts(0)}),
        case("create run response: status must be queued", "CreateRunResponse",
             {"run_id": rid(1), "conversation_id": CONV, "status": "routing",
              "events_url": f"/api/runs/{rid(1)}/events", "created_at": ts(0)}, False, True),
        case("run summary: completed", "RunSummary", run_summary()),
        case("run summary: failed carries error", "RunSummary", failed_summary),
        case("run summary: queued run is unfinished", "RunSummary",
             run_summary(status="queued", selected_model_id=None, finished_at=None, last_sequence=1,
                         metrics_status=None)),
        case("run summary: failed without error rejected", "RunSummary", run_summary(status="failed"), False, True),
        case("run summary: terminal without finished_at rejected", "RunSummary",
             run_summary(finished_at=None), False, True),
        case("run detail: completed, content not retained (privacy default)", "RunDetail", run_detail()),
        case("run detail: completed with opt-in retention", "RunDetail", run_detail(
            retention={"prompts": True, "responses": True}, messages=msgs,
            response_text="Rain falls softly.", transcript_replay_available=True)),
        case("run detail: cancel requested while generating", "RunDetail", run_detail(
            status="generating", finished_at=None, last_sequence=6, metrics=None, metrics_status=None,
            cancel_requested=True)),
        case("run detail: messages without prompt retention rejected", "RunDetail",
             run_detail(messages=msgs), False, True),
        case("run detail: response_text without response retention rejected", "RunDetail",
             run_detail(response_text="leak"), False, True),
        case("run detail: replay flag must follow retention", "RunDetail",
             run_detail(transcript_replay_available=True), False, True),
        case("run page: with next cursor", "RunPage",
             {"items": [page_item], "limit": 1, "next_cursor": "opaque-cursor-1", "has_more": True}),
        case("run page: last page", "RunPage", {"items": [page_item], "limit": 20, "next_cursor": None, "has_more": False}),
        case("run page: has_more requires next_cursor", "RunPage",
             {"items": [], "limit": 20, "next_cursor": None, "has_more": True}, False, True),
        case("run page: items may not exceed limit", "RunPage",
             {"items": [page_item, page_item], "limit": 1, "next_cursor": None, "has_more": False}, False, True),
        case("run page: limit over 100 rejected", "RunPage",
             {"items": [], "limit": 101, "next_cursor": None, "has_more": False}, False),
        case("list query: defaults", "ListRunsQuery", {}),
        case("list query: filters", "ListRunsQuery",
             {"limit": 50, "cursor": "opaque-cursor-1", "status": "completed", "conversation_id": CONV}),
        case("list query: limit 0 rejected", "ListRunsQuery", {"limit": 0}, False),
        case("cancel: cancelled (queued run)", "CancelResponse",
             {"run_id": rid(1), "status": "cancelled", "cancel_effect": "cancelled"}),
        case("cancel: requested (running)", "CancelResponse",
             {"run_id": rid(1), "status": "generating", "cancel_effect": "cancel_requested"}),
        case("cancel: already terminal (idempotent)", "CancelResponse",
             {"run_id": rid(1), "status": "completed", "cancel_effect": "already_terminal"}),
        case("cancel: unknown effect rejected", "CancelResponse",
             {"run_id": rid(1), "status": "completed", "cancel_effect": "maybe"}, False),
        case("conversation: memory only, bounded", "ConversationDetail", convo),
        case("conversation: turns above max rejected", "ConversationDetail",
             {**convo, "history": {"max_turns": 2, "turns": 3, "dropped_turns": 0}}, False, True),
        case("conversation: persistence must disclose memory_only", "ConversationDetail",
             {**convo, "persistence": "sqlite"}, False, True),
        case("conversation reset", "ConversationResetResponse",
             {"conversation_id": CONV, "reset": True, "cleared_turns": 1, "runs_retained": True}),
        case("conversation reset: idempotent second call clears 0", "ConversationResetResponse",
             {"conversation_id": CONV, "reset": True, "cleared_turns": 0, "runs_retained": True}),
        case("conversation reset: negative cleared_turns rejected", "ConversationResetResponse",
             {"conversation_id": CONV, "reset": True, "cleared_turns": -1, "runs_retained": True}, False),
    ]
    return cases_file("HTTP request/response/error schema cases.", cases)


# ---------------------------------------------------------------- main


def build_all():
    files = {
        "models.json": models_fixture(),
        "metrics.json": metrics_fixture(),
        "metrics_normalization.json": normalization_fixture(),
        "routing.json": routing_fixture(),
        "events.json": events_fixture(),
        "streams.json": streams_fixture(),
        "lifecycle.json": lifecycle_fixture(),
        "sse.json": sse_fixture(),
        "context.json": context_fixture(),
        "privacy.json": privacy_fixture(),
        "loopback.json": loopback_fixture(),
        "api.json": api_fixture(),
    }
    out = {OUT / name: json.dumps(data, indent=2, ensure_ascii=False) + "\n" for name, data in files.items()}
    example = app_config(4)
    out[EXAMPLE_CONFIG] = json.dumps(example, indent=2, ensure_ascii=False) + "\n"
    return out


def main(argv):
    out = build_all()
    if "--check" in argv:
        bad = [str(p) for p, text in out.items() if not p.exists() or p.read_text() != text]
        if bad:
            print("fixtures out of date:\n  " + "\n  ".join(bad), file=sys.stderr)
            return 1
        print(f"fixtures OK: {len(out)} files match build_fixtures.py")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    EXAMPLE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    for p, text in out.items():
        p.write_text(text)
    print(f"wrote {len(out)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
