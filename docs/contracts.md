# Contracts v1.0.0 (frozen by LH-001)

Source of truth: pydantic models in `backend/src/localhydra/contracts/`. They export JSON Schemas to `contracts/schemas/` (with `MANIFEST.json` hashes and `endpoints.json`), which generate the TypeScript types in `frontend/src/contracts/generated/`. SYNTHETIC fixtures in `contracts/fixtures/` (authored by `contracts/tools/build_fixtures.py`, stdlib only, independent of the models) are validated by pytest (pydantic, full semantics) and vitest (Ajv against the JSON Schemas). Cross-field rules JSON Schema cannot express are marked `schema_valid: true, valid: false` in fixtures and enforced by Python only. All fixture data is invented placeholder data, never measurements.

| Module | Defines |
|---|---|
| `common.py` | ID formats (`run_`/`conv_`/`ph_` + 32 hex), exact Ollama tag pattern, `Page`/`PageQuery`, versions |
| `errors.py` | `ErrorEnvelope {error:{code,message,retryable,details,request_id}}`, all codes and HTTP status map |
| `models.py` | Model profile (exact tag, enabled, context/output budgets, modalities, tools, task suitability with source, latency with source), 4-5 profile config, availability/loaded state, `resolve_model_statuses` |
| `config.py` | App config (server, provider, privacy, conversation); loopback URL/host/resolved-address rules |
| `routing.py` | Requirements (separate from model mapping), rejections, ranking components, `RoutingDecision` invariants |
| `context.py` | Context estimation method and fit rule |
| `metrics.py` | Raw Ollama metrics, normalized units, completeness, `normalize_ollama_metrics` |
| `lifecycle.py` | Status machine, compare-and-set `transition`, `decide_cancel`, `recover_status` |
| `events.py` | Event envelope, 12 typed events, `validate_event_stream`, `redact_for_persistence` |
| `sse.py` | Frame format, `Last-Event-ID` parsing/replay, `SequenceTracker` |
| `runs.py` | Create/list/detail/cancel/conversation shapes, pagination |
| `privacy.py` | Retention and privacy defaults |
| `api.py` | Health response and the endpoint registry |

## Frozen semantics (summary; the code and fixtures are normative)

- **Endpoints**: `GET /api/health`, `GET /api/models`, `POST /api/runs` (202), `GET /api/runs` (cursor pages, `limit` 1-100, default 20, newest first), `GET /api/runs/{id}`, `GET /api/runs/{id}/events` (SSE), `POST /api/runs/{id}/cancel`, `GET /api/conversations/{id}`, `POST /api/conversations/{id}/reset`. Only the first two are implemented (LH-001); the rest by LH-004. Every error is an `ErrorEnvelope`; validation errors list only `loc` and `type`, never input values.
- **Lifecycle**: `queued -> routing -> generating -> completed | failed | cancelled`; `interrupted` is produced only by startup recovery of unfinished persisted runs. The first terminal transition persisted wins; status write and terminal event append are one atomic step; later terminal attempts are rejected and append nothing. Cancel: terminal run -> 200 `already_terminal`, no event; queued -> `cancelled` immediately; routing/generating -> one `run.cancel_requested` event, effect `cancel_requested`, repeats add no event. If completion/failure is persisted before the worker honours cancel, that outcome stands (`cancel_requested` then `completed` is valid); a worker that sees the flag before persisting completion must finalize `cancelled`.
- **Events**: `sequence` 1.. gapless per run, equals the SSE `id`, persisted before delivery. A new `phase_id` starts at `run.queued`, `routing.started`, `generation.started`. Completion requires the final `generation.metrics` first (flagged incomplete when data is missing). `no_eligible_model` routing outcome must end in `run.failed` with that code. Live stage signals are limited to app-observed `generation.progress`; load/prefill/decode durations appear only in final metrics.
- **SSE**: `id/event/data` frames, `retry: 2000` preamble, `: keepalive` every 15 s, replay of sequences `> Last-Event-ID` (header beats `?last_event_id=`), 400 `invalid_last_event_id` for malformed or beyond-last-persisted ids, stream closes after the terminal event. Clients drop duplicates and reconnect on a gap (`SequenceTracker`, identical in Python and TypeScript, tested against the same fixture).
- **Context estimation** `ascii3_nonascii1_v1`: `point = ceil(ascii/3 + non_ascii) + 8*messages + 16`, `upper = ceil(point*1.25)`, fits iff `upper + output_reserve <= context_window`; reserve is the requested max output or the profile budget; requesting more than the profile budget is rejected, not clamped. The method and uncertainty ship with every estimate.
- **Metrics**: raw Ollama nanosecond values preserved verbatim; normalized to ms/tokens; throughput only from measured tokens and duration (null otherwise, never from words); missing inputs stay null and are listed in `completeness.missing` with a reason; no GPU memory, energy, cache hit rate or cost fields exist.
- **Privacy defaults**: all retention, content logging and content export are off. Streaming is always live; persisted delta copies are redacted unless response retention is on; `RunDetail` content fields are null unless the matching retention flag is on; conversation context is bounded in-memory only and disclosed as lost on restart.
- **Local-only**: bind host must be a loopback IP literal; provider URL must be a bare loopback origin without credentials; redirects are never followed; resolved addresses must all be loopback (`all_loopback`, to be applied by LH-002 at connect time); the `Host` header must be a loopback name; FastAPI's CDN-backed docs are disabled. This is an application boundary, not OS-enforced isolation.
- **Models**: 4-5 profiles, explicit `name:tag`, unique ids and tags (case-insensitive), matched exactly (no `:latest` inference). Not-installed models are `unavailable`, never pulled; no config field can enable downloads. An unprobed provider yields `unknown`, not `available`.

## Contract change process

1. Open the change with the owners of affected tickets (see `docs/development.md`); parallel workers must not edit contracts unilaterally.
2. Edit the pydantic models (and semantics functions) in `backend/src/localhydra/contracts/`; update `contracts/tools/build_fixtures.py` with valid and invalid cases for the change.
3. Versioning: additive, optional changes bump the minor version; any change that breaks existing payloads or semantics bumps the major version (`CONTRACT_VERSION` in `common.py`) and, for event payloads, `EVENT_SCHEMA_VERSION`. Patch for documentation/fixture-only fixes.
4. Regenerate (commands in `docs/development.md`): schemas + manifest, fixtures, TypeScript types. Commit all generated output with the source change.
5. Run all checks: backend pytest, `--check` for schemas/fixtures/types, frontend typecheck, vitest, build. The drift tests fail if generated files are stale, so an uncoordinated edit cannot go unnoticed.
6. Record the change in `contracts/CHANGELOG.md` with the new version and notify downstream owners; tickets already DONE that depend on the old contract must be re-verified.
