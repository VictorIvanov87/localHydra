# Development guide (LH-001)

Run everything from the repository root. Prerequisites: Python 3.11+ (checks were run on 3.12.12; `uv` is used if present), Node 20+ (run on 22.22.0) and npm. Dependency downloads happen only during install; the application never downloads models, weights or assets, and makes no cloud calls.

## Verified commands

These are the exact commands that were executed for LH-001 (see the ticket for results).

| Purpose | Command |
|---|---|
| Create backend venv | `uv venv backend/.venv --python 3.12` (or `python3.12 -m venv backend/.venv`) |
| Install backend | `uv pip install --python backend/.venv/bin/python -c backend/constraints.txt -e "backend[dev]"` (or `backend/.venv/bin/python -m pip install -c backend/constraints.txt -e "backend[dev]"`) |
| Install frontend | `npm ci --prefix frontend` |
| Start backend (127.0.0.1:8765) | `backend/.venv/bin/python -m localhydra [--config config/localhydra.example.json] [--port N]` |
| Start frontend dev (127.0.0.1:5173, proxies `/api`) | `npm run dev --prefix frontend` |
| Serve built frontend (127.0.0.1:4173, proxies `/api`) | `npm run build --prefix frontend && npm run preview --prefix frontend` |
| Backend tests | `backend/.venv/bin/python -m pytest backend -q` |
| Frontend typecheck | `npm run typecheck --prefix frontend` |
| Frontend contract-fixture validation | `npm run test --prefix frontend` |
| Frontend typecheck + build | `npm run build --prefix frontend` |
| Adapter tests only (LH-002; synthetic local fake servers) | `backend/.venv/bin/python -m pytest backend/tests -k adapter -q` |
| Live Ollama smoke (LH-002; read-only listing, never pulls) | `backend/.venv/bin/python -m localhydra.adapters.live_smoke --model <installed name:tag>` (exit 0 ran, 2 not run, 1 failed; without `--model` it only lists installed tags) |
| Contracts in sync (schemas, fixtures, TS types) | `backend/.venv/bin/python -m localhydra.contracts.registry --check` ; `backend/.venv/bin/python contracts/tools/build_fixtures.py --check` ; `npm run contracts:check --prefix frontend` |
| Regenerate contracts after an intended change | `backend/.venv/bin/python -m localhydra.contracts.registry` ; `backend/.venv/bin/python contracts/tools/build_fixtures.py` ; `npm run contracts:generate --prefix frontend` |

`scripts/` contains convenience wrappers (`install.sh`, `check.sh`, `contracts.sh`, `smoke.sh`, `dev-backend.sh`, `dev-frontend.sh`; run with `bash scripts/<name>`). Independent follow-up verification executed `bash scripts/check.sh` and `bash scripts/smoke.sh` successfully after enabling owner execute permission and fixing HTTP newline parsing in the smoke script. The check includes both test suites, contract checks and the frontend build; the smoke checks live loopback services and socket binding. Install/dev wrappers have not been independently executed; the individual commands above remain the verified install path.

Backend config: with no `--config` the backend runs "unconfigured" (`/api/models` returns `configured:false`, no models). `config/localhydra.example.json` is a SYNTHETIC example with placeholder tags (`example/model-N:synthetic`); copy it to `config/localhydra.json` (git-ignored) and replace tags and capability numbers with your real installed Ollama tags and measured/operator values. 4 to 5 profiles are required, tags must be explicit `name:tag`, unique, and are matched exactly.

## File ownership (parallel workers)

| Area | Owner | Rule |
|---|---|---|
| `backend/src/localhydra/contracts/`, `contracts/` (schemas, fixtures, tools), `frontend/src/contracts/` (incl. `generated/`), `docs/contracts.md` | LH-001 (frozen) | Change only through the process in `docs/contracts.md`. |
| `backend/src/localhydra/app.py`, `settings.py`, `__main__.py`, scaffold files, `scripts/`, `docs/development.md` | LH-001 | Downstream tickets may extend `create_app` by adding routers; do not alter health/models behaviour without a contract change. |
| `backend/src/localhydra/adapters/` (Ollama adapter, telemetry), `backend/tests/test_adapter_*.py`, `backend/tests/provider_fakes.py` | LH-002 | Reuse `contracts.metrics.normalize_ollama_metrics`, `contracts.config.all_loopback`. LH-004 may import `tests.provider_fakes` (synthetic fake Ollama server, `SyntheticProvider`). |
| `backend/src/localhydra/routing/` (pure module) | LH-003 | Produce `RoutingDecision` objects that satisfy the contract validators; use `contracts.context`. |
| `backend/src/localhydra/runtime/`, SQLite, API routes | LH-004 | Implement the lifecycle/SSE/privacy semantics in `contracts/` exactly; add routers to `create_app`. |
| `frontend/src/` except `contracts/` (dashboard, mock server) | LH-005 | Consume `frontend/src/contracts/generated` types; do not hand-edit them. |
| `backend/src/localhydra/classifier/` | LH-006 | Output `RoutingRequirements` only. |
| evaluation CLI / task set | LH-007 | Own area, e.g. `backend/src/localhydra/evaluation/`. |
| integration docs / runbook | LH-008 | |

Each owner adds tests beside their code (`backend/tests/test_<area>_*.py`, `frontend/tests/`). Do not edit another owner's files without coordination; contract needs go through the change process.

## Ollama adapter (LH-002)

`localhydra.adapters` is transport only (no routing, run lifecycle or UI). Downstream code should depend on the `ModelProvider` protocol (`adapters/provider.py`) and receive an `OllamaProvider` (or a test double) by injection.

| Capability | API |
|---|---|
| Discovery (installed tags, exact) | `await provider.list_models()` -> `InstalledModels` (`tags`, `local_tags`, `remote_tags`) |
| Loaded state | `await provider.list_loaded()` -> `LoadedModels`; `supported=False` (no `/api/ps`) means *unknown*, not "none loaded" |
| Profiles + provider state -> contract statuses | `await probe_models(provider, profiles)` -> `ModelProbe` (`statuses` from `resolve_model_statuses`, `probe_status`, `remote_tags`) |
| Streaming chat | `stream = provider.chat_stream(ChatRequest(...), cancel=event)`; `async for item in stream` yields `ChatDelta`... then exactly one `ChatFinished` or `ChatCancelled`, or raises `ProviderError`. Use `contextlib.aclosing(stream)` when you may stop early. |
| Structured output | `await provider.chat_structured(request, PydanticModel)` -> `StructuredResult` (schema sent as Ollama `format`, reply validated against the model) |
| Telemetry | `stream.telemetry.progress()` (live, app-observed only), `item.telemetry.normalized(routing_ms=...)` -> contract `NormalizedMetrics` |
| Timeouts | `ProviderTimeouts(connect_s, request_s, first_chunk_s, idle_s, total_s)`; defaults are operational, not capability claims |

Behaviour that downstream tickets can rely on:
- **Local only.** The target must be a loopback origin; `localhost` is resolved by the adapter on every request, all answers must be loopback, and the connection goes to the validated IP (no second lookup). Redirects are refused (`redirect_refused`), environment proxies are ignored. This is an application boundary, not OS isolation.
- **No downloads.** Only `GET /api/tags`, `GET /api/ps` and `POST /api/chat` exist in the code; a missing tag is `model_not_found` (-> `ErrorCode.model_unavailable`).
- **No cloud models.** Ollama can list cloud-proxied tags (`remote_host`/`remote_model` in `/api/tags`, or a `cloud` tag name). `chat_stream`/`chat_structured` refuse them (`remote_model_refused`) after one cached `/api/tags` preflight per tag, and `probe_models` reports them as unavailable plus in `remote_tags`. Operators should not list such tags in profiles.
- **Typed errors.** `ProviderError.kind` (`ProviderErrorKind`) maps onto the frozen `ErrorCode`s (`to_error_body()` yields a contract `ErrorBody`). Errors never contain prompt/response text and never forward Ollama's own error message.
- **Cancellation.** Setting the cancel event, `stream.cancel()`, `aclose()` or cancelling the consuming task closes the HTTP connection promptly. Ollama only sees a closed connection; whether it stops computing is **not confirmed** (`ChatCancelled.upstream_stop_confirmed` is always `False`).
- **Telemetry.** Raw Ollama values are kept verbatim; values that are not non-negative integers are dropped and named in `discarded_metric_fields`, never coerced. `wall_ms` starts when the chat request is about to be sent (it includes connection and, on a cold model, model load). `ttft_ms` is taken when the adapter hands the first non-blank `content` delta to its consumer (a lower bound of what the client sees; `thinking` text is not visible content). `routing_ms` belongs to the runtime: without it the contract lists it as missing. Structured calls are not streamed, so they have no `ttft_ms`.

Run a live smoke only against a model you already pulled yourself; it never downloads anything, and a chat loads that model into Ollama's memory like any normal request.

## Limits of the scaffold

Only `GET /api/health` and `GET /api/models` exist. `/api/models` still reports availability `unknown` (provider not probed): the adapter's `probe_models` exists but is deliberately not wired into the app (LH-004 owns the routes and probe scheduling). Anything else answers with the standard 404 error envelope.
