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
| Contracts in sync (schemas, fixtures, TS types) | `backend/.venv/bin/python -m localhydra.contracts.registry --check` ; `backend/.venv/bin/python contracts/tools/build_fixtures.py --check` ; `npm run contracts:check --prefix frontend` |
| Regenerate contracts after an intended change | `backend/.venv/bin/python -m localhydra.contracts.registry` ; `backend/.venv/bin/python contracts/tools/build_fixtures.py` ; `npm run contracts:generate --prefix frontend` |

`scripts/` contains convenience wrappers (`install.sh`, `check.sh`, `contracts.sh`, `smoke.sh`, `dev-backend.sh`, `dev-frontend.sh`; run with `bash scripts/<name>`). Independent follow-up verification executed `bash scripts/check.sh` and `bash scripts/smoke.sh` successfully after enabling owner execute permission and fixing HTTP newline parsing in the smoke script. The check includes both test suites, contract checks and the frontend build; the smoke checks live loopback services and socket binding. Install/dev wrappers have not been independently executed; the individual commands above remain the verified install path.

Backend config: with no `--config` the backend runs "unconfigured" (`/api/models` returns `configured:false`, no models). `config/localhydra.example.json` is a SYNTHETIC example with placeholder tags (`example/model-N:synthetic`); copy it to `config/localhydra.json` (git-ignored) and replace tags and capability numbers with your real installed Ollama tags and measured/operator values. 4 to 5 profiles are required, tags must be explicit `name:tag`, unique, and are matched exactly.

## File ownership (parallel workers)

| Area | Owner | Rule |
|---|---|---|
| `backend/src/localhydra/contracts/`, `contracts/` (schemas, fixtures, tools), `frontend/src/contracts/` (incl. `generated/`), `docs/contracts.md` | LH-001 (frozen) | Change only through the process in `docs/contracts.md`. |
| `backend/src/localhydra/app.py`, `settings.py`, `__main__.py`, scaffold files, `scripts/`, `docs/development.md` | LH-001 | Downstream tickets may extend `create_app` by adding routers; do not alter health/models behaviour without a contract change. |
| `backend/src/localhydra/adapters/` (Ollama adapter, telemetry) | LH-002 | Reuse `contracts.metrics.normalize_ollama_metrics`, `contracts.config.all_loopback`. |
| `backend/src/localhydra/routing/` (pure module) | LH-003 | Produce `RoutingDecision` objects that satisfy the contract validators; use `contracts.context`. |
| `backend/src/localhydra/runtime/`, SQLite, API routes | LH-004 | Implement the lifecycle/SSE/privacy semantics in `contracts/` exactly; add routers to `create_app`. |
| `frontend/src/` except `contracts/` (dashboard, mock server) | LH-005 | Consume `frontend/src/contracts/generated` types; do not hand-edit them. |
| `backend/src/localhydra/classifier/` | LH-006 | Output `RoutingRequirements` only. |
| evaluation CLI / task set | LH-007 | Own area, e.g. `backend/src/localhydra/evaluation/`. |
| integration docs / runbook | LH-008 | |

Each owner adds tests beside their code (`backend/tests/test_<area>_*.py`, `frontend/tests/`). Do not edit another owner's files without coordination; contract needs go through the change process.

## Limits of the scaffold

Only `GET /api/health` and `GET /api/models` exist. `/api/models` reports availability `unknown` (provider not probed) until LH-002/LH-004 add the Ollama probe. Anything else answers with the standard 404 error envelope.
