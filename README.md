# localHydra

Observable, local-only model routing with Ollama.

## Product goal
Choose among 4–5 user-configured local models according to task requirements, measured capability, latency, context limits, and loaded-model state. Show the actual route, model, progress, timing, tokens, errors, and results in a local dashboard.

## MVP boundaries
- Local inference only, including any router/classifier and embeddings. No cloud fallback, external telemetry, CDN assets, or automatic model downloads.
- Models and hardware are not selected yet. Use exact installed Ollama tags and explicit operator-provided capability/context profiles; do not invent model suitability or benchmark results.
- Start with deterministic routing and one solver per turn. Add an optional small local task classifier behind a replaceable interface.
- Include multi-turn chat, explicit model override, cancellation, durable traces, replay, and benchmark comparison.
- Exclude repository editing, arbitrary shell execution, autonomous tool agents, parallel critics, training a router, and production deployment from MVP. Treat these as later milestones requiring isolation and permission design.

## Proposed implementation defaults
Python 3.11+ / FastAPI / Pydantic / httpx / SQLite backend; React / TypeScript / Vite dashboard; pytest and Playwright tests. These are implementation defaults, not installed scaffolding. LH-001 may change a default with a written rationale before dependent work begins.

## Start here
Read docs/architecture.md and tickets/README.md. The numbered ticket files are the implementation backlog. LH-001 (scaffold and frozen contracts) and LH-002 (Ollama adapter and telemetry) are done; see docs/development.md for commands and the adapter guide, and docs/contracts.md for the contracts. Routing, the run runtime/API and the dashboard are not implemented yet (LH-003 onward).
