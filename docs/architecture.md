# MVP architecture and boundaries

## Defaults
Proposed stack: Python 3.11+ / FastAPI / Pydantic / httpx / SQLite; React / TypeScript / Vite; pytest and Playwright. LH-001 may change defaults with a written rationale before downstream implementation. No dependencies are installed by this planning package.

Flow: dashboard -> local run service -> requirements classifier + routing policy -> Ollama. Structured events feed durable history and the live dashboard. A classifier is optional; deterministic rules are the baseline.

## Contract to freeze in LH-001
- GET /api/health and /api/models.
- POST /api/runs: conversation ID optional, messages, routing mode, model override optional; returns run/conversation IDs.
- GET /api/runs and /api/runs/{id}; paginated history.
- GET /api/runs/{id}/events: SSE, durable sequence IDs, Last-Event-ID replay and client deduplication.
- POST /api/runs/{id}/cancel: idempotent.
- Conversation read/reset APIs and consistent error envelopes.
- Model profiles: exact tag, eligibility, configured context/output budgets, modality/tool support, task suitability, measured or configured latency, observed loaded state. Support 4–5 profiles without hardcoded names.
- Events: schema version, run/conversation/phase IDs, sequence, timestamp, type, payload. Cover routing, generation, deltas, metrics, cancellation and terminal outcomes.
- Lifecycle: queued -> routing -> generating -> completed/failed/cancelled; exactly one terminal outcome under races. Recover unfinished persisted runs as interrupted after restart.

## Measurement honesty
Preserve raw available Ollama measurements. Define normalized units for load, prompt processing, generation, wall time, routing time, time to first visible token, and prompt/generated tokens. Loading/prefill durations normally become available with final metrics: do not invent live stage signals. Incomplete stream metrics stay flagged incomplete. Throughput uses measured tokens/duration, not words. Do not infer GPU memory, energy, cache hit rate, or dollar savings.

## Routing and classification
Hard eligibility before ranking: installed allowlist, declared modality/features, conservative context fit with output reservation. Token estimates disclose their method/uncertainty. No eligible model produces an explicit error. Sticky conversations and manual overrides do not bypass eligibility. Keep classifier requirements separate from model mapping; classifier cannot name arbitrary endpoints or expand permissions. Decisions record rejected candidates, ranking components and policy version.

## Privacy/security
Only loopback Ollama targets in MVP; validate resolved addresses and redirects. Prompt/response retention is opt-in across DB, persisted events, logs and exports. Ephemeral streaming can show content without retaining it. Metadata history remains replayable; transcript replay is unavailable when not stored. Bound in-memory conversation retention and disclose loss on restart. Local-only is an application boundary, not a promise of OS-enforced network isolation.

## Not in MVP
Repository editing, shell tools, autonomous agent teams, parallel critics, router training, cloud providers, production deployment. Future cascade workflows require objective verification, bounded retries, explicit workspace isolation and permission design.
