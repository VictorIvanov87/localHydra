# localHydra implementation backlog

Eight implementation tickets are prepared. LH-001 and LH-002 are DONE; the rest are TODO: writing these tickets does not mean their implementation is done. Statuses: TODO, IN_PROGRESS, BLOCKED, DONE. DONE requires every acceptance criterion and recorded real execution evidence.

## Execution order and parallelism
1. LH-001 freezes scaffold and executable contracts.
2. LH-002 (adapter), LH-003 (routing), and LH-005 (dashboard with mocks) can run in parallel in separate owned areas.
3. LH-004 joins adapter/routing. LH-006 may start isolated classifier development after LH-002/LH-003, but requires LH-004 to finish integration. LH-004 never depends on LH-006.
4. LH-007 evaluates the runtime after LH-004; optional classifier comparison must not block the rules baseline.
5. LH-008 integrates and verifies everything, including live Ollama. Missing live models/hardware is an explicit blocker.

| ID | Task | Required dependencies for DONE | Status |
|---|---|---|---|
| [LH-001](LH-001.md) | Scaffold and contracts | None | DONE |
| [LH-002](LH-002.md) | Ollama adapter and telemetry | LH-001 | DONE |
| [LH-003](LH-003.md) | Explainable routing | LH-001 | TODO |
| [LH-004](LH-004.md) | Runtime, conversations and events | LH-002, LH-003 | TODO |
| [LH-005](LH-005.md) | Dashboard and controls | LH-001 | TODO |
| [LH-006](LH-006.md) | Optional local classifier | LH-002, LH-003, LH-004 | TODO |
| [LH-007](LH-007.md) | Evaluation and baselines | LH-004 | TODO |
| [LH-008](LH-008.md) | Integration and live acceptance | LH-004, LH-005, LH-006, LH-007 | TODO |

## Claude Code starting prompt
Read README.md, docs/architecture.md, tickets/README.md and tickets/LH-001.md. Implement LH-001 only. Add tests, execute required checks, and record actual commands/results in the ticket. Mark DONE only when every acceptance criterion passes; otherwise describe the blocker. Do not implement downstream tickets or claim live-model success from mocks.

For subsequent work, replace LH-001 with a ready ticket and verify its prerequisites. Once a Git repository is initialized, parallel workers should use separate branches/worktrees and avoid editing shared contracts without coordination. This planning package does not create a remote repository or launch Claude Code.
