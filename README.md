# Forage Search

**Forage** is a standalone, local-first, persistent web-research agent. It investigates user-defined **research watches** over repeated bounded sessions, saves evidence and research state, and reports **new or materially changed findings** instead of starting from scratch every time.

Repository: `forage-search` · Working product name: **Forage** · Stage: **pre-implementation**.

## Why it exists

A user can assign ongoing research such as “find MScAC-relevant AI/ML internships,” “monitor Toronto AI/technology events,” or any custom information need. Forage should work unattended, cite primary sources, use prior work to choose the next investigation, and avoid repeating old results.

Longer term, Tari/OpenClaw can use Forage as a **separate research worker** for Discord forum-based research watches. The research worker must remain independently operable without Tari, Discord, OpenClaw, or any particular cloud model.

A possible CSC2524 AI+HCI research extension adds **adaptive human-guidance checkpoints**: when a watch hits an objective-changing decision that cannot be resolved within the user's approved scope, request a review, set that watch to `AWAITING_REVIEW`, and allocate the shared local compute budget to other eligible watches. This is a hypothesis to implement and evaluate, **not an established feature or proven benefit**.

## Suggested development order

1. **Core standalone loop:** one CLI-created research watch, bounded search/fetch/extract/verify/report, SQLite persistence, JSON/Markdown artifacts, repeat sessions with useful memory. Real local LLM and real public web evidence, with test fixtures clearly labeled.
2. **Multi-watch runner:** several watches; durable sequential round-robin queue; standalone unattended execution; budgets and resumability.
3. **Human checkpoints:** watch lifecycle and explicit review requests; continuous, periodic, adaptive policies behind the same contract; tests and metrics.
4. **Demonstration / CSC2524 harness:** minimal local CLI or web UI supporting human goal definition, review, decisions, and a compelling reproducible demo. This must stand on its own without Tari.
5. **Optional Tari integration:** a thin adapter for Discord `#projects` forum threads and OpenClaw; never a requirement for the standalone deliverable.

## Design references

- [`docs/PRODUCT_SPEC.md`](docs/PRODUCT_SPEC.md) — scope, behaviors, lifecycle, first milestones.
- [`docs/ARCHITECTURE_CONTRACTS.md`](docs/ARCHITECTURE_CONTRACTS.md) — clean boundaries and proposed data contracts.
- [`docs/CSC2524_EVALUATION.md`](docs/CSC2524_EVALUATION.md) — optional research question, baselines, metrics, and limitations.
- [`docs/TARI_INTEGRATION_LATER.md`](docs/TARI_INTEGRATION_LATER.md) — separate future OpenClaw/Discord integration.
- [`CLAUDE.md`](CLAUDE.md) — coding-agent guardrails and working method.
- [`prompts/CLAUDE_CODE_LONGRUN.md`](prompts/CLAUDE_CODE_LONGRUN.md) — initial Claude Code assignment.

These are *design specifications*, not claims that implementation already exists. The coding agent may propose implementation details, but should not silently change product constraints or overclaim evaluation results.
