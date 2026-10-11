# Forage Product Specification (Working Draft)

## 1. Objective and design constraints

**Forage** is a local-first research worker running primarily on the MSI GS66. Its purpose is to autonomously revisit open-ended, user-owned research objectives and produce cumulative, evidence-backed discoveries from public web information. It must remain runnable on its own, with a CLI and a small persistent state store. It may later be invoked by Tari/OpenClaw or by a separate CSC2524 web UI.

Do **not** conflate Forage's core with Tari Scout's Discord forum workflow: Forage is the reusable worker and local watch controller; Tari becomes one optional front-end/controller.

**Non-goals for MVP:** Discord/OpenClaw wiring, personal Gmail, job applications, event registrations, arbitrary shell/web automation, large cloud LLM dependence, multi-agent swarm frameworks, social recommendations, full desktop/web UI, and proving HCI outcomes before evaluation.

## 2. A research watch

Each watch is a durable user-defined information need with:

- Stable ID, title and original objective.
- Approved inclusion/exclusion constraints, priorities, topical/geographical boundaries, freshness needs and optional source hints.
- Lifecycle status and dates; user amendments plus approval/provenance history.
- Durable research state: discovered sources, observations, extracted claims, observed timestamps, reported findings, open questions, search attempts, promising gaps, recheck plans, failures and cost accounting.
- Session/report history and pending human guidance requests.

Initial examples are (1) MScAC AI/ML internship opportunities and relevant companies/teams and (2) Toronto/U of T technology/career events. Add a custom arbitrary research topic to prove the engine is generalized. Domain presets supply hints; **do not build separate hard-coded agents**.

Never imply an internship satisfies official MScAC requirements without authoritative verification. Never invent event dates or deadlines.

## 3. One bounded research session

1. Load the watch and summarize prior research state for the local model within bounded context.
2. Identify high-value search directions, stale sources and open questions rather than repeating identical generic queries.
3. Search public sources within explicit time, request and token budgets.
4. Retrieve original pages where possible; extract claims, provenance and verification timestamps.
5. Compare with saved evidence: `NEW`, `UPDATED`, `RESOLVED`, `UNCHANGED`, `STALE`, `UNVERIFIED`, or `NO_MEANINGFUL_UPDATE` (final taxonomy may be refined with tests).
6. Rank relevance to currently approved constraints, noting uncertainty; produce **one structured report artifact** for the session. The report may contain several findings, but must not manufacture novelty.
7. Persist history, next investigative leads, metrics and run status even when no report should be displayed.
8. Evaluate whether progress needs human guidance; create a **review request** only when it concerns a decision outside established authorized bounds or clearly merits user choice. Do not ask for routine factual uncertainty that the agent can continue investigating independently.

First milestone: prove one watch can run twice, with session two informed by session one's saved state. Avoid claiming successful incremental research from synthetic fixtures alone.

## 4. Watch lifecycle and checkpoints

Minimum lifecycle:

```text
DRAFT -> PENDING_CONFIRMATION -> ACTIVE
ACTIVE -> PAUSED                 (explicit user pause; immediate)
ACTIVE -> AWAITING_REVIEW        (agent-initiated guidance request)
ACTIVE / PAUSED / AWAITING_REVIEW -> CLOSED
PAUSED -> REVIEW_PROPOSED -> ACTIVE             (user confirms resume plan)
AWAITING_REVIEW -> ACTIVE                       (user responds/approves plan)
CLOSED -> REVIEW_PROPOSED -> ACTIVE             (user reopens/confirms)
```

`CLOSED` means archived/retired, not deleted. Data and history are preserved. A watch in `AWAITING_REVIEW` is *not* auto-researchable and remains answerable by the user. Some system-level warnings may be delivered while awaiting review, but do not keep burning research sessions on it. State transitions and run/publish permissions must be transactional enough to survive restarts.

**Adaptive checkpoint candidate signals:** unclear conflict between approved constraints and newly discovered opportunities; sustained lack of marginal research utility; major source/market changes that invalidate an approved tactic; uncertainty about a necessary **user preference**; choice among materially different strategic directions; and deliberate change requested by the user. High model uncertainty alone is insufficient to interrupt if additional authorized research can resolve it.

A checkpoint request should include: what happened, evidence, why the decision cannot or should not be made automatically, 2–3 choices when appropriate, a safe default if the user is unavailable, and a structured way to respond. A wrong/premature checkpoint is a real failure, not a success.

**Safe default:** Continue within existing approved scope when safe and useful. When a consequential deviation requires approval and no safe productive work remains, suspend only that watch and let other eligible watches proceed.

## 5. Multi-watch scheduling

- Run one session at a time initially, using a shared Ollama/local compute resource.
- Store queue membership, eligibility and cursor durably. Round-robin ordering must persist across days and restarts.
- Initial standalone overnight target: **02:00–06:00 America/Toronto**. Admit no new run at/after 06:00; an already admitted bounded run may finish. Do not preload unlimited jobs.
- `PAUSED`, `CLOSED`, `AWAITING_REVIEW`, and unconfirmed watches never enter the research queue.
- When a watch requests review, release its future turns and allow other watches to proceed. A pending review must not monopolize the queue.
- Use bounded per-watch work, sensible cooldowns, cancellation and failure backoff to prevent endless searching, starvation or log spam.
- No paid-search assumptions; keep provider and budgets configurable. Use deterministic tests for time zone, DST, cursor persistence and cutoffs.
- Later, choose *one active scheduler authority* per deployment. Tari's bridge may eventually replace the standalone scheduler, not run alongside it.

## 6. User-visible minimum

For the CSC2524-viable standalone interface (CLI initially; local web UI optional), a user can:

- Create a watch from natural language, inspect and approve the structured objective.
- View its state, priorities, latest findings, sources, timestamps, uncertainty and pending decisions.
- Run a bounded session and see research progress/artifacts.
- Pause, close, and request resume/reopen with a refreshed plan and confirmation.
- Respond to a guidance checkpoint and reactivate the watch.
- Compare before/after research sessions and inspect where compute/search budget went.

## 7. Acceptance criteria

**Research:** public evidence actually fetched; no invented citations; second run consumes saved research context; duplicates suppressed; novel or changed findings highlighted; structured report validates; unknowns visible.

**Control:** no unconfirmed research; pause and review remove watches from future automatic admission; explicit confirmed user response is required to resume; other watches continue; no autonomous scope expansion.

**Reliability:** run IDs and state persistence; progress survives process failure; deterministic queue order; bounded time/search/token use; no surprise push/deployment changes.

**Reusability:** core, policies and storage run with neither Tari nor a UI framework installed. Interface-specific identifiers live in adapters, not core watch IDs.

## 8. Scope priority

- **P0** usable single-watch local research, SQLite/history, CLI/report artifact, fixtures + live smoke test.
- **P1** generic multi-watch control, fair overnight scheduling, robust unattended execution.
- **P2** checkpoint policy interface and adaptive review demo with 3 comparative policies.
- **P3** reproducible CSC2524 evaluation and an optional polished local UI.
- **Later** Tari/OpenClaw Discord adapter, optional stronger-model conversational planning, richer domain connectors.

P0 should be solid before sophisticated orchestration. The CSC2524 direction is still tentative and subject to instructor feedback.
