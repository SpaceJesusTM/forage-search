# Forage Implementation Plan

Status of each milestone is tracked in the table at the bottom and updated as work lands.
Observed facts (environment, measurements) are dated; anything not measured is labelled as an assumption.

## 1. Environment inspection (2026-10-11, development host)

| Item | Observed |
| --- | --- |
| Host | MacBook Air, Apple M4, 10 cores, 24 GB unified memory, macOS 27.0 (arm64), ~35 GB free disk |
| Python | 3.11.6 (Homebrew); `venv` available; no `uv` |
| Ollama | 0.40.0, server already running on `127.0.0.1:11434` (not started or reconfigured by Forage) |
| Installed models | `qwen3.5:4b-q4_K_M` (4.2B params, 3.3 GB), `qwen3.5:2b-q4_K_M` (1.9B, 1.9 GB). Both report `completion, vision, tools, thinking`; no embedding model installed |
| Inference speed | `qwen3.5:4b-q4_K_M`: ~22–28 output tok/s, schema-constrained JSON works with `think=false`. With `think=true` a 1200-token budget was consumed by reasoning before any answer, so thinking is disabled for structured calls |
| MSI GS66 | Reachable on the tailnet (`msi`, Windows, ping 10–33 ms via DERP relay). TCP 22 and 3389 open; Ollama port 11434 is **not** exposed on the tailnet. Non-interactive SSH as the default user is refused (no key installed), so MSI hardware, GPU and model inventory were **not inspected**. Nothing on the MSI was touched |
| Other local services | Docker daemon up with 0 containers; LM Studio, Discord, VS Code helpers listening on loopback. Forage binds no ports |
| Network quirk | AAAA (IPv6) DNS lookups stall for 5 s on this network; IPv4-only resolution is instant. Forage resolves IPv4 first |
| Search (no API key) | DuckDuckGo HTML endpoint: 6/6 queries OK when paced 20 s apart; 3 queries 1.5 s apart triggered HTTP 202 rate limiting. Brave HTML: 429. Mojeek: 403. Bing HTML: reachable but low-relevance results for non-browser clients. `ddgs` metasearch: works but slow here (15–76 s per query) |
| Fetch/extract | `httpx` + `trafilatura` extracted clean main text, title and date from Wikipedia, arXiv, utoronto.ca, ollama.com and a blog |

Consequences for the design:

- Default model is whatever is installed; on this host `qwen3.5:4b-q4_K_M`. A 4B model is weak at open-ended judgement, so the pipeline gives it narrow, schema-constrained tasks and uses deterministic code to verify what it returns.
- Search must be paced, budgeted and tolerant of rate limits. Keyless scraping is the least reliable component; the provider is an adapter so a keyed API or self-hosted SearXNG can replace it.
- No embedding model is installed, so semantic matching uses IDF-weighted lexical similarity with LLM adjudication for the ambiguous band.

## 2. Stack

| Concern | Choice | Reason |
| --- | --- | --- |
| Language | Python ≥ 3.11 | Available on host; `tomllib`, `zoneinfo` in stdlib |
| Storage | SQLite (stdlib `sqlite3`), WAL, numbered migrations | Spec recommendation; zero services; portable to the MSI |
| LLM | Ollama HTTP API via `httpx`, JSON-schema `format`, `think=false` | No SDK dependency; adapter is ~200 lines and replaceable |
| Schemas | `pydantic` v2 | Validates LLM output and the report contract; emits JSON Schema |
| Fetch | `httpx` with a connect-time address guard | Blocks private/loopback targets on every hop including redirects |
| Extraction | `trafilatura` (HTML, with `lxml_html_clean`), `pypdf` (PDF) | Clean main-text extraction; bounded PDF support for papers |
| Search | Native DuckDuckGo-HTML provider; optional `ddgs` and SearXNG adapters; fixture provider for tests | No paid API in P0 |
| CLI | stdlib `argparse` | No framework dependency |
| Tests | `pytest` | Deterministic fixtures by default; live tests behind a marker |

Deliberately not used: agent frameworks, vector databases, task queues, web frameworks, Docker.

## 3. Architecture

```text
forage/cli.py                     presentation (argparse); no business logic
forage/controller/                watch lifecycle, approvals, review responses   (Boundary B)
forage/scheduler/                 window, round-robin cursor, lease, run manager (Boundary D)
forage/engine/                    one bounded research session                   (Boundary A)
forage/checkpoint/                continuous | periodic | adaptive policies      (Boundary C)
forage/llm/  search/  fetch/      replaceable provider adapters (+ simulated fixtures)
forage/report/                    versioned JSON report schema, Markdown renderer, comparison
forage/storage/                   SQLite schema, migrations, repositories
```

The engine depends only on storage and the three adapter protocols. It never imports the CLI, scheduler or checkpoint packages, and holds no interface-specific identifiers.

### Session pipeline

1. **Recall**: build a bounded text view of prior state (findings, open questions, queries already tried with their yield, unfetched leads, sources due for recheck).
2. **Plan** (LLM): propose queries aimed at open questions and gaps. Deterministic code rejects exact and near-duplicate queries from earlier sessions.
3. **Search**: paced, budgeted, recorded. URLs are canonicalised; already-fetched URLs are skipped and counted.
4. **Triage** (LLM): score result snippets against the approved objective; choose what to fetch by index, never by URL text.
5. **Fetch**: robots.txt, size/time limits, address guard. Text snapshot stored with content hash. An unchanged hash skips re-extraction.
6. **Extract** (LLM): claims, each with a verbatim supporting quote.
7. **Verify** (deterministic, then LLM): the quote must be located in the stored page text; numbers in the claim must appear in the page; a batched support check rejects claims the quote does not back. Rejected claims are kept for audit but never reported.
8. **Integrate**: match to existing findings → `NEW`, `UPDATED`, `CORROBORATED`, `UNCHANGED`, `STALE`.
9. **Reflect** (LLM): resolve open questions against verified findings, record new gaps and next directions, optionally propose a guidance question.
10. **Report**: validated JSON (authority) plus Markdown; metrics and budget use.

### Evidence rules enforced in code

- A claim is reportable only if its source was fetched in this watch and its quote was found in the stored snapshot.
- The quote shown in reports is the page's own text at the matched location, not the model's transcription.
- Model-written text (summaries, open questions, next directions) is labelled as inference and kept separate from source-stated claims.
- "Quote verified" means the source says it, not that it is true. Independent corroboration is counted by distinct registrable domain.

### Untrusted content handling

- Page text and snippets appear only inside delimited data blocks in user-role messages, never in system prompts.
- The model has no tools. Its output is parsed into typed data; the only effects are search query strings and index-based selection among URLs the search provider returned.
- The planner and reflector never see raw page text, only short verified claims.
- The engine has read-only access to the approved objective. Scope-change ideas become checkpoint proposals that a human must confirm.
- Hidden elements, scripts and comments are stripped before extraction; instruction-like passages are flagged on the source record.
- Untrusted strings are neutralised before Markdown rendering.

## 4. Sequence and definition of done

### P0 — standalone single-watch research

1. Scaffold package, config, storage and migrations.
2. Ollama adapter with model discovery, schema-validated output and repair retry; simulated LLM for tests.
3. URL canonicalisation, guarded fetcher, extraction, search providers.
4. Watch controller with lifecycle and approval.
5. Engine pipeline, budgets, cancellation, crash-safe run records.
6. Report schema, Markdown renderer, session comparison.
7. CLI.
8. Automated tests; live two-session demonstration.

**Definition of done:** live local-model inference; live search and retrieval; a report whose every citation resolves to a fetched page containing the quoted text; state surviving a process restart; a second session that reads prior state, issues no repeated query, suppresses duplicates and reports honestly whether anything new was found; automated tests for watch state and approvals, URL handling, report schema, deduplication and change detection, persistence, budgets and injected-instruction pages.

### P1 — multi-watch runner

Durable round-robin cursor, eligibility, 02:00–06:00 America/Toronto admission window with DST tests, failure backoff, single-scheduler lease, signal-safe shutdown. No OS scheduler is installed.

### P2 — checkpoint policies

`CheckpointPolicy` interface; `continuous`, `periodic`, `adaptive`; `AWAITING_REVIEW` with request/response records; scope changes require explicit confirmation; scenario where one watch waits for review while another keeps running.

### P3 — evaluation support

Per-run structured logs of searches, LLM usage, findings, checkpoint decisions and queue admissions, exportable for analysis. Replayable simulated corpus for policy comparison.

## 5. Risks

| Risk | Mitigation |
| --- | --- |
| Small model fabricates or misattributes | Quote location check, number grounding, support check; reject rather than repair |
| Keyless search rate limiting | Pacing, cooldown, provider fallback, empty results recorded as such |
| Small model over-merges or over-splits findings | Conservative thresholds; adjudication only in the ambiguous band; change log kept per run for audit |
| Long pages exceed context | Lexical passage selection with a fixed character budget per page |
| Process killed mid-run | Incremental commits; stale `RUNNING` runs marked `INTERRUPTED` on next start |
| MSI differs from this host | No host-specific paths; model chosen at runtime; data directory configurable |

## 6. Milestone status (end of first implementation session, 2026-10-11)

| Milestone | Status | Verified by |
| --- | --- | --- |
| P0 single-watch research engine | Done | 219 deterministic tests; two live watches run for repeated sessions on the real web with the local model |
| P1 multi-watch runner | Done | Deterministic tests for rotation, cursor persistence, window and DST, backoff, lease, shutdown; one bounded live three-session queue run. A real overnight run inside the window has **not** been done |
| P2 checkpoint policies | Done | Deterministic tests for all three policies, gates, review responses and the one-waits-while-another-runs scenario; `periodic` exercised live once. `adaptive` has **not** fired on live data |
| P3 evaluation support | Partly done | Export and a simulated three-policy harness with one built-in scenario. No repeats or seeds, no live pilot, no relevance rubric, no UI |

Details, measured results and known defects are in `SESSION_HANDOFF.md`.

## 7. Changes made to this plan during implementation

Each of these came from running the system, not from the original design.

| Finding | Change |
| --- | --- |
| A page of generic prompt-engineering advice passed triage and filled a report with tangential claims | The claim audit now also judges bearing on the objective (direct / background / none) and drops everything that is not direct; triage prefers primary and data-bearing sources |
| The model quoted two cells of a table row; the cells were not adjacent in the page text, so a true claim was rejected | Table-cell quotes are matched against a single row and stored with the table header |
| A claim named a model the quote did not mention | Names and numbers are checked against the page; those found only outside the quote cap the claim at "partly supported" with a note |
| Related findings from different sources were never merged because wording differed | Filing is one model call per page over lexically shortlisted candidates, plus a deterministic shortcut when a claim restates one on record |
| One query used the whole page budget | The page budget is spread across the remaining queries; unread relevant results become leads for the next session |
| Reflection output was truncated before reaching "next directions" | Schema fields are ordered so essentials are generated first, and list lengths are capped in the grammar |
| Official-source research was throttled by a two-pages-per-site cap | Cap raised to three; sites the user named are exempt |
| By the third session, narrow model-written questions made the extractor judge relevant pages irrelevant | The user's standing questions always lead the extraction focus |
| In a simulated five-night run, watches ran out of distinct queries and went idle | Searches may be run again after `requery_after_hours`; within the interval repeats are still rejected |
| Budget "limit reached" flags depended on whether a gate happened to be asked | Flags are computed from counters, measured before the wrap-up phase |
| Timestamp ties were broken by hash-derived ids, so identical inputs could order differently | Ties break on insertion order; a test compares full traces across policies |
| A clean-environment install failed every session: `trafilatura` needs `lxml_html_clean`, which the development environment had only by accident | Declared as a dependency; the suite and the offline demo were re-run in a fresh environment |
| Output speed fell from about 28 to about 9 tokens per second during sustained runs on the fanless development laptop | Per-call timing from the model server is stored; budgets are time-bounded regardless |

## 8. Next safe steps

See section 9 and 10 of `SESSION_HANDOFF.md`.
