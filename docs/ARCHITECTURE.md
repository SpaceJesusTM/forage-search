# Forage Architecture (as built)

This describes the code in `src/forage/` as of the first implementation session. `ARCHITECTURE_CONTRACTS.md` is the original proposal; where the two differ, this file says so.

## Layers

```text
cli.py                       presentation only (argparse)
app.py                       composition root: builds adapters from settings
controller/watches.py        Boundary B: lifecycle, revisions, approval
controller/reviews.py        Boundary B: review requests and responses
scheduler/runner.py          run manager: admission, recovery, policy evaluation, outbox
scheduler/queue.py           Boundary D: round-robin, eligibility, lease, backoff
scheduler/window.py          night window arithmetic
checkpoint/{base,periodic,adaptive}.py   Boundary C: policies
engine/session.py            Boundary A: one bounded research session
engine/{memory,prompts,schemas,verify,matching,budget,safety}.py
llm/  search/  fetch/        replaceable adapters
report/{schema,build,markdown,compare}.py
storage/{db,store}.py  storage/migrations/*.sql
eval/{export,experiment,scenarios}.py
simulated.py                 SIMULATED web and model for tests and replayable experiments
```

Dependency direction is downward. `engine/` imports storage and the three adapter protocols only. It does not import the CLI, the scheduler or the checkpoint package, and it holds no interface-specific identifiers. The one upward reference is a lazy import of the report builder when a session finalises.

## One session (`engine/session.py`)

| Step | What happens | Model call |
| --- | --- | --- |
| Recall | Load findings, open questions, searches already run, saved directions, unread relevant pages, pages due for re-check. Snapshot the counts. | |
| Re-check | Re-fetch up to `max_rechecks_per_session` pages older than the watch's `freshness_days`. Unchanged hash: claims re-confirmed, no extraction. Changed: claims whose quotes vanished are superseded. 404/410: findings with no other source become `STALE`. | |
| Saved leads | Read relevant pages earlier sessions found but had no budget for (up to a third of the page budget before searching, the rest afterwards). | |
| Plan | Propose queries aimed at open questions and gaps. Code rejects repeats of searches run within `requery_after_hours`, then fills from saved directions, open questions, older productive searches and a last-resort recency cue. | `plan` |
| Search | Paced, budgeted, recorded. URLs canonicalised. Already-read pages are skipped and counted. | |
| Triage | Score snippets 0–3 and flag results outside the approved scope. Selection is by index. The page budget is spread across the remaining queries. | `triage` |
| Fetch | robots.txt, size and time limits, connect-time address guard. Text snapshot stored with content hash. Identical text on another URL is read once. | |
| Extract | Items, each with claims and a verbatim quote, from a passage selection bounded by `page_chars_per_call`. | `extract` |
| Verify | Deterministic: quote located in stored text, numbers and names present on the page. | |
| Audit | Per claim: supported / partial / unsupported, and direct / background / none bearing on the objective. Unsupported and non-direct claims are dropped. | `support` |
| File | Place each item with an existing finding or create one. A claim that restates one on record decides it without the model; otherwise lexical ranking, then the model for the uncertain band. | `match`, `novelty` |
| Reflect | Summary, resolved questions, new gaps, next directions, optional guidance proposal. Resolutions need real finding ids. | `reflect` |
| Report | Built from stored rows only. JSON validated against its own schema, then rendered to Markdown. | |

A heartbeat and cancellation check runs after every fetch and model call. A run always ends with a truthful row: `COMPLETED`, `FAILED`, `CANCELLED`, or `INTERRUPTED` (set on the next start when the heartbeat is stale; a report is still written from what was stored).

## Evidence model

- `source`: one row per canonical URL per watch. `source_snapshot`: one row per distinct content version, kept for audit and change detection.
- `claim`: model-written statement plus the page's own words at the matched location. Status `ACCEPTED`, `REJECTED` (kept with the reason, never reported) or `SUPERSEDED` (the source changed and no longer says it).
- `finding`: a group of accepted claims about one subject. `finding_change` records what each run did to each finding.

Change types reported: `NEW`, `UPDATED`, `CORROBORATED`, `STALE`. `UNCHANGED` is counted but not listed. This refines the taxonomy in the product spec: `RESOLVED` applies to open questions rather than findings, `UNVERIFIED` material is never reported, and `NO_MEANINGFUL_UPDATE` is the report-level flag `has_meaningful_update = false`.

Verification statuses a reader sees: `quote_verified` or `quote_verified_approximate` (at least 86% of quote words matched in sequence), with support `supported`, `partial` or `unchecked`. A claim whose numbers or names are on the page but outside its quote is capped at `partial` and says which.

**What verification does not establish.** It shows that a retrieved page contains the quoted words. It does not show that the claim is true, that the model attached a number to the right subject, or that the page is authoritative. The live runs produced one example of each residual error class; see `SESSION_HANDOFF.md`.

## Untrusted content

Structural controls, in order of importance:

1. The model has no tools. Its output becomes typed data. The only side effects are search query strings and index-based selection among URLs the search provider returned.
2. The engine reads the approved objective from a copy and has no code path that writes a revision or a watch status. Scope ideas reach the user only as a review request, and a response that changes scope is refused until explicitly confirmed.
3. Every claim must be located in stored page text. Text that reads like an instruction is rejected as a claim, as a query, as a question and as a guidance proposal.
4. The fetcher resolves the host at connect time and refuses non-public addresses on every hop, including redirects. That covers loopback, private ranges, link-local, CGNAT (so tailnet hosts), and IPv4-mapped IPv6. Non-HTTP schemes, credentials in URLs and unusual ports are refused earlier. Proxy environment variables are ignored.
5. Hidden elements, scripts, templates and comments are stripped before extraction. Sentences matching instruction patterns are withheld from the model; the stored snapshot keeps them.
6. Web text sits inside `<<<LABEL … LABEL>>>` blocks in user-role messages. The delimiter characters are removed from the content first.
7. Reports pass all web- and model-derived text through a Markdown neutraliser. Only fetched evidence URLs are rendered as links.

Pattern matching (item 5) is a tripwire and will miss novel phrasings. Items 1–4 are what the design relies on. The test `test_model_that_obeys_the_page_is_still_contained` scripts a model that does everything a hostile page asks and checks that nothing escapes.

## Watch lifecycle

Implemented transitions are the spec's minimum plus five that prevent dead ends:

```text
DRAFT -> PENDING_CONFIRMATION -> ACTIVE
ACTIVE -> PAUSED | AWAITING_REVIEW | CLOSED
PAUSED -> REVIEW_PROPOSED | CLOSED
AWAITING_REVIEW -> ACTIVE | CLOSED
CLOSED -> REVIEW_PROPOSED
REVIEW_PROPOSED -> ACTIVE

added:  DRAFT -> CLOSED, PENDING_CONFIRMATION -> CLOSED      abandon before approval
        REVIEW_PROPOSED -> PAUSED | CLOSED                   decline a resume/reopen plan
        AWAITING_REVIEW -> PAUSED                            answer a review with "pause"
```

Only `ACTIVE` watches with an approved revision are researchable. Every objective version is a `watch_revision` row with origin (`user`, `model_draft`, `resume_plan`, `checkpoint_response`), state and who decided it. An amendment to an active watch stays `PROPOSED`, with the old brief in force, until approved.

## Scheduling

- Rotation order is `queue_position`. The cursor is the position served last and is written in the same transaction as the admission.
- Eligibility: researchable, not backing off after failures (15 min doubling to 6 h), and not served within `min_minutes_between_sessions` (queued runs only).
- Window: local wall time in `night_timezone`, start inclusive, end exclusive. Nothing is admitted at or after the end; an admitted session finishes within its own budget.
- One session at a time is enforced twice: by the run manager and by a partial unique index on `session_run(status) WHERE status = 'RUNNING'`.
- One scheduler at a time: a lease row with an expiry. `scheduler_authority = "external"` disables the standalone runner entirely.
- A stop request (first SIGINT/SIGTERM) cancels at the next checkpoint and hands the interrupted turn back.

## Checkpoint policies

A policy is `evaluate(CheckpointContext) -> CheckpointDecision`, called by the run manager after a completed session. The context is loaded from the store. Every decision is logged, including `CONTINUE` and the reason a model proposal was declined.

| Policy | Requests review when |
| --- | --- |
| `continuous` | never |
| `periodic` | `every` sessions have completed since the last answered review (default 3) |
| `adaptive` | sustained scope conflict; a model-proposed user decision that recurs or is corroborated and reads like a decision; three consecutive sessions that searched successfully and verified nothing new |

A hard-stop gate (`max_total_sessions` in the watch budget) runs before the policy for all three and uses the signal name `common_hard_limit`.

The engine asks the model for a guidance proposal under every policy and stores it in the run metrics. Only the adaptive policy acts on it. That keeps the engine identical across conditions; `test_the_engine_behaves_identically_under_every_policy` compares searches, fetches, model prompts, findings and budgets.

## Report contract

`report/schema.py`, `schema_version = "1.0"`. Adapters should depend on this and on nothing else from the engine. Fields beyond the proposal's example: `simulated`, `headline`, `summary` (model inference), per-finding `independent_sources`, `uncertainties`, `retired_claims`, `claims_not_shown`; per-claim `quote`, `support`, `new_this_session`; `questions_answered/opened/open`, `next_directions`, `sources`, `searches`, `comparison`, `pending_review`, `budget`.

## Storage

SQLite, WAL mode, three numbered migrations applied in a transaction each. Tables: `watch`, `watch_revision`, `watch_event`, `session_run`, `search_attempt`, `source`, `source_snapshot`, `finding`, `claim`, `finding_change`, `open_question`, `lead`, `report`, `llm_call`, `run_event`, `checkpoint_decision`, `checkpoint_request`, `checkpoint_response`, `queue_state`, `watch_queue`, `queue_admission`, `scheduler_lease`, `outbox`. Identifiers for sources, claims, findings and questions are content-derived, so a retried write is a no-op.

## Decisions that are easy to reverse

| Decision | Alternative | To switch |
| --- | --- | --- |
| DuckDuckGo HTML scraping as the default search | Keyed API or self-hosted SearXNG | `search_provider` setting; a SearXNG adapter exists (mock-tested, not run live) |
| Lexical similarity plus model adjudication for deduplication | Embedding similarity | Replace `engine/matching.py`; needs an embedding model installed |
| `trafilatura` for main-text extraction | Another extractor | `fetch/extract.py` |
| One process, synchronous | Async or a worker service | The engine takes a run id and returns an outcome; wrap it |
| Data directory relative to the working directory | Fixed per-user location | `FORAGE_DATA_DIR` or `--data-dir` |
| Per-site cap of 3 pages per session | Different cap | `max_per_domain_per_session`; sites named as hints are exempt |

## Decisions that are harder to reverse

- The report schema and the evidence tables. Changes need a migration and a schema version bump.
- Content-derived identifiers. Changing normalisation changes ids.
