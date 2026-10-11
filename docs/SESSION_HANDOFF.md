# Session Handoff — first Forage implementation

Date: 2026-10-11 · Branch: `initial-implementation` · Nothing has been pushed.

**State:** Forage runs end to end as a standalone local research agent. P0, P1 and P2 are implemented and tested; P3 has export and a simulated policy-comparison harness. Two real watches were researched over repeated live sessions with a local 4B model and real web search, and later sessions demonstrably built on earlier ones.

**Not done:** no real overnight run inside the 02:00–06:00 window, nothing run on the MSI, the adaptive policy has not fired on live data, and no evaluation of the CSC2524 hypothesis has been carried out.

**Decisions waiting for you** are in section 9.

---

## 1. What was implemented

| Area | Delivered |
| --- | --- |
| Research engine (P0) | One bounded session: recall → plan → search → triage → fetch → extract → verify → audit → file → reflect → report. Budgets for time, searches, pages and model output tokens. Cancellation and crash recovery. |
| Evidence handling | A claim is reported only if its quote is located in the stored page text and its numbers and names appear on the page. Rejected claims are kept with the reason and never reported. Source-stated claims and model-written synthesis are separate in every report. |
| Memory | SQLite store of watches, revisions, runs, searches, sources, page snapshots, claims, findings, per-run changes, open questions, next directions, reports, model usage and events. |
| Longitudinal behaviour | Later sessions follow up unread relevant pages, skip pages already read, reject searches repeated within 20 hours, re-run productive searches after that, detect changed and vanished sources, and report `NEW` / `UPDATED` / `CORROBORATED` / `STALE`. |
| Watch control | Create from natural language with a model-drafted brief, explicit approval before any research, pause, resume and reopen through a confirmed plan, close without deletion, amendments that take effect only when approved, full history with actor. |
| Multi-watch runner (P1) | Persisted round-robin cursor, eligibility, night window with DST handling, failure backoff, minimum gap, single-scheduler lease, external-authority switch, clean shutdown. |
| Checkpoints (P2) | `continuous`, `periodic`, `adaptive` behind one interface; review requests and responses; scope changes need explicit confirmation; a shared hard-stop gate. |
| Evaluation support (P3) | `forage export` and `forage experiment` (simulated three-policy comparison). |
| CLI | `doctor`, `watch` (create, list, show, edit, approve, decline, pause, resume, reopen, close, policy), `run`, `report`, `runs`, `findings`, `sources`, `questions`, `compare`, `budget`, `log`, `review` (list, show, respond), `queue` (status, run, night), `export`, `experiment`. |

Not implemented: Tari/Discord integration (deferred by design), a web UI, embedding-based deduplication, a keyed search API adapter, link-following from fetched pages, JavaScript rendering.

## 2. Architecture and design decisions

The as-built architecture is in [`ARCHITECTURE.md`](ARCHITECTURE.md). The decisions that most affect future work:

1. **The model is given narrow, schema-constrained tasks and its output is checked by code.** With a 4B model this was necessary. Seven call types (`plan`, `triage`, `extract`, `support`, `match`, `novelty`, `reflect`) each return a validated pydantic object.
2. **Quotes are verified against stored page text, and the stored evidence is the page's wording.** This is the main defence against fabricated citations. It establishes "the source says this", not "this is true".
3. **The engine cannot change a watch.** It reads the approved revision and returns a candidate guidance proposal. Only the controller writes revisions and statuses: on a user action, or, for the move to `AWAITING_REVIEW`, on a checkpoint policy's decision.
4. **Checkpoint policies run after the session and read stored state.** The engine asks for a guidance proposal under every policy; only `adaptive` acts on it. A test compares complete traces across the three policies and requires them to be identical.
5. **Searches are remembered with their time.** A search is a repeat for 20 hours and legitimately re-runnable afterwards. The first version blocked repeats forever, which made a simulated five-night run go idle.
6. **Deterministic code owns** scheduling, state transitions, URL canonicalisation, exact deduplication, budgets, quote location and report assembly. Reports are rebuilt from rows, with no model call.
7. **No embedding model was installed**, so matching is IDF-weighted lexical ranking with one model call per page for the uncertain band. This is the weakest part of deduplication.
8. **Search is keyless DuckDuckGo HTML scraping**, chosen because nothing else worked without a key or a new service on this host. It is behind an adapter.

Reversible and harder-to-reverse decisions are tabulated at the end of `ARCHITECTURE.md`.

## 3. Models, dependencies and services

| Item | Used |
| --- | --- |
| Development host | MacBook Air M4, 24 GB, macOS 27.0, Python 3.11.6 |
| Model server | Ollama 0.40.0, already running on `127.0.0.1:11434`. Not started, stopped or reconfigured. No model pulled. |
| Model | `qwen3.5:4b-q4_K_M` (4.2B, 3.3 GB), auto-selected as the largest installed. `qwen3.5:2b-q4_K_M` was installed and not used. Reasoning mode off, `num_ctx` 8192, temperature 0.1. |
| Search | DuckDuckGo HTML endpoint, no key, 12 s minimum spacing |
| Python packages | httpx 0.28.1, pydantic 2.14.0, trafilatura 2.3.1, lxml 6.1.3, lxml_html_clean 0.4.5, pypdf 6.20.0; dev: pytest 9.1.1; optional: ddgs 9.16.0 |
| Services started by Forage | None. No ports bound, no containers, no OS scheduler entries. |
| MSI GS66 | Reachable on the tailnet (ping). I made one non-interactive SSH attempt as the default user, which was refused, and one TCP check of the Ollama port, which was closed. No other access was attempted. **Nothing on the MSI was inspected or changed.** |

Measured model throughput: 21–31 output tokens per second in the later runs (from the server's own timing). In the first two sessions, run back to back for about nine minutes each, individual calls fell to roughly 8–13 tokens per second of wall time. The laptop is fanless, so thermal throttling is the likely cause; I did not measure temperature.

## 4. Tests performed

### 4.1 Automated, deterministic (simulated web and model)

```bash
.venv/bin/python -m pytest          # 219 passed
```

| File | Tests | Covers |
| --- | ---: | --- |
| `test_urls.py` | 50 | canonicalisation, registrable domain, refused URLs, public/non-public addresses |
| `test_verify.py` | 22 | quote location (exact, elided, approximate, table rows), number and name grounding, instruction-like claims |
| `test_safety_matching_budget.py` | 16 | injection patterns, redaction, Markdown neutralising, repeat queries, similarity, passage selection, budget gates |
| `test_adapters.py` | 25 | JSON salvage, Ollama adapter (mocked), hidden-content stripping, fetcher limits and robots, address guard, DuckDuckGo parsing and throttling, fallback, SearXNG (mocked) |
| `test_storage_and_lifecycle.py` | 18 | migrations, transactions, single running session, run-id reuse, stale-run recovery, lifecycle and approvals, restart persistence |
| `test_session_e2e.py` | 32 | cited findings, second session reuse, re-query interval, corroboration, changed and vanished sources, budgets, cancellation, fabricated quotes and numbers, injected instructions, provider and model failures, comparison |
| `test_scheduler.py` | 27 | window boundaries, both DST nights, rotation, cursor across restart, skipping, cutoff, backoff, minimum gap, lease, external authority, stop, outbox |
| `test_checkpoints.py` | 18 | each policy, gates, review responses, confirmation of scope change, identical engine across policies, one watch waiting while another runs |
| `test_eval.py` | 5 | export, reproducible policy comparison, visible downside of pausing |
| `test_cli.py` | 6 | full CLI cycle, control commands, queue, review flow |

These prove pipeline behaviour on fixtures. They say nothing about research quality on the real web.

### 4.2 Live checks

```bash
.venv/bin/python -m pytest -m live   # 4 passed in 12 s (Ollama inference, DuckDuckGo search, page fetch, address guard)
.venv/bin/forage doctor --live       # inference ok, 3 search results, example.com fetched
```

### 4.3 Clean install

A fresh virtual environment with `pip install ".[dev]"` (non-editable, without the optional `ddgs` extra) passes all 219 tests, and the offline demonstration in section 5 runs end to end from a different working directory.

This check found a real defect: `trafilatura` needs `lxml_html_clean`, which my development environment only had because `ddgs` happened to pull it in. Without it every session failed with an import error. It is now a declared dependency.

### 4.4 Live research runs

All with `qwen3.5:4b-q4_K_M` and DuckDuckGo. I created, approved and answered reviews for these two test watches through the CLI; the history table records those actions as actor `user`. Reports are in `data/reports/` (not committed).

**Watch A — "What techniques make small local language models (under 8B parameters) produce reliable structured JSON output, and what measured evidence supports each technique?"**

| Session | Trigger | Wall s | Searches | Pages | Out tokens | Claims kept / rejected | New / updated | Saved leads read | Known URLs skipped |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| 1 | manual | 520.7 | 4 | 6 | 5670 | 50 / 3 | 18 / 0 | 0 | 0 |
| 2 | manual | 545.8 | 4 | 6 | 4786 | 25 / 2 | 8 / 1 | 2 | 3 |
| 3 | queue | 123.1 | 2 | 3 | 2143 | 0 / 4 | 0 / 0 | 1 | 1 |
| 4 | queue | 99.8 | 2 | 3 | 1091 | 0 / 0 | 0 / 0 | 1 | 2 |
| 5 | manual | 421.5 | 3 | 5 | 4461 | 14 / 14 | 2 / 3 | 2 | 2 |

**Watch B — "Find AI/ML internship opportunities and employers relevant to University of Toronto MScAC students for the 2027 internship cycle, including how the MScAC internship process works and its timeline."**

| Session | Trigger | Wall s | Searches | Pages | Out tokens | Claims kept / rejected | New / updated | Saved leads read | Known URLs skipped |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| 1 | manual | 179.9 | 4 | 4 | 3285 | 14 / 0 | 2 / 0 | 1 | 5 |
| 2 | manual | 204.6 | 4 | 6 | 3543 | 9 / 5 | 0 / 1 | 2 | 8 |
| 3 | queue | 114.0 | 2 | 3 | 1334 | 4 / 0 | 0 / 1 | 1 | 6 |
| 4 | manual | 222.3 | 3 | 5 | 2999 | 11 / 0 | 1 / 1 | 1 | 14 |

The code changed between sessions as defects were found, so the rows are not a controlled series. Sessions 1–2 of watch A ran before the relevance audit, name grounding and table-row matching existed. Watch B session 4 ran on the final code. Watch A session 5 ran just before the last audit change (section 7, item 4).

Totals across the nine live sessions: 124 model calls, 28 searches (all returned results), 41 page fetches (40 succeeded, one HTTP 403), 127 claims accepted and 28 discarded, 31 findings.

**Evidence that later sessions used earlier state**

- Across all nine live sessions of both watches, no search query was run twice (checked in the database). All sessions fell within one afternoon, so the 20-hour re-query rule was never exercised live.
- Watch A session 2 recalled 18 findings, 4 open questions and 4 earlier searches. Its first two page reads were leads saved by session 1 (a benchmark report and a fine-tuning repository), and its four searches targeted open questions, two of which session 1's review had written. It reported 8 new findings and added three claims from a second site to an existing finding.
- Watch B session 2 opened the 2026–27 MScAC handbook PDF, which session 1 had found in search results but not read (the per-site cap was two pages at the time). That added the claims "The MScAC Internship Expo takes place during the week beginning January 18, 2027" and "The offer system opens in early February", each with its quote from the handbook. It skipped 8 known URLs and suppressed 3 duplicate claims.
- Watch B session 4, on the final code, read the MScAC industry brochure PDF, kept 11 claims with none discarded, skipped 14 known URLs, and reported one new finding (funding and compensation) and seven new claims on an existing one.
- Watch A sessions 3 and 4 reported "No meaningful update" rather than repeating earlier findings.
- `forage compare <watch>` prints the before/after for any two sessions.

**Live queue and review**

`forage queue run --ignore-window --max-sessions 3` with watch B set to `periodic` (`every: 1`) admitted A, B, A. After its session B moved to `AWAITING_REVIEW`, a review request and outbox rows were written, and the third slot went to A. `forage review respond … --choice continue` returned B to `ACTIVE`. The run was started at about 16:48 local time with the window check overridden; admission inside the real window was tested only with a simulated clock.

### 4.5 What I checked by hand, and what I did not

I read the session reports and compared claims with their quotes for both watches. I did not open the cited URLs in a browser to confirm the stored text matches the live page, and I did not check every one of the 127 accepted claims. Treat the reports as the system's output, not as audited research.

### 4.6 Simulated policy comparison

`forage experiment --out <dir>` on the built-in `three-watches` scenario (5 nights × 6 slots, scripted user answering one night later):

| | continuous | periodic | adaptive |
| --- | ---: | ---: | ---: |
| Sessions run | 30 | 27 | 30 |
| Search requests | 42 | 36 | 38 |
| Relevant pages verified (of 23) | 17 | 22 | 22 |
| Relevant pages per 100 searches | 40.5 | 61.1 | 57.9 |
| Reviews requested | 0 | 9 | 3 |
| Reviews answered "continue" | 0 | 5 | 0 |
| Pages verified outside approved scope | 0 | 0 | 0 |

This scenario was designed to contain one watch that needs a scope decision, one that needs none and one that dries up. It shows the mechanism working and is reproducible. It is **not evidence for the CSC2524 hypothesis**: one deterministic run, a rule-based model, a scripted user, and a scenario built around the cases the adaptive policy targets. Note that periodic was slightly more efficient per search here, and that both asking policies missed a late page on the watch the user paused, which continuous found.

## 5. Install and run

```bash
cd forage-search
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/forage doctor --live

.venv/bin/forage watch create "<what to research>"
.venv/bin/forage watch approve <watch>
.venv/bin/forage run <watch> --max-seconds 600 --max-searches 4 --max-pages 6
.venv/bin/forage report <watch>
.venv/bin/forage run <watch>
.venv/bin/forage compare <watch>

.venv/bin/python -m pytest
.venv/bin/python -m pytest -m live
```

Fully offline demonstration with the simulated web and model (verified in a clean environment):

```bash
export FORAGE_DATA_DIR=/tmp/forage-demo FORAGE_MODEL=simulated FORAGE_SEARCH_PROVIDER=fixture \
       FORAGE_FIXTURE_WEB=examples/simulated_web.json FORAGE_SEARCH_MIN_INTERVAL_SECONDS=0
.venv/bin/forage watch create "How much tidal turbine energy capacity is installed and what does tidal energy cost?" \
    --no-llm --question "What tidal turbine capacity is installed?" --question "What does tidal energy cost?"
.venv/bin/forage watch approve how-much && .venv/bin/forage run how-much && .venv/bin/forage report how-much
.venv/bin/forage experiment --out /tmp/forage-experiment
```

The full command list and configuration table are in the README. The two live test watches are still in `data/`; `forage watch list` shows them, and you can close or keep them.

## 6. Milestone status

| Milestone | Status |
| --- | --- |
| P0 standalone research engine | **Complete.** Every item in the required validation list was demonstrated live or by test (table below). |
| P1 multi-watch autonomous research | **Complete in code and deterministic tests; one bounded live queue run.** Not yet run unattended through a real night window. No OS scheduler installed, as instructed. |
| P2 checkpoint policies | **Complete in code and deterministic tests.** `periodic` exercised live once. `adaptive` not yet triggered by live data. |
| P3 evaluation and interface | **Partly done.** Logs, export and a simulated harness exist. No repeats, live pilot, rubric or UI. |

Required P0 validation:

| Requirement | Result |
| --- | --- |
| Successful local-model inference | Yes: 124 live calls across 9 sessions; live test passes |
| Real web search and source retrieval | Yes: 28 live searches, 40 successful page fetches |
| Complete report with working source references | Yes: JSON and Markdown per session; every claim carries the fetched URL and the page's quote |
| State surviving process restart | Yes: every live session was a separate process; `test_state_survives_process_restart` |
| Two consecutive sessions, second using prior state | Yes, on two watches (section 4.4) |
| Duplicate handling and reporting of uncertainty | Yes: known URLs skipped, duplicate claims suppressed, identical content read once; per-finding and per-report uncertainty sections |
| Automated tests for key components | Yes: 219 |

## 7. Known bugs, limitations and security considerations

### Research quality

1. **Verified does not mean correct.** The quote check confirms the words are on the page. The model can still pair a number with the wrong subject. Claims whose names or numbers fall outside the quote are capped at "partly supported" and say so, but a wrong pairing inside one sentence is not caught.
2. **Model-written synthesis can be wrong.** Watch B session 2's summary said the Internship Expo "occurs in early January 2027"; the verified claim directly above it says "week beginning January 18, 2027". The summary sits in a section labelled as inference. A deterministic check of numbers and dates in summaries against the cited findings would catch this class and is not implemented.
3. **Scope adherence depends on explicit exclusions.** Watch A is about small local models, yet session 2 accepted claims about OpenAI, Anthropic and Gemini hosted APIs from a leaderboard page. The 4B model judged them on topic. Adding `--exclude "hosted cloud APIs"` is the current remedy.
4. **The relevance audit is noisy.** In watch A session 5 it discarded all ten claims from a directly relevant page (reported schema-validity rates for 3.8B–9B models). On replay, removing one sentence from the audit prompt ("Be strict …") gave 8 of 10 judged direct, while a generic prompt-engineering page was still rejected. I removed the sentence and changed the rule so background claims survive on pages that also have direct ones. That fix was tested on five stored pages only. `forage findings <watch> --rejected` lists everything discarded and why.
5. **Grouping into findings is rough.** In watch B, 27 claims sit in two findings. In watch A session 1, related findings from different sites stayed separate. There is no embedding model to do this well.
6. **Early findings are lower quality.** Watch A's first 18 findings include generic prompt-engineering advice accepted before the relevance audit existed. They remain in that watch's history.
7. **Passage selection is lexical.** On long pages the model sees about 7000 characters chosen by word overlap, so relevant passages with different vocabulary can be missed.
8. **A page read once is not re-read** unless `freshness_days` elapses and its content changed, so claims wrongly discarded from it stay discarded.

### Search and retrieval

9. **Keyless search is the least reliable component.** Three queries 1.5 s apart triggered throttling in probing; with 12–20 s spacing, all live queries succeeded. Markup changes would break parsing. For unattended nights, plan on a search service you control.
10. **Some sites refuse the bot user agent** (one HTTP 403 from medium.com). Forage identifies itself honestly and obeys robots.txt.
11. No JavaScript rendering, no link-following from fetched pages, PDFs limited to the first 20 pages and 4 MB.

### Engineering

12. **Defaults were tuned on one laptop.** Time budgets assume roughly 20–30 tokens per second. Throughput on the MSI is unknown.
13. `data_dir` defaults to `./data` relative to the working directory. Running `forage` from another directory creates a separate database. Set `FORAGE_DATA_DIR` for a fixed location.
14. Stale-run recovery uses a 10-minute heartbeat rule. A session killed with `SIGKILL` blocks new sessions until then; `Ctrl-C` on `forage run` recovers immediately.
15. The guarded transport replaces a private attribute of httpx's connection pool. It fails closed if the attribute disappears, and a test covers it, but an httpx upgrade could require a small change.
16. The SearXNG and `ddgs` adapters were tested with mocks only. `ddgs` did return results in probing but took 15–76 s per query on this network.
17. Windows was not tested. The code avoids POSIX-only calls except `SIGTERM` handling in `queue` commands.

### Security

- Retrieved content is treated as data throughout; the controls and their order of importance are in `ARCHITECTURE.md`. Instruction-pattern matching is a tripwire and will miss novel phrasing. The structural controls are what matter: no tools, index-based URL selection, read-only objective, quote verification, connect-time address guard.
- The address guard refuses loopback, private, link-local and CGNAT addresses on every connection, which includes tailnet hosts such as the MSI. DNS rebinding between check and connect is not possible because the check happens at connect.
- No credentials are read, stored or sent. Proxy environment variables are ignored by the fetcher. `.gitignore` excludes `data/`, databases, `forage.toml`, `.env*` and model files.
- Search queries are written by the model and sent to a third-party search engine. Objectives containing private information would be exposed that way.
- A model-proposed review question is shown to the user after an instruction-pattern scan. Its text still originates from a model that read web content, so read review options as suggestions.

## 8. Files changed and Git state

Branch `initial-implementation`, created from `main` at `b589407`. Local commits, none pushed:

```text
0893b9f Add Forage design specifications
b12dbb9 Implement P0: standalone single-watch research engine
590ebe1 Implement P1: multi-watch round-robin scheduler and night window
05818d4 Implement P2: interchangeable checkpoint policies and review handling
17e71c2 Add P3 evaluation support and fix long-run research behaviour
eebe9e4 Soften the relevance audit and expose discarded claims
(plus the commit that adds this document and the other docs)
```

- Commit 1 contains your design documents exactly as they were in the working tree (they were untracked, and `README.md` was modified).
- Later commits add `src/forage/` (about 8,500 lines), `tests/` (about 2,700 lines), `examples/simulated_web.json`, `pyproject.toml`, `.gitignore`, `forage.example.toml`, and `docs/IMPLEMENTATION_PLAN.md`, `docs/ARCHITECTURE.md`, `docs/SESSION_HANDOFF.md`.
- Edited existing documents: `README.md` (rewritten for the implemented state) and a two-line status note at the top of `docs/ARCHITECTURE_CONTRACTS.md`.
- Not committed, by design: `.venv/`, `data/` (database, reports, run logs).

Outside the repository I changed nothing persistent. A project memory note for future Claude Code sessions was written under `~/.claude/projects/…/memory/`.

## 9. Remaining work, by priority

**Needs your decision first**

1. **Search provider for unattended use.** Options: stay on keyless DuckDuckGo and accept occasional throttled nights; run SearXNG in Docker on the MSI (adapter exists, untested live, needs your approval because it is a new service); or a free-tier keyed API such as Brave (needs you to sign up and an adapter of about 40 lines). I recommend SearXNG if Docker on the MSI is acceptable to you.
2. **MSI access.** I could not log in, so model inventory, GPU and throughput there are unknown. Either add this laptop's SSH key or run `forage doctor --live` on the MSI yourself.
3. **Installing an embedding model** (for example `nomic-embed-text`, about 270 MB) would allow much better deduplication and grouping. It is a model download, so I did not do it.
4. **Whether to merge** `initial-implementation` into `main` and push.

**Engineering, most valuable first**

5. First supervised night: run `forage queue night --max-sessions 4` in a terminal during the real window with two watches and review resource use, before any OS scheduling.
6. Check numbers and dates in the model's summary against the cited findings; reject or strip mismatches.
7. Embedding-based matching for findings and claims (after decision 3).
8. Second-chance extraction: when the audit discards every claim from a page the extractor rated highly, re-audit once or keep the page as a lead.
9. Follow links from fetched pages to primary sources, within the same budget and guard.
10. Give triage and the audit the excluded scope as explicit examples, to reduce drift such as item 3 in section 7.
11. A report digest mode (a few lines per session) for future delivery adapters, alongside the full report.
12. Windows/WSL smoke test and a decision on where `data_dir` lives on the MSI's secondary SSD.

**For the CSC2524 project**

13. Confirm the framing with the instructor before investing further in evaluation.
14. Write the relevance rubric and label ground truth independently of the system.
15. More scenarios (source changes, premature-review traps, a watch where continuing is right), repeats, and a live overnight pilot with at least three watches.
16. A small local UI for the demo. The report schema and the store are the only things it needs.

## 10. Recommended next milestone

**A supervised multi-night live pilot on the MSI with three watches**, before any UI or Tari work. It is the shortest path to finding out whether the pieces hold up where Forage is meant to run: search reliability over several nights, real model throughput and budgets on that hardware, whether the adaptive policy ever fires on real data and whether its questions are sensible, and how the re-query interval behaves over days. It needs decisions 1 and 2 first. Everything it produces (`forage export`) feeds the CSC2524 evaluation directly.
