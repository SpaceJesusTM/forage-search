# Forage Search

**Forage** is a standalone, local-first, persistent web-research agent. It investigates user-defined **research watches** over repeated bounded sessions, saves evidence and research state, and reports **new or materially changed findings** instead of starting from scratch every time.

Repository: `forage-search` · Working product name: **Forage** · Stage: **first implementation (P0–P2 working, P3 evaluation support in place)**. See [`docs/SESSION_HANDOFF.md`](docs/SESSION_HANDOFF.md) for what was tested and what was not.

## Why it exists

A user can assign ongoing research such as “find MScAC-relevant AI/ML internships,” “monitor Toronto AI/technology events,” or any custom information need. Forage should work unattended, cite primary sources, use prior work to choose the next investigation, and avoid repeating old results.

Longer term, Tari/OpenClaw can use Forage as a **separate research worker** for Discord forum-based research watches. The research worker remains independently operable without Tari, Discord, OpenClaw, or any cloud model.

A possible CSC2524 AI+HCI research extension adds **adaptive human-guidance checkpoints**: when a watch hits an objective-changing decision that cannot be resolved within the user's approved scope, request a review, set that watch to `AWAITING_REVIEW`, and allocate the shared local compute budget to other eligible watches. This is a hypothesis to implement and evaluate, **not an established feature or proven benefit**. The mechanism is implemented; the evaluation has not been done.

## What it does today

- Turns a natural-language objective into a structured brief with the local model, and researches nothing until you approve it.
- Runs a bounded session: plans searches from what it already knows, searches the web, reads pages, extracts claims with verbatim quotes, and keeps only claims whose quote is found in the stored page text and whose numbers and names appear on the page.
- Remembers findings, sources, searches, open questions and next directions in SQLite. A later session follows up saved leads, skips pages it has read, rejects repeated searches, and reports only what is new, updated, newly corroborated or no longer confirmed.
- Writes one report per session as validated JSON and Markdown, with source-stated claims kept apart from model-written synthesis.
- Runs several watches in fair round-robin inside a night window, one session at a time.
- Supports three interchangeable checkpoint policies (`continuous`, `periodic`, `adaptive`) and a review flow in which scope changes need explicit confirmation.

It uses only a local Ollama model and keyless public web search. It needs no API keys and starts no services.

## Requirements

- Python 3.11 or newer
- [Ollama](https://ollama.com) running locally with at least one chat model installed. Forage picks the largest installed model under `max_model_gb` unless you name one. It never pulls or removes models.
- Internet access for search and page retrieval

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/forage doctor --live      # checks database, Ollama, model, search and fetch
```

## Use

```bash
# 1. Describe what to research. The local model drafts a brief; nothing runs yet.
forage watch create "What techniques make small local language models produce reliable JSON, and what measured evidence supports each?"

# 2. Read the brief, adjust it if needed, confirm it.
forage watch show <watch>
forage watch edit <watch> --exclude "hosted cloud APIs" --hint https://example.org/
forage watch approve <watch>

# 3. Run a bounded session. Budgets are hard limits.
forage run <watch> --max-seconds 600 --max-searches 4 --max-pages 6

# 4. Read the result.
forage report <watch>                 # Markdown; --format json for the structured report
forage findings <watch> --evidence    # everything known so far, with quotes and URLs
forage questions <watch>              # open and answered questions, saved directions

# 5. Run it again later. The second session builds on the first.
forage run <watch>
forage compare <watch>                # before/after: what was reused, repeated, new
forage runs <watch>                   # history with budget use
forage budget <watch>                 # where model time went
```

`<watch>` can be the full id or any unique prefix. Reports are also written to `data/reports/<watch>/`.

### Control

```bash
forage watch pause <watch>            # immediate, no confirmation
forage watch resume <watch>           # proposes a plan; needs `watch approve`
forage watch close <watch>            # archive; nothing is deleted
forage watch reopen <watch>           # proposes a plan; needs `watch approve`
forage watch policy <watch> adaptive  # or continuous, periodic --config '{"every": 3}'
```

### Several watches

```bash
forage queue status                             # rotation, cursor, window, why each watch is or is not eligible
forage queue run --max-sessions 3 --ignore-window   # bounded manual run, any time of day
forage queue night --max-sessions 12            # foreground runner; admits only 02:00–06:00 America/Toronto
```

`queue night` is an ordinary foreground process. **Forage does not install a cron job, launchd agent or Task Scheduler entry.** If you want unattended nights, schedule that one command yourself after reviewing resource use.

### Reviews

When a policy decides a choice belongs to you, the watch moves to `AWAITING_REVIEW` and gets no research time until you answer. Other watches keep their turns.

```bash
forage review list
forage review show <request>
forage review respond <request> --choice continue
forage review respond <request> --choice amend --remove-exclude "remote" --confirm-scope-change
```

An answer that changes the approved scope is refused, with a preview, unless `--confirm-scope-change` is given.

### Evaluation

```bash
forage export --out exports/               # JSONL of every log table plus per-watch summaries
forage experiment --out experiments/run1   # SIMULATED three-policy comparison on a built-in scenario
```

The experiment uses a fixture web, a rule-based stand-in for the model and a scripted user. It shows how the mechanism behaves. It is not evidence that any policy improves real research.

## Configuration

Defaults work on a machine with Ollama on its usual port. Override with `FORAGE_*` environment variables or a `forage.toml` in the working directory; see [`forage.example.toml`](forage.example.toml).

| Setting | Default | Meaning |
| --- | --- | --- |
| `data_dir` | `./data` | Database, reports, logs. Not committed. |
| `ollama_url` | `http://127.0.0.1:11434` | Local model server |
| `model` | largest installed under `max_model_gb` | Model tag |
| `search_provider` | `duckduckgo` | Also `ddgs`, `searxng`, `fixture`; chain with `+` |
| `max_duration_seconds` / `max_search_requests` / `max_fetched_pages` / `max_llm_output_tokens` | 900 / 6 / 8 / 9000 | Per-session budget |
| `requery_after_hours` | 20 | A search may be run again for newer results after this long |
| `night_start` / `night_end` / `night_timezone` | 02:00 / 06:00 / America/Toronto | Admission window for `queue night` |
| `scheduler_authority` | `standalone` | Set to `external` when another controller owns scheduling |

## Tests

```bash
.venv/bin/python -m pytest            # 219 deterministic tests; simulated web and model, no network
.venv/bin/python -m pytest -m live    # 4 small live checks: Ollama, search, fetch, address guard
```

## What to keep in mind

- **Verified means quoted, not true.** A reported claim's quote was found on the retrieved page. The model can still attach a number to the wrong subject or pick a weak source. Each claim is shown with its quote and URL so you can check.
- **The model is small.** With a 4B model, relevance judgement and grouping are rough. Summaries and answers are labelled as model inference.
- **Keyless search is fragile.** DuckDuckGo throttles bursts and its markup can change. Queries are paced; failures are reported, not hidden. For unattended use, configure a search service you control.
- **Retrieved pages are untrusted.** See the security section of [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Documentation

- [`docs/SESSION_HANDOFF.md`](docs/SESSION_HANDOFF.md) — what was built, how it was tested, live results, known defects, next steps.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the architecture as built, evidence rules, security model.
- [`docs/LANGGRAPH_ARCHITECTURE_DECISION.md`](docs/LANGGRAPH_ARCHITECTURE_DECISION.md) — evaluation of LangGraph as the session orchestrator, with a prototype and measurements (branch `experiment/langgraph`).
- [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) — environment findings, stack choices, milestone status.
- [`docs/PRODUCT_SPEC.md`](docs/PRODUCT_SPEC.md) — scope, behaviours, lifecycle, milestones (design specification).
- [`docs/ARCHITECTURE_CONTRACTS.md`](docs/ARCHITECTURE_CONTRACTS.md) — proposed boundaries and data contracts (design specification).
- [`docs/CSC2524_EVALUATION.md`](docs/CSC2524_EVALUATION.md) — research question, baselines, metrics, limitations (design specification).
- [`docs/TARI_INTEGRATION_LATER.md`](docs/TARI_INTEGRATION_LATER.md) — deferred OpenClaw/Discord integration.
- [`CLAUDE.md`](CLAUDE.md) and [`prompts/CLAUDE_CODE_LONGRUN.md`](prompts/CLAUDE_CODE_LONGRUN.md) — coding-agent guardrails and the first assignment.
