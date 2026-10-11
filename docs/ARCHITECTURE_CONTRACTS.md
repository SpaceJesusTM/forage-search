# Architecture and Contracts (Proposed, Not Yet Implemented)

The coding agent may choose the Python libraries and concrete modules. Preserve these **logical boundaries**:

```text
CLI / optional local demo UI / future Tari-Discord adapter
                        |
          Watch controller + approval state
                        |
          Durable scheduler and run manager
                        |
              Research-session API
             /        |          \
      local LLM   search/fetch   evidence store
             \        |          /
             Structured report + metrics
                        |
             checkpoint policy interface
             CONTINUE | REQUEST_REVIEW
```

### Boundary A: Research engine

Input: durable watch ID, approved objective+constraints, memory/context and explicit per-session budgets. Output: structured report, updated research memory, metrics and potentially a **candidate** checkpoint proposal. No Discord IDs, UI callbacks or access to private Tari state in the core.

### Boundary B: Watch/controller state

Controller is authoritative for approval, current lifecycle and eligibility; validates proposed updates; owns queue admission and cancellation. The research model can propose a new direction, but **cannot directly make an unauthorized change to user-approved scope**. Store provenance of changes and resolved checkpoints.

### Boundary C: Checkpoint policy (experimental variable)

The policy receives structured state/observations and returns one of:

- `CONTINUE` with an explanation (normal case).
- `REQUEST_REVIEW` with triggering evidence, precise human decision, choices and a cost/urgency indicator.

Policies to support under same runner:

1. `continuous`: never proactively requests strategic review (still honors explicit user pause, corrections and safety boundaries).
2. `periodic`: proposes review at fixed eligible-session intervals, holding control until answered under the experimental configuration.
3. `adaptive`: triggers review based on changes, uncertainty about **user intent**, evidence of stalled productivity or conflicts requiring choice. Avoid asking when more research within approved scope can resolve uncertainty.

An optional `CONSTRAINT_VIOLATION` or hard-stop gate should be **common to all three policies**, not counted as the adaptive research benefit.

### Boundary D: Scheduler

Select only eligible watch IDs, one bounded session at a time, with persisted fair rotation and a single scheduling authority. An `AWAITING_REVIEW` watch is excluded from future admissions. A watch paused during execution should cancel when practical and must not publish a stale report as an active update.

### Example abstract run input

```json
{
  "watch_id": "internships-demo",
  "run_id": "caller-issued-unique-id",
  "budget": {
    "max_duration_seconds": 900,
    "max_search_requests": 15,
    "max_fetched_pages": 25,
    "max_llm_output_tokens": 6000
  }
}
```

### Example abstract research report

```json
{
  "run_id": "caller-issued-unique-id",
  "watch_id": "internships-demo",
  "status": "completed",
  "has_meaningful_update": true,
  "findings": [
    {
      "finding_id": "stable-evidence-derived-id",
      "change_type": "NEW",
      "title": "Illustrative result (NOT a real finding)",
      "claims": [
        {
          "text": "Illustrative claim for schema only",
          "evidence_url": "https://example.com/illustrative",
          "checked_at": "2026-10-11T10:00:00Z",
          "verification_status": "unverified_example"
        }
      ],
      "uncertainties": []
    }
  ],
  "pending_review": null,
  "metrics": {
    "search_requests": 0,
    "fetched_pages": 0,
    "llm_tokens_estimate": 0,
    "wall_seconds": 0
  }
}
```

These are **illustrative contracts**, not locked database schema or validated live results. Schema design can evolve, but it must be stable, documented, versionable and testable before Tari integration.

### Storage

SQLite is a practical first choice. Minimum durable entities: `watch`, `watch_revision/approval`, `session_run`, `source`, `observation/claim`, `finding`, `report`, `search_attempt`, `checkpoint_request`, `checkpoint_response`, `queue_state`, and `delivery/outbox` if adapters publish externally. Use migrations and unique IDs to make retries idempotent.

### Research provider boundaries

- Local LLM adapter: discover installed Ollama models; do not hard-code tag/VRAM. Use validated structured outputs and retry/repair policy for malformed data.
- Search adapter: configurable public web search with errors, rate limits and budget accounting; deterministic fixture provider for offline tests. Do not require a paid API in P0.
- Fetch/extract adapter: prefer primary-source pages, record actual URL, timestamps and page provenance; handle robots/rate limits and prompt injection.
- Report renderer: structured JSON as authority; Markdown for human-readable local reports and optional future Discord formatting.

### Security boundaries

Network content is never a trusted command. Keep fetched text out of system-level instructions. Do not allow page text to modify objectives, invoke filesystem/shell access, or access secret environment variables. Narrow tool permissions, validate allowed URLs where applicable, rate-limit. The reusable research agent must never inherit Gmail or Discord credentials simply because Tari invokes it.

### Local runtime

Optimize for existing MSI GS66 environment and preserve running services. Store large model caches and growing persistent research datasets on the secondary SSD **if configured and available**, rather than hard-coding a Windows drive path. Support configurable data paths and portable deployment (host process or container), but avoid premature infrastructure churn.
