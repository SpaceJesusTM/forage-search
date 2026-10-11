# Future Tari/OpenClaw/Discord Integration — Deferred

This is a **future adapter plan**, not work for the first Forage milestone. Forage must remain fully usable without Tari.

## Intended user experience

Tari uses the existing Discord `#projects` **Forum**. **One user-created forum thread = one persistent research watch** (not a new thread per job/event/finding). In a new post the user tells Tari what to research; Tari interprets the intent, proposes a structured plan and asks the user to confirm once. After confirmation that watch is active.

Forage runs bounded research sessions and returns a single structured report per session. Tari publishes **at most one concise update** into the **same originating thread**, or publishes nothing for a no-change session. The user can ask Tari for summaries, comparisons or plans over the persisted research store.

Original workflow controls: 

- **Pause**: user says pause; immediate queue removal; thread stays visible, history kept; no confirmation needed.
- **Resume**: Tari reviews objective/history, proposes updated direction and asks for one confirmation *before* reactivating.
- **Close**: remove from queue and archive thread; research history remains; distinct from deletion.
- **Reopen**: unarchive/review/reconfirm, then reactivate; no destructive reset.
- **Adaptive guidance**: Forage may yield a `REQUEST_REVIEW` decision. Tari posts one specific question in that existing thread; watch becomes `AWAITING_REVIEW` and leaves the research queue. Other watches continue. User response resolves review and, when needed, confirms the revised scope before requeueing. Avoid overnight cloud-model calls for every queue step.

## Clean responsibility split

**Forage:** local LLM query planning, web/tool calls, evidence extraction/verification, source tracking, deduplication, longitudinal memory, structured reports, research checkpoint proposal. It does not hold Discord/Gmail credentials.

**Tari/controller:** interpret user intents with stronger model if appropriate, confirm goal changes, own forum-thread mappings, approve state transitions, schedule/dispatch research runs (when integrated), handle confirmations, publish reports and respond to user questions. Routine queue operations should be deterministic rather than model-powered.

## Queue semantics

Persist a sequential round-robin queue. Nightly window: 02:00–06:00 **America/Toronto**. Never admit a new job at/after 06:00. Allow one bounded previously admitted job to finish. Keep cursor across nights. Skip paused, closed, pending and awaiting-review watches without losing historical state or resetting fairness. An active standalone scheduler **must be disabled if Tari assumes scheduling authority**, to prevent duplicate work and delivery.

## Integration API boundary

Expect narrow start/status/report/cancel interfaces with stable `watch_id`/`run_id` and a versioned report schema; asynchronous execution is preferable. Tari's bridge verifies state and resolves a canonical `discord_thread_id` from its own mapping. **Never rely on a model-generated thread ID to decide the publication destination**. Use durable run IDs, an outbox and message IDs for idempotent delivery. If paused/closed mid-run, do not publish an active-status report accidentally.

## Deployment and safety

The MSI GS66 hosts Tari already. Do not disrupt its runtime or existing credentials. Forage may use Ollama (which may share resources with other projects), so enforce conservative compute limits and run times. Keep service endpoints private/loopback and use a restricted invocation contract. Discord forum integration and any stronger cloud model availability should be verified separately before implementation.

The two initial watches will likely be MScAC AI/ML internship opportunities and Toronto/U of T career/tech events, but the engine remains fully generic.
