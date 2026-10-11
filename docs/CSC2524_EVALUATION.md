# CSC2524 Research Extension — Candidate, Not Finalized

**Course framing:** AI+HCI; substantial working prototype, 6-page two-column report (max 1 page references), live presentation/demo; user study is **not required**. Confirm research framing and instructor expectations before treating these notes as the final project.

## Candidate question

**Can adaptive human-guidance checkpoints improve goal alignment and useful discovery per unit of research cost when a resource-constrained LLM agent manages multiple long-running research workflows?**

An asynchronous, local-first agent should sometimes stop consuming its limited overnight budget on a watch whose next meaningful decision belongs to the user, ask a specific question, put that watch in `AWAITING_REVIEW`, and continue other eligible watches. It must also avoid **premature** or **unnecessary** pauses. This is a falsifiable hypothesis, not a guaranteed improvement.

### Relationship to HCI work

- [**Cocoa: Co-Planning and Co-Execution with AI Agents** (CHI 2026)](https://dl.acm.org/doi/10.1145/3772318.3791673): primary co-planning anchor. Cocoa already supports revising plans, so **do not claim that editable objectives or re-planning are new**. The proposed extension is selective **re-engagement** when multiple autonomous research workflows persist across disconnected sessions and users are unavailable during execution.
- [**When Should Users Check? Modeling Confirmation Frequency in Multi-Step Agentic AI Tasks** (CHI 2026)](https://dl.acm.org/doi/10.1145/3772318.3790655): confirmation timing and intervention overhead. Examine actual evaluation/method carefully before novelty claims.
- [**Morae: Proactively Pausing UI Agents for User Choices** (UIST 2025)](https://dl.acm.org/doi/10.1145/3746059.3747797): selective user choice points; must distinguish their policy from ours.
- [**DroidRetriever: A Transparent and Steerable Automation System for Collaborative Mobile Information Seeking** (CHI 2026)](https://dl.acm.org/doi/10.1145/3772318.3790396): domain precedent for steerable information seeking.

The HCI contribution must be the **human–agent delegation/re-engagement mechanism** and its attention tradeoffs, not merely a better round-robin scheduler. Novelty remains to be confirmed by close readings/instructor approval. Avoid invented paper findings.

## Competing approaches (same research engine)

**A. Continuous baseline (original Tari Scout idea):** The user sets/approves a watch. It stays eligible and receives normal round-robin research turns until manually redirected/paused, time ends or common safety restrictions apply. It may receive explicit corrections; don't cripple it on purpose.

**B. Periodic review:** At fixed session boundaries, invite user review and move the watch to awaiting review until the decision is made. Same engine, queue, resources and manual controls as A/C.

**C. Adaptive review (candidate contribution):** Request guidance when **decision-relevant signals** indicate the research plan may be poorly defined, scope must change to continue productively, or further work has low marginal value relative to other watches. Skip research on a pending watch and allocate eligible time to other watches. Record why each checkpoint was requested. Safe default is no scope change without approval.

All conditions can receive explicit user edits. The intervention policy, not access to user information, must be the principal difference.

## Evaluation design

1. Use >=3 concurrent watch topics and multi-session temporal scenarios: internships, tech/career events, and at least one distinct custom research topic.
2. Create **replayable web snapshots/fixtures** with known arrivals, changes, source credibility and ambiguous choices. Use scripted user responses at consistent scheduled times and information availability; add a smaller live overnight pilot for ecological realism.
3. Include scenarios where human choice really matters (e.g., proposed broadening beyond approved priorities), where the agent should self-resolve uncertainty, where the strategy still works, where a source changes, and where a request for review would be premature.
4. Run multiple repeats/seeds where feasible. Fix available model, provider/search corpus, task mix, per-session and total overnight time/search/token budget, scheduler rules and response delay. Isolate variation to checkpoint policy as far as feasible. Include cost of policy inference itself.
5. Log every search, source, token estimate/latency, watch turn, triggered/declined checkpoint, user response, outcome and evidence. Predefine an evaluation rubric rather than judging outputs post hoc.

## Measurement plan

**Primary:** number of *novel, relevant, source-verified* discoveries per 100 search requests (and, if measurable, per token/GPU-minute). Report raw numerator and budget consumption as well as ratio.

**Secondary:** objective adherence; repeated/irrelevant searches; detection of changed findings; evidence correctness and freshness; per-watch useful coverage; time to useful finding; total useful findings across watches; compute redistribution; fairness/starvation across topics; missed/late needed reviews; unnecessary/premature checkpoints; human decisions/questions needed; false broadening; robustness after restarts.

**Critical confounds:** A policy that simply stops difficult watches may look efficient but miss valuable discoveries. Show both aggregate and per-watch quality, number of blocked watches, and utilization. A policy can redistribute all compute without reducing total compute expense. User attention is another scarce resource, but number of check-ins is only a rough proxy for experienced burden.

**Ground-truth review need:** Design scenarios where a user must decide an explicit tradeoff versus scenarios where evidence gathering alone is sufficient. The review labels should come from the approved objective and scenario/rubric, not solely from the agent being evaluated.

## Testable hypotheses (to refine)

- **H1:** Adaptive checkpoints yield more goal-aligned, source-verified novel findings per unit of research budget than uninterrupted round-robin research on mixed watch sets with genuine objective ambiguity.
- **H2:** Adaptive checkpoints require fewer unnecessary reviews than fixed-period review at comparable useful coverage.
- **H3:** Checkpoint-aware queue management reduces wasted work on blocked watches **without materially starving high-value watches**.

Negative outcomes matter. It is possible that continuous search performs better when no user choice is required, or that adaptive checkpoints are too eager and reduce total coverage.

## Demo requirements

A deterministic, rehearsable local demo should show: (1) creating/approving watches; (2) a research session with verifiable evidence and remembered history; (3) one watch surfacing a concrete tradeoff and entering `AWAITING_REVIEW`; (4) a separate watch continuing under the shared budget; (5) a user answer changing the first watch's approved scope; and (6) a comparison visualization/table of all three checkpoint policies. A custom UI is optional but may improve classroom delivery; Tari/Discord must not be necessary.

## What results can legitimately claim

A system-level evaluation may support differences in behavioral alignment, budget allocation, search quality and frequency of requested interventions under controlled simulated user actions. **It cannot prove users feel more control, prefer checkpoints, trust the system, or are less cognitively burdened** without participant evaluation. These should be proposed as future studies, not reported findings.

## Course delivery

Prioritize reproducible prototype and clear demo (oral/live presentation worth half the project grade), then 6-page UIST-format report: motivation/RQ; related work; interaction mechanism; architecture; evaluation setup; results/limitations/future work. Track artifacts/logs from early runs to avoid finishing with a demo but no results.
