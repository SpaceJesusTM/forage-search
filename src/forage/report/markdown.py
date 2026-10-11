"""Human-readable rendering of a ResearchReport.

All text that originated on the web or in the model is passed through `md_text`
so it cannot inject links, images or markup into the document.
"""

from __future__ import annotations

from forage.engine.safety import md_text, md_url
from forage.report.schema import ReportFinding, ReportQuestion, ResearchReport

_CHANGE_LABEL = {
    "NEW": "New",
    "UPDATED": "Updated",
    "CORROBORATED": "Corroborated by another source",
    "STALE": "No longer confirmed",
    "UNCHANGED": "Unchanged",
}


def _link(url: str) -> str:
    safe = md_url(url)
    return f"<{safe}>" if safe else "(url withheld: not http/https)"


def _finding(lines: list[str], finding: ReportFinding, sources_index: dict[str, int]) -> None:
    lines.append(f"### {_CHANGE_LABEL[finding.change_type]}: {md_text(finding.title, 160)}")
    lines.append("")
    meta = [f"id `{finding.finding_id}`", f"{finding.independent_sources} independent source(s)"]
    if finding.first_seen_session:
        meta.append(f"first seen in session {finding.first_seen_session}")
    lines.append("*" + " · ".join(meta) + "*")
    if finding.change_note:
        lines.append("")
        lines.append(f"Change: {md_text(finding.change_note, 200)}")
    lines.append("")
    for claim in finding.claims:
        tag = " **(new this session)**" if claim.new_this_session and finding.change_type != "NEW" else ""
        ref = sources_index.get(claim.source_id)
        cite = f"[S{ref}]" if ref else f"({md_text(claim.source_domain, 80)})"
        status = "quote verified" if claim.verification_status == "quote_verified" else "quote matched approximately"
        if claim.support == "partial":
            status += ", partly supported"
        elif claim.support == "unchecked":
            status += ", support not checked"
        lines.append(f"- {md_text(claim.text, 500)}{tag} {cite}")
        lines.append(f"  - Source text: \"{md_text(claim.quote, 500)}\"")
        lines.append(f"  - {status}; checked {claim.checked_at}; {_link(claim.evidence_url)}")
        if claim.note:
            lines.append(f"  - Note: {md_text(claim.note, 240)}")
    if finding.claims_not_shown:
        lines.append(f"- ({finding.claims_not_shown} more verified claim(s) on record; see `forage findings --evidence`)")
    if finding.retired_claims:
        lines.append("")
        lines.append("No longer stated by the source (retired):")
        for text in finding.retired_claims:
            lines.append(f"- ~~{md_text(text, 400)}~~")
    if finding.uncertainties:
        lines.append("")
        lines.append("Uncertainty:")
        for note in finding.uncertainties:
            lines.append(f"- {md_text(note, 320)}")
    lines.append("")


def _question_line(question: ReportQuestion) -> str:
    extra = f" (searched {question.searches_so_far}x)" if question.searches_so_far else ""
    return f"- {md_text(question.text, 300)}{extra} `{question.question_id}`"


def render_markdown(report: ResearchReport) -> str:
    lines: list[str] = []
    add = lines.append
    add(f"# {md_text(report.watch_title, 160)} — session {report.session_number}")
    add("")
    if report.simulated:
        add("> **SIMULATED RUN.** Produced from fixture pages and/or a simulated model. Nothing here is real research.")
        add("")
    add(f"**Objective:** {md_text(report.objective, 900)}")
    add("")
    add(f"**Result:** {report.headline}")
    add("")
    add(f"Run `{report.run_id}` · status {report.status} · started {report.started_at}"
        + (f" · finished {report.finished_at}" if report.finished_at else ""))
    add(f"Model `{report.model or 'unknown'}` · search provider `{report.search_provider or 'unknown'}`"
        + (f" · stopped: {md_text(report.stop_reason, 80)}" if report.stop_reason and report.stop_reason != "completed" else ""))
    if report.error:
        add("")
        add(f"**Error:** {md_text(report.error, 400)}")
    add("")

    if report.pending_review:
        review = report.pending_review
        add("## Decision needed from you")
        add("")
        add(f"{md_text(review.question, 500)}")
        add("")
        for choice in review.choices:
            default = " (default if you do not answer)" if choice.get("id") == review.safe_default else ""
            add(f"- `{md_text(str(choice.get('id', '')), 20)}` {md_text(str(choice.get('label', '')), 240)}{default}")
        add("")
        add(f"This watch is paused for review and uses no research budget until you answer. Request `{review.request_id}`.")
        add("")

    sources_index = {s.source_id: i + 1 for i, s in enumerate(report.sources)}
    # Sources cited by findings but read in an earlier session still need a reference number.
    for finding in report.findings:
        for claim in finding.claims:
            sources_index.setdefault(claim.source_id, len(sources_index) + 1)

    if report.findings:
        add("## At a glance")
        add("")
        for finding in report.findings:
            add(f"- {_CHANGE_LABEL[finding.change_type]}: {md_text(finding.title, 140)} "
                f"({len(finding.claims)} claim(s), {finding.independent_sources} source site(s))")
        add("")

    add("## Findings from sources")
    add("")
    add("Each statement below was extracted by the local model and then checked: the quoted source text was "
        "located on the retrieved page. That shows the source says it, not that it is true.")
    add("")
    if report.findings:
        for finding in report.findings:
            _finding(lines, finding, sources_index)
    else:
        add("No new, updated or newly corroborated findings this session.")
        add("")
    if report.unchanged_findings:
        add(f"{report.unchanged_findings} earlier finding(s) were seen again with nothing new and are not repeated here.")
        add("")

    add("## Model synthesis (inference, not source text)")
    add("")
    add(md_text(report.summary, 1200) if report.summary else "No synthesis was produced for this session.")
    add("")
    if report.questions_answered:
        add("Questions the model judges answered by the findings above:")
        add("")
        for question in report.questions_answered:
            add(f"- {md_text(question.text, 300)}")
            add(f"  - Answer (model inference): {md_text(question.answer or '', 400)}")
            if question.answered_by:
                add("  - Based on: " + ", ".join(f"`{fid}`" for fid in question.answered_by))
        add("")

    add("## Open questions")
    add("")
    if report.questions_open:
        opened = {q.question_id for q in report.questions_opened}
        for question in report.questions_open:
            add(_question_line(question) + (" — new this session" if question.question_id in opened else ""))
    else:
        add("None recorded.")
    add("")

    add("## Next research directions")
    add("")
    if report.next_directions:
        for direction in report.next_directions:
            reason = f" — {md_text(direction.rationale, 240)}" if direction.rationale else ""
            add(f"- {md_text(direction.query, 200)}{reason}")
    else:
        add("None proposed this session.")
    add("")

    add("## Uncertainty and limits")
    add("")
    for note in report.uncertainties:
        add(f"- {md_text(note, 500)}")
    add("")

    comparison = report.comparison
    add("## How this session used earlier research")
    add("")
    if comparison.prior_sessions == 0:
        add("This was the first session for this watch; there was no earlier state to build on.")
    else:
        add(f"- Started from {comparison.known_findings_before} known finding(s), {comparison.open_questions_before} open "
            f"question(s) and {comparison.queries_already_tried_before} earlier search(es) across {comparison.prior_sessions} prior session(s).")
        add(f"- Searches repeated from recent sessions: {comparison.queries_repeated}. "
            f"Older searches deliberately run again for newer results: {comparison.queries_rerun_after_interval}. "
            f"Proposed queries rejected as repeats: {comparison.queries_rejected_as_repeats}.")
        add(f"- Search results already visited and skipped: {comparison.known_urls_skipped}. "
            f"Saved leads followed up: {comparison.saved_leads_followed}. Sources re-checked: {comparison.sources_rechecked}.")
        add(f"- Duplicate claims suppressed: {comparison.duplicate_claims_suppressed}.")
        add(f"- Known findings now: {comparison.known_findings_after}. Open questions now: {comparison.open_questions_after}.")
    add("")

    add("## Searches")
    add("")
    if report.searches:
        add("| Query | Status | Results | New pages | Already known |")
        add("| --- | --- | ---: | ---: | ---: |")
        for search in report.searches:
            add(f"| {md_text(search.query, 120)} | {search.status.lower()} | {search.results} | {search.new_urls} | {search.known_urls} |")
    else:
        add("No searches were run.")
    add("")

    add("## Sources retrieved")
    add("")
    if report.sources:
        for source in report.sources:
            ref = sources_index.get(source.source_id, 0)
            detail = [source.outcome.replace("_", " ")]
            if source.claims_kept or source.claims_rejected:
                detail.append(f"{source.claims_kept} claim(s) kept, {source.claims_rejected} rejected")
            if source.flags:
                detail.append("flags: " + ", ".join(md_text(f, 40) for f in source.flags))
            if source.note:
                detail.append(md_text(source.note, 160))
            add(f"- [S{ref}] {md_text(source.title or source.domain, 140)} — {_link(source.url)}")
            add(f"  - {'; '.join(detail)}; retrieved {source.fetched_at}")
    else:
        add("No pages were retrieved.")
    add("")

    m, b = report.metrics, report.budget
    add("## Budget use")
    add("")
    add("| Resource | Used | Limit |")
    add("| --- | ---: | ---: |")
    add(f"| Wall time (s) | {m.get('wall_seconds', 0)} | {b.get('max_duration_seconds', '-')} |")
    add(f"| Search requests | {m.get('search_requests', 0)} | {b.get('max_search_requests', '-')} |")
    add(f"| Pages fetched | {m.get('fetched_pages', 0)} | {b.get('max_fetched_pages', '-')} |")
    add(f"| Model output tokens | {m.get('llm_output_tokens', 0)} | {b.get('max_llm_output_tokens', '-')} |")
    add(f"| Model prompt tokens | {m.get('llm_prompt_tokens', 0)} | - |")
    add(f"| Model calls | {m.get('llm_calls', 0)} | - |")
    add(f"| Model time (s) | {m.get('llm_seconds', 0)} | - |")
    add("")
    add(f"Claims extracted {m.get('claims_extracted', 0)}, kept {m.get('claims_accepted', 0)}, "
        f"rejected {m.get('claims_rejected', 0)}.")
    add("")
    return "\n".join(lines)
