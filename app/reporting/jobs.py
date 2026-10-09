"""Read-only HTML review dashboard over persisted jobs.

Purpose
-------
Make the pipeline legible to a person. Everything upstream of this point -
sources, adapters, ingestion, deduplication, eligibility - is invisible without
opening a JSONL file, and the run ledger records *counts* rather than
*content*. This renders one row per job with the decision context attached.

Design commitments
------------------
* **Read-only.** Nothing here writes to the store, mutates a job, or changes
  state. Rendering cannot alter what it renders.
* **No JavaScript.** The output is a single self-contained HTML file with no
  ``<script>`` and no external references. A report that will be opened from
  disk and shared should not carry an execution surface. Filtering and sorting
  are therefore applied at *render* time through parameters, not in the page.
* **Honest about gaps.** A column whose value does not exist yet says
  ``not yet evaluated`` rather than rendering blank, because a blank cell reads
  as "nothing to report" when it actually means "not computed".
* **Escape everything.** Every value interpolated into markup goes through
  ``html.escape``. Job descriptions come from third parties and are untrusted.

Not here: matching (increment 6), status decisions (increment 5), and any write
path. Application status appears as an explicit placeholder until increment 5
owns the real model.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from app.jobs.eligibility import EligibilityVerdict, evaluate_eligibility
from app.jobs.models import Job, RemoteStatus
from app.jobs.status import StatusError, StatusLog
from app.jobs.store import JobStore


#: Rendered wherever a value has not been computed yet.
NOT_EVALUATED = "not yet evaluated"

#: Application status is a placeholder until increment 5 defines the model.
STATUS_PLACEHOLDER = "—"

#: Colour bands for an eligibility verdict.
_VERDICT_CLASS = {
    "eligible": "ok",
    "not_eligible": "no",
    "unknown": "maybe",
}

#: Colour bands for a match tier. Kept visually distinct from eligibility:
#: a job can be ineligible on location and still be a strong match on skills,
#: and collapsing the two would hide that.
_TIER_CLASS = {
    "strong_match": "tier-strong",
    "credible_match": "tier-credible",
    "stretch": "tier-stretch",
    "unsuitable": "tier-unsuitable",
    "not_yet_evaluated": "tier-unevaluated",
}

#: Order tiers sort by, best first. Unevaluated results sort last rather than
#: being dropped, so an unassessed job is visible instead of invisible.
_TIER_ORDER = {
    "strong_match": 0,
    "credible_match": 1,
    "stretch": 2,
    "unsuitable": 3,
    "not_yet_evaluated": 4,
}

_STYLE = """
:root{color-scheme:light dark}
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:2rem;line-height:1.45}
h1{font-size:1.4rem;margin:0 0 .25rem}
p.meta{color:#6b7280;margin:0 0 1.25rem;font-size:.9rem}
.summary{display:flex;gap:1rem;flex-wrap:wrap;margin:0 0 1.25rem;font-size:.9rem}
.summary span{border:1px solid #d1d5db;border-radius:.4rem;padding:.3rem .6rem}
table{border-collapse:collapse;width:100%;font-size:.9rem}
caption{text-align:left;font-weight:600;padding:.4rem 0}
th,td{border:1px solid #d1d5db;padding:.45rem .55rem;text-align:left;vertical-align:top}
th{background:#111827;color:#f9fafb;position:sticky;top:0}
td.wrap{max-width:26rem}
.ok{background:#dcfce7}.no{background:#fee2e2}.maybe{background:#fef9c3}
.flag{background:#f3f4f6;color:#374151;font-size:.82rem;border-radius:.25rem;
      padding:.05rem .3rem;display:inline-block;margin:0 .2rem .2rem 0}
.badge{font-size:.82rem;color:#6b7280}
a{color:#2563eb}
.empty{padding:2rem;text-align:center;color:#6b7280;border:1px dashed #d1d5db}
.coverage{display:flex;gap:.75rem;flex-wrap:wrap;margin:0 0 1rem;font-size:.85rem;
          border:1px solid #d1d5db;border-radius:.4rem;padding:.5rem .6rem}
.notice{border-left:3px solid #b45309;background:#fffbeb;padding:.5rem .75rem;
        margin:0 0 1rem;font-size:.9rem}
.attribution{margin:1.25rem 0 0;font-size:.85rem;border-top:1px solid #d1d5db;
             padding-top:.75rem;color:#374151}
.attribution ul{margin:.4rem 0 0;padding-left:1.1rem}
.tier-strong{background:#dcfce7}
.tier-credible{background:#ecfccb}
.tier-stretch{background:#fef9c3}
.tier-unsuitable{background:#fee2e2}
.tier-unevaluated{background:#f3f4f6;color:#6b7280}
.badge.warn{display:inline-block;background:#b45309;color:#fff;border-radius:.25rem;
            padding:.05rem .35rem;margin:.15rem 0;font-weight:600}
.cite{font-size:.82rem;margin:.15rem 0;padding-left:.5rem;border-left:2px solid #d1d5db}
.cite .src{display:inline-block;background:#e5e7eb;border-radius:.2rem;
           padding:0 .3rem;margin-right:.3rem;font-size:.75rem;color:#374151}
.gap{font-size:.82rem;color:#b91c1c}
.gap.veto{font-weight:600}
.match-panel span{white-space:nowrap}
"""


@dataclass(frozen=True)
class JobView:
    """One job, flattened for display.

    ``verdict_reasons`` and ``evidence`` carry the *why* behind
    ``verdict``. A verdict with no stated reason is not reviewable, so the
    renderer never shows a verdict without them.

    The ``match_*`` fields carry the match assessment. They default to the
    explicit "not yet evaluated" state rather than to a blank or a zero, so an
    unassessed job can never be mistaken for a job scoring badly. A match
    result reorders and explains; it never excludes - see
    :mod:`app.jobs.match`.
    """

    job_id: str
    title: str
    company: str
    location: str
    url: str
    verdict: str
    verdict_reasons: List[str]
    evidence: List[str]
    flags: List[str]
    sources: List[str]
    source_urls: List[str]
    first_seen: str
    last_seen: str
    posted_date: str
    description: str
    description_complete: bool
    possible_duplicate: bool
    match_explanation: str
    application_status: str
    #: The source's own list of accepted countries, when it publishes one.
    #: Kept verbatim so coverage can be reported from the source's enumeration
    #: rather than from the eligibility verdict, which weighs other signals too.
    country: str = ""

    # --- match assessment -------------------------------------------
    match_tier: str = "not_yet_evaluated"
    match_score: Optional[float] = None
    match_confidence: str = "insufficient"
    match_evidence: List[Dict[str, str]] = field(default_factory=list)
    match_missing: List[str] = field(default_factory=list)
    match_concerns: List[str] = field(default_factory=list)
    match_insufficient_reason: str = ""
    match_vetoes: List[str] = field(default_factory=list)
    #: Was a match result supplied at all? Distinguishes "nobody ran an
    #: assessment" from "an assessment ran and found the evidence too thin to
    #: support a tier". Those are different facts and must not render alike.
    match_present: bool = False

    @property
    def verdict_class(self) -> str:
        return _VERDICT_CLASS.get(self.verdict, "")

    @property
    def match_class(self) -> str:
        return _TIER_CLASS.get(self.match_tier, "")

    @property
    def match_evaluated(self) -> bool:
        """Was this job actually assessed, as opposed to left unevaluated?"""
        return self.match_tier != "not_yet_evaluated"

    @property
    def match_uncertain(self) -> bool:
        """Assessed, but not firmly enough to lean on.

        True for a thin-evidence result, and for an assessment that ran but
        declined to assert a tier. ``not yet evaluated`` is deliberately not
        uncertain - it is *absent*, and claiming uncertainty would imply a
        conclusion was reached.
        """
        if not self.match_present:
            return False
        return (not self.match_evaluated) or self.match_confidence in (
            "low", "insufficient",
        )


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _flag_job(
    job: Mapping[str, Any],
    verdict: EligibilityVerdict,
    *,
    candidate_country: str = "KE",
) -> List[str]:
    """Deterministic review flags. No model involvement.

    Each flag is a fact about the *posting* a reviewer would otherwise have to
    derive by reading the row: an unstated salary, a description that is only a
    card snippet, and so on.

    Eligibility ``notes`` are deliberately **not** folded in. They are caveats
    attached to the verdict - "confirm with the employer whether Kenya-based
    applicants are considered" - and the verdict cell already shows them.
    Repeating them here made every caveat appear twice in the same row.
    """
    flags: List[str] = []
    if job.get("salary_min") is None and job.get("salary_max") is None:
        flags.append("no salary stated")
    if job.get("description_complete") is False:
        flags.append("description is a card snippet")
    if job.get("deadline"):
        flags.append(f"deadline {job['deadline']}")
    if job.get("posted_date") is None and job.get("posted_raw"):
        flags.append(f"posted date unparsed ({job['posted_raw']})")
    flags.extend(verdict.flags)
    for note in verdict.notes:
        if note.startswith("geography conflict"):
            flags.append(note)
    return flags


#: Phrases a source uses to claim unrestricted remote availability.
_GLOBAL_CLAIMS = ("anywhere in the world", "worldwide", "globally", "anywhere")


#: Display names for the ISO codes the eligibility policy uses, so a flag
#: reads "omits Kenya" rather than "omits KE". Keys are lowercase because
#: lookups casefold the code.
_COUNTRY_NAMES = {"ke": "Kenya"}


def _display_country(code: str) -> str:
    return _COUNTRY_NAMES.get(code.casefold(), code)


def _as_job(payload: Mapping[str, Any]) -> Job:
    """Rebuild a :class:`Job` from its stored JSON form.

    The store persists jobs as plain dictionaries, but ``evaluate_eligibility``
    reads attributes off a ``Job``. Round-tripping here rather than duplicating
    the eligibility logic keeps a single source of truth for what a job *is*.

    ``remote_status`` needs coercing back to its enum: JSON turns it back into a
    bare string, and the eligibility policy compares it against enum members.
    """
    fields = {
        name: value
        for name, value in payload.items()
        if name in Job.__dataclass_fields__
    }
    status = fields.get("remote_status")
    if isinstance(status, str):
        try:
            fields["remote_status"] = RemoteStatus(status)
        except ValueError:
            # An unrecognised status is data we did not anticipate, not a crash:
            # leave it as unknown rather than rejecting the whole job.
            fields["remote_status"] = RemoteStatus.UNKNOWN
    return Job(**fields)


def geography_conflict(
    job: Mapping[str, Any], *, candidate_country: str = "KE"
) -> str:
    """Describe a country list that omits the candidate despite a global claim.

    A thin reporting helper only. The *decision* belongs to
    :mod:`app.jobs.eligibility`, where an explicit country list is
    authoritative, so this never changes a verdict. It exists so a reader can
    see at a glance that a posting's label and its own enumeration disagree.
    """
    from app.jobs.eligibility import _enumerated_countries, _listed_in

    region = str(job.get("region") or "")
    if not _enumerated_countries(job):
        return ""
    if not any(claim in region.casefold() for claim in _GLOBAL_CLAIMS):
        return ""
    if _listed_in(job, candidate_country):
        return ""
    return f"source states worldwide, but its country list omits {_display_country(candidate_country)}"


def _match_fields(match: Any) -> Dict[str, Any]:
    """Map a :class:`~app.jobs.match.MatchResult` onto view fields.

    Accepts a ``MatchResult`` or its ``to_dict()`` output. Anything missing or
    unrecognised degrades to the explicit ``not_yet_evaluated`` state - never
    to a zero score, which would read as "scored and found wanting".
    """
    empty = {
        "match_tier": "not_yet_evaluated",
        "match_score": None,
        "match_confidence": "insufficient",
        "match_evidence": [],
        "match_missing": [],
        "match_concerns": [],
        "match_insufficient_reason": "",
        "match_vetoes": [],
    }
    if match is None:
        return empty

    if isinstance(match, Mapping):
        data = match
    elif hasattr(match, "to_dict"):
        # Preferred over reading __dict__: to_dict() is the object's own
        # contract and already unwraps enum members to their values.
        data = match.to_dict()
    else:
        data = getattr(match, "__dict__", None)
    if not isinstance(data, Mapping):
        return empty

    # `MatchTier` and `Confidence` are str-mixin enums, and str() on those
    # yields "MatchTier.UNSUITABLE", not "unsuitable". Unwrapping .value is what
    # keeps a real MatchResult from being degraded to not_yet_evaluated. Dict
    # inputs carry plain strings, so both paths must work.
    tier_raw = data.get("tier")
    tier = str(getattr(tier_raw, "value", tier_raw) or "not_yet_evaluated")
    if tier not in _TIER_ORDER:
        # An unrecognised tier is data we did not anticipate. Reporting it as
        # a real tier would be worse than saying it is not yet evaluated.
        tier = "not_yet_evaluated"

    score = data.get("score")
    try:
        score = None if score is None else float(score)
    except (TypeError, ValueError):
        score = None

    confidence_raw = data.get("confidence")
    confidence = str(
        getattr(confidence_raw, "value", confidence_raw) or "insufficient"
    )

    evidence = []
    for item in data.get("evidence") or []:
        if isinstance(item, Mapping):
            evidence.append({
                "claim": _text(item.get("claim")),
                "source": _text(item.get("source")),
                "quote": _text(item.get("quote")),
            })
        elif hasattr(item, "to_dict"):
            # EvidenceItem instances from a real MatchResult.
            payload = item.to_dict()
            evidence.append({
                "claim": _text(payload.get("claim")),
                "source": _text(payload.get("source")),
                "quote": _text(payload.get("quote")),
            })

    return {
        "match_tier": tier,
        "match_score": score,
        "match_confidence": confidence,
        "match_evidence": evidence,
        "match_missing": [_text(g) for g in (data.get("missing_requirements") or [])],
        "match_concerns": [_text(c) for c in (data.get("concerns") or [])],
        "match_insufficient_reason": _text(data.get("insufficient_reason")),
        "match_vetoes": [_text(v) for v in (data.get("vetoes") or [])],
        "match_present": True,
    }


def build_view(
    record: Mapping[str, Any],
    *,
    candidate_country: str = "KE",
    match_explanation: Optional[str] = None,
    application_status: Optional[str] = None,
    match: Any = None,
) -> JobView:
    """Flatten one stored record into a :class:`JobView`.

    ``match_explanation`` and ``application_status`` are supplied by the
    caller. ``match`` is a :class:`~app.jobs.match.MatchResult` (or its dict
    form) when an assessment exists; without one the job reports itself as
    ``not_yet_evaluated``, which is a real state rather than a blank.
    """
    job: Mapping[str, Any] = record.get("job", {}) or {}
    verdict = evaluate_eligibility(
        _as_job(job), candidate_country=candidate_country
    )
    sources = [str(entry.get("source", "")) for entry in record.get("sources", [])]
    source_urls = [str(entry.get("url", "")) for entry in record.get("sources", [])]

    return JobView(
        job_id=_text(record.get("job_id")),
        title=_text(job.get("title")),
        company=_text(job.get("company")),
        location=_text(job.get("location")) or "—",
        url=_text(job.get("url")),
        verdict=verdict.verdict,
        verdict_reasons=list(verdict.reasons) + list(verdict.notes),
        evidence=list(verdict.evidence_quotes),
        flags=_flag_job(job, verdict, candidate_country=candidate_country),
        sources=[name for name in sources if name],
        source_urls=[url for url in source_urls if url],
        first_seen=_text(record.get("first_seen")),
        last_seen=_text(record.get("last_seen")),
        posted_date=_text(job.get("posted_date")) or NOT_EVALUATED,
        description=_text(job.get("description")),
        description_complete=bool(job.get("description_complete")),
        possible_duplicate=bool(record.get("possible_duplicate")),
        match_explanation=match_explanation or NOT_EVALUATED,
        application_status=application_status or STATUS_PLACEHOLDER,
        country=_text(job.get("country")),
        **_match_fields(match),
    )


def _sorted(views: Sequence[JobView], sort: str) -> List[JobView]:
    """Stable, deterministic ordering. Ties fall back to company then title."""
    keys = {
        "company": lambda v: (v.company.casefold(), v.title.casefold()),
        "date": lambda v: (v.last_seen or "", v.company.casefold()),
        "verdict": lambda v: (v.verdict, v.company.casefold(), v.title.casefold()),
        "title": lambda v: (v.title.casefold(), v.company.casefold()),
        # Tiers order best-first; unevaluated jobs sort last rather than being
        # dropped, so an unassessed job stays visible.
        "match": lambda v: (
            _TIER_ORDER.get(v.match_tier, 9),
            -(v.match_score or -1.0),
            v.company.casefold(), v.title.casefold(),
        ),
    }
    return sorted(views, key=keys.get(sort, keys["company"]))


def _filtered(
    views: Iterable[JobView],
    *,
    verdict: Optional[str] = None,
    source: Optional[str] = None,
    query: Optional[str] = None,
    kenya_eligible: bool = False,
    match_tier: Optional[str] = None,
    status: Optional[str] = None,
    posted_after: Optional[str] = None,
    uncertain: bool = False,
) -> List[JobView]:
    """Apply filters at render time - there is no client-side scripting.

    ``kenya_eligible`` narrows to jobs the eligibility policy judged eligible.
    It is a *filter*, not a default: nothing is hidden unless the reader asks
    for it, so a source that returns nothing eligible reads as empty coverage
    rather than as a broken scraper.
    """
    result = []
    for view in views:
        if verdict and view.verdict != verdict:
            continue
        if source and source not in view.sources:
            continue
        if kenya_eligible and view.verdict != "eligible":
            continue
        if match_tier and view.match_tier != match_tier:
            continue
        if status and view.application_status != status:
            continue
        if uncertain and not view.match_uncertain:
            continue
        if posted_after and (view.posted_date or "") < posted_after:
            continue
        if query:
            haystack = " ".join(
                [view.title, view.company, view.location, view.description]
            ).casefold()
            if query.casefold() not in haystack:
                continue
        result.append(view)
    return result


def coverage_of(
    views: Sequence[JobView],
    runs: Sequence[Mapping[str, Any]],
    *,
    candidate_country: str = "KE",
) -> Dict[str, int]:
    """Source coverage: what arrived, what was stored, and how it judged.

    Computed from the jobs on screen plus the run ledger, so the reader can
    tell "nothing was eligible" apart from "nothing was collected".

    ``country_named`` counts jobs whose *published* country list names the
    candidate's country. It is reported separately from ``eligible`` because
    the two can disagree: the eligibility policy weighs worldwide wording and
    body text, while a country list is the source's own explicit enumeration.
    Showing only the verdict would hide that disagreement.
    """
    name = _display_country(candidate_country)
    totals = {
        "fetched": 0, "stored": 0, "rejected": 0,
        "duplicates": 0, "quarantined": 0,
        "eligible": 0, "not_eligible": 0, "unknown": 0,
        "country_named": 0, "country_lists": 0, "conflicts": 0,
        "no_active_sources": 0, "runs": len(runs),
    }
    for run in runs:
        totals["fetched"] += int(run.get("totals", {}).get("fetched", 0) or 0)
        totals["stored"] += int(run.get("totals", {}).get("stored", 0) or 0)
        totals["rejected"] += int(run.get("totals", {}).get("rejected", 0) or 0)
        if run.get("no_active_sources"):
            totals["no_active_sources"] += 1
        for outcome in run.get("sources", []):
            totals["duplicates"] += int(outcome.get("possible_duplicates", 0) or 0)
            totals["quarantined"] += int(outcome.get("rejected", 0) or 0)
    for view in views:
        if view.verdict in ("eligible", "not_eligible", "unknown"):
            totals[view.verdict] += 1
        if any("contested reading" in flag or "geography conflict" in flag
               for flag in view.flags):
            totals["conflicts"] += 1
        if view.country:
            totals["country_lists"] += 1
            if _country_list_names(view, name):
                totals["country_named"] += 1
    return totals


def _country_list_names(view: "JobView", name: str) -> bool:
    """Does this view's published country list name ``name``?"""
    haystack = f"{view.location} {view.country}"
    return name.casefold() in haystack.casefold()


def _match_panel(views: Sequence[JobView]) -> str:
    """A strip showing how many jobs were actually assessed, and how well.

    Reported separately from the eligibility counts because the two answer
    different questions: eligibility asks whether the candidate may take a
    role, the match asks how well they fit. A source can be full of
    ineligible roles that were never match-assessed at all, and collapsing
    those would make "not assessed" look like "scored and rejected".
    """
    by_tier: Dict[str, int] = {}
    uncertain = 0
    unranked = 0
    for view in views:
        by_tier[view.match_tier] = by_tier.get(view.match_tier, 0) + 1
        if view.match_uncertain:
            uncertain += 1
        if view.match_evaluated and view.match_score is None:
            unranked += 1

    assessed = sum(count for tier, count in by_tier.items()
                   if tier != "not_yet_evaluated")
    cells = [(tier, by_tier.get(tier, 0)) for tier in _TIER_ORDER]
    cells.append(("uncertain of those", uncertain))
    cells.append(("no score", unranked))

    parts = "".join(
        f"<span>{html.escape(label)}: <strong>{count}</strong></span>"
        for label, count in cells
    )

    note = ""
    if views and assessed == 0:
        note = (
            '<p class="notice">No match assessment has been run for any of these '
            "jobs. Every tier reads &quot;not yet evaluated&quot;; no score has "
            "been calculated and nothing has been ranked. This is missing "
            "analysis, not a poor result.</p>"
        )
    return f"<div class='coverage match-panel'>{parts}</div>{note}"


def _coverage_panel(totals: Mapping[str, int], views: Sequence[JobView]) -> str:
    """The coverage strip, plus honest notes where the numbers disagree."""
    cells = (
        ("fetched", totals["fetched"]),
        ("stored", totals["stored"]),
        ("eligible", totals["eligible"]),
        ("not eligible", totals["not_eligible"]),
        ("unknown", totals["unknown"]),
        ("duplicates", totals["duplicates"]),
        ("quarantined", totals["quarantined"]),
        ("country lists published", totals["country_lists"]),
        ("naming Kenya", totals["country_named"]),
        ("geography conflicts", totals["conflicts"]),
    )
    parts = "".join(
        f"<span>{html.escape(label)}: <strong>{value}</strong></span>"
        for label, value in cells
    )

    notes: List[str] = []
    if totals["no_active_sources"]:
        # Checked first and exclusive: "nothing was consulted" makes every
        # other observation vacuous, and stacking them reads as noise.
        notes.append(
            "No source was consulted in the most recent run. Nothing was "
            "collected; this is not an empty result."
        )
    elif not views:
        # Distinct from "jobs arrived but none are eligible": here the source
        # returned nothing at all, which is a collection fact, not a coverage
        # limit.
        notes.append(
            "The source returned no jobs in this run. Nothing was collected, "
            "so no eligibility could be assessed."
        )
    else:
        if totals["eligible"] == 0:
            notes.append(
                "0 Kenya-eligible jobs in this feed; the feed currently "
                f"excludes Kenya. {len(views)} job(s) were collected and judged "
                f"{totals['not_eligible']} not eligible / {totals['unknown']} "
                "unknown. This is a coverage limit, not a collection failure."
            )
        if totals["unknown"] and not totals["eligible"] and not totals["not_eligible"]:
            notes.append(
                "Eligibility could not be determined for any listing: the "
                "postings state no usable location information."
            )
        if totals["country_lists"] and totals["country_named"] == 0:
            # Only asserted when listings actually publish an enumeration.
            # Saying "no listing names Kenya" when none publishes a country
            # list at all would be true of a field that was never filled in.
            contested = (
                f" {totals['eligible']} were nonetheless judged eligible; "
                "treat those verdicts as contested."
                if totals["eligible"] else ""
            )
            notes.append(
                f"No published country list names Kenya "
                f"({totals['country_lists']} listing(s) publish a list). "
                f"The feed currently excludes Kenya.{contested}"
            )

    # Coverage notes and the match panel are independent, so the match note
    # is rendered alongside rather than inside this panel's note list.
    match_note = _match_panel(views)
    rendered = "".join(f'<p class="notice">{html.escape(n)}</p>' for n in notes)
    return f"<div class='coverage'>{parts}</div>{match_note}{rendered}"


def _attribution_footer(views: Sequence[JobView], attributions: Mapping[str, str]) -> str:
    """Credit sources whose permission requires attribution.

    We Work Remotely grants use of its feed on the condition that listings are
    attributed with a link back. That is enforced here rather than left as a
    convention, because losing attribution means losing access.
    """
    present = [
        name for name in attributions
        if any(name in view.sources for view in views)
    ]
    if not present:
        return ""
    lines = [
        "<div class='attribution'><strong>Attribution required</strong><ul>"
    ]
    for name in sorted(present):
        url = html.escape(attributions[name], quote=True)
        label = html.escape(name)
        lines.append(
            f"<li>Listings from <a href='{url}' rel='noopener'>{label}</a> "
            f"are provided under their feed's terms. Please link back to the "
            f"original posting.</li>"
        )
    lines.append("</ul></div>")
    return "".join(lines)


def _cell(text: str, *, css: str = "") -> str:
    """A cell of escaped plain text, or a placeholder dash when empty."""
    body = html.escape(text) if text else '<span class="badge">—</span>'
    attribute = f' class="{css}"' if css else ""
    return f"<td{attribute}>{body}</td>"


def _cell_markup(markup: str, *, css: str = "") -> str:
    """A cell whose content is *already* escaped markup.

    Distinct from :func:`_cell` on purpose. Composed content - flag chips, a
    source list with a trailing chip - is built by escaping each piece and then
    joining the pieces with tags. Running that result back through
    ``_cell`` would escape the tags too and render them as visible text.
    Callers of this function are responsible for escaping every value that
    reaches it.
    """
    attribute = f' class="{css}"' if css else ""
    return f"<td{attribute}>{markup}</td>"


#: How a tier reads to a person. The machine token stays visible alongside it
#: so a reader can match what they see against a filter or a stored value.
_TIER_LABEL = {
    "strong_match": "strong match",
    "credible_match": "credible match",
    "stretch": "stretch",
    "unsuitable": "unsuitable",
    "not_yet_evaluated": "not yet evaluated",
}


def _match_cell(view: JobView) -> str:
    """Render the match assessment for one row.

    Written so the three states cannot be confused:

    * **not yet evaluated** - no assessment exists. Says so; shows no score.
    * **evaluated, low confidence or thin evidence** - shows the tier but
      marks it uncertain and prints why, so a guess is never read as a verdict.
    * **evaluated with evidence** - shows tier, score, confidence, evidence
      with its source, missing requirements and concerns.

    Every value is escaped. Evidence quotes come from third-party job postings
    and are untrusted text.
    """
    if not view.match_present:
        # Nothing was ever run. This is absence of analysis, not a poor result.
        return (
            f'<td class="{view.match_class}">{html.escape("not yet evaluated")}'
            '<div class="badge">no match assessment has been run for this job'
            "</div></td>"
        )

    score = (
        "no score calculated"
        if view.match_score is None
        else f"score {view.match_score}"
    )
    parts = [
        f'<td class="{view.match_class}">',
        html.escape(_TIER_LABEL.get(view.match_tier, view.match_tier)),
        f'<div class="badge">tier: {html.escape(view.match_tier)}</div>',
        f'<div class="badge">{html.escape(score)} &middot; confidence: '
        f"{html.escape(view.match_confidence)}</div>",
    ]

    if view.match_uncertain:
        # Say plainly that this result should not be leaned on.
        parts.append('<div class="badge warn">uncertain</div>')

    if view.match_insufficient_reason:
        parts.append(
            f'<div class="badge">{html.escape(view.match_insufficient_reason)}</div>'
        )

    for item in view.match_evidence:
        claim = html.escape(item.get("claim", ""))
        source = html.escape(item.get("source", "") or "unknown source")
        quote = html.escape(item.get("quote", ""))
        parts.append(
            f'<div class="cite"><span class="src">{source}</span> '
            f"{claim} &mdash; &ldquo;{quote}&rdquo;</div>"
        )

    for gap in view.match_missing:
        parts.append(f'<div class="gap">missing: {html.escape(gap)}</div>')

    for concern in view.match_concerns:
        parts.append(f'<div class="gap">concern: {html.escape(concern)}</div>')

    # An exclusion is only ever reported, never acted on here.
    for veto in view.match_vetoes:
        parts.append(
            f'<div class="gap veto">excluded by eligibility: {html.escape(veto)}</div>'
        )

    parts.append("</td>")
    return "".join(parts)


def render_row(view: JobView) -> str:
    """One ``<tr>``. Split out so the row markup is directly testable."""
    # ";" so adjacent reasons do not read as one run-on sentence.
    verdict_detail = "; ".join(view.verdict_reasons) or "no reason recorded"
    evidence = (
        f'<div class="badge">evidence: {html.escape(" | ".join(view.evidence))}</div>'
        if view.evidence
        else ""
    )
    flags = "".join(
        f'<span class="flag">{html.escape(flag)}</span>' for flag in view.flags
    )
    sources = ", ".join(html.escape(name) for name in view.sources)
    chips = []
    if view.possible_duplicate:
        chips.append('<span class="flag">possible duplicate</span>')
    if view.description_complete:
        chips.append('<span class="flag">full description</span>')
    sources_cell = sources or ""
    if chips:
        sources_cell = (sources_cell + " " if sources_cell else "") + " ".join(chips)

    link = (
        f'<a href="{html.escape(view.url, quote=True)}">{html.escape(view.title)}</a>'
        if view.url
        else html.escape(view.title)
    )

    return (
        "<tr>"
        + f'<td class="{view.verdict_class}">{html.escape(view.verdict)}'
        + f'<div class="badge">{html.escape(verdict_detail)}</div>{evidence}</td>'
        + f"<td>{link}</td>"
        + _cell(view.company)
        + _cell(view.location)
        + _cell_markup(sources_cell or '<span class="badge">—</span>')
        + _cell_markup(flags or '<span class="badge">—</span>')
        + _match_cell(view)
        + _cell(view.application_status)
        + _cell(view.posted_date)
        + _cell(view.first_seen)
        + _cell(view.last_seen)
        + "</tr>"
    )


_HEADERS = (
    "Eligibility", "Role", "Company", "Location", "Sources", "Flags",
    "Match", "Status", "Posted", "First seen", "Last seen",
)


def render_dashboard_html(
    views: Sequence[JobView],
    *,
    generated_at: Optional[str] = None,
    title: str = "Job review",
    filters: Optional[Mapping[str, Any]] = None,
    runs: Sequence[Mapping[str, Any]] = (),
    attributions: Optional[Mapping[str, str]] = None,
) -> str:
    """Render the review dashboard. Pure function of its inputs."""
    stamp = generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    active = {key: value for key, value in (filters or {}).items() if value}

    counts: Dict[str, int] = {}
    for view in views:
        counts[view.verdict] = counts.get(view.verdict, 0) + 1

    summary = "".join(
        f"<span>{html.escape(name)}: {count}</span>"
        for name, count in sorted(counts.items())
    ) or "<span>no jobs</span>"

    totals = coverage_of(views, runs)
    coverage = _coverage_panel(totals, views)
    attribution = _attribution_footer(views, attributions or {})
    match_panel = _match_panel(views)

    if views:
        body = "".join(render_row(view) for view in views)
        table = (
            "<table><caption>Jobs</caption><thead><tr>"
            + "".join(f"<th>{html.escape(header)}</th>" for header in _HEADERS)
            + f"</tr></thead><tbody>{body}</tbody></table>"
        )
    else:
        table = '<div class="empty">No jobs match the current filters.</div>'

    # A <p> may not contain a <div>; the filters line is a sibling, not a child.
    applied = (
        f'<p class="meta">filters: {html.escape(json.dumps(active, sort_keys=True))}</p>'
        if active
        else ""
    )

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f"<p class='meta'>Generated {html.escape(stamp)} &middot; read-only</p>"
        f"{applied}"
        f"<div class='summary'>{summary}</div>"
        f"{coverage}"
        f"{table}"
        f"{attribution}</body></html>"
    )


def render_dashboard_file(
    store: JobStore,
    output_path,
    *,
    generated_at: Optional[str] = None,
    candidate_country: str = "KE",
    verdict: Optional[str] = None,
    source: Optional[str] = None,
    query: Optional[str] = None,
    sort: str = "company",
    match_explanations: Optional[Mapping[str, str]] = None,
    status_log: Optional[StatusLog] = None,
    kenya_eligible: bool = False,
    attributions: Optional[Mapping[str, str]] = None,
    matches: Optional[Mapping[str, Any]] = None,
    match_tier: Optional[str] = None,
    uncertain: bool = False,
    posted_after: Optional[str] = None,
) -> str:
    """Read the store, render, and write the dashboard. Returns the path written.

    This is the only function here that touches the filesystem, and it only
    ever *writes the report*. The store is read, never modified.

    ``status_log`` is an optional :class:`~app.jobs.status.StatusLog`. Supplying
    it replaces the status placeholder with the candidate's actual recorded
    disposition.

    ``matches`` maps job id to a :class:`~app.jobs.match.MatchResult` (or its
    dict form). A job with no entry reports ``not_yet_evaluated`` - a real
    state, not a blank, and never a zero score.
    """
    def status_of(job_id: str) -> Optional[str]:
        if status_log is None:
            return None
        try:
            return status_log.current(job_id).value
        except StatusError:
            # A status problem must not stop the dashboard rendering. The
            # column falls back to its placeholder rather than the report
            # failing wholesale. Only StatusError is caught - an unexpected
            # failure should surface rather than be silently reported as
            # "no status recorded".
            return None

    views = [
        build_view(
            record,
            candidate_country=candidate_country,
            match_explanation=(match_explanations or {}).get(str(record.get("job_id"))),
            application_status=status_of(str(record.get("job_id"))),
            match=(matches or {}).get(str(record.get("job_id"))),
        )
        for record in store.load_jobs()
    ]
    selected = _filtered(views, verdict=verdict, source=source, query=query,
                          kenya_eligible=kenya_eligible, match_tier=match_tier,
                          uncertain=uncertain, posted_after=posted_after)
    ordered = _sorted(selected, sort)
    applied = {"verdict": verdict, "source": source, "query": query,
               "sort": sort, "kenya_eligible": kenya_eligible or None,
               "match_tier": match_tier, "uncertain": uncertain or None,
               "posted_after": posted_after}

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        render_dashboard_html(
            ordered,
            generated_at=generated_at,
            filters={key: value for key, value in applied.items() if value},
            runs=store.load_runs(),
            attributions=attributions,
        ),
        encoding="utf-8",
    )
    return str(target)
