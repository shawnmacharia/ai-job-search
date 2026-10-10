"""One read-only report for reviewing every persisted job and every source.

What this is
------------
A single page that answers the questions you actually have when you sit down to
review jobs, without making a request and without changing anything:

- *What is new and worth acting on today?*
- *What is waiting for a decision from me?*
- *Is every source healthy, and is any of them refusing to run?*
- *What did I decide about this, when, and why?*
- *Why is this job being shown to me as it is?*

Read-only by construction
-------------------------
Everything below reads. Nothing here writes to jobs, decisions, freshness,
ledgers or run records, and nothing opens a socket. The report is a *view* over
state that other components own; if it disagreed with the store, the store would
still be right.

Why one report rather than the existing dashboard
-------------------------------------------------
The dashboard is a table of every job, which is the right shape for browsing and
the wrong shape for deciding what to do this morning. This report is organised
by *question* rather than by row, so the actionable set is separated from the
queues and neither is buried under the other.

The distinctions this report must never collapse
-------------------------------------------------
These look similar in a summary and mean different things. Merging any of them
turns a fact about the tool into a fact about the job market, or worse, hides a
job:

- **never assessed** vs **assessed, evidence too thin** vs **assessed, with
  evidence** - "not yet evaluated" is missing analysis, not a poor result
- **ineligible** vs **source failure** vs **successful zero-result** - the first
  is about a posting, the other two are about us failing to read one
- **stale** vs **expired** - different ages, and expired means "gone long ago",
  not "impossible"
- **unknown** vs **no evidence** - "we could not tell" is not "no"

Score and tier never hide a job. They reorder and explain. If a job is eligible
and not yet reviewed, it appears in the actionable queue regardless of whether
any assessment exists, and regardless of what that assessment said.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from app.jobs.freshness import FreshnessLedger, FreshnessState
from app.jobs.status import DEFAULT_STATUS, ReviewStatus, StatusError, StatusLog
from app.jobs.store import JobStore
from app.reporting.jobs import JobView, build_view

#: Sources this report knows how to describe. A source present in the store but
#: absent here is still reported - this list orders the display and marks which
#: ones the project considers live, it does not filter.
KNOWN_SOURCES = ("weworkremotely", "myjobmag.co.ke", "remotive.com")

#: Purposes each source uses in the shared attempt ledger. Used to keep one
#: source's traffic out of another's count; the ledger file is shared.
SOURCE_PURPOSE = {
    "weworkremotely": "feed",
    "myjobmag.co.ke": "feed",
    "remotive.com": "api",
}

#: Substrings that identify a refusal caused by a daily budget rather than by a
#: network or server fault. Matched against the recorded error text, so this is
#: a *reported inference*, labelled as such in the output rather than presented
#: as a first-class fact.
_REFUSAL_MARKERS = ("daily request limit", "could not be read")


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _e(value: Any) -> str:
    """Escape one value for HTML. Every rendered string goes through this."""
    return html.escape(_text(value), quote=True)


# ----------------------------------------------------------------------
# rows
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SourceHealthRow:
    """One source's health, from its own records only.

    ``last_run`` describes the most recent run; ``recent_failures`` counts
    consecutive failures; ``zero_result`` is set only when the source
    *succeeded* and returned nothing, which is a different fact from failing and
    is kept separate on the page for exactly that reason.
    """

    name: str
    access_level: str = "not recorded"
    attribution: str = ""
    fetched: int = 0
    stored: int = 0
    eligible: int = 0
    ineligible: int = 0
    unknown: int = 0
    last_run_state: str = "never run"
    last_run_at: str = ""
    consecutive_failures: int = 0
    zero_result: bool = False
    last_error: str = ""
    requests_last_24h: int = 0
    last_request_at: str = ""
    budget_refused: bool = False
    skipped_reasons: Tuple[str, ...] = ()

    @property
    def total_jobs(self) -> int:
        return self.eligible + self.ineligible + self.unknown


@dataclass(frozen=True)
class DecisionMemory:
    """What the candidate has recorded about one job."""

    job_id: str
    status: str
    last_note: str = ""
    decided_at: str = ""
    history: Tuple[Tuple[str, str], ...] = ()

    @property
    def reviewed(self) -> bool:
        return self.status != DEFAULT_STATUS.value


@dataclass(frozen=True)
class ReviewReport:
    """The whole report, computed. Rendering is a pure function of this."""

    generated_at: str
    actionable: Tuple[JobView, ...] = ()
    eligible_unreviewed: Tuple[JobView, ...] = ()
    uncertain: Tuple[JobView, ...] = ()
    contested: Tuple[JobView, ...] = ()
    duplicates: Tuple[JobView, ...] = ()
    stale: Tuple[JobView, ...] = ()
    expired: Tuple[JobView, ...] = ()
    needs_decision: Tuple[JobView, ...] = ()
    sources: Tuple[SourceHealthRow, ...] = ()
    decisions: Mapping[str, DecisionMemory] = field(default_factory=dict)
    total_jobs: int = 0
    #: Problems that must be shown rather than swallowed. A corrupt store is
    #: reported on the page; it is never dropped so the numbers look clean.
    errors: Tuple[str, ...] = ()
    #: The daily working queue after filters, with a reason per entry.
    queue: Tuple[Tuple[JobView, str], ...] = ()
    #: Filtered views, kept so the report can be re-rendered without re-reading.
    filtered: Tuple[JobView, ...] = ()
    filters: Mapping[str, Any] = field(default_factory=dict)
    #: Category counts over the *unfiltered* corpus, so a filter never makes a
    #: category look empty when it is not.
    categories: Mapping[str, int] = field(default_factory=dict)


# ----------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------


def _decision_memory(status_log: Optional[StatusLog], job_id: str) -> DecisionMemory:
    if status_log is None:
        return DecisionMemory(job_id=job_id, status=DEFAULT_STATUS.value)
    try:
        events = status_log.history(job_id)
    except StatusError:
        # A status problem must not stop the report rendering. It is recorded as
        # "unreadable" rather than reported as "no decisions", because those
        # are different facts and only one of them is reassuring.
        return DecisionMemory(job_id=job_id, status="unreadable")
    if not events:
        return DecisionMemory(job_id=job_id, status=DEFAULT_STATUS.value)
    latest = events[-1]
    return DecisionMemory(
        job_id=job_id,
        status=_text(latest.status),
        last_note=_text(latest.note),
        decided_at=_text(latest.at),
        history=tuple((_text(e.status), _text(e.at)) for e in events),
    )


def _run_state(source: str, runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The latest run facts for one source, from ``runs.jsonl`` only."""
    state: Dict[str, Any] = {
        "state": "never run",
        "at": "",
        "fetched": 0,
        "stored": 0,
        "error": "",
        "failures": 0,
        "zero_result": False,
        "skips": (),
    }
    for run in runs:
        finished = _text(run.get("finished_at") or run.get("started_at"))
        for entry in run.get("skipped", []) or []:
            if _text(entry.get("name")) == source:
                state["skips"] = tuple(state["skips"]) + (
                    _text(entry.get("reason") or entry.get("code")),
                )
        for outcome in run.get("sources", []) or []:
            if _text(outcome.get("name")) != source:
                continue
            state["at"] = finished
            state["fetched"] = int(outcome.get("fetched", 0) or 0)
            state["stored"] = int(outcome.get("stored", 0) or 0)
            state["error"] = _text(outcome.get("error"))
            if outcome.get("ok"):
                state["failures"] = 0
                # A successful run that returned nothing is not a failure and
                # not an empty market we verified - it is the weakest signal
                # there is, and it gets its own label.
                state["zero_result"] = state["fetched"] == 0
                state["state"] = "zero result" if state["fetched"] == 0 else "ok"
            else:
                state["failures"] = int(state["failures"]) + 1
                state["zero_result"] = False
                state["state"] = "failed"
    return state


def _request_state(ledger_holder: Any, source: str) -> Dict[str, Any]:
    """Requests in the last 24h for this source, from the attempt ledger."""
    purpose = SOURCE_PURPOSE.get(source)
    out = {"count": 0, "last": ""}
    if purpose is None or ledger_holder is None:
        return out
    try:
        attempts = ledger_holder.prior_attempts()
    except Exception:  # noqa: BLE001 - never let the ledger break the report
        return out
    now = datetime.now(timezone.utc)
    stamps: List[datetime] = []
    for attempt in attempts:
        if _text(getattr(attempt, "purpose", "")) != purpose:
            # Another source's traffic must not appear as this source's.
            continue
        if _text(getattr(attempt, "source", "")) not in (source, ""):
            continue
        raw = _text(getattr(attempt, "at", ""))
        try:
            from app.jobs.freshness import parse_at

            stamp = parse_at(raw)
        except ValueError:
            continue
        stamps.append(stamp)
    recent = [s for s in stamps if (now - s).total_seconds() < 86400]
    out["count"] = len(recent)
    if stamps:
        out["last"] = max(stamps).isoformat(timespec="seconds")
    return out


def _read_freshness(store: JobStore) -> Tuple[Mapping[str, Any], Optional[str]]:
    """Freshness state, plus an error when it cannot be trusted.

    An unreadable observation log must not silently produce "active" for
    everything. If we cannot read the record of what the sources actually
    returned, nothing vouches for these jobs, so they are reported as unknown
    and the problem is surfaced. Reporting them as current would be a claim
    nobody has evidence for.
    """
    path = store.data_dir / "freshness.jsonl"
    if path.exists():
        try:
            for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if line.strip() and not isinstance(json.loads(line), dict):
                    return {}, (
                        f"freshness log line {number} is not an object; jobs are "
                        "reported as unobserved rather than current"
                    )
        except json.JSONDecodeError as exc:
            return {}, (
                f"freshness log is unreadable ({exc.msg}); jobs are reported as "
                "unobserved rather than current"
            )
        except OSError as exc:
            return {}, f"freshness log could not be read ({exc}); jobs are unobserved"
    try:
        return FreshnessLedger(store).evaluate(), None
    except Exception as exc:  # noqa: BLE001
        return {}, f"could not evaluate freshness: {type(exc).__name__}: {exc}"


def _source_rows(
    store: JobStore,
    views: Sequence[JobView],
    runs: Sequence[Mapping[str, Any]],
    ledger: Any,
    attributions: Mapping[str, str],
    decisions_by_source: Mapping[str, Any],
) -> List[SourceHealthRow]:
    rows: List[SourceHealthRow] = []
    present = {s for view in views for s in view.sources}
    ordered = list(KNOWN_SOURCES) + sorted(present - set(KNOWN_SOURCES))

    for name in ordered:
        own = [v for v in views if name in v.sources]
        run = _run_state(name, runs)
        requests = _request_state(ledger, name)
        error = run["error"]
        refused = bool(error) and any(m in error.casefold() for m in _REFUSAL_MARKERS)
        access = decisions_by_source.get(name)
        rows.append(
            SourceHealthRow(
                name=name,
                access_level=getattr(access, "level", None) and _text(
                    getattr(access.level, "value", access.level)
                ) or "not recorded",
                attribution=attributions.get(name, ""),
                fetched=run["fetched"],
                stored=run["stored"],
                eligible=sum(1 for v in own if v.verdict == "eligible"),
                ineligible=sum(1 for v in own if v.verdict == "not_eligible"),
                unknown=sum(1 for v in own if v.verdict == "unknown"),
                last_run_state=run["state"],
                last_run_at=run["at"],
                consecutive_failures=run["failures"],
                zero_result=bool(run["zero_result"]),
                last_error=error,
                requests_last_24h=requests["count"],
                last_request_at=requests["last"],
                budget_refused=refused,
                skipped_reasons=run["skips"],
            )
        )
    return rows


# ----------------------------------------------------------------------
# actionable queue
# ----------------------------------------------------------------------

#: Statuses that close a job out of the queue. ``dismissed`` is the only one:
#: there is no "applied" or "rejected" state in this project, because acting is a
#: hard stop and a status implying an application would be a fiction.
CLOSED_STATUSES = frozenset({ReviewStatus.DISMISSED.value})

#: Statuses that mean the job is being actively pursued. Kept out of
#: "never reviewed" but still out of "needs a decision", because nothing is
#: being asked of the reader.
PURSUING_STATUSES = frozenset({
    ReviewStatus.INTERESTED.value, ReviewStatus.SHORTLISTED.value,
})

#: Tier order for sorting, best first. Unassessed sorts last rather than being
#: dropped: an unassessed job is still a job.
_TIER_SORT = {
    "strong_match": 0, "credible_match": 1, "stretch": 2,
    "unsuitable": 3, "not_yet_evaluated": 4,
}

_FRESHNESS_SORT = {
    FreshnessState.ACTIVE.value: 0,
    FreshnessState.STALE.value: 1,
    FreshnessState.EXPIRED.value: 2,
    "unknown": 3,
}

#: Status order for the queue: least-settled first.
#:
#: The queue exists to surface what still needs a decision, so an unexamined job
#: leads one that is already shortlisted. Ranking the other way would fill the
#: top of the list with jobs the candidate has already triaged, which is the
#: opposite of what a review queue is for. Both decided statuses still appear -
#: they are just lower down than work nobody has looked at.
_STATUS_SORT = {
    ReviewStatus.NEW.value: 0,
    ReviewStatus.REVIEWING.value: 1,
    ReviewStatus.INTERESTED.value: 2,
    ReviewStatus.SHORTLISTED.value: 3,
    ReviewStatus.DISMISSED.value: 4,
}


def _posted_rank(value: str) -> Optional[float]:
    """A posted date as a sortable epoch, or ``None`` when it is not one.

    Descending order on a string is not something a sort key can express, so
    the timestamp is reduced to a number and negated at the call site instead.
    An unparseable or missing date yields ``None`` rather than epoch zero, which
    would sort undated postings as if they were from 1970 and float them to the
    top of a queue.
    """
    text = _text(value).strip()
    if not text:
        return None
    from app.jobs.freshness import parse_at

    try:
        return parse_at(text).timestamp()
    except (ValueError, TypeError):
        return None


def queue_key(view: JobView) -> Tuple[Any, ...]:
    """The total order for the actionable queue.

    Every component is deterministic, and two of them are guarded rather than
    assumed:

    - **score** is consulted only when one was actually calculated. A job with
      no assessment must not sort as though it scored zero, which would quietly
      bury exactly the jobs nobody has looked at yet.
    - **job id** is the final tie-breaker. Without it the order would depend on
      input ordering, and a queue that reshuffles between two identical runs is
      a queue nobody trusts.
    """
    posted = _posted_rank(view.posted_date)
    return (
        _STATUS_SORT.get(view.application_status, 5),
        _FRESHNESS_SORT.get(view.freshness, 4),
        _TIER_SORT.get(view.match_tier, 5),
        # Missing score sorts after any real score, never as zero.
        (1, 0.0) if view.match_score is None else (0, -view.match_score),
        # Undated postings sort last, not as epoch zero.
        (1, 0.0) if posted is None else (0, -posted),
        view.job_id,
    )


def why_actionable(view: JobView) -> str:
    """The one-sentence reason this job is in the queue.

    A queue that does not say why is a list to re-derive by hand. Each clause
    names a condition that was checked, so a reader can tell whether a job
    belongs here without re-reading the policy.
    """
    bits = ["Kenya-eligible"]
    if view.application_status == ReviewStatus.NEW.value:
        bits.append("not yet reviewed")
    elif view.application_status == ReviewStatus.REVIEWING.value:
        bits.append("under review, undecided")
    else:
        bits.append(f"status: {view.application_status}")
    if view.freshness == FreshnessState.ACTIVE.value:
        bits.append("current")
    else:
        bits.append(f"freshness {view.freshness}")
    if not view.match_present:
        bits.append("no match assessment yet - still reviewable")
    elif not view.match_evaluated:
        bits.append("match assessed but evidence thin")
    if view.possible_duplicate:
        bits.append("possible duplicate - check before acting")
    return "; ".join(bits)


def build_actionable(views: Sequence[JobView]) -> List[JobView]:
    """The daily working queue, in a deterministic order.

    Membership: Kenya-eligible, not closed, not expired. A job with no match
    assessment is *in* the queue - missing analysis is not a reason to hide a
    job from someone deciding what to read.
    """
    selected = [
        v for v in views
        if v.verdict == "eligible"
        and v.application_status not in CLOSED_STATUSES
        and v.freshness != FreshnessState.EXPIRED.value
    ]
    return sorted(selected, key=queue_key)


def apply_filters(
    views: Sequence[JobView],
    *,
    source: Optional[str] = None,
    eligibility: Optional[str] = None,
    status: Optional[str] = None,
    match_tier: Optional[str] = None,
    uncertain: Optional[bool] = None,
    posted_after: Optional[str] = None,
    freshness: Optional[str] = None,
    possible_duplicate: Optional[bool] = None,
) -> List[JobView]:
    """Narrow a list by any combination of read-only filters.

    Every filter is conjunctive: supplying two narrows by both. ``uncertain``
    means "assessed but the evidence will not carry weight", which is a
    different question from "never assessed" - both are offered because
    neither subsumes the other.
    """
    result: List[JobView] = []
    for view in views:
        if source and source not in view.sources:
            continue
        if eligibility and view.verdict != eligibility:
            continue
        if status and view.application_status != status:
            continue
        if match_tier and view.match_tier != match_tier:
            continue
        if uncertain is True and not view.match_uncertain:
            continue
        if uncertain is False and view.match_uncertain:
            continue
        if posted_after is not None:
            # Compared as parsed dates, never as raw text. ``posted_date``
            # carries a "not yet evaluated" sentinel when the source published
            # no date; compared as a string that sentinel sorts *above* every
            # real ISO timestamp ("n" > "2"), so an undated job would otherwise
            # pass a filter meant to show only recent ones. A job whose date we
            # cannot read cannot be shown to be after a given date, so it does
            # not match - and this is the only place in the project where an
            # unreadable value narrows a result set rather than widening it.
            moment = _posted_rank(view.posted_date)
            cutoff = _posted_rank(posted_after)
            if moment is None or cutoff is None or moment < cutoff:
                continue
        if freshness and view.freshness != freshness:
            continue
        if possible_duplicate is True and not view.possible_duplicate:
            continue
        if possible_duplicate is False and view.possible_duplicate:
            continue
        result.append(view)
    return result


@dataclass(frozen=True)
class QueueFilters:
    """Read-only narrowing of the queue. Every field optional; all conjunctive."""

    source: Optional[str] = None
    eligibility: Optional[str] = None
    status: Optional[str] = None
    match_tier: Optional[str] = None
    uncertain: Optional[bool] = None
    posted_after: Optional[str] = None
    freshness: Optional[str] = None
    possible_duplicate: Optional[bool] = None

    def applied(self) -> Dict[str, Any]:
        return {
            key: value for key, value in (
                ("source", self.source), ("eligibility", self.eligibility),
                ("status", self.status), ("match_tier", self.match_tier),
                ("uncertain", self.uncertain), ("posted_after", self.posted_after),
                ("freshness", self.freshness),
                ("possible_duplicate", self.possible_duplicate),
            ) if value is not None
        }

    def narrow(self, views: Sequence[JobView]) -> List[JobView]:
        return apply_filters(
            views,
            source=self.source, eligibility=self.eligibility, status=self.status,
            match_tier=self.match_tier, uncertain=self.uncertain,
            posted_after=self.posted_after, freshness=self.freshness,
            possible_duplicate=self.possible_duplicate,
        )


# ----------------------------------------------------------------------
# building
# ----------------------------------------------------------------------


def build_report(
    store: JobStore,
    *,
    candidate_country: str = "KE",
    matches: Optional[Mapping[str, Any]] = None,
    status_log: Optional[StatusLog] = None,
    freshness: Optional[Mapping[str, Any]] = None,
    attributions: Optional[Mapping[str, str]] = None,
    generated_at: Optional[str] = None,
    queue_filters: Optional["QueueFilters"] = None,
) -> ReviewReport:
    """Assemble the report from everything already on disk.

    Reads only. ``strict=True`` on the store is deliberate: a record that cannot
    be folded is surfaced in :attr:`ReviewReport.errors` rather than quietly
    omitted, because a total that silently excludes rows is worse than useless
    during review.
    """
    errors: List[str] = []
    try:
        records = store.load_jobs(strict=True)
    except ValueError as exc:
        records = store.load_jobs()
        errors.append(f"unreadable job records: {exc}")
    except Exception as exc:  # noqa: BLE001 - report must still render
        records = []
        errors.append(f"could not read stored jobs: {type(exc).__name__}: {exc}")

    attributions = attributions or {}
    if status_log is None:
        try:
            status_log = StatusLog(store)
        except Exception:  # noqa: BLE001
            status_log = None

    if freshness is None:
        freshness, freshness_error = _read_freshness(store)
        if freshness_error:
            errors.append(freshness_error)

    decisions_by_source: Dict[str, Any] = {}
    try:
        from app.sources.access import load_decisions

        decisions_by_source = load_decisions(store.data_dir)
    except Exception:  # noqa: BLE001
        decisions_by_source = {}

    views: List[JobView] = []
    for record in records:
        job_id = _text(record.get("job_id"))
        memory = _decision_memory(status_log, job_id)
        try:
            views.append(build_view(
                record,
                candidate_country=candidate_country,
                application_status=memory.status,
                match=(matches or {}).get(job_id),
                freshness=(freshness or {}).get(job_id),
            ))
        except Exception as exc:  # noqa: BLE001
            errors.append(
                f"job {job_id or '(no id)'} could not be rendered: "
                f"{type(exc).__name__}: {exc}"
            )

    live = [v for v in views if v.freshness in (FreshnessState.ACTIVE.value, "unknown")]

    actionable = [
        v for v in live
        if v.verdict == "eligible"
        and v.application_status == DEFAULT_STATUS.value
    ]
    actionable.sort(key=lambda v: (v.last_seen or "", v.company.casefold()), reverse=True)

    # Every eligible job still awaiting a decision, *including* stale and
    # expired ones. The actionable list above is the subset that is also current;
    # this queue is the full outstanding set. Collapsing them would either hide
    # stale work or mislabel it as gone.
    eligible_unreviewed = [
        v for v in views
        if v.verdict == "eligible"
        and v.application_status == DEFAULT_STATUS.value
    ]
    uncertain = [v for v in views if v.verdict == "unknown"]
    contested = [v for v in views if _is_contested(v)]
    duplicates = [v for v in views if v.possible_duplicate]
    stale = [v for v in views if v.freshness == FreshnessState.STALE.value]
    expired = [v for v in views if v.freshness == FreshnessState.EXPIRED.value]
    needs_decision = [
        v for v in views
        if v.application_status == DEFAULT_STATUS.value and v.verdict == "eligible"
    ]

    # The practical daily queue: eligible, not closed, not expired, in a
    # deterministic order. A job with no match assessment stays in it.
    queue = build_actionable(views)
    filtered = queue_filters.narrow(queue) if queue_filters else list(queue)
    categories = {
        "never_reviewed": sum(
            1 for v in views if v.application_status == ReviewStatus.NEW.value),
        "reviewing_undecided": sum(
            1 for v in views if v.application_status == ReviewStatus.REVIEWING.value),
        "interested": sum(
            1 for v in views if v.application_status == ReviewStatus.INTERESTED.value),
        "shortlisted": sum(
            1 for v in views if v.application_status == ReviewStatus.SHORTLISTED.value),
        "dismissed": sum(
            1 for v in views if v.application_status == ReviewStatus.DISMISSED.value),
        "uncertain_eligibility": len(uncertain),
        "possible_duplicates": len(duplicates),
        "stale": len(stale),
        "expired": len(expired),
    }

    try:
        source_rows = _source_rows(
            store, views, store.load_runs(), _ledger_for(store),
            attributions, decisions_by_source,
        )
    except Exception as exc:  # noqa: BLE001
        source_rows = []
        errors.append(f"could not build source health: {type(exc).__name__}: {exc}")

    decisions = {v.job_id: _decision_memory(status_log, v.job_id) for v in views}

    return ReviewReport(
        generated_at=generated_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        actionable=tuple(actionable),
        eligible_unreviewed=tuple(eligible_unreviewed),
        uncertain=tuple(uncertain),
        contested=tuple(contested),
        duplicates=tuple(duplicates),
        stale=tuple(stale),
        expired=tuple(expired),
        needs_decision=tuple(needs_decision),
        sources=tuple(source_rows),
        decisions=decisions,
        total_jobs=len(views),
        errors=tuple(errors),
        queue=tuple((v, why_actionable(v)) for v in filtered),
        filtered=tuple(filtered),
        filters=dict(queue_filters.applied()) if queue_filters else {},
        categories=categories,
    )


def _is_contested(view: JobView) -> bool:
    """A job whose eligibility reading is internally inconsistent.

    Detected from the flags the existing policy already raises, not by a second
    opinion computed here.
    """
    return any("contested" in f or "geography conflict" in f for f in view.flags)


def _ledger_for(store: JobStore):
    """The attempt ledger for this store, or ``None`` when there is no path."""
    from app.sources.transport import Ledger

    path = store.data_dir / "access_attempts.jsonl"
    return Ledger(path) if path.exists() else None


# ----------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------

_STYLE = """
:root{color-scheme:light dark}
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:2rem;line-height:1.45}
h1{font-size:1.4rem;margin:0 0 .25rem}
h2{font-size:1.05rem;margin:1.75rem 0 .5rem;border-bottom:1px solid #d1d5db;padding-bottom:.25rem}
p.meta{color:#6b7280;margin:0 0 1rem;font-size:.9rem}
.panel{border:1px solid #d1d5db;border-radius:.4rem;padding:.6rem .8rem;margin:0 0 1rem;font-size:.88rem}
.error{border-left:3px solid #b91c1c;background:#fef2f2;padding:.6rem .8rem;margin:0 0 1rem}
.count{font-weight:600}
.empty{color:#6b7280;font-style:italic;padding:.4rem 0;font-size:.88rem}
table{border-collapse:collapse;width:100%;font-size:.86rem;margin:0 0 .5rem}
th,td{border:1px solid #d1d5db;padding:.4rem .5rem;text-align:left;vertical-align:top}
th{background:#111827;color:#f9fafb}
.badge{font-size:.8rem;color:#6b7280}
.state-ok{background:#ecfdf5}
.state-failed{background:#fee2e2}
.state-empty{background:#fef9c3}
.state-never{background:#f3f4f6;color:#6b7280}
.t-active{background:#ecfdf5}
.t-stale{background:#fffbeb}
.t-expired{background:#f3f4f6;color:#6b7280}
.t-unknown{background:#f9fafb;color:#9ca3af}
a{color:#2563eb}
details{margin:.2rem 0}
summary{cursor:pointer;font-size:.85rem}
.evidence{font-size:.82rem;color:#374151;border-left:2px solid #d1d5db;padding-left:.5rem;margin:.2rem 0}
"""


def _match_state(view: JobView) -> Tuple[str, str]:
    """Describe the match state in words, without collapsing the cases.

    Three genuinely different situations, kept apart:

    - no assessment has run at all - this is missing analysis
    - an assessment ran but could not assert a tier - this is thin evidence,
      which is a *result*, not an absence
    - an assessment ran and asserted a tier - this is a finding

    Collapsing the second into the first is how a job that was examined and
    judged thin gets reported as though nobody looked.
    """
    if not view.match_present:
        return "not assessed", "no assessment has been run"
    if not view.match_evaluated:
        reason = view.match_insufficient_reason or "evidence was too thin to assert a tier"
        label = "assessed, no tier asserted"
        if view.match_uncertain:
            label = "assessed, evidence insufficient"
        return label, reason
    if view.match_uncertain:
        return view.match_tier, "low confidence; treat with care"
    return view.match_tier, ""


def _hints_section(report: "ReviewReport", hints) -> str:
    """The hints block, placed above the queue.

    Deliberately high on the page: a reader who stops at the queue must still
    have seen that "not assessed" means unassessed, and that what follows is
    arithmetic rather than judgement.
    """
    from app.jobs.hints import LIMITATIONS

    assessed = sum(1 for view, _ in report.queue if view.match_present)
    parts = [
        "<section class='hints'>",
        "<h2>Review hints &mdash; not assessments</h2>",
        "<p>Shared words, title overlap and explicit seniority wording, "
        "computed from each posting and your profile. "
        "<strong>No tier, no score, no judgement about fit.</strong> "
        "A match assessment comes from verified evidence; these are arithmetic.</p>",
        f"<p class='meta'>Match assessments in this queue: {assessed} of "
        f"{len(report.queue)}. Every other row is unassessed by provider.</p>",
        "<ul class='limitations'>",
    ]
    for limitation in LIMITATIONS:
        parts.append(f"<li>{_e(limitation)}</li>")
    parts.append("</ul>")
    if hints:
        parts.append(
            "<table><tr><th>Job</th><th>Signals</th></tr>")
        for job_id in sorted(hints):
            parts.append(
                f"<tr><td>{_e(job_id)}</td>"
                f"<td>{_e(hints[job_id].summary())}</td></tr>")
        parts.append("</table>")
    else:
        parts.append("<p class='empty'>No review hints computed.</p>")
    parts.append("</section>")
    return "".join(parts)


def _queue_table(entries: Sequence[Tuple[JobView, str]]) -> str:
    """The daily queue, each row carrying the reason it is there."""
    if not entries:
        return '<p class="empty">Nothing in the queue.</p>'
    head = ("#", "Why it is here", "Role", "Eligibility", "Freshness", "Match",
            "Status", "Sources")
    out = ["<tr>" + "".join(f"<th>{_e(h)}</th>" for h in head) + "</tr>"]
    for position, (view, reason) in enumerate(entries, start=1):
        out.append(
            "<tr>"
            f"<td>{position}</td>"
            f"<td>{_e(reason)}</td>"
            f"<td>{_e(view.title)}<div class='badge'>{_e(view.company)}</div></td>"
            f"<td>{_e(view.verdict)}"
            f"<div class='badge'>{_e('; '.join(view.verdict_reasons) or 'no reason recorded')}</div>"
            + "".join(f'<div class="evidence">{_e(q)}</div>' for q in view.evidence)
            + "</td>"
            f"<td class='t-{_slug(view.freshness)}'>{_e(view.freshness)}"
            f"<div class='badge'>last seen {_e(view.last_seen or 'never')}</div></td>"
            f"<td>{_score_cell(view)}</td>"
            f"<td>{_e(view.application_status)}</td>"
            f"<td>{_e(', '.join(view.sources) or '—')}"
            f"<div class='badge'>{_e(', '.join(view.source_urls))}</div></td>"
            "</tr>"
        )
    return (
        '<table><thead>' + out[0] + "</thead><tbody>"
        + "".join(out[1:]) + "</tbody></table>"
    )


def _category_strip(report: "ReviewReport") -> str:
    """Counts per category, over the unfiltered corpus.

    Deliberately not affected by the queue filters: narrowing the queue must
    not make a category look empty when it is not, or a filter would quietly
    report "no duplicates" when duplicates exist.
    """
    order = (
        ("never reviewed", "never_reviewed"),
        ("reviewing, undecided", "reviewing_undecided"),
        ("interested", "interested"),
        ("shortlisted", "shortlisted"),
        ("dismissed", "dismissed"),
        ("uncertain eligibility", "uncertain_eligibility"),
        ("possible duplicates", "possible_duplicates"),
        ("stale", "stale"),
        ("expired", "expired"),
    )
    cells = "".join(
        f"<span>{_e(label)}: <strong>{report.categories.get(key, 0)}</strong></span>"
        for label, key in order
    )
    return f"<div class='panel'>{cells}</div>"


def _score_cell(view: JobView) -> str:
    """Tier and score. A score is shown only when one was actually calculated.

    Never a zero for a missing score: zero reads as "scored and found wanting",
    which is a claim the data does not support.
    """
    label, detail = _match_state(view)
    head = f'<span class="tier">{_e(label)}</span>'
    if view.match_present and view.match_evaluated and view.match_score is not None:
        return head + f'<div class="badge">score {_e(view.match_score)}</div>'
    if view.match_present and view.match_evaluated:
        return head + '<div class="badge">no score calculated</div>'
    return head + f'<div class="badge">{_e(detail)}</div>'


def _job_row(view: JobView, report: ReviewReport) -> str:
    memory = report.decisions.get(view.job_id)
    note = ""
    if memory and memory.last_note:
        note = f'<div class="badge">note: {_e(memory.last_note)}</div>'
    decided = ""
    if memory and memory.decided_at:
        decided = f'<div class="badge">decided {_e(memory.decided_at)}</div>'
    evidence = ""
    if view.evidence:
        evidence = "".join(
            f'<div class="evidence">evidence: {_e(quote)}</div>' for quote in view.evidence
        )
    return (
        "<tr>"
        f"<td>{_e(view.title)}<div class='badge'>{_e(view.company)}</div></td>"
        f"<td>{_e(view.verdict)}"
        f"<div class='badge'>{_e('; '.join(view.verdict_reasons) or 'no reason recorded')}</div>"
        f"{evidence}</td>"
        f"<td class='t-{_slug(view.freshness)}'>{_e(view.freshness)}"
        f"<div class='badge'>{_e(view.freshness_detail or 'no observation recorded')}</div>"
        f"<div class='badge'>last seen {_e(view.last_seen or 'never')}</div></td>"
        f"<td>{_score_cell(view)}</td>"
        f"<td>{_e(view.application_status)}{note}{decided}</td>"
        f"<td>{_e(', '.join(view.sources) or '—')}"
        f"<div class='badge'>{_e(', '.join(view.source_urls))}</div></td>"
        "</tr>"
    )


_JOB_HEADERS = ("Role", "Eligibility", "Freshness", "Match", "Status", "Sources")


def _job_table(views: Sequence[JobView], report: ReviewReport, note: str = "") -> str:
    if not views:
        return '<p class="empty">Nothing in this queue.</p>'
    head = "<tr>" + "".join(f"<th>{_e(h)}</th>" for h in _JOB_HEADERS) + "</tr>"
    body = "".join(_job_row(v, report) for v in views)
    table = f"<table><thead>{head}</thead><tbody>{body}</tbody></table>"
    detail = f'<details><summary>{_e(note)}</summary></details>' if note else ""
    return f'<p class="count">{len(views)} job(s)</p>{table}{detail}'


def _slug(value: Any) -> str:
    """A CSS-class-safe token.

    ``last_run_state`` holds values like ``"never run"``, and interpolating one
    straight into a class attribute produces ``class="state-never run"`` - two
    classes instead of one, so the row silently loses its styling.
    """
    text = _text(value).strip().casefold()
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in text)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "none"


def _source_table(rows: Sequence[SourceHealthRow]) -> str:
    if not rows:
        return '<p class="empty">No sources recorded.</p>'
    head = ("Source", "Access", "Last run", "Requests/24h", "Jobs (e/i/u)", "Attribution")
    out = ["<tr>" + "".join(f"<th>{_e(h)}</th>" for h in head) + "</tr>"]
    for row in rows:
        detail = ""
        if row.last_run_at:
            detail += f'<div class="badge">{_e(row.last_run_at)}</div>'
        if row.consecutive_failures:
            detail += f'<div class="badge">{row.consecutive_failures} consecutive failure(s)</div>'
        if row.budget_refused:
            detail += ('<div class="badge">refused by daily budget '
                       "(inferred from the recorded error)</div>")
        if row.zero_result:
            detail += ('<div class="badge">succeeded and returned nothing '
                       "&mdash; not a failure</div>")
        for reason in row.skipped_reasons:
            detail += f'<div class="badge">skipped: {_e(reason)}</div>'
        if row.last_error:
            detail += f'<div class="badge">{_e(row.last_error)}</div>'
        requests = _e(row.requests_last_24h)
        if row.last_request_at:
            requests += f'<div class="badge">last {_e(row.last_request_at)}</div>'
        attribution = (
            _e(row.attribution) if row.attribution
            else '<span class="badge">none recorded</span>'
        )
        out.append(
            "<tr>"
            f"<td>{_e(row.name)}{detail}</td>"
            f"<td>{_e(row.access_level)}</td>"
            f'<td class="state-{_slug(row.last_run_state)}">{_e(row.last_run_state)}</td>'
            f"<td>{requests}</td>"
            f"<td>{_e(row.eligible)}/{_e(row.ineligible)}/{_e(row.unknown)}</td>"
            f"<td>{attribution}</td>"
            "</tr>"
        )
    body = "".join(out)
    return (
        '<table><thead>' + out[0] + "</thead><tbody>"
        + "".join(out[1:]) + "</tbody></table>"
    )


def _decisions_table(report: ReviewReport) -> str:
    entries = [m for m in report.decisions.values() if m.reviewed or m.last_note]
    if not entries:
        return '<p class="empty">No decisions recorded yet.</p>'
    head = ("Job", "Status", "Decided", "Latest note", "History")
    out = ["<tr>" + "".join(f"<th>{_e(h)}</th>" for h in head) + "</tr>"]
    for memory in sorted(entries, key=lambda m: (m.decided_at or ""), reverse=True):
        history = ", ".join(f"{s}@{t}" for s, t in memory.history)
        out.append(
            "<tr>"
            f"<td>{_e(memory.job_id)}</td>"
            f"<td>{_e(memory.status)}</td>"
            f"<td>{_e(memory.decided_at or '—')}</td>"
            f"<td>{_e(memory.last_note or '—')}</td>"
            f"<td>{_e(history or '—')}</td>"
            "</tr>"
        )
    return f'<table><thead>{"".join(out[0])}</thead><tbody>{"".join(out[1:])}</tbody></table>'


def _evidence_table(views: Sequence[JobView]) -> str:
    if not views:
        return '<p class="empty">No jobs to show evidence for.</p>'
    head = ("Role", "Why", "Match evidence", "Source links", "Freshness basis")
    out = ["<tr>" + "".join(f"<th>{_e(h)}</th>" for h in head) + "</tr>"]
    for view in views[:50]:
        why = "; ".join(view.verdict_reasons) or "no reason recorded"
        match_bits = []
        for item in view.match_evidence:
            claim = _text(item.get("claim")) if isinstance(item, Mapping) else ""
            quote = _text(item.get("quote")) if isinstance(item, Mapping) else ""
            source = _text(item.get("source")) if isinstance(item, Mapping) else ""
            match_bits.append(f"{claim} ({source}): {quote}")
        if not match_evidence_line(view):
            match_bits.append("no assessment has been run")
        out.append(
            "<tr>"
            f"<td>{_e(view.title)}</td>"
            f"<td>{_e(why)}"
            + "".join(f'<div class="evidence">{_e(q)}</div>' for q in view.evidence)
            + "</td>"
            f"<td>{_e(' | '.join(match_bits))}</td>"
            f"<td>{_e(', '.join(view.source_urls) or '—')}</td>"
            f"<td>{_e(view.freshness_detail or 'no observation recorded')}"
            f"<div class='badge'>{_e(view.freshness_last_seen or 'never seen')}</div></td>"
            "</tr>"
        )
    return (
        '<table><thead>' + "".join(out[0]) + "</thead><tbody>"
        + "".join(out[1:]) + "</tbody></table>"
    )


def match_evidence_line(view: JobView) -> str:
    """The match state as one phrase, for the evidence table."""
    return _match_state(view)[0]


def render_report_html(
    report: ReviewReport,
    *,
    title: str = "Daily review",
    hints: Optional[Mapping[str, Any]] = None,
) -> str:
    """Render the report. A pure function of its input; no I/O, no scripts.

    ``hints`` carries transparent review signals. They are rendered in their
    own clearly-labelled section and never touch a match column, a tier, or
    queue membership.
    """
    errors = ""
    if report.errors:
        items = "".join(f"<li>{_e(e)}</li>" for e in report.errors)
        errors = (
            '<div class="error"><strong>Problems reading stored data</strong>'
            f"<ul>{items}</ul>"
            "<p>The counts below may be incomplete. Corrupt records are shown, "
            "not skipped.</p></div>"
        )

    counts = (
        f"jobs <strong>{report.total_jobs}</strong> · "
        f"actionable <strong>{len(report.actionable)}</strong> · "
        f"uncertain <strong>{len(report.uncertain)}</strong> · "
        f"stale <strong>{len(report.stale)}</strong> · "
        f"expired <strong>{len(report.expired)}</strong> · "
        f"duplicates <strong>{len(report.duplicates)}</strong>"
    )

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{_e(title)}</title><style>{_STYLE}</style></head><body>"
        f"<h1>{_e(title)}</h1>"
        f"<p class='meta'>Generated {_e(report.generated_at)} &middot; read-only"
        " &middot; no network requests were made</p>"
        f"{errors}"
        f"<div class='panel'>{counts}</div>"
        + (
            f"<p class='meta'>filters: {_e(json.dumps(report.filters, sort_keys=True))}</p>"
            if report.filters else ""
        )
        + _category_strip(report)
        + _hints_section(report, hints)
        + "<h2>Actionable queue</h2>"
        "<p class='meta'>Kenya-eligible, not dismissed, not expired. Ordered by "
        "status, then freshness, then match tier, then score where one was "
        "actually calculated, then posted date, then job id. A job with no "
        "match assessment stays in the queue: missing analysis is not a reason "
        "to hide a job.</p>"
        + f"{_queue_table(report.queue)}"
        + "<h2>New and actionable</h2>"
        "<p class='meta'>Eligible, not yet reviewed, and not stale or expired. "
        "A match tier is shown only where an assessment actually ran; a missing "
        "assessment never removes a job from this list.</p>"
        f"{_job_table(report.actionable, report)}"
        "<h2>Review queues</h2>"
        "<h3>Eligible but not yet reviewed</h3>"
        f"{_job_table(report.eligible_unreviewed, report)}"
        "<h3>Uncertain eligibility</h3>"
        "<p class='meta'>Could not be decided from what the source published. "
        "Uncertain is not ineligible.</p>"
        f"{_job_table(report.uncertain, report)}"
        "<h3>Contested readings</h3>"
        "<p class='meta'>The source's own signals disagree with each other.</p>"
        f"{_job_table(report.contested, report)}"
        "<h3>Possible duplicates</h3>"
        f"{_job_table(report.duplicates, report, 'Flagged only; a fingerprint match never merges.')}"
        "<h3>Stale</h3>"
        "<p class='meta'>No longer listed by a source inside its freshness "
        "window. The posting may still be open.</p>"
        f"{_job_table(report.stale, report)}"
        "<h3>Expired</h3>"
        "<p class='meta'>Absent beyond the freshness window. Flagged, never "
        "deleted. Still shown.</p>"
        f"{_job_table(report.expired, report)}"
        "<h3>Awaiting a manual decision</h3>"
        f"{_job_table(report.needs_decision, report)}"
        "<h2>Source health</h2>"
        f"{_source_table(report.sources)}"
        "<h2>Decision memory</h2>"
        f"{_decisions_table(report)}"
        "<h2>Evidence</h2>"
        f"{_evidence_table(report.actionable)}"
        "</body></html>"
    )


def render_report_file(report: ReviewReport, output_path: Path) -> str:
    """Write the report. The only filesystem call in this module."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_report_html(report), encoding="utf-8")
    return str(target)