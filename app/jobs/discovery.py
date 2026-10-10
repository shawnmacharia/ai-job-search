"""One manual command that runs every permitted source, then reports.

What this is for
----------------
Three sources, each with its own endpoint, its own attribution and - for two of
them - its own daily request allowance. Running them means remembering all of
that. So this module holds the coordination once, and the sources keep holding
their own rules.

Ordering, and why nothing is fetched until it is checked
-------------------------------------------------------
Access is checked first, then the persisted budget, then a fetch plan is built,
and only then does anything touch the network. A source that cannot be consulted
is recorded and skipped rather than attempted, so a refusal costs no request and
writes no attempt to the ledger. Attempting and catching would be simpler and
wrong: it would spend a request to discover a limit we already know.

Independence
------------
Sources run one at a time through :func:`~app.jobs.runner.run_sources`, which
isolates failures. A source that times out, raises, or trips over a corrupt
index fails on its own; the others still run. This module never lets one
source's problem remove another's work from the run.

Outcomes, kept apart
--------------------
Five outcomes look similar from the outside and mean different things, so each
gets its own label rather than a shared "did not work":

``fetched``
    the source was consulted and returned jobs
``zero_result``
    the source was consulted, succeeded, and returned **nothing** - a real
    answer, and a weak one, since an empty response also looks like truncation
``failed``
    the source was consulted and did not work
``budget_refused``
    the source was **not** consulted because its allowance is spent. Nothing was
    requested, so this is not a failure and does not affect the exit code
``skipped``
    the source is disabled, or its access is not recorded as permitted

Collapsing ``budget_refused`` into ``failed`` would be the most damaging of
these merges: a run that respects every agreed limit would exit non-zero and
look broken, which teaches an operator to ignore the exit code.

Exit codes
----------
``0``
    every source that was consulted worked. Refusals and skips are normal and
    do not make a run fail.
``1``
    at least one source failed, but others succeeded. **Partial failure is
    reported, not hidden** - and not reported as total failure either, because
    that would hide the work that did happen.
``2``
    nothing could be consulted at all.

Read-only reporting
-------------------
The review report this produces reads only. Nothing here writes to jobs,
decisions, freshness or ledgers except through the existing store and runner
layers, which is where those writes belong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from app.jobs.runner import (
    SKIP_NOT_VERIFIED,
    SkippedSource,
    SourceSpec,
    run_sources,
)
from app.jobs.sources import (
    PERMITTED,
    SourceConfig,
    SourceRegistry,
)
from app.jobs.store import JobStore

#: Why a source was not consulted because its allowance is spent. Distinct from
#: the access codes in :mod:`app.jobs.runner` because the reason is a cadence
#: we agreed to, not a permission we lack.
SKIP_BUDGET_REFUSED = "budget_refused"

#: Recognised refusal wording, used to classify an error that reaches us through
#: the runner rather than through the pre-check. Reported as an inference, never
#: as a first-class fact.
_REFUSAL_MARKERS = ("daily request limit", "could not be read")


class Outcome(str, Enum):
    """How one source ended up this run."""

    #: Consulted, returned jobs.
    FETCHED = "fetched"
    #: Consulted, succeeded, returned nothing.
    ZERO_RESULT = "zero_result"
    #: Consulted, and it did not work.
    FAILED = "failed"
    #: Not consulted: the agreed allowance is spent, or unreadable.
    REFUSED = "budget_refused"
    #: Not consulted: disabled, or access not recorded as permitted.
    SKIPPED = "skipped"
    #: Dry run only: would have been consulted.
    WOULD_RUN = "would_run"


#: A source appearing in any of these is not an error condition. A run that
#: respects its limits should succeed.
NON_FAILURE_OUTCOMES = frozenset({
    Outcome.FETCHED, Outcome.ZERO_RESULT, Outcome.REFUSED,
    Outcome.SKIPPED, Outcome.WOULD_RUN,
})


@dataclass(frozen=True)
class SourceResult:
    """One source's outcome, with enough context to act on it."""

    name: str
    outcome: Outcome
    reason: str = ""
    fetched: int = 0
    stored: int = 0
    updated: int = 0
    rejected: int = 0
    requests_made: int = 0
    budget_reason: str = ""

    @property
    def failed(self) -> bool:
        return self.outcome is Outcome.FAILED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.name,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "fetched": self.fetched,
            "stored": self.stored,
            "updated": self.updated,
            "rejected": self.rejected,
            "requests_made": self.requests_made,
            "budget_reason": self.budget_reason,
        }


@dataclass
class DiscoveryResult:
    """Everything one orchestration produced."""

    started_at: str
    results: List[SourceResult] = field(default_factory=list)
    run: Any = None
    report_path: Optional[str] = None
    report_errors: Tuple[str, ...] = ()

    @property
    def by_name(self) -> Dict[str, SourceResult]:
        return {r.name: r for r in self.results}

    @property
    def failed(self) -> List[SourceResult]:
        return [r for r in self.results if r.failed]

    @property
    def contacted(self) -> List[SourceResult]:
        """Sources we actually tried to reach, including the ones that failed."""
        return [
            r for r in self.results
            if r.outcome in (Outcome.FETCHED, Outcome.ZERO_RESULT, Outcome.FAILED)
        ]

    @property
    def productive(self) -> List[SourceResult]:
        """Sources that produced an answer.

        A failure is *not* productive: the source was reached but gave us
        nothing. Distinguishing "reached" from "answered" is what makes exit
        code 2 reachable, and a run where every source failed should not be
        reported as a partial success.
        """
        return [
            r for r in self.results
            if r.outcome in (Outcome.FETCHED, Outcome.ZERO_RESULT)
        ]

    @property
    def total_requests(self) -> int:
        return sum(r.requests_made for r in self.results)

    def exit_code(self) -> int:
        """Aggregate status.

        Partial failure is the case that matters most and is the easiest to get
        wrong: it must be visible, but it must not be reported as though nothing
        was collected.
        """
        failures = self.failed
        if not failures:
            return 0
        if not self.productive:
            # Every source we tried failed: nothing was collected.
            return 2
        return 1

    def exit_reason(self) -> str:
        """A sentence explaining :meth:`exit_code`, for the CLI to print."""
        code = self.exit_code()
        if code == 0:
            return (
                "every source that was consulted worked"
                if self.contacted else "nothing needed consulting"
            )
        if code == 2:
            return f"all {len(self.failed)} source(s) failed; nothing was collected"
        return (
            f"{len(self.failed)} source(s) failed while "
            f"{len(self.productive)} succeeded; partial results collected"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "exit_code": self.exit_code(),
            "requests_made": self.total_requests,
            "sources": [r.to_dict() for r in self.results],
            "report_path": self.report_path,
            "report_errors": list(self.report_errors),
        }


def _budget_state(adapter: Any) -> Dict[str, Any]:
    """Ask an adapter whether it may request, tolerating sources with no cap.

    A source without a daily allowance - We Work Remotely governs spacing with a
    rate limiter instead - simply has no budget to consult. Reporting "allowed"
    for those is correct rather than permissive: there is no agreed daily limit
    to breach, and the limiter still applies once a fetch is made.
    """
    probe = getattr(adapter, "budget_state", None)
    if probe is None:
        return {"allowed": True, "reason": "no daily limit is agreed for this source",
                "limit": None, "spent": 0, "untrusted": False}
    try:
        return probe()
    except Exception as exc:  # noqa: BLE001 - never let a probe stop the run
        # A budget that cannot even be inspected is treated as refusing. The
        # conservative direction is the one that does not spend access.
        return {
            "allowed": False,
            "reason": f"budget could not be inspected ({type(exc).__name__}: {exc})",
            "limit": None, "spent": 0, "untrusted": True,
        }


def build_plan(
    adapters: Mapping[str, Any],
    decisions: Mapping[str, Any],
) -> Tuple[SourceRegistry, List[SourceResult], List[SkippedSource]]:
    """Decide what each source will do, consulting no network at all.

    Returns the enforced registry, a per-source outcome for *every* configured
    source, and the skips to record in the run. Splitting this out is what makes
    ``--dry-run`` honest: it is the same decision the real run makes, with the
    fetch step omitted rather than approximated.

    A permitted, affordable source is recorded as :attr:`Outcome.WOULD_RUN`,
    never as FETCHED. What it becomes is only knowable after the fetch, and
    claiming ``fetched`` up front would report a success nobody observed - the
    same mistake as reading an unreadable ledger as "no requests were made".
    """
    registry = SourceRegistry([
        SourceConfig(name=name, access=PERMITTED) for name in adapters
    ])
    enforced = registry.enforce_recorded_decisions(decisions)

    results: List[SourceResult] = []
    skip_entries: List[SkippedSource] = []

    for config in enforced.configs():
        reason = config.skip_reason()
        if reason is not None:
            results.append(SourceResult(
                name=config.name, outcome=Outcome.SKIPPED, reason=reason.reason
            ))
            skip_entries.append(reason)
            continue

        state = _budget_state(adapters[config.name])
        if not state["allowed"]:
            results.append(SourceResult(
                name=config.name,
                outcome=Outcome.REFUSED,
                reason=state["reason"],
                budget_reason=state["reason"],
            ))
            skip_entries.append(SkippedSource(
                config.name, SKIP_BUDGET_REFUSED, state["reason"],
            ))
            continue

        results.append(SourceResult(
            name=config.name,
            outcome=Outcome.WOULD_RUN,
            reason=state["reason"],
            budget_reason=state["reason"],
        ))

    return enforced, results, skip_entries


def _classify(outcome: Any) -> Outcome:
    """Map a runner outcome onto ours.

    The refusal check is an inference from the error text: a refusal can also
    arrive this way when the budget is spent between planning and fetching. It
    is matched on wording and labelled as a refusal, which is accurate but
    worth knowing is inference rather than a flag set at the source.
    """
    if outcome.ok:
        return Outcome.FETCHED if outcome.fetched else Outcome.ZERO_RESULT
    message = (outcome.error or "").casefold()
    if any(marker in message for marker in _REFUSAL_MARKERS):
        return Outcome.REFUSED
    return Outcome.FAILED


def run_discovery(
    store: JobStore,
    adapters: Mapping[str, Any],
    *,
    dry_run: bool = False,
    observed_at: Optional[datetime] = None,
    report_path: Optional[Path] = None,
    attributions: Optional[Mapping[str, str]] = None,
    matches: Optional[Mapping[str, Any]] = None,
    candidate_country: str = "KE",
) -> DiscoveryResult:
    """Coordinate one pass over every permitted source, then report.

    ``dry_run`` plans and reports without building a single fetch spec, so no
    socket is opened and no ledger gains an attempt.
    """
    from app.sources.access import load_decisions

    started = (observed_at or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    decisions = load_decisions(store.data_dir)
    registry, planned, skips = build_plan(adapters, decisions)

    result = DiscoveryResult(started_at=started, results=list(planned))

    if dry_run:
        # Nothing is run and nothing is written. The plan *is* the answer.
        if report_path is not None:
            result.report_path = _write_report(
                store, report_path, attributions, matches, candidate_country
            )
        return result

    specs = [
        SourceSpec(name=planned_result.name,
                   fetch=adapters[planned_result.name].to_records)
        for planned_result in planned
        if planned_result.outcome is Outcome.WOULD_RUN
    ]

    run = run_sources(specs, store=store, observed_at=observed_at, skipped=skips)
    result.run = run

    by_name = result.by_name
    for outcome in run.sources:
        planned_result = by_name.get(outcome.name)
        if planned_result is None:
            continue
        adapter = adapters[outcome.name]
        by_name[outcome.name] = SourceResult(
            name=outcome.name,
            outcome=_classify(outcome),
            reason=outcome.error or planned_result.budget_reason,
            fetched=outcome.fetched,
            stored=outcome.stored,
            updated=outcome.updated,
            rejected=outcome.rejected,
            requests_made=_requests_made(adapter),
            budget_reason=planned_result.budget_reason,
        )
    result.results = list(by_name.values())

    for adapter in adapters.values():
        end_run = getattr(adapter, "end_run", None)
        if callable(end_run):
            end_run()

    if report_path is not None:
        result.report_path, result.report_errors = _write_report(
            store, report_path, attributions, matches, candidate_country
        )

    return result


def _requests_made(adapter: Any) -> int:
    """How many requests this source made, or 0 if it does not say.

    ``requests_made`` is a property on the real adapters, so reading the
    attribute already gives the number. A callable is supported too, so a test
    double or a future adapter can expose it either way.
    """
    probe = getattr(adapter, "requests_made", 0)
    if isinstance(probe, bool):
        return 0
    if isinstance(probe, int):
        return probe
    if callable(probe):
        try:
            value = probe()
        except Exception:  # noqa: BLE001 - a counter must never break the run
            return 0
        return int(value) if isinstance(value, int) else 0
    return 0


def _write_report(
    store: JobStore,
    report_path: Path,
    attributions: Optional[Mapping[str, str]],
    matches: Optional[Mapping[str, Any]],
    candidate_country: str,
) -> Tuple[str, Tuple[str, ...]]:
    """Build and write the consolidated report. Read-only over the store."""
    from app.jobs.freshness import FreshnessLedger, summarise as summarise_freshness
    from app.jobs.status import StatusLog
    from app.reporting.review import build_report, render_report_file

    ledger = FreshnessLedger(store)
    state = ledger.evaluate()
    report = build_report(
        store,
        candidate_country=candidate_country,
        matches=matches,
        status_log=StatusLog(store),
        freshness=state,
        attributions=dict(attributions or {}),
    )
    written = render_report_file(report, Path(report_path))
    return written, tuple(report.errors)