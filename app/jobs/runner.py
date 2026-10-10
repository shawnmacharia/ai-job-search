"""Run several sources and report honestly on each one.

Before this module, a scraper's failure modes were indistinguishable. A source
that returned nothing, a source that Cloudflare blocked, a source whose browser
binary was missing, and a source that raised mid-scrape all ended the same way:
a short list, or an exception nobody caught. That is why "why did I get zero
jobs?" was unanswerable.

This runner fixes the reporting half of that:

    specs ──▶ for each: try fetch ─▶ ingest ─▶ SourceOutcome
                    └─ on any error: SourceOutcome(ok=False, error=...)

Guarantees
----------
* **One source never aborts the batch.** Every source is wrapped individually.
  A timeout, an exception, or a corrupt store index fails that source alone;
  the rest still run and are still reported.
* **Three outcomes are kept apart.** A source that worked and found jobs, a
  source that worked and found none, and a source that failed are different
  facts, and :attr:`SourceOutcome.kind` names which one happened.
* **Exit code is meaningful.** ``0`` when at least one source succeeded —
  including a successful zero-result source — and ``2`` only when every source
  failed. A site with no vacancies is not an outage.
* **No wall-clock in tests.** Durations are measured with an injected
  ``now`` callable, so health reporting is deterministic under test and no test
  ever asserts a real duration.
* **No fetching here.** A spec's ``fetch`` callable is somebody else's code.
  This module performs no network, browser or subprocess access of its own,
  which is what lets the whole path run against saved fixtures.

Retry, backoff, rate limiting and caching are deliberately *not* here; they sit
on top of this once the reporting is trusted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from app.jobs.ingest import ingest
from app.jobs.store import JobStore, RunRecord, SourceOutcome


#: The three states a source can end a run in.
KIND_JOBS = "jobs"
KIND_EMPTY = "empty"
KIND_FAILED = "failed"


#: Why a source was not consulted.
SKIP_DISABLED = "disabled"
SKIP_ACCESS_UNKNOWN = "access_unknown"
SKIP_ACCESS_RESTRICTED = "access_restricted"
SKIP_ACCESS_NOT_PERMITTED = "access_not_permitted"
SKIP_NO_FETCHER = "no_fetcher"
SKIP_NOT_VERIFIED = "not_verified"


@dataclass(frozen=True)
class SkippedSource:
    """A source that was deliberately not consulted, and why.

    A skip is not a failure and never affects the exit code, but it is never
    silent either: a source that quietly disappears from a run is
    indistinguishable from one that was never configured.
    """

    name: str
    code: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "code": self.code, "reason": self.reason}


@dataclass(frozen=True)
class SourceSpec:
    """One source to consult.

    ``fetch`` is called with no arguments and returns an iterable of raw source
    records - the shape the adapters already consume. It is the only place any
    network or browser access may live; the runner never performs it itself.
    """

    name: str
    fetch: Callable[[], Iterable[Mapping[str, Any]]]


def classify(outcome: SourceOutcome) -> str:
    """Name a source outcome: ``jobs``, ``empty`` or ``failed``.

    Exposed as a function rather than baked into the ledger so that a reader of
    ``data/runs.jsonl`` and a caller holding a live outcome reach for the same
    definition.
    """
    if not outcome.ok:
        return KIND_FAILED
    return KIND_EMPTY if outcome.fetched == 0 else KIND_JOBS


def _elapsed_ms(clock: Callable[[], float], started: float) -> int:
    """Milliseconds since ``started``, never negative.

    A monotonic clock cannot go backwards, but a misconfigured or mocked one
    can. A negative duration in the health report would be nonsense data that a
    later reader would have to distrust, so it is clamped to zero here rather
    than written out.
    """
    return max(0, int((clock() - started) * 1000))


def run_sources(
    specs: Sequence[SourceSpec],
    *,
    store: JobStore,
    observed_at: Optional[datetime] = None,
    run_id: Optional[str] = None,
    now: Optional[Callable[[], float]] = None,
    skipped: Sequence[Any] = (),
    record_freshness: bool = True,
) -> RunRecord:
    """Consult every spec, ingest what each returns, and record one run.

    ``now`` is a monotonic clock in seconds, injected for testing; it defaults
    to :func:`time.monotonic`. Durations are computed from it and never
    asserted on directly.

    The returned :class:`~app.jobs.store.RunRecord` carries one
    :class:`~app.jobs.store.SourceOutcome` per source and is already appended to
    ``data/runs.jsonl``. Its :meth:`~app.jobs.store.RunRecord.exit_code` is
    ``2`` only when every source failed.

    ``skipped`` carries sources that were deliberately not consulted - typically
    from :mod:`app.jobs.sources`. The runner accepts them without knowing where
    they came from, so the registry stays decoupled. They are recorded, and they
    never influence the exit code.

    Every run also appends one observation per source to
    ``data/freshness.jsonl``, including the ones that failed or were skipped.
    Recording failures is the point: a source that could not be read is not
    evidence that its jobs are gone, and the only way to keep that distinction
    later is to have written the failure down now. Pass
    ``record_freshness=False`` to suppress the log.
    """
    clock = now or time.monotonic
    moment = observed_at or datetime.now(timezone.utc)
    stamp = moment.isoformat(timespec="seconds")

    run = RunRecord(
        run_id=run_id or f"run-{stamp}",
        started_at=stamp,
        finished_at=stamp,
        skipped=[
            entry.to_dict() if isinstance(entry, SkippedSource) else dict(entry)
            for entry in skipped
        ],
    )

    for spec in specs:
        started = clock()
        try:
            records = spec.fetch()
            outcome = ingest(
                records,
                source=spec.name,
                store=store,
                observed_at=moment,
            )
        except BaseException as exc:  # noqa: BLE001 - isolation is the point
            # Deliberately broad. A source that times out, raises, or trips
            # over a corrupt index must fail on its own without taking the
            # batch with it. KeyboardInterrupt and SystemExit still propagate
            # to BaseException subclasses that are not Exception, so an
            # interrupt is not swallowed.
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            outcome = store.failed_source(
                spec.name,
                exc,
                duration_ms=_elapsed_ms(clock, started),
            )
        else:
            outcome.duration_ms = _elapsed_ms(clock, started)
        run.sources.append(outcome)

    run.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    store.record_run(run)
    if record_freshness:
        # Imported here rather than at module scope: freshness depends on the
        # store's data layout, and a module-level import would make that
        # dependency part of every importer's load order for no benefit.
        from app.jobs.freshness import FreshnessLedger

        FreshnessLedger(store).record_run(run)
    return run


def summarise(run: RunRecord) -> list[dict[str, Any]]:
    """A flat, human-readable view of a run: one entry per source, then skips.

    Convenience for reporting and for tests; the authoritative record remains
    ``data/runs.jsonl``. Skipped sources are included precisely so that this
    view can never imply a source was never configured.
    """
    rows = [
        {
            "source": outcome.name,
            "kind": classify(outcome),
            "ok": outcome.ok,
            "fetched": outcome.fetched,
            "stored": outcome.stored,
            "updated": outcome.updated,
            "possible_duplicates": outcome.possible_duplicates,
            "rejected": outcome.rejected,
            "error": outcome.error,
            "duration_ms": outcome.duration_ms,
        }
        for outcome in run.sources
    ]
    rows.extend(
        {"source": entry["name"], "kind": "skipped", "reason": entry["reason"],
         "code": entry["code"]}
        for entry in run.skipped
    )
    if run.no_active_sources:
        rows.append({"source": None, "kind": "no_active_sources",
                     "reason": "no source was consulted in this run"})
    return rows
