"""Ingestion: a source's raw records in, durable canonical jobs out.

This is the seam that makes the pipeline real. Before it existed, the store
had no writer and the adapter had no caller, so a scrape produced a list of
dicts that vanished when the process exited.

    source records ──▶ adapt (P1) ──▶ store (P0) ──▶ data/jobs.jsonl
                             │
                             └─ unusable ─▶ data/rejected.jsonl + reason

Design constraints
------------------
* **Offline by construction.** This module contains no HTTP client, no browser,
  no subprocess and no model call. Fetching is somebody else's job; ingestion
  only consumes records it has already been handed. That is what lets the whole
  path be tested against saved fixtures.
* **One bad record never stops the run.** Every record is adapted inside its own
  ``try``; anything unusable is quarantined with a reason and ingestion
  continues.
* **Idempotent.** Replaying the same records produces no new canonical jobs.
  The store's URL identity decides "new" versus "seen again"; ingestion does
  not keep its own seen-set.
* **Reports honestly.** The returned :class:`~app.jobs.store.SourceOutcome`
  separates records received, stored, updated, flagged, quarantined and failed,
  so "found nothing" stays distinguishable from "could not look".

Not yet implemented here, and deliberately so: per-source health tracking,
retries, rate limiting, caching and incremental collection. Ingestion is the
narrow piece that has to be right before those are layered on.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional

from app.jobs.adapters import AdaptError, adapt_record
from app.jobs.store import JobStore, SourceOutcome, StoreResult


def _now(moment: Optional[datetime]) -> datetime:
    return moment or datetime.now(timezone.utc)


def ingest(
    records: Iterable[Mapping[str, Any]],
    *,
    source: str,
    store: JobStore,
    observed_at: Optional[datetime] = None,
) -> SourceOutcome:
    """Adapt ``records`` from ``source`` and persist the usable ones.

    Returns a :class:`SourceOutcome` describing exactly what happened:

    ``fetched``
        Records received from the source.
    ``stored``
        New canonical jobs written for the first time.
    ``updated``
        Records whose URL had been seen before, so the existing job was
        refreshed. On a replay of identical data this equals ``fetched`` and
        ``stored`` is zero.
    ``possible_duplicates``
        Newly stored records flagged as a likely duplicate of another posting
        at the same company with the same title. Flagged, never merged.
    ``rejected``
        Records that could not become a ``Job``. Each is written to
        ``data/rejected.jsonl`` with a reason; none is silently dropped.
    ``ok``
        ``False`` only when the source itself failed. A run that received
        nothing but failed nothing is ``ok=True, fetched=0``.

    A record that cannot be adapted is quarantined and skipped; the remaining
    records are still ingested.
    """
    moment = _now(observed_at)
    stamp = moment.isoformat(timespec="seconds")

    received = 0
    rejected = 0
    adapted: list[dict[str, Any]] = []

    for raw in records:
        received += 1
        try:
            job = adapt_record(source, raw, now=moment)
        except AdaptError as exc:
            store.quarantine(
                raw,
                reason=f"adaptation failed: {exc.reason}",
                source=source,
                observed_at=stamp,
            )
            rejected += 1
            continue
        except Exception as exc:  # pragma: no cover - defence in depth
            # A single pathological record must never abort ingestion. The
            # store is the only thing that knows how to quarantine, so this
            # stays broad on purpose and records the type for debugging.
            store.quarantine(
                raw,
                reason=f"unexpected {type(exc).__name__} during adaptation: {exc}",
                source=source,
                observed_at=stamp,
            )
            rejected += 1
            continue
        adapted.append(job.__dict__)

    result: StoreResult = (
        store.store(adapted, source=source, observed_at=stamp)
        if adapted
        else StoreResult()
    )

    return SourceOutcome(
        name=source,
        ok=True,
        fetched=received,
        stored=result.stored,
        updated=result.updated,
        rejected=rejected + result.rejected,
        possible_duplicates=result.possible_duplicates,
        error=None,
    )
