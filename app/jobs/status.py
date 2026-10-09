"""A record of what the candidate decided about each job.

What this is
------------
A person's disposition toward a posting: *I have not looked at this*, *I am
looking*, *I want this*, *I am not interested*. That judgement is not derivable
from the posting, so it has to be recorded somewhere.

What this is deliberately **not**
--------------------------------
``app.state.models.ApplicationStatus`` already exists and is a different thing:
it describes the execution state of the automated pipeline (``pending``,
``running``, ``failed``). ``WorkflowStage`` goes further, with stages such as
``DRAFTING``, ``COMPILING`` and ``READY_FOR_APPROVAL``. Neither is reused here,
because reusing them would drag a record of human judgement into the automated
application workflow, and submitting applications is a hard stop for this
project. No state in this module represents an application, and nothing here
can produce one.

Design commitments
------------------
**Append-only.** Every decision is an event in ``data/status.jsonl``. The
current status of a job is derived by folding its events forward. Nothing is
ever overwritten, so a mistaken decision is recoverable by replaying the log -
which is what makes :meth:`StatusLog.history` the authoritative record and the
current status merely a summary of it.

**An overlay, not an edit.** Status lives in its own file. ``data/jobs.jsonl``
is written only by ingestion, and a status change never touches a job record.
Job postings get re-ingested and merged; a review decision must survive that.

**Transitions are explicit.** Moving from one status to another goes through a
declared edge. An undeclared move raises rather than silently recording
something, because a status log that accepts any sequence of states cannot tell
you what happened.

**Notes are data.** A note is the candidate's own words. It is stored as an
opaque string and never interpreted, executed, or fed back in as an
instruction.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from app.jobs.store import JobStore


class ReviewStatus(str, Enum):
    """A candidate's disposition toward a posting."""

    NEW = "new"
    REVIEWING = "reviewing"
    INTERESTED = "interested"
    DISMISSED = "dismissed"


#: The status a job has before any decision is recorded about it.
DEFAULT_STATUS = ReviewStatus.NEW


class StatusError(ValueError):
    """A status transition or record is not valid as written."""


#: Every status is reachable from every other status, *including itself*.
#:
#: This is deliberate, and it is the one design decision in this module worth
#: arguing for. An irreversible transition destroys information: a job marked
#: "dismissed" yesterday and reconsidered today would be indistinguishable from
#: one dismissed and forgotten, and the moment of doubt is exactly what makes
#: the log useful. Nothing here is terminal.
#:
#: Including identity matters too. Re-recording the same status with a further
#: note is how a second thought gets written down, and blocking it would mean
#: the only way to annotate a job was to mislabel it.
#:
#: The guards that carry real weight are therefore elsewhere: an unknown status
#: value, an unknown job id, and an unparseable line are all refused rather
#: than absorbed. A connected graph costs nothing so long as the trail is
#: append-only and complete.
_ALL_STATUSES: Set[ReviewStatus] = set(ReviewStatus)

ALLOWED: Dict[ReviewStatus, Set[ReviewStatus]] = {
    status: set(_ALL_STATUSES) for status in ReviewStatus
}


def can_transition(current: ReviewStatus, target: ReviewStatus) -> bool:
    """Is ``current -> target`` a declared move?

    True for any pair of real statuses - see :data:`ALLOWED`. It is ``False``
    only when either side is not a :class:`ReviewStatus` at all, which is what
    stops a value from the wrong enum being recorded.
    """
    if not isinstance(current, ReviewStatus) or not isinstance(target, ReviewStatus):
        return False
    return target in ALLOWED.get(current, set())


@dataclass(frozen=True)
class StatusEvent:
    """One recorded decision. Immutable and append-only."""

    job_id: str
    status: str
    previous: Optional[str]
    at: str
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "previous": self.previous,
            "at": self.at,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StatusEvent":
        return cls(
            job_id=str(payload.get("job_id", "")),
            status=str(payload.get("status", DEFAULT_STATUS.value)),
            previous=payload.get("previous"),
            at=str(payload.get("at", "")),
            note=str(payload.get("note", "") or ""),
        )


class StatusLog:
    """Read and write ``data/status.jsonl``.

    Construct with a :class:`~app.jobs.store.JobStore` so status is stored
    beside the jobs it refers to. The store is only ever *read* from here - to
    check that a job exists - never modified.
    """

    def __init__(self, store: JobStore) -> None:
        self._store = store
        self.path = Path(store.data_dir) / "status.jsonl"

    # ------------------------------------------------------------------
    # reading
    # ------------------------------------------------------------------

    def _read(self) -> List[StatusEvent]:
        if not self.path.exists():
            return []
        events: List[StatusEvent] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(StatusEvent.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError) as error:
                    # A corrupt line must not make the whole log unreadable -
                    # that would silently hide every decision ever made. It is
                    # reported instead.
                    raise StatusError(
                        f"{self.path}:{number} is not a valid status record: {error}"
                    ) from error
        return events

    def events(self, job_id: Optional[str] = None) -> List[StatusEvent]:
        """Every event, in recorded order, optionally for one job."""
        if job_id is None:
            return self._read()
        return [event for event in self._read() if event.job_id == job_id]

    def history(self, job_id: str) -> List[StatusEvent]:
        """The full decision trail for one job. This is the source of truth."""
        return self.events(job_id)

    def current(self, job_id: str) -> ReviewStatus:
        """The latest status, folding the log forward from the default."""
        events = self.history(job_id)
        if not events:
            return DEFAULT_STATUS
        try:
            return ReviewStatus(events[-1].status)
        except ValueError as error:
            raise StatusError(
                f"job {job_id!r} has unrecognised status "
                f"{events[-1].status!r}; the log is inconsistent"
            ) from error

    def summary(self) -> Dict[str, List[str]]:
        """Job ids grouped by current status, each group in stable order."""
        grouped: Dict[str, List[str]] = {status.value: [] for status in ReviewStatus}
        for record in self._store.load_jobs():
            grouped[self.current(str(record.get("job_id", ""))).value].append(
                str(record.get("job_id", ""))
            )
        return grouped

    # ------------------------------------------------------------------
    # writing
    # ------------------------------------------------------------------

    def known_job(self, job_id: str) -> bool:
        """Is this a job we actually hold?

        A status for an unknown job would be a status for a posting this
        repository has never seen, which is almost always a typo or a stale
        id. Rejecting it keeps the log from filling with entries that can
        never be displayed.
        """
        return any(
            str(record.get("job_id", "")) == job_id
            for record in self._store.load_jobs()
        )

    def record(
        self,
        job_id: str,
        status: ReviewStatus,
        *,
        note: str = "",
        at: Optional[str] = None,
        require_known_job: bool = True,
    ) -> StatusEvent:
        """Append one decision and return the event written.

        Raises :class:`StatusError` if the job is unknown, the status is not a
        :class:`ReviewStatus`, or the move is not a declared transition.
        """
        if not isinstance(status, ReviewStatus):
            raise StatusError(
                f"status must be a ReviewStatus, got {type(status).__name__}"
            )
        if not str(job_id).strip():
            raise StatusError("job id must not be empty")
        if require_known_job and not self.known_job(job_id):
            raise StatusError(f"unknown job id: {job_id!r}")

        existing = self.history(job_id)
        previous = existing[-1].status if existing else None
        current = (
            ReviewStatus(previous) if previous else DEFAULT_STATUS
        )
        if not can_transition(current, status):
            raise StatusError(
                f"{current.value} -> {status.value} is not a declared transition "
                f"for job {job_id!r}; allowed: "
                f"{sorted(s.value for s in ALLOWED[current])}"
            )

        moment = at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        event = StatusEvent(
            job_id=job_id,
            status=status.value,
            previous=previous,
            at=moment,
            note=str(note or ""),
        )
        self._append(event)
        return event

    def _append(self, event: StatusEvent) -> None:
        """Append durably, mirroring ``JobStore._append``."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(
                json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())