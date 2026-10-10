"""Freshness and expiry for stored jobs, without deleting anything.

What this is for
----------------
A job that is no longer in a feed is not the same fact as a job that was never
there, and neither is the same as a source that failed. Collapsing those three
into "not found" is how a dashboard ends up quietly lying: every listing appears
to vanish because one source timed out, or because a feed returned a zero-item
page that was really a parse failure.

This module keeps them apart:

- A **failed** source observation says nothing about any job. It is recorded
  and it changes no freshness state, because a source that could not be read
  cannot testify that a job is gone.
- A **successful** observation is testimony. Jobs the source returned are
  refreshed. Jobs it did not return are candidates for going stale.
- A **successful zero-result** observation is treated far more cautiously than
  a successful observation that returned other jobs. An empty feed is exactly
  what a truncated response, a silently changed feed shape, or a blocked
  request also looks like, so it must not immediately expire a whole corpus.

What this deliberately does not do
----------------------------------
It never deletes a job, never rewrites ``data/jobs.jsonl``, and never touches
the review status log. Going stale or expired is a *label* applied at render
time from an append-only observation log; the record itself and the candidate's
decisions about it are untouched. That is what makes expiry reversible - a job
that reappears is simply refreshed again, and its history is still there.

Determinism
-----------
Evaluation is a pure function of (observation log, policies, now). Nothing here
reads the wall clock during evaluation, and nothing calls out to a network, so
every rule below is directly testable with injected timestamps.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from app.jobs.store import JobStore, _utcnow


OBSERVATION_VERSION = 1


class FreshnessState(str, Enum):
    """How current a stored job is, judged only from source testimony.

    These are ``str``-mixin enums: members compare equal to their value, so
    ``state == "active"`` works. Note that ``str(member)`` still yields
    ``FreshnessState.ACTIVE`` on Python 3.11+; use ``.value`` when serialising.
    """

    #: Seen in the most recent successful observation of at least one of its
    #: sources.
    ACTIVE = "active"
    #: Not returned by a source that has successfully reported since, for longer
    #: than that source's stale window. Still a real job; the posting may well
    #: be open, but the feed no longer lists it.
    STALE = "stale"
    #: Absent beyond the source's expiry window. Flagged, never deleted.
    EXPIRED = "expired"

    def rank(self) -> int:
        """Sort order, freshest first."""
        return {FreshnessState.ACTIVE: 0, FreshnessState.STALE: 1, FreshnessState.EXPIRED: 2}[self]


@dataclass(frozen=True)
class SourcePolicy:
    """Per-source freshness thresholds.

    The two live sources behave differently and are configured separately
    rather than sharing one default:

    ``weworkremotely``
        A continuously-updated feed of recent remote roles. A listing leaves it
        quickly once it is filled, so a shorter stale window is honest.

    ``myjobmag.co.ke``
        A feed polled at most once a day, holding far more older Kenyan
        listings. A role can legitimately sit there for weeks, so it gets a
        longer window - marking those stale after a fortnight would be wrong.

    ``stale_days`` and ``expire_days`` are deliberately time-based rather than
    count-based: the two sources are polled at different cadences by design
    (one is throttled to a single daily request), so "absent for three runs"
    means a day for one and a month for the other.
    """

    source: str
    stale_days: int = 14
    expire_days: int = 90
    #: Consecutive *successful* zero-result observations that must accumulate
    #: before any job from this source may go stale. Guards against a single
    #: empty or truncated response wiping out the corpus.
    zero_result_grace_runs: int = 2

    def __post_init__(self) -> None:
        if self.expire_days < self.stale_days:
            raise ValueError(
                f"{self.source}: expire_days ({self.expire_days}) must be at least "
                f"stale_days ({self.stale_days}); a job cannot expire before it is stale"
            )
        if self.zero_result_grace_runs < 1:
            raise ValueError(
                f"{self.source}: zero_result_grace_runs must be at least 1"
            )


#: The two live sources, configured to their real feed behaviour.
DEFAULT_POLICIES: Dict[str, SourcePolicy] = {
    "weworkremotely": SourcePolicy(
        source="weworkremotely", stale_days=14, expire_days=60, zero_result_grace_runs=2
    ),
    "myjobmag.co.ke": SourcePolicy(
        source="myjobmag.co.ke", stale_days=21, expire_days=120, zero_result_grace_runs=3
    ),
}

#: Used for any source without an explicit entry above.
FALLBACK_POLICY = SourcePolicy(source="unknown", stale_days=14, expire_days=60)


def policy_for(source: str, policies: Optional[Mapping[str, SourcePolicy]] = None) -> SourcePolicy:
    table = DEFAULT_POLICIES if policies is None else policies
    found = table.get(source)
    if found is not None:
        return found
    # A source with no declared policy still gets a real policy, not None, so
    # callers never have to special-case the unknown-source path.
    return SourcePolicy(
        source=source,
        stale_days=FALLBACK_POLICY.stale_days,
        expire_days=FALLBACK_POLICY.expire_days,
        zero_result_grace_runs=FALLBACK_POLICY.zero_result_grace_runs,
    )


@dataclass(frozen=True)
class SourceObservation:
    """One source's behaviour in one run.

    The three-way split is the whole point of this type:

    - ``ok=False`` - the source failed. Carries no evidence about any job.
    - ``ok=True`` with a non-empty ``returned_job_ids`` - the source spoke.
    - ``ok=True`` with an empty ``returned_job_ids`` - the source spoke and said
      "nothing", which is the weakest possible testimony and is counted
      separately for exactly that reason.
    """

    source: str
    at: str
    ok: bool
    returned_job_ids: Sequence[str] = ()
    error: str = ""
    #: True when the source was deliberately not consulted (disabled, or not
    #: cleared for access). A skip is not evidence of absence either.
    skipped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_version": OBSERVATION_VERSION,
            "source": self.source,
            "at": self.at,
            "ok": self.ok,
            "returned_job_ids": list(self.returned_job_ids),
            "error": self.error,
            "skipped": self.skipped,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SourceObservation":
        return cls(
            source=str(data.get("source", "")),
            at=str(data.get("at", "")),
            ok=bool(data.get("ok", False)),
            returned_job_ids=tuple(data.get("returned_job_ids", []) or ()),
            error=str(data.get("error", "")),
            skipped=bool(data.get("skipped", False)),
        )

    @property
    def is_failure(self) -> bool:
        return not self.ok and not self.skipped

    @property
    def is_zero_result(self) -> bool:
        """Successful, consulted, and returned nothing."""
        return self.ok and not self.skipped and not self.returned_job_ids

    @property
    def is_evidence(self) -> bool:
        """Does this observation say anything at all about job presence?

        A failure or a skip does not. Only a genuine successful fetch does -
        and a zero-result one counts, but only after a grace threshold.
        """
        return self.ok and not self.skipped


def parse_at(value: str) -> datetime:
    """Parse a stored timestamp, tolerating both ``Z`` and ``+00:00`` forms."""
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty timestamp")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        # A naive timestamp is assumed UTC rather than local: the writer always
        # emits UTC, and guessing local time would shift every threshold.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _days_between(later: datetime, earlier: datetime) -> int:
    return max(0, (later - earlier).days)


@dataclass
class SourceState:
    """Per-(job, source) freshness evidence, accumulated across runs."""

    source: str
    #: Most recent moment this source actually returned the job.
    last_seen: Optional[str] = None
    #: Consecutive successful observations by this source that did *not* return
    #: the job. Reset to 0 the moment the job reappears.
    consecutive_misses: int = 0
    #: Consecutive successful zero-result observations by this source. Counts up
    #: independently of ``consecutive_misses`` so that grace is consumed by
    #: empty feeds specifically, not by ordinary absence.
    consecutive_zero_results: int = 0
    stale_since: Optional[str] = None
    expired_since: Optional[str] = None
    #: Monotonic counter of successful observations. Never decremented, so a
    #: job's observation count only ever grows.
    observations: int = 0
    #: This pair's state as of the last evaluation, so readers agree with the
    #: job-level rollup. Recorded rather than recomputed on demand: re-deriving
    #: against the wall clock would let a per-source answer drift away from the
    #: "as of now" the caller actually asked about.
    state: str = FreshnessState.ACTIVE.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "last_seen": self.last_seen,
            "consecutive_misses": self.consecutive_misses,
            "consecutive_zero_results": self.consecutive_zero_results,
            "stale_since": self.stale_since,
            "expired_since": self.expired_since,
            "observations": self.observations,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SourceState":
        return cls(
            source=str(data.get("source", "")),
            last_seen=data.get("last_seen"),
            consecutive_misses=int(data.get("consecutive_misses", 0) or 0),
            consecutive_zero_results=int(data.get("consecutive_zero_results", 0) or 0),
            stale_since=data.get("stale_since"),
            expired_since=data.get("expired_since"),
            observations=int(data.get("observations", 0) or 0),
            state=str(data.get("state", FreshnessState.ACTIVE.value)),
        )


@dataclass
class JobFreshness:
    """A job's freshness, aggregated across every source that has seen it.

    The job-level state is the *worst* state among its sources. A job listed by
    both a healthy daily feed and a source that has gone quiet should not be
    hidden because one of the two stopped mentioning it.
    """

    job_id: str
    state: FreshnessState = FreshnessState.ACTIVE
    sources: Dict[str, SourceState] = field(default_factory=dict)
    last_seen: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "state": self.state.value,
            "last_seen": self.last_seen,
            "sources": {name: state.to_dict() for name, state in self.sources.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JobFreshness":
        return cls(
            job_id=str(data.get("job_id", "")),
            state=FreshnessState(data.get("state", "active")),
            sources={
                name: SourceState.from_dict(payload)
                for name, payload in (data.get("sources") or {}).items()
            },
            last_seen=data.get("last_seen"),
        )

    def state_for(self, source: str) -> str:
        """This job's state with respect to one source, as last evaluated.

        Reads the recorded value rather than recomputing, so it always agrees
        with :attr:`state`. ``active`` for an unrecorded source is a real
        answer, not a default: nothing has said otherwise.
        """
        state = self.sources.get(source)
        if state is None:
            return FreshnessState.ACTIVE.value
        return state.state


def _derive_state(source_state: SourceState, policy: SourcePolicy, now: datetime) -> FreshnessState:
    """Classify one (job, source) pair. Pure: no I/O, no wall clock of its own.

    The order matters and encodes the rules in priority order:

    1. No evidence yet -> active. Absence of evidence is not evidence.
    2. The source's most recent successful observation returned *nothing*, and
       that empty run has not yet repeated often enough -> active. An empty feed
       is weak testimony, so it does not move a job until the grace threshold
       is met. Note this only bites while the last observation was empty: once
       the feed comes back with content, ordinary absence is trusted on its own
       schedule.
    3. Otherwise, elapsed time since the source last returned the job decides.
    """
    if source_state.last_seen is None:
        return FreshnessState.ACTIVE

    if 0 < source_state.consecutive_zero_results < policy.zero_result_grace_runs:
        return FreshnessState.ACTIVE

    elapsed = _days_between(now, parse_at(source_state.last_seen))
    if elapsed >= policy.expire_days:
        return FreshnessState.EXPIRED
    if elapsed >= policy.stale_days:
        return FreshnessState.STALE
    return FreshnessState.ACTIVE


def _worst(states: Iterable[FreshnessState]) -> FreshnessState:
    worst = FreshnessState.ACTIVE
    for state in states:
        if state.rank() > worst.rank():
            worst = state
    return worst


def apply_observation(
    freshness: Mapping[str, JobFreshness],
    observation: SourceObservation,
    *,
    tracked_job_ids: Optional[Iterable[str]] = None,
    policies: Optional[Mapping[str, SourcePolicy]] = None,
    now: Optional[datetime] = None,
) -> Dict[str, JobFreshness]:
    """Fold one observation into the freshness map and return a new map.

    ``tracked_job_ids`` is every job the store knows about, including jobs this
    source has never seen. That is what lets a job which has dropped out of a
    feed age out: without it there would be no record to age.

    Pure - the input mapping is never mutated - so tests can assert that
    applying an observation leaves earlier results alone.
    """
    moment = now or parse_at(observation.at)
    policy = policy_for(observation.source, policies)
    returned = set(observation.returned_job_ids)
    result: Dict[str, JobFreshness] = {
        job_id: JobFreshness.from_dict(job.to_dict()) for job_id, job in freshness.items()
    }

    candidates = set(result)
    if tracked_job_ids is not None:
        candidates |= set(tracked_job_ids)

    for job_id in candidates:
        job = result.get(job_id) or JobFreshness(job_id=job_id)
        state = job.sources.get(observation.source) or SourceState(source=observation.source)

        # A failed or skipped source is not testimony. Recording the attempt is
        # still worthwhile, but no counter moves and no state may worsen.
        if not observation.is_evidence:
            result[job_id] = job
            continue

        if job_id in returned:
            state.last_seen = observation.at
            state.consecutive_misses = 0
            state.consecutive_zero_results = 0
            state.stale_since = None
            state.expired_since = None
        else:
            state.consecutive_misses += 1
            if observation.is_zero_result:
                state.consecutive_zero_results += 1
            else:
                # The feed spoke with content again, so the empty run is no
                # longer the latest testimony. Grace is spent by consecutive
                # *latest* empties - one recovered feed resets it for every job,
                # including ones that run still did not return.
                state.consecutive_zero_results = 0

        state.observations += 1
        job.sources[observation.source] = state

        derived = _derive_state(state, policy, moment)
        if derived is FreshnessState.STALE and state.stale_since is None:
            state.stale_since = observation.at
        if derived is FreshnessState.ACTIVE:
            # Recovery clears the marks. The miss counters remain, so how long
            # a job had been missing before it came back is still legible.
            state.stale_since = None
            state.expired_since = None
        if derived is FreshnessState.EXPIRED and state.expired_since is None:
            state.expired_since = observation.at

        job.last_seen = max(
            (entry.last_seen for entry in job.sources.values() if entry.last_seen),
            default=job.last_seen,
        )
        derived_states = {}
        for name, entry in job.sources.items():
            derived_states[name] = _derive_state(
                entry, policy_for(name, policies), moment
            )
        state.state = derived_states[observation.source].value
        job.state = _worst(derived_states.values())
        result[job_id] = job

    return result


def evaluate(
    observations: Sequence[SourceObservation],
    *,
    tracked_job_ids: Optional[Iterable[str]] = None,
    policies: Optional[Mapping[str, SourcePolicy]] = None,
    now: Optional[datetime] = None,
) -> Dict[str, JobFreshness]:
    """Replay a whole observation log forward.

    Deterministic and order-dependent - observations are applied oldest first -
    so the result is a function of the log alone. Replaying rather than keeping
    mutated state means there is no incremental-update bug to get wrong.
    """
    ordered = sorted(observations, key=lambda obs: obs.at)
    moment = now or (parse_at(ordered[-1].at) if ordered else datetime.now(timezone.utc))
    freshness: Dict[str, JobFreshness] = {}
    for observation in ordered:
        freshness = apply_observation(
            freshness,
            observation,
            tracked_job_ids=tracked_job_ids,
            policies=policies,
            now=parse_at(observation.at),
        )
    # Re-derive against the supplied ``now`` so a caller asking "as of now"
    # gets the state as of now, not as of the last logged run.
    if ordered and moment > parse_at(ordered[-1].at):
        freshness = _rederive(freshness, policies, moment)
    return freshness


def _rederive(
    freshness: Mapping[str, JobFreshness],
    policies: Optional[Mapping[str, SourcePolicy]],
    now: datetime,
) -> Dict[str, JobFreshness]:
    result: Dict[str, JobFreshness] = {}
    for job_id, job in freshness.items():
        clone = JobFreshness.from_dict(job.to_dict())
        if clone.sources:
            derived = {
                name: _derive_state(entry, policy_for(name, policies), now)
                for name, entry in clone.sources.items()
            }
            for name, entry_state in derived.items():
                clone.sources[name].state = entry_state.value
            clone.state = _worst(derived.values())
        else:
            clone.state = FreshnessState.ACTIVE
        result[job_id] = clone
    return result


class FreshnessLedger:
    """Append-only log of source observations in ``data/freshness.jsonl``.

    Observations are stored, never jobs. That separation is what lets a source
    fail without touching any job record: the failure is a line in this file and
    nothing else.
    """

    def __init__(self, store: JobStore):
        self.store = store
        self.path = store.data_dir / "freshness.jsonl"

    def record(self, observations: Iterable[SourceObservation]) -> int:
        """Append observations. Returns how many were written."""
        written = 0
        for observation in observations:
            self._append(observation.to_dict())
            written += 1
        return written

    def record_run(self, run: Any) -> int:
        """Record every source in a :class:`~app.jobs.store.RunRecord`.

        Successes carry the ids the source actually returned, so absence is
        derived later rather than assumed at write time. Failures and skips are
        recorded too - as explicit non-testimony, which is what keeps a failing
        source from being read as an empty one.

        The timestamp is the run's ``started_at`` - the same stamp the job
        records were written with. ``finished_at`` would read the wall clock
        even when the caller injected a fixed ``observed_at``, which would leave
        observations and ``last_seen`` fields disagreeing about when a run
        happened and skew every freshness window by the difference.
        """
        stamp = getattr(run, "started_at", None) or getattr(run, "finished_at", None) or _utcnow()
        observations = [
            SourceObservation(
                source=outcome.name,
                at=stamp,
                ok=outcome.ok,
                returned_job_ids=tuple(getattr(outcome, "seen_job_ids", ()) or ()),
                error=outcome.error or "",
            )
            for outcome in getattr(run, "sources", [])
        ]
        observations.extend(
            SourceObservation(
                source=str(entry.get("name", "")),
                at=stamp,
                ok=False,
                skipped=True,
                error=str(entry.get("reason", "")),
            )
            for entry in getattr(run, "skipped", []) or []
        )
        return self.record(observations)

    def load(self) -> List[SourceObservation]:
        """Read the log back. Unreadable lines are skipped, never fatal."""
        return [
            SourceObservation.from_dict(row)
            for row in self.store._read_jsonl(self.path)
        ]

    def state_for(self, job_id: str, **kwargs: Any) -> JobFreshness:
        """Freshness of one job, replayed from the log."""
        return self.evaluate(**kwargs).get(job_id) or JobFreshness(job_id=job_id)

    def evaluate(
        self,
        *,
        tracked_job_ids: Optional[Iterable[str]] = None,
        policies: Optional[Mapping[str, SourcePolicy]] = None,
        now: Optional[datetime] = None,
    ) -> Dict[str, JobFreshness]:
        observations = self.load()
        if tracked_job_ids is None:
            tracked_job_ids = [
                str(record.get("job_id"))
                for record in self.store.load_jobs()
                if record.get("job_id")
            ]
        return evaluate(
            observations,
            tracked_job_ids=tracked_job_ids,
            policies=policies,
            now=now,
        )

    def _append(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def source_report(observations: Sequence[SourceObservation]) -> List[Dict[str, Any]]:
    """Per-source outcome history, derived from the observations alone.

    This exists because job freshness cannot report a source that has *never*
    succeeded: there are no jobs behind it, so it would simply be missing from
    the per-job view - and a reader would see an empty list rather than "this
    source is failing". Absence of evidence about a source must not look like
    absence of the source.

    Reports the latest outcome, how long it has been failing, and when it last
    actually worked.
    """
    ordered = sorted(observations, key=lambda obs: obs.at)
    report: Dict[str, Dict[str, Any]] = {}
    for observation in ordered:
        row = report.setdefault(
            observation.source,
            {
                "source": observation.source,
                "outcome": "unknown",
                "consecutive_failures": 0,
                "last_success": None,
                "last_outcome_at": None,
                "jobs_returned": 0,
                "skipped": False,
            },
        )
        row["last_outcome_at"] = observation.at
        if observation.is_evidence:
            row["outcome"] = "ok" if observation.returned_job_ids else "empty"
            row["consecutive_failures"] = 0
            row["last_success"] = observation.at
            row["jobs_returned"] += len(observation.returned_job_ids)
            row["skipped"] = False
        elif observation.skipped:
            row["outcome"] = "skipped"
            row["skipped"] = True
        else:
            row["consecutive_failures"] += 1
            if row["outcome"] != "skipped":
                row["outcome"] = "failing"
    return sorted(report.values(), key=lambda row: row["source"])


def summarise(
    freshness: Mapping[str, JobFreshness],
    *,
    observations: Optional[Sequence[SourceObservation]] = None,
) -> Dict[str, Any]:
    """Counts per state, plus each source's most recent outcome.

    Source health and job freshness are reported separately on purpose. "Five
    jobs are stale" and "one source failed" are different facts with different
    causes, and merging them into a single health number would make a source
    outage look like a job-market change.

    ``observations`` lets sources that have never produced a job still be
    reported - see :func:`source_report`.
    """
    counts = {state.value: 0 for state in FreshnessState}
    for job in freshness.values():
        counts[job.state.value] += 1

    source_rows: List[Dict[str, Any]] = []
    seen_sources: Dict[str, Dict[str, Any]] = {}
    for job in freshness.values():
        for name, state in job.sources.items():
            row = seen_sources.setdefault(
                name,
                {"source": name, "jobs": 0, "last_seen": None,
                 "state": FreshnessState.ACTIVE.value},
            )
            row["jobs"] += 1
            if state.last_seen and (row["last_seen"] is None or state.last_seen > row["last_seen"]):
                row["last_seen"] = state.last_seen
            # Take the worst per-source state across jobs. Reads the recorded
            # value rather than re-deriving, so this summary and the job rows
            # can never disagree about the same source.
            if FreshnessState(state.state).rank() > FreshnessState(row["state"]).rank():
                row["state"] = state.state
    source_rows = sorted(seen_sources.values(), key=lambda row: row["source"])

    return {
        "jobs": counts,
        "sources": source_rows,
        "outcomes": source_report(observations) if observations else [],
        "total": len(freshness),
    }