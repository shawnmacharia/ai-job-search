"""Remotive Public API: one endpoint, one request a day, attribution always.

Scope
-----
This module collects from exactly one URL - the documented Public API jobs
endpoint - because that is precisely what was granted. Everything else Remotive
publishes stays off limits: no HTML pages, no job-detail pages, no pagination,
no query variants, no other endpoints. The scope is expressed as constants so
that widening it is a visible edit rather than a silent drift.

Permission
----------
Access rests on written permission from Remotive, recorded as repository-owner
attestation in ``data/ops/remotive_permission_evidence.md`` and recorded as a
``permitted`` decision through :mod:`app.sources.access`. The grant resolves an
otherwise blocking conflict: Remotive's ``robots.txt`` disallows ``/api/*``,
while their published terms grant API access to developers. The endpoint was
sampled only after that grant was recorded.

Terms we hold ourselves to
--------------------------
- **Attribution is not optional.** Every row links back to the original Remotive
  URL and credits Remotive as the source. Their terms say access will be
  terminated without it, so it is attached at ingest rather than left to the
  renderer to remember.
- **No republication.** Nothing here is written anywhere a third-party board
  could read it; the output is a local file for one person's review.
- **We stay stricter than permitted.** Their published allowance is roughly four
  requests a day with a two-per-minute ceiling. We take one a day, which is
  what was committed to them, and which is ample for one job seeker. Being
  under the ceiling is not timidity; it is the reason the ceiling is never
  actually approached.
- **The 24-hour delay is Remotive's to apply, not ours to undo.** We do not
  request fresher data, and the API returns what it returns.

No network access happens at import time. Parsing, adapting and attribution are
pure functions over a response body, which is what lets the tests here be
entirely offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Tuple

from app.jobs.models import Job
from app.sources.budget import DailyBudget, describe_budget, prior_stamps
from app.sources.transport import (
    AccessError,
    AccessFetcher,
    Ledger,
    RateLimiter,
)

#: The one permitted endpoint. Nothing else in this module may be fetched.
API_URL = "https://remotive.com/api/remote-jobs"

#: Where a reader should go to see the original listing.
SOURCE_PAGE = "https://remotive.com/remote-jobs"

#: Terms page whose wording governs this integration.
TERMS_PAGE = "https://remotive.com/remote-jobs/api"

#: Rendered on every row. Their terms require crediting Remotive as the source
#: and linking back; this text is the credit half, the URL is the link half.
ATTRIBUTION_TEXT = "Listing from Remotive"
ATTRIBUTION_REQUIRED = True

#: Our committed cadence - stricter than their published ~4/day allowance.
DAILY_LIMIT = 1

#: Floor between requests. Their published ceiling is two per minute; we commit
#: to at most one per day, and 86400s is that floor expressed as an interval.
#:
#: This is enforced through :class:`~app.sources.transport.RateLimiter` backed
#: by the *persisted* attempt ledger, which is the only part of the mechanism
#: that survives across processes. The in-process :class:`_DailyBudget` is a
#: second, redundant check. An earlier draft used a 30s floor here, which left
#: the daily limit resting on in-memory state alone - the limit would then have
#: reset on every restart, and "one request a day" would have been a claim the
#: code did not actually keep.
MIN_INTERVAL = 86400.0

MAX_ATTEMPTS = 3

#: RemoteStatus values this integration recognises. Anything else is preserved as
#: text rather than coerced, so an unexpected value cannot silently become a
#: claim the payload did not make.
_REMOTE_KNOWN = {"true", "false", "unknown"}


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _first(values: Any) -> str:
    """Remotive publishes some fields as a single string or a list.

    Either shape is legitimate, so both are accepted rather than assuming one.
    A list of one is the common case; a bare string is not a defect.
    """
    if isinstance(values, (list, tuple)):
        for item in values:
            text = _text(item)
            if text:
                return text
        return ""
    return _text(values)


def clean_html(raw: Any) -> str:
    """Flatten a description to plain text.

    Remotive returns HTML fragments. The dashboard is deliberately
    script-free, so anything executable must not survive into stored text.
    Tags are stripped rather than escaped so the text reads correctly; the
    result is never re-inserted as markup.
    """
    text = _text(raw)
    if not text:
        return ""
    import re

    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>", "\n\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    return re.sub(r"\n{3,}", "\n\n", text).strip()


@dataclass(frozen=True)
class ApiJob:
    """One job as the API described it, before adaptation."""

    external_id: str
    url: str
    title: str
    company: str
    description: str
    location: str
    candidate_requirements: str
    job_type: str
    remote: str
    tags: Tuple[str, ...]
    published_at: str
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)


def _remote_flag(value: Any) -> str:
    """Normalise the remote flag without inventing a claim.

    Booleans, ``"true"``/``"false"`` and the Remotive integers all appear in
    practice. Anything unrecognised becomes ``unknown`` rather than being
    forced to true or false, because this field drives eligibility.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    text = _text(value).casefold()
    if text in {"true", "1", "yes"}:
        return "true"
    if text in {"false", "0", "no"}:
        return "false"
    return "unknown"


def parse_response(body: str) -> List[ApiJob]:
    """Parse the Public API payload into jobs.

    Raises :class:`AccessError` on anything that is not the documented shape. A
    silent empty list would be indistinguishable from "Remotive has no jobs
    today", and that confusion is exactly what this project refuses to create:
    an unreadable response must be reported, not absorbed.
    """
    text = (body or "").strip()
    if not text:
        raise AccessError("Remotive returned an empty body")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        # Recorded as unreadable rather than empty. A changed or truncated
        # response must never read as a quiet day.
        raise AccessError(
            f"Remotive response was not valid JSON ({exc.msg} at position {exc.pos}); "
            "treating this as unreadable, not as zero jobs"
        ) from exc

    if not isinstance(payload, Mapping):
        raise AccessError("Remotive response was not a JSON object")

    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list):
        # No "jobs" key at all is a shape change, not an empty result set.
        raise AccessError(
            "Remotive response has no 'jobs' array; treating this as unreadable, "
            "not as zero jobs"
        )

    jobs: List[ApiJob] = []
    for entry in raw_jobs:
        if not isinstance(entry, Mapping):
            continue
        url = _text(entry.get("url"))
        title = _text(entry.get("title"))
        if not url or not title:
            # Unusable rows are skipped, but they are counted by the caller's
            # ledger through the fetched/rejected split, so a mass of them is
            # visible rather than silent.
            continue
        tags = entry.get("tags")
        tag_tuple = (
            tuple(_text(tag) for tag in tags if _text(tag))
            if isinstance(tags, list)
            else ()
        )
        jobs.append(
            ApiJob(
                external_id=_text(entry.get("id")),
                url=url,
                title=title,
                company=_first(entry.get("company_name")) or _text(entry.get("company")),
                description=clean_html(entry.get("description")),
                location=_text(entry.get("candidate_required_location")),
                candidate_requirements=_text(entry.get("candidate_requirements")),
                job_type=_text(entry.get("job_type")),
                remote=_remote_flag(entry.get("remote")),
                tags=tag_tuple,
                published_at=_text(entry.get("publication_date")),
                raw=dict(entry),
            )
        )
    return jobs


def record_to_job(record: Mapping[str, Any]) -> Job:
    """Adapt one parsed API entry to a :class:`Job`.

    The job id is the Remotive URL, which is stable and is the same identity the
    rest of the store uses for provenance. Remotive's numeric ``id`` is kept in
    the record but not used as identity: it would create a second id for the
    same vacancy if the URL ever changed.
    """
    from app.jobs.models import RemoteStatus

    remote_raw = _text(record.get("remote")) or "unknown"
    try:
        remote_status = RemoteStatus(remote_raw)
    except ValueError:
        # Preserved as unknown rather than coerced. Eligibility weighs this
        # field, so an unrecognised value must not become a false claim.
        remote_status = RemoteStatus.UNKNOWN

    return Job(
        job_id=_text(record.get("url")),
        title=_text(record.get("title")),
        company=_text(record.get("company")),
        url=_text(record.get("url")),
        location=_text(record.get("location")),
        description=_text(record.get("description")),
        remote_status=remote_status,
        # Remotive calls them ``tags``; the canonical field is ``skills``. The
        # rename happens at the boundary so the rest of the project sees one
        # name, and the original label is preserved in raw_excerpt.
        skills=list(record.get("tags") or ()),
        portal="remotive.com",
        posted_date=_text(record.get("posted")) or None,
        raw_excerpt={
            "remotive_id": _text(record.get("external_id")),
            "source_field_name": "tags",
            "job_type": _text(record.get("job_type")),
            "candidate_requirements": _text(record.get("candidate_requirements")),
            "attribution": _text(record.get("attribution")),
            "attribution_url": _text(record.get("attribution_url")),
        },
        description_complete=True,
    )


class RemotiveSourceAdapter:
    """A :class:`~app.jobs.adapters.SourceAdapter` over Remotive records.

    Parses only. Registering it performs no network access, so importing this
    module is safe in tests and on CI.
    """

    name = "remotive.com"
    consumed_keys = frozenset({
        "job_id", "title", "company", "url", "description", "location",
        "remote", "tags", "posted", "external_id", "job_type",
        "candidate_requirements", "attribution", "attribution_url",
    })

    def adapt(self, raw: Mapping[str, Any], *, now: datetime) -> Job:
        from app.jobs.adapters import AdaptError

        job = record_to_job(raw)
        if not job.url:
            raise AdaptError("Remotive record is missing a link")
        if not job.title:
            raise AdaptError(f"Remotive record has no title: {job.url!r}")
        if not job.job_id:
            job = replace(job, job_id=job.url)
        return job


def _prior_api_stamps(ledger: Ledger) -> Optional[List[float]]:
    """API request timestamps already on disk, inside the last day.

    Delegates to the shared reader so Remotive and MyJobMag cannot drift apart
    on how history is read - in particular on the fail-closed rule for an
    unreadable ledger, which Remotive previously handled on its own.
    """
    return prior_stamps(ledger, purpose="api")


def _seed_budget(ledger: Ledger, limit: int = DAILY_LIMIT) -> "_DailyBudget":
    """Build a budget seeded from the *persisted* ledger.

    Seeded from disk rather than from ``ledger.attempts``, which starts empty in
    every new process. Without this the limit would hold only inside a single
    run, and the first request after a restart would always be allowed.

    An unreadable ledger yields an untrusted budget that refuses every request:
    "cannot show the window is empty" is not evidence of compliance.
    """
    stamps = _prior_api_stamps(ledger)
    if stamps is None:
        return _DailyBudget(limit, untrusted=True)
    return _DailyBudget(limit, prior_stamps=stamps)


class RemotiveAdapter:
    """Fetches the Public API and turns it into ingestible records.

    Enforces the approved scope in code: one URL, one request per day, and a
    per-minute floor from the persisted ledger so both limits survive across
    processes.
    """

    name = "remotive.com"
    api_url = API_URL
    attribution_required = ATTRIBUTION_REQUIRED

    def __init__(
        self,
        fetcher: Optional[AccessFetcher] = None,
        *,
        source: str = "remotive.com",
        budget: Optional["_DailyBudget"] = None,
    ) -> None:
        if fetcher is None:
            fetcher = AccessFetcher(
                ledger=Ledger(),
                limiter=RateLimiter(MIN_INTERVAL),
                max_attempts=MAX_ATTEMPTS,
            )
        self._fetcher = fetcher
        self._source = source
        self._jobs: Optional[List[ApiJob]] = None
        # Seeded from the *persisted* ledger, not from this process's attempts.
        # A budget seeded from in-memory state would reset on every restart and
        # "one request per day" would be a claim the code did not keep.
        self._budget = budget or _seed_budget(self._fetcher.ledger, DAILY_LIMIT)

    @property
    def requests_made(self) -> int:
        return sum(
            1 for a in self._fetcher.ledger.attempts if a.purpose == "api"
        )

    def budget_state(self) -> Dict[str, Any]:
        """Whether this source may request now, and why not if it may not.

        Lets an orchestrator consult the persisted budget *before* building a
        fetch plan, so a refusal costs no network call and no attempt record.
        """
        return describe_budget(self._budget, source=self._source)

    def fetch(self) -> List[ApiJob]:
        """Fetch and parse. At most one HTTP request per day."""
        if self._jobs is None:
            if not self._budget.allow():
                raise AccessError(
                    "Remotive daily request limit reached; the approved scope "
                    "allows at most one API request per day"
                    if not self._budget.untrusted else
                    "Remotive request ledger exists but could not be read; "
                    "refusing rather than risk a request beyond the approved "
                    "one-per-day scope. Move the file aside to start a fresh "
                    "count."
                )
            response = self._fetcher.get(
                API_URL, source=self._source, purpose="api"
            )
            self._budget.record()
            self._jobs = parse_response(response.body)
        return self._jobs

    def end_run(self) -> None:
        self._jobs = None

    def to_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for item in self.fetch():
            records.append({
                "job_id": item.url,
                "external_id": item.external_id,
                "title": item.title,
                "company": item.company,
                "url": item.url,
                "description": item.description,
                "location": item.location,
                "remote": item.remote,
                "tags": list(item.tags),
                "posted": item.published_at,
                "job_type": item.job_type,
                "candidate_requirements": item.candidate_requirements,
                # Attached at ingest, not left to the renderer. Their terms make
                # attribution a condition of access, so it belongs in the record.
                "attribution": ATTRIBUTION_TEXT,
                "attribution_url": SOURCE_PAGE,
            })
        return records

    def access_evidence(self) -> Dict[str, str]:
        return {
            "approved_endpoint": f"{API_URL} (only this endpoint)",
            "permission": (
                "written permission from Remotive, recorded as repository-owner "
                "attestation (remotive_permission_evidence.md)"
            ),
            "robots": (
                "disallows /api/* for site crawlers; the grant resolves this "
                "for the Public API endpoint only"
            ),
            "cadence": (
                f"at most {DAILY_LIMIT} request per day "
                f"(stricter than the published 4/day; {MIN_INTERVAL:.0f}s floor)"
            ),
            "attribution_required": "true",
            "delay": "24 hours, applied by Remotive",
            "not_permitted": (
                "HTML scraping, job-detail pages, pagination, query variants, "
                "any other endpoint, redistribution to third-party boards, "
                "collecting signups or contact details"
            ),
        }


class _DailyBudget(DailyBudget):
    """The daily budget, named as this module has always named it.

    A thin name over the shared implementation in :mod:`app.sources.budget`, so
    the rule that survives a restart lives in exactly one place for every source
    carrying a daily cap, and the two cannot drift apart.
    """


# ``_now_from`` was removed with the shared budget. It only seeded the budget's
# clock from ``ledger.attempts``; the window is now seeded from the persisted
# ledger by :func:`_seed_budget`. Leaving it would imply the budget still reads
# per-process attempts.
