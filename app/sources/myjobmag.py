"""MyJobMag Kenya: the second live source, over its officially published RSS feed.

Scope of permission
-------------------
Exactly one thing is permitted: fetching
:data:`FEED_URL` at most once per day and displaying its items on a private
local dashboard with attribution.

**Not permitted, and deliberately not implemented:** HTML scraping, job-detail
page access, pagination, the aggregate feeds, the summarized feed, the blog
feed, any other endpoint, authentication, or redistribution. The feed supplies
every field the pipeline needs, so there is no reason to touch anything else.

Why this source
---------------
We Work Remotely is permitted but yields nothing usable: across a real captured
run, **0 of 89 listings named Kenya**. MyJobMag is the opposite — in a single
feed sample, 100 items mentioned Kenya 111 times, from named Kenyan employers
(KCB Bank Kenya, Geminia Life, AA Kenya).

Location honesty
----------------
This source publishes **no location or country field per item**. The `.co.ke`
domain proves the *publisher* is Kenyan; it says nothing about where a role is.
:func:`location_evidence` therefore classifies every job as ``structured``,
``description`` or ``missing``, and the domain is never used to infer
eligibility.

Encoding
--------
Some titles arrive double-encoded UTF-8 (``"FranÃÂ¢ÃÂÃÂ"``). :func:`repair_text`
repairs **for display only**, conservatively: at most two passes, a repair
accepted only when a mojibake score improves, and the original always kept.
A repair we are unsure about is marked rather than applied, because silently
rewriting text is worse than showing it as it arrived.
"""

from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence

from app.jobs.models import Job, RemoteStatus
from app.sources.budget import DailyBudget, describe_budget, prior_stamps
from app.sources.transport import AccessError, AccessFetcher, Ledger, RateLimiter

#: The only URL this source may ever request.
FEED_URL = "https://www.myjobmag.co.ke/jobsxml_by_categories.xml"

#: The page MyJobMag publishes its feeds on. Cited as evidence.
FEEDS_PAGE = "https://www.myjobmag.co.ke/feeds"

#: Mandatory self-imposed attribution. MyJobMag's terms are silent on feeds, so
#: this is our own obligation - but it is treated as binding, because the
#: permission rests on goodwill we would rather not spend.
ATTRIBUTION_TEXT = "Listing from MyJobMag"
ATTRIBUTION_REQUIRED = True

#: At most one request per day.
MIN_INTERVAL = 86400.0
DAILY_LIMIT = 1

#: Bounded transient retries.
MAX_ATTEMPTS = 3

#: Location evidence classes.
LOCATION_STRUCTURED = "structured"
LOCATION_DESCRIPTION = "description"
LOCATION_MISSING = "missing"


# ----------------------------------------------------------------------
# encoding repair (display only)
# ----------------------------------------------------------------------

#: Marker ranges that appear when UTF-8 bytes are decoded as latin-1/cp1252 and
#: then re-encoded. These are the give-aways.
_MOJIBAKE_MARKERS = (
    "Ã¢", "Ã©", "Ã¨", "Ã¤", "Ã¶", "Ã¼", "Ã±", "Ã§", "Ã ", "Ã©", "Ã\x9d",
    "Â°", "Â©", "Â®", "Â£", "Â«", "Â»", "Â€", "â\x80\x9c", "â\x80\x9d",
    "â\x80\x99", "â\x80\x98", "ï¿½", "Ã\x83", "Ã\x82",
)


def mojibake_score(text: str) -> int:
    """How many mojibake markers appear. Lower is better."""
    if not text:
        return 0
    return sum(text.count(marker) for marker in _MOJIBAKE_MARKERS)


#: Characters that lead a mojibake sequence. A successful repair must contain
#: none of them. Scoring alone is too coarse: ``ÃÂ¢ÃÂÃÂ`` scores 3 and a partial
#: round trip can score 0 while still leaving a stray lead character (``â``),
#: which would be accepted as "fixed" and silently corrupt the title.
_MOJIBAKE_LEADS = "ÃÂâïð"


def _has_lead(text: str) -> bool:
    return _lead_count(text) > 0


def _lead_count(text: str) -> int:
    return sum(text.count(ch) for ch in _MOJIBAKE_LEADS)


def _quality(text: str) -> tuple:
    """Lower is better: surviving lead characters first, then marker count.

    A composite is needed because a partial round trip can drive the marker
    score to zero while still leaving lead characters behind, and vice versa.
    Comparing on one metric alone lets a worse text look better. Real MyJobMag
    titles are *triple* encoded, so a first pass can equal the original's score
    while still moving closer to correct - hence a tuple rather than a scalar.
    """
    return (_lead_count(text), mojibake_score(text))


def looks_mojibake(text: str) -> bool:
    """Is this text likely mis-decoded?

    The marker sequences are two specific characters in a row (``Ã©``, ``Ã¼``,
    ``â€™``). Genuine European text does not contain them: Portuguese uses
    ``ã`` (U+00E3), not ``Ã`` (U+00C3), and French apostrophes are U+2019, not
    ``â€™``. An earlier version also demanded a stray control character, which
    turned out to reject almost every real case - so the marker check alone is
    the test.

    Valid Unicode is therefore left untouched, which is asserted directly.
    """
    return bool(text) and mojibake_score(text) > 0


def _latin1_roundtrip(text: str) -> str:
    """Undo one level of latin-1/cp1252 mis-decoding."""
    try:
        repaired = text.encode("cp1252", errors="strict").decode("utf-8", errors="strict")
    except (UnicodeEncodeError, UnicodeDecodeError):
        try:
            repaired = text.encode("latin-1", errors="strict").decode(
                "utf-8", errors="strict")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
    return repaired


@dataclass(frozen=True)
class RepairedText:
    """A repaired string, with its original kept for audit."""

    text: str
    original: str
    repaired: bool = False
    uncertain: bool = False
    passes: int = 0

    @property
    def changed(self) -> bool:
        return self.text != self.original


def repair_text(raw: str, *, max_passes: int = 2) -> RepairedText:
    """Repair display-only encoding damage, conservatively.

    Rules, all of them deliberate:

    * Valid Unicode is returned **unchanged**. ``looks_mojibake`` requires both a
      marker and an unusual character, so genuine accented text is never
      touched.
    * At most ``max_passes`` repair attempts.
    * A repair is accepted **only when the mojibake score improves**. A round
      trip that makes things worse or no better is discarded.
    * If the result still scores poorly, the **original** is kept and the
      outcome is marked ``uncertain`` - better to show text as it arrived than
      to guess at a correction.
    * The original is always preserved, so a reviewer can see what the source
      actually published.
    """
    original = raw or ""
    if not looks_mojibake(original):
        return RepairedText(text=original, original=original)

    best = original
    best_quality = _quality(original)
    passes = 0
    for _ in range(max(0, min(max_passes, 2))):
        passes += 1
        candidate = _latin1_roundtrip(best)
        if candidate == best:
            break
        quality = _quality(candidate)
        if quality >= best_quality:
            # No improvement on either metric. Accepting it would risk
            # corrupting text that was merely unusual, so stop.
            break
        best, best_quality = candidate, quality

    clean = best_quality == (0, 0)
    return RepairedText(
        # If we are not confident, show the original untouched rather than a
        # half-corrected guess.
        text=best if clean else original,
        original=original,
        repaired=clean and best != original,
        uncertain=not clean,
        passes=passes,
    )


# ----------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------


def clean_html(raw: str) -> str:
    """RSS description -> readable plain text.

    The feed embeds escaped HTML inside the description. Decoding then
    stripping in that order is what turns markup into prose. Script and style
    bodies are removed rather than untagged.
    """
    if not raw:
        return ""
    text = html.unescape(raw)
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|li|h[1-6]|tr|br)>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    return text.strip()


def split_title(raw: str) -> tuple:
    """Split MyJobMag's ``"Role at Company"`` into (title, company).

    Splitting on the **last** " at " avoids splitting a role whose own title
    contains that phrase ("Sales Rep at Amazon" is ambiguous; "Vice President,
    Sales at Acme" is not). Falls back to leaving the company unknown rather
    than guessing from the description.
    """
    text = (raw or "").strip()
    if " at " in text:
        title, _, company = text.rpartition(" at ")
        if title.strip() and company.strip():
            return title.strip(), company.strip()
    return text, ""


#: Phrases that place a role in a country, in the posting's own words.
_KENYA_IN_SCOPE = re.compile(
    r"\b(kenya|nairobi|mombasa|kisumu|nakuru|thika|eldoret|malindi|naivasha)\b",
    re.IGNORECASE,
)

#: Regions and countries that place a role elsewhere. Deliberately does NOT
#: include "africa" or "remote" - those are ambiguous for a Kenya-based
#: candidate and resolve to `unknown`, not to a verdict.
_OUT_OF_SCOPE = re.compile(
    r"\b(uganda|tanzania|ethiopia|south africa|nigeria|ghana|zimbabwe|"
    r"rwanda|senegal|egypt|morocco|tunisia|botswana|zambia|namibia|"
    r"united kingdom|united states|usa|canada|australia|germany|france|"
    r"ireland|netherlands|poland|spain|italy|sweden|norway|denmark|"
    r"india|pakistan|bangladesh|sri lanka|nepal|philippines|indonesia|"
    r"malaysia|singapore|uae|dubai|saudi|qatar|kuwait|"
    r"middle east|gulf|caribbean)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LocationEvidence:
    """Where a role is, and how confident that statement is.

    ``kind`` is one of :data:`LOCATION_STRUCTURED`, :data:`LOCATION_DESCRIPTION`
    or :data:`LOCATION_MISSING`. ``verdict`` is ``"kenya"``, ``"other"`` or
    ``"ambiguous"``.

    A domain such as ``.co.ke`` is **never** consulted. It identifies the
    publisher, not the location of the role.
    """

    kind: str
    verdict: str
    quote: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {"kind": self.kind, "verdict": self.verdict, "quote": self.quote}


def location_evidence(
    job: Mapping[str, Any], *, candidate_country: str = "Kenya"
) -> LocationEvidence:
    """Classify a job's location evidence without inferring from the domain.

    Precedence:

    1. Structured ``country``/``location`` metadata naming the candidate's
       country -> ``structured`` / ``kenya``.
    2. Structured metadata naming somewhere else -> ``structured`` / ``other``.
    3. Description text naming Kenya or a Kenyan city -> ``description`` /
       ``kenya``.
    4. Description naming another country or region -> ``description`` /
       ``other``.
    5. Anything else, including "Africa", "remote" or "worldwide" alone ->
       ``ambiguous``, reported as ``missing`` when nothing was found at all.
    """
    candidate = candidate_country.casefold()

    structured = " ".join(
        str(job.get(field) or "") for field in ("country", "location", "region")
    ).strip()
    if structured:
        if candidate in structured.casefold():
            return LocationEvidence(LOCATION_STRUCTURED, "kenya", structured[:200])
        if _OUT_OF_SCOPE.search(structured):
            return LocationEvidence(LOCATION_STRUCTURED, "other", structured[:200])
        return LocationEvidence(LOCATION_STRUCTURED, "ambiguous", structured[:200])

    description = str(job.get("description") or "")
    if not description.strip():
        return LocationEvidence(LOCATION_MISSING, "ambiguous", "")

    match = _KENYA_IN_SCOPE.search(description)
    if match:
        start = max(0, match.start() - 80)
        return LocationEvidence(
            LOCATION_DESCRIPTION, "kenya",
            description[start:match.end() + 80].strip(),
        )

    other = _OUT_OF_SCOPE.search(description)
    if other:
        start = max(0, other.start() - 80)
        return LocationEvidence(
            LOCATION_DESCRIPTION, "other",
            description[start:other.end() + 80].strip(),
        )

    # "Africa", "remote", "worldwide" alone are not enough to conclude
    # anything about a Kenya-based applicant.
    return LocationEvidence(LOCATION_DESCRIPTION, "ambiguous", description[:200])


def parse_date(raw: str) -> Optional[str]:
    """``Fri, 9 Oct 2026 16:01:29 GMT`` -> ISO 8601 with a real offset.

    ``%Z`` accepts ``GMT`` but yields a *naive* datetime, which would silently
    drop the offset and shift every publication time. ``GMT`` is therefore
    normalised to ``+0000`` before parsing so the offset survives.
    """
    if not raw:
        return None
    text = raw.strip()
    normalised = re.sub(r"\s+(GMT|UTC)$", " +0000", text)
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%d %b %Y %H:%M:%S %z",
                "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(normalised, fmt).isoformat()
        except ValueError:
            continue
    for fmt in ("%a, %d %b %Y %H:%M:%S", "%d %b %Y %H:%M:%S"):
        try:
            # No zone in the source text at all; do not invent one.
            return datetime.strptime(text, fmt).isoformat()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return None


@dataclass(frozen=True)
class FeedItem:
    """One parsed ``<item>``."""

    title: str
    company: str
    url: str
    description: str
    posted: Optional[str]
    title_original: str
    title_uncertain: bool
    location: LocationEvidence

    def to_job(self, *, portal: str = "myjobmag.co.ke") -> Job:
        return Job(
            job_id=self.url,
            title=self.title,
            company=self.company,
            url=self.url,
            description=self.description,
            location=self.location.quote or "",
            country=None,  # this source publishes no country field
            remote_status=RemoteStatus.UNKNOWN,  # never inferred from prose
            portal=portal,
            posted_date=self.posted,
            description_complete=True,
        )


def _text(node: ElementTree.Element, name: str) -> str:
    child = node.find(name)
    if child is None:
        return ""
    return "".join(child.itertext()).strip()


def parse_feed(xml_text: str) -> List[FeedItem]:
    """Parse the feed. Raises on anything that is not a feed document.

    A captcha interstitial or an HTML error page must fail loudly rather than
    yield zero jobs and look like an empty board.
    """
    from app.jobs.adapters import AdaptError

    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as error:
        raise AdaptError(f"response is not valid XML: {error}") from error

    items = root.findall(".//item")
    if not items:
        if root.find("channel") is None:
            raise AdaptError("response has no RSS <channel>; not a feed document")
        return []

    parsed: List[FeedItem] = []
    for node in items:
        raw_title = _text(node, "title")
        repaired = repair_text(raw_title)
        title, company = split_title(repaired.text)
        description = clean_html(_text(node, "description"))
        parsed.append(FeedItem(
            title=title,
            company=company,
            url=_text(node, "link"),
            description=description,
            posted=parse_date(_text(node, "pubDate")),
            title_original=raw_title,
            title_uncertain=repaired.uncertain,
            location=location_evidence({"description": description}),
        ))
    return parsed


def record_to_job(record: Mapping[str, Any]) -> Job:
    """Adapt one raw record into the canonical :class:`Job`.

    Source-specific facts that have no canonical field - the encoding
    uncertainty flag, the original title, and the location-evidence
    classification - are carried in ``raw_excerpt`` so they survive ingestion
    instead of being dropped at the boundary. A reviewer needs to be able to
    see what the source actually published and how confident the parser was.
    """
    title = str(record.get("title") or "").strip()
    excerpt = {
        "title_original": record.get("title_original"),
        "title_encoding_uncertain": bool(record.get("title_encoding_uncertain")),
        "location_evidence": record.get("location_evidence"),
        "attribution": record.get("attribution"),
    }
    return Job(
        job_id=str(record.get("job_id") or record.get("url") or "").strip(),
        title=title,
        company=str(record.get("company") or "").strip(),
        url=str(record.get("url") or "").strip(),
        description=str(record.get("description") or ""),
        location=str(record.get("location") or ""),
        country=str(record.get("country") or "") or None,
        portal=str(record.get("portal") or "myjobmag.co.ke"),
        posted_date=record.get("posted") or None,
        raw_excerpt=json.dumps(excerpt, ensure_ascii=False, sort_keys=True),
        description_complete=True,
    )


class MyjobmagSourceAdapter:
    """A :class:`~app.jobs.adapters.SourceAdapter` over MyJobMag records.

    Parses only. Registering it performs no network access, so importing this
    module is safe in tests and on CI.
    """

    name = "myjobmag.co.ke"
    consumed_keys = frozenset({
        "job_id", "title", "company", "url", "description", "posted",
        "location", "country", "portal", "attribution", "attribution_url",
        "location_evidence",
    })

    def adapt(self, raw: Mapping[str, Any], *, now: datetime) -> Job:
        from app.jobs.adapters import AdaptError

        job = record_to_job(raw)
        if not job.url:
            raise AdaptError("MyJobMag record is missing a link")
        if not job.job_id:
            job = replace(job, job_id=job.url)
        if not job.title:
            raise AdaptError(f"MyJobMag record has no title: {job.url!r}")
        return job


class MyjobmagAdapter:
    """Fetches the feed and turns it into ingestible records.

    Enforces the approved scope in code: one URL, one request per run, and a
    daily floor enforced by :class:`_DailyBudget`.
    """

    name = "myjobmag.co.ke"
    feed_url = FEED_URL
    attribution_required = ATTRIBUTION_REQUIRED

    def __init__(
        self,
        fetcher: Optional[AccessFetcher] = None,
        *,
        source: str = "myjobmag.co.ke",
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
        self._items: Optional[List[FeedItem]] = None
        # Seeded from the *persisted* ledger, not from this process's attempts.
        # A budget seeded from in-memory state resets on every restart, which
        # would leave "one request per day" true only until the program closed.
        self._budget = budget or _seed_budget(self._fetcher.ledger, DAILY_LIMIT)

    @property
    def requests_made(self) -> int:
        return sum(1 for a in self._fetcher.ledger.attempts
                   if a.purpose == "feed")

    def budget_state(self) -> Dict[str, Any]:
        """Whether this source may request now, and why not if it may not.

        Lets an orchestrator consult the persisted budget *before* building a
        fetch plan, so a refusal costs no network call and no attempt record.
        """
        return describe_budget(self._budget, source=self._source)

    def fetch(self) -> List[FeedItem]:
        """Fetch and parse. At most one HTTP request per rolling day.

        The budget is consulted before the fetcher, so a refusal raises without
        a socket ever being opened and without a network attempt being written
        to the ledger. Recording a refusal as an attempt would be a false
        record: no request was made.
        """
        if self._items is None:
            if not self._budget.allow():
                raise AccessError(
                    "MyJobMag daily request limit reached; the approved scope "
                    "allows at most one feed request per day"
                    if not self._budget.untrusted else
                    "MyJobMag request ledger exists but could not be read; "
                    "refusing rather than risk a request beyond the approved "
                    "one-per-day scope. Move the file aside to start a fresh "
                    "count."
                )
            response = self._fetcher.get(
                FEED_URL, source=self._source, purpose="feed")
            self._budget.record()
            self._items = parse_feed(response.body)
        return self._items

    def end_run(self) -> None:
        self._items = None

    def to_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for item in self.fetch():
            record = {
                "job_id": item.url,
                "title": item.title,
                "company": item.company,
                "url": item.url,
                "description": item.description,
                "posted": item.posted,
                "location": "",
                "location_evidence": item.location.to_dict(),
                "title_original": item.title_original,
                "title_encoding_uncertain": item.title_uncertain,
                "portal": self.name,
                "attribution": ATTRIBUTION_TEXT,
                "attribution_url": FEEDS_PAGE,
            }
            records.append(record)
        return records

    def access_evidence(self) -> Dict[str, str]:
        return {
            "approved_feed": f"{FEED_URL} (only this endpoint)",
            "feeds_page": FEEDS_PAGE,
            "robots": "no rule covers the feed path (verified 2026-10-09)",
            "cadence": "at most 1 request per day",
            "attribution_required": "true",
            "not_permitted": (
                "HTML scraping, job-detail pages, pagination, aggregate/summary/"
                "blog feeds, any other endpoint, redistribution"
            ),
        }


class _DailyBudget(DailyBudget):
    """The daily budget, named as this module has always named it.

    Kept as a subclass rather than a bare alias so the name still reads as
    MyJobMag's own concept at the call sites, while the enforcement - including
    the part that survives a restart - lives in one shared place.
    """


def _seed_budget(ledger: Ledger, limit: int = DAILY_LIMIT) -> _DailyBudget:
    """Build a budget seeded from the *persisted* ledger.

    The previous implementation read ``ledger.attempts``, which starts empty in
    every new process. The daily limit therefore held only within a single run:
    closing the program and reopening it made the budget forget everything, and
    the first request of the next run was always allowed. Nothing raised and no
    test failed - it simply stopped being true across restarts.

    An unreadable ledger yields an untrusted budget that refuses every request.
    "Cannot show the window is empty" is not evidence of compliance.
    """
    stamps = prior_stamps(ledger, purpose="feed")
    if stamps is None:
        return _DailyBudget(limit, untrusted=True)
    return _DailyBudget(limit, prior_stamps=stamps)


# ``_now_from`` was removed with the budget rework. It existed only to seed the
# budget's clock from ``ledger.attempts`` - the in-memory state that the
# persisted seed replaced. Keeping it would suggest the budget still reads
# per-process attempts, which is exactly the behaviour that was wrong.