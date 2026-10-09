"""We Work Remotely: the first live source, collected over its public RSS feed.

Why this source
---------------
We Work Remotely was selected over the alternatives because it is the only
candidate whose permission chain needs no inference: ``robots.txt`` allows the
path, *and* the operator explicitly offers the feed for exactly this purpose on
a first-party page:

    "Looking to fill a remote job feed and want to use openings listed on
     WWR? No problem! Our public rss feed can help you with that. Anyone can
     use the feed, all we ask is that you attribute the links back to We Work
     Remotely."

    -- https://weworkremotely.com/remote-job-rss-feed

**Attribution is a condition of the permission, not a courtesy.** Every rendered
and exported view must credit WWR and link back to the original posting. The
dashboard enforces this; :data:`ATTRIBUTION_REQUIRED` exists so that requirement
is a value in the code rather than a convention.

Access constraints, honoured here
---------------------------------
* **One request per run.** The feed is fetched exactly once; there is no
  pagination and no category fan-out.
* **At least three seconds between requests** to this host, enforced by
  :class:`~app.sources.transport.RateLimiter`.
* **No disk caching.** The feed sends ``Cache-Control: max-age=0,
  must-revalidate``, so responses are used within the life of a single run and
  then discarded. See :class:`_RunCache`.
* **No retry after a refusal.** A 403 or a challenge is a decision, not a fault;
  the transport layer does not retry those.

Coverage, stated honestly
-------------------------
The feed currently yields **zero Kenya-eligible jobs**. 87 of 89 items are
labelled "Anywhere in the World", but each carries an explicit ``<country>``
list of accepted countries, and none of them include Kenya.

This is why :attr:`JobView`-style helpers here preserve the country list
verbatim instead of collapsing it to a friendly label: the eligibility layer
needs to see which countries are actually named. WWR is being integrated as a
**pipeline-validation source** - it proves the pipeline works end to end against
real data - not as sufficient coverage for a Kenya-based search.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from app.jobs.models import Job, RemoteStatus
from app.sources.transport import AccessFetcher, Ledger, RateLimiter

# NOTE: app.jobs.adapters imports this module to register WwrSourceAdapter, so
# importing it here at module scope would be circular. ``AdaptError`` is
# therefore imported lazily, inside the functions that raise it. That keeps
# ``import app.sources.wwr`` working on its own, which is how the source is
# used by tools.collect_wwr.


def _adapt_error():
    from app.jobs.adapters import AdaptError

    return AdaptError

#: The only URL this source ever requests.
FEED_URL = "https://weworkremotely.com/remote-jobs.rss"

#: The first-party page granting use of the feed. Cited as evidence for the
#: recorded access decision.
PERMISSION_URL = "https://weworkremotely.com/remote-job-rss-feed"

#: Verbatim from the page above. Quoted so the decision rests on the operator's
#: own words rather than a paraphrase.
PERMISSION_STATEMENT = (
    "Anyone can use the feed, all we ask is that you attribute the links back "
    "to We Work Remotely."
)

#: The condition attached to that permission. Enforced, not optional.
ATTRIBUTION_REQUIRED = True

#: Human-readable credit shown in every rendered view.
ATTRIBUTION_TEXT = "Listing from We Work Remotely"

#: Minimum seconds between requests to this host.
MIN_INTERVAL = 3.0

#: Sentinel used when a job is offered anywhere.
WORLDWIDE = "Anywhere in the World"

#: Country whose presence in a job's country list decides Kenya eligibility.
CANDIDATE_COUNTRY = "Kenya"

_NAMESPACE = "{http://www.w3.org/2005/Atom}"


def strip_html(raw: str) -> str:
    """Turn an RSS description into readable plain text.

    The feed double-escapes: the description arrives HTML-escaped *inside* the
    XML, so ``&lt;p&gt;`` is what ElementTree yields, and real tags appear as
    literal text. Decoding then stripping in that order is what produces prose
    instead of markup.

    Script and style content is removed rather than untagged, so a description
    cannot smuggle a stylesheet or a script into a rendered view.
    """
    if not raw:
        return ""
    text = html.unescape(raw)
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|li|h[1-6]|tr)>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    return text.strip()


def split_title(raw: str) -> tuple:
    """Split ``"Company: Role"`` into company and title.

    WWR prefixes the title with the company name. Falling back to an unsplit
    title when no prefix exists is honest - the company is recorded as unknown
    rather than guessed from the description.
    """
    text = (raw or "").strip()
    if ":" in text:
        company, _, title = text.partition(":")
        if company.strip() and title.strip():
            return company.strip(), title.strip()
    return "", text


def parse_countries(raw: str) -> List[str]:
    """Parse the ``<country>`` list into plain country names.

    Flag emoji are stripped so ``"🇰🇪 Kenya"`` becomes ``"Kenya"`` and the
    eligibility check is a plain substring test. The list is preserved exactly
    as published - a country absent from it is never inferred to be included.
    """
    if not raw:
        return []
    cleaned = re.sub(r"[\U0001F1E6-\U0001F1FF]", " ", raw)
    return [part.strip() for part in cleaned.split(",") if part.strip()]


def countries_mention(countries: Sequence[str], country: str) -> bool:
    """Is ``country`` exactly named in this job's accepted-country list?

    Whole-entry comparison, not substring. A substring test would report
    ``"Kenya"`` as present for an entry reading ``"Kenyan"``, and a false
    "Kenya is open" is exactly the inference this project must not make.
    """
    target = country.casefold().strip()
    return any(entry.casefold().strip() == target for entry in countries)


def parse_date(raw: str) -> Optional[str]:
    """``Fri, 09 Oct 2026 13:49:21 +0000`` -> ISO 8601."""
    if not raw:
        return None
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z"):
        try:
            return datetime.strptime(raw.strip(), fmt).isoformat()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(raw.strip()).isoformat()
    except ValueError:
        return None


@dataclass(frozen=True)
class FeedItem:
    """One parsed ``<item>``, before it becomes a :class:`Job`."""

    title: str
    company: str
    url: str
    description: str
    region: str
    countries: List[str]
    state: str
    skills: List[str]
    category: str
    employment_type: str
    posted: Optional[str]
    guid: str

    def to_job(self, *, portal: str = "weworkremotely") -> Job:
        """Map onto the canonical :class:`Job`.

        The country list is folded into ``location`` and ``region`` verbatim so
        the eligibility layer can see which countries are actually named.
        """
        countries = ", ".join(self.countries)
        location = self.region
        if self.state and self.state not in self.region:
            location = f"{self.region} ({self.state})" if self.region else self.state
        return Job(
            job_id=self.guid or self.url,
            title=self.title,
            company=self.company,
            url=self.url,
            description=self.description,
            location=location,
            region=self.region,
            country=countries or None,
            # Only the explicit global sentinel counts as worldwide. Anything
            # else keeps the default UNKNOWN rather than being upgraded on the
            # strength of the word "remote".
            remote_status=(
                RemoteStatus.FULLY_REMOTE_GLOBAL
                if self.region == WORLDWIDE
                else RemoteStatus.UNKNOWN
            ),
            portal=portal,
            posted_date=self.posted,
            skills=list(self.skills),
            # Full descriptions, unlike the snippet-only hiring.cafe feed.
            description_complete=True,
        )


def parse_feed(xml_text: str) -> List[FeedItem]:
    """Parse an RSS document into :class:`FeedItem` objects.

    Raises :class:`AdaptError` on a document that is not parseable RSS, so a
    captcha interstitial or an HTML error page fails loudly instead of
    yielding zero jobs and looking like an empty feed.
    """
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as error:
        raise _adapt_error()(f"response is not valid XML: {error}") from error

    items = root.findall(".//item")
    if not items:
        # Distinguish "no jobs today" from "this was not a feed at all".
        channel = root.find("channel")
        if channel is None:
            raise _adapt_error()("response has no RSS <channel>; not a feed document")
        return []

    parsed: List[FeedItem] = []
    for node in items:
        company, title = split_title(_text(node, "title"))
        guid = _text(node, "guid")
        link = _text(node, "link")
        parsed.append(FeedItem(
            title=title,
            company=company,
            url=link,
            description=strip_html(_text(node, "description")),
            region=_text(node, "region"),
            countries=parse_countries(_text(node, "country")),
            state=_text(node, "state"),
            skills=[s.strip() for s in _text(node, "skills").split(",") if s.strip()],
            category=_text(node, "category"),
            employment_type=_text(node, "type"),
            posted=parse_date(_text(node, "pubDate")),
            guid=guid or link,
        ))
    return parsed


def _text(node: ElementTree.Element, name: str) -> str:
    """Read a child element's text, unwrapping CDATA and entity escaping."""
    child = node.find(name)
    if child is None:
        return ""
    return "".join(child.itertext()).strip()


class _RunCache:
    """In-process cache, discarded when the run ends.

    The feed forbids caching (``max-age=0, must-revalidate``), so this exists
    only to guarantee that a single run issues exactly one request even if the
    cache is consulted more than once. It never writes to disk.
    """

    def __init__(self) -> None:
        self._items: Optional[List[FeedItem]] = None

    def get(self, produce) -> List[FeedItem]:
        if self._items is None:
            self._items = produce()
        return self._items

    def clear(self) -> None:
        self._items = None


def record_to_job(record: Mapping[str, Any]) -> Job:
    """Adapt one raw record into the canonical :class:`Job`.

    A free function rather than a method so it can be tested directly, without
    constructing a fetcher.
    """
    countries = [str(c) for c in (record.get("countries") or [])]
    region = str(record.get("location") or "")
    location = region
    state = str(record.get("state") or "")
    if state and state not in region:
        location = f"{region} ({state})" if region else state

    return Job(
        job_id=str(record.get("job_id") or record.get("url") or "").strip(),
        title=str(record.get("title") or "").strip(),
        company=str(record.get("company") or record.get("company_name") or "").strip(),
        url=str(record.get("url") or "").strip(),
        description=str(record.get("description") or ""),
        location=location,
        region=region,
        country=", ".join(countries) or None,
        # Only the explicit global sentinel counts as worldwide. Anything else
        # stays UNKNOWN rather than being upgraded on the word "remote".
        remote_status=(
            RemoteStatus.FULLY_REMOTE_GLOBAL
            if region == WORLDWIDE
            else RemoteStatus.UNKNOWN
        ),
        portal=str(record.get("portal") or "weworkremotely"),
        posted_date=record.get("posted") or None,
        skills=[str(s) for s in (record.get("skills") or [])],
        description_complete=True,
    )


class WwrSourceAdapter:
    """A :class:`~app.jobs.adapters.SourceAdapter` over WWR records.

    Registered so the shared :func:`~app.jobs.adapters.adapt_record` path can
    find it by name. It performs no network access - it only parses records it
    is handed - so importing this module is safe in tests and on CI.
    """

    name = "weworkremotely"
    consumed_keys = frozenset({
        "title", "company", "company_name", "url", "location", "state",
        "description", "skills", "countries", "posted", "portal", "category",
        "employment_type", "attribution", "attribution_url", "job_id",
    })

    def adapt(self, raw: Mapping[str, Any], *, now: datetime) -> Job:
        job = record_to_job(raw)
        if not job.url:
            raise _adapt_error()("WWR record is missing a url or guid")
        if not job.job_id:
            job = replace(job, job_id=job.url)
        if not job.title:
            raise _adapt_error()(f"WWR record has no title: {raw.get('url')!r}")
        return job


class WwrAdapter:
    """Fetches the WWR feed and turns it into ingestible records.

    Holds the access policy in code: one request per run, at least three
    seconds apart, no disk cache, no retry after a refusal.
    """

    name = "weworkremotely"
    feed_url = FEED_URL
    attribution_required = ATTRIBUTION_REQUIRED

    def __init__(
        self,
        fetcher: Optional[AccessFetcher] = None,
        *,
        source: str = "weworkremotely",
    ) -> None:
        if fetcher is None:
            # Policy-compliant defaults: slow, ledgered, bounded retries.
            fetcher = AccessFetcher(
                ledger=Ledger(),
                limiter=RateLimiter(MIN_INTERVAL),
                max_attempts=3,
            )
        self._fetcher = fetcher
        self._source = source
        self._cache = _RunCache()

    @property
    def requests_made(self) -> int:
        """How many HTTP requests this adapter issued, for the run ledger."""
        return sum(1 for a in self._fetcher.ledger.attempts
                   if a.purpose == "feed")

    def fetch(self) -> List[FeedItem]:
        """Fetch and parse the feed. Exactly one HTTP request per run."""

        def produce() -> List[FeedItem]:
            response = self._fetcher.get(
                FEED_URL, source=self._source, purpose="feed"
            )
            return parse_feed(response.body)

        return self._cache.get(produce)

    def end_run(self) -> None:
        """Discard the run cache so the next run refetches."""
        self._cache.clear()

    def to_records(self) -> List[Dict[str, Any]]:
        """Records in the shape ingestion expects.

        Shaped like the ``hiring.cafe`` fixtures so the same ingest path and the
        same store work unchanged. ``countries`` is carried through verbatim so
        the eligibility layer can see which countries are actually named.
        """
        return [
            {
                "job_id": item.guid or item.url,
                "title": item.title,
                "company": item.company,
                "company_name": item.company,
                "url": item.url,
                "location": item.region,
                "state": item.state,
                "description": item.description,
                "description_snippet": item.description,
                "work_mode": "Remote",
                "employment_type": item.employment_type,
                "category": item.category,
                "skills": list(item.skills),
                "countries": list(item.countries),
                "posted": item.posted,
                "portal": self.name,
                "attribution": ATTRIBUTION_TEXT,
                "attribution_url": PERMISSION_URL,
            }
            for item in self.fetch()
        ]

    def access_evidence(self) -> Dict[str, str]:
        """Evidence for the recorded :data:`permitted` decision."""
        return {
            "robots": (
                "User-agent: * / Allow: / with no rule covering "
                "/remote-jobs.rss (checked 2026-10-09)"
            ),
            "robots_url": "https://weworkremotely.com/robots.txt",
            "permission_statement": PERMISSION_STATEMENT,
            "permission_url": PERMISSION_URL,
            "attribution_required": "true",
            "access_method": "public RSS feed; no HTML scraping, no authentication",
            "rate_limits": (
                "none published; self-limited to one request per run with a "
                "3s minimum interval"
            ),
            "caching": (
                "Cache-Control: max-age=0, must-revalidate - no disk cache; "
                "in-process cache lives only for one run"
            ),
        }
