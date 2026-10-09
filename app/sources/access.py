"""Decide whether a source may be collected from, and fail closed when unsure.

The problem
-----------
``app/jobs/sources.py`` records a *declared* access decision. A declaration is
only as good as whoever wrote it, and until this module existed nothing
gathered the evidence a declaration is supposed to rest on. This layer fetches
robots.txt and the terms page - politely, a handful of times - and turns what it
finds into an :class:`AccessDecision`.

Fail closed
-----------
The asymmetry is deliberate and is the whole design:

* robots.txt missing, unreachable, unparseable, or returning HTML
  -> :data:`~app.sources.transport.AccessLevel.UNKNOWN`
* robots.txt disallows the paths we would collect
  -> :data:`~app.sources.transport.AccessLevel.RESTRICTED`
* terms that prohibit automated collection
  -> :data:`~app.sources.transport.AccessLevel.RESTRICTED`
* the server actively challenges automated clients
  -> :data:`~app_sources.transport.AccessLevel.RESTRICTED`
* robots allows, terms reviewed, nothing prohibits
  -> :data:`~app.sources.transport.AccessLevel.PERMITTED`

Anything short of the last is not permission. In particular ``unknown`` is never
treated as "probably fine": a site we could not read is a site we have not been
cleared for.

robots.txt permission is necessary but not sufficient
------------------------------------------------------
A site may allow a path in robots.txt and still refuse the request. Cloudflare
in particular serves robots.txt to everyone, including clients it then
challenges. So a permissive robots.txt is treated as *one input among several*
rather than the answer, and a server that challenges us is recorded as
``restricted`` even when robots.txt said yes.

Terms review
------------
Terms pages are frequently unreadable to a plain HTTP client. That is reported
as :data:`TERMS_UNREVIEWABLE` rather than guessed at. We do not infer terms
content from a 403, and we never try to work around one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

from app.sources.transport import (
    AccessError,
    AccessFetcher,
    HttpResponse,
    response_from_refusal,
)

# The RFC 9309 evaluator already exists in tools/robots_check.py. Reused rather
# than reimplemented: two divergent parsers of the same spec is how a fail-open
# gets shipped.
try:  # pragma: no cover - exercised by import
    from tools.robots_check import allowed as robots_allowed
    from tools.robots_check import is_robots_body
except Exception:  # pragma: no cover - tools/ not importable as a package
    robots_allowed = None
    is_robots_body = None


class AccessLevel(str, Enum):
    """What we concluded about a source."""

    PERMITTED = "permitted"
    RESTRICTED = "restricted"
    UNKNOWN = "unknown"


#: robots.txt verdict for a path we intend to collect.
ROBOTS_ALLOWS = "allows"
ROBOTS_DISALLOWS = "disallows"
ROBOTS_UNREADABLE = "unreadable"
ROBOTS_AMBIGUOUS = "ambiguous"

#: Terms review statuses.
TERMS_REVIEWED_CLEAR = "reviewed: no prohibition found"
TERMS_PROHIBITS = "reviewed: prohibits automated collection"
TERMS_UNREVIEWABLE = "unreviewable"
TERMS_NOT_ATTEMPTED = "not attempted"

#: Phrases that indicate a terms page forbids automated collection. Matched
#: case-insensitively against the visible text of the terms page only. This is
#: a conservative heuristic: it can miss a prohibition, which is why a page we
#: cannot read is UNREVIEWABLE rather than CLEAR.
PROHIBITION_MARKERS = (
    "scrap", "crawl", "spider", "robot", "automated means", "automatically collect",
    "data mining", "harvest",
)
PROHIBITION_CONTEXT = (
    "prohibit", "not permitted", "not allowed", "forbidden", "may not",
    "without our", "prohibited", "unauthorised", "unauthorized", "disallow",
)


class TermsUnavailable(RuntimeError):
    """The terms page could not be read."""


@dataclass(frozen=True)
class PathDecision:
    """The robots verdict for one path we would collect."""

    path: str
    allowed: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AccessDecision:
    """A recorded, evidence-backed access decision for one source."""

    source: str
    level: AccessLevel
    reason: str
    robots: str
    robots_url: str
    terms: str
    terms_url: str
    checked_at: str
    paths: Tuple[PathDecision, ...] = ()
    evidence: Tuple[str, ...] = ()

    @property
    def permitted(self) -> bool:
        return self.level is AccessLevel.PERMITTED

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["level"] = self.level.value
        payload["paths"] = [p.to_dict() for p in self.paths]
        payload["evidence"] = list(self.evidence)
        return payload

    def summary_line(self) -> str:
        return f"{self.source}: {self.level.value} - {self.reason}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def robots_url_for(base_url: str) -> str:
    parts = urlsplit(base_url)
    return urlunsplit((parts.scheme, parts.netloc, "/robots.txt", "", ""))


def join(base_url: str, path: str) -> str:
    parts = urlsplit(base_url)
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def check_paths(
    robots_text: str, paths: Sequence[str], agent: str = "*"
) -> List[PathDecision]:
    """Evaluate each path against robots.txt.

    Delegates to the existing RFC 9309 evaluator in ``tools/robots_check.py``.
    """
    if robots_allowed is None:  # pragma: no cover - defensive
        raise RuntimeError(
            "tools/robots_check.py is not importable; RFC 9309 evaluation "
            "cannot be delegated and must not be reimplemented here"
        )
    return [
        PathDecision(path=path, allowed=bool(robots_allowed(robots_text, agent, path)))
        for path in paths
    ]


def review_terms(text: str) -> Tuple[str, List[str]]:
    """Classify a terms page. Returns ``(status, evidence)``.

    Only ever called with text that was actually read. A page we could not fetch
    is UNREVIEWABLE and never reaches this function, because guessing what
    unread terms say is precisely the fabrication this project avoids.
    """
    lowered = text.casefold()
    hits: List[str] = []
    for marker in PROHIBITION_MARKERS:
        index = lowered.find(marker)
        if index < 0:
            continue
        # Require the prohibition language near the automation language, so a
        # page merely *describing* robots.txt is not read as a prohibition.
        window = lowered[max(0, index - 400): index + 400]
        if any(context in window for context in PROHIBITION_CONTEXT):
            hits.append(marker)
    if hits:
        return TERMS_PROHIBITS, [f"terms page mentions: {', '.join(sorted(set(hits)))}"]
    return TERMS_REVIEWED_CLEAR, []


def fetch_robots(fetcher: AccessFetcher, base_url: str, source: str) -> Tuple[Optional[str], str, List[str]]:
    """Fetch robots.txt. Returns ``(text_or_None, verdict, evidence)``."""
    url = robots_url_for(base_url)
    try:
        response = fetcher.get(url, source=source, purpose="robots")
    except AccessError as error:
        if error.status in (401, 403, 404, 410, 451):
            # A refused robots.txt is not an empty one. Either the site has no
            # public robots policy, or it declined to give us one; both mean we
            # have not been told we are welcome.
            return None, ROBOTS_UNREADABLE, [
                f"{url} returned HTTP {error.status} (refusal, not an empty policy)"
            ]
        return None, ROBOTS_UNREADABLE, [f"{url} unreachable: {error}"]

    if is_robots_body is not None and not is_robots_body(response.body):
        return None, ROBOTS_AMBIGUOUS, [
            f"{url} returned {response.status} with a body that is not a robots policy "
            f"(likely an HTML error or interstitial page)"
        ]
    return response.body, ROBOTS_ALLOWS, [f"{url} ({len(response.body)} bytes)"]


def fetch_terms(fetcher: AccessFetcher, base_url: str, source: str) -> Tuple[Optional[str], str, List[str]]:
    """Fetch the terms page. Returns ``(text_or_None, status, evidence)``.

    The path is a documented convention rather than a discovery crawl: we do
    not fetch the homepage looking for links, because that is crawling.
    """
    url = join(base_url, "/terms-and-conditions")
    try:
        response = fetcher.get(url, source=source, purpose="terms")
    except AccessError as error:
        refused = response_from_refusal(error, url)
        detail = f"HTTP {error.status}"
        if refused.challenged:
            detail += " with an anti-bot challenge"
        elif error.status == 404:
            detail += " (no terms page at the conventional path)"
        return None, TERMS_UNREVIEWABLE, [f"{url} unreadable: {detail}"]

    status, evidence = review_terms(response.body)
    return response.body, status, [f"{url} ({len(response.body)} bytes)"] + evidence


def verify_access(
    source: str,
    base_url: str,
    fetcher: AccessFetcher,
    *,
    paths: Sequence[str] = ("/jobs/", "/job/"),
    terms_path: str = "/terms-and-conditions",
) -> AccessDecision:
    """Verify one source and return its decision. Never raises for a refusal."""
    robots_text, robots_verdict, robots_evidence = fetch_robots(
        fetcher, base_url, source
    )
    terms_text, terms_status, terms_evidence = fetch_terms(
        fetcher, base_url, source
    )

    evidence: List[str] = list(robots_evidence) + list(terms_evidence)
    path_decisions: List[PathDecision] = []
    level = AccessLevel.UNKNOWN
    reason = ""

    if robots_verdict in (ROBOTS_UNREADABLE, ROBOTS_AMBIGUOUS):
        reason = (
            f"robots.txt was {robots_verdict}; without a readable policy we "
            f"cannot establish permission, so this source stays unknown"
        )
    else:
        path_decisions = check_paths(robots_text or "", paths)
        disallowed = [p.path for p in path_decisions if not p.allowed]

        if disallowed:
            level = AccessLevel.RESTRICTED
            reason = (
                f"robots.txt disallows the path(s) we would collect: "
                f"{', '.join(disallowed)}"
            )
        elif terms_status == TERMS_PROHIBITS:
            level = AccessLevel.RESTRICTED
            reason = (
                "the site's terms prohibit automated collection, which "
                "overrides a permissive robots.txt"
            )
        elif terms_status == TERMS_UNREVIEWABLE:
            reason = (
                "robots.txt permits the job paths, but the terms could not be "
                "reviewed and the server refused automated access; we have "
                "not been cleared to collect, so this stays unknown"
            )
        else:
            level = AccessLevel.PERMITTED
            reason = (
                f"robots.txt permits {', '.join(p.path for p in path_decisions)} "
                f"and the terms were reviewed with no prohibition found"
            )

    # A challenge is decisive regardless of what the documents say, and it
    # applies even when robots.txt was unreadable. Recorded last so it can
    # override a permissive robots.txt, and so a challenge we saw while
    # *reading* robots is not discarded as merely "unknown".
    #
    # Reasoning: an unreadable policy is an absence of information, whereas a
    # challenge is positive evidence that the site screens automated access.
    # Both stop us; only the second justifies calling it restricted.
    challenge = _challenge_seen(fetcher, source, base_url)
    if challenge:
        evidence.append(challenge)
        level = AccessLevel.RESTRICTED
        reason = (
            "the server presents an anti-bot challenge to automated clients; "
            "collecting would require bypassing an access control, which "
            "this project will not do"
        )

    return AccessDecision(
        source=source,
        level=level,
        reason=reason,
        robots=robots_verdict,
        robots_url=robots_url_for(base_url),
        terms=terms_status,
        terms_url=join(base_url, terms_path),
        checked_at=_now(),
        paths=tuple(path_decisions),
        evidence=tuple(evidence),
    )


def _challenge_seen(fetcher: AccessFetcher, source: str, base_url: str) -> str:
    """Was an anti-bot challenge observed on any attempt for this source?

    Read back from the ledger rather than re-requesting, so deciding the policy
    costs no additional network traffic. Only an explicit challenge counts - a
    plain 403 is a refusal, which is absence of permission rather than evidence
    that the site screens bots.
    """
    for attempt in reversed(fetcher.ledger.for_source(source)):
        if attempt.challenged:
            return (
                f"HTTP {attempt.status} with an anti-bot challenge on {attempt.url}"
            )
    return ""


# ----------------------------------------------------------------------
# persistence
# ----------------------------------------------------------------------


def decisions_path(data_dir: Path) -> Path:
    return Path(data_dir) / "access.json"


def load_decisions(data_dir: Path) -> Dict[str, AccessDecision]:
    """Load recorded access decisions, keyed by source."""
    path = decisions_path(data_dir)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # Corrupt evidence file means we have no decisions, which fails closed.
        return {}
    loaded: Dict[str, AccessDecision] = {}
    for entry in raw.get("decisions", []):
        try:
            loaded[entry["source"]] = AccessDecision(
                source=entry["source"],
                level=AccessLevel(entry["level"]),
                reason=entry.get("reason", ""),
                robots=entry.get("robots", ""),
                robots_url=entry.get("robots_url", ""),
                terms=entry.get("terms", ""),
                terms_url=entry.get("terms_url", ""),
                checked_at=entry.get("checked_at", ""),
                paths=tuple(PathDecision(**p) for p in entry.get("paths", [])),
                evidence=tuple(entry.get("evidence", [])),
            )
        except (KeyError, ValueError):
            continue
    return loaded


def save_decision(decision: AccessDecision, data_dir: Path) -> Path:
    """Record a decision, replacing any earlier one for the same source."""
    data_dir = Path(data_dir)
    loaded = load_decisions(data_dir)
    loaded[decision.source] = decision
    payload = {
        "updated_at": _now(),
        "decisions": [d.to_dict() for d in loaded.values()],
    }
    path = decisions_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------


def render_report(decisions: Sequence[AccessDecision]) -> str:
    """A human-readable access report."""
    lines = [
        "Access verification report",
        "=" * 60,
        "",
        "Policy: a source is usable only when its recorded access decision is",
        "'permitted'. Anything else - unknown, restricted, or absent - is not",
        "permission, and the source cannot become active.",
        "",
    ]
    for decision in decisions:
        lines.append(f"Source: {decision.source}")
        lines.append(f"  Decision       : {decision.level.value}")
        lines.append(f"  Reason         : {decision.reason}")
        lines.append(f"  robots.txt     : {decision.robots}")
        lines.append(f"  Terms review   : {decision.terms}")
        lines.append(f"  Checked at     : {decision.checked_at}")
        if decision.paths:
            lines.append("  Path decisions :")
            for path in decision.paths:
                verdict = "ALLOW" if path.allowed else "DISALLOW"
                lines.append(f"      {verdict:8} {path.path}")
        lines.append("  Evidence       :")
        for item in decision.evidence:
            lines.append(f"      - {item}")
        lines.append("")

    permitted = [d for d in decisions if d.permitted]
    lines.append("-" * 60)
    lines.append(
        f"{len(permitted)} of {len(decisions)} source(s) permitted."
    )
    lines.append(
        "Job discovery remains disabled regardless: enabling collection is a "
        "separate decision that is not made here."
    )
    return "\n".join(lines)