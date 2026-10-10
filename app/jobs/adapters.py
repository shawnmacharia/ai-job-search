"""Source-record adapters: turn whatever a source emitted into a canonical ``Job``.

The contract this module exists to fix
--------------------------------------
``app/orchestrator/scrapers/hiring_cafe.py`` emits ``description_snippet``,
``posted``, ``salary`` and ``work_mode``. ``normalize_job`` reads
``description`` and ``date``. Between the two, eight of eleven emitted fields
were discarded and every downstream stage - eligibility, matching, drafting -
received an empty description. The store faithfully persisted nothing.

An adapter is the seam. It takes the raw record a source produced and either
returns a fully populated ``Job`` or raises :class:`AdaptError`. It never
returns a half-built record, so "no description" cannot pass silently into
storage.

Guarantees
----------
* **Nothing is silently discarded.** Every key in the raw record is either
  mapped onto a canonical ``Job`` field or preserved verbatim in
  ``Job.raw_excerpt``. :data:`SourceAdapter.consumed_keys` makes that
  checkable by tests, so a future scraper field fails a test rather than
  disappearing.
* **Loud failure.** A record missing title, company, url or description raises
  :class:`AdaptError`, which the caller quarantines (see
  ``data/rejected.jsonl``).
* **Deterministic.** No model calls, no network. The clock is injected via
  ``now=`` so relative dates such as "2d ago" are reproducible in tests.

Currency ambiguity
------------------
A bare ``$`` is *not* resolved to USD. It is equally likely to be AUD, CAD or
SGD, and silently guessing would put a wrong currency in front of a
compensation filter. ``salary_currency`` is left ``None`` for a bare ``$`` and
set only for unambiguous symbols: ``€`` EUR, ``£`` GBP, ``₹`` INR, and the
``KSh``/``KES`` prefix for Kenyan shillings. The numbers are still parsed.

Adding a source
---------------
Subclass :class:`SourceAdapter`, declare :attr:`name` and
:attr:`consumed_keys`, implement :meth:`adapt`, and register it. No other
module needs to change.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Protocol

from app.jobs.models import Job
from app.jobs.normalize import classify_remote
from app.jobs.store import normalize_url


class AdaptError(ValueError):
    """A source record cannot become a ``Job``.

    ``reason`` is human-readable and is what the caller writes to
    ``data/rejected.jsonl`` alongside the offending record.
    """

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# salary parsing
# ---------------------------------------------------------------------------

#: Symbols that map to exactly one currency. Deliberately excludes "$".
_UNAMBIGUOUS_SYMBOLS = {"€": "EUR", "£": "GBP", "₹": "INR", "¥": "JPY"}

#: Written currency names/codes we recognise unambiguously.
_CURRENCY_WORDS = {
    "eur": "EUR", "euro": "EUR", "euros": "EUR",
    "gbp": "GBP", "pound": "GBP", "pounds": "GBP",
    "inr": "INR", "rupee": "INR", "rupees": "INR",
    "jpy": "JPY", "yen": "JPY",
    "kes": "KES", "ksh": "KES",
    "ngn": "NGN", "naira": "NGN",
    "zar": "ZAR", "rand": "ZAR",
    "usd": "USD",
    "aud": "AUD", "cad": "CAD", "sgd": "SGD",
}

#: 80k -> 80_000 ; 1.2M -> 1_200_000 ; 95,000 -> 95_000
_NUMBER_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*([kKmM])?(?![a-zA-Z0-9])")
_SCALE = {"k": 1_000, "m": 1_000_000}

_PERIOD_PATTERNS = (
    ("year", r"/\s*yr\b|/\s*year|per\s+year|annually|per\s+annum|\bannum\b|\bp\.?a\.?\b|\ba\s+year\b|yearly"),
    ("month", r"/\s*mo\b|/\s*month|per\s+month|monthly|\bp\.?m\.?\b|a\s+month"),
    ("hour", r"/\s*hr\b|/\s*hour|hourly"),
    ("day", r"/\s*day|daily"),
    ("week", r"/\s*wk\b|/\s*week|weekly"),
)


def _parse_amount(digits: str, suffix: Optional[str]) -> Optional[int]:
    """Turn one numeric token into an integer amount.

    A dot followed by exactly three digits and a short integer part is read as
    a thousands separator, not a decimal: ``3.500`` is 3500, not 3.5. Salary
    strings are written both ways and the two cannot be told apart in
    general, so the grouped reading wins - it is the one that occurs in real
    postings far more often, and rounding it as a decimal is a 1000x error.
    ``1.2M`` is unaffected because the ``M`` suffix marks it as a decimal.

    This mirrors the guard added to ``tools/convert_salary_excel.py`` (see
    CHANGELOG #326), which had the same failure mode.
    """
    cleaned = digits.replace(",", "")
    if "." in cleaned:
        head, _, frac = cleaned.partition(".")
        if suffix is None and len(frac) == 3 and 1 <= len(head) <= 3 and head.isdigit():
            return int(head + frac)
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if suffix:
        value *= _SCALE[suffix.casefold()]
    return int(round(value))


def _detect_currency(text: str) -> Optional[str]:
    """Return an unambiguous currency code, or ``None`` when it is ambiguous.

    A bare ``$`` returns ``None`` on purpose - see the module docstring.
    """
    for symbol, code in _UNAMBIGUOUS_SYMBOLS.items():
        if symbol in text:
            return code
    lowered = text.casefold()
    for word, code in _CURRENCY_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            return code
    return None


def parse_salary(text: Optional[str]) -> Dict[str, Optional[Any]]:
    """Parse a free-text salary string into canonical salary fields.

    Returns a mapping with ``salary_min``, ``salary_max``, ``salary_currency``
    and ``salary_period``. Any of them may be ``None``: an unparseable or
    unstated salary yields ``None`` values rather than an error, because a
    missing salary must never reject a job.
    """
    empty: Dict[str, Optional[Any]] = {
        "salary_min": None,
        "salary_max": None,
        "salary_currency": None,
        "salary_period": None,
    }
    if not text or not str(text).strip():
        return empty

    raw = str(text)
    amounts: List[int] = []
    for match in _NUMBER_RE.finditer(raw):
        amount = _parse_amount(match.group(1), match.group(2))
        if amount is not None and amount > 0:
            amounts.append(amount)
    if not amounts:
        return empty

    lowered = raw.casefold()
    period = None
    for name, pattern in _PERIOD_PATTERNS:
        if re.search(pattern, lowered):
            period = name
            break

    return {
        "salary_min": min(amounts),
        "salary_max": max(amounts),
        "salary_currency": _detect_currency(raw),
        "salary_period": period,
    }


# ---------------------------------------------------------------------------
# posted-date parsing
# ---------------------------------------------------------------------------

_RELATIVE_RE = re.compile(r"^(\d+)\s*(h|hour|hours|d|day|days|w|week|weeks|m|month|months|y|year|years)\s+ago$")
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

_RELATIVE_UNITS = {
    "h": "hours", "hour": "hours", "hours": "hours",
    "d": "days", "day": "days", "days": "days",
    "w": "weeks", "week": "weeks", "weeks": "weeks",
    "m": "days", "month": "days", "months": "days",  # approximate: 30 days
    "y": "days", "year": "days", "years": "days",     # approximate: 365 days
}


def parse_posted_date(text: Optional[str], *, now: datetime) -> Optional[str]:
    """Convert a source's posted-date string into an ISO ``YYYY-MM-DD``.

    Handles relative forms ("2d ago", "3h ago", "1w ago"), "today", and an
    absolute ``YYYY-MM-DD``. Returns ``None`` for anything else; the caller
    keeps the original string in ``Job.posted_raw``.
    """
    if not text or not str(text).strip():
        return None
    raw = str(text).strip()
    reference = now.astimezone(timezone.utc)

    iso = _ISO_DATE_RE.search(raw)
    if iso:
        return f"{iso.group(1)}-{iso.group(2)}-{iso.group(3)}"

    lowered = raw.casefold()
    if lowered in {"today", "just posted", "new"}:
        return reference.date().isoformat()
    if lowered in {"yesterday"}:
        return (reference - timedelta(days=1)).date().isoformat()

    relative = _RELATIVE_RE.match(lowered)
    if relative:
        amount = int(relative.group(1))
        unit = _RELATIVE_UNITS[relative.group(2)]
        return (reference - timedelta(**{unit: amount})).date().isoformat()

    return None


# ---------------------------------------------------------------------------
# adapter contract
# ---------------------------------------------------------------------------


class SourceAdapter(Protocol):
    """The contract every source adapter implements."""

    #: Stable identifier for the source, stored as ``Job.portal``.
    name: str

    #: Keys this adapter reads. Anything else in a raw record is preserved
    #: verbatim in ``Job.raw_excerpt`` rather than dropped.
    consumed_keys: frozenset

    def adapt(self, raw: Mapping[str, Any], *, now: datetime) -> Job:
        """Return a fully populated ``Job`` or raise :class:`AdaptError`."""


def _require(raw: Mapping[str, Any], key: str) -> str:
    value = str(raw.get(key) or "").strip()
    if not value:
        raise AdaptError(f"missing required field: {key}")
    return value


class HiringCafeAdapter:
    """Adapter for the cards emitted by ``hiring_cafe._parse_card_text``.

    The card text is a *snippet*, so ``description_complete`` is always
    ``False``: downstream stages can tell a short description from a whole
    posting instead of assuming either.
    """

    name = "hiring.cafe"
    portal = "hiring.cafe"
    consumed_keys = frozenset({
        "title", "company", "url", "description_snippet",
        "location", "work_mode", "salary", "posted",
    })

    def adapt(self, raw: Mapping[str, Any], *, now: datetime) -> Job:
        if not isinstance(raw, Mapping):
            raise AdaptError("record is not a mapping")

        title = _require(raw, "title")
        company = _require(raw, "company")
        url = _require(raw, "url")
        description = str(raw.get("description_snippet") or "").strip()
        if not description:
            # Never hand a downstream stage an empty description. Quarantine
            # this record instead; the caller keeps it in rejected.jsonl.
            raise AdaptError("missing required field: description_snippet")

        url_key = normalize_url(url)
        location = str(raw.get("location") or "").strip() or None
        work_mode = str(raw.get("work_mode") or "").strip()

        posted_raw = str(raw.get("posted") or "").strip() or None
        salary = parse_salary(raw.get("salary"))

        remote_text = " ".join(
            part for part in (location, work_mode, description) if part
        )

        return Job(
            job_id=hashlib.sha256(url_key.encode("utf-8")).hexdigest()[:16],
            title=title,
            company=company,
            url=url,
            description=description,
            location=location,
            remote_status=classify_remote(remote_text),
            salary_min=salary["salary_min"],
            salary_max=salary["salary_max"],
            salary_currency=salary["salary_currency"],
            salary_period=salary["salary_period"],
            portal=self.portal,
            posted_date=parse_posted_date(posted_raw, now=now),
            raw_excerpt={
                key: value
                for key, value in raw.items()
                if key not in self.consumed_keys
            },
            posted_raw=posted_raw,
            description_complete=False,
        )


_REGISTRY: Dict[str, SourceAdapter] = {}


def register(adapter: SourceAdapter) -> SourceAdapter:
    """Register ``adapter`` so :func:`adapt_record` can find it by name."""
    _REGISTRY[adapter.name] = adapter
    return adapter


register(HiringCafeAdapter())

# We Work Remotely is registered but, unlike hiring.cafe, importing this module
# performs no network access: its adapter only parses records it is handed. The
# fetch happens in the caller, through app.sources.wwr.
from app.sources.wwr import WwrSourceAdapter  # noqa: E402

register(WwrSourceAdapter())

# MyJobMag is registered for the same reason: parsing only, no network on
# import. The fetch happens in the caller through app.sources.myjobmag.
from app.sources.myjobmag import MyjobmagSourceAdapter  # noqa: E402

register(MyjobmagSourceAdapter())

# Remotive is registered for the same reason: parsing only, no network on
# import. The fetch happens in the caller through app.sources.remotive, which
# is the only place that knows the approved endpoint and its daily limit.
from app.sources.remotive import RemotiveSourceAdapter  # noqa: E402

register(RemotiveSourceAdapter())


def adapt_record(source: str, raw: Mapping[str, Any], *, now: Optional[datetime] = None) -> Job:
    """Adapt ``raw`` from ``source`` into a canonical ``Job``.

    Raises :class:`AdaptError` for an unknown source or an unusable record, so
    a caller can quarantine the record and keep going.
    """
    adapter = _REGISTRY.get(source)
    if adapter is None:
        raise AdaptError(f"unknown source: {source!r}")
    return adapter.adapt(raw, now=now or datetime.now(timezone.utc))
