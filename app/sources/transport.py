"""Rate-limited HTTP for access verification only.

Scope
-----
This module exists to check whether a source *may* be collected from. It is not
a crawler: it is deliberately the only place in the codebase that performs
network requests, and it is designed so that using it well means using it a
handful of times.

What it will not do
-------------------
* **No parallelism.** Every call takes a turn; :class:`RateLimiter` enforces a
  minimum interval between requests and nothing runs concurrently.
* **No retry after a refusal.** A ``401``, ``403``, ``404`` or ``451`` is a
  decision by the server, not a transient fault. Retrying it would be
  hammering a site that just said no - and working around a bot challenge is a
  hard stop for this project, so those are reported, not circumvented.
* **No credential handling.** No cookies, no authentication, no secrets. The
  only identifying header is the single user agent, and the only thing it may
  carry beyond the product token is a contact the operator chose to publish.
  That contact is read from a gitignored file and never from source, so no
  personal detail enters the repository - see :data:`CONTACT_PATH`.
* **No unbounded retries.** Attempts are capped and backed off.

Retries happen only for genuinely transient conditions: a timeout, a
connection error, ``5xx``, or ``429`` with a ``Retry-After`` to honour.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence


#: The product token. Tracked in source and deliberately free of any personal
#: detail, so it can live in a public repository without publishing the
#: operator's contact.
USER_AGENT_PRODUCT = "ai-job-search-access-check/1.0"

#: Optional operator contact, read from a gitignored file. Absent by default,
#: in which case the User-Agent is the bare product token.
CONTACT_PATH = Path(__file__).resolve().parents[2] / "data" / "contact.json"

#: Anything a server might interpret as a header terminator. A User-Agent is a
#: single header value; a newline in one would let a crafted contact field
#: inject arbitrary headers into every request.
_CONTROL = re.compile(r"[\r\n\x00-\x1f\x7f]")

#: Generous enough for a name plus a contact, tight enough to catch a pasted
#: paragraph being mistaken for a contact field.
_MAX_FIELD = 120


def _clean_field(value: Any) -> str:
    """Sanitise one operator-supplied string, or return "" to drop it.

    Silently discarding rather than raising: this file is a convenience, and a
    malformed one must never stop the tool from running. The worst outcome of a
    bad value is a bare product token, which is a valid User-Agent.
    """
    if not isinstance(value, str):
        return ""
    cleaned = _CONTROL.sub("", value).strip()
    return cleaned[:_MAX_FIELD] if cleaned else ""


def _operator_comment() -> str:
    """The parenthesised comment describing who is behind this tool.

    Read at import time so the result is a single module constant, which is
    what makes the guarantee meaningful: one string, resolved once, used for
    every request. Reading it per-request would reintroduce exactly the
    variability that a constant User-Agent exists to prevent.
    """
    try:
        raw = CONTACT_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError):
        return ""
    if not isinstance(data, Mapping):
        return ""
    purpose = _clean_field(data.get("purpose"))
    contact = _clean_field(data.get("contact"))
    parts = [part for part in (purpose, f"contact: {contact}" if contact else "") if part]
    return "; ".join(parts)


_comment = _operator_comment()
USER_AGENT = f"{USER_AGENT_PRODUCT} ({_comment})" if _comment else USER_AGENT_PRODUCT

#: Statuses that mean "come back later", not "you are refused".
RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})

#: Statuses that mean the server made a decision. Never retried.
REFUSAL_STATUSES = frozenset({401, 402, 403, 404, 410, 451})


class AccessError(RuntimeError):
    """A verification request did not succeed."""

    def __init__(self, message: str, *, status: Optional[int] = None,
                 retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass(frozen=True)
class HttpResponse:
    """One response, with the headers access decisions actually depend on."""

    url: str
    status: int
    headers: Mapping[str, str]
    body: str
    elapsed_ms: int

    def header(self, name: str) -> str:
        """Case-insensitive header lookup."""
        lowered = name.casefold()
        for key, value in self.headers.items():
            if key.casefold() == lowered:
                return value
        return ""

    @property
    def challenged(self) -> bool:
        """Is this an anti-bot challenge rather than an ordinary refusal?

        Cloudflare sets ``Cf-Mitigated: challenge`` when it is presenting an
        interstitial. A plain 403 is a refusal; a challenge is an explicit
        statement that automated access is being screened. Both stop us; only
        the second tells us the site actively wants to keep bots out.
        """
        return self.header("cf-mitigated").casefold() == "challenge"

    @property
    def refusal(self) -> bool:
        return self.status in REFUSAL_STATUSES

    def retry_after_seconds(self) -> Optional[float]:
        """``Retry-After`` as seconds, when it is a plain integer."""
        raw = self.header("retry-after").strip()
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            # An HTTP-date form is legal but not worth a clock dependency here;
            # falling back to the normal backoff is conservative.
            return None


@dataclass
class Attempt:
    """One verification attempt, recorded whatever the outcome."""

    source: str
    at: str
    url: str
    purpose: str
    outcome: str
    status: Optional[int] = None
    duration_ms: int = 0
    error: str = ""
    #: True only for an explicit anti-bot challenge, not for an ordinary
    #: refusal. Recorded separately because the two mean different things: a
    #: refusal says "no", a challenge says "we screen automated clients".
    challenged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Ledger:
    """Append-only record of every verification attempt.

    Written as JSON lines so a partially written run still leaves a readable
    trail, and so an interrupted verification cannot destroy the evidence for
    the attempts that already happened.
    """

    def __init__(self, path: Optional[Any] = None) -> None:
        self.path = path
        self.attempts: List[Attempt] = []

    def record(self, attempt: Attempt) -> None:
        self.attempts.append(attempt)
        if self.path is None:
            return
        from pathlib import Path

        target = Path(self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8", newline="") as handle:
            handle.write(json.dumps(attempt.to_dict(), sort_keys=True) + "\n")
            handle.flush()

    def for_source(self, source: str) -> List[Attempt]:
        return [a for a in self.attempts if a.source == source]

    def prior_attempts(self) -> List[Attempt]:
        """Attempts already on disk, oldest first.

        ``__init__`` deliberately starts empty so a run never inherits another
        run's in-memory state, which is why this exists as an explicit read.
        Unreadable lines are skipped rather than fatal: the ledger is an
        append-only trail, and a torn final line must not hide the attempts
        that were recorded successfully before it.

        This is what makes a rate limit survive a restart. A limiter or a
        counter seeded only from ``self.attempts`` would happily allow the first
        request of every new process, however recently the last one was made.
        """
        if self.path is None:
            return []
        from pathlib import Path

        target = Path(self.path)
        if not target.exists():
            return []
        loaded: List[Attempt] = []
        try:
            with target.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        loaded.append(Attempt(**data))
                    except (json.JSONDecodeError, TypeError):
                        continue
        except OSError:
            # No readable history means no prior requests can be proven. A
            # caller enforcing a limit must treat that as "unknown", not as
            # "none happened" - see the budget in app.sources.remotive.
            return []
        return loaded


class RateLimiter:
    """Enforces a minimum interval between requests.

    ``clock`` and ``sleeper`` are injected so tests can prove the delay
    happens without actually waiting.
    """

    def __init__(
        self,
        min_interval: float = 2.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = max(0.0, float(min_interval))
        self._clock = clock
        self._sleep = sleeper
        self._last: Optional[float] = None

    def wait(self) -> float:
        """Block until the next request is allowed. Returns seconds slept."""
        if self._last is None:
            self._last = self._clock()
            return 0.0
        elapsed = self._clock() - self._last
        remaining = self.min_interval - elapsed
        if remaining > 0:
            self._sleep(remaining)
        self._last = self._clock()
        return max(0.0, remaining)

    def penalise(self, seconds: float) -> None:
        """Push the next allowed time out, e.g. after a ``Retry-After``.

        The last-request timestamp is moved *forward* into the future, so the
        next :meth:`wait` sees negative elapsed time and sleeps for the full
        penalty. Moving it backwards - the intuitive-looking mistake - would
        instead look like plenty of time had already passed, and the penalty
        would be silently discarded.
        """
        self._last = self._clock() + max(0.0, seconds)


class AccessFetcher:
    """Fetches a URL politely, with bounded retries, recording every attempt."""

    def __init__(
        self,
        *,
        ledger: Optional[Ledger] = None,
        limiter: Optional[RateLimiter] = None,
        max_attempts: int = 3,
        backoff_base: float = 1.0,
        jitter: Callable[[float], float] = lambda value: value,
        opener: Optional[Callable[[str, float], HttpResponse]] = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        user_agent: str = USER_AGENT,
        timeout: float = 20.0,
    ) -> None:
        self.ledger = ledger or Ledger()
        self.limiter = limiter or RateLimiter()
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_base = backoff_base
        self.jitter = jitter
        self._opener = opener or _urllib_open
        self._clock = clock
        self._sleep = sleeper
        self.user_agent = user_agent
        self.timeout = timeout

    def get(self, url: str, *, source: str, purpose: str) -> HttpResponse:
        """GET one URL. Raises :class:`AccessError` when it cannot be had."""
        last_error: Optional[AccessError] = None

        for attempt_number in range(1, self.max_attempts + 1):
            self.limiter.wait()
            started = self._clock()
            try:
                response = self._opener(url, self.timeout)
                elapsed_ms = int((self._clock() - started) * 1000)
                self._record(source, url, purpose, "ok", response.status,
                             elapsed_ms, challenged=response.challenged)
                return response

            except AccessError as error:
                elapsed_ms = int((self._clock() - started) * 1000)
                status = error.status

                if status is not None and status in REFUSAL_STATUSES:
                    # A decision, not a fault. Recorded and returned to the
                    # caller as a refusal; retrying would be hammering.
                    self._record(source, url, purpose, "refused", status,
                                 elapsed_ms, str(error),
                                 challenged=_was_challenged(error))
                    raise

                self._record(source, url, purpose, "transient_failure", status,
                             elapsed_ms, str(error))
                last_error = error

                if not error.retryable or attempt_number == self.max_attempts:
                    break

                delay = self.jitter(self.backoff_base * (2 ** (attempt_number - 1)))
                retry_after = getattr(error, "retry_after", None)
                if retry_after:
                    self.limiter.penalise(retry_after)
                else:
                    self._sleep(delay)

        self._record(source, url, purpose, "failed", None, 0,
                     str(last_error) if last_error else "unknown error")
        raise last_error or AccessError(f"{url}: could not be fetched")

    def _record(self, source: str, url: str, purpose: str, outcome: str,
                status: Optional[int], duration_ms: int, error: str = "",
                challenged: bool = False) -> None:
        self.ledger.record(Attempt(
            source=source,
            at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            url=url, purpose=purpose, outcome=outcome, status=status,
            duration_ms=duration_ms, error=error, challenged=challenged,
        ))


def _was_challenged(error: AccessError) -> bool:
    """Was this refusal actually an anti-bot challenge?

    Checked from the headers carried on the error, because the response object
    no longer exists by the time the refusal is recorded.
    """
    headers = getattr(error, "headers", None) or {}
    for key, value in headers.items():
        if key.casefold() == "cf-mitigated" and str(value).casefold() == "challenge":
            return True
    return False


def _urllib_open(url: str, timeout: float) -> HttpResponse:
    """The single network call in the project, for verification only."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return HttpResponse(
                url=response.geturl(),
                status=response.status,
                headers=dict(response.headers.items()),
                body=body,
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
    except urllib.error.HTTPError as error:
        body = ""
        try:
            body = error.read().decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - body is best-effort
            pass
        headers = dict(error.headers.items()) if error.headers else {}
        failure = AccessError(
            f"{url}: HTTP {error.code}", status=error.code,
            retryable=error.code in RETRYABLE_STATUSES,
        )
        failure.retry_after = _retry_after(headers)
        failure.headers = headers
        failure.body = body
        failure.elapsed_ms = int((time.monotonic() - started) * 1000)
        raise failure
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise AccessError(f"{url}: {type(error).__name__}: {error}",
                          retryable=True) from error


def _retry_after(headers: Mapping[str, str]) -> Optional[float]:
    for key, value in headers.items():
        if key.casefold() == "retry-after":
            try:
                return max(0.0, float(value.strip()))
            except ValueError:
                return None
    return None


def response_from_refusal(error: AccessError, url: str) -> HttpResponse:
    """Rebuild a response from a refusal so callers can inspect its headers.

    A refusal carries the evidence that decides the outcome - notably
    ``Cf-Mitigated: challenge`` - so it must not be flattened into a message.
    """
    headers = getattr(error, "headers", {}) or {}
    return HttpResponse(
        url=url, status=error.status or 0, headers=headers,
        body=getattr(error, "body", "") or "",
        elapsed_ms=getattr(error, "elapsed_ms", 0),
    )