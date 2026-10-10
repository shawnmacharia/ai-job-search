"""One request per day, enforced from a ledger that survives a restart.

Why this module exists
----------------------
Both live sources with a daily cap enforce it through the same rule, and the
rule has one property that is easy to get wrong and impossible to notice: it is
only real if it survives the process ending.

A budget seeded from in-memory state behaves perfectly all day and resets to
zero the moment the program exits. Nothing fails, no test turns red, and the
next invocation cheerfully makes a second request inside the window. For a
source whose access was granted on the condition of a specific cadence, that is
the worst possible failure: it is silent, and it spends access we were given.

So the budget is seeded from the **persisted** attempt ledger. The ledger is
append-only and survives restarts; the budget is derived from it on every
construction.

Failing closed
--------------
Two cases must refuse rather than proceed:

1. **No readable history.** If a ledger file exists but cannot be parsed, we
   cannot show that the window is empty, and an unreadable record is not
   evidence of compliance. The budget refuses.
2. **A timestamp we cannot read.** An attempt with an unparseable time is
   treated as *recent*, not as ancient. Assuming it was long ago is how a
   limit gets quietly exceeded.

The distinction that matters: **no ledger file** means nothing has been
recorded, so the first request is allowed. **An unreadable one** means something
was recorded and we cannot tell what, so nothing is allowed.

Not fail-closed by design
-------------------------
A ledger with no file path is an in-memory ledger, used in tests and by callers
that manage their own persistence. There is no history to trust or distrust, so
it is seeded as empty rather than blocked - blocking it would make the class
unusable and would not protect anything, since nothing was ever written.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional, Sequence

#: The window a "per day" limit means: a rolling 24 hours, not a calendar day.
#:
#: A calendar-day boundary would let two requests land either side of midnight
#: and be almost 24 hours apart while both counting against the same day, and
#: would silently permit a third across a long weekend.
DAY_SECONDS = 86400.0


class DailyBudget:
    """At most ``limit`` requests per rolling 24-hour window.

    ``now`` is injected so the rule is provable in tests without waiting a day.
    ``prior_stamps`` seeds the window from a persisted ledger, and is what makes
    the limit survive a restart.

    ``untrusted=True`` locks the budget shut. It is the response to a ledger
    that exists but cannot be read: we cannot show the window is empty, so we
    refuse rather than guess.
    """

    def __init__(
        self,
        limit: int = 1,
        *,
        now: Optional[Callable[[], float]] = None,
        prior_stamps: Optional[Sequence[float]] = None,
        untrusted: bool = False,
    ) -> None:
        self.limit = max(1, int(limit))
        self._now = now or (lambda: datetime.now(timezone.utc).timestamp())
        self._stamps: List[float] = list(prior_stamps or ())
        #: Set when history exists but could not be read. Latched: once we know
        #: we cannot verify, the answer stays "no" until a new budget is built
        #: from a ledger that can be read.
        self._untrusted = bool(untrusted)

    @property
    def untrusted(self) -> bool:
        """True when the budget is refusing because history was unreadable."""
        return self._untrusted

    def allow(self) -> bool:
        if self._untrusted:
            return False
        cutoff = self._now() - DAY_SECONDS
        self._stamps = [s for s in self._stamps if s > cutoff]
        return len(self._stamps) < self.limit

    def record(self) -> None:
        self._stamps.append(self._now())


def prior_stamps(
    ledger: Any,
    *,
    purpose: str,
    window_seconds: float = DAY_SECONDS,
    now: Optional[Callable[[], float]] = None,
) -> Optional[List[float]]:
    """Timestamps of this purpose's attempts already on disk, inside the window.

    Returns:

    - ``None`` when a ledger file exists but its content cannot be trusted. The
      caller must fail closed.
    - a list (possibly empty) when history is readable. An empty list means
      *nothing has been recorded*, which legitimately permits a first request.

    Attempts belonging to other purposes are ignored, so one source's traffic
    never consumes another's budget. That isolation is the point: a shared
    ledger file must not let Remotive's requests exhaust MyJobMag's allowance.
    """
    clock = now or (lambda: datetime.now(timezone.utc).timestamp())
    path = getattr(ledger, "path", None)
    if path is None:
        # In-memory ledger: no history has ever been written, so there is
        # nothing to distrust either.
        return []

    try:
        exists = path.exists()
    except OSError:
        # Cannot even stat the path. Treat as untrusted rather than absent.
        return None
    if not exists:
        # No ledger has ever been written. Nothing to seed, nothing to distrust.
        return []

    from pathlib import Path

    stamps: List[float] = []
    cutoff = clock() - window_seconds
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    # A torn or hand-edited line could be hiding a request we
                    # are supposed to count. We cannot prove it is safe, so we
                    # do not treat the ledger as readable.
                    return None
                if not isinstance(data, dict) or "at" not in data:
                    return None
                if str(data.get("purpose", "")) != purpose:
                    # Another source's attempt. Must not consume this budget.
                    continue
                try:
                    stamp = _parse(data["at"]).timestamp()
                except (TypeError, ValueError):
                    # An unreadable time cannot be shown to be old. Treat it as
                    # now so it counts against the window.
                    stamp = clock()
                if stamp > cutoff:
                    stamps.append(stamp)
    except OSError:
        return None
    return stamps


def _parse(value: Any) -> datetime:
    """Parse a ledger timestamp, tolerating both ``Z`` and ``+00:00``."""
    text = str(value).strip()
    if not text:
        raise ValueError("empty timestamp")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        # A naive stamp is assumed UTC: the ledger writer always emits UTC, and
        # guessing local time would shift the window.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed