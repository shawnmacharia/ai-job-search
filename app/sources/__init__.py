"""Source access policy: verification, decisions, and reporting.

Live network access is confined to :mod:`app.sources.transport`, which exists
solely to check whether a source *may* be collected from. Nothing in this
package collects jobs.
"""

from app.sources.access import (
    AccessDecision,
    AccessLevel,
    load_decisions,
    render_report,
    save_decision,
    verify_access,
)
from app.sources.transport import AccessFetcher, AccessError, Ledger, RateLimiter

__all__ = [
    "AccessDecision",
    "AccessLevel",
    "AccessFetcher",
    "AccessError",
    "Ledger",
    "RateLimiter",
    "load_decisions",
    "render_report",
    "save_decision",
    "verify_access",
]