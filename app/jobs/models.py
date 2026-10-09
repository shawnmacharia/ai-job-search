"""Provider-independent job records."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.state.models import RemoteStatus


@dataclass
class Job:
    job_id: str
    title: str
    company: str
    url: str
    description: str = ""
    location: Optional[str] = None
    remote_status: RemoteStatus = RemoteStatus.UNKNOWN
    country: Optional[str] = None
    region: Optional[str] = None
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    salary_currency: Optional[str] = None
    salary_period: Optional[str] = None
    portal: Optional[str] = None
    posted_date: Optional[str] = None
    deadline: Optional[str] = None
    skills: List[str] = field(default_factory=list)

    # --- provenance and completeness, added for source adapters ---
    # All three default to a safe, honest value, so every existing positional
    # and keyword construction of Job keeps working unchanged.

    #: Source fields this record could not map to a canonical field, kept
    #: verbatim. Present so that adapting a source never *silently* discards
    #: data: anything not mapped lands here instead of vanishing.
    raw_excerpt: Dict[str, Any] = field(default_factory=dict)

    #: The source's own date string, kept when parsing it to an absolute date
    #: was lossy or failed outright (e.g. "2d ago" -> "2026-10-07").
    posted_raw: Optional[str] = None

    #: Whether ``description`` is the complete posting. Defaults to False
    #: because most sources publish only a card snippet, and downstream
    #: matching must be able to tell "short" from "whole".
    description_complete: bool = False


def job_key(company: str, title: str) -> str:
    """Create a stable fallback key when a portal has no identifier."""
    return f"{company.strip().casefold()}::{title.strip().casefold()}"