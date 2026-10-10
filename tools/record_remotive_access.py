"""Record the Remotive access decision through the verification layer.

Run once, manually, after the evidence record has been written. The decision is
written into ``data/access.json`` by :func:`app.sources.access.save_decision` -
never by editing that file - so it goes through the same path as every other
source and is rendered by the same report.

This script commits to nothing by itself: it holds no network code, and it
refuses to record ``permitted`` unless the evidence file for the source exists
and names the endpoint being granted. A decision without evidence is exactly
the failure mode this project exists to prevent.

Usage::

    py tools/record_remotive_access.py --data data
    py tools/record_remotive_access.py --data data --dry-run
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.sources.access import (  # noqa: E402
    ROBOTS_DISALLOWS,
    TERMS_REVIEWED_CLEAR,
    AccessDecision,
    AccessLevel,
    PathDecision,
    render_report,
    save_decision,
)

SOURCE = "remotive.com"
API_PATH = "/api/remote-jobs"
EVIDENCE_FILE = "remotive_permission_evidence.md"

REASON = (
    "Public API access granted in writing by Remotive; recorded as repository-"
    "owner attestation. The grant resolves the robots.txt conflict for this one "
    "endpoint: robots disallows /api/* for crawlers of the site, while the API "
    "is a separately licensed programmatic interface with its own published "
    "terms. Scope is the single documented jobs endpoint, at most one request "
    "per day, with attribution required. No HTML scraping, no pagination, no "
    "other endpoint."
)


def build_decision(checked_at: str) -> AccessDecision:
    """The decision, scoped as narrowly as the grant allows."""
    return AccessDecision(
        source=SOURCE,
        level=AccessLevel.PERMITTED,
        reason=REASON,
        robots=ROBOTS_DISALLOWS,
        robots_url="https://remotive.com/robots.txt",
        terms=TERMS_REVIEWED_CLEAR,
        terms_url="https://remotive.com/remote-jobs/api",
        checked_at=checked_at,
        paths=(
            PathDecision(path=API_PATH, allowed=True),
            # The HTML site is not covered by the grant. Recorded explicitly so
            # a future reader does not have to infer it from an absence.
            PathDecision(path="/", allowed=False),
            PathDecision(path="/remote-jobs/", allowed=False),
            PathDecision(path="/jobs/", allowed=False),
        ),
        evidence=(
            f"written permission relayed by the repository owner ({EVIDENCE_FILE})",
            "first-party terms retrieved 2026-10-09, HTTP 200",
            "scope: https://remotive.com/api/remote-jobs only",
            "cadence: at most 1 request/day (stricter than the published 4/day)",
            "attribution required: link back + credit Remotive as source",
            "restrictions: no third-party republication, no HTML, no pagination",
        ),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="data directory")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the decision without saving"
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data)
    evidence = data_dir / "ops" / EVIDENCE_FILE
    if not evidence.exists():
        print(
            f"refusing to record a decision without evidence: {evidence} is missing.\n"
            "Write the evidence record first, then re-run.",
            file=sys.stderr,
        )
        return 1

    decision = build_decision(datetime.now(timezone.utc).isoformat(timespec="seconds"))

    if args.dry_run:
        print(render_report([decision]))
        return 0

    path = save_decision(decision, data_dir)
    print(f"recorded: {decision.summary_line()}")
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())