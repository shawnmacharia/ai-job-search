#!/usr/bin/env python3
"""Run one manual MyJobMag discovery pass over the approved RSS feed.

Manual only. No scheduler, no recurring job. One invocation fetches the feed at
most once per day and stops.

The approved scope is enforced in code, not by convention:
  * exactly one endpoint: jobsxml_by_categories.xml
  * at most one request per rolling day, enforced against the persisted ledger
  * no job-detail pages, no pagination, no scraping, no other feeds
  * attribution carried into the dashboard

Usage:
    python tools/collect_myjobmag.py [--data-dir data] [--dashboard out.html]

Exit codes:
    0  the run completed
    1  the run failed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.jobs.runner import SourceSpec, run_sources, summarise  # noqa: E402
from app.jobs.sources import PERMITTED, SourceConfig, SourceRegistry  # noqa: E402
from app.jobs.store import JobStore  # noqa: E402
from app.sources.access import load_decisions  # noqa: E402
from app.sources.myjobmag import FEEDS_PAGE, MAX_ATTEMPTS, MyjobmagAdapter  # noqa: E402
from app.sources.transport import AccessFetcher, Ledger, RateLimiter  # noqa: E402

SOURCE = "myjobmag.co.ke"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="collect_myjobmag.py",
        description="One manual MyJobMag discovery run (approved RSS feed only).",
    )
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--dashboard", type=Path, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir)
    store = JobStore(data_dir)

    ledger = Ledger(data_dir / "access_attempts.jsonl")
    adapter = MyjobmagAdapter(
        AccessFetcher(
            ledger=ledger,
            limiter=RateLimiter(86400.0),
            max_attempts=MAX_ATTEMPTS,
        ),
        source=SOURCE,
    )

    # Fail closed: the registry consults the recorded decision, so the source
    # can only run because verification recorded it permitted.
    registry = SourceRegistry([SourceConfig(name=SOURCE, access=PERMITTED)])
    enforced = registry.enforce_recorded_decisions(load_decisions(data_dir))
    if not enforced.active():
        print("collect failed: the recorded access decision does not permit "
              "this source, so nothing was fetched", file=sys.stderr)
        for skip in enforced.skipped():
            print(f"  skipped {skip.name}: {skip.reason}", file=sys.stderr)
        return 1

    run = run_sources(
        [SourceSpec(name=SOURCE, fetch=adapter.to_records)], store=store
    )
    adapter.end_run()

    print(json.dumps({
        "source": SOURCE,
        "exit_code": run.exit_code(),
        "requests_made": adapter.requests_made,
        "totals": {
            "fetched": run.total_fetched,
            "stored": run.total_stored,
            "updated": run.total_updated,
            "rejected": run.total_rejected,
        },
        "rows": summarise(run),
    }, indent=2))

    if args.dashboard:
        from app.jobs.status import StatusLog
        from app.reporting.jobs import render_dashboard_file

        path = render_dashboard_file(
            store, args.dashboard, status_log=StatusLog(store),
            attributions={SOURCE: FEEDS_PAGE},
        )
        print(f"dashboard: {path}")

    return run.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())