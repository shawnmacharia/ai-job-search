#!/usr/bin/env python3
"""Run one manual, rate-limited discovery pass against We Work Remotely.

This is deliberately a manual tool. There is no scheduler, no recurring job and
no retry loop across runs - one invocation fetches the feed once, ingests it
through the existing pipeline, and stops.

Access policy is enforced in code, not by convention:
  * exactly one request per run, at least 3s from any previous one
  * no disk caching (the feed sends ``Cache-Control: max-age=0``)
  * bounded retries for transient faults only; a refusal is never retried
  * attribution is carried into the dashboard, as the feed's terms require

Usage:
    python tools/collect_wwr.py [--data-dir data] [--dashboard out.html]

Exit codes:
    0  the run completed (even with zero eligible jobs)
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

from app.jobs.ingest import ingest  # noqa: E402
from app.jobs.runner import run_sources, summarise  # noqa: E402
from app.jobs.sources import PERMITTED, SourceConfig, SourceRegistry  # noqa: E402
from app.jobs.store import JobStore  # noqa: E402
from app.sources.access import (  # noqa: E402
    AccessLevel,
    load_decisions,
    save_decision,
)
from app.sources.transport import AccessFetcher, Ledger, RateLimiter  # noqa: E402
from app.sources.wwr import (  # noqa: E402
    ATTRIBUTION_TEXT,
    MIN_INTERVAL,
    PERMISSION_URL,
    WwrAdapter,
)

SOURCE = "weworkremotely"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="collect_wwr.py",
        description="One manual We Work Remotely discovery run.",
    )
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--dashboard", type=Path, default=None,
                        help="also write the review dashboard here")
    parser.add_argument("--min-interval", type=float, default=MIN_INTERVAL,
                        help=f"minimum seconds between requests (default {MIN_INTERVAL})")
    return parser


def record_permission(data_dir: Path, adapter: WwrAdapter) -> None:
    """Persist the permitted access decision with its evidence.

    Written from the access-verification layer rather than asserted here, so
    the decision carries the robots finding, the first-party permission
    statement and the attribution requirement together.
    """
    from app.sources.access import AccessDecision
    from datetime import datetime, timezone

    evidence = adapter.access_evidence()
    decision = AccessDecision(
        source=SOURCE,
        level=AccessLevel.PERMITTED,
        reason=(
            "robots.txt allows the feed path and the operator explicitly "
            "offers the public RSS feed for filling a remote job feed, "
            "conditional on attributing links back to We Work Remotely"
        ),
        robots="allows",
        robots_url=evidence["robots_url"],
        terms=evidence["permission_statement"],
        terms_url=evidence["permission_url"],
        checked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        evidence=tuple(f"{k}: {v}" for k, v in evidence.items()),
    )
    save_decision(decision, data_dir)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir)
    store = JobStore(data_dir)

    ledger = Ledger(data_dir / "access_attempts.jsonl")
    adapter = WwrAdapter(
        AccessFetcher(
            ledger=ledger,
            limiter=RateLimiter(args.min_interval),
            max_attempts=3,
        ),
        source=SOURCE,
    )

    record_permission(data_dir, adapter)

    # Fail closed: the registry consults the recorded decision, so the source
    # can only run because verification recorded it permitted.
    registry = SourceRegistry([SourceConfig(name=SOURCE, access=PERMITTED)])
    enforced = registry.enforce_recorded_decisions(load_decisions(data_dir))
    active = enforced.active()
    if not active:
        print("collect failed: the recorded access decision does not permit "
              "this source, so nothing was fetched", file=sys.stderr)
        for skip in enforced.skipped():
            print(f"  skipped {skip.name}: {skip.reason}", file=sys.stderr)
        return 1

    spec = run_sources(
        [
            _spec(adapter)
        ],
        store=store,
    )
    # The run cache is discarded here; the next run refetches.
    adapter.end_run()

    rows = summarise(spec)
    report = {
        "source": SOURCE,
        "exit_code": spec.exit_code(),
        "requests_made": adapter.requests_made,
        "totals": {
            "fetched": spec.total_fetched,
            "stored": spec.total_stored,
            "updated": spec.total_updated,
            "rejected": spec.total_rejected,
        },
        "rows": rows,
    }
    print(json.dumps(report, indent=2))
    print(f"\nfeed requests issued this run: {adapter.requests_made}")

    if args.dashboard:
        from app.reporting.jobs import render_dashboard_file
        from app.jobs.status import StatusLog

        path = render_dashboard_file(
            store,
            args.dashboard,
            status_log=StatusLog(store),
            attributions={SOURCE: PERMISSION_URL},
        )
        print(f"dashboard: {path}")

    return spec.exit_code()


def _spec(adapter: WwrAdapter):
    """Build a SourceSpec whose fetch ingests through the shared path."""
    from app.jobs.runner import SourceSpec

    def fetch():
        return adapter.to_records()

    return SourceSpec(name=SOURCE, fetch=fetch)


if __name__ == "__main__":
    raise SystemExit(main())