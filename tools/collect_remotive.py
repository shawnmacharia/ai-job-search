#!/usr/bin/env python3
"""Run one manual Remotive discovery pass over the approved Public API.

Manual only. No scheduler, no recurring job. One invocation makes at most one
API request and stops.

The approved scope is enforced in code, not by convention:
  * exactly one endpoint: /api/remote-jobs
  * at most one request per rolling day, enforced against the *persisted*
    attempt ledger so the limit survives a restart
  * no HTML pages, no job-detail pages, no pagination, no other endpoints
  * attribution carried into every stored record

Access rests on written permission from Remotive, recorded as repository-owner
attestation in data/ops/remotive_permission_evidence.md. This tool refuses to
run unless that decision has been recorded as permitted through
app.sources.access - it does not decide for itself that it may fetch.

Usage:
    python tools/collect_remotive.py [--data-dir data] [--dashboard out.html]

Exit codes:
    0  the run completed
    1  the run failed, or the recorded decision does not permit collection
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
from app.sources.remotive import (  # noqa: E402
    MAX_ATTEMPTS,
    MIN_INTERVAL,
    RemotiveAdapter,
    SOURCE_PAGE,
)
from app.sources.transport import AccessFetcher, Ledger, RateLimiter  # noqa: E402

SOURCE = "remotive.com"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="collect_remotive.py",
        description="One manual Remotive discovery run (approved Public API only).",
    )
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--dashboard", type=Path, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir)
    store = JobStore(data_dir)

    # The ledger is persisted and the limiter's floor is a full day. Together
    # they are what makes "one request a day" true across separate runs rather
    # than only within one.
    ledger = Ledger(data_dir / "access_attempts.jsonl")
    adapter = RemotiveAdapter(
        AccessFetcher(
            ledger=ledger,
            limiter=RateLimiter(MIN_INTERVAL),
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

    run = run_sources([SourceSpec(name=SOURCE, fetch=adapter.to_records)], store=store)
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
        "access_evidence": adapter.access_evidence(),
    }, indent=2))
    print(f"\nAPI requests issued this run: {adapter.requests_made}")

    if args.dashboard:
        from app.jobs.freshness import FreshnessLedger, summarise as freshness_summary
        from app.jobs.status import StatusLog
        from app.reporting.jobs import render_dashboard_file

        # Replayed from the append-only observation log, so re-rendering costs
        # no API requests. Jobs never observed render as "unknown" rather than
        # being assumed current.
        freshness_ledger = FreshnessLedger(store)
        state = freshness_ledger.evaluate()
        path = render_dashboard_file(
            store,
            args.dashboard,
            status_log=StatusLog(store),
            attributions={SOURCE: SOURCE_PAGE},
            freshness=state,
            source_health=freshness_summary(state, observations=freshness_ledger.load()),
        )
        print(f"dashboard: {path}")

    return run.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())