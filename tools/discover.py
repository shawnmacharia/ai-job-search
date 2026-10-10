#!/usr/bin/env python3
"""Run one manual discovery pass across every permitted source, then report.

Manual only. No scheduler, no recurring job. One invocation consults each source
at most once and stops.

What it guarantees
------------------
* Access is checked against the *recorded* decision before anything else. A
  source with no recorded permission is skipped, not attempted.
* Each source's persisted daily budget is consulted **before** a fetch is
  built, so a refusal costs no request and writes no attempt to the ledger.
* Sources run independently: one failing does not stop the others.
* Every source ends in one of five labelled outcomes - fetched, zero result,
  failed, budget-refused, or skipped - never a single blended "did not work".

Usage:
    python tools/discover.py [--data-dir data] [--report out.html] [--dry-run]
    python tools/discover.py --dry-run

Exit codes:
    0  every source consulted succeeded (refusals and skips are normal)
    1  partial failure: something failed, something else worked
    2  nothing could be consulted
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.jobs.discovery import run_discovery  # noqa: E402
from app.jobs.store import JobStore  # noqa: E402
from app.sources.myjobmag import FEEDS_PAGE, MAX_ATTEMPTS as MAG_ATTEMPTS  # noqa: E402
from app.sources.myjobmag import MIN_INTERVAL as MAG_INTERVAL  # noqa: E402
from app.sources.myjobmag import MyjobmagAdapter  # noqa: E402
from app.sources.remotive import MAX_ATTEMPTS as REM_ATTEMPTS  # noqa: E402
from app.sources.remotive import MIN_INTERVAL as REM_INTERVAL  # noqa: E402
from app.sources.remotive import RemotiveAdapter, SOURCE_PAGE as REMOTIVE_PAGE  # noqa: E402
from app.sources.transport import AccessFetcher, Ledger, RateLimiter  # noqa: E402
from app.sources.wwr import MIN_INTERVAL as WWR_INTERVAL  # noqa: E402
from app.sources.wwr import PERMISSION_URL as WWR_PERMISSION  # noqa: E402
from app.sources.wwr import WwrAdapter  # noqa: E402

#: source name -> (attribution page, per-source request floor)
ATTRIBUTIONS = {
    "weworkremotely": WWR_PERMISSION,
    "myjobmag.co.ke": FEEDS_PAGE,
    "remotive.com": REMOTIVE_PAGE,
}


def build_adapters(data_dir: Path) -> dict:
    """One adapter per live source, each sharing the persisted attempt ledger.

    The ledger is shared on purpose - it is the record of what this machine has
    requested from each site - and each adapter reads only its own ``purpose``
    from it, so one source's traffic never spends another's allowance.
    """
    ledger_path = data_dir / "access_attempts.jsonl"
    return {
        "weworkremotely": WwrAdapter(
            AccessFetcher(
                ledger=Ledger(ledger_path),
                limiter=RateLimiter(WWR_INTERVAL),
                max_attempts=3,
            ),
            source="weworkremotely",
        ),
        "myjobmag.co.ke": MyjobmagAdapter(
            AccessFetcher(
                ledger=Ledger(ledger_path),
                limiter=RateLimiter(MAG_INTERVAL),
                max_attempts=MAG_ATTEMPTS,
            ),
            source="myjobmag.co.ke",
        ),
        "remotive.com": RemotiveAdapter(
            AccessFetcher(
                ledger=Ledger(ledger_path),
                limiter=RateLimiter(REM_INTERVAL),
                max_attempts=REM_ATTEMPTS,
            ),
            source="remotive.com",
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="discover.py",
        description="One manual discovery pass across every permitted source.",
    )
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--report", type=Path, default=None,
                        help="write the consolidated review report here")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report what would run, skip or refuse, without any network request",
    )
    parser.add_argument("--candidate-country", default="KE")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir)
    store = JobStore(data_dir)
    adapters = build_adapters(data_dir)

    result = run_discovery(
        store,
        adapters,
        dry_run=args.dry_run,
        report_path=args.report,
        attributions=ATTRIBUTIONS,
        candidate_country=args.candidate_country,
    )

    if args.dry_run:
        print("dry run: no source was contacted and no ledger was written\n")
    print(json.dumps(result.to_dict(), indent=2))

    if result.report_path:
        print(f"\nreport: {result.report_path}")
    for error in result.report_errors:
        print(f"  report warning: {error}", file=sys.stderr)

    print(f"\nrequests issued this run: {result.total_requests}")
    return result.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())