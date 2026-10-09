#!/usr/bin/env python3
"""Verify whether a source may be collected from, and print an access report.

This is the only tool here that makes network requests, and it does so only to
answer a policy question. It fetches robots.txt and the terms page, nothing
else - no job listings, no pagination, no crawling.

Requests are rate limited and sequential. Refusals are reported, never retried
around: if a site challenges this client, that is the answer.

Usage:
    python tools/check_access.py verify [--source NAME --base-url URL]
    python tools/check_access.py report

Exit codes:
    0  a decision was recorded (the decision itself may be restricted/unknown)
    1  the check could not run at all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.sources.access import (  # noqa: E402
    load_decisions,
    render_report,
    save_decision,
    verify_access,
)
from app.sources.transport import AccessFetcher, Ledger, RateLimiter  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "data"

#: Default minimum seconds between requests. Deliberately slow: this is
#: verification, not collection.
DEFAULT_MIN_INTERVAL = 2.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_access.py",
        description="Verify source access and report the decision.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--min-interval", type=float, default=DEFAULT_MIN_INTERVAL)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=20.0)

    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify", help="run the live check for one source")
    verify.add_argument("--source", default="hiring.cafe")
    verify.add_argument("--base-url", default="https://hiring.cafe")
    sub.add_parser("report", help="print the report for recorded decisions")
    return parser


def command_verify(args) -> int:
    data_dir = Path(args.data_dir)
    ledger = Ledger(data_dir / "access_attempts.jsonl")
    fetcher = AccessFetcher(
        ledger=ledger,
        limiter=RateLimiter(min_interval=args.min_interval),
        max_attempts=args.max_attempts,
        timeout=args.timeout,
    )

    decision = verify_access(args.source, args.base_url, fetcher)
    save_decision(decision, data_dir)
    print(render_report([decision]))
    return 0


def command_report(args) -> int:
    decisions = load_decisions(Path(args.data_dir))
    if not decisions:
        print("no access decisions recorded yet; run 'verify' first")
        return 1
    print(render_report(list(decisions.values())))
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "verify":
            return command_verify(args)
        if args.command == "report":
            return command_report(args)
    except KeyboardInterrupt:
        print("access check interrupted", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"access check failed: {type(error).__name__}: {error}",
              file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())