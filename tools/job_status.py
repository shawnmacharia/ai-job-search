#!/usr/bin/env python3
"""Record and show a review status against a stored job.

The status model in ``app.jobs.status`` is append-only: this tool appends a
decision and prints the resulting trail. It never rewrites history, never
edits ``data/jobs.jsonl``, and never contacts anything.

Usage:
    python tools/job_status.py list
    python tools/job_status.py show <job_id>
    python tools/job_status.py set <job_id> <status> [--note "..."]
    python tools/job_status.py summary

Exit codes:
    0  the command succeeded
    1  the command failed (unknown job, unknown status, unreadable log)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.jobs.status import ReviewStatus, StatusError, StatusLog  # noqa: E402
from app.jobs.store import JobStore  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "data"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="job_status.py",
        description="Record and show a review status against a stored job.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"data directory (default: {DEFAULT_DATA_DIR})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list known jobs with their current status")
    show = sub.add_parser("show", help="show the full decision trail for a job")
    show.add_argument("job_id")

    setter = sub.add_parser("set", help="record a status for a job")
    setter.add_argument("job_id")
    setter.add_argument("status", help=f"one of: {', '.join(s.value for s in ReviewStatus)}")
    setter.add_argument("--note", default="", help="optional note, stored verbatim")

    sub.add_parser("summary", help="group jobs by current status")
    return parser


def command_list(log: StatusLog, store: JobStore) -> int:
    jobs = store.load_jobs()
    if not jobs:
        print("no jobs stored yet")
        return 0
    width = max(len(str(record.get("job_id", ""))) for record in jobs)
    for record in sorted(jobs, key=lambda r: str(r.get("job_id", ""))):
        job_id = str(record.get("job_id", ""))
        job = record.get("job", {})
        status = log.current(job_id).value
        print(
            f"{job_id:<{width}}  {status:<11}  "
            f"{job.get('company', '')} - {job.get('title', '')}"
        )
    return 0


def command_show(log: StatusLog, job_id: str) -> int:
    events = log.history(job_id)
    if not events:
        print(f"no decisions recorded for {job_id} (status: new)")
        return 0
    for event in events:
        previous = event.previous or "-"
        note = f"  # {event.note}" if event.note else ""
        print(f"{event.at}  {previous} -> {event.status}{note}")
    return 0


def command_set(log: StatusLog, job_id: str, status: str, note: str) -> int:
    try:
        target = ReviewStatus(status)
    except ValueError:
        print(
            f"error: unknown status {status!r}; expected one of: "
            f"{', '.join(s.value for s in ReviewStatus)}",
            file=sys.stderr,
        )
        return 1
    event = log.record(job_id, target, note=note)
    print(f"{job_id}: {event.previous or '-'} -> {event.status}")
    return 0


def command_summary(log: StatusLog) -> int:
    for status, job_ids in log.summary().items():
        print(f"{status:<11} {len(job_ids)}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    store = JobStore(args.data_dir)
    log = StatusLog(store)

    try:
        if args.command == "list":
            return command_list(log, store)
        if args.command == "show":
            return command_show(log, args.job_id)
        if args.command == "set":
            return command_set(log, args.job_id, args.status, args.note)
        if args.command == "summary":
            return command_summary(log)
    except StatusError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())