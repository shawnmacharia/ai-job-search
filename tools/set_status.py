#!/usr/bin/env python3
"""Record one review decision for one job.

Manual, single decision per invocation. This writes exactly one line to
``data/status.jsonl`` and nothing else anywhere.

What it will not do
-------------------
* It never touches ``jobs.jsonl``, ``seen.json``, the attempt ledger, or the run
  ledger. A review decision is an overlay; the store is owned by ingestion.
* It never submits anything to an employer. ``applied`` and ``rejected`` are not
  statuses here, and adding them would imply an application workflow this
  project does not have and must not perform.
* It never guesses. An unknown job id or an unrecognised status is refused
  *before* anything is written, so a typo cannot leave a half-record behind.

Validation happens in :meth:`~app.jobs.status.StatusLog.record`, which checks
the job exists, checks the status is real, checks the transition is declared,
and only then appends. This command is a thin, honest surface over it.

Usage:
    python tools/set_status.py --job-id <id> --status interested [--note "..."]
    python tools/set_status.py --job-id <id> --status shortlisted
    python tools/set_status.py --list-statuses

Exit codes:
    0  the decision was recorded
    1  refused: unknown job id, unknown status, or no decision supplied
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


#: One line per status, for ``--list-statuses``. Held here rather than on the
#: enum because Python cannot attach a docstring to an individual member, and
#: repeating the class docstring five times is noise rather than help.
DESCRIPTIONS = {
    ReviewStatus.NEW: "not looked at yet (the default before any decision)",
    ReviewStatus.REVIEWING: "opened and being read, no decision yet",
    ReviewStatus.INTERESTED: "worth pursuing",
    ReviewStatus.SHORTLISTED: "in the short set you would actually act on",
    ReviewStatus.DISMISSED: "closed: not for me",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="set_status.py",
        description="Record one review decision for one job.",
    )
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--job-id", default=None, help="the stored job id")
    parser.add_argument("--status", default=None,
                        choices=[s.value for s in ReviewStatus],
                        help="the new status")
    parser.add_argument("--note", default="",
                        help="free text, stored verbatim and never interpreted")
    parser.add_argument("--list-statuses", action="store_true",
                        help="print the valid statuses and exit")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_statuses:
        for status in ReviewStatus:
            print(f"{status.value:14} {DESCRIPTIONS.get(status, '')}".rstrip())
        print("\nthere is deliberately no 'applied' or 'rejected': acting is a "
              "hard stop for this project.")
        return 0

    if not args.job_id or not args.status:
        print("refusing: --job-id and --status are both required, or use "
              "--list-statuses", file=sys.stderr)
        return 1

    store = JobStore(Path(args.data_dir))
    log = StatusLog(store)

    # Parsed before anything is written, so a bad value cannot leave a partial
    # record behind. argparse already restricts --status to real members; this
    # is the belt to that braces, and the one that matters if the choices are
    # ever widened.
    try:
        status = ReviewStatus(args.status)
    except ValueError:
        print(f"refusing: {args.status!r} is not a status", file=sys.stderr)
        return 1

    try:
        event = log.record(args.job_id, status, note=args.note)
    except StatusError as error:
        # Nothing has been written at this point: record() validates first.
        print(f"refusing: {error}", file=sys.stderr)
        return 1

    print(f"recorded: job {event.job_id} -> {event.status} at {event.at}")
    if event.note:
        print(f"  note: {event.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())