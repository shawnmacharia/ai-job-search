#!/usr/bin/env python3
"""List the actionable review queue. Read-only, offline, deterministic.

```bash
python tools/review.py --queue
```

This calls the reporting library directly. It does not invoke discovery, does
not touch a source adapter, opens no socket, and writes nothing - not the
queue, not a report file, not a status, not an assessment.

That is the whole point of this script. Everything else in ``tools/`` either
fetches or records. When you want to *look* at the queue without changing
anything and without spending a request, this is the only command to reach for.

Order is stable
--------------
Rows come back in the reporting library's queue order, with the canonical
internal job ID as a final tie-break. Re-running this on unchanged data prints
byte-identical output, so a diff between two runs shows real change rather than
reordering noise.

Unreadable records are reported, not hidden
------------------------------------------
The store skips a malformed line rather than aborting the read, which is right
for robustness and wrong for trust: a silently dropped record looks exactly
like a job that was never there. This command counts what the store could not
parse and says so on stdout, so a truncated ``jobs.jsonl`` cannot pass for an
empty queue.

Exit codes:
    0  queue listed
    1  nothing to list, or the store could not be read
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.jobs.status import StatusLog  # noqa: E402
from app.jobs.store import JobStore  # noqa: E402
from app.reporting.review import build_actionable, build_report  # noqa: E402

#: Columns, in order. Fixed so output is diffable between runs.
COLUMNS = ("#", "JOB ID", "TITLE", "COMPANY", "SOURCE", "ELIGIBLE",
           "FRESHNESS", "STATUS", "MATCH")


def _clean(value: Any, limit: int) -> str:
    """One field, made safe to print in a terminal.

    Postings are untrusted text. Control characters are stripped so a crafted
    title cannot rewrite the line it sits on, and the result is truncated so
    one enormous description does not destroy the table's shape. Newlines and
    tabs collapse to spaces rather than disappearing, which keeps the columns
    aligned and the text readable.
    """
    text = "" if value is None else str(value)
    text = "".join(" " if (ch < " " or ch == "\x7f") else ch for ch in text)
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 1] + "\u2026"
    return text


def unreadable_lines(data_dir: Path) -> Tuple[int, int]:
    """``(readable, unreadable)`` counts for ``jobs.jsonl``.

    Read with the same tolerance as the store, so the number reported is the
    number of lines the store actually lost - not a second opinion about what
    is valid.
    """
    path = Path(data_dir) / "jobs.jsonl"
    if not path.exists():
        return 0, 0
    readable = unreadable = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            json.loads(line)
        except (json.JSONDecodeError, ValueError):
            unreadable += 1
        else:
            readable += 1
    return readable, unreadable


def queue_rows(views: Sequence[Any]) -> List[Tuple[str, ...]]:
    """Turn views into fixed-width, escaped, deterministically ordered rows."""
    rows: List[Tuple[str, ...]] = []
    for index, view in enumerate(views, 1):
        rows.append((
            str(index),
            _clean(view.job_id, 60),
            _clean(view.title, 44),
            _clean(view.company, 26),
            _clean(",".join(view.sources) or "-", 22),
            _clean(view.verdict, 10),
            _clean(view.freshness, 9),
            _clean(view.application_status, 11),
            _clean(view.match_tier or "not assessed", 18),
        ))
    return rows


def render(rows: Sequence[Sequence[str]]) -> str:
    """One aligned table. Width is derived from content, so it is stable."""
    if not rows:
        return ""
    widths = [max(len(row[i]) for row in rows) for i in range(len(COLUMNS))]
    rule = "  ".join("-" * w for w in widths)
    lines = [rule]
    for number, row in enumerate(rows):
        # The index column is right-aligned; the rest read better flush left.
        cells = [row[0].rjust(widths[0])] + [
            cell.ljust(widths[i + 1]) for i, cell in enumerate(row[1:])
        ]
        lines.append("  ".join(cells).rstrip())
    return "\n".join(lines)


def match_state(view: Any) -> Dict[str, Any]:
    """Match-assessment state, distinguishing absent from insufficient.

    "never assessed" and "assessed and the evidence was insufficient" are
    different facts and collapse into the same word too easily, so they are
    reported apart.
    """
    if not view.match_present:
        return {"state": "not assessed", "tier": None, "reason": ""}
    return {
        "state": "assessed" if view.match_tier != "not_yet_evaluated"
                 else "insufficient evidence",
        "tier": view.match_tier,
        "reason": view.match_insufficient_reason or "",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="review.py",
        description="List the actionable review queue. Read-only and offline.",
    )
    parser.add_argument("--queue", action="store_true",
                        help="list the actionable queue (the only mode)")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.queue:
        print("nothing to do: pass --queue", file=sys.stderr)
        return 1

    readable, unreadable = unreadable_lines(args.data_dir)
    if unreadable:
        # Loud, and before the table: a silently dropped record would look
        # exactly like a job that was never stored.
        print(f"WARNING: {unreadable} unreadable line(s) in jobs.jsonl were "
              f"skipped by the store.", file=sys.stderr)
        print(f"         {readable} readable, {unreadable} unreadable.",
              file=sys.stderr)
        print("         The queue below is missing those records.",
              file=sys.stderr)

    store = JobStore(Path(args.data_dir))
    report = build_report(store, status_log=StatusLog(store))
    queue = build_actionable([view for view, _ in report.queue])
    rows = queue_rows(queue)

    for error in report.errors:
        print(f"WARNING: {error}", file=sys.stderr)

    if not rows:
        print("the actionable queue is empty")
        return 1

    print(f"actionable queue: {len(rows)} job(s) of {report.total_jobs} stored")
    print("columns: " + " | ".join(COLUMNS))
    print()
    print(render(rows))
    print(f"\nread-only: no file written, no network request, nothing assessed.")
    print(f"match state: {sum(1 for v in queue if not v.match_present)} "
          f"not assessed, "
          f"{sum(1 for v in queue if match_state(v)['state'] == 'assessed')} "
          f"assessed, "
          f"{sum(1 for v in queue if match_state(v)['state'] == 'insufficient evidence')} "
          f"insufficient evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())