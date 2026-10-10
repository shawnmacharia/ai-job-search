#!/usr/bin/env python3
"""Assess match for one job, or a batch you chose - by hand.

Manual only. There is no "assess everything" mode, by design: each assessment
may send job and profile text to a provider, and that is a decision a person
makes each time.

Requires an explicitly registered and reachable provider. The registry ships
empty, so with nothing configured this command refuses and writes nothing.

What it can and cannot do
--------------------------
It can reorder and explain the queue. It cannot remove a job, dismiss one,
change a status, or alter eligibility or freshness. A match result that hides a
job would turn a judgement into a decision, and that is not this tool's to make.

Data boundary
-------------
Assessing sends the job's title, company, location, description and
requirements, plus the candidate profile text, to the configured provider.
Nothing is sent until a provider is registered and named below.

Usage:
    python tools/match.py --provider <name> --job-id <id>
    python tools/match.py --provider <name> --batch 5
    python tools/match.py --providers          # list registered providers
    python tools/match.py --boundary           # state the data boundary

Exit codes:
    0  every selected job was assessed (or honestly could not be)
    1  refused: no provider, no profile, or bad selection
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.jobs.assessment import (  # noqa: E402
    MAX_BATCH,
    AssessmentUnavailable,
    assess_batch,
    assess_one,
    describe_data_boundary,
    load_candidate_profile,
    registered_providers,
    resolve_provider,
)
from app.jobs.assessment import AssessmentStore  # noqa: E402
from app.jobs.models import Job  # noqa: E402
from app.jobs.store import JobStore  # noqa: E402
from app.reporting.review import (  # noqa: E402
    QueueFilters,
    build_actionable,
    build_report,
)
from app.jobs.status import StatusLog  # noqa: E402


def _job_from_record(record) -> Job:
    fields = {
        name: value for name, value in record.get("job", {}).items()
        if name in Job.__dataclass_fields__
    }
    return Job(**fields)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="match.py",
        description="Bounded, evidence-checked match assessment for jobs you choose.",
    )
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--provider", default=None,
                        help="an explicitly registered provider name")
    parser.add_argument("--model", default=None)
    parser.add_argument("--job-id", default=None,
                        help="assess exactly this job")
    parser.add_argument("--batch", type=int, default=None,
                        help=f"assess at most N queued jobs (ceiling {MAX_BATCH})")
    parser.add_argument("--source", default=None,
                        help="restrict the batch to one source")
    parser.add_argument("--providers", action="store_true",
                        help="list registered providers and exit")
    parser.add_argument("--boundary", action="store_true",
                        help="state the data boundary and exit")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.providers:
        names = registered_providers()
        if not names:
            print("no providers registered; register one explicitly before "
                  "assessing. There is no default.")
        for name in names:
            print(f"  {name}")
        return 0

    if args.boundary:
        print(describe_data_boundary())
        return 0

    if not args.job_id and args.batch is None:
        print("refusing: choose --job-id for one job, or --batch N for a "
              "chosen number. There is no assess-everything mode.", file=sys.stderr)
        return 1

    if args.job_id and args.batch is not None:
        print("refusing: use --job-id or --batch, not both", file=sys.stderr)
        return 1

    if args.batch is not None and (args.batch < 1 or args.batch > MAX_BATCH):
        print(f"refusing: --batch must be between 1 and {MAX_BATCH}",
              file=sys.stderr)
        return 1

    store = JobStore(Path(args.data_dir))
    profile = load_candidate_profile(REPO_ROOT)

    # Provider first, and unconditionally. An unavailable provider must make no
    # writes at all - not even a record saying it failed, because the request
    # never left the machine and there is nothing to record.
    try:
        provider_name, provider = resolve_provider(args.provider)
    except AssessmentUnavailable as error:
        print(f"refusing: {error}", file=sys.stderr)
        print("no assessment was written.", file=sys.stderr)
        return 1

    if not profile.usable:
        print("refusing: no usable candidate profile, so nothing could be "
              "evidenced. No assessment was written.", file=sys.stderr)
        return 1

    assessments = AssessmentStore(Path(args.data_dir))

    if args.job_id:
        record = next(
            (r for r in store.load_jobs() if str(r.get("job_id")) == args.job_id),
            None,
        )
        if record is None:
            print(f"refusing: no stored job with id {args.job_id!r}. "
                  "No assessment was written.", file=sys.stderr)
            return 1
        results = [assess_one(
            _job_from_record(record), profile=profile,
            provider_name=provider_name, provider=provider,
            store=assessments, model=args.model,
        )]
    else:
        report = build_report(store, status_log=StatusLog(store))
        queue = QueueFilters(source=args.source).narrow(
            build_actionable(list(report.filtered) or [])
        ) if args.source else build_actionable(
            [v for v, _ in report.queue]
        )
        if not queue:
            print("nothing in the actionable queue to assess", file=sys.stderr)
            return 1
        results = assess_batch(
            [ _job_from_record(r) for r in store.load_jobs()
              if str(r.get("job_id")) in {v.job_id for v in queue} ],
            profile=profile, provider_name=provider_name,
            provider=provider, store=assessments,
            max_n=args.batch, model=args.model,
        )

    print(f"provider: {provider_name}  ({results[0].model or 'model unknown'})")
    for record in results:
        print(
            f"  {record.job_id}: {record.status} tier={record.tier} "
            f"score={record.score if record.score is not None else '-'} "
            f"confidence={record.confidence} "
            f"evidence={len(record.evidence)} discarded={record.discarded_evidence} "
            f"calls={record.provider_calls}"
        )
        if record.error:
            print(f"    error: {record.error}")

    total_calls = sum(r.provider_calls for r in results)
    print(f"\n{len(results)} assessment(s) written; {total_calls} provider call(s)")
    print(f"ledger: {assessments.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())