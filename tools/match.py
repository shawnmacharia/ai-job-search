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
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.jobs.assessment import (  # noqa: E402
    LOCAL_OLLAMA_PROVIDER,
    LOCAL_OLLAMA_URL,
    LOCAL_TRIAL_KEEP_ALIVE,
    LOCAL_TRIAL_TIMEOUT_SECONDS,
    MAX_BATCH,
    AssessmentUnavailable,
    ProfileUnavailable,
    assess_batch,
    assess_one,
    build_local_trial,
    describe_data_boundary,
    load_candidate_profile,
    plan_only,
    register_local_ollama,
    registered_providers,
    resolve_provider,
    run_local_trial,
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
    parser.add_argument("--job-id", action="append", default=None,
                        help="assess exactly this job; repeat for several")
    parser.add_argument("--batch", type=int, default=None,
                        help=f"assess at most N queued jobs (ceiling {MAX_BATCH})")
    parser.add_argument("--source", default=None,
                        help="restrict the batch to one source")
    parser.add_argument("--providers", action="store_true",
                        help="list registered providers and exit")
    parser.add_argument("--boundary", action="store_true",
                        help="state the data boundary and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="describe the run in full; zero provider calls, "
                             "zero writes")
    parser.add_argument("--allow-remote", action="store_true",
                        help="permit a non-local provider (off by default)")
    parser.add_argument("--register-local-ollama", action="store_true",
                        help="register a local Ollama provider, then exit "
                             "without assessing anything")
    parser.add_argument("--confirm-local-provider-boundary", action="store_true",
                        help="confirm you have read the boundary being printed")
    parser.add_argument("--endpoint", default=LOCAL_OLLAMA_URL,
                        help="endpoint for --register-local-ollama "
                             "(must be loopback)")
    parser.add_argument("--local-trial", action="store_true",
                        help="assess through the local trial path, which loads "
                             "the model once and records the cold-start "
                             "duration in every record")
    parser.add_argument("--timeout", type=float, default=None,
                        help="request timeout for --local-trial")
    parser.add_argument("--keep-alive", default=None,
                        help="model residency for --local-trial, e.g. 30m")
    parser.add_argument("--preload", dest="preload", action="store_true",
                        default=True, help="load the model once (default)")
    parser.add_argument("--no-preload", dest="preload", action="store_false",
                        help="skip the load; cold start goes unrecorded")
    parser.add_argument("--allow-repair", action="store_true",
                        help="permit one bounded repair per job")
    parser.add_argument("--approve-local-trial", action="store_true",
                        help="confirm the local trial may send requests")
    return parser


#: A trial is three jobs: enough to see whether the output is worth trusting,
#: few enough that a bad answer costs little attention.
TRIAL_CEILING = 3


def _run_local_trial(args, jobs, profile, assessments):
    """Assess through the trial path so preload timing is recorded.

    Every refusal here happens before any request: an unapproved trial, a cloud
    model, a remote endpoint or a batch over the ceiling all return without the
    model being loaded or contacted.
    """
    model = args.model or ""
    if not model:
        print("refusing: --local-trial requires --model", file=sys.stderr)
        return None
    try:
        plan = build_local_trial(
            [str(job.job_id) for job in jobs],
            endpoint=args.endpoint,
            model=model,
            timeout_seconds=args.timeout if args.timeout is not None
            else LOCAL_TRIAL_TIMEOUT_SECONDS,
            keep_alive=args.keep_alive or LOCAL_TRIAL_KEEP_ALIVE,
            preload=args.preload,
        )
    except AssessmentUnavailable as error:
        print(f"refusing: {error}", file=sys.stderr)
        print("no request was made and nothing was written.", file=sys.stderr)
        return None

    print(f"local trial plan: {json.dumps(plan.to_dict(), sort_keys=True)}")
    try:
        results = run_local_trial(
            plan, profile=profile, store=assessments, jobs=jobs,
            provider_name=LOCAL_OLLAMA_PROVIDER,
            approved=args.approve_local_trial,
            allow_repair=args.allow_repair,
        )
    except (AssessmentUnavailable, ProfileUnavailable) as error:
        print(f"refusing: {error}", file=sys.stderr)
        print("no request was made and nothing was written.", file=sys.stderr)
        return None
    cold = results[0].cold_start_seconds
    print(f"cold start: {cold if cold is not None else 'not recorded'}s")
    return results


def _print_boundary(boundary) -> None:
    """The complete boundary, printed immediately before registration."""
    print("DATA BOUNDARY - what this provider may receive\n")
    print(f"endpoint            : {boundary['endpoint']}")
    print(f"model               : {boundary['model']}")
    print(f"local-only verdict  : {boundary['local_only_verdict']}")
    print(f"\nprofile sources ({len(boundary['profile_sources'])}):")
    for source in boundary["profile_sources"]:
        print(f"  - {source}")
    print(f"profile characters  : {boundary['profile_characters']}")
    print(f"\njob fields sent     : {', '.join(boundary['sent_job_fields'])}")
    print(f"job fields NOT sent : {', '.join(boundary['excluded_job_fields'])}")
    print(f"full descriptions   : {boundary['full_description_sent']}")
    print(f"\ncredentials sent    : {boundary['credentials_sent']}")
    print(f"persisted anywhere  : {boundary['persisted']}")
    print(f"\nlocally installed models ({len(boundary['installed_models'])}):")
    for model in boundary["installed_models"]:
        print(f"  - {model}")


def _register_local(args) -> int:
    """Register the approved local provider, then exit having assessed nothing."""
    profile = load_candidate_profile(REPO_ROOT)
    try:
        boundary = register_local_ollama(
            model=args.model,
            endpoint=args.endpoint,
            confirmed=args.confirm_local_provider_boundary,
            profile=profile,
        )
    except AssessmentUnavailable as error:
        print(f"refusing to register: {error}", file=sys.stderr)
        print("nothing was registered and no job was assessed.", file=sys.stderr)
        return 1

    _print_boundary(boundary)
    print(f"\nregistered {LOCAL_OLLAMA_PROVIDER!r} in this process only.")
    print("registration is NOT persisted and does NOT survive this command.")
    print("no job was assessed. Supply --job-id values to assess anything.")
    return 0


def _print_plan(plan) -> None:
    print(f"DRY RUN - no provider called, no match result written\n")
    print(f"provider (would be called): {plan['provider'] or 'none named'}")
    print(f"provider calls planned    : {plan['provider_calls_planned']}")
    print(f"schema version            : {plan['schema_version']}")
    print(f"prompt version            : {plan['prompt_version']}")
    print(f"full descriptions sent    : {plan['full_description_sent']}")
    print(f"\nprofile sources ({len(plan['profile_sources'])}):")
    for source in plan["profile_sources"]:
        print(f"  - {source}")
    print(f"profile characters sent   : {plan['profile_characters']}")
    print(f"\njob fields sent      : {', '.join(plan['sent_job_fields'])}")
    print(f"job fields NOT sent  : {', '.join(plan['excluded_job_fields'])}")
    print(f"\njobs ({len(plan['jobs'])}):")
    for job in plan["jobs"]:
        print(f"  {job['job_id']}")
        print(f"    source url        : {job['url']}")
        print(f"    characters        : {job['total_characters']}")
        for field, count in sorted(job["field_char_counts"].items()):
            print(f"      {field:<12} {count}")
    print(f"\nwrites: {plan['writes']}")


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

    if args.register_local_ollama:
        return _register_local(args)

    selected_ids = list(args.job_id or [])
    if not selected_ids and args.batch is None:
        print("refusing: choose --job-id for one job, or --batch N for a "
              "chosen number. There is no assess-everything mode.", file=sys.stderr)
        return 1

    if selected_ids and args.batch is not None:
        print("refusing: use --job-id or --batch, not both", file=sys.stderr)
        return 1

    if args.batch is not None and (args.batch < 1 or args.batch > MAX_BATCH):
        print(f"refusing: --batch must be between 1 and {MAX_BATCH}",
              file=sys.stderr)
        return 1

    # A trial is three jobs. Enforced here rather than left to the caller's
    # good intentions, because the point of the trial is that it stays small.
    if len(selected_ids) > TRIAL_CEILING:
        print(f"refusing: {len(selected_ids)} job ids given; a trial is at "
              f"most {TRIAL_CEILING}.", file=sys.stderr)
        return 1

    duplicates = {i for i in selected_ids if selected_ids.count(i) > 1}
    if duplicates:
        print(f"refusing: duplicate job ids: {', '.join(sorted(duplicates))}",
              file=sys.stderr)
        return 1

    store = JobStore(Path(args.data_dir))
    profile = load_candidate_profile(REPO_ROOT)

    if not profile.usable:
        print("refusing: no usable candidate profile, so nothing could be "
              "evidenced. No assessment was written.", file=sys.stderr)
        return 1

    if selected_ids:
        stored = {str(r.get("job_id")): r for r in store.load_jobs()}
        missing = [i for i in selected_ids if i not in stored]
        if missing:
            print(f"refusing: no stored job with id(s): {', '.join(missing)}. "
                  "No assessment was written.", file=sys.stderr)
            return 1
        chosen = [_job_from_record(stored[i]) for i in selected_ids]
    else:
        chosen = None

    # Dry run stops here: before any provider is resolved, contacted, or named
    # as reachable. Reviewing the boundary must not require doing the thing
    # being reviewed, and a dry run that health-checked the endpoint would have
    # made a network call it promised not to make.
    if args.dry_run:
        if chosen is None:
            print("refusing: --dry-run needs explicit --job-id values; a "
                  "dry run that silently picked jobs would be the arbitrary "
                  "selection this is meant to avoid.", file=sys.stderr)
            return 1
        try:
            plan = plan_only(chosen, profile=profile, provider_name=args.provider)
        except ProfileUnavailable as error:
            print(f"refusing: {error}", file=sys.stderr)
            return 1
        _print_plan(plan)
        return 0

    # Built before either branch: both need it, and constructing it after a
    # branch that uses it left the name unbound on that path.
    assessments = AssessmentStore(Path(args.data_dir))

    # The local trial builds and owns its own provider, so it must not be gated on
    # a registered one. Resolving --provider first would make the flag unusable
    # and, worse, would report the wrong reason for refusing.
    if args.local_trial:
        if chosen is None:
            print("refusing: --local-trial needs explicit --job-id values",
                  file=sys.stderr)
            return 1
        results = _run_local_trial(args, chosen, profile, assessments)
        return 0 if results is not None else 1

    # Provider first, and unconditionally. An unavailable provider must make no
    # writes at all - not even a record saying it failed, because the request
    # never left the machine and there is nothing to record.
    try:
        provider_name, provider = resolve_provider(
            args.provider, local_only=not args.allow_remote)
    except AssessmentUnavailable as error:
        print(f"refusing: {error}", file=sys.stderr)
        print("no assessment was written.", file=sys.stderr)
        return 1

    if chosen is not None:
        results = [assess_one(
            job, profile=profile,
            provider_name=provider_name, provider=provider,
            store=assessments, model=args.model,
        ) for job in chosen]
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