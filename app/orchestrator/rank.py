"""Rank persisted jobs with evidence-based match assessment.

What changed, and why
---------------------
This entry point used to load jobs, call ``rank_job`` with
``RankingEvidence(0, 0, 0, 0, eligibility="unknown")``, and print the result.
Four fabricated zeroes looked like an assessment and contained no information.
It also raised a raw ``TypeError`` on scraper-shaped records, because ``Job``
requires ``job_id`` and scrapers emit a different shape.

Now:

* Jobs load through the store's persisted shape, or fail with a named error
  identifying the offending record - never a raw traceback.
* There is no fabricated evidence. With no provider configured, this reports
  that matching is **unavailable** and exits non-zero rather than printing a
  table of zeroes.
* A provider is used only when explicitly enabled, and its output is validated
  with :mod:`app.llm.validation` under a bounded repair budget.
* Only a confirmed hard eligibility veto excludes. Scores reorder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from app.jobs.eligibility import evaluate_eligibility
from app.jobs.match import (
    EvidenceItem,
    MatchError,
    MatchResult,
    assess_match,
    rank_matches,
    summarise,
)
from app.jobs.models import Job
from app.jobs.status import ReviewStatus, StatusLog
from app.jobs.store import JobStore
from app.llm.exceptions import SchemaValidationError
from app.llm.provider import LLMRequest
from app.llm.validation import build_repair_prompt, validate_json_dict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"

#: Bounded repair budget. Model output is validated and, on failure, asked for
#: again a fixed number of times. Bounded because an unbounded retry loop
#: against a misconfigured model is indistinguishable from a hang.
MAX_REPAIRS = 2

REQUIRED_EVIDENCE_KEYS = ["technical", "experience", "behavioral", "career"]


class LoadError(ValueError):
    """A persisted job record could not be turned into a :class:`Job`."""


class MatchUnavailable(RuntimeError):
    """No usable provider, so no honest assessment can be produced."""


def load_jobs(store: JobStore) -> List[Job]:
    """Load persisted jobs, naming any record that cannot be read.

    Raises :class:`LoadError` listing the offending job ids instead of letting a
    ``TypeError`` escape from a dataclass constructor - a traceback from deep
    inside a constructor tells you nothing about which record is bad or why.
    """
    jobs: List[Job] = []
    failures: List[str] = []

    try:
        records = store.load_jobs(strict=True)
    except ValueError as error:
        # The store found records it could not fold. That is the same class of
        # problem as a record missing a required field, and is reported the
        # same way rather than escaping as a bare ValueError.
        raise LoadError(str(error)) from error

    for index, record in enumerate(records):
        payload = record.get("job") if isinstance(record, dict) else None
        if not isinstance(payload, dict):
            failures.append(f"record {index}: expected a 'job' object")
            continue
        if not str(payload.get("job_id") or record.get("job_id") or "").strip():
            failures.append(f"record {index}: missing job_id")
            continue
        try:
            jobs.append(_as_job(payload, record))
        except (TypeError, ValueError) as error:
            job_id = payload.get("job_id") or record.get("job_id") or f"index {index}"
            failures.append(f"job {job_id!r}: {error}")

    if failures:
        raise LoadError(
            f"{len(failures)} persisted record(s) could not be read:\n  "
            + "\n  ".join(failures)
        )
    return jobs


def _as_job(payload: Mapping[str, Any], record: Mapping[str, Any]) -> Job:
    """Rebuild a Job from its persisted form."""
    from app.jobs.models import RemoteStatus

    fields = {
        name: value
        for name, value in payload.items()
        if name in Job.__dataclass_fields__
    }
    if not str(fields.get("job_id", "")).strip():
        fields["job_id"] = str(record.get("job_id", ""))
    missing = [
        name for name in ("job_id", "title", "company", "url")
        if not str(fields.get(name, "")).strip()
    ]
    if missing:
        # Reported here rather than letting Job's constructor raise: a message
        # naming the absent fields is actionable, "missing 2 required
        # positional arguments" is not.
        raise ValueError(f"missing required field(s): {', '.join(missing)}")
    status = fields.get("remote_status")
    if isinstance(status, str):
        try:
            fields["remote_status"] = RemoteStatus(status)
        except ValueError:
            fields["remote_status"] = RemoteStatus.UNKNOWN
    return Job(**fields)


def evidence_from_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate model output into a shape :func:`assess_match` accepts.

    Raises :class:`SchemaValidationError` rather than coercing: a score the
    model returned as ``"high"`` or omitted entirely is a validation failure,
    and papering over it is how fabricated numbers get in.
    """
    data = validate_json_dict(
        json.dumps(payload), REQUIRED_EVIDENCE_KEYS,
        {key: int for key in REQUIRED_EVIDENCE_KEYS},
    )
    evidence = []
    for item in data.get("evidence", []) or []:
        if not isinstance(item, dict):
            continue
        evidence.append(
            EvidenceItem(
                claim=str(item.get("claim", "")),
                source=str(item.get("source", "")),
                quote=str(item.get("quote", "")),
            )
        )
    return {
        "scores": {key: int(data[key]) for key in REQUIRED_EVIDENCE_KEYS},
        "evidence": evidence,
        "gaps": [str(g) for g in data.get("gaps", []) or []],
        "concerns": [str(c) for c in data.get("concerns", []) or []],
    }


SYSTEM_PROMPT = (
    "You assess job postings against a candidate profile. You return evidence, "
    "not decisions.\n"
    "Hard rules:\n"
    "- Return ONLY a JSON object. No prose, no markdown fence.\n"
    "- Keys technical, experience, behavioral, career are integers 0-100.\n"
    "- Every evidence quote MUST be copied verbatim from the posting or the "
    "profile. Never paraphrase a quote.\n"
    "- Never invent a qualification, employer, date, or skill. If the posting "
    "does not state a requirement, leave it out and say so in gaps.\n"
    "- If you cannot tell, say so. An honest gap is more useful than a guess."
)


def request_evidence(job: Job, profile: str, provider: Any, *, model: str) -> Dict[str, Any]:
    """Ask a provider for structured evidence, validating with bounded repair.

    The provider supplies evidence and explanations only. The score is
    aggregated deterministically by :mod:`app.jobs.match`; the model is never
    asked for a final score and never supplies one.

    Repair is bounded at :data:`MAX_REPAIRS`. An unbounded retry loop against a
    misconfigured model is indistinguishable from a hang.
    """
    prompt = (
        f"CANDIDATE PROFILE:\n{profile}\n\n"
        f"JOB POSTING:\n{job.title} at {job.company}\n{job.location}\n{job.description}\n\n"
        "Return the JSON object described in your instructions."
    )
    last_error: Optional[Exception] = None
    last_raw = ""

    for attempt in range(MAX_REPAIRS + 1):
        request = LLMRequest(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt if attempt == 0 else build_repair_prompt(
                last_raw, [str(last_error)]
            ),
            model=model,
            temperature=0.0,
        )
        last_raw = provider.generate(request).text or ""
        try:
            return evidence_from_payload(json.loads(last_raw))
        except (SchemaValidationError, json.JSONDecodeError) as error:
            last_error = error

    raise SchemaValidationError(
        f"provider output failed validation after {MAX_REPAIRS} repair attempts: "
        f"{last_error}"
    )


def assess_jobs(
    store: JobStore,
    *,
    provider: Optional[Any] = None,
    model: str = "llama3.2",
    profile: Optional[str] = None,
    candidate_country: str = "KE",
) -> List[MatchResult]:
    """Assess every persisted job, or explain why it cannot be done.

    Eligibility is evaluated per job and passed through, so the single
    permitted automatic exclusion - a confirmed hard veto - is reachable in the
    real pipeline rather than only in tests.
    """
    jobs = load_jobs(store)

    if provider is None:
        # No provider means no evidence. Producing a table anyway is exactly
        # the defect this replaces, so say so and refuse.
        raise MatchUnavailable(
            "no provider is configured, so no match evidence can be gathered. "
            "Ranking without evidence would mean inventing it. Re-run with "
            "--enable-ai (and a reachable Ollama) to assess, or read the "
            "dashboard, which needs no provider."
        )

    results: List[MatchResult] = []
    for job in jobs:
        eligibility = evaluate_eligibility(job, candidate_country=candidate_country)
        if profile:
            try:
                gathered = request_evidence(job, profile, provider, model=model)
                result = assess_match(
                    job,
                    gathered["scores"],
                    evidence=gathered["evidence"],
                    gaps=gathered["gaps"],
                    concerns=gathered["concerns"],
                    profile=profile,
                    eligibility=eligibility,
                )
            except SchemaValidationError as error:
                # An unusable provider response is reported as unusable output,
                # never coerced into a score.
                result = assess_match(
                    job, None, profile=profile, eligibility=eligibility,
                    concerns=[f"provider output invalid: {error}"],
                )
        else:
            result = assess_match(job, None, profile=profile, eligibility=eligibility)
        results.append(result)
    return rank_matches(results)


def status_of(store: JobStore) -> Dict[str, str]:
    """Current review status per job, to annotate the ranking output."""
    try:
        log = StatusLog(store)
        return {
            str(record.get("job_id", "")): log.current(str(record.get("job_id", ""))).value
            for record in store.load_jobs()
        }
    except Exception:
        # Status is a convenience annotation here; its absence must not stop
        # ranking. A malformed status log is reported in the output instead.
        return {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assess persisted jobs with evidence-based matching"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", default="llama3.2")
    parser.add_argument(
        "--enable-ai",
        action="store_true",
        help="use the local Ollama provider (off unless explicitly enabled)",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="path to a candidate profile text file",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    store = JobStore(args.data_dir)

    profile_text = None
    if args.profile:
        try:
            profile_text = Path(args.profile).read_text(encoding="utf-8")
        except OSError as error:
            print(f"match failed: cannot read profile: {error}", file=sys.stderr)
            return 1

    provider = None
    if args.enable_ai:
        try:
            from app.llm.ollama import OllamaProvider

            candidate = OllamaProvider(model=args.model)
            if not candidate.health_check():
                print(
                    "match failed: --enable-ai was given but Ollama is not reachable",
                    file=sys.stderr,
                )
                return 1
            provider = candidate
        except Exception as error:
            print(f"match failed: could not initialise provider: {error}", file=sys.stderr)
            return 1

    try:
        results = assess_jobs(
            store, provider=provider, model=args.model, profile=profile_text
        )
    except (MatchUnavailable, LoadError, MatchError) as error:
        print(f"match failed: {error}", file=sys.stderr)
        return 1

    statuses = status_of(store)
    print(json.dumps({
        "summary": summarise(results),
        "results": [
            {**result.to_dict(), "review_status": statuses.get(result.job_id, "new")}
            for result in results
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())