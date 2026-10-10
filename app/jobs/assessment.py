"""Bounded, evidence-checked match assessment.

What this is
------------
Recruiter-style matching, run **by hand, on jobs a person chose**. There is no
automatic pass over the queue, and there is deliberately no way to ask for one:
assessment costs money of attention and, when a provider is configured, sends
job text off this machine. Both are things a person should authorise each time.

The hard rules, and why each exists
-----------------------------------
**No provider, no assessment.** A provider must be explicitly registered and
reachable. The registry ships *empty*. There is no default and no fallback, so
running this with nothing configured cannot silently reach anything - and it
makes no writes at all, not even a record saying it failed.

**Assessments are not jobs.** They live in their own append-only log. A match is
a judgement that can be wrong and can be redone; a job record is the thing that
was actually published. Letting an assessment rewrite one would put a mutable
judgement inside the most authoritative thing in the store.

**Evidence is checked, not believed.** Every quote a provider returns is looked
for in the actual job record or the actual profile. A quote that cannot be
found is discarded. If that leaves nothing to stand on, the result becomes
``not_yet_evaluated`` - never a tier nobody can support. A model asserting a
qualification is not evidence that the candidate has it.

**Bounded calls, always.** One generate, plus at most one repair, via the
existing :func:`app.llm.provider.generate_structured`. There is no retry loop
and no unbounded retry anywhere in this module.

**A match reorders and explains. Nothing more.** The record produced here cannot
remove a job, close one, change its status, or alter eligibility or freshness.
The queue reads assessments; it never filters on them to hide anything.

Data boundary
-------------
A configured provider means job text and profile text leave this machine. That
is a real disclosure and the operator is the one who decides to make it, by
configuring a provider at all. Nothing here sends anything until one exists, and
:func:`describe_data_boundary` states plainly what would be sent so that decision
can be made with the facts in hand.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from app.jobs.match import (
    Confidence,
    EvidenceItem,
    MatchTier,
    verify_evidence,
)
from app.jobs.models import Job
from app.llm.exceptions import LLMError, ProviderConnectionError
from app.llm.provider import LLMProvider, LLMRequest, generate_structured

#: Bumped when the record shape changes. An old record read under a new schema
#: is a different fact, not the same one interpreted differently.
ASSESSMENT_SCHEMA_VERSION = 1

#: Bumped when the prompt changes, so an old assessment is never silently
#: compared against a new one as though both answered the same question.
PROMPT_VERSION = "match-v1"

#: Refuse a batch larger than this outright rather than truncating it. A caller
#: asking for 500 has made a mistake, and quietly assessing 20 of them would
#: hide that.
MAX_BATCH = 25

ASSESSMENTS_FILENAME = "matches.jsonl"

#: Recorded so a reader can tell an assessment from a refusal without parsing
#: the whole record.
STATUS_ASSESSED = "assessed"
STATUS_INSUFFICIENT = "insufficient_evidence"
STATUS_NOT_EVALUATED = "not_yet_evaluated"
STATUS_REFUSED = "refused"
STATUS_ERROR = "error"


class AssessmentUnavailable(RuntimeError):
    """No usable provider, or no usable candidate profile.

    Raised *before* anything is written. A run that cannot assess must leave the
    store exactly as it found it.
    """


class ProfileUnavailable(AssessmentUnavailable):
    """No structured candidate profile, so nothing can be evidenced."""


# ----------------------------------------------------------------------
# providers
# ----------------------------------------------------------------------

#: Explicitly registered provider factories, keyed by name. **Empty by design.**
#:
#: There is no default provider and no automatic discovery. A provider becomes
#: available only because someone registered one, which is the moment the data
#: boundary in :func:`describe_data_boundary` starts to apply.
_REGISTRY: Dict[str, Callable[[], LLMProvider]] = {}


def register_provider(name: str, factory: Callable[[], LLMProvider]) -> None:
    """Register a provider factory under ``name``."""
    if not callable(factory):
        raise ValueError("provider factory must be callable")
    _REGISTRY[name] = factory


def registered_providers() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def resolve_provider(name: Optional[str]) -> Tuple[str, LLMProvider]:
    """Return ``(name, provider)`` for an explicitly named, reachable provider.

    Raises :class:`AssessmentUnavailable` when nothing is registered, when the
    name is unknown, or when the provider is not reachable. In every one of
    those cases the caller writes nothing.
    """
    if not name:
        raise AssessmentUnavailable(
            "no provider named. Register one explicitly and pass --provider; "
            "there is no default, because naming one is what authorises sending "
            "job and profile text to it."
        )
    factory = _REGISTRY.get(name)
    if factory is None:
        known = ", ".join(registered_providers()) or "none registered"
        raise AssessmentUnavailable(
            f"unknown provider {name!r}; registered providers: {known}"
        )
    try:
        provider = factory()
    except Exception as exc:  # noqa: BLE001 - construction failure is unavailability
        raise AssessmentUnavailable(
            f"provider {name!r} could not be constructed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    try:
        reachable = provider.health_check()
    except Exception as exc:  # noqa: BLE001
        raise AssessmentUnavailable(
            f"provider {name!r} is not reachable: {type(exc).__name__}: {exc}"
        ) from exc
    if not reachable:
        raise AssessmentUnavailable(
            f"provider {name!r} reports itself unreachable; no assessment made"
        )
    return name, provider


def describe_data_boundary() -> str:
    """Plain statement of what a provider call would disclose."""
    return (
        "Assessing a job sends the job's title, company, location, description "
        "and requirements, plus the candidate profile text, to the configured "
        "provider. Nothing is sent until a provider is registered and named on "
        "the command line."
    )


# ----------------------------------------------------------------------
# the candidate profile gate
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateProfile:
    """The profile text matching is allowed to cite, and where it came from."""

    text: str
    sources: Tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        """Usable only if it has real content.

        A present-but-empty profile is not a profile. Matching against it would
        produce confident-sounding findings with nothing behind them.
        """
        return bool(self.text.strip())


#: Plain-text profile sources, in the order the schema calls authoritative.
#:
#: Read directly rather than through
#: :func:`app.profile.extract.build_source_documents`, which shells out to
#: ``pdftotext`` for binary documents. One PDF that cannot be converted raises
#: out of that call, and a single raise there loses the master CV and the
#: preference file with it - so an unrelated attachment would report "no usable
#: profile" and refuse every assessment while the authoritative facts sit on
#: disk unread. Reading the text sources directly is deterministic and cannot be
#: broken by a file nobody asked for.
PROFILE_TEXT_PATHS = ("cv/main_example.tex", "CLAUDE.md")

#: Extensions safe to read as text. Anything else is skipped rather than
#: converted: converting is a different subsystem's job and this one must not
#: shell out.
PROFILE_TEXT_SUFFIXES = frozenset({".txt", ".md", ".tex"})


def load_candidate_profile(repo_root: Path) -> CandidateProfile:
    """Load the profile the rest of the project treats as authoritative.

    No fact is invented and none is defaulted. A source that is missing simply
    contributes nothing; if nothing usable comes back the profile is unusable
    and assessment refuses, which is the correct direction to fail.
    """
    root = Path(repo_root)
    parts: List[str] = []
    sources: List[str] = []

    for relative in PROFILE_TEXT_PATHS:
        path = root / relative
        try:
            if path.exists():
                text = path.read_text(encoding="utf-8")
                if text.strip():
                    parts.append(text)
                    sources.append(relative)
        except (OSError, UnicodeDecodeError):
            # An unreadable authoritative file is a gap, not a reason to
            # abandon the others.
            continue

    documents_dir = root / "documents"
    try:
        candidates = sorted(documents_dir.rglob("*")) if documents_dir.exists() else []
    except OSError:
        candidates = []
    for path in candidates:
        if not path.is_file() or path.suffix.casefold() not in PROFILE_TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if text.strip():
            parts.append(text)
            sources.append(str(path.relative_to(root)))

    return CandidateProfile(text="\n\n".join(parts), sources=tuple(sources))


# ----------------------------------------------------------------------
# records
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class AssessmentRecord:
    """One assessment attempt, whatever became of it.

    Failures are recorded too. A record saying "the provider was unreachable at
    this time" is evidence; its absence leaves a gap that looks like a decision
    not to assess.
    """

    job_id: str
    at: str
    status: str
    provider: str = ""
    model: str = ""
    schema_version: int = ASSESSMENT_SCHEMA_VERSION
    prompt_version: str = PROMPT_VERSION
    tier: str = MatchTier.NOT_YET_EVALUATED.value
    score: Optional[float] = None
    confidence: str = Confidence.INSUFFICIENT.value
    evidence: Tuple[Dict[str, str], ...] = ()
    missing_requirements: Tuple[str, ...] = ()
    concerns: Tuple[str, ...] = ()
    error: str = ""
    #: Provider calls actually made. Bounded by construction; recorded so the
    #: bound is visible rather than assumed.
    provider_calls: int = 0
    #: Evidence the provider returned that could not be found in the source
    #: records. Kept so a model that keeps inventing quotes is visible.
    discarded_evidence: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "at": self.at,
            "status": self.status,
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "tier": self.tier,
            "score": self.score,
            "confidence": self.confidence,
            "evidence": [dict(e) for e in self.evidence],
            "missing_requirements": list(self.missing_requirements),
            "concerns": list(self.concerns),
            "error": self.error,
            "provider_calls": self.provider_calls,
            "discarded_evidence": self.discarded_evidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AssessmentRecord":
        return cls(
            job_id=str(data.get("job_id", "")),
            at=str(data.get("at", "")),
            status=str(data.get("status", STATUS_NOT_EVALUATED)),
            provider=str(data.get("provider", "")),
            model=str(data.get("model", "")),
            schema_version=int(data.get("schema_version", ASSESSMENT_SCHEMA_VERSION) or 1),
            prompt_version=str(data.get("prompt_version", PROMPT_VERSION)),
            tier=str(data.get("tier", MatchTier.NOT_YET_EVALUATED.value)),
            score=data.get("score"),
            confidence=str(data.get("confidence", Confidence.INSUFFICIENT.value)),
            evidence=tuple(dict(e) for e in data.get("evidence", []) or ()),
            missing_requirements=tuple(data.get("missing_requirements", []) or ()),
            concerns=tuple(data.get("concerns", []) or ()),
            error=str(data.get("error", "")),
            provider_calls=int(data.get("provider_calls", 0) or 0),
            discarded_evidence=int(data.get("discarded_evidence", 0) or 0),
        )

    @property
    def assessed(self) -> bool:
        return self.status == STATUS_ASSESSED


class AssessmentStore:
    """Append-only ``data/matches.jsonl``.

    Deliberately not ``jobs.jsonl``. A job record is what the source published;
    an assessment is a judgement about it, revisable and sometimes wrong. Prior
    assessments are never rewritten, so the history of what was thought, and
    when, stays readable - which is the whole reason for keeping a log.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.path = self.data_dir / ASSESSMENTS_FILENAME

    def append(self, record: AssessmentRecord) -> None:
        """Append one record. Nothing is ever updated in place."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        # Opened per write and flushed, so a record is durable the moment it is
        # acknowledged and an interrupted run keeps the ones it completed.
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def load(self) -> List[AssessmentRecord]:
        """Every assessment on record, oldest first."""
        if not self.path.exists():
            return []
        records: List[AssessmentRecord] = []
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(AssessmentRecord.from_dict(json.loads(line)))
                    except (ValueError, TypeError):
                        # A torn final line must not hide the records before it.
                        continue
        except OSError:
            return []
        return records

    def latest(self) -> Dict[str, AssessmentRecord]:
        """The most recent record per job.

        Later records win because the log is append-only and the newest
        assessment is the current one - but the earlier ones remain on disk.
        """
        current: Dict[str, AssessmentRecord] = {}
        for record in self.load():
            current[record.job_id] = record
        return current


# ----------------------------------------------------------------------
# prompting and validation
# ----------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You assess how well a candidate matches a job posting. Return only JSON.\n"
    "Every claim you make must cite a quote copied verbatim from the JOB or "
    "the PROFILE. A claim with no citation is not allowed: if you cannot quote "
    "support for something, list it under missing_requirements instead.\n"
    "Do not invent qualifications, do not assume, and do not be encouraging. "
    "If the evidence does not reach a conclusion, return tier "
    "'not_yet_evaluated' with confidence 'insufficient'.\n"
    "Schema:\n"
    '{"tier": "strong_match|credible_match|stretch|unsuitable|not_yet_evaluated",'
    ' "score": <number between 0 and 1 or null>,'
    ' "confidence": "high|medium|low|insufficient",'
    ' "evidence": [{"claim": "...", "source": "job|profile", "quote": "verbatim"}],'
    ' "missing_requirements": ["..."], "concerns": ["..."]}'
)


def build_request(job: Job, profile: str, model: Optional[str] = None) -> LLMRequest:
    """The single bounded request for one job.

    Temperature is fixed at zero. An assessment that re-reads differently each
    time is not a record of anything.
    """
    job_text = "\n".join(filter(None, [
        f"TITLE: {job.title}",
        f"COMPANY: {job.company}",
        f"LOCATION: {job.location}",
        f"REQUIREMENTS: {job.skills}",
        f"DESCRIPTION: {job.description}",
    ]))
    return LLMRequest(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=(
            f"PROFILE:\n{profile}\n\nJOB:\n{job_text}\n\n"
            "Return the JSON object described in the system prompt."
        ),
        model=model,
        temperature=0.0,
        metadata={"prompt_version": PROMPT_VERSION,
                  "schema_version": str(ASSESSMENT_SCHEMA_VERSION)},
    )


def _parse_response(text: str) -> Dict[str, Any]:
    """Parse provider output into a mapping, or raise.

    Tolerates a fenced code block because models emit one even when told not
    to, but nothing else is forgiven: prose around the JSON is not a finding.
    """
    body = (text or "").strip()
    if body.startswith("```"):
        lines = body.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        body = "\n".join(lines).strip()
    if not body:
        raise ValueError("provider returned nothing")
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("provider output is not a JSON object")
    return data


def _coerce_tier(value: Any) -> MatchTier:
    try:
        return MatchTier(str(value))
    except ValueError:
        # An unrecognised tier is data we did not anticipate. Reporting it as a
        # real tier would be worse than admitting we do not know.
        return MatchTier.NOT_YET_EVALUATED


def _coerce_confidence(value: Any) -> Confidence:
    try:
        return Confidence(str(value))
    except ValueError:
        return Confidence.INSUFFICIENT


def _coerce_score(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    # A score outside 0..1 is nonsense, not a very good or very bad match.
    return score if 0.0 <= score <= 1.0 else None


def build_assessment(
    job: Job,
    profile: str,
    raw: Mapping[str, Any],
    *,
    provider: str,
    model: str,
    at: str,
    provider_calls: int,
) -> AssessmentRecord:
    """Turn raw provider output into a record, verifying every quote.

    The tier a provider asserts is never taken on trust. If it asserted a tier
    but nothing it said survives checking, the result is downgraded to
    ``not_yet_evaluated`` - the honest state is "we ran it and learned nothing",
    which is different from "it is a poor match".
    """
    claimed = [
        EvidenceItem(
            claim=str(item.get("claim", "")),
            source=str(item.get("source", "")),
            quote=str(item.get("quote", "")),
        )
        for item in (raw.get("evidence") or [])
        if isinstance(item, Mapping)
    ]
    verified = verify_evidence(claimed, job, profile)
    discarded = len(claimed) - len(verified)

    tier = _coerce_tier(raw.get("tier"))
    confidence = _coerce_confidence(raw.get("confidence"))
    score = _coerce_score(raw.get("score"))

    if tier is not MatchTier.NOT_YET_EVALUATED and not verified:
        # A finding with no surviving evidence is not a finding.
        tier = MatchTier.NOT_YET_EVALUATED
        score = None
        confidence = Confidence.INSUFFICIENT

    if tier is MatchTier.NOT_YET_EVALUATED:
        confidence = Confidence.INSUFFICIENT
        score = None

    status = (
        STATUS_ASSESSED if tier is not MatchTier.NOT_YET_EVALUATED
        else STATUS_INSUFFICIENT
    )
    return AssessmentRecord(
        job_id=str(job.job_id),
        at=at,
        status=status,
        provider=provider,
        model=model,
        tier=tier.value,
        score=score,
        confidence=confidence.value,
        evidence=tuple(
            {"claim": e.claim, "source": e.source, "quote": e.quote}
            for e in verified
        ),
        missing_requirements=tuple(
            str(m) for m in (raw.get("missing_requirements") or [])
        ),
        concerns=tuple(str(c) for c in (raw.get("concerns") or [])),
        provider_calls=provider_calls,
        discarded_evidence=discarded,
    )


def to_match_dict(record: AssessmentRecord) -> Dict[str, Any]:
    """The shape :func:`app.reporting.jobs.build_view` consumes.

    An unassessed record is passed as ``None`` rather than as a
    ``not_yet_evaluated`` result, because the two are different facts: one means
    nobody looked, the other means we looked and could not conclude.
    """
    if not record.assessed:
        return {"job_id": record.job_id, "tier": record.tier,
                "score": record.score, "confidence": record.confidence,
                "evidence": list(record.evidence),
                "missing_requirements": list(record.missing_requirements),
                "concerns": list(record.concerns),
                "insufficient_reason": "assessment recorded as insufficient"
                if record.status == STATUS_INSUFFICIENT else
                "assessment recorded as not yet evaluated",
                "vetoes": [], "eligibility": "unknown",
                "keyword_overlap": []}
    return {
        "job_id": record.job_id,
        "tier": record.tier,
        "score": record.score,
        "confidence": record.confidence,
        "evidence": list(record.evidence),
        "missing_requirements": list(record.missing_requirements),
        "concerns": list(record.concerns),
        "insufficient_reason": "",
        "vetoes": [],
        "eligibility": "unknown",
        "keyword_overlap": [],
    }


# ----------------------------------------------------------------------
# assessing
# ----------------------------------------------------------------------


class _CountingProvider:
    """Delegates to a provider and counts ``generate`` calls.

    Counting here rather than guessing afterwards is what makes the bound
    visible in the record. ``generate_structured`` may issue one repair call on
    top of the first, and a run that reports "1 call" while having made two
    would misrepresent its own cost.
    """

    def __init__(self, inner: LLMProvider):
        self._inner = inner
        self.calls = 0

    def generate(self, request: LLMRequest):
        self.calls += 1
        return self._inner.generate(request)

    def health_check(self) -> bool:
        return self._inner.health_check()

    def list_models(self) -> list:
        return self._inner.list_models()


def assess_one(
    job: Job,
    *,
    profile: CandidateProfile,
    provider_name: str,
    provider: LLMProvider,
    store: AssessmentStore,
    model: Optional[str] = None,
) -> AssessmentRecord:
    """Assess one job. Bounded to one generate plus at most one repair.

    Every failure path still writes a record. "We tried and could not" is a fact
    worth keeping; leaving no trace would make a gap look like a decision.
    """
    at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if not profile.usable:
        record = AssessmentRecord(
            job_id=str(job.job_id), at=at, status=STATUS_REFUSED,
            provider=provider_name, error=(
                "no usable candidate profile, so no claim could be evidenced"
            ),
        )
        store.append(record)
        return record

    counter = _CountingProvider(provider)
    request = build_request(job, profile.text, model=model)

    def validator(text: str) -> Dict[str, Any]:
        return _parse_response(text)

    try:
        payload = generate_structured(counter, request, validator)
    except (LLMError, ProviderConnectionError, ValueError, json.JSONDecodeError) as exc:
        record = AssessmentRecord(
            job_id=str(job.job_id), at=at, status=STATUS_ERROR,
            provider=provider_name, model=model,
            error=f"{type(exc).__name__}: {exc}",
            provider_calls=counter.calls,
        )
        store.append(record)
        return record

    resolved_model = _model_name(provider, request)
    record = build_assessment(
        job, profile.text, payload,
        provider=provider_name,
        model=resolved_model,
        at=at,
        provider_calls=counter.calls,
    )
    store.append(record)
    return record


def _model_name(provider: LLMProvider, request: LLMRequest) -> str:
    """The model actually used, for the audit trail.

    Best effort: a provider that will not say which model it used gets an empty
    string rather than a guess. An invented model name in an audit record is
    worse than an absent one.
    """
    if request.model:
        return str(request.model)
    inner = getattr(provider, "_inner", provider)
    for attribute in ("model", "default_model"):
        value = getattr(inner, attribute, None)
        if isinstance(value, str) and value:
            return value
    return ""


def assess_batch(
    jobs: Sequence[Job],
    *,
    profile: CandidateProfile,
    provider_name: str,
    provider: LLMProvider,
    store: AssessmentStore,
    max_n: int = MAX_BATCH,
    model: Optional[str] = None,
) -> List[AssessmentRecord]:
    """Assess at most ``max_n`` jobs, chosen explicitly by the caller.

    The bound is enforced here rather than trusted to the command line: a caller
    that passes the whole queue is capped, not obeyed.
    """
    if max_n is None or max_n < 1:
        raise ValueError("max_n must be at least 1")
    if max_n > MAX_BATCH:
        raise ValueError(
            f"max_n={max_n} exceeds the hard ceiling of {MAX_BATCH}. Assess a "
            "smaller, deliberately chosen batch."
        )
    selected = list(jobs)[:max_n]
    return [
        assess_one(job, profile=profile, provider_name=provider_name,
                   provider=provider, store=store, model=model)
        for job in selected
    ]