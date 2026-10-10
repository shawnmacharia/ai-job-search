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
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
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

#: The trial ceiling, shared by every path that can send a request. Duplicated
#: as a constant rather than read from the CLI so the library enforces it even
#: when the CLI is bypassed.
TRIAL_JOB_CEILING = 3

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

#: What a request was for. A repair is not a retry: it is a second attempt at
#: parsing the same answer, recorded separately so "the model needed fixing"
#: never reads as "the request was sent twice".
CALL_FIRST = "first_call"
CALL_REPAIR = "repair"
CALL_RETRY = "retry"


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


#: Hosts that cannot leave this machine.
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})

#: Substrings marking a model that runs on Ollama's cloud even though the client
#: talks to localhost. A `:cloud` tag means the endpoint is local but the
#: *inference is not* - data goes out and results come back. Locality has to be
#: checked on the model, not only the URL, or the guarantee is cosmetic.
CLOUD_MODEL_MARKERS = ("-cloud", ":cloud")


def is_local_endpoint(base_url: Optional[str]) -> bool:
    """True only for a URL that cannot leave this machine."""
    if not base_url:
        # No URL means no network endpoint at all (an in-process fake in tests).
        return True
    try:
        parsed = urlparse(base_url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https", ""):
        return False
    host = (parsed.hostname or "").casefold()
    return host in LOCAL_HOSTS


def is_local_model(model: Optional[str]) -> bool:
    """True unless the model is one Ollama proxies to its cloud.

    Checked because ``http://localhost:11434`` is also the address of a cloud
    routing endpoint. A model tagged ``:cloud`` leaves the machine; treating
    that as local would make this a promise the code cannot keep.
    """
    if not model:
        return True
    lowered = str(model).casefold()
    return not any(marker in lowered for marker in CLOUD_MODEL_MARKERS)


def assert_local_provider(provider: LLMProvider) -> None:
    """Refuse a provider that would send data off this machine.

    Checks both the endpoint and the model. Either being remote is enough to
    refuse: the trial this exists for is a local-only trial, and a cloud model
    behind a localhost URL would defeat it while looking local.
    """
    base_url = getattr(provider, "base_url", None)
    if not is_local_endpoint(base_url):
        raise AssessmentUnavailable(
            f"provider endpoint {base_url!r} is not local; refusing to send "
            f"job or profile text off this machine"
        )
    model = getattr(provider, "default_model", None) or getattr(provider, "model", None)
    if not is_local_model(model):
        raise AssessmentUnavailable(
            f"model {model!r} is a cloud model: a localhost endpoint can still "
            f"proxy inference off this machine. Choose a locally installed model."
        )


def resolve_provider(name: Optional[str], *, local_only: bool = True) -> Tuple[str, LLMProvider]:
    """Return ``(name, provider)`` for an explicitly named, reachable provider.

    Raises :class:`AssessmentUnavailable` when nothing is registered, when the
    name is unknown, when the provider is not reachable, or - unless
    ``local_only`` is turned off - when it is not local. In every one of those
    cases the caller writes nothing.

    ``local_only`` defaults to True. Turning it off is a deliberate, separate
    decision from naming a provider, and exists so a future non-local provider
    is an explicit act rather than a default someone inherits.
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
    if local_only:
        try:
            assert_local_provider(provider)
        except AssessmentUnavailable as exc:
            raise AssessmentUnavailable(f"provider {name!r}: {exc}") from exc
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


#: The one endpoint this project will register against. Named rather than
#: defaulted, so a caller has to choose it and a typo becomes a refusal.
LOCAL_OLLAMA_URL = "http://localhost:11434"

#: The name the local provider registers under. Deliberately not derived from
#: ``--provider``, so naming it on the command line can never register it.
LOCAL_OLLAMA_PROVIDER = "local-ollama"


def local_provider_boundary(profile: CandidateProfile, *, endpoint: str,
                           model: str, installed: Sequence[str]) -> Dict[str, Any]:
    """The boundary, stated in full, at the moment of registration.

    Returned rather than printed so it can be asserted on in tests, and printed
    by the caller immediately before registering. Registration is the moment the
    data boundary starts to apply, so this is the last point at which someone
    can read it and stop.
    """
    return {
        "endpoint": endpoint,
        "model": model,
        "endpoint_is_local": is_local_endpoint(endpoint),
        "model_is_local": is_local_model(model),
        "local_only_verdict": "LOCAL ONLY" if (
            is_local_endpoint(endpoint) and is_local_model(model)
        ) else "NOT LOCAL",
        "profile_sources": list(profile.sources),
        "profile_characters": len(profile.text),
        "sent_job_fields": sorted(SENT_JOB_FIELDS),
        "excluded_job_fields": sorted(EXCLUDED_JOB_FIELDS),
        "full_description_sent": True,
        "installed_models": list(installed),
        "credentials_sent": 0,
        "persisted": False,
    }


def register_local_ollama(
    *,
    model: str,
    endpoint: str = LOCAL_OLLAMA_URL,
    provider_name: str = LOCAL_OLLAMA_PROVIDER,
    confirmed: bool,
    profile: CandidateProfile,
) -> Dict[str, Any]:
    """Register a local Ollama provider after checking every claim about it.

    Order matters and is deliberate: configuration is validated before anything
    is contacted, locality is asserted before registration, and the boundary is
    built before the provider joins the registry. A caller that has not
    confirmed gets nothing registered and nothing contacted.

    Raises :class:`AssessmentUnavailable` on every refusal, having registered
    nothing in all cases.
    """
    if not confirmed:
        raise AssessmentUnavailable(
            "registration not confirmed. Re-run with "
            "--confirm-local-provider-boundary once you have read the boundary."
        )
    if not is_local_endpoint(endpoint):
        raise AssessmentUnavailable(
            f"endpoint {endpoint!r} is not a loopback address; this path "
            f"registers local providers only"
        )
    if not is_local_model(model):
        raise AssessmentUnavailable(
            f"model {model!r} is a remote-inference model; refused. A localhost "
            f"endpoint can still proxy inference off this machine."
        )
    if not str(model).strip():
        raise AssessmentUnavailable("no model named")

    # Imported here so a normal run does not import an HTTP client it will never
    # use, and so tests can substitute one without touching the network.
    from app.llm.ollama import OllamaProvider

    try:
        provider = OllamaProvider(base_url=endpoint, model=model)
    except Exception as exc:  # noqa: BLE001
        raise AssessmentUnavailable(
            f"provider configuration is malformed: {type(exc).__name__}: {exc}"
        ) from exc

    # The same assertion every other provider goes through, run before the
    # provider is trusted with anything.
    assert_local_provider(provider)

    try:
        reachable = provider.health_check()
    except Exception as exc:  # noqa: BLE001
        raise AssessmentUnavailable(
            f"local Ollama is not reachable: {type(exc).__name__}: {exc}"
        ) from exc
    if not reachable:
        raise AssessmentUnavailable(
            f"local Ollama at {endpoint} is not reachable; refusing to register"
        )

    try:
        installed = list(provider.list_models())
    except Exception as exc:  # noqa: BLE001
        raise AssessmentUnavailable(
            f"could not list local models: {type(exc).__name__}: {exc}"
        ) from exc
    if model not in installed:
        raise AssessmentUnavailable(
            f"model {model!r} is not installed locally. Installed: "
            f"{', '.join(installed) or 'none'}"
        )

    boundary = local_provider_boundary(
        profile, endpoint=endpoint, model=model, installed=installed)

    register_provider(provider_name, lambda: OllamaProvider(
        base_url=endpoint, model=model))
    return boundary


#: Job fields placed in the request. Everything not listed here is not sent -
#: stated explicitly so "what leaves the machine" is answerable by reading a
#: list rather than by reading the prompt-building code and hoping.
SENT_JOB_FIELDS = ("title", "company", "location", "skills", "description")

#: The complement, named so an operator can see what is withheld rather than
#: inferring it from silence.
EXCLUDED_JOB_FIELDS = (
    "job_id", "url", "remote_status", "country", "region",
    "salary_min", "salary_max", "salary_currency", "salary_period",
    "portal", "posted_date", "deadline", "raw_excerpt", "posted_raw",
    "description_complete",
)


def describe_job_payload(job: Job) -> Dict[str, Any]:
    """Exactly what would be sent for ``job``, and what would not.

    Full descriptions **are** sent. That is a deliberate choice - requirements
    are usually inside the description, and truncating it would be truncating
    the evidence - but it means the largest and most revealing field goes out,
    so it is named here rather than left to be discovered.
    """
    sent = {}
    for name in SENT_JOB_FIELDS:
        value = getattr(job, name, None)
        sent[name] = list(value) if isinstance(value, (list, tuple)) else (value or "")
    prompt = build_request(job, "x").user_prompt
    return {
        "job_id": str(job.job_id),
        "url": str(job.url),
        "sent_fields": sorted(sent),
        "excluded_fields": sorted(EXCLUDED_JOB_FIELDS),
        "field_char_counts": {k: len(str(v)) for k, v in sent.items()},
        "full_description_sent": bool(sent["description"]),
        "total_characters": len(prompt),
        "profile_included": True,
    }


def plan_only(
    jobs: Sequence[Job],
    *,
    profile: CandidateProfile,
    provider_name: Optional[str],
) -> Dict[str, Any]:
    """Describe a run in full **without calling a provider or writing anything**.

    The boundary cannot honestly be reviewed if showing it requires doing the
    thing being reviewed. This builds the exact requests that would be sent,
    reports their size, and returns - so what is approved is what will run.
    """
    if not profile.usable:
        raise ProfileUnavailable(
            "no usable candidate profile, so nothing could be evidenced"
        )
    return {
        "provider": provider_name or "",
        "provider_calls_planned": len(jobs),
        "profile_sources": list(profile.sources),
        "profile_characters": len(profile.text),
        "sent_job_fields": sorted(SENT_JOB_FIELDS),
        "excluded_job_fields": sorted(EXCLUDED_JOB_FIELDS),
        "full_description_sent": True,
        "schema_version": ASSESSMENT_SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "jobs": [describe_job_payload(job) for job in jobs],
        "writes": 0,
    }


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
class ProviderIdentity:
    """Who answered, and under what conditions.

    Obtained through :meth:`describe`, never by reaching into a provider's
    attributes. Guessing is how ``model: None`` reached the audit trail: a
    wrapper around a provider exposes none of the attributes being guessed for,
    so a real, working provider produced a record that could not say which model
    answered it. A wrapper now has to *declare* what it is.

    ``model`` is never empty. A provider that cannot name its model is a
    provider we cannot audit, and that is recorded as ``unknown`` rather than
    as a silent ``None``.
    """

    provider: str
    model: str
    endpoint: str = ""
    local_only: Optional[bool] = None
    timeout_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError(
                "provider identity requires a model name; use the literal "
                "'unknown' when the provider cannot supply one"
            )

    @property
    def audit_safe(self) -> bool:
        """False when the model is a placeholder rather than a real name."""
        return self.model.casefold() != UNKNOWN_MODEL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "endpoint": self.endpoint,
            "local_only": self.local_only,
            "timeout_seconds": self.timeout_seconds,
        }


#: What gets recorded when a provider genuinely cannot say which model it is.
#: A visible string, never ``None`` - a null in an audit record reads as a
#: field nobody filled in rather than a fact someone recorded.
UNKNOWN_MODEL = "unknown"


def identity_of(provider: LLMProvider, provider_name: str) -> ProviderIdentity:
    """Ask a provider who it is, via the explicit contract only.

    Falls back to :data:`UNKNOWN_MODEL` rather than guessing at ``model`` or
    ``default_model``: those names are conventions, not a contract, and a
    wrapper that exposes neither is exactly the case that produced the
    ``model: None`` records.
    """
    describe = getattr(provider, "describe", None)
    if not callable(describe):
        return ProviderIdentity(provider=provider_name, model=UNKNOWN_MODEL)
    try:
        data = describe() or {}
    except Exception:  # noqa: BLE001 - an uncooperative provider is unauditable
        return ProviderIdentity(provider=provider_name, model=UNKNOWN_MODEL)
    if not isinstance(data, Mapping):
        return ProviderIdentity(provider=provider_name, model=UNKNOWN_MODEL)
    model = str(data.get("model") or "").strip() or UNKNOWN_MODEL
    endpoint = str(data.get("endpoint") or "")
    local = data.get("local_only")
    timeout = data.get("timeout_seconds")
    return ProviderIdentity(
        provider=provider_name,
        model=model,
        endpoint=endpoint,
        local_only=bool(local) if local is not None else None,
        timeout_seconds=float(timeout) if timeout is not None else None,
    )


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
    #: Endpoint the request went to, and whether it could leave this machine.
    endpoint: str = ""
    local_only: Optional[bool] = None
    #: The timeout the request was actually given, in seconds. Recorded so a
    #: timeout can later be read as "the limit was too low" rather than as an
    #: unqualified provider failure.
    timeout_seconds: Optional[float] = None
    #: ``first_call``, ``repair``, or ``retry``. Distinguishes the call that
    #: did the work from the ones that tried to fix it.
    call_kind: str = CALL_FIRST
    #: Wall-clock seconds spent on the model load, kept apart from generation
    #: so a slow machine is not misread as an incapable model.
    cold_start_seconds: Optional[float] = None
    generation_seconds: Optional[float] = None
    error_type: str = ""
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
            "endpoint": self.endpoint,
            "local_only": self.local_only,
            "timeout_seconds": self.timeout_seconds,
            "call_kind": self.call_kind,
            "cold_start_seconds": self.cold_start_seconds,
            "generation_seconds": self.generation_seconds,
            "error_type": self.error_type,
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
            model=str(data.get("model", "")) or UNKNOWN_MODEL,
            endpoint=str(data.get("endpoint", "")),
            local_only=data.get("local_only"),
            timeout_seconds=data.get("timeout_seconds"),
            call_kind=str(data.get("call_kind", CALL_FIRST)),
            cold_start_seconds=data.get("cold_start_seconds"),
            generation_seconds=data.get("generation_seconds"),
            error_type=str(data.get("error_type", "")),
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
    endpoint: str = "",
    local_only: Optional[bool] = None,
    timeout_seconds: Optional[float] = None,
    call_kind: str = CALL_FIRST,
    cold_start_seconds: Optional[float] = None,
    generation_seconds: Optional[float] = None,
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
        endpoint=endpoint,
        local_only=local_only,
        timeout_seconds=timeout_seconds,
        call_kind=call_kind,
        cold_start_seconds=cold_start_seconds,
        generation_seconds=generation_seconds,
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

    This wrapper is also why identity is fetched through ``describe`` rather
    than by attribute guessing: it holds the real provider as ``_inner`` and
    exposes none of the attributes a guesser would look for, so a wrapper
    silently erased the model name from the audit trail.
    """

    def __init__(self, inner: LLMProvider):
        self._inner = inner
        self.calls = 0
        #: Wall-clock seconds spent inside ``generate``. Excludes any model
        #: load, which ``generate`` does not perform.
        self.generation_seconds = 0.0
        #: Set by whoever warmed the provider, if anyone did.
        self.cold_start_seconds: Optional[float] = None
        #: Whether the last call was a repair rather than the first attempt.
        self.last_call_kind = CALL_FIRST

    def generate(self, request: LLMRequest):
        self.calls += 1
        self.last_call_kind = (
            CALL_REPAIR if self.calls > 1 else CALL_FIRST)
        started = time.monotonic()
        try:
            return self._inner.generate(request)
        finally:
            self.generation_seconds += time.monotonic() - started

    def health_check(self) -> bool:
        return self._inner.health_check()

    def list_models(self) -> list:
        return self._inner.list_models()

    def describe(self) -> Dict[str, Any]:
        """Delegate identity explicitly.

        A wrapper that did not do this would force every caller to guess
        through it, which is the bug being fixed.
        """
        describe = getattr(self._inner, "describe", None)
        if callable(describe):
            return describe()
        return {"model": UNKNOWN_MODEL}


def assess_one(
    job: Job,
    *,
    profile: CandidateProfile,
    provider_name: str,
    provider: LLMProvider,
    store: AssessmentStore,
    model: Optional[str] = None,
    cold_start_seconds: Optional[float] = None,
) -> AssessmentRecord:
    """Assess one job. Bounded to one generate plus at most one repair.

    Every failure path still writes a record. "We tried and could not" is a fact
    worth keeping; leaving no trace would make a gap that looks like a decision.
    """
    at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Identity first, through the contract. A provider that cannot name itself
    # is recorded as ``unknown`` - never ``None``, which reads as a field
    # nobody filled in rather than a fact someone wrote down.
    identity = identity_of(provider, provider_name)

    if not profile.usable:
        record = AssessmentRecord(
            job_id=str(job.job_id), at=at, status=STATUS_REFUSED,
            provider=identity.provider, model=identity.model,
            endpoint=identity.endpoint, local_only=identity.local_only,
            timeout_seconds=identity.timeout_seconds,
            error="no usable candidate profile, so no claim could be evidenced",
            error_type="ProfileUnavailable",
        )
        store.append(record)
        return record

    counter = _CountingProvider(provider)
    if cold_start_seconds is not None:
        counter.cold_start_seconds = round(float(cold_start_seconds), 3)
    request = build_request(job, profile.text, model=model)

    def validator(text: str) -> Dict[str, Any]:
        return _parse_response(text)

    try:
        payload = generate_structured(counter, request, validator)
    except (LLMError, ProviderConnectionError, ValueError,
            json.JSONDecodeError) as exc:
        record = AssessmentRecord(
            job_id=str(job.job_id), at=at, status=STATUS_ERROR,
            provider=identity.provider, model=identity.model,
            endpoint=identity.endpoint, local_only=identity.local_only,
            timeout_seconds=identity.timeout_seconds
            or getattr(request, "timeout_seconds", None),
            call_kind=counter.last_call_kind,
            cold_start_seconds=counter.cold_start_seconds,
            generation_seconds=round(counter.generation_seconds, 3),
            error=f"{type(exc).__name__}: {exc}",
            error_type=type(exc).__name__,
            provider_calls=counter.calls,
        )
        store.append(record)
        return record

    record = build_assessment(
        job, profile.text, payload,
        provider=identity.provider,
        model=identity.model,
        at=at,
        provider_calls=counter.calls,
        endpoint=identity.endpoint,
        local_only=identity.local_only,
        timeout_seconds=identity.timeout_seconds
        or getattr(request, "timeout_seconds", None),
        call_kind=counter.last_call_kind,
        cold_start_seconds=counter.cold_start_seconds,
        generation_seconds=round(counter.generation_seconds, 3),
    )
    store.append(record)
    return record


def _model_name(provider: LLMProvider, request: LLMRequest) -> str:
    """Deprecated. Kept only so old callers fail loudly rather than silently.

    Replaced by :func:`identity_of`. Attribute guessing is what wrote
    ``model: None`` into three real assessment records.
    """
    raise NotImplementedError(
        "_model_name guessed at provider attributes and produced records with "
        "model=None; use identity_of() and the describe() contract instead."
    )
    if request.model:  # pragma: no cover - unreachable, kept for the old diff
        return str(request.model)
    inner = getattr(provider, "_inner", provider)
    for attribute in ("model", "default_model"):
        value = getattr(inner, attribute, None)
        if isinstance(value, str) and value:
            return value
    return ""


#: Local-model execution defaults. Chosen to be explicit rather than inherited,
#: because the three 120s timeouts were the direct result of a limit nobody
#: chose on purpose.
LOCAL_TRIAL_TIMEOUT_SECONDS = 600.0
LOCAL_TRIAL_KEEP_ALIVE = "30m"


@dataclass(frozen=True)
class LocalTrialPlan:
    """What a local-model batch would do, decided before it does anything.

    Exists so the plan can be reviewed and printed without touching the model.
    It carries no job text and makes no request.
    """

    endpoint: str
    model: str
    timeout_seconds: float
    keep_alive: str
    job_ids: Tuple[str, ...]
    preload: bool

    @property
    def provider_calls_planned(self) -> int:
        # One generate per job; a repair only happens if the first answer is
        # unparseable, and the ceiling below bounds that.
        return len(self.job_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "keep_alive": self.keep_alive,
            "preload": self.preload,
            "job_count": len(self.job_ids),
            "provider_calls_planned": self.provider_calls_planned,
            "local_only": bool(is_local_endpoint(self.endpoint)
                               and is_local_model(self.model)),
            "writes_before_approval": 0,
        }


def build_local_trial(
    job_ids: Sequence[str],
    *,
    endpoint: str = LOCAL_OLLAMA_URL,
    model: str,
    timeout_seconds: float = LOCAL_TRIAL_TIMEOUT_SECONDS,
    keep_alive: str = LOCAL_TRIAL_KEEP_ALIVE,
    preload: bool = True,
) -> LocalTrialPlan:
    """Describe a local batch. No request, no write, no model load.

    Refuses a cloud model or a non-loopback endpoint here rather than at
    send time, so an unsafe plan cannot even be described.
    """
    if not is_local_endpoint(endpoint):
        raise AssessmentUnavailable(
            f"endpoint {endpoint!r} is not loopback; this mode is local-only")
    if not is_local_model(model):
        raise AssessmentUnavailable(
            f"model {model!r} is remote-inference; refused")
    ids = tuple(str(j) for j in job_ids)
    if not ids:
        raise AssessmentUnavailable("no jobs named")
    if len(set(ids)) != len(ids):
        raise AssessmentUnavailable("duplicate job ids")
    if len(ids) > TRIAL_JOB_CEILING:
        raise AssessmentUnavailable(
            f"{len(ids)} jobs given; a trial is at most {TRIAL_JOB_CEILING}")
    return LocalTrialPlan(
        endpoint=endpoint, model=model, timeout_seconds=timeout_seconds,
        keep_alive=keep_alive, job_ids=ids, preload=preload,
    )


def run_local_trial(
    plan: LocalTrialPlan,
    *,
    profile: CandidateProfile,
    store: AssessmentStore,
    jobs: Sequence[Job],
    provider_name: str = LOCAL_OLLAMA_PROVIDER,
    approved: bool,
) -> List[AssessmentRecord]:
    """Run a prepared local batch, once, with the model loaded first.

    ``approved`` must be true. Planning and running are separate on purpose:
    the plan is printed and reviewed before any of this executes, so a request
    is never implied by the mere existence of a plan.
    """
    if not approved:
        raise AssessmentUnavailable(
            "local trial not approved; no request was made")
    if not profile.usable:
        raise ProfileUnavailable("no usable candidate profile")
    if len(jobs) > TRIAL_JOB_CEILING:
        raise AssessmentUnavailable(
            f"{len(jobs)} jobs given; a trial is at most {TRIAL_JOB_CEILING}")

    from app.llm.ollama import OllamaProvider

    provider = OllamaProvider(
        base_url=plan.endpoint, model=plan.model,
        timeout_seconds=plan.timeout_seconds, keep_alive=plan.keep_alive,
    )
    # Locality re-asserted immediately before anything is sent.
    assert_local_provider(provider)

    # One load, timed on its own. Recorded per assessment so a slow model load
    # is never read as slow generation.
    cold_start = provider.preload() if plan.preload else None

    return [
        assess_one(
            job, profile=profile, provider_name=provider_name,
            provider=provider, store=store, model=plan.model,
            cold_start_seconds=cold_start,
        )
        for job in jobs[:TRIAL_JOB_CEILING]
    ]


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