"""Evidence-based match assessment: what a recruiter would actually ask.

Why this exists
---------------
``app/orchestrator/rank.py`` used to call ``rank_job`` with
``RankingEvidence(0, 0, 0, 0, eligibility="unknown")`` - four fabricated zeroes
presented as an assessment. The resulting table looked like a ranking and
contained no information whatsoever. This module replaces that with an honest
account: either there is evidence, and the tier follows from it, or there is
not, and the result says so.

The governing policy
--------------------
**Scores reorder. They never exclude.**

A match result may rank, prioritise, label, flag, and explain. It may not hide,
discard, drop, or mark a job unavailable. Low scores, missing evidence,
uncertain fit, and absent keywords all yield a lower rank, a flag, or an
"uncertain" classification - never removal.

The single exception is a **confirmed hard eligibility veto** from
:mod:`app.jobs.eligibility` - a role demonstrably anchored to a region that
excludes the candidate - and even then the reason and its evidence are carried
through to output, because an exclusion nobody can audit is indistinguishable
from a bug.

**Absence of evidence is not evidence of unsuitability.** ``unknown`` and
``not_yet_evaluated`` are first-class results. An empty posting with no
requirements stated is ``not_yet_evaluated``, never ``unsuitable``: calling a
role unsuitable because we could not read it would be inventing a judgement.

**Keyword overlap informs, it never decides.** Overlap may appear in
``evidence`` as a weak signal and may never alone produce a tier or an
exclusion. A term matching twice is not a qualification.

**Nothing is fabricated.** Every material claim cites evidence drawn from the
job record or the candidate profile. A claim with no citation cannot be made,
and scores with no evidence are refused rather than defaulted to zero.

Tiers
-----
``strong_match``    evidence supports the role and essential requirements appear met
``credible_match``  most requirements appear met, with gaps or uncertainty
``stretch``         meaningful gaps, but possibly still worth considering
``unsuitable``      an identified essential requirement or eligibility constraint
                    makes the role inappropriate - **requires** a cited veto
``not_yet_evaluated`` insufficient information; no tier asserted
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

from app.jobs.eligibility import EligibilityVerdict
from app.jobs.models import Job


class MatchTier(str, Enum):
    """The outcome of an assessment. Exactly these five values."""

    STRONG_MATCH = "strong_match"
    CREDIBLE_MATCH = "credible_match"
    STRETCH = "stretch"
    UNSUITABLE = "unsuitable"
    NOT_YET_EVALUATED = "not_yet_evaluated"


class Confidence(str, Enum):
    """How much weight the assessment can bear."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class EvidenceItem:
    """One claim, and where it came from.

    ``quote`` must be text that actually appears in the job record or the
    candidate profile. That is the whole point: a claim whose quotation cannot
    be found in a source is an assertion, not evidence.
    """

    claim: str
    source: str  # "job" | "profile"
    quote: str

    def to_dict(self) -> Dict[str, str]:
        return {"claim": self.claim, "source": self.source, "quote": self.quote}


class MatchError(ValueError):
    """A match input or result is not valid as written."""


class NoEvidenceError(MatchError):
    """A tier was requested that the evidence cannot support.

    Raised by strict callers. :func:`assess_match` itself reports the honest
    ``not_yet_evaluated`` outcome instead of raising, so a bulk run continues.
    """


def _clamp(value: int) -> int:
    return max(0, min(100, int(value)))


@dataclass(frozen=True)
class MatchResult:
    """One assessed job.

    ``score`` is ``None`` whenever no evidence was supplied. A score with no
    evidence is a fabricated number, so this module refuses to produce one -
    and the field being optional rather than defaulting to ``0`` makes that
    visible at every call site.
    """

    job_id: str
    tier: MatchTier
    score: Optional[float]
    confidence: Confidence
    evidence: List[EvidenceItem] = field(default_factory=list)
    missing_requirements: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    vetoes: List[str] = field(default_factory=list)
    eligibility: str = "unknown"
    insufficient_reason: str = ""
    keyword_overlap: List[str] = field(default_factory=list)

    @property
    def excluded(self) -> bool:
        """Is this the one permitted automatic exclusion?

        True only for a confirmed hard eligibility veto. Never true because of
        a low score, a weak tier, or missing evidence - that is the policy this
        module exists to enforce, so it is expressed as a property rather than
        a convention someone has to remember.
        """
        return bool(self.vetoes)

    @property
    def sorts_before_others(self) -> bool:
        """Unranked results sort last rather than being dropped."""
        return self.score is not None

    @property
    def sort_key(self) -> tuple:
        """Deterministic ordering: ranked results descending, unranked last."""
        return (0, -self.score, self.job_id) if self.score is not None else (1, 0.0, self.job_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "tier": self.tier.value,
            "score": self.score,
            "confidence": self.confidence.value,
            "evidence": [item.to_dict() for item in self.evidence],
            "missing_requirements": list(self.missing_requirements),
            "concerns": list(self.concerns),
            "vetoes": list(self.vetoes),
            "eligibility": self.eligibility,
            "insufficient_reason": self.insufficient_reason,
            "keyword_overlap": list(self.keyword_overlap),
            "excluded": self.excluded,
        }

    def explanation(self) -> str:
        """A human-readable summary, citing evidence for every material claim."""
        lines = [f"{self.job_id}: {self.tier.value}"]
        if self.score is None:
            lines.append(f"  score: not calculated ({self.insufficient_reason})")
        else:
            lines.append(f"  score: {self.score} (confidence: {self.confidence.value})")
        for item in self.evidence:
            lines.append(f"  + {item.claim} [job: \"{item.quote}\"]")
        for gap in self.missing_requirements:
            lines.append(f"  - missing: {gap}")
        for concern in self.concerns:
            lines.append(f"  ! {concern}")
        for veto in self.vetoes:
            lines.append(f"  x excluded: {veto}")
        if self.keyword_overlap:
            lines.append(f"  ~ keyword overlap (informs only): {', '.join(self.keyword_overlap)}")
        return "\n".join(lines)


#: Scores below this band cannot be asserted as a real fit. A job may sit here
#: and be reported as a ``stretch`` - that is a ranking outcome, not removal.
BELOW_FIT_FLOOR = 55.0


def _sources_text(job: Job, profile: Optional[str]) -> Dict[str, str]:
    """The citable text of each source, used to verify every quotation."""
    job_text = " ".join(
        str(value or "")
        for value in (
            job.title,
            job.company,
            job.location,
            job.description,
            job.deadline,
            " ".join(str(skill) for skill in (job.skills or [])),
        )
    )
    return {"job": job_text.casefold(), "profile": (profile or "").casefold()}


def verify_evidence(
    evidence: Sequence[EvidenceItem],
    job: Job,
    profile: Optional[str] = None,
) -> List[EvidenceItem]:
    """Drop any item whose quote does not appear in the source it cites.

    This is the mechanism that makes "every material claim must cite evidence"
    enforceable rather than aspirational. Model-produced evidence is checked
    against the actual record: a quote that cannot be found is discarded, and if
    that empties the set the result becomes ``not_yet_evaluated`` rather than
    inheriting a tier nobody can support.
    """
    sources = _sources_text(job, profile)
    verified: List[EvidenceItem] = []
    for item in evidence:
        if item.source not in sources:
            continue
        quote = (item.quote or "").strip().casefold()
        # An empty quote cites nothing, so it is not evidence either.
        if quote and quote in sources[item.source]:
            verified.append(item)
    return verified


def keyword_overlap(job: Job, profile: Optional[str]) -> List[str]:
    """Terms appearing in both the posting and the profile.

    Reported for transparency and usable as *weak supporting* evidence. Never
    sufficient on its own - see :func:`assess_match`.
    """
    if not profile:
        return []
    haystack = _sources_text(job, profile)["job"]
    terms = {
        word.strip(".,;:()[]'\"")
        for word in profile.casefold().split()
        if len(word.strip(".,;:()[]'\"")) > 3
    }
    return sorted(term for term in terms if term in haystack)


def _tier_for(score: float) -> MatchTier:
    if score >= 75.0:
        return MatchTier.STRONG_MATCH
    if score >= 60.0:
        return MatchTier.CREDIBLE_MATCH
    return MatchTier.STRETCH


def assess_match(
    job: Job,
    scores: Optional[Mapping[str, int]] = None,
    *,
    evidence: Sequence[EvidenceItem] = (),
    gaps: Sequence[str] = (),
    concerns: Sequence[str] = (),
    profile: Optional[str] = None,
    eligibility: Optional[EligibilityVerdict] = None,
) -> MatchResult:
    """Assess one job. ``scores`` of ``None`` means "no evidence available".

    There is no default for ``scores``. The caller must either supply measured
    scores or pass ``None`` to say it has none, which yields
    ``not_yet_evaluated``. Omitting the argument is an error rather than a
    silent zero - that is the specific defect this replaces.
    """
    if eligibility is not None and not isinstance(eligibility, EligibilityVerdict):
        raise MatchError(
            f"eligibility must be an EligibilityVerdict, got {type(eligibility).__name__}"
        )
    for item in evidence:
        if not isinstance(item, EvidenceItem):
            raise MatchError(f"evidence must be EvidenceItem, got {type(item).__name__}")

    verified = verify_evidence(evidence, job, profile)
    dropped = len(evidence) - len(verified)
    overlap = keyword_overlap(job, profile)

    # The one permitted automatic exclusion: a confirmed hard eligibility
    # veto. Its reasons and evidence travel with it so the exclusion is
    # auditable rather than a silent disappearance.
    vetoes: List[str] = []
    if eligibility is not None and eligibility.verdict == "not_eligible":
        vetoes = list(eligibility.reasons) + list(eligibility.evidence_quotes)
        return MatchResult(
            job_id=job.job_id,
            tier=MatchTier.UNSUITABLE,
            score=None,
            confidence=Confidence.HIGH,
            evidence=verified,
            missing_requirements=list(gaps),
            concerns=list(concerns),
            vetoes=vetoes,
            eligibility=eligibility.verdict,
            insufficient_reason="",
            keyword_overlap=overlap,
        )

    if scores is None:
        return MatchResult(
            job_id=job.job_id,
            tier=MatchTier.NOT_YET_EVALUATED,
            score=None,
            confidence=Confidence.INSUFFICIENT,
            evidence=verified,
            missing_requirements=list(gaps),
            concerns=list(concerns),
            eligibility=eligibility.verdict if eligibility else "unknown",
            insufficient_reason=(
                "no candidate profile was available, so no requirement could be "
                "checked and no score was calculated"
            ),
            keyword_overlap=overlap,
        )

    # A tier needs verified evidence or an explicitly declared gap. Keyword
    # overlap is deliberately NOT accepted as a substitute: an earlier draft of
    # this function allowed `overlap` to stand in for evidence, which is exactly
    # the "a keyword count may decide the outcome" behaviour this module exists
    # to forbid.
    #
    # This reports rather than raises. A bulk run should not abort because one
    # posting was too thin to assess - the honest outcome for that posting is
    # "not yet evaluated", and returning it is more useful to a caller than an
    # exception it has to catch and translate anyway.
    if not verified and not gaps:
        return MatchResult(
            job_id=job.job_id,
            tier=MatchTier.NOT_YET_EVALUATED,
            score=None,
            confidence=Confidence.INSUFFICIENT,
            evidence=[],
            missing_requirements=[],
            concerns=list(concerns),
            eligibility=eligibility.verdict if eligibility else "unknown",
            insufficient_reason=(
                "scores were supplied but no evidence could be verified against "
                "the job record or profile; keyword overlap alone cannot "
                "justify a tier, so no score was calculated"
            ),
            keyword_overlap=overlap,
        )

    weighted = sum(_clamp(scores.get(name, 0)) * weight for name, weight in _WEIGHTS.items())
    score = round(weighted, 2)
    confidence = _confidence(len(verified), dropped, len(gaps))

    tier = _tier_for(score)
    if confidence in (Confidence.LOW, Confidence.INSUFFICIENT):
        # Too little verified evidence to assert a tier. The score is retained
        # for ordering only, and the result reports itself as uncertain rather
        # than dressing a guess up as a conclusion.
        return MatchResult(
            job_id=job.job_id,
            tier=MatchTier.NOT_YET_EVALUATED,
            score=score,
            confidence=confidence,
            evidence=verified,
            missing_requirements=list(gaps),
            concerns=list(concerns),
            eligibility=eligibility.verdict if eligibility else "unknown",
            insufficient_reason=(
                "evidence supplied was too thin to support a tier"
                if confidence is Confidence.LOW
                else "no verifiable evidence was supplied"
            ),
            keyword_overlap=overlap,
        )

    return MatchResult(
        job_id=job.job_id,
        tier=tier,
        score=score,
        confidence=confidence,
        evidence=verified,
        missing_requirements=list(gaps),
        concerns=list(concerns),
        eligibility=eligibility.verdict if eligibility else "unknown",
        keyword_overlap=overlap,
    )


def _confidence(verified: int, dropped: int, gaps: int) -> Confidence:
    if verified == 0:
        return Confidence.INSUFFICIENT
    if dropped > verified or gaps > verified:
        return Confidence.LOW
    if verified < 3:
        return Confidence.MEDIUM
    return Confidence.HIGH


#: Weights mirror app.ranking.scoring.WEIGHTS and are summed over whatever
#: dimensions the caller supplied, so a caller assessing only some dimensions
#: is not silently scored as though the rest were zero.
_WEIGHTS: Dict[str, float] = {
    "technical": 0.30,
    "experience": 0.25,
    "behavioral": 0.15,
    "career": 0.30,
}


def rank_matches(results: Sequence[MatchResult]) -> List[MatchResult]:
    """Order results for display. Reorders only - nothing is removed.

    Unranked (``not_yet_evaluated`` with no score) results sort last rather than
    being filtered out, so an unreadable posting still appears in the list.
    """
    return sorted(results, key=lambda result: result.sort_key)


def summarise(results: Sequence[MatchResult]) -> Dict[str, Any]:
    """Counts per tier, plus the counts a reader needs to trust them."""
    by_tier: Dict[str, int] = {tier.value: 0 for tier in MatchTier}
    unranked = 0
    for result in results:
        by_tier[result.tier.value] += 1
        if not result.sorts_before_others:
            unranked += 1
    return {
        "total": len(results),
        "by_tier": by_tier,
        "unranked": unranked,
        "excluded": sum(1 for result in results if result.excluded),
        "note": (
            "Ranking reorders and labels. It never removes a job; only a "
            "confirmed hard eligibility veto excludes, and it carries its "
            "reason."
        ),
    }