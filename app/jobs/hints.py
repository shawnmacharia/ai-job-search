"""Review hints: transparent signals, deliberately not a match assessment.

What this is
------------
A recruiter's eye, approximated with arithmetic you can check. For each job it
reports the terms the posting and your profile have in common, whether the
title words appear in the profile, whether the posting says "required" or
"preferred", and any explicit seniority term. You can read the whole reasoning
in a sentence.

What this is not
----------------
**It is not an assessment.** There is no tier, no score, and no judgement about
whether you are a good fit. Those come from :mod:`app.jobs.match`, which
requires verified evidence and, in practice, a provider result. A hint is not
weak evidence - it is not evidence at all. It is arithmetic.

That distinction is enforced, not merely documented. :class:`ReviewHint` has no
tier field, no score field, and no ``MatchResult`` anywhere in its shape, so a
hint cannot be promoted into a match by passing it somewhere convenient. Adding
a score to this type is the one change that would break the guarantee.

Why bother
----------
The queue is 101 jobs, all identical on every assessment-shaped column. That is
an honest reflection of reality - nobody has assessed them - but it gives a
reader nothing to sort by. Hints make the queue browsable without pretending
the browsing is an assessment.

The limits are real and worth stating plainly
----------------------------------------------
* Term overlap measures shared words, not shared skill. "Kubernetes" appearing
  in both documents is evidence the word is there, not evidence you can do it.
* Title similarity rewards keyword-rich titles over better ones.
* "Required" and "preferred" are only found when the posting writes those words.
  Most postings do not, so this signal is usually absent rather than neutral -
  an absent signal is not a negative one.
* Seniority terms come from a fixed list. "Lead", "Principal" and "Head" are
  detected; a seniority conveyed any other way is not.
* Nothing here reads your CV's meaning, only its words.

None of these can be wrong in the sense of inventing a fact. They can absolutely
be wrong in the sense of pointing you at the wrong job first, which is why the
output says what it is on every line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.jobs.models import Job

#: Words too common to say anything. "Experience" appears in almost every
#: posting and in every CV; counting it as shared vocabulary is noise.
STOPWORDS = frozenset("""
and the with for you your from that this will have has been were are was not
but all can could should would may might must more most other such than then
they them their there here when where which who whom what how why into over
under about after before during while both each few some any own same
work working works role roles job jobs position positions team teams
experience experienced year years required required strong excellent good
ability able knowledge skills skill candidate candidates company companies
including include includes etc via using use used new well plus
""".split())

#: Explicit seniority markers. Reported as *text found*, never interpreted:
#: "lead" is reported as the word "lead", not as a verdict about seniority.
SENIORITY_TERMS = (
    "intern", "internship", "graduate", "entry", "junior", "associate",
    "mid", "mid-level", "senior", "lead", "principal", "staff", "head",
    "chief", "director", "executive", "vp", "vice president", "manager",
)

#: Terms indicating something is optional rather than required.
PREFERRED_MARKERS = (
    "preferred", "nice to have", "nice-to-have", "desirable", "advantageous",
    "a plus", "bonus", "optional",
)

#: Terms indicating something is required.
REQUIRED_MARKERS = (
    "required", "must have", "must-have", "essential", "mandatory",
    "you will need", "minimum",
)

_WORD = re.compile(r"[a-z0-9+#.]+")


def _tokens(text: str) -> List[str]:
    """Lowercase word tokens, stopwords removed.

    Kept simple and total: a hint is a reading aid, and a tokeniser that can
    fail is a hint that can be wrong in a new way.
    """
    if not text:
        return []
    return [
        w for w in _WORD.findall(text.casefold())
        if len(w) > 2 and w not in STOPWORDS
    ]


@dataclass(frozen=True)
class ReviewHint:
    """Transparent signals about one job.

    Deliberately has no ``tier``, no ``score`` and no match state. A hint that
    could be read as an assessment would eventually be displayed as one, and
    "the queue said stretch" is a claim this type is built to be unable to make.
    """

    job_id: str
    #: Terms in both the posting and the profile, sorted for determinism.
    shared_terms: Tuple[str, ...] = ()
    #: Posting title words that also appear in the profile.
    title_terms: Tuple[str, ...] = ()
    #: Seniority words found in the posting text, quoted rather than interpreted.
    seniority_terms: Tuple[str, ...] = ()
    #: True only when the posting literally writes "preferred"/"nice to have".
    mentions_preferred: bool = False
    #: True only when the posting literally writes "required"/"essential".
    mentions_required: bool = False
    #: Why these signals are weaker than an assessment. Shown, not buried.
    limitations: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "shared_terms": list(self.shared_terms),
            "title_terms": list(self.title_terms),
            "seniority_terms": list(self.seniority_terms),
            "mentions_preferred": self.mentions_preferred,
            "mentions_required": self.mentions_required,
            "limitations": list(self.limitations),
        }

    def summary(self) -> str:
        """One line, honest about what it is."""
        parts: List[str] = []
        if self.shared_terms:
            parts.append(f"{len(self.shared_terms)} shared term(s): "
                         f"{', '.join(self.shared_terms[:6])}")
        else:
            parts.append("no shared terms")
        if self.title_terms:
            parts.append(f"title terms: {', '.join(self.title_terms)}")
        if self.seniority_terms:
            parts.append(f"posting says: {', '.join(self.seniority_terms)}")
        if self.mentions_preferred and not self.mentions_required:
            parts.append("posting distinguishes preferred, not required")
        if not self.shared_terms and not self.title_terms:
            parts.append("nothing to compare against the profile")
        return "; ".join(parts)


#: Shown on every hint. A limitation that is not visible is not a limitation.
LIMITATIONS: Tuple[str, ...] = (
    "Shared words are not shared skill.",
    "This is arithmetic over text, not a judgement about fit.",
    "It is not a recruiter assessment and carries no tier or score.",
    "Absent signal means the posting did not say, not that the answer is no.",
)


def build_review_hint(job: Job, profile: Optional[str]) -> ReviewHint:
    """Compute the transparent signals for one job.

    Every field is either a word that literally appears in both documents, or a
    flag for a word the posting literally contains. Nothing is inferred.
    """
    job_text = " ".join(filter(None, [
        job.title, job.company, job.location, job.description,
        " ".join(job.skills or []),
    ]))
    job_tokens = set(_tokens(job_text))
    profile_tokens = set(_tokens(profile or ""))

    shared = sorted(t for t in job_tokens & profile_tokens if t not in STOPWORDS)
    # Written out rather than as a set operation with a subtraction, because
    # operator precedence there is easy to get wrong and hard to notice.
    title_terms = sorted(
        {t for t in _tokens(job.title)
         if t in profile_tokens and t not in STOPWORDS}
    )

    lowered = job_text.casefold()
    seniority = tuple(
        term for term in SENIORITY_TERMS if re.search(rf"\b{re.escape(term)}\b", lowered)
    )
    preferred = any(m in lowered for m in PREFERRED_MARKERS)
    required = any(m in lowered for m in REQUIRED_MARKERS)

    return ReviewHint(
        job_id=str(job.job_id),
        shared_terms=tuple(shared),
        title_terms=tuple(title_terms),
        seniority_terms=seniority,
        mentions_preferred=preferred,
        mentions_required=required,
        limitations=LIMITATIONS,
    )


def build_review_hints(
    jobs: Sequence[Job], profile: Optional[str]
) -> Dict[str, ReviewHint]:
    """Hints for many jobs, keyed by id.

    Ordering inside each hint is sorted, so two runs over unchanged input
    produce identical output - otherwise a diff could not distinguish a real
    change from reshuffling.
    """
    return {str(job.job_id): build_review_hint(job, profile) for job in jobs}