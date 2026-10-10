"""Review hints: transparent signals that are not assessments.

The risk these tests exist to prevent is a hint quietly acquiring the authority
of an assessment. That failure is silent and slow: a term overlap starts
reading as "good match", someone sorts by it, and a queue nobody assessed gets
presented as though someone did.

So the tests here are mostly negative. They assert that hints *cannot* become a
tier or a score, cannot move a job, cannot change eligibility, and cannot invent
a quote that is not in the source text. One positive test covers determinism,
because an unstable hint is noise wearing a useful hat.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from app.jobs.hints import (
    LIMITATIONS,
    STOPWORDS,
    ReviewHint,
    build_review_hint,
    build_review_hints,
)
from app.jobs.match import Confidence, MatchTier
from app.jobs.models import Job
from app.jobs.runner import SourceSpec, run_sources
from app.jobs.status import ReviewStatus, StatusLog
from app.jobs.store import JobStore
from app.reporting.review import build_actionable, build_report, render_report_html

PROFILE = (
    "Backend engineer with Go, Python, PostgreSQL and Kubernetes. "
    "Five years experience. Degree in Computer Science. "
    "Senior level work on payments systems."
)


def _job(**kwargs) -> Job:
    base = dict(
        job_id="j1",
        title="Senior Backend Engineer (Go/Kubernetes)",
        company="Acme",
        url="https://example.test/jobs/j1",
        location="Nairobi, Kenya",
        description=(
            "You will need Go and PostgreSQL. Kubernetes required. "
            "Experience with payments is preferred. Five years experience."
        ),
        skills=["go", "postgresql", "kubernetes"],
    )
    base.update(kwargs)
    return Job(**base)


class SeparationTests(unittest.TestCase):
    """A hint is not a match, and the type makes that structural."""

    def test_a_hint_carries_no_tier(self):
        import dataclasses

        fields = {f.name for f in dataclasses.fields(ReviewHint)}
        for forbidden in ("tier", "match_tier", "score", "match_score",
                          "confidence", "verdict", "rating"):
            self.assertNotIn(forbidden, fields,
                             f"ReviewHint must not carry {forbidden}")

    def test_a_hints_dict_contains_no_match_keys(self):
        hint = build_review_hint(_job(), PROFILE)
        for forbidden in ("tier", "score", "confidence", "match"):
            self.assertNotIn(forbidden, hint.to_dict())

    def test_a_hint_is_not_a_matchresult(self):
        from app.jobs.match import MatchResult

        self.assertFalse(isinstance(build_review_hint(_job(), PROFILE),
                                    MatchResult))

    def test_no_import_of_the_match_module(self):
        """Hints must not reach for a tier, even indirectly."""
        import ast

        tree = ast.parse(
            Path("app/jobs/hints.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
        self.assertNotIn("app.jobs.match", imported)
        self.assertNotIn("app.jobs.assessment", imported)

    def test_it_does_not_import_a_provider_or_transport(self):
        import ast

        tree = ast.parse(
            Path("app/jobs/hints.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
        for forbidden in ("app.llm", "socket", "urllib.request",
                          "http", "subprocess", "requests"):
            self.assertNotIn(forbidden, imported)

    def test_the_summary_says_it_is_not_an_assessment(self):
        hint = build_review_hint(_job(), PROFILE)
        self.assertNotIn("strong_match", hint.summary())
        self.assertNotIn("credible_match", hint.summary())
        self.assertNotIn("unsuitable", hint.summary())
        self.assertIsNone(getattr(hint, "score", None))


class SignalTests(unittest.TestCase):
    def test_shared_terms_are_words_in_both_documents(self):
        hint = build_review_hint(_job(), PROFILE)
        for term in hint.shared_terms:
            self.assertIn(term, _job().description.casefold()
                          + _job().title.casefold())
            self.assertIn(term, PROFILE.casefold())

    def test_stopwords_are_excluded(self):
        hint = build_review_hint(_job(), PROFILE)
        self.assertNotIn("the", hint.shared_terms)
        self.assertNotIn("and", hint.shared_terms)
        for term in hint.shared_terms:
            self.assertNotIn(term, STOPWORDS)

    def test_title_overlap_is_reported(self):
        hint = build_review_hint(_job(), PROFILE)
        self.assertIn("backend", hint.title_terms)
        self.assertIn("senior", hint.title_terms)

    def test_an_unrelated_job_reports_no_shared_terms(self):
        hint = build_review_hint(
            _job(title="Waiter - Hospitality",
                 description="Serve guests in a busy restaurant.",
                 skills=[]),
            PROFILE)
        self.assertEqual(hint.shared_terms, ())
        self.assertEqual(hint.title_terms, ())
        self.assertIn("nothing to compare", hint.summary())

    def test_seniority_terms_are_quoted_not_interpreted(self):
        hint = build_review_hint(_job(), PROFILE)
        self.assertIn("senior", hint.seniority_terms)

    def test_required_and_preferred_are_detected(self):
        hint = build_review_hint(_job(), PROFILE)
        self.assertIs(hint.mentions_required, True)
        self.assertIs(hint.mentions_preferred, True)

    def test_absent_wording_is_absent_not_negative(self):
        hint = build_review_hint(
            _job(description="Some work. No particular requirements stated."),
            PROFILE)
        self.assertIs(hint.mentions_required, False)
        self.assertIs(hint.mentions_preferred, False)

    def test_no_profile_means_no_claims(self):
        hint = build_review_hint(_job(), None)
        self.assertEqual(hint.shared_terms, ())
        self.assertEqual(hint.title_terms, ())

    def test_every_hint_carries_its_limitations(self):
        hint = build_review_hint(_job(), PROFILE)
        self.assertEqual(hint.limitations, LIMITATIONS)
        self.assertIn("not shared skill", " ".join(hint.limitations))

    def test_evidence_is_never_fabricated(self):
        """Terms come from the two documents; nothing is invented."""
        job = _job()
        hint = build_review_hint(job, PROFILE)
        source = (f"{job.title} {job.company} {job.location} "
                  f"{job.description} {' '.join(job.skills)} "
                  f"{PROFILE}").casefold()
        for term in hint.shared_terms + hint.title_terms:
            self.assertIn(term, source)
        for term in hint.seniority_terms:
            self.assertIn(term, source)


class DeterminismTests(unittest.TestCase):
    def test_two_builds_are_identical(self):
        jobs = [_job(job_id=f"j{i}") for i in range(5)]
        first = build_review_hints(jobs, PROFILE)
        second = build_review_hints(jobs, PROFILE)
        self.assertEqual({k: v.to_dict() for k, v in first.items()},
                         {k: v.to_dict() for k, v in second.items()})

    def test_terms_are_sorted(self):
        hint = build_review_hint(_job(), PROFILE)
        self.assertEqual(list(hint.shared_terms), sorted(hint.shared_terms))
        self.assertEqual(list(hint.title_terms), sorted(hint.title_terms))

    def test_input_order_does_not_change_the_result(self):
        jobs = [_job(job_id="a"), _job(job_id="b"), _job(job_id="c")]
        forward = build_review_hints(jobs, PROFILE)
        backward = build_review_hints(list(reversed(jobs)), PROFILE)
        self.assertEqual({k: v.to_dict() for k, v in forward.items()},
                         {k: v.to_dict() for k, v in backward.items()})


class NonInterferenceTests(unittest.TestCase):
    """A hint changes nothing about the job."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data = Path(self._tmp.name) / "data"
        self.store = JobStore(self.data)

    def _ingest(self, records):
        run_sources([SourceSpec(name="weworkremotely",
                                fetch=lambda: list(records))],
                    store=self.store)

    def _snapshot(self):
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(self.data.rglob("*")) if p.is_file()}

    def _record(self, name, **kwargs):
        base = {"url": f"https://weworkremotely/jobs/{name}",
                "title": "Backend Engineer", "company": "Acme",
                "location": "Nairobi, Kenya", "description": "Go and Postgres."}
        base.update(kwargs)
        return base

    def test_computing_hints_writes_nothing(self):
        self._ingest([self._record("a1"), self._record("a2")])
        before = self._snapshot()
        jobs = [Job(**{k: v for k, v in r["job"].items()
                       if k in Job.__dataclass_fields__})
                for r in self.store.load_jobs()]
        build_review_hints(jobs, PROFILE)
        self.assertEqual(before, self._snapshot())

    def test_hints_do_not_change_queue_membership(self):
        self._ingest([self._record("a1"), self._record("a2"),
                      self._record("a3")])
        before = build_actionable([v for v, _ in
                                   build_report(self.store).queue])
        ids = {v.job_id for v in before}
        jobs = [Job(**{k: v for k, v in r["job"].items()
                       if k in Job.__dataclass_fields__})
                for r in self.store.load_jobs()]
        build_review_hints(jobs, PROFILE)
        after = build_actionable([v for v, _ in
                                  build_report(self.store).queue])
        self.assertEqual({v.job_id for v in after}, ids)

    def test_hints_do_not_change_status(self):
        self._ingest([self._record("a1")])
        job_id = self.store.load_jobs()[0]["job_id"]
        StatusLog(self.store).record(job_id, ReviewStatus.DISMISSED)
        build_review_hints([_job(job_id=job_id)], PROFILE)
        self.assertEqual(StatusLog(self.store).current(job_id).value, "dismissed")

    def test_hints_do_not_change_eligibility(self):
        self._ingest([self._record("a1", location="Berlin, Germany")])
        before = build_report(self.store).eligible_unreviewed
        build_review_hints([_job()], PROFILE)
        after = build_report(self.store).eligible_unreviewed
        self.assertEqual(len(before), len(after))
        self.assertEqual(len(after), 0, "a German posting is not Kenya-eligible")

    def test_match_state_stays_not_yet_evaluated(self):
        """No provider result means no assessment, hints notwithstanding."""
        self._ingest([self._record("a1")])
        report = build_report(self.store)
        jobs = [Job(**{k: v for k, v in r["job"].items()
                       if k in Job.__dataclass_fields__})
                for r in self.store.load_jobs()]
        hints = build_review_hints(jobs, PROFILE)
        self.assertTrue(hints)
        for view, _ in report.queue:
            if not view.match_present:
                self.assertEqual(view.match_tier, MatchTier.NOT_YET_EVALUATED.value)
                self.assertIsNone(view.match_score)
                self.assertFalse(view.match_present)

    def test_no_match_file_is_created(self):
        self._ingest([self._record("a1")])
        build_review_hints([_job()], PROFILE)
        self.assertFalse((self.data / "matches.jsonl").exists())

    def test_no_provider_is_registered(self):
        from app.jobs.assessment import registered_providers

        build_review_hints([_job()], PROFILE)
        self.assertEqual(registered_providers(), ())


class RenderingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = JobStore(Path(self._tmp.name) / "data")
        run_sources([SourceSpec(name="weworkremotely", fetch=lambda: [{
            "url": "https://weworkremotely/jobs/a1",
            "title": "Backend Engineer", "company": "Acme",
            "location": "Nairobi, Kenya",
            "description": "Go and PostgreSQL required."}])], store=self.store)

    def test_the_section_states_it_is_not_an_assessment(self):
        report = build_report(self.store)
        html = render_report_html(report, hints={
            "a": build_review_hint(_job(), PROFILE)})
        self.assertIn("Review hints", html)
        self.assertIn("not assessments", html)
        self.assertIn("No tier, no score", html)

    def test_limitations_are_rendered(self):
        report = build_report(self.store)
        html = render_report_html(report, hints={})
        for limitation in LIMITATIONS:
            self.assertIn(limitation.split(".")[0], html)

    def test_every_job_is_labelled_unassessed(self):
        report = build_report(self.store)
        html = render_report_html(report, hints={})
        self.assertIn("unassessed by provider", html)

    def test_a_hint_never_renders_a_tier(self):
        report = build_report(self.store)
        html = render_report_html(report, hints={
            "j1": build_review_hint(_job(), PROFILE)})
        self.assertNotIn("credible_match", html)
        self.assertNotIn("strong_match", html)
        self.assertNotIn(">unsuitable<", html)

    def test_hostile_text_is_escaped(self):
        """Both the row key and the rendered signals must be escaped."""
        hostile_id = "<script>alert(1)</script>"
        hint = ReviewHint(
            job_id=hostile_id,
            shared_terms=("<img src=x onerror=alert(1)>",))
        html = render_report_html(build_report(self.store),
                                  hints={hostile_id: hint})
        self.assertNotIn("<script>", html)
        self.assertNotIn("<img src=x", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", html)

    def test_the_report_is_a_pure_function_of_its_input(self):
        report = build_report(self.store)
        hints = {"j1": build_review_hint(_job(), PROFILE)}
        self.assertEqual(render_report_html(report, hints=hints),
                         render_report_html(report, hints=hints))


if __name__ == "__main__":
    unittest.main()