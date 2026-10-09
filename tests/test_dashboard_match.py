"""Tests for match explanations in the read-only dashboard.

Fully offline: a temporary store plus synthetic ``MatchResult`` objects. No
network, no provider, no model.

The behaviour that matters here is not layout, it is *honesty*. A dashboard that
shows a match result has to keep three states apart - never evaluated, evaluated
but uncertain, and evaluated with evidence - because collapsing them turns
"we did not look" into "we looked and found it wanting".
"""

import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from app.jobs.ingest import ingest
from app.jobs.match import (
    Confidence,
    EvidenceItem,
    MatchResult,
    MatchTier,
    assess_match,
)
from app.jobs.models import Job, RemoteStatus
from app.jobs.store import JobStore
from app.reporting.jobs import (
    _filtered,
    _match_cell,
    _match_panel,
    _sorted,
    build_view,
    render_dashboard_file,
    render_dashboard_html,
    render_row,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hiring_cafe"
NOW = datetime(2026, 10, 9, 12, 0, 0, tzinfo=timezone.utc)

PROFILE = (
    "Data engineer with 8 years building streaming ingestion pipelines in "
    "Python and Kafka, strong dbt and Snowflake experience, and ELT "
    "ownership on a large analytics platform."
)


def load(name):
    import json

    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def job_for(name="basic.json"):
    record = load(name)
    return Job(
        job_id=record.get("url", "j-1").rsplit("/", 1)[-1],
        title=record["title"],
        company=record["company"],
        url=record["url"],
        location=record.get("location") or "",
        description=record.get("description_snippet") or "",
    )


def store_with(*names):
    directory = tempfile.TemporaryDirectory()
    store = JobStore(Path(directory.name) / "data")
    ingest([load(n) for n in names], source="weworkremotely", store=store,
           observed_at=NOW)
    store._test_directory = directory  # keep alive
    return store


class MatchCellTests(unittest.TestCase):
    def view(self, match=None, **overrides):
        """A JobView for one job, optionally carrying a match result.

        ``match`` is passed to ``build_view`` rather than splatted into view
        fields, because that is the only supported entry point - the view
        derives its match fields from a real result, which is the path under
        test.
        """
        record = {
            "job_id": "j-1",
            "job": {
                "job_id": "j-1", "title": "Data Engineer", "company": "Acme",
                "url": "https://x.test/1", "location": "Remote",
            },
        }
        view = build_view(record, match=match)
        return replace(view, **overrides) if overrides else view

    def test_full_evidence_renders_tier_score_evidence_and_concerns(self):
        result = assess_match(
            job_for(), {"technical": 85, "experience": 80,
                        "behavioral": 75, "career": 80},
            profile=PROFILE,
            evidence=[
                EvidenceItem("streaming ingestion", "profile", "streaming ingestion"),
                EvidenceItem("ELT pipelines", "job", "ELT pipelines"),
                EvidenceItem("dbt", "job", "dbt"),
            ],
            concerns=["posting does not state a salary range"],
        )
        markup = _match_cell(self.view(match=result))
        self.assertIn("strong_match", markup)
        self.assertIn("score", markup)
        self.assertIn("streaming ingestion", markup)
        self.assertIn("ELT pipelines", markup)
        self.assertIn("does not state a salary range", markup)

    def test_no_match_result_renders_not_yet_evaluated(self):
        markup = _match_cell(self.view())
        self.assertIn("not yet evaluated", markup)
        self.assertNotIn("score", markup)

    def test_an_unevaluated_job_never_shows_a_zero_score(self):
        result = MatchResult(job_id="j-1", tier=MatchTier.NOT_YET_EVALUATED,
                             score=None, confidence=Confidence.INSUFFICIENT,
                             insufficient_reason="no profile available")
        markup = _match_cell(self.view(match=result))
        self.assertIn("not yet evaluated", markup)
        self.assertIn("no profile available", markup)
        self.assertNotIn("score 0", markup)

    def test_insufficient_evidence_renders_uncertainty_honestly(self):
        result = MatchResult(
            job_id="j-1", tier=MatchTier.NOT_YET_EVALUATED, score=91.0,
            confidence=Confidence.LOW,
            insufficient_reason="evidence supplied was too thin to support a tier",
        )
        markup = _match_cell(self.view(match=result))
        self.assertIn("uncertain", markup)
        self.assertIn("too thin to support a tier", markup)
        # The tier is not asserted, so it must not be shown as a real one.
        self.assertNotIn("strong_match", markup)

    def test_an_ineligible_job_shows_the_veto_and_its_reason(self):
        job = job_for("europe_restricted.json")
        result = assess_match(
            job, {"technical": 90, "experience": 90,
                  "behavioral": 90, "career": 90},
            profile=PROFILE,
            eligibility=__import__(
                "app.jobs.eligibility", fromlist=["evaluate_eligibility"]
            ).evaluate_eligibility(job, candidate_country="KE"),
        )
        markup = _match_cell(self.view(match=result))
        self.assertIn("unsuitable", markup)
        self.assertIn("excluded by eligibility", markup)
        self.assertIn("EU", markup)

    def test_evidence_quotes_are_escaped(self):
        result = MatchResult(
            job_id="j-1", tier=MatchTier.STRONG_MATCH, score=80.0,
            confidence=Confidence.HIGH,
            evidence=[EvidenceItem('<script>alert(1)</script>', "job",
                                   '"><img src=x onerror=alert(1)>')],
        )
        markup = _match_cell(self.view(match=result))
        # Assert on the payload itself, not on fragments like "onerror=" that
        # also occur in legitimately escaped text. The raw payload must not
        # survive, and its escaped form must.
        payload = '"><img src=x onerror=alert(1)>'
        self.assertNotIn(payload, markup)
        self.assertNotIn("<script>", markup)
        self.assertNotIn("<img", markup)
        self.assertIn("&lt;script&gt;", markup)
        self.assertIn("&quot;&gt;&lt;img src=x onerror=alert(1)&gt;", markup)

    def test_evidence_shows_its_source(self):
        result = MatchResult(
            job_id="j-1", tier=MatchTier.STRONG_MATCH, score=80.0,
            confidence=Confidence.HIGH,
            evidence=[EvidenceItem("Kafka", "profile", "Kafka")],
        )
        markup = _match_cell(self.view(match=result))
        self.assertIn("profile", markup)
        self.assertIn("Kafka", markup)

    def test_missing_requirements_are_shown(self):
        result = MatchResult(
            job_id="j-1", tier=MatchTier.STRETCH, score=42.0,
            confidence=Confidence.HIGH, missing_requirements=["no Go experience stated"],
        )
        markup = _match_cell(self.view(match=result))
        self.assertIn("missing:", markup)
        self.assertIn("no Go experience stated", markup)


def _fields(result) -> dict:
    """Turn a MatchResult (or its dict form) into build_view kwargs."""
    data = result if isinstance(result, dict) else result.to_dict()
    return {
        "match_tier": data["tier"],
        "match_score": data["score"],
        "match_confidence": data["confidence"],
        "match_evidence": data["evidence"],
        "match_missing": data["missing_requirements"],
        "match_concerns": data["concerns"],
        "match_insufficient_reason": data["insufficient_reason"],
        "match_vetoes": data["vetoes"],
    }


class MatchMappingTests(unittest.TestCase):
    def test_a_match_result_object_is_accepted(self):
        result = MatchResult(job_id="j", tier=MatchTier.STRONG_MATCH, score=88.0,
                             confidence=Confidence.HIGH)
        record = {"job_id": "j", "job": {"job_id": "j", "title": "t", "company": "Acme",
                                         "url": "https://x.test/j"}}
        view = build_view(record, match=result)
        self.assertEqual(view.match_tier, "strong_match")
        self.assertEqual(view.match_score, 88.0)

    def test_a_dict_is_accepted(self):
        record = {"job_id": "j", "job": {"job_id": "j", "title": "t", "company": "Acme",
                                         "url": "https://x.test/j"}}
        view = build_view(record, match={"tier": "stretch", "score": 40.0,
                                         "confidence": "high"})
        self.assertEqual(view.match_tier, "stretch")
        self.assertEqual(view.match_score, 40.0)

    def test_an_unrecognised_tier_degrades_to_not_yet_evaluated(self):
        # Better to say we do not know than to render a tier we cannot honour.
        record = {"job_id": "j", "job": {"job_id": "j", "title": "t", "company": "Acme",
                                         "url": "https://x.test/j"}}
        view = build_view(record, match={"tier": "excellent-fit", "score": 99.0})
        self.assertEqual(view.match_tier, "not_yet_evaluated")
        self.assertFalse(view.match_evaluated)

    def test_a_non_numeric_score_degrades_to_none(self):
        record = {"job_id": "j", "job": {"job_id": "j", "title": "t", "company": "Acme",
                                         "url": "https://x.test/j"}}
        view = build_view(record, match={"tier": "stretch", "score": "high"})
        self.assertIsNone(view.match_score)

    def test_uncertainty_is_derived_not_stored(self):
        record = {"job_id": "j", "job": {"job_id": "j", "title": "t", "company": "Acme",
                                         "url": "https://x.test/j"}}
        view = build_view(record, match={"tier": "stretch", "confidence": "low"})
        self.assertTrue(view.match_uncertain)
        view = build_view(record, match={"tier": "stretch", "confidence": "high"})
        self.assertFalse(view.match_uncertain)


class MatchPanelTests(unittest.TestCase):
    def test_no_assessments_is_stated_as_missing_analysis(self):
        views = [build_view({"job_id": "j", "job": {
            "job_id": "j", "title": "t", "company": "Acme",
            "url": "https://x.test/j"}})]
        markup = _match_panel(views)
        self.assertIn("No match assessment has been run", markup)
        self.assertIn("missing analysis", markup)

    def test_the_panel_counts_each_tier(self):
        records = [{"job_id": f"j{i}", "job": {
            "job_id": f"j{i}", "title": "t", "company": "Acme",
            "url": f"https://x.test/{i}"}}
            for i in range(3)]
        views = [
            build_view(records[0], match={"tier": "strong_match", "confidence": "high"}),
            build_view(records[1], match={"tier": "stretch", "confidence": "low"}),
            build_view(records[2]),
        ]
        markup = _match_panel(views)
        self.assertIn("strong_match", markup)
        self.assertIn("stretch", markup)
        self.assertIn("not_yet_evaluated", markup)
        self.assertIn("uncertain of those", markup)


class FilterAndSortTests(unittest.TestCase):
    def views(self):
        records = [{"job_id": f"j{i}", "job": {
            "job_id": f"j{i}", "title": f"T{i}", "company": f"C{i}",
            "url": f"https://x.test/{i}", "location": "Remote",
            "posted_date": f"2026-0{i}-01"}} for i in range(1, 5)]
        return [
            build_view(records[0], match={"tier": "stretch", "confidence": "high"}),
            build_view(records[1], match={"tier": "strong_match", "confidence": "high"}),
            build_view(records[2], match={"tier": "credible_match",
                                         "confidence": "low"}),
            build_view(records[3]),
        ]

    def test_filter_by_tier(self):
        self.assertEqual(len(_filtered(self.views(), match_tier="strong_match")), 1)

    def test_filter_to_uncertain_only(self):
        selected = _filtered(self.views(), uncertain=True)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].match_tier, "credible_match")

    def test_filter_by_posted_after(self):
        selected = _filtered(self.views(), posted_after="2026-02-15")
        self.assertEqual([v.job_id for v in selected], ["j3", "j4"])

    def test_sorting_by_match_orders_best_first(self):
        ordered = _sorted(self.views(), "match")
        self.assertEqual(
            [v.match_tier for v in ordered],
            ["strong_match", "credible_match", "stretch", "not_yet_evaluated"],
        )

    def test_sorting_by_match_keeps_unevaluated_visible(self):
        ordered = _sorted(self.views(), "match")
        self.assertIn("not_yet_evaluated", [v.match_tier for v in ordered])

    def test_sorting_is_stable_against_input_order(self):
        views = self.views()
        first = [v.job_id for v in _sorted(views, "match")]
        second = [v.job_id for v in _sorted(list(reversed(views)), "match")]
        self.assertEqual(first, second)


class DashboardIntegrationTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.store = JobStore(self.root / "data")
        ingest([load(n) for n in ("basic.json", "salary_ksh.json")],
               source="weworkremotely", store=self.store, observed_at=NOW)
        self.job_id = str(self.store.load_jobs()[0]["job_id"])

    def render(self, **kwargs):
        path = self.root / "d.html"
        render_dashboard_file(self.store, path, generated_at="2026-10-09", **kwargs)
        return path.read_text(encoding="utf-8")

    def test_the_match_column_is_present(self):
        self.assertIn("<th>Match</th>", self.render())

    def test_a_supplied_match_appears_in_the_page(self):
        result = MatchResult(job_id=self.job_id, tier=MatchTier.STRONG_MATCH,
                             score=91.0, confidence=Confidence.HIGH,
                             evidence=[EvidenceItem("ELT", "job", "ELT")])
        markup = self.render(matches={self.job_id: result})
        self.assertIn("strong_match", markup)
        self.assertIn("91.0", markup)

    def test_jobs_without_a_match_report_not_yet_evaluated(self):
        markup = self.render()
        self.assertIn("not yet evaluated", markup)

    def test_the_dashboard_stays_read_only(self):
        before = self.store.jobs_path.read_text(encoding="utf-8")
        self.render(matches={self.job_id: MatchResult(
            job_id=self.job_id, tier=MatchTier.STRONG_MATCH, score=90.0,
            confidence=Confidence.HIGH)})
        self.assertEqual(before, self.store.jobs_path.read_text(encoding="utf-8"))

    def test_no_scripts_or_external_references(self):
        markup = self.render().casefold()
        self.assertNotIn("<script", markup)
        self.assertNotIn("src=", markup)
        self.assertNotIn("<link", markup)
        self.assertNotIn("javascript:", markup)

    def test_no_secrets_or_internal_paths_are_rendered(self):
        markup = self.render()
        for marker in ("password", "api_key", "BEGIN PRIVATE", "C:\\", "/Users/"):
            with self.subTest(marker=marker):
                self.assertNotIn(marker, markup)


class NoSecretsTests(unittest.TestCase):
    def test_the_reporting_module_imports_nothing_that_performs_io(self):
        import ast

        path = Path(__file__).resolve().parent.parent / "app" / "reporting" / "jobs.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])
        forbidden = {"playwright", "requests", "httpx", "urllib", "socket",
                     "subprocess", "ollama", "aiohttp", "selenium", "asyncio"}
        self.assertEqual(imported & forbidden, set())

    def test_no_reviewer_can_be_misled_by_an_exclusion_hiding_a_row(self):
        # An unsuitable job must still be present in the rendered output.
        store = store_with("basic.json")
        result = MatchResult(job_id=str(store.load_jobs()[0]["job_id"]),
                             tier=MatchTier.UNSUITABLE, score=None,
                             confidence=Confidence.HIGH, vetoes=["region restricted"])
        views = [build_view(r, match=result) for r in store.load_jobs()]
        markup = render_dashboard_html(views)
        self.assertIn("unsuitable", markup)
        self.assertIn("region restricted", markup)


if __name__ == "__main__":
    unittest.main()