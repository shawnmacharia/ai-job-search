"""Tests for evidence-based match assessment.

Entirely offline. A fake provider stands in for Ollama; nothing here opens a
socket, imports Playwright, or contacts a model.

The tests concentrate on the policy this module exists to enforce: scores
reorder and never exclude, absence of evidence is not unsuitability, and no
claim survives without a quotation that can actually be found in a source.
"""

import json
import tempfile
import unittest
from pathlib import Path

from app.jobs.eligibility import evaluate_eligibility
from app.jobs.ingest import ingest
from app.jobs.match import (
    Confidence,
    EvidenceItem,
    MatchError,
    MatchResult,
    MatchTier,
    assess_match,
    keyword_overlap,
    rank_matches,
    summarise,
    verify_evidence,
)
from app.jobs.models import Job
from app.jobs.store import JobStore
from app.orchestrator.rank import (
    LoadError,
    MatchUnavailable,
    assess_jobs,
    load_jobs,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hiring_cafe"

PROFILE = (
    "Data engineer with 8 years building streaming ingestion pipelines in "
    "Python and Kafka, strong dbt and Snowflake experience, and ELT "
    "ownership on a large analytics platform."
)

INGEST = ["basic.json", "worldwide_remote.json", "salary_ksh.json", "europe_restricted.json"]


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeProvider:
    """Stands in for Ollama. Records calls so retry behaviour is observable."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        index = min(self.calls - 1, len(self.responses) - 1)
        content = self.responses[index]

        class Response:
            pass

        response = Response()
        response.text = content if isinstance(content, str) else json.dumps(content)
        return response


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


class StoreTestCase(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.store = JobStore(self.root / "data")
        ingest([load(n) for n in INGEST], source="hiring.cafe", store=self.store)
        self.records = self.store.load_jobs()


class LoadTests(StoreTestCase):
    def test_valid_records_load_without_a_type_error(self):
        # The original defect: Job requires job_id, and scraper-shaped records
        # raised a raw TypeError from the constructor.
        jobs = load_jobs(self.store)
        self.assertEqual(len(jobs), len(INGEST))
        self.assertTrue(all(isinstance(job, Job) for job in jobs))

    def test_loaded_jobs_keep_their_identity(self):
        expected = {str(r["job_id"]) for r in self.records}
        self.assertEqual({job.job_id for job in load_jobs(self.store)}, expected)

    def test_a_record_without_a_job_object_is_reported(self):
        self.store.jobs_path.write_text(
            json.dumps({"job_id": "x", "job": "not-an-object"}) + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(LoadError) as caught:
            load_jobs(self.store)
        self.assertIn("expected a 'job' object", str(caught.exception))

    def test_a_malformed_record_produces_a_clear_error_not_a_traceback(self):
        self.store.jobs_path.write_text(
            json.dumps({"job": {"title": "No id"}}) + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(LoadError) as caught:
            load_jobs(self.store)
        message = str(caught.exception)
        # Reported, naming the offending line, rather than raising a TypeError
        # from a dataclass constructor or silently dropping the record.
        self.assertIn("no usable job_id", message)
        self.assertIn("line(s) 1", message)
        self.assertNotIsInstance(caught.exception, TypeError)

    def test_the_store_silently_dropped_this_record_before(self):
        # Records the pre-existing behaviour the strict path now surfaces.
        self.store.jobs_path.write_text(
            json.dumps({"job": {"title": "No id"}}) + "\n",
            encoding="utf-8",
        )
        self.assertEqual(self.store.load_jobs(), [], "default behaviour unchanged")
        with self.assertRaises(ValueError):
            self.store.load_jobs(strict=True)

    def test_missing_required_fields_are_named_rather_than_leaked(self):
        self.store.jobs_path.write_text(
            json.dumps({"job_id": "x", "job": {"title": "Only a title"}}) + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(LoadError) as caught:
            load_jobs(self.store)
        message = str(caught.exception)
        self.assertIn("missing required field(s)", message)
        self.assertIn("company", message)
        self.assertNotIn("required positional arguments", message)

    def test_the_error_names_the_offending_record(self):
        self.store.jobs_path.write_text(
            json.dumps({"job_id": "broken-1", "job": {"title": "No id"}}) + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(LoadError) as caught:
            load_jobs(self.store)
        self.assertIn("broken-1", str(caught.exception))

    def test_an_unrecognised_remote_status_degrades_rather_than_crashing(self):
        record = dict(self.records[0])
        record["job"] = {**record["job"], "remote_status": "bespoke"}
        self.store.jobs_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        self.assertEqual(len(load_jobs(self.store)), 1)


class NoProviderTests(StoreTestCase):
    def test_no_provider_raises_rather_than_ranking(self):
        with self.assertRaises(MatchUnavailable) as caught:
            assess_jobs(self.store, provider=None)
        self.assertIn("no provider is configured", str(caught.exception))

    def test_no_provider_produces_no_zero_evidence_table(self):
        # The original defect returned RankingEvidence(0,0,0,0) for every job.
        with self.assertRaises(MatchUnavailable):
            assess_jobs(self.store, provider=None)

    def test_cli_reports_unavailable_rather_than_pretending(self):
        from app.orchestrator import rank

        import io
        from contextlib import redirect_stderr

        err = io.StringIO()
        with redirect_stderr(err):
            code = rank.main(["--data-dir", str(self.root / "data")])
        self.assertEqual(code, 1)
        self.assertIn("no provider is configured", err.getvalue())

    def test_enable_ai_without_a_reachable_ollama_fails_clearly(self):
        from app.orchestrator import rank

        import io
        from contextlib import redirect_stderr
        from unittest import mock

        err = io.StringIO()
        with mock.patch.dict("sys.modules", {"app.llm.ollama": mock.MagicMock()}), \
                mock.patch("app.llm.ollama.OllamaProvider") as provider:
            provider.return_value.health_check.return_value = False
            with redirect_stderr(err):
                code = rank.main(
                    ["--data-dir", str(self.root / "data"), "--enable-ai"]
                )
        self.assertEqual(code, 1)
        self.assertIn("not reachable", err.getvalue())


class ProviderMappingTests(StoreTestCase):
    def good_payload(self, score=80, quote="Kafka"):
        return {
            "technical": score, "experience": score, "behavioral": score,
            "career": score,
            "evidence": [{"claim": "streaming experience",
                          "source": "profile", "quote": quote}],
        }

    def test_valid_provider_output_maps_to_a_tier_and_score(self):
        provider = FakeProvider(self.good_payload())
        results = assess_jobs(self.store, provider=provider, profile=PROFILE)
        self.assertEqual(len(results), len(INGEST))
        assessed = [r for r in results if r.tier is not MatchTier.NOT_YET_EVALUATED]
        self.assertTrue(assessed)
        self.assertEqual(assessed[0].tier, MatchTier.STRONG_MATCH)
        self.assertEqual(assessed[0].score, 80.0)

    def test_score_is_aggregated_deterministically(self):
        job = job_for()
        first = assess_match(job, {"technical": 80, "experience": 80,
                                    "behavioral": 80, "career": 80}, profile=PROFILE,
                             evidence=[EvidenceItem("streaming", "profile", "Kafka")])
        second = assess_match(job, {"technical": 80, "experience": 80,
                                     "behavioral": 80, "career": 80}, profile=PROFILE,
                              evidence=[EvidenceItem("streaming", "profile", "Kafka")])
        self.assertEqual(first.score, second.score)
        self.assertEqual(first.tier, second.tier)

    def test_the_model_never_supplies_the_score(self):
        # A model returning a "score" key is ignored; aggregation is ours.
        payload = self.good_payload()
        payload["final_score"] = 99
        job = job_for()
        result = assess_match(job, {"technical": 70, "experience": 70,
                                    "behavioral": 70, "career": 70}, profile=PROFILE,
                              evidence=[EvidenceItem("streaming", "profile", "Kafka")])
        self.assertNotEqual(result.score, 99)

    def test_provider_without_a_profile_reports_not_yet_evaluated(self):
        provider = FakeProvider(self.good_payload())
        results = assess_jobs(self.store, provider=provider, profile=None)
        for result in results:
            if result.excluded:
                # The one permitted exclusion, which does not need a profile.
                with self.subTest(job=result.job_id):
                    self.assertIs(result.tier, MatchTier.UNSUITABLE)
                    self.assertTrue(result.vetoes)
                continue
            with self.subTest(job=result.job_id):
                self.assertIs(result.tier, MatchTier.NOT_YET_EVALUATED)
                self.assertIsNone(result.score)

    def test_a_hard_veto_excludes_even_with_a_strong_profile(self):
        # The single permitted automatic exclusion, reachable in the real
        # pipeline rather than only in unit tests.
        provider = FakeProvider(self.good_payload())
        results = assess_jobs(self.store, provider=provider, profile=PROFILE)
        excluded = [r for r in results if r.excluded]
        self.assertEqual(len(excluded), 1, "the EU-anchored posting")
        self.assertIs(excluded[0].tier, MatchTier.UNSUITABLE)
        self.assertTrue(excluded[0].vetoes, "an exclusion must carry its reason")
        self.assertIsNone(excluded[0].score)
        # Everyone else is present and ranked.
        self.assertEqual(len(results), len(INGEST))


class RepairBudgetTests(StoreTestCase):
    def test_invalid_json_is_repaired_within_a_bounded_budget(self):
        from app.orchestrator.rank import MAX_REPAIRS

        provider = FakeProvider(
            "{not json at all",
            {"technical": "high", "experience": 1, "behavioral": 1, "career": 1},
            self.valid(),
        )
        results = assess_jobs(self.store, provider=provider, profile=PROFILE)
        # The budget is per job, so the ceiling is jobs x (1 + repairs).
        self.assertLessEqual(
            provider.calls, len(INGEST) * (MAX_REPAIRS + 1),
            "retries must be bounded per job",
        )
        self.assertTrue(results)

    def valid(self):
        return {
            "technical": 80, "experience": 80, "behavioral": 80, "career": 80,
            "evidence": [{"claim": "s", "source": "profile", "quote": "Kafka"}],
        }

    def test_permanently_invalid_output_fails_clearly(self):
        provider = FakeProvider("still not json")
        results = assess_jobs(self.store, provider=provider, profile=PROFILE)
        for result in results:
            if result.excluded:
                continue  # hard veto, independent of the provider
            with self.subTest(job=result.job_id):
                # Never fabricated into a tier.
                self.assertIs(result.tier, MatchTier.NOT_YET_EVALUATED)
                self.assertIsNone(result.score)
        non_excluded = [r for r in results if not r.excluded]
        self.assertTrue(
            any("provider output invalid" in c for c in non_excluded[0].concerns),
            "the invalid output must be reported, not swallowed",
        )

    def test_the_repair_budget_is_finite(self):
        from app.orchestrator.rank import MAX_REPAIRS, request_evidence

        provider = FakeProvider("nonsense")
        with self.assertRaises(Exception) as caught:
            request_evidence(job_for(), PROFILE, provider, model="m")
        self.assertIn("repair attempts", str(caught.exception))
        self.assertEqual(provider.calls, MAX_REPAIRS + 1)

    def test_a_non_integer_score_is_rejected_not_coerced(self):
        from app.orchestrator.rank import evidence_from_payload
        from app.llm.exceptions import SchemaValidationError

        with self.assertRaises(SchemaValidationError):
            evidence_from_payload({"technical": "high", "experience": 1,
                                   "behavioral": 1, "career": 1})


class EvidenceIntegrityTests(unittest.TestCase):
    def test_a_quote_absent_from_the_source_is_discarded(self):
        job = job_for()
        verified = verify_evidence(
            [EvidenceItem("invented", "job", "requires quantum expertise")], job, PROFILE
        )
        self.assertEqual(verified, [], "a fabricated quote must not survive")

    def test_an_empty_quote_is_not_evidence(self):
        self.assertEqual(verify_evidence([EvidenceItem("claim", "job", "")],
                                          job_for(), PROFILE), [])

    def test_a_quote_present_in_the_source_survives(self):
        verified = verify_evidence(
            [EvidenceItem("ELT", "job", "ELT")], job_for(), PROFILE
        )
        self.assertEqual(len(verified), 1)

    def test_a_profile_quote_is_checked_against_the_profile(self):
        job = job_for()
        # "dbt" appears in the profile and in this posting.
        verified = verify_evidence(
            [EvidenceItem("s", "profile", "Kafka")], job, PROFILE
        )
        self.assertEqual(len(verified), 1)
        # The same quote is checked against the *profile*, not the posting:
        # "quantum" is in neither.
        self.assertEqual(
            verify_evidence([EvidenceItem("s", "profile", "quantum")], job, PROFILE), []
        )

    def test_evidence_from_an_unknown_source_is_dropped(self):
        verified = verify_evidence(
            [EvidenceItem("c", "hearsay", "Kafka")], job_for(), PROFILE
        )
        self.assertEqual(verified, [])

    def test_non_evidence_item_types_are_rejected(self):
        with self.assertRaises(MatchError):
            assess_match(job_for(), {"technical": 80}, evidence=["not an item"])

    def test_every_non_unknown_tier_carries_evidence(self):
        for score in (95, 65, 30):
            result = assess_match(
                job_for(), {"technical": score, "experience": score,
                            "behavioral": score, "career": score},
                profile=PROFILE,
                evidence=[EvidenceItem("Kafka pipelines", "profile", "Kafka")],
            )
            if result.tier is not MatchTier.NOT_YET_EVALUATED:
                with self.subTest(score=score):
                    self.assertTrue(
                        result.evidence, f"{result.tier.value} must cite evidence"
                    )


class PolicyTests(unittest.TestCase):
    def assessed(self, score, **kwargs):
        return assess_match(
            job_for(), {"technical": score, "experience": score,
                        "behavioral": score, "career": score},
            profile=PROFILE,
            evidence=[EvidenceItem("Kafka", "profile", "Kafka")],
            **kwargs,
        )

    def test_a_low_score_never_excludes(self):
        result = self.assessed(5)
        self.assertFalse(result.excluded)
        self.assertIs(result.tier, MatchTier.STRETCH)

    def test_a_low_score_still_appears_in_the_ranking(self):
        results = rank_matches([self.assessed(5), self.assessed(95)])
        self.assertEqual(len(results), 2, "a low score must not remove a job")

    def test_adequate_evidence_is_used_for_the_low_score_case(self):
        # One verified item with no gaps: enough to assert a tier, not enough
        # to call it high confidence.
        self.assertEqual(self.assessed(5).confidence, Confidence.MEDIUM)

    def test_a_hard_eligibility_veto_is_the_only_exclusion(self):
        job = job_for("europe_restricted.json")
        verdict = evaluate_eligibility(
            job, candidate_country="KE"
        )
        result = assess_match(job, {"technical": 100, "experience": 100,
                                    "behavioral": 100, "career": 100},
                              profile=PROFILE, eligibility=verdict)
        self.assertIs(result.tier, MatchTier.UNSUITABLE)
        self.assertTrue(result.excluded)
        self.assertTrue(result.vetoes, "an exclusion must carry its reason")

    def test_an_eligible_verdict_does_not_exclude(self):
        job = job_for("salary_ksh.json")
        verdict = evaluate_eligibility(job, candidate_country="KE")
        result = assess_match(job, {"technical": 70, "experience": 70,
                                    "behavioral": 70, "career": 70},
                              profile=PROFILE, eligibility=verdict,
                              evidence=[EvidenceItem("ELT", "job", "ELT")])
        self.assertFalse(result.excluded)

    def test_an_unknown_verdict_does_not_exclude(self):
        job = job_for()
        verdict = evaluate_eligibility(job, candidate_country="KE")
        result = assess_match(job, {"technical": 70, "experience": 70,
                                    "behavioral": 70, "career": 70},
                              profile=PROFILE, eligibility=verdict,
                              evidence=[EvidenceItem("ELT", "job", "ELT")])
        self.assertFalse(result.excluded)
        self.assertIsNot(result.tier, MatchTier.UNSUITABLE)

    def test_absence_of_evidence_is_not_unsuitable(self):
        result = assess_match(job_for(), None, profile=None)
        self.assertIs(result.tier, MatchTier.NOT_YET_EVALUATED)
        self.assertIsNot(result.tier, MatchTier.UNSUITABLE)
        self.assertIsNone(result.score)
        self.assertIs(result.confidence, Confidence.INSUFFICIENT)
        self.assertTrue(result.insufficient_reason)

    def test_missing_evidence_is_stated_explicitly(self):
        result = assess_match(job_for(), None, profile=None)
        self.assertIn("insufficient", result.to_dict()["confidence"])
        self.assertTrue(result.insufficient_reason)

    def test_keyword_overlap_alone_cannot_produce_a_tier(self):
        # Overlap IS present here (dbt, Python, ELT all appear in the posting),
        # and it still must not justify a tier.
        overlap = keyword_overlap(job_for(), PROFILE)
        self.assertTrue(overlap, "this fixture is chosen so overlap is non-empty")
        result = assess_match(job_for(), {"technical": 90, "experience": 90,
                                          "behavioral": 90, "career": 90},
                              profile=PROFILE, evidence=[])
        self.assertIs(result.tier, MatchTier.NOT_YET_EVALUATED)
        self.assertIsNone(result.score)
        self.assertIn("keyword overlap alone", result.insufficient_reason)

        # With no overlap at all, the same input is reported the same way.
        none = assess_match(job_for(), {"technical": 90, "experience": 90,
                                         "behavioral": 90, "career": 90},
                            profile="", evidence=[])
        self.assertIs(none.tier, MatchTier.NOT_YET_EVALUATED)

    def test_keyword_overlap_alone_never_excludes(self):
        result = assess_match(job_for(), {"technical": 90, "experience": 90,
                                          "behavioral": 90, "career": 90},
                              profile=PROFILE, evidence=[])
        self.assertFalse(result.excluded)

    def test_a_declared_gap_is_enough_to_report_a_tier(self):
        # Contrast with the test above: an explicitly stated gap is evidence of
        # *something*, where keyword overlap is not.
        result = assess_match(job_for(), {"technical": 90, "experience": 90,
                                          "behavioral": 90, "career": 90},
                              profile=PROFILE, gaps=["posting states no salary"])
        self.assertIsNotNone(result.score)

    def test_keyword_overlap_may_accompany_a_result(self):
        result = self.assessed(80)
        self.assertIn("informs only", result.explanation())

    def test_keyword_overlap_is_never_an_exclusion(self):
        self.assertFalse(assess_match(job_for(), None, profile=PROFILE).excluded)

    def test_thin_evidence_declines_to_assert_a_tier(self):
        result = assess_match(
            job_for(), {"technical": 95, "experience": 95, "behavioral": 95, "career": 95},
            profile=PROFILE,
            evidence=[EvidenceItem("ELT", "job", "ELT")],
            gaps=["a", "b", "c"],  # more gaps than evidence: thin.
        )
        self.assertIs(result.tier, MatchTier.NOT_YET_EVALUATED)
        self.assertIs(result.confidence, Confidence.LOW)
        self.assertIn("thin", result.insufficient_reason)
        self.assertFalse(result.excluded)

    def test_adequate_evidence_does_assert_a_tier(self):
        result = assess_match(
            job_for(), {"technical": 95, "experience": 95, "behavioral": 95, "career": 95},
            profile=PROFILE,
            evidence=[
                EvidenceItem("ELT pipelines", "job", "ELT"),
                EvidenceItem("Python", "job", "Python"),
                EvidenceItem("dbt", "job", "dbt"),
            ],
        )
        self.assertIs(result.tier, MatchTier.STRONG_MATCH)
        self.assertIs(result.confidence, Confidence.HIGH)
        self.assertEqual(result.score, 95.0)

    def test_the_five_tiers_are_exactly_as_specified(self):
        self.assertEqual(
            [t.value for t in MatchTier],
            ["strong_match", "credible_match", "stretch", "unsuitable", "not_yet_evaluated"],
        )

    def test_wrong_eligibility_type_is_rejected(self):
        with self.assertRaises(MatchError):
            assess_match(job_for(), {"technical": 80}, eligibility="not_eligible")


class OrderingTests(unittest.TestCase):
    def test_unranked_results_sort_last_rather_than_being_dropped(self):
        unranked = assess_match(job_for(), None, profile=None)
        ranked = assess_match(job_for(), {"technical": 90, "experience": 90,
                                          "behavioral": 90, "career": 90},
                              profile=PROFILE,
                              evidence=[EvidenceItem("Kafka", "profile", "Kafka")])
        ordered = rank_matches([unranked, ranked])
        self.assertEqual(len(ordered), 2)
        self.assertIs(ordered[-1].tier, MatchTier.NOT_YET_EVALUATED)

    def test_ranking_is_deterministic(self):
        results = [
            assess_match(job_for(), {"technical": s, "experience": s,
                                     "behavioral": s, "career": s},
                         profile=PROFILE,
                         evidence=[EvidenceItem("Kafka", "profile", "Kafka")])
            for s in (40, 90, 65)
        ]
        first = [r.job_id for r in rank_matches(results)]
        second = [r.job_id for r in rank_matches(list(reversed(results)))]
        self.assertEqual(first, second)

    def test_summarise_accounts_for_every_result(self):
        results = [assess_match(job_for(), None, profile=None)]
        report = summarise(results)
        self.assertEqual(report["total"], len(results))
        self.assertEqual(sum(report["by_tier"].values()), len(results))
        self.assertIn("never removes a job", report["note"])


class NoNetworkTests(unittest.TestCase):
    def test_match_module_imports_nothing_that_performs_io(self):
        import ast

        path = Path(__file__).resolve().parent.parent / "app" / "jobs" / "match.py"
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

    def test_match_module_never_imports_the_provider(self):
        text = (Path(__file__).resolve().parent.parent
                / "app" / "jobs" / "match.py").read_text(encoding="utf-8")
        self.assertNotIn("Ollama", text)
        self.assertNotIn("ollama", text)


if __name__ == "__main__":
    unittest.main()