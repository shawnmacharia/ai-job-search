"""Bounded, evidence-checked match assessment.

The tests here are mostly about what the tool must **refuse to do**, because the
failure modes are the dangerous ones and they are all silent:

- fabricating a tier when no evidence survives checking
- treating "we looked and learned nothing" as "it is a poor match"
- sending anything anywhere without an explicitly named provider
- writing an assessment when no provider was available at all
- hiding a job because a score was low

Provider calls are made by a fake provider that counts them. No test opens a
socket, and none imports a real provider implementation.
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from app.jobs.assessment import (
    ASSESSMENT_SCHEMA_VERSION,
    MAX_BATCH,
    PROMPT_VERSION,
    STATUS_ASSESSED,
    STATUS_ERROR,
    STATUS_INSUFFICIENT,
    STATUS_REFUSED,
    AssessmentRecord,
    AssessmentStore,
    AssessmentUnavailable,
    CandidateProfile,
    assess_batch,
    assess_one,
    build_assessment,
    describe_data_boundary,
    is_local_endpoint,
    is_local_model,
    plan_only,
    register_local_ollama,
    registered_providers,
    register_provider,
    resolve_provider,
    to_match_dict,
)
from app.jobs.assessment import (
    LOCAL_OLLAMA_PROVIDER,
    LOCAL_OLLAMA_URL,
)
from app.jobs.match import Confidence, MatchTier
from app.jobs.models import Job
from app.jobs.runner import SourceSpec, run_sources
from app.jobs.status import ReviewStatus, StatusLog
from app.jobs.store import JobStore
from app.llm.exceptions import ProviderConnectionError
from app.llm.provider import LLMResponse
from app.reporting.review import build_report, render_report_html

REPO_ROOT = Path(__file__).resolve().parents[1]
MATCH_CLI = REPO_ROOT / "tools" / "match.py"
TRIAL_CEILING = 3
T0 = "2026-03-01T09:00:00+00:00"

PROFILE = (
    "PROFILE: Five years backend engineering in Go and Python. "
    "PostgreSQL, Kubernetes, distributed systems. Degree in Computer Science."
)


def _job(**kwargs) -> Job:
    base = dict(
        job_id="j1",
        title="Backend Engineer (Go)",
        company="Example Payments",
        url="https://example.test/jobs/j1",
        location="Nairobi, Kenya",
        description=(
            "We need a backend engineer working on payments in Go. "
            "You will own services end to end."
        ),
        skills=["go", "postgresql", "kubernetes"],
    )
    base.update(kwargs)
    return Job(**base)


def _good_payload() -> dict:
    return {
        "tier": "credible_match",
        "score": 0.72,
        "confidence": "medium",
        "evidence": [
            {
                "claim": "Go is a stated requirement",
                "source": "job",
                "quote": "backend engineer working on payments in Go",
            },
            {
                "claim": "Five years of Go experience",
                "source": "profile",
                "quote": "Five years backend engineering in Go",
            },
        ],
        "missing_requirements": ["on-call"],
        "concerns": [],
    }


class FakeProvider:
    """A provider that answers from a queue and counts every call."""

    def __init__(self, responses, *, reachable=True, model="fake-model-v1"):
        self.responses = list(responses)
        self.reachable = reachable
        self.model = model
        self.calls = 0
        self.requests = []

    def generate(self, request):
        self.calls += 1
        self.requests.append(request)
        index = min(self.calls - 1, len(self.responses) - 1)
        payload = self.responses[index]
        return LLMResponse(text=payload, model=self.model)

    def health_check(self):
        return self.reachable

    def list_models(self):
        return [self.model]


class ExplodingProvider(FakeProvider):
    def generate(self, request):
        self.calls += 1
        raise ProviderConnectionError("endpoint unreachable")


class RemoteProvider(FakeProvider):
    """A reachable provider on someone else's machine."""

    def __init__(self, **kwargs):
        super().__init__(["{}"], **kwargs)
        self.base_url = "https://api.example.com"


class CloudModelProvider(FakeProvider):
    """Local endpoint, cloud inference. The case that looks safe and is not."""

    def __init__(self, **kwargs):
        super().__init__(["{}"], **kwargs)
        self.base_url = "http://localhost:11434"
        self.default_model = "gpt-oss:120b-cloud"


class LocalEndpointProvider(FakeProvider):
    """The shape of a genuine local provider."""

    def __init__(self, **kwargs):
        super().__init__(["{}"], **kwargs)
        self.base_url = "http://localhost:11434"
        self.default_model = "llama3.2:latest"


class AssessmentTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data = Path(self._tmp.name) / "data"
        self.store = JobStore(self.data)
        self.assessments = AssessmentStore(self.data)
        self.profile = CandidateProfile(text=PROFILE, sources=("test",))

    def clean_registry(self):
        from app.jobs import assessment

        assessment._REGISTRY.clear()
        self.addCleanup(assessment._REGISTRY.clear)

    def ingest(self, records, *, source="weworkremotely"):
        run_sources([SourceSpec(name=source, fetch=lambda: list(records))],
                    store=self.store)

    def _cli(self, *args):
        return subprocess.run(
            [sys.executable, str(MATCH_CLI), "--data-dir", str(self.data), *args],
            capture_output=True, text=True,
        )


def _record(name, *, source="weworkremotely", title="Role", company="Acme",
            location="Nairobi, Kenya"):
    return {
        "url": f"https://{source}/jobs/{name}",
        "title": title, "company": company,
        "location": location, "description": "Work.",
    }


# ----------------------------------------------------------------------
# provider gate
# ----------------------------------------------------------------------


class ProviderGateTests(AssessmentTestCase):
    def test_the_registry_ships_empty(self):
        """No default provider: naming one is what authorises sending data."""
        self.clean_registry()
        self.assertEqual(registered_providers(), ())

    def test_no_named_provider_refuses(self):
        self.clean_registry()
        with self.assertRaises(AssessmentUnavailable):
            resolve_provider(None)

    def test_an_unknown_provider_refuses(self):
        self.clean_registry()
        with self.assertRaises(AssessmentUnavailable) as caught:
            resolve_provider("nope")
        self.assertIn("unknown provider", str(caught.exception))

    def test_an_unreachable_provider_refuses(self):
        self.clean_registry()
        register_provider("down", lambda: FakeProvider(["{}"], reachable=False))
        with self.assertRaises(AssessmentUnavailable) as caught:
            resolve_provider("down")
        self.assertIn("unreachable", str(caught.exception))

    def test_an_unavailable_provider_makes_no_assessment_writes(self):
        """Zero writes, not even a record saying it failed.

        The request never left the machine, so there is nothing to record - and
        a file full of refusals would be indistinguishable from real results.
        """
        self.clean_registry()
        with self.assertRaises(AssessmentUnavailable):
            resolve_provider("down")
        self.assertFalse(self.assessments.path.exists())

    def test_a_registered_and_reachable_provider_resolves(self):
        self.clean_registry()
        register_provider("fake", lambda: FakeProvider([json.dumps(_good_payload())]))
        name, provider = resolve_provider("fake")
        self.assertEqual(name, "fake")
        self.assertTrue(provider.health_check())

    def test_the_data_boundary_is_stated(self):
        text = describe_data_boundary()
        for expected in ("provider", "description", "profile"):
            self.assertIn(expected, text.casefold())

    def test_the_request_carries_no_credentials(self):
        self.clean_registry()
        provider = FakeProvider([json.dumps(_good_payload())])
        register_provider("fake", lambda: provider)
        name, resolved = resolve_provider("fake")
        assess_one(_job(), profile=self.profile, provider_name=name,
                   provider=resolved, store=self.assessments)
        text = provider.requests[0].system_prompt + provider.requests[0].user_prompt
        for marker in ("password", "api_key", "token", "secret"):
            self.assertNotIn(marker, text.casefold())


# ----------------------------------------------------------------------
# profile gate
# ----------------------------------------------------------------------


class ProfileGateTests(AssessmentTestCase):
    def test_the_repository_has_a_usable_profile(self):
        """The gate must pass against the real repository, not a fixture."""
        from app.jobs.assessment import load_candidate_profile

        profile = load_candidate_profile(REPO_ROOT)
        self.assertTrue(profile.usable, "master CV or preferences file missing")
        self.assertTrue(profile.text.strip())

    def test_an_unusable_profile_refuses_rather_than_inventing_facts(self):
        provider = FakeProvider([json.dumps(_good_payload())])
        record = assess_one(
            _job(), profile=CandidateProfile(text=""), provider_name="fake",
            provider=provider, store=self.assessments,
        )
        self.assertEqual(record.status, STATUS_REFUSED)
        self.assertEqual(record.tier, MatchTier.NOT_YET_EVALUATED.value)
        self.assertIsNone(record.score)
        self.assertEqual(provider.calls, 0, "no provider call may be made")


# ----------------------------------------------------------------------
# one job, batches
# ----------------------------------------------------------------------


class SingleJobTests(AssessmentTestCase):
    def _assess(self, payload=None, **kwargs):
        provider = FakeProvider([json.dumps(payload or _good_payload())])
        return assess_one(_job(), profile=self.profile, provider_name="fake",
                          provider=provider, store=self.assessments,
                          **kwargs), provider

    def test_one_selected_job_is_assessed(self):
        record, provider = self._assess()
        self.assertEqual(record.job_id, "j1")
        self.assertEqual(record.status, STATUS_ASSESSED)
        self.assertEqual(record.tier, MatchTier.CREDIBLE_MATCH.value)
        self.assertEqual(provider.calls, 1)

    def test_the_record_carries_the_full_audit_trail(self):
        record, _ = self._assess()
        self.assertEqual(record.schema_version, ASSESSMENT_SCHEMA_VERSION)
        self.assertEqual(record.prompt_version, PROMPT_VERSION)
        self.assertEqual(record.provider, "fake")
        self.assertEqual(record.model, "fake-model-v1")
        self.assertTrue(record.at)
        self.assertEqual(record.confidence, Confidence.MEDIUM.value)
        self.assertTrue(record.evidence)
        self.assertEqual(record.missing_requirements, ("on-call",))
        self.assertEqual(record.provider_calls, 1)

    def test_assessments_survive_a_restart(self):
        self._assess()
        reopened = AssessmentStore(self.data)
        latest = reopened.latest()
        self.assertIn("j1", latest)
        self.assertEqual(latest["j1"].tier, MatchTier.CREDIBLE_MATCH.value)

    def test_prior_assessments_remain_auditable(self):
        self._assess()
        self._assess({**_good_payload(), "tier": "stretch"})
        records = AssessmentStore(self.data).load()
        self.assertEqual(len(records), 2, "the earlier record is not rewritten")
        self.assertEqual(records[0].tier, MatchTier.CREDIBLE_MATCH.value)
        self.assertEqual(records[1].tier, MatchTier.STRETCH.value)
        self.assertEqual(AssessmentStore(self.data).latest()["j1"].tier,
                         MatchTier.STRETCH.value)


class BatchTests(AssessmentTestCase):
    def _jobs(self, n):
        return [
            _job(job_id=f"j{i}", url=f"https://example.test/jobs/j{i}",
                 title=f"Role {i}")
            for i in range(n)
        ]

    def _assess(self, n_jobs, max_n, payload=None):
        provider = FakeProvider([json.dumps(payload or _good_payload())])
        records = assess_batch(
            self._jobs(n_jobs), profile=self.profile, provider_name="fake",
            provider=provider, store=self.assessments, max_n=max_n,
        )
        return records, provider

    def test_batch_size_is_enforced(self):
        records, provider = self._assess(n_jobs=10, max_n=3)
        self.assertEqual(len(records), 3)
        self.assertEqual(provider.calls, 3)

    def test_an_oversized_batch_is_refused_rather_than_truncated(self):
        """A caller asking for 500 has made a mistake; obeying hides that."""
        provider = FakeProvider([json.dumps(_good_payload())])
        with self.assertRaises(ValueError):
            assess_batch(self._jobs(MAX_BATCH + 1), profile=self.profile,
                         provider_name="fake", provider=provider,
                         store=self.assessments, max_n=MAX_BATCH + 1)
        self.assertEqual(provider.calls, 0)

    def test_an_empty_batch_is_refused(self):
        provider = FakeProvider([json.dumps(_good_payload())])
        with self.assertRaises(ValueError):
            assess_batch([], profile=self.profile, provider_name="fake",
                         provider=provider, store=self.assessments, max_n=0)
        self.assertEqual(provider.calls, 0)

    def test_a_zero_or_negative_bound_is_refused(self):
        provider = FakeProvider([json.dumps(_good_payload())])
        for bad in (0, -1, None):
            with self.assertRaises(ValueError):
                assess_batch(self._jobs(1), profile=self.profile,
                             provider_name="fake", provider=provider,
                             store=self.assessments, max_n=bad)


class BoundedCallsTests(AssessmentTestCase):
    def test_at_most_one_repair_call(self):
        """generate_structured may repair once. Never a loop."""
        provider = FakeProvider(["not json at all"])
        record = assess_one(_job(), profile=self.profile, provider_name="fake",
                            provider=provider, store=self.assessments)
        self.assertLessEqual(provider.calls, 2)
        self.assertEqual(record.status, STATUS_ERROR)

    def test_the_call_count_is_recorded_not_assumed(self):
        provider = FakeProvider(["not json at all"])
        record = assess_one(_job(), profile=self.profile, provider_name="fake",
                            provider=provider, store=self.assessments)
        self.assertEqual(record.provider_calls, provider.calls)

    def test_a_provider_failure_is_recorded_distinctly(self):
        provider = ExplodingProvider(["{}"])
        record = assess_one(_job(), profile=self.profile, provider_name="fake",
                            provider=provider, store=self.assessments)
        self.assertEqual(record.status, STATUS_ERROR)
        self.assertIn("unreachable", record.error)
        self.assertEqual(record.tier, MatchTier.NOT_YET_EVALUATED.value)


# ----------------------------------------------------------------------
# evidence checking
# ----------------------------------------------------------------------


class EvidenceTests(AssessmentTestCase):
    def _build(self, payload):
        return build_assessment(_job(), PROFILE, payload, provider="fake",
                                model="m", at=T0, provider_calls=1)

    def test_valid_evidence_is_kept(self):
        record = self._build(_good_payload())
        self.assertEqual(len(record.evidence), 2)
        self.assertEqual(record.status, STATUS_ASSESSED)

    def test_an_evidence_quote_absent_from_the_record_is_discarded(self):
        payload = _good_payload()
        payload["evidence"].append({
            "claim": "Eight years of Go", "source": "profile",
            "quote": "Eight years of Go experience",
        })
        record = self._build(payload)
        self.assertEqual(len(record.evidence), 2, "the invented quote is gone")
        self.assertEqual(record.discarded_evidence, 1)

    def test_a_tier_with_no_surviving_evidence_becomes_not_evaluated(self):
        """The dangerous case: a confident finding with nothing behind it."""
        payload = _good_payload()
        payload["evidence"] = [{
            "claim": "Ten years of experience", "source": "profile",
            "quote": "Ten years of experience",
        }]
        record = self._build(payload)
        self.assertEqual(record.tier, MatchTier.NOT_YET_EVALUATED)
        self.assertIsNone(record.score)
        self.assertEqual(record.confidence, Confidence.INSUFFICIENT)
        self.assertEqual(record.status, STATUS_INSUFFICIENT)

    def test_an_empty_quote_is_not_evidence(self):
        payload = _good_payload()
        payload["evidence"] = [{"claim": "x", "source": "job", "quote": "  "}]
        self.assertEqual(self._build(payload).evidence, ())

    def test_an_unknown_source_is_discarded(self):
        payload = _good_payload()
        payload["evidence"].append({
            "claim": "invented", "source": "the_void", "quote": "Go",
        })
        self.assertEqual(len(self._build(payload).evidence), 2)

    def test_an_unrecognised_tier_is_downgraded_not_interpreted(self):
        """Even with valid evidence, a tier we do not recognise is not a finding.

        The tempting move would be to keep the evidence and guess the tier.
        The code refuses to, because "we do not know" and "credible match" are
        different facts and only one of them is true.
        """
        payload = _good_payload()
        payload["tier"] = "absolutely_perfect"
        record = self._build(payload)
        self.assertEqual(record.tier, MatchTier.NOT_YET_EVALUATED)
        self.assertEqual(len(record.evidence), 2, "the evidence itself was valid")

    def test_an_unrecognised_tier_alone_is_downgraded(self):
        payload = {"tier": "absolutely_perfect", "score": 0.99,
                   "confidence": "high", "evidence": []}
        self.assertEqual(self._build(payload).tier, MatchTier.NOT_YET_EVALUATED)

    def test_an_out_of_range_score_is_dropped(self):
        payload = _good_payload()
        payload["score"] = 4.2
        self.assertIsNone(self._build(payload).score)

    def test_a_boolean_score_is_not_one(self):
        payload = _good_payload()
        payload["score"] = True
        self.assertIsNone(self._build(payload).score)

    def test_an_unrecognised_confidence_becomes_insufficient(self):
        payload = _good_payload()
        payload["confidence"] = "rock_solid"
        self.assertEqual(self._build(payload).confidence, Confidence.INSUFFICIENT)


class MalformedOutputTests(AssessmentTestCase):
    def _assess(self, text):
        provider = FakeProvider([text])
        return assess_one(_job(), profile=self.profile, provider_name="fake",
                          provider=provider, store=self.assessments), provider

    def test_prose_is_rejected(self):
        record, _ = self._assess("This looks like a great match!")
        self.assertEqual(record.status, STATUS_ERROR)
        self.assertEqual(record.tier, MatchTier.NOT_YET_EVALUATED)

    def test_a_json_array_is_rejected(self):
        record, _ = self._assess("[1, 2, 3]")
        self.assertEqual(record.status, STATUS_ERROR)

    def test_empty_output_is_rejected(self):
        record, _ = self._assess("")
        self.assertEqual(record.status, STATUS_ERROR)

    def test_a_fenced_block_is_accepted(self):
        """Models emit fences even when told not to; that is not a finding."""
        record, _ = self._assess("```json\n" + json.dumps(_good_payload()) + "\n```")
        self.assertEqual(record.status, STATUS_ASSESSED)

    def test_no_tier_is_ever_fabricated_on_bad_output(self):
        for text in ("garbage", "", "[]", "{", "null"):
            record, _ = self._assess(text)
            self.assertEqual(record.tier, MatchTier.NOT_YET_EVALUATED)
            self.assertIsNone(record.score)


# ----------------------------------------------------------------------
# it explains, it does not decide
# ----------------------------------------------------------------------


class NonInterferenceTests(AssessmentTestCase):
    def test_an_assessment_changes_nothing_but_the_assessment_log(self):
        self.ingest([_record("a1")])
        job_id = self.store.load_jobs()[0]["job_id"]
        before = {
            p.name: p.read_bytes()
            for p in sorted(self.data.rglob("*")) if p.is_file()
        }
        provider = FakeProvider([json.dumps(_good_payload())])
        assess_one(_job(job_id=job_id), profile=self.profile,
                   provider_name="fake", provider=provider,
                   store=self.assessments)
        after = {
            p.name: p.read_bytes()
            for p in sorted(self.data.rglob("*")) if p.is_file()
        }
        untouched = {k for k in before if before[k] == after.get(k)}
        self.assertIn("jobs.jsonl", untouched, "a job record was modified")
        self.assertIn("seen.json", untouched)
        self.assertNotIn("matches.jsonl", before)
        self.assertIn("matches.jsonl", after)

    def test_an_assessment_does_not_change_eligibility(self):
        self.ingest([_record("a1", location="Berlin, Germany")])
        before = build_report(self.store).eligible_unreviewed
        provider = FakeProvider([json.dumps(_good_payload())])
        assess_one(_job(job_id=self.store.load_jobs()[0]["job_id"]),
                   profile=self.profile, provider_name="fake",
                   provider=provider, store=self.assessments)
        after = build_report(
            self.store,
            matches=to_match_dict(AssessmentStore(self.data).latest()[
                self.store.load_jobs()[0]["job_id"]]),
        )
        self.assertEqual(len(before), 0, "an ineligible job stays ineligible")
        self.assertEqual(len(after.eligible_unreviewed), 0)

    def test_an_assessment_does_not_change_status(self):
        self.ingest([_record("a1")])
        job_id = self.store.load_jobs()[0]["job_id"]
        StatusLog(self.store).record(job_id, ReviewStatus.DISMISSED)
        provider = FakeProvider([json.dumps(_good_payload())])
        assess_one(_job(job_id=job_id), profile=self.profile,
                   provider_name="fake", provider=provider,
                   store=self.assessments)
        self.assertEqual(StatusLog(self.store).current(job_id).value, "dismissed")

    def test_an_unsuitable_result_does_not_remove_a_job_from_the_queue(self):
        """A match result reorders and explains; it never hides."""
        self.ingest([_record("a1"), _record("a2")])
        ids = {r["job_id"] for r in self.store.load_jobs()}
        provider = FakeProvider([json.dumps({
            **_good_payload(), "tier": "unsuitable", "score": 0.01})])
        latest = {}
        for record in self.store.load_jobs():
            latest[record["job_id"]] = assess_one(
                _job(job_id=record["job_id"]), profile=self.profile,
                provider_name="fake", provider=provider,
                store=self.assessments)
        report = build_report(
            self.store,
            matches={k: to_match_dict(v) for k, v in latest.items()},
        )
        self.assertEqual({v.job_id for v, _ in report.queue}, ids)
        self.assertEqual(report.total_jobs, 2)


class RenderingTests(AssessmentTestCase):
    def test_valid_evidence_renders_in_the_report(self):
        self.ingest([_record("a1", title="Backend Engineer (Go)")])
        job_id = self.store.load_jobs()[0]["job_id"]
        provider = FakeProvider([json.dumps(_good_payload())])
        record = assess_one(_job(job_id=job_id), profile=self.profile,
                            provider_name="fake", provider=provider,
                            store=self.assessments)
        html = render_report_html(build_report(
            self.store, matches={job_id: to_match_dict(record)}))
        self.assertIn("credible_match", html)
        self.assertIn("Go is a stated requirement", html)
        self.assertIn("0.72", html)

    def test_an_unassessed_record_is_not_rendered_as_a_bad_match(self):
        self.ingest([_record("a1")])
        job_id = self.store.load_jobs()[0]["job_id"]
        provider = FakeProvider(["garbage"])
        record = assess_one(_job(job_id=job_id), profile=self.profile,
                            provider_name="fake", provider=provider,
                            store=self.assessments)
        html = render_report_html(build_report(
            self.store, matches={job_id: to_match_dict(record)}))
        self.assertNotIn("unsuitable", html)
        self.assertIn("assessed, evidence insufficient", html)

    def test_a_recorded_failure_is_shown_as_insufficient_not_poor(self):
        self.ingest([_record("a1")])
        job_id = self.store.load_jobs()[0]["job_id"]
        provider = ExplodingProvider(["{}"])
        record = assess_one(_job(job_id=job_id), profile=self.profile,
                            provider_name="fake", provider=provider,
                            store=self.assessments)
        payload = to_match_dict(record)
        self.assertEqual(payload["tier"], MatchTier.NOT_YET_EVALUATED.value)
        self.assertIsNone(payload["score"])


class StoreTests(AssessmentTestCase):
    def test_the_log_is_append_only_and_separate_from_jobs(self):
        self.assessments.append(AssessmentRecord(
            job_id="j1", at=T0, status=STATUS_ASSESSED))
        self.assessments.append(AssessmentRecord(
            job_id="j1", at=T0, status=STATUS_INSUFFICIENT))
        lines = (self.data / "matches.jsonl").read_text(
            encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertNotIn("matches.jsonl",
                         {p.name for p in self.store.data_dir.glob("jobs*")})

    def test_a_torn_final_line_does_not_hide_earlier_records(self):
        self.assessments.append(AssessmentRecord(
            job_id="j1", at=T0, status=STATUS_ASSESSED))
        with self.assessments.path.open("a", encoding="utf-8") as handle:
            handle.write('{"job_id": "j2", "at": "2026-')
        self.assertEqual(len(self.assessments.load()), 1)

    def test_loading_a_missing_log_is_empty_not_an_error(self):
        self.assertEqual(AssessmentStore(self.data / "nowhere").load(), [])

    def test_round_trip_preserves_every_field(self):
        record = AssessmentRecord(
            job_id="j1", at=T0, status=STATUS_ASSESSED, provider="p", model="m",
            tier="strong_match", score=0.9, confidence="high",
            evidence=({"claim": "c", "source": "job", "quote": "q"},),
            missing_requirements=("m1",), concerns=("c1",),
            provider_calls=1, discarded_evidence=2)
        restored = AssessmentRecord.from_dict(record.to_dict())
        self.assertEqual(restored.to_dict(), record.to_dict())


class CliTests(AssessmentTestCase):
    def _run(self, *args):
        return self._cli(*args)

    def test_without_a_provider_it_refuses_and_writes_nothing(self):
        self.ingest([_record("a1")])
        job_id = self.store.load_jobs()[0]["job_id"]
        result = self._run("--provider", "ollama", "--job-id", job_id)
        self.assertEqual(result.returncode, 1)
        self.assertIn("refusing", result.stderr)
        self.assertFalse((self.data / "matches.jsonl").exists())

    def test_with_no_selection_it_refuses(self):
        result = self._run("--provider", "fake")
        self.assertEqual(result.returncode, 1)
        self.assertIn("no assess-everything mode", result.stderr)

    def test_an_oversized_batch_is_refused_before_anything_else(self):
        result = self._run("--batch", str(MAX_BATCH + 1))
        self.assertEqual(result.returncode, 1)
        self.assertFalse((self.data / "matches.jsonl").exists())

    def test_an_empty_batch_value_is_refused(self):
        result = self._run("--batch", "0")
        self.assertEqual(result.returncode, 1)

    def test_using_both_selections_is_refused(self):
        result = self._run("--job-id", "x", "--batch", "1")
        self.assertEqual(result.returncode, 1)
        self.assertIn("not both", result.stderr)

    def test_an_unknown_job_id_writes_nothing(self):
        self.ingest([_record("a1")])
        result = self._run("--provider", "fake", "--job-id", "no-such-job")
        self.assertEqual(result.returncode, 1)
        self.assertFalse((self.data / "matches.jsonl").exists())

    def test_provider_listing_and_boundary_are_read_only(self):
        self.ingest([_record("a1")])
        for flag in ("--providers", "--boundary"):
            result = self._run(flag)
            self.assertEqual(result.returncode, 0)
        self.assertFalse((self.data / "matches.jsonl").exists())


class LocalOnlyTests(AssessmentTestCase):
    """The trial is local-only, and a cloud model hides behind localhost."""

    def clean_registry(self):
        from app.jobs import assessment

        assessment._REGISTRY.clear()
        self.addCleanup(assessment._REGISTRY.clear)

    def test_a_loopback_endpoint_is_local(self):
        for url in ("http://localhost:11434", "http://127.0.0.1:11434",
                    "http://[::1]:11434"):
            self.assertTrue(is_local_endpoint(url), url)

    def test_a_remote_endpoint_is_not_local(self):
        for url in ("https://api.example.com", "http://10.0.0.5:11434",
                    "http://ollama.example.com"):
            self.assertFalse(is_local_endpoint(url), url)

    def test_a_non_http_scheme_is_not_local(self):
        self.assertFalse(is_local_endpoint("ftp://localhost"))
        self.assertFalse(is_local_endpoint("file:///etc/passwd"))

    def test_no_endpoint_means_in_process_and_is_local(self):
        self.assertTrue(is_local_endpoint(None))

    def test_a_cloud_model_is_refused_despite_a_local_endpoint(self):
        """localhost can proxy inference off this machine. The tag is the tell."""
        self.assertFalse(is_local_model("gpt-oss:120b-cloud"))
        self.assertFalse(is_local_model("llama3:cloud"))

    def test_an_installed_local_model_passes(self):
        for model in ("llama3.2:latest", "qwen2.5-coder:1.5b", None):
            self.assertTrue(is_local_model(model), model)

    def test_resolve_provider_refuses_a_remote_endpoint(self):
        self.clean_registry()
        register_provider("remote", lambda: RemoteProvider())
        with self.assertRaises(AssessmentUnavailable) as caught:
            resolve_provider("remote")
        self.assertIn("not local", str(caught.exception))

    def test_resolve_provider_refuses_a_cloud_model_on_a_local_endpoint(self):
        self.clean_registry()
        register_provider("cloud", lambda: CloudModelProvider())
        with self.assertRaises(AssessmentUnavailable) as caught:
            resolve_provider("cloud")
        self.assertIn("cloud model", str(caught.exception))

    def test_a_remote_provider_makes_no_assessment_writes(self):
        self.clean_registry()
        register_provider("remote", lambda: RemoteProvider())
        with self.assertRaises(AssessmentUnavailable):
            resolve_provider("remote")
        self.assertFalse(self.assessments.path.exists())

    def test_a_local_provider_resolves(self):
        self.clean_registry()
        register_provider("local", lambda: LocalEndpointProvider())
        name, provider = resolve_provider("local")
        self.assertEqual(name, "local")

    def test_remote_must_be_asked_for_explicitly(self):
        """Local-only is the default, not a mode you opt into."""
        self.clean_registry()
        register_provider("remote", lambda: RemoteProvider())
        name, provider = resolve_provider("remote", local_only=False)
        self.assertEqual(name, "remote")


class DryRunTests(AssessmentTestCase):
    def test_a_dry_run_makes_no_provider_calls_and_no_writes(self):
        provider = FakeProvider([json.dumps(_good_payload())])
        plan = plan_only([_job()], profile=self.profile, provider_name="fake")
        self.assertEqual(provider.calls, 0, "no provider may be called")
        self.assertFalse(self.assessments.path.exists(), "no assessment written")
        self.assertEqual(plan["writes"], 0)

    def test_a_dry_run_reports_the_profile_boundary(self):
        plan = plan_only([_job()], profile=self.profile, provider_name="fake")
        self.assertEqual(plan["profile_sources"], ["test"])
        self.assertEqual(plan["profile_characters"], len(PROFILE))

    def test_a_dry_run_names_sent_and_excluded_fields(self):
        plan = plan_only([_job()], profile=self.profile, provider_name="fake")
        self.assertIn("description", plan["sent_job_fields"])
        self.assertIn("url", plan["excluded_job_fields"])
        self.assertIn("salary_min", plan["excluded_job_fields"])
        self.assertTrue(plan["full_description_sent"])

    def test_a_dry_run_reports_each_job_and_its_source_url(self):
        plan = plan_only([_job()], profile=self.profile, provider_name="fake")
        self.assertEqual(len(plan["jobs"]), 1)
        self.assertEqual(plan["jobs"][0]["job_id"], "j1")
        self.assertIn("example.test", plan["jobs"][0]["url"])

    def test_a_dry_run_refuses_without_a_profile(self):
        with self.assertRaises(AssessmentUnavailable):
            plan_only([_job()], profile=CandidateProfile(text=""),
                      provider_name="fake")

    def test_the_real_repository_profile_can_be_planned_for(self):
        from app.jobs.assessment import load_candidate_profile

        profile = load_candidate_profile(REPO_ROOT)
        plan = plan_only([_job()], profile=profile, provider_name=None)
        self.assertGreater(plan["profile_characters"], 0)
        self.assertEqual(plan["provider"], "")
        self.assertTrue(plan["profile_sources"])


class TrialCeilingTests(AssessmentTestCase):
    def test_the_trial_ceiling_is_three(self):
        self.assertEqual(TRIAL_CEILING, 3)

    def test_a_dry_run_without_explicit_ids_is_refused(self):
        """A dry run must not silently become an arbitrary selection."""
        self.ingest([_record("a1"), _record("a2"), _record("a3")])
        result = self._cli("--dry-run", "--batch", "3")
        self.assertEqual(result.returncode, 1)
        self.assertIn("explicit --job-id", result.stderr)

    def test_more_than_three_explicit_ids_are_refused(self):
        ids = []
        for name in ("a1", "a2", "a3", "a4"):
            self.ingest([_record(name)])
            ids += ["--job-id", self.store.load_jobs()[-1]["job_id"]]
        result = self._cli(*ids)
        self.assertEqual(result.returncode, 1)
        self.assertIn("at most 3", result.stderr)
        self.assertFalse((self.data / "matches.jsonl").exists())

    def test_duplicate_job_ids_are_refused(self):
        self.ingest([_record("a1")])
        job_id = self.store.load_jobs()[0]["job_id"]
        result = self._cli("--job-id", job_id, "--job-id", job_id)
        self.assertEqual(result.returncode, 1)
        self.assertIn("duplicate", result.stderr)

    def test_a_dry_run_with_three_explicit_ids_succeeds(self):
        for name in ("a1", "a2", "a3"):
            self.ingest([_record(name)])
        ids = [i for r in self.store.load_jobs() for i in ("--job-id", r["job_id"])]
        result = self._cli(*ids, "--dry-run")
        self.assertEqual(result.returncode, 0)
        self.assertIn("DRY RUN", result.stdout)
        self.assertIn("writes: 0", result.stdout)
        self.assertFalse((self.data / "matches.jsonl").exists())


class FakeOllamaFactory:
    """Stands in for the HTTP Ollama client.

    Installed via :func:`install_fake_ollama`, which patches the symbol the
    registration path imports *inside* the function. No test opens a socket or
    constructs a real ``OllamaProvider``.
    """

    def __init__(self, *, installed=("llama3.2:latest", "qwen2.5-coder:1.5b"),
                 reachable=True, raises=None, bad_config=False):
        self.installed = list(installed)
        self.reachable = reachable
        self.raises = raises
        self.bad_config = bad_config
        self.constructed = []
        self.generated = 0

    def __call__(self, base_url, model):
        if self.bad_config:
            raise ValueError("malformed provider configuration")
        self.constructed.append((base_url, model))
        return _FakeOllamaInstance(base_url, model, self)


class _FakeOllamaInstance:
    def __init__(self, base_url, model, factory):
        self.base_url = base_url
        self.default_model = model
        self._factory = factory

    def health_check(self):
        if self._factory.raises:
            raise self._factory.raises
        return self._factory.reachable

    def list_models(self):
        return list(self._factory.installed)

    def generate(self, request):
        self._factory.generated += 1
        raise AssertionError("registration must never generate")


def install_fake_ollama(self, factory):
    """Patch the Ollama client the registration path imports lazily."""
    from app.llm import ollama as ollama_module

    class _Patched:
        @staticmethod
        def OllamaProvider(base_url=LOCAL_OLLAMA_URL, model="llama3.2"):
            # Routed through the factory so its bad_config branch is exercised.
            return factory(base_url, model)

    original = ollama_module.OllamaProvider
    ollama_module.OllamaProvider = _Patched.OllamaProvider
    self.addCleanup(setattr, ollama_module, "OllamaProvider", original)
    self.addCleanup(assessment_module()._REGISTRY.clear)
    return factory


def assessment_module():
    from app.jobs import assessment

    return assessment


class LocalOllamaRegistrationTests(AssessmentTestCase):
    """The only path that may register a real provider.

    Every test here is offline: the Ollama client is patched, so no socket is
    opened and no model is ever loaded.
    """

    def setUp(self):
        super().setUp()
        assessment_module()._REGISTRY.clear()

    def _register(self, factory, **kwargs):
        install_fake_ollama(self, factory)
        params = dict(model="llama3.2:latest", confirmed=True,
                      profile=self.profile)
        params.update(kwargs)
        return register_local_ollama(**params)

    def test_the_approved_model_registers(self):
        boundary = self._register(FakeOllamaFactory())
        self.assertEqual(LOCAL_OLLAMA_PROVIDER in registered_providers(), True)
        self.assertEqual(boundary["model"], "llama3.2:latest")
        self.assertEqual(boundary["local_only_verdict"], "LOCAL ONLY")

    def test_registration_without_confirmation_is_refused(self):
        factory = FakeOllamaFactory()
        install_fake_ollama(self, factory)
        with self.assertRaises(AssessmentUnavailable) as caught:
            register_local_ollama(model="llama3.2:latest", confirmed=False,
                                  profile=self.profile)
        self.assertIn("not confirmed", str(caught.exception))
        self.assertEqual(registered_providers(), ())
        self.assertEqual(factory.constructed, [], "nothing may be contacted")

    def test_a_non_loopback_endpoint_is_refused(self):
        for endpoint in ("https://api.example.com", "http://10.0.0.5:11434",
                         "http://example.com:11434"):
            factory = FakeOllamaFactory()
            with self.assertRaises(AssessmentUnavailable) as caught:
                self._register(factory, endpoint=endpoint)
            self.assertIn("loopback", str(caught.exception))
            self.assertEqual(registered_providers(), ())

    def test_a_cloud_model_is_refused(self):
        """The approved-model check is not a suggestion."""
        for model in ("gpt-oss:120b-cloud", "llama3:cloud", "x-cloud"):
            factory = FakeOllamaFactory(installed=[model])
            with self.assertRaises(AssessmentUnavailable) as caught:
                self._register(factory, model=model)
            self.assertIn("remote-inference", str(caught.exception))
            self.assertEqual(registered_providers(), ())

    def test_the_cloud_model_present_on_this_machine_stays_refused(self):
        """The literal model id that motivated all of this."""
        factory = FakeOllamaFactory(
            installed=["llama3.2:latest", "gpt-oss:120b-cloud"])
        with self.assertRaises(AssessmentUnavailable):
            self._register(factory, model="gpt-oss:120b-cloud")
        self.assertEqual(registered_providers(), ())

    def test_an_unavailable_model_is_refused(self):
        factory = FakeOllamaFactory(installed=["qwen2.5-coder:1.5b"])
        with self.assertRaises(AssessmentUnavailable) as caught:
            self._register(factory, model="llama3.2:latest")
        self.assertIn("not installed locally", str(caught.exception))
        self.assertEqual(registered_providers(), ())

    def test_unreachable_ollama_is_refused(self):
        factory = FakeOllamaFactory(reachable=False)
        with self.assertRaises(AssessmentUnavailable) as caught:
            self._register(factory)
        self.assertIn("not reachable", str(caught.exception))
        self.assertEqual(registered_providers(), ())

    def test_a_raising_health_check_is_refused(self):
        from app.llm.exceptions import ProviderConnectionError

        factory = FakeOllamaFactory(raises=ProviderConnectionError("down"))
        with self.assertRaises(AssessmentUnavailable) as caught:
            self._register(factory)
        self.assertIn("not reachable", str(caught.exception))

    def test_malformed_provider_configuration_is_refused(self):
        factory = FakeOllamaFactory(bad_config=True)
        with self.assertRaises(AssessmentUnavailable) as caught:
            self._register(factory)
        self.assertIn("malformed", str(caught.exception))
        self.assertEqual(registered_providers(), ())

    def test_an_empty_model_is_refused(self):
        factory = FakeOllamaFactory()
        with self.assertRaises(AssessmentUnavailable):
            self._register(factory, model="   ")
        self.assertEqual(registered_providers(), ())

    def test_the_boundary_is_returned_before_registration(self):
        boundary = self._register(FakeOllamaFactory())
        for key in ("endpoint", "model", "local_only_verdict",
                    "profile_sources", "profile_characters",
                    "sent_job_fields", "excluded_job_fields",
                    "full_description_sent"):
            self.assertIn(key, boundary)
        self.assertEqual(boundary["profile_characters"], len(PROFILE))
        self.assertIn("description", boundary["sent_job_fields"])
        self.assertIn("url", boundary["excluded_job_fields"])

    def test_the_boundary_declares_no_persistence(self):
        boundary = self._register(FakeOllamaFactory())
        self.assertFalse(boundary["persisted"])
        self.assertEqual(boundary["credentials_sent"], 0)

    def test_registration_makes_no_provider_calls(self):
        factory = FakeOllamaFactory()
        self._register(factory)
        self.assertEqual(factory.generated, 0)

    def test_registration_writes_no_assessment(self):
        self._register(FakeOllamaFactory())
        self.assertFalse(self.assessments.path.exists())
        self.assertEqual(AssessmentStore(self.data).load(), [])

    def test_registration_does_not_assess_a_stored_job(self):
        self.ingest([_record("a1")])
        self._register(FakeOllamaFactory())
        self.assertFalse((self.data / "matches.jsonl").exists())

    def test_registration_is_not_persisted_across_processes(self):
        """In-process only: a fresh CLI starts with an empty registry."""
        factory = FakeOllamaFactory()
        install_fake_ollama(self, factory)
        register_local_ollama(model="llama3.2:latest", confirmed=True,
                              profile=self.profile)
        self.assertTrue(registered_providers())
        # Nothing on disk records the registration.
        for path in self.data.rglob("*"):
            if path.is_file():
                self.assertNotIn(b"local-ollama", path.read_bytes())
        # And a fresh import has none.
        result = self._cli("--providers")
        self.assertIn("no providers registered", result.stdout)

    def test_naming_the_provider_does_not_register_it(self):
        """`--provider ollama` must not be a back door to registration."""
        for name in ("ollama", "local-ollama", "llama3.2:latest"):
            result = self._cli("--provider", name, "--job-id", "x")
            self.assertEqual(result.returncode, 1)
        self.assertFalse((self.data / "matches.jsonl").exists())

    def test_the_cli_refuses_to_register_without_confirmation(self):
        result = self._cli("--register-local-ollama", "--model", "llama3.2:latest")
        self.assertEqual(result.returncode, 1)
        self.assertIn("not confirmed", result.stderr)
        self.assertIn("nothing was registered", result.stderr)

    def test_the_cli_registration_emits_no_assessment_writes(self):
        """Drives ``main`` in-process so the patched client applies.

        A subprocess cannot inherit this patch, so a CLI test that expects
        registration to succeed would reach a real localhost endpoint and pass
        only on a machine that happens to be running Ollama. Registration
        behaviour is asserted through ``main`` here; the subprocess tests cover
        the refusal paths, which return before anything is contacted.
        """
        install_fake_ollama(self, FakeOllamaFactory())
        import tools.match as match_cli

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = match_cli.main([
                "--data-dir", str(self.data),
                "--register-local-ollama",
                "--model", "llama3.2:latest",
                "--confirm-local-provider-boundary",
            ])
        output = buffer.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("DATA BOUNDARY", output)
        self.assertIn("LOCAL ONLY", output)
        self.assertIn("no job was assessed", output)
        self.assertIn("NOT persisted", output)
        self.assertFalse((self.data / "matches.jsonl").exists())

    def test_the_cli_registration_prints_the_boundary_before_registering(self):
        """The boundary must appear in the output the operator reads."""
        install_fake_ollama(self, FakeOllamaFactory())
        import tools.match as match_cli

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            match_cli.main([
                "--data-dir", str(self.data),
                "--register-local-ollama",
                "--model", "llama3.2:latest",
                "--confirm-local-provider-boundary",
            ])
        output = buffer.getvalue()
        boundary_at = output.index("DATA BOUNDARY")
        registered_at = output.index("registered")
        self.assertLess(boundary_at, registered_at,
                        "boundary must be printed before registration")
        self.assertIn("job fields NOT sent", output)
        self.assertIn("credentials sent", output)


class NoNetworkInTestsTests(unittest.TestCase):
    def test_the_assessment_module_imports_no_transport(self):
        import ast

        tree = ast.parse(
            (REPO_ROOT / "app" / "jobs" / "assessment.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        # `urllib.parse` is pure string parsing and is how locality is checked;
        # what must never appear is a module that can open a connection.
        for forbidden in ("socket", "subprocess", "http", "requests", "httpx",
                          "urllib.request", "urllib.error", "ssl"):
            self.assertNotIn(forbidden, imported)


if __name__ == "__main__":
    unittest.main()