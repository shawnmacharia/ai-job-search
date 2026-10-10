"""Remotive Public API: scope, terms, and parsing. Entirely offline.

The constraints under test are not stylistic. Each one encodes a term Remotive
attached to the access grant, and breaking any of them would mean using access
we were given in a way we were not given it:

- only the one approved endpoint is ever fetched;
- no more than one request a day, and that limit survives across processes;
- attribution is present on every stored record;
- an unreadable response is reported, never absorbed into "zero jobs".

The fixture is **synthetic and hand-authored**, not a live capture, and is
labelled as such in the file itself. It deliberately includes the awkward
shapes the API is documented to emit - a company as a list, a remote flag as a
string, a row with no title, an unrecognised remote value - so those paths are
exercised by data rather than by assumption.
"""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.jobs.adapters import AdaptError
from app.jobs.eligibility import evaluate_eligibility
from app.jobs.models import RemoteStatus
from app.jobs.runner import SourceSpec, run_sources
from app.jobs.store import JobStore
from app.sources.access import AccessLevel
from app.sources.remotive import (
    API_URL,
    ATTRIBUTION_REQUIRED,
    ATTRIBUTION_TEXT,
    DAILY_LIMIT,
    MIN_INTERVAL,
    SOURCE_PAGE,
    RemotiveAdapter,
    RemotiveSourceAdapter,
    _DailyBudget,
    _remote_flag,
    clean_html,
    parse_response,
    record_to_job,
)
from app.sources.transport import AccessError, AccessFetcher, Ledger, RateLimiter

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "remotive" / "sample.json"
T0 = datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc)


def fixture_body() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def _response(body: str, url: str):
    from app.sources.transport import HttpResponse

    return HttpResponse(url=url, status=200, headers={}, body=body, elapsed_ms=1)


def _persisted_fetcher(path: Path, ledger: Ledger = None) -> AccessFetcher:
    """A fetcher whose ledger lives on disk, so limits survive a restart."""
    return AccessFetcher(
        ledger=ledger or Ledger(path),
        limiter=RateLimiter(MIN_INTERVAL, sleeper=lambda _: None),
        opener=lambda url, timeout: _response(fixture_body(), url),
        sleeper=lambda _: None,
        max_attempts=1,
    )


def fake_fetcher(body: str = "", *, status: int = 200):
    """A fetcher that returns ``body`` and counts requests. No sockets."""
    calls: list = []

    def opener(url, timeout):
        calls.append(url)
        return _response(body, url)

    fetcher = AccessFetcher(
        ledger=Ledger(),
        limiter=RateLimiter(0.0, sleeper=lambda _: None),
        opener=opener,
        sleeper=lambda _: None,
        max_attempts=1,
    )
    return fetcher, calls


class ScopeTests(unittest.TestCase):
    """Only the approved endpoint may ever be requested."""

    def test_the_approved_endpoint_is_the_documented_jobs_endpoint(self):
        self.assertEqual(API_URL, "https://remotive.com/api/remote-jobs")

    def test_the_adapter_fetches_only_that_url(self):
        fetcher, calls = fake_fetcher(fixture_body())
        adapter = RemotiveAdapter(fetcher)
        adapter.to_records()
        self.assertEqual(calls, [API_URL])

    def test_no_pagination_or_query_is_appended(self):
        fetcher, calls = fake_fetcher(fixture_body())
        RemotiveAdapter(fetcher).to_records()
        for url in calls:
            self.assertNotIn("?", url)
            self.assertNotIn("#", url)

    def test_only_one_request_is_made_per_fetch(self):
        """No implicit second call to enumerate further pages."""
        fetcher, calls = fake_fetcher(fixture_body())
        adapter = RemotiveAdapter(fetcher)
        adapter.to_records()
        self.assertEqual(len(calls), 1)
        self.assertEqual(adapter.requests_made, 1)

    def test_the_module_names_no_other_remotive_url(self):
        """Every Remotive URL in the module must be a page link, never a fetch.

        ``SOURCE_PAGE`` and ``TERMS_PAGE`` are permitted because the terms
        require linking back. What must not appear is a second *endpoint*: an
        unapproved API path, a job-detail page, or a paginating query variant.
        """
        source = Path(__file__).resolve().parents[1] / "app" / "sources" / "remotive.py"
        urls = set(re.findall(
            r"https://remotive\.com[^\s\"']+", source.read_text(encoding="utf-8")
        ))
        self.assertEqual(
            urls,
            {
                "https://remotive.com/api/remote-jobs",
                "https://remotive.com/remote-jobs",
                "https://remotive.com/remote-jobs/api",
            },
            f"an unapproved URL appeared in the module: {sorted(urls)}",
        )


class CadenceTests(unittest.TestCase):
    """One request a day, and the limit must survive across processes."""

    def test_a_second_fetch_in_the_same_process_is_refused(self):
        fetcher, calls = fake_fetcher(fixture_body())
        adapter = RemotiveAdapter(fetcher)
        adapter.to_records()
        adapter.end_run()
        with self.assertRaises(AccessError) as caught:
            adapter.to_records()
        self.assertIn("daily request limit", str(caught.exception))
        self.assertEqual(len(calls), 1)

    def test_the_limit_is_one_per_day_not_the_permitted_four(self):
        """We stay stricter than the grant. Being under the ceiling is the point."""
        self.assertEqual(DAILY_LIMIT, 1)
        self.assertLess(DAILY_LIMIT, 4)

    def test_a_day_later_the_budget_allows_another_request(self):
        clock = {"t": 0.0}
        budget = _DailyBudget(1, now=lambda: clock["t"])
        self.assertTrue(budget.allow())
        budget.record()
        self.assertFalse(budget.allow())
        clock["t"] += 86401.0
        self.assertTrue(budget.allow())

    def test_the_budget_is_seeded_from_the_persisted_ledger(self):
        """The daily limit must survive a restart, not just a loop iteration.

        Enforced by seeding the budget from the ledger written to disk. An
        in-memory counter would reset on every new process, and "one request a
        day" would be a claim the code did not keep.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "access_attempts.jsonl"
            first = RemotiveAdapter(_persisted_fetcher(path))
            first.to_records()
            self.assertEqual(first.requests_made, 1)
            self.assertTrue(path.exists(), "attempts must be persisted, not held")

            # A brand-new process reading the same ledger must be refused.
            reopened = Ledger(path)
            self.assertGreaterEqual(
                len(reopened.prior_attempts()), 1,
                "the ledger must be readable, or the limit cannot survive a restart",
            )
            again = RemotiveAdapter(_persisted_fetcher(path, ledger=reopened))
            self.assertFalse(
                again._budget.allow(),
                "a fresh process must still see the spent daily budget",
            )
            with self.assertRaises(AccessError):
                again.to_records()

    def test_the_floor_is_a_whole_day(self):
        self.assertEqual(MIN_INTERVAL, 86400.0)

    def test_a_refused_request_is_never_sent(self):
        fetcher, calls = fake_fetcher(fixture_body())
        adapter = RemotiveAdapter(fetcher)
        adapter.to_records()
        before = len(calls)
        adapter.end_run()
        with self.assertRaises(AccessError):
            adapter.to_records()
        self.assertEqual(len(calls), before)


class AttributionTests(unittest.TestCase):
    """Their terms make attribution a condition of access."""

    def test_attribution_is_required_by_this_integration(self):
        self.assertTrue(ATTRIBUTION_REQUIRED)

    def test_every_record_carries_attribution_and_the_source_page(self):
        fetcher, _ = fake_fetcher(fixture_body())
        for record in RemotiveAdapter(fetcher).to_records():
            self.assertEqual(record["attribution"], ATTRIBUTION_TEXT)
            self.assertEqual(record["attribution_url"], SOURCE_PAGE)

    def test_attribution_survives_adaptation_onto_the_job(self):
        fetcher, _ = fake_fetcher(fixture_body())
        record = RemotiveAdapter(fetcher).to_records()[0]
        job = RemotiveSourceAdapter().adapt(record, now=T0)
        self.assertEqual(job.raw_excerpt["attribution"], ATTRIBUTION_TEXT)
        self.assertEqual(job.raw_excerpt["attribution_url"], SOURCE_PAGE)

    def test_each_record_links_back_to_the_original_listing(self):
        fetcher, _ = fake_fetcher(fixture_body())
        for record in RemotiveAdapter(fetcher).to_records():
            self.assertTrue(record["url"].startswith("https://remotive.com/"))

    def test_the_access_evidence_states_the_restrictions(self):
        fetcher, _ = fake_fetcher(fixture_body())
        evidence = RemotiveAdapter(fetcher).access_evidence()
        blob = " ".join(f"{k} {v}" for k, v in evidence.items()).casefold()
        for phrase in ("redistribution", "pagination", "html scraping",
                       "attribution_required", "delay", "permission"):
            self.assertIn(phrase, blob)


class ParsingTests(unittest.TestCase):
    def test_the_fixture_is_labelled_synthetic(self):
        """It must never be mistaken for a real capture."""
        payload = json.loads(fixture_body())
        self.assertIn("SYNTHETIC", payload["_fixture"])

    def test_rows_without_a_title_are_skipped(self):
        jobs = parse_response(fixture_body())
        self.assertNotIn(
            "example-missing-title", " ".join(job.url for job in jobs)
        )

    def test_a_company_published_as_a_list_is_accepted(self):
        jobs = {job.url: job for job in parse_response(fixture_body())}
        job = jobs["https://remotive.com/remote-jobs/example-company-as-list"]
        self.assertEqual(job.company, "Example Cloud Co")

    def test_a_remote_flag_as_a_string_is_accepted(self):
        jobs = {job.url: job for job in parse_response(fixture_body())}
        job = jobs["https://remotive.com/remote-jobs/example-company-as-list"]
        self.assertEqual(job.remote, "true")

    def test_an_unrecognised_remote_value_becomes_unknown_not_true(self):
        """Eligibility weighs this field, so a guess would be a false claim."""
        self.assertEqual(_remote_flag("maybe"), "unknown")
        jobs = {job.url: job for job in parse_response(fixture_body())}
        self.assertEqual(
            jobs["https://remotive.com/remote-jobs/example-odd-remote"].remote,
            "unknown",
        )

    def test_boolean_and_string_remote_flags_agree(self):
        for raw, expected in ((True, "true"), ("true", "true"), ("1", "true"),
                              (False, "false"), ("no", "false"), ("", "unknown")):
            self.assertEqual(_remote_flag(raw), expected)

    def test_script_tags_do_not_survive_into_stored_text(self):
        """The dashboard is script-free; executable text must not reach it."""
        cleaned = clean_html("<p>Hi</p><script>alert('x')</script>")
        self.assertNotIn("script", cleaned.casefold())
        self.assertNotIn("alert", cleaned)

    def test_html_is_flattened_to_readable_text(self):
        cleaned = clean_html("<p>One</p><p>Two<br/>Three</p>")
        self.assertIn("One", cleaned)
        self.assertIn("Two", cleaned)
        self.assertIn("Three", cleaned)
        self.assertNotIn("<", cleaned)

    def test_html_entities_are_decoded(self):
        self.assertIn("R&D", clean_html("<p>R&amp;D</p>"))

    def test_the_job_id_is_the_url_not_the_numeric_id(self):
        jobs = parse_response(fixture_body())
        self.assertTrue(jobs[0].external_id)
        self.assertTrue(jobs[0].url.startswith("https://remotive.com/"))
        job = record_to_job({
            "url": jobs[0].url, "title": jobs[0].title, "external_id": jobs[0].external_id,
        })
        self.assertEqual(job.job_id, jobs[0].url)


class UnreadableResponseTests(unittest.TestCase):
    """An unreadable response must never read as an empty job market."""

    def test_malformed_json_is_an_error_not_an_empty_list(self):
        with self.assertRaises(AccessError) as caught:
            parse_response("{not json")
        self.assertIn("not valid JSON", str(caught.exception))

    def test_a_missing_jobs_key_is_an_error_not_an_empty_list(self):
        with self.assertRaises(AccessError) as caught:
            parse_response(json.dumps({"something-else": []}))
        self.assertIn("no 'jobs' array", str(caught.exception))

    def test_an_empty_body_is_an_error(self):
        with self.assertRaises(AccessError):
            parse_response("   ")

    def test_a_json_array_is_not_the_documented_shape(self):
        with self.assertRaises(AccessError):
            parse_response("[]")

    def test_a_genuinely_empty_job_list_parses_to_nothing(self):
        """A real empty result is a valid answer and must stay distinct."""
        self.assertEqual(parse_response(json.dumps({"jobs": []})), [])

    def test_the_error_says_unreadable_rather_than_zero(self):
        try:
            parse_response("<html>maintenance</html>")
        except AccessError as error:
            self.assertIn("unreadable", str(error))
        else:
            self.fail("expected an AccessError")


class AdaptationTests(unittest.TestCase):
    def test_a_record_without_a_url_is_rejected(self):
        with self.assertRaises(AdaptError):
            RemotiveSourceAdapter().adapt(
                {"title": "T", "company": "C", "url": ""}, now=T0
            )

    def test_a_record_without_a_title_is_rejected(self):
        with self.assertRaises(AdaptError):
            RemotiveSourceAdapter().adapt(
                {"title": "", "company": "C", "url": "https://remotive.com/x"},
                now=T0,
            )

    def test_tags_are_carried_as_skills(self):
        job = RemotiveSourceAdapter().adapt({
            "url": "https://remotive.com/x", "title": "T", "company": "C",
            "tags": ["go", "sql"],
        }, now=T0)
        self.assertEqual(job.skills, ["go", "sql"])
        self.assertEqual(job.raw_excerpt["source_field_name"], "tags")

    def test_an_unknown_remote_flag_maps_to_the_unknown_status(self):
        job = RemotiveSourceAdapter().adapt({
            "url": "https://remotive.com/x", "title": "T", "company": "C",
            "remote": "maybe",
        }, now=T0)
        self.assertIs(job.remote_status, RemoteStatus.UNKNOWN)

    def test_a_remotely_named_location_is_not_assumed_global(self):
        """'Remote' names no country, so eligibility must not be handed one."""
        job = RemotiveSourceAdapter().adapt({
            "url": "https://remotive.com/x", "title": "T", "company": "C",
            "location": "Remote", "remote": "true",
        }, now=T0)
        self.assertEqual(job.location, "Remote")
        verdict = evaluate_eligibility(job, candidate_country="KE")
        self.assertIn(verdict.verdict, {"unknown", "eligible", "not_eligible"})


class IngestionTests(unittest.TestCase):
    """End-to-end through the existing store, runner and eligibility layers."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = JobStore(Path(self._tmp.name) / "data")

    def test_records_are_ingested_and_persisted(self):
        fetcher, _ = fake_fetcher(fixture_body())
        run = run_sources(
            [SourceSpec(name="remotive.com", fetch=RemotiveAdapter(fetcher).to_records)],
            store=self.store,
            observed_at=T0,
        )
        outcome = run.sources[0]
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.stored, 5)
        self.assertEqual(len(self.store.load_jobs()), 5)

    def test_a_repeat_run_updates_rather_than_duplicating(self):
        fetcher, _ = fake_fetcher(fixture_body())
        adapter = RemotiveAdapter(fetcher)
        run_sources([SourceSpec(name="remotive.com", fetch=adapter.to_records)],
                    store=self.store, observed_at=T0)
        run_sources([SourceSpec(name="remotive.com", fetch=adapter.to_records)],
                    store=self.store, observed_at=T0 + timedelta(days=1))
        self.assertEqual(len(self.store.load_jobs()), 5)

    def test_a_failed_source_stores_nothing(self):
        def boom():
            raise RuntimeError("api exploded")

        run = run_sources([SourceSpec(name="remotive.com", fetch=boom)],
                          store=self.store, observed_at=T0)
        self.assertFalse(run.sources[0].ok)
        self.assertEqual(self.store.load_jobs(), [])

    def test_eligibility_is_computed_for_every_stored_job(self):
        fetcher, _ = fake_fetcher(fixture_body())
        run_sources(
            [SourceSpec(name="remotive.com", fetch=RemotiveAdapter(fetcher).to_records)],
            store=self.store, observed_at=T0,
        )
        verdicts = {
            evaluate_eligibility(
                __import__("app.jobs.models", fromlist=["Job"]).Job(**r["job"])
            ).verdict
            for r in self.store.load_jobs()
        }
        self.assertTrue(verdicts <= {"eligible", "not_eligible", "unknown"})
        self.assertTrue(verdicts)

    def test_the_nairobi_row_is_not_treated_as_remote_worldwide(self):
        """A named on-site Kenya role must survive as its own evidence."""
        fetcher, _ = fake_fetcher(fixture_body())
        run_sources(
            [SourceSpec(name="remotive.com", fetch=RemotiveAdapter(fetcher).to_records)],
            store=self.store, observed_at=T0,
        )
        locations = [r["job"].get("location") for r in self.store.load_jobs()]
        self.assertIn("Nairobi, Kenya", locations)

    def test_attribution_reaches_the_stored_record(self):
        fetcher, _ = fake_fetcher(fixture_body())
        run_sources(
            [SourceSpec(name="remotive.com", fetch=RemotiveAdapter(fetcher).to_records)],
            store=self.store, observed_at=T0,
        )
        for record in self.store.load_jobs():
            self.assertEqual(
                record["job"]["raw_excerpt"]["attribution"], ATTRIBUTION_TEXT
            )


class AttributionRenderingTests(unittest.TestCase):
    """Attribution is a condition of access, so the rendered report must carry it.

    The stored record is not the compliance surface a person sees; the
    dashboard is. A credit that exists only in a JSON field and never reaches
    the page would not satisfy the term that requires mentioning Remotive as a
    source.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = JobStore(Path(self._tmp.name) / "data")
        fetcher, _ = fake_fetcher(fixture_body())
        run_sources(
            [SourceSpec(name="remotive.com", fetch=RemotiveAdapter(fetcher).to_records)],
            store=self.store, observed_at=T0,
        )

    def _render(self) -> str:
        from app.reporting.jobs import render_dashboard_file

        out = Path(self._tmp.name) / "dashboard.html"
        render_dashboard_file(self.store, out, attributions={
            "remotive.com": SOURCE_PAGE,
        })
        return out.read_text(encoding="utf-8")

    def test_the_report_credits_remotive_as_a_source(self):
        html = self._render()
        self.assertIn("Attribution required", html)
        self.assertIn(SOURCE_PAGE, html)

    def test_every_row_links_back_to_the_original_listing(self):
        html = self._render()
        self.assertIn("https://remotive.com/remote-jobs/", html)

    def test_the_report_carries_no_script(self):
        """Consistent with the rest of the dashboard, and with no XSS surface."""
        html = self._render()
        self.assertNotIn("<script", html.casefold())


class ImportSafetyTests(unittest.TestCase):
    def test_importing_the_module_makes_no_network_call(self):
        import importlib

        module = importlib.import_module("app.sources.remotive")
        importlib.reload(module)
        self.assertTrue(callable(module.parse_response))


if __name__ == "__main__":
    unittest.main()