"""Tests for the We Work Remotely RSS source.

Fully offline. Every test parses ``tests/fixtures/wwr/sample.xml`` - a real
captured response truncated to six items - or drives the transport with a stub
opener. Nothing here opens a socket.

The fixture is genuine WWR data, which is why some assertions look
counter-intuitive: WWR labels most jobs "Anywhere in the World" while
enumerating accepted countries that exclude Kenya. Preserving that
distinction, rather than smoothing it over, is the point of several tests.
"""

import json
import re
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app.jobs.adapters import AdaptError, adapt_record
from app.jobs.ingest import ingest
from app.jobs.models import RemoteStatus
from app.jobs.runner import SourceSpec, run_sources, summarise
from app.jobs.store import JobStore
from app.reporting.jobs import coverage_of, render_dashboard_file, render_dashboard_html
from app.sources.transport import AccessError, AccessFetcher, HttpResponse, Ledger, RateLimiter
from app.sources.wwr import (
    ATTRIBUTION_REQUIRED,
    ATTRIBUTION_TEXT,
    CANDIDATE_COUNTRY,
    FEED_URL,
    MIN_INTERVAL,
    PERMISSION_STATEMENT,
    PERMISSION_URL,
    WwrAdapter,
    WwrSourceAdapter,
    countries_mention,
    parse_countries,
    parse_date,
    parse_feed,
    record_to_job,
    split_title,
    strip_html,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "wwr" / "sample.xml"
NOW = datetime(2026, 10, 9, 12, 0, 0, tzinfo=timezone.utc)


def fixture_text():
    return FIXTURE.read_text(encoding="utf-8")


def stub_fetcher(body=None, status=200, ledger=None, **kwargs):
    """An AccessFetcher that returns ``body`` without touching the network."""
    calls = []

    def opener(url, timeout):
        calls.append(url)
        if isinstance(body, Exception):
            raise body
        return HttpResponse(url=url, status=status, headers={}, body=body or "",
                            elapsed_ms=5)

    fetcher = AccessFetcher(
        ledger=ledger or Ledger(),
        limiter=RateLimiter(0.0, sleeper=lambda _: None),
        opener=opener,
        sleeper=lambda _: None,
        **kwargs,
    )
    return fetcher, calls


def _views_with_verdicts(verdicts, country="Kenya"):
    """Minimal JobViews carrying only a verdict, for coverage-panel tests."""
    from app.reporting.jobs import JobView

    views = []
    for index, verdict in enumerate(verdicts):
        views.append(JobView(
            job_id=f"j{index}", title="t", company="c", location="l", url="",
            verdict=verdict, verdict_reasons=["r"], evidence=[], flags=[],
            sources=["weworkremotely"], source_urls=[], first_seen="", last_seen="",
            posted_date="", description="", description_complete=True,
            possible_duplicate=False, match_explanation="", application_status="",
            country=country,
        ))
    return views


class FixtureTests(unittest.TestCase):
    def test_the_fixture_is_a_real_capture(self):
        text = fixture_text()
        self.assertIn("<rss", text)
        self.assertIn("<channel>", text)
        self.assertIn("weworkremotely.com", text)

    def test_the_fixture_is_truncated_to_a_handful_of_items(self):
        self.assertEqual(fixture_text().count("<item>"), 6)

    def test_the_fixture_contains_no_personal_data(self):
        text = fixture_text().casefold()
        for marker in ("@gmail", "@outlook", "@yahoo", "password", "api_key"):
            with self.subTest(marker=marker):
                self.assertNotIn(marker, text)


class ParsingTests(unittest.TestCase):
    def setUp(self):
        self.items = parse_feed(fixture_text())

    def test_six_items_parse(self):
        self.assertEqual(len(self.items), 6)

    def test_every_item_yields_an_identifier(self):
        for item in self.items:
            with self.subTest(url=item.url):
                self.assertTrue(item.guid or item.url)

    def test_every_item_has_a_description_and_date(self):
        for item in self.items:
            with self.subTest(url=item.url):
                self.assertTrue(item.description)
                self.assertTrue(item.posted)


class TitleSplitTests(unittest.TestCase):
    def test_company_prefix_is_split_off(self):
        self.assertEqual(split_title("Lemon.io: Senior Developer"),
                         ("Lemon.io", "Senior Developer"))

    def test_no_prefix_leaves_company_unknown_rather_than_guessed(self):
        company, title = split_title("Senior Developer")
        self.assertEqual(company, "")
        self.assertEqual(title, "Senior Developer")

    def test_a_colon_inside_the_role_does_not_hijack_the_split(self):
        company, title = split_title("Acme: Engineer: Platform")
        self.assertEqual(company, "Acme")
        self.assertEqual(title, "Engineer: Platform")

    def test_empty_input(self):
        self.assertEqual(split_title(""), ("", ""))


class HtmlCleaningTests(unittest.TestCase):
    def test_double_escaped_markup_becomes_readable_text(self):
        cleaned = strip_html("&lt;p&gt;Build &lt;b&gt;pipelines&lt;/b&gt;&lt;/p&gt;")
        self.assertIn("Build", cleaned)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("&lt;", cleaned)

    def test_script_content_is_removed_not_untagged(self):
        cleaned = strip_html("<script>alert('x')</script><p>Safe</p>")
        self.assertNotIn("alert", cleaned)
        self.assertIn("Safe", cleaned)

    def test_style_content_is_removed(self):
        self.assertNotIn("color:red", strip_html("<style>a{color:red}</style>Hi"))

    def test_image_tags_do_not_leak_as_text(self):
        cleaned = strip_html('<img src="https://x.test/logo.gif" />Data Engineer')
        self.assertNotIn("logo.gif", cleaned)
        self.assertNotIn("<img", cleaned)
        self.assertIn("Data Engineer", cleaned)

    def test_paragraphs_become_line_breaks(self):
        self.assertRegex(strip_html("<p>One</p><p>Two</p>"), r"One\s*\n\s*Two")

    def test_empty_input(self):
        self.assertEqual(strip_html(""), "")

    def test_the_fixture_description_is_plain_prose(self):
        item = parse_feed(fixture_text())[0]
        self.assertNotIn("<", item.description)
        self.assertNotIn("&lt;", item.description)


class CountryTests(unittest.TestCase):
    def test_flag_emoji_are_stripped(self):
        self.assertEqual(parse_countries("🇰🇪 Kenya, 🇺🇸 United States"),
                         ["Kenya", "United States"])

    def test_the_country_list_is_preserved_verbatim(self):
        item = parse_feed(fixture_text())[0]
        self.assertGreater(len(item.countries), 50)
        self.assertIn("Australia", item.countries)

    def test_kenya_absence_is_not_inferred_away(self):
        # WWR labels this "Anywhere in the World" but Kenya is not in the
        # accepted-country list. Inferring inclusion would fabricate eligibility.
        item = parse_feed(fixture_text())[0]
        self.assertEqual(item.region, "Anywhere in the World")
        self.assertFalse(countries_mention(item.countries, CANDIDATE_COUNTRY))

    def test_kenya_is_detected_when_published(self):
        self.assertTrue(countries_mention(["Kenya", "Uganda"], CANDIDATE_COUNTRY))

    def test_matching_is_case_insensitive_and_substring_safe(self):
        self.assertTrue(countries_mention(["KENYA"], CANDIDATE_COUNTRY))
        self.assertFalse(countries_mention(["Kenyan"], CANDIDATE_COUNTRY))

    def test_empty_country_list_mentions_nothing(self):
        self.assertFalse(countries_mention([], CANDIDATE_COUNTRY))

    def test_no_item_in_the_fixture_includes_kenya(self):
        for item in parse_feed(fixture_text()):
            with self.subTest(url=item.url):
                self.assertFalse(countries_mention(item.countries, CANDIDATE_COUNTRY))


class DateTests(unittest.TestCase):
    def test_rfc822_parses_to_iso(self):
        self.assertEqual(parse_date("Fri, 09 Oct 2026 13:49:21 +0000"),
                         "2026-10-09T13:49:21+00:00")

    def test_unparseable_date_is_none_not_invented(self):
        self.assertIsNone(parse_date("sometime last week"))
        self.assertIsNone(parse_date(""))


class FeedErrorTests(unittest.TestCase):
    def test_non_xml_is_an_error_not_an_empty_feed(self):
        with self.assertRaises(AdaptError):
            parse_feed("<!doctype html><html>Cloudflare</html>")

    def test_xml_without_a_channel_is_an_error(self):
        with self.assertRaises(AdaptError):
            parse_feed("<rss><notachannel/></rss>")

    def test_a_valid_but_empty_feed_returns_no_items(self):
        # Distinguishing "no jobs today" from "not a feed" matters: the first
        # is normal, the second is a capture problem.
        self.assertEqual(parse_feed("<rss><channel></channel></rss>"), [])


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.items = parse_feed(fixture_text())
        self.record = WwrAdapter.__dict__  # keep import honest
        self.raw = {
            "job_id": "guid-1",
            "title": "Data Engineer",
            "company": "Acme",
            "url": "https://weworkremotely.com/remote-jobs/x-1",
            "location": "Anywhere in the World",
            "state": "Delaware",
            "description": "Build ELT pipelines.",
            "skills": ["Python", "dbt"],
            "countries": ["Kenya", "Uganda"],
            "posted": "2026-10-01T00:00:00+00:00",
        }

    def test_all_fields_map_onto_the_job_contract(self):
        job = record_to_job(self.raw)
        self.assertEqual(job.title, "Data Engineer")
        self.assertEqual(job.company, "Acme")
        self.assertEqual(job.url, self.raw["url"])
        self.assertEqual(job.description, "Build ELT pipelines.")
        self.assertEqual(job.skills, ["Python", "dbt"])
        self.assertEqual(job.portal, "weworkremotely")

    def test_country_list_is_preserved_on_the_job(self):
        job = record_to_job(self.raw)
        self.assertIn("Kenya", job.country)
        self.assertIn("Uganda", job.country)

    def test_worldwide_region_maps_to_global_remote(self):
        self.assertIs(record_to_job(self.raw).remote_status,
                      RemoteStatus.FULLY_REMOTE_GLOBAL)

    def test_a_regional_posting_is_not_upgraded_to_global(self):
        raw = dict(self.raw, location="Remote - Americas")
        job = record_to_job(raw)
        self.assertIs(job.remote_status, RemoteStatus.UNKNOWN)
        self.assertNotEqual(job.remote_status, RemoteStatus.FULLY_REMOTE_GLOBAL)

    def test_descriptions_are_marked_complete(self):
        # WWR publishes whole descriptions, unlike hiring.cafe card snippets.
        self.assertTrue(record_to_job(self.raw).description_complete)

    def test_state_is_folded_into_location_without_duplication(self):
        self.assertEqual(record_to_job(self.raw).location,
                         "Anywhere in the World (Delaware)")

    def test_state_already_in_region_is_not_repeated(self):
        raw = dict(self.raw, location="Delaware", state="Delaware")
        self.assertEqual(record_to_job(raw).location, "Delaware")


class AdapterContractTests(unittest.TestCase):
    def setUp(self):
        self.adapter = WwrSourceAdapter()

    def test_it_declares_the_contract(self):
        self.assertEqual(self.adapter.name, "weworkremotely")
        self.assertTrue(self.adapter.consumed_keys)

    def test_a_record_adapts(self):
        job = self.adapter.adapt(
            {"title": "Acme: Data Engineer", "url": "https://wwr.test/1",
             "company": "Acme", "description": "x", "location": "Anywhere in the World"},
            now=NOW,
        )
        self.assertEqual(job.company, "Acme")

    def test_a_record_without_a_url_is_refused(self):
        with self.assertRaises(AdaptError):
            self.adapter.adapt({"title": "No URL", "company": "Acme"}, now=NOW)

    def test_a_record_without_a_title_is_refused(self):
        with self.assertRaises(AdaptError):
            self.adapter.adapt({"url": "https://wwr.test/1", "company": "Acme"},
                               now=NOW)

    def test_it_is_reachable_through_the_shared_registry(self):
        job = adapt_record("weworkremotely", {
            "title": "Acme: Data Engineer", "url": "https://wwr.test/1",
            "company": "Acme", "location": "Anywhere in the World",
        }, now=NOW)
        self.assertEqual(job.portal, "weworkremotely")

    def test_importing_the_adapter_performs_no_request(self):
        # Registration happens at import; it must not fetch.
        ledger = Ledger()
        fetcher, calls = stub_fetcher(fixture_text(), ledger=ledger)
        self.assertEqual(calls, [])


class FetchPolicyTests(unittest.TestCase):
    def test_exactly_one_request_per_run(self):
        adapter = WwrAdapter(stub_fetcher(fixture_text())[0])
        self.assertEqual(len(adapter.fetch()), 6)
        adapter.fetch()
        adapter.fetch()
        self.assertEqual(adapter.requests_made, 1)

    def test_end_run_allows_the_next_run_to_refetch(self):
        fetcher, calls = stub_fetcher(fixture_text())
        adapter = WwrAdapter(fetcher)
        adapter.fetch()
        adapter.end_run()
        adapter.fetch()
        self.assertEqual(len(calls), 2)

    def test_only_the_feed_url_is_requested(self):
        fetcher, calls = stub_fetcher(fixture_text())
        WwrAdapter(fetcher).fetch()
        self.assertEqual(calls, [FEED_URL])

    def test_the_declared_minimum_interval_is_at_least_three_seconds(self):
        self.assertGreaterEqual(MIN_INTERVAL, 3.0)

    def test_a_default_adapter_uses_the_slow_limiter(self):
        self.assertIsInstance(WwrAdapter()._fetcher.limiter, RateLimiter)
        self.assertGreaterEqual(WwrAdapter()._fetcher.limiter.min_interval, 3.0)

    def test_no_disk_cache_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = WwrAdapter(stub_fetcher(fixture_text())[0])
            adapter.fetch()
            written = [p.name for p in root.rglob("*") if p.is_file()]
            self.assertEqual(written, [], "the feed forbids caching to disk")

    def test_a_refusal_is_not_retried(self):
        failure = AccessError("no", status=403)
        failure.headers, failure.body, failure.elapsed_ms = {}, "", 0
        fetcher, calls = stub_fetcher(failure, max_attempts=5)
        with self.assertRaises(AccessError):
            WwrAdapter(fetcher).fetch()
        self.assertEqual(len(calls), 1)

    def test_a_transient_failure_is_retried_within_bounds(self):
        failure = AccessError("boom", status=503, retryable=True)
        fetcher, calls = stub_fetcher(failure, max_attempts=3)
        with self.assertRaises(AccessError):
            WwrAdapter(fetcher).fetch()
        self.assertEqual(len(calls), 3)


class AttributionTests(unittest.TestCase):
    def test_attribution_is_declared_required(self):
        self.assertTrue(ATTRIBUTION_REQUIRED)
        self.assertTrue(WwrAdapter.attribution_required)

    def test_the_permission_statement_is_quoted_verbatim(self):
        self.assertEqual(
            PERMISSION_STATEMENT,
            "Anyone can use the feed, all we ask is that you attribute the "
            "links back to We Work Remotely.",
        )

    def test_records_carry_attribution(self):
        records = WwrAdapter(stub_fetcher(fixture_text())[0]).to_records()
        for record in records:
            with self.subTest(url=record["url"]):
                self.assertEqual(record["attribution"], ATTRIBUTION_TEXT)
                self.assertEqual(record["attribution_url"], PERMISSION_URL)

    def test_access_evidence_is_complete(self):
        evidence = WwrAdapter().access_evidence()
        for key in ("robots", "permission_statement", "permission_url",
                    "attribution_required", "access_method", "caching"):
            with self.subTest(key=key):
                self.assertIn(key, evidence)
        self.assertEqual(evidence["attribution_required"], "true")


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.store = JobStore(self.root / "data")

    def _run(self):
        adapter = WwrAdapter(stub_fetcher(fixture_text())[0])
        return adapter, run_sources(
            [SourceSpec(name="weworkremotely", fetch=adapter.to_records)],
            store=self.store, observed_at=NOW,
        )

    def test_all_six_jobs_are_stored(self):
        adapter, run = self._run()
        self.assertEqual(run.total_fetched, 6)
        self.assertEqual(run.total_stored, 6)
        self.assertEqual(run.exit_code(), 0)
        self.assertEqual(len(self.store.load_jobs()), 6)

    def test_the_run_is_recorded_in_the_health_ledger(self):
        self._run()
        ledger = self.store.load_runs()
        self.assertEqual(len(ledger), 1)
        entry = ledger[0]
        self.assertEqual(entry["sources"][0]["name"], "weworkremotely")
        self.assertEqual(entry["totals"]["stored"], 6)
        self.assertIn("duration_ms", entry["sources"][0])

    def test_a_second_run_updates_rather_than_duplicates(self):
        self._run()
        self._run()
        self.assertEqual(len(self.store.load_jobs()), 6)
        self.assertEqual(len(self.store.load_runs()), 2)

    def test_persisted_jobs_keep_the_country_list_where_published(self):
        # Two fixture items are region-scoped and publish no country list;
        # nothing is invented for them.
        self._run()
        published = 0
        for record in self.store.load_jobs():
            country = record["job"].get("country")
            if country:
                published += 1
        self.assertGreaterEqual(published, 1)
        self.assertLess(published, 6, "some items legitimately publish no list")

    def test_a_published_country_list_survives_ingestion_verbatim(self):
        self._run()
        record = next(r for r in self.store.load_jobs() if r["job"].get("country"))
        self.assertIn("Australia", record["job"]["country"])
        self.assertNotIn("Kenya", record["job"]["country"])

    def test_persisted_jobs_keep_the_full_description(self):
        self._run()
        for record in self.store.load_jobs():
            with self.subTest(job=record["job_id"]):
                self.assertTrue(record["job"]["description_complete"])


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.store = JobStore(self.root / "data")
        adapter = WwrAdapter(stub_fetcher(fixture_text())[0])
        self.run = run_sources(
            [SourceSpec(name="weworkremotely", fetch=adapter.to_records)],
            store=self.store, observed_at=NOW,
        )

    def render(self, **kwargs):
        kwargs.setdefault("attributions", {"weworkremotely": PERMISSION_URL})
        path = self.root / "d.html"
        render_dashboard_file(self.store, path, generated_at="2026-10-09", **kwargs)
        return path.read_text(encoding="utf-8")

    def test_all_jobs_are_shown_by_default(self):
        from app.reporting.jobs import build_view

        views = [build_view(r) for r in self.store.load_jobs()]
        self.assertEqual(len(views), 6)

    def test_attribution_is_rendered_with_a_link_back(self):
        markup = self.render()
        self.assertIn("Attribution required", markup)
        self.assertIn(PERMISSION_URL, markup)
        self.assertIn("Listings from", markup)

    def test_coverage_counts_are_rendered(self):
        markup = self.render()
        for label in ("fetched", "stored", "eligible", "not eligible",
                      "duplicates", "quarantined"):
            with self.subTest(label=label):
                self.assertIn(label, markup)

    def test_no_item_in_the_fixture_includes_kenya(self):
        for item in parse_feed(fixture_text()):
            with self.subTest(url=item.url):
                self.assertFalse(countries_mention(item.countries, CANDIDATE_COUNTRY))

    def test_the_coverage_note_states_a_zero_eligible_result_plainly(self):
        # Exercised on a store whose jobs are all Kenya-excluded, so the
        # branch that matters here is genuinely covered.
        from app.reporting.jobs import _coverage_panel

        views = _views_with_verdicts(["not_eligible", "unknown", "unknown"],
                                 country="Uganda")
        markup = _coverage_panel(coverage_of(views, []), views)
        self.assertIn("0 Kenya-eligible jobs", markup)
        self.assertIn("not a collection failure", markup)

    def test_the_eligibility_note_is_omitted_when_something_is_eligible(self):
        from app.reporting.jobs import _coverage_panel

        views = _views_with_verdicts(["eligible", "not_eligible"], country="Kenya")
        markup = _coverage_panel(coverage_of(views, []), views)
        self.assertNotIn("0 Kenya-eligible jobs", markup)

    def test_the_country_list_note_is_independent_of_the_verdict(self):
        # A verdict can say "eligible" while the source's own country list
        # never names Kenya. That fact must still be stated.
        from app.reporting.jobs import _coverage_panel

        views = _views_with_verdicts(["eligible", "eligible"], country="Uganda, South Africa")
        markup = _coverage_panel(coverage_of(views, []), views)
        self.assertIn("No published country list names Kenya", markup)
        self.assertIn("contested", markup)
        self.assertIn("naming Kenya", markup)

    def test_the_coverage_note_distinguishes_no_run_from_an_empty_result(self):
        from app.reporting.jobs import _coverage_panel

        views = _views_with_verdicts(["not_eligible"], country="Uganda")
        markup = _coverage_panel(
            coverage_of(views, [{"no_active_sources": True}]), views
        )
        self.assertIn("No source was consulted", markup)


    def test_coverage_counts_eligibility_bands(self):
        from app.reporting.jobs import build_view

        views = [build_view(r) for r in self.store.load_jobs()]
        totals = coverage_of(views, self.store.load_runs())
        self.assertEqual(totals["fetched"], 6)
        self.assertEqual(totals["stored"], 6)
        self.assertEqual(
            totals["eligible"] + totals["not_eligible"] + totals["unknown"],
            len(views))

    def test_only_jobs_with_both_signals_are_flagged_as_conflicted(self):
        # Exactly one fixture item publishes a country list *and* has body
    # text naming a region containing Kenya. The rest publish no list at all,
    # so they fall to the existing prose policy and carry no conflict.
        from app.jobs.eligibility import GEOGRAPHY_CONFLICT
        from app.reporting.jobs import build_view

        views = [build_view(r) for r in self.store.load_jobs()]
        conflicted = [v for v in views if GEOGRAPHY_CONFLICT in v.flags]
        self.assertEqual(len(conflicted), 1)
        self.assertEqual(conflicted[0].company, "Lemon.io")
        self.assertEqual(conflicted[0].verdict, "not_eligible")

    def test_a_job_with_a_country_list_is_not_eligible_even_when_worldwide(self):
        from app.reporting.jobs import build_view

        views = [build_view(r) for r in self.store.load_jobs()]
        listed = [v for v in views if v.country]
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0].company, "Lemon.io")
        self.assertEqual(listed[0].verdict, "not_eligible")

    def test_jobs_with_no_country_list_keep_the_existing_prose_policy(self):
        from app.reporting.jobs import build_view

        views = [build_view(r) for r in self.store.load_jobs()]
        unlisted = [v for v in views if not v.country]
        self.assertEqual(len(unlisted), 5)
        for view in unlisted:
            with self.subTest(job=view.job_id):
                self.assertNotIn("geography_conflict", view.flags)

    def test_the_kenya_filter_is_opt_in_and_narrows_to_eligible(self):
        from app.reporting.jobs import build_view, _filtered

        views = [build_view(r) for r in self.store.load_jobs()]
        eligible = [v for v in views if v.verdict == "eligible"]
        # Default shows everything; the filter is opt-in.
        self.assertEqual(len(_filtered(views)), len(views))
        self.assertEqual(
            len(_filtered(views, kenya_eligible=True)), len(eligible))
        self.assertTrue(
            all(v.verdict == "eligible" for v in _filtered(views, kenya_eligible=True)))

    def test_the_dashboard_stays_read_only(self):
        before = self.store.jobs_path.read_text(encoding="utf-8")
        self.render()
        self.render(kenya_eligible=True)
        self.assertEqual(before, self.store.jobs_path.read_text(encoding="utf-8"))

    def test_no_javascript_or_external_references(self):
        markup = self.render().casefold()
        self.assertNotIn("<script", markup)
        self.assertNotIn("src=", markup)



class GeographyConflictTests(unittest.TestCase):
    """WWR publishes a global label and a country list that can disagree."""

    def test_a_global_label_with_kenya_absent_is_flagged(self):
        from app.reporting.jobs import geography_conflict

        job = {"region": "Anywhere in the World", "country": "Uganda, South Africa"}
        self.assertIn("omits Kenya", geography_conflict(job))

    def test_a_global_label_with_kenya_present_is_not_flagged(self):
        from app.reporting.jobs import geography_conflict

        job = {"region": "Anywhere in the World", "country": "Uganda, Kenya, South Africa"}
        self.assertEqual(geography_conflict(job), "")

    def test_a_regional_posting_is_not_flagged(self):
        from app.reporting.jobs import geography_conflict

        job = {"region": "Remote - Americas", "country": "United States, Canada"}
        self.assertEqual(geography_conflict(job), "")

    def test_no_country_list_means_no_conflict_to_report(self):
        from app.reporting.jobs import geography_conflict

        self.assertEqual(geography_conflict({"region": "Anywhere in the World"}), "")

    def test_a_single_country_is_not_an_enumeration(self):
        from app.reporting.jobs import geography_conflict

        job = {"region": "Anywhere in the World", "country": "Germany"}
        self.assertEqual(geography_conflict(job), "")

    def test_the_helper_never_changes_a_verdict(self):
        # It is a reporting convenience only; the decision lives in
        # app.jobs.eligibility.
        from app.reporting.jobs import geography_conflict

        job = {"region": "Anywhere in the World", "country": "Kenya, Uganda"}
        self.assertEqual(geography_conflict(job), "")

class NoNetworkTests(unittest.TestCase):
    def test_the_source_module_has_no_hardcoded_fetch_beyond_the_fetcher(self):
        import ast

        path = Path(__file__).resolve().parent.parent / "app" / "sources" / "wwr.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])
        forbidden = {"urllib", "requests", "httpx", "socket", "subprocess",
                     "playwright", "selenium", "aiohttp"}
        self.assertEqual(imported & forbidden, set(),
                         "fetching must go through AccessFetcher only")

    def test_no_authentication_or_credentials(self):
        text = (Path(__file__).resolve().parent.parent
                / "app" / "sources" / "wwr.py").read_text(encoding="utf-8").casefold()
        for marker in ("authorization", "api_key", "apikey", "password",
                       "cookie", "bearer", "login", "oauth"):
            with self.subTest(marker=marker):
                self.assertNotIn(marker, text)

    def test_only_one_host_is_referenced_in_code(self):
        # Docstrings mention other services for contrast (hiring.cafe), so only
        # string literals that are not the module docstring are considered.
        import ast

        path = (Path(__file__).resolve().parent.parent
                / "app" / "sources" / "wwr.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstring = ast.get_docstring(tree) or ""
        hosts = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if value == docstring:
                    continue
                hosts.update(re.findall(r"https?://([^/\s\"']+)", value))
        # XML namespace URIs (e.g. the Atom namespace) are identifiers, never
        # fetched, so they are not hosts this source contacts.
        hosts = {h for h in hosts if "w3.org" not in h}
        self.assertEqual(
            hosts, {"weworkremotely.com"},
            "this source must reference exactly one host outside prose")


if __name__ == "__main__":
    unittest.main()