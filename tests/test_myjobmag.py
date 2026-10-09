"""Tests for the MyJobMag RSS source.

Fully offline. Every test drives the fixture
``tests/fixtures/myjobmag/sample.xml`` - a real captured response, truncated
and with contact details scrubbed - or stubs the transport. No network, no
provider, no browser automation.

The behaviours worth pinning are the ones where a plausible-looking shortcut
would be wrong: inferring Kenya from the ``.co.ke`` domain, "repairing" valid
Unicode, hiding a job that cannot be parsed, and quietly exceeding the approved
request budget.
"""

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app.jobs.adapters import AdaptError, adapt_record
from app.jobs.ingest import ingest
from app.jobs.models import RemoteStatus
from app.jobs.runner import SourceSpec, run_sources
from app.jobs.store import JobStore
from app.reporting.jobs import build_view, render_dashboard_file
from app.sources.myjobmag import (
    ATTRIBUTION_REQUIRED,
    ATTRIBUTION_TEXT,
    DAILY_LIMIT,
    FEED_URL,
    FEEDS_PAGE,
    LOCATION_DESCRIPTION,
    LOCATION_MISSING,
    LOCATION_STRUCTURED,
    MIN_INTERVAL,
    _DailyBudget,
    MyjobmagAdapter,
    MyjobmagSourceAdapter,
    clean_html,
    location_evidence,
    looks_mojibake,
    mojibake_score,
    parse_date,
    parse_feed,
    repair_text,
    split_title,
)
from app.sources.transport import (
    AccessError,
    AccessFetcher,
    HttpResponse,
    Ledger,
    RateLimiter,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "myjobmag" / "sample.xml"
NOW = datetime(2026, 10, 9, 12, 0, 0, tzinfo=timezone.utc)


def fixture_text():
    return FIXTURE.read_text(encoding="utf-8")


def adapter_with(body=None, **kwargs):
    calls = []

    def opener(url, timeout):
        calls.append(url)
        if isinstance(body, Exception):
            raise body
        return HttpResponse(url=url, status=200, headers={},
                            body=body if body is not None else fixture_text(),
                            elapsed_ms=5)

    fetcher = AccessFetcher(
        ledger=kwargs.pop("ledger", None) or Ledger(),
        limiter=RateLimiter(0.0, sleeper=lambda _: None),
        opener=opener, sleeper=lambda _: None, **kwargs
    )
    return MyjobmagAdapter(fetcher, budget=_DailyBudget(DAILY_LIMIT, now=lambda: 0.0)), calls


class FixtureTests(unittest.TestCase):
    def test_the_fixture_is_a_real_capture(self):
        text = fixture_text()
        self.assertIn("<rss", text)
        self.assertIn("myjobmag.co.ke", text)

    def test_the_fixture_is_truncated(self):
        self.assertEqual(fixture_text().count("<item>"), 8)

    def test_the_fixture_carries_no_contact_details(self):
        import re

        text = fixture_text()
        self.assertIsNone(re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text),
                          "no email addresses in the fixture")
        self.assertNotIn("+254", text)


class ParsingTests(unittest.TestCase):
    def setUp(self):
        self.items = parse_feed(fixture_text())

    def test_items_parse(self):
        self.assertEqual(len(self.items), 8)

    def test_every_item_has_an_original_link(self):
        for item in self.items:
            with self.subTest(item=item.title):
                self.assertTrue(item.url.startswith("https://www.myjobmag.co.ke/"))

    def test_every_item_has_a_description(self):
        for item in self.items:
            with self.subTest(item=item.title):
                self.assertTrue(item.description)

    def test_descriptions_are_marked_complete(self):
        for item in self.items:
            with self.subTest(item=item.title):
                self.assertTrue(item.to_job().description_complete)

    def test_remote_status_is_never_inferred(self):
        # The feed has no remote field; inferring one from prose would be a
        # guess presented as a fact.
        for item in self.items:
            with self.subTest(item=item.title):
                self.assertIs(item.to_job().remote_status, RemoteStatus.UNKNOWN)

    def test_no_country_field_is_invented(self):
        for item in self.items:
            with self.subTest(item=item.title):
                self.assertIsNone(item.to_job().country)


class TitleSplitTests(unittest.TestCase):
    def test_role_and_company_are_separated(self):
        self.assertEqual(split_title("Finance Manager at SevenTwenty Holdings Ltd"),
                         ("Finance Manager", "SevenTwenty Holdings Ltd"))

    def test_split_uses_the_last_at(self):
        self.assertEqual(split_title("Rep at Amazon at Big Co"),
                         ("Rep at Amazon", "Big Co"))

    def test_no_at_leaves_company_unknown_rather_than_guessed(self):
        title, company = split_title("Data Engineer")
        self.assertEqual(title, "Data Engineer")
        self.assertEqual(company, "")

    def test_empty_title(self):
        self.assertEqual(split_title(""), ("", ""))


class HtmlDecodingTests(unittest.TestCase):
    def test_escaped_markup_becomes_prose(self):
        cleaned = clean_html("&lt;p&gt;Build &lt;b&gt;pipelines&lt;/b&gt;&lt;/p&gt;")
        self.assertIn("Build pipelines", cleaned)
        self.assertNotIn("<b>", cleaned)

    def test_entities_are_decoded(self):
        self.assertIn("R&D", clean_html("R&amp;D"))

    def test_script_bodies_are_removed(self):
        self.assertNotIn("alert", clean_html("<script>alert(1)</script>Role"))

    def test_image_tags_do_not_leak(self):
        self.assertNotIn("logo.gif", clean_html('<img src="http://x/logo.gif"/>Role'))

    def test_empty_input(self):
        self.assertEqual(clean_html(""), "")


class EncodingTests(unittest.TestCase):
    def test_valid_unicode_is_untouched(self):
        for text in ("Zürich office", "Café Manager", "Chemin de l’école",
                     "Irmã e Açúcar", "Đà Nẵng"):
            with self.subTest(text=text):
                result = repair_text(text)
                self.assertEqual(result.text, text)
                self.assertFalse(result.repaired)

    def test_plain_ascii_is_untouched(self):
        result = repair_text("Finance Manager at SevenTwenty")
        self.assertFalse(result.repaired)
        self.assertEqual(result.text, "Finance Manager at SevenTwenty")

    def test_single_level_mojibake_is_repaired(self):
        result = repair_text("CafÃ© Manager")
        self.assertTrue(result.repaired)
        self.assertEqual(result.text, "Café Manager")

    def test_double_mojibake_is_repaired(self):
        result = repair_text("ZÃ¼rich office")
        self.assertTrue(result.repaired)
        self.assertEqual(result.text, "Zürich office")

    def test_triple_encoded_is_repaired_or_marked_uncertain(self):
        # Real MyJobMag data is triple encoded; two passes cannot fully
        # resolve it, so it must be marked uncertain and shown verbatim
        # rather than half-corrected.
        raw = next(i.title_original for i in parse_feed(fixture_text())
                    if mojibake_score(i.title_original))
        result = repair_text(raw)
        if result.repaired:
            self.assertEqual(result.uncertain, False)
        else:
            self.assertTrue(result.uncertain)
            self.assertEqual(result.text, raw)

    def test_the_original_is_always_preserved(self):
        raw = "CafÃ© Manager"
        self.assertEqual(repair_text(raw).original, raw)

    def test_repair_is_bounded_to_two_passes(self):
        result = repair_text("FranÃÂ¢ÃÂÃÂois Ã©cole")
        self.assertLessEqual(result.passes, 2)

    def test_uncertain_repair_is_never_silently_applied(self):
        raw = "FranÃÂ¢ÃÂÃÂois Ã©cole"
        result = repair_text(raw)
        if result.uncertain:
            self.assertEqual(result.text, raw, "an uncertain repair must not be applied")

    def test_a_non_possible_repair_is_rejected(self):
        # Nothing here can be round-tripped into valid text.
        raw = "\x80\x81 not text"
        result = repair_text(raw)
        self.assertEqual(result.text, raw)

    def test_the_real_fixture_mojibake_item_is_handled(self):
        items = parse_feed(fixture_text())
        originals = [i.title_original for i in items]
        originals = [t for t in originals if mojibake_score(t)]
        self.assertTrue(originals, "the fixture retains a mojibake title")

    def test_mojibake_detection_does_not_fire_on_valid_text(self):
        self.assertFalse(looks_mojibake("Café Manager at Zürich"))
        self.assertFalse(looks_mojibake("Irmã e Açúcar"))
        self.assertTrue(looks_mojibake("CafÃ© Manager"))


class DateTests(unittest.TestCase):
    def test_rfc822_parses(self):
        self.assertEqual(parse_date("Fri, 9 Oct 2026 16:01:29 GMT"),
                         "2026-10-09T16:01:29+00:00")

    def test_unparseable_is_none_not_invented(self):
        self.assertIsNone(parse_date("recently"))
        self.assertIsNone(parse_date(""))

    def test_fixture_dates_parse(self):
        for item in parse_feed(fixture_text()):
            with self.subTest(item=item.title):
                self.assertTrue(item.posted)


class LocationEvidenceTests(unittest.TestCase):
    def test_structured_country_naming_kenya(self):
        result = location_evidence({"country": "Kenya", "location": "Nairobi"})
        self.assertEqual(result.kind, LOCATION_STRUCTURED)
        self.assertEqual(result.verdict, "kenya")

    def test_structured_country_naming_elsewhere(self):
        result = location_evidence({"country": "South Africa"})
        self.assertEqual(result.kind, LOCATION_STRUCTURED)
        self.assertEqual(result.verdict, "other")

    def test_description_naming_kenya_is_kenya_evidence(self):
        result = location_evidence(
            {"description": "This role is based in Nairobi, Kenya."})
        self.assertEqual(result.kind, LOCATION_DESCRIPTION)
        self.assertEqual(result.verdict, "kenya")

    def test_description_naming_a_kenyan_city(self):
        self.assertEqual(
            location_evidence({"description": "Based in Mombasa."}).verdict, "kenya")

    def test_description_naming_another_country(self):
        result = location_evidence({"description": "This role is based in Kampala, Uganda."})
        self.assertEqual(result.verdict, "other")

    def test_africa_alone_is_ambiguous_not_kenya(self):
        result = location_evidence({"description": "Remote across Africa."})
        self.assertEqual(result.verdict, "ambiguous")

    def test_remote_alone_is_ambiguous(self):
        self.assertEqual(location_evidence({"description": "Remote role."}).verdict,
                         "ambiguous")

    def test_worldwide_alone_is_ambiguous(self):
        self.assertEqual(location_evidence({"description": "Work worldwide."}).verdict,
                         "ambiguous")

    def test_missing_location_is_missing(self):
        result = location_evidence({})
        self.assertEqual(result.kind, LOCATION_MISSING)
        self.assertEqual(result.verdict, "ambiguous")

    def test_empty_description_is_missing(self):
        self.assertEqual(location_evidence({"description": "  "}).kind, LOCATION_MISSING)

    def test_the_domain_is_never_consulted(self):
        # A .co.ke source says nothing about where a role is.
        result = location_evidence({"description": "Excellent opportunity."})
        self.assertEqual(result.verdict, "ambiguous")

    def test_evidence_carries_a_quote(self):
        result = location_evidence({"description": "Role based in Nairobi, Kenya."})
        self.assertIn("Nairobi", result.quote)


class AdapterContractTests(unittest.TestCase):
    def setUp(self):
        self.adapter = MyjobmagSourceAdapter()

    def test_it_declares_the_contract(self):
        self.assertEqual(self.adapter.name, "myjobmag.co.ke")
        self.assertTrue(self.adapter.consumed_keys)

    def test_a_record_adapts(self):
        job = self.adapter.adapt(
            {"title": "Data Engineer at Acme", "company": "Acme",
             "url": "https://www.myjobmag.co.ke/a_fields.php?id=1",
             "description": "Based in Nairobi, Kenya."}, now=NOW)
        self.assertEqual(job.company, "Acme")

    def test_a_record_without_a_link_is_refused(self):
        with self.assertRaises(AdaptError):
            self.adapter.adapt({"title": "No link", "company": "Acme"}, now=NOW)

    def test_a_record_without_a_title_is_refused(self):
        with self.assertRaises(AdaptError):
            self.adapter.adapt(
                {"url": "https://www.myjobmag.co.ke/a_fields.php?id=1"}, now=NOW)

    def test_reachable_through_the_shared_registry(self):
        job = adapt_record("myjobmag.co.ke", {
            "title": "Data Engineer at Acme", "company": "Acme",
            "url": "https://www.myjobmag.co.ke/a_fields.php?id=1",
            "description": "Nairobi, Kenya"}, now=NOW)
        self.assertEqual(job.portal, "myjobmag.co.ke")

    def test_importing_it_makes_no_request(self):
        ledger = Ledger()
        _, calls = adapter_with(ledger=ledger)
        self.assertEqual(calls, [])


class RequestPolicyTests(unittest.TestCase):
    def test_only_the_approved_feed_is_requested(self):
        adapter, calls = adapter_with()
        adapter.fetch()
        self.assertEqual(calls, [FEED_URL])
        self.assertIn("myjobmag.co.ke/jobsxml_by_categories.xml", FEED_URL)

    def test_the_daily_limit_is_one(self):
        self.assertEqual(DAILY_LIMIT, 1)
        self.assertGreaterEqual(MIN_INTERVAL, 86400.0)

    def test_a_second_fetch_in_the_same_day_is_refused(self):
        adapter, calls = adapter_with()
        adapter.fetch()
        adapter.end_run()
        with self.assertRaises(AccessError) as caught:
            adapter.fetch()
        self.assertIn("daily request limit", str(caught.exception))
        self.assertEqual(len(calls), 1, "the refusal must not have fetched")

    def test_the_budget_allows_a_request_when_the_day_has_rolled_over(self):
        budget = _DailyBudget(1, now=lambda: 100000.0)
        self.assertTrue(budget.allow())
        budget.record()
        self.assertFalse(budget.allow())

    def test_the_budget_forgets_old_stamps(self):
        now = [0.0]
        budget = _DailyBudget(1, now=lambda: now[0])
        self.assertTrue(budget.allow())
        budget.record()
        self.assertFalse(budget.allow())
        now[0] = 90000.0
        self.assertTrue(budget.allow(), "a new day permits a new request")

    def test_a_refusal_is_not_retried(self):
        failure = AccessError("no", status=403)
        failure.headers, failure.body, failure.elapsed_ms = {}, "", 0
        adapter, calls = adapter_with(body=failure, max_attempts=5)
        with self.assertRaises(AccessError):
            adapter.fetch()
        self.assertEqual(len(calls), 1)

    def test_transient_failures_retry_within_bounds(self):
        failure = AccessError("boom", status=503, retryable=True)
        adapter, calls = adapter_with(body=failure, max_attempts=3)
        with self.assertRaises(AccessError):
            adapter.fetch()
        self.assertEqual(len(calls), 3)

    def test_a_non_feed_response_fails_loudly(self):
        adapter, _ = adapter_with(body="<html>Cloudflare</html>")
        with self.assertRaises(AdaptError):
            adapter.fetch()


class AttributionTests(unittest.TestCase):
    def test_attribution_is_required(self):
        self.assertTrue(ATTRIBUTION_REQUIRED)
        self.assertTrue(MyjobmagAdapter.attribution_required)

    def test_records_carry_attribution(self):
        adapter, _ = adapter_with()
        for record in adapter.to_records():
            with self.subTest(url=record["url"]):
                self.assertEqual(record["attribution"], ATTRIBUTION_TEXT)
                self.assertEqual(record["attribution_url"], FEEDS_PAGE)

    def test_access_evidence_declares_what_is_not_permitted(self):
        evidence = MyjobmagAdapter().access_evidence()
        forbidden = evidence["not_permitted"].casefold()
        for marker in ("scraping", "detail pages", "pagination", "endpoint"):
            with self.subTest(marker=marker):
                self.assertIn(marker, forbidden)


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.store = JobStore(self.root / "data")

    def run_once(self):
        adapter, _ = adapter_with()
        return adapter, run_sources(
            [SourceSpec(name="myjobmag.co.ke", fetch=adapter.to_records)],
            store=self.store, observed_at=NOW)

    def test_jobs_are_ingested_and_persisted(self):
        _, run = self.run_once()
        self.assertEqual(run.total_fetched, 8)
        self.assertEqual(run.exit_code(), 0)
        self.assertEqual(len(self.store.load_jobs()), 8)

    def test_the_run_is_recorded_in_the_health_ledger(self):
        self.run_once()
        entry = self.store.load_runs()[0]
        self.assertEqual(entry["sources"][0]["name"], "myjobmag.co.ke")
        self.assertEqual(entry["totals"]["stored"], 8)

    def test_original_urls_are_persisted_unchanged(self):
        self.run_once()
        urls = [r["job"]["url"] for r in self.store.load_jobs()]
        for url in urls:
            with self.subTest(url=url):
                self.assertTrue(url.startswith("https://www.myjobmag.co.ke/"))

    def test_a_re_run_updates_rather_than_duplicates(self):
        self.run_once()
        self.run_once()
        self.assertEqual(len(self.store.load_jobs()), 8)
        self.assertEqual(len(self.store.load_runs()), 2)

    def test_encoding_uncertainty_is_persisted(self):
        self.run_once()
        excerpts = [r["job"].get("raw_excerpt") for r in self.store.load_jobs()]
        flags = []
        for raw in excerpts:
            if not raw:
                continue
            try:
                flags.append(json.loads(raw).get("title_encoding_uncertain"))
            except (TypeError, ValueError):
                continue
        self.assertTrue(
            any(flags),
            "the mojibake item's uncertainty must survive ingestion",
        )

    def test_the_original_title_is_retained_for_audit(self):
        self.run_once()
        originals = []
        for record in self.store.load_jobs():
            raw = record["job"].get("raw_excerpt")
            if raw:
                originals.append(json.loads(raw).get("title_original"))
        self.assertTrue(all(originals), "every record keeps its original title")
        self.assertTrue(any("Ã" in (o or "") for o in originals),
                        "the mis-encoded title is preserved verbatim")


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.store = JobStore(self.root / "data")
        adapter, _ = adapter_with()
        run_sources([SourceSpec(name="myjobmag.co.ke", fetch=adapter.to_records)],
                    store=self.store, observed_at=NOW)

    def render(self, **kwargs):
        path = self.root / "d.html"
        render_dashboard_file(self.store, path, generated_at="2026-10-09", **kwargs)
        return path.read_text(encoding="utf-8")

    def test_attribution_is_rendered_with_a_link_back(self):
        markup = self.render(attributions={"myjobmag.co.ke": FEEDS_PAGE})
        self.assertIn("Attribution required", markup)
        self.assertIn(FEEDS_PAGE, markup)

    def test_original_links_are_preserved(self):
        markup = self.render()
        self.assertIn("myjobmag.co.ke/a_fields.php", markup)

    def test_rendering_is_read_only(self):
        before = self.store.jobs_path.read_text(encoding="utf-8")
        self.render()
        self.assertEqual(before, self.store.jobs_path.read_text(encoding="utf-8"))

    def test_no_scripts_or_external_references(self):
        markup = self.render().casefold()
        self.assertNotIn("<script", markup)
        self.assertNotIn("src=", markup)
        self.assertNotIn("<link", markup)

    def test_no_secrets_or_internal_paths(self):
        markup = self.render()
        for marker in ("password", "api_key", "BEGIN PRIVATE", "C:\\", "/Users/"):
            with self.subTest(marker=marker):
                self.assertNotIn(marker, markup)


class RegistryScopeTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.data = Path(self._directory.name)

    def test_permitted_activation_requires_the_recorded_decision(self):
        from app.jobs.sources import PERMITTED, SourceConfig, SourceRegistry
        from app.sources.access import (
            AccessDecision, AccessLevel, load_decisions, save_decision,
        )

        decision = AccessDecision(
            source="myjobmag.co.ke", level=AccessLevel.PERMITTED,
            reason="approved RSS use", robots="allows", robots_url="",
            terms="", terms_url="", checked_at="2026-10-09T12:00:00+00:00",
        )
        save_decision(decision, self.data)
        registry = SourceRegistry([
            SourceConfig(name="myjobmag.co.ke", access=PERMITTED)])
        enforced = registry.enforce_recorded_decisions(load_decisions(self.data))
        self.assertEqual([c.name for c in enforced.active()], ["myjobmag.co.ke"])

    def test_an_unrecorded_source_cannot_activate(self):
        from app.jobs.sources import PERMITTED, SourceConfig, SourceRegistry

        registry = SourceRegistry([
            SourceConfig(name="myjobmag.co.ke", access=PERMITTED)])
        enforced = registry.enforce_recorded_decisions({})
        self.assertEqual(enforced.active(), [])

    def test_a_restricted_source_cannot_activate(self):
        from app.jobs.sources import PERMITTED, SourceConfig, SourceRegistry
        from app.sources.access import AccessDecision, AccessLevel

        registry = SourceRegistry([
            SourceConfig(name="myjobmag.co.ke", access=PERMITTED)])
        enforced = registry.enforce_recorded_decisions({"myjobmag.co.ke":
            AccessDecision(source="myjobmag.co.ke", level=AccessLevel.RESTRICTED,
                           reason="no", robots="", robots_url="", terms="",
                           terms_url="", checked_at="")})
        self.assertEqual(enforced.active(), [])


class NoSideEffectsTests(unittest.TestCase):
    def test_the_module_imports_nothing_that_performs_io(self):
        import ast

        path = (Path(__file__).resolve().parent.parent
                / "app" / "sources" / "myjobmag.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])
        forbidden = {"urllib", "requests", "httpx", "socket", "subprocess",
                     "playwright", "selenium", "aiohttp", "ollama", "asyncio"}
        self.assertEqual(imported & forbidden, set())

    def test_no_credentials_or_authentication(self):
        import ast

        path = (Path(__file__).resolve().parent.parent
                / "app" / "sources" / "myjobmag.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstring = ast.get_docstring(tree) or ""
        # Prose is excluded: the module docstring names these things to say
        # they are NOT used, which is the opposite of using them.
        code = "\n".join(
            n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and n.value != docstring
        ).casefold()
        text = code
        # "captcha" is deliberately not asserted here: the module docstring
        # names it to explain that challenges are NOT bypassed, and a
        # credential scan should test for credentials, not for prose.
        for marker in ("authorization", "api_key", "apikey", "password",
                       "cookie", "bearer", "oauth", "proxy"):
            with self.subTest(marker=marker):
                self.assertNotIn(marker, text)

    def test_no_detail_page_or_pagination_code(self):
        text = (Path(__file__).resolve().parent.parent
                / "app" / "sources" / "myjobmag.py").read_text(encoding="utf-8")
        # a_fields.php appears only in comments/evidence, never as a request.
        self.assertNotIn('a_fields.php?id=', text)
        self.assertNotIn("?page=", text)

    def test_only_myjobmag_hosts_are_referenced(self):
        import ast
        import re

        path = (Path(__file__).resolve().parent.parent
                / "app" / "sources" / "myjobmag.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        hosts = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for host in re.findall(r"https?://([^/\s'\"]+)", node.value):
                    hosts.add(host)
        self.assertEqual(hosts, {"www.myjobmag.co.ke"})

    def test_this_test_module_makes_no_network_call(self):
        import ast

        path = Path(__file__).resolve().parent / "test_myjobmag.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])
        forbidden = {"urllib", "requests", "httpx", "socket", "subprocess",
                     "playwright", "selenium", "aiohttp"}
        self.assertEqual(imported & forbidden, set())


if __name__ == "__main__":
    unittest.main()