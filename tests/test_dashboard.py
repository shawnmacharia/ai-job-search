"""Tests for the read-only review dashboard.

Entirely offline: fixtures and a temporary store. No network, Playwright,
Ollama, subprocess, or live scraping.

The dashboard is the first thing a person sees, so the tests focus on the
properties that make it trustworthy rather than on markup shape: it must
escape everything, never claim a value it does not have, never omit a source,
and never mutate what it reads.
"""

import json
import re
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app.jobs.ingest import ingest
from app.jobs.store import JobStore
from app.reporting.jobs import (
    NOT_EVALUATED,
    STATUS_PLACEHOLDER,
    build_view,
    render_dashboard_html,
    render_dashboard_file,
    render_row,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hiring_cafe"
NOW = datetime(2026, 10, 9, 12, 0, 0, tzinfo=timezone.utc)

INGEST = [
    ("basic.json", "hiring.cafe"),
    ("worldwide_remote.json", "hiring.cafe"),
    ("salary_ksh.json", "hiring.cafe"),
    ("europe_restricted.json", "hiring.cafe"),
]


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class DashboardTestCase(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.store = JobStore(self.root / "data")
        ingest(
            [load(name) for name, _ in INGEST],
            source="hiring.cafe",
            store=self.store,
            observed_at=NOW,
        )
        self.records = self.store.load_jobs()
        self.views = [build_view(record) for record in self.records]

    def view(self, company):
        return next(v for v in self.views if v.company == company)


class ViewTests(DashboardTestCase):
    def test_every_ingested_job_appears(self):
        self.assertEqual(len(self.views), len(INGEST))

    def test_verdict_and_reason_are_both_present(self):
        for view in self.views:
            with self.subTest(company=view.company):
                self.assertIn(view.verdict, {"eligible", "not_eligible", "unknown"})
                self.assertTrue(
                    view.verdict_reasons, "a verdict must never appear without a reason"
                )

    def test_restricted_region_is_shown_as_not_eligible_with_a_reason(self):
        view = self.view("Fjord Analytics")
        self.assertEqual(view.verdict, "not_eligible")
        self.assertTrue(
            any("EU" in reason for reason in view.verdict_reasons),
            f"the reason must name the region that excludes: {view.verdict_reasons}",
        )

    def test_worldwide_remote_is_shown_as_eligible(self):
        self.assertEqual(self.view("Globex").verdict, "eligible")

    def test_source_provenance_is_carried_through(self):
        view = self.view("Acme Analytics")
        self.assertIn("hiring.cafe", view.sources)
        self.assertTrue(view.source_urls)

    def test_flags_are_derived_not_invented(self):
        # No salary stated -> flagged.
        self.assertIn("no salary stated", self.view("Acme Analytics").flags)
        # Snippet description -> flagged.
        self.assertIn("description is a card snippet", self.view("Acme Analytics").flags)

    def test_a_job_with_a_stated_salary_is_not_flagged_for_it(self):
        self.assertNotIn("no salary stated", self.view("Safi Health").flags)

    def test_match_explanation_says_not_yet_evaluated_by_default(self):
        for view in self.views:
            self.assertEqual(view.match_explanation, NOT_EVALUATED)

    def test_match_explanation_is_used_when_supplied(self):
        record = self.records[0]
        view = build_view(record, match_explanation="Strong overlap on ELT and dbt.")
        self.assertEqual(view.match_explanation, "Strong overlap on ELT and dbt.")

    def test_application_status_is_an_explicit_placeholder(self):
        for view in self.views:
            self.assertEqual(view.application_status, STATUS_PLACEHOLDER)

    def test_first_and_last_seen_are_present(self):
        for view in self.views:
            self.assertTrue(view.first_seen)
            self.assertTrue(view.last_seen)


class RenderingTests(DashboardTestCase):
    def test_escapes_every_value_it_renders(self):
        hostile = dict(
            self.records[0],
            job={
                **self.records[0]["job"],
                "title": '<script>alert("x")</script>',
                "company": "A & B <b>bold</b>",
                "description": '"><img src=x onerror=alert(1)>',
            },
        )
        markup = render_dashboard_html([build_view(hostile)])
        self.assertNotIn("<script>", markup)
        self.assertNotIn("<b>bold</b>", markup)
        self.assertNotIn("onerror=", markup)
        self.assertIn("&lt;script&gt;", markup)
        self.assertIn("A &amp; B", markup)

    def test_contains_no_javascript(self):
        markup = render_dashboard_html(self.views)
        self.assertNotIn("<script", markup.casefold())
        self.assertNotIn("javascript:", markup.casefold())
        self.assertNotIn("onclick", markup.casefold())

    def test_contains_no_external_references(self):
        markup = render_dashboard_html(self.views)
        self.assertNotIn("<link", markup.casefold())
        self.assertNotIn("src=", markup.casefold())

    def test_every_job_gets_a_row(self):
        markup = render_dashboard_html(self.views)
        body = markup.split("<tbody>")[1].split("</tbody>")[0]
        self.assertEqual(body.count("<tr>"), len(self.views))

    def test_eligibility_verdict_is_rendered_per_row(self):
        markup = render_dashboard_html(self.views)
        for verdict in ("eligible", "not_eligible"):
            self.assertIn(verdict, markup)

    def test_original_url_is_linked_and_attribute_is_escaped(self):
        record = dict(self.records[0])
        record["job"] = {**record["job"], "url": 'https://x.test/?a=1&b="2"'}
        markup = render_dashboard_html([build_view(record)])
        self.assertNotIn('b="2"', markup)
        self.assertIn("&quot;2&quot;", markup)

    def test_empty_input_renders_an_explicit_empty_state(self):
        markup = render_dashboard_html([])
        self.assertIn("No jobs match", markup)
        self.assertNotIn("<tbody>", markup)

    def test_summary_counts_each_verdict(self):
        markup = render_dashboard_html(self.views)
        for verdict in ("eligible", "not_eligible"):
            count = sum(1 for v in self.views if v.verdict == verdict)
            self.assertIn(f"{verdict}: {count}", markup)

    def test_generated_at_is_stamped_and_escaped(self):
        markup = render_dashboard_html(self.views, generated_at="2026-10-09T12:00:00+00:00")
        self.assertIn("2026-10-09T12:00:00+00:00", markup)

    def test_row_never_shows_a_verdict_without_a_reason(self):
        markup = render_dashboard_html(self.views)
        self.assertNotIn("no reason recorded", markup)


class FileRenderingTests(DashboardTestCase):
    def test_writes_a_file_and_returns_its_path(self):
        target = self.root / "out" / "dashboard.html"
        written = render_dashboard_file(self.store, target, generated_at="2026-10-09")
        self.assertTrue(Path(written).exists())
        self.assertIn("Acme Analytics", Path(written).read_text(encoding="utf-8"))

    def test_rendering_is_read_only(self):
        before = (
            self.store.jobs_path.read_text(encoding="utf-8"),
            json.dumps(self.store.load_seen(), sort_keys=True),
            len(self.store.load_jobs()),
        )
        render_dashboard_file(self.store, self.root / "d.html")
        render_dashboard_file(self.store, self.root / "d2.html", verdict="eligible")
        after = (
            self.store.jobs_path.read_text(encoding="utf-8"),
            json.dumps(self.store.load_seen(), sort_keys=True),
            len(self.store.load_jobs()),
        )
        self.assertEqual(before, after, "the dashboard must not mutate the store")

    def test_repeated_rendering_is_byte_identical(self):
        first = render_dashboard_file(
            self.store, self.root / "a.html", generated_at="2026-10-09"
        )
        second = render_dashboard_file(
            self.store, self.root / "b.html", generated_at="2026-10-09"
        )
        self.assertEqual(
            Path(first).read_bytes(), Path(second).read_bytes(),
            "rendering must be deterministic",
        )


class FilteringAndSortingTests(DashboardTestCase):
    def test_verdict_filter_is_applied_at_render_time(self):
        markup = render_dashboard_html(
            self.views, filters={"verdict": "not_eligible"}
        )
        self.assertIn("filters", markup)
        # The filter is metadata; selection happens before rendering.
        from app.reporting.jobs import _filtered

        selected = _filtered(self.views, verdict="not_eligible")
        self.assertTrue(selected)
        self.assertTrue(all(v.verdict == "not_eligible" for v in selected))

    def test_source_filter(self):
        from app.reporting.jobs import _filtered

        self.assertEqual(len(_filtered(self.views, source="hiring.cafe")), len(self.views))
        self.assertEqual(_filtered(self.views, source="nowhere"), [])

    def test_query_filter_matches_title_company_and_description(self):
        from app.reporting.jobs import _filtered

        self.assertTrue(_filtered(self.views, query="Globex"))
        self.assertTrue(_filtered(self.views, query="ELT"))
        self.assertEqual(_filtered(self.views, query="zzzz-not-present"), [])

    def test_sorting_is_stable_and_deterministic(self):
        from app.reporting.jobs import _sorted

        for sort in ("company", "date", "verdict", "title"):
            with self.subTest(sort=sort):
                first = _sorted(self.views, sort)
                second = _sorted(list(reversed(self.views)), sort)
                self.assertEqual(
                    [v.job_id for v in first], [v.job_id for v in second],
                    "sorting must not depend on input order",
                )

    def test_company_sort_orders_alphabetically(self):
        from app.reporting.jobs import _sorted

        companies = [v.company for v in _sorted(self.views, "company")]
        self.assertEqual(companies, sorted(companies, key=str.casefold))

    def test_unknown_sort_falls_back_to_company(self):
        from app.reporting.jobs import _sorted

        self.assertEqual(
            [v.job_id for v in _sorted(self.views, "nonsense")],
            [v.job_id for v in _sorted(self.views, "company")],
        )


class DuplicateVisibilityTests(DashboardTestCase):
    def test_possible_duplicate_is_visible_in_the_row(self):
        mirrored = dict(load("basic.json"))
        mirrored["url"] = "https://hiring.cafe/job/basic-001-mirror"
        ingest([mirrored], source="hiring.cafe", store=self.store, observed_at=NOW)

        views = [build_view(record) for record in self.store.load_jobs()]
        flagged = [v for v in views if v.possible_duplicate]
        self.assertEqual(len(flagged), 1)
        markup = render_row(flagged[0])
        self.assertIn("possible duplicate", markup)


class MarkupRegressionsTests(DashboardTestCase):
    """Bugs found by reading the rendered output, not by the property tests.

    Each of these passed the escaping/no-JS suite above while still producing
    visibly broken HTML.
    """

    def test_flag_chips_render_as_elements_not_as_visible_markup(self):
        markup = render_dashboard_html(self.views)
        self.assertIn('<span class="flag">no salary stated</span>', markup)
        # The earlier bug escaped the chip markup itself, so the tags showed
        # up as literal text in the cell.
        self.assertNotIn("&lt;span class=&quot;flag&quot;&gt;", markup)

    def test_duplicate_and_completeness_chips_are_elements(self):
        record = dict(self.records[0])
        record["job"] = {**record["job"], "description_complete": True}
        markup = render_dashboard_html([build_view(record)])
        self.assertIn('<span class="flag">full description</span>', markup)
        self.assertNotIn("&lt;span", markup)

    def test_adjacent_verdict_reasons_are_separated(self):
        # Joining reasons with a space produced "restriction remote eligibility"
        # - two distinct reasons reading as one run-on clause.
        view = self.view("Globex")
        markup = render_row(view)
        self.assertNotIn("restriction remote", markup)
        self.assertIn(";", markup)

    def test_no_block_element_is_nested_inside_a_paragraph(self):
        markup = render_dashboard_html(self.views, filters={"verdict": "eligible"})
        for paragraph in re.findall(r"<p[^>]*>(.*?)</p>", markup, re.DOTALL):
            self.assertNotIn("<div", paragraph, "a <p> may not contain a <div>")

    def test_eligibility_notes_are_not_repeated_as_flags(self):
        # Notes are caveats on the verdict, which the verdict cell already
        # renders. Showing them in both cells said the same thing twice.
        for view in self.views:
            for reason in view.verdict_reasons:
                self.assertNotIn(
                    reason, view.flags,
                    "an eligibility reason must not also be a review flag",
                )

    def test_flags_remain_for_posting_facts(self):
        self.assertIn("no salary stated", self.view("Acme Analytics").flags)
        self.assertNotIn("no salary stated", self.view("Safi Health").flags)

    def test_chip_values_are_still_escaped(self):
        record = dict(self.records[0])
        record["job"] = {**record["job"], "deadline": '<b>2026-12-01</b>'}
        markup = render_dashboard_html([build_view(record)])
        self.assertIn("&lt;b&gt;2026-12-01&lt;/b&gt;", markup)
        self.assertNotIn("<b>2026-12-01</b>", markup)


class AsJobTests(unittest.TestCase):
    """The stored record is a dict; the eligibility policy reads a ``Job``."""

    def rebuild(self, **overrides):
        from app.reporting.jobs import _as_job

        payload = {
            "job_id": "j-1",
            "title": "Data Engineer",
            "company": "Acme Analytics",
            "url": "https://hiring.cafe/job/basic-001",
        }
        payload.update(overrides)
        return _as_job(payload)

    def test_remote_status_is_coerced_back_to_its_enum(self):
        from app.jobs.models import RemoteStatus

        rebuilt = self.rebuild(remote_status="fully_remote_global")
        self.assertIs(rebuilt.remote_status, RemoteStatus.FULLY_REMOTE_GLOBAL)

    def test_an_unrecognised_remote_status_becomes_unknown(self):
        # Data we did not anticipate must not reject the whole job.
        from app.jobs.models import RemoteStatus

        rebuilt = self.rebuild(remote_status="bespoke")
        self.assertIs(rebuilt.remote_status, RemoteStatus.UNKNOWN)

    def test_unknown_keys_are_dropped_not_passed_to_the_constructor(self):
        self.assertEqual(self.rebuild(not_a_job_field=1).title, "Data Engineer")


class NoSideEffectsTests(unittest.TestCase):
    def test_reporting_module_imports_nothing_that_performs_io(self):
        import ast

        path = Path(__file__).resolve().parent.parent / "app" / "reporting" / "jobs.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])

        forbidden = {
            "playwright", "requests", "httpx", "urllib", "socket", "subprocess",
            "ollama", "aiohttp", "selenium", "asyncio",
        }
        offenders = imported & forbidden
        self.assertEqual(offenders, set(), f"must stay offline: {sorted(offenders)}")


if __name__ == "__main__":
    unittest.main()
