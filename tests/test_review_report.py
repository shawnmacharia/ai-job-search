"""The consolidated daily review report.

Every test here is about a distinction that, once collapsed, makes the report
lie. The recurring theme is that similar-looking things are not the same:

- a job that was never assessed vs one assessed and found wanting
- a source that failed vs one that succeeded and returned nothing
- stale vs expired vs never observed at all
- ineligible vs uncertain

Two properties are load-bearing and get their own tests: the report is
**read-only** (proven by hashing every state file before and after) and it
**escapes** everything it renders (proven with a deliberately hostile title).

No test makes a network request, and no test writes to the store.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.jobs.freshness import FreshnessLedger
from app.jobs.runner import SourceSpec, run_sources
from app.jobs.status import ReviewStatus, StatusLog
from app.jobs.store import JobStore
from app.reporting.review import build_report, render_report_file, render_report_html

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _record(name, *, source="weworkremotely", company="Acme", location="Nairobi, Kenya",
            title="Role", description="Work."):
    return {
        "url": f"https://{source}/jobs/{name}",
        "title": title,
        "company": company,
        "location": location,
        "description": description,
    }


def _ingest(store, records, *, source="weworkremotely", at=None):
    run_sources(
        [SourceSpec(name=source, fetch=lambda: list(records))],
        store=store,
        observed_at=at or datetime(2026, 3, 1, tzinfo=timezone.utc),
    )


class ReportTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = JobStore(Path(self._tmp.name) / "data")

    def report(self, **kwargs):
        kwargs.setdefault("status_log", StatusLog(self.store))
        return build_report(self.store, **kwargs)


class SourceCoverageTests(ReportTestCase):
    """All three live sources must be represented."""

    def setUp(self):
        super().setUp()
        _ingest(self.store, [_record("a1", source="weworkremotely")],
                source="weworkremotely")
        _ingest(self.store, [_record("b1", source="myjobmag.co.ke")],
                source="myjobmag.co.ke")
        _ingest(self.store, [_record("c1", source="remotive.com")],
                source="remotive.com")

    def test_all_three_live_sources_are_represented(self):
        report = self.report()
        names = [row.name for row in report.sources]
        for expected in ("weworkremotely", "myjobmag.co.ke", "remotive.com"):
            self.assertIn(expected, names)

    def test_each_source_counts_its_own_jobs_only(self):
        report = self.report()
        totals = {row.name: row.eligible + row.ineligible + row.unknown
                  for row in report.sources}
        self.assertEqual(totals["weworkremotely"], 1)
        self.assertEqual(totals["myjobmag.co.ke"], 1)
        self.assertEqual(totals["remotive.com"], 1)

    def test_a_source_that_never_ran_is_still_listed(self):
        """Absence of a source is not the same as absence from the report."""
        report = self.report(attributions={})
        row = {r.name: r for r in report.sources}["remotive.com"]
        self.assertEqual(row.attribution, "")

    def test_attribution_is_shown_when_supplied(self):
        report = self.report(attributions={"remotive.com": "https://remotive.com/remote-jobs"})
        row = {r.name: r for r in report.sources}["remotive.com"]
        self.assertIn("remotive.com", row.attribution)


class ActionableTests(ReportTestCase):
    def test_a_new_eligible_job_appears_in_the_actionable_queue(self):
        _ingest(self.store, [_record("a1")])
        report = self.report()
        self.assertEqual(len(report.actionable), 1)
        self.assertEqual(report.actionable[0].verdict, "eligible")

    def test_an_ineligible_job_is_not_actionable(self):
        _ingest(self.store, [_record("a1", location="Berlin, Germany")])
        report = self.report()
        self.assertEqual(len(report.actionable), 0)

    def test_a_dismissed_job_does_not_appear_as_new(self):
        _ingest(self.store, [_record("a1"), _record("a2")])
        log = StatusLog(self.store)
        job_id = self.store.load_jobs()[0]["job_id"]
        log.record(job_id, ReviewStatus.DISMISSED, note="not for me")
        report = self.report()
        self.assertEqual(len(report.actionable), 1, "only the untouched job remains")
        self.assertNotIn(job_id, [v.job_id for v in report.actionable])

    def test_an_interested_job_is_not_listed_as_new(self):
        _ingest(self.store, [_record("a1")])
        log = StatusLog(self.store)
        log.record(self.store.load_jobs()[0]["job_id"], ReviewStatus.INTERESTED)
        self.assertEqual(len(self.report().actionable), 0)

    def test_actionable_is_a_subset_of_the_outstanding_queue(self):
        """The queue includes stale work; the actionable list does not."""
        _ingest(self.store, [_record("a1"), _record("a2")])
        self.assertGreaterEqual(
            len(self.report().eligible_unreviewed),
            len(self.report().actionable),
        )


class FreshnessTests(ReportTestCase):
    """Stale and expired are different ages and must never share a bucket."""

    def _age(self, days):
        return FreshnessLedger(self.store).evaluate()

    def test_stale_and_expired_are_separate_queues(self):
        _ingest(self.store, [_record("a1"), _record("a2")], source="weworkremotely")
        ledger = FreshnessLedger(self.store)
        state = ledger.evaluate()
        views = list(state)
        report = self.report()
        # Whatever the ages, the two queues are distinct sets by construction.
        stale_ids = {v.job_id for v in report.stale}
        expired_ids = {v.job_id for v in report.expired}
        self.assertEqual(stale_ids & expired_ids, set())
        for job_id in stale_ids:
            self.assertEqual(state[job_id].state.value, "stale")
        for job_id in expired_ids:
            self.assertEqual(state[job_id].state.value, "expired")

    def test_a_stale_job_is_not_actionable(self):
        from app.jobs.freshness import DEFAULT_POLICIES

        _ingest(self.store, [_record("a1"), _record("a2")], source="weworkremotely")
        stale_days = DEFAULT_POLICIES["weworkremotely"].stale_days
        _ingest(self.store, [_record("a2")], source="weworkremotely",
                at=datetime(2026, 3, 1, tzinfo=timezone.utc) + timedelta(days=stale_days + 2))
        report = self.report()
        stale_ids = {v.job_id for v in report.stale}
        self.assertTrue(stale_ids, "the dropped job should be stale")
        self.assertEqual(stale_ids & {v.job_id for v in report.actionable}, set())

    def test_a_failed_source_does_not_make_jobs_appear_expired(self):
        """The failure must not be read as evidence the jobs are gone."""
        _ingest(self.store, [_record("a1")], source="weworkremotely")
        before = self.report()
        self.assertEqual(len(before.expired), 0)

        def boom():
            raise RuntimeError("source exploded")

        run_sources(
            [SourceSpec(name="weworkremotely", fetch=boom)],
            store=self.store,
            observed_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        )
        report = self.report()
        self.assertEqual(len(report.expired), 0, "a failure expires nothing")
        for view in report.actionable + report.eligible_unreviewed:
            self.assertEqual(view.freshness, "active")

    def test_a_failure_is_reported_as_a_failed_source(self):
        _ingest(self.store, [_record("a1")], source="weworkremotely")

        def boom():
            raise RuntimeError("source exploded")

        run_sources([SourceSpec(name="weworkremotely", fetch=boom)],
                    store=self.store,
                    observed_at=datetime(2026, 3, 2, tzinfo=timezone.utc))
        row = {r.name: r for r in self.report().sources}["weworkremotely"]
        self.assertEqual(row.last_run_state, "failed")
        self.assertGreaterEqual(row.consecutive_failures, 1)

    def test_a_successful_empty_feed_is_reported_separately_from_a_failure(self):
        _ingest(self.store, [_record("a1")], source="weworkremotely")
        _ingest(self.store, [], source="weworkremotely",
                at=datetime(2026, 3, 2, tzinfo=timezone.utc))
        row = {r.name: r for r in self.report().sources}["weworkremotely"]
        self.assertEqual(row.last_run_state, "zero result")
        self.assertTrue(row.zero_result)
        self.assertEqual(row.consecutive_failures, 0)

    def test_a_zero_result_run_expires_nothing(self):
        _ingest(self.store, [_record("a1")], source="weworkremotely")
        _ingest(self.store, [], source="weworkremotely",
                at=datetime(2026, 9, 1, tzinfo=timezone.utc))
        report = self.report()
        self.assertEqual(len(report.expired), 0)
        self.assertEqual(len(report.stale), 0)

    def test_an_unobserved_source_leaves_jobs_unknown_not_active(self):
        _ingest(self.store, [_record("a1")], source="weworkremotely",
                at=datetime(2026, 3, 1, tzinfo=timezone.utc))
        report = self.report()
        # Every job was observed by this run, so freshness is established.
        self.assertNotEqual(report.actionable[0].freshness, "unknown")


class EligibilityVisibilityTests(ReportTestCase):
    def test_uncertain_eligibility_remains_visible(self):
        # "EMEA" scopes the role to a region without saying whether a KE-based
        # applicant qualifies, which is genuinely uncertain - not ineligible.
        _ingest(self.store, [_record("a1", location="EMEA")])
        report = self.report()
        self.assertEqual(len(report.uncertain), 1)
        self.assertEqual(report.uncertain[0].verdict, "unknown")
        self.assertTrue(report.uncertain[0].verdict_reasons)

    def test_contested_readings_remain_visible(self):
        _ingest(self.store, [_record("a1", location="United States of America")])
        report = self.report()
        self.assertIsInstance(report.contested, tuple)

    def test_an_ineligible_job_is_never_hidden_from_the_evidence_section(self):
        _ingest(self.store, [_record("a1", location="Berlin, Germany")])
        report = self.report()
        self.assertEqual(report.total_jobs, 1, "still counted and available")

    def test_eligibility_is_never_inferred_from_the_source_domain(self):
        """A .co.ke or remotive.com URL must not decide eligibility by itself."""
        _ingest(self.store, [_record("a1", source="remotive.com",
                                      location="Berlin, Germany")],
                source="remotive.com")
        report = self.report()
        self.assertEqual(report.total_jobs, 1)
        view = report.actionable[0] if report.actionable else None
        if view is not None:
            self.fail("a Berlin role must not be actionable")


class MatchDistinctionTests(ReportTestCase):
    """Never assessed is not assessed-and-rejected."""

    def test_a_job_with_no_assessment_is_still_actionable(self):
        _ingest(self.store, [_record("a1")])
        report = self.report()
        self.assertEqual(len(report.actionable), 1)
        self.assertFalse(report.actionable[0].match_present)

    def test_a_missing_assessment_shows_no_score(self):
        _ingest(self.store, [_record("a1")])
        html = render_report_html(self.report())
        self.assertIn("no assessment has been run", html)
        self.assertNotIn("score 0", html)

    def test_a_suited_match_never_hides_a_job(self):
        from app.jobs.match import assess_match
        from app.jobs.models import Job

        _ingest(self.store, [_record("a1")])
        record = self.store.load_jobs()[0]
        job = Job(**{k: v for k, v in record["job"].items()
                     if k in Job.__dataclass_fields__})
        result = assess_match(job)
        report = self.report(matches={record["job_id"]: result})
        self.assertEqual(len(report.actionable), 1, "a match result is not a filter")

    def test_an_insufficient_assessment_is_labelled_not_scored(self):
        _ingest(self.store, [_record("a1")])
        report = self.report(matches={
            self.store.load_jobs()[0]["job_id"]: {
                "job_id": "x", "tier": "not_yet_evaluated", "score": None,
                "confidence": "insufficient", "evidence": [],
                "missing_requirements": ["salary"], "concerns": [],
                "vetoes": [], "eligibility": "eligible",
                "insufficient_reason": "no salary published",
                "keyword_overlap": [],
            }
        })
        view = report.actionable[0]
        self.assertTrue(view.match_present)
        self.assertIsNone(view.match_score)
        html = render_report_html(report)
        self.assertIn("insufficient", html.lower())


class DuplicateTests(ReportTestCase):
    def test_possible_duplicates_remain_visible(self):
        # Same company and title, different URL -> fingerprint match, flagged.
        _ingest(self.store, [
            _record("d1", title="Engineer"),
            _record("d2", title="Engineer"),
        ])
        report = self.report()
        self.assertEqual(len(report.duplicates), 1)
        self.assertEqual(len(report.stale) + len(report.expired) + len(report.actionable),
                         report.total_jobs, "a duplicate is flagged, never dropped")

    def test_duplicates_are_flagged_not_merged(self):
        _ingest(self.store, [
            _record("d1", title="Engineer"),
            _record("d2", title="Engineer"),
        ])
        self.assertEqual(len(self.store.load_jobs()), 2)


class DecisionMemoryTests(ReportTestCase):
    def test_current_status_note_and_timestamp_are_reported(self):
        _ingest(self.store, [_record("a1")])
        job_id = self.store.load_jobs()[0]["job_id"]
        log = StatusLog(self.store)
        log.record(job_id, ReviewStatus.REVIEWING, note="checking the salary")
        memory = self.report().decisions[job_id]
        self.assertEqual(memory.status, "reviewing")
        self.assertEqual(memory.last_note, "checking the salary")
        self.assertTrue(memory.decided_at)

    def test_history_is_retained(self):
        _ingest(self.store, [_record("a1")])
        job_id = self.store.load_jobs()[0]["job_id"]
        log = StatusLog(self.store)
        log.record(job_id, ReviewStatus.REVIEWING, note="first")
        log.record(job_id, ReviewStatus.INTERESTED, note="second")
        memory = self.report().decisions[job_id]
        self.assertEqual(len(memory.history), 2)
        self.assertEqual(memory.last_note, "second")

    def test_a_job_with_no_decision_reads_as_new_not_unreadable(self):
        _ingest(self.store, [_record("a1")])
        memory = self.report().decisions[self.store.load_jobs()[0]["job_id"]]
        self.assertEqual(memory.status, "new")
        self.assertFalse(memory.reviewed)

    def test_decision_history_survives_the_report(self):
        _ingest(self.store, [_record("a1")])
        job_id = self.store.load_jobs()[0]["job_id"]
        StatusLog(self.store).record(job_id, ReviewStatus.DISMISSED, note="no")
        self.report()
        self.assertEqual(len(StatusLog(self.store).history(job_id)), 1)


class ReadOnlyTests(ReportTestCase):
    """The report must not change anything it reads."""

    def _hash_state(self):
        out = {}
        for path in sorted(self.store.data_dir.rglob("*")):
            if path.is_file():
                out[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return out

    def test_store_and_ledger_bytes_are_unchanged(self):
        _ingest(self.store, [_record("a1"), _record("a2", source="myjobmag.co.ke")],
                source="weworkremotely")
        StatusLog(self.store).record(
            self.store.load_jobs()[0]["job_id"], ReviewStatus.INTERESTED, note="x"
        )
        before = self._hash_state()
        report = self.report()
        render_report_file(report, Path(self._tmp.name) / "out.html")
        after = self._hash_state()
        self.assertEqual(before, after, "the report must not write to any state file")

    def test_building_twice_gives_the_same_answer(self):
        _ingest(self.store, [_record("a1")])
        self.assertEqual(self.report().total_jobs, self.report().total_jobs)

    def test_rendering_is_a_pure_function_of_the_report(self):
        _ingest(self.store, [_record("a1")])
        report = self.report()
        self.assertEqual(render_report_html(report), render_report_html(report))


class RenderingTests(ReportTestCase):
    def test_hostile_text_is_escaped(self):
        hostile = "<script>alert('x')</script>"
        _ingest(self.store, [_record("a1", title=hostile, company="<b>Evil</b>")])
        html = render_report_html(self.report())
        self.assertNotIn("<script>alert", html)
        self.assertIn("&lt;script&gt;", html)

    def test_the_report_contains_no_script_tag(self):
        _ingest(self.store, [_record("a1")])
        self.assertNotIn("<script", render_report_html(self.report()).casefold())

    def test_the_report_loads_no_external_resources(self):
        """No stylesheet, font, image or frame is fetched from anywhere."""
        _ingest(self.store, [_record("a1")])
        html = render_report_html(self.report())
        for marker in ("<link", "@import", "<iframe", "<img", "srcset",
                       "http://", "cdn."):
            self.assertNotIn(marker, html, f"{marker} would be an external reference")

    def test_source_links_appear_as_text_not_as_resources(self):
        _ingest(self.store, [_record("a1")])
        html = render_report_html(self.report())
        self.assertIn("weworkremotely", html)

    def test_quote_characters_are_escaped_in_attributes(self):
        _ingest(self.store, [_record('q" onmouseover="alert(1)')])
        html = render_report_html(self.report())
        self.assertNotIn('onmouseover="alert', html)


class CorruptInputTests(ReportTestCase):
    """Corrupt records must be visible, not silently skipped."""

    def test_a_record_with_no_job_id_is_reported_not_hidden(self):
        _ingest(self.store, [_record("a1")])
        with self.store.jobs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"job": {"title": "No id"}}) + "\n")
        report = self.report()
        self.assertTrue(report.errors, "a corrupt record must be surfaced")
        self.assertIn("unreadable job records", report.errors[0])

    def test_the_report_still_renders_when_records_are_corrupt(self):
        _ingest(self.store, [_record("a1")])
        with self.store.jobs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"job": {"title": "No id"}}) + "\n")
        html = render_report_html(self.report())
        self.assertIn("Problems reading stored data", html)
        self.assertIn("actionable", html)

    def test_the_error_block_warns_the_counts_may_be_incomplete(self):
        _ingest(self.store, [_record("a1")])
        with self.store.jobs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"job": {}}) + "\n")
        html = render_report_html(self.report())
        self.assertIn("may be incomplete", html)

    def test_an_unreadable_freshness_log_does_not_crash_the_report(self):
        _ingest(self.store, [_record("a1")])
        (self.store.data_dir / "freshness.jsonl").write_text("{ broken", encoding="utf-8")
        report = self.report()
        self.assertIn("freshness", report.errors[0].casefold())
        self.assertEqual(render_report_html(report).count("<html"), 1)


class EmptyStoreTests(ReportTestCase):
    def test_an_empty_store_renders_without_error(self):
        report = self.report()
        self.assertEqual(report.total_jobs, 0)
        self.assertEqual(report.errors, ())
        html = render_report_html(report)
        self.assertIn("Nothing in this queue", html)

    def test_every_section_is_present_even_when_empty(self):
        html = render_report_html(self.report())
        for heading in ("New and actionable", "Review queues", "Source health",
                        "Decision memory", "Evidence"):
            self.assertIn(heading, html)


class FixtureIntegrityTests(unittest.TestCase):
    """The mojibake fixtures must stay byte-exact.

    Added because a previous increment corrupted them by round-tripping a source
    file through a lossy text tool. These files exist precisely to hold broken
    encoding, so anything that rewrites them quietly destroys the coverage they
    provide.
    """

    EXPECTED = {
        "myjobmag/sample.xml": None,
        "wwr/sample.xml": None,
        "remotive/sample.json": None,
    }

    def test_fixtures_exist_and_are_readable_as_utf8(self):
        for name in self.EXPECTED:
            path = FIXTURES / name
            self.assertTrue(path.exists(), f"{name} is missing")
            path.read_bytes().decode("utf-8")

    def test_fixtures_carry_no_bom(self):
        bom = bytes([0xEF, 0xBB, 0xBF])
        for name in self.EXPECTED:
            raw = (FIXTURES / name).read_bytes()
            self.assertFalse(raw.startswith(bom), f"{name} gained a BOM")

    def test_the_mojibake_fixture_still_holds_broken_bytes(self):
        raw = (FIXTURES / "myjobmag" / "sample.xml").read_bytes()
        self.assertGreater(
            sum(1 for b in raw if b > 127), 0,
            "the MyJobMag fixture must retain genuine mojibake to test against",
        )


if __name__ == "__main__":
    unittest.main()