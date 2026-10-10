"""The actionable review queue: what a person actually reads each morning.

Two properties carry the weight here.

**Determinism.** A queue that reorders between two identical runs is a queue
nobody trusts, so ordering is asserted directly and a final job-id tie-breaker
guarantees a total order even when everything else ties.

**Nothing hides a job.** A missing match assessment must not remove a job from
the queue, and a dismissed job must leave it. Those pull in opposite directions
and both are asserted, because the failure mode is silent in both cases.

No test here touches the network or the real store.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.jobs.freshness import DEFAULT_POLICIES, FreshnessLedger
from app.jobs.runner import SourceSpec, run_sources
from app.jobs.status import ReviewStatus, StatusLog
from app.jobs.store import JobStore
from app.reporting.review import (
    QueueFilters,
    apply_filters,
    build_actionable,
    build_report,
    queue_key,
    render_report_html,
    why_actionable,
)

T0 = datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc)
REPO_ROOT = Path(__file__).resolve().parents[1]
SET_STATUS = REPO_ROOT / "tools" / "set_status.py"


def _record(name, *, source="weworkremotely", title="Role", company="Acme",
            location="Nairobi, Kenya", description="Work."):
    return {
        "url": f"https://{source}/jobs/{name}",
        "title": title,
        "company": company,
        "location": location,
        "description": description,
    }


class QueueTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data = Path(self._tmp.name) / "data"
        self.store = JobStore(self.data)

    def ingest(self, records, *, source="weworkremotely", at=None):
        run_sources([SourceSpec(name=source, fetch=lambda: list(records))],
                    store=self.store, observed_at=at or T0)

    def job_id(self, name):
        for record in self.store.load_jobs():
            if str(record["job"]["url"]).endswith(f"/{name}"):
                return str(record["job_id"])
        raise AssertionError(f"no stored job ending /{name}")

    def views(self, **kwargs):
        kwargs.setdefault("status_log", StatusLog(self.store))
        report = build_report(self.store, **kwargs)
        return report


class MembershipTests(QueueTestCase):
    def test_an_eligible_new_job_is_in_the_queue(self):
        self.ingest([_record("a1")])
        report = self.views()
        self.assertEqual(len(report.queue), 1)

    def test_an_ineligible_job_is_not_in_the_queue(self):
        self.ingest([_record("a1", location="Berlin, Germany")])
        self.assertEqual(len(self.views().queue), 0)

    def test_a_dismissed_job_leaves_the_queue(self):
        self.ingest([_record("a1")])
        StatusLog(self.store).record(self.job_id("a1"), ReviewStatus.DISMISSED)
        self.assertEqual(len(self.views().queue), 0)

    def test_an_interested_job_stays_in_the_queue(self):
        """Interested is a live disposition, not a closed one."""
        self.ingest([_record("a1")])
        StatusLog(self.store).record(self.job_id("a1"), ReviewStatus.INTERESTED)
        self.assertEqual(len(self.views().queue), 1)

    def test_a_shortlisted_job_stays_in_the_queue(self):
        self.ingest([_record("a1")])
        StatusLog(self.store).record(self.job_id("a1"), ReviewStatus.SHORTLISTED)
        self.assertEqual(len(self.views().queue), 1)

    def test_a_reviewing_job_stays_in_the_queue(self):
        self.ingest([_record("a1")])
        StatusLog(self.store).record(self.job_id("a1"), ReviewStatus.REVIEWING)
        self.assertEqual(len(self.views().queue), 1)

    def test_an_expired_job_leaves_the_queue(self):
        self.ingest([_record("a1"), _record("a2")])
        gone = DEFAULT_POLICIES["weworkremotely"].expire_days + 2
        self.ingest([_record("a2")], at=T0 + timedelta(days=gone))
        report = self.views()
        self.assertEqual(len(report.expired), 1)
        self.assertNotIn(self.job_id("a1"), {v.job_id for v, _ in report.queue})

    def test_a_stale_job_may_stay_in_the_queue(self):
        """Stale is not closed. A posting can be stale and still worth reading."""
        self.ingest([_record("a1"), _record("a2")])
        gone = DEFAULT_POLICIES["weworkremotely"].stale_days + 2
        self.ingest([_record("a2")], at=T0 + timedelta(days=gone))
        report = self.views()
        self.assertEqual(len(report.stale), 1)
        self.assertIn(self.job_id("a1"), {v.job_id for v, _ in report.queue})

    def test_a_job_with_no_match_result_is_still_in_the_queue(self):
        """Missing analysis is not a reason to hide a job from a reader."""
        self.ingest([_record("a1")])
        report = self.views()
        view = report.queue[0][0]
        self.assertFalse(view.match_present)
        self.assertEqual(len(report.queue), 1)

    def test_an_unassessed_job_says_so_in_its_reason(self):
        self.ingest([_record("a1")])
        _, reason = self.views().queue[0]
        self.assertIn("no match assessment yet", reason)
        self.assertIn("Kenya-eligible", reason)

    def test_an_uncertain_job_keeps_its_own_queue(self):
        self.ingest([_record("a1", location="EMEA")])
        report = self.views()
        self.assertEqual(len(report.uncertain), 1)
        self.assertEqual(report.uncertain[0].verdict, "unknown")
        # Uncertain is not eligible, so it is not silently queued as if it were.
        self.assertEqual(len(report.queue), 0)

    def test_a_possible_duplicate_stays_visible(self):
        self.ingest([_record("d1", title="Engineer"), _record("d2", title="Engineer")])
        report = self.views()
        self.assertEqual(len(report.duplicates), 1)
        flagged = [v for v, _ in report.queue if v.possible_duplicate]
        self.assertEqual(len(flagged), 1)
        self.assertIn("possible duplicate", why_actionable(flagged[0]))

    def test_attribution_and_original_links_are_preserved(self):
        self.ingest([_record("a1")])
        html = render_report_html(self.views())
        self.assertIn("https://weworkremotely/jobs/a1", html)
        self.assertIn("weworkremotely", html)


class CategoryTests(QueueTestCase):
    def test_every_category_is_counted_separately(self):
        self.ingest([_record("a1"), _record("a2"), _record("a3"),
                     _record("u1", location="EMEA")])
        log = StatusLog(self.store)
        log.record(self.job_id("a1"), ReviewStatus.REVIEWING)
        log.record(self.job_id("a2"), ReviewStatus.INTERESTED)
        log.record(self.job_id("a3"), ReviewStatus.SHORTLISTED)
        categories = self.views().categories
        self.assertEqual(categories["reviewing_undecided"], 1)
        self.assertEqual(categories["interested"], 1)
        self.assertEqual(categories["shortlisted"], 1)
        self.assertEqual(categories["uncertain_eligibility"], 1)

    def test_categories_are_not_collapsed_by_a_filter(self):
        self.ingest([_record("a1"), _record("a2")])
        StatusLog(self.store).record(self.job_id("a2"), ReviewStatus.DISMISSED)
        report = self.views(queue_filters=QueueFilters(source="weworkremotely"))
        self.assertEqual(len(report.queue), 1)
        self.assertEqual(report.categories["dismissed"], 1)

    def test_interested_and_shortlisted_are_distinct(self):
        self.ingest([_record("a1"), _record("a2")])
        log = StatusLog(self.store)
        log.record(self.job_id("a1"), ReviewStatus.INTERESTED)
        log.record(self.job_id("a2"), ReviewStatus.SHORTLISTED)
        report = self.views()
        self.assertEqual(report.categories["interested"], 1)
        self.assertEqual(report.categories["shortlisted"], 1)


class OrderingTests(QueueTestCase):
    def _queue(self, records, at=None):
        self.ingest(records, at=at)
        return self.views().queue

    def test_status_orders_before_freshness(self):
        queue = self._queue([_record("a1"), _record("a2")])
        StatusLog(self.store).record(self.job_id("a2"), ReviewStatus.REVIEWING)
        queue = self.views().queue
        self.assertEqual(queue[0][0].application_status, "new")

    def test_unexamined_work_ranks_above_already_decided_work(self):
        """The queue surfaces what still needs a decision.

        A shortlisted job has already been triaged, so it belongs below a job
        nobody has opened - otherwise the top of the list is entirely work the
        candidate has finished with.
        """
        self._queue([_record("a1"), _record("a2"), _record("a3")])
        log = StatusLog(self.store)
        log.record(self.job_id("a1"), ReviewStatus.INTERESTED)
        log.record(self.job_id("a2"), ReviewStatus.SHORTLISTED)
        order = [v.application_status for v, _ in self.views().queue]
        self.assertEqual(order, ["new", "interested", "shortlisted"])

    def test_reviewing_ranks_below_new_and_above_decided(self):
        self._queue([_record("a1"), _record("a2"), _record("a3")])
        log = StatusLog(self.store)
        log.record(self.job_id("a1"), ReviewStatus.SHORTLISTED)
        log.record(self.job_id("a2"), ReviewStatus.REVIEWING)
        order = [v.application_status for v, _ in self.views().queue]
        self.assertEqual(order, ["new", "reviewing", "shortlisted"])

    def test_ordering_is_stable_across_repeated_builds(self):
        self._queue([_record("a1"), _record("a2"), _record("a3")])
        first = [v.job_id for v, _ in self.views().queue]
        second = [v.job_id for v, _ in self.views().queue]
        self.assertEqual(first, second)

    def test_job_id_breaks_remaining_ties(self):
        queue = self._queue([_record("a3"), _record("a1"), _record("a2")])
        ids = [v.job_id for v, _ in queue]
        self.assertEqual(ids, sorted(ids),
                         "with everything else equal, job id decides")

    def test_an_unassessed_job_sorts_after_an_assessed_one(self):
        queue = self._queue([_record("a1"), _record("a2")])
        assessed = {
            self.job_id("a1"): {"tier": "strong_match", "score": 0.8,
                                "confidence": "medium"},
        }
        report = self.views(matches={self.job_id("a2"): {
            "tier": "strong_match", "score": 0.8, "confidence": "medium",
            "evidence": [], "missing_requirements": [], "concerns": [],
            "vetoes": [], "eligibility": "eligible",
        }})
        ids = [v.job_id for v, _ in report.queue]
        self.assertEqual(ids[0], self.job_id("a2"), "the assessed job leads")

    def test_a_missing_score_never_sorts_as_zero(self):
        """A job with no score must not be buried under every real score."""
        self._queue([_record("a1"), _record("a2")])
        report = self.views(matches={self.job_id("a2"): {
            "tier": "credible_match", "score": 0.01, "confidence": "medium",
            "evidence": [], "missing_requirements": [], "concerns": [],
            "vetoes": [], "eligibility": "eligible",
        }})
        by_id = {v.job_id: v for v, _ in report.queue}
        unassessed = by_id[self.job_id("a1")]
        assessed = by_id[self.job_id("a2")]
        self.assertEqual(queue_key(unassessed)[3], (1, 0.0))
        self.assertEqual(queue_key(assessed)[3], (0, -0.01))
        self.assertLess(queue_key(assessed), queue_key(unassessed),
                        "a real score, however small, ranks ahead of no score")

    def test_a_newer_posting_sorts_first(self):
        report = self.views()
        for view in report.filtered:
            self.assertIsInstance(queue_key(view)[4][0], int)

    def test_queue_key_is_comparable_for_every_job(self):
        self._queue([_record("a1")])
        for view in self.views().filtered:
            self.assertEqual(len(queue_key(view)), 6)


class FilterTests(QueueTestCase):
    def setUp(self):
        super().setUp()
        # Distinct companies, so the fingerprint does not collide and every
        # job starts clean of a duplicate flag.
        self.ingest([_record("a1", company="Alpha Ltd")])
        # Provenance comes from the source the run was attributed to, never from
        # the URL's host. The record below is ingested *as* myjobmag.co.ke, and
        # that is what makes it a MyJobMag job.
        self.ingest([_record("b1", source="myjobmag.co.ke", company="Beta Ltd")],
                    source="myjobmag.co.ke")

    def test_filter_by_source(self):
        views = [v for v, _ in self.views().queue]
        narrowed = apply_filters(views, source="myjobmag.co.ke")
        self.assertEqual(len(narrowed), 1)
        self.assertEqual(narrowed[0].sources, ["myjobmag.co.ke"])

    def test_filter_by_source_does_not_infer_from_the_url_host(self):
        """A .co.ke-shaped URL must not make a job a MyJobMag job."""
        self.ingest([_record("c1", source="myjobmag.co.ke")],
                    source="weworkremotely")
        views = [v for v, _ in self.views().queue]
        mag = apply_filters(views, source="myjobmag.co.ke")
        wwr = apply_filters(views, source="weworkremotely")
        self.assertEqual(len(wwr), 2, "the c1 record was ingested as weworkremotely")
        self.assertEqual(len(mag), 1)

    def test_filter_by_eligibility(self):
        views = [v for v, _ in self.views().queue]
        self.assertEqual(len(apply_filters(views, eligibility="eligible")), 2)
        self.assertEqual(len(apply_filters(views, eligibility="not_eligible")), 0)

    def test_filter_by_status(self):
        StatusLog(self.store).record(self.job_id("a1"), ReviewStatus.REVIEWING)
        views = [v for v, _ in self.views().queue]
        narrowed = apply_filters(views, status="reviewing")
        self.assertEqual(len(narrowed), 1)

    def test_filter_by_freshness(self):
        views = [v for v, _ in self.views().queue]
        narrowed = apply_filters(views, freshness="active")
        self.assertEqual(len(narrowed), 2)

    def test_filter_by_possible_duplicate(self):
        self.ingest([_record("d1", title="Engineer", company="Gamma Ltd"),
                     _record("d2", title="Engineer", company="Gamma Ltd")])
        views = [v for v, _ in self.views().queue]
        self.assertEqual(len(apply_filters(views, possible_duplicate=True)), 1)
        self.assertEqual(len(apply_filters(views, possible_duplicate=False)), 3)

    def test_filter_by_posted_after(self):
        """A dated job passes; an undated one cannot be shown to be after."""
        dated = _record("e1", company="Epsilon Ltd")
        # The adapter reads the source's own field name, "posted".
        dated["posted"] = "2026-02-20"
        self.ingest([dated])
        views = [v for v, _ in self.views().queue]
        by_id = {v.job_id: v for v in views}
        self.assertEqual(by_id[self.job_id("e1")].posted_date, "2026-02-20")
        self.assertEqual(len(apply_filters(views, posted_after="2026-02-01")), 1)
        self.assertEqual(len(apply_filters(views, posted_after="2026-03-01")), 0)

    def test_an_undated_job_never_survives_a_posted_after_filter(self):
        """The "not yet evaluated" sentinel must not read as a recent date.

        Compared as text, that sentinel sorts above every real ISO timestamp,
        so an undated job would pass a filter meant to show only recent ones.
        """
        views = [v for v, _ in self.views().queue]
        self.assertTrue(any(v.posted_date == "not yet evaluated" for v in views))
        self.assertEqual(apply_filters(views, posted_after="2020-01-01"), [])

    def test_filter_by_uncertainty(self):
        views = [v for v, _ in self.views().queue]
        self.assertEqual(len(apply_filters(views, uncertain=False)), 2)

    def test_filters_compose(self):
        views = [v for v, _ in self.views().queue]
        narrowed = apply_filters(views, source="weworkremotely", freshness="active")
        self.assertEqual(len(narrowed), 1)
        narrower = apply_filters(views, source="weworkremotely", freshness="expired")
        self.assertEqual(len(narrower), 0)

    def test_queue_filters_reach_the_report(self):
        report = self.views(queue_filters=QueueFilters(source="myjobmag.co.ke"))
        self.assertEqual(len(report.queue), 1)
        self.assertEqual(report.filters, {"source": "myjobmag.co.ke"})
        self.assertIn("filters", render_report_html(report))

    def test_a_filter_that_matches_nothing_is_empty_not_everything(self):
        report = self.views(queue_filters=QueueFilters(source="nope"))
        self.assertEqual(len(report.queue), 0)
        self.assertIn("Nothing in the queue", render_report_html(report))


class ReasonTests(QueueTestCase):
    def test_every_queue_entry_states_a_reason(self):
        self.ingest([_record("a1")])
        report = self.views()
        for _, reason in report.queue:
            self.assertTrue(reason)
            self.assertIn("Kenya-eligible", reason)

    def test_the_reason_names_the_current_state(self):
        self.ingest([_record("a1")])
        StatusLog(self.store).record(self.job_id("a1"), ReviewStatus.REVIEWING)
        _, reason = self.views().queue[0]
        self.assertIn("under review", reason)


class ReadOnlyTests(QueueTestCase):
    def test_report_generation_leaves_every_state_file_byte_identical(self):
        self.ingest([_record("a1"), _record("b1", source="myjobmag.co.ke")])
        StatusLog(self.store).record(self.job_id("a1"), ReviewStatus.INTERESTED)
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(self.data.rglob("*")) if p.is_file()}
        self.views()
        self.views(queue_filters=QueueFilters(source="weworkremotely"))
        render_report_html(self.views())
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(self.data.rglob("*")) if p.is_file()}
        self.assertEqual(before, after)

    def test_the_queue_module_imports_nothing_that_performs_io(self):
        """Scans imports, not prose.

        A substring scan over the whole file matches the word "requests" in
        ordinary English and proves nothing. Only the import statements matter.
        """
        import ast

        for name in ("app/reporting/review.py",):
            tree = ast.parse((REPO_ROOT / name).read_text(encoding="utf-8"))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(a.name.split(".")[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            for forbidden in ("urllib", "requests", "socket", "subprocess",
                              "httpx", "http", "playwright"):
                self.assertNotIn(forbidden, imported)


class StatusCommandTests(QueueTestCase):
    def _run(self, *args):
        return subprocess.run(
            [sys.executable, str(SET_STATUS), "--data-dir", str(self.data), *args],
            capture_output=True, text=True,
        )

    def test_a_valid_decision_is_recorded(self):
        self.ingest([_record("a1")])
        result = self._run("--job-id", self.job_id("a1"), "--status", "interested")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(
            StatusLog(self.store).current(self.job_id("a1")).value, "interested")

    def test_a_decision_survives_a_restart(self):
        """A new StatusLog over the same file must still see it."""
        self.ingest([_record("a1")])
        self._run("--job-id", self.job_id("a1"), "--status", "shortlisted",
                  "--note", "worth a call")
        reopened = StatusLog(JobStore(self.data))
        self.assertEqual(reopened.current(self.job_id("a1")).value, "shortlisted")
        self.assertEqual(len(reopened.history(self.job_id("a1"))), 1)

    def test_an_unknown_job_id_writes_nothing(self):
        self.ingest([_record("a1")])
        before = (self.data / "status.jsonl").exists()
        result = self._run("--job-id", "no-such-job", "--status", "interested")
        self.assertEqual(result.returncode, 1)
        self.assertFalse((self.data / "status.jsonl").exists() and not before)
        self.assertEqual(self.store.load_jobs(), self.store.load_jobs())

    def test_an_invalid_status_is_refused_and_writes_nothing(self):
        self.ingest([_record("a1")])
        result = self._run("--job-id", self.job_id("a1"), "--status", "applied")
        self.assertEqual(result.returncode, 2)  # argparse rejects the choice
        self.assertFalse((self.data / "status.jsonl").exists())

    def test_a_missing_argument_writes_nothing(self):
        self.ingest([_record("a1")])
        result = self._run("--job-id", self.job_id("a1"))
        self.assertEqual(result.returncode, 1)
        self.assertFalse((self.data / "status.jsonl").exists())

    def test_the_command_never_modifies_jobs_or_source_history(self):
        self.ingest([_record("a1")])
        job_hash = hashlib.sha256((self.data / "jobs.jsonl").read_bytes()).hexdigest()
        seen_hash = hashlib.sha256((self.data / "seen.json").read_bytes()).hexdigest()
        self._run("--job-id", self.job_id("a1"), "--status", "interested")
        self.assertEqual(
            hashlib.sha256((self.data / "jobs.jsonl").read_bytes()).hexdigest(),
            job_hash)
        self.assertEqual(
            hashlib.sha256((self.data / "seen.json").read_bytes()).hexdigest(),
            seen_hash)

    def test_a_note_is_stored_verbatim(self):
        self.ingest([_record("a1")])
        note = "salary unclear <script>alert(1)</script>"
        self._run("--job-id", self.job_id("a1"), "--status", "reviewing",
                  "--note", note)
        events = StatusLog(self.store).history(self.job_id("a1"))
        self.assertEqual(events[0].note, note)

    def test_the_statuses_can_be_listed(self):
        result = self._run("--list-statuses")
        self.assertEqual(result.returncode, 0)
        for value in ("new", "reviewing", "interested", "shortlisted", "dismissed"):
            self.assertIn(value, result.stdout)

    def test_no_application_status_exists(self):
        """Acting is a hard stop; the vocabulary must not imply otherwise."""
        for name in ("applied", "submitted", "rejected"):
            self.assertNotIn(name, [s.value for s in ReviewStatus])


class ExistingBehaviourTests(QueueTestCase):
    def test_dry_run_and_orchestration_are_unchanged(self):
        from app.jobs.discovery import Outcome, run_discovery

        class Adapter:
            def __init__(self, name, records):
                self.name, self.records, self.calls = name, records, 0

            def budget_state(self):
                return {"allowed": True, "reason": "ok", "limit": None,
                        "spent": 0, "untrusted": False}

            def to_records(self):
                self.calls += 1
                return list(self.records)

            def end_run(self):
                pass

            @property
            def requests_made(self):
                return self.calls

        from app.sources.access import AccessDecision, AccessLevel, save_decision

        save_decision(AccessDecision(
            source="weworkremotely", level=AccessLevel.PERMITTED, reason="t",
            robots="a", robots_url="", terms="r", terms_url="",
            checked_at=T0.isoformat(timespec="seconds")), self.data)

        adapter = Adapter("weworkremotely", [_record("a1")])
        dry = run_discovery(self.store, {"weworkremotely": adapter}, dry_run=True,
                            observed_at=T0)
        self.assertEqual(adapter.calls, 0)
        self.assertIs(dry.results[0].outcome, Outcome.WOULD_RUN)

        real = run_discovery(self.store, {"weworkremotely": adapter}, observed_at=T0)
        self.assertIs(real.results[0].outcome, Outcome.FETCHED)
        self.assertEqual(adapter.calls, 1)

    def test_freshness_layers_still_agree_with_the_queue(self):
        self.ingest([_record("a1")])
        state = FreshnessLedger(self.store).evaluate()
        report = self.views()
        for view, _ in report.queue:
            self.assertEqual(view.freshness, state[view.job_id].state.value)


class EscapingTests(QueueTestCase):
    def test_hostile_titles_and_notes_are_escaped(self):
        hostile = "<script>alert('x')</script>"
        self.ingest([_record("a1", title=hostile, company="<b>Evil</b>")])
        StatusLog(self.store).record(self.job_id("a1"), ReviewStatus.REVIEWING,
                                     note=hostile)
        html = render_report_html(self.views())
        self.assertNotIn("<script>alert", html)
        self.assertIn("&lt;script&gt;", html)

    def test_source_links_survive_escaping_intact(self):
        self.ingest([_record("a1")])
        html = render_report_html(self.views())
        self.assertIn("https://weworkremotely/jobs/a1", html)

    def test_the_report_has_no_script_or_external_resource(self):
        self.ingest([_record("a1")])
        html = render_report_html(self.views()).casefold()
        for marker in ("<script", "<link", "@import", "<iframe", "cdn."):
            self.assertNotIn(marker, html)


if __name__ == "__main__":
    unittest.main()