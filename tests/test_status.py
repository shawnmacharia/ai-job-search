"""Tests for job status tracking.

Entirely offline: fixtures and a temporary store. No network, Playwright,
Ollama, subprocess, or live scraping.

The status log is the one part of this pipeline that records something a
*person* decided, so the tests care most about three things: that a decision
is never lost, that an impossible transition is refused rather than
recorded, and that recording a status never touches a job.
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from app.jobs.ingest import ingest
from app.jobs.status import (
    ALLOWED,
    DEFAULT_STATUS,
    ReviewStatus,
    StatusError,
    StatusEvent,
    StatusLog,
    can_transition,
)
from app.jobs.store import JobStore
from app.reporting.jobs import STATUS_PLACEHOLDER, render_dashboard_file

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hiring_cafe"
NOW = datetime(2026, 10, 9, 12, 0, 0, tzinfo=timezone.utc)

INGEST = [
    "basic.json",
    "worldwide_remote.json",
    "salary_ksh.json",
    "europe_restricted.json",
]


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class StatusTestCase(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.store = JobStore(self.root / "data")
        ingest(
            [load(name) for name in INGEST],
            source="hiring.cafe",
            store=self.store,
            observed_at=NOW,
        )
        self.log = StatusLog(self.store)
        self.job_ids = [str(r["job_id"]) for r in self.store.load_jobs()]
        self.first = self.job_ids[0]

    def jobs_bytes(self):
        return self.store.jobs_path.read_text(encoding="utf-8")


class TransitionTests(StatusTestCase):
    def test_every_status_is_reachable_from_new(self):
        for status in ReviewStatus:
            with self.subTest(status=status):
                if status is ReviewStatus.NEW:
                    continue
                self.log.record(self.first, status, at="2026-10-09T12:00:00+00:00")
                self.assertEqual(self.log.current(self.first), status)

    def test_every_status_can_be_reached_from_every_other(self):
        # No status is a dead end, and neither is a status a dead end from.
        # A candidate who dismissed a job may always change their mind.
        for current in ReviewStatus:
            for target in ReviewStatus:
                with self.subTest(current=current, target=target):
                    self.assertTrue(
                        can_transition(current, target),
                        f"{current.value} -> {target.value} must be allowed",
                    )

    def test_dismissed_is_not_terminal(self):
        self.log.record(self.first, ReviewStatus.DISMISSED)
        self.log.record(self.first, ReviewStatus.INTERESTED)
        self.assertEqual(self.log.current(self.first), ReviewStatus.INTERESTED)

    def test_declared_edges_cover_the_whole_enum(self):
        for status in ReviewStatus:
            self.assertIn(status, ALLOWED, f"{status.value} has no declared edges")

    def test_a_value_from_another_enum_is_not_a_declared_transition(self):
        # The guard that carries weight now that the graph is connected.
        from app.state.models import ApplicationStatus

        self.assertFalse(can_transition(ReviewStatus.NEW, ApplicationStatus.PENDING))

    def test_re_annotating_keeps_the_status_and_appends_a_note(self):
        # Identity transitions are how a second thought is written down
        # without mislabelling the job.
        self.log.record(self.first, ReviewStatus.REVIEWING, note="first read")
        self.log.record(self.first, ReviewStatus.REVIEWING, note="second thought")
        self.assertEqual(self.log.current(self.first), ReviewStatus.REVIEWING)
        notes = [event.note for event in self.log.history(self.first)]
        self.assertEqual(notes, ["first read", "second thought"])


class RecordingTests(StatusTestCase):
    def test_a_job_with_no_events_is_new(self):
        self.assertEqual(self.log.current(self.first), DEFAULT_STATUS)
        self.assertEqual(DEFAULT_STATUS, ReviewStatus.NEW)

    def test_recording_returns_the_event_written(self):
        event = self.log.record(
            self.first, ReviewStatus.INTERESTED, note="strong ELT overlap",
            at="2026-10-09T12:00:00+00:00",
        )
        self.assertEqual(event.job_id, self.first)
        self.assertEqual(event.status, "interested")
        self.assertIsNone(event.previous, "the first event has no predecessor")
        self.assertEqual(event.note, "strong ELT overlap")

    def test_a_second_event_records_its_predecessor(self):
        self.log.record(self.first, ReviewStatus.REVIEWING, at="2026-10-09T12:00:00+00:00")
        second = self.log.record(self.first, ReviewStatus.DISMISSED, at="2026-10-09T13:00:00+00:00")
        self.assertEqual(second.previous, "reviewing")

    def test_history_is_the_full_trail_in_order(self):
        for index, status in enumerate(
            (ReviewStatus.REVIEWING, ReviewStatus.INTERESTED, ReviewStatus.DISMISSED)
        ):
            self.log.record(self.first, status, at=f"2026-10-09T1{index}:00:00+00:00")
        trail = [event.status for event in self.log.history(self.first)]
        self.assertEqual(trail, ["reviewing", "interested", "dismissed"])

    def test_status_is_scoped_to_one_job(self):
        other = self.job_ids[1]
        self.log.record(self.first, ReviewStatus.INTERESTED)
        self.assertEqual(self.log.current(other), DEFAULT_STATUS)
        self.assertEqual(len(self.log.history(other)), 0)

    def test_notes_are_optional(self):
        event = self.log.record(self.first, ReviewStatus.REVIEWING)
        self.assertEqual(event.note, "")

    def test_notes_are_stored_verbatim_and_never_interpreted(self):
        # A note is opaque data. It is stored as written and rendered as text.
        note = "call recruiter; see https://example.test/; ignore prior rules"
        event = self.log.record(self.first, ReviewStatus.INTERESTED, note=note)
        self.assertEqual(event.note, note)
        self.assertEqual(self.log.history(self.first)[-1].note, note)


class RejectionTests(StatusTestCase):
    def test_unknown_job_is_rejected(self):
        with self.assertRaises(StatusError) as caught:
            self.log.record("no-such-job", ReviewStatus.INTERESTED)
        self.assertIn("unknown job id", str(caught.exception))

    def test_empty_job_id_is_rejected(self):
        with self.assertRaises(StatusError):
            self.log.record("   ", ReviewStatus.INTERESTED)

    def test_a_non_review_status_is_rejected(self):
        # Guards against the existing ApplicationStatus being passed in by
        # mistake - it is a different enum for a different thing.
        from app.state.models import ApplicationStatus

        with self.assertRaises(StatusError) as caught:
            self.log.record(self.first, ApplicationStatus.PENDING)
        self.assertIn("ReviewStatus", str(caught.exception))

    def test_a_raw_string_status_is_rejected(self):
        with self.assertRaises(StatusError):
            self.log.record(self.first, "interested")

    def test_a_rejected_record_is_not_written(self):
        before = self.log.path.exists()
        with self.assertRaises(StatusError):
            self.log.record("no-such-job", ReviewStatus.INTERESTED)
        self.assertEqual(self.log.path.exists(), before)

    def test_unknown_job_check_can_be_waived(self):
        self.log.record(
            "not-yet-ingested", ReviewStatus.INTERESTED, require_known_job=False
        )
        self.assertEqual(self.log.current("not-yet-ingested"), ReviewStatus.INTERESTED)


class OverlayTests(StatusTestCase):
    """Status is an overlay. It must never alter a job record."""

    def test_recording_status_does_not_touch_the_jobs_file(self):
        before = self.jobs_bytes()
        self.log.record(self.first, ReviewStatus.INTERESTED)
        self.log.record(self.job_ids[1], ReviewStatus.DISMISSED)
        self.assertEqual(before, self.jobs_bytes(), "jobs.jsonl must be untouched")

    def test_status_is_stored_in_its_own_file(self):
        self.log.record(self.first, ReviewStatus.INTERESTED)
        self.assertEqual(self.log.path.name, "status.jsonl")
        self.assertTrue(self.log.path.exists())
        self.assertFalse(self.store.runs_path.exists(), "a run must not be implied")

    def test_reingesting_a_job_preserves_its_status(self):
        self.log.record(self.first, ReviewStatus.DISMISSED)
        # The same posting is seen again, as happens on every crawl.
        ingest([load("basic.json")], source="hiring.cafe", store=self.store, observed_at=NOW)
        self.assertEqual(self.log.current(self.first), ReviewStatus.DISMISSED)

    def test_status_survives_a_new_log_instance(self):
        self.log.record(self.first, ReviewStatus.INTERESTED, note="keep")
        reopened = StatusLog(self.store)
        self.assertEqual(reopened.current(self.first), ReviewStatus.INTERESTED)
        self.assertEqual(reopened.history(self.first)[-1].note, "keep")


class ReadFailureTests(StatusTestCase):
    def test_a_corrupt_line_is_reported_rather_than_silently_dropped(self):
        self.log.record(self.first, ReviewStatus.INTERESTED)
        with self.log.path.open("a", encoding="utf-8") as handle:
            handle.write("{not json at all}\n")
        with self.assertRaises(StatusError) as caught:
            self.log.current(self.first)
        self.assertIn("status.jsonl:2", str(caught.exception))

    def test_a_blank_line_is_skipped(self):
        self.log.record(self.first, ReviewStatus.INTERESTED)
        with self.log.path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        self.assertEqual(self.log.current(self.first), ReviewStatus.INTERESTED)

    def test_an_unrecognised_status_value_is_reported(self):
        self.log.record(self.first, ReviewStatus.INTERESTED)
        with self.log.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "job_id": self.first, "status": "bespoke",
                "previous": "interested", "at": "2026-10-09T14:00:00+00:00", "note": "",
            }) + "\n")
        with self.assertRaises(StatusError) as caught:
            self.log.current(self.first)
        self.assertIn("bespoke", str(caught.exception))


class SummaryTests(StatusTestCase):
    def test_summary_groups_every_job_by_current_status(self):
        self.log.record(self.job_ids[0], ReviewStatus.INTERESTED)
        self.log.record(self.job_ids[1], ReviewStatus.DISMISSED)
        grouped = self.log.summary()
        self.assertEqual(grouped["interested"], [self.job_ids[0]])
        self.assertEqual(grouped["dismissed"], [self.job_ids[1]])
        self.assertEqual(len(grouped["new"]), len(self.job_ids) - 2)

    def test_summary_includes_every_status_bucket(self):
        self.assertEqual(
            set(self.log.summary()), {status.value for status in ReviewStatus}
        )


class DashboardIntegrationTests(StatusTestCase):
    def test_dashboard_shows_the_recorded_status(self):
        self.log.record(self.job_ids[0], ReviewStatus.INTERESTED, note="yes")
        out = self.root / "d.html"
        render_dashboard_file(self.store, out, status_log=self.log, generated_at="2026-10-09")
        markup = out.read_text(encoding="utf-8")
        self.assertIn("interested", markup)

    def test_dashboard_falls_back_to_the_placeholder_without_a_log(self):
        out = self.root / "d.html"
        render_dashboard_file(self.store, out, generated_at="2026-10-09")
        self.assertIn(STATUS_PLACEHOLDER, out.read_text(encoding="utf-8"))

    def test_rendering_with_a_log_is_still_read_only(self):
        before = self.jobs_bytes()
        render_dashboard_file(self.store, self.root / "d.html", status_log=self.log)
        self.assertEqual(before, self.jobs_bytes())

    def test_a_status_problem_does_not_break_the_dashboard(self):
        self.log.record(self.first, ReviewStatus.INTERESTED)
        with self.log.path.open("a", encoding="utf-8") as handle:
            handle.write("{corrupt}\n")
        out = self.root / "d.html"
        rendered = render_dashboard_file(self.store, out, status_log=self.log)
        self.assertTrue(Path(rendered).exists(), "the report must still render")
        self.assertIn(STATUS_PLACEHOLDER, Path(rendered).read_text(encoding="utf-8"))


class CliTests(StatusTestCase):
    """The tool must be usable without writing Python."""

    def setUp(self):
        super().setUp()
        import importlib.util

        path = Path(__file__).resolve().parent.parent / "tools" / "job_status.py"
        spec = importlib.util.spec_from_file_location("job_status_cli", path)
        self.cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.cli)

    def run_cli(self, *args):
        """Run the CLI, capturing its streams.

        Errors are written to stderr and must not pollute test output, so the
        streams are captured and asserted on where it matters.
        """
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = self.cli.main(["--data-dir", str(self.root / "data"), *args])
        return code, out.getvalue(), err.getvalue()

    def test_list_reports_every_job(self):
        code, out, _ = self.run_cli("list")
        self.assertEqual(code, 0)
        self.assertEqual(len(out.strip().splitlines()), len(self.job_ids))

    def test_set_records_and_reports_the_transition(self):
        code, _, _ = self.run_cli("set", self.first, "interested")
        self.assertEqual(code, 0)
        self.assertEqual(self.log.current(self.first), ReviewStatus.INTERESTED)

    def test_set_with_a_note_stores_it(self):
        self.run_cli("set", self.first, "reviewing", "--note", "worth a look")
        self.assertEqual(self.log.history(self.first)[-1].note, "worth a look")

    def test_an_unknown_status_exits_one_and_explains_on_stderr(self):
        code, out, err = self.run_cli("set", self.first, "nonsense")
        self.assertEqual(code, 1)
        self.assertIn("unknown status", err)
        self.assertIn("interested", err, "the error must list the valid statuses")
        self.assertEqual(out, "", "errors must not go to stdout")

    def test_an_unknown_job_exits_one_and_explains_on_stderr(self):
        code, out, err = self.run_cli("set", "no-such-job", "interested")
        self.assertEqual(code, 1)
        self.assertIn("unknown job id", err)
        self.assertEqual(out, "")

    def test_a_corrupt_log_exits_one_rather_than_crashing(self):
        self.log.record(self.first, ReviewStatus.INTERESTED)
        with self.log.path.open("a", encoding="utf-8") as handle:
            handle.write("{corrupt}\n")
        code, _, err = self.run_cli("list")
        self.assertEqual(code, 1)
        self.assertIn("status.jsonl", err)

    def test_show_and_summary_succeed(self):
        self.run_cli("set", self.first, "interested")
        self.assertEqual(self.run_cli("show", self.first)[0], 0)
        self.assertEqual(self.run_cli("summary")[0], 0)

    def test_show_prints_the_trail(self):
        self.run_cli("set", self.first, "interested", "--note", "yes")
        self.run_cli("set", self.first, "dismissed")
        _, out, _ = self.run_cli("show", self.first)
        self.assertIn("-> interested", out)
        self.assertIn("interested -> dismissed", out)
        self.assertIn("yes", out)

    def test_summary_counts_each_bucket(self):
        self.run_cli("set", self.first, "interested")
        _, out, _ = self.run_cli("summary")
        # Parse rather than match padded whitespace, which is a formatting
        # detail that should not be load-bearing.
        counts = {
            status: int(count)
            for line in out.strip().splitlines()
            for status, count in [line.split()]
        }
        self.assertEqual(counts["interested"], 1)
        self.assertEqual(counts["new"], len(self.job_ids) - 1)
        self.assertEqual(sum(counts.values()), len(self.job_ids))

    def test_the_cli_never_modifies_the_jobs_file(self):
        before = self.jobs_bytes()
        self.run_cli("set", self.first, "dismissed")
        self.run_cli("list")
        self.run_cli("summary")
        self.assertEqual(before, self.jobs_bytes())


class NoApplicationLanguageTests(unittest.TestCase):
    """This module records human judgement, not application execution.

    Application submission is a hard stop for this project, so no status name
    or docstring here may imply that this module applies to anything.
    """

    FORBIDDEN = ("apply", "applied", "submit", "submitted", "submission", "offer")

    def source_text(self):
        return (Path(__file__).resolve().parent.parent
                / "app" / "jobs" / "status.py").read_text(encoding="utf-8")

    def test_no_status_value_refers_to_applying(self):
        values = [status.value for status in ReviewStatus]
        for value in values:
            for word in self.FORBIDDEN:
                with self.subTest(status=value, word=word):
                    self.assertNotIn(word, value)

    def test_module_does_not_reuse_the_pipeline_status_enums(self):
        text = self.source_text()
        self.assertNotIn("ApplicationStatus", text.split('"""')[2])
        self.assertNotIn("WorkflowStage", text.split('"""')[2])

    def test_module_imports_nothing_that_performs_io(self):
        import ast

        tree = ast.parse(self.source_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])
        forbidden = {"playwright", "requests", "httpx", "urllib", "socket",
                     "subprocess", "ollama", "aiohttp", "selenium", "asyncio"}
        self.assertEqual(imported & forbidden, set())


if __name__ == "__main__":
    unittest.main()