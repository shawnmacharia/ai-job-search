"""MyJobMag's one-request-per-day promise must survive closing the program.

The bug these tests exist for was invisible. The budget was seeded from
``ledger.attempts``, a list that starts empty in every new process, so the limit
held perfectly within a run and reset the moment the program exited. Nothing
raised, no existing test failed, and the next invocation made a second request
inside the window anyway.

For a source whose access was granted on the condition of a specific cadence,
that is the worst kind of failure: silent, and it spends access we were given.
So the assertions below are about *process boundaries*, not about counters.

No test here touches the network. Nothing in this file makes a live MyJobMag
request.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.jobs.runner import SourceSpec, run_sources
from app.jobs.store import JobStore
from app.sources.budget import DAY_SECONDS, DailyBudget, prior_stamps
from app.sources.myjobmag import DAILY_LIMIT, FEED_URL, MyjobmagAdapter, _seed_budget
from app.sources.transport import AccessError, AccessFetcher, Ledger, RateLimiter

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "myjobmag" / "sample.xml"


def fixture_body() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def recording_fetcher(path: Path, ledger: Ledger = None):
    """A fetcher whose ledger lives on disk, counting the URLs it is asked for."""
    calls: list = []

    def opener(url, timeout):
        from app.sources.transport import HttpResponse

        calls.append(url)
        return HttpResponse(
            url=url, status=200, headers={}, body=fixture_body(), elapsed_ms=1
        )

    return AccessFetcher(
        ledger=ledger or Ledger(path),
        limiter=RateLimiter(0.0, sleeper=lambda _: None),
        opener=opener,
        sleeper=lambda _: None,
        max_attempts=1,
    ), calls


def write_ledger(path: Path, entries) -> Path:
    """Write a ledger by hand, so tests control the recorded history exactly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return path


def attempt_row(at: str, *, purpose: str = "feed", source: str = "myjobmag.co.ke"):
    return {
        "source": source, "at": at, "url": FEED_URL, "purpose": purpose,
        "outcome": "ok", "status": 200, "duration_ms": 1, "error": "",
        "challenged": False,
    }


def hours_ago(hours: float) -> str:
    moment = datetime.now(timezone.utc) - timedelta(hours=hours)
    return moment.isoformat(timespec="seconds")


class FirstRequestTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "access_attempts.jsonl"

    def test_the_first_request_is_allowed(self):
        fetcher, calls = recording_fetcher(self.path)
        adapter = MyjobmagAdapter(fetcher)
        adapter.to_records()
        self.assertEqual(calls, [FEED_URL])
        self.assertEqual(adapter.requests_made, 1)

    def test_no_prior_ledger_means_nothing_has_been_recorded(self):
        """An absent ledger is not a failure; it is a first run."""
        fetcher, _ = recording_fetcher(self.path)
        self.assertTrue(MyjobmagAdapter(fetcher)._budget.allow())

    def test_an_empty_ledger_file_permits_the_first_request(self):
        write_ledger(self.path, [])
        fetcher, calls = recording_fetcher(self.path)
        MyjobmagAdapter(fetcher).to_records()
        self.assertEqual(len(calls), 1)


class SameProcessRefusalTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "access_attempts.jsonl"

    def test_a_second_request_in_the_same_process_is_refused(self):
        fetcher, _ = recording_fetcher(self.path)
        adapter = MyjobmagAdapter(fetcher)
        adapter.to_records()
        adapter.end_run()
        with self.assertRaises(AccessError) as caught:
            adapter.to_records()
        self.assertIn("daily request limit", str(caught.exception))

    def test_refusal_makes_zero_network_calls(self):
        fetcher, calls = recording_fetcher(self.path)
        adapter = MyjobmagAdapter(fetcher)
        adapter.to_records()
        adapter.end_run()
        before = len(calls)
        with self.assertRaises(AccessError):
            adapter.to_records()
        self.assertEqual(len(calls), before, "a refusal must not touch the network")

    def test_refusal_is_not_recorded_as_a_network_attempt(self):
        """A refusal sent nothing, so writing an attempt would be a false record."""
        fetcher, _ = recording_fetcher(self.path)
        adapter = MyjobmagAdapter(fetcher)
        adapter.to_records()
        adapter.end_run()
        attempts_before = len(Ledger(self.path).prior_attempts())
        with self.assertRaises(AccessError):
            adapter.to_records()
        self.assertEqual(len(Ledger(self.path).prior_attempts()), attempts_before)

    def test_a_refusal_is_still_reported_in_the_run_ledger(self):
        """The operator must be told the source did not run, and why."""
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp) / "data")
            fetcher, _ = recording_fetcher(self.path)
            adapter = MyjobmagAdapter(fetcher)
            adapter.to_records()
            adapter.end_run()
            run = run_sources(
                [SourceSpec(name="myjobmag.co.ke", fetch=adapter.to_records)],
                store=store,
            )
            outcome = run.sources[0]
            self.assertFalse(outcome.ok)
            self.assertEqual(outcome.fetched, 0)
            self.assertIn("daily request limit", outcome.error or "")


class NewProcessTests(unittest.TestCase):
    """The limit has to outlive the process, or it is not a limit."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "access_attempts.jsonl"

    def test_a_new_process_refuses_based_on_the_persisted_ledger(self):
        first = MyjobmagAdapter(recording_fetcher(self.path)[0])
        first.to_records()
        self.assertTrue(self.path.exists())

        # A separate Ledger object is what a separate process would build.
        fetcher, _ = recording_fetcher(self.path, ledger=Ledger(self.path))
        adapter = MyjobmagAdapter(fetcher)
        self.assertFalse(
            adapter._budget.allow(),
            "a new process must still see the spent budget",
        )

    def test_a_new_process_refuses_without_making_a_request(self):
        MyjobmagAdapter(recording_fetcher(self.path)[0]).to_records()
        fetcher, calls = recording_fetcher(self.path, ledger=Ledger(self.path))
        adapter = MyjobmagAdapter(fetcher)
        with self.assertRaises(AccessError):
            adapter.to_records()
        self.assertEqual(calls, [], "no request may be made after a restart")

    def test_the_persisted_attempt_is_still_readable(self):
        MyjobmagAdapter(recording_fetcher(self.path)[0]).to_records()
        self.assertGreaterEqual(len(Ledger(self.path).prior_attempts()), 1)


class WindowTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "access_attempts.jsonl"

    def test_a_request_after_the_window_is_allowed(self):
        write_ledger(self.path, [attempt_row(hours_ago(25))])
        fetcher, calls = recording_fetcher(self.path, ledger=Ledger(self.path))
        MyjobmagAdapter(fetcher).to_records()
        self.assertEqual(len(calls), 1)

    def test_a_request_just_inside_the_window_is_refused(self):
        write_ledger(self.path, [attempt_row(hours_ago(23))])
        fetcher, calls = recording_fetcher(self.path, ledger=Ledger(self.path))
        with self.assertRaises(AccessError):
            MyjobmagAdapter(fetcher).to_records()
        self.assertEqual(calls, [])

    def test_the_window_is_a_rolling_day_not_a_calendar_day(self):
        self.assertEqual(DAY_SECONDS, 86400.0)

    def test_a_stale_attempt_outside_the_window_is_forgotten(self):
        write_ledger(self.path, [attempt_row(hours_ago(48))])
        fetcher, _ = recording_fetcher(self.path, ledger=Ledger(self.path))
        self.assertTrue(MyjobmagAdapter(fetcher)._budget.allow())


class SourceIsolationTests(unittest.TestCase):
    """A shared ledger file must not let one source spend another's budget."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "access_attempts.jsonl"

    def test_unrelated_source_attempts_do_not_consume_the_budget(self):
        write_ledger(self.path, [
            attempt_row(hours_ago(1), purpose="api", source="remotive.com"),
            attempt_row(hours_ago(1), purpose="robots", source="weworkremotely"),
            attempt_row(hours_ago(1), purpose="terms", source="myjobmag.co.ke"),
        ])
        fetcher, calls = recording_fetcher(self.path, ledger=Ledger(self.path))
        MyjobmagAdapter(fetcher).to_records()
        self.assertEqual(len(calls), 1, "only a feed attempt may consume the budget")

    def test_myjobmag_attempts_still_consume_the_budget(self):
        write_ledger(self.path, [attempt_row(hours_ago(1))])
        fetcher, calls = recording_fetcher(self.path, ledger=Ledger(self.path))
        with self.assertRaises(AccessError):
            MyjobmagAdapter(fetcher).to_records()
        self.assertEqual(calls, [])


class MalformedLedgerTests(unittest.TestCase):
    """Unreadable history must refuse, not assume compliance."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "access_attempts.jsonl"

    def test_a_truncated_line_makes_the_ledger_untrusted(self):
        write_ledger(self.path, [attempt_row(hours_ago(1))])
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write('{"source": "myjobmag.co.ke", "at": "2026-0')  # torn write
        fetcher, calls = recording_fetcher(self.path, ledger=Ledger(self.path))
        adapter = MyjobmagAdapter(fetcher)
        self.assertTrue(adapter._budget.untrusted)
        self.assertFalse(adapter._budget.allow())

    def test_an_untrusted_budget_refuses_without_making_a_request(self):
        self.path.write_text("{not json at all", encoding="utf-8")
        fetcher, calls = recording_fetcher(self.path, ledger=Ledger(self.path))
        with self.assertRaises(AccessError) as caught:
            MyjobmagAdapter(fetcher).to_records()
        self.assertEqual(calls, [])
        self.assertIn("could not be read", str(caught.exception))

    def test_an_unreadable_timestamp_counts_as_recent(self):
        """A time we cannot read cannot be shown to be old, so it must count."""
        write_ledger(self.path, [{
            "source": "myjobmag.co.ke", "at": "not-a-timestamp",
            "url": FEED_URL, "purpose": "feed", "outcome": "ok",
        }])
        fetcher, calls = recording_fetcher(self.path, ledger=Ledger(self.path))
        with self.assertRaises(AccessError):
            MyjobmagAdapter(fetcher).to_records()
        self.assertEqual(calls, [])

    def test_a_row_missing_its_timestamp_is_untrusted(self):
        write_ledger(self.path, [{
            "source": "myjobmag.co.ke", "url": FEED_URL, "purpose": "feed",
            "outcome": "ok",
        }])
        fetcher, _ = recording_fetcher(self.path, ledger=Ledger(self.path))
        self.assertTrue(MyjobmagAdapter(fetcher)._budget.untrusted)

    def test_a_json_array_line_is_untrusted(self):
        self.path.write_text("[1, 2, 3]\n", encoding="utf-8")
        fetcher, _ = recording_fetcher(self.path, ledger=Ledger(self.path))
        self.assertTrue(MyjobmagAdapter(fetcher)._budget.untrusted)

    def test_an_in_memory_ledger_is_not_untrusted(self):
        """No file means nothing was ever written, so there is nothing to distrust."""
        fetcher, _ = recording_fetcher(Path("unused-never-written.jsonl"),
                                       ledger=Ledger())
        self.assertFalse(MyjobmagAdapter(fetcher)._budget.untrusted)


class SharedBudgetTests(unittest.TestCase):
    """The rule lives in one place, so the sources cannot drift apart."""

    def test_the_daily_limit_is_one(self):
        self.assertEqual(DAILY_LIMIT, 1)

    def test_an_untrusted_budget_refuses_everything(self):
        budget = DailyBudget(1, untrusted=True)
        self.assertFalse(budget.allow())
        budget.record()
        self.assertFalse(budget.allow(), "the refusal must latch")

    def test_prior_stamps_returns_none_for_unreadable_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.jsonl"
            path.write_text("{{{", encoding="utf-8")
            self.assertIsNone(prior_stamps(Ledger(path), purpose="feed"))

    def test_prior_stamps_returns_a_list_for_readable_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_ledger(Path(tmp) / "attempts.jsonl",
                                [attempt_row(hours_ago(2))])
            stamps = prior_stamps(Ledger(path), purpose="feed")
            self.assertIsNotNone(stamps)
            self.assertEqual(len(stamps), 1)

    def test_seeding_from_an_unreadable_ledger_produces_a_refusing_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.jsonl"
            path.write_text("not json", encoding="utf-8")
            self.assertTrue(_seed_budget(Ledger(path)).untrusted)


class PreservationTests(unittest.TestCase):
    """The fix must not disturb anything it was not asked to touch."""

    def test_the_approved_endpoint_is_unchanged(self):
        self.assertEqual(FEED_URL, "https://www.myjobmag.co.ke/jobsxml_by_categories.xml")

    def test_attribution_is_unchanged(self):
        from app.sources.myjobmag import ATTRIBUTION_REQUIRED, ATTRIBUTION_TEXT

        self.assertTrue(ATTRIBUTION_REQUIRED)
        self.assertEqual(ATTRIBUTION_TEXT, "Listing from MyJobMag")

    def test_the_access_evidence_still_states_the_scope(self):
        fetcher, _ = recording_fetcher(Path("unused.jsonl"), ledger=Ledger())
        blob = " ".join(
            f"{k} {v}" for k, v in MyjobmagAdapter(fetcher).access_evidence().items()
        ).casefold()
        for phrase in ("1 request per day", "attribution_required", "html scraping"):
            self.assertIn(phrase, blob)

    def test_the_recorded_access_decision_is_untouched(self):
        from app.sources.access import AccessLevel, load_decisions

        decisions = load_decisions(Path(__file__).resolve().parents[1] / "data")
        decision = decisions.get("myjobmag.co.ke")
        if decision is not None:
            self.assertIs(decision.level, AccessLevel.PERMITTED)


if __name__ == "__main__":
    unittest.main()