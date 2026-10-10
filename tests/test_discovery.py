"""Daily discovery orchestration: run every permitted source, then report.

The property under test throughout is that a source which cannot be consulted is
*not consulted*. Not "consulted and caught" - that would spend a request to
discover a limit we already know. So the network-call counters in these tests
are the real assertion, and they are checked before the outcome labels are.

The five outcomes are kept apart deliberately. Merging ``budget_refused`` into
``failed`` would make a run that honours every agreed limit exit non-zero, which
teaches an operator to ignore the exit code - the opposite of what it is for.

Every test is offline. Adapters are fakes with their own call counters, and no
test opens a socket.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app.jobs.discovery import (
    SKIP_BUDGET_REFUSED,
    DiscoveryResult,
    Outcome,
    SourceResult,
    build_plan,
    run_discovery,
)
from app.jobs.freshness import FreshnessLedger
from app.jobs.runner import SKIP_ACCESS_UNKNOWN
from app.jobs.status import StatusLog
from app.jobs.store import JobStore
from app.sources.access import AccessDecision, AccessLevel, save_decision

T0 = datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc)


def _record(name, source, *, title="Role", company="Acme", location="Nairobi, Kenya"):
    return {
        "url": f"https://{source}/jobs/{name}",
        "title": title,
        "company": company,
        "location": location,
        "description": "Work.",
    }


class FakeAdapter:
    """Stands in for a real source adapter.

    Counts calls so a test can prove a refusal made no network request, which is
    the assertion that matters and the one a mocked exception would not catch.
    """

    def __init__(self, name, *, records=(), budget=None, raises=None):
        self.name = name
        self._records = list(records)
        self._budget = budget
        self._raises = raises
        self.calls = 0
        self.to_records_calls = 0
        self.ended = False

    def budget_state(self):
        if self._budget is None:
            return {"allowed": True, "reason": "no daily limit is agreed",
                    "limit": None, "spent": 0, "untrusted": False}
        return dict(self._budget)

    def to_records(self):
        self.to_records_calls += 1
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return list(self._records)

    def end_run(self):
        self.ended = True

    @property
    def requests_made(self):
        return self.calls


class NoBudgetAdapter(FakeAdapter):
    """A source with no agreed daily cap - We Work Remotely's shape."""

    budget_state = None  # deliberately absent, like WWR's adapter

    def __init__(self, name, *, records=(), raises=None):
        super().__init__(name, records=records, raises=raises)
        # Remove the attribute entirely so getattr() finds nothing.
        del self.__dict__["_budget"]


class DiscoveryTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = JobStore(Path(self._tmp.name) / "data")

    def permit(self, *names):
        for name in names:
            save_decision(AccessDecision(
                source=name,
                level=AccessLevel.PERMITTED,
                reason="test",
                robots="allows", robots_url="",
                terms="reviewed", terms_url="",
                checked_at=T0.isoformat(timespec="seconds"),
            ), self.store.data_dir)

    def spent_budget(self, source="myjobmag.co.ke", reason=None):
        return {
            "source": source,
            "allowed": False,
            "reason": reason or "the approved allowance of 1 request per day is already spent",
            "limit": 1, "spent": 1, "untrusted": False,
        }


class AllSourcesTests(DiscoveryTestCase):
    def test_all_three_sources_run_in_one_orchestration(self):
        self.permit("weworkremotely", "myjobmag.co.ke", "remotive.com")
        adapters = {
            "weworkremotely": FakeAdapter("weworkremotely",
                                           records=[_record("a", "weworkremotely")]),
            "myjobmag.co.ke": FakeAdapter("myjobmag.co.ke",
                                          records=[_record("b", "myjobmag.co.ke")]),
            "remotive.com": FakeAdapter("remotive.com",
                                        records=[_record("c", "remotive.com")]),
        }
        result = run_discovery(self.store, adapters, observed_at=T0)
        self.assertEqual(len(result.results), 3)
        self.assertEqual({r.outcome for r in result.results}, {Outcome.FETCHED})
        self.assertEqual(len(self.store.load_jobs()), 3)

    def test_each_source_ingests_through_the_shared_store(self):
        self.permit("weworkremotely", "myjobmag.co.ke", "remotive.com")
        adapters = {
            "weworkremotely": FakeAdapter("weworkremotely",
                                           records=[_record("a", "weworkremotely")]),
            "myjobmag.co.ke": FakeAdapter("myjobmag.co.ke",
                                          records=[_record("b", "myjobmag.co.ke")]),
            "remotive.com": FakeAdapter("remotive.com",
                                        records=[_record("c", "remotive.com")]),
        }
        run_discovery(self.store, adapters, observed_at=T0)
        sources = {s for r in self.store.load_jobs() for s in
                   [e.get("source") for e in r["sources"]]}
        self.assertEqual(sources, {"weworkremotely", "myjobmag.co.ke", "remotive.com"})

    def test_attribution_is_carried_through_to_the_report(self):
        self.permit("remotive.com")
        out = Path(self._tmp.name) / "report.html"
        run_discovery(
            self.store,
            {"remotive.com": FakeAdapter("remotive.com",
                                         records=[_record("c", "remotive.com")])},
            observed_at=T0,
            report_path=out,
            attributions={"remotive.com": "https://remotive.com/remote-jobs"},
        )
        self.assertIn("remotive.com", out.read_text(encoding="utf-8"))


class BudgetTests(DiscoveryTestCase):
    def test_a_spent_budget_refuses_before_any_network_call(self):
        self.permit("myjobmag.co.ke")
        adapter = FakeAdapter("myjobmag.co.ke", records=[_record("b", "myjobmag.co.ke")],
                              budget=self.spent_budget())
        result = run_discovery(self.store, {"myjobmag.co.ke": adapter}, observed_at=T0)
        self.assertEqual(adapter.calls, 0, "a refusal must not touch the network")
        self.assertEqual(adapter.to_records_calls, 0)
        self.assertEqual(result.results[0].outcome, Outcome.REFUSED)

    def test_a_refusal_stores_nothing(self):
        self.permit("myjobmag.co.ke")
        adapter = FakeAdapter("myjobmag.co.ke", records=[_record("b", "myjobmag.co.ke")],
                              budget=self.spent_budget())
        run_discovery(self.store, {"myjobmag.co.ke": adapter}, observed_at=T0)
        self.assertEqual(self.store.load_jobs(), [])

    def test_each_source_has_its_own_budget(self):
        """One source's spent budget must not silence the others."""
        self.permit("weworkremotely", "myjobmag.co.ke", "remotive.com")
        adapters = {
            "weworkremotely": NoBudgetAdapter(
                "weworkremotely", records=[_record("a", "weworkremotely")]),
            "myjobmag.co.ke": FakeAdapter("myjobmag.co.ke",
                                          records=[_record("b", "myjobmag.co.ke")],
                                          budget=self.spent_budget()),
            "remotive.com": FakeAdapter("remotive.com",
                                        records=[_record("c", "remotive.com")]),
        }
        result = run_discovery(self.store, adapters, observed_at=T0)
        by = result.by_name
        self.assertIs(by["weworkremotely"].outcome, Outcome.FETCHED)
        self.assertIs(by["myjobmag.co.ke"].outcome, Outcome.REFUSED)
        self.assertIs(by["remotive.com"].outcome, Outcome.FETCHED)
        self.assertEqual(adapters["myjobmag.co.ke"].calls, 0)

    def test_an_unreadable_budget_refuses_rather_than_spending_access(self):
        self.permit("remotive.com")
        adapter = FakeAdapter("remotive.com", records=[_record("c", "remotive.com")],
                              budget={"source": "remotive.com", "allowed": False,
                                      "reason": "ledger could not be read",
                                      "limit": 1, "spent": 0, "untrusted": True})
        result = run_discovery(self.store, {"remotive.com": adapter}, observed_at=T0)
        self.assertEqual(adapter.calls, 0)
        self.assertIs(result.results[0].outcome, Outcome.REFUSED)
        self.assertIn("could not be read", result.results[0].reason)

    def test_a_budget_that_cannot_be_inspected_refuses(self):
        class Broken(FakeAdapter):
            def budget_state(self):
                raise RuntimeError("disk gone")

        self.permit("remotive.com")
        adapter = Broken("remotive.com", records=[_record("c", "remotive.com")])
        result = run_discovery(self.store, {"remotive.com": adapter}, observed_at=T0)
        self.assertEqual(adapter.calls, 0)
        self.assertIs(result.results[0].outcome, Outcome.REFUSED)


class OutcomeDistinctionTests(DiscoveryTestCase):
    def test_a_successful_empty_source_is_a_zero_result_not_a_failure(self):
        self.permit("weworkremotely")
        adapter = FakeAdapter("weworkremotely", records=[])
        result = run_discovery(self.store, {"weworkremotely": adapter}, observed_at=T0)
        self.assertIs(result.results[0].outcome, Outcome.ZERO_RESULT)
        self.assertEqual(result.exit_code(), 0)

    def test_a_zero_result_and_a_refusal_are_different_outcomes(self):
        self.permit("weworkremotely", "myjobmag.co.ke")
        adapters = {
            "weworkremotely": FakeAdapter("weworkremotely", records=[]),
            "myjobmag.co.ke": FakeAdapter("myjobmag.co.ke",
                                          records=[_record("b", "myjobmag.co.ke")],
                                          budget=self.spent_budget()),
        }
        result = run_discovery(self.store, adapters, observed_at=T0)
        self.assertIs(result.by_name["weworkremotely"].outcome, Outcome.ZERO_RESULT)
        self.assertIs(result.by_name["myjobmag.co.ke"].outcome, Outcome.REFUSED)

    def test_a_failure_is_reported_as_a_failure(self):
        self.permit("remotive.com")
        adapter = FakeAdapter("remotive.com", raises=RuntimeError("api exploded"))
        result = run_discovery(self.store, {"remotive.com": adapter}, observed_at=T0)
        self.assertIs(result.results[0].outcome, Outcome.FAILED)
        self.assertIn("api exploded", result.results[0].reason)

    def test_an_unpermitted_source_is_skipped_not_attempted(self):
        # No decision recorded for this source.
        adapter = FakeAdapter("hiring.cafe", records=[_record("h", "hiring.cafe")])
        result = run_discovery(self.store, {"hiring.cafe": adapter}, observed_at=T0)
        self.assertEqual(adapter.calls, 0)
        self.assertIs(result.results[0].outcome, Outcome.SKIPPED)

    def test_a_restricted_source_is_skipped(self):
        save_decision(AccessDecision(
            source="hiring.cafe", level=AccessLevel.RESTRICTED, reason="challenge",
            robots="disallows", robots_url="", terms="unreviewable", terms_url="",
            checked_at=T0.isoformat(timespec="seconds"),
        ), self.store.data_dir)
        adapter = FakeAdapter("hiring.cafe", records=[_record("h", "hiring.cafe")])
        result = run_discovery(self.store, {"hiring.cafe": adapter}, observed_at=T0)
        self.assertEqual(adapter.calls, 0)
        self.assertIs(result.results[0].outcome, Outcome.SKIPPED)


class IndependenceTests(DiscoveryTestCase):
    def test_one_failure_does_not_stop_the_others(self):
        self.permit("weworkremotely", "myjobmag.co.ke", "remotive.com")
        adapters = {
            "weworkremotely": FakeAdapter("weworkremotely", raises=RuntimeError("boom")),
            "myjobmag.co.ke": FakeAdapter("myjobmag.co.ke",
                                          records=[_record("b", "myjobmag.co.ke")]),
            "remotive.com": FakeAdapter("remotive.com",
                                        records=[_record("c", "remotive.com")]),
        }
        result = run_discovery(self.store, adapters, observed_at=T0)
        by = result.by_name
        self.assertIs(by["weworkremotely"].outcome, Outcome.FAILED)
        self.assertIs(by["myjobmag.co.ke"].outcome, Outcome.FETCHED)
        self.assertIs(by["remotive.com"].outcome, Outcome.FETCHED)
        self.assertEqual(len(self.store.load_jobs()), 2)

    def test_adapters_are_released_after_the_run(self):
        self.permit("weworkremotely")
        adapter = FakeAdapter("weworkremotely", records=[_record("a", "weworkremotely")])
        run_discovery(self.store, {"weworkremotely": adapter}, observed_at=T0)
        self.assertTrue(adapter.ended)


class ExitCodeTests(DiscoveryTestCase):
    def _result(self, results):
        return DiscoveryResult(started_at=T0.isoformat(), results=results)

    def test_everything_worked_is_zero(self):
        result = self._result([
            SourceResult("a", Outcome.FETCHED),
            SourceResult("b", Outcome.ZERO_RESULT),
            SourceResult("c", Outcome.REFUSED),
        ])
        self.assertEqual(result.exit_code(), 0)

    def test_partial_failure_is_one(self):
        result = self._result([
            SourceResult("a", Outcome.FAILED),
            SourceResult("b", Outcome.FETCHED),
        ])
        self.assertEqual(result.exit_code(), 1)

    def test_total_failure_is_two(self):
        result = self._result([SourceResult("a", Outcome.FAILED)])
        self.assertEqual(result.exit_code(), 2)

    def test_a_refusal_alone_is_not_a_failure(self):
        """A run that respects every limit must not look broken."""
        result = self._result([
            SourceResult("a", Outcome.REFUSED),
            SourceResult("b", Outcome.REFUSED),
        ])
        self.assertEqual(result.exit_code(), 0)

    def test_a_failure_alongside_a_refusal_is_reported(self):
        result = self._result([
            SourceResult("a", Outcome.FAILED),
            SourceResult("b", Outcome.REFUSED),
        ])
        self.assertEqual(result.exit_code(), 2, "nothing succeeded, so nothing worked")


class DryRunTests(DiscoveryTestCase):
    def test_dry_run_makes_zero_network_calls(self):
        self.permit("weworkremotely", "myjobmag.co.ke")
        adapters = {
            "weworkremotely": FakeAdapter("weworkremotely",
                                           records=[_record("a", "weworkremotely")]),
            "myjobmag.co.ke": FakeAdapter("myjobmag.co.ke",
                                          records=[_record("b", "myjobmag.co.ke")]),
        }
        result = run_discovery(self.store, adapters, dry_run=True, observed_at=T0)
        for adapter in adapters.values():
            self.assertEqual(adapter.calls, 0)
            self.assertEqual(adapter.to_records_calls, 0)
        self.assertEqual(result.total_requests, 0)

    def test_dry_run_writes_nothing(self):
        self.permit("weworkremotely")
        before = {p.name for p in self.store.data_dir.rglob("*") if p.is_file()}
        run_discovery(
            self.store,
            {"weworkremotely": FakeAdapter("weworkremotely",
                                            records=[_record("a", "weworkremotely")])},
            dry_run=True, observed_at=T0,
        )
        after = {p.name for p in self.store.data_dir.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(self.store.load_jobs(), [])

    def test_dry_run_reports_what_would_run(self):
        self.permit("weworkremotely", "myjobmag.co.ke")
        adapters = {
            "weworkremotely": FakeAdapter("weworkremotely", records=[]),
            "myjobmag.co.ke": FakeAdapter("myjobmag.co.ke", records=[],
                                          budget=self.spent_budget()),
        }
        result = run_discovery(self.store, adapters, dry_run=True, observed_at=T0)
        self.assertIs(result.by_name["weworkremotely"].outcome, Outcome.WOULD_RUN)
        self.assertIs(result.by_name["myjobmag.co.ke"].outcome, Outcome.REFUSED)

    def test_dry_run_reports_skips_too(self):
        adapter = FakeAdapter("hiring.cafe", records=[])
        result = run_discovery(self.store, {"hiring.cafe": adapter}, dry_run=True,
                               observed_at=T0)
        self.assertIs(result.results[0].outcome, Outcome.SKIPPED)

    def test_dry_run_and_the_real_run_agree_on_the_plan(self):
        """The dry run must not approximate the decision the real run makes."""
        self.permit("weworkremotely", "myjobmag.co.ke")
        adapters = {
            "weworkremotely": FakeAdapter("weworkremotely", records=[]),
            "myjobmag.co.ke": FakeAdapter("myjobmag.co.ke", records=[],
                                          budget=self.spent_budget()),
        }
        plan = run_discovery(self.store, adapters, dry_run=True, observed_at=T0)
        self.assertIs(plan.by_name["myjobmag.co.ke"].outcome, Outcome.REFUSED)
        real = run_discovery(self.store, adapters, observed_at=T0)
        self.assertIs(real.by_name["myjobmag.co.ke"].outcome, Outcome.REFUSED)


class LayerTests(DiscoveryTestCase):
    """Run and freshness records go through the existing layers."""

    def test_a_run_record_is_written(self):
        self.permit("weworkremotely")
        run_discovery(
            self.store,
            {"weworkremotely": FakeAdapter("weworkremotely",
                                           records=[_record("a", "weworkremotely")])},
            observed_at=T0,
        )
        runs = self.store.load_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["sources"][0]["name"], "weworkremotely")

    def test_freshness_observations_are_written(self):
        self.permit("weworkremotely")
        run_discovery(
            self.store,
            {"weworkremotely": FakeAdapter("weworkremotely",
                                           records=[_record("a", "weworkremotely")])},
            observed_at=T0,
        )
        self.assertTrue((self.store.data_dir / "freshness.jsonl").exists())
        state = FreshnessLedger(self.store).evaluate()
        self.assertEqual(len(state), 1)
        self.assertEqual(list(state.values())[0].state.value, "active")

    def test_a_refusal_is_recorded_without_becoming_a_network_attempt(self):
        self.permit("myjobmag.co.ke")
        run_discovery(
            self.store,
            {"myjobmag.co.ke": FakeAdapter("myjobmag.co.ke", records=[],
                                           budget=self.spent_budget())},
            observed_at=T0,
        )
        runs = self.store.load_runs()
        self.assertEqual(runs[0]["sources"], [], "no source was consulted")
        codes = [s["code"] for s in runs[0]["skipped"]]
        self.assertIn(SKIP_BUDGET_REFUSED, codes)

    def test_a_skipped_source_is_recorded_with_its_reason(self):
        run_discovery(self.store, {"hiring.cafe": FakeAdapter("hiring.cafe", records=[])},
                      observed_at=T0)
        runs = self.store.load_runs()
        codes = [s["code"] for s in runs[0]["skipped"]]
        self.assertIn(SKIP_ACCESS_UNKNOWN, codes)


class ReportTests(DiscoveryTestCase):
    def test_the_consolidated_report_is_generated_after_the_run(self):
        self.permit("weworkremotely")
        out = Path(self._tmp.name) / "nested" / "report.html"
        result = run_discovery(
            self.store,
            {"weworkremotely": FakeAdapter("weworkremotely",
                                           records=[_record("a", "weworkremotely")])},
            observed_at=T0, report_path=out,
        )
        self.assertEqual(result.report_path, str(out))
        html = out.read_text(encoding="utf-8")
        self.assertIn("New and actionable", html)
        self.assertIn("Source health", html)

    def test_report_generation_alone_writes_nothing(self):
        """Rendering the report must not touch a single state file.

        Proved with a dry run, so the only work that happens is the report. A
        real run would legitimately write a run record and observations, and
        those writes would mask a report quietly mutating something.
        """
        self.permit("weworkremotely")
        run_discovery(
            self.store,
            {"weworkremotely": FakeAdapter("weworkremotely",
                                           records=[_record("a", "weworkremotely")])},
            observed_at=T0,
        )
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(self.store.data_dir.rglob("*")) if p.is_file()}
        run_discovery(
            self.store, {}, dry_run=True, observed_at=T0,
            report_path=Path(self._tmp.name) / "again.html",
        )
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(self.store.data_dir.rglob("*")) if p.is_file()}
        self.assertEqual(before, after)

    def test_a_run_with_no_sources_is_still_recorded(self):
        """An empty run must not be indistinguishable from no run at all."""
        run_discovery(self.store, {}, observed_at=T0)
        runs = self.store.load_runs()
        self.assertEqual(len(runs), 1)
        self.assertTrue(runs[0]["no_active_sources"])

    def test_report_errors_are_surfaced_not_swallowed(self):
        self.permit("weworkremotely")
        run_discovery(
            self.store,
            {"weworkremotely": FakeAdapter("weworkremotely",
                                           records=[_record("a", "weworkremotely")])},
            observed_at=T0,
        )
        with self.store.jobs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"job": {"title": "no id"}}) + "\n")
        result = run_discovery(
            self.store, {}, observed_at=T0,
            report_path=Path(self._tmp.name) / "broken.html",
        )
        self.assertTrue(result.report_errors)


class PlanTests(DiscoveryTestCase):
    def test_build_plan_classifies_without_any_network(self):
        self.permit("weworkremotely")
        adapter = FakeAdapter("weworkremotely", records=[_record("a", "weworkremotely")])
        _, results, skips = build_plan({"weworkremotely": adapter}, {"weworkremotely": _permitted("weworkremotely")})
        self.assertEqual(adapter.calls, 0)
        self.assertIs(results[0].outcome, Outcome.WOULD_RUN)
        self.assertEqual(skips, [])

    def test_every_configured_source_appears_exactly_once(self):
        self.permit("weworkremotely", "myjobmag.co.ke")
        _, results, _ = build_plan(
            {
                "weworkremotely": FakeAdapter("weworkremotely", records=[]),
                "myjobmag.co.ke": FakeAdapter("myjobmag.co.ke", records=[]),
            },
            {"weworkremotely": _permitted("weworkremotely"),
             "myjobmag.co.ke": _permitted("myjobmag.co.ke")},
        )
        self.assertEqual([r.name for r in results],
                         ["weworkremotely", "myjobmag.co.ke"])


def _permitted(name):
    return AccessDecision(
        source=name, level=AccessLevel.PERMITTED, reason="test",
        robots="allows", robots_url="", terms="reviewed", terms_url="",
        checked_at=T0.isoformat(timespec="seconds"),
    )


if __name__ == "__main__":
    unittest.main()
