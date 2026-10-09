"""Tests for multi-source health reporting.

Entirely offline: fixtures and fake fetch callables. No network, Playwright,
Ollama, subprocess, socket, or live scraping.

Time is injected, so durations are exact and no test asserts wall-clock
behaviour. Relative ordering is asserted too - a source the clock advances
further for must report a larger ``duration_ms``.

Source naming note: a source that actually returns records must carry the name
of a *registered* adapter (``hiring.cafe`` is the only one shipped). A source
that returns nothing, or that fails before ingestion, is never adapted, so any
name is correct for it - and using descriptive names there is what makes the
health report readable.
"""

import ast
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app.jobs.runner import (
    KIND_EMPTY,
    KIND_FAILED,
    KIND_JOBS,
    SourceSpec,
    classify,
    run_sources,
    summarise,
)
from app.jobs.store import JobStore, SourceOutcome

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hiring_cafe"
NOW = datetime(2026, 10, 9, 12, 0, 0, tzinfo=timezone.utc)
USABLE = ["basic.json", "worldwide_remote.json", "salary_ksh.json"]

#: The one adapter that ships with the repository.
LIVE_SOURCE = "hiring.cafe"


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def raises(exc):
    """A fetch callable that raises ``exc``."""

    def fetch():
        raise exc

    return fetch


class FakeClock:
    """A monotonic clock that advances by a scripted amount per call.

    ``run_sources`` calls the clock exactly twice per source (start, end), so a
    two-source run consumes the first four ticks.
    """

    def __init__(self, ticks):
        self.ticks = list(ticks)
        self.index = 0

    def __call__(self):
        value = self.ticks[min(self.index, len(self.ticks) - 1)]
        self.index += 1
        return value


def specs(sources) -> list[SourceSpec]:
    """Build specs from a ``name -> fetch`` mapping or a sequence of pairs.

    A mapping is the common case, but source names legitimately contain dots
    (``hiring.cafe``) and so cannot be keyword arguments. A sequence of pairs is
    also accepted so the same source can be consulted twice with different
    queries - two live sources need two adapters, one source twice needs one.
    """
    items = sources.items() if isinstance(sources, dict) else sources
    return [SourceSpec(name=name, fetch=fetch) for name, fetch in items]


class RunnerTestCase(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.store = JobStore(Path(self._directory.name) / "data")

    def consult(self, sources: dict, *, ticks=(0.0, 0.01, 0.0, 0.01, 0.0, 0.01), run_id=None):
        # NB: not named `run` - unittest.TestCase.run(result) is the entry point
        # the test runner itself calls, and shadowing it breaks discovery.
        return run_sources(
            specs(sources),
            store=self.store,
            observed_at=NOW,
            run_id=run_id,
            now=FakeClock(list(ticks)),
        )

    def outcome(self, run, name):
        return next(o for o in run.sources if o.name == name)


class OutcomeClassificationTests(unittest.TestCase):
    # NB: deliberately not named `_outcome` - unittest.TestCase.__init__ sets an
    # `_outcome` attribute on every instance for its subtest machinery, so a
    # helper by that name is silently shadowed by None.
    @staticmethod
    def make_outcome(**kwargs):
        return SourceOutcome(name="probe", **kwargs)

    def test_three_outcomes_are_distinguished(self):
        self.assertEqual(classify(self.make_outcome(ok=True, fetched=7)), KIND_JOBS)
        self.assertEqual(classify(self.make_outcome(ok=True, fetched=0)), KIND_EMPTY)
        self.assertEqual(
            classify(self.make_outcome(ok=False, error="boom")), KIND_FAILED
        )

    def test_a_failed_source_is_never_classified_as_empty(self):
        # The bug this whole increment exists to prevent: "found nothing" and
        # "could not look" must not share a value.
        failed = self.make_outcome(ok=False, error="TimeoutError: x")
        empty = self.make_outcome(ok=True, fetched=0)
        self.assertNotEqual(classify(failed), classify(empty))


class RunSourcesTests(RunnerTestCase):
    def test_mixed_batch_reports_each_source_distinctly(self):
        run = self.consult({
            LIVE_SOURCE: lambda: [load(name) for name in USABLE],
            "empty-board": lambda: [],
            "broken-board": raises(RuntimeError("upstream unavailable")),
        })

        kinds = {row["source"]: row["kind"] for row in summarise(run)}
        self.assertEqual(kinds, {
            LIVE_SOURCE: KIND_JOBS,
            "empty-board": KIND_EMPTY,
            "broken-board": KIND_FAILED,
        })

        good = self.outcome(run, LIVE_SOURCE)
        self.assertEqual(good.fetched, 3)
        self.assertEqual(good.stored, 3)
        self.assertEqual(len(self.store.load_jobs()), 3)

        empty = self.outcome(run, "empty-board")
        self.assertTrue(empty.ok)
        self.assertEqual(empty.fetched, 0)
        self.assertIsNone(empty.error)

        broken = self.outcome(run, "broken-board")
        self.assertFalse(broken.ok)
        self.assertIn("upstream unavailable", broken.error)

    def test_a_failing_source_does_not_prevent_the_others(self):
        # The healthy sources are the same adapter consulted twice with
        # different queries - two live sources would need two adapters.
        run = self.consult([
            ("broken-board", raises(TimeoutError("page.goto timed out"))),
            (LIVE_SOURCE, lambda: [load("basic.json")]),
            (LIVE_SOURCE, lambda: [load("worldwide_remote.json")]),
        ])

        self.assertEqual(len(run.sources), 3, "every source still reports")
        self.assertEqual(sum(1 for o in run.sources if o.ok), 2)
        self.assertEqual(len(self.store.load_jobs()), 2, "healthy sources still stored")

    def test_a_failing_source_does_not_lose_records_already_stored(self):
        self.consult({
            LIVE_SOURCE: lambda: [load("basic.json"), load("worldwide_remote.json")],
            "broken-board": raises(RuntimeError("late failure")),
        })
        self.assertEqual(len(self.store.load_jobs()), 2)
        ledger = self.store.load_runs()
        self.assertEqual(len(ledger), 1, "the run is still recorded")
        self.assertEqual(len(ledger[0]["sources"]), 2)

    def test_exit_code_zero_when_any_source_succeeds(self):
        cases = {
            "one good": {LIVE_SOURCE: lambda: [load("basic.json")]},
            "one empty": {"empty-board": lambda: []},
            "partial": {
                LIVE_SOURCE: lambda: [load("basic.json")],
                "broken-board": raises(RuntimeError("x")),
            },
        }
        for label, sources in cases.items():
            with self.subTest(case=label):
                run = self.consult(sources)
                self.assertEqual(run.exit_code(), 0, "a zero-result source is not an outage")

    def test_exit_code_two_only_when_every_source_fails(self):
        run = self.consult({
            "a": raises(RuntimeError("down")),
            "b": raises(TimeoutError("timeout")),
            "c": raises(OSError("dns")),
        })
        self.assertTrue(run.all_failed)
        self.assertEqual(run.exit_code(), 2)

    def test_a_successful_zero_result_source_alone_is_exit_zero(self):
        run = self.consult({"empty-board": lambda: []})
        self.assertFalse(run.all_failed)
        self.assertEqual(run.exit_code(), 0)

    def test_no_sources_requested_is_not_a_total_failure(self):
        run = self.consult({})
        self.assertEqual(run.exit_code(), 0, "nothing was asked, so nothing failed")


class DurationTests(RunnerTestCase):
    def test_durations_come_from_the_injected_clock(self):
        # Ticks pair up per source: (start, end) for each, in spec order.
        run = self.consult(
            {
                "slow-board": raises(RuntimeError("x")),
                LIVE_SOURCE: lambda: [load("basic.json")],
                "medium-board": lambda: [load("worldwide_remote.json")],
            },
            ticks=[0.0, 2.0, 0.0, 0.01, 0.0, 0.75],
        )
        by_name = {o.name: o.duration_ms for o in run.sources}
        self.assertEqual(by_name["slow-board"], 2000)
        self.assertEqual(by_name[LIVE_SOURCE], 10)
        self.assertEqual(by_name["medium-board"], 750)

    def test_relative_ordering_holds(self):
        run = self.consult(
            {
                "quick-board": lambda: [load("basic.json")],
                "sluggish-board": raises(RuntimeError("x")),
            },
            ticks=[0.0, 0.05, 0.0, 1.5],
        )
        by_name = {o.name: o.duration_ms for o in run.sources}
        self.assertGreater(by_name["sluggish-board"], by_name["quick-board"])

    def test_duration_is_recorded_for_failed_sources_too(self):
        run = self.consult({"broken-board": raises(RuntimeError("x"))}, ticks=[0.0, 1.25])
        self.assertEqual(run.sources[0].duration_ms, 1250)

    def test_duration_is_never_negative(self):
        # A misbehaving clock must not put nonsense into the health report.
        run = self.consult({LIVE_SOURCE: lambda: [load("basic.json")]}, ticks=[5.0, 1.0])
        self.assertGreaterEqual(run.sources[0].duration_ms, 0)


class ErrorReportingTests(RunnerTestCase):
    def test_errors_carry_type_and_a_useful_message(self):
        run = self.consult({"broken-board": raises(TimeoutError("selector a[href='/job/'] timed out"))})
        error = run.sources[0].error
        self.assertIn("TimeoutError", error)
        self.assertIn("selector", error)

    def test_error_message_is_only_type_and_message(self):
        # The runner records the exception's own type and message; it never
        # serialises environment, configuration, or store contents.
        run = self.consult({"broken-board": raises(ValueError("bad payload"))})
        self.assertEqual(run.sources[0].error, "ValueError: bad payload")

    def test_non_exception_failures_are_handled(self):
        class ExplodingFetch:
            def __call__(self):
                raise SystemError("catastrophic")

        run = self.consult({"broken-board": ExplodingFetch()})
        self.assertFalse(run.sources[0].ok)
        self.assertIn("SystemError", run.sources[0].error)

    def test_keyboard_interrupt_is_not_swallowed(self):
        with self.assertRaises(KeyboardInterrupt):
            self.consult({
                "stopped": raises(KeyboardInterrupt()),
                LIVE_SOURCE: lambda: [load("basic.json")],
            })


class QuarantineAndIdempotencyTests(RunnerTestCase):
    def test_malformed_records_are_quarantined_and_counted(self):
        run = self.consult([
            (LIVE_SOURCE, lambda: [load("basic.json"), load("missing_description.json")]),
            (LIVE_SOURCE, lambda: [load("worldwide_remote.json")]),
        ])

        mixed = run.sources[0]
        self.assertEqual(mixed.fetched, 2)
        self.assertEqual(mixed.stored, 1)
        self.assertEqual(mixed.rejected, 1)

        rejected = self.store.load_rejected()
        self.assertEqual(len(rejected), 1)
        self.assertIn("adaptation failed", rejected[0]["reason"])

        self.assertEqual(run.sources[1].stored, 1,
                         "the other source was unaffected")

    def test_replaying_the_same_records_does_not_duplicate_jobs(self):
        sources = {LIVE_SOURCE: lambda: [load(name) for name in USABLE]}

        first = self.consult(sources)
        self.assertEqual(first.total_stored, len(USABLE))
        self.assertEqual(first.total_updated, 0)

        second = self.consult(sources)
        self.assertEqual(second.total_stored, 0)
        self.assertEqual(second.total_updated, len(USABLE))
        self.assertEqual(len(self.store.load_jobs()), len(USABLE))
        self.assertEqual(second.exit_code(), 0)

    def test_two_sources_offering_the_same_job_are_flagged_not_merged(self):
        # A genuine cross-source duplicate needs a second registered adapter.
        # Registering through the public adapters.register() also exercises the
        # extension point P1 built.
        from app.jobs.adapters import HiringCafeAdapter, register

        class _TestAggregator(HiringCafeAdapter):
            name = "test-aggregator"
            portal = "test-aggregator"

        register(_TestAggregator())

        shared = load("basic.json")
        mirrored = dict(shared, url="https://aggregator.example/jobs/basic-001")
        run = self.consult({
            LIVE_SOURCE: lambda: [shared],
            "test-aggregator": lambda: [mirrored],
        })

        self.assertEqual(run.total_stored, 2, "both listings are kept")
        self.assertEqual(run.total_stored + run.total_updated + run.total_rejected, 2)
        flagged = [j for j in self.store.load_jobs() if j["possible_duplicate"]]
        self.assertEqual(len(flagged), 1, "one record is flagged, not merged")
        origins = {s["source"] for job in self.store.load_jobs() for s in job["sources"]}
        self.assertEqual(origins, {LIVE_SOURCE, "test-aggregator"},
                         "both source names survive as provenance")


class LedgerTests(RunnerTestCase):
    def test_run_is_written_to_the_ledger_with_every_source(self):
        run = self.consult(
            {
                LIVE_SOURCE: lambda: [load("basic.json")],
                "empty-board": lambda: [],
                "broken-board": raises(RuntimeError("x")),
            },
            run_id="run-42",
        )
        self.assertEqual(run.run_id, "run-42")

        ledger = self.store.load_runs()
        self.assertEqual(len(ledger), 1)
        entry = ledger[0]
        self.assertEqual(entry["run_id"], "run-42")
        self.assertEqual(len(entry["sources"]), 3)
        self.assertFalse(entry["all_failed"])
        self.assertTrue(entry["partially_failed"])
        self.assertEqual(entry["exit_code"], 0)
        self.assertEqual(entry["totals"]["fetched"], 1)
        self.assertEqual(entry["totals"]["stored"], 1)
        self.assertEqual(entry["totals"]["rejected"], 0)

        by_name = {s["name"]: s for s in entry["sources"]}
        self.assertTrue(by_name[LIVE_SOURCE]["ok"])
        self.assertTrue(by_name["empty-board"]["ok"])
        self.assertFalse(by_name["broken-board"]["ok"])
        self.assertIn("RuntimeError", by_name["broken-board"]["error"])
        for source in entry["sources"]:
            self.assertIn("duration_ms", source)


class NoSideEffectsTests(unittest.TestCase):
    FORBIDDEN = {
        "playwright", "requests", "httpx", "urllib", "urllib3", "socket",
        "http", "subprocess", "ollama", "aiohttp", "selenium", "shutil",
        "asyncio", "multiprocessing",
    }

    def test_runner_module_imports_nothing_that_reaches_outside_the_process(self):
        path = Path(__file__).resolve().parent.parent / "app" / "jobs" / "runner.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))

        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])

        # `time` is allowed: it is the default clock and is injected away in
        # every test. Nothing that performs I/O is permitted.
        offenders = (imported & self.FORBIDDEN) - {"time"}
        self.assertEqual(
            offenders, set(),
            f"runner.py must not perform I/O, but imports {sorted(offenders)}",
        )

    def test_running_sources_imports_no_browser_or_model_module(self):
        import sys

        with tempfile.TemporaryDirectory() as directory:
            run_sources(
                [SourceSpec(name=LIVE_SOURCE, fetch=lambda: [load("basic.json")])],
                store=JobStore(Path(directory) / "data"),
                observed_at=NOW,
                now=FakeClock([0.0, 0.0]),
            )
        for module in ("playwright", "ollama", "requests"):
            with self.subTest(module=module):
                self.assertNotIn(module, sys.modules)

    def test_runner_contains_no_shell_out(self):
        path = Path(__file__).resolve().parent.parent / "app" / "jobs" / "runner.py"
        source = path.read_text(encoding="utf-8")
        for forbidden in ("os.system", "os.popen", "eval(", "exec(", "urlopen("):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
