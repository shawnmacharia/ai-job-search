"""Tests for the conservative source registry.

Entirely offline. The registry performs no robots fetch, HTTP request, DNS
lookup, network access, browser launch, or subprocess call - it enforces a
decision that was declared elsewhere. That is asserted, not assumed.
"""

import ast
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app.jobs.runner import (
    SKIP_ACCESS_RESTRICTED,
    SKIP_ACCESS_UNKNOWN,
    SKIP_DISABLED,
    SKIP_NO_FETCHER,
    SkippedSource,
    classify,
    run_sources,
    summarise,
)
from app.jobs.sources import (
    PERMITTED,
    RESTRICTED,
    UNKNOWN,
    SourceConfig,
    SourceConfigError,
    SourceRegistry,
    default_registry,
)
from app.jobs.store import JobStore

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hiring_cafe"
NOW = datetime(2026, 10, 9, 12, 0, 0, tzinfo=timezone.utc)
LIVE_SOURCE = "hiring.cafe"


def load(name):
    import json

    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def fetcher(name="basic.json"):
    return lambda: [load(name)]


class FixedClock:
    """A clock that advances by a fixed step; run_sources calls it twice per source."""

    def __init__(self, step: float = 0.01):
        self.step = step
        self.now = 0.0

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


class SourceConfigTests(unittest.TestCase):
    def test_enabled_and_permitted_is_active(self):
        config = SourceConfig(name="a", enabled=True, access=PERMITTED)
        self.assertTrue(config.is_permitted)
        self.assertTrue(config.is_active)
        self.assertIsNone(config.skip_reason())

    def test_access_defaults_to_unknown_so_a_new_source_is_not_active(self):
        config = SourceConfig(name="a")
        self.assertEqual(config.access, UNKNOWN)
        self.assertFalse(config.is_active, "fail closed by default")

    def test_disabled_source_is_skipped_with_reason(self):
        config = SourceConfig(name="a", enabled=False, access=PERMITTED)
        skip = config.skip_reason()
        self.assertIsNotNone(skip)
        self.assertEqual(skip.code, SKIP_DISABLED)
        self.assertIn("disabled", skip.reason.casefold())

    def test_unknown_access_is_skipped_with_reason(self):
        skip = SourceConfig(name="a", access=UNKNOWN).skip_reason()
        self.assertEqual(skip.code, SKIP_ACCESS_UNKNOWN)
        self.assertIn("not been decided", skip.reason.casefold())

    def test_restricted_access_is_skipped_with_reason(self):
        skip = SourceConfig(name="a", access=RESTRICTED).skip_reason()
        self.assertEqual(skip.code, SKIP_ACCESS_RESTRICTED)
        self.assertIn("restricted", skip.reason.casefold())

    def test_disabled_wins_over_permitted_and_gets_the_disabled_reason(self):
        # A source switched off must not look like an access problem.
        skip = SourceConfig(name="a", enabled=False, access=RESTRICTED).skip_reason()
        self.assertEqual(skip.code, SKIP_DISABLED)

    def test_note_is_used_as_the_human_readable_reason(self):
        skip = SourceConfig(
            name="a", access=RESTRICTED, note="terms forbid scraping"
        ).skip_reason()
        self.assertEqual(skip.reason, "terms forbid scraping")

    def test_invalid_access_is_rejected(self):
        with self.assertRaises(SourceConfigError):
            SourceConfig(name="a", access="allowed-ish")

    def test_empty_name_is_rejected(self):
        with self.assertRaises(SourceConfigError):
            SourceConfig(name="   ")


class RegistryTests(unittest.TestCase):
    def test_duplicate_source_name_is_rejected(self):
        registry = SourceRegistry([SourceConfig(name="a", access=PERMITTED)])
        with self.assertRaises(SourceConfigError) as caught:
            registry.register(SourceConfig(name="a", access=PERMITTED))
        self.assertIn("already registered", str(caught.exception))

    def test_registering_a_non_config_is_rejected(self):
        with self.assertRaises(SourceConfigError):
            SourceRegistry().register({"name": "a"})

    def test_unknown_source_lookup_raises(self):
        with self.assertRaises(SourceConfigError):
            SourceRegistry().get("nope")

    def test_active_returns_only_enabled_and_permitted(self):
        registry = SourceRegistry([
            SourceConfig(name="live", access=PERMITTED),
            SourceConfig(name="off", enabled=False, access=PERMITTED),
            SourceConfig(name="unknown", access=UNKNOWN),
            SourceConfig(name="blocked", access=RESTRICTED),
        ])
        self.assertEqual([c.name for c in registry.active()], ["live"])

    def test_skipped_reports_every_inactive_source_with_a_reason(self):
        registry = SourceRegistry([
            SourceConfig(name="live", access=PERMITTED),
            SourceConfig(name="off", enabled=False, access=PERMITTED),
            SourceConfig(name="unknown", access=UNKNOWN),
            SourceConfig(name="blocked", access=RESTRICTED),
        ])
        skips = {s.name: s for s in registry.skipped()}
        self.assertEqual(set(skips), {"off", "unknown", "blocked"})
        for skip in skips.values():
            self.assertTrue(skip.reason, "every skip needs a reason")
        # No silent omission: nothing disappears.
        self.assertEqual(len(skips) + len(registry.active()), len(registry))

    def test_active_sources_are_returned_in_stable_registration_order(self):
        names = ["zulu", "alpha", "mike", "bravo"]
        registry = SourceRegistry(
            [SourceConfig(name=name, access=PERMITTED) for name in names]
        )
        for _ in range(5):
            self.assertEqual([c.name for c in registry.active()], names)

    def test_registry_basics(self):
        registry = SourceRegistry([SourceConfig(name="a", access=PERMITTED)])
        self.assertIn("a", registry)
        self.assertEqual(len(registry), 1)
        self.assertEqual(registry.get("a").name, "a")


class PlanTests(unittest.TestCase):
    def test_plan_pairs_active_specs_with_skipped_reasons(self):
        registry = SourceRegistry([
            SourceConfig(name="live", access=PERMITTED),
            SourceConfig(name="off", enabled=False, access=PERMITTED),
        ])
        plan = registry.plan({"live": fetcher()})
        self.assertEqual([s.name for s in plan.specs], ["live"])
        self.assertEqual([s.name for s in plan.skipped], ["off"])
        self.assertTrue(plan)
        self.assertFalse(plan.is_empty)

    def test_active_source_without_a_fetcher_is_skipped_not_raised(self):
        registry = SourceRegistry([SourceConfig(name="live", access=PERMITTED)])
        plan = registry.plan({})
        self.assertFalse(plan.specs, "no fetcher means nothing runs")
        self.assertEqual(plan.skipped[0].code, SKIP_NO_FETCHER)
        self.assertTrue(plan.skipped[0].reason)

    def test_a_fetch_for_an_inactive_source_is_ignored(self):
        registry = SourceRegistry([SourceConfig(name="off", enabled=False, access=PERMITTED)])
        plan = registry.plan({"off": fetcher()})
        self.assertFalse(plan.specs)
        self.assertEqual(plan.skipped[0].code, SKIP_DISABLED)

    def test_empty_plan_is_falsy(self):
        registry = SourceRegistry([SourceConfig(name="off", enabled=False, access=PERMITTED)])
        plan = registry.plan({})
        self.assertFalse(plan, "an empty plan reads as 'no work to do'")


class DefaultRegistryTests(unittest.TestCase):
    def test_shipped_registry_fails_closed(self):
        registry = default_registry()
        self.assertIn("hiring.cafe", registry)
        self.assertEqual(registry.active(), [], "nothing runs until access is declared")
        skips = registry.skipped()
        self.assertEqual(len(skips), 1)
        self.assertEqual(skips[0].code, SKIP_ACCESS_UNKNOWN)
        self.assertTrue(skips[0].reason)


class RegistryRunnerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.store = JobStore(Path(self._directory.name) / "data")

    def run_plan(self, plan, *, ticks=(0.0, 0.01)):
        return run_sources(
            plan.specs,
            store=self.store,
            observed_at=NOW,
            now=FixedClock(),
            skipped=plan.skipped,
        )

    def test_skipped_sources_reach_the_run_ledger(self):
        registry = SourceRegistry([
            SourceConfig(name=LIVE_SOURCE, access=PERMITTED),
            SourceConfig(name="off", enabled=False, access=PERMITTED, note="switched off"),
            SourceConfig(name="blocked", access=RESTRICTED),
        ])
        plan = registry.plan({LIVE_SOURCE: fetcher()})
        run = self.run_plan(plan)

        ledger = self.store.load_runs()[0]
        skipped = {entry["name"]: entry for entry in ledger["skipped"]}
        self.assertEqual(set(skipped), {"off", "blocked"})
        self.assertEqual(skipped["off"]["reason"], "switched off")
        self.assertEqual(skipped["blocked"]["code"], SKIP_ACCESS_RESTRICTED)
        self.assertEqual(len(ledger["sources"]), 1, "only the active source ran")

    def test_a_skip_is_not_a_failure_and_does_not_change_the_exit_code(self):
        registry = SourceRegistry([
            SourceConfig(name=LIVE_SOURCE, access=PERMITTED),
            SourceConfig(name="off", enabled=False, access=PERMITTED),
        ])
        run = self.run_plan(registry.plan({LIVE_SOURCE: fetcher()}))
        self.assertEqual(run.exit_code(), 0)
        self.assertFalse(run.all_failed)
        self.assertFalse(run.partially_failed)

    def test_one_active_source_and_several_skipped_works_normally(self):
        registry = SourceRegistry([
            SourceConfig(name=LIVE_SOURCE, access=PERMITTED),
            SourceConfig(name="off", enabled=False, access=PERMITTED),
            SourceConfig(name="unknown", access=UNKNOWN),
            SourceConfig(name="blocked", access=RESTRICTED),
        ])
        run = self.run_plan(registry.plan({LIVE_SOURCE: fetcher()}))
        self.assertEqual(len(run.sources), 1)
        self.assertEqual(run.sources[0].stored, 1)
        self.assertEqual(len(run.skipped), 3)
        self.assertEqual(run.exit_code(), 0)
        self.assertEqual(len(self.store.load_jobs()), 1)

    def test_no_active_sources_is_reported_clearly_not_as_success(self):
        registry = SourceRegistry([
            SourceConfig(name="off", enabled=False, access=PERMITTED),
            SourceConfig(name="unknown", access=UNKNOWN),
        ])
        plan = registry.plan({})
        run = self.run_plan(plan)

        self.assertTrue(run.no_active_sources)
        self.assertEqual(run.exit_code(), 0, "a skip is not a failure")
        self.assertEqual(run.total_fetched, 0)

        kinds = [row["kind"] for row in summarise(run)]
        self.assertIn("no_active_sources", kinds, "the run must say it consulted nothing")

        entry = self.store.load_runs()[0]
        self.assertTrue(entry["no_active_sources"])
        self.assertEqual(len(entry["skipped"]), 2, "nothing is silently omitted")

    def test_every_active_source_failing_still_exits_two(self):
        def boom():
            raise RuntimeError("site down")

        registry = SourceRegistry([SourceConfig(name="live", access=PERMITTED)])
        plan = registry.plan({"live": boom})
        run = self.run_plan(plan)
        self.assertTrue(run.all_failed)
        self.assertEqual(run.exit_code(), 2, "skipped sources must not mask a total failure")

    def test_a_skipped_source_alongside_a_failure_does_not_soften_the_failure(self):
        registry = SourceRegistry([
            SourceConfig(name="live", access=PERMITTED),
            SourceConfig(name="off", enabled=False, access=PERMITTED),
        ])
        plan = registry.plan({"live": lambda: (_ for _ in ()).throw(OSError("dns"))})
        run = self.run_plan(plan)
        self.assertTrue(run.all_failed)
        self.assertEqual(run.exit_code(), 2)

    def test_skipped_sources_appear_in_summarise(self):
        registry = SourceRegistry([
            SourceConfig(name=LIVE_SOURCE, access=PERMITTED),
            SourceConfig(name="off", enabled=False, access=PERMITTED),
        ])
        run = self.run_plan(registry.plan({LIVE_SOURCE: fetcher()}))
        rows = {row["source"]: row for row in summarise(run)}
        self.assertEqual(rows[LIVE_SOURCE]["kind"], "jobs")
        self.assertEqual(rows["off"]["kind"], "skipped")
        self.assertEqual(rows["off"]["code"], SKIP_DISABLED)

    def test_run_sources_accepts_plain_dicts_for_skipped(self):
        run = run_sources(
            [],
            store=self.store,
            observed_at=NOW,
            skipped=[{"name": "x", "code": "disabled", "reason": "off"}],
        )
        self.assertEqual(run.skipped[0]["name"], "x")
        self.assertTrue(run.no_active_sources)


class SkippedSourceTests(unittest.TestCase):
    def test_round_trips_to_a_dict(self):
        skip = SkippedSource("a", "disabled", "because")
        self.assertEqual(
            skip.to_dict(), {"name": "a", "code": "disabled", "reason": "because"}
        )

    def test_is_hashable_and_comparable(self):
        # frozen dataclass: usable in sets, which reporting code wants.
        a = SkippedSource("a", "disabled", "because")
        self.assertEqual(len({a, SkippedSource("a", "disabled", "because")}), 1)


class NoSideEffectsTests(unittest.TestCase):
    FORBIDDEN = {
        "playwright", "requests", "httpx", "urllib", "urllib3", "socket",
        "http", "subprocess", "ollama", "aiohttp", "selenium", "shutil",
        "asyncio", "multiprocessing", "ssl", "ftplib", "smtplib", "telnetlib",
    }

    def _imports(self, module_path: str) -> set:
        tree = ast.parse(Path(module_path).read_text(encoding="utf-8"))
        names: set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names.add(node.module.split(".")[0])
        return names

    def test_sources_module_imports_nothing_that_performs_io(self):
        path = Path(__file__).resolve().parent.parent / "app" / "jobs" / "sources.py"
        offenders = self._imports(str(path)) & self.FORBIDDEN
        self.assertEqual(offenders, set(), f"sources.py must stay offline: {sorted(offenders)}")

    def test_sources_module_contains_no_url_or_fetch_primitives(self):
        path = Path(__file__).resolve().parent.parent / "app" / "jobs" / "sources.py"
        source = path.read_text(encoding="utf-8")
        for forbidden in ("urlopen", "requests.get", "socket.", "subprocess", "os.system"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_registry_never_calls_the_fetcher_of_an_inactive_source(self):
        # Fail closed, proven by execution: an inactive source's fetcher must
        # not be invoked even when one is supplied.
        calls = []

        def spy():
            calls.append(1)
            return []

        registry = SourceRegistry([
            SourceConfig(name="blocked", access=RESTRICTED),
            SourceConfig(name="off", enabled=False, access=PERMITTED),
        ])
        plan = registry.plan({"blocked": spy, "off": spy})
        self.assertEqual(plan.specs, [])
        self.assertEqual(calls, [], "no fetcher ran for an inactive source")

    def test_building_a_plan_imports_no_network_module(self):
        # Measured as a delta: other test modules legitimately import
        # `urllib.request` (app.llm.ollama does), so an absolute check on
        # sys.modules would fail for reasons unrelated to this one.
        import sys

        network = {
            "playwright", "ollama", "requests", "urllib.request",
            "http.client", "socket", "ssl", "asyncio",
        }
        before = {name for name in sys.modules if name in network}
        SourceRegistry([SourceConfig(name="a", access=PERMITTED)]).plan({})
        after = {name for name in sys.modules if name in network}
        self.assertEqual(
            after - before, set(),
            "building a plan must not pull in a network module",
        )


if __name__ == "__main__":
    unittest.main()
