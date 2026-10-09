"""Tests for connecting the hiring.cafe source to the P0 store.

Fully offline: fixtures only. No network, no Playwright, no Ollama, no
subprocess, no live scraping. The same nine saved hiring.cafe records used by
``test_adapters`` are ingested end to end, which is the first time the chain
source -> adapter -> store runs as a unit.
"""

import ast
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app.jobs.ingest import ingest
from app.jobs.store import JobStore, SourceOutcome

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hiring_cafe"
NOW = datetime(2026, 10, 9, 12, 0, 0, tzinfo=timezone.utc)

#: Every fixture on disk, including the two deliberately malformed ones.
ALL_FIXTURES = sorted(path.name for path in FIXTURES.glob("*.json"))
USABLE_FIXTURES = [
    name for name in ALL_FIXTURES
    if name not in {"missing_description.json", "missing_company.json"}
]


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class IngestTestCase(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.store = JobStore(Path(self._directory.name) / "data")

    def ingest_fixtures(self, names):
        return ingest(
            [load(name) for name in names],
            source="hiring.cafe",
            store=self.store,
            observed_at=NOW,
        )


class IngestionTests(IngestTestCase):
    def test_successful_ingestion_of_all_usable_fixtures(self):
        outcome = self.ingest_fixtures(USABLE_FIXTURES)
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.fetched, len(USABLE_FIXTURES))
        self.assertEqual(outcome.stored, len(USABLE_FIXTURES))
        self.assertEqual(outcome.updated, 0)
        self.assertEqual(outcome.rejected, 0)
        self.assertEqual(len(self.store.load_jobs()), len(USABLE_FIXTURES))

    def test_replaying_the_same_records_creates_no_duplicate_jobs(self):
        first = self.ingest_fixtures(USABLE_FIXTURES)
        self.assertEqual(first.stored, len(USABLE_FIXTURES))

        second = self.ingest_fixtures(USABLE_FIXTURES)
        self.assertEqual(second.stored, 0, "a replay stores nothing new")
        self.assertEqual(second.updated, len(USABLE_FIXTURES), "a replay counts as updates")
        self.assertEqual(second.fetched, len(USABLE_FIXTURES))
        self.assertEqual(
            len(self.store.load_jobs()), len(USABLE_FIXTURES),
            "canonical job count must not grow on replay",
        )

    def test_replay_three_times_is_still_idempotent(self):
        for _ in range(3):
            self.ingest_fixtures(USABLE_FIXTURES)
        self.assertEqual(len(self.store.load_jobs()), len(USABLE_FIXTURES))

    def test_malformed_record_is_quarantined_while_valid_records_still_ingest(self):
        outcome = self.ingest_fixtures(ALL_FIXTURES)
        self.assertEqual(outcome.fetched, len(ALL_FIXTURES))
        self.assertEqual(outcome.stored, len(USABLE_FIXTURES))
        self.assertEqual(outcome.rejected, 2, "both malformed fixtures are quarantined")

        rejected = self.store.load_rejected()
        self.assertEqual(len(rejected), 2)
        reasons = " ".join(entry["reason"] for entry in rejected)
        self.assertIn("adaptation failed", reasons)
        self.assertIn("description_snippet", reasons)
        self.assertIn("company", reasons)
        for entry in rejected:
            self.assertEqual(entry["source"], "hiring.cafe")

    def test_one_bad_record_does_not_stop_the_rest(self):
        # Malformed record deliberately placed in the middle of a valid batch.
        batch = [
            load("basic.json"),
            {"nonsense": True},
            load("worldwide_remote.json"),
            "not even a mapping",
            load("salary_ksh.json"),
        ]
        outcome = ingest(batch, source="hiring.cafe", store=self.store, observed_at=NOW)
        self.assertEqual(outcome.fetched, 5)
        self.assertEqual(outcome.stored, 3)
        self.assertEqual(outcome.rejected, 2)
        self.assertEqual(len(self.store.load_jobs()), 3)

    def test_source_provenance_records_source_and_original_url(self):
        raw = load("salary_year.json")
        self.ingest_fixtures(["salary_year.json"])
        record = self.store.load_jobs()[0]
        self.assertEqual(record["sources"][0]["source"], "hiring.cafe")
        self.assertEqual(record["sources"][0]["url"], raw["url"])
        self.assertEqual(record["job"]["portal"], "hiring.cafe")

    def test_store_round_trip_preserves_enriched_job_fields(self):
        raw = load("salary_year.json")
        self.ingest_fixtures(["salary_year.json"])
        stored = self.store.load_jobs()[0]["job"]

        self.assertEqual(stored["description"], raw["description_snippet"])
        self.assertEqual(stored["posted_date"], "2026-10-09")  # "5h ago"
        self.assertEqual(stored["posted_raw"], "5h ago")
        self.assertFalse(stored["description_complete"])
        self.assertEqual(stored["salary_min"], 80000)
        self.assertEqual(stored["salary_max"], 120000)
        self.assertIsNone(stored["salary_currency"])
        self.assertEqual(stored["salary_period"], "year")
        self.assertEqual(stored["raw_excerpt"]["company_blurb"], raw["company_blurb"])
        self.assertEqual(stored["raw_excerpt"]["employment_type"], raw["employment_type"])

    def test_source_outcome_counts_are_accurate(self):
        outcome = self.ingest_fixtures(ALL_FIXTURES)
        self.assertEqual(
            outcome.fetched, len(ALL_FIXTURES), "fetched counts records received"
        )
        self.assertEqual(
            outcome.stored + outcome.updated + outcome.rejected, outcome.fetched,
            "every received record is accounted for exactly once",
        )
        self.assertIsInstance(outcome, SourceOutcome)
        self.assertIsNone(outcome.error)

    def test_cross_source_duplicate_is_flagged_not_merged(self):
        # Same company and title, different URL - exactly the case P0 flags
        # rather than merges.
        first = load("basic.json")
        second = load("basic.json")
        second["url"] = "https://hiring.cafe/job/basic-001-mirror"
        outcome = ingest([first, second], source="hiring.cafe", store=self.store, observed_at=NOW)

        self.assertEqual(outcome.stored, 2, "both are kept as separate records")
        self.assertEqual(outcome.possible_duplicates, 1, "the second is flagged")

        jobs = self.store.load_jobs()
        self.assertEqual(len(jobs), 2, "no merge happened")
        flagged = [job for job in jobs if job["possible_duplicate"]]
        self.assertEqual(len(flagged), 1)
        self.assertEqual(len(flagged[0]["duplicate_of"]), 1)

    def test_empty_source_is_ok_and_distinguishable_from_failure(self):
        outcome = ingest([], source="hiring.cafe", store=self.store, observed_at=NOW)
        self.assertTrue(outcome.ok, "an empty result is not a failure")
        self.assertEqual(outcome.fetched, 0)
        self.assertIsNone(outcome.error)
        self.assertEqual(self.store.load_jobs(), [])
        # Nothing was written at all - no empty index, no empty log.
        self.assertFalse(self.store.jobs_path.exists())
        self.assertFalse(self.store.seen_path.exists())

    def test_failed_source_outcome_is_still_distinguishable(self):
        failure = self.store.failed_source("hiring.cafe", TimeoutError("timed out"))
        self.assertFalse(failure.ok)
        self.assertIn("TimeoutError", failure.error)
        self.assertEqual(failure.fetched, 0)


class NoSideEffectsTests(unittest.TestCase):
    """The ingestion path must not reach the network or a browser."""

    FORBIDDEN = {
        "playwright", "requests", "httpx", "urllib", "urllib3", "socket",
        "http", "subprocess", "ollama", "asyncio", "aiohttp", "selenium",
        "multiprocessing", "shutil",
    }

    def test_ingest_module_imports_nothing_that_reaches_outside_the_process(self):
        path = Path(__file__).resolve().parent.parent / "app" / "jobs" / "ingest.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))

        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])

        offenders = imported & self.FORBIDDEN
        self.assertEqual(
            offenders, set(),
            f"ingest.py must stay offline, but imports {sorted(offenders)}",
        )

    def test_ingest_contains_no_bare_expressions_that_shell_out(self):
        path = Path(__file__).resolve().parent.parent / "app" / "jobs" / "ingest.py"
        source = path.read_text(encoding="utf-8")
        for forbidden in ("os.system", "os.popen", "eval(", "exec(", "__import__"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_ingestion_runs_without_any_of_those_modules_imported(self):
        import sys
        for module in ("playwright", "ollama"):
            with self.subTest(module=module):
                self.assertNotIn(
                    module, sys.modules,
                    f"{module} must never be imported by the ingestion path",
                )


if __name__ == "__main__":
    unittest.main()
