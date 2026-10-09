"""Tests for the source-record -> canonical ``Job`` adapter contract.

Regression under test: ``hiring_cafe`` emits ``description_snippet``,
``posted``, ``salary`` and ``work_mode``, while ``normalize_job`` reads
``description`` and ``date``. Eight of eleven emitted fields were discarded and
every downstream stage received an empty description.

Fixtures are saved parser output, not live scrapes: no network, no Playwright,
no Ollama. They are byte-compatible with what ``_parse_card_text`` emits, so a
change to that shape breaks these tests.
"""

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app.jobs.adapters import (
    AdaptError,
    HiringCafeAdapter,
    adapt_record,
    parse_posted_date,
    parse_salary,
)
from app.jobs.eligibility import evaluate_eligibility
from app.jobs.models import Job
from app.jobs.store import JobStore
from app.state.models import RemoteStatus

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hiring_cafe"
NOW = datetime(2026, 10, 9, 12, 0, 0, tzinfo=timezone.utc)

#: Every key ``hiring_cafe._parse_card_text`` emits, plus the field added
#: afterwards by the keyword filter. TestEmittedFieldsAreAccountedFor asserts
#: this set stays fully accounted for, so a new scraper field fails a test
#: rather than being silently dropped.
EMITTED_KEYS = frozenset({
    "title", "company", "company_blurb", "location", "salary", "work_mode",
    "employment_type", "posted", "description_snippet", "url",
    "matched_query", "skill_match_count",
})


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def adapt(name, *, now=NOW):
    return adapt_record("hiring.cafe", load(name), now=now)


class HeadlineRegressionTests(unittest.TestCase):
    def test_saved_record_produces_non_empty_description_and_posted_date(self):
        job = adapt("basic.json")
        self.assertTrue(job.description.strip(), "description must survive adaptation")
        self.assertTrue(job.posted_date, "posted_date must survive adaptation")
        self.assertEqual(job.description, load("basic.json")["description_snippet"])
        self.assertEqual(job.posted_date, "2026-10-07")

    def test_card_fields_survive_where_normalize_job_dropped_them(self):
        # The exact failure this increment exists to fix: normalize_job on the
        # same record yields an empty description.
        from app.jobs.normalize import normalize_job

        raw = load("basic.json")
        legacy = normalize_job(raw)
        self.assertEqual(legacy.description, "", "precondition: normalize_job still drops it")

        job = adapt("basic.json")
        self.assertNotEqual(job.description, "")
        self.assertNotEqual(job.posted_date, None)


class FieldAccountingTests(unittest.TestCase):
    def test_every_emitted_field_is_mapped_or_preserved_as_raw_evidence(self):
        adapter = HiringCafeAdapter()
        for fixture in sorted(FIXTURES.glob("*.json")):
            with self.subTest(fixture=fixture.name):
                raw = load(fixture.name)
                try:
                    job = adapter.adapt(raw, now=NOW)
                except AdaptError:
                    continue  # malformed fixtures are covered separately
                accounted = set(adapter.consumed_keys) | set(job.raw_excerpt)
                emitted = {key for key in raw if not key.startswith("_")}
                self.assertEqual(
                    emitted - accounted, set(),
                    f"{fixture.name}: emitted fields silently discarded: "
                    f"{sorted(emitted - accounted)}",
                )

    def test_declared_emitted_keys_are_all_consumed_or_preserved(self):
        adapter = HiringCafeAdapter()
        job = adapt("salary_year.json")
        accounted = set(adapter.consumed_keys) | set(job.raw_excerpt)
        self.assertEqual(EMITTED_KEYS - accounted, set(),
                         "a key the scraper emits is neither mapped nor preserved")

    def test_unmapped_fields_survive_in_raw_excerpt(self):
        raw = load("salary_year.json")
        job = adapt("salary_year.json")
        self.assertEqual(job.raw_excerpt["company_blurb"], raw["company_blurb"])
        self.assertEqual(job.raw_excerpt["employment_type"], raw["employment_type"])
        self.assertEqual(job.raw_excerpt["matched_query"], raw["matched_query"])
        self.assertEqual(job.raw_excerpt["skill_match_count"], raw["skill_match_count"])

    def test_consumed_keys_are_not_duplicated_into_raw_excerpt(self):
        job = adapt("basic.json")
        for key in HiringCafeAdapter().consumed_keys:
            self.assertNotIn(key, job.raw_excerpt, f"{key} was mapped, not preserved raw")


class SalaryParsingTests(unittest.TestCase):
    def test_salary_parsing_table(self):
        cases = [
            # (text, min, max, currency, period)
            ("$80k-$120k/yr", 80000, 120000, None, "year"),
            ("€4k-€6k/mo", 4000, 6000, "EUR", "month"),
            ("$95,000", 95000, 95000, None, None),
            ("£45k", 45000, 45000, "GBP", None),
            ("KSh 1.2M/month", 1200000, 1200000, "KES", "month"),
            ("USD 70,000 - 90,000 annually", 70000, 90000, "USD", "year"),
            ("AUD 120k", 120000, 120000, "AUD", None),
            ("Naira 400,000 monthly", 400000, 400000, "NGN", "month"),
            ("£52,000 per annum", 52000, 52000, "GBP", "year"),
            ("$1.2M-$1.5M", 1200000, 1500000, None, None),
            ("$30/hour", 30, 30, None, "hour"),
            ("$100k", 100000, 100000, None, None),
            # European thousands separator, not a decimal (CHANGELOG #326)
            ("€3.500 per month", 3500, 3500, "EUR", "month"),
            ("€80.000 per year", 80000, 80000, "EUR", "year"),
            # Nothing parseable -> explicit None, never an error
            ("Competitive", None, None, None, None),
            ("Competitive - DOE", None, None, None, None),
            ("", None, None, None, None),
            (None, None, None, None, None),
        ]
        for text, low, high, currency, period in cases:
            with self.subTest(text=text):
                result = parse_salary(text)
                self.assertEqual(result["salary_min"], low)
                self.assertEqual(result["salary_max"], high)
                self.assertEqual(result["salary_currency"], currency)
                self.assertEqual(result["salary_period"], period)

    def test_bare_dollar_does_not_claim_a_currency(self):
        # Explicitly decided: "$" is ambiguous, so the field stays empty
        # rather than asserting USD.
        self.assertIsNone(parse_salary("$80k-$120k/yr")["salary_currency"])
        self.assertIsNone(parse_salary("$120,000")["salary_currency"])

    def test_unambiguous_symbols_do_set_a_currency(self):
        self.assertEqual(parse_salary("€3.500 per month")["salary_currency"], "EUR")
        self.assertEqual(parse_salary("£45k")["salary_currency"], "GBP")
        self.assertEqual(parse_salary("KSh 1.2M/month")["salary_currency"], "KES")

    def test_salary_reaches_the_job_from_a_fixture(self):
        job = adapt("salary_year.json")
        self.assertEqual(job.salary_min, 80000)
        self.assertEqual(job.salary_max, 120000)
        self.assertEqual(job.salary_period, "year")
        self.assertIsNone(job.salary_currency)


class PostedDateParsingTests(unittest.TestCase):
    def test_posted_date_table(self):
        cases = [
            ("2d ago", "2026-10-07"),
            ("3h ago", "2026-10-09"),
            ("1w ago", "2026-10-02"),
            ("Today", "2026-10-09"),
            ("Yesterday", "2026-10-08"),
            ("2026-09-01", "2026-09-01"),
            ("recently", None),
            ("", None),
            (None, None),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(parse_posted_date(text, now=NOW), expected)

    def test_relative_dates_are_deterministic_under_the_injected_clock(self):
        early = datetime(2026, 10, 1, tzinfo=timezone.utc)
        late = datetime(2026, 10, 9, tzinfo=timezone.utc)
        self.assertEqual(parse_posted_date("2d ago", now=early), "2026-09-29")
        self.assertEqual(parse_posted_date("2d ago", now=late), "2026-10-07")

    def test_unparseable_posted_keeps_the_original_string(self):
        job = adapt("unparseable_posted.json")
        self.assertIsNone(job.posted_date)
        self.assertEqual(job.posted_raw, "recently")


class FailureTests(unittest.TestCase):
    def test_missing_description_raises_rather_than_returning_empty(self):
        with self.assertRaises(AdaptError) as caught:
            adapt("missing_description.json")
        self.assertIn("description_snippet", caught.exception.reason)

    def test_missing_company_raises(self):
        with self.assertRaises(AdaptError) as caught:
            adapt("missing_company.json")
        self.assertIn("company", caught.exception.reason)

    def test_missing_title_or_url_raises(self):
        base = load("basic.json")
        for field in ("title", "url"):
            with self.subTest(field=field):
                broken = dict(base, **{field: None})
                with self.assertRaises(AdaptError):
                    adapt_record("hiring.cafe", broken, now=NOW)

    def test_non_mapping_record_raises(self):
        for value in ("a string", 42, None, ["list"]):
            with self.subTest(value=value):
                with self.assertRaises(AdaptError):
                    adapt_record("hiring.cafe", value, now=NOW)

    def test_unknown_source_raises(self):
        with self.assertRaises(AdaptError) as caught:
            adapt_record("nonexistent-source", load("basic.json"), now=NOW)
        self.assertIn("unknown source", caught.exception.reason)

    def test_adapt_never_raises_an_unexpected_exception_type(self):
        """Contract discipline: data problems become AdaptError, not anything else."""
        broken_values = [
            {}, {"title": "x"}, {"title": "x", "company": "y"},
            {"title": "x", "company": "y", "url": "z", "description_snippet": "d"},
            {"title": "", "company": "y", "url": "z", "description_snippet": "d"},
            "not a mapping", 7, None,
        ]
        for value in broken_values:
            with self.subTest(value=value):
                try:
                    adapt_record("hiring.cafe", value, now=NOW)
                except AdaptError:
                    pass
                except Exception as exc:  # pragma: no cover - the assertion
                    self.fail(f"raised {type(exc).__name__} instead of AdaptError: {exc}")


class ProvenanceAndCompletenessTests(unittest.TestCase):
    def test_portal_is_set_from_the_adapter_name(self):
        self.assertEqual(adapt("basic.json").portal, "hiring.cafe")

    def test_snippets_are_marked_incomplete(self):
        for name in ("basic.json", "salary_year.json", "worldwide_remote.json"):
            with self.subTest(name=name):
                self.assertFalse(
                    adapt(name).description_complete,
                    "a card snippet must never claim to be a complete posting",
                )

    def test_remote_status_is_derived_from_card_text(self):
        self.assertEqual(adapt("basic.json").remote_status, RemoteStatus.FULLY_REMOTE_COUNTRY_RESTRICTED)
        self.assertEqual(adapt("salary_ksh.json").remote_status, RemoteStatus.ONSITE)

    def test_restricted_region_survives_into_eligibility(self):
        # The P1 description fix makes P0's eligibility judgement possible:
        # with an empty description this could only ever return "unknown".
        verdict = evaluate_eligibility(adapt("europe_restricted.json"), candidate_country="KE")
        self.assertEqual(verdict.verdict, "not_eligible")

        worldwide = evaluate_eligibility(adapt("worldwide_remote.json"), candidate_country="KE")
        self.assertEqual(worldwide.verdict, "eligible")


class BackwardCompatibilityTests(unittest.TestCase):
    def test_existing_job_construction_still_works(self):
        # Positional construction, exactly as normalize_job does it.
        job = Job("id-1", "Data Engineer", "Acme", "https://example.test/1")
        self.assertEqual(job.title, "Data Engineer")
        self.assertEqual(job.description, "")
        self.assertFalse(job.description_complete)
        self.assertEqual(job.raw_excerpt, {})
        self.assertIsNone(job.posted_raw)

    def test_new_fields_default_safely_and_are_not_shared(self):
        first = Job("a", "T", "C", "u")
        second = Job("b", "T2", "C2", "u2")
        first.raw_excerpt["x"] = 1
        self.assertEqual(second.raw_excerpt, {}, "mutable defaults must not be shared")

    def test_normalize_job_output_is_unchanged(self):
        from app.jobs.normalize import normalize_job

        job = normalize_job({"id": "x", "title": "T", "company": "C", "url": "u", "description": "d"})
        self.assertEqual(job.description, "d")
        self.assertEqual(job.raw_excerpt, {})
        self.assertFalse(job.description_complete)


class StoreIntegrationTests(unittest.TestCase):
    def test_store_round_trips_an_enriched_job_without_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory) / "data")
            raw = load("salary_year.json")
            job = adapt_record("hiring.cafe", raw, now=NOW)

            result = store.store([job.__dict__], source="hiring.cafe", observed_at=NOW.isoformat())
            self.assertEqual(result.stored, 1)

            stored = store.load_jobs()[0]["job"]
            for field in (
                "title", "company", "url", "description", "location",
                "salary_min", "salary_max", "salary_currency", "salary_period",
                "posted_date", "portal", "raw_excerpt", "posted_raw",
                "description_complete",
            ):
                with self.subTest(field=field):
                    self.assertEqual(
                        stored[field], getattr(job, field),
                        f"{field} was lost between adapter and storage",
                    )
            self.assertEqual(stored["description"], raw["description_snippet"])
            self.assertEqual(stored["raw_excerpt"]["company_blurb"], raw["company_blurb"])

    def test_malformed_records_are_quarantined_by_the_store_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory) / "data")
            store.store(
                [load("basic.json"), load("missing_description.json")],
                source="hiring.cafe",
                observed_at=NOW.isoformat(),
            )
            # The malformed one was stored raw; the adapter is what rejects it.
            # Either way it is not lost - it is visible in the store.
            self.assertEqual(len(store.load_jobs()), 2)


if __name__ == "__main__":
    unittest.main()
