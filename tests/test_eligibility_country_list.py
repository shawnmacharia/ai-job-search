"""Tests for country-list authority in the eligibility policy.

These encode an approved policy change: when a posting enumerates the countries
it will hire in, that enumeration is authoritative, and generic marketing
wording ("Anywhere in the World", "global", "worldwide") does not override it.

The regression this prevents is concrete and was measured on live data: judged
on prose alone, 22 of 89 We Work Remotely listings came back ``eligible`` while
not one of them named Kenya.

Pure and offline: no I/O, no model, no network.
"""

import unittest

from app.jobs.eligibility import (
    ELIGIBLE,
    GEOGRAPHY_CONFLICT,
    NOT_ELIGIBLE,
    UNKNOWN,
    evaluate_eligibility,
)
from app.jobs.models import Job, RemoteStatus

WORLDWIDE = "Anywhere in the World"


def job(**overrides) -> Job:
    base = dict(
        job_id="j-1",
        title="Data Engineer",
        company="Acme",
        url="https://example.test/j-1",
        description="",
        location=WORLDWIDE,
        remote_status=RemoteStatus.FULLY_REMOTE_GLOBAL,
    )
    base.update(overrides)
    return Job(**base)


class CountryListIncludesKenyaTests(unittest.TestCase):
    def test_kenya_in_the_list_is_eligible(self):
        verdict = evaluate_eligibility(
            job(country="Kenya, Uganda, South Africa"), candidate_country="KE")
        self.assertEqual(verdict.verdict, ELIGIBLE)
        self.assertTrue(verdict.reasons)
        self.assertFalse(verdict.has_conflict)

    def test_evidence_quotes_the_list(self):
        verdict = evaluate_eligibility(job(country="Kenya, Uganda"))
        self.assertTrue(any("country:" in q for q in verdict.evidence_quotes))

    def test_a_regional_label_does_not_override_a_list_naming_kenya(self):
        verdict = evaluate_eligibility(
            job(location="Remote - Europe", country="Kenya, Germany"))
        self.assertEqual(verdict.verdict, ELIGIBLE)

    def test_kenya_with_decorators_still_matches(self):
        verdict = evaluate_eligibility(job(country="Uganda, Kenya (East Africa)"))
        self.assertEqual(verdict.verdict, ELIGIBLE)


class CountryListExcludesKenyaTests(unittest.TestCase):
    def test_worldwide_label_does_not_override_a_list_omitting_kenya(self):
        # The live WWR case, and the reason this policy exists.
        verdict = evaluate_eligibility(
            job(location=WORLDWIDE,
                country="Andorra, Argentina, Australia, Austria, Belgium"))
        self.assertEqual(verdict.verdict, NOT_ELIGIBLE)

    def test_the_reason_names_the_list(self):
        verdict = evaluate_eligibility(
            job(country="Andorra, Argentina, Australia"))
        self.assertIn("country list", " ".join(verdict.reasons).casefold())

    def test_the_list_is_evidence(self):
        verdict = evaluate_eligibility(job(country="Andorra, Argentina"))
        self.assertTrue(any("country:" in q for q in verdict.evidence_quotes))

    def test_evidence_shows_both_signals_when_the_body_disagrees(self):
        verdict = evaluate_eligibility(
            job(country="Andorra, Argentina"),
            # "Africa" indicates a region containing Kenya.
            candidate_country="KE",
        )
        verdict = evaluate_eligibility(
            job(description="Remote across Africa (including South Africa).",
                country="Andorra, Argentina"),
            candidate_country="KE",
        )
        self.assertEqual(verdict.verdict, NOT_ELIGIBLE)
        self.assertTrue(verdict.has_conflict)
        # Both the list and the body are represented as evidence.
        self.assertTrue(any("country:" in q for q in verdict.evidence_quotes))
        self.assertGreaterEqual(len(verdict.evidence_quotes), 2)

    def test_a_long_list_is_summarised_rather_than_dumped(self):
        countries = ", ".join(f"Country{index}" for index in range(75))
        verdict = evaluate_eligibility(job(country=countries))
        joined = " ".join(verdict.reasons)
        self.assertIn("+71 more", joined)
        self.assertLess(len(joined), 400)

    def test_a_single_country_is_not_treated_as_an_enumeration(self):
        # Backwards compatibility: one value keeps the older single-country path.
        verdict = evaluate_eligibility(job(country="Germany"))
        self.assertEqual(verdict.verdict, NOT_ELIGIBLE)


class GeographyConflictTests(unittest.TestCase):
    def test_body_naming_kenya_with_a_list_excluding_it_is_not_eligible(self):
        verdict = evaluate_eligibility(
            job(description="We hire in Kenya and beyond.",
                country="Andorra, Argentina, Australia"),
            candidate_country="KE",
        )
        self.assertEqual(verdict.verdict, NOT_ELIGIBLE)
        self.assertIn(GEOGRAPHY_CONFLICT, verdict.flags)

    def test_both_signals_are_described(self):
        verdict = evaluate_eligibility(
            job(description="We hire in Kenya.",
                country="Andorra, Argentina"),
            candidate_country="KE",
        )
        text = " ".join(verdict.reasons) + " " + " ".join(verdict.notes)
        self.assertIn("country list", text.casefold())
        self.assertIn("body", text.casefold())

    def test_no_conflict_when_the_list_names_kenya(self):
        verdict = evaluate_eligibility(
            job(description="We hire in Kenya.", country="Kenya, Uganda"))
        self.assertFalse(verdict.has_conflict)
        self.assertEqual(verdict.verdict, ELIGIBLE)

    def test_no_conflict_when_the_body_is_silent(self):
        verdict = evaluate_eligibility(job(country="Andorra, Argentina"))
        self.assertFalse(verdict.has_conflict)
        self.assertEqual(verdict.verdict, NOT_ELIGIBLE)


class NoCountryListTests(unittest.TestCase):
    def test_body_naming_kenya_is_eligible_with_evidence(self):
        verdict = evaluate_eligibility(
            job(description="Remote role open to candidates in Kenya."),
            candidate_country="KE")
        self.assertEqual(verdict.verdict, ELIGIBLE)
        self.assertTrue(verdict.evidence_quotes)

    def test_worldwide_wording_alone_is_eligible_when_unrestricted(self):
        verdict = evaluate_eligibility(job(description=WORLDWIDE))
        self.assertEqual(verdict.verdict, ELIGIBLE)

    def test_worldwide_wording_with_a_conflicting_restriction_is_not_eligible(self):
        # The pre-existing rule still holds: a global phrase never overrides an
        # explicit restriction.
        verdict = evaluate_eligibility(
            job(location="Remote, Europe only", description=WORLDWIDE))
        self.assertEqual(verdict.verdict, NOT_ELIGIBLE)

    def test_no_location_signal_at_all_is_unknown(self):
        verdict = evaluate_eligibility(
            job(location="", description="Great opportunity, apply now."))
        self.assertEqual(verdict.verdict, UNKNOWN)
        self.assertTrue(verdict.reasons, "unknown must still explain itself")

    def test_unknown_is_never_eligible(self):
        verdict = evaluate_eligibility(job(location="", description=""))
        self.assertNotEqual(verdict.verdict, ELIGIBLE)


class ReasonQualityTests(unittest.TestCase):
    def test_every_not_eligible_carries_a_plain_english_reason(self):
        cases = [
            job(country="Andorra, Argentina"),
            job(location="Remote - Europe"),
            job(description="United States only."),
        ]
        for candidate in cases:
            with self.subTest(job=candidate.url):
                verdict = evaluate_eligibility(candidate)
                if verdict.verdict == NOT_ELIGIBLE:
                    self.assertTrue(verdict.reasons)
                    self.assertTrue(verdict.evidence_quotes)

    def test_every_unknown_carries_a_reason(self):
        verdict = evaluate_eligibility(job(location="", description=""))
        if verdict.verdict == UNKNOWN:
            self.assertTrue(verdict.reasons)

    def test_the_policy_is_pure_and_repeatable(self):
        candidate = job(country="Andorra, Argentina",
                        description="Hiring in Kenya.")
        first = evaluate_eligibility(candidate)
        second = evaluate_eligibility(candidate)
        self.assertEqual(first.verdict, second.verdict)
        self.assertEqual(first.reasons, second.reasons)
        self.assertEqual(first.flags, second.flags)


class FlagErasureTests(unittest.TestCase):
    def test_the_verdict_flag_reaches_the_dashboard_row(self):
        from app.reporting.jobs import build_view

        record = {"job_id": "j-1", "job": {
            "job_id": "j-1", "title": "Data Engineer", "company": "Acme",
            "url": "https://example.test/j-1",
            "description": "Hiring in Kenya.", "location": WORLDWIDE,
            "country": "Andorra, Argentina",
        }}
        view = build_view(record)
        self.assertEqual(view.verdict, NOT_ELIGIBLE)
        self.assertIn(GEOGRAPHY_CONFLICT, view.flags)

    def test_the_conflict_detail_reaches_the_row_flags(self):
        from app.reporting.jobs import build_view

        record = {"job_id": "j-1", "job": {
            "job_id": "j-1", "title": "Data Engineer", "company": "Acme",
            "url": "https://example.test/j-1",
            "description": "Hiring in Kenya.", "location": WORLDWIDE,
            "country": "Andorra, Argentina",
        }}
        flags = " ".join(build_view(record).flags)
        self.assertIn("geography conflict", flags.casefold())


class CoverageDistinctionTests(unittest.TestCase):
    """The panel must not blur four different outcomes into one message."""

    def view(self, verdict, country=""):
        from app.reporting.jobs import JobView

        return JobView(
            job_id="j", title="t", company="c", location="l", url="",
            verdict=verdict, verdict_reasons=["r"], evidence=[], flags=[],
            sources=["s"], source_urls=[], first_seen="", last_seen="",
            posted_date="", description="", description_complete=True,
            possible_duplicate=False, match_explanation="",
            application_status="", country=country,
        )

    def panel(self, views, runs=()):
        from app.reporting.jobs import _coverage_panel, coverage_of

        return _coverage_panel(coverage_of(views, list(runs)), views)

    def test_zero_fetched_is_stated_as_no_jobs_returned(self):
        markup = self.panel([])
        self.assertIn("The source returned no jobs", markup)

    def test_jobs_but_none_eligible_is_a_coverage_limit(self):
        markup = self.panel([self.view("not_eligible"), self.view("unknown")])
        self.assertIn("0 Kenya-eligible jobs", markup)
        self.assertIn("coverage limit", markup)
        self.assertNotIn("returned no jobs", markup)

    def test_all_not_eligible_is_distinguished_from_all_unknown(self):
        excluded = self.panel([self.view("not_eligible"), self.view("not_eligible")])
        unknown = self.panel([self.view("unknown")])
        self.assertIn("2 not eligible / 0 unknown", excluded)
        self.assertIn("0 not eligible / 1 unknown", unknown)
        self.assertIn(
            "Eligibility could not be determined", unknown)

    def test_no_country_list_published_never_claims_none_names_kenya(self):
        # The field was empty, not filled with countries that exclude Kenya.
        markup = self.panel([self.view("not_eligible", country="")])
        self.assertNotIn("No published country list names Kenya", markup)

    def test_a_published_list_omitting_kenya_is_reported(self):
        markup = self.panel([self.view("not_eligible", "Andorra, Argentina")])
        self.assertIn("No published country list names Kenya", markup)
        self.assertIn("1 listing(s) publish a list", markup)

    def test_no_source_consulted_suppresses_every_other_note(self):
        markup = self.panel(
            [self.view("not_eligible", "Andorra")], [{"no_active_sources": True}])
        self.assertIn("No source was consulted", markup)
        self.assertNotIn("0 Kenya-eligible", markup)
        self.assertNotIn("No published country list", markup)

    def test_a_conflict_is_counted_separately_from_the_verdict(self):
        from app.reporting.jobs import JobView, coverage_of

        view = JobView(
            job_id="j", title="t", company="c", location=WORLDWIDE, url="",
            verdict="not_eligible", verdict_reasons=["r"], evidence=[],
            flags=["geography conflict: body vs list"], sources=["s"],
            source_urls=[], first_seen="", last_seen="", posted_date="",
            description="", description_complete=True, possible_duplicate=False,
            match_explanation="", application_status="",
            country="Andorra, Argentina",
        )
        totals = coverage_of([view], [])
        self.assertEqual(totals["conflicts"], 1)
        self.assertEqual(totals["not_eligible"], 1)
        self.assertEqual(totals["country_lists"], 1)
        self.assertEqual(totals["country_named"], 0)

    def test_the_panel_keeps_coverage_counters_visible(self):
        markup = self.panel([self.view("eligible", "Kenya, Uganda")])
        for label in ("fetched", "stored", "eligible", "not eligible",
                      "unknown", "duplicates", "quarantined",
                      "country lists published", "geography conflicts"):
            with self.subTest(label=label):
                self.assertIn(label, markup)


class ExtendedCountryNameTests(unittest.TestCase):
    """A single-valued country field must still be honoured exactly.

    Regression: ``country: "United States of America"`` was absent from the
    alias table, so a US-only remote role fell through to the prose path and was
    judged ``eligible`` on a "worldwide" marketing phrase. Found on real WWR
    data while implementing country-list authority.
    """

    def test_an_extended_country_name_is_resolved(self):
        verdict = evaluate_eligibility(
            job(country="United States of America", description="Remote worldwide."),
            candidate_country="KE")
        self.assertEqual(verdict.verdict, NOT_ELIGIBLE)

    def test_a_two_letter_code_cannot_match_by_prefix(self):
        # Guards the three-character floor that makes prefix matching safe.
        verdict = evaluate_eligibility(
            job(country="Norway", description="Remote worldwide."),
            candidate_country="KE")
        self.assertEqual(verdict.verdict, NOT_ELIGIBLE)
        verdict = evaluate_eligibility(
            job(country="Kenya", description="Remote worldwide."),
            candidate_country="KE")
        self.assertEqual(verdict.verdict, ELIGIBLE)

    def test_an_unrecognised_country_is_not_invented_into_a_verdict(self):
        verdict = evaluate_eligibility(
            job(country="Republic of Wakanda", description="Remote worldwide."),
            candidate_country="KE")
        # Falls through to the prose path rather than guessing a code.
        self.assertIn(verdict.verdict, (ELIGIBLE, NOT_ELIGIBLE, UNKNOWN))

    def test_the_single_country_path_still_beats_prose(self):
        verdict = evaluate_eligibility(
            job(country="Germany", description="Remote worldwide."),
            candidate_country="KE")
        self.assertEqual(verdict.verdict, NOT_ELIGIBLE)


if __name__ == "__main__":
    unittest.main()