"""Tests for access verification.

No live requests. Every HTTP interaction is a stub, so this suite is offline
and deterministic. The real network call lives in exactly one function,
``app.sources.transport._urllib_open``.

The behaviour under test is a policy, and the policy has one shape: everything
short of explicit permission fails closed. These tests spend most of their
weight on the failure paths, because those are the ones that decide whether the
system is safe to point at somebody else's website.
"""

import json
import tempfile
import unittest
from pathlib import Path

from app.jobs.sources import (
    PERMITTED,
    RESTRICTED,
    UNKNOWN,
    SourceConfig,
    SourceRegistry,
)
from app.sources.access import (
    ROBOTS_ALLOWS,
    ROBOTS_AMBIGUOUS,
    ROBOTS_DISALLOWS,
    ROBOTS_UNREADABLE,
    TERMS_PROHIBITS,
    TERMS_REVIEWED_CLEAR,
    TERMS_UNREVIEWABLE,
    AccessDecision,
    AccessLevel,
    check_paths,
    load_decisions,
    render_report,
    review_terms,
    robots_url_for,
    save_decision,
    verify_access,
)
from app.sources.transport import (
    REFUSAL_STATUSES,
    AccessError,
    AccessFetcher,
    HttpResponse,
    Ledger,
    RateLimiter,
    response_from_refusal,
)

BASE = "https://example.test"

ROBOTS_ALLOWING = """
User-agent: *
Allow: /jobs/
Allow: /job/
Disallow: /admin/
Disallow: /*?page=*
"""

ROBOTS_DISALLOWING = """
User-agent: *
Disallow: /
"""

TERMS_CLEAR = "You may browse this site. Contact us for permission regarding bulk use."

TERMS_PROHIBITING = (
    "Automated means of collecting data from this site, including scraping, "
    "crawling and harvesting, are prohibited without our prior written consent."
)


def response(url, status=200, body="", headers=None, elapsed_ms=12):
    return HttpResponse(
        url=url, status=status, headers=dict(headers or {}), body=body,
        elapsed_ms=elapsed_ms,
    )


class ScriptedOpener:
    """Returns queued responses; records the URLs requested, in order."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.urls = []

    def __call__(self, url, timeout):
        self.urls.append(url)
        item = self.responses[min(len(self.urls) - 1, len(self.responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


def fetcher_for(*responses, ledger=None, jitter=None, **kwargs):
    opener = ScriptedOpener(*responses)
    return AccessFetcher(
        ledger=ledger or Ledger(),
        limiter=RateLimiter(0.0, sleeper=lambda _: None),
        opener=opener,
        sleeper=lambda _: None,
        # Identity jitter by default so backoff growth is observable. Tests
        # that are not about backoff pass jitter=lambda v: 0.0.
        jitter=jitter if jitter is not None else (lambda value: value),
        **kwargs,
    ), opener


def robots_and_terms(robots_body=ROBOTS_ALLOWING, terms_body=TERMS_CLEAR):
    return (
        response(f"{BASE}/robots.txt", body=robots_body),
        response(f"{BASE}/terms-and-conditions", body=terms_body),
    )


class RobotsEvaluationTests(unittest.TestCase):
    def test_relevant_path_allowed_by_robots(self):
        decisions = check_paths(ROBOTS_ALLOWS, ["/jobs/", "/job/"])
        self.assertTrue(all(d.allowed for d in decisions))

    def test_disallowed_path_is_detected(self):
        decisions = check_paths(ROBOTS_DISALLOWING, ["/jobs/"])
        self.assertFalse(decisions[0].allowed)

    def test_pagination_is_disallowed_on_an_allowing_site(self):
        # The live hiring.cafe policy disallows paginated listings; a check
        # that ignored query strings would miss it.
        decisions = check_paths(ROBOTS_ALLOWING, ["/jobs/?page=2"])
        self.assertFalse(decisions[0].allowed)

    def test_robots_url_is_derived_from_the_base(self):
        self.assertEqual(robots_url_for(BASE), f"{BASE}/robots.txt")

    def test_the_existing_rfc9309_evaluator_is_reused(self):
        # Delegation, not reimplementation: two parsers of one spec is how a
        # fail-open ships.
        import app.sources.access as module

        self.assertIsNotNone(module.robots_allowed, "tools/robots_check must be used")
        self.assertTrue(callable(module.robots_allowed))


class TermsReviewTests(unittest.TestCase):
    def test_clear_terms_are_reviewed_as_clear(self):
        status, _ = review_terms(TERMS_CLEAR)
        self.assertEqual(status, TERMS_REVIEWED_CLEAR)

    def test_prohibiting_terms_are_detected(self):
        status, evidence = review_terms(TERMS_PROHIBITING)
        self.assertEqual(status, TERMS_PROHIBITS)
        self.assertTrue(evidence)

    def test_a_page_that_only_mentions_robots_is_not_a_prohibition(self):
        status, _ = review_terms(
            "This site has a robots.txt file. See our help page for details."
        )
        self.assertEqual(status, TERMS_REVIEWED_CLEAR)


class DecisionTests(unittest.TestCase):
    def verify(self, *responses, **kwargs):
        fetcher, opener = fetcher_for(*responses)
        return verify_access("example.test", BASE, fetcher, **kwargs), opener

    def test_allowed_robots_and_clear_terms_permits(self):
        decision, _ = self.verify(*robots_and_terms())
        self.assertIs(decision.level, AccessLevel.PERMITTED)
        self.assertTrue(decision.permitted)
        self.assertTrue(decision.evidence)
        self.assertTrue(decision.checked_at)

    def test_disallowed_robots_restricts(self):
        decision, _ = self.verify(*robots_and_terms(ROBOTS_DISALLOWING))
        self.assertIs(decision.level, AccessLevel.RESTRICTED)
        self.assertIn("disallow", decision.reason.casefold())

    def test_unreachable_robots_is_unknown(self):
        fetcher, _ = fetcher_for(AccessError("connection refused", retryable=True))
        decision = verify_access("example.test", BASE, fetcher)
        self.assertIs(decision.level, AccessLevel.UNKNOWN)
        self.assertEqual(decision.robots, ROBOTS_UNREADABLE)

    def test_robots_returning_html_is_ambiguous_not_permitted(self):
        html = "<!doctype html><html><body>Cloudflare</body></html>"
        fetcher, _ = fetcher_for(
            response(f"{BASE}/robots.txt", body=html),
            response(f"{BASE}/terms-and-conditions", body=TERMS_CLEAR),
        )
        decision = verify_access("example.test", BASE, fetcher)
        self.assertIs(decision.level, AccessLevel.UNKNOWN)
        self.assertEqual(decision.robots, ROBOTS_AMBIGUOUS)

    def test_a_refused_robots_is_unreadable_not_empty(self):
        failure = AccessError("refused", status=403)
        failure.headers, failure.body, failure.elapsed_ms = {}, "", 5
        fetcher, _ = fetcher_for(failure)
        decision = verify_access("example.test", BASE, fetcher)
        self.assertIs(decision.level, AccessLevel.UNKNOWN)
        self.assertIn("refusal", decision.evidence[0])

    def test_prohibiting_terms_restrict_even_with_permissive_robots(self):
        decision, _ = self.verify(*robots_and_terms(terms_body=TERMS_PROHIBITING))
        self.assertIs(decision.level, AccessLevel.RESTRICTED)
        self.assertIn("terms", decision.reason.casefold())

    def test_unreadable_terms_yield_unknown_not_permitted(self):
        failure = AccessError("refused", status=403)
        failure.headers, failure.body, failure.elapsed_ms = {}, "", 5
        fetcher, _ = fetcher_for(
            response(f"{BASE}/robots.txt", body=ROBOTS_ALLOWING), failure
        )
        decision = verify_access("example.test", BASE, fetcher)
        self.assertIs(decision.level, AccessLevel.UNKNOWN)
        self.assertEqual(decision.terms, TERMS_UNREVIEWABLE)

    def test_a_permissive_robots_does_not_override_a_server_challenge(self):
        # The live hiring.cafe case: robots.txt says allow, the server says
        # challenge. The server wins.
        failure = AccessError("refused", status=403)
        failure.headers = {"Cf-Mitigated": "challenge", "Server": "cloudflare"}
        failure.body, failure.elapsed_ms = "", 5
        fetcher, _ = fetcher_for(failure, failure)
        decision = verify_access("hiring.cafe", BASE, fetcher)
        self.assertIs(decision.level, AccessLevel.RESTRICTED)
        self.assertIn("challenge", decision.reason.casefold())

    def test_only_two_requests_are_made(self):
        _, opener = self.verify(*robots_and_terms())
        self.assertEqual(len(opener.urls), 2)
        self.assertTrue(opener.urls[0].endswith("/robots.txt"))
        self.assertTrue(opener.urls[1].endswith("/terms-and-conditions"))

    def test_no_job_listing_is_ever_fetched(self):
        _, opener = self.verify(*robots_and_terms())
        for url in opener.urls:
            self.assertNotIn("/jobs", url)
            self.assertNotIn("/job/", url)


class ChallengeDetectionTests(unittest.TestCase):
    def test_cf_mitigated_header_is_detected(self):
        self.assertTrue(response("u", 403, headers={"Cf-Mitigated": "challenge"}).challenged)

    def test_a_plain_403_is_a_refusal_but_not_a_challenge(self):
        self.assertFalse(response("u", 403).challenged)
        self.assertTrue(response("u", 403).refusal)

    def test_header_lookup_is_case_insensitive(self):
        self.assertEqual(response("u", 403, headers={"CF-MITIGATED": "challenge"})
                         .header("cf-mitigated"), "challenge")

    def test_a_refusal_is_rebuilt_with_its_evidence(self):
        failure = AccessError("refused", status=403)
        failure.headers = {"Cf-Mitigated": "challenge"}
        failure.body, failure.elapsed_ms = "", 0
        rebuilt = response_from_refusal(failure, "u")
        self.assertTrue(rebuilt.challenged)


class RetryTests(unittest.TestCase):
    def test_429_honours_retry_after(self):
        # The penalty is what matters, so test the limiter directly: after a
        # Retry-After the next request waits at least that long.
        slept = []
        limiter = RateLimiter(1.0, clock=lambda: 100.0, sleeper=slept.append)
        limiter.wait()                       # first request establishes the clock
        limiter.penalise(30.0)
        limiter.wait()                       # must absorb the 30s penalty
        self.assertTrue(any(s >= 29.0 for s in slept),
                        f"Retry-After not honoured: slept {slept}")

    def test_a_429_is_retried_and_the_penalty_replaces_the_normal_backoff(self):
        failure = AccessError("slow down", status=429, retryable=True)
        failure.retry_after = 30.0
        slept = []
        fetcher, opener = fetcher_for(
            failure, response(f"{BASE}/robots.txt", body=ROBOTS_ALLOWING),
            max_attempts=3,
        )
        fetcher._sleep = slept.append
        fetcher.limiter = RateLimiter(0.0, sleeper=lambda _: None)
        result = fetcher.get(f"{BASE}/robots.txt", source="s", purpose="robots")
        self.assertEqual(result.status, 200)
        self.assertEqual(len(opener.urls), 2)
        # Retry-After was honoured via the limiter, so the exponential backoff
        # sleep was not also used.
        self.assertEqual(slept, [])

    def test_retry_after_parsing(self):
        self.assertEqual(response("u", 429, headers={"Retry-After": "12"}).retry_after_seconds(), 12.0)
        self.assertIsNone(response("u", 429).retry_after_seconds())
        self.assertIsNone(
            response("u", 429, headers={"Retry-After": "Wed, 21 Oct"}).retry_after_seconds()
        )

    def test_transient_5xx_retries_with_bounded_backoff(self):
        fetcher, opener = fetcher_for(
            AccessError("boom", status=503, retryable=True),
            AccessError("boom", status=503, retryable=True),
            response(f"{BASE}/robots.txt", body=ROBOTS_ALLOWING),
            max_attempts=3,
        )
        delays = []
        fetcher._sleep = delays.append
        fetcher.get(f"{BASE}/robots.txt", source="s", purpose="robots")
        self.assertEqual(len(opener.urls), 3)
        # Exponential growth, and bounded by max_attempts: exactly two waits
        # for three attempts.
        self.assertEqual(delays, [1.0, 2.0])

    def test_retries_stop_at_max_attempts(self):
        fetcher, opener = fetcher_for(
            AccessError("boom", status=503, retryable=True), max_attempts=2
        )
        with self.assertRaises(AccessError):
            fetcher.get(f"{BASE}/robots.txt", source="s", purpose="robots")
        self.assertEqual(len(opener.urls), 2)

    def test_a_refusal_is_never_retried(self):
        for status in sorted(REFUSAL_STATUSES):
            with self.subTest(status=status):
                failure = AccessError("no", status=status)
                failure.headers, failure.body, failure.elapsed_ms = {}, "", 0
                fetcher, opener = fetcher_for(failure, max_attempts=5)
                with self.assertRaises(AccessError):
                    fetcher.get(f"{BASE}/jobs/", source="s", purpose="probe")
                self.assertEqual(len(opener.urls), 1, f"{status} must not be retried")

    def test_a_timeout_is_retried(self):
        fetcher, opener = fetcher_for(
            AccessError("timed out", retryable=True),
            response(f"{BASE}/robots.txt", body=ROBOTS_ALLOWING),
        )
        fetcher._sleep = lambda _: None
        fetcher.get(f"{BASE}/robots.txt", source="s", purpose="robots")
        self.assertEqual(len(opener.urls), 2)


class RateLimitTests(unittest.TestCase):
    def test_an_immediate_repeat_request_waits(self):
        slept = []
        now = [0.0]
        limiter = RateLimiter(
            2.0, clock=lambda: now[0], sleeper=slept.append
        )
        limiter.wait()
        now[0] = 0.1
        limiter.wait()
        self.assertEqual(len(slept), 1)
        self.assertAlmostEqual(slept[0], 1.9, places=6)

    def test_no_wait_when_enough_time_has_passed(self):
        slept = []
        now = [0.0]
        limiter = RateLimiter(2.0, clock=lambda: now[0], sleeper=slept.append)
        limiter.wait()
        now[0] = 5.0
        limiter.wait()
        self.assertEqual(slept, [])

    def test_the_first_request_never_waits(self):
        slept = []
        limiter = RateLimiter(5.0, clock=lambda: 0.0, sleeper=slept.append)
        self.assertEqual(limiter.wait(), 0.0)
        self.assertEqual(slept, [])

    def test_two_requests_are_never_concurrent(self):
        # The limiter is inherently sequential: it advances a single
        # timestamp. Assert the property that makes that true.
        now = [0.0]
        limiter = RateLimiter(1.0, clock=lambda: now[0], sleeper=lambda s: now.__setitem__(0, now[0] + s))
        limiter.wait()
        self.assertAlmostEqual(limiter.wait(), 1.0, places=6)


class LedgerTests(unittest.TestCase):
    def test_every_attempt_is_recorded(self):
        ledger = Ledger()
        fetcher, _ = fetcher_for(
            AccessError("boom", status=503, retryable=True),
            response(f"{BASE}/robots.txt", body=ROBOTS_ALLOWING),
            ledger=ledger,
        )
        fetcher._sleep = lambda _: None
        fetcher.get(f"{BASE}/robots.txt", source="src", purpose="robots")
        outcomes = [a.outcome for a in ledger.for_source("src")]
        self.assertEqual(outcomes, ["transient_failure", "ok"])

    def test_an_attempt_records_source_time_purpose_outcome_and_duration(self):
        ledger = Ledger()
        fetcher, _ = fetcher_for(
            response(f"{BASE}/robots.txt", body=ROBOTS_ALLOWING), ledger=ledger
        )
        fetcher.get(f"{BASE}/robots.txt", source="src", purpose="robots")
        attempt = ledger.for_source("src")[0]
        self.assertEqual(attempt.source, "src")
        self.assertEqual(attempt.purpose, "robots")
        self.assertEqual(attempt.outcome, "ok")
        self.assertTrue(attempt.at)
        self.assertGreaterEqual(attempt.duration_ms, 0)

    def test_a_refusal_is_recorded_as_refused(self):
        failure = AccessError("no", status=403)
        failure.headers, failure.body, failure.elapsed_ms = {}, "", 0
        ledger = Ledger()
        fetcher, _ = fetcher_for(failure, ledger=ledger)
        with self.assertRaises(AccessError):
            fetcher.get(f"{BASE}/jobs/", source="src", purpose="probe")
        self.assertEqual(ledger.for_source("src")[0].outcome, "refused")

    def test_the_ledger_is_written_as_append_only_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempts.jsonl"
            ledger = Ledger(path)
            fetcher, _ = fetcher_for(
                response(f"{BASE}/robots.txt", body=ROBOTS_ALLOWING), ledger=ledger
            )
            fetcher.get(f"{BASE}/robots.txt", source="s", purpose="robots")
            fetcher.get(f"{BASE}/robots.txt", source="s", purpose="robots")
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["purpose"], "robots")


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.data = Path(self._directory.name)

    def decision(self, level=AccessLevel.PERMITTED):
        return AccessDecision(
            source="example.test", level=level, reason="because",
            robots=ROBOTS_ALLOWS, robots_url=f"{BASE}/robots.txt",
            terms=TERMS_REVIEWED_CLEAR, terms_url=f"{BASE}/terms-and-conditions",
            checked_at="2026-10-09T12:00:00+00:00",
            evidence=("something",),
        )

    def test_a_decision_persists_and_reloads(self):
        save_decision(self.decision(), self.data)
        loaded = load_decisions(self.data)
        self.assertIn("example.test", loaded)
        self.assertIs(loaded["example.test"].level, AccessLevel.PERMITTED)

    def test_a_restricted_decision_persists(self):
        save_decision(self.decision(AccessLevel.RESTRICTED), self.data)
        self.assertIs(
            load_decisions(self.data)["example.test"].level, AccessLevel.RESTRICTED
        )

    def test_a_corrupt_file_fails_closed_to_no_decisions(self):
        (self.data / "access.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(load_decisions(self.data), {})

    def test_no_file_means_no_decisions(self):
        self.assertEqual(load_decisions(self.data), {})


class RegistryEnforcementTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.data = Path(self._directory.name)

    def registry(self):
        return SourceRegistry([SourceConfig(name="example.test", access=PERMITTED)])

    def decision(self, level):
        return AccessDecision(
            source="example.test", level=level, reason="verified",
            robots=ROBOTS_ALLOWS, robots_url="", terms=TERMS_REVIEWED_CLEAR,
            terms_url="", checked_at="2026-10-09T12:00:00+00:00",
        )

    def test_a_permitted_decision_leaves_the_source_active(self):
        enforced = self.registry().enforce_recorded_decisions(
            {"example.test": self.decision(AccessLevel.PERMITTED)}
        )
        self.assertEqual([c.name for c in enforced.active()], ["example.test"])

    def test_a_restricted_source_cannot_become_active(self):
        enforced = self.registry().enforce_recorded_decisions(
            {"example.test": self.decision(AccessLevel.RESTRICTED)}
        )
        self.assertEqual(enforced.active(), [])
        self.assertEqual(enforced.skipped()[0].code, "access_restricted")

    def test_an_unknown_source_cannot_become_active(self):
        enforced = self.registry().enforce_recorded_decisions(
            {"example.test": self.decision(AccessLevel.UNKNOWN)}
        )
        self.assertEqual(enforced.active(), [])
        self.assertEqual(enforced.skipped()[0].code, "access_unknown")

    def test_a_source_with_no_recorded_decision_cannot_become_active(self):
        # A bare config claim is not a permission.
        enforced = self.registry().enforce_recorded_decisions({})
        self.assertEqual(enforced.active(), [])
        self.assertEqual(enforced.skipped()[0].code, "access_unknown")

    def test_the_skip_reason_carries_the_recorded_reason(self):
        decision = self.decision(AccessLevel.RESTRICTED)
        decision = AccessDecision(**{**decision.__dict__, "reason": "cloudflare challenge"})
        enforced = self.registry().enforce_recorded_decisions({"example.test": decision})
        self.assertIn("cloudflare challenge", enforced.skipped()[0].reason)

    def test_enforcement_does_not_mutate_the_original_registry(self):
        original = self.registry()
        original.enforce_recorded_decisions({})
        self.assertEqual([c.name for c in original.active()], ["example.test"])

    def test_a_recorded_decision_is_reachable_from_disk(self):
        save_decision(self.decision(AccessLevel.RESTRICTED), self.data)
        enforced = self.registry().enforce_recorded_decisions(load_decisions(self.data))
        self.assertEqual(enforced.active(), [], "persisted restriction must be honoured")

    def test_the_shipped_registry_stays_inactive(self):
        from app.jobs.sources import default_registry

        enforced = default_registry().enforce_recorded_decisions({})
        self.assertEqual(enforced.active(), [])


class ReportTests(unittest.TestCase):
    def decision(self, level=AccessLevel.PERMITTED, source="example.test"):
        return AccessDecision(
            source=source, level=level, reason="a reason", robots=ROBOTS_ALLOWS,
            robots_url="https://example.test/robots.txt",
            terms=TERMS_REVIEWED_CLEAR, terms_url="https://example.test/terms-and-conditions",
            checked_at="2026-10-09T12:00:00+00:00",
            evidence=("robots.txt (1234 bytes)",),
        )

    def test_the_report_states_the_decision_and_evidence(self):
        text = render_report([self.decision()])
        self.assertIn("permitted", text)
        self.assertIn("a reason", text)
        self.assertIn("robots.txt (1234 bytes)", text)
        self.assertIn("2026-10-09T12:00:00+00:00", text)

    def test_the_report_states_that_discovery_remains_disabled(self):
        self.assertIn("Job discovery remains disabled",
                      render_report([self.decision()]))

    def test_the_report_shows_path_decisions(self):
        from app.sources.access import PathDecision

        decision = self.decision()
        with_paths = AccessDecision(
            **{
                **decision.__dict__,
                "paths": (PathDecision("/jobs/", True), PathDecision("/admin/", False)),
            }
        )
        text = render_report([with_paths])
        self.assertIn("Path decisions", text)
        self.assertIn("ALLOW", text)
        self.assertIn("DISALLOW", text)
        self.assertIn("/jobs/", text)
        self.assertIn("/admin/", text)


class NoSecretsTests(unittest.TestCase):
    FORBIDDEN = ("cookie", "authorization", "password", "api_key", "apikey",
                 "bearer", "token", "session", "login", "auth")

    def sources(self):
        return [
            Path(__file__).resolve().parent.parent / "app" / "sources" / "transport.py",
            Path(__file__).resolve().parent.parent / "app" / "sources" / "access.py",
        ]

    def test_no_credential_or_cookie_headers_are_sent(self):
        # The only header this project ever sets is its identifying user agent.
        text = self.sources()[0].read_text(encoding="utf-8")
        self.assertIn('"User-Agent": USER_AGENT', text)
        self.assertEqual(text.count('"User-Agent":'), 1)

    def test_no_login_or_authentication_flow_exists(self):
        for path in self.sources():
            lowered = path.read_text(encoding="utf-8").casefold()
            with self.subTest(module=path.name):
                self.assertNotIn("login", lowered)
                self.assertNotIn("set-cookie", lowered)

    def test_no_personal_data_is_recorded(self):
        for path in self.sources():
            text = path.read_text(encoding="utf-8").casefold()
            with self.subTest(module=path.name):
                for token in ("email", "phone", "address", "username"):
                    self.assertNotIn(token, text)

    def test_only_one_network_call_exists(self):
        text = self.sources()[0].read_text(encoding="utf-8")
        self.assertEqual(text.count("urllib.request.urlopen"), 1)

    def test_the_user_agent_is_identifying_and_constant(self):
        from app.sources.transport import USER_AGENT

        self.assertIn("access-check", USER_AGENT)


if __name__ == "__main__":
    unittest.main()