"""Freshness and expiry: a job that stops appearing must not be confused with
a source that stopped working.

Every test here pins one of the distinctions the feature exists to preserve.
The dangerous cases are the negative ones - the run where the source timed out,
the run where the feed came back empty - because both look like "no jobs" to
anything that only looks at counts.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.jobs.freshness import (
    DEFAULT_POLICIES,
    FreshnessLedger,
    FreshnessState,
    JobFreshness,
    SourceObservation,
    SourcePolicy,
    SourceState,
    apply_observation,
    evaluate,
    policy_for,
    source_report,
    summarise,
)
from app.jobs.runner import SourceSpec, run_sources
from app.jobs.status import ReviewStatus, StatusLog
from app.jobs.store import JobStore
from app.reporting.jobs import (
    build_view,
    render_dashboard_file,
    render_dashboard_html,
    render_row,
)

T0 = datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc)


def at(days: float = 0.0, hours: float = 0.0) -> str:
    """A timestamp offset from the fixed test epoch."""
    moment = T0 + timedelta(days=days, hours=hours)
    return moment.isoformat(timespec="seconds")


def record(job_id="j1", title="Role", company="Acme", url=None, source="weworkremotely"):
    return {
        "url": url or f"https://example.test/{job_id}",
        "title": title,
        "company": company,
        "location": "Nairobi, Kenya",
        "description": "A role.",
    }


def make_store(tmp: str) -> JobStore:
    return JobStore(Path(tmp) / "data")


def job_ids(store: JobStore) -> Dict[str, str]:
    """Map the test's logical names to the ids the store actually assigned.

    The store derives ids from the URL, so they are hashes. Tests must never
    hard-code them - that would break the moment id derivation changes, and
    would quietly assert against the wrong job in the meantime.
    """
    mapping = {}
    for record in store.load_jobs():
        key = str(record["job"]["url"]).rsplit("/", 1)[-1]
        mapping[key] = str(record["job_id"])
    return mapping


def outcome(name, ok=True, fetched=0, stored=0, error=None):
    from app.jobs.store import SourceOutcome

    return SourceOutcome(
        name=name, ok=ok, fetched=fetched, stored=stored, error=error
    )


class SourcePolicyTests(unittest.TestCase):
    def test_expiry_cannot_precede_staleness(self):
        with self.assertRaises(ValueError):
            SourcePolicy(source="x", stale_days=30, expire_days=10)

    def test_zero_result_grace_must_allow_at_least_one_run(self):
        with self.assertRaises(ValueError):
            SourcePolicy(source="x", zero_result_grace_runs=0)

    def test_an_unknown_source_still_gets_a_real_policy(self):
        """No source may fall through to ``None``.

        A caller that forgot to configure a source should get the conservative
        default, not a crash and not an unbounded window.
        """
        policy = policy_for("a-source-nobody-configured")
        self.assertGreater(policy.stale_days, 0)
        self.assertGreaterEqual(policy.expire_days, policy.stale_days)

    def test_the_two_live_sources_have_different_windows(self):
        """WWR rotates fast; MyJobMag is polled once a day and holds older work."""
        wwr = policy_for("weworkremotely")
        mag = policy_for("myjobmag.co.ke")
        self.assertLess(wwr.stale_days, mag.stale_days)


class ObservationShapeTests(unittest.TestCase):
    def test_failure_is_not_evidence(self):
        observation = SourceObservation(source="weworkremotely", at=at(), ok=False)
        self.assertTrue(observation.is_failure)
        self.assertFalse(observation.is_evidence)

    def test_a_skip_is_not_evidence_either(self):
        observation = SourceObservation(
            source="weworkremotely", at=at(), ok=False, skipped=True
        )
        self.assertFalse(observation.is_failure)
        self.assertFalse(observation.is_evidence)

    def test_a_zero_result_is_evidence_but_the_weakest_kind(self):
        observation = SourceObservation(source="weworkremotely", at=at(), ok=True)
        self.assertTrue(observation.is_evidence)
        self.assertTrue(observation.is_zero_result)


class UpdateOnSightTests(unittest.TestCase):
    def test_a_successful_observation_updates_last_seen(self):
        state = apply_observation(
            {},
            SourceObservation(
                source="weworkremotely", at=at(days=3), ok=True,
                returned_job_ids=("j1",),
            ),
            tracked_job_ids=["j1"],
        )
        self.assertEqual(state["j1"].sources["weworkremotely"].last_seen, at(days=3))
        self.assertEqual(state["j1"].last_seen, at(days=3))
        self.assertIs(state["j1"].state, FreshnessState.ACTIVE)

    def test_last_seen_advances_on_each_sighting(self):
        state = {}
        for day in (0, 5, 10):
            state = apply_observation(
                state,
                SourceObservation(
                    source="weworkremotely", at=at(days=day), ok=True,
                    returned_job_ids=("j1",),
                ),
                tracked_job_ids=["j1"],
            )
        self.assertEqual(state["j1"].sources["weworkremotely"].last_seen, at(days=10))
        self.assertEqual(state["j1"].sources["weworkremotely"].observations, 3)

    def test_a_second_source_sees_the_same_job(self):
        state = apply_observation(
            {},
            SourceObservation(
                source="weworkremotely", at=at(), ok=True, returned_job_ids=("j1",)
            ),
            tracked_job_ids=["j1"],
        )
        state = apply_observation(
            state,
            SourceObservation(
                source="myjobmag.co.ke", at=at(days=1), ok=True, returned_job_ids=("j1",)
            ),
            tracked_job_ids=["j1"],
        )
        self.assertEqual(
            sorted(state["j1"].sources), ["myjobmag.co.ke", "weworkremotely"]
        )
        self.assertEqual(state["j1"].last_seen, at(days=1))


class FailureLeavesEverythingAloneTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {
            "j1": JobFreshness(
                job_id="j1",
                state=FreshnessState.ACTIVE,
                last_seen=at(days=40),
                sources={
                    "weworkremotely": apply_observation(
                        {},
                        SourceObservation(
                            source="weworkremotely", at=at(days=40), ok=True,
                            returned_job_ids=("j1",),
                        ),
                        tracked_job_ids=["j1"],
                    )["j1"].sources["weworkremotely"]
                },
            )
        }

    def test_a_failed_source_changes_nothing(self):
        """A job last seen 40 days ago stays exactly as it was after a failure.

        Without this, one timeout would expire the entire corpus.
        """
        after = apply_observation(
            self.baseline,
            SourceObservation(
                source="weworkremotely", at=at(days=60), ok=False, error="timeout"
            ),
            tracked_job_ids=["j1"],
            now=T0 + timedelta(days=60),
        )
        source = after["j1"].sources["weworkremotely"]
        self.assertEqual(source.last_seen, at(days=40))
        self.assertEqual(source.consecutive_misses, 0)
        self.assertEqual(source.consecutive_zero_results, 0)
        self.assertIsNone(source.stale_since)
        self.assertIsNone(source.expired_since)

    def test_a_failed_source_cannot_push_a_job_past_expiry(self):
        after = apply_observation(
            self.baseline,
            SourceObservation(source="weworkremotely", at=at(days=999), ok=False),
            tracked_job_ids=["j1"],
            now=T0 + timedelta(days=999),
        )
        self.assertIsNot(after["j1"].state, FreshnessState.EXPIRED)

    def test_a_skipped_source_changes_nothing(self):
        after = apply_observation(
            self.baseline,
            SourceObservation(
                source="weworkremotely", at=at(days=60), ok=False, skipped=True
            ),
            tracked_job_ids=["j1"],
            now=T0 + timedelta(days=60),
        )
        self.assertEqual(
            after["j1"].sources["weworkremotely"].last_seen, at(days=40)
        )

    def test_a_failure_does_not_erase_the_miss_count(self):
        """Counters survive a failure; they are not reset by it either."""
        aged = apply_observation(
            self.baseline,
            SourceObservation(source="weworkremotely", at=at(days=45), ok=True),
            tracked_job_ids=["j1"],
            now=T0 + timedelta(days=45),
        )
        missed = aged["j1"].sources["weworkremotely"].consecutive_misses
        after = apply_observation(
            aged,
            SourceObservation(source="weworkremotely", at=at(days=46), ok=False),
            tracked_job_ids=["j1"],
            now=T0 + timedelta(days=46),
        )
        self.assertEqual(
            after["j1"].sources["weworkremotely"].consecutive_misses, missed
        )


class StalenessTests(unittest.TestCase):
    def test_absence_within_the_window_stays_active(self):
        state = apply_observation(
            {},
            SourceObservation(
                source="weworkremotely", at=at(), ok=True, returned_job_ids=("j1", "j2")
            ),
            tracked_job_ids=["j1"],
        )
        state = apply_observation(
            state,
            SourceObservation(
                source="weworkremotely", at=at(days=5), ok=True,
                returned_job_ids=("j2",),
            ),
            tracked_job_ids=["j1"],
            now=T0 + timedelta(days=5),
        )
        self.assertIs(state["j1"].state, FreshnessState.ACTIVE)

    def test_absence_past_the_stale_window_marks_stale(self):
        policy = DEFAULT_POLICIES["weworkremotely"]
        state = apply_observation(
            {},
            SourceObservation(
                source="weworkremotely", at=at(), ok=True,
                returned_job_ids=("j1", "j2"),
            ),
            tracked_job_ids=["j1"],
        )
        state = apply_observation(
            state,
            SourceObservation(
                source="weworkremotely", at=at(days=policy.stale_days + 1), ok=True,
                returned_job_ids=("j2",),
            ),
            tracked_job_ids=["j1"],
            now=T0 + timedelta(days=policy.stale_days + 1),
        )
        self.assertIs(state["j1"].state, FreshnessState.STALE)
        self.assertIsNotNone(state["j1"].sources["weworkremotely"].stale_since)

    def test_absence_past_the_expiry_window_marks_expired(self):
        policy = DEFAULT_POLICIES["weworkremotely"]
        state = apply_observation(
            {},
            SourceObservation(
                source="weworkremotely", at=at(), ok=True, returned_job_ids=("j1", "j2")
            ),
            tracked_job_ids=["j1"],
        )
        state = apply_observation(
            state,
            SourceObservation(
                source="weworkremotely", at=at(days=policy.expire_days + 1), ok=True,
                returned_job_ids=("j2",),
            ),
            tracked_job_ids=["j1"],
            now=T0 + timedelta(days=policy.expire_days + 1),
        )
        self.assertIs(state["j1"].state, FreshnessState.EXPIRED)
        self.assertIsNotNone(state["j1"].sources["weworkremotely"].expired_since)

    def test_repeated_absence_keeps_the_job_marked(self):
        """Repeated successful absence must not quietly clear the mark."""
        policy = DEFAULT_POLICIES["weworkremotely"]
        state = apply_observation(
            {},
            SourceObservation(
                source="weworkremotely", at=at(), ok=True,
                returned_job_ids=("j1", "j2"),
            ),
            tracked_job_ids=["j1"],
        )
        for day in range(policy.stale_days + 1, policy.stale_days + 6):
            state = apply_observation(
                state,
                SourceObservation(
                    source="weworkremotely", at=at(days=day), ok=True,
                    returned_job_ids=("j2",),
                ),
                tracked_job_ids=["j1"],
                now=T0 + timedelta(days=day),
            )
        self.assertIs(state["j1"].state, FreshnessState.STALE)
        self.assertGreaterEqual(
            state["j1"].sources["weworkremotely"].consecutive_misses, 5
        )


class ZeroResultThresholdTests(unittest.TestCase):
    def test_one_empty_feed_does_not_stale_anything(self):
        """A single empty response must not touch the corpus, however old."""
        policy = DEFAULT_POLICIES["weworkremotely"]
        state = apply_observation(
            {},
            SourceObservation(
                source="weworkremotely", at=at(), ok=True,
                returned_job_ids=("j1", "j2"),
            ),
            tracked_job_ids=["j1"],
        )
        later = policy.stale_days + 10
        state = apply_observation(
            state,
            SourceObservation(source="weworkremotely", at=at(days=later), ok=True),
            tracked_job_ids=["j1"],
            now=T0 + timedelta(days=later),
        )
        self.assertIs(state["j1"].state, FreshnessState.ACTIVE)

    def test_the_configured_threshold_releases_the_grace(self):
        policy = DEFAULT_POLICIES["weworkremotely"]
        state = apply_observation(
            {},
            SourceObservation(
                source="weworkremotely", at=at(), ok=True,
                returned_job_ids=("j1", "j2"),
            ),
            tracked_job_ids=["j1"],
        )
        # One empty feed: still in grace.
        state = apply_observation(
            state,
            SourceObservation(source="weworkremotely", at=at(days=20), ok=True),
            tracked_job_ids=["j1"],
            now=T0 + timedelta(days=20),
        )
        self.assertIs(state["j1"].state, FreshnessState.ACTIVE)
        # A second consecutive empty feed exhausts the grace; now it can go stale.
        state = apply_observation(
            state,
            SourceObservation(source="weworkremotely", at=at(days=21), ok=True),
            tracked_job_ids=["j1"],
            now=T0 + timedelta(days=21),
        )
        self.assertIs(state["j1"].state, FreshnessState.STALE)

    def test_a_non_empty_feed_resets_the_grace_counter(self):
        """A recovered feed clears the empty-feed grace for every job.

        Without the reset, one empty response would leave the grace counter
        primed and the *next* empty response would immediately stale the corpus
        - the single failure would be counted twice.
        """
        state = apply_observation(
            {},
            SourceObservation(
                source="weworkremotely", at=at(), ok=True, returned_job_ids=("j1", "j2")
            ),
            tracked_job_ids=["j1"],
        )
        state = apply_observation(
            state,
            SourceObservation(source="weworkremotely", at=at(days=2), ok=True),
            tracked_job_ids=["j1"],
            now=T0 + timedelta(days=2),
        )
        self.assertEqual(
            state["j1"].sources["weworkremotely"].consecutive_zero_results, 1
        )
        # The feed comes back with content but still without j1.
        state = apply_observation(
            state,
            SourceObservation(
                source="weworkremotely", at=at(days=3), ok=True, returned_job_ids=("j2",)
            ),
            tracked_job_ids=["j1"],
            now=T0 + timedelta(days=3),
        )
        self.assertEqual(
            state["j1"].sources["weworkremotely"].consecutive_zero_results, 0
        )
        # Still well inside the stale window, and no grace is pending.
        self.assertIs(state["j1"].state, FreshnessState.ACTIVE)

    def test_zero_results_never_expire_immediately(self):
        policy = DEFAULT_POLICIES["weworkremotely"]
        state = apply_observation(
            {},
            SourceObservation(
                source="weworkremotely", at=at(), ok=True,
                returned_job_ids=("j1",),
            ),
            tracked_job_ids=["j1"],
        )
        state = apply_observation(
            state,
            SourceObservation(source="weworkremotely", at=at(days=1), ok=True),
            tracked_job_ids=["j1"],
            now=T0 + timedelta(days=1),
        )
        self.assertIsNot(state["j1"].state, FreshnessState.EXPIRED)


class ReappearanceTests(unittest.TestCase):
    def _stale_job(self):
        policy = DEFAULT_POLICIES["weworkremotely"]
        state = apply_observation(
            {},
            SourceObservation(
                source="weworkremotely", at=at(), ok=True,
                returned_job_ids=("j1", "j2"),
            ),
            tracked_job_ids=["j1"],
        )
        state = apply_observation(
            state,
            SourceObservation(
                source="weworkremotely", at=at(days=policy.stale_days + 1), ok=True,
                returned_job_ids=("j2",),
            ),
            tracked_job_ids=["j1"],
            now=T0 + timedelta(days=policy.stale_days + 1),
        )
        self.assertIs(state["j1"].state, FreshnessState.STALE)
        return state

    def test_a_job_that_returns_becomes_active_again(self):
        state = self._stale_job()
        back = policy_for("weworkremotely").stale_days + 2
        state = apply_observation(
            state,
            SourceObservation(
                source="weworkremotely", at=at(days=back), ok=True,
                returned_job_ids=("j1", "j2"),
            ),
            tracked_job_ids=["j1"],
            now=T0 + timedelta(days=back),
        )
        self.assertIs(state["j1"].state, FreshnessState.ACTIVE)

    def test_reappearance_clears_the_stale_mark(self):
        state = self._stale_job()
        back = policy_for("weworkremotely").stale_days + 2
        state = apply_observation(
            state,
            SourceObservation(
                source="weworkremotely", at=at(days=back), ok=True, returned_job_ids=("j1",)
            ),
            tracked_job_ids=["j1"],
            now=T0 + timedelta(days=back),
        )
        source = state["j1"].sources["weworkremotely"]
        self.assertIsNone(source.stale_since)
        self.assertEqual(source.consecutive_misses, 0)

    def test_reappearance_keeps_the_history(self):
        """How long the job had been missing must stay legible after it returns."""
        state = self._stale_job()
        back = policy_for("weworkremotely").stale_days + 2
        state = apply_observation(
            state,
            SourceObservation(
                source="weworkremotely", at=at(days=back), ok=True, returned_job_ids=("j1",)
            ),
            tracked_job_ids=["j1"],
            now=T0 + timedelta(days=back),
        )
        source = state["j1"].sources["weworkremotely"]
        self.assertGreaterEqual(source.observations, 3)

    def test_an_expired_job_can_also_come_back(self):
        policy = DEFAULT_POLICIES["weworkremotely"]
        state = apply_observation(
            {},
            SourceObservation(
                source="weworkremotely", at=at(), ok=True, returned_job_ids=("j1", "j2")
            ),
            tracked_job_ids=["j1"],
        )
        state = apply_observation(
            state,
            SourceObservation(
                source="weworkremotely", at=at(days=policy.expire_days + 1), ok=True,
                returned_job_ids=("j2",),
            ),
            tracked_job_ids=["j1"],
            now=T0 + timedelta(days=policy.expire_days + 1),
        )
        self.assertIs(state["j1"].state, FreshnessState.EXPIRED)
        back = policy.expire_days + 2
        state = apply_observation(
            state,
            SourceObservation(
                source="weworkremotely", at=at(days=back), ok=True, returned_job_ids=("j1",)
            ),
            tracked_job_ids=["j1"],
            now=T0 + timedelta(days=back),
        )
        self.assertIs(state["j1"].state, FreshnessState.ACTIVE)


class MultiSourceTests(unittest.TestCase):
    def test_per_source_states_are_kept_apart(self):
        """The job takes the worst state, but each source keeps its own answer.

        A reviewer needs to know *which* source went quiet: one source quietly
        dropping a listing is a different fact from every source agreeing it is
        gone. The rollup says "stale"; the detail says which source to trust.
        """
        wwr = DEFAULT_POLICIES["weworkremotely"]
        state = apply_observation(
            {},
            SourceObservation(
                source="weworkremotely", at=at(), ok=True, returned_job_ids=("j1", "j2")
            ),
            tracked_job_ids=["j1"],
        )
        state = apply_observation(
            state,
            SourceObservation(
                source="myjobmag.co.ke", at=at(), ok=True, returned_job_ids=("j1",)
            ),
            tracked_job_ids=["j1"],
        )
        gone = wwr.stale_days + 1
        state = apply_observation(
            state,
            SourceObservation(
                source="weworkremotely", at=at(days=gone), ok=True, returned_job_ids=("j2",)
            ),
            tracked_job_ids=["j1"],
            now=T0 + timedelta(days=gone),
        )
        self.assertIs(state["j1"].state, FreshnessState.STALE)
        self.assertIs(state["j1"].state_for("weworkremotely"), "stale")
        self.assertIs(state["j1"].state_for("myjobmag.co.ke"), "active")

    def test_a_failed_source_does_not_degrade_a_healthy_one(self):
        """One source failing must not drag the other's currency down with it."""
        wwr = DEFAULT_POLICIES["weworkremotely"]
        state = apply_observation(
            {},
            SourceObservation(
                source="weworkremotely", at=at(), ok=True, returned_job_ids=("j1",)
            ),
            tracked_job_ids=["j1"],
        )
        state = apply_observation(
            state,
            SourceObservation(
                source="myjobmag.co.ke", at=at(), ok=True, returned_job_ids=("j1",)
            ),
            tracked_job_ids=["j1"],
        )
        later = wwr.stale_days + 1
        state = apply_observation(
            state,
            SourceObservation(
                source="myjobmag.co.ke", at=at(days=later), ok=False, error="timeout"
            ),
            tracked_job_ids=["j1"],
            now=T0 + timedelta(days=later),
        )
        self.assertIs(state["j1"].state, FreshnessState.ACTIVE)
        self.assertIs(state["j1"].state_for("weworkremotely"), "active")
        self.assertIs(state["j1"].state_for("myjobmag.co.ke"), "active")

    def test_the_job_goes_stale_only_once_every_source_has_gone_quiet(self):
        wwr = DEFAULT_POLICIES["weworkremotely"]
        mag = DEFAULT_POLICIES["myjobmag.co.ke"]
        state = apply_observation(
            {},
            SourceObservation(
                source="weworkremotely", at=at(), ok=True, returned_job_ids=("j1", "j2")
            ),
            tracked_job_ids=["j1"],
        )
        state = apply_observation(
            state,
            SourceObservation(
                source="myjobmag.co.ke", at=at(), ok=True, returned_job_ids=("j1",)
            ),
            tracked_job_ids=["j1"],
        )
        day = max(wwr.stale_days, mag.stale_days) + 1
        state = apply_observation(
            state,
            SourceObservation(
                source="weworkremotely", at=at(days=day), ok=True, returned_job_ids=("j2",)
            ),
            tracked_job_ids=["j1"],
            now=T0 + timedelta(days=day),
        )
        state = apply_observation(
            state,
            SourceObservation(source="myjobmag.co.ke", at=at(days=day), ok=True),
            tracked_job_ids=["j1"],
            now=T0 + timedelta(days=day),
        )
        self.assertIs(state["j1"].state, FreshnessState.STALE)


class SourceReportTests(unittest.TestCase):
    """A source that has never produced a job must still be reportable."""

    def test_a_source_that_always_fails_is_still_listed(self):
        observations = [
            SourceObservation(source="s", at=at(days=1), ok=False, error="timeout"),
            SourceObservation(source="s", at=at(days=2), ok=False, error="timeout"),
        ]
        rows = source_report(observations)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["outcome"], "failing")
        self.assertEqual(rows[0]["consecutive_failures"], 2)
        self.assertIsNone(rows[0]["last_success"])

    def test_a_success_clears_the_failure_run(self):
        observations = [
            SourceObservation(source="s", at=at(days=1), ok=False, error="timeout"),
            SourceObservation(source="s", at=at(days=2), ok=True, returned_job_ids=("a",)),
        ]
        rows = source_report(observations)
        self.assertEqual(rows[0]["outcome"], "ok")
        self.assertEqual(rows[0]["consecutive_failures"], 0)
        self.assertEqual(rows[0]["last_success"], at(days=2))

    def test_an_empty_but_successful_run_is_not_a_failure(self):
        rows = source_report([SourceObservation(source="s", at=at(), ok=True)])
        self.assertEqual(rows[0]["outcome"], "empty")
        self.assertEqual(rows[0]["consecutive_failures"], 0)

    def test_a_skip_does_not_count_as_a_failure(self):
        """Deliberately not consulting a source is not the same as it breaking."""
        rows = source_report([
            SourceObservation(source="s", at=at(), ok=False, skipped=True,
                              error="not cleared for access"),
        ])
        self.assertEqual(rows[0]["outcome"], "skipped")
        self.assertEqual(rows[0]["consecutive_failures"], 0)

    def test_a_skipped_source_still_counts_as_consulted_before(self):
        rows = source_report([
            SourceObservation(source="s", at=at(days=1), ok=True, returned_job_ids=("a",)),
            SourceObservation(source="s", at=at(days=2), ok=False, skipped=True),
        ])
        self.assertEqual(rows[0]["last_success"], at(days=1))
        self.assertEqual(rows[0]["consecutive_failures"], 0)

    def test_sources_are_reported_independently(self):
        rows = source_report([
            SourceObservation(source="a", at=at(), ok=True, returned_job_ids=("x",)),
            SourceObservation(source="b", at=at(days=1), ok=False, error="timeout"),
        ])
        self.assertEqual([row["source"] for row in rows], ["a", "b"])
        self.assertEqual(rows[1]["outcome"], "failing")

    def test_summarise_carries_outcomes_for_jobless_sources(self):
        report = summarise({}, observations=[
            SourceObservation(source="never-worked", at=at(), ok=False, error="timeout")
        ])
        self.assertEqual(len(report["outcomes"]), 1)
        self.assertEqual(report["outcomes"][0]["outcome"], "failing")

    def test_summarise_without_observations_reports_none(self):
        report = summarise({})
        self.assertEqual(report["outcomes"], [])
        self.assertEqual(report["sources"], [])


class PurityTests(unittest.TestCase):
    def test_apply_observation_does_not_mutate_its_input(self):
        original = apply_observation(
            {},
            SourceObservation(
                source="weworkremotely", at=at(), ok=True, returned_job_ids=("j1",)
            ),
            tracked_job_ids=["j1"],
        )
        before = json.dumps({k: v.to_dict() for k, v in original.items()}, sort_keys=True)
        apply_observation(
            original,
            SourceObservation(source="weworkremotely", at=at(days=90), ok=True),
            tracked_job_ids=["j1"],
            now=T0 + timedelta(days=90),
        )
        after = json.dumps({k: v.to_dict() for k, v in original.items()}, sort_keys=True)
        self.assertEqual(before, after)

    def test_evaluate_is_order_independent_of_the_input_listing(self):
        """Observations are sorted by time, so a shuffled log gives the same answer."""
        observations = [
            SourceObservation(source="weworkremotely", at=at(days=2), ok=True,
                              returned_job_ids=("j1",)),
            SourceObservation(source="weworkremotely", at=at(), ok=True,
                              returned_job_ids=("j1", "j2")),
        ]
        forward = evaluate(observations, tracked_job_ids=["j1"],
                           now=T0 + timedelta(days=2))
        backward = evaluate(list(reversed(observations)), tracked_job_ids=["j1"],
                            now=T0 + timedelta(days=2))
        self.assertEqual(
            forward["j1"].to_dict(), backward["j1"].to_dict()
        )

    def test_round_trip_through_dict(self):
        state = apply_observation(
            {},
            SourceObservation(
                source="weworkremotely", at=at(days=4), ok=True, returned_job_ids=("j1",)
            ),
            tracked_job_ids=["j1"],
        )
        restored = JobFreshness.from_dict(state["j1"].to_dict())
        self.assertEqual(restored.to_dict(), state["j1"].to_dict())


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = make_store(self._tmp.name)

    def test_observations_are_appended_not_replaced(self):
        ledger = FreshnessLedger(self.store)
        ledger.record([SourceObservation(source="s", at=at(), ok=True,
                                         returned_job_ids=("a",))])
        ledger.record([SourceObservation(source="s", at=at(days=1), ok=True,
                                         returned_job_ids=("b",))])
        rows = [
            json.loads(line)
            for line in (self.store.data_dir / "freshness.jsonl")
            .read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["returned_job_ids"], ["a"])
        self.assertEqual(rows[1]["returned_job_ids"], ["b"])

    def test_a_failed_source_is_recorded_as_non_testimony(self):
        ledger = FreshnessLedger(self.store)
        ledger.record([
            SourceObservation(source="s", at=at(), ok=True, returned_job_ids=("a",)),
            SourceObservation(source="s", at=at(days=99), ok=False, error="timeout"),
        ])
        loaded = ledger.load()
        self.assertEqual(len(loaded), 2)
        self.assertFalse(loaded[-1].is_evidence)

    def test_evaluate_defaults_to_every_stored_job(self):
        self.store.store([record("j1"), record("j2")], source="weworkremotely")
        ids = job_ids(self.store)
        ledger = FreshnessLedger(self.store)
        ledger.record([SourceObservation(
            source="weworkremotely", at=at(), ok=True,
            returned_job_ids=(ids["j1"], ids["j2"]),
        )])
        state = ledger.evaluate(now=parse_now(at()))
        self.assertIn(ids["j1"], state)
        self.assertIn(ids["j2"], state)
        self.assertIs(state[ids["j1"]].state, FreshnessState.ACTIVE)
        self.assertEqual(state[ids["j1"]].last_seen, at())

    def test_summarise_separates_job_counts_from_source_health(self):
        self.store.store([record("j1"), record("j2")], source="weworkremotely")
        ledger = FreshnessLedger(self.store)
        ledger.record([SourceObservation(
            source="weworkremotely", at=at(), ok=True, returned_job_ids=("j1", "j2")
        )])
        report = summarise(ledger.evaluate(now=parse_now(at())))
        self.assertIn("jobs", report)
        self.assertIn("sources", report)
        self.assertEqual(report["jobs"]["active"], 2)
        self.assertEqual(report["sources"][0]["source"], "weworkremotely")


def parse_now(value: str) -> datetime:
    from app.jobs.freshness import parse_at

    return parse_at(value)


class RunnerIntegrationTests(unittest.TestCase):
    """The runner is the only place that knows a run really happened."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = make_store(self._tmp.name)

    def test_a_run_writes_one_observation_per_source(self):
        run_sources(
            [SourceSpec(name="weworkremotely", fetch=lambda: [record("j1")])],
            store=self.store,
            observed_at=T0,
            record_freshness=True,
        )
        observations = FreshnessLedger(self.store).load()
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].source, "weworkremotely")
        self.assertTrue(observations[0].is_evidence)

    def test_a_failing_source_is_recorded_as_a_failure_not_an_empty_feed(self):
        def boom():
            raise RuntimeError("source exploded")

        run_sources(
            [SourceSpec(name="weworkremotely", fetch=boom)],
            store=self.store,
            observed_at=T0,
            record_freshness=True,
        )
        observations = FreshnessLedger(self.store).load()
        self.assertEqual(len(observations), 1)
        self.assertTrue(observations[0].is_failure)
        self.assertFalse(observations[0].is_evidence)

    def test_a_skipped_source_is_recorded_as_a_skip(self):
        run_sources(
            [],
            store=self.store,
            observed_at=T0,
            skipped=[{"name": "myjobmag.co.ke", "reason": "not cleared",
                      "code": "restricted"}],
            record_freshness=True,
        )
        observations = FreshnessLedger(self.store).load()
        self.assertEqual(len(observations), 1)
        self.assertTrue(observations[0].skipped)
        self.assertFalse(observations[0].is_evidence)

    def test_an_empty_source_is_recorded_as_a_zero_result(self):
        run_sources(
            [SourceSpec(name="weworkremotely", fetch=lambda: [])],
            store=self.store,
            observed_at=T0,
            record_freshness=True,
        )
        observations = FreshnessLedger(self.store).load()
        self.assertTrue(observations[0].is_evidence)
        self.assertTrue(observations[0].is_zero_result)

    def test_a_failed_source_leaves_stored_freshness_untouched(self):
        run_sources(
            [SourceSpec(name="weworkremotely", fetch=lambda: [record("j1")])],
            store=self.store,
            observed_at=T0,
            record_freshness=True,
        )
        first = job_ids(self.store)["j1"]
        before = FreshnessLedger(self.store).evaluate()

        def boom():
            raise RuntimeError("source exploded")

        run_sources(
            [SourceSpec(name="weworkremotely", fetch=boom)],
            store=self.store,
            observed_at=T0 + timedelta(days=400),
            record_freshness=True,
        )
        after = FreshnessLedger(self.store).evaluate()
        self.assertEqual(
            before[first].sources["weworkremotely"].last_seen,
            after[first].sources["weworkremotely"].last_seen,
        )
        self.assertEqual(
            before[first].sources["weworkremotely"].consecutive_misses,
            after[first].sources["weworkremotely"].consecutive_misses,
        )
        # 400 days have passed and a failure still must not expire anything.
        self.assertIsNot(after[first].state, FreshnessState.EXPIRED)

    def test_reappearance_via_repeated_runs_makes_a_job_active_again(self):
        policy = DEFAULT_POLICIES["weworkremotely"]
        run_sources(
            [SourceSpec(name="weworkremotely", fetch=lambda: [record("j1"), record("j2")])],
            store=self.store,
            observed_at=T0,
            record_freshness=True,
        )
        first = job_ids(self.store)["j1"]
        gone = policy.stale_days + 1
        run_sources(
            [SourceSpec(name="weworkremotely", fetch=lambda: [record("j2")])],
            store=self.store,
            observed_at=T0 + timedelta(days=gone),
            record_freshness=True,
        )
        state = FreshnessLedger(self.store).evaluate()
        self.assertIs(state[first].state, FreshnessState.STALE)

        run_sources(
            [SourceSpec(name="weworkremotely", fetch=lambda: [record("j1"), record("j2")])],
            store=self.store,
            observed_at=T0 + timedelta(days=gone + 1),
            record_freshness=True,
        )
        state = FreshnessLedger(self.store).evaluate()
        self.assertIs(state[first].state, FreshnessState.ACTIVE)

    def test_a_repeat_run_does_not_create_a_second_job(self):
        run_sources(
            [SourceSpec(name="weworkremotely", fetch=lambda: [record("j1")])],
            store=self.store,
            observed_at=T0,
            record_freshness=True,
        )
        first = job_ids(self.store)["j1"]
        run_sources(
            [SourceSpec(name="weworkremotely", fetch=lambda: [record("j1")])],
            store=self.store,
            observed_at=T0 + timedelta(days=1),
            record_freshness=True,
        )
        self.assertEqual(len(self.store.load_jobs()), 1)
        state = FreshnessLedger(self.store).evaluate()
        self.assertIs(state[first].state, FreshnessState.ACTIVE)
        self.assertEqual(state[first].sources["weworkremotely"].observations, 2)


class ExpiryNeverDestroysTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = make_store(self._tmp.name)

    def test_an_expired_job_is_still_persisted_in_full(self):
        run_sources(
            [SourceSpec(name="weworkremotely", fetch=lambda: [record("j1"), record("j2")])],
            store=self.store,
            observed_at=T0,
            record_freshness=True,
        )
        first = job_ids(self.store)["j1"]
        policy = DEFAULT_POLICIES["weworkremotely"]
        gone = policy.expire_days + 5
        run_sources(
            [SourceSpec(name="weworkremotely", fetch=lambda: [record("j2")])],
            store=self.store,
            observed_at=T0 + timedelta(days=gone),
            record_freshness=True,
        )
        state = FreshnessLedger(self.store).evaluate()
        self.assertIs(state[first].state, FreshnessState.EXPIRED)
        # Still there, in full, with its history.
        stored = {r["job_id"]: r for r in self.store.load_jobs()}
        self.assertIn(first, stored)
        self.assertEqual(stored[first]["job"]["title"], "Role")
        self.assertTrue(stored[first]["first_seen"])

    def test_expiry_does_not_reset_the_first_seen_timestamp(self):
        run_sources(
            [SourceSpec(name="weworkremotely", fetch=lambda: [record("j1")])],
            store=self.store,
            observed_at=T0,
            record_freshness=True,
        )
        first = self.store.load_jobs()[0]["first_seen"]
        for day in (1, 2, 3):
            run_sources(
                [SourceSpec(name="weworkremotely", fetch=lambda: [])],
                store=self.store,
                observed_at=T0 + timedelta(days=day),
                record_freshness=True,
            )
        self.assertEqual(self.store.load_jobs()[0]["first_seen"], first)

    def test_expiry_leaves_the_review_status_untouched(self):
        run_sources(
            [SourceSpec(name="weworkremotely", fetch=lambda: [record("j1"), record("j2")])],
            store=self.store,
            observed_at=T0,
            record_freshness=True,
        )
        job_id = job_ids(self.store)["j1"]
        log = StatusLog(self.store)
        log.record(job_id, ReviewStatus.INTERESTED, note="worth a call")
        before = log.current(job_id)

        policy = DEFAULT_POLICIES["weworkremotely"]
        run_sources(
            [SourceSpec(name="weworkremotely", fetch=lambda: [record("j2")])],
            store=self.store,
            observed_at=T0 + timedelta(days=policy.expire_days + 5),
            record_freshness=True,
        )
        self.assertIs(
            FreshnessLedger(self.store).evaluate()[job_id].state, FreshnessState.EXPIRED
        )
        after = log.current(job_id)
        self.assertEqual(before.value, after.value)
        self.assertEqual([event.note for event in log.history(job_id)], ["worth a call"])
        self.assertEqual(len(log.history(job_id)), 1)

    def test_expiry_writes_nothing_back_into_the_job_record(self):
        """Freshness is derived; the job record must not gain a mutable field."""
        run_sources(
            [SourceSpec(name="weworkremotely", fetch=lambda: [record("j1"), record("j2")])],
            store=self.store,
            observed_at=T0,
            record_freshness=True,
        )
        before = self.store.load_jobs()
        policy = DEFAULT_POLICIES["weworkremotely"]
        run_sources(
            [SourceSpec(name="weworkremotely", fetch=lambda: [record("j2")])],
            store=self.store,
            observed_at=T0 + timedelta(days=policy.expire_days + 5),
            record_freshness=True,
        )
        after = self.store.load_jobs()
        self.assertEqual(len(before), len(after))
        for record_before, record_after in zip(before, after):
            self.assertNotIn("freshness", record_after)
            self.assertNotIn("expired", record_after)


def source_state(name: str, last_seen: str, misses: int, state: str) -> SourceState:
    """A per-source freshness record, built the way the module builds them."""
    return SourceState(
        source=name,
        last_seen=last_seen,
        consecutive_misses=misses,
        consecutive_zero_results=0,
        observations=misses + 1,
        state=state,
    )


def job_state(job_id: str, state: str, last_seen: str, **sources) -> JobFreshness:
    return JobFreshness(
        job_id=job_id, state=FreshnessState(state), last_seen=last_seen,
        sources=dict(sources),
    )


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = make_store(self._tmp.name)
        run_sources(
            [SourceSpec(name="weworkremotely", fetch=lambda: [record("j1")])],
            store=self.store,
            observed_at=T0,
            record_freshness=False,
        )

    def _stored(self):
        return self.store.load_jobs()[0]

    def test_an_unobserved_job_reads_unknown_not_active(self):
        view = build_view(self._stored())
        self.assertEqual(view.freshness, "unknown")

    def test_freshness_reaches_the_view(self):
        freshness = job_state(
            "x", "stale", at(days=30),
            weworkremotely=source_state("weworkremotely", at(days=30), 3, "stale"),
        )
        view = build_view(self._stored(), freshness=freshness)
        self.assertEqual(view.freshness, "stale")
        self.assertEqual(view.freshness_last_seen, at(days=30))
        self.assertEqual(view.freshness_missing_sources, ["weworkremotely"])
        self.assertIn("no longer listed by", view.freshness_detail)

    def test_a_still_listed_source_is_named_separately(self):
        freshness = job_state(
            "x", "stale", at(days=30),
            **{
                "weworkremotely": source_state("weworkremotely", at(days=30), 3, "stale"),
                "myjobmag.co.ke": source_state("myjobmag.co.ke", at(days=2), 0, "active"),
            },
        )
        view = build_view(self._stored(), freshness=freshness)
        self.assertEqual(view.freshness_sources, ["myjobmag.co.ke"])
        self.assertEqual(view.freshness_missing_sources, ["weworkremotely"])
        self.assertIn("listed by myjobmag.co.ke", view.freshness_detail)

    def test_the_row_shows_the_state_and_the_last_seen(self):
        freshness = job_state(
            "x", "expired", at(days=100),
            weworkremotely=source_state("weworkremotely", at(days=100), 6, "expired"),
        )
        row = render_row(build_view(self._stored(), freshness=freshness))
        self.assertIn("expired", row)
        self.assertIn("last seen", row)
        self.assertIn("fresh-expired", row)

    def test_each_state_renders_distinctly(self):
        rows = {}
        for state in ("active", "stale", "expired", "unknown"):
            freshness = None if state == "unknown" else job_state(
                "x", state, at(),
                weworkremotely=source_state("weworkremotely", at(), 0, state),
            )
            rows[state] = render_row(build_view(self._stored(), freshness=freshness))
        self.assertEqual(len(set(rows.values())), 4)

    def test_the_panel_reports_freshness_and_source_health_separately(self):
        freshness = job_state(
            "j", "active", at(),
            weworkremotely=source_state("weworkremotely", at(), 0, "active"),
        )
        health = {"sources": [{"source": "weworkremotely", "state": "active",
                               "jobs": 1, "last_seen": at()}]}
        html = render_dashboard_html(
            [build_view(self._stored(), freshness=freshness)],
            source_health=health,
        )
        self.assertIn("fresh-panel", html)
        self.assertIn("weworkremotely", html)
        # Job counts live in their own strip...
        self.assertIn("active: <strong>1</strong>", html)

    def test_a_failed_source_shows_up_as_its_own_fact(self):
        health = {"sources": [{"source": "myjobmag.co.ke", "state": "unknown",
                               "jobs": 0, "last_seen": ""}]}
        html = render_dashboard_html([build_view(self._stored())], source_health=health)
        self.assertIn("myjobmag.co.ke", html)
        # Source health is not presented as a job-freshness verdict.
        self.assertIn("unknown: <strong>1</strong>", html)

    def test_a_failing_source_is_shown_even_with_no_jobs_behind_it(self):
        """The worst case: a source that has never worked at all."""
        health = {"sources": [], "outcomes": [
            {"source": "myjobmag.co.ke", "outcome": "failing",
             "consecutive_failures": 4, "last_success": None, "jobs_returned": 0,
             "skipped": False, "last_outcome_at": at()},
        ]}
        html = render_dashboard_html([build_view(self._stored())], source_health=health)
        self.assertIn("myjobmag.co.ke", html)
        self.assertIn("failing", html)
        self.assertIn("4 consecutive failures", html)

    def test_the_dashboard_writes_freshness_end_to_end(self):
        run_sources(
            [SourceSpec(name="weworkremotely", fetch=lambda: [record("j1")])],
            store=self.store,
            observed_at=T0,
            record_freshness=True,
        )
        state = FreshnessLedger(self.store).evaluate()
        out = Path(self._tmp.name) / "dashboard.html"
        render_dashboard_file(
            self.store, out,
            generated_at=at(),
            freshness=state,
            source_health=summarise(state),
        )
        html = out.read_text(encoding="utf-8")
        self.assertIn("active", html)
        self.assertIn("weworkremotely", html)

    def test_expired_jobs_remain_on_screen(self):
        """Expiry is a label. Hiding the row would be an unsupportable claim."""
        policy = DEFAULT_POLICIES["weworkremotely"]
        run_sources(
            [SourceSpec(name="weworkremotely", fetch=lambda: [record("j1"), record("j2")])],
            store=self.store,
            observed_at=T0,
            record_freshness=True,
        )
        run_sources(
            [SourceSpec(name="weworkremotely", fetch=lambda: [record("j2")])],
            store=self.store,
            observed_at=T0 + timedelta(days=policy.expire_days + 5),
            record_freshness=True,
        )
        state = FreshnessLedger(self.store).evaluate()
        out = Path(self._tmp.name) / "dashboard.html"
        render_dashboard_file(
            self.store, out, generated_at=at(), freshness=state,
            source_health=summarise(state),
        )
        html = out.read_text(encoding="utf-8")
        self.assertIn("expired", html)
        self.assertIn("Role", html)

    def test_filtering_by_freshness_narrows_the_table(self):
        run_sources(
            [SourceSpec(name="weworkremotely", fetch=lambda: [record("j1"), record("j2")])],
            store=self.store,
            observed_at=T0,
            record_freshness=True,
        )
        state = FreshnessLedger(self.store).evaluate()
        out = Path(self._tmp.name) / "dashboard.html"
        render_dashboard_file(
            self.store, out, generated_at=at(), freshness=state,
            freshness_state="expired",
        )
        html = out.read_text(encoding="utf-8")
        # Nothing is expired yet, so the filter empties the table rather than
        # quietly showing everything.
        self.assertIn("No jobs match the current filters", html)


if __name__ == "__main__":
    unittest.main()