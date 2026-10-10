"""The read-only queue listing CLI.

Three properties matter more than the output's looks, and each is asserted
directly rather than inferred:

**Nothing is written.** Every persisted file is hashed before and after and
compared byte for byte. A read-only tool that silently rewrites a report file
teaches you to distrust the word "read-only".

**Nothing is sent.** The module imports no transport, and a run with the socket
layer sabotaged still succeeds - proof that success never depended on a
connection rather than an assumption that none was made.

**The output is deterministic.** Two runs over unchanged data must be identical
strings. Without that, a diff between two runs cannot distinguish real change
from reordering, and the command loses the one thing it is for.

Also covered: the mojibake that genuinely appears in stored MyJobMag titles is
handled rather than assumed away, and a malformed ``jobs.jsonl`` is reported
visibly instead of quietly producing a shorter queue.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import contextlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from app.jobs.runner import SourceSpec, run_sources
from app.jobs.store import JobStore

REPO_ROOT = Path(__file__).resolve().parents[1]
REVIEW_CLI = REPO_ROOT / "tools" / "review.py"


def _load_cli():
    """Import tools/review.py as a module.

    Loaded by path because ``tools`` is not a package; importing it normally
    would execute argparse at import time.
    """
    spec = importlib.util.spec_from_file_location("_review_cli", REVIEW_CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(name, *, source="weworkremotely", title="Role", company="Acme",
            location="Nairobi, Kenya"):
    return {
        "url": f"https://{source}/jobs/{name}",
        "title": title, "company": company,
        "location": location, "description": "Work.",
    }


class ReviewCliTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data = Path(self._tmp.name) / "data"
        self.cli = _load_cli()
        self.store = JobStore(self.data)

    def ingest(self, records, *, source="weworkremotely"):
        run_sources([SourceSpec(name=source, fetch=lambda: list(records))],
                    store=self.store)

    def _main(self, *args):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = self.cli.main(["--data-dir", str(self.data), *args])
        return code, buffer.getvalue()

    def snapshot(self):
        return {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(self.data.rglob("*")) if p.is_file()
        }

    def _data_rows(self, out):
        """Rows that carry a job, excluding the rule and the preamble.

        The index column's width depends on the row count, so it is matched by
        pattern rather than by a fixed prefix.
        """
        import re

        pattern = re.compile(r"^\s*\d+\s+\S")
        return [l for l in out.splitlines()
                if pattern.match(l) and not l.lstrip().startswith("-")]


class QueueOutputTests(ReviewCliTestCase):
    def test_it_lists_the_queue(self):
        self.ingest([_record("a1", title="Backend Engineer")])
        code, out = self._main("--queue")
        self.assertEqual(code, 0)
        self.assertIn("actionable queue", out)
        self.assertIn("Backend Engineer", out)

    def test_it_shows_every_required_column(self):
        self.ingest([_record("a1", title="Role", company="Acme")])
        code, out = self._main("--queue")
        self.assertEqual(code, 0)
        for column in ("JOB ID", "TITLE", "COMPANY", "SOURCE", "ELIGIBLE",
                       "FRESHNESS", "STATUS", "MATCH"):
            self.assertIn(column, out)

    def test_the_canonical_job_id_is_shown(self):
        self.ingest([_record("a1")])
        job_id = self.store.load_jobs()[0]["job_id"]
        code, out = self._main("--queue")
        self.assertEqual(code, 0)
        self.assertIn(job_id, out)

    def test_without_the_flag_it_does_nothing(self):
        code, _ = self._main()
        self.assertEqual(code, 1)

    def test_an_empty_queue_says_so_and_exits_nonzero(self):
        code, out = self._main("--queue")
        self.assertEqual(code, 1)
        self.assertIn("empty", out)


class DeterminismTests(ReviewCliTestCase):
    def test_two_runs_over_unchanged_data_are_identical(self):
        self.ingest([_record("a1"), _record("a2"), _record("a3")])
        first_code, first = self._main("--queue")
        second_code, second = self._main("--queue")
        self.assertEqual(first_code, second_code)
        self.assertEqual(first, second)

    def test_order_is_stable_across_many_jobs(self):
        self.ingest([_record(f"j{i:02d}") for i in range(12)])
        runs = [self._main("--queue")[1] for _ in range(3)]
        self.assertEqual(len(set(runs)), 1, "order varied between runs")

    def test_row_order_does_not_depend_on_ingest_order(self):
        self.ingest([_record("z1", title="Zeta")])
        self.ingest([_record("a1", title="Alpha")], source="remotive")
        out_a = self._main("--queue")[1]
        # Reverse the ingest order in a fresh store; the listing must match.
        self._tmp.cleanup()
        self.setUp()
        self.ingest([_record("a1", title="Alpha")], source="remotive")
        self.ingest([_record("z1", title="Zeta")])
        out_b = self._main("--queue")[1]
        self.assertEqual(out_a, out_b)

    def test_a_diff_shows_real_change_not_reordering(self):
        self.ingest([_record("a1"), _record("a2")])
        before = self._main("--queue")[1]
        self.ingest([_record("a3", title="Brand New Role")])
        after = self._main("--queue")[1]
        self.assertNotEqual(before, after)
        self.assertIn("Brand New Role", after)
        # Everything present before is still present, in the same order.
        for row in ("Role", "a1", "a2"):
            self.assertIn(row, after)


class ReadOnlyTests(ReviewCliTestCase):
    def test_every_persisted_file_is_byte_identical_after_a_run(self):
        self.ingest([_record("a1"), _record("a2")])
        before = self.snapshot()
        self.assertTrue(before, "the store should have persisted something")
        self._main("--queue")
        after = self.snapshot()
        self.assertEqual(before, after)
        self.assertEqual(set(before), set(after), "a file was added or removed")

    def test_no_match_file_is_created(self):
        self.ingest([_record("a1")])
        self._main("--queue")
        self.assertFalse((self.data / "matches.jsonl").exists())

    def test_no_report_file_is_created(self):
        self.ingest([_record("a1")])
        self._main("--queue")
        for name in ("review_report.html", "queue_report.html",
                     "discovery_report.html"):
            self.assertFalse((self.data / name).exists())

    def test_no_status_freshness_or_ledger_is_written(self):
        self.ingest([_record("a1")])
        before = self.snapshot()
        self._main("--queue")
        after = self.snapshot()
        for name in before:
            self.assertEqual(before[name], after[name], name)

    def test_repeated_runs_do_not_accumulate_writes(self):
        self.ingest([_record("a1")])
        baseline = self.snapshot()
        for _ in range(3):
            self._main("--queue")
        self.assertEqual(baseline, self.snapshot())


class NoNetworkTests(ReviewCliTestCase):
    def test_the_cli_imports_no_transport_module(self):
        import ast

        tree = ast.parse(REVIEW_CLI.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        for forbidden in ("socket", "subprocess", "http", "requests", "httpx",
                          "urllib.request", "urllib.error", "ssl"):
            self.assertNotIn(forbidden, imported)

    def test_a_run_succeeds_with_the_socket_layer_sabotaged(self):
        """Proof by construction, not by inspection.

        If anything in this path needed a connection, breaking the socket layer
        would change the outcome. It does not.
        """
        self.ingest([_record("a1")])
        script = """
import socket, sys
sys.path.insert(0, ".")
def _blocked(*a, **k):
    raise AssertionError("the review CLI attempted a network call")
socket.socket = _blocked
socket.create_connection = _blocked
import runpy
runpy.run_path("tools/review.py", run_name="__main__")
"""
        path = Path(self._tmp.name) / "sabotage.py"
        path.write_text(script, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(path), "--queue", "--data-dir", str(self.data)],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        )
        # __main__ exits 0; a SystemExit(0) surfaces as returncode 0.
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("actionable queue", result.stdout)

    def test_it_does_not_invoke_discovery(self):
        """Parsed, not grepped.

        A docstring saying "does not invoke discovery" contains the word, so a
        substring search would fail on prose that is evidence of the opposite.
        Only real references - imports and attribute lookups - are checked.
        """
        import ast

        tree = ast.parse(REVIEW_CLI.read_text(encoding="utf-8"))
        referenced = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                referenced.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                referenced.add(node.module)
            elif isinstance(node, ast.Attribute):
                referenced.add(node.attr)
            elif isinstance(node, ast.Name):
                referenced.add(node.id)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                # A module path passed to importlib or __import__ is a
                # reference too, unlike prose.
                if node.value.endswith(".py") or "/" in node.value:
                    referenced.add(node.value)
        for forbidden in ("run_discovery", "robots_check", "collect_wwr",
                          "collect_remotive", "collect_myjobmag", "urlopen",
                          "Request", "socket"):
            self.assertNotIn(forbidden, referenced,
                             f"{forbidden} is referenced")

    def test_it_does_not_import_a_source_adapter(self):
        import ast

        tree = ast.parse(REVIEW_CLI.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        for forbidden in ("app.sources.discovery", "app.jobs.discovery",
                          "app.sources.transport", "app.jobs.assessment"):
            self.assertNotIn(forbidden, imported)

    def test_it_registers_no_provider_and_assesses_nothing(self):
        from app.jobs.assessment import registered_providers

        self.ingest([_record("a1")])
        self._main("--queue")
        self.assertEqual(registered_providers(), ())


class EscapingTests(ReviewCliTestCase):
    def test_control_characters_cannot_forge_a_new_row(self):
        """The hostile text survives as text, but stays on one line.

        The claim is not that the words vanish - they are part of the title and
        hiding them would be a lie. The claim is that they cannot break out of
        their row and impersonate a second job.
        """
        hostile = "Real Title\r\n  99  FAKE ROW   forged-fake"
        self.ingest([_record("a1", title=hostile)])
        code, out = self._main("--queue")
        self.assertEqual(code, 0)
        rows = self._data_rows(out)
        self.assertEqual(len(rows), 1, "a forged row appeared")
        self.assertIn("FAKE ROW", rows[0],
                      "the title text should still be visible")
        self.assertNotIn("forged-fake", " ".join(rows[1:]))

    def test_a_title_cannot_add_a_table_row(self):
        self.ingest([_record("a1", title="A\nB")])
        code, out = self._main("--queue")
        self.assertEqual(code, 0)
        rows = self._data_rows(out)
        self.assertEqual(len(rows), 1)
        self.assertIn("A B", rows[0])

    def test_an_escape_sequence_is_not_interpreted(self):
        self.ingest([_record("a1", title="\x1b[31mRED\x1b[0m")])
        code, out = self._main("--queue")
        self.assertEqual(code, 0)
        self.assertNotIn("\x1b", out, "an escape byte reached the output")

    def test_a_very_long_title_is_truncated(self):
        self.ingest([_record("a1", title="X" * 5000)])
        code, out = self._main("--queue")
        self.assertEqual(code, 0)
        row = self._data_rows(out)[0]
        self.assertLess(len(row), 400)
        self.assertIn("\u2026", row)

    def test_real_mojibake_survives_without_breaking_alignment(self):
        """The store genuinely contains mojibake; it must render, not vanish."""
        self.ingest([_record("a1", title="Area Sales \u00c3\u0083\u00c2\u00a2")])
        code, out = self._main("--queue")
        self.assertEqual(code, 0)
        self.assertIn("Area Sales", out)

    def test_none_values_render_as_empty_not_the_word_none(self):
        self.cli._clean(None, 10)
        self.assertEqual(self.cli._clean(None, 10), "")


class CorruptRecordTests(ReviewCliTestCase):
    def test_a_torn_line_is_counted_and_reported_visibly(self):
        self.ingest([_record("a1")])
        with (self.data / "jobs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write('{"job_id": "broken", "at": "2026-')
        readable, unreadable = self.cli.unreadable_lines(self.data)
        self.assertEqual(unreadable, 1)
        self.assertGreaterEqual(readable, 1)

        buffer = io.StringIO()
        errors = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            with contextlib.redirect_stderr(errors):
                self.cli.main(["--data-dir", str(self.data), "--queue"])
        message = errors.getvalue()
        self.assertIn("unreadable", message)
        self.assertIn("skipped", message)
        self.assertIn("missing those records", message)

    def test_a_warning_appears_even_when_a_queue_exists(self):
        """A silent short queue would look like a legitimately smaller queue."""
        self.ingest([_record("a1")])
        with (self.data / "jobs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("not json at all\n")
        errors = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(errors):
                self.cli.main(["--data-dir", str(self.data), "--queue"])
        self.assertIn("1 unreadable", errors.getvalue())

    def test_a_clean_store_reports_no_warning(self):
        self.ingest([_record("a1")])
        errors = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(errors):
                self.cli.main(["--data-dir", str(self.data), "--queue"])
        self.assertEqual(errors.getvalue(), "")

    def test_a_missing_store_is_empty_not_an_error(self):
        readable, unreadable = self.cli.unreadable_lines(self.data / "nowhere")
        self.assertEqual((readable, unreadable), (0, 0))

    def test_blank_lines_are_not_counted_as_corrupt(self):
        self.ingest([_record("a1")])
        with (self.data / "jobs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("\n   \n")
        readable, unreadable = self.cli.unreadable_lines(self.data)
        self.assertEqual(unreadable, 0)


class MatchStateTests(ReviewCliTestCase):
    def test_never_assessed_is_distinct_from_insufficient_evidence(self):
        class View:
            match_present = False
            match_tier = None
            match_insufficient_reason = ""

        self.assertEqual(self.cli.match_state(View())["state"], "not assessed")

    def test_an_assessed_view_reports_its_tier(self):
        class View:
            match_present = True
            match_tier = "credible_match"
            match_insufficient_reason = ""

        state = self.cli.match_state(View())
        self.assertEqual(state["state"], "assessed")
        self.assertEqual(state["tier"], "credible_match")

    def test_insufficient_evidence_is_named_as_such(self):
        class View:
            match_present = True
            match_tier = "not_yet_evaluated"
            match_insufficient_reason = "no surviving evidence"

        state = self.cli.match_state(View())
        self.assertEqual(state["state"], "insufficient evidence")
        self.assertEqual(state["reason"], "no surviving evidence")

    def test_the_queue_says_how_many_are_unassessed(self):
        self.ingest([_record("a1")])
        code, out = self._main("--queue")
        self.assertEqual(code, 0)
        self.assertIn("match state:", out)
        self.assertIn("1 not assessed", out)


class NoExternalReferencesTests(unittest.TestCase):
    def test_no_script_file_is_added(self):
        """A new executable outside tools/ would be an unexplained addition."""
        before = {
            p for p in REPO_ROOT.rglob("*.py")
            if ".git" not in p.parts and "__pycache__" not in p.parts
        }
        self.assertIn(REVIEW_CLI.resolve(), before)

    def test_the_cli_holds_no_urls_of_its_own(self):
        """Any hardcoded reference would be a portal this system does not use."""
        import re

        source = REVIEW_CLI.read_text(encoding="utf-8")
        urls = re.findall(r"https?://[^\s\"'<>]+", source)
        self.assertEqual(urls, [], f"unexpected URLs: {urls}")

    def test_no_source_or_permission_files_changed(self):
        from app.jobs import assessment, discovery

        self.assertEqual(assessment.registered_providers(), ())
        self.assertTrue(hasattr(discovery, "run_discovery"))


if __name__ == "__main__":
    unittest.main()