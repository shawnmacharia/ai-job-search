"""The User-Agent is a promise this project makes to other people's servers.

It is stated to Remotive in writing: "each request sends an identifying
User-Agent in the form: [...]". That claim is only true if the string in
:mod:`app.sources.transport` matches the one in the email, and these tests
are what keep the two honest.

Two properties matter as much as the value itself:

- the contact is operator-supplied and must never be able to inject a header
  terminator into every outgoing request;
- a missing or malformed contact file must degrade to a bare, still-valid
  product token rather than stopping the tool.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.sources import transport
from app.sources.transport import USER_AGENT, USER_AGENT_PRODUCT


class UserAgentCompositionTests(unittest.TestCase):
    """The effective User-Agent is composed, not hardcoded."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._original = transport.CONTACT_PATH

    def _compose(self, body):
        """Compose a User-Agent against a temporary contact file."""
        path = Path(self._tmp.name) / "contact.json"
        if body is not None:
            path.write_text(body, encoding="utf-8")
        transport.CONTACT_PATH = path
        try:
            comment = transport._operator_comment()
        finally:
            transport.CONTACT_PATH = self._original
        return transport.USER_AGENT_PRODUCT + (f" ({comment})" if comment else "")

    def test_the_product_token_carries_no_personal_data(self):
        """It is tracked in source and the repository is public."""
        self.assertNotIn("@", USER_AGENT_PRODUCT)
        self.assertNotIn("http", USER_AGENT_PRODUCT)
        self.assertIn("access-check", USER_AGENT_PRODUCT)

    def test_the_declared_product_token_is_the_one_used(self):
        self.assertTrue(USER_AGENT.startswith(USER_AGENT_PRODUCT))

    def test_no_contact_file_yields_the_bare_product_token(self):
        self.assertEqual(self._compose(None), USER_AGENT_PRODUCT)

    def test_an_empty_object_yields_the_bare_product_token(self):
        self.assertEqual(self._compose("{}"), USER_AGENT_PRODUCT)

    def test_purpose_and_contact_are_combined_in_order(self):
        composed = self._compose(json.dumps({
            "purpose": "personal non-commercial research",
            "contact": "someone@example.test",
        }))
        self.assertEqual(
            composed,
            "ai-job-search-access-check/1.0 "
            "(personal non-commercial research; contact: someone@example.test)",
        )

    def test_purpose_alone_is_still_a_valid_comment(self):
        composed = self._compose(json.dumps({"purpose": "personal research"}))
        self.assertEqual(
            composed, "ai-job-search-access-check/1.0 (personal research)"
        )

    def test_contact_alone_is_still_a_valid_comment(self):
        composed = self._compose(json.dumps({"contact": "someone@example.test"}))
        self.assertEqual(
            composed, "ai-job-search-access-check/1.0 (contact: someone@example.test)"
        )

    def test_the_value_is_a_single_header_line(self):
        """A newline here would inject headers into every request."""
        composed = self._compose(json.dumps({
            "purpose": "research", "contact": "someone@example.test",
        }))
        self.assertNotIn("\r", composed)
        self.assertNotIn("\n", composed)


class MalformedContactFileTests(unittest.TestCase):
    """A convenience file must never be able to stop the tool running."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._original = transport.CONTACT_PATH

    def _comment(self, body):
        path = Path(self._tmp.name) / "contact.json"
        path.write_text(body, encoding="utf-8")
        transport.CONTACT_PATH = path
        try:
            return transport._operator_comment()
        finally:
            transport.CONTACT_PATH = self._original

    def test_malformed_json_degrades_to_nothing(self):
        self.assertEqual(self._comment("{not json"), "")

    def test_a_json_array_is_not_an_object(self):
        self.assertEqual(self._comment(json.dumps(["a", "b"])), "")

    def test_wrong_value_types_are_dropped(self):
        self.assertEqual(self._comment(json.dumps({"purpose": 123, "contact": None})), "")

    def test_control_characters_are_stripped(self):
        """Header injection must be neutralised even from a local file."""
        cleaned = transport._clean_field("a\r\nX-Evil: yes")
        self.assertNotIn("\r", cleaned)
        self.assertNotIn("\n", cleaned)

    def test_an_injection_attempt_cannot_forge_a_second_header(self):
        path = Path(self._tmp.name) / "contact.json"
        path.write_text(
            json.dumps({"purpose": "x\r\nX-Evil: yes", "contact": "a@b.test"}),
            encoding="utf-8",
        )
        transport.CONTACT_PATH = path
        try:
            comment = transport._operator_comment()
        finally:
            transport.CONTACT_PATH = self._original
        self.assertNotIn("\r", comment)
        self.assertNotIn("\n", comment)

    def test_an_absurdly_long_field_is_truncated(self):
        cleaned = transport._clean_field("z" * 5000)
        self.assertLessEqual(len(cleaned), transport._MAX_FIELD)

    def test_a_null_byte_is_stripped(self):
        self.assertNotIn("\x00", transport._clean_field("a\x00b"))


class SingleCallSiteTests(unittest.TestCase):
    """The UA must not be varied per request."""

    def test_the_single_network_call_sends_the_constant(self):
        source = Path(transport.__file__).read_text(encoding="utf-8")
        self.assertIn('"User-Agent": USER_AGENT', source)

    def test_no_other_module_sends_a_hardcoded_user_agent(self):
        """A second UA would mean a second identity to the servers we contact.

        Scans for literal User-Agent header construction anywhere except the
        single sanctioned call site. Browser-spoofing strings count too: an
        agent that claims to be Chrome while describing itself as a job
        checker is exactly the kind of misrepresentation this project avoids.
        """
        sanctioned = Path(transport.__file__).resolve()
        offenders = []
        for path in sanctioned.parents[1].rglob("*.py"):
            if path.resolve() == sanctioned:
                continue
            text = path.read_text(encoding="utf-8")
            if '"User-Agent"' in text or "'User-Agent'" in text:
                offenders.append(path.name)
        self.assertEqual(
            offenders, [], f"unexpected User-Agent headers outside transport: {offenders}"
        )


if __name__ == "__main__":
    unittest.main()