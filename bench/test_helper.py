#!/usr/bin/env python3
"""Offline unit tests for the helper-model plumbing (no network).   python3 bench/test_helper.py"""
import sys
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lowctx  # noqa: E402

BODY = {"model": "m", "messages": [], "max_tokens": 10}


class HelperBody(unittest.TestCase):
    def test_deepseek_disables_thinking(self):
        out = lowctx.helper_body(BODY, "https://api.deepseek.com/v1", {})
        self.assertEqual(out["thinking"], {"type": "disabled"})
        self.assertNotIn("enable_thinking", out)

    def test_localhost_sends_enable_thinking(self):
        out = lowctx.helper_body(BODY, "http://127.0.0.1:8081/v1", {})
        self.assertIs(out["enable_thinking"], False)
        self.assertNotIn("thinking", out)

    def test_openai_gets_no_extras(self):
        self.assertEqual(lowctx.helper_body(BODY, "https://api.openai.com/v1", {}), BODY)

    def test_extra_body_merged_and_input_untouched(self):
        out = lowctx.helper_body(BODY, "https://openrouter.ai/api/v1", {"reasoning": {"effort": "low"}})
        self.assertEqual(out["reasoning"], {"effort": "low"})
        self.assertEqual(out["model"], "m")
        self.assertNotIn("reasoning", BODY)

    def test_lookalike_host_is_not_deepseek(self):
        self.assertNotIn("thinking", lowctx.helper_body(BODY, "https://notdeepseek.com.evil.io/v1", {}))


class HelperPost(unittest.TestCase):
    def test_retries_once_without_extras_on_unknown_field(self):
        sent = []

        def fake(url, key, body, timeout=0):
            sent.append(body)
            if len(sent) == 1:
                raise lowctx.HTTPFail(400, 'HTTP 400: {"error":"Unrecognized request argument supplied: thinking"}')
            return {"ok": True}

        with mock.patch.object(lowctx, "post_chat", fake), mock.patch.object(lowctx, "COMP_URL", "https://api.deepseek.com/v1"):
            self.assertEqual(lowctx.helper_post(dict(BODY)), {"ok": True})
        self.assertIn("thinking", sent[0])
        self.assertEqual(sent[1], BODY)

    def test_other_400_is_not_retried(self):
        fake = mock.Mock(side_effect=lowctx.HTTPFail(400, "HTTP 400: max_tokens too large"))
        with mock.patch.object(lowctx, "post_chat", fake), mock.patch.object(lowctx, "COMP_URL", "https://api.deepseek.com/v1"):
            self.assertRaises(lowctx.HTTPFail, lowctx.helper_post, dict(BODY))
        self.assertEqual(fake.call_count, 1)


class HelperOn(unittest.TestCase):
    def test_key_only(self):
        self.assertTrue(lowctx.helper_on("sk-x", ""))

    def test_url_only(self):  # keyless local server
        self.assertTrue(lowctx.helper_on("", "http://127.0.0.1:8081/v1"))

    def test_neither(self):
        self.assertFalse(lowctx.helper_on("", ""))


class Moderation(unittest.TestCase):
    def test_deepseek_body(self):
        txt = ('HTTP 400 from https://api.deepseek.com/v1: {"error":{"message":"Content Exists Risk",'
               '"type":"invalid_request_error"}}')
        self.assertTrue(lowctx.MODERATION.search(txt))

    def test_openai_content_policy(self):
        txt = ('HTTP 400 from https://api.openai.com/v1: {"error":{"message":"Your request was rejected as a '
               'result of our safety system.","code":"content_policy_violation"}}')
        self.assertTrue(lowctx.MODERATION.search(txt))

    def test_plain_error_does_not_match(self):
        self.assertFalse(lowctx.MODERATION.search('HTTP 400: {"error":{"message":"max_tokens is too large"}}'))


if __name__ == "__main__":
    unittest.main()
