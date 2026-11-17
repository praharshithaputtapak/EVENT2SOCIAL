import logging
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import groq_transport
from groq_transport import GroqTransportError


class _StatusError(Exception):
    def __init__(self, status, body, headers=None):
        self.response = SimpleNamespace(
            status_code=status,
            headers=headers or {"content-type": "application/json", "server": "groq"},
            content=body,
            text=body.decode("utf-8", errors="replace"),
        )


class GroqTransportTests(unittest.TestCase):
    def _fake_sdk(self, behavior):
        instances = []

        class FakeGroq:
            def __init__(self, **kwargs):
                self.options = kwargs
                self.closed = False
                instances.append(self)

            def close(self):
                self.closed = True

        module = ModuleType("groq")
        module.Groq = FakeGroq
        return module, instances

    def test_sdk_uses_host_base_and_disables_automatic_retries(self):
        module, instances = self._fake_sdk(None)
        with patch.dict(sys.modules, {"groq": module}):
            result = groq_transport.call_groq(
                "unit-test-key", "https://api.groq.com/openai/v1", "test", 45,
                lambda client: "ok")
        self.assertEqual(result, "ok")
        self.assertEqual(instances[0].options["base_url"], "https://api.groq.com")
        self.assertEqual(instances[0].options["timeout"], 45)
        self.assertEqual(instances[0].options["max_retries"], 0)
        self.assertTrue(instances[0].closed)

    def test_transient_503_is_retried_once(self):
        module, instances = self._fake_sdk(None)
        attempts = []

        def operation(client):
            attempts.append(1)
            if len(attempts) == 1:
                raise _StatusError(503, b'{"error":{"message":"temporarily busy"}}',
                                   {"content-type": "application/json", "retry-after": "0"})
            return "recovered"

        with patch.dict(sys.modules, {"groq": module}), patch.object(groq_transport.time, "sleep"):
            self.assertEqual(groq_transport.call_groq("unit-key", None, "test", 45, operation), "recovered")
        self.assertEqual(len(attempts), 2)

    def test_429_is_not_retried_and_logs_sanitized_response_diagnostics(self):
        module, _ = self._fake_sdk(None)
        attempts = []

        def operation(client):
            attempts.append(1)
            raise _StatusError(429,
                b'{"error":{"message":"daily limit reached; token=unit-test-key"}}',
                {"content-type": "application/json", "server": "groq"})

        with patch.dict(sys.modules, {"groq": module}), self.assertLogs(groq_transport.logger, logging.WARNING) as logs:
            with self.assertRaises(GroqTransportError) as caught:
                groq_transport.call_groq("unit-test-key", None, "test", 45, operation)
        self.assertEqual(len(attempts), 1)
        self.assertIn("HTTP 429", str(caught.exception))
        output = "\n".join(logs.output)
        self.assertIn("status=429", output)
        self.assertIn("application/json", output)
        self.assertIn("daily limit reached", output)
        self.assertNotIn("unit-test-key", output)

    def test_cloudflare_1010_is_reported_as_non_api_block(self):
        module, _ = self._fake_sdk(None)

        def operation(client):
            raise _StatusError(403, b"error code: 1010", {
                "content-type": "text/plain; charset=UTF-8", "server": "cloudflare"})

        with patch.dict(sys.modules, {"groq": module}):
            with self.assertRaises(GroqTransportError) as caught:
                groq_transport.call_groq("unit-key", None, "text test", 45, operation)
        self.assertIn("Cloudflare error 1010", str(caught.exception))
        self.assertIn("does not establish whether the API key", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
