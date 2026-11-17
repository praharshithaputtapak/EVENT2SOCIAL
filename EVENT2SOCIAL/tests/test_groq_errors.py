import unittest

from groq_errors import format_groq_http_error


class GroqErrorFormattingTests(unittest.TestCase):
    def test_json_permission_error_preserves_message_type_and_code(self):
        result = format_groq_http_error(
            403,
            b'{"error":{"message":"Model blocked","type":"permissions_error","code":"model_permission_blocked_project"}}',
            api_key="test-secret")
        self.assertIn("HTTP 403", result)
        self.assertIn("permissions_error", result)
        self.assertIn("model_permission_blocked_project", result)
        self.assertIn("Model blocked", result)
        self.assertIn("project permissions", result)
        self.assertNotIn("test-secret", result)

    def test_401_is_identified_as_authentication_failure(self):
        result = format_groq_http_error(401, b'{"error":{"message":"invalid key","type":"authentication_error"}}')
        self.assertIn("HTTP 401", result)
        self.assertIn("Verify that the configured GROQ_API_KEY", result)

    def test_cloudflare_1010_non_json_response_is_distinguished(self):
        result = format_groq_http_error(403, b"error code: 1010", content_type="text/plain", server="cloudflare")
        self.assertIn("Cloudflare error 1010", result)
        self.assertIn("does not establish whether the API key", result)


if __name__ == "__main__":
    unittest.main()
