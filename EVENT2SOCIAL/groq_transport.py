"""Groq SDK transport shared by text generation and media analysis."""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

from groq_errors import format_groq_http_error

logger = logging.getLogger(__name__)


class GroqTransportError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 0, headers: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers or {}


def api_root(base_or_endpoint: str | None) -> str:
    """Convert the app's OpenAI-compatible base URL to the SDK's host base URL."""
    base = (base_or_endpoint or "https://api.groq.com/openai/v1").rstrip("/")
    for suffix in ("/chat/completions", "/audio/transcriptions", "/models", "/openai/v1"):
        if base.endswith(suffix):
            base = base[:-len(suffix)].rstrip("/")
            break
    return base


def _retry_delay(headers: Any) -> float:
    try:
        return min(max(float(headers.get("Retry-After", "2")), 0), 8)
    except (AttributeError, TypeError, ValueError):
        return 2.0


def call_groq(api_key: str, base_url: str | None, service: str, timeout: int,
              operation: Callable[[Any], Any]) -> Any:
    """Run one SDK operation; retry only one transient 503, never retry 429."""
    if not api_key:
        raise GroqTransportError("Groq is not configured. Set GROQ_API_KEY in server secrets.")
    try:
        from groq import Groq
    except ImportError as exc:
        raise GroqTransportError(
            "The Groq SDK is not installed in the active environment. Install requirements.txt.") from exc

    client = Groq(api_key=api_key, base_url=api_root(base_url), timeout=timeout,
                  max_retries=0)
    try:
        for attempt in range(2):
            try:
                return operation(client)
            except Exception as exc:
                response = getattr(exc, "response", None)
                if response is None:
                    logger.warning("Groq %s transport error: %s", service, type(exc).__name__)
                    raise GroqTransportError(
                        f"Groq {service} could not complete the HTTP request ({type(exc).__name__}). "
                        "Check DNS, TLS, firewall, and the configured API endpoint.") from None

                status = int(getattr(response, "status_code", 0) or 0)
                headers = getattr(response, "headers", {})
                content_type = str(headers.get("content-type", ""))
                body = getattr(response, "content", b"")
                if not body:
                    body = getattr(response, "text", "")
                message = format_groq_http_error(
                    status, body, content_type=content_type,
                    server=str(headers.get("server", "")), api_key=api_key, service=service)
                # The formatted diagnostic contains only provider error details and is
                # redacted before logging; credentials and request headers are omitted.
                sanitized = re.sub(r"(?i)(gsk_[A-Za-z0-9_-]+)", "[redacted]", message)[:800]
                logger.warning("Groq HTTP failure status=%s content_type=%r sanitized_error=%r",
                               status, content_type or "unknown", sanitized)

                if status == 503 and attempt == 0:
                    time.sleep(_retry_delay(headers))
                    continue
                raise GroqTransportError(message, status_code=status, headers=headers) from None
        raise GroqTransportError(f"Groq {service} failed after one bounded HTTP 503 retry.")
    finally:
        client.close()
