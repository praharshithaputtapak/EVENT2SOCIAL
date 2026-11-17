"""Safe, actionable formatting for errors returned by Groq or its HTTP edge."""
from __future__ import annotations

import json
import re
from typing import Any


def _redact(value: Any, api_key: str = "", limit: int = 600) -> str:
    text = str(value or "")
    if api_key:
        text = text.replace(api_key, "[redacted]")
    text = re.sub(r"(?i)(api[_ -]?key|token|authorization)\s*[:=]\s*[^\s,;]+",
                  r"\1=[redacted]", text)
    return text.strip()[:limit]


def format_groq_http_error(status: int, body: bytes | str, *,
                           content_type: str = "", server: str = "",
                           api_key: str = "", service: str = "request") -> str:
    """Preserve Groq's JSON error fields, or identify non-JSON edge errors."""
    raw = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else str(body or "")
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        payload = None

    if isinstance(payload, dict):
        error = payload.get("error", payload)
        if not isinstance(error, dict):
            error = {"message": error}
        message = _redact(error.get("message", ""), api_key)
        error_type = _redact(error.get("type", ""), api_key, 120)
        code = _redact(error.get("code", ""), api_key, 120)
        fields = [f"HTTP {status}"]
        if error_type:
            fields.append(f"type={error_type}")
        if code:
            fields.append(f"code={code}")
        detail = f" Groq message: {message}" if message else ""
        if status == 401:
            advice = " Verify that the configured GROQ_API_KEY is valid and active."
        elif status == 403:
            advice = " Groq authenticated the request but denied access; check model or organization/project permissions."
        elif status == 404:
            advice = " Check the API endpoint and model ID."
        elif status == 429:
            advice = " Check the account's rate and daily quota limits; do not retry an exhausted daily quota."
        else:
            advice = " Check the request details and Groq service status."
        return f"Groq {service} failed ({', '.join(fields)}).{detail}{advice}"

    detail = _redact(raw, api_key)
    if "1010" in detail and ("cloudflare" in server.lower() or "error code: 1010" in detail.lower()):
        return (f"Groq {service} was blocked before reaching Groq's JSON API (HTTP {status}, "
                f"Cloudflare error 1010; response content-type: {content_type or 'unknown'}). "
                "Cloudflare documents 1010 as access denied based on the client's browser signature. "
                "This does not establish whether the API key or model permissions are valid; "
                "try an approved network/client path or contact Groq support with the timestamp and response details.")
    if detail:
        return (f"Groq {service} failed (HTTP {status}; non-JSON response; "
                f"content-type: {content_type or 'unknown'}; server: {server or 'unknown'}). "
                f"Response: {detail}")
    return (f"Groq {service} failed (HTTP {status}; empty or non-JSON response; "
            f"content-type: {content_type or 'unknown'}; server: {server or 'unknown'}).")
