import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import media_processing as media
import storage


class _Response:
    def __init__(self, body):
        self.body = json.dumps(body).encode("utf-8")
    def __enter__(self): return self
    def __exit__(self, *_): return False
    def read(self): return self.body


class GeminiQuotaHandlingTests(unittest.TestCase):
    def test_interaction_429_preserves_safe_daily_quota_detail_and_does_not_retry(self):
        calls = []
        def respond(request, timeout):
            calls.append(1)
            body = {"error": {"message": "Rate limit exceeded (limit: 20 requests per day; api_key=test-secret-key). retry in 2h"}}
            raise HTTPError(request.full_url, 429, "quota", {"Retry-After": "3600"}, BytesIO(json.dumps(body).encode()))
        with patch.object(media.urllib.request, "urlopen", respond):
            with self.assertRaises(media.GeminiQuotaError) as caught:
                media._interaction("test-secret-key", "model-x", [{"type": "text", "text": "hello"}], "https://example.test")
        self.assertEqual(len(calls), 1)
        self.assertTrue(caught.exception.daily_quota)
        self.assertIn("20 requests per day", str(caught.exception))
        self.assertNotIn("test-secret-key", str(caught.exception))
        self.assertTrue(caught.exception.retry_after)

    def test_media_interaction_retries_503_once_then_succeeds(self):
        calls = []
        def respond(request, timeout):
            calls.append(1)
            if len(calls) == 1:
                raise HTTPError(request.full_url, 503, "busy", {"Retry-After": "0"}, BytesIO(b"{}"))
            return _Response({"output_text": "analyzed"})
        with patch.object(media.urllib.request, "urlopen", respond), patch.object(media.time, "sleep"):
            self.assertEqual(media._interaction("key", "model-x", [{"type": "text", "text": "hello"}], "https://example.test"), "analyzed")
        self.assertEqual(len(calls), 2)

    def test_media_interaction_stops_after_two_503_responses(self):
        calls = []
        def respond(request, timeout):
            calls.append(1)
            raise HTTPError(request.full_url, 503, "busy", {"Retry-After": "0"}, BytesIO(b"{}"))
        with patch.object(media.urllib.request, "urlopen", respond), patch.object(media.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
                media._interaction("key", "model-x", [{"type": "text", "text": "hello"}], "https://example.test")
        self.assertEqual(len(calls), 2)

    def test_429_stops_remaining_batch_and_blocks_repeated_requests(self):
        calls = []
        def worker(item):
            calls.append(item)
            if item == "second":
                raise media.GeminiQuotaError(
                    "Gemini media analysis daily quota reached (HTTP 429). Provider message: 20 requests per day",
                    retry_after="", daily_quota=True)
            return f"result-{item}"
        results = media.run_media_batch(["first", "second", "third"], worker)
        self.assertEqual(calls, ["first", "second"])
        self.assertEqual([item.status for item in results], ["complete", "deferred", "deferred"])
        docs = [{"filename": name, "sha256": name} for name in ("first", "second", "third")]
        cache = {
            "first": {"sha256": "first", "status": "complete", "text": "cached"},
            "second": {"sha256": "second", "status": "deferred", "error": results[1].error},
            "third": {"sha256": "third", "status": "deferred", "error": results[2].error},
        }
        work, deferred, cached = media.filter_media_batch_items(docs, cache)
        self.assertEqual(work, [])
        self.assertEqual(deferred, ["second", "third"])
        self.assertEqual(cached, ["first"])
        forced_work, forced_deferred, _ = media.filter_media_batch_items(docs, cache, force_retry=True)
        self.assertEqual([d["filename"] for d in forced_work], ["second", "third"])
        self.assertEqual(forced_deferred, [])
        self.assertTrue(media.media_retry_is_blocked("failed", "HTTP 429: 20 requests per day"))

    def test_retry_timestamp_allows_request_only_after_due_time(self):
        retry_at = "2026-11-13T10:00:00+00:00"
        before = datetime(2026, 11, 13, 9, 59, tzinfo=timezone.utc)
        after = datetime(2026, 11, 13, 10, 1, tzinfo=timezone.utc)
        self.assertTrue(media.media_retry_is_blocked("deferred", "quota", retry_at, now=before))
        self.assertFalse(media.media_retry_is_blocked("deferred", "quota", retry_at, now=after))

    def test_retry_after_schema_round_trip_in_temporary_database(self):
        with tempfile.TemporaryDirectory(prefix="event2social-test-") as temp:
            root = Path(temp)
            db_path = root / "test.db"
            old_db = sqlite3.connect(db_path)
            try:
                old_db.execute("""CREATE TABLE media_analysis(
                    owner_id TEXT NOT NULL, filename TEXT NOT NULL, sha256 TEXT NOT NULL,
                    status TEXT NOT NULL, text TEXT NOT NULL DEFAULT '',
                    observations TEXT NOT NULL DEFAULT '[]', timestamps TEXT NOT NULL DEFAULT '[]',
                    error TEXT NOT NULL DEFAULT '', model TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(owner_id, filename))""")
                old_db.commit()
            finally:
                old_db.close()
            with patch.object(storage, "DATA_DIR", root), patch.object(storage, "DB_PATH", db_path):
                storage.save_media_analysis("sample.jpg", "hash", "deferred", error="HTTP 429 quota",
                                            model="model-x", retry_after="2026-11-13T10:00:00+00:00")
                record = storage.load_media_analysis()["sample.jpg"]
                self.assertEqual(record["status"], "deferred")
                self.assertEqual(record["retry_after"], "2026-11-13T10:00:00+00:00")
                check_db = sqlite3.connect(db_path)
                try:
                    columns = {row[1] for row in check_db.execute("PRAGMA table_info(media_analysis)")}
                finally:
                    check_db.close()
                self.assertIn("retry_after", columns)


if __name__ == "__main__":
    unittest.main()
