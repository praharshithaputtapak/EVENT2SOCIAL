import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import generation
import media_processing as media
import rag
from groq_transport import GroqTransportError


def fake_chat(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def fake_client(chat_result=None, audio_result=None):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: chat_result)),
        audio=SimpleNamespace(transcriptions=SimpleNamespace(create=lambda **kwargs: audio_result)),
    )


class GroqProviderTests(unittest.TestCase):
    def test_selected_provider_reads_groq_from_secrets(self):
        self.assertEqual(rag.configured_provider({"EVENT2SOCIAL_PROVIDER": "groq"}), "groq")

    def test_generation_config_reads_groq_secret_and_selects_models(self):
        key, base, model = generation.generation_config(
            {"GROQ_API_KEY": "test-groq-key", "GROQ_MODEL": "openai/gpt-oss-20b"},
            provider="groq")
        self.assertEqual(key, "test-groq-key")
        self.assertEqual(base, "https://api.groq.com/openai/v1")
        self.assertEqual(model, "openai/gpt-oss-20b")

    def test_groq_generation_uses_sdk_chat_and_returns_content(self):
        client = fake_client(chat_result=fake_chat("A grounded event caption."))
        with patch.object(generation, "call_groq", side_effect=lambda *args: args[-1](client)) as call:
            result = generation.generate(
                {"platform": "Instagram", "event": {"name": "Community Day"}}, [],
                api_key="test-groq-key", base_url="https://api.groq.com/openai/v1",
                model="openai/gpt-oss-120b", provider="groq")
        self.assertEqual(result, "A grounded event caption.")
        self.assertEqual(call.call_count, 1)
        self.assertEqual(call.call_args.args[:4],
                         ("test-groq-key", "https://api.groq.com/openai/v1", "text generation", 90))

    def test_groq_generation_surfaces_auth_error_without_leaking_key(self):
        key = "test-groq-secret"
        error = GroqTransportError("Groq text generation failed (HTTP 401). Verify key.", status_code=401)
        with patch.object(generation, "call_groq", side_effect=error):
            with self.assertRaises(generation.ProviderError) as caught:
                generation.generate({}, [], api_key=key,
                    base_url="https://api.groq.com/openai/v1",
                    model="openai/gpt-oss-120b", provider="groq")
        self.assertIn("HTTP 401", str(caught.exception))
        self.assertNotIn(key, str(caught.exception))

    def test_groq_generation_does_not_retry_429(self):
        with patch.object(generation, "call_groq", side_effect=GroqTransportError(
                "Groq text generation failed (HTTP 429)", status_code=429)) as call:
            with self.assertRaises(generation.ProviderError):
                generation.generate({}, [], api_key="test-key",
                    base_url="https://api.groq.com/openai/v1",
                    model="openai/gpt-oss-120b", provider="groq")
        self.assertEqual(call.call_count, 1)

    def test_carousel_overlay_generation_requests_short_slide_text_only(self):
        client = fake_client(chat_result=fake_chat("Build together"))
        captured = {}

        def execute(key, base, service, timeout, operation):
            def capture(**kwargs):
                captured.update(kwargs)
                return client.chat.completions.create(**kwargs)
            fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=capture)))
            return operation(fake)

        with patch.object(generation, "call_groq", side_effect=execute):
            result = generation.generate(
                {"output_mode": "slide_overlay", "slide_position": 2,
                 "platform": "Instagram", "event": {"name": "Community Day"},
                 "campaign_focus": "Meet the community"}, [], api_key="test-key",
                base_url="https://api.groq.com/openai/v1", model="openai/gpt-oss-120b", provider="groq")
        self.assertEqual(result, "Build together")
        prompt = captured["messages"][1]["content"]
        self.assertIn("2–7 words", prompt)
        self.assertIn("no hashtags", prompt)
        self.assertNotIn("Retrieved evidence", prompt)

    def test_post_package_generation_requests_platform_variants_and_exact_slide_count(self):
        client = fake_client(chat_result=fake_chat('{"slide_texts":["Welcome","Join us"],"hashtags":[],"platform_captions":{}}'))
        captured = {}

        def execute(key, base, service, timeout, operation):
            def capture(**kwargs):
                captured.update(kwargs)
                return client.chat.completions.create(**kwargs)
            return operation(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=capture))))

        with patch.object(generation, "call_groq", side_effect=execute):
            result = generation.generate({"output_mode": "post_package", "slide_count": 2,
                    "event": {"name": "Community Day"}, "platform": "Instagram"}, [],
                api_key="test-key", base_url="https://api.groq.com/openai/v1",
                model="openai/gpt-oss-120b", provider="groq")
        self.assertIn("platform_captions", captured["messages"][1]["content"])
        self.assertIn("exactly 2", captured["messages"][1]["content"])
        self.assertIn("slide_texts", result)

    def test_groq_image_uses_sdk_chat_and_preserves_image_data_url(self):
        image_buffer = BytesIO()
        Image.new("RGB", (8, 8), color="red").save(image_buffer, format="JPEG")
        client = fake_client(chat_result=fake_chat("A red event poster."))
        captured = {}

        def execute(key, base, service, timeout, operation):
            captured.update(key=key, base=base, service=service, timeout=timeout)
            return operation(client)

        with patch.object(media, "call_groq", side_effect=execute):
            result = media.analyze_image(
                image_buffer.getvalue(), "poster.jpg", api_key="test-key",
                model="qwen/qwen3.8-27b", endpoint="https://api.groq.com/openai/v1",
                provider="groq")
        self.assertEqual(result, "A red event poster.")
        self.assertEqual(captured["service"], "image analysis")
        self.assertEqual(captured["base"], "https://api.groq.com/openai/v1")

    def test_groq_image_429_raises_deferred_quota_error_without_retry(self):
        image_buffer = BytesIO()
        Image.new("RGB", (8, 8), color="blue").save(image_buffer, format="JPEG")
        with patch.object(media, "call_groq", side_effect=GroqTransportError(
                "Groq image analysis failed (HTTP 429). 20 requests per day",
                status_code=429, headers={"Retry-After": "60"})) as call:
            with self.assertRaises(media.GeminiQuotaError) as caught:
                media.analyze_image(image_buffer.getvalue(), "poster.jpg", api_key="test-key",
                    model="qwen/qwen3.8-27b", endpoint="https://api.groq.com/openai/v1",
                    provider="groq")
        self.assertEqual(call.call_count, 1)
        self.assertTrue(caught.exception.daily_quota)
        self.assertTrue(caught.exception.retry_after)

    def test_video_transcription_uses_sdk_and_keeps_segment_timestamps(self):
        transcription = SimpleNamespace(model_dump=lambda: {
            "text": "Welcome to the event.",
            "segments": [{"start": 5.0, "text": "Welcome to the event."}],
        })
        client = fake_client(audio_result=transcription)
        with patch.object(media, "call_groq", side_effect=lambda *args: args[-1](client)) as call:
            result = media._groq_transcribe(b"test-key", b"mock-mp3", "event-0.mp3",
                                             "https://api.groq.com/openai/v1", "whisper-large-v3-turbo")
        self.assertEqual(result["segments"][0]["start"], 5.0)
        self.assertEqual(call.call_count, 1)

    def test_groq_video_uses_local_frames_and_whisper_timestamps(self):
        with tempfile.TemporaryDirectory(prefix="event2social-groq-test-") as temp:
            source = Path(temp) / "event.mp4"
            source.write_bytes(b"mock source")
            frame_calls = []
            audio_calls = []

            def extract_frame(path, ffmpeg, timestamp):
                frame_calls.append(timestamp)
                return b"mock-jpeg"

            def extract_audio(path, ffmpeg, start, duration):
                audio_calls.append((start, duration))
                return b"mock-mp3"

            with patch.object(media, "_video_tools", return_value=("ffmpeg.exe", "ffprobe.exe")), \
                 patch.object(media, "_probe_video", return_value=(120.0, True)), \
                 patch.object(media, "_extract_video_frame", side_effect=extract_frame), \
                 patch.object(media, "_extract_audio_chunk", side_effect=extract_audio), \
                 patch.object(media, "_groq_transcribe", return_value={
                     "text": "Welcome to the event.",
                     "segments": [{"start": 5.0, "text": "Welcome to the event."}]}), \
                 patch.object(media, "_groq_chat", return_value="At 01:00, attendees enter the hall.") as chat:
                result = media.analyze_video(
                    source, "event.mp4", api_key="test-key",
                    model="qwen/qwen3.8-27b", audio_model="whisper-large-v3-turbo",
                    endpoint="https://api.groq.com/openai/v1", provider="groq")
            self.assertEqual(len(frame_calls), 3)
            self.assertEqual(audio_calls, [(0, 120)])
            self.assertIn("Welcome to the event.", result)
            self.assertIn("[00:05]", result)
            self.assertIn("At 01:00", result)
            self.assertEqual(len(chat.call_args.args[2][0]["content"]), 7)

    def test_rag_embeddings_remain_separate_from_groq_generation(self):
        with patch.dict("os.environ", {"EVENT2SOCIAL_EMBEDDING_PROVIDER": ""}, clear=False):
            embedding_provider = rag.configured_embedding_provider(
                {"GEMINI_API_KEY": "test-embedding-key"}, "groq")
        self.assertEqual(embedding_provider, "gemini")
        with self.assertRaisesRegex(rag.ProviderError, "does not currently expose an embeddings endpoint"):
            rag.embedding_config({}, "groq")


if __name__ == "__main__":
    unittest.main()
