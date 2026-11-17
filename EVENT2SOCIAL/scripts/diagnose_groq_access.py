"""Run a single, explicit Groq SDK check without ever printing the API key.

Examples from the project root:
  .venv\\Scripts\\python.exe scripts\\diagnose_groq_access.py --auth
  .venv\\Scripts\\python.exe scripts\\diagnose_groq_access.py --text
  .venv\\Scripts\\python.exe scripts\\diagnose_groq_access.py --audio path\\to\\consented-audio.mp3
Each mode makes only its named request; --text and --audio consume provider quota.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from groq_transport import GroqTransportError, call_groq  # noqa: E402


def _secret(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    secret_path = ROOT / ".streamlit" / "secrets.toml"
    if not secret_path.is_file():
        return ""
    try:
        import tomllib
        with secret_path.open("rb") as stream:
            return str(tomllib.load(stream).get(name, "")).strip()
    except Exception:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--auth", action="store_true", help="authenticate with /models and list accessible model IDs")
    modes.add_argument("--text", action="store_true", help="send one short text-generation request")
    modes.add_argument("--audio", metavar="FILE", help="send the selected audio file for one transcription request")
    parser.add_argument("--audio-model", help="override only this diagnostic's audio model; never changes application configuration")
    args = parser.parse_args()

    key = _secret("GROQ_API_KEY")
    if not key:
        print("GROQ_API_KEY is not configured in the process environment or Streamlit secrets.")
        return 2
    base = _secret("GROQ_BASE_URL") or "https://api.groq.com/openai/v1"
    generation_model = _secret("GROQ_MODEL") or "openai/gpt-oss-120b"
    audio_model = args.audio_model or _secret("GROQ_AUDIO_MODEL") or "whisper-large-v3"

    try:
        if args.auth:
            print("Checking Groq authentication and accessible model IDs (key hidden)...")
            models = call_groq(key, base, "authentication/model-list check", 30,
                               lambda client: client.models.list())
            ids = sorted(item.id for item in models.data)
            for model_id in ids:
                print(f"  {model_id}")
            print(f"Configured generation model {generation_model!r}: {'listed' if generation_model in ids else 'not listed'}")
            print(f"Configured audio model {audio_model!r}: {'listed' if audio_model in ids else 'not listed'}")
            print("A model listing confirms access to /models; it does not guarantee that every operation is enabled for a listed model.")
        elif args.text:
            print(f"Sending one Groq text-generation check with {generation_model!r} (key hidden)...")
            response = call_groq(
                key, base, "text-generation diagnostic", 60,
                lambda client: client.chat.completions.create(
                    model=generation_model,
                    messages=[{"role": "user", "content": "Reply with exactly: Groq text check passed."}],
                    max_completion_tokens=256, timeout=60),
            )
            text = str(response.choices[0].message.content or "").strip()
            if not text:
                print("Groq accepted the request but returned no visible text content.")
                return 1
            print("Text generation succeeded:")
            print(text)
        else:
            audio_path = Path(args.audio).expanduser()
            if not audio_path.is_file():
                print("The selected audio file does not exist.")
                return 2
            print(f"Sending selected audio to Groq transcription model {audio_model!r} (key hidden)...")
            with audio_path.open("rb") as stream:
                response = call_groq(
                    key, base, "audio-transcription diagnostic", 180,
                    lambda client: client.audio.transcriptions.create(
                        file=(audio_path.name, stream), model=audio_model,
                        response_format="verbose_json", timestamp_granularities=["segment"], timeout=180),
                )
            print("Audio transcription succeeded:")
            print(str(getattr(response, "text", "")).strip())
    except GroqTransportError as exc:
        print(str(exc))
        return 1
    except Exception as exc:
        print(f"Groq diagnostic failed ({type(exc).__name__}). Check the SDK version and selected request mode.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
