"""Consent-gated provider-backed image and video understanding."""
from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import json
import os
import shutil
import subprocess
import re
import socket
import urllib.error
import urllib.request
from groq_transport import GroqTransportError, call_groq
import time
import mimetypes
from functools import lru_cache
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from PIL import Image

GEMINI_INTERACTIONS = "https://generativelanguage.googleapis.com/v1beta/interactions"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_VISION_MODEL = "qwen/qwen3.8-27b"
GROQ_AUDIO_MODEL = "whisper-large-v3"
MAX_INLINE_REQUEST = 18 * 1024 * 1024
MAX_VIDEO_BYTES_DEFAULT = 2 * 1024 * 1024 * 1024


class ProviderQuotaError(RuntimeError):
    """A provider 429. The caller must stop its batch and defer remaining work."""

    def __init__(self, message: str, *, retry_after: str = "", daily_quota: bool = False):
        super().__init__(message)
        self.retry_after = retry_after
        self.daily_quota = daily_quota


# Keep the historic name for existing internal callers and tests.
GeminiQuotaError = ProviderQuotaError


@dataclass
class MediaBatchOutcome:
    item: Any
    status: str
    value: Any = None
    error: str = ""
    retry_after: str = ""


@dataclass
class MediaAnalysisSkip:
    reason: str


def extract_pdf_embedded_images(content: bytes) -> tuple[list[dict[str, Any]], list[str]]:
    """Extract image objects from PDF pages as normalized PNGs with page metadata."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(content), strict=False)
    except Exception as exc:
        raise ValueError(f"Could not open PDF for embedded image extraction: {exc}") from exc

    extracted: list[dict[str, Any]] = []
    errors: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            images = page.images
        except Exception as exc:
            errors.append(f"Page {page_number}: {type(exc).__name__}: {exc}")
            continue
        for image_index, image_file in enumerate(images, start=1):
            try:
                source_image = image_file.image
                source_image.load()
                image = source_image.copy()
                if image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA"}:
                    image = image.convert("RGB")
                out = BytesIO()
                image.save(out, format="PNG", optimize=True)
                image_bytes = out.getvalue()
                extracted.append({
                    "page_number": page_number,
                    "image_index": image_index,
                    "content": image_bytes,
                    "mime_type": "image/png",
                    "width": image.width,
                    "height": image.height,
                    "sha256": hashlib.sha256(image_bytes).hexdigest(),
                })
            except Exception as exc:
                errors.append(f"Page {page_number}, image {image_index}: {type(exc).__name__}: {exc}")
    return extracted, errors


def is_daily_quota_error(message: str) -> bool:
    return bool(re.search(r"(?i)(requests?\s+per\s+day|daily\s+(?:request\s+)?quota|quota_exceeded|per-day\s+limit)", message or ""))


def retry_time_from_429(headers: Any, message: str) -> str:
    """Return a UTC retry timestamp when the provider supplies a usable delay."""
    now = datetime.now(timezone.utc)
    header = str(headers.get("Retry-After", "")).strip() if headers else ""
    if header:
        try:
            return (now + timedelta(seconds=max(0.0, float(header)))).isoformat(timespec="seconds")
        except ValueError:
            try:
                parsed = parsedate_to_datetime(header)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc).isoformat(timespec="seconds")
            except (TypeError, ValueError, OverflowError):
                pass
    if "retry" in (message or "").casefold():
        units = {"h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
                 "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
                 "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1}
        matches = re.findall(r"(\d+)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b", message, re.I)
        seconds = sum(int(value) * units[unit.lower()] for value, unit in matches)
        if seconds:
            return (now + timedelta(seconds=seconds)).isoformat(timespec="seconds")
    return ""


def media_retry_is_blocked(status: str, error: str, retry_after: str = "", *,
                           force_retry: bool = False, now: datetime | None = None) -> bool:
    """Do not repeat quota-failed media requests without elapsed/provider or user confirmation."""
    protected = status == "deferred" or (status == "failed" and is_daily_quota_error(error))
    if not protected or force_retry:
        return False
    if not retry_after:
        return True
    try:
        due = datetime.fromisoformat(retry_after.replace("Z", "+00:00"))
        if due.tzinfo is None:
            due = due.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        return current < due
    except (TypeError, ValueError):
        return True


def filter_media_batch_items(items: list[dict[str, Any]], cache: dict[str, dict[str, Any]], *,
                             force_retry: bool = False) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Separate work, quota-deferred items, and current cached results before any request."""
    work, deferred, cached_results = [], [], []
    for item in items:
        record = cache.get(item["filename"], {})
        same_source = record.get("sha256") == item.get("sha256")
        if same_source and (record.get("status") == "not_needed"
                            or (record.get("status") == "complete" and record.get("text"))):
            cached_results.append(item["filename"])
        elif same_source and media_retry_is_blocked(record.get("status", ""), record.get("error", ""),
                                                    record.get("retry_after", ""),
                                                    force_retry=force_retry):
            deferred.append(item["filename"])
        else:
            work.append(item)
    return work, deferred, cached_results


def run_media_batch(items: list[Any], worker) -> list[MediaBatchOutcome]:
    """Run independent media jobs; a provider 429 defers the rest of this batch."""
    outcomes: list[MediaBatchOutcome] = []
    for index, item in enumerate(items):
        try:
            result = worker(item)
        except ProviderQuotaError as exc:
            outcomes.append(MediaBatchOutcome(item, "deferred", error=str(exc), retry_after=exc.retry_after))
            outcomes.extend(MediaBatchOutcome(pending, "deferred", error=str(exc), retry_after=exc.retry_after)
                           for pending in items[index + 1:])
            break
        except Exception as exc:
            outcomes.append(MediaBatchOutcome(item, "failed", error=str(exc)))
        else:
            if isinstance(result, MediaAnalysisSkip):
                outcomes.append(MediaBatchOutcome(item, "not_needed", error=result.reason))
            else:
                outcomes.append(MediaBatchOutcome(item, "complete", value=result))
    return outcomes


def media_analysis_model(secrets: Any = None, provider: str = "gemini") -> str:
    """Resolve the media model independently from the text-generation model."""
    from rag import _secret
    if provider == "groq":
        return (os.getenv("GROQ_MEDIA_MODEL", "").strip()
                or _secret(secrets, "GROQ_MEDIA_MODEL") or GROQ_VISION_MODEL)
    if provider != "gemini":
        return ""
    return (os.getenv("GEMINI_MEDIA_MODEL", "").strip()
            or _secret(secrets, "GEMINI_MEDIA_MODEL")
            or os.getenv("GEMINI_MODEL", "").strip()
            or _secret(secrets, "GEMINI_MODEL")
            or "gemini-3.8-flash")


def media_audio_model(secrets: Any = None, provider: str = "groq") -> str:
    if provider != "groq":
        return ""
    from rag import _secret
    return (os.getenv("GROQ_AUDIO_MODEL", "").strip()
            or _secret(secrets, "GROQ_AUDIO_MODEL") or GROQ_AUDIO_MODEL)


def _video_upload_limit() -> int:
    try:
        mb = int(os.getenv("EVENT2SOCIAL_MAX_VIDEO_MB", "2048"))
        return min(max(mb, 1), 20 * 1024) * 1024 * 1024
    except ValueError:
        return MAX_VIDEO_BYTES_DEFAULT


def _response_text(value: Any) -> str:
    if isinstance(value, dict):
        if isinstance(value.get("output_text"), str):
            return value["output_text"].strip()
        pieces = []
        for key, item in value.items():
            if key == "text" and isinstance(item, str):
                pieces.append(item)
            else:
                pieces.append(_response_text(item))
        return "\n".join(x for x in pieces if x).strip()
    if isinstance(value, list):
        return "\n".join(x for x in (_response_text(v) for v in value) if x).strip()
    return ""


def _safe_provider_detail(value: str, api_key: str) -> str:
    detail = str(value or "")
    if api_key:
        detail = detail.replace(api_key, "[redacted]")
    detail = re.sub(r"(?i)(api[_ -]?key|token|authorization)\s*[:=]\s*[^\s,;]+", r"\1=[redacted]", detail)
    return detail[:500]


def _call_groq_media(api_key: str, endpoint: str | None, service: str,
                     operation, *, timeout: int = 180):
    try:
        return call_groq(api_key, endpoint or GROQ_BASE_URL, service, timeout, operation)
    except GroqTransportError as exc:
        if exc.status_code == 429:
            detail = str(exc)
            daily_quota = is_daily_quota_error(detail)
            raise GeminiQuotaError(
                f"Groq {service} rate limit or quota reached (HTTP 429). {detail}",
                retry_after=retry_time_from_429(exc.headers, detail),
                daily_quota=daily_quota,
            ) from None
        raise RuntimeError(str(exc)) from None


def _groq_chat(api_key: str, model: str, messages: list[dict[str, Any]],
               endpoint: str | None = None, *, service: str = "media analysis") -> str:
    body = json.dumps({"model": model, "messages": messages, "stream": False},
                      ensure_ascii=False).encode("utf-8")
    if len(body) > MAX_INLINE_REQUEST:
        raise RuntimeError("The selected media exceeds the 18 MB Groq request limit. Reduce image size or video duration.")
    result = _call_groq_media(
        api_key, endpoint, service,
        lambda client: client.chat.completions.create(
            model=model, messages=messages, stream=False, timeout=180),
        timeout=180)
    try:
        text = result.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        text = ""
    text = str(text or "").strip()
    if not text:
        raise RuntimeError("Groq returned no media description.")
    return text


def _interaction(api_key: str, model: str, inputs: list[dict[str, Any]], endpoint: str | None = None) -> str:
    if not api_key:
        raise RuntimeError("Gemini media analysis is unavailable until the administrator configures GEMINI_API_KEY on the server.")
    body = json.dumps({"model": model, "input": inputs}).encode("utf-8")
    if len(body) > MAX_INLINE_REQUEST:
        raise RuntimeError("The sampled media request exceeds the 18 MB safety limit. Reduce image size or video duration.")
    endpoint = endpoint or os.getenv("GEMINI_INTERACTIONS_URL", "").strip() or GEMINI_INTERACTIONS
    req = urllib.request.Request(endpoint, data=body,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"})
    try:
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=180) as response:
                    result = json.loads(response.read())
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 503 or attempt:
                    raise
                try:
                    delay = min(max(float(exc.headers.get("Retry-After", "2")), 0), 8)
                except (TypeError, ValueError):
                    delay = 2
                time.sleep(delay)
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read())
            detail = str(payload.get("error", {}).get("message", ""))
        except Exception:
            detail = ""
        detail = _safe_provider_detail(detail, api_key)
        if exc.code == 401 or exc.code == 403:
            raise RuntimeError("Gemini media analysis was rejected. The administrator should check server credential/API access." + (f" Provider message: {detail}" if detail else "")) from None
        if exc.code == 429:
            daily_quota = is_daily_quota_error(detail)
            label = "daily quota" if daily_quota else "rate limit or quota"
            message = f"Gemini media analysis {label} reached (HTTP 429)."
            if detail:
                message += f" Provider message: {detail}"
            raise GeminiQuotaError(message, retry_after=retry_time_from_429(exc.headers, detail),
                                   daily_quota=daily_quota) from None
        raise RuntimeError(f"Gemini media analysis failed (HTTP {exc.code})." + (f" Provider message: {detail}" if detail else "")) from None
    except urllib.error.URLError as exc:
        reason = str(exc.reason).replace(api_key, "[redacted]")
        raise RuntimeError(f"Could not connect to Gemini for media analysis: {reason}. Check DNS, proxy/firewall access and endpoint configuration.") from None
    except TimeoutError:
        raise RuntimeError("Gemini media analysis timed out after 180 seconds. Retry or check provider/network status.") from None
    except Exception as exc:
        raise RuntimeError(f"Gemini returned an unreadable media response ({exc.__class__.__name__}).") from None
    text = _response_text(result)
    if not text:
        raise RuntimeError("Gemini returned no media description.")
    return text


def _optimized_image(content: bytes, mime_type: str) -> tuple[bytes, str]:
    from io import BytesIO
    with Image.open(BytesIO(content)) as im:
        im.thumbnail((1600, 1600))
        out = BytesIO()
        if mime_type == "image/png" and im.mode in ("RGBA", "LA"):
            im.save(out, format="PNG", optimize=True)
            return out.getvalue(), "image/png"
        im.convert("RGB").save(out, format="JPEG", quality=84, optimize=True)
        return out.getvalue(), "image/jpeg"


def analyze_image(content: bytes, filename: str, *, api_key: str, model: str,
                  endpoint: str | None = None, provider: str = "gemini") -> str:
    suffix = Path(filename).suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(suffix)
    if not mime:
        raise RuntimeError("Unsupported image type for media analysis.")
    image, mime = _optimized_image(content, mime)
    prompt = ("Analyze this event source image. Transcribe visible text exactly and describe visible activities. "
              "Never guess a name, date, venue, organizer, attendance or outcome. Where a field is directly readable, "
              "return a separate line in the exact form Event name: value, Event date and time: value, Venue: value, "
              "Organizing team: value, Event theme: value, Target audience: value, Event objective: value, "
              "Chief guests and speakers: value, Activities and programmes: value, or Registration link: value. "
              "Omit fields not directly supported by visible text. Put uncertain readings only under Uncertain: and do not present them as facts. "
              "Finish with a short Visual description: line for activities/context.")
    if provider == "groq":
        data_url = f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"
        return _groq_chat(api_key, model, [{
            "role": "user",
            "content": [{"type": "text", "text": f"Source filename: {filename}\n{prompt}"},
                        {"type": "image_url", "image_url": {"url": data_url}}],
        }], endpoint=endpoint, service="image analysis")
    if provider != "gemini":
        raise RuntimeError(f"Image analysis is not supported for {provider.title()}.")
    return _interaction(api_key, model, [
        {"type": "text", "text": f"Source filename: {filename}\n{prompt}"},
        {"type": "image", "data": base64.b64encode(image).decode("ascii"), "mime_type": mime},
    ], endpoint=endpoint)


def rank_photos(media: list[dict[str, Any]], objective: str,
                analysis_cache: dict[str, dict[str, Any]], status_callback=None) -> list[dict[str, Any]]:
    """Rank originals by deduplication, legibility, size, exposure, and available scene text."""
    import re
    query = {w for w in re.findall(r"[a-z0-9]{3,}", objective.casefold())}
    scored, hashes = [], []
    for index, doc in enumerate(media):
        path = Path(doc.get("_path", ""))
        try:
            stat = path.stat()
            width, height, quality, phash = _cached_photo_metrics(str(path), stat.st_mtime_ns, stat.st_size)
        except OSError:
            width, height, quality, phash = 0, 0, 0.0, ()
        duplicate = bool(phash) and any(sum(a != b for a, b in zip(phash, old)) <= 3 for old in hashes)
        if phash:
            hashes.append(phash)
        analysis = analysis_cache.get(doc["filename"], {})
        desc = analysis.get("text", "") if analysis.get("sha256") == doc.get("sha256") and analysis.get("status", "complete") == "complete" else ""
        text = (doc["filename"] + " " + desc).casefold()
        words = set(re.findall(r"[a-z0-9]{3,}", text))
        relevance = len(query & words) / max(1, len(query))
        score = round(quality * 0.62 + relevance * 0.38 - (0.8 if duplicate else 0), 4)
        reasons = []
        if width and height:
            reasons.append(f"{width}×{height} original")
        if duplicate:
            reasons.append("near-duplicate; ranked lower")
        elif quality >= 0.8:
            reasons.append("strong measured sharpness and lighting")
        elif quality < 0.45:
            reasons.append("lower measured quality or resolution")
        if desc and relevance > 0:
            matched = sorted(query & words)
            reasons.append("scene analysis matches focus: " + ", ".join(matched[:4]))
        elif not desc:
            reasons.append("quality-only ranking; not yet analysed by Gemini")
        scored.append({**doc, "score": score, "quality": round(quality, 3),
                       "relevance": round(relevance, 3), "duplicate": duplicate,
                       "resolution": f"{width}×{height}" if width and height else "Unknown",
                       "reason": "; ".join(reasons), "analysis": desc})
        if status_callback:
            status_callback(index + 1, len(media), doc.get("display_name", doc.get("filename", "image")))
    return sorted(scored, key=lambda x: (x["duplicate"], -x["score"], x["filename"].casefold()))


@lru_cache(maxsize=2048)
def _cached_photo_metrics(path: str, modified_ns: int, byte_count: int) -> tuple[int, int, float, tuple[bool, ...]]:
    """Cache CPU-heavy image decoding and quality metrics across Streamlit reruns."""
    del modified_ns, byte_count  # Cache-key components invalidate changed files.
    from PIL import Image, ImageFilter, ImageStat
    try:
        with Image.open(path) as raw:
            im = raw.convert("RGB")
            width, height = im.size
            thumb = im.resize((32, 32)).convert("L")
            pixels = list(thumb.getdata())
            avg = sum(pixels) / len(pixels)
            phash = tuple(pixel >= avg for pixel in pixels)
            luminance = ImageStat.Stat(im.convert("L")).mean[0]
            exposure = max(0.0, 1.0 - abs(luminance - 128) / 128)
            edge_std = ImageStat.Stat(im.convert("L").filter(ImageFilter.FIND_EDGES)).stddev[0]
            quality = min(width * height / 1_000_000, 2.0) * 0.22 + min(edge_std / 45, 1.0) * 0.48 + exposure * 0.30
            return width, height, quality, phash
    except Exception:
        return 0, 0, 0.0, ()


def select_recommended_photos(ranked: list[dict[str, Any]], *, max_slides: int = 8,
                              minimum_quality: float = 0.35) -> list[dict[str, Any]]:
    """Select a small quality-first subset; weak images never pad the carousel."""
    candidates = [item for item in ranked if not item.get("duplicate")
                  and float(item.get("quality", 0.0)) >= minimum_quality]
    if not candidates:
        candidates = [item for item in ranked if not item.get("duplicate")]
    return candidates[:max(1, min(int(max_slides), 10))]


def rank_video_moments(analysis_text: str, objective: str = "") -> list[dict[str, Any]]:
    """Extract timestamped moments from Gemini notes and rank by text relevance."""
    stamp_re = re.compile(r"(?<!\d)(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?!\d)")
    query = {w for w in re.findall(r"[a-z0-9]{3,}", objective.casefold())}
    lines = analysis_text.splitlines()
    moments = []
    for index, line in enumerate(lines):
        match = stamp_re.search(line)
        if not match:
            continue
        hours, minutes, seconds = match.groups()
        start = (int(hours or 0) * 3600) + int(minutes) * 60 + int(seconds)
        description = (line[:match.start()] + " " + line[match.end():]).strip(" -*•	:—–")
        if len(description) < 8 and index + 1 < len(lines):
            next_line = stamp_re.sub("", lines[index + 1]).strip(" -*•	:—–")
            if next_line:
                description = next_line
        if len(description) < 8:
            continue
        words = set(re.findall(r"[a-z0-9]{3,}", description.casefold()))
        overlap = len(query & words) / max(1, len(query))
        moments.append({"start_seconds": start, "timestamp": match.group(0),
                        "description": description[:240], "score": round(overlap, 3),
                        "reason": (f"{len(query & words)} focus terms match in AI notes" if query else "timestamped AI note; visual quality not measured")})
    return sorted(moments, key=lambda moment: (-moment["score"], moment["start_seconds"]))


def extract_video_preview(path: str | Path, start_seconds: int, *, duration: int = 6,
                          output_dir: str | Path) -> Path:
    """Create a short local MP4 preview from an AI-timestamped moment."""
    source = Path(path)
    ffmpeg = os.getenv("FFMPEG_PATH", "").strip() or shutil.which("ffmpeg")
    if not ffmpeg or not Path(ffmpeg).is_file():
        raise RuntimeError("FFmpeg is not available. Install FFmpeg and add it to PATH or set FFMPEG_PATH.")
    if not source.is_file() or source.suffix.lower() not in {".mp4", ".mov"}:
        raise RuntimeError("The source video is missing or not a supported MP4/MOV file.")
    start = max(0, int(start_seconds))
    length = min(max(int(duration), 2), 12)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(f"{source.resolve()}:{start}:{length}".encode()).hexdigest()[:20]
    target = target_dir / f"moment-{digest}.mp4"
    if target.is_file() and target.stat().st_size:
        return target
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start),
               "-i", str(source), "-t", str(length), "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "28", "-c:a", "aac", "-movflags", "+faststart", str(target)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("FFmpeg did not finish making the short preview within 90 seconds.") from exc
    if result.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        detail = (result.stderr or "FFmpeg returned no details.").strip()[-400:]
        raise RuntimeError(f"FFmpeg could not create this moment preview: {detail}")
    return target


def _video_tools() -> tuple[str, str]:
    ffmpeg = os.getenv("FFMPEG_PATH", "").strip() or shutil.which("ffmpeg")
    ffprobe = os.getenv("FFPROBE_PATH", "").strip() or shutil.which("ffprobe")
    if not ffmpeg or not Path(ffmpeg).is_file():
        raise RuntimeError("Groq video analysis needs FFmpeg. Install it and add ffmpeg to PATH or set FFMPEG_PATH.")
    if not ffprobe or not Path(ffprobe).is_file():
        raise RuntimeError("Groq video analysis needs FFprobe to locate sample timestamps and audio tracks. Install it with FFmpeg or set FFPROBE_PATH.")
    return ffmpeg, ffprobe


def _probe_video(path: Path, ffprobe: str) -> tuple[float, bool]:
    command = [ffprobe, "-v", "error", "-show_entries",
               "format=duration:stream=codec_type", "-of", "json", str(path)]
    try:
        result = subprocess.run(command, capture_output=True, timeout=30, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("FFprobe did not inspect the video within 30 seconds.") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace")[-300:]
        raise RuntimeError(f"FFprobe could not inspect the video: {detail or 'unknown error'}")
    try:
        info = json.loads(result.stdout)
        duration = float(info.get("format", {}).get("duration", 0))
        has_audio = any(stream.get("codec_type") == "audio" for stream in info.get("streams", []))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("FFprobe returned invalid video metadata.") from exc
    if duration <= 0:
        raise RuntimeError("The video duration could not be determined.")
    return duration, has_audio


def _extract_video_frame(path: Path, ffmpeg: str, at_seconds: float) -> bytes:
    command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-ss",
               f"{at_seconds:.3f}", "-i", str(path), "-frames:v", "1",
               "-vf", "scale='min(1280,iw)':-2", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"]
    try:
        result = subprocess.run(command, capture_output=True, timeout=60, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"FFmpeg did not extract the frame at {at_seconds:.1f}s within 60 seconds.") from exc
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", "replace")[-300:]
        raise RuntimeError(f"FFmpeg could not extract a video frame at {at_seconds:.1f}s: {detail or 'no frame returned'}")
    return result.stdout


def _extract_audio_chunk(path: Path, ffmpeg: str, start: int, duration: int) -> bytes:
    command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-ss",
               str(start), "-i", str(path), "-t", str(duration), "-vn", "-ac", "1",
               "-ar", "16000", "-b:a", "48k", "-f", "mp3", "pipe:1"]
    try:
        result = subprocess.run(command, capture_output=True, timeout=max(90, duration // 2), check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"FFmpeg did not extract audio starting at {start}s in time.") from exc
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", "replace")[-300:]
        raise RuntimeError(f"FFmpeg could not extract audio starting at {start}s: {detail or 'no audio returned'}")
    return result.stdout


def _groq_transcribe(api_key: str, audio: bytes, filename: str, endpoint: str | None,
                     model: str) -> dict[str, Any]:
    safe_filename = re.sub(r"[^A-Za-z0-9._-]", "_", filename) or "event-audio.mp3"
    result = _call_groq_media(
        api_key, endpoint, "video audio transcription",
        lambda client: client.audio.transcriptions.create(
            file=(safe_filename, audio, "audio/mpeg"), model=model,
            response_format="verbose_json", timestamp_granularities=["segment"],
            timeout=180), timeout=180)
    return result.model_dump() if hasattr(result, "model_dump") else dict(result)


def _format_video_transcript(result: dict[str, Any], offset: int) -> str:
    segments = result.get("segments")
    lines = []
    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, dict) or not str(segment.get("text", "")).strip():
                continue
            stamp_seconds = offset + max(0, int(float(segment.get("start", 0))))
            stamp = f"{stamp_seconds // 60:02d}:{stamp_seconds % 60:02d}"
            lines.append(f"[{stamp}] {str(segment['text']).strip()}")
    if lines:
        return "\n".join(lines)
    text = str(result.get("text", "")).strip()
    if not text:
        return ""
    stamp = f"{offset // 60:02d}:{offset % 60:02d}"
    return f"[{stamp}] {text}"


def _analyze_video_groq(path: Path, filename: str, *, api_key: str, model: str,
                        audio_model: str, endpoint: str | None, status_callback=None) -> str:
    if not api_key:
        raise RuntimeError("Groq video analysis is unavailable until the administrator configures GROQ_API_KEY.")
    if not path.is_file() or not path.stat().st_size:
        raise RuntimeError("The uploaded video file is missing or empty.")
    ffmpeg, ffprobe = _video_tools()
    duration, has_audio = _probe_video(path, ffprobe)
    sample_times = sorted(set(min(max(duration * fraction, 0), max(0, duration - 0.1))
                              for fraction in (0.05, 0.5, 0.95)))
    frames = []
    if status_callback:
        status_callback("Extracting up to three timestamped frames locally with FFmpeg…")
    for timestamp in sample_times:
        frames.append((timestamp, _extract_video_frame(path, ffmpeg, timestamp)))

    transcript = ""
    if has_audio:
        if status_callback:
            status_callback("Extracting and transcribing video audio with Groq Whisper…")
        transcript_parts = []
        chunk_seconds = 600
        for start in range(0, int(duration + 0.999), chunk_seconds):
            chunk_duration = min(chunk_seconds, max(1, int(duration - start)))
            audio = _extract_audio_chunk(path, ffmpeg, start, chunk_duration)
            if len(audio) > 24 * 1024 * 1024:
                raise RuntimeError("An extracted audio segment exceeds Groq's 25 MB transcription upload limit.")
            result = _groq_transcribe(api_key, audio, f"{Path(filename).stem}-{start}.mp3",
                                      endpoint, audio_model)
            transcript_parts.append(_format_video_transcript(result, start))
        transcript = "\n".join(part for part in transcript_parts if part)

    prompt = (f"Analyze up to three timestamped frames sampled from the event video {filename}. "
              "Describe visible activities, useful moments, and readable signs. Return concise notes with approximate MM:SS timestamps "
              "based on the timestamps attached to each frame. Do not infer what happens between sampled frames. "
              "Do not guess names, event facts, attendance, or outcomes. Label uncertain details.\n")
    if transcript:
        prompt += "\nAudio transcription from Groq Whisper, with approximate timestamps:\n" + transcript[:24000]
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for timestamp, frame in frames:
        stamp = f"{int(timestamp) // 60:02d}:{int(timestamp) % 60:02d}"
        content.append({"type": "text", "text": f"Video frame sampled at approximately {stamp}."})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(frame).decode("ascii")}})
    if status_callback:
        status_callback("Analyzing timestamped video frames with Groq vision…")
    visual_notes = _groq_chat(api_key, model, [{"role": "user", "content": content}],
                              endpoint=endpoint, service="video frame analysis")
    if transcript:
        return f"{visual_notes}\n\nAudio transcription (Groq Whisper; approximate timestamps):\n{transcript}"
    return f"{visual_notes}\n\nNo audio stream was found in this video."


def analyze_video(path: str | Path, filename: str, *, api_key: str, model: str,
                  ffmpeg: str | None = None, status_callback=None, timeout_seconds: int = 300,
                  endpoint: str | None = None, provider: str = "gemini",
                  audio_model: str | None = None) -> str:
    if provider == "groq":
        return _analyze_video_groq(Path(path), filename, api_key=api_key, model=model,
                                   audio_model=audio_model or GROQ_AUDIO_MODEL,
                                   endpoint=endpoint, status_callback=status_callback)
    if provider != "gemini":
        raise RuntimeError(f"Video analysis is not supported for {provider.title()}.")
    """Upload a source video with Gemini Files API, wait for ACTIVE, then analyze it."""
    del ffmpeg  # direct Gemini file input avoids local transcoding and preserves timestamps
    if not api_key:
        raise RuntimeError("Gemini video analysis is unavailable until the administrator configures the server key.")
    source = Path(path)
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError("The uploaded video file is missing or empty.")
    if source.stat().st_size > _video_upload_limit():
        limit_mb = _video_upload_limit() // (1024 * 1024)
        raise RuntimeError(f"Video is larger than the configured Gemini upload limit ({limit_mb} MB). Set EVENT2SOCIAL_MAX_VIDEO_MB on the server if your Gemini plan supports a higher limit.")
    suffix = source.suffix.lower()
    mime_type = {".mov": "video/mov", ".mp4": "video/mp4"}.get(suffix) or mimetypes.guess_type(filename)[0]
    if mime_type not in {"video/mp4", "video/quicktime"}:
        raise RuntimeError(f"Unsupported video format: {mime_type}.")
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    metadata = json.dumps({"file": {"display_name": filename}}).encode("utf-8")
    start = urllib.request.Request("https://generativelanguage.googleapis.com/upload/v1beta/files", data=metadata,
        headers={**headers, "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                 "X-Goog-Upload-Header-Content-Length": str(source.stat().st_size),
                 "X-Goog-Upload-Header-Content-Type": mime_type}, method="POST")
    remote_name = ""
    try:
        with urllib.request.urlopen(start, timeout=60) as response:
            upload_url = response.headers.get("X-Goog-Upload-URL") or response.headers.get("Location")
        if not upload_url:
            raise RuntimeError("Gemini did not return a resumable upload URL for the video.")
        if status_callback:
            status_callback("Uploading video to Gemini…")
        def video_chunks():
            with source.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    yield block
        upload = urllib.request.Request(upload_url, data=video_chunks(), headers={
            "Content-Length": str(source.stat().st_size), "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize", "Content-Type": mime_type}, method="POST")
        with urllib.request.urlopen(upload, timeout=180) as response:
            file_record = json.loads(response.read()).get("file", {})
        remote_name = file_record.get("name", "")
        uri = file_record.get("uri", "")
        remote_mime = file_record.get("mimeType", mime_type)
        if not remote_name or not uri:
            raise RuntimeError("Gemini accepted the upload but returned no file reference.")
        if status_callback:
            status_callback("Gemini is processing the video…")
        deadline = time.monotonic() + timeout_seconds
        state = file_record.get("state", "PROCESSING")
        while state in (None, "PROCESSING") and time.monotonic() < deadline:
            time.sleep(3)
            poll = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/{remote_name}", headers=headers)
            with urllib.request.urlopen(poll, timeout=30) as response:
                status_payload = json.loads(response.read())
                file_record = status_payload.get("file", status_payload)
            state = file_record.get("state")
        if state == "FAILED":
            raise RuntimeError(f"Gemini video processing failed: {file_record.get('error', 'provider reported FAILED state')}.")
        if state != "ACTIVE":
            raise RuntimeError(f"Gemini video processing did not finish within {timeout_seconds} seconds (state: {state}).")
        if status_callback:
            status_callback("Video is ready; requesting timestamped activity and speech analysis…")
        prompt = (f"Analyze the complete event video {filename}. Describe event activities and useful moments with approximate MM:SS timestamps. "
                  "Transcribe clearly audible speech only when possible. Identify readable visible text. Do not guess names, event facts, attendance or outcomes; label uncertainty.")
        return _interaction(api_key, model, [
            {"type": "video", "uri": uri, "mime_type": remote_mime, "processing": "agentic"},
            {"type": "text", "text": prompt},
        ], endpoint=endpoint)
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read())
            detail = str(payload.get("error", {}).get("message", ""))
        except Exception:
            detail = ""
        detail = _safe_provider_detail(detail, api_key)
        if exc.code == 429:
            daily_quota = is_daily_quota_error(detail)
            label = "daily quota" if daily_quota else "rate limit or quota"
            message = f"Gemini video upload/processing {label} reached (HTTP 429)."
            if detail:
                message += f" Provider message: {detail}"
            raise GeminiQuotaError(message, retry_after=retry_time_from_429(exc.headers, detail),
                                   daily_quota=daily_quota) from None
        raise RuntimeError(f"Gemini video upload/processing failed (HTTP {exc.code}). {detail}".strip()) from None
    except (urllib.error.URLError, TimeoutError) as exc:
        reason = getattr(exc, "reason", "network timeout")
        raise RuntimeError(f"Could not upload or analyze the video with Gemini: {reason}.") from None
    finally:
        if remote_name:
            delete = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/{remote_name}",
                headers={"x-goog-api-key": api_key}, method="DELETE")
            try:
                urllib.request.urlopen(delete, timeout=15).close()
            except Exception:
                pass
