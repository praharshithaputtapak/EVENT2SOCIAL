"""Local FFmpeg composition for short vertical event reels and slideshows."""
from __future__ import annotations

import os
import logging
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from typing import Any

logger = logging.getLogger(__name__)


def _ffmpeg_path() -> str:
    executable = os.getenv("FFMPEG_PATH", "").strip() or shutil.which("ffmpeg")
    if not executable or not Path(executable).is_file():
        raise RuntimeError("FFmpeg is required to export Reels/Shorts. Install FFmpeg, add it to PATH, or set FFMPEG_PATH.")
    return executable


def _make_overlay(path: Path, text: str, width: int, height: int) -> None:
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font_paths = (Path("C:/Windows/Fonts/segoeui.ttf"), Path("C:/Windows/Fonts/arial.ttf"),
                  Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    font = next((ImageFont.truetype(str(font_path), max(20, width // 16))
                 for font_path in font_paths if font_path.is_file()), ImageFont.load_default())
    clean = " ".join(text.split())[:160]
    lines, line = [], ""
    for word in clean.split():
        candidate = (line + " " + word).strip()
        if draw.textlength(candidate, font=font) > width - 100 and line:
            lines.append(line); line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    lines = lines[:3]
    if lines:
        line_height = max(34, width // 18)
        top = int(height * 0.72)
        box_height = line_height * len(lines) + 36
        draw.rounded_rectangle((25, top, width - 25, top + box_height), radius=18,
                               fill=(0, 0, 0, 155))
        y = top + 16
        for text_line in lines:
            draw.text((width / 2, y), text_line, font=font, fill="white", anchor="ma")
            y += line_height
    image.save(path, format="PNG")


def _make_title_card(path: Path, text: str, width: int, height: int) -> None:
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new("RGB", (width, height), (18, 41, 65))
    draw = ImageDraw.Draw(image)
    draw.ellipse((width * 0.55, -height * 0.12, width * 1.3, height * 0.53), fill=(33, 99, 140))
    draw.ellipse((-width * 0.5, height * 0.72, width * 0.55, height * 1.25), fill=(35, 82, 113))
    font_path = next((p for p in (Path("C:/Windows/Fonts/segoeui.ttf"), Path("C:/Windows/Fonts/arial.ttf"),
                                   Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")) if p.is_file()), None)
    font = ImageFont.truetype(str(font_path), max(28, width // 11)) if font_path else ImageFont.load_default()
    words, lines, line = " ".join(text.split())[:100].split(), [], ""
    for word in words:
        candidate = (line + " " + word).strip()
        if draw.textlength(candidate, font=font) > width - 100 and line:
            lines.append(line); line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    lines = lines[:4] or ["Event highlights"]
    line_height = max(44, width // 9)
    y = (height - line_height * len(lines)) / 2
    for line in lines:
        draw.text((width / 2, y), line, font=font, fill="white", anchor="mm")
        y += line_height
    image.save(path, format="PNG")


def create_vertical_video(segments: list[dict[str, Any]], output_path: str | Path, *,
                          overlays: list[str] | None = None, width: int = 1080,
                          height: int = 1920, transition_seconds: float = 0.35,
                          cover_text: str = "", opening_title: str = "",
                          closing_text: str = "", progress_callback=None) -> tuple[Path, Path]:
    """Compose local photo/video segments to vertical H.264 MP4 with fades and overlays.

    Original video audio is intentionally omitted to avoid mixing incompatible tracks;
    the user can add licensed platform audio after download.
    """
    if not segments:
        raise ValueError("Select at least one video or photograph for the Reel.")
    if len(segments) > 12:
        raise ValueError("Choose at most 12 source segments per Reel export.")
    ffmpeg = _ffmpeg_path()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    durations = []
    for item in segments:
        source = Path(item["path"])
        if not source.is_file():
            raise FileNotFoundError(f"Reel source is unavailable: {source.name}")
        duration = min(max(float(item.get("duration", 3 if item.get("kind") == "image" else 5)), 2), 12)
        durations.append(duration)
    fade = min(max(float(transition_seconds), 0.0), 0.8)

    with tempfile.TemporaryDirectory(prefix="event2social-reel-") as temp_dir:
        temp = Path(temp_dir)
        prepared_segments = []
        prepared_overlays = []
        if opening_title.strip():
            title_card = temp / "opening-title.png"
            _make_title_card(title_card, opening_title, width, height)
            prepared_segments.append({"path": str(title_card), "kind": "image", "duration": 2.5})
            prepared_overlays.append("")
        prepared_segments.extend(segments)
        prepared_overlays.extend(list(overlays or []))
        if closing_text.strip():
            ending_card = temp / "closing-card.png"
            _make_title_card(ending_card, closing_text, width, height)
            prepared_segments.append({"path": str(ending_card), "kind": "image", "duration": 2.5})
            prepared_overlays.append("")
        segments = prepared_segments
        overlays = prepared_overlays
        durations = [min(max(float(item.get("duration", 3 if item.get("kind") == "image" else 5)), 2), 12)
                     for item in segments]
        command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
        for item, duration in zip(segments, durations):
            if item.get("kind") == "image":
                command.extend(["-loop", "1", "-framerate", "30", "-t", str(duration), "-i", str(item["path"])])
            else:
                command.extend(["-ss", str(max(float(item.get("start", 0)), 0)), "-t", str(duration), "-i", str(item["path"])])

        overlay_files = []
        for index, _item in enumerate(segments):
            overlay_text = str((overlays or [])[index] if index < len(overlays or []) else "").strip()
            overlay_file = temp / f"overlay-{index}.png"
            _make_overlay(overlay_file, overlay_text, width, height)
            overlay_files.append(overlay_file)
            command.extend(["-loop", "1", "-framerate", "30", "-t", str(durations[index]), "-i", str(overlay_file)])

        filters = []
        for index, (item, duration) in enumerate(zip(segments, durations)):
            source_label = f"s{index}"
            chain = (f"[{index}:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
                     f"crop={width}:{height},fps=30,setsar=1,trim=duration={duration},settb=AVTB,setpts=PTS-STARTPTS[{source_label}];"
                     f"[{source_label}][{len(segments)+index}:v]overlay=0:0:shortest=1,format=yuv420p,"
                     f"trim=duration={duration},settb=AVTB,setpts=PTS-STARTPTS[v{index}]")
            filters.append(chain)

        current = "v0"
        elapsed = durations[0]
        for index in range(1, len(segments)):
            out_label = f"x{index}"
            if fade > 0:
                offset = max(elapsed - fade * index, 0)
                filters.append(f"[{current}][v{index}]xfade=transition=fade:duration={fade}:offset={offset}[{out_label}]")
            else:
                filters.append(f"[{current}][v{index}]concat=n=2:v=1:a=0[{out_label}]")
            elapsed += durations[index]
            current = out_label
        total_duration = sum(durations) - fade * max(0, len(segments) - 1)
        command.extend(["-filter_complex", ";".join(filters), "-map", f"[{current}]", "-an",
                        "-t", str(total_duration), "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        "-stats_period", "0.5", "-progress", "pipe:1", str(output)])
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        timeout_guard = threading.Timer(900, process.kill)
        timeout_guard.daemon = True
        timeout_guard.start()
        diagnostics = []
        try:
            if process.stdout:
                for line in process.stdout:
                    line = line.strip()
                    if line.startswith("out_time_ms=") and progress_callback:
                        try:
                            fraction = int(line.partition("=")[2]) / max(total_duration * 1_000_000, 1)
                            progress_callback(min(max(fraction, 0.0), 0.99))
                        except ValueError:
                            pass
                    elif line and not line.startswith(("frame=", "fps=", "stream_", "dup_frames=", "drop_frames=", "speed=", "progress=", "out_time=")):
                        diagnostics.append(line)
            return_code = process.wait()
        finally:
            timeout_guard.cancel()
            if process.stdout:
                process.stdout.close()
        if return_code != 0 or not output.is_file() or output.stat().st_size == 0:
            detail = "\n".join(diagnostics) or "FFmpeg did not create an output file."
            logger.error("FFmpeg Reel composition failed (exit code %s). Captured FFmpeg output:\n%s",
                         return_code, detail)
            raise RuntimeError(f"FFmpeg could not compose the Reel: {detail[-1200:]}")

    cover = output.with_name(output.stem + "-cover.jpg")
    cover_result = subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", "0", "-i", str(output), "-frames:v", "1", "-q:v", "2", str(cover)],
        capture_output=True, text=True, timeout=60)
    if cover_result.returncode != 0 or not cover.is_file():
        cover_error = (cover_result.stderr or "Cover image was not created.").strip()
        logger.error("FFmpeg Reel cover extraction failed (exit code %s):\n%s",
                     cover_result.returncode, cover_error)
        raise RuntimeError(f"Reel exported, but cover extraction failed: {cover_error[-500:]}")
    if cover_text.strip():
        from PIL import Image, ImageDraw, ImageFont
        with Image.open(cover) as source:
            image = source.convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")
        font_paths = (Path("C:/Windows/Fonts/segoeui.ttf"), Path("C:/Windows/Fonts/arial.ttf"),
                      Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
        font = next((ImageFont.truetype(str(path), 74) for path in font_paths if path.is_file()), ImageFont.load_default())
        text = " ".join(cover_text.split())[:80]
        lines, line = [], ""
        for word in text.split():
            candidate = (line + " " + word).strip()
            if draw.textlength(candidate, font=font) > image.width - 100 and line:
                lines.append(line); line = word
            else:
                line = candidate
        if line:
            lines.append(line)
        lines = lines[:3]
        top = max(60, int(image.height * 0.68))
        box_height = 62 * len(lines) + 40
        draw.rounded_rectangle((30, top, image.width - 30, top + box_height), radius=18,
                               fill=(0, 0, 0, 165))
        y = top + 20
        for line in lines:
            draw.text((image.width / 2, y), line, font=font, fill="white", anchor="ma")
            y += 62
        image.convert("RGB").save(cover, quality=95)
    return output, cover
