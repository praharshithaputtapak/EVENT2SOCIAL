"""Local FFmpeg composition for short vertical event reels and slideshows."""
from __future__ import annotations

import os
import logging
import json
from pathlib import Path
import shlex
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


def _ffprobe_path(ffmpeg: str) -> str | None:
    configured = os.getenv("FFPROBE_PATH", "").strip()
    if configured and Path(configured).is_file():
        return configured
    sibling_name = "ffprobe.exe" if Path(ffmpeg).suffix.lower() == ".exe" else "ffprobe"
    sibling = Path(ffmpeg).with_name(sibling_name)
    if sibling.is_file():
        return str(sibling)
    return shutil.which("ffprobe")


def _render_profile(requested: str | None) -> str:
    profile = requested or os.getenv("EVENT2SOCIAL_REEL_PROFILE", "").strip().lower()
    if not profile:
        # This is the source root shown in Streamlit Community Cloud logs.
        profile = "cloud" if Path(__file__).resolve().as_posix().startswith("/mount/src/") else "local"
    if profile not in {"local", "cloud"}:
        raise ValueError("Reel profile must be 'local' or 'cloud'.")
    return profile


def _probe_media(ffprobe: str | None, path: Path) -> dict[str, Any] | None:
    if not ffprobe:
        logger.info("Media probe unavailable for %s (ffprobe not found).", path)
        return None
    command = [ffprobe, "-v", "error", "-show_entries",
               "format=duration:stream=codec_type,codec_name,width,height,duration",
               "-of", "json", str(path)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
        data = json.loads(result.stdout or "{}")
        streams = data.get("streams", [])
        video = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
        metadata = {
            "duration": (data.get("format") or {}).get("duration") or video.get("duration"),
            "resolution": f"{video['width']}x{video['height']}" if video.get("width") and video.get("height") else None,
            "codec": video.get("codec_name"),
        }
        logger.info("Reel input metadata path=%s metadata=%s", path, metadata)
        return metadata
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Could not probe Reel input %s: %s", path, exc)
        return None


def _build_filter_graph(durations: list[float], width: int, height: int,
                        fade: float) -> tuple[str, str]:
    """Build scene and transition filters with normalized xfade inputs."""
    segment_count = len(durations)
    filters = []
    for index, duration in enumerate(durations):
        source_label = f"s{index}"
        overlay_label = f"o{index}"
        filters.append(
            f"[{index}:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},trim=duration={duration},settb=AVTB,"
            f"setpts=PTS-STARTPTS,format=yuv420p,setsar=1,fps=30[{source_label}]"
        )
        filters.append(
            f"[{segment_count + index}:v]trim=duration={duration},settb=AVTB,"
            f"setpts=PTS-STARTPTS,format=rgba,setsar=1,fps=30[{overlay_label}]"
        )
        # Reset timestamps and time base before the final fps filter. In
        # FFmpeg 7.1, setpts clears frame-rate metadata, so no setpts may follow
        # this fps on a path that feeds xfade.
        filters.append(
            f"[{source_label}][{overlay_label}]overlay=0:0:shortest=1,"
            f"trim=duration={duration},settb=AVTB,setpts=PTS-STARTPTS,"
            f"scale={width}:{height},crop={width}:{height},format=yuv420p,setsar=1,fps=30[v{index}]"
        )

    current = "v0"
    elapsed = durations[0]
    for index in range(1, segment_count):
        raw_label = f"xraw{index}"
        output_label = f"x{index}"
        if fade > 0:
            offset = max(elapsed - fade * index, 0)
            filters.append(
                f"[{current}][v{index}]xfade=transition=fade:duration={fade}:"
                f"offset={offset}[{raw_label}]"
            )
        else:
            filters.append(f"[{current}][v{index}]concat=n=2:v=1:a=0[{raw_label}]")
        # Normalize composites too, since they become the first input to the
        # next xfade in a multi-scene Reel.
        filters.append(
            f"[{raw_label}]scale={width}:{height},crop={width}:{height},"
            f"format=yuv420p,setsar=1,settb=AVTB,setpts=PTS-STARTPTS,fps=30[{output_label}]"
        )
        elapsed += durations[index]
        current = output_label
    return ";".join(filters), current


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
                          closing_text: str = "", progress_callback=None,
                          profile: str | None = None) -> tuple[Path, Path]:
    """Compose local photo/video segments to vertical H.264 MP4 with fades and overlays.

    Original video audio is intentionally omitted to avoid mixing incompatible tracks;
    the user can add licensed platform audio after download.
    """
    if not segments:
        raise ValueError("Select at least one video or photograph for the Reel.")
    if len(segments) > 12:
        raise ValueError("Choose at most 12 source segments per Reel export.")
    ffmpeg = _ffmpeg_path()
    ffprobe = _ffprobe_path(ffmpeg)
    render_profile = _render_profile(profile)
    if render_profile == "cloud":
        width, height = 720, 1280
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cover = output.with_name(output.stem + "-cover.jpg")
    durations = []
    input_diagnostics = []
    for item in segments:
        source = Path(item["path"])
        if not source.is_file():
            raise FileNotFoundError(f"Reel source is unavailable: {source.name}")
        duration = min(max(float(item.get("duration", 3 if item.get("kind") == "image" else 5)), 2), 12)
        durations.append(duration)
        size_bytes = source.stat().st_size
        metadata = _probe_media(ffprobe, source)
        input_diagnostics.append({"path": str(source), "kind": item.get("kind", "video"),
                                  "size_bytes": size_bytes, "requested_seconds": duration,
                                  "metadata": metadata})
    logger.info("Starting Reel render profile=%s resolution=%dx%d sources=%s output=%s",
                render_profile, width, height, json.dumps(input_diagnostics, ensure_ascii=False), output)
    fade = min(max(float(transition_seconds), 0.0), 0.8)

    # Keep all intermediates beside the destination so they share its writable
    # filesystem and are removed automatically on every success/failure path.
    with tempfile.TemporaryDirectory(prefix="event2social-reel-", dir=str(output.parent)) as temp_dir:
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
        if render_profile == "cloud":
            command.extend(["-filter_threads", "1", "-filter_complex_threads", "1"])
        for item, duration in zip(segments, durations):
            if item.get("kind") == "image":
                command.extend(["-loop", "1", "-framerate", "30", "-t", str(duration), "-i", str(item["path"])])
            else:
                if render_profile == "cloud":
                    command.extend(["-threads", "1"])
                command.extend(["-ss", str(max(float(item.get("start", 0)), 0)), "-t", str(duration), "-i", str(item["path"])])

        overlay_files = []
        for index, _item in enumerate(segments):
            overlay_text = str((overlays or [])[index] if index < len(overlays or []) else "").strip()
            overlay_file = temp / f"overlay-{index}.png"
            _make_overlay(overlay_file, overlay_text, width, height)
            overlay_files.append(overlay_file)
            command.extend(["-loop", "1", "-framerate", "30", "-t", str(durations[index]), "-i", str(overlay_file)])

        filter_graph, current = _build_filter_graph(durations, width, height, fade)
        total_duration = sum(durations) - fade * max(0, len(segments) - 1)
        temp_output = temp / "render.mp4"
        temp_cover = temp / "render-cover.jpg"
        command.extend(["-filter_complex", filter_graph, "-map", f"[{current}]", "-an",
                        "-t", str(total_duration), "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        *(["-threads:v", "1"] if render_profile == "cloud" else []),
                        "-stats_period", "0.5", "-progress", "pipe:1", str(temp_output)])
        logger.info("Starting FFmpeg process: %s", shlex.join(command))
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, encoding="utf-8", errors="replace", bufsize=1)
        except OSError:
            logger.exception("Could not start FFmpeg for Reel output %s", output)
            raise

        stderr_lines: list[str] = []

        def collect_stderr() -> None:
            if process.stderr:
                for stderr_line in process.stderr:
                    stderr_lines.append(stderr_line.rstrip())

        stderr_reader = threading.Thread(target=collect_stderr, name="reel-ffmpeg-stderr", daemon=True)
        stderr_reader.start()
        timed_out = threading.Event()

        def terminate_for_timeout() -> None:
            timed_out.set()
            try:
                if process.poll() is None:
                    process.kill()
            except OSError:
                logger.debug("FFmpeg process exited while the timeout handler was running.")

        timeout_guard = threading.Timer(900, terminate_for_timeout)
        timeout_guard.daemon = True
        timeout_guard.start()
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
            return_code = process.wait()
        except BaseException:
            if process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass
            process.wait()
            raise
        finally:
            timeout_guard.cancel()
            if process.stdout:
                process.stdout.close()
            stderr_reader.join(timeout=5)
            if process.stderr:
                process.stderr.close()

        stderr_output = "\n".join(stderr_lines).strip()
        logger.info("FFmpeg render process exited with status %s for %s", return_code, output)
        if timed_out.is_set():
            logger.error("FFmpeg render timed out after 900 seconds. stderr:\n%s", stderr_output or "<empty>")
            raise TimeoutError("FFmpeg rendering exceeded the 15-minute limit.")
        if stderr_output:
            logger.warning("FFmpeg render stderr for %s:\n%s", output, stderr_output)
        if return_code != 0:
            logger.error("FFmpeg render failed with exit status %s. stderr:\n%s",
                         return_code, stderr_output or "<empty>")
            raise RuntimeError(f"FFmpeg could not compose the Reel (exit status {return_code}).")
        if not temp_output.is_file() or temp_output.stat().st_size == 0:
            logger.error("FFmpeg exited successfully but did not create a non-empty MP4: %s", temp_output)
            raise RuntimeError("FFmpeg finished without creating a valid Reel video.")
        logger.info("FFmpeg created temporary Reel video path=%s size_bytes=%d",
                    temp_output, temp_output.stat().st_size)
        output_metadata = _probe_media(ffprobe, temp_output)
        if output_metadata and output_metadata.get("resolution") != f"{width}x{height}":
            raise RuntimeError(f"Rendered video resolution validation failed: {output_metadata.get('resolution')}.")

        cover_command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", "0", "-i", str(temp_output), "-frames:v", "1", "-q:v", "2", str(temp_cover)]
        logger.info("Starting FFmpeg cover extraction: %s", shlex.join(cover_command))
        try:
            cover_result = subprocess.run(cover_command, capture_output=True, text=True,
                                          timeout=60, check=False)
        except (OSError, subprocess.SubprocessError):
            logger.exception("FFmpeg cover extraction could not complete for %s", output)
            raise
        cover_stderr = (cover_result.stderr or "").strip()
        logger.info("FFmpeg cover extraction exited with status %s for %s", cover_result.returncode, output)
        if cover_stderr:
            logger.warning("FFmpeg cover extraction stderr for %s:\n%s", output, cover_stderr)
        if cover_result.returncode != 0 or not temp_cover.is_file() or temp_cover.stat().st_size == 0:
            logger.error("FFmpeg cover extraction failed status=%s stderr:\n%s",
                         cover_result.returncode, cover_stderr or "<empty>")
            raise RuntimeError(f"Reel cover extraction failed (FFmpeg exit status {cover_result.returncode}).")

        if cover_text.strip():
            from PIL import Image, ImageDraw, ImageFont
            with Image.open(temp_cover) as source:
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
            image.convert("RGB").save(temp_cover, quality=95)

        from PIL import Image
        with Image.open(temp_cover) as verified_cover:
            verified_cover.verify()
        logger.info("Validated temporary Reel outputs video_bytes=%d cover_bytes=%d",
                    temp_output.stat().st_size, temp_cover.stat().st_size)

        # Preserve any same-named prior outputs until both new files publish.
        backups: dict[Path, Path] = {}
        published: list[Path] = []
        try:
            for destination, backup_name in ((output, "previous-video.mp4"),
                                             (cover, "previous-cover.jpg")):
                if destination.exists():
                    backup = temp / backup_name
                    os.replace(destination, backup)
                    backups[destination] = backup
            os.replace(temp_cover, cover)
            published.append(cover)
            os.replace(temp_output, output)
            published.append(output)
        except Exception:
            logger.exception("Could not publish staged Reel outputs; restoring prior files.")
            for destination in published:
                try:
                    destination.unlink(missing_ok=True)
                except OSError:
                    logger.exception("Could not remove incomplete Reel output %s during rollback.", destination)
            for destination, backup in backups.items():
                if backup.exists():
                    try:
                        os.replace(backup, destination)
                    except OSError:
                        logger.exception("Could not restore prior Reel output %s from %s.", destination, backup)
            raise
        logger.info("Published Reel outputs video=%s video_bytes=%d cover=%s cover_bytes=%d",
                    output, output.stat().st_size, cover, cover.stat().st_size)
    return output, cover
