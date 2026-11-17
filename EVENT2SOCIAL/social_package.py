"""Formatting and ZIP export for locally generated social media packages."""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import re
import zipfile
from typing import Any


_HASHTAG_RE = re.compile(r"(?<![\w#])#[\w\u00A1-\uFFFF]+", re.UNICODE)


def edit_slide_order(selected: list[str], action: str, index: int,
                    replacement: str = "") -> list[str]:
    """Return an edited photo-key ordering without mutating caller or widget state."""
    updated = list(selected)
    if action == "up" and 0 < index < len(updated):
        updated[index - 1], updated[index] = updated[index], updated[index - 1]
    elif action == "down" and 0 <= index < len(updated) - 1:
        updated[index], updated[index + 1] = updated[index + 1], updated[index]
    elif action == "remove" and 0 <= index < len(updated):
        updated.pop(index)
    elif action == "replace" and 0 <= index < len(updated) and replacement:
        updated[index] = replacement
        updated = list(dict.fromkeys(updated))
    return updated


def extract_hashtags(text: str) -> list[str]:
    """Return distinct hashtags in appearance order."""
    found = []
    seen = set()
    for tag in _HASHTAG_RE.findall(text or ""):
        key = tag.casefold()
        if key not in seen:
            found.append(tag)
            seen.add(key)
    return found


def render_caption_file(package: dict[str, Any]) -> str:
    """Build a readable publishing text file from editable package fields."""
    caption = str(package.get("caption", "")).strip()
    adaptation = str(package.get("caption_adaptation", "")).strip() or caption
    tags_value = package.get("hashtags", [])
    if isinstance(tags_value, str):
        tags = extract_hashtags(tags_value)
    else:
        tags = [str(item).strip() for item in tags_value if str(item).strip()]
    tag_text = " ".join(tag if tag.startswith("#") else f"#{tag}" for tag in tags)
    publish_caption = _HASHTAG_RE.sub("", adaptation)
    publish_caption = re.sub(r"[ \t]+", " ", publish_caption)
    publish_caption = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", publish_caption).strip()
    if tag_text:
        publish_caption = f"{publish_caption}\n\n{tag_text}".strip()
    cta = str(package.get("cta", "")).strip()
    if cta:
        publish_caption = f"{publish_caption}\n\n{cta}".strip()

    lines = [f"Platform: {package.get('platform', '')}",
             f"Post type: {package.get('post_type', 'Single Image Post')}",
             f"Aspect ratio: {package.get('aspect', '')}", "",
             "Main caption:", caption, "", "Hashtags:", tag_text or "(none)", "",
             "Optional call to action:", cta or "(none)", "",
             f"Caption adapted for {package.get('platform', 'the selected platform')}:",
             adaptation, "", "Publish-ready caption:", publish_caption, "",
             "Platform-specific versions:"]
    for platform, value in package.get("platform_captions", {}).items():
        if value:
            lines.extend([f"{platform}:", str(value), ""])
    lines.extend([
             "Slide text and source images:"]
    )
    slides = sorted(package.get("slides", []), key=lambda item: item.get("position", 0))
    for index, slide in enumerate(slides, 1):
        source = slide.get("display_name") or slide.get("photo_key") or "Selected photograph"
        overlay = str(slide.get("slide_text", "")).strip() or "(no slide overlay)"
        lines.append(f"Slide {index:02d}: {source}\nOverlay: {overlay}")
    return "\n".join(lines).rstrip() + "\n"


def build_package_zip(package: dict[str, Any]) -> bytes:
    """Create an in-memory ZIP with separate slide files and the caption text."""
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        slides = sorted(package.get("slides", []), key=lambda item: item.get("position", 0))
        if not slides:
            raise ValueError("Create at least one visual slide before downloading a package.")
        for index, slide in enumerate(slides, 1):
            path = Path(slide.get("creative_path", ""))
            if not path.is_file():
                raise ValueError(f"Slide {index} image is missing: {path.name or 'unknown file'}")
            zipped.write(path, f"images/slide-{index:02d}.jpg")
            original = Path(slide.get("original_path", ""))
            if original.is_file():
                suffix = original.suffix.lower() if original.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} else ".bin"
                zipped.write(original, f"originals/original-{index:02d}{suffix}")
        zipped.writestr("caption.txt", render_caption_file(package))
        manifest = {key: value for key, value in package.items() if key != "slides"}
        manifest["slides"] = [{key: value for key, value in slide.items()
                               if key not in {"creative_path", "original_path"}} for slide in slides]
        zipped.writestr("package.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return archive.getvalue()
