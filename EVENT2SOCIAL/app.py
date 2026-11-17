"""EVENT2SOCIAL — event information to evidence-based social content."""
from __future__ import annotations

import json
import re
import io
import hashlib
import hmac
import os
import unicodedata
import uuid
import zipfile
from datetime import datetime, date, time
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
import streamlit.components.v1 as components

from generation import generate, generation_config
from media_processing import (analyze_image, analyze_video, rank_photos,
                              select_recommended_photos,
                              rank_video_moments, extract_video_preview, media_analysis_model,
                              media_audio_model,
                              MediaAnalysisSkip, is_daily_quota_error,
                              filter_media_batch_items, run_media_batch)
from pdf_visuals import extract_and_store_pdf_images
from social_package import build_package_zip, edit_slide_order, extract_hashtags, render_caption_file
from reel_creator import create_vertical_video
from rag import (ProviderError, configured_provider, configured_embedding_provider,
                 embed_texts, embedding_config,
                 embedding_index_matches, embedding_request_text, extract_document,
                 file_hash, normalize_provider, prepare_chunks, render_scanned_pdf_pages, retrieve)
from storage import (get_campaign, list_campaigns, list_documents, load_chunks,
                     load_documents, load_state, save_campaign, save_chunks,
                     save_state, save_uploaded_document,
                     update_campaign, update_campaign_selection, UPLOAD_DIR, DATA_DIR,
                     load_global_state, save_global_state, set_active_user, load_media_analysis,
                     save_media_analysis, save_document_extraction, list_document_extractions,
                     PDF_IMAGE_DIR, get_pdf_image_extraction,
                     list_extracted_pdf_images, get_extracted_pdf_image,
                     save_post_package)

st.set_page_config(page_title="EVENT2SOCIAL", page_icon="✳️", layout="wide")

if st.session_state.pop("_clear_auth_fields", False):
    for auth_key in ("login_identity", "login_password", "signup_identity", "signup_password", "signup_confirm"):
        st.session_state.pop(auth_key, None)

try:
    APP_SECRETS = st.secrets
except Exception:
    APP_SECRETS = None

DEFAULT_PROVIDER = configured_provider(APP_SECRETS)
AI_PROVIDER = normalize_provider(DEFAULT_PROVIDER)
EMBED_PROVIDER = configured_embedding_provider(APP_SECRETS, AI_PROVIDER)
_EMBED_KEY, EMBED_BASE, EMBED_MODEL = embedding_config(APP_SECRETS, EMBED_PROVIDER)
_GEN_KEY, GEN_BASE, GEN_MODEL = generation_config(APP_SECRETS, AI_PROVIDER)
MEDIA_MODEL = media_analysis_model(APP_SECRETS, AI_PROVIDER)
MEDIA_AUDIO_MODEL = media_audio_model(APP_SECRETS, AI_PROVIDER)
EMBED_KEY = _EMBED_KEY
GEN_KEY = _GEN_KEY or (_EMBED_KEY if AI_PROVIDER == "gemini" else "")

# Keep the existing SQLite database and uploads intact. Existing workspace
# records stay with the legacy owner while new accounts receive isolated data.
if st.session_state.get("authenticated") and not st.session_state.get("user_id"):
    st.session_state["authenticated"] = False
if st.session_state.get("authenticated") and st.session_state.get("user_id"):
    active_user_id = st.session_state["user_id"]
    if st.session_state.get("_active_session_user") != active_user_id:
        private_prefixes = ("event_form_", "event_suggestion_", "event_voice", "campaign_", "draft_",
                            "content_", "validation_", "approval_", "photo_", "package_", "reel_", "kb_uploads_")
        private_keys = {"theme_mode", "last_retrieval", "active_campaign_id", "media_ai_consent",
                        "pending_navigation", "_last_event_voice", "_last_campaign_voice",
                        "kb_selected_visual_photo", "kb_pdf_photo_gallery_page"}
        for key in list(st.session_state.keys()):
            if key in private_keys or key.startswith(private_prefixes):
                st.session_state.pop(key, None)
        st.session_state["_active_session_user"] = active_user_id
    set_active_user(st.session_state["user_id"], legacy_owner=bool(st.session_state.get("legacy_owner")))
    active_campaign_id = st.session_state.get("active_campaign_id")
    if active_campaign_id and get_campaign(active_campaign_id) is None:
        st.session_state.pop("active_campaign_id", None)
    next_page = st.session_state.pop("pending_navigation", None)
    if next_page:
        st.session_state["route"] = next_page
else:
    set_active_user(None)

ROUTES = {
    "Dashboard": "Dashboard", "My Events": "My Events", "Workspace": "Workspace",
    "Settings": "Settings", "Event Details": "Event Details",
    "Knowledge Base": "Knowledge Base", "RAG Retrieval": "RAG Retrieval",
    "Campaign Creation": "Campaign Creation", "Generated Content": "Generated Content",
    "Content Validation": "Content Validation", "Review & Approval": "Review & Approval",
    "Publishing": "Publishing", "Reel Creator": "Reel Creator",
}
page = st.session_state.get("route", "Dashboard")
if page not in ROUTES:
    page = "Dashboard"
    st.session_state["route"] = page


def go(destination: str) -> None:
    st.session_state["pending_navigation"] = destination
    st.rerun()


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="topbar"><div><strong>EVENT2SOCIAL</strong>'
        f'<span class="topbar-page">{title}</span></div>'
        '<span class="workspace-pill">Local workspace · Saved on this device</span></div>'
        f'<div class="hero"><div class="eyebrow">EVENT2SOCIAL</div>'
        f'<h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def metric_card(icon: str, value: int, label: str) -> None:
    st.markdown(
        f'<div class="metric-card"><span class="metric-icon">{icon}</span>'
        f'<div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True)


def event_details() -> dict:
    return load_state("event_details", {})


def load_dark_preference() -> bool:
    return bool(load_state("appearance_dark", False))


def save_dark_preference(dark: bool) -> None:
    if bool(dark) != load_dark_preference():
        save_state("appearance_dark", bool(dark))


_speech_component = components.declare_component("event2social_speech", path=str(Path(__file__).parent / "components" / "speech"))


def speech_input(key: str) -> str:
    value = _speech_component(key=key, default="")
    return value if isinstance(value, str) else ""


PASSWORD_ROUNDS = 600_000


def password_record(password: str) -> dict[str, str | int]:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ROUNDS)
    return {"salt": salt.hex(), "hash": digest.hex(), "rounds": PASSWORD_ROUNDS}


def password_matches(password: str, account: dict) -> bool:
    try:
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                     bytes.fromhex(account["salt"]), int(account["rounds"]))
        return hmac.compare_digest(digest.hex(), account["hash"])
    except (KeyError, ValueError, TypeError):
        return False


def show_login() -> None:
    hero("EVENT2SOCIAL", "An AI-powered event content creation tool for your organization.")
    accounts = load_global_state("accounts", None)
    if accounts is None:
        accounts = []
    if not any(a.get("user_id") == "legacy" for a in accounts):
        legacy = load_global_state("local_account", None)
        if legacy:
            accounts.insert(0, {"user_id": "legacy", "legacy_owner": True, **legacy})
            save_global_state("accounts", accounts)
    login_tab, signup_tab = st.tabs(["Login", "Create Account / Sign Up"])
    with login_tab:
        with st.form("login_account", border=True):
            identity = st.text_input("Username or email", key="login_identity")
            password = st.text_input("Password", type="password", key="login_password")
            login = st.form_submit_button("Sign in", type="primary")
        if login:
            account = next((a for a in accounts if a.get("identity") == identity.strip().casefold()), None)
            if account and password_matches(password, account):
                st.session_state.update(authenticated=True, user_id=account["user_id"],
                                        legacy_owner=bool(account.get("legacy_owner")), route="Dashboard")
                st.session_state["_clear_auth_fields"] = True
                set_active_user(account["user_id"], legacy_owner=bool(account.get("legacy_owner")))
                st.rerun()
            else:
                st.error("The username/email or password is incorrect.")
    with signup_tab:
        st.caption("Each account has a separate event workspace. Existing unassigned workspace data stays with the original local account.")
        with st.form("register_account", border=True):
            new_identity = st.text_input("Username or email", key="signup_identity")
            new_password = st.text_input("Password", type="password", help="Use at least 12 characters.", key="signup_password")
            confirm = st.text_input("Confirm password", type="password", key="signup_confirm")
            create = st.form_submit_button("Create account", type="primary")
        if create:
            normalized = new_identity.strip().casefold()
            if not normalized or len(new_password) < 12:
                st.error("Enter a username or email and a password of at least 12 characters.")
            elif new_password != confirm:
                st.error("The passwords do not match.")
            elif any(a.get("identity") == normalized for a in accounts):
                st.error("An account with that username/email already exists.")
            else:
                is_legacy_owner = not any(a.get("user_id") == "legacy" for a in accounts)
                account = {"user_id": "legacy" if is_legacy_owner else uuid.uuid4().hex,
                           "legacy_owner": is_legacy_owner, "identity": normalized, **password_record(new_password)}
                accounts.append(account)
                save_global_state("accounts", accounts)
                st.session_state.update(authenticated=True, user_id=account["user_id"], legacy_owner=is_legacy_owner, route="Dashboard")
                st.session_state["_clear_auth_fields"] = True
                set_active_user(account["user_id"], legacy_owner=is_legacy_owner)
                st.rerun()
    st.caption("Passwords are hashed locally. Network access still requires HTTPS to protect credentials in transit.")


def event_date(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value) if value else None
    except (TypeError, ValueError):
        return None


def event_time(value: str | None) -> time | None:
    try:
        return time.fromisoformat(value) if value else None
    except (TypeError, ValueError):
        return None


def _creative_overlay_text(value: Any) -> str:
    """Keep poster text readable with installed fonts; leave saved caption untouched."""
    value = re.sub(r"[*_`~]{1,3}", "", str(value or ""))
    value = "".join(" " if unicodedata.category(char) == "So" or char in "\u200d\ufe0f" else char
                    for char in value)
    return re.sub(r"\s+", " ", value).strip()


def compose_post_creative(caption: str, photo_name: str, aspect: str, campaign_id: int,
                          *, slide_index: int | None = None) -> Path:
    """Compose a downloadable post using an uploaded or PDF-extracted photo."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    source = next((d for d in list_documents() if d["filename"] == photo_name), None)
    if source and Path(source["filename"]).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        input_path = UPLOAD_DIR / source["stored_filename"]
    else:
        extracted = get_extracted_pdf_image(photo_name)
        if not extracted:
            raise ValueError("Choose an uploaded or PDF-extracted event photograph first.")
        input_path = PDF_IMAGE_DIR / extracted["stored_filename"]
    if not input_path.is_file():
        raise ValueError("The selected event photograph is no longer available in local storage.")
    ratios = {"Square (1:1)": (1080, 1080), "Portrait (4:5)": (1080, 1350),
              "Landscape (1.91:1)": (1200, 628)}
    ratio = ratios.get(aspect, ratios["Square (1:1)"])
    image = ImageOps.fit(Image.open(input_path).convert("RGB"), ratio, centering=(0.5, 0.42))
    overlay = Image.new("RGBA", ratio, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    header_height = int(ratio[1] * 0.20)
    draw.rectangle((0, 0, ratio[0], header_height), fill=(13, 35, 40, 190))
    band = int(ratio[1] * 0.32)
    for y in range(band):
        alpha = int(205 * y / band)
        draw.line((0, ratio[1] - band + y, ratio[0], ratio[1] - band + y), fill=(13, 35, 40, alpha))
    image = Image.alpha_composite(image.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(image)
    def unicode_font(size: int):
        for font_path in (Path("C:/Windows/Fonts/segoeui.ttf"),
                          Path("C:/Windows/Fonts/arial.ttf"),
                          Path("C:/Windows/Fonts/calibri.ttf")):
            if font_path.is_file():
                try:
                    return ImageFont.truetype(str(font_path), size)
                except OSError:
                    continue
        return ImageFont.load_default()
    font, small = unicode_font(42), unicode_font(26)
    title_font, meta_font = unicode_font(42), unicode_font(23)
    event = event_details()
    headline = _creative_overlay_text(event.get("name") or "Event Highlights")
    title_lines, title_line = [], ""
    for word in headline.split():
        candidate = (title_line + " " + word).strip()
        if draw.textlength(candidate, font=title_font) > ratio[0] - 100 and title_line:
            title_lines.append(title_line)
            title_line = word
        else:
            title_line = candidate
    if title_line:
        title_lines.append(title_line)
    if len(title_lines) > 2:
        title_lines = title_lines[:2]
        title_lines[-1] = title_lines[-1].rstrip(" .") + "…"
    y_title = 24
    for title_line in title_lines:
        draw.text((50, y_title), title_line, font=title_font, fill="white")
        y_title += 48
    meta = " · ".join(str(event.get(key, "")).strip() for key in ("date", "venue") if event.get(key))
    if meta:
        draw.text((50, header_height - 35), meta[:90], font=meta_font, fill=(220, 235, 231))
    lines, line = [], ""
    for word in _creative_overlay_text(caption).split():
        candidate = (line + " " + word).strip()
        if draw.textlength(candidate, font=font) > ratio[0] - 100 and line:
            lines.append(line); line = word
        else:
            line = candidate
    if line: lines.append(line)
    y = ratio[1] - band + 40
    for text in lines[:4]:
        draw.text((50, y), text, font=font, fill="white", stroke_width=1, stroke_fill=(0, 0, 0, 100)); y += 52
    draw.text((50, ratio[1] - 54), "EVENT2SOCIAL", font=small, fill=(235, 244, 239))
    suffix = f"-slide-{slide_index + 1:02d}" if slide_index is not None else ""
    out = DATA_DIR / f"creative-{campaign_id}{suffix}-{ratio[0]}x{ratio[1]}.jpg"
    image.convert("RGB").save(out, quality=92)
    return out


def visual_photo_sources() -> list[dict]:
    """Return current uploaded photos and current-version PDF image assets."""
    docs = list_documents()
    pdf_hashes = {d["filename"]: d["sha256"] for d in docs
                  if Path(d["filename"]).suffix.lower() == ".pdf"}
    photos = []
    for doc in docs:
        if Path(doc["filename"]).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            path = UPLOAD_DIR / doc["stored_filename"]
            if path.is_file():
                photos.append({**doc, "photo_key": doc["filename"], "display_name": doc["filename"],
                               "source_type": "upload", "_path": str(path)})
    for image in list_extracted_pdf_images():
        if pdf_hashes.get(image["source_filename"]) != image["source_hash"]:
            continue
        path = PDF_IMAGE_DIR / image["stored_filename"]
        if not path.is_file():
            continue
        label = f"{image['source_filename']} · page {image['page_number']} · image {image['image_index']}"
        photos.append({"filename": image["photo_key"], "photo_key": image["photo_key"],
                       "sha256": image["image_hash"], "stored_filename": image["stored_filename"],
                       "display_name": label, "source_type": "pdf", "source_filename": image["source_filename"],
                       "page_number": image["page_number"], "image_index": image["image_index"],
                       "width": image["width"], "height": image["height"], "_path": str(path)})
    return photos


def show_pdf_photo_gallery(docs: list[dict]) -> None:
    """Extract each current PDF once and let the user select an image for a post."""
    pdf_docs = [d for d in docs if Path(d["filename"]).suffix.lower() == ".pdf"]
    if not pdf_docs:
        return
    st.subheader("Photographs embedded in PDFs")
    st.caption("Extracted images stay on this device and remain linked to their source PDF and page. They are not sent to an AI service by extraction.")
    for doc in pdf_docs:
        run = get_pdf_image_extraction(doc["filename"], doc["sha256"])
        images = list_extracted_pdf_images(doc["filename"], doc["sha256"])
        files_present = all((PDF_IMAGE_DIR / item["stored_filename"]).is_file() for item in images)
        already_done = bool(run and run["status"] in {"complete", "no_images"} and files_present)
        if already_done:
            if images:
                st.success(f"{doc['filename']}: {len(images)} embedded image(s) extracted; repeated processing is skipped.")
            else:
                st.info(f"{doc['filename']}: no embedded images were found.")
        else:
            button_label = "Retry PDF image extraction" if run else f"Extract embedded photographs from {doc['filename']}"
            if run and run.get("error"):
                st.warning(f"Previous extraction issue for {doc['filename']}: {run['error']}")
            if st.button(button_label, key=f"extract_pdf_images_{doc['sha256'][:16]}"):
                path = UPLOAD_DIR / doc["stored_filename"]
                try:
                    with st.spinner(f"Extracting images from {doc['filename']}…"):
                        result = extract_and_store_pdf_images(doc["filename"], doc["sha256"], path)
                    if result["images"]:
                        st.success(f"Extracted or reused {len(result['images'])} unique embedded image(s) from {doc['filename']}.")
                        for error in result["errors"][:5]:
                            st.warning(error)
                    elif result["errors"]:
                        st.error(f"Could not extract images from {doc['filename']}: {result['errors'][0]}")
                    else:
                        st.info(f"No embedded images were found in {doc['filename']}.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not extract images from {doc['filename']}: {exc}")

    current_photos = [p for p in visual_photo_sources() if p["source_type"] == "pdf"]
    selected = st.session_state.get("kb_selected_visual_photo", "")
    selected_photo = next((p for p in current_photos if p["photo_key"] == selected), None)
    if selected_photo:
        st.info(f"Selected for the next visual post: {selected_photo['display_name']}")
    if current_photos:
        st.markdown("#### Extracted image gallery")
        page_count = max(1, (len(current_photos) + 11) // 12)
        if int(st.session_state.get("kb_pdf_photo_gallery_page", 1)) > page_count:
            st.session_state["kb_pdf_photo_gallery_page"] = 1
        gallery_page = st.number_input("Gallery page", min_value=1, max_value=page_count,
                                       step=1, key="kb_pdf_photo_gallery_page", label_visibility="collapsed")
        page_start = (int(gallery_page) - 1) * 12
        visible_photos = current_photos[page_start:page_start + 12]
        st.caption(f"Showing {page_start + 1}–{page_start + len(visible_photos)} of {len(current_photos)} extracted photos.")
        columns = st.columns(3)
        for index, photo in enumerate(visible_photos):
            with columns[index % len(columns)]:
                st.image(photo["_path"], caption=photo["display_name"], use_container_width=True)
                st.caption(f"{photo['width']} × {photo['height']} · source PDF page {photo['page_number']}")
                if st.button("Use for Instagram visual", key=f"select_pdf_photo_{photo['photo_key']}"):
                    st.session_state["kb_selected_visual_photo"] = photo["photo_key"]
                    st.rerun()


EVENT_FIELDS = {
    "name": ("Event name", ("event name", "name of event", "event title", "title")),
    "description": ("About the event", ("event description", "description", "about the event", "overview", "summary")),
    "date": ("Event date and time", ("event date and time", "event date", "date and time", "date", "when")),
    "venue": ("Venue or location", ("venue", "location", "event location", "where")),
    "organizer": ("Organizing team", ("organizer", "organiser", "organizing team", "organising team", "host", "organized by", "organised by")),
    "theme": ("Event theme", ("event theme", "theme")),
    "audience": ("Target audience", ("target audience", "audience", "who should attend")),
    "objective": ("Event objective", ("event objective", "objective", "purpose", "goal", "goals")),
    "speakers": ("Chief guests and speakers", ("chief guest", "chief guests", "chief guests and speakers", "speaker", "speakers", "keynote speaker", "guest of honour", "guest of honor")),
    "activities": ("Activities and programmes", ("activities", "activity", "activities and programmes", "programmes", "programs", "programme highlights")),
    "url": ("Registration or event link", ("registration link", "event link", "website", "url")),
    "extra": ("Other confirmed information", ("other information", "additional information", "confirmed details")),
}


def extract_event_suggestions(documents: list[tuple[str, bytes, str]]) -> dict:
    """Extract clearly labelled profile values using the existing document readers.

    This is intentionally a review aid, not a fact checker. Only labelled text is
    proposed; values retain their filenames and PDF page numbers for comparison.
    """
    candidates: dict[str, list[dict]] = {key: [] for key in EVENT_FIELDS}
    errors: list[str] = []
    all_labels = tuple(alias for _, aliases in EVENT_FIELDS.values() for alias in aliases)
    label_to_field = {alias: field for field, (_, aliases) in EVENT_FIELDS.items() for alias in aliases}
    label_pattern = re.compile(r"^\s*([A-Za-z][A-Za-z /_-]{0,50}?)\s*(?::|\||–|—|-)\s*(.*)$")

    for filename, content, _digest in documents:
        media_item = load_media_analysis().get(filename, {})
        media_text = media_item.get("text", "") if media_item.get("sha256") == _digest else ""
        if Path(filename).suffix.lower() in {".mp4", ".mov"}:
            units = [{"page_number": None, "text": media_text}] if media_text else []
            if not units:
                errors.append(f"{filename}: media analysis has not completed.")
                continue
        else:
            try:
                units = extract_document(filename, content)
                for unit in units:
                    save_document_extraction(filename, _digest, unit.get("page_number"), "text", 
                        "complete" if unit.get("text", "").strip() else "empty", text=unit.get("text", ""),
                        error="" if unit.get("text", "").strip() else (unit.get("error") or "No readable text on this page."))
                if media_text:
                    units.append({"page_number": None, "text": media_text})
            except Exception as exc:
                if media_text:
                    units = [{"page_number": None, "text": media_text}]
                    save_document_extraction(filename, _digest, None, "text", "failed", error=str(exc))
                else:
                    errors.append(f"{filename}: {exc}")
                    save_document_extraction(filename, _digest, None, "text", "failed", error=str(exc))
                    continue
        found_any_text = False
        for unit in units:
            text = unit.get("text", "")
            if text.strip():
                found_any_text = True
            lines = text.splitlines()
            for i, line in enumerate(lines):
                match = label_pattern.match(line)
                raw_label = match.group(1) if match else line.strip()
                alias = re.sub(r"\s+", " ", raw_label.lower())
                if alias not in label_to_field:
                    continue
                field = label_to_field[alias]
                value = match.group(2).strip() if match else ""
                if not value:
                    section = []
                    for next_line in lines[i + 1:]:
                        if not next_line.strip():
                            if section:
                                break
                            continue
                        next_match = label_pattern.match(next_line)
                        next_alias = (re.sub(r"\s+", " ", next_match.group(1).strip().lower())
                                      if next_match else "")
                        if next_alias in all_labels:
                            break
                        section.append(next_line.strip())
                    value = " ".join(section).strip()
                if not value:
                    continue
                page_number = unit.get("page_number")
                if page_number is None and Path(filename).suffix.lower() == ".pdf":
                    page_match = re.search(r"Source:\s*.+?\bpage\s+(\d+)", text, re.IGNORECASE)
                    if page_match:
                        page_number = int(page_match.group(1))
                source = {"filename": filename, "page_number": page_number}
                normalized = re.sub(r"\s+", " ", value).casefold()
                existing = next((item for item in candidates[field]
                                 if re.sub(r"\s+", " ", item["value"]).casefold() == normalized), None)
                if existing:
                    if source not in existing["sources"]:
                        existing["sources"].append(source)
                else:
                    candidates[field].append({"value": value, "sources": [source]})
        if not found_any_text:
            errors.append(f"{filename}: no readable text was extracted. Use scanned-page OCR or the configured vision provider where available.")

    return {"fields": {key: items for key, items in candidates.items() if items}, "errors": errors,
            "source_hashes": {name: digest for name, _content, digest in documents},
            "extracted_at": datetime.now().isoformat(timespec="minutes")}


def document_search_service() -> str:
    """Show only the configured API host and model; never display credentials."""
    host = urlparse(EMBED_BASE).hostname or "Configured API endpoint"
    return f"{EMBED_PROVIDER.title()} · {host} · {EMBED_MODEL}"


def campaign_selectbox(key: str = "campaign_selector") -> dict | None:
    campaigns = list_campaigns()
    if not campaigns:
        st.info("No saved content yet. Create a campaign to see it here.")
        if st.button("Create campaign", key=f"{key}_create", type="primary"):
            go("Campaign Creation")
        return None
    labels = {f"{c['event_name']} · {c['platform']} · {c['created_at']} · #{c['id']}": c for c in campaigns}
    return labels[st.selectbox("Choose a saved post", list(labels), key=key)]


def show_event_page() -> None:
    hero("Event Details", "Review details found in your documents, then confirm the information for your event profile.")
    saved = event_details()
    extracted = load_state("pending_event_extraction", {})
    suggestions = extracted.get("fields", {})
    current_hashes = {doc["filename"]: doc["sha256"] for doc in list_documents()}
    if suggestions and extracted.get("source_hashes") != current_hashes:
        st.info("The uploaded documents have changed since these suggestions were created. Re-extract details in Knowledge Base before using them.")
        suggestions = {}
    for field in EVENT_FIELDS:
        key = f"event_form_{field}"
        candidates = suggestions.get(field, [])
        suggestion = candidates[0]["value"] if len(candidates) == 1 else ""
        if key not in st.session_state or (not st.session_state[key] and not saved.get(field, "")):
            st.session_state[key] = saved.get(field, "") or suggestion

    if suggestions:
        with st.container(border=True):
            st.subheader("Information found in your documents")
            st.caption("Document values are suggestions, not verified facts. Review the source and confirm the values you want to use in your event profile.")
            selected: dict[str, dict] = {}
            confirmed_sources = load_state("event_profile_sources", {})
            for field, candidates in suggestions.items():
                label = EVENT_FIELDS[field][0]
                if len(candidates) == 1:
                    item = candidates[0]
                    sources = ", ".join(source["filename"] + (f" (page {source['page_number']})" if source.get("page_number") else "") for source in item["sources"])
                    status = " · previously confirmed" if (saved.get(field) == item["value"] and confirmed_sources.get(field, {}).get("value") == item["value"]) else " · suggested"
                    st.write(f"**{label}{status}:** {item['value']}  \nSource: {sources}")
                    selected[field] = item
                else:
                    choices = []
                    for item in candidates:
                        sources = ", ".join(source["filename"] + (f" (page {source['page_number']})" if source.get("page_number") else "") for source in item["sources"])
                        choices.append(f"{item['value']} — {sources}")
                    choice_key = f"event_suggestion_choice_{field}"
                    if st.session_state.get(choice_key) not in choices:
                        st.session_state.pop(choice_key, None)
                    if saved.get(field):
                        st.caption(f"Current saved {label.lower()}: {saved[field]}")
                    choice = st.selectbox(f"{label} · choose the correct value", choices, key=choice_key)
                    selected[field] = candidates[choices.index(choice)]
            for error in extracted.get("errors", []):
                st.warning(error)
            st.caption("Using suggestions only fills the editable form. Your saved event profile changes only when you save it below.")
            if st.button("Use selected document suggestions", key="apply_event_suggestions"):
                for field, item in selected.items():
                    st.session_state[f"event_form_{field}"] = item["value"]
                st.rerun()

    st.caption("Voice entry needs a supported browser and HTTPS when accessed over the network. Your browser may use its own speech service; recording starts only when you press the microphone. Manual entry is always available.")
    voice_text = speech_input("event_voice")
    if voice_text and st.session_state.get("_last_event_voice") != voice_text:
        st.session_state["event_form_description"] = (st.session_state.get("event_form_description", "") + " " + voice_text).strip()
        st.session_state["_last_event_voice"] = voice_text
        st.rerun()

    status_options = ["Upcoming", "Ongoing", "Completed"]
    event_status = st.selectbox("Event status", status_options,
                                index=status_options.index(saved.get("status", "Upcoming"))
                                if saved.get("status", "Upcoming") in status_options else 0,
                                key="event_form_status")
    objective_options = {
        "Upcoming": ["Announcement", "Promotion", "Invitation", "Registration campaign"],
        "Ongoing": ["Live update", "Activity highlight", "Event coverage"],
        "Completed": ["Post-event report", "Event highlights", "Achievements",
                      "Participant experiences", "Thank-you post"],
    }[event_status]
    if st.session_state.get("_event_objective_status") != event_status:
        st.session_state.pop("event_form_objective", None)
        st.session_state["_event_objective_status"] = event_status
    old_objective = saved.get("objective", "")
    default_objective = "Post-event report" if event_status == "Completed" else (old_objective if old_objective in objective_options else objective_options[0])
    with st.form("event_form", border=True):
        name = st.text_input("Event name *", placeholder="Enter the event name", key="event_form_name")
        description = st.text_area("About the event", placeholder="What is the event about? What can attendees expect?", height=115, key="event_form_description")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Dates and times")
            start_date = st.date_input("Start date", value=event_date(saved.get("start_date")), key="event_form_start_date")
            end_date = st.date_input("End date", value=event_date(saved.get("end_date")), key="event_form_end_date")
            has_start_time = st.checkbox("Start time is known", value=bool(saved.get("start_time")), key="event_form_has_start_time")
            start_time = st.time_input("Start time", value=event_time(saved.get("start_time")) or time(9, 0), key="event_form_start_time", disabled=not has_start_time) if has_start_time else None
            has_end_time = st.checkbox("End time is known", value=bool(saved.get("end_time")), key="event_form_has_end_time")
            end_time = st.time_input("End time", value=event_time(saved.get("end_time")) or time(17, 0), key="event_form_end_time", disabled=not has_end_time) if has_end_time else None
            if saved.get("date") and not saved.get("start_date"):
                st.caption(f"Previously saved date: {saved['date']}")
            venue = st.text_input("Venue name", value=saved.get("venue_name", saved.get("venue", "")), placeholder="Venue or institution", key="event_form_venue")
            location = st.text_input("City and location", value=saved.get("location", ""), placeholder="City, region", key="event_form_location")
            maps_link = st.text_input("Google Maps location link", value=saved.get("maps_link", ""), placeholder="Optional map link", key="event_form_maps_link")
            organizer = st.text_input("Organizing team", placeholder="Organization or contact person", key="event_form_organizer")
        with c2:
            audience = st.text_input("Target audience", placeholder="Who should this post reach?", key="event_form_audience")
            objective = st.selectbox("Post objective", objective_options, index=objective_options.index(default_objective), key="event_form_objective")
            url = st.text_input("Registration or event link", placeholder="Paste a public event link, if available", key="event_form_url")
        theme = st.text_input("Event theme", placeholder="The main subject or focus of the event", key="event_form_theme")
        speakers = st.text_area("Confirmed chief guests and speakers", value=saved.get("speakers", ""), height=70, key="event_form_speakers")
        activities = st.text_area("Confirmed activities and programmes", value=saved.get("activities", ""), height=70, key="event_form_activities")
        extra = st.text_area("Other confirmed information", placeholder="Add confirmed speakers, activities, access details, or ticket information", height=85, key="event_form_extra")
        c1, c2 = st.columns(2)
        submitted = c1.form_submit_button("Save event details", type="secondary", use_container_width=True)
        proceed = c2.form_submit_button("Save and add documents", type="primary", use_container_width=True)
    if submitted or proceed:
        if not name.strip():
            st.error("Add an event name before saving.")
        else:
            profile = {"name": name.strip(), "description": description.strip(),
                "date": (f"{start_date.isoformat() if start_date else ''} {start_time.strftime('%H:%M') if start_time else ''}–{end_date.isoformat() if end_date else ''} {end_time.strftime('%H:%M') if end_time else ''}".strip(" –") if start_date or end_date else saved.get("date", "")),
                "start_date": start_date.isoformat() if start_date else "", "end_date": end_date.isoformat() if end_date else "",
                "start_time": start_time.strftime("%H:%M") if start_time else "", "end_time": end_time.strftime("%H:%M") if end_time else "",
                "status": event_status, "venue": venue.strip(), "venue_name": venue.strip(),
                "location": location.strip(), "maps_link": maps_link.strip(), "organizer": organizer.strip(),
                "audience": audience.strip(), "objective": objective.strip(), "url": url.strip(),
                "theme": theme.strip(), "speakers": speakers.strip(), "activities": activities.strip(), "extra": extra.strip()}
            save_state("event_details", profile)
            source_map = load_state("event_profile_sources", {})
            for field in EVENT_FIELDS:
                candidates = suggestions.get(field, [])
                matching = next((item for item in candidates
                                 if item["value"].casefold() == profile.get(field, "").casefold()), None)
                if matching:
                    source_map[field] = {"value": matching["value"], "sources": matching["sources"]}
                elif saved.get(field, "") != profile.get(field, ""):
                    source_map.pop(field, None)
            save_state("event_profile_sources", source_map)
            if proceed:
                go("Knowledge Base")
            st.success("Event details saved on this device.")
            st.rerun()


def show_knowledge_page() -> None:
    hero("Knowledge Base", "Add your event documents so posts can use relevant, source-based information.")
    st.caption("Reading files and extracting labelled event details happen locally. Preparing semantic document search requires an external API.")
    if not EMBED_KEY:
        st.warning("Document search needs a server-configured embedding credential. Ask the administrator to configure it once; ordinary users do not enter API keys.")
        if st.button("View administrator setup", key="configure_embedding"):
            go("Settings")
    else:
        st.caption(f"Document search API is configured · {document_search_service()}")
    uploads = st.file_uploader("Choose event files", type=["txt", "pdf", "docx", "jpg", "jpeg", "png", "webp", "mp4", "mov"], accept_multiple_files=True,
        key=f"kb_uploads_{st.session_state.get('user_id', 'anonymous')}",
        help="Add documents, images, and videos together. Images and scanned PDF pages use local Tesseract OCR when installed.")
    if uploads and st.button("Save uploaded documents", type="secondary"):
        saved_count = 0
        for upload in uploads:
            try:
                content = upload.getvalue()
                save_uploaded_document(upload.name, content, file_hash(content))
                saved_count += 1
            except Exception as exc:
                st.error(f"Could not save {upload.name}: {exc}")
        if saved_count:
            st.success(f"Saved {saved_count} file(s) to local storage.")
            st.rerun()

    docs = list_documents()
    index = load_state("knowledge_index", {})
    show_pdf_photo_gallery(docs)
    current_hashes = {doc["filename"]: doc["sha256"] for doc in docs}
    index_current = embedding_index_matches(index, provider=EMBED_PROVIDER, base_url=EMBED_BASE,
                                            model=EMBED_MODEL, source_hashes=current_hashes)
    if docs:
        if st.button("Extract event details from documents", key="extract_event_profile"):
            with st.spinner("Reading labelled event information from your documents…"):
                result = extract_event_suggestions(load_documents())
                save_state("pending_event_extraction", result)
            if result.get("fields"):
                st.success("Event details were found. Review the suggestions before saving them to your event profile.")
            else:
                st.info("No clearly labelled event details were found. You can still enter your event information below.")
            st.rerun()
        extraction = load_state("pending_event_extraction", {})
        current_hashes = {doc["filename"]: doc["sha256"] for doc in docs}
        if extraction.get("source_hashes") == current_hashes and extraction.get("extracted_at"):
            if not extraction.get("fields"):
                st.info("No clearly labelled event details were found. You can still enter your event information below.")
            for error in extraction.get("errors", []):
                st.warning(error)
        if extraction.get("fields") and extraction.get("source_hashes") == current_hashes:
            count = sum(len(items) for items in extraction["fields"].values())
            st.info(f"{count} suggested value(s) are ready for review. They are not confirmed event facts.")
            if st.button("Review extracted details in Event Details", key="review_extracted_event"):
                go("Event Details")
        elif extraction.get("fields"):
            st.caption("New documents have been added since the last extraction. Extract again to refresh the suggestions.")
    with st.container(border=True):
        st.subheader("Your event documents")
        if not docs:
            st.info("No documents added yet. Choose event files above, then save them here.")
        media_cache = load_media_analysis()
        extraction_records = list_document_extractions()
        for doc in docs:
            current = index_current and doc["sha256"] == index.get("source_hashes", {}).get(doc["filename"])
            ready = current and doc["filename"] in index.get("indexed_filenames", [])
            suffix = Path(doc['filename']).suffix.lower()
            cached_media = media_cache.get(doc["filename"], {})
            analyzed = cached_media.get("sha256") == doc["sha256"] and cached_media.get("status") == "complete" and bool(cached_media.get("text"))
            extraction_rows = [r for r in extraction_records if r["filename"] == doc["filename"] and r["sha256"] == doc["sha256"]]
            extracted_ok = any(r["status"] == "complete" and r["text"] for r in extraction_rows)
            extraction_failed = any(r["status"] in {"failed", "empty"} for r in extraction_rows)
            status = ('Ready to search' if ready else 'Analysis in progress' if cached_media.get("status") == "processing" and cached_media.get("sha256") == doc["sha256"]
                      else 'Analysed · available to campaigns' if analyzed
                      else 'Analysis deferred · provider limit' if cached_media.get("sha256") == doc["sha256"] and (cached_media.get("status") == "deferred" or (cached_media.get("status") == "failed" and is_daily_quota_error(cached_media.get("error", ""))))
                      else 'Analysis failed' if cached_media.get("sha256") == doc["sha256"] and cached_media.get("status") == "failed"
                      else 'Media analysis not needed' if cached_media.get("sha256") == doc["sha256"] and cached_media.get("status") == "not_needed"
                      else 'Text extracted · needs indexing' if extracted_ok
                      else 'No readable text · media analysis available' if extraction_failed
                      else 'Uploaded · needs media analysis' if suffix in {'.mp4', '.mov', '.jpg', '.jpeg', '.png', '.webp'}
                      else 'Uploaded · needs processing')
            st.markdown(f"**{doc['filename']}** · {suffix.lstrip('.').upper()} · {doc['byte_count']:,} bytes · {status}")
            file_path = UPLOAD_DIR / doc['stored_filename']
            if suffix in {'.jpg', '.jpeg', '.png', '.webp'} and file_path.is_file():
                st.image(str(file_path), caption=doc['filename'], width=320)
            elif suffix in {'.mp4', '.mov'} and file_path.is_file():
                try:
                    st.video(str(file_path))
                except Exception:
                    st.caption("This browser cannot preview the video format.")
            if cached_media.get("sha256") == doc["sha256"] and cached_media.get("status") == "complete" and cached_media.get("text"):
                st.caption(f"{AI_PROVIDER.title()} media analysis saved as independent event context · available directly to campaign generation")
                if cached_media.get("error"):
                    st.warning(f"The latest analysis retry failed, so the previously successful result is retained. {cached_media['error']}")
                with st.expander("View media analysis", expanded=False):
                    st.write(cached_media["text"])
                if cached_media.get("timestamps"):
                    st.caption("Video timestamps: " + ", ".join(cached_media["timestamps"]))
                if suffix in {".mp4", ".mov"}:
                    profile = event_details()
                    focus_terms = " ".join(str(profile.get(k, "")) for k in ("name", "description", "objective"))
                    moments = rank_video_moments(cached_media["text"], focus_terms)
                    if moments:
                        st.caption("Timestamped moments are ranked by overlap with confirmed event profile wording. This is a text relevance recommendation; visual quality is not measured.")
                        selected_moment = st.selectbox(
                            "Recommended moment for a future reel · choose a timestamp",
                            options=range(len(moments)),
                            format_func=lambda i: f"{moments[i]['timestamp']} · {moments[i]['description'][:90]}",
                            key=f"video_moment_{doc['filename']}")
                        moment = moments[selected_moment]
                        st.caption("Why it ranks here: " + moment["reason"] + ". Preview shows six seconds from this timestamp; it does not modify the original video.")
                        if st.button("Preview this 6-second moment", key=f"preview_video_moment_{doc['filename']}"):
                            try:
                                preview_path = extract_video_preview(file_path, moment["start_seconds"],
                                                                    output_dir=DATA_DIR / "video_previews")
                                st.video(str(preview_path))
                            except Exception as exc:
                                st.error(f"Could not create this video moment preview: {exc}")
            elif cached_media.get("sha256") == doc["sha256"] and (cached_media.get("status") == "deferred" or (cached_media.get("status") == "failed" and is_daily_quota_error(cached_media.get("error", "")))):
                st.warning(f"Media analysis deferred to avoid repeat quota requests · {cached_media.get('error') or 'Provider quota limit reached.'}")
                if cached_media.get("retry_after"):
                    st.caption(f"Provider retry time: {cached_media['retry_after']} UTC")
            elif cached_media.get("sha256") == doc["sha256"] and cached_media.get("status") == "failed":
                st.error(f"Media analysis failed · {cached_media.get('error') or 'Unknown provider error'}")
    media_docs = [d for d in docs if Path(d["filename"]).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".pdf"}]
    if media_docs:
        st.subheader("Understand event photos and videos")
        st.caption("With your consent, selected images and rendered scanned-PDF pages are analyzed by the configured vision provider. Groq video analysis samples frames locally with FFmpeg and transcribes available audio with Groq Whisper; original local files are kept unchanged.")
        selected_names = st.multiselect("Select up to 20 files for this analysis run",
            options=[d["filename"] for d in media_docs],
            default=[d["filename"] for d in media_docs[:20]],
            key=f"media_analysis_selection_{st.session_state.get('user_id', 'anonymous')}")
        analysis_docs = [d for d in media_docs if d["filename"] in selected_names]
        if len(analysis_docs) > 20:
            st.warning("Select at most 20 files per run to keep analysis bounded.")
            analysis_docs = analysis_docs[:20]
        media_consent = st.checkbox("I approve sending the selected event media to the configured AI service for analysis.", key="media_ai_consent")
        media_provider_supported = AI_PROVIDER in {"gemini", "groq"}
        if not media_provider_supported:
            st.info("Event media analysis currently requires Groq or Gemini.")
        elif not GEN_KEY:
            secret_name = "GROQ_API_KEY" if AI_PROVIDER == "groq" else "GEMINI_API_KEY"
            st.warning(f"{AI_PROVIDER.title()} media analysis is unavailable until the administrator configures {secret_name} in server secrets.")
        cache = load_media_analysis()
        quota_deferred = [d for d in analysis_docs
                          if cache.get(d["filename"], {}).get("sha256") == d["sha256"]
                          and (cache.get(d["filename"], {}).get("status") == "deferred"
                               or (cache.get(d["filename"], {}).get("status") == "failed"
                                   and is_daily_quota_error(cache.get(d["filename"], {}).get("error", ""))))]
        if quota_deferred:
            retry_values = sorted({cache[d["filename"]].get("retry_after", "") for d in quota_deferred
                                   if cache[d["filename"]].get("retry_after")})
            retry_note = f" Provider retry time: {retry_values[0]} UTC." if retry_values else " No retry time was supplied; use the explicit retry button only after confirming quota is available."
            st.warning(f"{len(quota_deferred)} selected media file(s) are deferred after an AI provider rate or quota limit.{retry_note}")
        can_analyze = bool(media_consent and analysis_docs and media_provider_supported and GEN_KEY)
        analyze_clicked = st.button(f"Analyze event media with {AI_PROVIDER.title()}", type="secondary",
                                    disabled=not can_analyze, key="analyze_event_media")
        retry_clicked = False
        if quota_deferred:
            retry_clicked = st.button("Retry quota-deferred media (only after quota is available)",
                                      disabled=not can_analyze, key="retry_deferred_media")
        if analyze_clicked or retry_clicked:
            work_docs, skipped_deferred, cached_results = filter_media_batch_items(
                analysis_docs, cache, force_retry=retry_clicked)
            for name in skipped_deferred:
                record = cache[name]
                if record.get("status") != "deferred":
                    matching_doc = next(d for d in analysis_docs if d["filename"] == name)
                    save_media_analysis(name, matching_doc["sha256"], "deferred", text="",
                        error=record.get("error", "Provider daily quota limit previously reached."),
                        model=record.get("model", MEDIA_MODEL), retry_after=record.get("retry_after", ""))
                    cache[name] = {**record, "status": "deferred"}
            if skipped_deferred:
                st.info("Skipped quota-deferred media to avoid another request: " + ", ".join(skipped_deferred))
            if not work_docs:
                if cached_results and not skipped_deferred:
                    st.info("All selected media files already have current results.")
            else:
                progress = st.progress(0, text="Preparing media analysis…")

                def analyze_one_media(doc: dict) -> str | MediaAnalysisSkip:
                    name = doc["filename"]
                    suffix = Path(name).suffix.lower()
                    path = UPLOAD_DIR / doc["stored_filename"]
                    cached = cache.get(name, {})
                    save_media_analysis(name, doc["sha256"], "processing", text="",
                                        observations=cached.get("observations", []), timestamps=cached.get("timestamps", []),
                                        model=MEDIA_MODEL)
                    def update_media_status(message: str) -> None:
                        progress.progress(0, text=f"{name} · {message}")
                    if suffix in {".mp4", ".mov"}:
                        return analyze_video(path, name, api_key=GEN_KEY, model=MEDIA_MODEL,
                                             status_callback=update_media_status, endpoint=GEN_BASE,
                                             provider=AI_PROVIDER, audio_model=MEDIA_AUDIO_MODEL)
                    if suffix == ".pdf":
                        page_images = render_scanned_pdf_pages(name, path.read_bytes(), max_pages=3)
                        if not page_images:
                            return MediaAnalysisSkip("No scanned or text-empty pages were found; document text extraction is the applicable path.")
                        descriptions = []
                        for page in page_images:
                            page_ref = f"{name} · page {page['page_number']}"
                            ocr_text = page.get("ocr_text", "").strip()
                            if ocr_text:
                                save_document_extraction(name, doc["sha256"], page["page_number"], "ocr", "complete", text=ocr_text)
                            elif page.get("ocr_error"):
                                save_document_extraction(name, doc["sha256"], page["page_number"], "ocr", "failed", error=page["ocr_error"])
                                st.warning(f"{page_ref}: OCR unavailable ({page['ocr_error']}). Continuing with {AI_PROVIDER.title()} vision.")
                            if len(ocr_text) >= 20:
                                continue
                            update_media_status(f"Analyzing page {page['page_number']} with {AI_PROVIDER.title()} vision…")
                            description = analyze_image(page["content"], page_ref + ".jpg", api_key=GEN_KEY,
                                                        model=MEDIA_MODEL, endpoint=GEN_BASE, provider=AI_PROVIDER)
                            descriptions.append(f"Source: {page_ref}\n{description}")
                            save_document_extraction(name, doc["sha256"], page["page_number"], f"{AI_PROVIDER}_vision", "complete", text=description)
                        if not descriptions:
                            return MediaAnalysisSkip("Scanned-page OCR found readable text, so vision analysis was not needed for the sampled pages.")
                        return "\n\n".join(descriptions)
                    update_media_status(f"Analyzing image with {AI_PROVIDER.title()} vision…")
                    return analyze_image(path.read_bytes(), name, api_key=GEN_KEY,
                                         model=MEDIA_MODEL, endpoint=GEN_BASE, provider=AI_PROVIDER)

                outcomes = run_media_batch(work_docs, analyze_one_media)
                quota_outcomes = [result for result in outcomes if result.status == "deferred"]
                completed = 0
                for index, result in enumerate(outcomes):
                    doc = result.item
                    name = doc["filename"]
                    if result.status == "complete":
                        description = str(result.value or "")
                        times = sorted(set(re.findall(r"(?<!\d)(?:\d{1,2}:)?\d{1,2}:\d{2}(?!\d)", description)))
                        observations = [line.strip(" -*•") for line in description.splitlines() if line.strip()][:30]
                        save_media_analysis(name, doc["sha256"], "complete", text=description,
                                            observations=observations, timestamps=times, model=MEDIA_MODEL)
                        cache[name] = {"sha256": doc["sha256"], "status": "complete", "text": description,
                                       "observations": observations, "timestamps": times, "model": MEDIA_MODEL}
                        completed += 1
                        st.success(f"{name}: media analysis completed and saved.")
                    elif result.status == "not_needed":
                        save_media_analysis(name, doc["sha256"], "not_needed", model=MEDIA_MODEL)
                        cache[name] = {"sha256": doc["sha256"], "status": "not_needed", "text": "", "model": MEDIA_MODEL}
                        st.info(f"{name}: {result.error}")
                    elif result.status == "deferred":
                        save_media_analysis(name, doc["sha256"], "deferred", text="", error=result.error,
                                            model=MEDIA_MODEL, retry_after=result.retry_after)
                        cache[name] = {"sha256": doc["sha256"], "status": "deferred", "text": "",
                                       "error": result.error, "model": MEDIA_MODEL, "retry_after": result.retry_after}
                    else:
                        save_media_analysis(name, doc["sha256"], "failed", text="", error=result.error, model=MEDIA_MODEL)
                        cache[name] = {"sha256": doc["sha256"], "status": "failed", "text": "", "error": result.error,
                                       "model": MEDIA_MODEL}
                        st.error(f"{name}: media analysis failed: {result.error}")
                    progress.progress((index + 1) / len(outcomes), text=f"Processed {index + 1} of {len(outcomes)} selected files")
                if quota_outcomes:
                    quota_error = quota_outcomes[0].error
                    st.error(f"{AI_PROVIDER.title()} rate limit or quota reached. The media batch stopped; {len(quota_outcomes)} file(s) were deferred. Provider response: {quota_error}")
                    st.info("Quota-deferred files will not be sent again automatically. Retry them after quota becomes available, using the explicit retry button.")
                if completed:
                    st.success(f"Analysis completed for {completed} file(s). Results are saved independently of document text extraction.")
    can_prepare = bool(docs and EMBED_KEY)
    if st.button("Prepare documents for search", type="primary", disabled=not can_prepare):
        with st.spinner("Reading your files and preparing them for search…"):
            extracted, problems, hashes = [], [], {}
            for filename, content, digest in load_documents():
                hashes[filename] = digest
                try:
                    if Path(filename).suffix.lower() in {".mp4", ".mov"}:
                        analyzed = load_media_analysis().get(filename, {})
                        if analyzed.get("sha256") != digest or analyzed.get("status") != "complete":
                            problems.append(f"{filename}: video is not included in document-text RAG. Analyze it separately with {AI_PROVIDER.title()} if you want media context in a campaign.")
                        continue
                    all_units = extract_document(filename, content)
                    for unit in all_units:
                        readable = bool(unit["text"].strip())
                        save_document_extraction(filename, digest, unit.get("page_number"), "text",
                            "complete" if readable else "empty", text=unit.get("text", ""),
                            error="" if readable else (unit.get("error") or "No readable text on this page."))
                    units = [unit for unit in all_units if unit["text"].strip()]
                    if not units:
                        ocr_detail = next((unit.get("error") for unit in all_units if unit.get("error")), "")
                        detail = f" OCR status: {ocr_detail}" if ocr_detail else ""
                        problems.append(f"{filename}: no readable text was extracted.{detail} This file will not be indexed as document text; event details and media analysis can still be used.")
                    for unit in units:
                        extracted.append({"source_filename": filename, "source_hash": digest,
                                          "page_number": unit["page_number"], "text": unit["text"]})
                except Exception as exc:
                    problems.append(f"{filename}: {exc}")
                    save_document_extraction(filename, digest, None, "text", "failed", error=str(exc))
            chunks = prepare_chunks(extracted)
            try:
                if not chunks:
                    st.info("No readable document text is available for RAG indexing. Event media is stored separately, and you can continue using the confirmed event profile and analyzed media without RAG.")
                    chunks = None
                if chunks is None:
                    pass
                else:
                    embedding_inputs = [embedding_request_text(
                        row["text"], provider=EMBED_PROVIDER, role="document",
                        title=row["source_filename"], model=EMBED_MODEL) for row in chunks]
                    vectors = embed_texts(embedding_inputs, api_key=EMBED_KEY, base_url=EMBED_BASE,
                                          model=EMBED_MODEL, provider=EMBED_PROVIDER)
                    for row, vector in zip(chunks, vectors):
                        row["vector"] = vector
                    save_chunks(chunks)
                    expected_format = "gemini-embedding-2-search-v1" if EMBED_PROVIDER == "gemini" and EMBED_MODEL == "gemini-embedding-2" else "plain-v1"
                    indexed_filenames = sorted({r["source_filename"] for r in chunks})
                    index = {"source_hashes": {name: hashes[name] for name in indexed_filenames}, "indexed_filenames": indexed_filenames,
                        "chunk_count": len(chunks), "embedding_model": EMBED_MODEL, "vector_dimensions": len(vectors[0]),
                        "embedding_provider": EMBED_PROVIDER, "embedding_base_url": EMBED_BASE.rstrip("/"),
                        "embedding_format": expected_format, "source_policy": "document-text-v1",
                        "indexed_at": datetime.now().isoformat(timespec="minutes")}
                    save_state("knowledge_index", index)
                    index_current = embedding_index_matches(index, provider=EMBED_PROVIDER, base_url=EMBED_BASE,
                                                            model=EMBED_MODEL, source_hashes=current_hashes)
                    st.success(f"Prepared {len(chunks)} passages from {len(index['indexed_filenames'])} readable source(s).")
            except Exception as exc:
                st.error(f"Document preparation failed: {exc}")
            for problem in problems:
                st.warning(problem)
    chunks = load_chunks()
    if chunks and index.get("embedding_model") and index_current:
        st.success(f"Ready to search · {len(chunks)} passages · {index.get('embedding_model')} · updated {index.get('indexed_at', 'unknown')}")
        pending = [d["filename"] for d in docs if index.get("source_hashes", {}).get(d["filename"]) != d["sha256"]]
        if pending:
            st.info(f"{len(pending)} new or changed file(s) are not in this RAG index yet. Media analysis remains available independently.")
        if st.button("Continue to Campaign Creation", key="knowledge_to_campaign"):
            go("Campaign Creation")
        if st.button("Search event documents", key="kb_search"):
            go("RAG Retrieval")
    elif chunks and index.get("embedding_model"):
        st.warning("The saved index uses a different provider, endpoint, model, format, or document version. Re-prepare the documents before searching.")
        if st.button("Continue to Campaign Creation", key="knowledge_to_campaign_stale_index"):
            go("Campaign Creation")
    elif docs and not EMBED_KEY:
        st.warning("Indexing is blocked because no server-side document-embedding credential is configured. The administrator can set it through environment variables or Streamlit secrets. The button requires this credential and at least one saved file.")
        if st.button("Continue to Campaign Creation", key="knowledge_to_campaign_no_rag"):
            go("Campaign Creation")
    elif docs:
        st.info("Files are saved locally. Document text extraction, media analysis, and RAG indexing are separate steps; you can continue to Campaign Creation without an index.")
        if st.button("Continue to Campaign Creation", key="knowledge_to_campaign_no_index"):
            go("Campaign Creation")
    st.caption(f"Text extraction, {AI_PROVIDER.title()} media analysis, and semantic RAG indexing are independent. A file with no readable text does not block media analysis or campaign generation.")


def show_retrieval_page() -> None:
    hero("RAG Retrieval", "Search your event documents and review the passages that can support a post.")
    chunks = load_chunks()
    index = load_state("knowledge_index", {})
    docs = list_documents()
    hashes = {d["filename"]: d["sha256"] for d in docs}
    if not chunks or not embedding_index_matches(index, provider=EMBED_PROVIDER, base_url=EMBED_BASE,
                                                 model=EMBED_MODEL, source_hashes=hashes):
        st.info("Prepare the current documents with the selected embedding provider and model before searching.")
        if st.button("Open Knowledge Base", type="primary", key="retrieval_open_kb"):
            go("Knowledge Base")
        return
    if not EMBED_KEY:
        st.warning("Semantic search needs a server-configured embedding credential. Ask the administrator to configure it once.")
        if st.button("View administrator setup", key="configure_retrieval"):
            go("Settings")
        return
    query = st.text_input("What information are you looking for?", placeholder="For example, what activities are planned for young people?")
    if st.button("Search documents", type="primary", disabled=not bool(query.strip()), key="run_retrieval"):
        try:
            with st.spinner("Searching your event documents…"):
                query_input = embedding_request_text(query, provider=EMBED_PROVIDER, role="query",
                                                     model=index["embedding_model"])
                vector = embed_texts([query_input], api_key=EMBED_KEY, base_url=EMBED_BASE,
                                     model=index["embedding_model"], provider=EMBED_PROVIDER)[0]
                if len(vector) != index.get("vector_dimensions"):
                    raise ProviderError("The query embedding dimensions do not match the saved index. Re-prepare the documents.")
                results = retrieve(query, chunks, vector, top_k=5)
            st.session_state["last_retrieval"] = results
            st.success(f"Found {len(results)} relevant passages.")
        except Exception as exc:
            st.error(f"Search could not be completed: {exc}")
    for i, item in enumerate(st.session_state.get("last_retrieval", []), 1):
        page_label = f" · page {item['page_number']}" if item.get("page_number") else ""
        with st.container(border=True):
            st.markdown(f"**{i}. {item.get('source_filename', 'Source document')}{page_label}** · relevance {item.get('score', 0):.3f}")
            st.write(item["text"])


def _post_package_from_campaign(campaign: dict, caption: str) -> dict:
    try:
        package = json.loads(campaign.get("post_package") or "{}")
    except (TypeError, ValueError):
        package = {}
    return {**package, "campaign_id": campaign["id"], "platform": campaign["platform"],
            "caption": caption, "hashtags": package.get("hashtags", extract_hashtags(caption)),
            "cta": package.get("cta", ""),
            "caption_adaptation": package.get("caption_adaptation", caption),
            "post_type": package.get("post_type", "Single Image Post"),
            "aspect": package.get("aspect", "Square (1:1)"),
            "slides": package.get("slides", [])}


def _edit_package_selection(selection_key: str, action: str, index: int,
                           replacement_widget_key: str = "", focus_key: str = "") -> None:
    """Widget callback runs before rerender, so updating the multiselect is safe."""
    replacement = st.session_state.get(replacement_widget_key, "") if replacement_widget_key else ""
    selected = edit_slide_order(st.session_state.get(selection_key, []), action, index, replacement)
    st.session_state[selection_key] = selected
    if focus_key and selected:
        if action == "replace" and replacement in selected:
            st.session_state[focus_key] = replacement
        elif action == "remove" and st.session_state.get(focus_key) not in selected:
            st.session_state[focus_key] = selected[min(index, len(selected) - 1)]


def _set_package_focus(focus_key: str, photo_key: str) -> None:
    st.session_state[focus_key] = photo_key


def _apply_recommended_package_selection(selection_key: str, type_key: str,
                                         recommended: list[str], recommended_type: str) -> None:
    st.session_state[selection_key] = list(recommended)
    st.session_state[type_key] = recommended_type


def show_post_package_editor(campaign: dict, caption: str) -> None:
    """Compose, persist and download a single-image or carousel publishing package."""
    cid = campaign["id"]
    media = visual_photo_sources()
    if not media:
        st.info("Upload event photographs or extract them from a PDF in the Knowledge Base to create a visual post.")
        return
    by_key = {item["photo_key"]: item for item in media}
    package = _post_package_from_campaign(campaign, caption)
    prefix = f"package_{cid}"
    saved_slides = sorted(package["slides"], key=lambda item: item.get("position", 0))
    saved_keys = [slide.get("photo_key") for slide in saved_slides if slide.get("photo_key") in by_key]
    brief = json.loads(campaign.get("brief") or "{}")
    event = event_details()
    objective = " ".join(str(value or "") for value in (
        event.get("name"), event.get("description"), event.get("theme"),
        brief.get("objective"), brief.get("campaign_focus")))
    media_cache = load_media_analysis()
    inventory = [(item["photo_key"], item.get("sha256", ""), item.get("_path", "")) for item in media]
    analysis_signatures = [(name, value.get("sha256", ""),
                            hashlib.sha1(str(value.get("text", "")).encode("utf-8")).hexdigest())
                           for name, value in sorted(media_cache.items())
                           if value.get("status") == "complete"]
    rank_signature = hashlib.sha256(json.dumps([inventory, analysis_signatures, objective],
                                               ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    ranking_key = f"{prefix}_ranking_{rank_signature}"
    if ranking_key not in st.session_state:
        ranking_progress = st.progress(0, text="Locally assessing photo quality and duplicates…")
        try:
            ranked = rank_photos(media, objective, media_cache,
                                 status_callback=lambda done, total, name: ranking_progress.progress(
                                     done / max(total, 1), text=f"Assessing {done} of {total}: {name}"))
            st.session_state[ranking_key] = [{"key": item["photo_key"], "score": item["score"],
                "quality": item["quality"], "duplicate": item["duplicate"],
                "reason": item["reason"], "analysis": item["analysis"]} for item in ranked]
        finally:
            ranking_progress.empty()
    ranking = st.session_state[ranking_key]
    ranked_by_key = {item["key"]: item for item in ranking}
    recommended_keys = [item["key"] for item in select_recommended_photos(
        [{**item, "filename": item["key"]} for item in ranking], max_slides=8)]
    saved_package_exists = bool(saved_slides or campaign.get("creative_path"))
    recommended_type = "Multiple Image / Carousel Post" if len(recommended_keys) >= 2 else "Single Image Post"
    selected_key = f"{prefix}_selection"
    if selected_key not in st.session_state:
        legacy_keys = [key for key in json.loads(campaign.get("selected_media") or "[]") if key in by_key]
        selected = saved_keys or (legacy_keys if saved_package_exists else recommended_keys)
        st.session_state[selected_key] = selected[:1] if package["post_type"] == "Single Image Post" and saved_package_exists else selected
    type_key = f"{prefix}_type"
    if type_key not in st.session_state:
        st.session_state[type_key] = package["post_type"] if saved_package_exists else recommended_type

    st.markdown(f"#### {campaign['platform']} visual post package")
    post_type = st.selectbox("Content type", ["Single Image Post", "Multiple Image / Carousel Post"], key=type_key)
    st.caption(f"Automatic recommendation: {recommended_type} · {len(recommended_keys)} of {len(media)} candidate photos. Ranking uses image resolution, sharpness, exposure, near-duplicate detection, event wording and saved AI vision descriptions when available. No AI request is made by this local ranking step.")
    for number, key in enumerate(recommended_keys, 1):
        detail = ranked_by_key.get(key, {})
        st.caption(f"Recommended {number}: {by_key[key]['display_name']} · {detail.get('reason') or 'highest available local quality score'}")
    consent_key = f"{prefix}_vision_consent"
    if consent_key not in st.session_state:
        st.session_state[consent_key] = False
    vision_consent = st.checkbox(f"I consent to send candidate photographs to {AI_PROVIDER.title()} for event-scene analysis.",
                                 key=consent_key)
    vision_supported = AI_PROVIDER in {"groq", "gemini"} and bool(GEN_KEY)
    if not vision_supported:
        st.caption("Semantic photo analysis is unavailable without a configured vision provider. Local quality and duplicate ranking still work.")
    analysis_cache = load_media_analysis()
    analysis_candidates = []
    for row in ranking:
        item = by_key[row["key"]]
        record = analysis_cache.get(item["filename"], {})
        if record.get("sha256") == item.get("sha256") and record.get("status") == "complete" and record.get("text"):
            continue
        analysis_candidates.append({"filename": item["filename"], "sha256": item["sha256"],
                                    "_path": item["_path"], "display_name": item["display_name"]})
    if analysis_candidates:
        st.caption(f"{len(analysis_candidates)} candidate photo(s) lack current vision descriptions. Analyze the top 20 ranked candidates (one request per image); quota failures stop the batch and are saved to avoid repeat calls.")
    if st.button("Analyze top photo candidates with AI", key=f"{prefix}_analyze_photos",
                 disabled=not (vision_consent and vision_supported and bool(analysis_candidates))):
        cache = load_media_analysis()
        work, deferred, cached = filter_media_batch_items(analysis_candidates[:20], cache)
        if deferred:
            st.warning("Skipped previously quota-deferred photos: " + ", ".join(deferred))
        if not work:
            st.info("No new photo analysis requests were sent. Existing results or quota deferrals cover the selected candidates.")
        else:
            progress = st.progress(0, text="Analyzing event photographs…")
            def analyze_candidate(item: dict) -> str:
                index_now = work.index(item) + 1
                progress.progress((index_now - 1) / len(work), text=f"Analyzing {index_now} of {len(work)}: {item['display_name']}")
                return analyze_image(Path(item["_path"]).read_bytes(), item["display_name"], api_key=GEN_KEY,
                                     model=MEDIA_MODEL, endpoint=GEN_BASE, provider=AI_PROVIDER)
            outcomes = run_media_batch(work, analyze_candidate)
            for outcome in outcomes:
                item = outcome.item
                if outcome.status == "complete":
                    save_media_analysis(item["filename"], item["sha256"], "complete", text=str(outcome.value), model=MEDIA_MODEL)
                    st.success(f"Vision analysis saved for {item['display_name']}.")
                elif outcome.status == "deferred":
                    save_media_analysis(item["filename"], item["sha256"], "deferred", error=outcome.error,
                                        model=MEDIA_MODEL, retry_after=outcome.retry_after)
                    st.warning(f"Analysis deferred for {item['display_name']}: {outcome.error}")
                else:
                    save_media_analysis(item["filename"], item["sha256"], "failed", error=outcome.error, model=MEDIA_MODEL)
                    st.error(f"Analysis failed for {item['display_name']}: {outcome.error}")
            progress.progress(1, text="Photo analysis batch complete")
            st.rerun()
    st.button("Use automatic photo recommendation", key=f"{prefix}_apply_recommendation",
              on_click=_apply_recommended_package_selection,
              args=(selected_key, type_key, recommended_keys, recommended_type))
    aspects = ["Square (1:1)", "Portrait (4:5)", "Landscape (1.91:1)"]
    aspect_key = f"{prefix}_aspect"
    if aspect_key not in st.session_state:
        st.session_state[aspect_key] = package["aspect"] if package["aspect"] in aspects else "Square (1:1)"
    aspect = st.selectbox("Slide aspect ratio", aspects, key=aspect_key,
                          help="Square and portrait suit Instagram; landscape is also available for Facebook and LinkedIn.")
    if post_type == "Single Image Post" and len(st.session_state.get(selected_key, [])) > 1:
        st.session_state[selected_key] = st.session_state[selected_key][:1]

    upload_key = f"{prefix}_uploads"
    uploads = st.file_uploader("Add photographs manually", type=["jpg", "jpeg", "png", "webp"],
                               accept_multiple_files=True, key=upload_key)
    if uploads and st.button("Add uploaded photographs to Knowledge Base gallery", key=f"{prefix}_add_uploads"):
        existing = {doc["filename"] for doc in list_documents()}
        existing_hashes = {doc["sha256"] for doc in list_documents()}
        added = 0
        for uploaded in uploads:
            raw = uploaded.getvalue()
            digest = hashlib.sha256(raw).hexdigest()
            if digest in existing_hashes:
                continue
            suffix = Path(uploaded.name).suffix.lower() or ".jpg"
            name = f"campaign-{cid}-{uuid.uuid4().hex[:10]}{suffix}"
            if name not in existing:
                save_uploaded_document(name, raw, digest)
                added += 1
                existing_hashes.add(digest)
        st.session_state.pop(selected_key, None)
        st.success(f"Added {added} photograph(s) to the local gallery.")
        st.rerun()

    options = list(by_key)
    labels = {key: by_key[key]["display_name"] for key in options}
    pending_order = st.session_state.pop(f"{prefix}_pending_order", None)
    if pending_order and all(key in options for key in pending_order):
        st.session_state[selected_key] = list(pending_order)
    selected = st.multiselect("Select photographs in carousel order", options, key=selected_key,
                              format_func=lambda key: labels.get(key, key))
    if len(selected) > 12:
        st.warning("A post package is limited to 12 images for responsive preview and export. Remove some images or use the automatic recommendation.")
        return
    if post_type == "Multiple Image / Carousel Post" and len(selected) < 2:
        st.caption("Choose at least two photographs for a carousel. The chooser preserves selection order; use the controls below to adjust it.")
    if not selected:
        return
    sequence_reasons = st.session_state.get(f"{prefix}_selection_reasons", {})
    if sequence_reasons:
        st.caption("AI sequence reasoning: " + " · ".join(
            f"{index + 1}. {sequence_reasons[key]}" for index, key in enumerate(selected)
            if sequence_reasons.get(key)))

    hashtags_key, cta_key, adaptation_key = (f"{prefix}_hashtags", f"{prefix}_cta", f"{prefix}_adaptation")
    platform_keys = {platform_name: f"{prefix}_caption_{platform_name.casefold()}"
                     for platform_name in ("Instagram", "Facebook", "LinkedIn")}
    pending_copy = st.session_state.pop(f"{prefix}_pending_copy", None)
    if pending_copy:
        if pending_order:
            st.session_state[f"{prefix}_selection_reasons"] = dict(zip(
                pending_order, pending_copy.get("selection_reasons", [])))
        st.session_state[hashtags_key] = " ".join("#" + str(tag).lstrip("#") for tag in pending_copy.get("hashtags", []))
        st.session_state[cta_key] = pending_copy.get("call_to_action", "")
        captions = pending_copy.get("platform_captions", {})
        for platform_name, widget_key in platform_keys.items():
            st.session_state[widget_key] = str(captions.get(platform_name, ""))
        st.session_state[adaptation_key] = str(captions.get(campaign["platform"], caption))
        for photo_key, overlay_text in zip(selected, pending_copy.get("slide_texts", [])):
            slide_widget_key = f"{prefix}_slide_text_{hashlib.sha1(photo_key.encode()).hexdigest()[:12]}"
            st.session_state[slide_widget_key] = str(overlay_text)
    if hashtags_key not in st.session_state:
        st.session_state[hashtags_key] = " ".join(package.get("hashtags") or extract_hashtags(caption))
    if cta_key not in st.session_state:
        st.session_state[cta_key] = package.get("cta", "")
    if adaptation_key not in st.session_state:
        st.session_state[adaptation_key] = package.get("caption_adaptation") or caption
    saved_captions = package.get("platform_captions", {})
    if platform_keys[campaign["platform"]] not in st.session_state:
        st.session_state[platform_keys[campaign["platform"]]] = str(saved_captions.get(campaign["platform"], caption))
    with st.expander("Platform-specific captions", expanded=bool(saved_captions)):
        for platform_name, widget_key in platform_keys.items():
            if widget_key not in st.session_state:
                st.session_state[widget_key] = str(saved_captions.get(platform_name, ""))
            st.text_area(f"{platform_name} caption", key=widget_key, height=90)
    if st.button("Generate complete post copy with AI", key=f"{prefix}_generate_copy", disabled=not bool(GEN_KEY)):
        scene_notes = []
        for photo_key in selected:
            note = media_cache.get(by_key[photo_key]["filename"], {})
            if note.get("sha256") == by_key[photo_key].get("sha256") and note.get("text"):
                scene_notes.append({"image": by_key[photo_key]["display_name"], "description": note["text"][:1000]})
        generation_brief = {"output_mode": "post_package", "platform": campaign["platform"],
            "event": event, "campaign_focus": caption, "selected_media": [by_key[k]["display_name"] for k in selected],
            "media_descriptions": scene_notes, "slide_count": len(selected)}
        try:
            with st.spinner("Generating platform captions, hashtags, CTA and slide text…"):
                result_text = generate(generation_brief, json.loads(campaign.get("evidence") or "[]"),
                    api_key=GEN_KEY, base_url=GEN_BASE, model=GEN_MODEL, provider=AI_PROVIDER)
            result = json.loads(result_text)
            if not isinstance(result, dict) or not isinstance(result.get("platform_captions"), dict):
                raise ValueError("The AI response did not contain the requested structured package copy.")
            if not isinstance(result.get("slide_texts"), list):
                result["slide_texts"] = []
            result["slide_texts"] = (result["slide_texts"] + [""] * len(selected))[:len(selected)]
            order = result.get("slide_order", [])
            if (isinstance(order, list) and len(order) == len(selected)
                    and sorted(order) == list(range(1, len(selected) + 1))):
                result["slide_order"] = [selected[position - 1] for position in order]
            else:
                result["slide_order"] = list(selected)
            reasons = result.get("selection_reasons", [])
            if not isinstance(reasons, list):
                reasons = []
            result["selection_reasons"] = (reasons + [""] * len(selected))[:len(selected)]
            ordered_overlays = result["slide_texts"]
            original_order = list(selected)
            result["slide_texts"] = [ordered_overlays[original_order.index(key)]
                                      for key in result["slide_order"]]
            result["selection_reasons"] = [result["selection_reasons"][original_order.index(key)]
                                            for key in result["slide_order"]]
            st.session_state[f"{prefix}_pending_order"] = result["slide_order"]
            st.session_state[f"{prefix}_pending_copy"] = result
            st.rerun()
        except Exception as exc:
            st.error(f"Could not generate package copy: {exc}. Your existing caption and manually editable fields remain available.")
    hashtags_text = st.text_input("Relevant hashtags", key=hashtags_key, help="Separate hashtags with spaces.")
    cta = st.text_input("Optional call to action", key=cta_key)
    adaptation = st.text_area(f"Caption adaptation for {campaign['platform']}", key=adaptation_key, height=110)
    platform_captions = {platform_name: st.session_state.get(widget_key, "")
                         for platform_name, widget_key in platform_keys.items()}
    platform_captions[campaign["platform"]] = adaptation

    prior_by_key = {slide.get("photo_key"): slide for slide in saved_slides}
    focus_key = f"{prefix}_focus_slide"
    if focus_key not in st.session_state or st.session_state[focus_key] not in selected:
        st.session_state[focus_key] = selected[0]
    st.markdown("##### Carousel preview · publishing order")
    for row_start in range(0, len(selected), 4):
        thumb_columns = st.columns(4)
        for offset, key in enumerate(selected[row_start:row_start + 4]):
            index = row_start + offset
            photo = by_key[key]
            with thumb_columns[offset]:
                st.image(photo["_path"], caption=f"Slide {index + 1} · {photo['display_name']}",
                         use_container_width=True)
                st.button(f"Edit slide {index + 1}", key=f"{prefix}_focus_{index}",
                          type="primary" if st.session_state[focus_key] == key else "secondary",
                          on_click=_set_package_focus, args=(focus_key, key), use_container_width=True)

    key = st.session_state[focus_key]
    index = selected.index(key)
    photo = by_key[key]
    previous = prior_by_key.get(key, {})
    text_key = f"{prefix}_slide_text_{hashlib.sha1(key.encode()).hexdigest()[:12]}"
    pending_text_key = f"{text_key}_pending"
    if pending_text_key in st.session_state:
        st.session_state[text_key] = st.session_state.pop(pending_text_key)
    if text_key not in st.session_state:
        st.session_state[text_key] = previous.get("slide_text", "")
    preview_col, edit_col = st.columns([1.15, 1])
    with preview_col:
        st.markdown(f"**Slide {index + 1} of {len(selected)} · {photo['display_name']}**")
        existing = previous.get("creative_path")
        st.image(existing if existing and Path(existing).is_file() else photo["_path"],
                 caption="Current designed slide" if existing and Path(existing).is_file() else "Selected original photo",
                 use_container_width=True)
    with edit_col:
        slide_text = st.text_input("Slide-specific text", key=text_key,
                                   placeholder="Short optional overlay; generated suggestion appears here")
        movement = st.columns(3)
        movement[0].button("Move up", key=f"{prefix}_up", disabled=index == 0,
            on_click=_edit_package_selection, args=(selected_key, "up", index, "", focus_key))
        movement[1].button("Move down", key=f"{prefix}_down", disabled=index == len(selected)-1,
            on_click=_edit_package_selection, args=(selected_key, "down", index, "", focus_key))
        movement[2].button("Remove slide", key=f"{prefix}_remove",
            on_click=_edit_package_selection, args=(selected_key, "remove", index, "", focus_key))
        replacement_key = f"{prefix}_replace_choice_{hashlib.sha1(key.encode()).hexdigest()[:12]}"
        if replacement_key not in st.session_state or st.session_state[replacement_key] not in options:
            st.session_state[replacement_key] = key
        replacement = st.selectbox("Replace this photograph", options, key=replacement_key,
                                   format_func=lambda item: labels.get(item, item))
        if replacement != key:
            st.button("Apply replacement", key=f"{prefix}_replace",
                on_click=_edit_package_selection,
                args=(selected_key, "replace", index, replacement_key, focus_key))
        ai_col, local_col = st.columns(2)
        if ai_col.button("Regenerate this slide text with AI", key=f"{prefix}_regen", disabled=not bool(GEN_KEY)):
            brief = json.loads(campaign.get("brief") or "{}")
            brief.update({"output_mode": "slide_overlay", "platform": campaign["platform"],
                          "content_type": "short carousel slide overlay", "campaign_focus": caption,
                          "selected_media": [key], "slide_position": index + 1, "event": event})
            try:
                with st.spinner(f"Generating text for slide {index + 1}…"):
                    generated = generate(brief, [], api_key=GEN_KEY, base_url=GEN_BASE,
                                         model=GEN_MODEL, provider=AI_PROVIDER)
                    generated_overlay = generated.strip().strip('"')
                    st.session_state[pending_text_key] = generated_overlay
                    image_path = compose_post_creative(generated_overlay, key, aspect, cid, slide_index=index)
                    generated_slide = {"position": index, "photo_key": key, "display_name": photo["display_name"],
                        "source_type": photo["source_type"], "source_filename": photo.get("source_filename", photo["filename"]),
                        "page_number": photo.get("page_number"), "slide_text": generated_overlay,
                        "selection_reason": sequence_reasons.get(key, ""), "creative_path": str(image_path),
                        "original_path": photo["_path"]}
                    slides = []
                    for position, selected_photo_key in enumerate(selected):
                        existing_slide = generated_slide if selected_photo_key == key else dict(prior_by_key.get(selected_photo_key, {}))
                        if existing_slide:
                            existing_slide["position"] = position
                            slides.append(existing_slide)
                    update_campaign(cid, caption, campaign["status"])
                    save_post_package(cid, dict(package, post_type=post_type, platform=campaign["platform"], aspect=aspect,
                        caption=caption, hashtags=[tag for tag in hashtags_text.split() if tag], cta=cta,
                        caption_adaptation=adaptation, platform_captions=platform_captions, slides=slides))
                st.success(f"Slide {index + 1} text and image updated.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not regenerate slide {index + 1}: {exc}")
        if local_col.button("Update this slide only", key=f"{prefix}_update"):
            try:
                image_path = compose_post_creative(slide_text, key, aspect, cid, slide_index=index)
                slide = {"position": index, "photo_key": key, "display_name": photo["display_name"],
                         "source_type": photo["source_type"], "source_filename": photo.get("source_filename", photo["filename"]),
                         "page_number": photo.get("page_number"), "slide_text": slide_text,
                         "selection_reason": sequence_reasons.get(key, ""), "creative_path": str(image_path),
                         "original_path": photo["_path"]}
                slides = []
                for position, selected_photo_key in enumerate(selected):
                    current_slide = slide if selected_photo_key == key else dict(prior_by_key.get(selected_photo_key, {}))
                    if current_slide:
                        current_slide["position"] = position
                        slides.append(current_slide)
                update_campaign(cid, caption, campaign["status"])
                save_post_package(cid, dict(package, post_type=post_type, platform=campaign["platform"], aspect=aspect,
                    caption=caption, hashtags=[tag for tag in hashtags_text.split() if tag], cta=cta,
                    caption_adaptation=adaptation, platform_captions=platform_captions, slides=slides))
                st.success(f"Slide {index + 1} image updated.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not update slide {index + 1}: {exc}")

    if st.button("Create / update complete post package", type="primary", key=f"{prefix}_compose_all",
                 disabled=post_type.startswith("Multiple") and len(selected) < 2):
        progress = st.progress(0, text="Preparing separate visual slides…")
        try:
            slides = []
            for index, key in enumerate(selected):
                photo = by_key[key]
                overlay = st.session_state.get(f"{prefix}_slide_text_{hashlib.sha1(key.encode()).hexdigest()[:12]}", "")
                image_path = compose_post_creative(overlay, key, aspect, cid,
                                                   slide_index=index if post_type.startswith("Multiple") else None)
                slides.append({"position": index, "photo_key": key, "display_name": photo["display_name"],
                    "source_type": photo["source_type"], "source_filename": photo.get("source_filename", photo["filename"]),
                    "page_number": photo.get("page_number"), "slide_text": overlay,
                    "selection_reason": sequence_reasons.get(key, ""), "creative_path": str(image_path),
                    "original_path": photo["_path"]})
                progress.progress((index + 1) / len(selected), text=f"Composed slide {index + 1} of {len(selected)}")
            result = {"version": 1, "campaign_id": cid, "post_type": post_type, "platform": campaign["platform"],
                      "aspect": aspect, "caption": caption, "hashtags": [tag for tag in hashtags_text.split() if tag],
                      "cta": cta, "caption_adaptation": adaptation,
                      "platform_captions": platform_captions, "slides": slides}
            update_campaign(cid, caption, campaign["status"])
            save_post_package(cid, result)
            st.success(f"Created {len(slides)} separate slide image(s) and saved the publishing package.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not create the post package: {exc}")
        finally:
            progress.empty()

    current = get_campaign(cid) or campaign
    try:
        saved_package = json.loads(current.get("post_package") or "{}")
    except (TypeError, ValueError):
        saved_package = {}
    if saved_package.get("slides"):
        for slide in saved_package["slides"]:
            if not slide.get("original_path") and slide.get("photo_key") in by_key:
                slide["original_path"] = by_key[slide["photo_key"]]["_path"]
        saved_package.setdefault("caption", caption)
        saved_package.setdefault("campaign_id", cid)
        st.markdown("##### Download publishing package")
        slides = sorted(saved_package["slides"], key=lambda item: item.get("position", 0))
        for index, slide in enumerate(slides, 1):
            image_path = Path(slide.get("creative_path", ""))
            if image_path.is_file():
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.image(str(image_path), caption=f"Slide {index} · {slide.get('display_name', '')}", use_container_width=True)
                with col_b:
                    st.download_button(f"Download slide {index}", image_path.read_bytes(),
                        file_name=f"event2social-{cid}-slide-{index:02d}.jpg", mime="image/jpeg", key=f"{prefix}_dl_slide_{index}")
                    original_path = Path(slide.get("original_path", ""))
                    if original_path.is_file():
                        st.download_button(f"Download original {index}", original_path.read_bytes(),
                            file_name=f"event2social-{cid}-original-{index:02d}{original_path.suffix.lower()}",
                            mime="application/octet-stream", key=f"{prefix}_dl_original_{index}")
        caption_data = render_caption_file(saved_package)
        st.download_button("Download caption and hashtags (.txt)", caption_data.encode("utf-8"),
                           file_name=f"event2social-{cid}-caption.txt", mime="text/plain", key=f"{prefix}_dl_caption")
        try:
            zip_data = build_package_zip(saved_package)
            st.download_button("Download complete package (.zip)", zip_data,
                               file_name=f"event2social-{cid}-publishing-package.zip", mime="application/zip",
                               key=f"{prefix}_dl_zip")
        except ValueError as exc:
            st.warning(f"Finish composing all slides before downloading the ZIP: {exc}")


def show_create_page() -> None:
    hero("Campaign Creation", "Create an event post using confirmed event details and relevant source material when available.")
    event = event_details()
    if not event:
        st.info("Add your event details before creating a post.")
        if st.button("Enter event details", type="primary", key="create_missing_event"):
            go("Event Details")
        return
    rows = load_chunks()
    index = load_state("knowledge_index", {})
    hashes = {doc["filename"]: doc["sha256"] for doc in list_documents()}
    index_valid = bool(rows and index.get("embedding_model") and embedding_index_matches(
        index, provider=EMBED_PROVIDER, base_url=EMBED_BASE, model=EMBED_MODEL, source_hashes=hashes))
    can_retrieve = bool(index_valid and EMBED_KEY)
    if can_retrieve:
        st.success(f"Knowledge Base ready · {len(rows)} passages will be searched automatically.")
    elif rows:
        st.warning("The saved Knowledge Base index is missing, out of date, or unavailable. A draft can still use confirmed event details; prepare the Knowledge Base to add source passages.")
        if st.button("Open Knowledge Base", key="create_open_kb"):
            go("Knowledge Base")
    else:
        st.info("No searchable passages are ready. You can still create a draft from the confirmed event profile; readable source files can improve it.")
        if st.button("Add source material", key="create_add_sources"):
            go("Knowledge Base")
    if not GEN_KEY:
        st.warning("AI generation is unavailable until the administrator configures the server provider credential. See Settings → Administrator setup.")
    else:
        st.caption(f"Generation is configured · {AI_PROVIDER.title()} · {GEN_MODEL}. Availability and quota are verified only when a request is made.")
    st.caption(f"Event: **{event.get('name', 'Untitled')}** · confirmed profile + {'automatic retrieval' if can_retrieve else 'profile only'}")
    voice_campaign = speech_input("campaign_voice")
    if voice_campaign and st.session_state.get("_last_campaign_voice") != voice_campaign:
        st.session_state["campaign_focus"] = (st.session_state.get("campaign_focus", "") + " " + voice_campaign).strip()
        st.session_state["_last_campaign_voice"] = voice_campaign
        st.rerun()
    with st.form("campaign_form", border=True):
        c1, c2 = st.columns(2)
        with c1:
            platform = st.selectbox("Platform", ["Instagram", "LinkedIn", "Facebook"])
            content_type = st.selectbox("Content format", {
                "Instagram": ["Caption", "Event announcement", "Countdown post"],
                "LinkedIn": ["Professional event post", "Registration announcement", "Program highlight"],
                "Facebook": ["Promotional post", "Event announcement", "Community invitation"]}[platform])
            audience = st.text_input("Audience", event.get("audience", ""), placeholder="Who should this post reach?")
        with c2:
            tone = st.selectbox("Tone", ["Warm and welcoming", "Professional", "Energetic", "Informative", "Playful"])
            objective = st.text_input("Post objective", event.get("objective", ""), placeholder="What should readers do or know?")
            focus = st.text_input("What should this post focus on?", placeholder="For example, highlight confirmed activities", key="campaign_focus")
            if platform == "Instagram":
                aspect = st.selectbox("Post size", ["Square (1:1)", "Portrait (4:5)"])
            else:
                aspect = "Square (1:1)"
        submit = st.form_submit_button("Find useful information and create draft", type="primary", use_container_width=True,
                                       disabled=not bool(GEN_KEY))
    if submit:
        query = "\n".join([event.get("name", ""), event.get("description", ""), event.get("date", ""),
            event.get("venue", ""), event.get("organizer", ""), event.get("theme", ""), audience, objective,
            content_type, platform, tone, focus, event.get("extra", "")])
        try:
            evidence, retrieval_note = [], ""
            with st.spinner("Checking the Knowledge Base when available, then writing your event post…"):
                if can_retrieve:
                    try:
                        query_input = embedding_request_text(query, provider=EMBED_PROVIDER, role="query",
                                                             model=index["embedding_model"])
                        query_vector = embed_texts([query_input], api_key=EMBED_KEY, base_url=EMBED_BASE,
                                                   model=index["embedding_model"], provider=EMBED_PROVIDER)[0]
                        if len(query_vector) != index.get("vector_dimensions"):
                            raise ProviderError("Query dimensions do not match the saved index. Re-prepare the Knowledge Base.")
                        evidence = retrieve(query, rows, query_vector, top_k=4)
                        if not evidence:
                            retrieval_note = "No relevant Knowledge Base passages were found; the draft uses the confirmed event profile."
                    except Exception as exc:
                        retrieval_note = f"Knowledge Base search did not complete ({exc}). The draft uses the confirmed event profile."
                media_cache = load_media_analysis()
                photo_docs = visual_photo_sources()
                ranked_photos = rank_photos(photo_docs, " ".join([objective, focus, content_type]), media_cache)
                selected_key = st.session_state.get("kb_selected_visual_photo", "")
                ranked_names = {item["filename"] for item in ranked_photos}
                selected_photo = (selected_key if selected_key in ranked_names else
                                  ranked_photos[0]["filename"] if ranked_photos else "")
                media_context = [{"filename": d["filename"], "description": media_cache[d["filename"]]["text"][:4000]}
                                 for d in list_documents()
                                 if d["filename"] in media_cache
                                 and media_cache[d["filename"]].get("status") == "complete"
                                 and media_cache[d["filename"]].get("sha256") == d["sha256"]]
                brief = {"event": event, "confirmed_event_profile": True, "platform": platform, "content_type": content_type, "audience": audience,
                         "tone": tone, "objective": objective, "campaign_focus": focus,
                         "aspect": aspect, "retrieval_status": "used" if evidence else "profile only",
                         "selected_media": [selected_photo] if selected_photo else [],
                         "photo_rankings": [{"filename": p["filename"], "score": p["score"],
                                             "resolution": p["resolution"], "duplicate": p["duplicate"]}
                                            for p in ranked_photos],
                         "media_descriptions": media_context,
                         "embedding_provider": EMBED_PROVIDER, "generation_provider": AI_PROVIDER,
                         "embedding_model": index.get("embedding_model", ""), "generation_model": GEN_MODEL}
                content = generate(brief, evidence, api_key=GEN_KEY, base_url=GEN_BASE,
                                  model=GEN_MODEL, provider=AI_PROVIDER)
                cid = save_campaign(event.get("name", ""), platform, json.dumps(brief, ensure_ascii=False), content, "Draft", evidence)
                if selected_photo:
                    update_campaign_selection(cid, [selected_photo])
                st.session_state["active_campaign_id"] = cid
                st.session_state["campaign_just_created"] = cid
            st.success("Draft generated and saved." if evidence else "Draft generated and saved from the confirmed event profile.")
            if retrieval_note:
                st.warning(retrieval_note)
        except Exception as exc:
            st.error(f"Could not generate a grounded draft: {exc}")
    active = get_campaign(st.session_state.get("active_campaign_id"))
    if active:
        st.divider()
        st.subheader("Draft preview")
        st.caption(f"{active['platform']} · {active['event_name']} · Saved locally")
        if st.session_state.get("campaign_just_created") == active["id"]:
            if st.button("Continue to Generated Content", key="create_to_content"):
                st.session_state.pop("campaign_just_created", None)
                go("Generated Content")
        editable = st.text_area("Edit your draft", active["content"], height=240, key=f"draft_{active['id']}")
        a, b, c = st.columns(3)
        if a.button("Save draft edits", type="primary", key="save_draft"):
            update_campaign(active["id"], editable, "Draft")
            st.success("Draft edits saved.")
            st.rerun()
        b.download_button("Download .txt", editable, file_name="event2social-draft.txt", mime="text/plain")
        if c.button("Review this draft", key="review_draft"):
            update_campaign(active["id"], editable, "Draft")
            go("Content Validation")
        show_evidence(active)
        if active["platform"] in {"Instagram", "LinkedIn", "Facebook"}:
            show_post_package_editor(active, editable)


def show_evidence(campaign: dict) -> None:
    evidence = json.loads(campaign.get("evidence") or "[]")
    with st.expander("Sources used for this draft", expanded=True):
        if not evidence:
            st.info("No Knowledge Base passages were retrieved. This draft uses the confirmed event profile.")
        for i, item in enumerate(evidence, 1):
            page_label = f" · page {item['page_number']}" if item.get("page_number") else ""
            st.markdown(f"**{i}. {item.get('source_filename', 'Source document')}{page_label}** · relevance {item.get('score', 0):.3f}")
            st.write(item.get("text", ""))


def show_dashboard() -> None:
    hero("Welcome to EVENT2SOCIAL", "Plan event communications and create social posts grounded in your own information.")
    event, campaigns, docs = event_details(), list_campaigns(), list_documents()
    drafts = sum(c["status"] == "Draft" for c in campaigns)
    approved = sum(c["status"] == "Approved by human" for c in campaigns)
    cols = st.columns(4)
    for col, icon, value, label in [(cols[0], "E", 1 if event else 0, "Events"),
            (cols[1], "C", len(campaigns), "Posts created"), (cols[2], "R", drafts, "Awaiting review"),
            (cols[3], "✓", approved, "Approved posts")]:
        with col:
            metric_card(icon, value, label)
    a, b = st.columns([2, 1])
    with a, st.container(border=True):
        st.subheader("Your workspace")
        st.write("Add event details, gather reference documents, and prepare clear posts for your community.")
        if st.button("+ Create New Event" if not event else "Open Event Details", type="primary", key="dashboard_event"):
            go("Event Details")
    with b, st.container(border=True):
        st.subheader("Workspace overview")
        st.write(f"{len(docs)} event document(s) · {len(load_chunks())} searchable passage(s)")
        if st.button("Open Workspace", key="dashboard_workspace"):
            go("Workspace")
    st.markdown("### My Events")
    if not event:
        st.info("No events yet. Create an event to get started; nothing is pre-filled.")
        if st.button("Create your first event", key="dashboard_empty_event"):
            go("Event Details")
    else:
        with st.container(border=True):
            st.markdown(f"**{event.get('name', 'Untitled event')}**")
            st.caption(" · ".join(x for x in [event.get("date", ""), event.get("venue", "")] if x) or "Event details saved locally")
            if st.button("Manage event", key="dashboard_manage_event"):
                go("My Events")
    st.markdown("### Recent content")
    if not campaigns:
        st.info("No saved drafts yet. Create confirmed event details; an administrator-configured generation service is required to draft posts.")
    else:
        for campaign in campaigns[:3]:
            with st.container(border=True):
                st.markdown(f"**{campaign['event_name']} · {campaign['platform']}**")
                st.caption(f"{campaign['created_at']} · {campaign['status']}")
                st.write(campaign["content"][:180] + ("…" if len(campaign["content"]) > 180 else ""))


def show_events() -> None:
    hero("My Events", "View and update the event details saved in this local workspace.")
    event = event_details()
    if not event:
        st.info("No event details yet. Create an event to begin.")
    else:
        with st.container(border=True):
            st.subheader(event.get("name", "Untitled event"))
            st.write(event.get("description") or "No event description added.")
            st.caption(" · ".join(x for x in [event.get("date", ""), event.get("venue", ""), event.get("organizer", "")] if x))
    if st.button("+ Create New Event" if not event else "Edit Event Details", type="primary", key="events_edit"):
        go("Event Details")
    st.caption("This version stores one active event at a time. Multi-event workspaces are not implemented yet.")


def show_workspace() -> None:
    hero("Workspace", "Choose a step to prepare, create, and review event communications.")
    cards = [
        ("Event Details", "Add the event facts your team has confirmed.", "Event Details", "Local processing"),
        ("Knowledge Base", "Save and prepare event documents for search.", "Knowledge Base", "API required for embeddings"),
        ("RAG Retrieval", "Search prepared documents and inspect relevant excerpts.", "RAG Retrieval", "API required for semantic search"),
        ("Campaign Creation", "Create a grounded draft for Instagram, LinkedIn, or Facebook.", "Campaign Creation", "API required for writing"),
        ("Generated Content", "Open, edit, and download saved drafts.", "Generated Content", "Local processing"),
        ("Reel Creator", "Build a local vertical video or photo slideshow.", "Reel Creator", "FFmpeg required"),
        ("Content Validation", "Check a saved post before your team reviews it.", "Content Validation", "Local checks"),
        ("Review & Approval", "Mark a post approved or return it to draft.", "Review & Approval", "Team decision recorded locally"),
        ("Publishing", "Review the handoff status for approved content.", "Publishing", "Not connected to social accounts"),
    ]
    for row in range(0, len(cards), 2):
        cols = st.columns(2)
        for col, (title, description, target, badge) in zip(cols, cards[row:row+2]):
            with col, st.container(border=True):
                st.subheader(title)
                st.write(description)
                st.caption(badge)
                if st.button(f"Open {title}", key=f"workspace_{target}", use_container_width=True):
                    go(target)


def show_generated_content() -> None:
    hero("Generated Content", "Edit and download posts saved for your event.")
    campaign = campaign_selectbox("content_campaign")
    if not campaign:
        return
    brief = json.loads(campaign.get("brief") or "{}")
    st.caption(f"{campaign['platform']} · {brief.get('content_type', 'Social post')} · {campaign['created_at']} · {campaign['status']}")
    if brief.get("media_descriptions"):
        with st.expander("AI-analyzed media context · unconfirmed", expanded=False):
            for item in brief["media_descriptions"]:
                st.markdown(f"**{item.get('filename', 'Media source')}**")
                st.write(item.get("description", ""))
    selected_media = json.loads(campaign.get("selected_media") or "[]")
    if selected_media:
        st.caption("Original media: " + ", ".join(selected_media))
    text = st.text_area("Social post", campaign["content"], height=260, key=f"content_{campaign['id']}")
    a, b = st.columns(2)
    if a.button("Save changes", type="primary", key="content_save"):
        update_campaign(campaign["id"], text, campaign["status"])
        st.success("Changes saved.")
        st.rerun()
    b.download_button("Download .txt", text, file_name=f"event2social-{campaign['id']}.txt", mime="text/plain")
    if st.button("Continue to Content Validation", key="generated_to_validation"):
        update_campaign(campaign["id"], text, campaign["status"])
        go("Content Validation")
    show_post_package_editor(campaign, text)
    show_evidence(campaign)
    if campaign.get("creative_path") and Path(campaign["creative_path"]).is_file():
        st.image(campaign["creative_path"], caption=f"First post slide · {campaign.get('creative_format', '')}")


def show_reel_creator() -> None:
    hero("Reel / Short Video Creator", "Prepare a vertical event video or photo slideshow locally for Instagram Reels or YouTube Shorts.")
    event = event_details()
    if not event:
        st.info("Add confirmed event details first so the script and cover can use verified information.")
        return
    videos = [doc for doc in list_documents() if Path(doc["filename"]).suffix.lower() in {".mp4", ".mov"}
              and (UPLOAD_DIR / doc["stored_filename"]).is_file()]
    photos = visual_photo_sources()
    assets = {}
    for doc in videos:
        assets[doc["filename"]] = {"key": doc["filename"], "label": doc["filename"],
            "path": str(UPLOAD_DIR / doc["stored_filename"]), "kind": "video", "doc": doc}
    for photo in photos:
        assets[photo["photo_key"]] = {"key": photo["photo_key"], "label": photo["display_name"],
            "path": photo["_path"], "kind": "image", "doc": photo}
    if not assets:
        st.info("Upload event videos or photos in the Knowledge Base before creating a Reel.")
        return
    media_cache = load_media_analysis()
    event_terms = set(re.findall(r"[a-z0-9]{3,}", " ".join(str(v or "") for v in event.values()).casefold()))
    candidate_rows = []
    for asset in assets.values():
        doc = asset["doc"]
        notes = media_cache.get(doc.get("filename", asset["key"]), {})
        description = notes.get("text", "") if notes.get("sha256") == doc.get("sha256") else ""
        overlap = len(event_terms & set(re.findall(r"[a-z0-9]{3,}", description.casefold())))
        size = int(doc.get("byte_count", 0)) if asset["kind"] == "video" else int(doc.get("width", 0) or 0) * int(doc.get("height", 0) or 0)
        candidate_rows.append((overlap, size, asset))
    candidate_rows.sort(key=lambda row: (-row[0], -row[1], row[2]["label"].casefold()))
    recommended = [row[2]["key"] for row in candidate_rows[:min(6, len(candidate_rows))]]
    st.caption("Automatic source suggestions favour existing event-analysis matches and larger local sources. This screen does not send media to an AI service unless you explicitly request AI copy generation; video editing uses local FFmpeg.")
    selected_key = "reel_source_selection"
    if selected_key not in st.session_state:
        st.session_state[selected_key] = recommended
    platform = st.selectbox("Target platform", ["Instagram Reels", "YouTube Shorts"], key="reel_platform")
    selected = st.multiselect("Video clips and photos · publishing order", list(assets), key=selected_key,
        format_func=lambda key: assets[key]["label"], help="The suggested starting selection is editable. Use photos, videos, or a mixture.")
    if not selected:
        st.warning("Select at least one source to render a video.")
        return
    st.caption(f"Selected {len(selected)} of {len(assets)} available source(s); order shown above is the scene order.")
    script_key, caption_key, cover_key, closing_key = "reel_script", "reel_caption", "reel_cover_text", "reel_closing_text"
    def reel_text(value: object, fallback: str = "") -> str:
        """Keep text-area and text-input state valid even after malformed AI JSON."""
        return value if isinstance(value, str) else fallback

    overlay_keys = [f"reel_overlay_{hashlib.sha1(key.encode()).hexdigest()[:12]}" for key in selected]
    pending = st.session_state.pop("reel_copy_pending", None)
    if pending:
        st.session_state[script_key] = reel_text(pending.get("script"))
        st.session_state[caption_key] = reel_text(pending.get("caption"))
        st.session_state[cover_key] = reel_text(pending.get("cover_text"))
        st.session_state[closing_key] = reel_text(pending.get("closing_text"), "Thank you for joining us")
        for key, value in zip(overlay_keys, pending.get("scene_overlays", [])):
            st.session_state[key] = reel_text(value)
        st.session_state["reel_hashtags"] = " ".join("#" + str(tag).lstrip("#") for tag in pending.get("hashtags", []))
        st.session_state["reel_cta"] = pending.get("call_to_action", "")
    if script_key not in st.session_state:
        st.session_state[script_key] = f"Opening: {event.get('name', 'Event highlights')}\nShow the selected event moments in order.\nClosing: Thank the community and invite viewers to learn more."
    if caption_key not in st.session_state:
        st.session_state[caption_key] = str(event.get("description", "")).strip()
    if cover_key not in st.session_state:
        st.session_state[cover_key] = str(event.get("name", "Event Highlights"))[:70]
    if closing_key not in st.session_state:
        st.session_state[closing_key] = "Thank you for joining us"
    # Session state can outlive a rerun (and AI providers may return valid JSON
    # with values of the wrong shape). Streamlit text widgets require strings.
    for key in (script_key, caption_key, cover_key, closing_key, "reel_hashtags", "reel_cta", *overlay_keys):
        if key in st.session_state and not isinstance(st.session_state[key], str):
            st.session_state[key] = ""
    ai_col, progress_col = st.columns([1, 2])
    if ai_col.button("Generate script, captions and scene text with AI", key="reel_generate_copy", disabled=not bool(GEN_KEY)):
        selected_notes = []
        for key in selected:
            doc = assets[key]["doc"]
            note = media_cache.get(doc.get("filename", key), {})
            if note.get("sha256") == doc.get("sha256") and note.get("text"):
                selected_notes.append({"source": assets[key]["label"], "description": note["text"][:1500]})
        brief = {"output_mode": "reel_script", "platform": platform, "slide_count": len(selected),
                 "event": event, "media_descriptions": selected_notes,
                 "campaign_focus": st.session_state.get(caption_key, "")}
        try:
            with st.spinner("Generating the short-video script and copy…"):
                raw = generate(brief, [], api_key=GEN_KEY, base_url=GEN_BASE,
                               model=GEN_MODEL, provider=AI_PROVIDER)
            try:
                copy = json.loads(raw)
                if not isinstance(copy, dict):
                    raise ValueError("Expected a JSON object")
            except (TypeError, ValueError):
                copy = {"script": raw, "caption": raw, "hashtags": [],
                        "call_to_action": "", "scene_overlays": [], "cover_text": event.get("name", "Event highlights")}
            st.session_state["reel_copy_pending"] = copy
            st.rerun()
        except Exception as exc:
            st.error(f"AI copy generation failed; you can still edit the local script and render the selected media. {exc}")
    if not GEN_KEY:
        progress_col.caption("AI copy generation is not configured. The local Reel export remains available with editable event-based text.")
    script = st.text_area("Scene script", key=script_key, height=100)
    caption = st.text_area("Caption", key=caption_key, height=100)
    hashtags = st.text_input("Hashtags", key="reel_hashtags")
    cta = st.text_input("Optional call to action", key="reel_cta")
    cover_text = st.text_input("Cover title", key=cover_key)
    closing_text = st.text_input("Closing card text", key=closing_key)
    overlays = []
    for index, key in enumerate(selected):
        overlay_key = overlay_keys[index]
        if overlay_key not in st.session_state:
            st.session_state[overlay_key] = cover_text if index == 0 else ""
        overlays.append(st.text_input(f"Scene {index + 1} overlay", key=overlay_key))
    duration = st.slider("Seconds per source segment (video clips are trimmed from the start)", 2, 12, 4, key="reel_segment_duration")
    st.info("Export uses 9:16 vertical H.264 MP4 with fade transitions and text overlays. Original clip audio is omitted; add licensed platform audio after download. No new AI-generated footage is created.")
    if st.button("Render Reel / Short MP4", type="primary", key="reel_render", disabled=not bool(selected)):
        output_dir = DATA_DIR / "reels"
        output_path = output_dir / f"event2social-reel-{datetime.now().strftime('%Y%m%d-%H%M%S')}.mp4"
        segments = [{"path": assets[key]["path"], "kind": assets[key]["kind"], "duration": duration} for key in selected]
        try:
            progress = st.progress(0, text="Rendering vertical video with FFmpeg…")
            with st.spinner("FFmpeg is composing the selected scenes. Larger videos may take a few minutes."):
                video_path, cover_path = create_vertical_video(segments, output_path, overlays=overlays,
                    opening_title=cover_text, closing_text=closing_text,
                    progress_callback=lambda fraction: progress.progress(
                        fraction, text=f"Encoding vertical video · {int(fraction * 100)}%"))
            progress.progress(1, text="Video and cover ready")
            details = [f"Platform: {platform}", "Format: 9:16 vertical MP4", f"Cover title: {cover_text}",
                       f"Closing card: {closing_text}",
                       "", "Script:", script, "", "Scene overlays:"]
            details.extend(f"Scene {index + 1}: {text}" for index, text in enumerate(overlays) if text.strip())
            details.extend(["", "Caption:", caption,
                       "", "Hashtags:", hashtags, "", "Call to action:", cta])
            caption_path = output_dir / f"{output_path.stem}-caption.txt"
            caption_path.write_text("\n".join(details).strip() + "\n", encoding="utf-8")
            st.session_state["reel_output"] = {"video": str(video_path), "cover": str(cover_path),
                "caption": str(caption_path), "platform": platform}
            st.success("Vertical video, cover image and caption file created locally.")
        except Exception as exc:
            st.error(f"Could not render the Reel: {exc}")
        finally:
            if "progress" in locals():
                progress.empty()
    result = st.session_state.get("reel_output", {})
    if result.get("video") and Path(result["video"]).is_file():
        video_path, cover_path, caption_path = Path(result["video"]), Path(result["cover"]), Path(result["caption"])
        st.video(str(video_path))
        st.image(str(cover_path), caption="Reel cover image", width=360)
        col_video, col_cover, col_caption = st.columns(3)
        col_video.download_button("Download MP4", video_path.read_bytes(), file_name=video_path.name,
                                  mime="video/mp4", key="reel_download_video")
        col_cover.download_button("Download cover", cover_path.read_bytes(), file_name=cover_path.name,
                                  mime="image/jpeg", key="reel_download_cover")
        col_caption.download_button("Download caption", caption_path.read_bytes(), file_name=caption_path.name,
                                    mime="text/plain", key="reel_download_caption")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(video_path, video_path.name)
            archive.write(cover_path, cover_path.name)
            archive.write(caption_path, caption_path.name)
        st.download_button("Download complete Reel package (.zip)", buffer.getvalue(),
            file_name=f"{video_path.stem}-package.zip", mime="application/zip", key="reel_download_zip")


def show_validation() -> None:
    hero("Content Validation", "Run quick checks for clarity and platform length before sending a post for review.")
    campaign = campaign_selectbox("validation_campaign")
    if not campaign:
        return
    text = st.text_area("Post to check", campaign["content"], height=230, key=f"validation_{campaign['id']}")
    limits = {"Instagram": 2200, "LinkedIn": 3000, "Facebook": 63206}
    if st.button("Run content checks", type="primary", key="validate_content"):
        checks = [("Post has content", bool(text.strip()), "Add text before review."),
                  (f"Within {campaign['platform']} character guide", len(text) <= limits.get(campaign["platform"], 3000),
                   f"This post has {len(text):,} characters; the guide is {limits.get(campaign['platform'], 3000):,}."),
                  ("Event name appears in the post", campaign["event_name"].casefold() in text.casefold(),
                   "Consider naming the event so readers have context.")]
        st.session_state[f"validation_results_{campaign['id']}"] = checks
        st.session_state[f"validation_text_hash_{campaign['id']}"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    checks = st.session_state.get(f"validation_results_{campaign['id']}", [])
    for label, passed, note in checks:
        (st.success if passed else st.warning)(f"{'Pass' if passed else 'Review'} · {label}" + ("" if passed else f" — {note}"))
    if checks and st.button("Continue to Review & Approval", key="validation_to_review"):
        current_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if current_hash != st.session_state.get(f"validation_text_hash_{campaign['id']}"):
            st.warning("The text changed after the last check. Run content checks again to validate this exact version before review.")
        elif not text.strip():
            st.error("Add post text before sending it for review.")
        else:
            update_campaign(campaign["id"], text, campaign["status"])
            go("Review & Approval")
    if st.button("Save edits", key="validation_save"):
        update_campaign(campaign["id"], text, campaign["status"])
        st.success("Edits saved.")
        st.rerun()


def show_approval() -> None:
    hero("Review & Approval", "Record your team’s decision before preparing a post for publishing.")
    campaign = campaign_selectbox("approval_campaign")
    if not campaign:
        return
    st.caption(f"{campaign['platform']} · current status: {campaign['status']}")
    st.text_area("Post for review", campaign["content"], height=230, disabled=True, key=f"approval_{campaign['id']}")
    show_evidence(campaign)
    left, right = st.columns(2)
    if left.button("Approve for publishing", type="primary", key="approve_campaign", disabled=campaign["status"] == "Approved by human"):
        update_campaign(campaign["id"], campaign["content"], "Approved by human")
        st.success("Approval recorded locally.")
        st.rerun()
    if right.button("Return to draft", key="return_campaign", disabled=campaign["status"] == "Draft"):
        update_campaign(campaign["id"], campaign["content"], "Draft")
        st.success("Post returned to Draft.")
        st.rerun()


def show_publishing() -> None:
    hero("Publishing", "Review posts approved by your team and download them for sharing.")
    st.info("Publishing directly to Instagram, LinkedIn, or Facebook is not connected in this version. No post will be sent from this page.")
    approved = [c for c in list_campaigns() if c["status"] == "Approved by human"]
    if not approved:
        st.info("No posts have been approved yet. Review a generated post first.")
        if st.button("Open Review & Approval", key="publishing_review"):
            go("Review & Approval")
        return
    for campaign in approved:
        with st.container(border=True):
            st.markdown(f"**{campaign['event_name']} · {campaign['platform']}**")
            st.caption("Approved · ready for your team to copy or download")
            st.write(campaign["content"])
            st.download_button("Download approved post", campaign["content"], file_name=f"event2social-approved-{campaign['id']}.txt",
                               mime="text/plain", key=f"publish_download_{campaign['id']}")
            if campaign.get("creative_path") and Path(campaign["creative_path"]).is_file():
                st.image(campaign["creative_path"], caption=f"Approved visual · {campaign.get('creative_format', '')}")
                st.download_button("Download approved image", Path(campaign["creative_path"]).read_bytes(),
                                   file_name=f"event2social-approved-{campaign['id']}.jpg", mime="image/jpeg",
                                   key=f"publish_image_{campaign['id']}")


def show_settings() -> None:
    hero("Settings", "AI services are configured by the person who runs this tool.")
    _, embed_base, embed_model = embedding_config(APP_SECRETS, EMBED_PROVIDER)
    _, gen_base, gen_model = generation_config(APP_SECRETS, AI_PROVIDER)
    st.info("API credentials are read from server environment variables or Streamlit secrets. They are never entered by ordinary users or stored in this workspace database.")
    st.subheader("Configured services")
    a, b, c = st.columns(3)
    a.metric("Document search · embeddings", "Configured" if EMBED_KEY else "Not configured")
    b.metric("Content generation", "Configured" if GEN_KEY else "Not configured")
    media_configured = AI_PROVIDER in {"groq", "gemini"} and bool(GEN_KEY)
    c.metric(f"{AI_PROVIDER.title()} image/video analysis", "Configured" if media_configured else "Not configured")
    st.caption(f"Generation: {AI_PROVIDER.title()} · Embeddings: {EMBED_PROVIDER.title()} ({urlparse(embed_base).hostname} / {embed_model}) · Generation endpoint: {urlparse(gen_base).hostname} / {GEN_MODEL} · Media analysis model: {MEDIA_MODEL or 'Not available'}")
    st.caption("Configured means a credential is present; this screen does not send test requests or claim current service availability.")
    if not (EMBED_KEY and GEN_KEY and media_configured):
        st.warning("Some AI services are not configured. An administrator must configure the required provider credential on the server.")
    with st.expander("Administrator setup", expanded=not bool(EMBED_KEY and GEN_KEY and media_configured)):
        st.write("For local development, create this file in the project root and keep it out of version control:")
        st.code(str(Path(__file__).parent / ".streamlit" / "secrets.toml"), language="text")
        st.write("Add the Groq key in the server secrets file, then restart Streamlit:")
        st.code('EVENT2SOCIAL_PROVIDER = "groq"\nGROQ_API_KEY = "set-this-in-the-server-secret-store"\n# Optional overrides\nGROQ_MODEL = "openai/gpt-oss-120b"\nGROQ_MEDIA_MODEL = "qwen/qwen3.8-27b"\nGROQ_AUDIO_MODEL = "whisper-large-v3"\n# Groq does not provide embeddings; existing Gemini key is used for RAG\nGEMINI_API_KEY = "keep-configured-for-document-search"', language="toml")
        st.caption("Generation, image analysis, and video-frame/audio analysis use Groq. Video processing uses local FFmpeg for sampled frames and audio extraction, then Groq vision and Whisper; a video file is not sent to Groq as a native video input. The browser speech-input control continues to use the browser's speech service. Groq currently provides no embeddings endpoint, so semantic document search continues to use the separately selected Gemini or OpenAI embedding credential. Keep GEMINI_API_KEY if you want to retain the current Gemini RAG embedding provider. Do not commit secrets.toml or enter keys in the UI.")
        st.caption("Embedding or model changes require re-indexing. Provider requests send document passages and generation briefs to the configured provider.")
    st.markdown("**Storage**")
    st.caption("Event details, uploaded files, passages, and drafts are stored in this tool’s local SQLite database, scoped to the signed-in account.")


# Shared visual system based on the reference recording: clean light surfaces, blue accent,
# compact typography, clear cards and generous page spacing.
with st.sidebar:
    st.markdown('<div class="brand-row"><div class="brand-mark">E2S</div><div><strong>EVENT2SOCIAL</strong><span>EVENT WORKSPACE</span></div></div>', unsafe_allow_html=True)
    st.caption("Local workspace · Saved on this device")
    if st.session_state.get("authenticated"):
        if st.button("Sign out", key="logout", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state.pop("active_campaign_id", None)
            st.session_state.pop("user_id", None)
            st.session_state.pop("legacy_owner", None)
            st.session_state.pop("_active_session_user", None)
            st.session_state["_clear_auth_fields"] = True
            set_active_user(None)
            st.session_state["route"] = "Dashboard"
            st.rerun()
        st.markdown('<div class="side-label">MAIN MENU</div>', unsafe_allow_html=True)
        for item in ("Dashboard", "My Events", "Workspace", "Settings"):
            if st.button(item, key=f"main_{item}", use_container_width=True, type="primary" if page == item else "secondary"):
                go(item)
        with st.expander("WORKSPACE PAGES", expanded=page not in ("Dashboard", "My Events", "Workspace", "Settings")):
            for item in ("Event Details", "Knowledge Base", "RAG Retrieval", "Campaign Creation",
                         "Generated Content", "Reel Creator", "Content Validation", "Review & Approval", "Publishing"):
                if st.button(item, key=f"nav_{item}", use_container_width=True, type="primary" if page == item else "secondary"):
                    go(item)
    st.divider()
    saved_dark = load_dark_preference()
    dark = st.toggle("Dark appearance", value=saved_dark, key="theme_mode")
    save_dark_preference(dark)
    st.caption("Storage · local SQLite")

if dark:
    theme = {"bg":"#081321", "rail":"#1c3158", "panel":"#172943", "card":"#172943", "text":"#f5f7fb", "muted":"#b9c7da", "border":"#4b5f7c", "accent":"#4bc8f1", "accent_text":"#071522", "input":"#101e32", "alert":"#12243a"}
else:
    theme = {"bg":"#ffffff", "rail":"#f0f1ef", "panel":"#f0f1ef", "card":"#f0f1ef", "text":"#171a20", "muted":"#4a5361", "border":"#bac3cf", "accent":"#3263e8", "accent_text":"#ffffff", "input":"#ffffff", "alert":"#eef2fb"}
st.markdown(f"""
<style>
  :root {{ color-scheme: {'dark' if dark else 'light'}; }}
  .stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{{background:{theme['bg']};color:{theme['text']}}}
  [data-testid="stSidebar"]{{background:{theme['rail']};border-right:1px solid {theme['border']}}}
  [data-testid="stSidebar"] *{{color:{theme['text']}}}
  [data-testid="stHeader"],.stAppHeader{{background:transparent!important;box-shadow:none!important;backdrop-filter:none!important;-webkit-backdrop-filter:none!important}}
  [data-testid="stHeader"]>div,.stAppHeader>div{{background:transparent!important}}[data-testid="stHeader"] *,.stAppHeader *{{color:{theme['text']}}}
  .block-container{{max-width:1240px;padding-top:1.15rem;padding-bottom:3.5rem}}
  h1,h2,h3,h4,p,label,th,td,[data-testid="stMarkdownContainer"],[data-testid="stWidgetLabel"]{{color:{theme['text']}}}
  h1{{font-size:1.85rem;font-weight:700}}h2{{font-size:1.35rem;font-weight:700}}h3{{font-size:1.08rem;font-weight:650}}
  [data-testid="stCaptionContainer"],.stCaption{{color:{theme['muted']}}}
  .brand-row{{display:flex;align-items:center;gap:.65rem;margin:.4rem 0 .45rem}}
  .brand-mark{{display:flex;align-items:center;justify-content:center;width:2.25rem;height:2.25rem;border-radius:.42rem;background:linear-gradient(140deg,#3263e8,#35b9dd);color:#fff!important;font-size:.68rem;font-weight:800;letter-spacing:.04em}}
  .brand-row strong{{display:block;font-size:.92rem}}.brand-row span{{display:block;margin-top:.13rem;color:{theme['muted']}!important;font-size:.65rem;letter-spacing:.1em}}
  .side-label{{margin:.8rem 0 .35rem;font-size:.66rem;font-weight:750;letter-spacing:.1em;color:{theme['muted']}!important}}
  .topbar{{display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:.65rem .85rem;margin:0 0 2rem;border:1px solid {theme['border']};border-left:4px solid {theme['accent']};border-radius:.35rem;background:{theme['panel']}}}
  .topbar strong{{display:block;font-size:.93rem}}.topbar-page{{display:block;margin-top:.12rem;color:{theme['muted']};font-size:.75rem}}.workspace-pill{{color:{theme['muted']};font-size:.75rem}}
  .hero{{padding:1.3rem 1.55rem;margin-bottom:1.45rem;border:1px solid {theme['border']};border-left:4px solid {theme['accent']};border-radius:.42rem;background:{theme['panel']}}}
  .hero h1{{color:{theme['text']};margin:.3rem 0 .45rem;font-size:1.8rem}}.hero p{{color:{theme['text']};margin:0;font-size:.96rem}}.eyebrow{{color:{theme['accent']};font-size:.66rem;font-weight:750;letter-spacing:.11em}}
  .metric-card{{min-height:8.1rem;padding:1rem 1.05rem;border:1px solid {theme['border']};border-radius:.42rem;background:{theme['card']}}}
  .metric-icon{{display:flex;align-items:center;justify-content:center;width:1.8rem;height:1.8rem;border-radius:.36rem;background:{theme['accent']};color:{theme['accent_text']};font-size:.75rem;font-weight:800}}
  .metric-value{{margin-top:.58rem;color:{theme['accent']};font-size:1.48rem;font-weight:750;line-height:1.2}}.metric-label{{margin-top:.14rem;color:{theme['text']};font-size:.82rem}}
  [data-testid="stVerticalBlockBorderWrapper"],[data-testid="stForm"]{{border-color:{theme['border']};border-radius:.45rem;background:{theme['card']}}}
  div.stButton>button,div.stFormSubmitButton>button{{min-height:2.45rem;border-radius:.35rem;border-color:{theme['border']};color:{theme['text']}}}
  div.stButton>button[kind="primary"],div.stFormSubmitButton>button[kind="primary"]{{border-color:{theme['accent']};background:{theme['accent']};color:{theme['accent_text']};font-weight:650}}
  [data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea,[data-testid="stNumberInput"] input,[data-testid="stDateInput"] input,[data-testid="stSelectbox"] div[data-baseweb="select"]>div,[data-testid="stMultiSelect"] div[data-baseweb="select"]>div{{background:{theme['input']};color:{theme['text']};border-color:{theme['border']};border-radius:.35rem}}
  input::placeholder,textarea::placeholder{{color:{theme['muted']};opacity:1}}[data-baseweb="popover"],[data-baseweb="menu"],[role="listbox"]{{background:{theme['card']};color:{theme['text']}}}[role="option"]{{color:{theme['text']}}}
  [data-testid="stFileUploader"] section{{background:{theme['card']};border-color:{theme['border']}}}[data-testid="stAlert"]{{background:{theme['alert']};border-color:{theme['border']};border-radius:.38rem}}[data-testid="stAlert"] *{{color:{theme['text']}}}
  [data-testid="stExpander"],[data-testid="stTabs"] [role="tablist"]{{border-color:{theme['border']};border-radius:.4rem}}[data-testid="stDataFrame"],[data-testid="stTable"]{{border-color:{theme['border']}}}table,th,td{{border-color:{theme['border']}}}a{{color:{theme['accent']}}}hr{{border-color:{theme['border']}}}
</style>
""", unsafe_allow_html=True)

ROUTE_HANDLERS = {
    "Dashboard": show_dashboard, "My Events": show_events, "Workspace": show_workspace,
    "Settings": show_settings, "Event Details": show_event_page,
    "Knowledge Base": show_knowledge_page, "RAG Retrieval": show_retrieval_page,
    "Campaign Creation": show_create_page, "Generated Content": show_generated_content,
    "Reel Creator": show_reel_creator,
    "Content Validation": show_validation, "Review & Approval": show_approval,
    "Publishing": show_publishing,
}
if st.session_state.get("authenticated"):
    ROUTE_HANDLERS[page]()
else:
    show_login()
