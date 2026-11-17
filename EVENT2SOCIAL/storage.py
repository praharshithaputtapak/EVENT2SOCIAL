"""SQLite persistence and local source-file storage."""
from __future__ import annotations

import json
import hashlib
import sqlite3
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "event2social.db"
UPLOAD_DIR = DATA_DIR / "uploads"
PDF_IMAGE_DIR = UPLOAD_DIR / "pdf_extracted_images"
_ACTIVE_USER_ID: ContextVar[str | None] = ContextVar("event2social_user_id", default=None)
_LEGACY_OWNER: ContextVar[bool] = ContextVar("event2social_legacy_owner", default=False)


class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc, tb):
        try:
            return super().__exit__(exc_type, exc, tb)
        finally:
            self.close()


def set_active_user(user_id: str | None, *, legacy_owner: bool = False) -> None:
    _ACTIVE_USER_ID.set(user_id)
    _LEGACY_OWNER.set(bool(legacy_owner))


def save_global_state(key: str, value: Any) -> None:
    with connect() as db:
        db.execute("INSERT INTO app_state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                   (key, json.dumps(value, ensure_ascii=False)))


def load_global_state(key: str, default: Any = None) -> Any:
    with connect() as db:
        row = db.execute("SELECT value FROM app_state WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=30, factory=_ClosingConnection)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS app_state(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS documents(
            id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL UNIQUE,
            stored_filename TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            byte_count INTEGER NOT NULL,
            added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS chunks(
            id INTEGER PRIMARY KEY,
            source TEXT,
            text TEXT,
            vector TEXT,
            source_filename TEXT,
            source_hash TEXT,
            page_number INTEGER,
            chunk_index INTEGER
        );
        CREATE TABLE IF NOT EXISTS campaigns(
            id INTEGER PRIMARY KEY,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            event_name TEXT,
            platform TEXT,
            brief TEXT,
            content TEXT,
            status TEXT DEFAULT 'Draft',
            evidence TEXT,
            publish_status TEXT DEFAULT 'Not simulated'
        );
        CREATE TABLE IF NOT EXISTS media_analysis(
            owner_id TEXT NOT NULL, filename TEXT NOT NULL, sha256 TEXT NOT NULL,
            status TEXT NOT NULL, text TEXT NOT NULL DEFAULT '',
            observations TEXT NOT NULL DEFAULT '[]', timestamps TEXT NOT NULL DEFAULT '[]',
            error TEXT NOT NULL DEFAULT '', model TEXT NOT NULL DEFAULT '',
            retry_after TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(owner_id, filename)
        );
        CREATE TABLE IF NOT EXISTS document_extractions(
            id INTEGER PRIMARY KEY, owner_id TEXT NOT NULL, filename TEXT NOT NULL,
            sha256 TEXT NOT NULL, page_number INTEGER NOT NULL DEFAULT -1,
            extraction_kind TEXT NOT NULL, status TEXT NOT NULL,
            text TEXT NOT NULL DEFAULT '', error TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(owner_id, filename, sha256, page_number, extraction_kind)
        );
        CREATE TABLE IF NOT EXISTS pdf_image_extraction_runs(
            owner_id TEXT NOT NULL, source_filename TEXT NOT NULL, source_hash TEXT NOT NULL,
            status TEXT NOT NULL, image_count INTEGER NOT NULL DEFAULT 0,
            error TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(owner_id, source_filename, source_hash)
        );
        CREATE TABLE IF NOT EXISTS extracted_pdf_images(
            id INTEGER PRIMARY KEY, owner_id TEXT NOT NULL, photo_key TEXT NOT NULL,
            source_filename TEXT NOT NULL, source_hash TEXT NOT NULL,
            page_number INTEGER NOT NULL, image_index INTEGER NOT NULL, image_hash TEXT NOT NULL,
            stored_filename TEXT NOT NULL, mime_type TEXT NOT NULL,
            width INTEGER NOT NULL, height INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(owner_id, source_filename, source_hash, page_number, image_hash),
            UNIQUE(owner_id, photo_key)
        );
    """)
    # Add metadata columns when opening a database created by the old prototype.
    existing = {row[1] for row in db.execute("PRAGMA table_info(chunks)")}
    for name, kind in (("source_filename", "TEXT"), ("source_hash", "TEXT"),
                       ("page_number", "INTEGER"), ("chunk_index", "INTEGER")):
        if name not in existing:
            db.execute(f"ALTER TABLE chunks ADD COLUMN {name} {kind}")
    existing_campaigns = {row[1] for row in db.execute("PRAGMA table_info(campaigns)")}
    if "publish_status" not in existing_campaigns:
        db.execute("ALTER TABLE campaigns ADD COLUMN publish_status TEXT DEFAULT 'Not simulated'")
    for name, kind in (("creative_path", "TEXT"), ("creative_format", "TEXT"),
                       ("selected_media", "TEXT"), ("post_package", "TEXT NOT NULL DEFAULT '{}'")):
        if name not in existing_campaigns:
            db.execute(f"ALTER TABLE campaigns ADD COLUMN {name} {kind}")
    for table in ("documents", "chunks", "campaigns"):
        cols = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
        if "owner_id" not in cols:
            db.execute(f"ALTER TABLE {table} ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'legacy'")
    existing_media = {row[1] for row in db.execute("PRAGMA table_info(media_analysis)")}
    if "retry_after" not in existing_media:
        db.execute("ALTER TABLE media_analysis ADD COLUMN retry_after TEXT NOT NULL DEFAULT ''")
    return db


def load_media_analysis() -> dict[str, dict[str, Any]]:
    owner_id = _ACTIVE_USER_ID.get() or "legacy"
    with connect() as db:
        rows = db.execute("SELECT * FROM media_analysis WHERE owner_id=?", (owner_id,)).fetchall()
    result = {row["filename"]: {"sha256": row["sha256"], "status": row["status"],
             "text": row["text"], "observations": json.loads(row["observations"]),
             "timestamps": json.loads(row["timestamps"]), "error": row["error"],
             "model": row["model"], "retry_after": row["retry_after"],
             "updated_at": row["updated_at"]} for row in rows}
    # Migrate the previous JSON cache lazily without discarding successful analyses.
    legacy = load_state("media_analysis", {})
    for filename, item in legacy.items():
        if filename not in result and item.get("sha256"):
            save_media_analysis(filename, item["sha256"], "complete" if item.get("text") else "failed",
                text=item.get("text", ""), error=item.get("error", ""), model=item.get("model", ""),
                retry_after=item.get("retry_after", ""))
            result[filename] = {**item, "status": "complete" if item.get("text") else "failed"}
    return result


def save_media_analysis(filename: str, sha256: str, status: str, *, text: str = "",
                        observations: list[str] | None = None, timestamps: list[str] | None = None,
                        error: str = "", model: str = "", retry_after: str = "") -> None:
    owner_id = _ACTIVE_USER_ID.get() or "legacy"
    with connect() as db:
        db.execute("""INSERT INTO media_analysis(owner_id,filename,sha256,status,text,observations,timestamps,error,model,retry_after,updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(owner_id,filename) DO UPDATE SET sha256=excluded.sha256,status=excluded.status,
            text=excluded.text,observations=excluded.observations,timestamps=excluded.timestamps,
            error=excluded.error,model=excluded.model,retry_after=excluded.retry_after,updated_at=CURRENT_TIMESTAMP""",
            (owner_id, filename, sha256, status, text, json.dumps(observations or [], ensure_ascii=False),
             json.dumps(timestamps or [], ensure_ascii=False), error, model, retry_after))


def save_document_extraction(filename: str, sha256: str, page_number: int | None,
                             extraction_kind: str, status: str, *, text: str = "", error: str = "") -> None:
    owner_id = _ACTIVE_USER_ID.get() or "legacy"
    page_key = page_number if page_number is not None else -1
    with connect() as db:
        db.execute("""INSERT INTO document_extractions(owner_id,filename,sha256,page_number,extraction_kind,status,text,error,updated_at)
            VALUES(?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(owner_id,filename,sha256,page_number,extraction_kind) DO UPDATE SET
            status=excluded.status,text=excluded.text,error=excluded.error,updated_at=CURRENT_TIMESTAMP""",
            (owner_id, filename, sha256, page_key, extraction_kind, status, text, error))


def list_document_extractions(filename: str | None = None) -> list[dict[str, Any]]:
    owner_id = _ACTIVE_USER_ID.get() or "legacy"
    with connect() as db:
        if filename is None:
            rows = db.execute("SELECT * FROM document_extractions WHERE owner_id=? ORDER BY filename,page_number", (owner_id,)).fetchall()
        else:
            rows = db.execute("SELECT * FROM document_extractions WHERE owner_id=? AND filename=? ORDER BY page_number", (owner_id, filename)).fetchall()
    return [dict(row) for row in rows]


def save_state(key: str, value: Any) -> None:
    user_id = _ACTIVE_USER_ID.get()
    scoped = f"user:{user_id}:{key}" if user_id else key
    with connect() as db:
        db.execute("INSERT INTO app_state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                   (scoped, json.dumps(value, ensure_ascii=False)))


def load_state(key: str, default: Any = None) -> Any:
    user_id = _ACTIVE_USER_ID.get()
    scoped = f"user:{user_id}:{key}" if user_id else key
    with connect() as db:
        row = db.execute("SELECT value FROM app_state WHERE key=?", (scoped,)).fetchone()
        if row is None and user_id and _LEGACY_OWNER.get():
            row = db.execute("SELECT value FROM app_state WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def save_uploaded_document(name: str, content: bytes, sha256: str) -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    owner_id = _ACTIVE_USER_ID.get() or "legacy"
    db_name = name if owner_id == "legacy" else f"{owner_id}::{name}"
    with connect() as db:
        row = db.execute("SELECT stored_filename FROM documents WHERE filename=? AND owner_id=?", (db_name, owner_id)).fetchone()
    stored_name = row["stored_filename"] if row else f"{uuid.uuid4().hex}{Path(name).suffix.lower()}"
    (UPLOAD_DIR / stored_name).write_bytes(content)
    with connect() as db:
        db.execute("""INSERT INTO documents(filename,stored_filename,sha256,byte_count,owner_id) VALUES(?,?,?,?,?)
                    ON CONFLICT(filename) DO UPDATE SET stored_filename=excluded.stored_filename,
                    sha256=excluded.sha256,byte_count=excluded.byte_count,added_at=CURRENT_TIMESTAMP,owner_id=excluded.owner_id""",
                   (db_name, stored_name, sha256, len(content), owner_id))


def list_documents() -> list[dict[str, Any]]:
    owner_id = _ACTIVE_USER_ID.get() or "legacy"
    with connect() as db:
        if owner_id == "legacy" or _LEGACY_OWNER.get():
            rows = db.execute("SELECT * FROM documents WHERE owner_id IN (?, 'legacy') ORDER BY filename COLLATE NOCASE", (owner_id,)).fetchall()
        else:
            rows = db.execute("SELECT * FROM documents WHERE owner_id=? ORDER BY filename COLLATE NOCASE", (owner_id,)).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["filename"] = item["filename"].split("::", 1)[-1]
        result.append(item)
    return result


def save_pdf_image_extraction(source_filename: str, source_hash: str, status: str,
                              image_count: int, error: str = "") -> None:
    owner_id = _ACTIVE_USER_ID.get() or "legacy"
    with connect() as db:
        db.execute("""INSERT INTO pdf_image_extraction_runs(owner_id,source_filename,source_hash,status,image_count,error,updated_at)
            VALUES(?,?,?,?,?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(owner_id,source_filename,source_hash) DO UPDATE SET
            status=excluded.status,image_count=excluded.image_count,error=excluded.error,updated_at=CURRENT_TIMESTAMP""",
            (owner_id, source_filename, source_hash, status, image_count, error))


def get_pdf_image_extraction(source_filename: str, source_hash: str) -> dict[str, Any] | None:
    owner_id = _ACTIVE_USER_ID.get() or "legacy"
    with connect() as db:
        row = db.execute("""SELECT * FROM pdf_image_extraction_runs
            WHERE owner_id=? AND source_filename=? AND source_hash=?""",
            (owner_id, source_filename, source_hash)).fetchone()
    return dict(row) if row else None


def save_extracted_pdf_image(source_filename: str, source_hash: str, page_number: int,
                             image_index: int, image_hash: str, content: bytes,
                             mime_type: str, width: int, height: int) -> dict[str, Any]:
    owner_id = _ACTIVE_USER_ID.get() or "legacy"
    identity = f"{owner_id}\0{source_filename}\0{source_hash}\0{page_number}\0{image_hash}"
    photo_key = "pdfimg_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
    stored_filename = f"{photo_key}.png"
    PDF_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    image_path = PDF_IMAGE_DIR / stored_filename
    image_path.write_bytes(content)
    with connect() as db:
        db.execute("""INSERT OR IGNORE INTO extracted_pdf_images(
            owner_id,photo_key,source_filename,source_hash,page_number,image_index,image_hash,
            stored_filename,mime_type,width,height)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (owner_id, photo_key, source_filename, source_hash, page_number, image_index,
             image_hash, stored_filename, mime_type, width, height))
        row = db.execute("SELECT * FROM extracted_pdf_images WHERE owner_id=? AND photo_key=?",
                         (owner_id, photo_key)).fetchone()
    return dict(row)


def list_extracted_pdf_images(source_filename: str | None = None,
                              source_hash: str | None = None) -> list[dict[str, Any]]:
    owner_id = _ACTIVE_USER_ID.get() or "legacy"
    conditions, values = ["owner_id=?"], [owner_id]
    if source_filename is not None:
        conditions.append("source_filename=?")
        values.append(source_filename)
    if source_hash is not None:
        conditions.append("source_hash=?")
        values.append(source_hash)
    with connect() as db:
        rows = db.execute("SELECT * FROM extracted_pdf_images WHERE " + " AND ".join(conditions) +
                          " ORDER BY source_filename,page_number,image_index", values).fetchall()
    return [dict(row) for row in rows]


def get_extracted_pdf_image(photo_key: str) -> dict[str, Any] | None:
    owner_id = _ACTIVE_USER_ID.get() or "legacy"
    with connect() as db:
        row = db.execute("SELECT * FROM extracted_pdf_images WHERE owner_id=? AND photo_key=?",
                         (owner_id, photo_key)).fetchone()
    return dict(row) if row else None


def load_documents() -> list[tuple[str, bytes, str]]:
    result = []
    for row in list_documents():
        path = UPLOAD_DIR / row["stored_filename"]
        if path.is_file():
            result.append((row["filename"], path.read_bytes(), row["sha256"]))
    return result


def save_chunks(rows: list[dict[str, Any]]) -> None:
    owner_id = _ACTIVE_USER_ID.get() or "legacy"
    with connect() as db:
        db.execute("DELETE FROM chunks WHERE owner_id=?", (owner_id,))
        db.executemany("""INSERT INTO chunks(source,text,vector,source_filename,source_hash,page_number,chunk_index,owner_id)
                          VALUES(?,?,?,?,?,?,?,?)""",
                       [(r["source_filename"], r["text"], json.dumps(r["vector"]),
                         r["source_filename"], r["source_hash"], r.get("page_number"), r["chunk_index"], owner_id)
                        for r in rows])


def load_chunks() -> list[dict[str, Any]]:
    owner_id = _ACTIVE_USER_ID.get() or "legacy"
    with connect() as db:
        if owner_id == "legacy" or _LEGACY_OWNER.get():
            rows = db.execute("SELECT * FROM chunks WHERE owner_id IN (?, 'legacy') ORDER BY id", (owner_id,)).fetchall()
        else:
            rows = db.execute("SELECT * FROM chunks WHERE owner_id=? ORDER BY id", (owner_id,)).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        if not item["source_filename"]:
            item["source_filename"] = item.get("source") or "Unknown source"
        item["vector"] = json.loads(item["vector"] or "[]")
        result.append(item)
    return result


def save_campaign(event_name: str, platform: str, brief: str, content: str,
                  status: str = "Draft", evidence: list[dict[str, Any]] | None = None) -> int:
    saved_evidence = [{key: item[key] for key in ("source_filename", "page_number", "chunk_index", "text", "score") if key in item}
                      for item in (evidence or [])]
    owner_id = _ACTIVE_USER_ID.get() or "legacy"
    with connect() as db:
        cur = db.execute("INSERT INTO campaigns(event_name,platform,brief,content,status,evidence,owner_id) VALUES(?,?,?,?,?,?,?)",
                         (event_name, platform, brief, content, status,
                          json.dumps(saved_evidence, ensure_ascii=False), owner_id))
        return int(cur.lastrowid)


def update_campaign(campaign_id: int, content: str, status: str) -> None:
    owner_id = _ACTIVE_USER_ID.get() or "legacy"
    with connect() as db:
        db.execute("UPDATE campaigns SET content=?,status=? WHERE id=? AND owner_id=?", (content, status, campaign_id, owner_id))


def update_creative(campaign_id: int, creative_path: str, creative_format: str,
                    selected_media: list[str]) -> None:
    owner_id = _ACTIVE_USER_ID.get() or "legacy"
    with connect() as db:
        db.execute("UPDATE campaigns SET creative_path=?,creative_format=?,selected_media=? WHERE id=? AND owner_id=?",
                   (creative_path, creative_format, json.dumps(selected_media), campaign_id, owner_id))


def save_post_package(campaign_id: int, package: dict[str, Any]) -> None:
    """Persist the full visual package while keeping legacy first-slide columns current."""
    owner_id = _ACTIVE_USER_ID.get() or "legacy"
    slides = sorted(package.get("slides", []), key=lambda item: item.get("position", 0))
    first_path = slides[0].get("creative_path", "") if slides else ""
    selected_media = [item.get("photo_key", "") for item in slides if item.get("photo_key")]
    with connect() as db:
        db.execute("""UPDATE campaigns SET post_package=?,creative_path=?,creative_format=?,selected_media=?
            WHERE id=? AND owner_id=?""",
            (json.dumps(package, ensure_ascii=False), first_path,
             package.get("aspect", ""), json.dumps(selected_media, ensure_ascii=False),
             campaign_id, owner_id))


def update_campaign_selection(campaign_id: int, selected_media: list[str]) -> None:
    owner_id = _ACTIVE_USER_ID.get() or "legacy"
    with connect() as db:
        db.execute("UPDATE campaigns SET selected_media=? WHERE id=? AND owner_id=?",
                   (json.dumps(selected_media), campaign_id, owner_id))


def get_campaign(campaign_id: int | None) -> dict[str, Any] | None:
    if not campaign_id:
        return None
    owner_id = _ACTIVE_USER_ID.get() or "legacy"
    with connect() as db:
        row = db.execute("SELECT * FROM campaigns WHERE id=? AND owner_id=?", (campaign_id, owner_id)).fetchone()
    return dict(row) if row else None


def list_campaigns(limit: int = 50) -> list[dict[str, Any]]:
    owner_id = _ACTIVE_USER_ID.get() or "legacy"
    with connect() as db:
        return [dict(row) for row in db.execute("SELECT * FROM campaigns WHERE owner_id=? ORDER BY id DESC LIMIT ?", (owner_id, limit))]


def remove_legacy_demo_data() -> dict[str, int]:
    """One-time, narrowly scoped cleanup of the old bundled Green Futures fixture.

    Campaigns are only removed when both their event name and their saved evidence
    identify the deleted bundled sample document. Other user campaigns, even ones
    with the same event name, are retained. No tables or unrelated state are reset.
    """
    marker = "legacy_green_futures_fixture_removed_v1"
    demo_name = "Green Futures Summit 2026"
    sample_source = "sample_event.txt"
    removed_campaigns = 0
    removed_chunks = 0
    with connect() as db:
        if db.execute("SELECT 1 FROM app_state WHERE key=?", (marker,)).fetchone():
            return {"campaigns": 0, "chunks": 0}

        campaign_ids = []
        for row in db.execute("SELECT id,event_name,evidence FROM campaigns"):
            if row["event_name"] != demo_name:
                continue
            try:
                evidence = json.loads(row["evidence"] or "[]")
            except (TypeError, json.JSONDecodeError):
                continue
            is_fixture = any(
                item.get("source") == sample_source
                and (demo_name in item.get("text", "")
                     or "Riverside Convention Centre, Pune" in item.get("text", ""))
                for item in evidence if isinstance(item, dict)
            )
            if is_fixture:
                campaign_ids.append((row["id"],))
        if campaign_ids:
            db.executemany("DELETE FROM campaigns WHERE id=?", campaign_ids)
            removed_campaigns = len(campaign_ids)

        demo_chunk_markers = (
            "%EVENT: Green Futures Summit 2026%",
            "%Doors open at 9:00 AM. The keynote begins at 10:00 AM%",
        )
        chunk_ids = [
            (row[0],) for row in db.execute(
                "SELECT id FROM chunks WHERE source=? AND (text LIKE ? OR text LIKE ?)",
                (sample_source, *demo_chunk_markers))
        ]
        if chunk_ids:
            db.executemany("DELETE FROM chunks WHERE id=?", chunk_ids)
            removed_chunks = len(chunk_ids)

        # The old demo event was never needed as saved form data, but clear an exact
        # legacy copy if one exists; leave all other event state untouched.
        state_row = db.execute("SELECT value FROM app_state WHERE key='event_details'").fetchone()
        if state_row:
            try:
                event = json.loads(state_row["value"])
            except (TypeError, json.JSONDecodeError):
                event = None
            if isinstance(event, dict) and (
                event.get("name") == demo_name
                and event.get("date") == "14 November 2026"
                and event.get("venue") == "Riverside Convention Centre, Pune"
                and event.get("description") == "A one-day gathering about practical climate action for resilient cities."
            ):
                db.execute("DELETE FROM app_state WHERE key='event_details'")

        db.execute("INSERT INTO app_state(key,value) VALUES(?,?)", (marker, json.dumps(True)))
    return {"campaigns": removed_campaigns, "chunks": removed_chunks}
