"""Extract, persist, and reuse embedded PDF images without duplicate work."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import storage
from media_processing import extract_pdf_embedded_images


def extract_and_store_pdf_images(source_filename: str, source_hash: str,
                                 source_path: str | Path) -> dict[str, Any]:
    """Extract a PDF version once and persist images with source page metadata."""
    assets = storage.list_extracted_pdf_images(source_filename, source_hash)
    run = storage.get_pdf_image_extraction(source_filename, source_hash)
    files_present = all((storage.PDF_IMAGE_DIR / item["stored_filename"]).is_file() for item in assets)
    if run and run["status"] in {"complete", "no_images"} and files_present:
        return {"skipped": True, "status": run["status"], "images": assets,
                "errors": [line for line in run.get("error", "").splitlines() if line]}

    try:
        extracted, errors = extract_pdf_embedded_images(Path(source_path).read_bytes())
        for image in extracted:
            storage.save_extracted_pdf_image(
                source_filename, source_hash, image["page_number"], image["image_index"],
                image["sha256"], image["content"], image["mime_type"], image["width"], image["height"])
        assets = storage.list_extracted_pdf_images(source_filename, source_hash)
        status = "complete" if assets and not errors else "partial" if assets else "failed" if errors else "no_images"
        error_text = "\n".join(errors[:20])
        storage.save_pdf_image_extraction(source_filename, source_hash, status, len(assets), error_text)
        return {"skipped": False, "status": status, "images": assets, "errors": errors}
    except Exception as exc:
        storage.save_pdf_image_extraction(source_filename, source_hash, "failed", len(assets), str(exc))
        raise
