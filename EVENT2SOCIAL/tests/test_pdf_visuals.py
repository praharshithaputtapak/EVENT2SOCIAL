import hashlib
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import pdf_visuals
import storage


class PdfVisualAssetTests(unittest.TestCase):
    def test_image_records_keep_pdf_page_and_repeat_processing_is_skipped(self):
        image_buffer = BytesIO()
        Image.new("RGB", (320, 240), "navy").save(image_buffer, format="PNG")
        image_bytes = image_buffer.getvalue()
        record = {"page_number": 3, "image_index": 1, "content": image_bytes,
                  "mime_type": "image/png", "width": 320, "height": 240,
                  "sha256": hashlib.sha256(image_bytes).hexdigest()}

        with tempfile.TemporaryDirectory(prefix="event2social-pdf-visual-") as folder:
            root = Path(folder)
            db_path = root / "test.db"
            upload_dir = root / "uploads"
            pdf_dir = upload_dir / "pdf_extracted_images"
            pdf_path = root / "kb.pdf"
            pdf_path.write_bytes(b"test-pdf")
            with patch.object(storage, "DATA_DIR", root), \
                 patch.object(storage, "DB_PATH", db_path), \
                 patch.object(storage, "UPLOAD_DIR", upload_dir), \
                 patch.object(storage, "PDF_IMAGE_DIR", pdf_dir), \
                 patch.object(pdf_visuals, "extract_pdf_embedded_images", return_value=([record], [])) as extractor:
                first = pdf_visuals.extract_and_store_pdf_images("kb.pdf", "pdf-hash", pdf_path)
                second = pdf_visuals.extract_and_store_pdf_images("kb.pdf", "pdf-hash", pdf_path)
                saved = storage.list_extracted_pdf_images("kb.pdf", "pdf-hash")

            self.assertFalse(first["skipped"])
            self.assertTrue(second["skipped"])
            self.assertEqual(extractor.call_count, 1)
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["page_number"], 3)
            self.assertEqual(saved[0]["source_filename"], "kb.pdf")
            self.assertTrue((pdf_dir / saved[0]["stored_filename"]).is_file())


if __name__ == "__main__":
    unittest.main()
