import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import storage
from social_package import build_package_zip, edit_slide_order, extract_hashtags, render_caption_file


class SocialPackageTests(unittest.TestCase):
    def test_slide_move_remove_and_replace_preserve_order(self):
        original = ["a", "b", "c"]
        self.assertEqual(edit_slide_order(original, "up", 2), ["a", "c", "b"])
        self.assertEqual(edit_slide_order(original, "down", 0), ["b", "a", "c"])
        self.assertEqual(edit_slide_order(original, "remove", 1), ["a", "c"])
        self.assertEqual(edit_slide_order(original, "replace", 1, "d"), ["a", "d", "c"])
        self.assertEqual(edit_slide_order(original, "replace", 1, "a"), ["a", "c"])
        self.assertEqual(original, ["a", "b", "c"])

    def test_caption_file_includes_platform_caption_tags_cta_and_slide_sources(self):
        package = {
            "platform": "Instagram", "post_type": "Multiple Image / Carousel Post",
            "aspect": "Portrait (4:5)", "caption": "A confirmed event caption #OpenDay #openDay",
            "hashtags": ["#OpenDay", "#Community"], "cta": "Register today",
            "caption_adaptation": "Join our event.",
            "slides": [{"position": 0, "display_name": "kb.pdf · page 2 · image 1",
                        "slide_text": "Meet the community"}],
        }
        text = render_caption_file(package)
        self.assertIn("Platform: Instagram", text)
        self.assertIn("#OpenDay #Community", text)
        self.assertIn("Register today", text)
        self.assertIn("Slide 01: kb.pdf · page 2 · image 1", text)
        self.assertEqual(extract_hashtags("#OpenDay #openday #Art_Week"), ["#OpenDay", "#Art_Week"])

    def test_zip_contains_each_slide_caption_and_manifest(self):
        with tempfile.TemporaryDirectory(prefix="event2social-package-") as folder:
            root = Path(folder)
            paths = []
            for index, color in enumerate(("red", "blue"), 1):
                path = root / f"slide-{index}.jpg"
                Image.new("RGB", (80, 100), color).save(path, format="JPEG", quality=95)
                paths.append(path)
            package = {"campaign_id": 17, "platform": "Instagram", "post_type": "Multiple Image / Carousel Post",
                       "aspect": "Portrait (4:5)", "caption": "Event day", "hashtags": ["#Event"],
                       "cta": "Join us", "caption_adaptation": "Event day",
                       "slides": [{"position": i, "photo_key": f"photo-{i}", "display_name": f"photo {i}",
                                   "slide_text": "", "creative_path": str(path), "original_path": str(path)}
                                  for i, path in enumerate(paths)]}
            archive_bytes = build_package_zip(package)
            with zipfile.ZipFile(__import__("io").BytesIO(archive_bytes)) as zipped:
                self.assertEqual(zipped.namelist(), ["images/slide-01.jpg", "originals/original-01.jpg",
                    "images/slide-02.jpg", "originals/original-02.jpg", "caption.txt", "package.json"])
                self.assertIn("#Event", zipped.read("caption.txt").decode("utf-8"))
                self.assertEqual(json.loads(zipped.read("package.json"))["campaign_id"], 17)
                self.assertGreater(len(zipped.read("images/slide-01.jpg")), 100)

    def test_campaign_package_storage_is_additive_and_keeps_first_slide_legacy_fields(self):
        with tempfile.TemporaryDirectory(prefix="event2social-package-db-") as folder:
            root = Path(folder)
            with patch.object(storage, "DATA_DIR", root), patch.object(storage, "DB_PATH", root / "test.db"):
                campaign_id = storage.save_campaign("Event", "Instagram", "{}", "Caption", "Draft", [])
                first = root / "first.jpg"
                first.write_bytes(b"local test image")
                package = {"post_type": "Multiple Image / Carousel Post", "aspect": "Square (1:1)",
                           "slides": [{"position": 0, "photo_key": "photo-a", "creative_path": str(first)},
                                      {"position": 1, "photo_key": "photo-b", "creative_path": "second.jpg"}]}
                storage.save_post_package(campaign_id, package)
                saved = storage.get_campaign(campaign_id)
                self.assertEqual(json.loads(saved["post_package"])["post_type"], package["post_type"])
                self.assertEqual(saved["creative_path"], str(first))
                self.assertEqual(json.loads(saved["selected_media"]), ["photo-a", "photo-b"])


if __name__ == "__main__":
    unittest.main()
