import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from media_processing import _cached_photo_metrics, rank_photos, select_recommended_photos


class PhotoRecommendationTests(unittest.TestCase):
    def test_recommendations_exclude_duplicates_and_weak_images_and_cap_carousel(self):
        ranked = [{"filename": f"photo-{index}", "quality": 0.9 - index * 0.01,
                   "score": 0.9 - index * 0.01, "duplicate": index in {2, 9}}
                  for index in range(48)]
        chosen = select_recommended_photos(ranked, max_slides=8)
        self.assertEqual(len(chosen), 8)
        self.assertFalse(any(item["duplicate"] for item in chosen))
        self.assertNotIn("photo-2", [item["filename"] for item in chosen])

    def test_local_quality_metrics_are_cached_across_rank_calls(self):
        with tempfile.TemporaryDirectory(prefix="event2social-photo-rank-") as folder:
            paths = []
            for index in range(3):
                path = Path(folder) / f"photo-{index}.png"
                image = Image.new("RGB", (180, 150), (50 + index * 40, 110, 160))
                draw = ImageDraw.Draw(image)
                draw.rectangle((20 + index, 20, 80 + index, 90), fill="white")
                image.save(path)
                paths.append(path)
            photos = [{"filename": p.name, "photo_key": p.name, "sha256": f"hash-{i}",
                       "_path": str(p)} for i, p in enumerate(paths)]
            _cached_photo_metrics.cache_clear()
            first = rank_photos(photos, "event workshop", {})
            before = _cached_photo_metrics.cache_info().hits
            second = rank_photos(photos, "event workshop", {})
            after = _cached_photo_metrics.cache_info().hits
            self.assertEqual([item["filename"] for item in first], [item["filename"] for item in second])
            self.assertEqual(after - before, len(paths))


if __name__ == "__main__":
    unittest.main()
