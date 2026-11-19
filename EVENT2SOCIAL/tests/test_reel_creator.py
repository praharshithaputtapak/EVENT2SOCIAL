
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import reel_creator
from reel_creator import create_vertical_video


FFMPEG = os.getenv("FFMPEG_PATH", "").strip() or shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


class ReelCreatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not FFMPEG or not Path(FFMPEG).is_file():
            raise unittest.SkipTest("FFmpeg is not installed")

    def make_photo(self, folder: Path) -> Path:
        photo = folder / "photo.png"
        Image.new("RGB", (240, 320), "navy").save(photo)
        return photo

    def make_video(self, folder: Path, photo: Path) -> Path:
        source_video = folder / "source.mp4"
        subprocess.run([FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-loop", "1", "-t", "2", "-i", str(photo), "-vf", "scale=240:320,fps=15",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source_video)],
            check=True, capture_output=True, timeout=60)
        return source_video

    def render_and_check(self, root: Path, segments, *, expected_resolution="180x320", **kwargs):
        output = root / "result.mp4"
        progress = []
        video, cover = create_vertical_video(segments, output, width=180, height=320,
            transition_seconds=0.2, progress_callback=progress.append, **kwargs)
        self.assertTrue(video.is_file() and video.stat().st_size > 1000)
        self.assertTrue(cover.is_file() and cover.stat().st_size > 1000)
        self.assertTrue(progress)
        self.assertTrue(all(0 <= value <= 0.99 for value in progress))
        if FFPROBE:
            info = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", str(video)],
                check=True, capture_output=True, text=True)
            self.assertEqual(info.stdout.strip(), expected_resolution)

    def test_image_only_export(self):
        with tempfile.TemporaryDirectory(prefix="event2social-reel-image-test-") as folder:
            root = Path(folder)
            photo = self.make_photo(root)
            self.render_and_check(root, [{"path": str(photo), "kind": "image", "duration": 2}],
                                  overlays=["Image scene"], cover_text="Image cover")

    def test_video_only_export(self):
        with tempfile.TemporaryDirectory(prefix="event2social-reel-video-test-") as folder:
            root = Path(folder)
            source_video = self.make_video(root, self.make_photo(root))
            self.render_and_check(root, [{"path": str(source_video), "kind": "video", "duration": 2}],
                                  overlays=["Video scene"], cover_text="Video cover")

    def test_mixed_media_export_with_title_cards_and_fades(self):
        with tempfile.TemporaryDirectory(prefix="event2social-reel-mixed-test-") as folder:
            root = Path(folder)
            photo = self.make_photo(root)
            source_video = self.make_video(root, photo)
            self.render_and_check(root, [
                {"path": str(photo), "kind": "image", "duration": 2},
                {"path": str(source_video), "kind": "video", "duration": 2},
            ], overlays=["Photo scene", "Video scene"], opening_title="Community Day",
               closing_text="Thank you", cover_text="Community Day")

    def test_cloud_profile_uses_720_by_1280_resolution(self):
        with tempfile.TemporaryDirectory(prefix="event2social-reel-cloud-test-") as folder:
            root = Path(folder)
            photo = self.make_photo(root)
            source_video = self.make_video(root, photo)
            self.render_and_check(root, [
                {"path": str(photo), "kind": "image", "duration": 2},
                {"path": str(source_video), "kind": "video", "duration": 2},
            ], expected_resolution="720x1280", profile="cloud", overlays=["Photo", "Video"])

    def test_local_profile_keeps_1080_by_1920_resolution(self):
        with tempfile.TemporaryDirectory(prefix="event2social-reel-local-profile-test-") as folder:
            root = Path(folder)
            photo = self.make_photo(root)
            output = root / "result.mp4"
            video, cover = create_vertical_video(
                [{"path": str(photo), "kind": "image", "duration": 2}], output, profile="local")
            self.assertTrue(video.is_file() and video.stat().st_size > 1000)
            self.assertTrue(cover.is_file() and cover.stat().st_size > 1000)
            if FFPROBE:
                info = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", str(video)],
                    check=True, capture_output=True, text=True)
                self.assertEqual(info.stdout.strip(), "1080x1920")

    def test_failed_render_cleans_temporary_outputs(self):
        with tempfile.TemporaryDirectory(prefix="event2social-reel-failure-test-") as folder:
            root = Path(folder)
            bad_source = root / "not-a-video.mp4"
            bad_source.write_bytes(b"not a video")
            output = root / "result.mp4"
            with self.assertRaises(RuntimeError):
                create_vertical_video([{"path": str(bad_source), "kind": "video", "duration": 2}], output,
                                      width=180, height=320)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_name("result-cover.jpg").exists())
            self.assertEqual(list(root.glob("event2social-reel-*")), [])

    def test_failed_publish_restores_previous_valid_outputs(self):
        with tempfile.TemporaryDirectory(prefix="event2social-reel-publish-test-") as folder:
            root = Path(folder)
            photo = self.make_photo(root)
            output = root / "result.mp4"
            cover = root / "result-cover.jpg"
            output.write_bytes(b"previous-video")
            cover.write_bytes(b"previous-cover")
            original_replace = os.replace

            def fail_video_publish(source, destination):
                if Path(source).name == "render.mp4" and Path(destination) == output:
                    raise OSError("simulated final video publish failure")
                return original_replace(source, destination)

            with patch.object(reel_creator.os, "replace", side_effect=fail_video_publish):
                with self.assertRaisesRegex(OSError, "simulated final video publish failure"):
                    create_vertical_video(
                        [{"path": str(photo), "kind": "image", "duration": 2}], output,
                        width=180, height=320)
            self.assertEqual(output.read_bytes(), b"previous-video")
            self.assertEqual(cover.read_bytes(), b"previous-cover")
            self.assertEqual(list(root.glob("event2social-reel-*")), [])


if __name__ == "__main__":
    unittest.main()
