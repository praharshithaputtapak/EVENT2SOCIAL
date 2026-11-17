import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from PIL import Image

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

    def render_and_check(self, root: Path, segments, **kwargs):
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
            self.assertEqual(info.stdout.strip(), "180x320")

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


if __name__ == "__main__":
    unittest.main()
