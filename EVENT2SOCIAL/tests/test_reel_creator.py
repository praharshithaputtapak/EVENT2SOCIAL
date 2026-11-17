import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from reel_creator import create_vertical_video


class ReelCreatorTests(unittest.TestCase):
    def test_photo_and_video_segments_export_vertical_mp4_and_titled_cover(self):
        if not shutil.which("ffmpeg") and not __import__("os").getenv("FFMPEG_PATH"):
            self.skipTest("FFmpeg is not installed")
        with tempfile.TemporaryDirectory(prefix="event2social-reel-test-") as folder:
            root = Path(folder)
            photo = root / "photo.png"
            Image.new("RGB", (240, 320), "navy").save(photo)
            source_video = root / "source.mp4"
            ffmpeg = shutil.which("ffmpeg") or __import__("os").environ["FFMPEG_PATH"]
            subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-loop", "1", "-t", "2", "-i", str(photo), "-vf", "scale=240:320,fps=15",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source_video)],
                check=True, capture_output=True, timeout=60)
            output = root / "result.mp4"
            progress = []
            video, cover = create_vertical_video([
                {"path": str(photo), "kind": "image", "duration": 2},
                {"path": str(source_video), "kind": "video", "duration": 2},
            ], output, overlays=["Event highlights", "Join the event"],
                width=180, height=320, transition_seconds=0.2, cover_text="Community Day",
                opening_title="Community Day", closing_text="Thank you", progress_callback=progress.append)
            self.assertTrue(video.is_file() and video.stat().st_size > 1000)
            self.assertTrue(cover.is_file() and cover.stat().st_size > 1000)
            self.assertTrue(progress)
            self.assertTrue(all(0 <= value <= 0.99 for value in progress))
            probe = shutil.which("ffprobe")
            if probe:
                info = subprocess.run([probe, "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", str(video)],
                    check=True, capture_output=True, text=True)
                self.assertEqual(info.stdout.strip(), "180x320")


if __name__ == "__main__":
    unittest.main()
