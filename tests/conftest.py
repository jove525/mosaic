import pytest
from pathlib import Path
import tempfile
import subprocess


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def tiny_video(tmp_dir):
    """Create a 5-second silent test video using ffmpeg."""
    video_path = tmp_dir / "test.mp4"
    subprocess.run([
        "ffmpeg", "-f", "lavfi", "-i", "color=c=black:s=640x360:d=5",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-shortest", "-y", str(video_path)
    ], check=True, capture_output=True)
    return video_path


@pytest.fixture
def sample_transcript():
    return [
        {"start": 0.0, "end": 3.5, "text": "In 2008, something remarkable happened."},
        {"start": 3.5, "end": 7.0, "text": "Seven hundred billion dollars vanished overnight."},
        {"start": 7.0, "end": 11.2, "text": "But the real story is what happened next."},
    ]


@pytest.fixture
def sample_frames(tmp_dir):
    """Create 3 tiny PNG frames."""
    from PIL import Image
    frames = []
    for i in range(3):
        img = Image.new("RGB", (64, 36), color=(i * 80, 0, 0))
        path = tmp_dir / f"frame_{i:04d}.png"
        img.save(path)
        frames.append(path)
    return frames
