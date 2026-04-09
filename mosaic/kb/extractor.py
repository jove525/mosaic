import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from mosaic.utils.fs import ensure_dir


@dataclass
class FrameSet:
    video_path: Path
    frame_dir: Path
    frame_paths: list[Path] = field(default_factory=list)


def extract_frames(video_path: Path, frame_dir: Path, interval_seconds: int = 8) -> FrameSet:
    """Extract one frame every interval_seconds from video_path into frame_dir."""
    video_frame_dir = frame_dir / video_path.stem
    ensure_dir(video_frame_dir)

    existing = sorted(video_frame_dir.glob("frame_*.png"))
    if existing:
        return FrameSet(video_path=video_path, frame_dir=video_frame_dir, frame_paths=existing)

    output_pattern = str(video_frame_dir / "frame_%04d.png")
    subprocess.run([
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps=1/{interval_seconds}",
        "-vsync", "vfr",
        output_pattern,
        "-y", "-loglevel", "error"
    ], check=True)

    frame_paths = sorted(video_frame_dir.glob("frame_*.png"))
    return FrameSet(video_path=video_path, frame_dir=video_frame_dir, frame_paths=frame_paths)
