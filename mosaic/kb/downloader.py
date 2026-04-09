from dataclasses import dataclass
from pathlib import Path
import yt_dlp
from mosaic.utils.fs import ensure_dir


@dataclass
class VideoMeta:
    video_id: str
    title: str
    duration_seconds: int
    channel: str
    url: str
    local_path: Path


def download_video(url: str, output_dir: Path, cookies_file: Path | None = None) -> VideoMeta:
    """Download a video by URL. Skip if already downloaded. Return VideoMeta."""
    ensure_dir(output_dir)

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }
    if cookies_file and cookies_file.exists():
        ydl_opts["cookiefile"] = str(cookies_file)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        video_id = info["id"]
        local_path = Path(ydl.prepare_filename(info))

        if not local_path.exists():
            ydl.download([url])

    return VideoMeta(
        video_id=video_id,
        title=info["title"],
        duration_seconds=int(info.get("duration", 0)),
        channel=info.get("channel", ""),
        url=info.get("webpage_url", url),
        local_path=local_path,
    )
