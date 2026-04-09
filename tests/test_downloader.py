from unittest.mock import patch, MagicMock
from pathlib import Path
from mosaic.kb.downloader import download_video, VideoMeta


def test_download_video_returns_meta(tmp_dir):
    fake_info = {
        "id": "abc123",
        "title": "Test Video",
        "duration": 600,
        "channel": "Test Channel",
        "webpage_url": "https://youtube.com/watch?v=abc123",
    }
    fake_video_path = tmp_dir / "abc123.mp4"
    fake_video_path.touch()

    with patch("mosaic.kb.downloader.yt_dlp.YoutubeDL") as MockDL:
        instance = MockDL.return_value.__enter__.return_value
        instance.extract_info.return_value = fake_info
        instance.prepare_filename.return_value = str(fake_video_path)

        meta = download_video("https://youtube.com/watch?v=abc123", tmp_dir)

    assert isinstance(meta, VideoMeta)
    assert meta.video_id == "abc123"
    assert meta.title == "Test Video"
    assert meta.duration_seconds == 600
    assert meta.local_path == fake_video_path


def test_download_video_skips_if_exists(tmp_dir):
    existing = tmp_dir / "abc123.mp4"
    existing.touch()

    with patch("mosaic.kb.downloader.yt_dlp.YoutubeDL") as MockDL:
        instance = MockDL.return_value.__enter__.return_value
        instance.extract_info.return_value = {
            "id": "abc123", "title": "T", "duration": 100,
            "channel": "C", "webpage_url": "https://youtube.com/watch?v=abc123"
        }
        instance.prepare_filename.return_value = str(existing)

        meta = download_video("https://youtube.com/watch?v=abc123", tmp_dir)

    # yt-dlp download should not be called — file already exists
    instance.download.assert_not_called()
    assert meta.local_path == existing
