from unittest.mock import patch, MagicMock
from mosaic.kb.extractor import extract_frames, FrameSet, transcribe_video, TranscriptSegment


def test_extract_frames_returns_frameset(tiny_video, tmp_dir):
    frame_dir = tmp_dir / "frames"
    result = extract_frames(tiny_video, frame_dir, interval_seconds=2)

    assert isinstance(result, FrameSet)
    assert len(result.frame_paths) >= 2  # 5s video at 2s interval = 2-3 frames
    assert all(p.exists() for p in result.frame_paths)
    assert all(p.suffix == ".png" for p in result.frame_paths)
    assert result.video_path == tiny_video


def test_extract_frames_skips_if_already_extracted(tiny_video, tmp_dir):
    frame_dir = tmp_dir / "frames"
    result1 = extract_frames(tiny_video, frame_dir, interval_seconds=2)
    result2 = extract_frames(tiny_video, frame_dir, interval_seconds=2)

    assert len(result1.frame_paths) == len(result2.frame_paths)


def test_transcribe_video_returns_segments(tiny_video):
    fake_result = {
        "segments": [
            {"start": 0.0, "end": 2.5, "text": " Hello world."},
            {"start": 2.5, "end": 5.0, "text": " This is a test."},
        ]
    }
    with patch("mosaic.kb.extractor.whisper") as mock_whisper:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = fake_result
        mock_whisper.load_model.return_value = mock_model

        segments = transcribe_video(tiny_video, model_name="base")

    assert len(segments) == 2
    assert isinstance(segments[0], TranscriptSegment)
    assert segments[0].start == 0.0
    assert segments[0].end == 2.5
    assert segments[0].text == "Hello world."  # leading space stripped
