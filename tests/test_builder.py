from unittest.mock import patch, MagicMock
from pathlib import Path
from mosaic.kb.builder import KBBuilder
from mosaic.kb.downloader import VideoMeta
from mosaic.kb.extractor import FrameSet, TranscriptSegment
from mosaic.kb.analyzer import VideoInsights
from mosaic.kb.store import KBStore


def _fake_meta(tmp_dir, video_id="vid1"):
    p = tmp_dir / f"{video_id}.mp4"
    p.touch()
    return VideoMeta(
        video_id=video_id, title="Test", duration_seconds=300,
        channel="wendover", url="https://example.com", local_path=p
    )


def test_builder_processes_video(tmp_path):
    store = KBStore(persist_dir=tmp_path / "kb")
    builder = KBBuilder(
        store=store,
        clips_dir=tmp_path / "clips",
        frames_dir=tmp_path / "frames",
        api_key="fake",
        whisper_model="base",
        frame_interval=8,
    )

    fake_meta = _fake_meta(tmp_path)
    fake_frameset = FrameSet(video_path=fake_meta.local_path, frame_dir=tmp_path, frame_paths=[])
    fake_transcript = [TranscriptSegment(0.0, 3.0, "Test narration.")]
    fake_insights = VideoInsights(
        video_id="vid1", channel="wendover", title="Test",
        hook={}, story_arc={}, emotional_beats=[], visual_patterns=[],
        pacing={}, narration_style={}, key_insights=["Test insight one"]
    )

    with patch("mosaic.kb.builder.download_video", return_value=fake_meta), \
         patch("mosaic.kb.builder.extract_frames", return_value=fake_frameset), \
         patch("mosaic.kb.builder.transcribe_video", return_value=fake_transcript), \
         patch("mosaic.kb.builder.analyze_video", return_value=fake_insights):

        builder.process_video("https://example.com", channel_key="wendover")

    assert store.is_processed("vid1")


def test_builder_skips_already_processed(tmp_path):
    store = KBStore(persist_dir=tmp_path / "kb")
    builder = KBBuilder(
        store=store, clips_dir=tmp_path / "clips", frames_dir=tmp_path / "frames",
        api_key="fake", whisper_model="base", frame_interval=8,
    )

    # Pre-populate store so vid1 is already processed
    store.add(VideoInsights(
        video_id="vid1", channel="wendover", title="T",
        hook={}, story_arc={}, emotional_beats=[], visual_patterns=[],
        pacing={}, narration_style={}, key_insights=["existing insight"]
    ))

    fake_meta = _fake_meta(tmp_path, "vid1")

    with patch("mosaic.kb.builder.download_video", return_value=fake_meta), \
         patch("mosaic.kb.builder.analyze_video") as mock_analyze:

        builder.process_video("https://example.com", channel_key="wendover")

    mock_analyze.assert_not_called()


def test_builder_process_channel(tmp_path):
    store = KBStore(persist_dir=tmp_path / "kb")
    builder = KBBuilder(
        store=store, clips_dir=tmp_path / "clips", frames_dir=tmp_path / "frames",
        api_key="fake", whisper_model="base", frame_interval=8,
    )

    urls = ["https://example.com/1", "https://example.com/2"]

    with patch.object(builder, "process_video") as mock_process:
        builder.process_channel(urls, channel_key="wendover")
        assert mock_process.call_count == 2
