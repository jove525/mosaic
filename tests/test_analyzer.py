from unittest.mock import patch, MagicMock
from pathlib import Path
from mosaic.kb.analyzer import analyze_video, VideoInsights
from mosaic.kb.downloader import VideoMeta
from mosaic.kb.extractor import FrameSet, TranscriptSegment


def _make_meta(tmp_dir):
    return VideoMeta(
        video_id="test123",
        title="Why Incentives Rule Everything",
        duration_seconds=600,
        channel="Test Channel",
        url="https://youtube.com/watch?v=test123",
        local_path=tmp_dir / "test123.mp4",
    )


def _make_frameset(tmp_dir, sample_frames):
    return FrameSet(
        video_path=tmp_dir / "test123.mp4",
        frame_dir=tmp_dir / "frames",
        frame_paths=sample_frames,
    )


def _make_transcript():
    return [
        TranscriptSegment(0.0, 3.5, "In 2008, something remarkable happened."),
        TranscriptSegment(3.5, 7.0, "Seven hundred billion dollars vanished overnight."),
    ]


def test_analyze_video_returns_insights(tmp_dir, sample_frames):
    meta = _make_meta(tmp_dir)
    frameset = _make_frameset(tmp_dir, sample_frames)
    transcript = _make_transcript()

    fake_response_text = """{
        "hook": {"structure": "statistic_shock", "technique": "opens with a number", "why_it_works": "numbers create immediate credibility", "duration_seconds": 28},
        "story_arc": {"shape": "problem_reveal_reframe", "phases": []},
        "emotional_beats": [],
        "visual_patterns": [],
        "pacing": {"avg_cut_interval_seconds": 4.0, "music_shift_points": [], "silence_moments": []},
        "narration_style": {"avg_sentence_length_words": 10, "rhythm": "punchy", "notable_techniques": []},
        "key_insights": ["Opens with a shocking statistic to create immediate stakes"]
    }"""

    with patch("mosaic.kb.analyzer.anthropic.Anthropic") as MockClient:
        mock_client = MockClient.return_value
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=fake_response_text)]
        mock_client.messages.create.return_value = mock_msg

        insights = analyze_video(meta, frameset, transcript, api_key="fake")

    assert isinstance(insights, VideoInsights)
    assert insights.video_id == "test123"
    assert insights.channel == "Test Channel"
    assert insights.hook["structure"] == "statistic_shock"
    assert len(insights.key_insights) == 1


def test_analyze_video_includes_channel_context(tmp_dir, sample_frames):
    meta = _make_meta(tmp_dir)
    frameset = _make_frameset(tmp_dir, sample_frames)
    transcript = _make_transcript()

    with patch("mosaic.kb.analyzer.anthropic.Anthropic") as MockClient:
        mock_client = MockClient.return_value
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text='{"hook":{},"story_arc":{},"emotional_beats":[],"visual_patterns":[],"pacing":{},"narration_style":{},"key_insights":[]}')]
        mock_client.messages.create.return_value = mock_msg

        analyze_video(meta, frameset, transcript, api_key="fake", channel_focus="storytelling and emotional arc")

    call_kwargs = mock_client.messages.create.call_args
    prompt_text = str(call_kwargs)
    assert "storytelling and emotional arc" in prompt_text
