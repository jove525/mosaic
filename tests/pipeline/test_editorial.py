import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from mosaic.pipeline.editorial import (
    parse_claude_segments,
    build_watch_prompt,
    trim_segment,
)


def test_parse_claude_segments_valid():
    response = json.dumps({
        "usable_segments": [
            {"start": 10.5, "end": 24.0, "description": "civilians at booth", "relevance": "direct match"},
            {"start": 55.0, "end": 70.0, "description": "woman handing money", "relevance": "supports line"}
        ],
        "verdict": "useful",
        "why": "Contains relevant WWII bond drive footage"
    })
    result = parse_claude_segments(response)
    assert result["verdict"] == "useful"
    assert len(result["usable_segments"]) == 2
    assert result["usable_segments"][0]["start"] == 10.5


def test_parse_claude_segments_not_useful():
    response = json.dumps({
        "usable_segments": [],
        "verdict": "not_useful",
        "why": "Japanese army footage, unrelated to bond drives"
    })
    result = parse_claude_segments(response)
    assert result["verdict"] == "not_useful"
    assert result["usable_segments"] == []


def test_parse_claude_segments_malformed():
    result = parse_claude_segments("this is not json {{{")
    assert result["verdict"] == "not_useful"
    assert result["usable_segments"] == []


def test_build_watch_prompt_contains_narration():
    prompt = build_watch_prompt(
        narration_text="Your grandparents funded the war.",
        emotion="TENSION",
        visual_description="Civilians at bond booth.",
        transcript_text="[0.0s-5.0s] Welcome to the bond drive.",
        video_duration=300.0,
    )
    assert "Your grandparents funded the war." in prompt
    assert "TENSION" in prompt
    assert "bond drive" in prompt


def test_trim_segment_calls_ffmpeg(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake")
    dest = tmp_path / "out.mp4"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        dest.write_bytes(b"fake output")  # simulate ffmpeg creating the file
        from mosaic.pipeline.editorial import trim_segment
        result = trim_segment(source, dest, start=10.5, end=24.0)
        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert "-ss" in cmd
        assert "10.5" in cmd


def test_watch_candidate_not_useful_on_api_error(tmp_path):
    """watch_candidate returns not_useful verdict when Claude call fails."""
    from mosaic.pipeline.editorial import watch_candidate
    # Create a minimal fake video file (ffmpeg will fail, extractor will fail gracefully)
    fake_video = tmp_path / "fake.mp4"
    fake_video.write_bytes(b"not a real video")
    line = {
        "narration_text": "Test narration.",
        "emotion": "TENSION",
        "visual_description": "Test visual.",
    }
    result = watch_candidate(fake_video, line, frame_dir=tmp_path / "frames")
    # Should return not_useful gracefully, never raise
    assert result["verdict"] == "not_useful"


def test_run_editorial_all_gaps(tmp_path):
    """When no candidates exist for any line, all entries are flagged needs_generated_visual."""
    import json
    from mosaic.pipeline.editorial import run_editorial

    # Write raw_candidates.json with no downloaded candidates
    raw = [
        {
            "narration_line_ref": 1,
            "narration_text": "Test narration.",
            "emotion": "TENSION",
            "visual_description": "Some visual.",
            "candidates": []
        }
    ]
    (tmp_path / "raw_candidates.json").write_text(json.dumps(raw), encoding="utf-8")
    (tmp_path / "clips").mkdir()
    (tmp_path / "clips" / "raw").mkdir()
    (tmp_path / "clips" / "final").mkdir()

    cache_dir = tmp_path / "cache"
    result = run_editorial(tmp_path, cache_dir=cache_dir)
    assert result == {"sourced": 0, "gaps": 1, "total": 1}

    manifest = json.loads((tmp_path / "clip_manifest.json").read_text())
    assert len(manifest) == 1
    assert manifest[0]["needs_generated_visual"] is True
    assert manifest[0]["cuts"] == []
