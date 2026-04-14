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
