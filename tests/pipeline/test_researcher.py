import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from mosaic.pipeline.researcher import (
    build_researcher_prompt,
    parse_brief_angle,
    run_researcher,
)

def test_build_researcher_prompt_contains_topic_and_kb():
    profile = {
        "north_star": "make the viewer distrust something they trusted before",
        "domain": "economics, incentives",
        "arc_template": "surface_reality → hidden_incentive → reframe",
        "core_thesis": "Show me the incentive and I'll show you the outcome",
        "channel_north_star": "Create a video so compelling...",
    }
    kb_insights = [
        {"insight": "Johnny Harris opens with a question that has no obvious answer", "channel": "johnny_harris"},
    ]
    prompt = build_researcher_prompt("war-financing", profile, kb_insights, [])
    # north_star goes in the system prompt; user prompt contains topic + KB insights
    assert "war-financing" in prompt
    assert "johnny_harris" in prompt
    assert "Johnny Harris opens with a question" in prompt

def test_parse_brief_angle_extracts_angle():
    brief_text = """# Topic Brief: War Financing

## Story Angle
Governments never vote to raise taxes for wars — they vote to print money instead.

## Why This Serves the North Star
Viewers trust governments as transparent actors.
"""
    angle = parse_brief_angle(brief_text)
    assert "print money" in angle

def test_parse_brief_angle_missing_returns_unknown():
    angle = parse_brief_angle("no angle section here")
    assert angle == "unknown"

def test_run_researcher_writes_brief(tmp_path):
    topic_dir = tmp_path / "war-financing"
    topic_dir.mkdir()
    profile = {
        "north_star": "make the viewer distrust something they trusted before",
        "domain": "economics, incentives",
        "arc_template": "surface_reality → hidden_incentive → reframe",
        "core_thesis": "Show me the incentive",
        "channel_north_star": "Create a compelling video",
        "reference_channels": ["johnny_harris"],
    }
    fake_brief = "# Topic Brief: War Financing\n\n## Story Angle\nTest angle.\n\n## Sources\n- http://example.com — fact\n"

    with patch("mosaic.pipeline.researcher.KBStore") as MockStore, \
         patch("mosaic.pipeline.researcher.TasteStore") as MockTaste, \
         patch("mosaic.pipeline.researcher._call_claude_with_search") as mock_claude:
        mock_store = MagicMock()
        mock_store.query.return_value = []
        MockStore.return_value = mock_store
        MockTaste.return_value = MagicMock(query=MagicMock(return_value=[]))
        mock_claude.return_value = fake_brief

        result = run_researcher(topic_dir, "war-financing", profile)

    brief_path = topic_dir / "brief.md"
    assert brief_path.exists()
    assert "Story Angle" in brief_path.read_text()
    assert "angle" in result
