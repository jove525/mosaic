import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from mosaic.pipeline.delta_analyzer import (
    build_agent_context,
    _extract_eval_score,
    _extract_duration_result,
    run_delta_analysis,
)

SAMPLE_REVIEW_NOTES = """\
Self-eval: 4/4 PASS — all checks passed.
Duration: 10m 45s — PASS (above 10min gate)
"""

SAMPLE_METADATA = {
    "title": "Why Rome Really Fell",
    "description": "Hook sentence.",
    "tags": ["economics", "history"],
    "thumbnail_brief": "Coin on black background.",
}

def test_extract_eval_score():
    score = _extract_eval_score(SAMPLE_REVIEW_NOTES)
    assert "Self-eval" in score

def test_extract_duration_result():
    result = _extract_duration_result(SAMPLE_REVIEW_NOTES)
    assert "Duration" in result

def test_extract_eval_score_missing():
    score = _extract_eval_score("no eval info here")
    assert score == "unknown"

def test_build_agent_context():
    ctx = build_agent_context("war-financing", SAMPLE_METADATA, SAMPLE_REVIEW_NOTES)
    assert ctx["topic"] == "war-financing"
    assert ctx["title_chosen"] == "Why Rome Really Fell"
    assert "Self-eval" in ctx["eval_score"]
    assert "Duration" in ctx["duration_result"]

def test_run_delta_analysis_writes_files(tmp_path):
    topic_dir = tmp_path / "war-financing"
    topic_dir.mkdir()
    (topic_dir / "your_feedback.md").write_text(
        "The video felt too long and the hook didn't land.", encoding="utf-8"
    )
    (topic_dir / "youtube_metadata.json").write_text(
        json.dumps(SAMPLE_METADATA), encoding="utf-8"
    )
    (topic_dir / "review_notes.md").write_text(SAMPLE_REVIEW_NOTES, encoding="utf-8")

    fake_entries = [
        {
            "agent": "scriptwriter",
            "decision": "Script ran 11 minutes",
            "agent_confidence": "high",
            "user_signal": "User felt it was too long",
            "delta": "overconfidence",
            "learning": "Prefer 9-minute scripts when topic is well-understood",
        }
    ]

    with patch("mosaic.pipeline.delta_analyzer._call_claude_for_delta") as mock_claude, \
         patch("mosaic.pipeline.delta_analyzer.TasteStore") as MockTaste:
        mock_claude.return_value = fake_entries
        MockTaste.return_value = MagicMock(add=MagicMock())

        result = run_delta_analysis(topic_dir, "incentiveslab", "war-financing")

    assert result["stored"] == 1
    assert (topic_dir / "delta_analysis.md").exists()
    content = (topic_dir / "delta_analysis.md").read_text()
    assert "overconfidence" in content.lower() or "OVERCONFIDENCE" in content
    assert "scriptwriter" in content
