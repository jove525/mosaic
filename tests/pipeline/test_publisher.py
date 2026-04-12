import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from mosaic.pipeline.publisher import (
    check_duration_gate_passed,
    generate_youtube_metadata,
    run_publisher,
    PublisherError,
)

SAMPLE_PROFILE = {
    "north_star": "make the viewer distrust something they trusted before",
    "channel_north_star": "Create a video so compelling that viewers distrust a trusted institution",
    "core_thesis": "Show me the incentive and I'll show you the outcome",
}

def test_check_duration_gate_passed_clean():
    notes = "Self-eval: 4/4 PASS\nDuration: 10m 30s — PASS"
    assert check_duration_gate_passed(notes) is True

def test_check_duration_gate_passed_with_fail():
    notes = "⚠ Duration gate FAIL — Scriptwriter rerun cap reached (2 attempts)."
    assert check_duration_gate_passed(notes) is False

def test_generate_youtube_metadata_parses_json():
    fake_response = json.dumps({
        "title": "Why Rome Really Fell",
        "description": "Hook sentence here. Core argument. Why it matters. Subscribe CTA.",
        "tags": ["economics", "Roman Empire", "war financing"],
        "thumbnail_brief": "Close-up of devalued Roman coin on black background.",
    })
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=fake_response)]

    with patch("mosaic.pipeline.publisher.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = mock_msg
        result = generate_youtube_metadata("war-financing", "script text", "angle", SAMPLE_PROFILE)

    assert result["title"] == "Why Rome Really Fell"
    assert isinstance(result["tags"], list)
    assert "thumbnail_brief" in result

def test_generate_youtube_metadata_strips_code_fence():
    fake_response = '```json\n{"title": "Test", "description": "desc", "tags": [], "thumbnail_brief": "brief"}\n```'
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=fake_response)]

    with patch("mosaic.pipeline.publisher.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = mock_msg
        result = generate_youtube_metadata("test-topic", "", "angle", SAMPLE_PROFILE)

    assert result["title"] == "Test"

def test_run_publisher_writes_metadata(tmp_path):
    topic_dir = tmp_path / "war-financing"
    topic_dir.mkdir()
    (topic_dir / "final_draft.mp4").write_bytes(b"fake_video")
    (topic_dir / "review_notes.md").write_text("Self-eval: 4/4 PASS\nDuration: 10m 30s — PASS", encoding="utf-8")
    (topic_dir / "script.md").write_text("# Script\n\n[NARRATION] Test narration.", encoding="utf-8")

    fake_metadata = {
        "title": "Why Rome Really Fell",
        "description": "Hook. Argument. Matters. CTA.",
        "tags": ["economics", "history"],
        "thumbnail_brief": "Coin on black.",
    }

    with patch("mosaic.pipeline.publisher.generate_youtube_metadata") as mock_gen, \
         patch("mosaic.pipeline.publisher.TasteStore") as MockTaste:
        mock_gen.return_value = fake_metadata
        MockTaste.return_value = MagicMock(query=MagicMock(return_value=[]))

        result = run_publisher(topic_dir, "war-financing", SAMPLE_PROFILE)

    assert result["status"] == "ready_for_review"
    metadata_path = topic_dir / "youtube_metadata.json"
    assert metadata_path.exists()
    data = json.loads(metadata_path.read_text())
    assert data["title"] == "Why Rome Really Fell"

def test_run_publisher_raises_on_duration_fail(tmp_path):
    topic_dir = tmp_path / "war-financing"
    topic_dir.mkdir()
    (topic_dir / "final_draft.mp4").write_bytes(b"fake_video")
    (topic_dir / "review_notes.md").write_text(
        "⚠ Duration gate FAIL — Scriptwriter rerun cap reached (2 attempts).",
        encoding="utf-8",
    )

    with pytest.raises(PublisherError, match="duration_gate_fail"):
        run_publisher(topic_dir, "war-financing", SAMPLE_PROFILE)

def test_run_publisher_raises_on_missing_final_draft(tmp_path):
    topic_dir = tmp_path / "war-financing"
    topic_dir.mkdir()
    # final_draft.mp4 does NOT exist

    with pytest.raises(PublisherError, match="final_draft.mp4"):
        run_publisher(topic_dir, "war-financing", SAMPLE_PROFILE)
