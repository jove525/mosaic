import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from mosaic.pipeline.curator import (
    parse_narration_lines,
    score_clip_match,
    select_music_track,
    run_curator,
)

SAMPLE_SCRIPT = """\
# Script: War Financing

## Section: Hook (0:00–0:30)

[NARRATION] The Roman Empire didn't fall because of barbarians.
[EMOTION]   REVELATION
[VISUAL]    Wide aerial of Roman ruins
[DIRECTION] Thesis inversion
[PACING]    Hold 2s. Hard cut.

[NARRATION] It fell because it ran out of money to pay its soldiers.
[EMOTION]   TENSION
[VISUAL]    Close-up of devalued Roman coins
[DIRECTION] Pattern begins
[PACING]    Steady.

## Section: Context (0:30–2:00)

[NARRATION] Every government since has faced the same problem.
[EMOTION]   MOMENTUM
[VISUAL]    Montage of war budgets
[DIRECTION] Stakes
[PACING]    Drive forward.
"""

def test_parse_narration_lines_extracts_all_fields():
    lines = parse_narration_lines(SAMPLE_SCRIPT)
    assert len(lines) == 3
    assert lines[0]["narration_text"] == "The Roman Empire didn't fall because of barbarians."
    assert lines[0]["emotion"] == "REVELATION"
    assert lines[0]["visual"] == "Wide aerial of Roman ruins"
    assert lines[0]["line_ref"] == 1

def test_score_clip_match_semantic_required():
    score = score_clip_match(semantic=False, emotional=True, narrative=True)
    assert score == 0

def test_score_clip_match_all_layers():
    score = score_clip_match(semantic=True, emotional=True, narrative=True)
    assert score == 3

def test_score_clip_match_minimum_passing():
    score = score_clip_match(semantic=True, emotional=False, narrative=True)
    assert score >= 2

def test_select_music_track_tension():
    track = select_music_track(dominant_emotion="TENSION")
    assert track["tempo"] == "slow"
    assert "ambient" in track["genre"].lower()

def test_select_music_track_revelation():
    track = select_music_track(dominant_emotion="REVELATION")
    assert "cinematic" in track["genre"].lower()

def test_run_curator_writes_raw_candidates(tmp_path):
    topic_dir = tmp_path / "war-financing"
    topic_dir.mkdir()
    (topic_dir / "clips").mkdir()
    (topic_dir / "script.md").write_text(SAMPLE_SCRIPT, encoding="utf-8")

    with patch("mosaic.pipeline.curator._generate_narration") as mock_narr, \
         patch("mosaic.pipeline.curator._generate_search_queries") as mock_queries, \
         patch("mosaic.pipeline.curator._search_internet_archive") as mock_ia, \
         patch("mosaic.pipeline.curator._search_wikimedia") as mock_wm, \
         patch("mosaic.pipeline.curator._resolve_ia_download_url") as mock_resolve, \
         patch("mosaic.pipeline.curator._download_clip") as mock_dl, \
         patch("mosaic.pipeline.curator.TasteStore") as MockTaste:

        mock_narr.return_value = b"fake_mp3_bytes"
        mock_queries.return_value = ["WWII war bonds 1943"]
        mock_ia.return_value = [{"identifier": "test-id", "url": None, "duration": 8.0, "source_type": "internet_archive", "license": "public_domain", "license_tier": "SAFE", "title": "Test"}]
        mock_wm.return_value = []
        mock_resolve.return_value = "https://archive.org/download/test-id/test.mp4"
        dest = topic_dir / "clips" / "raw" / "test.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x" * 20000)
        mock_dl.return_value = dest
        MockTaste.return_value = MagicMock(query=MagicMock(return_value=[]))

        result = run_curator(topic_dir, {"north_star": "test"})

    raw_candidates_path = topic_dir / "raw_candidates.json"
    assert raw_candidates_path.exists()
    raw = json.loads(raw_candidates_path.read_text())
    assert len(raw) == 3
    assert "narration_line_ref" in raw[0]
    assert "candidates" in raw[0]
    assert (topic_dir / "narration.mp3").exists()
