import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from mosaic.pipeline.curator import (
    parse_narration_lines,
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


def test_parse_narration_lines_all_tags():
    lines = parse_narration_lines(SAMPLE_SCRIPT)
    assert lines[0]["direction"] == "Thesis inversion"
    assert lines[0]["pacing"] == "Hold 2s. Hard cut."


def test_select_music_track_tension():
    track = select_music_track("TENSION")
    assert track["tempo"] == "slow"
    assert "ambient" in track["genre"].lower()


def test_select_music_track_revelation():
    track = select_music_track("REVELATION")
    assert "cinematic" in track["genre"].lower()


def test_select_music_track_unknown_defaults_to_momentum():
    track = select_music_track("UNKNOWN")
    assert "cinematic" in track["genre"].lower()


def _make_fake_clip(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * 20000)
    return path


def test_run_curator_writes_clip_manifest(tmp_path):
    topic_dir = tmp_path / "war-financing"
    topic_dir.mkdir()
    (topic_dir / "clips").mkdir()
    (topic_dir / "script.md").write_text(SAMPLE_SCRIPT, encoding="utf-8")

    fake_clip = _make_fake_clip(topic_dir / "clips" / "line01_test.mp4")
    fake_qc = {
        "passed": True,
        "technical": {"passed": True, "reason": "ok"},
        "semantic": {"passed": True, "score": 0.31},
        "contextual": {"passed": True, "emotional": True, "narrative": True, "why": "good match"},
    }

    with patch("mosaic.pipeline.curator._generate_narration", return_value=b"fake_mp3"), \
         patch("mosaic.pipeline.curator._generate_search_queries", return_value=["WWII war bonds"]), \
         patch("mosaic.pipeline.curator._search_internet_archive", return_value=[]), \
         patch("mosaic.pipeline.curator._search_wikimedia", return_value=[]), \
         patch("mosaic.pipeline.curator._search_youtube_cc", return_value=[]), \
         patch("mosaic.pipeline.curator._generate_clip_kling", return_value=fake_clip), \
         patch("mosaic.pipeline.curator.run_qc", return_value=fake_qc), \
         patch("mosaic.pipeline.curator.TasteStore") as MockTaste:

        MockTaste.return_value = MagicMock(query=MagicMock(return_value=[]))
        result = run_curator(topic_dir, {"north_star": "test"})

    manifest_path = topic_dir / "clip_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest) == 3
    assert (topic_dir / "narration.mp3").exists()


def test_run_curator_manifest_has_required_fields(tmp_path):
    topic_dir = tmp_path / "war-financing"
    topic_dir.mkdir()
    (topic_dir / "clips").mkdir()
    (topic_dir / "script.md").write_text(SAMPLE_SCRIPT, encoding="utf-8")

    fake_clip = _make_fake_clip(topic_dir / "clips" / "line01_test.mp4")
    fake_qc = {
        "passed": True,
        "technical": {"passed": True, "reason": "ok"},
        "semantic": {"passed": True, "score": 0.31},
        "contextual": {"passed": True, "emotional": True, "narrative": True, "why": "good match"},
    }

    with patch("mosaic.pipeline.curator._generate_narration", return_value=b"fake_mp3"), \
         patch("mosaic.pipeline.curator._generate_search_queries", return_value=["test query"]), \
         patch("mosaic.pipeline.curator._search_internet_archive", return_value=[]), \
         patch("mosaic.pipeline.curator._search_wikimedia", return_value=[]), \
         patch("mosaic.pipeline.curator._search_youtube_cc", return_value=[]), \
         patch("mosaic.pipeline.curator._generate_clip_kling", return_value=fake_clip), \
         patch("mosaic.pipeline.curator.run_qc", return_value=fake_qc), \
         patch("mosaic.pipeline.curator.TasteStore") as MockTaste:

        MockTaste.return_value = MagicMock(query=MagicMock(return_value=[]))
        run_curator(topic_dir, {})

    manifest = json.loads((topic_dir / "clip_manifest.json").read_text())
    entry = manifest[0]
    required = ["narration_line_ref", "narration_text", "status", "source_url",
                "local_path", "source_type", "license", "license_tier",
                "timestamp_window", "qc"]
    for field in required:
        assert field in entry, f"missing field: {field}"


def test_run_curator_gap_when_all_sources_fail_and_ai_cap_hit(tmp_path):
    topic_dir = tmp_path / "war-financing"
    topic_dir.mkdir()
    (topic_dir / "clips").mkdir()
    (topic_dir / "script.md").write_text(SAMPLE_SCRIPT, encoding="utf-8")

    with patch("mosaic.pipeline.curator._generate_narration", return_value=b"fake_mp3"), \
         patch("mosaic.pipeline.curator._generate_search_queries", return_value=["test"]), \
         patch("mosaic.pipeline.curator._diversify_queries", return_value=[]), \
         patch("mosaic.pipeline.curator._search_internet_archive", return_value=[]), \
         patch("mosaic.pipeline.curator._search_wikimedia", return_value=[]), \
         patch("mosaic.pipeline.curator._search_youtube_cc", return_value=[]), \
         patch("mosaic.pipeline.curator._generate_clip_kling", return_value=None), \
         patch("mosaic.pipeline.curator.TasteStore") as MockTaste:

        MockTaste.return_value = MagicMock(query=MagicMock(return_value=[]))
        result = run_curator(topic_dir, {})

    assert result["gaps"] == 3
    manifest = json.loads((topic_dir / "clip_manifest.json").read_text())
    assert all(e["status"] == "gap" for e in manifest)


def test_run_curator_coverage_pct_reported(tmp_path):
    topic_dir = tmp_path / "war-financing"
    topic_dir.mkdir()
    (topic_dir / "clips").mkdir()
    (topic_dir / "script.md").write_text(SAMPLE_SCRIPT, encoding="utf-8")

    fake_clip = _make_fake_clip(topic_dir / "clips" / "line01_test.mp4")
    fake_qc = {
        "passed": True,
        "technical": {"passed": True, "reason": "ok"},
        "semantic": {"passed": True, "score": 0.31},
        "contextual": {"passed": True, "emotional": True, "narrative": True, "why": "ok"},
    }

    with patch("mosaic.pipeline.curator._generate_narration", return_value=b"fake_mp3"), \
         patch("mosaic.pipeline.curator._generate_search_queries", return_value=["test"]), \
         patch("mosaic.pipeline.curator._search_internet_archive", return_value=[]), \
         patch("mosaic.pipeline.curator._search_wikimedia", return_value=[]), \
         patch("mosaic.pipeline.curator._search_youtube_cc", return_value=[]), \
         patch("mosaic.pipeline.curator._generate_clip_kling", return_value=fake_clip), \
         patch("mosaic.pipeline.curator.run_qc", return_value=fake_qc), \
         patch("mosaic.pipeline.curator.TasteStore") as MockTaste:

        MockTaste.return_value = MagicMock(query=MagicMock(return_value=[]))
        result = run_curator(topic_dir, {})

    assert "coverage_pct" in result
    assert 0.0 <= result["coverage_pct"] <= 100.0
