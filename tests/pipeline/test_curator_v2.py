import json
from pathlib import Path
import pytest


def test_raw_candidates_json_structure(tmp_path):
    """raw_candidates.json must have per-line candidates with local_path in clips/raw/."""
    raw_candidates = [
        {
            "narration_line_ref": 1,
            "narration_text": "Test narration line.",
            "emotion": "TENSION",
            "visual_description": "Civilians at bond booth.",
            "candidates": [
                {
                    "identifier": "test-video-001",
                    "source": "prelinger",
                    "url": "https://archive.org/download/test/test.mp4",
                    "local_path": "clips/raw/test-video-001.mp4",
                    "title": "Test Video",
                    "license": "public_domain"
                }
            ]
        }
    ]
    out = tmp_path / "raw_candidates.json"
    out.write_text(json.dumps(raw_candidates), encoding="utf-8")
    loaded = json.loads(out.read_text())
    assert len(loaded) == 1
    entry = loaded[0]
    assert "narration_line_ref" in entry
    assert "candidates" in entry
    assert entry["candidates"][0]["local_path"].startswith("clips/raw/")
    assert "identifier" in entry["candidates"][0]
