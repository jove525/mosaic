from mosaic.pipeline.assembler import _collect_valid_cuts


def test_collect_valid_cuts_resolved(tmp_path):
    """Resolved entries with existing cut files are collected."""
    cut_file = tmp_path / "clips" / "final" / "clip_001_a.mp4"
    cut_file.parent.mkdir(parents=True)
    cut_file.write_bytes(b"fake" * 500)  # >1000 bytes

    manifest = [
        {
            "narration_line_ref": 1,
            "cuts": [{"local_path": "clips/final/clip_001_a.mp4", "duration_seconds": 10.0, "description": "test"}],
            "needs_generated_visual": False,
        }
    ]
    cuts = _collect_valid_cuts(manifest, tmp_path)
    assert len(cuts) == 1
    assert cuts[0]["local_path"] == "clips/final/clip_001_a.mp4"


def test_collect_valid_cuts_gap(tmp_path):
    """Flagged entries with needs_generated_visual=True produce no cuts."""
    manifest = [
        {
            "narration_line_ref": 5,
            "cuts": [],
            "needs_generated_visual": True,
            "visual_description": "Bar chart",
        }
    ]
    cuts = _collect_valid_cuts(manifest, tmp_path)
    assert cuts == []


def test_collect_valid_cuts_missing_file(tmp_path):
    """Cuts whose files don't exist are silently skipped."""
    manifest = [
        {
            "narration_line_ref": 1,
            "cuts": [{"local_path": "clips/final/missing.mp4", "duration_seconds": 10.0, "description": "test"}],
            "needs_generated_visual": False,
        }
    ]
    cuts = _collect_valid_cuts(manifest, tmp_path)
    assert cuts == []
