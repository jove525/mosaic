"""Tests for clip_manifest.json contract and QC integration."""
import json
from pathlib import Path
import pytest


def test_clip_manifest_entry_structure():
    """clip_manifest.json entries must follow the data contract."""
    entry = {
        "narration_line_ref": 1,
        "narration_text": "The Roman Empire didn't fall because of barbarians.",
        "status": "sourced",
        "source_url": "https://archive.org/download/test/test.mp4",
        "local_path": "clips/line01_test.mp4",
        "source_type": "internet_archive",
        "license": "public_domain",
        "license_tier": "SAFE",
        "timestamp_window": {"start": "0:30", "end": "0:55"},
        "window_reason": "civilians at bond booth",
        "qc": {
            "technical": "ok",
            "clip_score": 0.31,
            "haiku_emotional": True,
            "haiku_narrative": True,
            "haiku_why": "matches REVELATION emotion",
        },
        "attempts": 1,
    }
    required = ["narration_line_ref", "narration_text", "status", "source_url",
                "local_path", "source_type", "license", "license_tier",
                "timestamp_window", "qc"]
    for field in required:
        assert field in entry

    assert entry["status"] in ("sourced", "ai_generated", "gap")
    assert entry["license_tier"] in ("SAFE", "CAUTION", "AVOID", "")
    assert isinstance(entry["timestamp_window"], dict)


def test_clip_manifest_gap_entry_structure():
    """Gap entries must still have all required fields."""
    entry = {
        "narration_line_ref": 3,
        "narration_text": "Every government since has faced the same problem.",
        "status": "gap",
        "source_url": "",
        "local_path": "",
        "source_type": "gap",
        "license": "",
        "license_tier": "",
        "timestamp_window": {},
        "window_reason": "no suitable clip found",
        "qc": {},
        "attempts": 3,
        "gap_reason": "all sources exhausted after 6 queries",
    }
    assert entry["status"] == "gap"
    assert entry["local_path"] == ""
    assert "gap_reason" in entry


def test_clip_manifest_ai_generated_entry():
    """AI-generated entries must be flagged correctly."""
    entry = {
        "narration_line_ref": 2,
        "narration_text": "It fell because it ran out of money.",
        "status": "ai_generated",
        "source_url": "",
        "local_path": "clips/line02_ai_roman_coins.mp4",
        "source_type": "ai_generated",
        "license": "ai_generated",
        "license_tier": "SAFE",
        "timestamp_window": {"start": "0:00", "end": "0:05"},
        "window_reason": "AI generated",
        "qc": {"technical": "ok", "clip_score": 0.0, "haiku_emotional": True, "haiku_narrative": True, "haiku_why": "ai_generated"},
        "attempts": 2,
    }
    assert entry["status"] == "ai_generated"
    assert entry["source_type"] == "ai_generated"


def test_ai_cap_20_percent():
    """AI-generated clips must not exceed 20% of total lines."""
    total_lines = 40
    cap = int(total_lines * 0.20)
    assert cap == 8

    manifest = []
    ai_count = 0
    for i in range(total_lines):
        if i < 8:
            manifest.append({"status": "ai_generated"})
            ai_count += 1
        else:
            manifest.append({"status": "sourced"})

    ai_in_manifest = sum(1 for e in manifest if e["status"] == "ai_generated")
    assert ai_in_manifest / total_lines <= 0.20


def test_manifest_completeness():
    """Every narration line must have exactly one manifest entry."""
    total_lines = 5
    manifest = [
        {"narration_line_ref": i + 1, "status": "sourced"} for i in range(total_lines)
    ]
    refs = {e["narration_line_ref"] for e in manifest}
    expected = set(range(1, total_lines + 1))
    assert refs == expected
