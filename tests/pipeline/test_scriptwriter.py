import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from mosaic.pipeline.scriptwriter import (
    parse_script_stats,
    build_scriptwriter_prompt,
    run_scriptwriter,
)

SAMPLE_SCRIPT = """\
# Script: The Hidden Tax That Funds Every War

## Section: Hook (0:00–0:30)

[NARRATION] The Roman Empire didn't fall because of barbarians.
[EMOTION]   REVELATION
[VISUAL]    Wide aerial of ruins
[DIRECTION] Thesis inversion — viewer's assumption crumbles
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
[DIRECTION] Stakes established
[PACING]    Drive forward.
"""

def test_parse_script_stats_counts_narration_lines():
    stats = parse_script_stats(SAMPLE_SCRIPT)
    assert stats["line_count"] == 3
    assert stats["section_count"] == 2

def test_parse_script_stats_estimates_minutes():
    stats = parse_script_stats(SAMPLE_SCRIPT)
    assert stats["est_minutes"] > 0

def test_build_scriptwriter_prompt_contains_arc():
    profile = {
        "arc_template": "surface_reality → hidden_incentive → reframe",
        "north_star": "make the viewer distrust something they trusted before",
        "channel_north_star": "Create a compelling video",
        "reference_channels": ["johnny_harris"],
    }
    brief_text = "# Topic Brief: War Financing\n\n## Story Angle\nTest angle.\n"
    prompt = build_scriptwriter_prompt(brief_text, profile, [], [])
    assert "War Financing" in prompt
    assert "Test angle" in prompt

def test_run_scriptwriter_writes_script(tmp_path):
    topic_dir = tmp_path / "war-financing"
    topic_dir.mkdir()
    (topic_dir / "brief.md").write_text("# Brief\n\n## Story Angle\nTest.\n")

    profile = {
        "arc_template": "surface_reality → hidden_incentive → reframe",
        "north_star": "make the viewer distrust something they trusted before",
        "channel_north_star": "Create a compelling video",
        "reference_channels": ["johnny_harris"],
    }

    with patch("mosaic.pipeline.scriptwriter.KBStore") as MockStore, \
         patch("mosaic.pipeline.scriptwriter.TasteStore") as MockTaste, \
         patch("mosaic.pipeline.scriptwriter._call_claude") as mock_claude:
        mock_store = MagicMock()
        mock_store.query.return_value = []
        MockStore.return_value = mock_store
        MockTaste.return_value = MagicMock(query=MagicMock(return_value=[]))
        mock_claude.return_value = SAMPLE_SCRIPT

        result = run_scriptwriter(topic_dir, profile)

    assert (topic_dir / "script.md").exists()
    assert result["line_count"] == 3
    assert result["est_minutes"] > 0
