import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from mosaic.pipeline.assembler import (
    check_duration_gate,
    DurationGateResult,
    build_self_eval_checks,
    parse_eval_result,
    build_ass_header,
)


def test_duration_gate_fail():
    result = check_duration_gate(duration_seconds=450)  # 7.5 min
    assert result == DurationGateResult.FAIL


def test_duration_gate_warn():
    result = check_duration_gate(duration_seconds=510)  # 8.5 min
    assert result == DurationGateResult.WARN


def test_duration_gate_pass():
    result = check_duration_gate(duration_seconds=620)  # 10.3 min
    assert result == DurationGateResult.PASS


def test_build_self_eval_checks_returns_four():
    checks = build_self_eval_checks()
    assert len(checks) == 4
    assert all("question" in c for c in checks)


def test_parse_eval_result_pass():
    eval_text = """
1. Hook question in first 15 seconds: PASS — unanswered question established at 0:12
2. Every visual earns screen time: PASS — all clips serve narrative
3. Two unexpected feeling moments: PASS — timestamps 2:30 and 7:15
4. Ending resolves hook tension: PASS — full circle at 9:28
"""
    result = parse_eval_result(eval_text)
    assert result["passed"] == 4
    assert result["failed"] == 0
    assert result["score"] == "4/4"


def test_parse_eval_result_with_failures():
    eval_text = """
1. Hook question in first 15 seconds: FAIL — hook is too slow
2. Every visual earns screen time: PASS
3. Two unexpected feeling moments: PASS
4. Ending resolves hook tension: FAIL — resolution is weak
"""
    result = parse_eval_result(eval_text)
    assert result["passed"] == 2
    assert result["failed"] == 2
    assert result["score"] == "2/4"


def test_build_ass_header_contains_style():
    header = build_ass_header()
    assert "[Script Info]" in header
    assert "[V4+ Styles]" in header
    assert "IncentivesLab" in header
