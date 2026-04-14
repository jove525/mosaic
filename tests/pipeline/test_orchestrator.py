import pytest
from pathlib import Path
from unittest.mock import MagicMock
from mosaic.pipeline.orchestrator import Orchestrator, PipelineError

@pytest.fixture
def topic_dir(tmp_path):
    return tmp_path / "output" / "incentiveslab" / "war-financing"

def make_orchestrator(topic_dir):
    return Orchestrator(
        channel="incentiveslab",
        topic_slug="war-financing",
        output_root=topic_dir.parent.parent,
    )

def test_orchestrator_creates_topic_dir(topic_dir):
    orch = make_orchestrator(topic_dir)
    orch._ensure_dirs()
    assert (topic_dir).exists()
    assert (topic_dir / "clips").exists()

def test_from_flag_skips_earlier_agents(topic_dir):
    orch = make_orchestrator(topic_dir)
    orch._ensure_dirs()
    # Create required input file for scriptwriter
    (topic_dir / "brief.md").write_text("# Brief")

    called = []
    orch._run_researcher = lambda: called.append("researcher")
    orch._run_scriptwriter = lambda: called.append("scriptwriter")
    orch._run_curator = lambda: called.append("curator")
    orch._run_editorial = lambda: called.append("editorial")
    orch._run_assembler = lambda: called.append("assembler")
    orch._run_publisher = lambda: called.append("publisher")

    orch.run(from_agent="scriptwriter")
    assert "researcher" not in called
    assert called == ["scriptwriter", "curator", "editorial", "assembler", "publisher"]

def test_from_flag_missing_input_raises(topic_dir):
    orch = make_orchestrator(topic_dir)
    orch._ensure_dirs()
    # brief.md does NOT exist
    with pytest.raises(PipelineError, match="brief.md"):
        orch.run(from_agent="scriptwriter")

def test_full_run_calls_all_agents(topic_dir):
    orch = make_orchestrator(topic_dir)
    called = []
    orch._run_researcher = lambda: called.append("researcher")
    orch._run_scriptwriter = lambda: called.append("scriptwriter")
    orch._run_curator = lambda: called.append("curator")
    orch._run_editorial = lambda: called.append("editorial")
    orch._run_assembler = lambda: called.append("assembler")
    orch._run_publisher = lambda: called.append("publisher")

    orch.run(from_agent=None)
    assert called == ["researcher", "scriptwriter", "curator", "editorial", "assembler", "publisher"]

def test_unknown_channel_raises(tmp_path):
    with pytest.raises(PipelineError, match="Unknown channel"):
        Orchestrator(
            channel="nonexistent",
            topic_slug="test-topic",
            output_root=tmp_path,
        )


def test_agent_failure_includes_resume_command(topic_dir):
    orch = make_orchestrator(topic_dir)

    def fail():
        raise RuntimeError("KB offline")

    orch._run_researcher = fail
    with pytest.raises(PipelineError, match="--from researcher"):
        orch.run(from_agent=None)
