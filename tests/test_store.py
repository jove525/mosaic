import pytest
from mosaic.kb.store import KBStore
from mosaic.kb.analyzer import VideoInsights


def _make_insights(video_id: str = "vid1") -> VideoInsights:
    return VideoInsights(
        video_id=video_id,
        channel="wendover",
        title="Why Planes Fly",
        hook={"structure": "curiosity_gap", "technique": "opens with paradox", "why_it_works": "creates unanswered question", "duration_seconds": 25},
        story_arc={"shape": "question_answer", "phases": []},
        emotional_beats=[],
        visual_patterns=[{"narrative_moment": "revelation", "shot_type": "zoom_out", "why": "creates release"}],
        pacing={"avg_cut_interval_seconds": 4.0, "music_shift_points": [], "silence_moments": []},
        narration_style={"avg_sentence_length_words": 11, "rhythm": "punchy", "notable_techniques": []},
        key_insights=[
            "Opens with a paradox that demands resolution",
            "Uses zoom-out shots at every revelation moment to create emotional release",
        ],
    )


def test_store_and_retrieve_insights(tmp_dir):
    store = KBStore(persist_dir=tmp_dir / "kb")
    insights = _make_insights()

    store.add(insights)
    results = store.query("zoom out shot for emotional release", n_results=1)

    assert len(results) == 1
    assert results[0]["video_id"] == "vid1"
    assert "zoom-out" in results[0]["insight"].lower() or "zoom out" in results[0]["insight"].lower()


def test_store_skip_duplicate(tmp_dir):
    store = KBStore(persist_dir=tmp_dir / "kb")
    insights = _make_insights()

    store.add(insights)
    store.add(insights)  # should not raise or duplicate

    results = store.query("paradox hook", n_results=10)
    video_ids = [r["video_id"] for r in results]
    assert video_ids.count("vid1") <= 2  # one entry per key_insight, not doubled


def test_store_is_already_processed(tmp_dir):
    store = KBStore(persist_dir=tmp_dir / "kb")
    assert not store.is_processed("vid1")

    store.add(_make_insights("vid1"))
    assert store.is_processed("vid1")
