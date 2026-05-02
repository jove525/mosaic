import pytest
from pathlib import Path
from mosaic.pipeline.taste_store import TasteStore, TasteEntry

@pytest.fixture
def store(tmp_path):
    return TasteStore(persist_dir=tmp_path / "taste")

def test_add_and_query(store):
    entry = TasteEntry(
        video_id="war-financing",
        channel="incentiveslab",
        agent="assembler",
        decision="slow dissolve at thesis inversion (0:28)",
        agent_confidence="high",
        user_signal="negative — felt slow",
        delta="overconfidence",
        learning="use hard cut at thesis inversion, not dissolve",
    )
    store.add(entry)
    results = store.query("transition at thesis inversion", n_results=1)
    assert len(results) == 1
    assert results[0]["learning"] == "use hard cut at thesis inversion, not dissolve"
    assert results[0]["delta"] == "overconfidence"

def test_empty_store_returns_empty(store):
    results = store.query("any query", n_results=5)
    assert results == []


def test_duplicate_add_is_idempotent(store):
    entry = TasteEntry(
        video_id="war-financing",
        channel="incentiveslab",
        agent="assembler",
        decision="slow dissolve at thesis inversion (0:28)",
        agent_confidence="high",
        user_signal="felt slow",
        delta="overconfidence",
        learning="use hard cut, not dissolve",
    )
    store.add(entry)
    store.add(entry)  # should not raise
    assert store._collection.count() == 1
