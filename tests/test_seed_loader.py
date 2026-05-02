from mosaic.kb.seed_loader import load_seed_guide
from mosaic.kb.store import KBStore


def test_load_seed_guide_populates_store(tmp_path):
    store = KBStore(persist_dir=tmp_path / "kb")
    seed_path = tmp_path / "guide.md"

    seed_path.write_text("""# Test Guide

## Hook Principles

### Curiosity Gap Hook
Open with an unanswered question. Works because it creates cognitive dissonance.
Application: state the paradox within 15 seconds.

## Visual Principles

### Close-Up for Emotion
Use close-ups at emotionally charged moments to force viewer intimacy.
""")

    load_seed_guide(seed_path, store)

    results = store.query("curiosity gap hook technique", n_results=3)
    assert len(results) >= 1
    assert any("curiosity" in r["insight"].lower() or "question" in r["insight"].lower() for r in results)


def test_load_seed_guide_is_idempotent(tmp_path):
    store = KBStore(persist_dir=tmp_path / "kb")
    seed_path = tmp_path / "guide.md"
    seed_path.write_text("# Guide\n\n## Hooks\n\n### Simple Hook\nUse a hook. Works because attention.\n")

    load_seed_guide(seed_path, store)
    count_after_first = store._collection.count()

    load_seed_guide(seed_path, store)
    count_after_second = store._collection.count()

    assert count_after_first == count_after_second
