import json
from pathlib import Path
import pytest
from mosaic.pipeline.clip_cache import ClipCache, CachedVideo, CachedSegment

def test_save_and_load(tmp_path):
    cache = ClipCache(tmp_path)
    video = CachedVideo(
        identifier="test-001",
        source="prelinger",
        title="Test War Bonds",
        source_url="https://archive.org/download/test/test.mp4",
        license="public_domain",
        analyzed_at="2026-04-14",
        duration_seconds=300.0,
        segments=[
            CachedSegment(start=10.0, end=25.0, description="civilians at booth", tags=["WWII", "bond drive"])
        ]
    )
    cache.save(video)
    loaded = cache.load("test-001")
    assert loaded is not None
    assert loaded.identifier == "test-001"
    assert len(loaded.segments) == 1
    assert loaded.segments[0].start == 10.0

def test_load_missing_returns_none(tmp_path):
    cache = ClipCache(tmp_path)
    assert cache.load("nonexistent") is None

def test_exists(tmp_path):
    cache = ClipCache(tmp_path)
    assert not cache.exists("test-001")
    video = CachedVideo(
        identifier="test-001", source="prelinger", title="T",
        source_url="https://x.com", license="public_domain",
        analyzed_at="2026-04-14", duration_seconds=60.0, segments=[]
    )
    cache.save(video)
    assert cache.exists("test-001")
