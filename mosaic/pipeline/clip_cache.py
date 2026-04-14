"""Persistent cache of analyzed video files. Shared across topics and runs.
Phase 3: this cache becomes the queryable clip library."""
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class CachedSegment:
    start: float
    end: float
    description: str
    tags: list[str] = field(default_factory=list)


@dataclass
class CachedVideo:
    identifier: str
    source: str
    title: str
    source_url: str
    license: str
    analyzed_at: str
    duration_seconds: float
    segments: list[CachedSegment] = field(default_factory=list)


class ClipCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, identifier: str) -> Path:
        safe = re.sub(r"[^\w\-]", "_", identifier)[:80]
        return self.cache_dir / f"{safe}.json"

    def exists(self, identifier: str) -> bool:
        return self._path(identifier).exists()

    def save(self, video: CachedVideo) -> None:
        data = asdict(video)
        self._path(video.identifier).write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    def load(self, identifier: str) -> Optional[CachedVideo]:
        p = self._path(identifier)
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        segments = [CachedSegment(**s) for s in data.pop("segments", [])]
        return CachedVideo(**data, segments=segments)
