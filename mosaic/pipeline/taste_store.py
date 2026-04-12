import chromadb
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mosaic.utils.fs import ensure_dir


@dataclass
class TasteEntry:
    video_id: str
    channel: str
    agent: str
    decision: str
    agent_confidence: Literal["high", "medium", "low"]
    user_signal: str        # raw user feedback quote or paraphrase
    delta: Literal["overconfidence", "underconfidence", "calibrated"]
    learning: str           # one-line actionable rule


class TasteStore:
    """ChromaDB collection storing human feedback vs agent self-eval deltas."""

    COLLECTION_NAME = "mosaic_taste"

    def __init__(self, persist_dir: Path):
        ensure_dir(persist_dir)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, entry: TasteEntry) -> None:
        """Store a taste entry. Document text is the learning rule for semantic search.
        Silently skips if an entry with the same (video_id, agent, decision) already exists.
        """
        decision_hash = hashlib.sha1(entry.decision.encode()).hexdigest()[:8]
        doc_id = f"{entry.video_id}__{entry.agent}__{decision_hash}"
        existing = self._collection.get(ids=[doc_id])
        if existing["ids"]:
            return
        self._collection.add(
            ids=[doc_id],
            documents=[entry.learning],
            metadatas=[{
                "video_id": entry.video_id,
                "channel": entry.channel,
                "agent": entry.agent,
                "decision": entry.decision,
                "agent_confidence": entry.agent_confidence,
                "user_signal": entry.user_signal,
                "delta": entry.delta,
                "learning": entry.learning,
            }],
        )

    def query(self, query_text: str, n_results: int = 5) -> list[dict]:
        """Return top n_results taste entries matching the query."""
        count = self._collection.count()
        if count == 0:
            return []
        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(n_results, count),
        )
        output = []
        for meta in results["metadatas"][0]:
            output.append(dict(meta))
        return output
