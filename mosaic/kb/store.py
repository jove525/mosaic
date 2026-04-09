import chromadb
import json
from pathlib import Path
from mosaic.kb.analyzer import VideoInsights
from mosaic.utils.fs import ensure_dir


class KBStore:
    """Persistent ChromaDB store for editorial insights."""

    COLLECTION_NAME = "mosaic_kb"

    def __init__(self, persist_dir: Path):
        ensure_dir(persist_dir)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, insights: VideoInsights) -> None:
        """Store each key_insight as a searchable document. Skip if already stored."""
        for i, insight_text in enumerate(insights.key_insights):
            doc_id = f"{insights.video_id}__insight_{i}"
            existing = self._collection.get(ids=[doc_id])
            if existing["ids"]:
                continue

            self._collection.add(
                ids=[doc_id],
                documents=[insight_text],
                metadatas=[{
                    "video_id": insights.video_id,
                    "channel": insights.channel,
                    "title": insights.title,
                    "hook_structure": insights.hook.get("structure", ""),
                    "arc_shape": insights.story_arc.get("shape", ""),
                    "full_json": json.dumps({
                        "hook": insights.hook,
                        "story_arc": insights.story_arc,
                        "emotional_beats": insights.emotional_beats,
                        "visual_patterns": insights.visual_patterns,
                        "pacing": insights.pacing,
                        "narration_style": insights.narration_style,
                    }),
                }],
            )

    def query(self, query_text: str, n_results: int = 5) -> list[dict]:
        """Return top n_results insights matching the query."""
        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(n_results, max(1, self._collection.count())),
        )
        output = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            output.append({
                "insight": doc,
                "video_id": meta["video_id"],
                "channel": meta["channel"],
                "title": meta["title"],
                "full_json": json.loads(meta["full_json"]),
            })
        return output

    def is_processed(self, video_id: str) -> bool:
        """Check if any insights for this video_id are already stored."""
        results = self._collection.get(where={"video_id": video_id})
        return len(results["ids"]) > 0
