import re
from pathlib import Path
from mosaic.kb.store import KBStore
from mosaic.kb.analyzer import VideoInsights


def _parse_seed_insights(text: str) -> list[str]:
    """Extract insight paragraphs from markdown. Each ### section → one insight."""
    insights = []
    sections = re.split(r"^###\s+", text, flags=re.MULTILINE)
    for section in sections[1:]:  # skip content before first ###
        lines = section.strip().splitlines()
        if not lines:
            continue
        title = lines[0].strip()
        body = " ".join(line.strip() for line in lines[1:] if line.strip())
        if body:
            insights.append(f"{title}: {body}")
    return insights


def load_seed_guide(seed_path: Path, store: KBStore) -> None:
    """Parse seed guide markdown and load insights into the KB store."""
    text = seed_path.read_text(encoding="utf-8")
    insights = _parse_seed_insights(text)

    seed_insights = VideoInsights(
        video_id="__seed__",
        channel="seed",
        title="Editorial Seed Guide",
        hook={},
        story_arc={},
        emotional_beats=[],
        visual_patterns=[],
        pacing={},
        narration_style={},
        key_insights=insights,
    )
    store.add(seed_insights)
