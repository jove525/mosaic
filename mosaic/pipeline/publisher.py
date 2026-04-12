"""Agent 5 — Publisher.

Reads final_draft.mp4 + review_notes.md. Generates YouTube metadata via Claude.
Does NOT upload — pipeline is in review mode. Human uploads manually.
"""
import json
import logging
import re
from pathlib import Path

import anthropic

from mosaic.config.settings import settings
from mosaic.pipeline.taste_store import TasteStore

logger = logging.getLogger(__name__)

PIPELINE_MODE = "review"

_METADATA_SYSTEM = """\
You are a YouTube metadata specialist for IncentivesLab, a documentary channel about \
hidden incentives and how they shape the world.

Channel north star: {channel_north_star}
Core thesis: {core_thesis}

Produce YouTube metadata in EXACTLY this JSON format — no extra keys, no preamble:

{{
  "title": "<title under 70 chars — pattern: [What X Really Means] or [Why X Actually...]>",
  "description": "<3-4 paragraphs. Hook first sentence. Second paragraph: core argument. Third: why this matters. Fourth: subscribe CTA. 200-400 words total.>",
  "tags": ["<tag1>", "<tag2>", ...],
  "thumbnail_brief": "<one sentence describing the thumbnail concept for a designer>"
}}

Tags: 8-12 tags. Mix broad (economics, history) and specific (war financing, Roman Empire).
"""


class PublisherError(Exception):
    pass


def generate_youtube_metadata(
    topic_slug: str,
    script_text: str,
    angle: str,
    channel_profile: dict,
) -> dict:
    """Call Claude to generate YouTube title, description, tags, thumbnail brief."""
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    system = _METADATA_SYSTEM.format(
        channel_north_star=channel_profile.get("channel_north_star", ""),
        core_thesis=channel_profile.get("core_thesis", ""),
    )
    user = (
        f"Topic: {topic_slug}\n\n"
        f"Story angle: {angle}\n\n"
        f"Script excerpt (first 1000 chars):\n{script_text[:1000]}"
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError as e:
        raise PublisherError(f"Claude returned invalid JSON for metadata: {e}") from e


def check_duration_gate_passed(review_notes_text: str) -> bool:
    """Return False if review_notes contains a duration gate FAIL."""
    return "Duration gate FAIL" not in review_notes_text


def run_publisher(topic_dir: Path, topic_slug: str, channel_profile: dict) -> dict:
    """Run Publisher agent. Writes youtube_metadata.json. Returns status dict."""
    final_draft = topic_dir / "final_draft.mp4"
    review_notes_path = topic_dir / "review_notes.md"
    script_path = topic_dir / "script.md"

    if not final_draft.exists():
        raise PublisherError(f"final_draft.mp4 not found in {topic_dir}")
    if not review_notes_path.exists():
        raise PublisherError(f"review_notes.md not found in {topic_dir}")

    review_text = review_notes_path.read_text(encoding="utf-8")
    if not check_duration_gate_passed(review_text):
        raise PublisherError("duration_gate_fail")

    if script_path.exists():
        script_text = script_path.read_text(encoding="utf-8")
    else:
        logger.warning("Publisher: script.md not found — metadata will have no script context")
        script_text = ""

    # Query taste store for publisher learnings
    taste_store = TasteStore(settings.taste_dir)
    taste_learnings = taste_store.query("publisher metadata title description", n_results=3)
    angle = channel_profile.get("north_star", topic_slug)

    logger.info("Publisher: generating YouTube metadata for '%s'", topic_slug)
    metadata = generate_youtube_metadata(topic_slug, script_text, angle, channel_profile)

    # Attach taste learnings as internal notes (not uploaded)
    if taste_learnings:
        metadata["_taste_notes"] = [t["learning"] for t in taste_learnings]

    metadata_path = topic_dir / "youtube_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Publisher: youtube_metadata.json written")

    return {"status": "ready_for_review", "metadata_path": str(metadata_path)}
