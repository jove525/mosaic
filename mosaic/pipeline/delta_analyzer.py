"""Delta Analyzer — post-publish feedback loop.

Reads your_feedback.md, compares against agent decisions stored in review_notes.md
and youtube_metadata.json, uses Claude to identify calibration gaps, stores
TasteEntry objects in TasteStore.
"""
import json
import logging
import re
from pathlib import Path

import anthropic

from mosaic.config.settings import settings
from mosaic.pipeline.taste_store import TasteStore, TasteEntry

logger = logging.getLogger(__name__)

_DELTA_SYSTEM = """\
You are a calibration analyst for an AI video production pipeline.

You will receive:
1. User feedback on a finished video (1-2 sentences, honest reaction)
2. Agent decisions: what the AI pipeline chose (title, description, eval score, duration)
3. The story angle used

Your job: identify where the agent was overconfident, underconfident, or calibrated.

Respond in EXACTLY this JSON format — no extra keys, no preamble:

[
  {
    "agent": "<researcher|scriptwriter|curator|assembler|publisher>",
    "decision": "<what the agent decided — one sentence>",
    "agent_confidence": "<high|medium|low>",
    "user_signal": "<what the user feedback reveals about this decision>",
    "delta": "<overconfidence|underconfidence|calibrated>",
    "learning": "<one-sentence rule for future runs>"
  }
]

Produce 2-4 entries. Only include agents where a real calibration gap exists.
"""


def _call_claude_for_delta(user_feedback: str, agent_context: dict) -> list[dict]:
    """Call Claude to produce delta entries. Returns list of dicts."""
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    user = (
        f"User feedback: {user_feedback}\n\n"
        f"Agent context:\n{json.dumps(agent_context, indent=2, ensure_ascii=False)}"
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=_DELTA_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    raw = response.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw.strip())


def build_agent_context(
    topic_slug: str,
    metadata: dict,
    review_notes_text: str,
) -> dict:
    """Build the agent context dict passed to Claude for delta analysis."""
    return {
        "topic": topic_slug,
        "title_chosen": metadata.get("title", ""),
        "tags_chosen": metadata.get("tags", []),
        "thumbnail_brief": metadata.get("thumbnail_brief", ""),
        "eval_score": _extract_eval_score(review_notes_text),
        "duration_result": _extract_duration_result(review_notes_text),
    }


def _extract_eval_score(review_notes_text: str) -> str:
    """Extract self-eval score line from review_notes.md."""
    for line in review_notes_text.split("\n"):
        if "Self-eval" in line or "eval score" in line.lower():
            return line.strip()
    return "unknown"


def _extract_duration_result(review_notes_text: str) -> str:
    """Extract duration result line from review_notes.md."""
    for line in review_notes_text.split("\n"):
        if "Duration" in line or "duration" in line:
            return line.strip()
    return "unknown"


def run_delta_analysis(topic_dir: Path, channel: str, topic_slug: str) -> dict:
    """Run delta analysis. Reads your_feedback.md, writes delta_analysis.md, stores TasteEntries."""
    feedback_path = topic_dir / "your_feedback.md"
    metadata_path = topic_dir / "youtube_metadata.json"
    review_notes_path = topic_dir / "review_notes.md"

    user_feedback = feedback_path.read_text(encoding="utf-8").strip()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    review_notes = review_notes_path.read_text(encoding="utf-8") if review_notes_path.exists() else ""

    agent_context = build_agent_context(topic_slug, metadata, review_notes)

    logger.info("Delta: calling Claude for calibration analysis on '%s'", topic_slug)
    delta_entries = _call_claude_for_delta(user_feedback, agent_context)

    taste_store = TasteStore(settings.taste_dir)
    video_id = f"{channel}__{topic_slug}"

    stored = 0
    for entry_dict in delta_entries:
        try:
            entry = TasteEntry(
                video_id=video_id,
                channel=channel,
                agent=entry_dict["agent"],
                decision=entry_dict["decision"],
                agent_confidence=entry_dict["agent_confidence"],
                user_signal=entry_dict["user_signal"],
                delta=entry_dict["delta"],
                learning=entry_dict["learning"],
            )
            taste_store.add(entry)
            stored += 1
        except (KeyError, Exception) as e:
            logger.warning("Delta: skipping malformed entry: %s — %s", entry_dict, e)

    # Write delta_analysis.md
    lines = [
        f"# Delta Analysis: {topic_slug}",
        f"\nUser feedback: {user_feedback}\n",
        "## Calibration Entries\n",
    ]
    for e in delta_entries:
        lines.append(f"### [{e.get('agent', '?')}] {e.get('delta', '?').upper()}")
        lines.append(f"- Decision: {e.get('decision', '')}")
        lines.append(f"- Confidence: {e.get('agent_confidence', '')}")
        lines.append(f"- User signal: {e.get('user_signal', '')}")
        lines.append(f"- Learning: **{e.get('learning', '')}**\n")

    analysis_path = topic_dir / "delta_analysis.md"
    analysis_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Delta: delta_analysis.md written, %d entries stored", stored)

    return {"stored": stored, "entries": delta_entries}
