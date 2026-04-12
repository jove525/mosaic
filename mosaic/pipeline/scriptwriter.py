"""Agent 2 — Scriptwriter.

Reads brief.md, queries KB for rhythm/beat patterns, writes script.md.
Every line is narration + emotion tag + visual instruction + pacing.
"""
import logging
import re
from pathlib import Path

import anthropic

from mosaic.config.settings import settings
from mosaic.kb.store import KBStore
from mosaic.pipeline.taste_store import TasteStore

logger = logging.getLogger(__name__)

# Average words per narration line → average seconds of narration
_WORDS_PER_MINUTE = 140

_SCRIPTWRITER_SYSTEM = """\
You are the Scriptwriter for IncentivesLab. You write scripts that function as a \
director's blueprint — every line is narration, emotion, visual instruction, and \
story direction simultaneously.

Channel north star (viewer experience): {north_star}
Channel north star (quality bar): {channel_north_star}
Arc template: {arc_template}

EMOTION vocabulary — use exactly these words: TENSION | REVELATION | MOMENTUM | GRIEF | DEFIANCE | IRONY

Script format — use EXACTLY this structure for every narration block:

[NARRATION] The text of the narration line. Use punctuation and <break time="Xs"/> for pacing.
[EMOTION]   REVELATION
[VISUAL]    Specific visual description — shot type, subject, energy
[DIRECTION] Editorial note — what this line does to the viewer
[PACING]    Cut instruction — e.g. "Hold 2s. Hard cut." or "Steady pace."

Section headers:
## Section: Hook (0:00–0:30)
## Section: Context (0:30–2:00)
## Section: Tension (2:00–7:00)
## Section: Resolution (7:00–9:30)
## Section: CTA (9:30–end)

Target: 10–12 minutes of narration. That is ~1,400–1,700 words of narration text.

Before finalizing: check — "Does this script make the viewer distrust something they \
trusted before?" If not, rewrite the hook section.
"""


def build_scriptwriter_prompt(
    brief_text: str,
    channel_profile: dict,
    kb_insights: list[dict],
    taste_learnings: list[dict],
) -> str:
    kb_section = ""
    if kb_insights:
        kb_section = "\n\nKB rhythm/beat patterns from reference channels:\n"
        for item in kb_insights[:6]:
            kb_section += f"- [{item['channel']}] {item['insight']}\n"

    taste_section = ""
    if taste_learnings:
        taste_section = "\n\nPast taste learnings — apply these:\n"
        for item in taste_learnings[:3]:
            taste_section += f"- {item['learning']} (delta: {item['delta']})\n"

    return (
        f"Write the full script from this brief:\n\n{brief_text}"
        f"{kb_section}{taste_section}"
    )


def parse_script_stats(script_text: str) -> dict:
    """Count narration lines, sections, and estimate runtime."""
    narration_lines = re.findall(r"^\[NARRATION\](.+)$", script_text, re.MULTILINE)
    sections = re.findall(r"^## Section:", script_text, re.MULTILINE)
    total_words = sum(len(line.split()) for line in narration_lines)
    est_minutes = round(total_words / _WORDS_PER_MINUTE, 1)
    return {
        "line_count": len(narration_lines),
        "section_count": len(sections),
        "est_minutes": est_minutes,
        "word_count": total_words,
    }


def _call_claude(system: str, user: str) -> str:
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text.strip()


def run_scriptwriter(topic_dir: Path, channel_profile: dict) -> dict:
    """Run the Scriptwriter agent. Reads brief.md, writes script.md."""
    brief_path = topic_dir / "brief.md"
    brief_text = brief_path.read_text(encoding="utf-8")

    kb_store = KBStore(settings.kb_dir)
    taste_store = TasteStore(settings.taste_dir)

    # Query KB for narration rhythm, emotional beat maps from reference channels
    kb_insights = kb_store.query(
        "narration rhythm emotional beats pacing documentary",
        n_results=8,
    )
    taste_learnings = taste_store.query("scriptwriter pacing script structure", n_results=3)

    system = _SCRIPTWRITER_SYSTEM.format(
        north_star=channel_profile.get("north_star", ""),
        channel_north_star=channel_profile.get("channel_north_star", ""),
        arc_template=channel_profile.get("arc_template", ""),
    )
    user = build_scriptwriter_prompt(brief_text, channel_profile, kb_insights, taste_learnings)

    logger.info("Scriptwriter: calling Claude to write script")
    script_text = _call_claude(system, user)

    script_path = topic_dir / "script.md"
    script_path.write_text(script_text, encoding="utf-8")

    stats = parse_script_stats(script_text)
    logger.info(
        "Scriptwriter: script.md written (%d narration lines, ~%.1f min)",
        stats["line_count"], stats["est_minutes"],
    )
    return stats
