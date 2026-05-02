"""Agent 1 — Researcher.

Queries the KB for hook/arc patterns, then uses Claude with web search
to surface the specific angle worth telling, and writes brief.md.
"""
import logging
from pathlib import Path

import anthropic

from mosaic.config.settings import settings
from mosaic.kb.store import KBStore
from mosaic.pipeline.taste_store import TasteStore

logger = logging.getLogger(__name__)

_RESEARCHER_SYSTEM = """\
You are a documentary researcher for IncentivesLab. Your job is to find the specific \
framing of a topic that will make a viewer distrust something they trusted before.

Channel north star: {channel_north_star}
Channel north star (viewer experience): {north_star}
Channel domain: {domain}
Arc template: {arc_template}
Core thesis: {core_thesis}

You have access to a web search tool. Use it to find primary sources, academic papers, \
news articles, and data. Aim for 5-10 high-quality sources.

Produce a brief in EXACTLY this format — no extra sections, no preamble:

# Topic Brief: [title]

## Story Angle
[The specific framing — not the topic, the angle. One sentence that would make someone \
who knows the topic say "I never thought of it that way".]

## Why This Serves the North Star
[How this makes the viewer distrust something they trusted before. Be specific about what \
belief is being challenged.]

## Hook Candidate
[Hook structure pulled from KB — name + technique + why it works for this topic]

## Arc Shape
[Arc shape: surface_reality → hidden_incentive → reframe — mapped to this topic's phases]

## Sources
- [URL] — [key fact or quote]
(5-10 sources minimum)

## Surprising Framing Test
[Answer: "Would someone who already knows about this topic be surprised by this framing?" \
If no, rewrite the Story Angle first.]
"""


def build_researcher_prompt(
    topic_slug: str,
    channel_profile: dict,
    kb_insights: list[dict],
    taste_learnings: list[dict],
) -> str:
    kb_section = ""
    if kb_insights:
        kb_section = "\n\nRelevant KB insights (hook/arc patterns from reference channels):\n"
        for item in kb_insights[:5]:
            kb_section += f"- [{item['channel']}] {item['insight']}\n"

    taste_section = ""
    if taste_learnings:
        taste_section = "\n\nPast taste learnings (apply these to your decisions):\n"
        for item in taste_learnings[:3]:
            taste_section += f"- {item['learning']} (delta: {item['delta']})\n"

    return (
        f"Research this topic for IncentivesLab: **{topic_slug}**\n\n"
        f"Find the angle that passes the surprising framing test. "
        f"Use web search to gather sources.{kb_section}{taste_section}"
    )


def parse_brief_angle(brief_text: str) -> str:
    """Extract the Story Angle value from brief.md text."""
    lines = brief_text.split("\n")
    in_angle = False
    for line in lines:
        if line.strip() == "## Story Angle":
            in_angle = True
            continue
        if in_angle:
            if line.startswith("##"):
                break
            if line.strip():
                return line.strip()
    return "unknown"


def _call_claude_with_search(system: str, user: str) -> str:
    """Call Claude with web search tool enabled, return text response."""
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    messages = [{"role": "user", "content": user}]
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=system,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 10}],
        messages=messages,
    )
    # Extract text from response (may contain tool_use blocks)
    text_parts = [
        block.text for block in response.content if hasattr(block, "text")
    ]
    return "\n".join(text_parts).strip()


def run_researcher(topic_dir: Path, topic_slug: str, channel_profile: dict) -> dict:
    """Run the Researcher agent. Writes topic_dir/brief.md. Returns dict with 'angle' key."""
    kb_store = KBStore(settings.kb_dir)
    taste_store = TasteStore(settings.taste_dir)

    # Query KB for hook/arc patterns matching the domain
    kb_insights = kb_store.query(
        f"hook structure arc shape {channel_profile.get('domain', '')}",
        n_results=8,
    )

    # Query taste store for researcher learnings from past runs
    taste_learnings = taste_store.query("researcher angle framing", n_results=3)

    system = _RESEARCHER_SYSTEM.format(
        channel_north_star=channel_profile.get("channel_north_star", ""),
        north_star=channel_profile.get("north_star", ""),
        domain=channel_profile.get("domain", ""),
        arc_template=channel_profile.get("arc_template", ""),
        core_thesis=channel_profile.get("core_thesis", ""),
    )
    user = build_researcher_prompt(topic_slug, channel_profile, kb_insights, taste_learnings)

    logger.info("Researcher: calling Claude with web search for topic '%s'", topic_slug)
    brief_text = _call_claude_with_search(system, user)

    brief_path = topic_dir / "brief.md"
    brief_path.write_text(brief_text, encoding="utf-8")
    logger.info("Researcher: brief.md written (%d chars)", len(brief_text))

    angle = parse_brief_angle(brief_text)
    return {"angle": angle}
