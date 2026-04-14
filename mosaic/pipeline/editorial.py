"""Agent 3.5 — Editorial.

Reads raw_candidates.json. For each narration line:
- Checks clip cache for already-analyzed videos
- Watches new candidates via frame extraction + Whisper + Claude Haiku
- Trims usable segments via ffmpeg
- Caches analysis results
- Writes clip_manifest.json for Assembler

Source waterfall: cache → Prelinger → Internet Archive → Wikimedia → Pexels → needs_generated_visual
"""
import base64
import json
import logging
import re
import subprocess
from datetime import date
from pathlib import Path
from typing import Optional

import anthropic

from mosaic.config.settings import settings
from mosaic.kb.extractor import extract_frames, transcribe_video
from mosaic.pipeline.clip_cache import ClipCache, CachedVideo, CachedSegment

logger = logging.getLogger(__name__)

_WATCH_SYSTEM = """\
You are an editorial director reviewing archival footage for a documentary video.
Given frames from a video and its transcript, identify timestamp ranges that would
work as B-roll under the given narration line.

Be selective — only flag segments that genuinely match or complement the narration.
A segment showing Japanese soldiers is NOT a match for WWII American bond drives.
Multiple short segments are better than one long one.

Respond ONLY with valid JSON:
{
  "usable_segments": [
    {"start": <float seconds>, "end": <float seconds>, "description": "<what is shown>", "relevance": "<why it fits>"}
  ],
  "verdict": "useful" | "not_useful",
  "why": "<one sentence>"
}
"""


def build_watch_prompt(
    narration_text: str,
    emotion: str,
    visual_description: str,
    transcript_text: str,
    video_duration: float,
) -> str:
    return (
        f"Narration line: \"{narration_text}\"\n"
        f"Emotion: {emotion}\n"
        f"Intended visual: {visual_description}\n\n"
        f"Video duration: {video_duration:.1f}s\n"
        f"Video transcript:\n{transcript_text}\n\n"
        f"Review the frames and identify usable segments (minimum 5s, maximum 30s each). "
        f"If nothing matches the narration context, return verdict: not_useful."
    )


def parse_claude_segments(response_text: str) -> dict:
    """Parse Claude Haiku JSON response. Returns safe default on any parse error."""
    try:
        data = json.loads(response_text.strip())
        return {
            "verdict": data.get("verdict", "not_useful"),
            "usable_segments": data.get("usable_segments", []),
            "why": data.get("why", ""),
        }
    except (json.JSONDecodeError, AttributeError):
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return {
                    "verdict": data.get("verdict", "not_useful"),
                    "usable_segments": data.get("usable_segments", []),
                    "why": data.get("why", ""),
                }
            except json.JSONDecodeError:
                pass
    return {"verdict": "not_useful", "usable_segments": [], "why": "parse error"}


def trim_segment(source: Path, dest: Path, start: float, end: float) -> Optional[Path]:
    """Trim a segment from source video using ffmpeg. Returns dest path or None on failure."""
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(source),
        "-t", str(duration),
        "-c:v", "libx264", "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        str(dest),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        if dest.exists() and dest.stat().st_size > 1000:
            return dest
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else ""
        logger.warning("ffmpeg trim failed (%s → %s): %s", source.name, dest.name, stderr)
    return None
