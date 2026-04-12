"""Agent 3 — Curator.

Reads script.md, finds SAFE-licensed clips per narration line,
generates narration.mp3 via ElevenLabs, selects music.mp3.
Writes clip_manifest.json.
"""
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import anthropic
import requests

from mosaic.config.settings import settings
from mosaic.pipeline.taste_store import TasteStore

logger = logging.getLogger(__name__)

MUSIC_TRACKS = {
    "TENSION": {"genre": "dark ambient", "tempo": "slow", "track": "youtube_audio_lib_dark_ambient_01"},
    "GRIEF": {"genre": "dark ambient", "tempo": "slow", "track": "youtube_audio_lib_grief_01"},
    "REVELATION": {"genre": "cinematic build", "tempo": "moderate", "track": "youtube_audio_lib_cinematic_01"},
    "MOMENTUM": {"genre": "cinematic build", "tempo": "moderate", "track": "youtube_audio_lib_momentum_01"},
    "DEFIANCE": {"genre": "sparse unconventional", "tempo": "irregular", "track": "youtube_audio_lib_sparse_01"},
    "IRONY": {"genre": "sparse unconventional", "tempo": "irregular", "track": "youtube_audio_lib_irony_01"},
}

_CURATOR_MATCHING_SYSTEM = """\
You are evaluating whether a video clip is suitable for a specific narration line \
in a documentary. Assess three layers:

Layer 1 — Semantic match (REQUIRED): Does the clip relate to what's being said?
Layer 2 — Emotional match: Does the clip feel like what this moment needs?
  (Consider: shot type, energy, human presence, color temperature, motion speed)
Layer 3 — Narrative match: Does the clip serve where we are in the story arc?

Respond in JSON only:
{
  "semantic": true/false,
  "emotional": true/false,
  "narrative": true/false,
  "why": "one sentence explanation"
}
"""


def parse_narration_lines(script_text: str) -> list[dict]:
    """Parse script.md into list of narration line dicts with all tags."""
    lines = script_text.split("\n")
    results = []
    current: dict = {}
    line_ref = 0

    for line in lines:
        line = line.strip()
        if line.startswith("[NARRATION]"):
            if current.get("narration_text"):
                results.append(current)
            line_ref += 1
            current = {
                "line_ref": line_ref,
                "narration_text": line[len("[NARRATION]"):].strip(),
                "emotion": "",
                "visual": "",
                "direction": "",
                "pacing": "",
            }
        elif line.startswith("[EMOTION]") and current:
            current["emotion"] = line[len("[EMOTION]"):].strip()
        elif line.startswith("[VISUAL]") and current:
            current["visual"] = line[len("[VISUAL]"):].strip()
        elif line.startswith("[DIRECTION]") and current:
            current["direction"] = line[len("[DIRECTION]"):].strip()
        elif line.startswith("[PACING]") and current:
            current["pacing"] = line[len("[PACING]"):].strip()

    if current.get("narration_text"):
        results.append(current)
    return results


def score_clip_match(semantic: bool, emotional: bool, narrative: bool) -> int:
    """Return match score. Layer 1 (semantic) must be True or score is 0."""
    if not semantic:
        return 0
    return int(semantic) + int(emotional) + int(narrative)


def select_music_track(dominant_emotion: str) -> dict:
    """Select music track based on dominant emotion in script."""
    return MUSIC_TRACKS.get(dominant_emotion, MUSIC_TRACKS["MOMENTUM"])


def _detect_dominant_emotion(narration_lines: list[dict]) -> str:
    from collections import Counter
    emotions = [line.get("emotion", "") for line in narration_lines if line.get("emotion")]
    if not emotions:
        return "MOMENTUM"
    return Counter(emotions).most_common(1)[0][0]


def _generate_narration(narration_lines: list[dict]) -> bytes:
    """Call ElevenLabs API to generate narration MP3 from all narration text."""
    full_text = " ".join(line["narration_text"] for line in narration_lines)
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{settings.elevenlabs_voice_id}"
    headers = {
        "xi-api-key": settings.elevenlabs_api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": full_text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.content


def _search_internet_archive(query: str, max_results: int = 5) -> list[dict]:
    """Search Internet Archive for public domain video clips."""
    url = "https://archive.org/advancedsearch.php"
    params = {
        "q": f"{query} AND mediatype:movies",
        "fl[]": ["identifier", "title", "description"],
        "rows": max_results,
        "output": "json",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        docs = resp.json().get("response", {}).get("docs", [])
        results = []
        for doc in docs:
            identifier = doc.get("identifier", "")
            if identifier:
                results.append({
                    "url": f"https://archive.org/download/{identifier}/{identifier}.mp4",
                    "duration": 10.0,
                    "title": doc.get("title", ""),
                    "source_type": "internet_archive",
                    "license": "public_domain",
                    "license_tier": "SAFE",
                })
        return results
    except Exception as e:
        logger.warning("Internet Archive search failed for '%s': %s", query, e)
        return []


def _download_clip(url: str, dest_path: Path) -> Optional[Path]:
    """Download a clip via yt-dlp or direct HTTP."""
    import subprocess
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["yt-dlp", "-o", str(dest_path), "--no-playlist", url],
            capture_output=True, timeout=60,
        )
        if result.returncode == 0 and dest_path.exists():
            return dest_path
    except Exception:
        pass
    try:
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return dest_path
    except Exception as e:
        logger.warning("Failed to download clip from %s: %s", url, e)
        return None


def _call_claude_for_matching(narration_line: dict, clip_url: str) -> dict:
    """Ask Claude whether a clip matches the narration line across 3 layers."""
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    user = (
        f"Narration: \"{narration_line['narration_text']}\"\n"
        f"Emotion tag: {narration_line['emotion']}\n"
        f"Visual tag: {narration_line['visual']}\n"
        f"Story direction: {narration_line['direction']}\n\n"
        f"Clip URL: {clip_url}\n\n"
        f"Evaluate the three-layer match."
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        system=_CURATOR_MATCHING_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    text = response.content[0].text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"semantic": False, "emotional": False, "narrative": False, "why": "parse error"}


def run_curator(topic_dir: Path, channel_profile: dict) -> dict:
    """Run Curator agent. Reads script.md, writes clip_manifest.json, narration.mp3, music.mp3."""
    taste_store = TasteStore(settings.taste_dir)
    taste_learnings = taste_store.query("curator clip selection footage", n_results=3)
    if taste_learnings:
        logger.info("Curator: applying %d taste learnings", len(taste_learnings))

    script_path = topic_dir / "script.md"
    script_text = script_path.read_text(encoding="utf-8")
    narration_lines = parse_narration_lines(script_text)
    clips_dir = topic_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    manifest: list[dict] = []
    gaps = 0
    sourced = 0

    for line in narration_lines:
        line_ref = line["line_ref"]
        clip_filename = f"clip_{line_ref:03d}.mp4"
        clip_dest = clips_dir / clip_filename

        candidates = _search_internet_archive(line["visual"], max_results=5)
        selected = None

        for candidate in candidates:
            if candidate["license_tier"] != "SAFE":
                continue
            match = _call_claude_for_matching(line, candidate["url"])
            score = score_clip_match(
                semantic=match.get("semantic", False),
                emotional=match.get("emotional", False),
                narrative=match.get("narrative", False),
            )
            if score >= 2:
                downloaded = _download_clip(candidate["url"], clip_dest)
                if downloaded:
                    selected = {
                        "narration_line_ref": line_ref,
                        "narration_text": line["narration_text"],
                        "source_url": candidate["url"],
                        "local_path": f"clips/{clip_filename}",
                        "source_type": candidate["source_type"],
                        "license": candidate["license"],
                        "license_tier": candidate["license_tier"],
                        "semantic_match": match.get("semantic", False),
                        "emotional_match": match.get("emotional", False),
                        "narrative_match": match.get("narrative", False),
                        "duration_seconds": candidate.get("duration", 10.0),
                        "why_selected": match.get("why", ""),
                    }
                    sourced += 1
                    break
            time.sleep(0.5)

        if selected is None:
            gaps += 1
            logger.warning("Curator: no SAFE clip found for line %d — logging gap", line_ref)
            manifest.append({
                "narration_line_ref": line_ref,
                "narration_text": line["narration_text"],
                "source_url": None,
                "local_path": None,
                "source_type": "gap",
                "license": None,
                "license_tier": "GAP",
                "semantic_match": False,
                "emotional_match": False,
                "narrative_match": False,
                "duration_seconds": 3.0,
                "why_selected": "No SAFE clip found — Assembler will insert 3s black frame",
            })
        else:
            manifest.append(selected)

    manifest_path = topic_dir / "clip_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("Curator: generating narration via ElevenLabs")
    narration_bytes = _generate_narration(narration_lines)
    narration_path = topic_dir / "narration.mp3"
    narration_path.write_bytes(narration_bytes)

    dominant_emotion = _detect_dominant_emotion(narration_lines)
    track = select_music_track(dominant_emotion)
    music_path = topic_dir / "music.mp3"
    if not music_path.exists():
        logger.info(
            "Curator: music.mp3 not found — download '%s' from YouTube Audio Library and place at %s",
            track["track"], music_path,
        )
        music_path.write_bytes(b"")

    narration_size_kb = narration_path.stat().st_size / 1024
    est_duration_sec = narration_size_kb / 16
    minutes = int(est_duration_sec // 60)
    seconds = int(est_duration_sec % 60)
    narration_duration = f"{minutes}:{seconds:02d}"

    return {
        "clips_sourced": sourced,
        "clips_total": len(narration_lines),
        "gaps": gaps,
        "narration_duration": narration_duration,
        "music_track": track["track"],
    }
