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
    resp = requests.post(url, headers=headers, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.content


def _generate_search_queries(visual_description: str, narration_text: str) -> list[str]:
    """Ask Claude to convert a [VISUAL] description into short archival search queries."""
    client = anthropic.Anthropic()
    prompt = (
        f"You are helping source archival footage for a documentary video.\n\n"
        f"Narration: \"{narration_text}\"\n"
        f"Visual description: \"{visual_description}\"\n\n"
        f"Generate 3 short search queries (3-5 words each) suitable for searching "
        f"Internet Archive or Wikimedia Commons for real archival footage that would "
        f"match this moment. Prioritize historically specific terms (e.g. 'WWII war bond drive 1943', "
        f"'Norman Rockwell war poster', 'FDR cabinet meeting'). "
        f"Return ONLY a JSON array of 3 strings, nothing else."
    )
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            try:
                queries = json.loads(match.group())
                return [q for q in queries if isinstance(q, str)][:3]
            except json.JSONDecodeError:
                pass
    except Exception as e:
        logger.warning("Query generation failed: %s", e)
    # Fallback: extract key nouns from visual description
    words = visual_description.split()[:6]
    return [" ".join(words)]


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
                # Try to find a real downloadable video file via metadata
                results.append({
                    "identifier": identifier,
                    "url": None,  # resolved in _resolve_ia_download_url
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


def _resolve_ia_download_url(identifier: str) -> Optional[str]:
    """Fetch IA item metadata and return the first downloadable mp4 URL."""
    try:
        meta_url = f"https://archive.org/metadata/{identifier}"
        resp = requests.get(meta_url, timeout=15)
        resp.raise_for_status()
        files = resp.json().get("files", [])
        for f in files:
            name = f.get("name", "")
            if name.lower().endswith(".mp4"):
                return f"https://archive.org/download/{identifier}/{name}"
        # fallback: try .ogv or .mpeg
        for f in files:
            name = f.get("name", "")
            if name.lower().endswith((".ogv", ".mpeg", ".mov")):
                return f"https://archive.org/download/{identifier}/{name}"
    except Exception as e:
        logger.warning("IA metadata fetch failed for '%s': %s", identifier, e)
    return None


def _search_wikimedia(query: str, max_results: int = 5) -> list[dict]:
    """Search Wikimedia Commons for CC-licensed video clips."""
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": f"{query} filetype:video",
        "srnamespace": "6",  # File namespace
        "srlimit": max_results,
        "format": "json",
    }
    headers = {"User-Agent": "MosaicPipeline/1.0 (documentary video tool; contact@example.com)"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        items = resp.json().get("query", {}).get("search", [])
        results = []
        for item in items:
            title = item.get("title", "")
            if not title.startswith("File:"):
                continue
            filename = title[len("File:"):]
            # Build direct Commons file URL
            file_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{requests.utils.quote(filename)}"
            results.append({
                "url": file_url,
                "duration": 10.0,
                "title": filename,
                "source_type": "wikimedia_commons",
                "license": "cc_licensed",
                "license_tier": "SAFE",
            })
        return results
    except Exception as e:
        logger.warning("Wikimedia search failed for '%s': %s", query, e)
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
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"semantic": False, "emotional": False, "narrative": False, "why": "parse error"}


def run_curator(topic_dir: Path, channel_profile: dict) -> dict:
    """Run Curator agent. Reads script.md, writes raw_candidates.json, narration.mp3, music.mp3."""
    taste_store = TasteStore(settings.taste_dir)
    taste_learnings = taste_store.query("curator clip selection footage", n_results=3)
    if taste_learnings:
        logger.info("Curator: applying %d taste learnings", len(taste_learnings))

    script_path = topic_dir / "script.md"
    script_text = script_path.read_text(encoding="utf-8")
    narration_lines = parse_narration_lines(script_text)
    clips_dir = topic_dir / "clips" / "raw"
    clips_dir.mkdir(parents=True, exist_ok=True)
    (topic_dir / "clips" / "final").mkdir(parents=True, exist_ok=True)

    raw_candidates: list[dict] = []

    for line in narration_lines:
        line_ref = line["line_ref"]
        queries = []
        try:
            queries = _generate_search_queries(line["visual"], line["narration_text"])
        except Exception as e:
            logger.warning("Curator: query generation error line %d: %s", line_ref, e)
        if not queries:
            queries = ["WWII archival footage"]
        logger.info("Curator: line %d queries: %s", line_ref, queries)

        candidates = []
        for query in queries:
            # Prelinger Archives first
            prelinger_results = _search_internet_archive(
                f"collection:prelinger {query}", max_results=2
            )
            for r in prelinger_results:
                if r.get("identifier"):
                    url = _resolve_ia_download_url(r["identifier"])
                    if url:
                        r["url"] = url
                        r["source"] = "prelinger"
                        candidates.append(r)

            # General Internet Archive
            ia_results = _search_internet_archive(query, max_results=2)
            for r in ia_results:
                if r.get("identifier"):
                    url = _resolve_ia_download_url(r["identifier"])
                    if url:
                        r["url"] = url
                        candidates.append(r)

            # Wikimedia Commons
            wm_results = _search_wikimedia(query, max_results=2)
            candidates.extend(wm_results)

        # Download all unique candidates for this line
        downloaded_candidates = []
        seen_identifiers = set()
        for candidate in candidates:
            identifier = candidate.get("identifier") or candidate.get("title", "unknown")
            if identifier in seen_identifiers:
                continue
            seen_identifiers.add(identifier)
            safe_name = re.sub(r"[^\w\-]", "_", identifier)[:80]
            dest = clips_dir / f"{safe_name}.mp4"
            if dest.exists() and dest.stat().st_size > 10_000:
                logger.info("Curator: %s already cached", safe_name)
            else:
                downloaded = _download_clip(candidate["url"], dest)
                if not downloaded:
                    continue
            downloaded_candidates.append({
                "identifier": identifier,
                "source": candidate.get("source", candidate.get("source_type", "unknown")),
                "url": candidate.get("url", ""),
                "local_path": f"clips/raw/{dest.name}",
                "title": candidate.get("title", ""),
                "license": candidate.get("license", "public_domain"),
            })

        raw_candidates.append({
            "narration_line_ref": line_ref,
            "narration_text": line["narration_text"],
            "emotion": line.get("emotion", ""),
            "visual_description": line.get("visual", ""),
            "candidates": downloaded_candidates,
        })

    raw_candidates_path = topic_dir / "raw_candidates.json"
    raw_candidates_path.write_text(json.dumps(raw_candidates, indent=2), encoding="utf-8")

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
        "candidates_total": len(raw_candidates),
        "candidates_with_downloads": sum(1 for r in raw_candidates if r["candidates"]),
        "narration_duration": narration_duration,
        "music_track": track["track"],
    }
