"""Agent 3 — Curator.

Reads script.md, finds SAFE-licensed clips per narration line,
generates narration.mp3 via ElevenLabs, selects music.mp3.
Writes clip_manifest.json, narration.mp3, music.mp3.

Clip sourcing flow per narration line:
  1. Generate search queries (Haiku)
  2. Per source in hierarchy: get transcript (no download) → identify timestamp window
     → download only that window → run 3-layer QC
  3. If all sources exhausted: diversify queries, retry once
  4. If still no clip and AI cap not hit: Kling generation
  5. If AI cap hit or Kling unavailable: declare gap
"""
import base64
import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from shutil import which

# Resolve yt-dlp from the active venv's Scripts folder, falling back to PATH
_YT_DLP = str(Path(sys.executable).parent / "yt-dlp.exe") if sys.platform == "win32" else str(Path(sys.executable).parent / "yt-dlp")
if not Path(_YT_DLP).exists():
    _YT_DLP = which("yt-dlp") or "yt-dlp"
import time
from typing import Optional

import anthropic
import requests

from mosaic.config.settings import settings
from mosaic.pipeline.qc import run_qc
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


# ---------------------------------------------------------------------------
# Script parsing
# ---------------------------------------------------------------------------

def parse_narration_lines(script_text: str) -> list[dict]:
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


# ---------------------------------------------------------------------------
# Search query generation
# ---------------------------------------------------------------------------

def _generate_search_queries(visual_description: str, narration_text: str) -> list[str]:
    client = anthropic.Anthropic()
    prompt = (
        f"You are helping source archival footage for a documentary video.\n\n"
        f"Narration: \"{narration_text}\"\n"
        f"Visual description: \"{visual_description}\"\n\n"
        f"Generate 3 short search queries (3-5 words each) suitable for searching "
        f"Internet Archive, Wikimedia Commons, or YouTube for real footage. "
        f"Prioritize historically specific terms. "
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
            queries = json.loads(match.group())
            return [q for q in queries if isinstance(q, str)][:3]
    except Exception as e:
        logger.warning("Curator: query generation failed: %s", e)
    words = visual_description.split()[:6]
    return [" ".join(words)]


def _diversify_queries(
    visual_description: str,
    narration_text: str,
    failed_queries: list[str],
) -> list[str]:
    """Ask Haiku for alternative queries after initial ones failed."""
    client = anthropic.Anthropic()
    prompt = (
        f"Previous search queries failed to find suitable footage:\n"
        f"{json.dumps(failed_queries)}\n\n"
        f"Narration: \"{narration_text}\"\n"
        f"Visual: \"{visual_description}\"\n\n"
        f"Generate 3 alternative search queries using different keywords, "
        f"synonyms, or broader/narrower terms. "
        f"Return ONLY a JSON array of 3 strings."
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
            queries = json.loads(match.group())
            return [q for q in queries if isinstance(q, str)][:3]
    except Exception as e:
        logger.warning("Curator: query diversification failed: %s", e)
    return []


# ---------------------------------------------------------------------------
# Source search
# ---------------------------------------------------------------------------

def _search_internet_archive(query: str, max_results: int = 5) -> list[dict]:
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
                    "identifier": identifier,
                    "title": doc.get("title", ""),
                    "description": doc.get("description", ""),
                    "source_type": "internet_archive",
                    "license": "public_domain",
                    "license_tier": "SAFE",
                    "url": None,
                })
        return results
    except Exception as e:
        logger.warning("Curator: IA search failed for '%s': %s", query, e)
        return []


def _resolve_ia_item(identifier: str) -> Optional[dict]:
    """Fetch IA metadata. Returns {url, duration_seconds} or None."""
    try:
        resp = requests.get(f"https://archive.org/metadata/{identifier}", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        meta = data.get("metadata", {})
        files = data.get("files", [])

        # Duration from metadata
        duration = 0.0
        raw_duration = meta.get("runtime") or meta.get("duration") or ""
        if raw_duration:
            parts = str(raw_duration).split(":")
            try:
                if len(parts) == 3:
                    duration = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                elif len(parts) == 2:
                    duration = int(parts[0]) * 60 + float(parts[1])
                else:
                    duration = float(parts[0])
            except (ValueError, IndexError):
                duration = 0.0

        # Find downloadable video file
        for f in files:
            name = f.get("name", "")
            if name.lower().endswith(".mp4"):
                file_duration = float(f.get("length", duration) or duration)
                return {
                    "url": f"https://archive.org/download/{identifier}/{name}",
                    "duration_seconds": file_duration,
                }
        for f in files:
            name = f.get("name", "")
            if name.lower().endswith((".ogv", ".mpeg", ".mov")):
                file_duration = float(f.get("length", duration) or duration)
                return {
                    "url": f"https://archive.org/download/{identifier}/{name}",
                    "duration_seconds": file_duration,
                }
    except Exception as e:
        logger.warning("Curator: IA metadata failed for '%s': %s", identifier, e)
    return None


def _search_wikimedia(query: str, max_results: int = 5) -> list[dict]:
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": f"{query} filetype:video",
        "srnamespace": "6",
        "srlimit": max_results,
        "format": "json",
    }
    headers = {"User-Agent": "MosaicPipeline/1.0"}
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
            file_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{requests.utils.quote(filename)}"
            results.append({
                "url": file_url,
                "title": filename,
                "description": item.get("snippet", ""),
                "source_type": "wikimedia_commons",
                "license": "cc_licensed",
                "license_tier": "SAFE",
                "duration_seconds": None,
            })
        return results
    except Exception as e:
        logger.warning("Curator: Wikimedia search failed for '%s': %s", query, e)
        return []


def _search_youtube_cc(query: str, max_results: int = 5) -> list[dict]:
    """Search YouTube for CC-licensed clips. Requires YOUTUBE_API_KEY."""
    if not settings.youtube_api_key:
        return []
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "videoLicense": "creativeCommon",
        "maxResults": max_results,
        "key": settings.youtube_api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        results = []
        for item in items:
            video_id = item.get("id", {}).get("videoId", "")
            snippet = item.get("snippet", {})
            if video_id:
                results.append({
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "title": snippet.get("title", ""),
                    "description": snippet.get("description", ""),
                    "source_type": "cc_youtube",
                    "license": "cc_licensed",
                    "license_tier": "SAFE",
                    "duration_seconds": None,
                    "has_captions": True,
                })
        return results
    except Exception as e:
        logger.warning("Curator: YouTube CC search failed — %s (flag: YOUTUBE_API_KEY may be missing)", e)
        return []


# ---------------------------------------------------------------------------
# Transcript + window selection
# ---------------------------------------------------------------------------

def _get_youtube_transcript(url: str, tmp_dir: Path) -> Optional[str]:
    """Pull auto-generated captions only, no video download."""
    vtt_path = tmp_dir / "transcript.vtt"
    try:
        result = subprocess.run(
            [_YT_DLP, "--write-auto-sub", "--sub-lang", "en",
             "--sub-format", "vtt", "--skip-download",
             "-o", str(tmp_dir / "transcript"), url],
            capture_output=True, timeout=60,
        )
        # yt-dlp may add .en.vtt suffix
        for candidate in tmp_dir.glob("transcript*.vtt"):
            return candidate.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning("Curator: transcript fetch failed for %s: %s", url, e)
    return None


def _select_window_from_transcript(
    transcript_text: str,
    narration_text: str,
    visual_tag: str,
) -> Optional[dict]:
    """Ask Haiku to identify best timestamp window from transcript text."""
    client = anthropic.Anthropic()
    prompt = (
        f"You are selecting a 20-30 second clip from a video transcript.\n\n"
        f"Narration to match: \"{narration_text}\"\n"
        f"Visual needed: \"{visual_tag}\"\n\n"
        f"Transcript (with timestamps):\n{transcript_text[:4000]}\n\n"
        f"Identify the single best 20-30 second window. "
        f"Return JSON only: {{\"start\": \"MM:SS\", \"end\": \"MM:SS\", \"reason\": \"one sentence\"}}"
    )
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        logger.warning("Curator: window selection failed: %s", e)
    return None


def _select_window_from_context(
    title: str,
    description: str,
    duration_seconds: float,
    narration_text: str,
    visual_tag: str,
) -> dict:
    """For items without transcripts: Haiku uses title/description to pick window,
    defaulting to the middle third."""
    client = anthropic.Anthropic()
    duration_str = f"{int(duration_seconds // 60)}:{int(duration_seconds % 60):02d}"
    prompt = (
        f"A video has no transcript. Use the metadata to estimate the best 20-30s window.\n\n"
        f"Title: \"{title}\"\n"
        f"Description: \"{description[:500]}\"\n"
        f"Total duration: {duration_str}\n\n"
        f"Narration to match: \"{narration_text}\"\n"
        f"Visual needed: \"{visual_tag}\"\n\n"
        f"If you cannot determine a specific window, use the middle third. "
        f"Return JSON only: {{\"start\": \"MM:SS\", \"end\": \"MM:SS\", \"reason\": \"one sentence\"}}"
    )
    default_start = duration_seconds / 3
    default_end = min(default_start + 25, duration_seconds * 2 / 3)

    def _fmt(s: float) -> str:
        return f"{int(s // 60)}:{int(s % 60):02d}"

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        logger.warning("Curator: context window selection failed: %s", e)

    return {"start": _fmt(default_start), "end": _fmt(default_end), "reason": "middle-third default"}


def _download_section(url: str, start: str, end: str, dest: Path) -> Optional[Path]:
    """Download a specific time window via yt-dlp --download-sections."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [_YT_DLP,
             "--download-sections", f"*{start}-{end}",
             "--no-playlist",
             "-o", str(dest), url],
            capture_output=True, timeout=120,
        )
        if result.returncode == 0 and dest.exists() and dest.stat().st_size > 10_000:
            return dest
    except Exception as e:
        logger.warning("Curator: yt-dlp section download failed: %s", e)
    return None


def _download_and_trim(url: str, start: str, end: str, dest: Path) -> Optional[Path]:
    """For direct HTTP sources: full download then ffmpeg trim."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".raw.mp4")
    try:
        # Download full file
        result = subprocess.run(
            [_YT_DLP, "-o", str(tmp), "--no-playlist", url],
            capture_output=True, timeout=300,
        )
        if result.returncode != 0 or not tmp.exists():
            resp = requests.get(url, timeout=120, stream=True)
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)

        if not tmp.exists() or tmp.stat().st_size < 10_000:
            return None

        # Trim with ffmpeg — scale filter ensures even dimensions (archival footage fix)
        result = subprocess.run(
            ["ffmpeg", "-ss", start, "-to", end, "-i", str(tmp),
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
             "-c:v", "libx264", "-c:a", "aac",
             str(dest), "-y"],
            capture_output=True, timeout=60,
        )
        if result.returncode != 0:
            logger.warning("ffmpeg trim failed (%s → %s): %s", tmp.name, dest.name, result.stderr.decode(errors="ignore")[:200])
        tmp.unlink(missing_ok=True)
        if dest.exists() and dest.stat().st_size > 10_000:
            return dest
    except Exception as e:
        logger.warning("Curator: full download + trim failed for %s: %s", url, e)
        tmp.unlink(missing_ok=True)
    return None


# ---------------------------------------------------------------------------
# AI generation (Kling fallback)
# ---------------------------------------------------------------------------

def _generate_clip_kling(visual_tag: str, narration_text: str, dest: Path) -> Optional[Path]:
    """Generate a clip via Kling API. Returns dest path or None."""
    if not settings.kling_api_key or not settings.kling_api_secret:
        logger.warning("Curator: Kling API keys not configured — skipping AI generation")
        return None

    dest.parent.mkdir(parents=True, exist_ok=True)
    prompt_text = f"{visual_tag}. {narration_text[:120]}"

    try:
        import hmac
        import hashlib

        # Kling API v1 — JWT auth (HS256, header.payload.signature)
        api_url = "https://api.klingai.com/v1/videos/text2video"
        now = int(time.time())

        def _b64url(data: bytes) -> str:
            import base64
            return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

        header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
        payload = _b64url(json.dumps({
            "iss": settings.kling_api_key,
            "exp": now + 1800,
            "nbf": now - 5,
        }).encode())
        sig = _b64url(hmac.new(
            settings.kling_api_secret.encode(),
            f"{header}.{payload}".encode(),
            hashlib.sha256,
        ).digest())
        jwt_token = f"{header}.{payload}.{sig}"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {jwt_token}",
        }
        payload = {
            "prompt": prompt_text,
            "duration": 5,
            "aspect_ratio": "16:9",
            "mode": "std",
        }

        resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        task_id = resp.json().get("data", {}).get("task_id")
        if not task_id:
            logger.warning("Curator: Kling no task_id returned")
            return None

        # Poll for completion (max 3 min)
        for _ in range(36):
            time.sleep(5)
            poll = requests.get(
                f"https://api.klingai.com/v1/videos/text2video/{task_id}",
                headers=headers, timeout=15,
            )
            poll.raise_for_status()
            data = poll.json().get("data", {})
            status = data.get("task_status")
            if status == "succeed":
                video_url = (
                    data.get("task_result", {})
                    .get("videos", [{}])[0]
                    .get("url")
                )
                if video_url:
                    r = requests.get(video_url, timeout=60, stream=True)
                    r.raise_for_status()
                    with open(dest, "wb") as f:
                        for chunk in r.iter_content(chunk_size=65536):
                            f.write(chunk)
                    logger.info("Curator: Kling clip generated → %s", dest.name)
                    return dest
                break
            elif status in ("failed", "cancelled"):
                logger.warning("Curator: Kling task %s failed/cancelled", task_id)
                break

    except Exception as e:
        logger.warning("Curator: Kling generation error: %s", e)
    return None


# ---------------------------------------------------------------------------
# Per-line sourcing orchestrator
# ---------------------------------------------------------------------------

def _source_clip_for_line(
    line: dict,
    clips_dir: Path,
    ai_generated_count: int,
    total_lines: int,
) -> dict:
    """Try all sources for one narration line. Returns manifest entry dict."""
    line_ref = line["line_ref"]
    visual_tag = line.get("visual", "")
    narration_text = line["narration_text"]
    emotion_tag = line.get("emotion", "")
    direction_tag = line.get("direction", "")

    all_queries_tried: list[str] = []
    diversification_attempts = 0

    def _try_queries(queries: list[str]) -> Optional[dict]:
        """Run source hierarchy for a set of queries. Returns manifest entry or None."""
        for query in queries:
            candidates: list[dict] = []

            # Source hierarchy
            prelinger = _search_internet_archive(f"collection:prelinger {query}", max_results=3)
            candidates.extend(prelinger)
            candidates.extend(_search_internet_archive(query, max_results=3))
            candidates.extend(_search_wikimedia(query, max_results=3))
            candidates.extend(_search_youtube_cc(query, max_results=3))

            for candidate in candidates[:settings.curator_max_candidates_per_source]:
                result = _try_candidate(candidate, line_ref, clips_dir, visual_tag, narration_text, emotion_tag, direction_tag)
                if result:
                    result["query_used"] = query
                    return result
        return None

    def _try_candidate(
        candidate: dict,
        line_ref: int,
        clips_dir: Path,
        visual_tag: str,
        narration_text: str,
        emotion_tag: str,
        direction_tag: str,
    ) -> Optional[dict]:
        source_type = candidate.get("source_type", "")
        is_youtube = source_type == "cc_youtube" or (
            candidate.get("url", "") and "youtube.com" in candidate.get("url", "")
        )

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_dir = Path(tmp_str)
            url = candidate.get("url", "")
            title = candidate.get("title", "")
            description = candidate.get("description", "")

            # --- Resolve IA item ---
            if source_type == "internet_archive" and not url:
                identifier = candidate.get("identifier", "")
                resolved = _resolve_ia_item(identifier)
                if not resolved:
                    return None
                url = resolved["url"]
                candidate["duration_seconds"] = resolved["duration_seconds"]
                title = title or identifier
                is_youtube = False

            duration = candidate.get("duration_seconds") or 0.0

            # Duration cap for non-YouTube sources
            if not is_youtube and duration and duration > settings.ia_max_duration_seconds:
                logger.info(
                    "Curator: line %d — skipping %s (%.0fs > %ds cap)",
                    line_ref, title[:40], duration, settings.ia_max_duration_seconds,
                )
                return None

            # --- Determine timestamp window ---
            window = None
            if is_youtube:
                transcript = _get_youtube_transcript(url, tmp_dir)
                if transcript:
                    window = _select_window_from_transcript(transcript, narration_text, visual_tag)
            if not window:
                # Use title/description context (IA items, Wikimedia, no-transcript YouTube)
                window = _select_window_from_context(
                    title, description,
                    duration or 30.0,
                    narration_text, visual_tag,
                )

            start = window.get("start", "0:00")
            end = window.get("end", "0:25")
            window_reason = window.get("reason", "")

            # --- Download ---
            safe_name = re.sub(r"[^\w\-]", "_", (title or "clip")[:50])
            dest = clips_dir / f"line{line_ref:02d}_{safe_name}_{hash(url) & 0xFFFF:04x}.mp4"

            if dest.exists() and dest.stat().st_size > 10_000:
                logger.info("Curator: line %d — using cached %s", line_ref, dest.name)
                downloaded = dest
            elif is_youtube:
                downloaded = _download_section(url, start, end, dest)
            else:
                downloaded = _download_and_trim(url, start, end, dest)

            if not downloaded:
                return None

            # --- QC ---
            qc_result = run_qc(downloaded, visual_tag, narration_text, emotion_tag, direction_tag)
            if not qc_result["passed"]:
                logger.info(
                    "Curator: line %d — QC failed (%s): tech=%s sem=%.2f ctx=%s",
                    line_ref,
                    title[:30],
                    qc_result["technical"]["reason"] or "ok",
                    qc_result["semantic"]["score"],
                    qc_result["contextual"]["why"],
                )
                downloaded.unlink(missing_ok=True)
                return None

            logger.info("Curator: line %d — clip accepted: %s", line_ref, dest.name)
            return {
                "narration_line_ref": line_ref,
                "narration_text": narration_text,
                "status": "sourced",
                "source_url": url,
                "local_path": f"clips/{dest.name}",
                "source_type": source_type,
                "license": candidate.get("license", "public_domain"),
                "license_tier": candidate.get("license_tier", "SAFE"),
                "timestamp_window": {"start": start, "end": end},
                "window_reason": window_reason,
                "qc": {
                    "technical": qc_result["technical"]["reason"] or "ok",
                    "clip_score": qc_result["semantic"]["score"],
                    "haiku_emotional": qc_result["contextual"]["emotional"],
                    "haiku_narrative": qc_result["contextual"]["narrative"],
                    "haiku_why": qc_result["contextual"]["why"],
                },
                "attempts": 1,
            }

    # --- Main sourcing loop ---
    queries = _generate_search_queries(visual_tag, narration_text)
    all_queries_tried.extend(queries)

    result = _try_queries(queries)
    if result:
        return result

    # Query diversification
    while diversification_attempts < settings.curator_max_query_diversifications:
        diversification_attempts += 1
        alt_queries = _diversify_queries(visual_tag, narration_text, all_queries_tried)
        if not alt_queries:
            break
        all_queries_tried.extend(alt_queries)
        logger.info("Curator: line %d — diversification attempt %d", line_ref, diversification_attempts)
        result = _try_queries(alt_queries)
        if result:
            result["query_diversifications"] = diversification_attempts
            return result

    # --- AI generation ---
    ai_cap = int(total_lines * settings.ai_generated_cap_pct)
    if ai_generated_count < ai_cap:
        logger.info("Curator: line %d — escalating to Kling AI generation", line_ref)
        safe_name = re.sub(r"[^\w\-]", "_", visual_tag[:40])
        dest = clips_dir / f"line{line_ref:02d}_ai_{safe_name}.mp4"
        generated = _generate_clip_kling(visual_tag, narration_text, dest)
        if generated:
            return {
                "narration_line_ref": line_ref,
                "narration_text": narration_text,
                "status": "ai_generated",
                "source_url": "",
                "local_path": f"clips/{dest.name}",
                "source_type": "ai_generated",
                "license": "ai_generated",
                "license_tier": "SAFE",
                "timestamp_window": {"start": "0:00", "end": "0:05"},
                "window_reason": "AI generated",
                "qc": {"technical": "ok", "clip_score": 0.0, "haiku_emotional": True, "haiku_narrative": True, "haiku_why": "ai_generated"},
                "attempts": diversification_attempts + 1,
            }
    else:
        logger.warning(
            "Curator: line %d — AI cap reached (%d/%d) — declaring gap",
            line_ref, ai_generated_count, ai_cap,
        )

    # --- Gap ---
    logger.warning("Curator: line %d — GAP declared. Queries tried: %s", line_ref, all_queries_tried)
    return {
        "narration_line_ref": line_ref,
        "narration_text": narration_text,
        "status": "gap",
        "source_url": "",
        "local_path": "",
        "source_type": "gap",
        "license": "",
        "license_tier": "",
        "timestamp_window": {},
        "window_reason": "no suitable clip found",
        "qc": {},
        "attempts": diversification_attempts + 1,
        "gap_reason": f"all sources exhausted after {len(all_queries_tried)} queries",
    }


# ---------------------------------------------------------------------------
# Narration + music
# ---------------------------------------------------------------------------

def _generate_narration(narration_lines: list[dict]) -> bytes:
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


def _detect_dominant_emotion(narration_lines: list[dict]) -> str:
    from collections import Counter
    emotions = [line.get("emotion", "") for line in narration_lines if line.get("emotion")]
    if not emotions:
        return "MOMENTUM"
    return Counter(emotions).most_common(1)[0][0]


def select_music_track(dominant_emotion: str) -> dict:
    return MUSIC_TRACKS.get(dominant_emotion, MUSIC_TRACKS["MOMENTUM"])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_curator(topic_dir: Path, channel_profile: dict) -> dict:
    """Run Curator agent. Writes clip_manifest.json, narration.mp3, music.mp3."""
    taste_store = TasteStore(settings.taste_dir)
    taste_learnings = taste_store.query("curator clip selection footage", n_results=3)
    if taste_learnings:
        logger.info("Curator: applying %d taste learnings", len(taste_learnings))

    script_path = topic_dir / "script.md"
    script_text = script_path.read_text(encoding="utf-8")
    narration_lines = parse_narration_lines(script_text)
    if not narration_lines:
        raise ValueError(f"No narration lines parsed from {script_path}")

    clips_dir = topic_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    total_lines = len(narration_lines)
    ai_generated_count = 0
    gap_count = 0
    manifest: list[dict] = []

    for line in narration_lines:
        entry = _source_clip_for_line(line, clips_dir, ai_generated_count, total_lines)
        if entry["status"] == "ai_generated":
            ai_generated_count += 1
        elif entry["status"] == "gap":
            gap_count += 1
        manifest.append(entry)

    # Validate completeness — every line must have an entry
    manifest_refs = {e["narration_line_ref"] for e in manifest}
    for line in narration_lines:
        if line["line_ref"] not in manifest_refs:
            logger.error("Curator: line %d missing from manifest — pipeline integrity error", line["line_ref"])

    manifest_path = topic_dir / "clip_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Narration
    narration_path = topic_dir / "narration.mp3"
    if narration_path.exists() and narration_path.stat().st_size > 1000:
        logger.info("Curator: narration.mp3 exists — skipping ElevenLabs")
    else:
        narration_bytes = _generate_narration(narration_lines)
        narration_path.write_bytes(narration_bytes)

    # Music
    dominant_emotion = _detect_dominant_emotion(narration_lines)
    track = select_music_track(dominant_emotion)
    music_path = topic_dir / "music.mp3"
    if not music_path.exists():
        logger.info(
            "Curator: music.mp3 not found — download '%s' from YouTube Audio Library → %s",
            track["track"], music_path,
        )
        music_path.write_bytes(b"")

    sourced = sum(1 for e in manifest if e["status"] == "sourced")
    coverage_pct = round((sourced + ai_generated_count) / total_lines * 100, 1)

    logger.info(
        "Curator: %d/%d sourced | %d AI-generated | %d gaps | %.1f%% coverage",
        sourced, total_lines, ai_generated_count, gap_count, coverage_pct,
    )

    if gap_count > 0:
        gap_lines = [e["narration_line_ref"] for e in manifest if e["status"] == "gap"]
        logger.warning("Curator: GAPS at lines %s — manual sourcing required", gap_lines)

    narration_size_kb = narration_path.stat().st_size / 1024
    est_duration_sec = narration_size_kb / 16
    minutes = int(est_duration_sec // 60)
    seconds = int(est_duration_sec % 60)

    return {
        "total_lines": total_lines,
        "sourced": sourced,
        "ai_generated": ai_generated_count,
        "gaps": gap_count,
        "coverage_pct": coverage_pct,
        "gap_lines": [e["narration_line_ref"] for e in manifest if e["status"] == "gap"],
        "narration_duration": f"{minutes}:{seconds:02d}",
        "music_track": track["track"],
    }
