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


def watch_candidate(
    video_path: Path,
    line: dict,
    frame_dir: Path,
) -> dict:
    """Extract frames + transcribe + ask Claude Haiku to find usable segments.

    Returns parsed segment dict. Never raises — returns not_useful on any error.
    """
    not_useful = {"verdict": "not_useful", "usable_segments": [], "why": ""}

    # Extract frames
    try:
        frame_set = extract_frames(video_path, frame_dir, interval_seconds=4)
        frame_paths = frame_set.frame_paths[:40]  # cap at 40 frames (~2.7 min coverage)
    except Exception as e:
        logger.warning("Frame extraction failed for %s: %s", video_path.name, e)
        return not_useful

    # Transcribe audio
    transcript_text = ""
    try:
        segments = transcribe_video(video_path)
        transcript_text = "\n".join(
            f"[{s.start:.1f}s-{s.end:.1f}s] {s.text}" for s in segments[:60]
        )
    except Exception as e:
        logger.warning("Whisper transcription failed for %s: %s", video_path.name, e)
        # Continue with frame-only analysis

    # Get video duration via ffprobe
    video_duration = 0.0
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(video_path)],
            capture_output=True, text=True, check=True,
        )
        video_duration = float(json.loads(result.stdout).get("format", {}).get("duration", 0))
    except Exception:
        pass

    # Build Claude message content with frames
    content = []
    for frame_path in frame_paths:
        try:
            img_data = base64.standard_b64encode(frame_path.read_bytes()).decode("utf-8")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": img_data},
            })
        except Exception:
            continue

    prompt = build_watch_prompt(
        narration_text=line["narration_text"],
        emotion=line.get("emotion", ""),
        visual_description=line.get("visual_description", ""),
        transcript_text=transcript_text,
        video_duration=video_duration,
    )
    content.append({"type": "text", "text": prompt})

    # Call Claude Haiku
    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=_WATCH_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
        return parse_claude_segments(response.content[0].text)
    except Exception as e:
        logger.warning("Claude Haiku call failed for %s: %s", video_path.name, e)
        return not_useful


def _load_manual_sources(topic_dir: Path) -> dict[int, list[dict]]:
    """Load manual_sources.json if present. Returns dict keyed by narration_line_ref."""
    manual_path = topic_dir / "manual_sources.json"
    if not manual_path.exists():
        return {}
    try:
        entries = json.loads(manual_path.read_text(encoding="utf-8"))
        result: dict[int, list[dict]] = {}
        for entry in entries:
            line_ref = entry.get("narration_line_ref")
            if line_ref is None:
                continue
            candidate = {
                "identifier": entry.get("identifier") or f"manual_{line_ref}",
                "url": entry["url"],
                "local_path": None,  # will be downloaded
                "title": entry.get("title", "Manual source"),
                "source": entry.get("source", "manual"),
                "source_type": entry.get("source", "manual"),
                "license": entry.get("license", "public_domain"),
            }
            result.setdefault(line_ref, []).append(candidate)
        logger.info("Editorial: loaded %d manual source entries from manual_sources.json",
                    sum(len(v) for v in result.values()))
        return result
    except Exception as e:
        logger.warning("Editorial: failed to load manual_sources.json: %s", e)
        return {}


def _download_manual_candidate(candidate: dict, clips_raw_dir: Path) -> Optional[dict]:
    """Download a manually specified clip URL into clips/raw/. Returns updated candidate or None."""
    import requests
    url = candidate["url"]
    identifier = candidate["identifier"]
    safe_name = re.sub(r"[^\w\-]", "_", identifier)[:72] + f"_{hash(identifier) & 0xFFFF:04x}"
    dest = clips_raw_dir / f"{safe_name}.mp4"
    if dest.exists() and dest.stat().st_size > 10_000:
        logger.info("Editorial: manual clip already cached — %s", safe_name)
    else:
        import subprocess as _sp
        try:
            result = _sp.run(
                ["yt-dlp", "-o", str(dest), "--no-playlist", url],
                capture_output=True, timeout=120,
            )
            if result.returncode != 0 or not dest.exists():
                raise RuntimeError("yt-dlp failed")
        except Exception:
            try:
                resp = requests.get(url, timeout=120, stream=True)
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                logger.warning("Editorial: failed to download manual clip %s: %s", url, e)
                return None
    if not dest.exists() or dest.stat().st_size < 10_000:
        return None
    return {**candidate, "local_path": f"clips/raw/{dest.name}"}


def _build_search_urls(visual_description: str) -> list[str]:
    """Generate suggested search URLs for a flagged line based on its visual description."""
    words = re.sub(r"[^\w\s]", "", visual_description).split()[:6]
    query = "+".join(words)
    return [
        f"https://archive.org/search?query={query}&and[]=mediatype%3A%22movies%22",
        f"https://archive.org/search?query=collection%3Aprelinger+{query}",
        f"https://commons.wikimedia.org/w/index.php?search={query}&ns6=1",
    ]


def run_editorial(topic_dir: Path, cache_dir: Optional[Path] = None) -> dict:
    """Run Editorial agent. Reads raw_candidates.json, writes clip_manifest.json.

    Also reads manual_sources.json (if present) — manually specified clip URLs are
    prepended to each line's candidate list and tried before automated candidates.
    """
    if cache_dir is None:
        _base = Path(__file__).parent.parent.parent
        cache_dir = _base / "data" / "clip_cache"
    cache = ClipCache(cache_dir)

    raw_path = topic_dir / "raw_candidates.json"
    raw_candidates = json.loads(raw_path.read_text(encoding="utf-8"))

    # Load manual overrides — keyed by narration_line_ref
    manual_by_line = _load_manual_sources(topic_dir)
    clips_raw_dir = topic_dir / "clips" / "raw"
    clips_raw_dir.mkdir(parents=True, exist_ok=True)

    # Download any manual candidates that haven't been fetched yet
    for line_ref, candidates in manual_by_line.items():
        resolved = []
        for c in candidates:
            if c.get("local_path") is None:
                downloaded = _download_manual_candidate(c, clips_raw_dir)
                if downloaded:
                    resolved.append(downloaded)
                    logger.info("Editorial: manual clip downloaded for line %d — %s",
                                line_ref, downloaded["local_path"])
            else:
                resolved.append(c)
        manual_by_line[line_ref] = resolved

    final_dir = topic_dir / "clips" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    frame_base = topic_dir / "clips" / "frames"
    frame_base.mkdir(parents=True, exist_ok=True)

    manifest = []
    sourced = 0
    gaps = 0

    for line_entry in raw_candidates:
        line_ref = line_entry["narration_line_ref"]
        line = {
            "narration_text": line_entry["narration_text"],
            "emotion": line_entry.get("emotion", ""),
            "visual_description": line_entry.get("visual_description", ""),
        }
        # Manual candidates go first — tried before any automated candidates
        manual_candidates = manual_by_line.get(line_ref, [])
        candidates = manual_candidates + line_entry.get("candidates", [])
        cuts = []
        winning_candidate = None

        for candidate in candidates:
            identifier = candidate.get("identifier", "unknown")
            local_path = topic_dir / candidate["local_path"]

            if not local_path.exists():
                logger.warning("Editorial: candidate file missing — %s", local_path)
                continue

            # Check cache first
            if cache.exists(identifier):
                cached = cache.load(identifier)
                logger.info("Editorial: cache hit for %s", identifier)
                analysis = {
                    "verdict": "useful" if cached.segments else "not_useful",
                    "usable_segments": [
                        {"start": s.start, "end": s.end,
                         "description": s.description, "relevance": "cached"}
                        for s in cached.segments
                    ],
                    "why": "from cache",
                }
            else:
                logger.info("Editorial: watching %s for line %d", identifier, line_ref)
                frame_dir = frame_base / re.sub(r"[^\w\-]", "_", identifier)[:40]
                analysis = watch_candidate(local_path, line, frame_dir)

                # Cache the result regardless of verdict
                cached_video = CachedVideo(
                    identifier=identifier,
                    source=candidate.get("source", "unknown"),
                    title=candidate.get("title", ""),
                    source_url=candidate.get("url", ""),
                    license=candidate.get("license", "public_domain"),
                    analyzed_at=date.today().isoformat(),
                    duration_seconds=0.0,
                    segments=[
                        CachedSegment(
                            start=s["start"], end=s["end"],
                            description=s["description"],
                            tags=[line.get("emotion", ""), "auto-tagged"],
                        )
                        for s in analysis.get("usable_segments", [])
                    ],
                )
                cache.save(cached_video)

            if analysis["verdict"] != "useful":
                logger.info("Editorial: %s — not useful for line %d (%s)",
                            identifier, line_ref, analysis.get("why", ""))
                continue

            # Trim each usable segment
            source_type = candidate.get("source", candidate.get("source_type", "unknown"))
            do_crop = _needs_timecode_crop(source_type)
            for i, seg in enumerate(analysis["usable_segments"]):
                cut_name = f"clip_{line_ref:03d}_{chr(97 + i)}.mp4"  # clip_001_a.mp4
                cut_path = final_dir / cut_name
                trimmed = trim_segment(local_path, cut_path, seg["start"], seg["end"],
                                       crop_timecode=do_crop)
                if trimmed:
                    cuts.append({
                        "local_path": f"clips/final/{cut_name}",
                        "duration_seconds": round(seg["end"] - seg["start"], 2),
                        "description": seg["description"],
                    })

            if cuts:
                winning_candidate = candidate
                logger.info("Editorial: line %d — %d cuts from %s", line_ref, len(cuts), identifier)
                break  # found usable cuts from this candidate, move to next line

        if cuts:
            sourced += 1
            manifest.append({
                "narration_line_ref": line_ref,
                "narration_text": line_entry["narration_text"],
                "cuts": cuts,
                "source_url": winning_candidate.get("url", "") if winning_candidate else "",
                "license": winning_candidate.get("license", "public_domain") if winning_candidate else "public_domain",
                "needs_generated_visual": False,
            })
        else:
            gaps += 1
            logger.info("Editorial: line %d — no usable footage found, flagging", line_ref)
            manifest.append({
                "narration_line_ref": line_ref,
                "narration_text": line_entry["narration_text"],
                "cuts": [],
                "source_url": None,
                "license": None,
                "needs_generated_visual": True,
                "visual_description": line_entry.get("visual_description", ""),
            })

    manifest_path = topic_dir / "clip_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total = len(raw_candidates)
    coverage = sourced / total if total else 0.0
    logger.info("Editorial: %d/%d lines sourced (%.0f%%), %d flagged",
                sourced, total, coverage * 100, gaps)

    # Write gap report for all flagged lines
    flagged_entries = [e for e in manifest if e.get("needs_generated_visual")]
    gap_report_path = topic_dir / "gap_report.md"
    if flagged_entries:
        report_lines = [
            "# Editorial Gap Report\n\n",
            f"**Coverage:** {sourced}/{total} lines ({coverage * 100:.0f}%)"
            f" — minimum required: {settings.min_clip_coverage * 100:.0f}%\n\n",
            "## How to resolve\n\n",
            "1. Find a clip for each `manual_source` line below\n",
            "2. Add it to `manual_sources.json` in this topic directory:\n\n",
            "```json\n",
            "[\n",
            '  {"narration_line_ref": N, "url": "https://...", "title": "...",\n',
            '   "source": "internet_archive", "license": "public_domain"}\n',
            "]\n",
            "```\n\n",
            "3. Re-run: `python run_pipeline.py --channel incentiveslab"
            f" --topic {topic_dir.name} --from editorial`\n\n",
            "---\n\n",
            "## Flagged Lines\n\n",
        ]
        for entry in flagged_entries:
            visual = entry.get("visual_description", "")
            action = _infer_action(visual)
            search_urls = _build_search_urls(visual) if action == "manual_source" else []
            report_lines.append(
                f"### Line {entry['narration_line_ref']} — `{action}`\n"
                f"**Narration:** {entry['narration_text']}\n"
                f"**Visual:** {visual or '(none)'}\n"
            )
            if search_urls:
                report_lines.append("**Suggested searches:**\n")
                for url in search_urls:
                    report_lines.append(f"- {url}\n")
            report_lines.append("\n")
        gap_report_path.write_text("".join(report_lines), encoding="utf-8")
        logger.info("Editorial: gap_report.md written (%d flagged lines)", len(flagged_entries))

    # Coverage gate — block Assembler if below threshold
    if coverage < settings.min_clip_coverage:
        raise ValueError(
            f"Editorial: coverage too low ({sourced}/{total} = {coverage * 100:.0f}%) — "
            f"minimum is {settings.min_clip_coverage * 100:.0f}%.\n"
            f"  1. Review: {gap_report_path}\n"
            f"  2. Add clips: {topic_dir / 'manual_sources.json'}\n"
            f"  3. Re-run: python run_pipeline.py --from editorial"
        )

    cuts_total = sum(len(e["cuts"]) for e in manifest)
    cache_hits = sum(1 for e in raw_candidates
                     for c in e.get("candidates", [])
                     if cache.exists(c.get("identifier", "")))
    return {
        "lines_resolved": sourced,
        "lines_flagged": gaps,
        "cuts_total": cuts_total,
        "cache_hits": cache_hits,
    }


_IA_SOURCES = {"internet_archive", "prelinger"}

_GENERATED_VISUAL_KEYWORDS = [
    "bar chart", "chart", "graph", "infographic", "diagram", "illustration",
    "animated", "text overlay", "data visualization", "title card", "black screen",
    "cta", "subscribe", "typewriter",
]


def _needs_timecode_crop(source_type: str) -> bool:
    """Return True if the clip source typically has a burned-in timecode overlay."""
    return source_type in _IA_SOURCES


def _infer_action(visual_description: str) -> str:
    """Classify a flagged line as manual_source or generated_visual based on visual tag."""
    desc = visual_description.lower()
    if any(kw in desc for kw in _GENERATED_VISUAL_KEYWORDS):
        return "generated_visual"
    return "manual_source"


def trim_segment(source: Path, dest: Path, start: float, end: float,
                 crop_timecode: bool = False) -> Optional[Path]:
    """Trim a segment from source video using ffmpeg. Returns dest path or None on failure.

    crop_timecode: if True, crops bottom 12% of frame (removes IA window-burn timecodes)
    and scales back to original dimensions.
    """
    duration = end - start
    vf = "crop=iw:ih*0.88:0:0,scale=iw:ih/0.88" if crop_timecode else None
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(source),
        "-t", str(duration),
    ]
    if vf:
        cmd += ["-vf", vf]
    cmd += [
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
