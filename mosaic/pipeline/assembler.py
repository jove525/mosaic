"""Agent 4 — Assembler.

Transcribes narration.mp3 with Whisper, places clips at audio timestamps,
applies ASS subtitles, runs self-eval, enforces duration gate, renders final_draft.mp4.
Audio is master timeline — all visual sync derives from Whisper word timestamps.
"""
import json
import logging
import subprocess
from enum import Enum
from pathlib import Path

import anthropic
import whisper

from mosaic.config.settings import settings
from mosaic.pipeline.taste_store import TasteStore

logger = logging.getLogger(__name__)

_SELF_EVAL_SYSTEM = """\
You are evaluating a documentary video script + assembly plan for IncentivesLab.
Check each criterion and respond PASS or FAIL with a timestamp or reason.

Format each line as:
N. [criterion]: PASS/FAIL — [brief explanation and timestamp if applicable]
"""

_EXPANSION_BRIEF = """\
The current script runs under 8 minutes. Expand the Tension section by adding one of:
- A concrete case study that deepens the hidden incentive mechanism
- A historical parallel that shows the pattern repeating
- A data-driven breakdown of who benefits from the hidden incentive

Do NOT add new sections. Expand within the existing Tension section only.
Target: bring total duration above 10 minutes.
"""


class DurationGateResult(Enum):
    FAIL = "FAIL"
    WARN = "WARN"
    PASS = "PASS"


def check_duration_gate(duration_seconds: float) -> DurationGateResult:
    if duration_seconds < 480:  # 8 min
        return DurationGateResult.FAIL
    if duration_seconds < 600:  # 10 min
        return DurationGateResult.WARN
    return DurationGateResult.PASS


def build_self_eval_checks() -> list[dict]:
    return [
        {"question": "Does the hook create an unanswered question in the first 15 seconds?"},
        {"question": "Is every visual earning its screen time (serves story momentum)?"},
        {"question": "Are there at least 2 moments designed to create an unexpected feeling?"},
        {"question": "Does the ending resolve the tension the hook created?"},
    ]


def parse_eval_result(eval_text: str) -> dict:
    """Parse self-eval Claude response into pass/fail counts."""
    lines = eval_text.strip().split("\n")
    passed = 0
    failed = 0
    for line in lines:
        upper = line.upper()
        if ": PASS" in upper:
            passed += 1
        elif ": FAIL" in upper:
            failed += 1
    return {
        "passed": passed,
        "failed": failed,
        "total": passed + failed,
        "score": f"{passed}/{passed + failed}",
        "raw": eval_text,
    }


def build_ass_header() -> str:
    """Return ASS subtitle file header with IncentivesLab style."""
    return """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
Collisions: Normal

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: IncentivesLab,Arial,52,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,20,20,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _seconds_to_ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _transcribe_narration(narration_path: Path) -> dict:
    """Transcribe narration.mp3 with Whisper. Returns full result with word-level timestamps."""
    model = whisper.load_model(settings.whisper_model)
    result = model.transcribe(str(narration_path), word_timestamps=True)
    return result


def _build_ass_events(whisper_result: dict) -> str:
    """Build ASS subtitle events from Whisper word timestamps."""
    lines = []
    for segment in whisper_result.get("segments", []):
        words = segment.get("words", [])
        if not words:
            start = _seconds_to_ass_time(segment["start"])
            end = _seconds_to_ass_time(segment["end"])
            text = segment["text"].strip()
            lines.append(f"Dialogue: 0,{start},{end},IncentivesLab,,0,0,0,,{text}")
            continue
        group: list = []
        for word in words:
            group.append(word)
            if len(group) >= 8 or word == words[-1]:
                start = _seconds_to_ass_time(group[0]["start"])
                end = _seconds_to_ass_time(group[-1]["end"])
                text = " ".join(w["word"].strip() for w in group)
                lines.append(f"Dialogue: 0,{start},{end},IncentivesLab,,0,0,0,,{text}")
                group = []
    return "\n".join(lines)


def _run_self_eval(script_text: str, manifest: list[dict]) -> dict:
    """Ask Claude to self-evaluate the video plan against channel north star checks."""
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    checks = build_self_eval_checks()
    checks_text = "\n".join(f"{i+1}. {c['question']}" for i, c in enumerate(checks))
    gaps = [m for m in manifest if m.get("license_tier") == "GAP"]
    gap_note = f"\n\nNote: {len(gaps)} narration lines have no clip (black frame placeholder)." if gaps else ""

    user = (
        f"Script excerpt (first 2000 chars):\n\n{script_text[:2000]}\n\n"
        f"Clip manifest: {len(manifest)} clips, {len(gaps)} gaps.{gap_note}\n\n"
        f"Evaluate each check:\n{checks_text}"
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=_SELF_EVAL_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    return parse_eval_result(response.content[0].text)


def _render_video(topic_dir: Path, narration_path: Path, music_path: Path,
                  clip_manifest: list[dict], ass_path: Path,
                  whisper_result: dict) -> Path:
    """Render final_draft.mp4 via ffmpeg. Audio is master timeline."""
    output_path = topic_dir / "final_draft.mp4"
    clips_dir = topic_dir / "clips"

    segments = whisper_result.get("segments", [])
    if segments:
        total_duration = segments[-1].get("end", 0.0)
        if total_duration == 0.0:
            raise RuntimeError(
                "Whisper returned no segment end time — cannot determine narration duration"
            )
    else:
        total_duration = 60.0

    valid_clips = [
        m for m in clip_manifest
        if m.get("local_path") and (topic_dir / m["local_path"]).exists()
    ]

    if not valid_clips:
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=black:size=1920x1080:duration={total_duration}",
            "-i", str(narration_path),
            "-vf", f"ass={ass_path}",
            "-c:v", "libx264", "-c:a", "aac", "-shortest",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            logger.error("ffmpeg failed (no-clips branch): %s", stderr)
            raise
        return output_path

    concat_list_path = topic_dir / "concat_list.txt"
    with open(concat_list_path, "w") as f:
        for entry in clip_manifest:
            local = entry.get("local_path")
            if local and (topic_dir / local).exists():
                f.write(f"file '{(topic_dir / local).absolute()}'\n")
            else:
                black_path = topic_dir / "black_3s.mp4"
                if not black_path.exists():
                    try:
                        subprocess.run([
                            "ffmpeg", "-y", "-f", "lavfi",
                            "-i", "color=c=black:size=1920x1080:duration=3",
                            "-c:v", "libx264", str(black_path),
                        ], check=True, capture_output=True)
                    except subprocess.CalledProcessError as e:
                        stderr = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
                        logger.error("ffmpeg failed (black frame generation): %s", stderr)
                        raise
                f.write(f"file '{black_path.absolute()}'\n")

    visual_path = topic_dir / "visual_track.mp4"
    try:
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list_path),
            "-c:v", "libx264", "-an", str(visual_path),
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        logger.error("ffmpeg failed (concat visual track): %s", stderr)
        raise

    if music_path.exists() and music_path.stat().st_size > 0:
        audio_filter = (
            "[0:a]volume=1.0[narr];"
            "[1:a]volume=0.126[music];"
            "[narr][music]amix=inputs=2:duration=first[aout]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(visual_path),
            "-i", str(narration_path),
            "-i", str(music_path),
            "-filter_complex", audio_filter,
            "-map", "0:v", "-map", "[aout]",
            "-vf", f"ass={ass_path}",
            "-c:v", "libx264", "-c:a", "aac", "-shortest",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(visual_path),
            "-i", str(narration_path),
            "-map", "0:v", "-map", "1:a",
            "-vf", f"ass={ass_path}",
            "-c:v", "libx264", "-c:a", "aac", "-shortest",
            str(output_path),
        ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        logger.error("ffmpeg failed (final render): %s", stderr)
        raise
    return output_path


def _get_duration(video_path: Path) -> float:
    """Get video duration in seconds via ffprobe."""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", str(video_path),
        ], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error("ffprobe failed: %s", e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else e.stderr)
        raise RuntimeError(f"ffprobe failed on {video_path}: {e.stderr}") from e
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ffprobe returned invalid JSON for {video_path}") from e
    raw = data.get("format", {}).get("duration")
    if raw is None:
        raise RuntimeError(f"ffprobe output missing 'duration' key for {video_path}")
    return float(raw)


def run_assembler(
    topic_dir: Path,
    channel_profile: dict,
    rerun_count: int = 0,
) -> dict:
    """Run Assembler agent. Reads clip_manifest.json + narration.mp3, renders final_draft.mp4."""
    taste_store = TasteStore(settings.taste_dir)
    taste_learnings = taste_store.query("assembler edit transition pacing", n_results=3)
    if taste_learnings:
        logger.info("Assembler: applying %d taste learnings", len(taste_learnings))

    narration_path = topic_dir / "narration.mp3"
    music_path = topic_dir / "music.mp3"
    script_path = topic_dir / "script.md"
    manifest_path = topic_dir / "clip_manifest.json"

    script_text = script_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    logger.info("Assembler: transcribing narration with Whisper")
    whisper_result = _transcribe_narration(narration_path)

    ass_content = build_ass_header() + _build_ass_events(whisper_result)
    ass_path = topic_dir / "subtitles.ass"
    ass_path.write_text(ass_content, encoding="utf-8")

    logger.info("Assembler: running self-evaluation")
    eval_result = _run_self_eval(script_text, manifest)
    gaps = [m for m in manifest if m.get("license_tier") == "GAP"]

    logger.info("Assembler: rendering final_draft.mp4")
    output_path = _render_video(
        topic_dir, narration_path, music_path, manifest, ass_path, whisper_result
    )

    duration_sec = _get_duration(output_path)
    size_mb = round(output_path.stat().st_size / (1024 * 1024), 1)
    minutes = int(duration_sec // 60)
    seconds = int(duration_sec % 60)
    duration_str = f"{minutes}:{seconds:02d}"

    gate = check_duration_gate(duration_sec)
    review_lines = [
        f"# Assembler Review Notes",
        f"",
        f"## Self-Evaluation",
        f"Score: {eval_result['score']}",
        f"",
        eval_result["raw"],
        f"",
        f"## Duration Gate",
        f"Duration: {duration_str} ({gate.value})",
        f"",
    ]
    if gaps:
        review_lines += [
            f"## Clip Gaps ({len(gaps)} total)",
            "Narration lines with no SAFE clip found (3s black frame inserted):",
        ]
        for g in gaps:
            review_lines.append(f"- Line {g['narration_line_ref']}: {g['narration_text'][:80]}")
        review_lines.append("")

    if gate == DurationGateResult.FAIL:
        if rerun_count >= 2:
            review_lines.append(
                "⚠ Duration gate FAIL — Scriptwriter rerun cap reached (2 attempts). "
                "Manual intervention required."
            )
            logger.warning("Assembler: duration gate FAIL, rerun cap reached")
        else:
            review_lines.append(
                f"⚠ Duration gate FAIL — triggering Scriptwriter rerun "
                f"(attempt {rerun_count + 1}/2)"
            )
            logger.warning("Assembler: duration gate FAIL — re-running Scriptwriter")
            audit_path = topic_dir / f"review_notes_attempt_{rerun_count}.md"
            audit_path.write_text("\n".join(review_lines), encoding="utf-8")
            logger.info("Assembler: audit trail written to %s", audit_path)
            (topic_dir / "expansion_brief.md").write_text(_EXPANSION_BRIEF, encoding="utf-8")
            from mosaic.pipeline.scriptwriter import run_scriptwriter
            from mosaic.pipeline.curator import run_curator
            run_scriptwriter(topic_dir, channel_profile)
            run_curator(topic_dir, channel_profile)
            return run_assembler(topic_dir, channel_profile, rerun_count=rerun_count + 1)

    elif gate == DurationGateResult.WARN:
        review_lines.append(f"⚠ Duration WARN — {duration_str} is between 8-10 min. Review if acceptable.")

    (topic_dir / "review_notes.md").write_text("\n".join(review_lines), encoding="utf-8")

    return {
        "eval_result": gate.value,
        "eval_score": eval_result["score"],
        "duration": duration_str,
        "size_mb": size_mb,
    }
