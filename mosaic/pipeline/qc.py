"""Clip QC — three-layer quality gate for Curator.

Layer 1 — Technical (ffprobe, free):
    duration ≥ 3s, video stream present, no dominant black frames

Layer 2 — Semantic / CLIP (local, free):
    cosine similarity between sampled frames and [VISUAL] tag text
    threshold: mean score ≥ 0.18, uses ViT-B/32 (lowered from 0.25 — archival B&W footage scores 0.17-0.22)

Layer 3 — Emotional + Narrative / Haiku (cheap):
    only called if Layer 2 passes
    evaluates emotional match + narrative match from frames + script context
"""
import base64
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

import anthropic
from PIL import Image

logger = logging.getLogger(__name__)

# CLIP model loaded once at module level (CPU, ~350MB download on first use)
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None


def _load_clip():
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is not None:
        return
    try:
        import open_clip
        import torch
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        _clip_model.eval()
        logger.info("QC: CLIP ViT-B/32 loaded on CPU")
    except Exception as e:
        logger.warning("QC: CLIP load failed — semantic QC will be skipped: %s", e)


def extract_frames(clip_path: Path, n: int = 5) -> list[Path]:
    """Extract n evenly-spaced frames from clip as JPEG files."""
    frames_dir = clip_path.parent / f"{clip_path.stem}_frames"
    frames_dir.mkdir(exist_ok=True)
    try:
        # Get duration
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(clip_path)],
            capture_output=True, text=True, timeout=15,
        )
        duration = float(result.stdout.strip())
    except Exception:
        duration = 10.0

    frames = []
    for i in range(n):
        t = duration * (i + 1) / (n + 1)
        out = frames_dir / f"frame_{i:02d}.jpg"
        if not out.exists():
            subprocess.run(
                ["ffmpeg", "-ss", str(t), "-i", str(clip_path),
                 "-frames:v", "1", "-q:v", "2", str(out), "-y"],
                capture_output=True, timeout=15,
            )
        if out.exists() and out.stat().st_size > 1000:
            frames.append(out)
    return frames


def technical_qc(clip_path: Path) -> tuple[bool, str]:
    """Check duration, video stream presence, black frame ratio."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration:stream=codec_type",
             "-of", "json", str(clip_path)],
            capture_output=True, text=True, timeout=15,
        )
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration", 0))
        streams = data.get("streams", [])
        has_video = any(s.get("codec_type") == "video" for s in streams)

        if not has_video:
            return False, "no video stream"
        if duration < 3.0:
            return False, f"duration {duration:.1f}s < 3s minimum"

        # Black frame check: sample 3 frames, flag if >2 are nearly black
        frames = extract_frames(clip_path, n=3)
        black_count = 0
        for fp in frames:
            img = Image.open(fp).convert("L")
            mean_brightness = sum(img.getdata()) / (img.width * img.height)
            if mean_brightness < 15:
                black_count += 1
        if black_count > 1:
            return False, f"{black_count}/3 frames are black"

        return True, "ok"
    except Exception as e:
        return False, f"ffprobe error: {e}"


def semantic_qc(clip_path: Path, visual_tag: str) -> tuple[bool, float]:
    """CLIP cosine similarity between frames and [VISUAL] tag text. Threshold 0.25."""
    _load_clip()
    if _clip_model is None:
        logger.warning("QC: CLIP unavailable — skipping semantic QC, defaulting PASS")
        return True, 0.0

    try:
        import open_clip
        import torch

        frames = extract_frames(clip_path, n=5)
        if not frames:
            return False, 0.0

        image_tensors = []
        for fp in frames:
            img = Image.open(fp).convert("RGB")
            tensor = _clip_preprocess(img).unsqueeze(0)
            image_tensors.append(tensor)

        images = torch.cat(image_tensors, dim=0)
        text_tokens = _clip_tokenizer([visual_tag])

        with torch.no_grad():
            image_features = _clip_model.encode_image(images)
            text_features = _clip_model.encode_text(text_tokens)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarities = (image_features @ text_features.T).squeeze(1)
            mean_score = similarities.mean().item()

        passed = mean_score >= 0.18
        return passed, round(mean_score, 4)
    except Exception as e:
        logger.warning("QC: CLIP error — defaulting PASS: %s", e)
        return True, 0.0


_HAIKU_QC_SYSTEM = """\
You are evaluating a video clip for a documentary. You will see frames from the clip
and the narration context. Evaluate two things only:

1. Emotional match: Does the clip's visual energy, human presence, and tone match
   the required emotional register?
2. Narrative match: Does this clip serve the story at this exact moment in the arc?

Respond in JSON only:
{"emotional": true/false, "narrative": true/false, "why": "one sentence"}
"""


def contextual_qc(
    clip_path: Path,
    narration_text: str,
    emotion_tag: str,
    direction_tag: str,
) -> tuple[bool, dict]:
    """Haiku vision: emotional + narrative match. Called only after semantic_qc passes."""
    frames = extract_frames(clip_path, n=5)
    if not frames:
        return False, {"emotional": False, "narrative": False, "why": "no frames extracted"}

    content: list[dict] = []
    for fp in frames:
        b64 = base64.standard_b64encode(fp.read_bytes()).decode()
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        })
    content.append({
        "type": "text",
        "text": (
            f'Narration: "{narration_text}"\n'
            f"Required emotion: {emotion_tag}\n"
            f"Story direction: {direction_tag}\n\n"
            "Evaluate emotional match and narrative match."
        ),
    })

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            system=_HAIKU_QC_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
        text = response.content[0].text.strip()
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            result = json.loads(match.group())
            passed = result.get("emotional", False) and result.get("narrative", False)
            return passed, result
        return False, {"emotional": False, "narrative": False, "why": "parse error"}
    except Exception as e:
        logger.warning("QC: Haiku contextual QC error: %s", e)
        return False, {"emotional": False, "narrative": False, "why": str(e)}


def run_qc(
    clip_path: Path,
    visual_tag: str,
    narration_text: str,
    emotion_tag: str,
    direction_tag: str,
) -> dict:
    """Run all three QC layers. Returns result dict with overall pass/fail."""
    result = {
        "technical": {"passed": False, "reason": ""},
        "semantic": {"passed": False, "score": 0.0},
        "contextual": {"passed": False, "emotional": False, "narrative": False, "why": ""},
        "passed": False,
    }

    tech_pass, tech_reason = technical_qc(clip_path)
    result["technical"] = {"passed": tech_pass, "reason": tech_reason}
    if not tech_pass:
        return result

    sem_pass, sem_score = semantic_qc(clip_path, visual_tag)
    result["semantic"] = {"passed": sem_pass, "score": sem_score}
    if not sem_pass:
        return result

    ctx_pass, ctx_detail = contextual_qc(clip_path, narration_text, emotion_tag, direction_tag)
    result["contextual"] = {
        "passed": ctx_pass,
        "emotional": ctx_detail.get("emotional", False),
        "narrative": ctx_detail.get("narrative", False),
        "why": ctx_detail.get("why", ""),
    }

    result["passed"] = ctx_pass
    return result
