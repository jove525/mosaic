import anthropic
import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from mosaic.kb.downloader import VideoMeta
from mosaic.kb.extractor import FrameSet, TranscriptSegment


@dataclass
class VideoInsights:
    video_id: str
    channel: str
    title: str
    hook: dict
    story_arc: dict
    emotional_beats: list
    visual_patterns: list
    pacing: dict
    narration_style: dict
    key_insights: list[str] = field(default_factory=list)


def _encode_frame(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("utf-8")


def _build_prompt(
    meta: VideoMeta,
    transcript: list[TranscriptSegment],
    channel_focus: str,
) -> str:
    transcript_text = "\n".join(
        f"[{seg.start:.1f}s–{seg.end:.1f}s] {seg.text}"
        for seg in transcript
    )
    return f"""You are an expert documentary film analyst and editorial director.

Analyze this video from the channel "{meta.channel}" (focus: {channel_focus}).
Title: "{meta.title}" | Duration: {meta.duration_seconds}s

TRANSCRIPT (timestamped):
{transcript_text}

FRAMES: Attached above, one every ~8 seconds.

Your task: extract INTENT-RICH editorial insights — not just what happens, but WHY it works and what it produces in the viewer emotionally.

Return ONLY valid JSON with this exact structure:
{{
  "hook": {{
    "structure": "name the hook type",
    "technique": "describe exactly what is done in the first 30 seconds",
    "why_it_works": "explain the psychological mechanism at work",
    "duration_seconds": 0
  }},
  "story_arc": {{
    "shape": "name the arc pattern",
    "phases": [
      {{"name": "phase name", "start": 0, "end": 0, "purpose": "why this phase exists in the arc"}}
    ]
  }},
  "emotional_beats": [
    {{"timestamp": 0, "emotion": "TENSION|REVELATION|MOMENTUM|GRIEF|DEFIANCE|IRONY", "trigger": "what causes it", "why": "what it produces in the viewer"}}
  ],
  "visual_patterns": [
    {{"narrative_moment": "describe the story moment", "shot_type": "describe the shot", "why": "why this visual serves this moment"}}
  ],
  "pacing": {{
    "avg_cut_interval_seconds": 0.0,
    "music_shift_points": [],
    "silence_moments": []
  }},
  "narration_style": {{
    "avg_sentence_length_words": 0,
    "rhythm": "describe the sentence rhythm pattern",
    "notable_techniques": ["list specific rhetorical techniques used"]
  }},
  "key_insights": [
    "3-5 specific, actionable insights about what makes this video work — insights another creator could actually apply"
  ]
}}"""


def analyze_video(
    meta: VideoMeta,
    frameset: FrameSet,
    transcript: list[TranscriptSegment],
    api_key: str,
    channel_focus: str = "",
) -> VideoInsights:
    """Send frames + transcript to Claude. Return structured VideoInsights."""
    client = anthropic.Anthropic(api_key=api_key)

    # Build image content blocks (max 20 frames for cost control)
    frames_to_send = frameset.frame_paths[::max(1, len(frameset.frame_paths) // 20)][:20]
    image_blocks = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": _encode_frame(f),
            },
        }
        for f in frames_to_send
    ]

    text_block = {"type": "text", "text": _build_prompt(meta, transcript, channel_focus)}
    content = image_blocks + [text_block]

    data = None
    for attempt in range(2):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8096,
            messages=[{"role": "user", "content": content}],
        )
        raw = response.content[0].text
        # Strip markdown code fences if present
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        # Find JSON object boundaries
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]
        try:
            data = json.loads(raw.strip())
            break
        except json.JSONDecodeError:
            if attempt == 1:
                raise


    return VideoInsights(
        video_id=meta.video_id,
        channel=meta.channel,
        title=meta.title,
        hook=data.get("hook", {}),
        story_arc=data.get("story_arc", {}),
        emotional_beats=data.get("emotional_beats", []),
        visual_patterns=data.get("visual_patterns", []),
        pacing=data.get("pacing", {}),
        narration_style=data.get("narration_style", {}),
        key_insights=data.get("key_insights", []),
    )
