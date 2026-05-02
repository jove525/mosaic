# Mosaic

**6-agent AI video production pipeline.** Give it a topic — it outputs a publish-ready YouTube video.

Built with Claude API, ElevenLabs, ffmpeg, and Pexels. Used in production for [@PredictiveEchoes](https://youtube.com/@PredictiveEchoes) and [@IncentivesLab](https://youtube.com/@IncentivesLab).

---

## What It Does

Mosaic turns a topic slug into a finished YouTube video through a sequential agent pipeline:

```
Topic brief
    │
    ▼
[Researcher]      — Pulls source material, writes a structured brief (angle, key claims, sources)
    │
    ▼
[Scriptwriter]    — Writes the full narration script with SEO title, description, tags
    │
    ▼
[Curator]         — Sources b-roll clips (Pexels), generates voiceover (ElevenLabs), selects music
    │
    ▼
[Editorial]       — Resolves clip-to-line mapping, applies manual overrides, flags gaps
    │
    ▼
[Assembler]       — Renders final video via ffmpeg (clips + narration + music, sync'd to script)
    │
    ▼
[Publisher]       — Uploads to YouTube with metadata; writes review_notes.md for human gate
    │
    ▼
Published video
```

Each agent reads from and writes to a structured `output/<channel>/<topic>/` directory. You can resume from any agent using `--from` if a step fails or needs to be re-run.

---

## Knowledge Base (KB) System

Before running the pipeline, Mosaic builds a **channel-specific editorial knowledge base** from reference YouTube channels.

The KB builder (`build_kb.py`):
1. Downloads videos from curated reference channels (e.g. Johnny Harris, Kurzgesagt, Wendover)
2. Extracts frames at configurable intervals
3. Transcribes audio via Whisper
4. Analyzes each video with Claude — extracts hook structure, pacing patterns, b-roll style, retention techniques
5. Stores insights in a local KB (`data/kb/`)

The pipeline then uses this KB to calibrate script structure, hook writing, and visual pacing to match the editorial style of top-performing channels in your niche.

Reference channels are configured in `mosaic/config/channels.py`. Each channel profile defines focus areas (e.g. "curiosity-gap hooks, surprise reveals" for Veritasium).

---

## Channel Profiles

Mosaic is channel-aware. Each production channel has a profile defining its domain, tone, arc template, visual style, and editorial north star:

```python
"incentiveslab": {
    "domain": "economics, human behavior, incentive structures",
    "tone": "curious, slightly provocative, Munger-esque",
    "arc_template": "surface_reality → hidden_incentive → reframe",
    "north_star": "make the viewer distrust something they trusted before",
    "reference_channels": ["johnny_harris", "kurzgesagt"],
}
```

Add a new channel profile to `mosaic/config/channels.py` → `PRODUCTION_CHANNELS` to target a new niche.

---

## Stack

| Layer | Tool |
|---|---|
| Research & scripting | Claude API (Sonnet) |
| Voiceover | ElevenLabs |
| Video assembly | ffmpeg |
| B-roll footage | Pexels API |
| Transcription (KB) | Whisper (local) |
| Frame extraction (KB) | ffmpeg |
| KB analysis | Claude API (Sonnet) |

---

## Setup

```bash
# 1. Clone and install
git clone https://github.com/jove525/mosaic.git
cd mosaic
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Fill in: ANTHROPIC_API_KEY, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, PEXELS_API_KEY
# Optional: YOUTUBE_API_KEY (for publisher agent)

# 3. Build the editorial KB (optional but recommended)
python build_kb.py --channel all --limit 10

# 4. Run the pipeline
python run_pipeline.py --channel incentiveslab --topic war-financing
```

### Resume from a specific agent

```bash
# Re-run just the assembler (e.g. after editing clip_manifest.json)
python run_pipeline.py --channel incentiveslab --topic war-financing --from assembler

# Run only research + scripting, stop before clip sourcing
python run_pipeline.py --channel incentiveslab --topic war-financing --to scriptwriter
```

### Delta analysis (post-publish feedback loop)

After watching a published video, write one line of feedback to `output/<channel>/<topic>/your_feedback.md`, then run:

```bash
python run_pipeline.py --channel incentiveslab --topic war-financing --delta
```

Mosaic compares the published video against the script and your feedback to generate `delta_analysis.md` — a structured breakdown of what worked, what didn't, and what to adjust next time.

---

## Output Structure

```
output/
└── incentiveslab/
    └── war-financing/
        ├── brief.md              # Researcher output
        ├── script.md             # Full narration script
        ├── raw_candidates.json   # Curator clip candidates
        ├── clip_manifest.json    # Editorial-resolved clip plan
        ├── narration.mp3         # ElevenLabs voiceover
        ├── final_draft.mp4       # Assembled video
        ├── review_notes.md       # Publisher gate — human reviews before upload
        ├── pipeline.log          # Per-run log
        └── your_feedback.md      # (you write this) post-publish feedback
```

---

## Status

**Production use:** Running. Used for PredictiveEchoes (geopolitics) and IncentivesLab (behavioral economics).

**Clip sourcing note:** Pexels works well for niches that don't require historically or politically specific footage. For finance/behavioral economics content, generic stock footage can feel mismatched — this is an active area of improvement (exploring AI-generated clips via Kling API as an alternative source).

**Tests:** `pytest tests/` — covers all pipeline agents and KB modules.

---

## Project Context

Mosaic is the reusable production layer. Channel-specific scheduling, topic ideation, and Telegram approval gates live in the channel's own orchestrator (e.g. the PredictiveEchoes orchestrator at [jove525/predictive-echoes](https://github.com/jove525/predictive-echoes)).

---

Built by [Joven Baring](https://jovenbaring.com) — operations systems + AI automation.
