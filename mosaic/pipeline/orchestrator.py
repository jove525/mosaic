import logging
from datetime import datetime
from pathlib import Path

from mosaic.config.channels import PRODUCTION_CHANNELS

logger = logging.getLogger(__name__)

AGENT_ORDER = ["researcher", "scriptwriter", "curator", "editorial", "assembler", "publisher"]

# Required input files each agent needs before it can run
AGENT_REQUIRED_INPUTS: dict[str, list[str]] = {
    "researcher": [],
    "scriptwriter": ["brief.md"],
    "curator": ["script.md"],
    "editorial": ["raw_candidates.json"],
    "assembler": ["clip_manifest.json", "narration.mp3"],
    "publisher": ["final_draft.mp4", "review_notes.md"],
}


class PipelineError(Exception):
    pass


class Orchestrator:
    def __init__(self, channel: str, topic_slug: str, output_root: Path):
        if channel not in PRODUCTION_CHANNELS:
            raise PipelineError(f"Unknown channel: {channel}. Add it to PRODUCTION_CHANNELS.")
        self.channel = channel
        self.topic_slug = topic_slug
        self.topic_dir = output_root / channel / topic_slug
        self.log_path = self.topic_dir / "pipeline.log"
        self._channel_profile = PRODUCTION_CHANNELS[channel]

    def _ensure_dirs(self):
        self.topic_dir.mkdir(parents=True, exist_ok=True)
        (self.topic_dir / "clips").mkdir(exist_ok=True)

    def _log(self, agent: str, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {agent:<12} → {message}"
        logger.info(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _check_required_inputs(self, from_agent: str):
        required = AGENT_REQUIRED_INPUTS.get(from_agent, [])
        for fname in required:
            fpath = self.topic_dir / fname
            if not fpath.exists():
                raise PipelineError(
                    f"Cannot resume from '{from_agent}': required file '{fname}' not found "
                    f"at {fpath}. Run the pipeline from an earlier agent first."
                )

    def _run_researcher(self):
        from mosaic.pipeline.researcher import run_researcher
        result = run_researcher(self.topic_dir, self.topic_slug, self._channel_profile)
        self._log("RESEARCHER", f"brief.md written (angle: \"{result.get('angle', '?')}\")")

    def _run_scriptwriter(self):
        from mosaic.pipeline.scriptwriter import run_scriptwriter
        result = run_scriptwriter(self.topic_dir, self._channel_profile)
        self._log("SCRIPTWRITER", f"script.md written ({result.get('line_count', '?')} lines, ~{result.get('est_minutes', '?')}min estimated)")

    def _run_editorial(self):
        from mosaic.pipeline.editorial import run_editorial
        result = run_editorial(self.topic_dir)
        self._log("EDITORIAL", f"clip_manifest.json written ({result.get('lines_resolved', 0)} resolved, {result.get('lines_flagged', 0)} flagged, {result.get('cuts_total', 0)} cuts, {result.get('cache_hits', 0)} cache hits)")

    def _run_curator(self):
        from mosaic.pipeline.curator import run_curator
        result = run_curator(self.topic_dir, self._channel_profile)
        self._log("CURATOR", f"{result.get('clips_sourced', '?')}/{result.get('clips_total', '?')} clips sourced ({result.get('gaps', 0)} flagged gaps)")
        self._log("CURATOR", f"narration.mp3 generated ({result.get('narration_duration', '?')} duration)")
        self._log("CURATOR", f"music.mp3 selected ({result.get('music_track', '?')})")

    def _run_assembler(self):
        from mosaic.pipeline.assembler import run_assembler
        result = run_assembler(self.topic_dir, self._channel_profile)
        self._log("ASSEMBLER", f"self-eval {result.get('eval_result', '?')} ({result.get('eval_score', '?')} checks)")
        self._log("ASSEMBLER", f"final_draft.mp4 rendered ({result.get('duration', '?')} duration, {result.get('size_mb', '?')}MB)")

    def _run_publisher(self):
        from mosaic.pipeline.publisher import run_publisher
        result = run_publisher(self.topic_dir, self.topic_slug, self._channel_profile)
        self._log("PUBLISHER", f"youtube_metadata.json written — status: {result.get('status', '?')}")

    def run(self, from_agent: str | None = None, to_agent: str | None = None):
        self._ensure_dirs()
        start_idx = 0
        if from_agent:
            if from_agent not in AGENT_ORDER:
                raise PipelineError(f"Unknown --from value: '{from_agent}'. Valid: {AGENT_ORDER}")
            self._check_required_inputs(from_agent)
            start_idx = AGENT_ORDER.index(from_agent)

        end_idx = len(AGENT_ORDER)
        if to_agent:
            if to_agent not in AGENT_ORDER:
                raise PipelineError(f"Unknown --to value: '{to_agent}'. Valid: {AGENT_ORDER}")
            end_idx = AGENT_ORDER.index(to_agent) + 1

        agents_to_run = AGENT_ORDER[start_idx:end_idx]
        dispatch = {
            "researcher": self._run_researcher,
            "scriptwriter": self._run_scriptwriter,
            "curator": self._run_curator,
            "editorial": self._run_editorial,
            "assembler": self._run_assembler,
            "publisher": self._run_publisher,
        }
        for agent_name in agents_to_run:
            try:
                dispatch[agent_name]()
            except Exception as e:
                self._log(agent_name.upper(), f"FAILED: {e}")
                raise PipelineError(
                    f"Agent '{agent_name}' failed: {e}\n"
                    f"Resume with: python run_pipeline.py --channel {self.channel} "
                    f"--topic {self.topic_slug} --from {agent_name}"
                ) from e

        self._log("─" * 45, "")
        review_path = self.topic_dir / "final_draft.mp4"
        self._log("Review", str(review_path))
        self._log("Prompt", "Did this video make you feel something you didn't expect to feel?")
        print(f"\n{'─'*50}")
        print(f"Your video is ready:")
        print(f"  {review_path}")
        print()
        print("Watch it first. Write one line in your_feedback.md before opening anything else.")
        print()
        print("Did this video make you feel something you didn't expect to feel?")
        print(f"{'─'*50}\n")
