from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

_BASE = Path(__file__).parent.parent.parent  # D:/Mosaic


class Settings:
    def __init__(self):
        self.anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
        self.elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY", "")
        self.elevenlabs_voice_id: str = os.getenv("ELEVENLABS_VOICE_ID", "")
        self.whisper_model: str = os.getenv("WHISPER_MODEL", "base")
        self.frame_interval_seconds: int = int(os.getenv("FRAME_INTERVAL_SECONDS", "8"))
        self.min_clip_coverage: float = float(os.getenv("MIN_CLIP_COVERAGE", "0.70"))

        # Clip sourcing
        self.kling_api_key: str = os.getenv("KLING_API_KEY", "")
        self.kling_api_secret: str = os.getenv("KLING_API_SECRET", "")
        self.youtube_api_key: str = os.getenv("YOUTUBE_API_KEY", "")
        self.pexels_api_key: str = os.getenv("PEXELS_API_KEY", "")

        # Curator constraints
        self.ia_max_duration_seconds: int = int(os.getenv("IA_MAX_DURATION_SECONDS", "300"))  # 5 min
        self.ai_generated_cap_pct: float = float(os.getenv("AI_GENERATED_CAP_PCT", "0.20"))
        self.curator_max_candidates_per_source: int = 5
        self.curator_max_query_diversifications: int = 2
        self.curator_max_retries_before_ai: int = 3

        self.data_dir: Path = _BASE / "data"
        self.kb_dir: Path = self.data_dir / "kb"
        self.clips_dir: Path = self.data_dir / "clips"
        self.frames_dir: Path = self.data_dir / "frames"
        self.seed_dir: Path = _BASE / "seed"
        self.output_dir: Path = _BASE / "output"
        self.taste_dir: Path = self.data_dir / "taste"


settings = Settings()
