from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

_BASE = Path(__file__).parent.parent.parent  # D:/Mosaic


class Settings:
    def __init__(self):
        self.anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
        self.whisper_model: str = os.getenv("WHISPER_MODEL", "base")
        self.frame_interval_seconds: int = int(os.getenv("FRAME_INTERVAL_SECONDS", "8"))

        self.data_dir: Path = _BASE / "data"
        self.kb_dir: Path = self.data_dir / "kb"
        self.clips_dir: Path = self.data_dir / "clips"
        self.frames_dir: Path = self.data_dir / "frames"
        self.seed_dir: Path = _BASE / "seed"


settings = Settings()
