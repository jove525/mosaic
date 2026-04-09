import logging
from pathlib import Path
from mosaic.kb.downloader import download_video
from mosaic.kb.extractor import extract_frames, transcribe_video
from mosaic.kb.analyzer import analyze_video
from mosaic.kb.store import KBStore
from mosaic.config.channels import REFERENCE_CHANNELS

logger = logging.getLogger(__name__)


class KBBuilder:
    def __init__(
        self,
        store: KBStore,
        clips_dir: Path,
        frames_dir: Path,
        api_key: str,
        whisper_model: str = "base",
        frame_interval: int = 8,
    ):
        self.store = store
        self.clips_dir = clips_dir
        self.frames_dir = frames_dir
        self.api_key = api_key
        self.whisper_model = whisper_model
        self.frame_interval = frame_interval

    def process_video(self, url: str, channel_key: str) -> None:
        """Download, extract, transcribe, analyze, and store one video."""
        channel_config = REFERENCE_CHANNELS.get(channel_key, {})
        channel_focus = channel_config.get("focus", "")

        logger.info(f"Downloading: {url}")
        meta = download_video(url, self.clips_dir / channel_key)

        if self.store.is_processed(meta.video_id):
            logger.info(f"Skipping {meta.video_id} — already in KB")
            return

        logger.info(f"Extracting frames: {meta.title}")
        frameset = extract_frames(meta.local_path, self.frames_dir / channel_key, self.frame_interval)

        logger.info(f"Transcribing: {meta.title}")
        transcript = transcribe_video(meta.local_path, self.whisper_model)

        logger.info(f"Analyzing: {meta.title}")
        insights = analyze_video(meta, frameset, transcript, self.api_key, channel_focus)

        logger.info(f"Storing insights: {meta.video_id} ({len(insights.key_insights)} insights)")
        self.store.add(insights)

    def process_channel(self, video_urls: list[str], channel_key: str) -> None:
        """Process all videos for a channel."""
        total = len(video_urls)
        for i, url in enumerate(video_urls, 1):
            logger.info(f"[{i}/{total}] Processing {channel_key}: {url}")
            try:
                self.process_video(url, channel_key)
            except Exception as e:
                logger.error(f"Failed {url}: {e} — continuing")
