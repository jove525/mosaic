"""
Mosaic KB Builder — CLI entry point.

Usage:
  python build_kb.py --channel wendover --limit 5
  python build_kb.py --channel all --limit 20
  python build_kb.py --seed-only

Before running:
  1. Add video URLs to mosaic/config/channels.py
  2. Set ANTHROPIC_API_KEY in .env
"""
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import os
from mosaic.config.settings import settings
from mosaic.config.channels import REFERENCE_CHANNELS
from mosaic.kb.store import KBStore
from mosaic.kb.builder import KBBuilder
from mosaic.kb.seed_loader import load_seed_guide
from mosaic.utils.fs import ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build the Mosaic editorial knowledge base.")
    parser.add_argument("--channel", default="all", help="Channel key to process, or 'all'")
    parser.add_argument("--limit", type=int, default=None, help="Max videos per channel")
    parser.add_argument("--seed-only", action="store_true", help="Load seed guide only, skip video analysis")
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key and not args.seed_only:
        raise ValueError("ANTHROPIC_API_KEY not set in .env")

    for d in [settings.kb_dir, settings.clips_dir, settings.frames_dir]:
        ensure_dir(d)

    store = KBStore(persist_dir=settings.kb_dir)

    seed_path = settings.seed_dir / "editorial_guide.md"
    if seed_path.exists():
        logger.info("Loading seed editorial guide...")
        load_seed_guide(seed_path, store)
        logger.info(f"Seed loaded. KB now has {store._collection.count()} entries.")
    else:
        logger.warning(f"Seed guide not found at {seed_path}")

    if args.seed_only:
        logger.info("--seed-only flag set. Done.")
        return

    builder = KBBuilder(
        store=store,
        clips_dir=settings.clips_dir,
        frames_dir=settings.frames_dir,
        api_key=api_key,
        whisper_model=settings.whisper_model,
        frame_interval=settings.frame_interval_seconds,
    )

    channels_to_process = (
        list(REFERENCE_CHANNELS.keys())
        if args.channel == "all"
        else [args.channel]
    )

    for channel_key in channels_to_process:
        if channel_key not in REFERENCE_CHANNELS:
            logger.error(f"Unknown channel: {channel_key}. Available: {list(REFERENCE_CHANNELS.keys())}")
            continue

        urls = REFERENCE_CHANNELS[channel_key]["video_urls"]
        if not urls:
            logger.warning(f"No video URLs configured for {channel_key}. Add them to mosaic/config/channels.py")
            continue

        if args.limit:
            urls = urls[:args.limit]

        logger.info(f"Processing {channel_key}: {len(urls)} videos")
        builder.process_channel(urls, channel_key)

    logger.info(f"KB build complete. Total entries: {store._collection.count()}")


if __name__ == "__main__":
    main()
