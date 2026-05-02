#!/usr/bin/env python3
"""Mosaic Phase 2 — Production Pipeline CLI."""
import argparse
import logging
from pathlib import Path

from mosaic.config.settings import settings
from mosaic.pipeline.orchestrator import AGENT_ORDER, Orchestrator, PipelineError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="Mosaic production pipeline")
    parser.add_argument("--channel", required=True, help="Channel key (e.g. incentiveslab)")
    parser.add_argument("--topic", required=True, help="Topic slug (e.g. war-financing)")
    parser.add_argument(
        "--from",
        dest="from_agent",
        choices=AGENT_ORDER,
        default=None,
        help="Resume pipeline from this agent (skips earlier agents)",
    )
    parser.add_argument(
        "--to",
        dest="to_agent",
        choices=AGENT_ORDER,
        default=None,
        help="Stop pipeline after this agent (inclusive)",
    )
    parser.add_argument(
        "--delta",
        action="store_true",
        help="Run delta analysis (requires your_feedback.md to exist)",
    )
    args = parser.parse_args()

    if args.delta:
        from mosaic.pipeline.delta_analyzer import run_delta_analysis
        topic_dir = settings.output_dir / args.channel / args.topic
        feedback_path = topic_dir / "your_feedback.md"
        if not feedback_path.exists():
            print(f"\n[ERROR] your_feedback.md not found at {topic_dir}")
            print("Watch the video first, then write one line in your_feedback.md.")
            raise SystemExit(1)
        run_delta_analysis(topic_dir, args.channel, args.topic)
        print(f"\nDelta analysis complete: {topic_dir / 'delta_analysis.md'}")
        return

    try:
        orch = Orchestrator(
            channel=args.channel,
            topic_slug=args.topic,
            output_root=settings.output_dir,
        )
        orch.run(from_agent=args.from_agent, to_agent=args.to_agent)
    except PipelineError as e:
        print(f"\n[ERROR] {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
