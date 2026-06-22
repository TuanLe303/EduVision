"""CLI entry point for the EduVision Report Generator.

Usage examples
--------------
# Tiếng Việt, dùng Gemini, in ra stdout:
python -m services.report_generator.main --source session.jsonl --llm gemini

# Tiếng Anh, dùng GPT, lưu ra file:
python -m services.report_generator.main --source session.jsonl --llm gpt --language en --output report.md

# Dùng cấu hình mặc định trong config.yaml:
python -m services.report_generator.main --source session.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from loguru import logger

from .src.aggregator import aggregate_jsonl
from .src.llm_client import build_llm_client
from .src.prompt_builder import build_prompt

_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "services"
    / "report_generator"
    / "config.yaml"
)


def _load_config(cfg_path: Path) -> dict:
    import yaml

    if not cfg_path.exists():
        return {}
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m services.report_generator.main",
        description="Generate an AI session report from EduVision vision-AI output.",
    )
    parser.add_argument(
        "--source",
        required=True,
        metavar="SESSION.JSONL",
        help="Path to the JSONL file produced by the vision_ai pipeline (--output-jsonl).",
    )
    parser.add_argument(
        "--llm",
        default=None,
        choices=["gemini", "gpt"],
        help="LLM provider.  Overrides llm_provider in config.yaml.",
    )
    parser.add_argument(
        "--language",
        default=None,
        choices=["vi", "en"],
        help="Report language: 'vi' (Vietnamese, default) or 'en' (English).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="REPORT.MD",
        help="Path to write the Markdown report.  Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=None,
        metavar="N",
        help="Minimum frames a track must appear in to be included (default: 5).",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the LLM prompt and exit without calling the API (useful for debugging).",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    cfg = _load_config(_CONFIG_PATH)

    provider: str = args.llm or cfg.get("llm_provider", "gemini")
    language: str = args.language or cfg.get("report_language", "vi")
    min_frames: int = (
        args.min_frames
        if args.min_frames is not None
        else cfg.get("min_present_frames", 5)
    )

    source_path = Path(args.source)
    if not source_path.exists():
        logger.error("Source file not found: {}", source_path)
        return 2

    logger.info("Aggregating session data from {}", source_path)
    summary = aggregate_jsonl(source_path, min_present_frames=min_frames)

    cs = summary.class_stats
    logger.info(
        "Session aggregated — {} students | {} frames | {:.1f}% avg attention",
        cs.total_students,
        cs.total_frames,
        cs.avg_attention_score * 100,
    )

    prompt = build_prompt(summary, language=language)

    if args.print_prompt:
        print(prompt)
        return 0

    logger.info("Calling {} API to generate report...", provider)
    client = build_llm_client(provider, cfg)
    report = client.generate(prompt)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        logger.info("Report saved to {}", out_path)
    else:
        print(report)

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logger.error("Report generation failed: {}", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
