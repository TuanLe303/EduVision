"""Command-line E2E accuracy evaluator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.evaluate.metrics import evaluate_records


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EduVision E2E output JSONL.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--event-iou", type=float, default=0.5)
    parser.add_argument("--bbox-iou", type=float, default=0.5)
    parser.add_argument(
        "--behavior-output",
        choices=["final", "frame"],
        default="final",
        help="Score temporally aggregated final_behavior (E2E) or raw frame_behavior",
    )
    parser.add_argument("--quiet", action="store_true", help="Write output without printing full JSON")
    parser.add_argument("--performance", type=Path, help="Optional JSON from benchmark.py")
    parser.add_argument("--case", choices=["best", "normal", "worst"], help="Label only; no case rules yet")
    args = parser.parse_args()

    performance: dict[str, Any] = {}
    if args.performance:
        performance = json.loads(args.performance.read_text(encoding="utf-8"))
    result = evaluate_records(
        _jsonl(args.predictions),
        _jsonl(args.ground_truth),
        fps=args.fps,
        event_iou_threshold=args.event_iou,
        bbox_iou_threshold=args.bbox_iou,
        behavior_output=args.behavior_output,
        runtime_seconds=performance.get("runtime_seconds"),
        video_duration_seconds=performance.get("video_duration_seconds"),
        peak_ram_mb=performance.get("peak_ram_mb"),
        peak_vram_mb=performance.get("peak_vram_mb"),
    )
    result["case"] = args.case
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not args.quiet:
        print(rendered)


if __name__ == "__main__":
    main()
