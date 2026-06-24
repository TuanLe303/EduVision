"""Run the EduVision vision pipeline and collect E2E performance metrics."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import cv2

from services.vision_ai.src.main import VisionPipeline, build_parser, _source_value


def _p95(values: list[float]) -> float:
    values = sorted(values)
    position = (len(values) - 1) * 0.95
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _ram_mb() -> float | None:
    try:
        import psutil  # type: ignore[import-not-found]

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return None


def main() -> None:
    parser = build_parser()
    parser.description = "Benchmark the complete EduVision vision pipeline."
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--warmup-frames", type=int, default=5)
    args = parser.parse_args()

    source = _source_value(args.source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    pipeline = VisionPipeline(args)
    latencies_ms: list[float] = []
    processed = 0
    peak_ram = _ram_mb()
    measured_ended: float | None = None
    output_handle = None
    if args.output_jsonl:
        path = Path(args.output_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = path.open("w", encoding="utf-8")

    measured_started: float | None = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            processed += 1
            timestamp = max(0.0, float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
            started = perf_counter()
            result = pipeline.process_frame(frame, processed, timestamp)
            latency_ms = (perf_counter() - started) * 1000.0
            result["processing_ms"] = latency_ms
            if processed > args.warmup_frames:
                if measured_started is None:
                    measured_started = started
                latencies_ms.append(latency_ms)
            if output_handle:
                output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            current_ram = _ram_mb()
            if current_ram is not None:
                peak_ram = max(peak_ram or 0.0, current_ram)
            if processed > args.warmup_frames:
                measured_ended = perf_counter()
            if args.max_frames and processed >= args.max_frames:
                break
    finally:
        capture.release()
        pipeline.reset()
        if output_handle:
            output_handle.close()

    runtime = (
        measured_ended - measured_started
        if measured_started is not None and measured_ended is not None
        else 0.0
    )
    measured_frames = len(latencies_ms)
    video_duration = measured_frames / source_fps if source_fps > 0 else None
    peak_vram = None
    try:
        import torch

        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except (ImportError, RuntimeError):
        pass
    metrics = {
        "processed_frames": processed,
        "measured_frames": measured_frames,
        "warmup_frames": min(processed, args.warmup_frames),
        "runtime_seconds": runtime,
        "video_duration_seconds": video_duration,
        "effective_fps": measured_frames / runtime if runtime > 0 else None,
        "real_time_factor": runtime / video_duration if video_duration else None,
        "mean_latency_ms": mean(latencies_ms) if latencies_ms else None,
        "p95_latency_ms": _p95(latencies_ms) if latencies_ms else None,
        "peak_ram_mb": peak_ram,
        "peak_vram_mb": peak_vram,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
