"""
EduVision — Vision pipeline → Database bridge.

Usage (run alongside the vision pipeline):

    # Standalone — wraps the CLI pipeline and pushes events to the running API
    python -m services.backend_api.event_pusher \\
        --session-id 1 \\
        --source rtsp://192.168.x.x:8554/live \\
        --api-url http://localhost:8000

Or import and call push_frame_result() from your own pipeline loop.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import Any, Optional

import requests


# ---------------------------------------------------------------------------
# Core pusher
# ---------------------------------------------------------------------------


class EventPusher:
    """
    Buffers final_behavior events produced by VisionPipeline.process_frame()
    and periodically flushes them to the backend API.
    """

    def __init__(
        self,
        session_id: int,
        api_url: str = "http://localhost:8000",
        flush_every: int = 30,  # flush after N accumulated frames
        timeout: float = 5.0,
    ) -> None:
        self.session_id = session_id
        self._url = api_url.rstrip("/")
        self._flush_every = flush_every
        self._timeout = timeout
        self._buffer: list[dict[str, Any]] = []
        self._flush_count = 0
        self._error_count = 0

    def push_frame_result(self, result: dict[str, Any], timestamp: float) -> None:
        """
        Call this with the dict returned by VisionPipeline.process_frame().
        final_behavior entries are extracted and buffered.
        """
        tracks = result.get("tracks", [])
        final_behavior = result.get("final_behavior", [])
        
        if tracks:
            def _push_tracks():
                behavior_map = {item.get("track_id"): item for item in final_behavior}
                merged_tracks = []
                for t in tracks:
                    tid = t.get("track_id")
                    b = behavior_map.get(tid, {})
                    merged_tracks.append({
                        "track_id": tid,
                        "bbox": t.get("bbox"),
                        "state": b.get("state", "unknown"),
                        "name": b.get("name"),
                        "student_id": b.get("student_id")
                    })
                payload = {
                    "type": "frame",
                    "tracks": merged_tracks,
                }
                try:
                    requests.post(f"{self._url}/api/ws/frame", json=payload, timeout=1.0)
                except Exception:
                    pass
            threading.Thread(target=_push_tracks, daemon=True).start()

        for item in final_behavior:
            self._buffer.append(
                {
                    "student_id": item.get("student_id"),
                    "track_id": item.get("track_id"),
                    "state": item.get("state", "focused"),
                    "confidence": float(item.get("confidence", 0.0)),
                    "source": item.get("source"),
                    "ts": timestamp,
                }
            )

        self._flush_count += 1
        if self._flush_count >= self._flush_every:
            self.flush()
            self._flush_count = 0

    def flush(self) -> int:
        """Send buffered events to the API. Returns number of events sent."""
        if not self._buffer:
            return 0
        payload = {"events": self._buffer}
        try:
            resp = requests.post(
                f"{self._url}/api/sessions/{self.session_id}/events",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            sent = len(self._buffer)
            self._buffer.clear()
            return sent
        except Exception as exc:
            self._error_count += 1
            print(f"[EventPusher] flush error #{self._error_count}: {exc}", file=sys.stderr)
            # Keep buffer to retry on next flush
            return 0

    def close(self) -> None:
        """Flush remaining events and clean up."""
        self.flush()


# ---------------------------------------------------------------------------
# Standalone CLI runner
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run EduVision pipeline and push events to the backend API."
    )
    p.add_argument("--session-id", type=int, required=True, help="Active session ID.")
    p.add_argument(
        "--source", default="0",
        help="Video path, RTSP URL, or webcam index (default: 0).",
    )
    p.add_argument("--api-url", default="http://localhost:8000", help="Base URL of the EduVision API.")
    p.add_argument("--detector", default="yolo11n", choices=["yolo11n", "yolo11s", "yolo26n", "yolo26s"])
    p.add_argument("--behavior-model", default="weights/behavior_yolo26n.pt")
    p.add_argument("--enrollment-path", default=None, help="Path to enrollment JSON (auto-detected if omitted).")
    p.add_argument("--flush-every", type=int, default=30, help="Flush to API every N frames.")
    p.add_argument("--show", action="store_true", help="Display annotated video.")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--face-device", default=None)
    p.add_argument("--device", default=None)
    return p


def run(args: argparse.Namespace) -> int:
    import cv2

    from pathlib import Path

    from services.vision_ai.src.main import VisionPipeline, annotate_frame, build_parser as _vp_parser

    # Auto-detect enrollment JSON
    enrollment_path = args.enrollment_path
    if enrollment_path is None:
        candidate = Path(__file__).resolve().parents[2] / "data" / "enrollments.json"
        enrollment_path = str(candidate) if candidate.exists() else None

    # Build a Namespace that VisionPipeline expects
    vp_args_list = [
        "--source", str(args.source),
        "--detector", args.detector,
        "--behavior-model", args.behavior_model,
        "--enable-face",
    ]
    if enrollment_path:
        vp_args_list += ["--enrollment-path", enrollment_path]
    if args.show:
        vp_args_list.append("--show")
    if args.max_frames:
        vp_args_list += ["--max-frames", str(args.max_frames)]
    if args.device:
        vp_args_list += ["--device", args.device]
    if args.face_device:
        vp_args_list += ["--face-device", args.face_device]

    vp_args = _vp_parser().parse_args(vp_args_list)

    pipeline = VisionPipeline(vp_args)
    pipeline.start_class()
    pusher = EventPusher(
        session_id=args.session_id,
        api_url=args.api_url,
        flush_every=args.flush_every,
    )

    source_val: Any = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        print(f"Failed to open source: {args.source}", file=sys.stderr)
        return 2

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            ts = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if isinstance(source_val, int) or str(source_val).lower().startswith(("rtsp://", "http://")):
                ts = time.time()

            result = pipeline.process_frame(frame, frame_index, ts)
            pusher.push_frame_result(result, ts)

            if args.show:
                annotated = annotate_frame(frame, result)
                cv2.imshow("EduVision", annotated)
                if cv2.waitKey(1) & 0xFF in {27, ord("q")}:
                    break

            if args.max_frames and frame_index >= args.max_frames:
                break
    finally:
        pusher.close()
        pipeline.reset()
        cap.release()
        if args.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    return run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
