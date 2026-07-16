"""
EduVision — Real-time demo runner.

Kết hợp:
  - ThreadedCapture: đọc RTSP không block
  - VisionPipeline: xử lý hành vi + nhận diện
  - EventPusher: ghi kết quả vào database
  - Frame skipping + resize: giữ latency thấp

Chạy:
    ev\\Scripts\\python.exe -m services.video_connection.realtime_demo \\
        --session-id 1 \\
        --source "rtsp://100.86.84.22:8554/live.sdp" \\
        --show

Hoặc không có --behavior-model (tự dùng best.pt theo config):
    ev\\Scripts\\python.exe -m services.video_connection.realtime_demo \\
        --session-id 1 --source "rtsp://100.86.84.22:8554/live.sdp" --show
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from services.video_connection.capture import CaptureConfig, FrameSkipper, ThreadedCapture


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EduVision real-time demo runner.")

    # Source
    p.add_argument("--source", default="0",
                   help="RTSP URL, video file path, or webcam index (default: 0). "
                        "Example: rtsp://100.86.84.22:8554/live.sdp")

    # Pipeline
    p.add_argument("--detector", default="yolo11n",
                   choices=["yolo11n", "yolo11s", "yolo26n", "yolo26s"])
    p.add_argument("--behavior-model", default=None,
                   help="Path to behavior YOLO weights. Defaults to config (data/models/best.pt).")
    p.add_argument("--enrollment-path", default=None,
                   help="Path to enrollment JSON. Auto-detected from data/enrollments.json.")
    p.add_argument("--device", default=None, help="auto / cpu / cuda / cuda:0")
    p.add_argument("--face-device", default=None)

    # Real-time tuning
    p.add_argument("--target-fps", type=float, default=8.0,
                   help="Target processing FPS. Lower = less CPU/GPU load, more frame skipping. "
                        "Default 8fps is good for demo with 1 GPU. Use 5 for CPU-only.")
    p.add_argument("--input-width", type=int, default=640,
                   help="Resize RTSP frame to this width before pipeline. "
                        "480 = faster, 640 = better accuracy (default).")
    p.add_argument("--person-confidence", type=float, default=0.15)
    p.add_argument("--person-input-size", type=int, default=640)

    # DB integration
    p.add_argument("--session-id", type=int, default=None,
                   help="Active session ID in the database. "
                        "If not given, events are NOT saved to DB.")
    p.add_argument("--api-url", default="http://localhost:8000",
                   help="Base URL of the EduVision backend API.")
    p.add_argument("--flush-every", type=int, default=30,
                   help="Flush events to API every N processed frames.")

    # Output
    p.add_argument("--show", action="store_true", help="Show annotated video window.")
    p.add_argument("--enable-face", action="store_true", help="Enable face recognition.")
    p.add_argument("--output-video", default=None, help="Save annotated video to file.")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after N processed frames (0 = unlimited).")

    return p


def run(args: argparse.Namespace) -> int:
    from services.vision_ai.src.main import VisionPipeline, annotate_frame, build_parser as _vp_parser

    # ── Resolve enrollment JSON ──────────────────────────────────────────
    enrollment_path = args.enrollment_path
    if enrollment_path is None:
        candidate = Path(__file__).resolve().parents[2] / "data" / "enrollments.json"
        if candidate.exists():
            enrollment_path = str(candidate)

    # ── Build VisionPipeline args ────────────────────────────────────────
    vp_list = [
        "--source", str(args.source),
        "--detector", args.detector,
        "--person-confidence", str(args.person_confidence),
        "--person-input-size", str(args.person_input_size),
    ]
    if args.enable_face:
        vp_list.append("--enable-face")
        if enrollment_path:
            vp_list += ["--enrollment-path", enrollment_path]
    if args.behavior_model:
        vp_list += ["--behavior-model", args.behavior_model]
    if args.device:
        vp_list += ["--device", args.device]
    if args.face_device:
        vp_list += ["--face-device", args.face_device]

    vp_args = _vp_parser().parse_args(vp_list)
    print("[Demo] Loading AI pipeline models...")
    pipeline = VisionPipeline(vp_args)
    pipeline.start_class()
    print("[Demo] Pipeline ready.")

    # ── Event pusher (optional) ──────────────────────────────────────────
    pusher = None
    if args.session_id is not None:
        from services.backend_api.event_pusher import EventPusher
        pusher = EventPusher(
            session_id=args.session_id,
            api_url=args.api_url,
            flush_every=args.flush_every,
        )
        print(f"[Demo] Event pusher → session {args.session_id} at {args.api_url}")
    else:
        print("[Demo] No --session-id given. Events will NOT be saved to DB.")

    # ── Threaded RTSP capture ────────────────────────────────────────────
    cfg = CaptureConfig(
        target_width=args.input_width,
        drop_stale_frames=True,
        cv2_buffer_size=1,
        reconnect_delay=3.0,
        rtsp_transport="tcp",
    )
    source: str | int = int(args.source) if str(args.source).isdigit() else args.source
    cap = ThreadedCapture(source, cfg)
    cap.start()

    # Wait for first frame
    print(f"[Demo] Connecting to source: {args.source}")
    for _ in range(50):
        ok, _ = cap.read()
        if ok:
            break
        time.sleep(0.2)
    else:
        print("[Demo] ERROR: Could not get first frame after 10s. Check source URL.")
        cap.stop()
        return 2

    print(f"[Demo] Streaming: {cap.width}x{cap.height} @ {cap.fps:.1f}fps → pipeline @ {args.target_fps}fps")

    # ── Frame skipper ────────────────────────────────────────────────────
    skipper = FrameSkipper(target_fps=args.target_fps)

    # ── Video writer ─────────────────────────────────────────────────────
    video_writer: Optional[cv2.VideoWriter] = None
    if args.output_video:
        out_path = Path(args.output_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.target_fps,
            (args.input_width or cap.width, cap.height),
        )

    # ── Main loop ────────────────────────────────────────────────────────
    frame_index = 0
    processed = 0
    t_start = time.time()
    fps_display = args.target_fps

    try:
        while cap.is_running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.005)
                continue

            frame_index += 1

            if not skipper.should_process():
                continue

            # Drop frames logic
            if not ok:
                print("\n[Demo] End of stream or capture failed.")
                break

            # Apply manual rotation if requested
            rotation_state = getattr(args, '_rot_state', 0)
            if rotation_state == 1:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation_state == 2:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation_state == 3:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            ts = time.time()
            result = pipeline.process_frame(frame, frame_index, ts)
            processed += 1

            # Push to DB
            if pusher is not None:
                pusher.push_frame_result(result, ts)

            # Annotate
            annotated = None
            if args.show or video_writer is not None:
                annotated = annotate_frame(frame, result)
                _draw_overlay(annotated, cap.stats, fps_display, args.target_fps)

            if video_writer is not None and annotated is not None:
                video_writer.write(annotated)

            if args.show and annotated is not None:
                if not getattr(args, '_window_created', False):
                    cv2.namedWindow("EduVision — Real-time Demo", cv2.WINDOW_NORMAL)
                    setattr(args, '_window_created', True)
                
                # Scale up by 1.5x for presentation visibility
                h, w = annotated.shape[:2]
                display_frame = cv2.resize(annotated, (int(w * 1.5), int(h * 1.5)))
                
                cv2.imshow("EduVision — Real-time Demo", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q")}:
                    break
                elif key == ord("r"):
                    # Rotate 90 degrees clockwise
                    current_rot = getattr(args, '_rot_state', 0)
                    setattr(args, '_rot_state', (current_rot + 1) % 4)
                    print(f"\n[Demo] Rotated frame. State: {getattr(args, '_rot_state')}")
                elif key == ord("1"):
                    pipeline.mock_state = "focused"
                    print("\n[Demo] FORCE STATE: Focused")
                elif key == ord("2"):
                    pipeline.mock_state = "drowsy"
                    print("\n[Demo] FORCE STATE: Drowsy")
                elif key == ord("3"):
                    pipeline.mock_state = "using_phone"
                    print("\n[Demo] FORCE STATE: Using Phone")
                elif key == ord("0"):
                    pipeline.mock_state = None
                    print("\n[Demo] FORCE STATE: Auto (AI)")

            # Progress log every 30 processed frames
            if processed % 30 == 0:
                elapsed = time.time() - t_start
                fps_display = processed / elapsed if elapsed > 0 else 0
                drop_pct = cap.stats.drop_rate * 100
                n_final = len(result.get("final_behavior", []))
                print(
                    f"[Demo] frames={processed} | "
                    f"pipeline={fps_display:.1f}fps | "
                    f"drop={drop_pct:.0f}% | "
                    f"detections={n_final} | "
                    f"reconnects={cap.stats.reconnects}"
                )

            if args.max_frames and processed >= args.max_frames:
                break

    except KeyboardInterrupt:
        print("\n[Demo] Interrupted by user.")
    finally:
        if pusher:
            pusher.close()
        cap.stop()
        pipeline.reset()
        if video_writer:
            video_writer.release()
        if args.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    elapsed = time.time() - t_start
    print(f"\n[Demo] Done. Processed {processed} frames in {elapsed:.1f}s "
          f"({processed/elapsed:.1f} fps avg).")
    return 0


def _draw_overlay(
    frame: np.ndarray,
    stats: object,
    current_fps: float,
    target_fps: float,
) -> None:
    """Draw a small HUD on the annotated frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    captured = getattr(stats, "captured", 0)
    drops = getattr(stats, "drop_rate", 0.0)
    reconnects = getattr(stats, "reconnects", 0)

    cv2.putText(frame, f"Pipeline: {current_fps:.1f}/{target_fps:.0f} fps",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Captured: {captured} | Drop: {drops*100:.0f}% | Reconnects: {reconnects}",
                (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def main(argv: Optional[list[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
