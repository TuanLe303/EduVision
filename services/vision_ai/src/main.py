from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from time import time
from typing import Any, Optional, Sequence

import numpy as np

from services.vision_ai.src.behavior_detection import BehaviorDetector
from services.vision_ai.src.face_detection import FaceDetector
from services.vision_ai.src.face_recognition import FaceRecognizer
from services.vision_ai.src.object_detection import OffTaskObjectDetector
from services.vision_ai.src.seat_monitor import SeatMonitor
from services.vision_ai.src.tracking import Tracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the EduVision Vision AI pipeline.")
    parser.add_argument("--source", default="0", help="Video path, RTSP URL, or webcam index.")
    parser.add_argument("--detector", default="yolo11n", choices=["yolo11n", "yolo11s","yolo26n","yolo26s"])
    parser.add_argument(
        "--behavior-model",
        default=None,
        help="Path to custom YOLO weights. If not set, reads from config (best.pt).",
    )
    parser.add_argument(
        "--behavior-window",
        type=int,
        default=None,
        help="Override the temporal window size N from behavior config.",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack",
        choices=["bytetrack", "bytetrack_classroom", "botsort"],
    )
    parser.add_argument(
        "--person-input-size",
        type=int,
        default=640,
        help="Person detector image size. Use 1280 for distant students in Full HD classroom video.",
    )
    parser.add_argument(
        "--person-confidence",
        type=float,
        default=0.10,
        help="Minimum person detection confidence passed into the tracker.",
    )
    parser.add_argument(
        "--enable-face",
        action="store_true",
        help="Enable face detection. Also enabled by --enrollment-path or --start-class.",
    )
    parser.add_argument("--face-detector", default="scrfd", choices=["scrfd", "retinaface"])
    parser.add_argument("--recognizer", default="insightface", choices=["insightface"])
    parser.add_argument(
        "--recognition-model",
        choices=["buffalo_s", "buffalo_l"],
        help="InsightFace model pack. Defaults to face_recognition config.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="auto, cpu, cuda, cuda:0, etc. Defaults to each module config.",
    )
    parser.add_argument(
        "--face-device",
        default=None,
        help="cpu, cuda, or cuda:<index>. Defaults to each InsightFace module config.",
    )
    parser.add_argument("--enrollment-path", help="JSON file with enrolled student embeddings.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames. 0 means no limit.")
    parser.add_argument("--output-jsonl", help="Optional path to write one JSON object per processed frame.")
    parser.add_argument("--output-video", help="Optional path for an annotated output video.")
    parser.add_argument("--show", action="store_true", help="Display the annotated video stream.")
    parser.add_argument(
        "--start-class",
        action="store_true",
        help="Start attendance and seat calibration immediately for CLI runs.",
    )
    parser.add_argument("--print-every", type=int, default=1, help="Print every Nth processed frame.")
    parser.add_argument("--skip-objects", action="store_true", help="Skip off-task object detection stage.")
    parser.add_argument("--skip-face-recognition", action="store_true", help="Skip recognition even with enrollments.")
    return parser


class VisionPipeline:
    """Connect the implemented Vision AI modules into a per-frame pipeline."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._tracker = Tracker(
            model_name=args.detector,
            tracker=args.tracker,
            confidence_threshold=args.person_confidence,
            input_size=args.person_input_size,
            device=args.device,
        )
        self._behavior_detector = BehaviorDetector(
            model_path=args.behavior_model,
            device=args.device,
            window_size=args.behavior_window,
        )
        face_enabled = bool(args.enable_face or args.enrollment_path or args.start_class)
        self._face_detector = (
            FaceDetector(backend=args.face_detector, device=args.face_device)
            if face_enabled
            else None
        )
        self._recognizer = self._build_recognizer(args)
        self._object_detector = None if args.skip_objects else OffTaskObjectDetector(
            model_name=args.detector,
            device=args.device,
        )
        self._seat_monitor = SeatMonitor()
        if args.start_class:
            self.start_class()

    def _build_recognizer(self, args: argparse.Namespace) -> Optional[FaceRecognizer]:
        if args.skip_face_recognition or not args.enrollment_path:
            return None
        return FaceRecognizer(
            backend=args.recognizer,
            model_name=args.recognition_model,
            enrollment_path=args.enrollment_path,
            device=args.face_device,
        )

    def process_frame(self, frame: np.ndarray, frame_index: int, timestamp: float) -> dict[str, Any]:
        canonical_tracks = self._tracker.update(frame)
        tracks = canonical_tracks
        faces_by_track = self._detect_faces_for_tracks(frame, tracks)
        recognition_by_track = self._recognize_faces(
            frame,
            faces_by_track,
            [track.track_id for track in tracks],
        )

        objects = self._object_detector.detect(frame, persons=tracks) if self._object_detector else []
        objects_by_track: dict[int, list[Any]] = {track.track_id: [] for track in tracks}
        for detected_object in objects:
            person_id = getattr(detected_object, "person_id", None)
            if person_id in objects_by_track:
                objects_by_track[person_id].append(detected_object)

        behavior_overrides: dict[int, tuple[str, float]] = {}
        for track_id, detected_objects in objects_by_track.items():
            phone_scores = [
                float(getattr(item, "confidence", 0.0))
                for item in detected_objects
                if getattr(item, "label", "") == "cell phone"
            ]
            if phone_scores:
                behavior_overrides[track_id] = ("using_phone", max(phone_scores))

        _, frame_behaviors, behavior_results = self._behavior_detector.update(
            frame,
            frame_index,
            tracks,
            state_overrides=behavior_overrides,
        )

        seat_results = self._seat_monitor.update(
            tracks=tracks,
            recognition_by_track=recognition_by_track,
            frame_index=frame_index,
            timestamp=timestamp,
        )
        final_behavior = _build_final_behavior(
            behavior_results,
            seat_results,
            recognition_by_track,
        )

        return {
            "frame_index": frame_index,
            "timestamp": timestamp,
            "tracks": [_to_json(track) for track in tracks],
            "faces": {str(track_id): _to_json(face) for track_id, face in faces_by_track.items()},
            "recognition": {
                str(track_id): _to_json(result) for track_id, result in recognition_by_track.items()
            },
            "objects": [_to_json(detected_object) for detected_object in objects],
            "frame_behavior": [_to_json(result) for result in frame_behaviors],
            "behavior": [_to_json(result) for result in behavior_results],
            "seat": [_to_json(result) for result in seat_results],
            "final_behavior": final_behavior,
        }

    def start_class(self, frame_index: int = 0, timestamp: float = 0.0) -> None:
        self._seat_monitor.start_session(frame_index, timestamp)

    def end_class(self) -> None:
        self._seat_monitor.end_session()

    def reset(self) -> None:
        self._tracker.reset()
        self._behavior_detector.reset()
        self._seat_monitor.reset()
        if self._recognizer is not None:
            self._recognizer.reset()

    def _detect_faces_for_tracks(self, frame: np.ndarray, tracks: Sequence[Any]) -> dict[int, Any]:
        if self._face_detector is None:
            return {}
        faces_by_track: dict[int, Any] = {}
        for track in tracks:
            crop, offset = _crop(frame, track.bbox)
            if crop.size == 0:
                continue

            faces = self._face_detector.detect(crop)
            if not faces:
                continue

            face = _select_face_for_person_crop(faces)
            face.bbox = _offset_bbox(face.bbox, offset)
            face.landmarks = [
                [float(point[0] + offset[0]), float(point[1] + offset[1])]
                for point in face.landmarks
            ]
            faces_by_track[track.track_id] = face
        return faces_by_track

    def _recognize_faces(
        self,
        frame: np.ndarray,
        faces_by_track: dict[int, Any],
        active_track_ids: Sequence[int],
    ) -> dict[int, Any]:
        if self._recognizer is None:
            return {}

        self._recognizer.update_active_tracks(active_track_ids)
        recognition_by_track: dict[int, Any] = {}
        for track_id, face in faces_by_track.items():
            if len(face.landmarks) == 5:
                recognition_by_track[track_id] = self._recognizer.recognize(
                    frame,
                    landmarks=face.landmarks,
                    track_id=track_id,
                )
                continue

            crop, _ = _crop(frame, _expand_bbox(face.bbox, 0.2))
            if crop.size:
                recognition_by_track[track_id] = self._recognizer.recognize(
                    crop,
                    track_id=track_id,
                )
        return recognition_by_track


def _build_final_behavior(
    behavior_results: Sequence[Any],
    seat_results: Sequence[Any],
    recognition_by_track: dict[int, Any],
) -> list[dict[str, Any]]:
    """Expose only temporally ready labels; seat-away has explicit priority."""

    ready_by_track = {
        int(result.track_id): result
        for result in behavior_results
        if getattr(result, "ready", False) and getattr(result, "state", None) is not None
    }
    seat_by_track = {
        int(result.track_id): result
        for result in seat_results
        if getattr(result, "track_id", None) is not None
    }
    output: list[dict[str, Any]] = []
    for track_id in sorted(set(ready_by_track) | set(seat_by_track)):
        temporal = ready_by_track.get(track_id)
        seat = seat_by_track.get(track_id)
        if seat is not None and seat.state == "away_from_seat":
            state = "away_from_seat"
            confidence = seat.confidence
            reason = seat.reason
            source = "seat_monitor"
        elif temporal is not None:
            state = temporal.state
            confidence = temporal.confidence
            reason = temporal.reason
            source = "behavior_yolo_temporal"
        else:
            continue
        output.append(
            {
                "track_id": track_id,
                "student_id": _recognized_identity(recognition_by_track.get(track_id)),
                "state": state,
                "confidence": float(confidence),
                "reason": reason,
                "source": source,
            }
        )
    return output


def annotate_frame(frame: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    """Draw canonical boxes and only behavior labels that passed temporal gating."""

    import cv2

    annotated = frame.copy()
    final_by_track = {
        int(item["track_id"]): item for item in result.get("final_behavior", [])
    }
    recognition = result.get("recognition", {})
    for track in result.get("tracks", []):
        track_id = int(track["track_id"])
        x1, y1, x2, y2 = (int(round(value)) for value in track["bbox"])
        final = final_by_track.get(track_id)
        identity = _recognized_identity(recognition.get(str(track_id)))
        label_parts = [identity or f"ID {track_id}"]
        color = (160, 160, 160)
        if final is not None:
            label_parts.append(str(final["state"]))
            label_parts.append(f"{float(final['confidence']):.2f}")
            color = _state_color(str(final["state"]))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            " | ".join(label_parts),
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def _recognized_identity(result: Any) -> Optional[str]:
    if result is None:
        return None
    if isinstance(result, dict):
        if result.get("recognized") is False or result.get("matched") is False:
            return None
        value = result.get("student_id") or result.get("identity") or result.get("label")
    else:
        if getattr(result, "recognized", True) is False or getattr(result, "matched", True) is False:
            return None
        value = (
            getattr(result, "student_id", None)
            or getattr(result, "identity", None)
            or getattr(result, "label", None)
        )
    return str(value) if value not in {None, "", "unknown"} else None


def _state_color(state: str) -> tuple[int, int, int]:
    if state in {"away_from_seat", "sleeping", "using_phone"}:
        return (0, 0, 255)
    if state in {"drowsy", "off_task", "side_talking"}:
        return (0, 165, 255)
    if state == "raising_hand":
        return (255, 180, 0)
    return (0, 200, 0)


def run(args: argparse.Namespace) -> int:
    import cv2

    source = _source_value(args.source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"Failed to open video source: {args.source}", file=sys.stderr)
        return 2

    pipeline = VisionPipeline(args)
    output_handle = None
    video_writer = None
    show_enabled = bool(args.show)
    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_path.open("w", encoding="utf-8")
    if args.output_video:
        output_path = Path(args.output_video)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_index += 1
            media_timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            # File timestamps start at 0. Using wall-clock time for only that
            # first frame would make the session duration billions of seconds.
            timestamp = (
                time()
                if isinstance(source, int) or str(source).lower().startswith(("rtsp://", "http://", "https://"))
                else max(0.0, media_timestamp_ms / 1000.0)
            )
            result = pipeline.process_frame(frame, frame_index, timestamp)
            line = json.dumps(result, ensure_ascii=False)
            annotated = annotate_frame(frame, result) if show_enabled or video_writer is not None else None

            if args.print_every > 0 and frame_index % args.print_every == 0:
                print(line)
            if output_handle is not None:
                output_handle.write(line + "\n")
            if video_writer is not None and annotated is not None:
                video_writer.write(annotated)
            if show_enabled and annotated is not None:
                try:
                    cv2.imshow("EduVision", annotated)
                    if cv2.waitKey(1) & 0xFF in {27, ord("q")}:
                        break
                except cv2.error as exc:
                    show_enabled = False
                    print(
                        "OpenCV GUI is unavailable; disabling --show and continuing "
                        f"the pipeline. Details: {exc}",
                        file=sys.stderr,
                    )

            if args.max_frames and frame_index >= args.max_frames:
                break
    finally:
        pipeline.reset()
        capture.release()
        if video_writer is not None:
            video_writer.release()
        if show_enabled:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        if output_handle is not None:
            output_handle.close()

    return 0


def _source_value(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def _crop(frame: np.ndarray, bbox: Sequence[float]) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(bbox, width, height)
    return frame[y1:y2, x1:x2], (x1, y1)


def _clip_bbox(bbox: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    left = max(0, min(width, int(round(x1))))
    top = max(0, min(height, int(round(y1))))
    right = max(0, min(width, int(round(x2))))
    bottom = max(0, min(height, int(round(y2))))
    return left, top, right, bottom


def _offset_bbox(bbox: Sequence[float], offset: tuple[int, int]) -> list[float]:
    return [
        float(bbox[0] + offset[0]),
        float(bbox[1] + offset[1]),
        float(bbox[2] + offset[0]),
        float(bbox[3] + offset[1]),
    ]


def _expand_bbox(bbox: Sequence[float], margin: float) -> list[float]:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return [
        x1 - width * margin,
        y1 - height * margin,
        x2 + width * margin,
        y2 + height * margin,
    ]


def _select_face_for_person_crop(faces: Sequence[Any]) -> Any:
    """Prefer the largest face so a partial bystander face is less likely to win."""
    return max(
        faces,
        key=lambda face: (
            _bbox_area(getattr(face, "bbox", [])),
            float(getattr(face, "confidence", 0.0)),
        ),
    )


def _bbox_area(bbox: Sequence[float]) -> float:
    if len(bbox) < 4:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(
        0.0, float(bbox[3]) - float(bbox[1])
    )


def _to_json(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_json(item) for key, item in asdict(value).items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
