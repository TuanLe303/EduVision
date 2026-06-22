from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from time import time
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from services.vision_ai.src.behavior import BehaviorAnalyzer, StudentFrameSignal
from services.vision_ai.src.face_detection import FaceDetector
from services.vision_ai.src.face_recognition import FaceRecognizer
from services.vision_ai.src.head_pose import HeadPoseEstimator
from services.vision_ai.src.object_detection import OffTaskObjectDetector
from services.vision_ai.src.tracking import Tracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the EduVision Vision AI pipeline.")
    parser.add_argument("--source", default="0", help="Video path, RTSP URL, or webcam index.")
    parser.add_argument("--detector", default="yolo11n", choices=["yolo11n", "yolo11s","yolo26n","yolo26s"])
    parser.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "botsort"])
    parser.add_argument("--face-detector", default="scrfd", choices=["scrfd", "retinaface"])
    parser.add_argument("--recognizer", default="insightface", choices=["insightface"])
    parser.add_argument(
        "--recognition-model",
        choices=["buffalo_s", "buffalo_l"],
        help="InsightFace model pack. Defaults to face_recognition config.",
    )
    parser.add_argument("--pose", default="yolo11n-pose", choices=["yolo11n-pose", "yolo11s-pose"])
    parser.add_argument(
        "--head-pose",
        default="mediapipe-solvepnp",
        choices=["mediapipe-solvepnp"],
        help="Head-pose backend. Currently only MediaPipe + solvePnP is implemented.",
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
    parser.add_argument("--print-every", type=int, default=1, help="Print every Nth processed frame.")
    parser.add_argument("--skip-pose", action="store_true", help="Skip pose estimation stage.")
    parser.add_argument("--skip-head-pose", action="store_true", help="Skip head-pose/gaze stage.")
    parser.add_argument("--skip-objects", action="store_true", help="Skip off-task object detection stage.")
    parser.add_argument("--skip-face-recognition", action="store_true", help="Skip recognition even with enrollments.")
    return parser


class VisionPipeline:
    """Connect the implemented Vision AI modules into a per-frame pipeline."""

    def __init__(self, args: argparse.Namespace) -> None:
        tracking_model = args.detector if args.skip_pose else args.pose
        self._tracker = Tracker(
            model_name=tracking_model,
            tracker=args.tracker,
            device=args.device,
        )
        self._face_detector = FaceDetector(
            backend=args.face_detector,
            device=args.face_device,
        )
        self._recognizer = self._build_recognizer(args)
        self._head_pose_estimator = None if args.skip_head_pose else HeadPoseEstimator()
        self._object_detector = None if args.skip_objects else OffTaskObjectDetector(
            model_name=args.detector,
            device=args.device,
        )
        self._behavior_analyzer = BehaviorAnalyzer()

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
        tracks, poses_by_track = self._tracker.update_with_poses(frame)
        faces_by_track = self._detect_faces_for_tracks(frame, tracks)
        recognition_by_track = self._recognize_faces(
            frame,
            faces_by_track,
            [track.track_id for track in tracks],
        )

        head_poses = self._head_pose_estimator.estimate(frame) if self._head_pose_estimator else []
        head_poses_by_track = {
            track.track_id: _best_head_pose(track.bbox, faces_by_track.get(track.track_id), head_poses)
            for track in tracks
        }

        objects = self._object_detector.detect(frame, persons=tracks) if self._object_detector else []
        objects_by_track: dict[int, list[Any]] = {track.track_id: [] for track in tracks}
        for detected_object in objects:
            person_id = getattr(detected_object, "person_id", None)
            if person_id in objects_by_track:
                objects_by_track[person_id].append(detected_object)

        behavior_results = []
        for track in tracks:
            recognition = recognition_by_track.get(track.track_id)
            face = faces_by_track.get(track.track_id)
            behavior_results.append(
                self._behavior_analyzer.analyze(
                    StudentFrameSignal(
                        track_id=track.track_id,
                        student_id=getattr(recognition, "student_id", None),
                        timestamp=timestamp,
                        face_detected=face is not None,
                        recognized=bool(getattr(recognition, "matched", False)),
                        seated=True,
                        head_pose=head_poses_by_track.get(track.track_id),
                        pose=poses_by_track.get(track.track_id),
                        objects=objects_by_track.get(track.track_id, []),
                    )
                )
            )

        return {
            "frame_index": frame_index,
            "timestamp": timestamp,
            "tracks": [_to_json(track) for track in tracks],
            "faces": {str(track_id): _to_json(face) for track_id, face in faces_by_track.items()},
            "recognition": {
                str(track_id): _to_json(result) for track_id, result in recognition_by_track.items()
            },
            "poses": {
                str(track_id): _to_json(pose)
                for track_id, pose in poses_by_track.items()
                if pose is not None
            },
            "head_poses": {
                str(track_id): _to_json(head_pose)
                for track_id, head_pose in head_poses_by_track.items()
                if head_pose is not None
            },
            "objects": [_to_json(detected_object) for detected_object in objects],
            "behavior": [_to_json(result) for result in behavior_results],
        }

    def reset(self) -> None:
        self._tracker.reset()
        if self._recognizer is not None:
            self._recognizer.reset()
        self._behavior_analyzer.reset()

    def _detect_faces_for_tracks(self, frame: np.ndarray, tracks: Sequence[Any]) -> dict[int, Any]:
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


def run(args: argparse.Namespace) -> int:
    import cv2

    capture = cv2.VideoCapture(_source_value(args.source))
    if not capture.isOpened():
        print(f"Failed to open video source: {args.source}", file=sys.stderr)
        return 2

    pipeline = VisionPipeline(args)
    output_handle = None
    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_path.open("w", encoding="utf-8")

    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_index += 1
            result = pipeline.process_frame(frame, frame_index, time())
            line = json.dumps(result, ensure_ascii=False)

            if args.print_every > 0 and frame_index % args.print_every == 0:
                print(line)
            if output_handle is not None:
                output_handle.write(line + "\n")

            if args.max_frames and frame_index >= args.max_frames:
                break
    finally:
        pipeline.reset()
        capture.release()
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


def _best_head_pose(
    track_bbox: Sequence[float],
    face: Optional[Any],
    candidates: Iterable[Any],
) -> Optional[Any]:
    match_bbox = getattr(face, "bbox", None) or track_bbox
    best = None
    best_score = 0.0
    for candidate in candidates:
        candidate_bbox = getattr(candidate, "face_bbox", None)
        if candidate_bbox is None:
            continue
        score = _iou(match_bbox, candidate_bbox)
        if score > best_score:
            best = candidate
            best_score = score
    return best if best_score > 0 else None


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0:
        return 0.0
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    return intersection / max(1e-6, area_a + area_b - intersection)


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
