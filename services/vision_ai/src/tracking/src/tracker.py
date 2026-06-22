from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from services.vision_ai.src.pose_estimation import (
    SUPPORTED_POSE_MODELS,
    PoseResult,
    build_pose_result,
    extract_pose_arrays,
    load_pose_config,
)


@dataclass
class TrackResult:
    track_id: int
    bbox: List[float]   # [x1, y1, x2, y2] in pixel coordinates
    confidence: float


_SUPPORTED_TRACKERS = ("bytetrack", "botsort")

# configs/services/tracking/ relative to project root (EduVision/)
# __file__ is at services/vision_ai/src/tracking/src/tracker.py → parents[5] = EduVision/
_CONFIGS_DIR = Path(__file__).resolve().parents[5] / "configs" / "services" / "tracking"


def _validate_threshold(name: str, value: Any) -> float:
    """Validate a confidence/IoU threshold and return it as a float."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number in [0, 1]")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _validate_frame(frame: np.ndarray) -> None:
    """Fail early with a clear error when a frame is not a BGR uint8 image."""
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a NumPy array")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"frame must have shape (height, width, 3), got {frame.shape}"
        )
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("frame height and width must be greater than zero")
    if frame.dtype != np.uint8:
        raise TypeError(f"frame dtype must be uint8, got {frame.dtype}")


def _resolve_config(tracker: str) -> str:
    """Return the project config path for the given tracker name.

    Falls back to the Ultralytics built-in config name if the project
    config file does not exist yet.
    """
    project_cfg = _CONFIGS_DIR / f"{tracker}.yaml"
    if project_cfg.exists():
        return str(project_cfg)
    return f"{tracker}.yaml"  # Ultralytics built-in fallback


class Tracker:
    """
    Wraps Ultralytics ByteTrack / BoT-SORT tracking.

    Combines YOLO person detection and multi-object tracking in one pass
    so that each person in the frame receives a persistent track_id across
    frames.  Switch between trackers via the `tracker` argument — no other
    code changes required.

    Tracker parameters are loaded from:
        configs/services/tracking/{tracker}.yaml

    Args:
        model_name:           YOLO variant — "yolo11n" (default) or "yolo11s".
        tracker:              Tracking algorithm — "bytetrack" (default) or "botsort".
        confidence_threshold: Minimum detection confidence to keep.
        iou_threshold:        IoU threshold for NMS inside YOLO.
        device:               "auto" lets Ultralytics pick GPU/CPU automatically.
                              Pass "cpu", "cuda:0", etc. to force a specific device.
    """

    def __init__(
        self,
        model_name: str = "yolo11n",
        tracker: str = "bytetrack",
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        input_size: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        model_name = model_name.strip()

        if not isinstance(tracker, str):
            raise TypeError("tracker must be a string")
        if tracker not in _SUPPORTED_TRACKERS:
            raise ValueError(
                f"tracker must be one of {list(_SUPPORTED_TRACKERS)}, got '{tracker}'"
            )

        pose_config = (
            load_pose_config(model_name)
            if model_name in SUPPORTED_POSE_MODELS
            else None
        )
        self._expects_pose = pose_config is not None
        configured_confidence = (
            pose_config.confidence_threshold if pose_config is not None else 0.1
        )
        configured_iou = pose_config.iou_threshold if pose_config is not None else 0.5
        configured_input_size = pose_config.input_size if pose_config is not None else 640
        configured_device = pose_config.device if pose_config is not None else "auto"
        selected_device = device if device is not None else configured_device
        if not isinstance(selected_device, str) or not selected_device.strip():
            raise ValueError("device must be a non-empty string")
        selected_device = selected_device.strip()

        selected_input_size = input_size if input_size is not None else configured_input_size
        if isinstance(selected_input_size, bool) or not isinstance(selected_input_size, int):
            raise TypeError("input_size must be a positive integer")
        if selected_input_size <= 0:
            raise ValueError("input_size must be greater than zero")

        self._tracker_cfg = _resolve_config(tracker)
        self._conf = _validate_threshold(
            "confidence_threshold",
            confidence_threshold
            if confidence_threshold is not None
            else configured_confidence,
        )
        self._iou = _validate_threshold(
            "iou_threshold", iou_threshold if iou_threshold is not None else configured_iou
        )
        self._input_size = selected_input_size
        self._keypoint_threshold = (
            pose_config.keypoint_threshold if pose_config is not None else 0.0
        )
        # Ultralytics accepts None to mean "auto-select"
        self._device: Optional[str] = (
            None if selected_device == "auto" else selected_device
        )
        from ultralytics import YOLO

        weight = pose_config.model if pose_config is not None else f"{model_name}.pt"
        self._model = YOLO(weight)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray) -> List[TrackResult]:
        """
        Run detection + tracking on a single BGR frame.

        Args:
            frame: BGR image as a NumPy array (H × W × 3, uint8).

        Returns:
            List of TrackResult.  Empty list when no persons are tracked
            or when the tracker has not yet assigned IDs (e.g. first frame).
        """
        tracks, _ = self.update_with_poses(frame)
        return tracks

    def update_with_poses(
        self, frame: np.ndarray
    ) -> tuple[List[TrackResult], dict[int, PoseResult]]:
        """Track people and return pose keypoints from the same inference pass.

        The pose mapping is empty when the configured YOLO model does not
        produce keypoints. This keeps ``update()`` backward compatible while
        allowing the end-to-end pipeline to avoid a second pose inference.
        """
        _validate_frame(frame)

        results = self._model.track(
            source=frame,
            persist=True,           # keep track state across calls
            tracker=self._tracker_cfg,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._input_size,
            classes=[0],            # COCO class 0 = person
            device=self._device,
            verbose=False,
        )

        tracks: List[TrackResult] = []
        poses_by_track: dict[int, PoseResult] = {}

        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                # tracker has not yet assigned IDs (can happen on frame 1)
                continue

            ids   = boxes.id.int().cpu().numpy()
            bboxes = boxes.xyxy.cpu().numpy()
            confs  = boxes.conf.cpu().numpy()
            if not (len(ids) == len(bboxes) == len(confs)):
                raise RuntimeError(
                    "tracking result returned different numbers of IDs, boxes, and scores"
                )
            pose_points, pose_confidences = extract_pose_arrays(
                result,
                len(ids),
                required=self._expects_pose,
            )

            for index, (track_id, bbox, conf) in enumerate(zip(ids, bboxes, confs)):
                normalized_track_id = int(track_id)
                tracks.append(
                    TrackResult(
                        track_id=normalized_track_id,
                        bbox=bbox.tolist(),
                        confidence=float(conf),
                    )
                )

                if pose_points is not None and pose_confidences is not None:
                    poses_by_track[normalized_track_id] = build_pose_result(
                        bbox,
                        conf,
                        pose_points[index],
                        pose_confidences[index],
                        self._keypoint_threshold,
                    )

        return tracks, poses_by_track

    def reset(self) -> None:
        """
        Clear all active tracks.

        Call this between sessions (e.g. when switching to a new video
        source) to prevent stale track IDs from carrying over.
        """
        self._model.predictor = None
