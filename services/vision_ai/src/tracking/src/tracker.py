from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class TrackResult:
    track_id: int
    bbox: List[float]   # [x1, y1, x2, y2] in pixel coordinates
    confidence: float


_SUPPORTED_TRACKERS = ("bytetrack", "botsort")

# configs/services/tracking/ relative to project root (EduVision/)
# __file__ is at services/vision_ai/src/tracking/src/tracker.py → parents[5] = EduVision/
_CONFIGS_DIR = Path(__file__).resolve().parents[5] / "configs" / "services" / "tracking"


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
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        device: str = "auto",
    ) -> None:
        if tracker not in _SUPPORTED_TRACKERS:
            raise ValueError(
                f"tracker must be one of {list(_SUPPORTED_TRACKERS)}, got '{tracker}'"
            )

        self._tracker_cfg = _resolve_config(tracker)
        self._conf = confidence_threshold
        self._iou = iou_threshold
        # Ultralytics accepts None to mean "auto-select"
        self._device: Optional[str] = None if device == "auto" else device
        from ultralytics import YOLO

        self._model = YOLO(f"{model_name}.pt")

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
        results = self._model.track(
            source=frame,
            persist=True,           # keep track state across calls
            tracker=self._tracker_cfg,
            conf=self._conf,
            iou=self._iou,
            classes=[0],            # COCO class 0 = person
            device=self._device,
            verbose=False,
        )

        tracks: List[TrackResult] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                # tracker has not yet assigned IDs (can happen on frame 1)
                continue

            ids   = boxes.id.int().cpu().numpy()
            bboxes = boxes.xyxy.cpu().numpy()
            confs  = boxes.conf.cpu().numpy()

            for track_id, bbox, conf in zip(ids, bboxes, confs):
                tracks.append(
                    TrackResult(
                        track_id=int(track_id),
                        bbox=bbox.tolist(),
                        confidence=float(conf),
                    )
                )

        return tracks

    def reset(self) -> None:
        """
        Clear all active tracks.

        Call this between sessions (e.g. when switching to a new video
        source) to prevent stale track IDs from carrying over.
        """
        self._model.predictor = None
