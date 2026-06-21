from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class Keypoint:
    name: str
    x: float
    y: float
    confidence: float


@dataclass
class PoseResult:
    bbox: List[float]
    confidence: float
    keypoints: List[Keypoint]


_SUPPORTED_MODELS = ("yolo11n-pose", "yolo11s-pose")

_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

_CONFIGS_DIR = Path(__file__).resolve().parents[5] / "configs" / "services" / "pose_estimation"


def _load_config(model_name: str) -> dict:
    import yaml

    path = _CONFIGS_DIR / f"{model_name}.yaml"
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _weight_name(model_name: str, cfg: dict) -> str:
    model = cfg.get("model") or model_name
    return model if str(model).endswith(".pt") else f"{model}.pt"


class PoseEstimator:
    """YOLO pose-estimation wrapper returning first-party pose records."""

    def __init__(
        self,
        model_name: str = "yolo11n-pose",
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        input_size: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        if model_name not in _SUPPORTED_MODELS:
            raise ValueError(
                f"model_name must be one of {list(_SUPPORTED_MODELS)}, got '{model_name}'"
            )

        cfg = _load_config(model_name)
        self._conf = (
            confidence_threshold
            if confidence_threshold is not None
            else cfg.get("confidence_threshold", 0.35)
        )
        self._iou = iou_threshold if iou_threshold is not None else cfg.get("iou_threshold", 0.5)
        self._input_size = input_size if input_size is not None else cfg.get("input_size", 640)

        selected_device = device if device is not None else cfg.get("device", "auto")
        self._device: Optional[str] = None if selected_device == "auto" else selected_device

        from ultralytics import YOLO

        self._model = YOLO(_weight_name(model_name, cfg))

    def estimate(self, frame: np.ndarray) -> List[PoseResult]:
        """
        Estimate body keypoints for persons in a BGR frame.

        Returns an empty list when no pose result is available.
        """
        results = self._model.predict(
            source=frame,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._input_size,
            device=self._device,
            verbose=False,
        )

        poses: List[PoseResult] = []
        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints
            if boxes is None or keypoints is None:
                continue

            bboxes = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            xy = keypoints.xy.cpu().numpy()
            kp_conf = _keypoint_confidences(keypoints, xy)

            for bbox, conf, points, point_conf in zip(bboxes, confs, xy, kp_conf):
                poses.append(
                    PoseResult(
                        bbox=bbox.tolist(),
                        confidence=float(conf),
                        keypoints=[
                            Keypoint(
                                name=_KEYPOINT_NAMES[index],
                                x=float(point[0]),
                                y=float(point[1]),
                                confidence=float(point_conf[index]),
                            )
                            for index, point in enumerate(points[: len(_KEYPOINT_NAMES)])
                        ],
                    )
                )

        poses.sort(key=lambda item: item.confidence, reverse=True)
        return poses

    def reset(self) -> None:
        """No persistent state to clear; provided for API consistency."""
        pass


def _keypoint_confidences(keypoints, xy: np.ndarray) -> np.ndarray:
    conf = getattr(keypoints, "conf", None)
    if conf is None:
        return np.ones((xy.shape[0], xy.shape[1]), dtype=np.float32)
    return conf.cpu().numpy()

