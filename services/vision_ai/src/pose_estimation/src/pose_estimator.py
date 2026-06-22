from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, List, Optional

import numpy as np


@dataclass
class Keypoint:
    name: str
    x: float
    y: float
    confidence: float
    visible: bool


@dataclass
class PoseResult:
    bbox: List[float]
    confidence: float
    keypoints: List[Keypoint]


SUPPORTED_POSE_MODELS = ("yolo11n-pose", "yolo11s-pose")

COCO_KEYPOINT_NAMES = (
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
)

_CONFIGS_DIR = Path(__file__).resolve().parents[5] / "configs" / "services" / "pose_estimation"


@dataclass(frozen=True)
class PoseModelConfig:
    model: str
    confidence_threshold: float
    iou_threshold: float
    keypoint_threshold: float
    input_size: int
    device: str


def _threshold(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number in [0, 1]")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {result}")
    return result


def load_pose_config(model_name: str) -> PoseModelConfig:
    """Load and validate the pose model settings consumed by Tracker."""
    if model_name not in SUPPORTED_POSE_MODELS:
        raise ValueError(
            f"pose model must be one of {list(SUPPORTED_POSE_MODELS)}, got '{model_name}'"
        )
    import yaml

    path = _CONFIGS_DIR / f"{model_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Pose config not found: {path}")
    with path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Pose config must be a YAML mapping: {path}")

    model = raw.get("model", model_name)
    if not isinstance(model, str) or not model.strip():
        raise ValueError("pose config model must be a non-empty string")
    model = model.strip()
    weight = model if model.endswith(".pt") else f"{model}.pt"

    input_size = raw.get("input_size", 640)
    if isinstance(input_size, bool) or not isinstance(input_size, int):
        raise TypeError("pose input_size must be a positive integer")
    if input_size <= 0:
        raise ValueError("pose input_size must be greater than zero")

    device = raw.get("device", "auto")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("pose device must be a non-empty string")

    return PoseModelConfig(
        model=weight,
        confidence_threshold=_threshold(
            "pose confidence_threshold", raw.get("confidence_threshold", 0.1)
        ),
        iou_threshold=_threshold("pose iou_threshold", raw.get("iou_threshold", 0.5)),
        keypoint_threshold=_threshold(
            "keypoint_threshold", raw.get("keypoint_threshold", 0.3)
        ),
        input_size=input_size,
        device=device.strip(),
    )


def extract_pose_arrays(
    result: Any,
    expected_count: int,
    *,
    required: bool,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Validate Ultralytics keypoint tensors and convert them to NumPy."""
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        if required:
            raise RuntimeError("pose model did not return keypoints")
        return None, None

    xy = np.asarray(keypoints.xy.cpu().numpy())
    if xy.shape != (expected_count, len(COCO_KEYPOINT_NAMES), 2):
        raise RuntimeError(
            "pose coordinates must have shape "
            f"({expected_count}, {len(COCO_KEYPOINT_NAMES)}, 2), got {xy.shape}"
        )
    if not np.isfinite(xy).all():
        raise RuntimeError("pose coordinates contain NaN or infinity")

    confidence_tensor = getattr(keypoints, "conf", None)
    if confidence_tensor is None:
        raise RuntimeError("pose model did not return keypoint confidence scores")
    confidences = np.asarray(confidence_tensor.cpu().numpy())
    expected_shape = (expected_count, len(COCO_KEYPOINT_NAMES))
    if confidences.shape != expected_shape:
        raise RuntimeError(
            f"pose confidence must have shape {expected_shape}, got {confidences.shape}"
        )
    if not np.isfinite(confidences).all():
        raise RuntimeError("pose confidence contains NaN or infinity")
    if np.any((confidences < 0.0) | (confidences > 1.0)):
        raise RuntimeError("pose confidence must be in [0, 1]")
    return xy, confidences


def build_pose_result(
    bbox: np.ndarray,
    confidence: float,
    points: np.ndarray,
    point_confidences: np.ndarray,
    keypoint_threshold: float,
) -> PoseResult:
    """Build a stable 17-keypoint result without dropping low-score slots."""
    bbox = np.asarray(bbox).reshape(-1)
    if bbox.size < 4 or not np.isfinite(bbox[:4]).all():
        raise RuntimeError("pose bbox must contain four finite coordinates")
    if float(bbox[2]) <= float(bbox[0]) or float(bbox[3]) <= float(bbox[1]):
        raise RuntimeError("pose bbox must have positive width and height")
    confidence = float(confidence)
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise RuntimeError("pose box confidence must be in [0, 1]")
    threshold = _threshold("keypoint_threshold", keypoint_threshold)
    if points.shape != (len(COCO_KEYPOINT_NAMES), 2):
        raise RuntimeError("one pose must contain exactly 17 (x, y) keypoints")
    if point_confidences.shape != (len(COCO_KEYPOINT_NAMES),):
        raise RuntimeError("one pose must contain exactly 17 keypoint confidence scores")
    if not np.isfinite(points).all():
        raise RuntimeError("pose coordinates contain NaN or infinity")
    if not np.isfinite(point_confidences).all():
        raise RuntimeError("pose confidence contains NaN or infinity")
    if np.any((point_confidences < 0.0) | (point_confidences > 1.0)):
        raise RuntimeError("pose confidence must be in [0, 1]")

    return PoseResult(
        bbox=bbox[:4].astype(float).tolist(),
        confidence=confidence,
        keypoints=[
            Keypoint(
                name=name,
                x=float(points[index][0]),
                y=float(points[index][1]),
                confidence=float(point_confidences[index]),
                visible=float(point_confidences[index]) >= threshold,
            )
            for index, name in enumerate(COCO_KEYPOINT_NAMES)
        ],
    )
