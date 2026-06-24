from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, List, Optional

import numpy as np


@dataclass
class PersonDetection:
    bbox: List[float]  # [x1, y1, x2, y2] in pixel coordinates
    confidence: float
    class_id: int = 0  # COCO class 0 = person


_SUPPORTED_MODELS = ("yolo11n", "yolo11s","yolo26n","yolo26s")

# configs/services/person_detection/ relative to project root (EduVision/)
# __file__ is at services/vision_ai/src/person_detection/src/person_detector.py
# -> parents[5] = EduVision/
_CONFIGS_DIR = Path(__file__).resolve().parents[5] / "configs" / "services" / "person_detection"


def _load_config(model_name: str) -> dict:
    import yaml

    path = _CONFIGS_DIR / f"{model_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Person detector config not found: {path}")
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Person detector config must be a YAML mapping: {path}")
    return config


def _weight_name(model_name: str, cfg: dict) -> str:
    model = cfg.get("model") or model_name
    if not isinstance(model, str) or not model.strip():
        raise ValueError("'model' in the person detector config must be a non-empty string")
    model = model.strip()
    return model if model.endswith(".pt") else f"{model}.pt"


def _validate_threshold(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number in [0, 1]")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _validate_input_size(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("input_size must be a positive integer")
    if value <= 0:
        raise ValueError(f"input_size must be positive, got {value}")
    return value


def _validate_frame(frame: np.ndarray) -> None:
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


class PersonDetector:
    """
    Wraps Ultralytics YOLO person detection.

    Detects COCO class 0 (person) in a single BGR frame and returns
    first-party PersonDetection records. Switch between yolo11n and yolo11s
    via the model_name argument.

    Detector parameters are loaded from:
        configs/services/person_detection/{model_name}.yaml

    Args:
        model_name:           YOLO variant: "yolo11n" (default) or "yolo11s".
        confidence_threshold: Minimum detection confidence to keep.
                              When None, the value in config is used.
        iou_threshold:        IoU threshold for NMS. When None, config is used.
        input_size:           Inference image size. When None, config is used.
        device:               "auto" lets Ultralytics pick GPU/CPU automatically.
                              Pass "cpu", "cuda:0", etc. to force a device.
    """

    def __init__(
        self,
        model_name: str = "yolo11n",
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        input_size: Optional[int] = None,
        device: Optional[str] = None,
        model: Optional[Any] = None,
    ) -> None:
        if model_name not in _SUPPORTED_MODELS:
            raise ValueError(
                f"model_name must be one of {list(_SUPPORTED_MODELS)}, got '{model_name}'"
            )

        cfg = _load_config(model_name)
        self._model_name = model_name
        confidence = (
            confidence_threshold
            if confidence_threshold is not None
            else cfg.get("confidence_threshold", 0.4)
        )
        iou = iou_threshold if iou_threshold is not None else cfg.get("iou_threshold", 0.5)
        image_size = input_size if input_size is not None else cfg.get("input_size", 640)
        self._conf = _validate_threshold("confidence_threshold", confidence)
        self._iou = _validate_threshold("iou_threshold", iou)
        self._input_size = _validate_input_size(image_size)

        cfg_device = cfg.get("device", "auto")
        selected_device = device if device is not None else cfg_device
        if not isinstance(selected_device, str) or not selected_device.strip():
            raise ValueError("device must be a non-empty string")
        selected_device = selected_device.strip()
        self._device: Optional[str] = None if selected_device == "auto" else selected_device

        if model is not None:
            if not callable(getattr(model, "predict", None)):
                raise TypeError("model must provide a callable predict() method")
            self._model = model
        else:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise ImportError(
                    "PersonDetector requires Ultralytics and its runtime dependencies. "
                    "Install the project dependencies with 'pip install -r requirements.txt'."
                ) from exc
            self._model = YOLO(_weight_name(model_name, cfg))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        """
        Detect persons in a single BGR frame.

        Args:
            frame: BGR image as a NumPy array (H x W x 3, uint8).

        Returns:
            List of PersonDetection sorted by descending confidence.
            Empty list when no persons are detected.
        """
        _validate_frame(frame)
        results = self._model.predict(
            source=frame,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._input_size,
            classes=[0],
            device=self._device,
            verbose=False,
        )

        detections: List[PersonDetection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            bboxes = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.int().cpu().numpy() if boxes.cls is not None else [0] * len(bboxes)

            for bbox, conf, class_id in zip(bboxes, confs, class_ids):
                if int(class_id) != 0:
                    continue
                detections.append(
                    PersonDetection(
                        bbox=bbox.tolist(),
                        confidence=float(conf),
                        class_id=0,
                    )
                )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def reset(self) -> None:
        """No persistent state to clear; provided for API consistency."""
        pass
