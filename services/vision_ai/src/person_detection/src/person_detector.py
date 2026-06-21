from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class PersonDetection:
    bbox: List[float]  # [x1, y1, x2, y2] in pixel coordinates
    confidence: float
    class_id: int = 0  # COCO class 0 = person


_SUPPORTED_MODELS = ("yolo11n", "yolo11s")

# configs/services/person_detection/ relative to project root (EduVision/)
# __file__ is at services/vision_ai/src/person_detection/src/person_detector.py
# -> parents[5] = EduVision/
_CONFIGS_DIR = Path(__file__).resolve().parents[5] / "configs" / "services" / "person_detection"


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
    ) -> None:
        if model_name not in _SUPPORTED_MODELS:
            raise ValueError(
                f"model_name must be one of {list(_SUPPORTED_MODELS)}, got '{model_name}'"
            )

        cfg = _load_config(model_name)
        self._model_name = model_name
        self._conf = (
            confidence_threshold
            if confidence_threshold is not None
            else cfg.get("confidence_threshold", 0.4)
        )
        self._iou = iou_threshold if iou_threshold is not None else cfg.get("iou_threshold", 0.5)
        self._input_size = input_size if input_size is not None else cfg.get("input_size", 640)

        cfg_device = cfg.get("device", "auto")
        selected_device = device if device is not None else cfg_device
        self._device: Optional[str] = None if selected_device == "auto" else selected_device

        from ultralytics import YOLO

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

