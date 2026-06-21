from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class FaceDetection:
    bbox: List[float]             # [x1, y1, x2, y2] in pixel coordinates
    confidence: float
    landmarks: List[List[float]]  # [[x, y], ...] 5 keypoints; empty list if unavailable


_SUPPORTED_BACKENDS = ("scrfd", "retinaface")

# configs/services/face_detection/ relative to project root (EduVision/)
# __file__ is at services/vision_ai/src/face_detection/src/face_detector.py → parents[5] = EduVision/
_CONFIGS_DIR = Path(__file__).resolve().parents[5] / "configs" / "services" / "face_detection"

_DEFAULT_MODELS = {
    "scrfd": "scrfd_500m_bnkps",
    "retinaface": "retinaface_r50_v1",
}


def _load_config(backend: str) -> dict:
    import yaml

    path = _CONFIGS_DIR / f"{backend}.yaml"
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _ctx_id(device: str) -> int:
    return -1 if device == "cpu" else 0


class FaceDetector:
    """
    Wraps InsightFace SCRFD / RetinaFace face detection.

    Detects faces in a frame and returns bounding boxes, confidence scores,
    and 5-point facial landmarks for each detected face.  Switch between
    SCRFD and RetinaFace via the `backend` argument — no other code changes
    required.

    Detector parameters are loaded from:
        configs/services/face_detection/{backend}.yaml

    Args:
        backend:              Detection model — "scrfd" (default) or "retinaface".
        model:                Override the model name from config
                              (e.g. "scrfd_10g_bnkps" for higher accuracy).
                              When None, the value in the config file is used.
        confidence_threshold: Minimum face confidence to keep.
                              When None, the value in the config file is used.
        device:               "cpu" (default) or "cuda".
    """

    def __init__(
        self,
        backend: str = "scrfd",
        model: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        device: str = "cpu",
    ) -> None:
        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"backend must be one of {list(_SUPPORTED_BACKENDS)}, got '{backend}'"
            )

        cfg = _load_config(backend)
        self._backend = backend
        self._conf = (
            confidence_threshold
            if confidence_threshold is not None
            else cfg.get("confidence_threshold", 0.5)
        )
        self._device = device

        model_name = model or cfg.get("model") or _DEFAULT_MODELS[backend]
        self._model = self._build_model(backend, model_name, cfg)

    def _build_model(self, backend: str, model_name: str, cfg: dict):
        from insightface.model_zoo import get_model

        ctx = _ctx_id(self._device)
        detector = get_model(model_name)

        if backend == "scrfd":
            input_size = tuple(cfg.get("input_size", [640, 640]))
            detector.prepare(ctx_id=ctx, input_size=input_size)
        else:
            detector.prepare(ctx_id=ctx)

        return detector

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in a single BGR frame.

        Args:
            frame: BGR image as a NumPy array (H × W × 3, uint8).

        Returns:
            List of FaceDetection sorted by descending confidence.
            Empty list when no faces are detected.
        """
        bboxes, kpss = self._model.detect(frame, thresh=self._conf)

        if bboxes is None or len(bboxes) == 0:
            return []

        results: List[FaceDetection] = []
        for i, raw in enumerate(bboxes):
            x1, y1, x2, y2, conf = raw.tolist()
            landmarks = kpss[i].tolist() if kpss is not None else []
            results.append(
                FaceDetection(
                    bbox=[x1, y1, x2, y2],
                    confidence=conf,
                    landmarks=landmarks,
                )
            )

        results.sort(key=lambda d: d.confidence, reverse=True)
        return results

    def reset(self) -> None:
        """No persistent state to clear — provided for API consistency with other modules."""
        pass
