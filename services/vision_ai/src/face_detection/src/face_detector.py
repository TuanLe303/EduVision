from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, List, Optional

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
        raise FileNotFoundError(f"Face detector config not found: {path}")
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Face detector config must be a YAML mapping: {path}")
    return config


def _ctx_id(device: str) -> int:
    if device == "cpu":
        return -1
    if device == "cuda":
        return 0
    return int(device.split(":", 1)[1])


def _validate_device(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<index>'")
    device = value.strip().lower()
    if device in {"cpu", "cuda"}:
        return device
    if device.startswith("cuda:") and device[5:].isdigit():
        return device
    raise ValueError(
        f"device must be 'cpu', 'cuda', or 'cuda:<index>', got '{value}'"
    )


def _validate_threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("confidence_threshold must be a number in [0, 1]")
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"confidence_threshold must be in [0, 1], got {threshold}"
        )
    return threshold


def _validate_input_size(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("input_size must contain exactly [width, height]")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise TypeError("input_size width and height must be integers")
    width, height = value
    if width <= 0 or height <= 0:
        raise ValueError("input_size width and height must be greater than zero")
    return width, height


def _validate_model_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("model must be a non-empty string")
    return value.strip()


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
        device:               "cpu", "cuda", or "cuda:<index>". When None,
                              the value in the config file is used.
    """

    def __init__(
        self,
        backend: str = "scrfd",
        model: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        device: Optional[str] = None,
    ) -> None:
        if not isinstance(backend, str):
            raise TypeError("backend must be a string")
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
        self._conf = _validate_threshold(self._conf)
        selected_device = device if device is not None else cfg.get("device", "cpu")
        self._device = _validate_device(selected_device)

        model_name = _validate_model_name(
            model if model is not None else cfg.get("model", _DEFAULT_MODELS[backend])
        )
        self._model = self._build_model(backend, model_name, cfg)

    def _build_model(self, backend: str, model_name: str, cfg: dict):
        from insightface.model_zoo import get_model

        if self._device != "cpu":
            import onnxruntime

            if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
                raise RuntimeError(
                    "CUDA face detection requires onnxruntime-gpu with "
                    "CUDAExecutionProvider available"
                )

        providers = (
            ["CPUExecutionProvider"]
            if self._device == "cpu"
            else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        try:
            detector = get_model(model_name, download=True, providers=providers)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load face detector model '{model_name}'"
            ) from exc
        if detector is None:
            raise RuntimeError(
                f"InsightFace could not find or download face detector model '{model_name}'"
            )

        ctx = _ctx_id(self._device)
        if backend == "scrfd":
            input_size = _validate_input_size(cfg.get("input_size", [640, 640]))
            detector.prepare(
                ctx_id=ctx,
                input_size=input_size,
                det_thresh=self._conf,
            )
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
        _validate_frame(frame)
        if self._backend == "scrfd":
            bboxes, kpss = self._model.detect(frame)
        else:
            bboxes, kpss = self._model.detect(frame, threshold=self._conf)

        if bboxes is None or len(bboxes) == 0:
            return []
        if kpss is not None and len(kpss) != len(bboxes):
            raise RuntimeError(
                "Face detector returned different numbers of boxes and landmarks"
            )

        results: List[FaceDetection] = []
        for i, raw in enumerate(bboxes):
            values = np.asarray(raw).reshape(-1)
            if values.size < 5:
                raise RuntimeError(
                    f"Face detector bbox must contain at least 5 values, got {values.size}"
                )
            x1, y1, x2, y2, conf = (float(value) for value in values[:5])
            if not np.isfinite([x1, y1, x2, y2, conf]).all():
                raise RuntimeError("Face detector returned non-finite bbox values")
            if x2 <= x1 or y2 <= y1:
                continue

            landmarks: List[List[float]] = []
            if kpss is not None:
                landmark_array = np.asarray(kpss[i], dtype=np.float32)
                if landmark_array.ndim != 2 or landmark_array.shape[1] != 2:
                    raise RuntimeError(
                        "Face detector landmarks must have shape (count, 2)"
                    )
                if not np.isfinite(landmark_array).all():
                    raise RuntimeError("Face detector returned non-finite landmarks")
                landmarks = landmark_array.tolist()
            results.append(
                FaceDetection(
                    bbox=[x1, y1, x2, y2],
                    confidence=float(conf),
                    landmarks=landmarks,
                )
            )

        results.sort(key=lambda d: d.confidence, reverse=True)
        return results

    def reset(self) -> None:
        """No persistent state to clear — provided for API consistency with other modules."""
        pass
