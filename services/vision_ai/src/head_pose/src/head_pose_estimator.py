from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np


@dataclass
class HeadPoseResult:
    yaw: float
    pitch: float
    roll: float
    gaze_direction: str
    face_bbox: Optional[List[float]] = None
    confidence: float = 1.0


_CONFIGS_DIR = Path(__file__).resolve().parents[5] / "configs" / "services" / "head_pose"
_FACE_MESH_POINTS = [1, 152, 33, 263, 61, 291]

_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),          # nose tip
        (0.0, -63.6, -12.5),      # chin
        (-43.3, 32.7, -26.0),     # left eye outer corner
        (43.3, 32.7, -26.0),      # right eye outer corner
        (-28.9, -28.9, -24.1),    # left mouth corner
        (28.9, -28.9, -24.1),     # right mouth corner
    ],
    dtype=np.float64,
)


def _load_config() -> dict:
    import yaml

    path = _CONFIGS_DIR / "mediapipe.yaml"
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


class HeadPoseEstimator:
    """MediaPipe Face Mesh + OpenCV solvePnP head-pose estimator."""

    def __init__(
        self,
        yaw_threshold: Optional[float] = None,
        pitch_threshold: Optional[float] = None,
        max_num_faces: Optional[int] = None,
        min_detection_confidence: Optional[float] = None,
    ) -> None:
        cfg = _load_config()
        self._yaw_threshold = yaw_threshold if yaw_threshold is not None else cfg.get("yaw_threshold", 25.0)
        self._pitch_threshold = (
            pitch_threshold if pitch_threshold is not None else cfg.get("pitch_threshold", 20.0)
        )
        self._max_num_faces = max_num_faces if max_num_faces is not None else cfg.get("max_num_faces", 4)
        self._min_detection_confidence = (
            min_detection_confidence
            if min_detection_confidence is not None
            else cfg.get("min_detection_confidence", 0.5)
        )

        import mediapipe as mp

        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self._max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=self._min_detection_confidence,
        )

    def estimate(self, frame: np.ndarray) -> List[HeadPoseResult]:
        """
        Estimate head pose for faces in a BGR frame.

        Returns one HeadPoseResult per detected face mesh.
        """
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_results = self._face_mesh.process(rgb)
        if not mesh_results.multi_face_landmarks:
            return []

        height, width = frame.shape[:2]
        output: List[HeadPoseResult] = []
        for face_landmarks in mesh_results.multi_face_landmarks:
            points = [
                (
                    face_landmarks.landmark[index].x * width,
                    face_landmarks.landmark[index].y * height,
                )
                for index in _FACE_MESH_POINTS
            ]
            bbox = _face_bbox(face_landmarks.landmark, width, height)
            result = self.estimate_from_landmarks(points, frame.shape, face_bbox=bbox)
            if result is not None:
                output.append(result)
        return output

    def estimate_from_landmarks(
        self,
        landmarks: Sequence[Sequence[float]],
        frame_shape: Sequence[int],
        face_bbox: Optional[List[float]] = None,
    ) -> Optional[HeadPoseResult]:
        """Estimate pose from six 2D landmarks matching _MODEL_POINTS order."""
        if len(landmarks) < 6:
            return None

        import cv2

        image_points = np.asarray(landmarks[:6], dtype=np.float64)
        height, width = int(frame_shape[0]), int(frame_shape[1])
        focal_length = float(width)
        camera_matrix = np.array(
            [
                [focal_length, 0, width / 2],
                [0, focal_length, height / 2],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vector, _ = cv2.solvePnP(
            _MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pitch, yaw, roll = _rotation_matrix_to_euler(rotation_matrix)
        return HeadPoseResult(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            gaze_direction=self._classify_gaze(yaw, pitch),
            face_bbox=face_bbox,
        )

    def reset(self) -> None:
        """No temporal state to clear; provided for API consistency."""
        pass

    def _classify_gaze(self, yaw: float, pitch: float) -> str:
        if pitch > self._pitch_threshold:
            return "down"
        if pitch < -self._pitch_threshold:
            return "up"
        if yaw > self._yaw_threshold:
            return "right"
        if yaw < -self._yaw_threshold:
            return "left"
        return "center"


def _rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0.0

    pitch = math.degrees(x)
    yaw = math.degrees(y)
    roll = math.degrees(z)
    return pitch, yaw, roll


def _face_bbox(landmarks: Iterable, width: int, height: int) -> List[float]:
    xs = [point.x * width for point in landmarks]
    ys = [point.y * height for point in landmarks]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]

