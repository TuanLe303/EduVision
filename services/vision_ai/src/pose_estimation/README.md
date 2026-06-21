# Pose Estimation

YOLO pose wrapper for body keypoints.

## Usage

```python
from services.vision_ai.src.pose_estimation import PoseEstimator

estimator = PoseEstimator()
poses = estimator.estimate(frame)
```

## Output

```python
@dataclass
class PoseResult:
    bbox: List[float]
    confidence: float
    keypoints: List[Keypoint]
```

Keypoints use the 17-point COCO order.

## Config

Config files live in `configs/services/pose_estimation/`.

- `yolo11n-pose.yaml`
- `yolo11s-pose.yaml`

