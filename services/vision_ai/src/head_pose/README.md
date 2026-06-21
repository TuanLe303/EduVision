# Head Pose / Gaze

MediaPipe Face Mesh plus OpenCV `solvePnP` wrapper.

## Usage

```python
from services.vision_ai.src.head_pose import HeadPoseEstimator

estimator = HeadPoseEstimator()
poses = estimator.estimate(frame)
```

## Output

```python
@dataclass
class HeadPoseResult:
    yaw: float
    pitch: float
    roll: float
    gaze_direction: str
    face_bbox: Optional[List[float]]
    confidence: float
```

`gaze_direction` is one of `center`, `left`, `right`, `down`, or `up`.

