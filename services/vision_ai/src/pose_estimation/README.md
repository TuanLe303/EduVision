# Pose Estimation

Pose schemas, validated YOLO-pose output parsing, and model configuration for
the realtime tracking pipeline. Inference is intentionally owned by `Tracker`,
so detection, track IDs, bounding boxes, and keypoints come from one
`model.track()` call.

## Output

```python
@dataclass
class PoseResult:
    bbox: List[float]
    confidence: float
    keypoints: List[Keypoint]
```

Every result contains exactly 17 keypoints in COCO order. Low-confidence
keypoints remain in their stable slots and have `visible=False`; the threshold
comes from `keypoint_threshold` in the selected pose config.

## Config

Config files live in `configs/services/pose_estimation/`.

- `yolo11n-pose.yaml`
- `yolo11s-pose.yaml`

These settings are loaded by `Tracker` whenever a supported pose model is
selected. Runtime constructor/CLI values override config values only when they
are explicitly supplied.
