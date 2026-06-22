# Face Detection

Wrapper around InsightFace SCRFD and RetinaFace for detecting faces in a video frame.

## Models

| Backend | Default model | Notes |
|---|---|---|
| `scrfd` | `scrfd_500m_bnkps` | Fast, lightweight; alternative: `scrfd_2.5g_bnkps`, `scrfd_10g_bnkps` |
| `retinaface` | `retinaface_r50_v1` | Higher accuracy; alternative: `retinaface_mnet025_v1` |

Models are downloaded automatically by InsightFace on first use to `~/.insightface/models/`.
The first startup therefore requires network access; later runs use the local cache.

## Installation

```bash
uv pip install -r requirements.txt            # CPU
uv pip uninstall onnxruntime
uv pip install onnxruntime-gpu                # GPU (use instead of onnxruntime)
```

## Input / Output

**Input:** BGR frame as `np.ndarray` (H × W × 3, uint8).

**Output:** `List[FaceDetection]`, sorted by descending confidence.

```python
@dataclass
class FaceDetection:
    bbox: List[float]             # [x1, y1, x2, y2] pixel coordinates
    confidence: float             # face detection score [0, 1]
    landmarks: List[List[float]]  # [[x, y], ...] 5 keypoints:
                                  #   left_eye, right_eye, nose, left_mouth, right_mouth
                                  #   empty list if the model does not return keypoints
```

## Config

Config files are at `configs/services/face_detection/`.

**scrfd.yaml**
```yaml
model: scrfd_500m_bnkps
input_size: [640, 640]
confidence_threshold: 0.5
device: cpu
```

**retinaface.yaml**
```yaml
model: retinaface_r50_v1
confidence_threshold: 0.5
device: cpu
```

## Usage

```python
from services.vision_ai.src.face_detection import FaceDetector

# Default: SCRFD, CPU
detector = FaceDetector()
faces = detector.detect(frame)

for face in faces:
    print(face.bbox, face.confidence, face.landmarks)

# Switch to RetinaFace
detector = FaceDetector(backend="retinaface")

# Higher accuracy SCRFD model
detector = FaceDetector(backend="scrfd", model="scrfd_10g_bnkps")

# GPU; requires onnxruntime-gpu instead of onnxruntime
detector = FaceDetector(device="cuda")

# Select a specific GPU
detector = FaceDetector(device="cuda:1")

# Override confidence threshold at runtime
detector = FaceDetector(confidence_threshold=0.7)
```

## Hardware recommendations

| Config | GPU VRAM | Notes |
|---|---|---|
| `scrfd_500m_bnkps` CPU | — | Suitable for dev/testing |
| `scrfd_500m_bnkps` GPU | ≥ 2 GB | Real-time on classroom video |
| `scrfd_10g_bnkps` GPU | ≥ 4 GB | Higher recall on distant/small faces |
| `retinaface_r50_v1` GPU | ≥ 4 GB | Best accuracy, slower |

## Integration

This module sits between **Tracking** and **Face Recognition** in the pipeline:

```
Tracker (TrackResult with bbox)
    ↓  crop person ROI
FaceDetector.detect(person_crop)
    ↓  FaceDetection with bbox + landmarks
Face Recognition
```
