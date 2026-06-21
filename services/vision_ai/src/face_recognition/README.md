# Face Recognition

InsightFace-based face recognition and attendance wrapper.

## Usage

```python
from services.vision_ai.src.face_recognition import FaceRecognizer, RegisteredFace

recognizer = FaceRecognizer(enrollment_path="data/enrollments/students.json")
result = recognizer.recognize(face_crop)

if result.matched:
    print(result.student_id, result.name, result.confidence)
```

## Enrollment Format

```json
{
  "students": [
    {
      "student_id": "S001",
      "name": "Nguyen Van A",
      "embedding": [0.01, 0.02]
    }
  ]
}
```

`FaceRecognizer.add_enrollment(...)` can also be used at runtime.

## Config

Config files live in `configs/services/face_recognition/`.

- `insightface.yaml`: default `buffalo_s`
- `arcface.yaml`: aliases the same InsightFace embedding path with `buffalo_l`

## Output

```python
@dataclass
class RecognitionResult:
    student_id: Optional[str]
    name: Optional[str]
    confidence: float
    distance: float
    matched: bool
```
