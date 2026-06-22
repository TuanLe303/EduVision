# Face Recognition

InsightFace-based face recognition and enrollment wrapper. It supports the
`buffalo_s` and `buffalo_l` model packs, prefers CUDA when available, and
stabilizes identities across frames using tracking IDs.

## Usage

```python
from services.vision_ai.src.face_recognition import FaceRecognizer, RegisteredFace

recognizer = FaceRecognizer(
    model_name="buffalo_s",
    enrollment_path="data/enrollments/students.json",
)
result = recognizer.recognize(frame, landmarks=face.landmarks, track_id=track_id)

if result.matched:
    print(result.student_id, result.name, result.similarity)
```

## Enrollment Format

```json
{
  "students": [
    {
      "student_id": "S001",
      "name": "Nguyen Van A",
      "embeddings": [
        [0.01, 0.02],
        [0.03, 0.04]
      ]
    }
  ]
}
```

Multiple embeddings for the same student are intentionally supported for
different viewing angles and lighting. Legacy records containing one
`embedding` are also accepted.

```python
recognizer.enroll(
    student_id="S001",
    name="Nguyen Van A",
    face_image=frame,
    landmarks=face.landmarks,
)
recognizer.save_enrollments("data/enrollments/students.json")
```

## Config

Config files live in `configs/services/face_recognition/`.

- Backend: `insightface`
- Models: `buffalo_s` (faster) and `buffalo_l` (more accurate)
- `device: auto` prefers CUDA and falls back to CPU

## Output

```python
@dataclass
class RecognitionResult:
    student_id: Optional[str]
    name: Optional[str]
    similarity: float  # cosine similarity, not a probability
    distance: float
    matched: bool
```

`result.confidence` remains as a compatibility alias of `similarity`.
