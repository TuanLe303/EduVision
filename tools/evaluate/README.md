# EduVision E2E evaluation

This folder measures only the public end-to-end result. It does not score the
individual detector, tracker, recognizer, or temporal aggregator.

## Ground-truth JSONL

Use one object per evaluated frame. `students` contains every visible student;
`state` may be omitted when only attendance/count is annotated.

```json
{"frame_index": 1, "timestamp": 0.0, "person_count": 2, "students": [{"student_id": "S001", "state": "focused"}, {"student_id": "S002", "state": "using_phone"}]}
{"frame_index": 2, "timestamp": 0.04, "person_count": 2, "students": [{"student_id": "S001", "state": "focused"}, {"student_id": "S002", "state": "using_phone"}]}
```

The prediction file is the JSONL produced by `VisionPipeline`. Files are joined
by `frame_index`. Prediction identity is recovered from matched recognition and
retained for later frames on the same track.

## Accuracy evaluation

```powershell
uv run python -m tools.evaluate.evaluate `
  --predictions outputs/session.jsonl `
  --ground-truth data/ground_truth/session.jsonl `
  --fps 25 `
  --output outputs/evaluation.json
```

The output contains attendance F1, student-behavior macro F1 and per-class F1,
event F1 at temporal IoU 0.5, behavior-duration MAE, and student-count MAE.

`--case best|normal|worst` currently adds a label only. Case definitions and
acceptance thresholds are intentionally not implemented yet.

## Performance benchmark

The benchmark accepts all regular pipeline arguments and adds three options:
`--metrics-output`, `--warmup-frames`, and `--max-frames`.

```powershell
uv run python -m tools.evaluate.benchmark `
  --source classroom.mp4 `
  --behavior-model weights/behavior_yolo26n.pt `
  --output-jsonl outputs/session.jsonl `
  --metrics-output outputs/performance.json `
  --warmup-frames 5
```

It reports effective FPS, real-time factor, mean/P95 frame latency, peak VRAM,
and peak RAM when the optional `psutil` package is installed. Combine accuracy
and performance by passing `--performance outputs/performance.json` to the
accuracy evaluator.
