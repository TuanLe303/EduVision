# EduVision E2E evaluation

This folder measures only the public end-to-end result. It does not score the
individual detector, tracker, recognizer, or temporal aggregator.

## Ground-truth JSONL

Use one object per evaluated frame. The ground-truth file may be sparse: only
frames present in this file are evaluated. A missing frame means **not
annotated**; an annotated empty frame must be written explicitly with
`"students": []` and `"person_count": 0`.

`students` must contain every visible student in an evaluated frame. `state`
may be omitted when only attendance/count is annotated. An optional absolute
`xyxy` `bbox` enables person-detection precision/recall/F1 at the configured
IoU threshold. Supply a bbox for every student or detection metrics are
disabled to avoid scoring partially annotated frames.

```json
{"frame_index": 25, "timestamp": 1.0, "person_count": 2, "students": [{"student_id": "S001", "state": "focused", "bbox": [10, 20, 110, 220]}, {"student_id": "S002", "state": "using_phone", "bbox": [130, 20, 230, 220]}]}
{"frame_index": 250, "timestamp": 10.0, "person_count": 0, "students": []}
```

The prediction file is the JSONL produced by `VisionPipeline`. Files are joined
by `frame_index`. All prediction records are read so identity history can be
retained, but accuracy is scored only at ground-truth frame indexes.

## Accuracy evaluation

```powershell
uv run python -m tools.evaluate.evaluate `
  --predictions outputs/session.jsonl `
  --ground-truth data/ground_truth/session.jsonl `
  --fps 25 `
  --bbox-iou 0.5 `
  --output outputs/evaluation.json
```

The output contains session attendance F1, per-frame student-identity F1,
student-behavior macro/per-class F1, a behavior confusion matrix, optional
person-detection metrics, student-count MAE, and `frame_results` with correct
IDs, false positives, false negatives, behavior errors, and count error for
every annotated frame.

Event F1 and behavior-duration MAE are emitted only when ground truth has at
least two consecutive frame indexes. Sparse snapshot labels cannot establish
event start/end times, so `events.available` is `false` for them. Annotate a
separate continuous clip when temporal event metrics are required.

## Sampled YOLO behavior labels

YOLO files containing `class_id cx cy width height` have anonymous behavior
boxes, not student identities. Convert them using the source video and the FPS
used when the labeled images were extracted:

```powershell
uv run python -m tools.evaluate.convert_yolo_ground_truth `
  --labels data/test/test4/label `
  --video data/test/test4/test4.mp4 `
  --sample-fps 2 `
  --output data/test/test4/test4_ground_truth.jsonl
```

This format enables only `bbox_behavior`. By default, converted labels are
treated as annotated boxes rather than a complete list of people, so unmatched
model predictions are ignored and person count/detection/identity metrics are
disabled. Evaluate with:

```powershell
uv run python -m tools.evaluate.evaluate `
  --predictions data/test/test4/test4_session_v2.jsonl `
  --ground-truth data/test/test4/test4_ground_truth.jsonl `
  --fps 29.970128322687405 `
  --bbox-iou 0.5 `
  --behavior-output final `
  --output outputs/test4_e2e_no_identity.json `
  --quiet
```

`--behavior-output final` evaluates the pipeline's temporally aggregated
`final_behavior` while using the raw behavior bbox for spatial matching. This
is the E2E behavior score without student identity. Use `--behavior-output
frame` only when evaluating the raw per-frame behavior detector itself.

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
