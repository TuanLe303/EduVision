# YOLO Behavior Detection

The model performs one full-frame detection pass and never creates track IDs.
Behavior boxes are associated with canonical person tracks using maximum-weight
Hungarian matching over IoU, containment, and center distance.

Visual labels are `focused`, `drowsy`, `sleeping`, `using_phone`, `off_task`,
`side_talking`, and `raising_hand`. `away_from_seat` is handled by the fixed-camera
`SeatMonitor`, not by YOLO.

Train with `python tools/train_behavior.py` after preparing the YOLO detection
dataset configured in `configs/services/behavior_detection/dataset.yaml`.

Run the pipeline with:

```powershell
python -m services.vision_ai.src.main --behavior-model models/behavior_yolo.pt --source video.mp4
```

Temporal hyperparameters live in
`configs/services/behavior_detection/yolo_behavior.yaml`. Predictions remain
internal until the configured window has enough evidence; no warm-up label is
drawn on the frame. After a state is ready, a short missed-detection gap retains
that state with linearly decaying confidence. Retained results expose
`observed=false` and `detection_age`; they expire after `max_detection_gap`.

When `--behavior-window` overrides the configured window, `min_history` and
`min_state_frames` are scaled proportionally and clamped to the new window.
