# EduVision — Tools

Utility scripts for the EduVision data pipeline. All scripts use **YOLO26** (latest, NMS-free, backward-compatible annotation format).

## Scripts

| Script | Purpose | Usage |
|---|---|---|
| `extract_frames.py` | Extract frames from raw `.mp4` videos | `python tools/extract_frames.py --fps 2` |
| `augment_data.py` | Augment frames with transforms (flip, rotate, etc.) | `python tools/augment_data.py --multiplier 2` |
| `auto_annotate.py` | Auto-annotate frames using YOLO26 person detector | `python tools/auto_annotate.py --conf 0.35` |
| `merge_datasets.py` | Merge EduVision + Ambient Intelligence datasets | `python tools/merge_datasets.py` |
| `split_dataset.py` | Split merged dataset into train/val/test | `python tools/split_dataset.py --ratio 70 20 10` |
| `verify_dataset.py` | Verify dataset integrity + generate statistics | `python tools/verify_dataset.py --stage dataset` |
| `upload_to_hf.py` | Upload processed dataset to HuggingFace Hub | `python tools/upload_to_hf.py --token hf_...` |

## Full Pipeline

```bash
# Step 1: Extract frames from collected videos (2 fps, JPEG quality 85)
python tools/extract_frames.py --fps 2 --quality 85

# Step 2: Augment frames for diversity (2x multiplier = 3x total data)
python tools/augment_data.py --multiplier 2

# Step 3: Auto-annotate with YOLO26n person detector
python tools/auto_annotate.py --model yolo26n --conf 0.35

# Step 4: Merge with Ambient Intelligence Classroom dataset
python tools/merge_datasets.py

# Step 5: Split into train/val/test (stratified by source)
python tools/split_dataset.py --ratio 70 20 10 --seed 42

# Step 6: Verify dataset integrity
python tools/verify_dataset.py --stage dataset

# Step 7 (optional): Upload to HuggingFace
python tools/upload_to_hf.py --token hf_YOUR_TOKEN --repo annghoang/EduVision
```

## Requirements

```
ultralytics>=8.4.0    # YOLO26 support
opencv-python
albumentations        # Data augmentation
matplotlib            # Verification charts
tqdm
huggingface_hub
```

Install all at once:
```bash
pip install -r requirements.txt
```

## Behavior Classes

| ID | Class | Description |
|---|---|---|
| 0 | `person` | Generic person detection (wide shot) |
| 1 | `focused` | Student focused on lesson |
| 2 | `drowsy` | Student drowsy/sleepy |
| 3 | `sleeping` | Student sleeping |
| 4 | `using_phone` | Student using phone |
| 5 | `off_task` | Off-task behavior (eating, laptop, looking down) |
| 6 | `side_talking` | Side-talking with neighbor |
| 7 | `away_from_seat` | Away from assigned seat |
| 8 | `raising_hand` | Raising hand |

## Output Structure

```
data/
├── raw_frames/                  # Extracted + augmented frames
│   ├── wide_shot/               # 3 camera angles
│   │   ├── goc_cheo/            # frame_00001.jpg, frame_00001_aug1.jpg, ...
│   │   ├── goc_thang_phai/
│   │   └── goc_thang_trai/
│   ├── expression/              # 8 behaviors × 3 positions
│   │   ├── pos_1/{behavior}/
│   │   ├── pos_2/{behavior}/
│   │   └── pos_3/{behavior}/
│   └── augmentation_report.json
│
├── annotated/                   # YOLO-format annotations
│   ├── wide_shot/images + labels/
│   ├── expression/images + labels/
│   ├── dataset.yaml
│   └── annotation_report.json
│
├── merged/                      # Merged with external datasets
│   ├── images/
│   ├── labels/
│   ├── dataset.yaml
│   └── merge_report.json
│
├── dataset/                     # Train/val/test split (ready to train)
│   ├── train/images + labels/
│   ├── val/images + labels/
│   ├── test/images + labels/
│   └── dataset.yaml
│
└── verification/                # Quality reports
    ├── samples/                 # Annotated sample images
    ├── class_distribution.png
    ├── bbox_histogram.png
    └── verification_report.json
```

## Notes

- **Data is NOT stored in this repository** — all `data/` folders are gitignored
- Raw video files (`.mp4`) in `Data Collection/` are also gitignored
- All scripts support `--help` for full option listing
- YOLO26 is backward-compatible with YOLOv8/v11 annotation format
