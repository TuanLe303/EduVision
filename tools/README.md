# EduVision — Tools

Utility scripts for the EduVision data pipeline.

## Scripts

| Script | Purpose | Usage |
|---|---|---|
| `extract_frames.py` | Extract frames from raw `.MOV`/`.mp4` videos | `python tools/extract_frames.py --fps 2` |
| `auto_annotate.py` | Auto-annotate Wide Shot frames using YOLO11 | `python tools/auto_annotate.py --conf 0.35` |
| `split_dataset.py` | Split annotated dataset into train/val/test | `python tools/split_dataset.py --ratio 70 20 10` |
| `upload_to_hf.py` | Upload processed dataset to HuggingFace Hub | `python tools/upload_to_hf.py --token hf_...` |

## Full Pipeline

```bash
# Step 1: Extract frames from collected videos (2 fps, JPEG quality 85)
python tools/extract_frames.py --fps 2 --quality 85

# Step 2: Auto-annotate wide shot frames with YOLO11n person detector
python tools/auto_annotate.py --model yolo11n --conf 0.35

# Step 3: Split into train/val/test (stratified by camera angle)
python tools/split_dataset.py --ratio 70 20 10 --seed 42

# Step 4: Upload to HuggingFace (replace with your token)
python tools/upload_to_hf.py --token hf_YOUR_TOKEN --repo annghoang/EduVision
```

## Requirements

```
opencv-python
ultralytics
huggingface_hub
```

Install all at once:
```bash
pip install opencv-python ultralytics huggingface_hub
```

## Output Structure

```
data/
├── raw_frames/                  # Extracted frames
│   ├── wide_shot/               # 681 frames, 3 camera angles
│   │   ├── goc_cheo/
│   │   ├── goc_thang_phai/
│   │   └── goc_thang_trai/
│   ├── expression/              # 654 frames, 8 behaviors × 3 positions
│   │   ├── pos_1/{behavior}/
│   │   ├── pos_2/{behavior}/
│   │   └── pos_3/{behavior}/
│   └── extraction_report.json
│
├── annotated/                   # YOLO-format annotations
│   ├── wide_shot/
│   │   ├── images/              # 681 .jpg frames
│   │   └── labels/              # 681 .txt files (7,806 person boxes)
│   ├── expression/              # Classification-style labels
│   ├── dataset.yaml
│   └── annotation_report.json
│
└── dataset/                     # Train/val/test split (ready to train)
    ├── train/images + labels/   # 475 frames (70%)
    ├── val/images + labels/     # 135 frames (20%)
    ├── test/images + labels/    # 71 frames (10%)
    ├── dataset.yaml             # YOLO training config
    └── split_report.json
```

## Notes

- **Data is NOT stored in this repository** — see HuggingFace: `annghoang/EduVision`
- Raw video files (`.MOV`, `.mp4`) are excluded from both GitHub and HuggingFace
- All scripts support `--help` for full option listing
