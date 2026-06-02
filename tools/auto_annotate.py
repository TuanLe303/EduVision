"""
EduVision -- Auto-Annotation Tool
====================================
Uses a pretrained YOLO11 model to automatically detect persons in Wide Shot
frames and generates YOLO-format annotation (.txt) files.

The output is a Roboflow/YOLO-compatible dataset structure that can be:
  1. Directly used to train a YOLO person-detector
  2. Uploaded to Roboflow for manual review / label refinement
  3. Extended with behavior labels after human review

Output structure:
    data/
    └── annotated/
        ├── wide_shot/
        │   ├── images/        <- frame .jpg files (copied)
        │   └── labels/        <- YOLO .txt files  (auto-generated)
        ├── dataset.yaml       <- YOLO dataset config
        └── annotation_report.json

YOLO label format (one line per detected object):
    <class_id> <cx> <cy> <w> <h>   (all normalized 0-1)

Usage:
    python tools/auto_annotate.py                     # default settings
    python tools/auto_annotate.py --conf 0.4          # stricter confidence
    python tools/auto_annotate.py --model yolo11s     # larger model
    python tools/auto_annotate.py --preview           # show sample frames
    python tools/auto_annotate.py --help
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics is not installed.")
    print("  Run:  pip install ultralytics")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("[ERROR] opencv-python is not installed.")
    print("  Run:  pip install opencv-python")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
FRAMES_ROOT   = PROJECT_ROOT / "data" / "raw_frames"
OUTPUT_ROOT   = PROJECT_ROOT / "data" / "annotated"

WIDE_SHOT_DIR = FRAMES_ROOT / "wide_shot"
EXPR_DIR      = FRAMES_ROOT / "expression"

# COCO class id for "person" = 0
PERSON_CLASS_ID = 0

# Behavior classes for the dataset YAML
BEHAVIOR_CLASSES = [
    "person",          # 0 — generic person (wide shot)
    "focused",         # 1
    "drowsy",          # 2
    "sleeping",        # 3
    "using_phone",     # 4
    "off_task",        # 5
    "side_talking",    # 6
    "away_from_seat",  # 7
    "raising_hand",    # 8
]

BEHAVIOR_CLASS_ID = {name: i for i, name in enumerate(BEHAVIOR_CLASSES)}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float,
                 img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert absolute xyxy bbox to YOLO normalized cx,cy,w,h."""
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return cx, cy, w, h


def draw_preview(image_path: Path, label_path: Path, class_names: list[str]) -> None:
    """Display an annotated preview frame (press any key to close)."""
    img = cv2.imread(str(image_path))
    if img is None:
        return
    h, w = img.shape[:2]
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow(f"Preview: {image_path.name}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# Annotation functions
# ─────────────────────────────────────────────────────────────────────────────

def annotate_wide_shot(model: YOLO, conf: float, preview: bool) -> dict:
    """
    Run person detection on all Wide Shot frames.
    Copies images to annotated/wide_shot/images/ and writes YOLO labels.
    Class = 0 (person) for all detections.
    """
    images_out = OUTPUT_ROOT / "wide_shot" / "images"
    labels_out = OUTPUT_ROOT / "wide_shot" / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    stats = {"processed": 0, "total_detections": 0, "skipped": 0, "angles": {}}

    for angle_dir in sorted(WIDE_SHOT_DIR.iterdir()):
        if not angle_dir.is_dir():
            continue
        angle_name  = angle_dir.name
        angle_stats = {"frames": 0, "detections": 0}

        print(f"\n  [ANGLE] {angle_name}")
        frame_files = sorted(angle_dir.glob("*.jpg"))

        for img_path in frame_files:
            # Copy image
            dest_img = images_out / f"{angle_name}__{img_path.name}"
            shutil.copy2(img_path, dest_img)

            # Read dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                stats["skipped"] += 1
                continue
            img_h, img_w = img.shape[:2]

            # Run inference — filter class=person only
            results = model.predict(
                source=str(img_path),
                conf=conf,
                classes=[PERSON_CLASS_ID],
                verbose=False,
            )

            label_lines = []
            n_det = 0
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
                    # Clamp to [0, 1]
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    bw = max(0.001, min(1.0, bw))
                    bh = max(0.001, min(1.0, bh))
                    label_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    n_det += 1

            # Write label file
            dest_label = labels_out / f"{angle_name}__{img_path.stem}.txt"
            with open(dest_label, "w") as f:
                f.write("\n".join(label_lines))

            angle_stats["frames"]     += 1
            angle_stats["detections"] += n_det
            stats["processed"]        += 1
            stats["total_detections"] += n_det

        print(f"      {angle_stats['frames']} frames, "
              f"{angle_stats['detections']} person detections")
        stats["angles"][angle_name] = angle_stats

        # Preview one frame per angle
        if preview:
            sample = next(iter(frame_files), None)
            if sample:
                dest_img   = images_out / f"{angle_name}__{sample.name}"
                dest_label = labels_out / f"{angle_name}__{sample.stem}.txt"
                draw_preview(dest_img, dest_label, BEHAVIOR_CLASSES)

    return stats


def annotate_expression(preview: bool) -> dict:
    """
    For expression frames, the label IS the folder name.
    Copies images and writes YOLO labels using a full-frame bbox
    (class = behavior class, box covers the whole image = 0.5 0.5 1.0 1.0).

    NOTE: These are 'classification-style' labels. If you want precise
    per-person bounding boxes, run the wide_shot detector on these too
    (add --expr-detect flag) or refine on Roboflow.
    """
    stats = {"processed": 0, "behaviors": {}}

    for pos_dir in sorted(EXPR_DIR.iterdir()):
        if not pos_dir.is_dir():
            continue
        pos_name = pos_dir.name
        print(f"\n  [POS] {pos_name}")

        for behavior_dir in sorted(pos_dir.iterdir()):
            if not behavior_dir.is_dir():
                continue
            behavior = behavior_dir.name
            cls_id   = BEHAVIOR_CLASS_ID.get(behavior, 0)

            images_out = OUTPUT_ROOT / "expression" / pos_name / behavior / "images"
            labels_out = OUTPUT_ROOT / "expression" / pos_name / behavior / "labels"
            images_out.mkdir(parents=True, exist_ok=True)
            labels_out.mkdir(parents=True, exist_ok=True)

            frame_files = sorted(behavior_dir.glob("*.jpg"))
            n = 0
            for img_path in frame_files:
                shutil.copy2(img_path, images_out / img_path.name)
                label_path = labels_out / f"{img_path.stem}.txt"
                # Full-frame bbox (whole image = one subject performing behavior)
                with open(label_path, "w") as f:
                    f.write(f"{cls_id} 0.500000 0.500000 1.000000 1.000000\n")
                n += 1

            stats["processed"] += n
            stats["behaviors"][f"{pos_name}/{behavior}"] = n
            print(f"    {behavior:20s}  {n:3d} frames  (class_id={cls_id})")

            if preview and frame_files:
                sample     = frame_files[0]
                dest_img   = images_out / sample.name
                dest_label = labels_out / f"{sample.stem}.txt"
                draw_preview(dest_img, dest_label, BEHAVIOR_CLASSES)

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Dataset YAML
# ─────────────────────────────────────────────────────────────────────────────

def write_dataset_yaml() -> None:
    """Write a YOLO dataset.yaml pointing to the annotated wide_shot split."""
    yaml_path = OUTPUT_ROOT / "dataset.yaml"
    content = f"""# EduVision -- YOLO Dataset Configuration
# Generated by tools/auto_annotate.py
#
# After splitting into train/val/test, update the paths below.
# Use tools/split_dataset.py (coming soon) to create the split.

path: {OUTPUT_ROOT.as_posix()}  # dataset root

train: wide_shot/images   # relative to 'path'
val:   wide_shot/images   # update after split
test:  wide_shot/images   # update after split

# Class definitions
nc: {len(BEHAVIOR_CLASSES)}
names:
"""
    for i, name in enumerate(BEHAVIOR_CLASSES):
        content += f"  {i}: {name}\n"

    content += """
# Notes:
#   class 0 = 'person'  (wide-shot auto-detected bounding boxes)
#   classes 1-8 = behavior labels (expression data, full-frame boxes)
#   After Roboflow review, merge and re-export with precise per-person boxes.
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n  [DONE] dataset.yaml -> {yaml_path.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(wide_stats: dict, expr_stats: dict, model_name: str) -> None:
    print("\n" + "-" * 60)
    print("  AUTO-ANNOTATION SUMMARY")
    print("-" * 60)
    print(f"  Model used        : {model_name}")
    print(f"  Wide shot frames  : {wide_stats.get('processed', 0)}")
    print(f"  Person detections : {wide_stats.get('total_detections', 0)}")
    print(f"  Expr frames       : {expr_stats.get('processed', 0)}")
    avg = (wide_stats.get('total_detections', 0) /
           max(1, wide_stats.get('processed', 1)))
    print(f"  Avg persons/frame : {avg:.1f}")
    print(f"  Output            : {OUTPUT_ROOT.relative_to(PROJECT_ROOT)}")
    print("-" * 60)
    print()
    print("  NEXT STEPS:")
    print("  1. Upload data/annotated/wide_shot/ to Roboflow for review")
    print("     -> Fix missed detections & add behavior labels")
    print("  2. Run: python tools/split_dataset.py --ratio 70 20 10")
    print("  3. Train: yolo train data=data/annotated/dataset.yaml model=yolo11n.pt")
    print("-" * 60)


def save_report(wide_stats: dict, expr_stats: dict, model_name: str,
                conf: float) -> None:
    report = {
        "model": model_name,
        "confidence_threshold": conf,
        "wide_shot": wide_stats,
        "expression": expr_stats,
        "behavior_classes": BEHAVIOR_CLASSES,
    }
    report_path = OUTPUT_ROOT / "annotation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  [DONE] Report -> {report_path.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EduVision -- Auto-annotate frames using pretrained YOLO11.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", default="yolo11n",
        choices=["yolo11n", "yolo11s", "yolo11m", "yolo11l"],
        help="YOLO11 model size (default: yolo11n — fastest)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.35,
        help="Confidence threshold for person detection (default: 0.35)"
    )
    parser.add_argument(
        "--wide-only", action="store_true",
        help="Only annotate Wide Shot frames"
    )
    parser.add_argument(
        "--expr-only", action="store_true",
        help="Only annotate Expression frames"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Show preview of annotated frames (requires display)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("-" * 60)
    print("  EduVision -- Auto-Annotation Tool")
    print("-" * 60)
    print(f"  Model   : {args.model}.pt")
    print(f"  Conf    : {args.conf}")
    print(f"  Output  : {OUTPUT_ROOT.relative_to(PROJECT_ROOT)}")
    print()

    if not FRAMES_ROOT.exists():
        print(f"[ERROR] Frames not found at: {FRAMES_ROOT}")
        print("  Run tools/extract_frames.py first.")
        sys.exit(1)

    # Load model (downloads automatically on first run)
    print(f"  Loading {args.model}.pt ...")
    model = YOLO(f"{args.model}.pt")
    print(f"  Model loaded OK\n")

    t0 = time.time()
    wide_stats = {}
    expr_stats = {}

    if not args.expr_only:
        print("-- Wide Shot Auto-Detection -----------------------------------")
        wide_stats = annotate_wide_shot(model, args.conf, args.preview)

    if not args.wide_only:
        print("\n-- Expression Data Label Copy --------------------------------")
        expr_stats = annotate_expression(args.preview)

    write_dataset_yaml()
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    print_summary(wide_stats, expr_stats, args.model)
    save_report(wide_stats, expr_stats, args.model, args.conf)


if __name__ == "__main__":
    main()
