"""
EduVision — Auto-Annotation Tool (YOLO26)
============================================
Uses a pretrained YOLO26 model to automatically detect persons in Wide Shot
frames and generates YOLO-format annotation (.txt) files.

Also annotates Expression Data frames by running person detection first,
then assigning behavior class labels based on folder names.

Output structure:
    data/
    └── annotated/
        ├── wide_shot/
        │   ├── images/        <- frame .jpg files (copied)
        │   └── labels/        <- YOLO .txt files  (auto-generated)
        ├── expression/
        │   ├── images/        <- cropped person .jpg files
        │   └── labels/        <- YOLO .txt files with behavior class
        ├── dataset.yaml       <- YOLO dataset config
        └── annotation_report.json

YOLO label format (one line per detected object):
    <class_id> <cx> <cy> <w> <h>   (all normalized 0-1)

Usage:
    python tools/auto_annotate.py                     # default settings
    python tools/auto_annotate.py --conf 0.4          # stricter confidence
    python tools/auto_annotate.py --model yolo26n     # YOLO26 nano (default)
    python tools/auto_annotate.py --model yolo26s     # YOLO26 small
    python tools/auto_annotate.py --preview           # show sample frames
    python tools/auto_annotate.py --help
"""

import argparse
import io
import json
import shutil
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics is not installed.")
    print("  Run:  pip install -U ultralytics")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("[ERROR] opencv-python is not installed.")
    print("  Run:  pip install opencv-python")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback: simple progress without tqdm
    def tqdm(iterable, **kwargs):
        return iterable

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

# Behavior classes for the EduVision dataset
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

# Folder name variations → canonical behavior name
# Handles extract_frames.py output like "using_phone_1" → "using_phone"
BEHAVIOR_FOLDER_REMAP = {
    "using_phone_1": "using_phone",
    "using_phone_2": "using_phone",
}


def resolve_behavior(folder_name: str) -> tuple[str, int]:
    """Resolve folder name to (canonical_behavior, class_id)."""
    canonical = BEHAVIOR_FOLDER_REMAP.get(folder_name, folder_name)
    cls_id = BEHAVIOR_CLASS_ID.get(canonical, 0)
    return canonical, cls_id

# Class colors for preview visualization (BGR)
CLASS_COLORS = [
    (255, 100, 0),    # person: blue
    (0, 200, 0),      # focused: green
    (0, 220, 255),    # drowsy: yellow
    (0, 165, 255),    # sleeping: orange
    (0, 0, 255),      # using_phone: red
    (200, 0, 200),    # off_task: purple
    (255, 255, 0),    # side_talking: cyan
    (200, 0, 200),    # away_from_seat: magenta
    (0, 255, 128),    # raising_hand: lime
]


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


def draw_preview(image_path: Path, label_path: Path, class_names: list[str],
                 save_path: Path | None = None) -> None:
    """Draw annotated preview and optionally save to file."""
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
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                color = CLASS_COLORS[cls_id] if cls_id < len(CLASS_COLORS) else (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                # Background for text
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(img, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), img)
    else:
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

        # Collect both original and augmented frames
        frame_files = sorted(angle_dir.glob("*.jpg"))
        if not frame_files:
            print(f"  [WARN] No .jpg files in {angle_name}")
            continue

        print(f"\n  [ANGLE] {angle_name}  ({len(frame_files)} frames)")

        for img_path in tqdm(frame_files, desc=f"    {angle_name}", unit="frame"):
            # Copy image with angle prefix to avoid name collisions
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
        if preview and frame_files:
            sample = frame_files[0]
            dest_img   = images_out / f"{angle_name}__{sample.name}"
            dest_label = labels_out / f"{angle_name}__{sample.stem}.txt"
            preview_path = OUTPUT_ROOT / "previews" / f"wide_{angle_name}.jpg"
            draw_preview(dest_img, dest_label, BEHAVIOR_CLASSES, save_path=preview_path)
            print(f"      Preview saved -> {preview_path.relative_to(PROJECT_ROOT)}")

    return stats


def annotate_expression(model: YOLO, conf: float, preview: bool) -> dict:
    """
    For expression frames:
    1. Run person detection to get actual person bounding box
    2. Assign behavior class based on the folder name
    
    This produces proper per-person bounding boxes with behavior labels,
    not full-frame classification boxes.
    """
    images_out = OUTPUT_ROOT / "expression" / "images"
    labels_out = OUTPUT_ROOT / "expression" / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    stats = {"processed": 0, "behaviors": {}, "total_detections": 0}

    for pos_dir in sorted(EXPR_DIR.iterdir()):
        if not pos_dir.is_dir():
            continue
        pos_name = pos_dir.name
        print(f"\n  [POS] {pos_name}")

        for behavior_dir in sorted(pos_dir.iterdir()):
            if not behavior_dir.is_dir():
                continue
            behavior = behavior_dir.name
            behavior_canonical, cls_id = resolve_behavior(behavior)

            frame_files = sorted(behavior_dir.glob("*.jpg"))
            if not frame_files:
                continue

            n = 0
            n_det = 0
            for img_path in tqdm(frame_files, desc=f"    {pos_name}/{behavior}", unit="frame"):
                # Unique filename: pos_behavior_frame.jpg
                dest_name = f"{pos_name}__{behavior}__{img_path.name}"
                dest_img = images_out / dest_name
                shutil.copy2(img_path, dest_img)

                # Run person detection to get actual bounding box
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img_h, img_w = img.shape[:2]

                results = model.predict(
                    source=str(img_path),
                    conf=conf,
                    classes=[PERSON_CLASS_ID],
                    verbose=False,
                )

                label_lines = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
                        cx = max(0.0, min(1.0, cx))
                        cy = max(0.0, min(1.0, cy))
                        bw = max(0.001, min(1.0, bw))
                        bh = max(0.001, min(1.0, bh))
                        # Use BEHAVIOR class id instead of person (0)
                        label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                        n_det += 1

                if not label_lines:
                    # Fallback: if no person detected, use full-frame box
                    label_lines.append(f"{cls_id} 0.500000 0.500000 1.000000 1.000000")

                dest_label = labels_out / f"{pos_name}__{behavior}__{img_path.stem}.txt"
                with open(dest_label, "w") as f:
                    f.write("\n".join(label_lines))
                n += 1

            stats["processed"] += n
            stats["total_detections"] += n_det
            stats["behaviors"][f"{pos_name}/{behavior}"] = {
                "frames": n,
                "detections": n_det,
                "class_id": cls_id,
            }
            print(f"    {behavior:20s}  {n:3d} frames, {n_det:3d} detections  (class_id={cls_id})")

            if preview and frame_files:
                sample = frame_files[0]
                dest_name = f"{pos_name}__{behavior}__{sample.name}"
                dest_img   = images_out / dest_name
                dest_label = labels_out / f"{pos_name}__{behavior}__{sample.stem}.txt"
                preview_path = OUTPUT_ROOT / "previews" / f"expr_{pos_name}_{behavior}.jpg"
                draw_preview(dest_img, dest_label, BEHAVIOR_CLASSES, save_path=preview_path)

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Dataset YAML
# ─────────────────────────────────────────────────────────────────────────────

def write_dataset_yaml() -> None:
    """Write a YOLO dataset.yaml for the annotated data."""
    yaml_path = OUTPUT_ROOT / "dataset.yaml"
    content = f"""# EduVision -- YOLO Dataset Configuration
# Generated by tools/auto_annotate.py
# Model: YOLO26
#
# After merging with external datasets and splitting, use:
#   python tools/merge_datasets.py
#   python tools/split_dataset.py --ratio 70 20 10

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
#   classes 1-8 = behavior labels (expression data, person-detected boxes)
#   Use tools/merge_datasets.py to combine with external datasets.
#   Use tools/split_dataset.py to create train/val/test split.
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n  [DONE] dataset.yaml -> {yaml_path.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(wide_stats: dict, expr_stats: dict, model_name: str) -> None:
    print("\n" + "-" * 60)
    print("  AUTO-ANNOTATION SUMMARY (YOLO26)")
    print("-" * 60)
    print(f"  Model used        : {model_name}")
    print(f"  Wide shot frames  : {wide_stats.get('processed', 0)}")
    print(f"  Person detections : {wide_stats.get('total_detections', 0)}")
    print(f"  Expr frames       : {expr_stats.get('processed', 0)}")
    print(f"  Expr detections   : {expr_stats.get('total_detections', 0)}")
    w_proc = wide_stats.get('processed', 0)
    if w_proc:
        avg = wide_stats.get('total_detections', 0) / w_proc
        print(f"  Avg persons/frame : {avg:.1f}")
    print(f"  Output            : {OUTPUT_ROOT.relative_to(PROJECT_ROOT)}")
    print("-" * 60)
    print()
    print("  NEXT STEPS:")
    print("  1. Review annotations: python tools/verify_dataset.py --stage annotated")
    print("  2. Merge datasets:    python tools/merge_datasets.py")
    print("  3. Split dataset:     python tools/split_dataset.py --ratio 70 20 10")
    print("  4. Train:             yolo train data=data/dataset/dataset.yaml"
          " model=yolo26n.pt epochs=50 imgsz=640")
    print("-" * 60)


def save_report(wide_stats: dict, expr_stats: dict, model_name: str,
                conf: float) -> None:
    report = {
        "model": model_name,
        "model_version": "YOLO26",
        "confidence_threshold": conf,
        "wide_shot": wide_stats,
        "expression": expr_stats,
        "behavior_classes": BEHAVIOR_CLASSES,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
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
        description="EduVision -- Auto-annotate frames using pretrained YOLO26.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", default="yolo26n",
        choices=["yolo26n", "yolo26s", "yolo26m", "yolo26l",
                 "yolo11n", "yolo11s"],  # backward compat
        help="YOLO model size (default: yolo26n — fastest, NMS-free)"
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
        help="Save preview images with drawn bounding boxes"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("-" * 60)
    print("  EduVision -- Auto-Annotation Tool (YOLO26)")
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
        print("\n-- Expression Data Auto-Detection -----------------------------")
        expr_stats = annotate_expression(model, args.conf, args.preview)

    write_dataset_yaml()
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    print_summary(wide_stats, expr_stats, args.model)
    save_report(wide_stats, expr_stats, args.model, args.conf)


if __name__ == "__main__":
    main()
