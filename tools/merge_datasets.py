"""
EduVision — Dataset Merge Tool
=================================
Merges the EduVision self-collected dataset with external datasets
(Ambient Intelligence Classroom) into a unified YOLO-format dataset.

Class mapping from Ambient Intelligence -> EduVision:
  Ambient 0 (Drowsy/Sleeping)      -> EduVision 2 (drowsy)
  Ambient 1 (Eating/Drinking)      -> EduVision 5 (off_task)
  Ambient 2 (Focused/Thinking)     -> EduVision 1 (focused)
  Ambient 3 (Looking down)         -> EduVision 5 (off_task)
  Ambient 4 (Looking upfront)      -> EduVision 1 (focused)
  Ambient 5 (Raising Hand)         -> EduVision 8 (raising_hand)
  Ambient 6 (Student)              -> EduVision 0 (person)
  Ambient 7 (Using-Laptop/Writing) -> EduVision 5 (off_task)
  Ambient 8 (Using-Phone)          -> EduVision 4 (using_phone)

Output structure:
    data/
    └── merged/
        ├── images/          # All images from both datasets
        ├── labels/          # All labels (remapped to EduVision classes)
        ├── dataset.yaml     # YOLO dataset config
        └── merge_report.json

Usage:
    python tools/merge_datasets.py
    python tools/merge_datasets.py --ev-only      # only EduVision data
    python tools/merge_datasets.py --ambient-only  # only Ambient data
    python tools/merge_datasets.py --help
"""

import argparse
import io
import json
import shutil
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EV_ANNOTATED = PROJECT_ROOT / "data" / "annotated"
EV_WIDE      = EV_ANNOTATED / "wide_shot"
EV_EXPR      = EV_ANNOTATED / "expression"
AMBIENT_ROOT = PROJECT_ROOT / "Data Collection" / "Recommended" / "Ambient Intelligence Classroom.yolo26"
OUTPUT_ROOT  = PROJECT_ROOT / "data" / "merged"

# ─────────────────────────────────────────────────────────────────────────────
# Class definitions
# ─────────────────────────────────────────────────────────────────────────────

EDUVISION_CLASSES = [
    "person",          # 0
    "focused",         # 1
    "drowsy",          # 2
    "sleeping",        # 3
    "using_phone",     # 4
    "off_task",        # 5
    "side_talking",    # 6
    "away_from_seat",  # 7
    "raising_hand",    # 8
]

# Ambient Intelligence class ID -> EduVision class ID
AMBIENT_TO_EDUVISION = {
    0: 2,   # Drowsy/Sleeping   -> drowsy
    1: 5,   # Eating/Drinking   -> off_task
    2: 1,   # Focused/Thinking  -> focused
    3: 5,   # Looking down      -> off_task
    4: 1,   # Looking upfront   -> focused
    5: 8,   # Raising Hand      -> raising_hand
    6: 0,   # Student           -> person
    7: 5,   # Using-Laptop/Writing -> off_task
    8: 4,   # Using-Phone       -> using_phone
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def remap_label_file(src_label: Path, dst_label: Path,
                     class_map: dict[int, int]) -> dict:
    """
    Read a YOLO label file, remap class IDs, write to destination.
    Returns stats: {remapped: int, invalid: int, lines: int}
    """
    stats = {"remapped": 0, "invalid": 0, "lines": 0}
    output_lines = []

    with open(src_label, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                stats["invalid"] += 1
                continue

            try:
                old_cls = int(parts[0])
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                stats["invalid"] += 1
                continue

            # Validate coordinates are in [0, 1]
            if not all(0.0 <= c <= 1.0 for c in coords):
                stats["invalid"] += 1
                continue

            new_cls = class_map.get(old_cls, old_cls)
            output_lines.append(
                f"{new_cls} {coords[0]:.6f} {coords[1]:.6f} "
                f"{coords[2]:.6f} {coords[3]:.6f}"
            )
            stats["remapped"] += 1

    stats["lines"] = len(output_lines)
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_label, "w") as f:
        f.write("\n".join(output_lines))

    return stats


def copy_image_label_pair(
    img_src: Path,
    lbl_src: Path,
    images_out: Path,
    labels_out: Path,
    prefix: str,
    class_map: dict[int, int] | None = None,
) -> dict | None:
    """Copy an image+label pair with optional class remapping."""
    dest_img = images_out / f"{prefix}__{img_src.name}"
    dest_lbl = labels_out / f"{prefix}__{lbl_src.stem}.txt"

    shutil.copy2(img_src, dest_img)

    if class_map:
        stats = remap_label_file(lbl_src, dest_lbl, class_map)
    else:
        shutil.copy2(lbl_src, dest_lbl)
        # Count lines
        with open(lbl_src) as f:
            n = sum(1 for line in f if line.strip())
        stats = {"remapped": n, "invalid": 0, "lines": n}

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Processing
# ─────────────────────────────────────────────────────────────────────────────

def merge_eduvision_data(images_out: Path, labels_out: Path) -> dict:
    """Copy EduVision annotated data (wide shot + expression) to merged dir."""
    stats = {"wide_shot": {"images": 0, "labels": 0, "detections": 0},
             "expression": {"images": 0, "labels": 0, "detections": 0}}

    # Wide shot
    ws_images = EV_WIDE / "images"
    ws_labels = EV_WIDE / "labels"
    if ws_images.exists() and ws_labels.exists():
        img_files = sorted(ws_images.glob("*.jpg"))
        print(f"\n  [EV Wide Shot] {len(img_files)} images")

        for img_path in tqdm(img_files, desc="    Wide shot", unit="img"):
            lbl_path = ws_labels / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            s = copy_image_label_pair(
                img_path, lbl_path, images_out, labels_out,
                prefix="ev", class_map=None  # Already EduVision format
            )
            stats["wide_shot"]["images"] += 1
            stats["wide_shot"]["labels"] += 1
            stats["wide_shot"]["detections"] += s["lines"]
    else:
        print(f"  [WARN] EduVision wide shot not found at: {EV_WIDE}")

    # Expression
    expr_images = EV_EXPR / "images"
    expr_labels = EV_EXPR / "labels"
    if expr_images.exists() and expr_labels.exists():
        img_files = sorted(expr_images.glob("*.jpg"))
        print(f"\n  [EV Expression] {len(img_files)} images")

        for img_path in tqdm(img_files, desc="    Expression", unit="img"):
            lbl_path = expr_labels / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            s = copy_image_label_pair(
                img_path, lbl_path, images_out, labels_out,
                prefix="ev_expr", class_map=None
            )
            stats["expression"]["images"] += 1
            stats["expression"]["labels"] += 1
            stats["expression"]["detections"] += s["lines"]
    else:
        print(f"  [INFO] EduVision expression data not found at: {EV_EXPR}")

    return stats


def merge_ambient_data(images_out: Path, labels_out: Path) -> dict:
    """Copy Ambient Intelligence Classroom data with class remapping."""
    stats = {"images": 0, "labels": 0, "detections": 0,
             "invalid_lines": 0, "splits": {}}

    if not AMBIENT_ROOT.exists():
        print(f"  [WARN] Ambient Intelligence dataset not found at: {AMBIENT_ROOT}")
        return stats

    # Process train, valid, test splits
    for split_name, split_dir_name in [("train", "train"), ("valid", "valid"), ("test", "test")]:
        split_images = AMBIENT_ROOT / split_dir_name / "images"
        split_labels = AMBIENT_ROOT / split_dir_name / "labels"

        if not split_images.exists():
            continue

        img_files = sorted(split_images.glob("*"))
        img_files = [f for f in img_files if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        print(f"\n  [Ambient {split_name}] {len(img_files)} images")

        split_stats = {"images": 0, "labels": 0, "detections": 0}

        for img_path in tqdm(img_files, desc=f"    {split_name}", unit="img"):
            lbl_path = split_labels / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            # Use prefix with split name to ensure uniqueness
            prefix = f"ambient_{split_name}"
            s = copy_image_label_pair(
                img_path, lbl_path, images_out, labels_out,
                prefix=prefix, class_map=AMBIENT_TO_EDUVISION
            )

            split_stats["images"] += 1
            split_stats["labels"] += 1
            split_stats["detections"] += s["lines"]
            stats["invalid_lines"] += s["invalid"]

        stats["splits"][split_name] = split_stats
        stats["images"] += split_stats["images"]
        stats["labels"] += split_stats["labels"]
        stats["detections"] += split_stats["detections"]

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Dataset YAML
# ─────────────────────────────────────────────────────────────────────────────

def write_merged_yaml() -> None:
    """Write dataset.yaml for the merged dataset."""
    yaml_path = OUTPUT_ROOT / "dataset.yaml"
    lines = [
        "# EduVision -- Merged Dataset Configuration",
        "# Generated by tools/merge_datasets.py",
        "# Sources: EduVision self-collected + Ambient Intelligence Classroom",
        "",
        f"path: {OUTPUT_ROOT.as_posix()}",
        "train: images  # will be split by tools/split_dataset.py",
        "val:   images",
        "test:  images",
        "",
        f"nc: {len(EDUVISION_CLASSES)}",
        "names:",
    ]
    for i, name in enumerate(EDUVISION_CLASSES):
        lines.append(f"  {i}: {name}")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  [DONE] dataset.yaml -> {yaml_path.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary & Report
# ─────────────────────────────────────────────────────────────────────────────

def count_class_distribution(labels_dir: Path) -> dict[str, int]:
    """Count how many bounding boxes per class in the merged labels."""
    counts = {name: 0 for name in EDUVISION_CLASSES}
    for lbl_file in labels_dir.glob("*.txt"):
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    if 0 <= cls_id < len(EDUVISION_CLASSES):
                        counts[EDUVISION_CLASSES[cls_id]] += 1
    return counts


def print_summary(ev_stats: dict, ambient_stats: dict,
                  class_dist: dict[str, int]) -> None:
    ev_total = (ev_stats["wide_shot"]["images"] +
                ev_stats["expression"]["images"])
    amb_total = ambient_stats["images"]
    total = ev_total + amb_total

    print("\n" + "-" * 60)
    print("  DATASET MERGE SUMMARY")
    print("-" * 60)
    print(f"  EduVision images  : {ev_total:,}")
    print(f"    Wide shot       : {ev_stats['wide_shot']['images']:,}")
    print(f"    Expression      : {ev_stats['expression']['images']:,}")
    print(f"  Ambient images    : {amb_total:,}")
    print(f"  Total merged      : {total:,}")
    if ambient_stats["invalid_lines"]:
        print(f"  Invalid lines     : {ambient_stats['invalid_lines']}")

    print(f"\n  Class distribution:")
    total_boxes = sum(class_dist.values())
    for name, count in class_dist.items():
        pct = count / total_boxes * 100 if total_boxes else 0
        bar = "|" * int(pct / 2)
        print(f"    {name:20s} {count:6,}  ({pct:5.1f}%)  {bar}")

    print("-" * 60)
    print()
    print("  NEXT STEPS:")
    print("  1. Verify: python tools/verify_dataset.py --stage merged")
    print("  2. Split:  python tools/split_dataset.py --ratio 70 20 10")
    print("-" * 60)


def save_report(ev_stats: dict, ambient_stats: dict,
                class_dist: dict) -> None:
    report = {
        "eduvision": ev_stats,
        "ambient_intelligence": ambient_stats,
        "class_mapping": {
            f"ambient_{k}": f"eduvision_{v}"
            for k, v in AMBIENT_TO_EDUVISION.items()
        },
        "class_distribution": class_dist,
        "eduvision_classes": EDUVISION_CLASSES,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    report_path = OUTPUT_ROOT / "merge_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  [DONE] Report -> {report_path.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EduVision — Merge self-collected and external datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ev-only", action="store_true",
        help="Only include EduVision self-collected data"
    )
    parser.add_argument(
        "--ambient-only", action="store_true",
        help="Only include Ambient Intelligence data"
    )
    parser.add_argument(
        "--clean", action="store_true", default=True,
        help="Remove existing merged data before merging (default: True)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("-" * 60)
    print("  EduVision -- Dataset Merge Tool")
    print("-" * 60)
    print(f"  Output : {OUTPUT_ROOT.relative_to(PROJECT_ROOT)}")
    print()

    # Clean output
    if args.clean and OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    images_out = OUTPUT_ROOT / "images"
    labels_out = OUTPUT_ROOT / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ev_stats = {"wide_shot": {"images": 0, "labels": 0, "detections": 0},
                "expression": {"images": 0, "labels": 0, "detections": 0}}
    ambient_stats = {"images": 0, "labels": 0, "detections": 0,
                     "invalid_lines": 0, "splits": {}}

    if not args.ambient_only:
        print("-- EduVision Data --------------------------------------------")
        ev_stats = merge_eduvision_data(images_out, labels_out)

    if not args.ev_only:
        print("\n-- Ambient Intelligence Classroom ----------------------------")
        ambient_stats = merge_ambient_data(images_out, labels_out)

    write_merged_yaml()

    # Count class distribution
    print("\n  Counting class distribution ...")
    class_dist = count_class_distribution(labels_out)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    print_summary(ev_stats, ambient_stats, class_dist)
    save_report(ev_stats, ambient_stats, class_dist)


if __name__ == "__main__":
    main()
