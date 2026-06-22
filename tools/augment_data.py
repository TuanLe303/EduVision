"""
EduVision — Data Augmentation Tool
=====================================
Applies image augmentations to extracted frames to increase dataset diversity.
Augmented images are saved alongside originals in the same directory structure.

Augmentations applied (using albumentations):
  - Horizontal flip
  - Random brightness / contrast
  - Random hue / saturation shifts
  - Gaussian blur (light)
  - Random rotation (small angles, ±10°)
  - Random perspective transform (light)
  - Additive Gaussian noise
  - CLAHE (contrast limited adaptive histogram equalization)

Output:
  Augmented files named: frame_00001_aug1.jpg, frame_00001_aug2.jpg, etc.
  Saved in the same directory as originals.

Usage:
    python tools/augment_data.py                    # default: 2 augmented copies
    python tools/augment_data.py --multiplier 3     # 3 augmented copies per image
    python tools/augment_data.py --wide-only        # only augment wide shot
    python tools/augment_data.py --expr-only        # only augment expression
    python tools/augment_data.py --help
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    import cv2
except ImportError:
    print("[ERROR] opencv-python is not installed.")
    print("  Run:  pip install opencv-python")
    sys.exit(1)

try:
    import albumentations as A
except ImportError:
    print("[ERROR] albumentations is not installed.")
    print("  Run:  pip install albumentations")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAMES_ROOT  = PROJECT_ROOT / "data" / "raw_frames"
WIDE_SHOT_DIR = FRAMES_ROOT / "wide_shot"
EXPR_DIR      = FRAMES_ROOT / "expression"


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_augmentation_pipeline() -> A.Compose:
    """
    Build an augmentation pipeline that produces visually diverse variants
    while keeping the subject recognizable.

    All transforms are image-only at this stage. Bounding box annotations
    are generated AFTER augmentation by auto_annotate.py, so we do not need
    to transform coordinates here.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.4),
        A.Perspective(scale=(0.02, 0.06), p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=1.0
            ),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.6),
        A.OneOf([
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=1.0
            ),
            A.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=1.0
            ),
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.GaussNoise(p=1.0),
        ], p=0.3),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Processing
# ─────────────────────────────────────────────────────────────────────────────

def augment_directory(
    directory: Path,
    pipeline: A.Compose,
    multiplier: int,
    quality: int,
) -> dict:
    """
    Augment all .jpg images in a directory (non-recursive).
    Skips files that already have '_aug' in their name.

    Returns stats dict.
    """
    originals = sorted([
        f for f in directory.glob("*.jpg")
        if "_aug" not in f.stem
    ])

    stats = {"originals": len(originals), "augmented": 0, "skipped": 0}

    for img_path in originals:
        img = cv2.imread(str(img_path))
        if img is None:
            stats["skipped"] += 1
            continue

        # Convert BGR -> RGB for albumentations, then back
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for i in range(1, multiplier + 1):
            aug_name = f"{img_path.stem}_aug{i}.jpg"
            aug_path = directory / aug_name

            # Skip if already augmented
            if aug_path.exists():
                continue

            augmented = pipeline(image=img_rgb)["image"]
            aug_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(aug_path), aug_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            stats["augmented"] += 1

    return stats


def process_wide_shot(pipeline: A.Compose, multiplier: int, quality: int) -> dict:
    """Augment all wide shot angle directories."""
    results = {}

    if not WIDE_SHOT_DIR.exists():
        print(f"  [WARN] Wide shot directory not found: {WIDE_SHOT_DIR}")
        return results

    for angle_dir in sorted(WIDE_SHOT_DIR.iterdir()):
        if not angle_dir.is_dir():
            continue
        angle_name = angle_dir.name
        print(f"\n  [ANGLE] {angle_name}")

        stats = augment_directory(angle_dir, pipeline, multiplier, quality)
        results[angle_name] = stats
        print(f"      {stats['originals']} originals -> "
              f"{stats['augmented']} augmented "
              f"({stats['skipped']} skipped)")

    return results


def process_expression(pipeline: A.Compose, multiplier: int, quality: int) -> dict:
    """Augment all expression data directories."""
    results = {}

    if not EXPR_DIR.exists():
        print(f"  [WARN] Expression directory not found: {EXPR_DIR}")
        return results

    for pos_dir in sorted(EXPR_DIR.iterdir()):
        if not pos_dir.is_dir():
            continue
        pos_name = pos_dir.name
        print(f"\n  [POS] {pos_name}")

        for behavior_dir in sorted(pos_dir.iterdir()):
            if not behavior_dir.is_dir():
                continue
            behavior = behavior_dir.name

            stats = augment_directory(behavior_dir, pipeline, multiplier, quality)
            key = f"{pos_name}/{behavior}"
            results[key] = stats
            print(f"    {behavior:20s}  {stats['originals']:3d} orig -> "
                  f"{stats['augmented']:3d} aug")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Summary & Report
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(wide_results: dict, expr_results: dict, multiplier: int) -> None:
    total_orig = sum(s["originals"] for s in wide_results.values())
    total_orig += sum(s["originals"] for s in expr_results.values())
    total_aug = sum(s["augmented"] for s in wide_results.values())
    total_aug += sum(s["augmented"] for s in expr_results.values())

    print("\n" + "-" * 60)
    print("  AUGMENTATION SUMMARY")
    print("-" * 60)
    print(f"  Multiplier          : {multiplier}x")
    print(f"  Original images     : {total_orig:,}")
    print(f"  Augmented created   : {total_aug:,}")
    print(f"  Total after augment : {total_orig + total_aug:,}")
    print(f"  Expansion ratio     : {(total_orig + total_aug) / max(1, total_orig):.1f}x")
    print("-" * 60)


def save_report(wide_results: dict, expr_results: dict,
                multiplier: int, quality: int) -> None:
    report = {
        "settings": {"multiplier": multiplier, "jpeg_quality": quality},
        "wide_shot": wide_results,
        "expression": expr_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    report_path = FRAMES_ROOT / "augmentation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  [DONE] Report -> {report_path.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EduVision — Augment extracted frames for dataset diversity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--multiplier", type=int, default=2,
        help="Number of augmented copies per original image (default: 2)"
    )
    parser.add_argument(
        "--quality", type=int, default=85,
        help="JPEG quality for augmented images 1-100 (default: 85)"
    )
    parser.add_argument(
        "--wide-only", action="store_true",
        help="Only augment Wide Shot frames"
    )
    parser.add_argument(
        "--expr-only", action="store_true",
        help="Only augment Expression frames"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("-" * 60)
    print("  EduVision -- Data Augmentation Tool")
    print("-" * 60)
    print(f"  Source     : {FRAMES_ROOT.relative_to(PROJECT_ROOT)}")
    print(f"  Multiplier : {args.multiplier}x")
    print(f"  Quality    : {args.quality}")
    print()

    if not FRAMES_ROOT.exists():
        print(f"[ERROR] Frames not found at: {FRAMES_ROOT}")
        print("  Run tools/extract_frames.py first.")
        sys.exit(1)

    pipeline = build_augmentation_pipeline()
    t0 = time.time()

    wide_results = {}
    expr_results = {}

    if not args.expr_only:
        print("-- Wide Shot Augmentation -------------------------------------")
        wide_results = process_wide_shot(pipeline, args.multiplier, args.quality)

    if not args.wide_only:
        print("\n-- Expression Data Augmentation --------------------------------")
        expr_results = process_expression(pipeline, args.multiplier, args.quality)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    print_summary(wide_results, expr_results, args.multiplier)
    save_report(wide_results, expr_results, args.multiplier, args.quality)


if __name__ == "__main__":
    main()
