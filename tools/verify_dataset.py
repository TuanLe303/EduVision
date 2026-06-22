"""
EduVision — Dataset Verification Tool
========================================
Verifies dataset integrity, generates statistics, and creates visual samples.

Stages:
  raw_frames : Verify extracted frames in data/raw_frames/
  annotated  : Verify annotated data in data/annotated/
  merged     : Verify merged data in data/merged/
  dataset    : Verify final split dataset in data/dataset/

Checks:
  - Image-label pairing (each image has a label and vice versa)
  - Label format validation (5 fields, normalized coords)
  - Image readability (not corrupted)
  - Class distribution analysis
  - Bounding box statistics

Output:
  data/verification/
  ├── samples/                   # Annotated sample images
  ├── class_distribution.png     # Bar chart
  ├── bbox_histogram.png         # Bounding boxes per image
  └── verification_report.json

Usage:
    python tools/verify_dataset.py --stage raw_frames
    python tools/verify_dataset.py --stage annotated
    python tools/verify_dataset.py --stage merged
    python tools/verify_dataset.py --stage dataset
    python tools/verify_dataset.py --stage dataset --samples 20
    python tools/verify_dataset.py --help
"""

import argparse
import io
import json
import random
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
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    print("[ERROR] matplotlib is not installed.")
    print("  Run:  pip install matplotlib")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "data" / "verification"

BEHAVIOR_CLASSES = [
    "person", "focused", "drowsy", "sleeping", "using_phone",
    "off_task", "side_talking", "away_from_seat", "raising_hand",
]

CLASS_COLORS_BGR = [
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

CLASS_COLORS_RGB = [  # For matplotlib
    "#0064FF", "#00C800", "#FFDC00", "#FFA500",
    "#FF0000", "#C800C8", "#00FFFF", "#FF00FF", "#80FF00",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ─────────────────────────────────────────────────────────────────────────────
# Stage paths
# ─────────────────────────────────────────────────────────────────────────────

def get_stage_paths(stage: str) -> list[tuple[Path, Path | None]]:
    """
    Return list of (images_dir, labels_dir) for a given stage.
    labels_dir can be None (for raw_frames stage).
    """
    data_root = PROJECT_ROOT / "data"

    if stage == "raw_frames":
        paths = []
        rf = data_root / "raw_frames"
        # Wide shot subdirs
        ws = rf / "wide_shot"
        if ws.exists():
            for d in sorted(ws.iterdir()):
                if d.is_dir():
                    paths.append((d, None))
        # Expression subdirs
        expr = rf / "expression"
        if expr.exists():
            for pos in sorted(expr.iterdir()):
                if pos.is_dir():
                    for beh in sorted(pos.iterdir()):
                        if beh.is_dir():
                            paths.append((beh, None))
        return paths

    elif stage == "annotated":
        paths = []
        ann = data_root / "annotated"
        ws = ann / "wide_shot"
        if ws.exists():
            paths.append((ws / "images", ws / "labels"))
        expr = ann / "expression"
        if expr.exists():
            paths.append((expr / "images", expr / "labels"))
        return paths

    elif stage == "merged":
        merged = data_root / "merged"
        return [(merged / "images", merged / "labels")]

    elif stage == "dataset":
        ds = data_root / "dataset"
        paths = []
        for split in ["train", "val", "test"]:
            sp = ds / split
            if sp.exists():
                paths.append((sp / "images", sp / "labels"))
        return paths

    return []


# ─────────────────────────────────────────────────────────────────────────────
# Verification logic
# ─────────────────────────────────────────────────────────────────────────────

def verify_images(images_dir: Path) -> dict:
    """Check all images in a directory."""
    stats = {
        "total": 0, "readable": 0, "corrupted": 0,
        "resolutions": {}, "files": [],
    }

    if not images_dir.exists():
        return stats

    img_files = [f for f in sorted(images_dir.iterdir())
                 if f.suffix.lower() in IMAGE_EXTS]
    stats["total"] = len(img_files)

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            stats["corrupted"] += 1
        else:
            stats["readable"] += 1
            h, w = img.shape[:2]
            res = f"{w}x{h}"
            stats["resolutions"][res] = stats["resolutions"].get(res, 0) + 1
        stats["files"].append(img_path.name)

    return stats


def verify_labels(labels_dir: Path) -> dict:
    """Check all label files in a directory."""
    stats = {
        "total": 0, "valid": 0, "empty": 0, "invalid_format": 0,
        "class_counts": {i: 0 for i in range(len(BEHAVIOR_CLASSES))},
        "bboxes_per_image": [],
        "files": [],
    }

    if not labels_dir or not labels_dir.exists():
        return stats

    lbl_files = sorted(labels_dir.glob("*.txt"))
    stats["total"] = len(lbl_files)

    for lbl_path in lbl_files:
        stats["files"].append(lbl_path.name)
        n_boxes = 0
        valid = True

        with open(lbl_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            stats["empty"] += 1
            stats["bboxes_per_image"].append(0)
            continue

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                valid = False
                continue

            try:
                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                valid = False
                continue

            if not (0 <= cls_id < len(BEHAVIOR_CLASSES)):
                valid = False
                continue

            if not all(0.0 <= c <= 1.0 for c in coords):
                valid = False
                continue

            stats["class_counts"][cls_id] += 1
            n_boxes += 1

        if valid and n_boxes > 0:
            stats["valid"] += 1
        else:
            stats["invalid_format"] += 1

        stats["bboxes_per_image"].append(n_boxes)

    return stats


def check_pairing(img_stats: dict, lbl_stats: dict) -> dict:
    """Check if images and labels are properly paired."""
    img_stems = {Path(f).stem for f in img_stats["files"]}
    lbl_stems = {Path(f).stem for f in lbl_stats["files"]}

    paired = img_stems & lbl_stems
    imgs_without_labels = img_stems - lbl_stems
    labels_without_imgs = lbl_stems - img_stems

    return {
        "paired": len(paired),
        "images_without_labels": sorted(imgs_without_labels)[:20],
        "labels_without_images": sorted(labels_without_imgs)[:20],
        "n_images_without_labels": len(imgs_without_labels),
        "n_labels_without_images": len(labels_without_imgs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def draw_sample(image_path: Path, label_path: Path, output_path: Path) -> None:
    """Draw bounding boxes on an image and save."""
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
                try:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                except ValueError:
                    continue

                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                color = CLASS_COLORS_BGR[cls_id] if cls_id < len(CLASS_COLORS_BGR) else (0, 255, 0)
                label = BEHAVIOR_CLASSES[cls_id] if cls_id < len(BEHAVIOR_CLASSES) else str(cls_id)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(img, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


def generate_samples(stage_paths: list[tuple[Path, Path | None]],
                     n_samples: int) -> int:
    """Generate annotated sample images."""
    samples_dir = OUTPUT_DIR / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    for images_dir, labels_dir in stage_paths:
        if not images_dir or not images_dir.exists():
            continue
        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl_path = None
            if labels_dir:
                lbl_path = labels_dir / f"{img_path.stem}.txt"
                if not lbl_path.exists():
                    lbl_path = None
            all_pairs.append((img_path, lbl_path))

    if not all_pairs:
        return 0

    # Random sample
    rng = random.Random(42)
    samples = rng.sample(all_pairs, min(n_samples, len(all_pairs)))

    for i, (img_path, lbl_path) in enumerate(samples):
        out_path = samples_dir / f"sample_{i + 1:03d}.jpg"
        if lbl_path:
            draw_sample(img_path, lbl_path, out_path)
        else:
            import shutil
            shutil.copy2(img_path, out_path)

    return len(samples)


def generate_class_chart(class_counts: dict[int, int]) -> None:
    """Generate a class distribution bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    names = []
    counts = []
    colors = []
    for i, name in enumerate(BEHAVIOR_CLASSES):
        c = class_counts.get(i, 0)
        names.append(f"{name}\n(id={i})")
        counts.append(c)
        colors.append(CLASS_COLORS_RGB[i] if i < len(CLASS_COLORS_RGB) else "#888888")

    bars = ax.bar(names, counts, color=colors, edgecolor="black", linewidth=0.5)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                    f"{count:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("EduVision — Class Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Bounding Boxes", fontsize=11)
    ax.set_xlabel("Behavior Class", fontsize=11)
    ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()

    chart_path = OUTPUT_DIR / "class_distribution.png"
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(chart_path), dpi=150)
    plt.close()
    print(f"  [CHART] Class distribution -> {chart_path.relative_to(PROJECT_ROOT)}")


def generate_bbox_histogram(bboxes_per_image: list[int]) -> None:
    """Generate histogram of bounding boxes per image."""
    if not bboxes_per_image:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(bboxes_per_image, bins=max(10, max(bboxes_per_image) + 1),
            color="#4A90D9", edgecolor="black", linewidth=0.5)
    ax.set_title("EduVision — Bounding Boxes per Image", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Bounding Boxes", fontsize=11)
    ax.set_ylabel("Number of Images", fontsize=11)

    avg = sum(bboxes_per_image) / len(bboxes_per_image)
    ax.axvline(avg, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {avg:.1f}")
    ax.legend()
    plt.tight_layout()

    hist_path = OUTPUT_DIR / "bbox_histogram.png"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(hist_path), dpi=150)
    plt.close()
    print(f"  [CHART] BBox histogram -> {hist_path.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main verification
# ─────────────────────────────────────────────────────────────────────────────

def run_verification(stage: str, n_samples: int) -> dict:
    """Run full verification for a given stage."""
    stage_paths = get_stage_paths(stage)

    if not stage_paths:
        print(f"  [ERROR] No data found for stage '{stage}'")
        return {}

    report = {
        "stage": stage,
        "directories": [],
        "total_images": 0,
        "total_labels": 0,
        "total_corrupted": 0,
        "total_empty_labels": 0,
        "total_invalid_labels": 0,
        "class_distribution": {i: 0 for i in range(len(BEHAVIOR_CLASSES))},
        "all_bboxes_per_image": [],
    }

    for images_dir, labels_dir in stage_paths:
        dir_name = str(images_dir.relative_to(PROJECT_ROOT)) if images_dir.exists() else "N/A"
        print(f"\n  [DIR] {dir_name}")

        # Verify images
        img_stats = verify_images(images_dir)
        print(f"    Images: {img_stats['readable']}/{img_stats['total']} readable, "
              f"{img_stats['corrupted']} corrupted")
        if img_stats["resolutions"]:
            for res, count in img_stats["resolutions"].items():
                print(f"      Resolution {res}: {count} images")

        report["total_images"] += img_stats["total"]
        report["total_corrupted"] += img_stats["corrupted"]

        # Verify labels
        if labels_dir:
            lbl_stats = verify_labels(labels_dir)
            print(f"    Labels: {lbl_stats['valid']}/{lbl_stats['total']} valid, "
                  f"{lbl_stats['empty']} empty, {lbl_stats['invalid_format']} invalid")

            report["total_labels"] += lbl_stats["total"]
            report["total_empty_labels"] += lbl_stats["empty"]
            report["total_invalid_labels"] += lbl_stats["invalid_format"]

            for cls_id, count in lbl_stats["class_counts"].items():
                report["class_distribution"][cls_id] += count
            report["all_bboxes_per_image"].extend(lbl_stats["bboxes_per_image"])

            # Check pairing
            pairing = check_pairing(img_stats, lbl_stats)
            print(f"    Pairing: {pairing['paired']} paired, "
                  f"{pairing['n_images_without_labels']} imgs missing labels, "
                  f"{pairing['n_labels_without_images']} labels missing imgs")

        report["directories"].append({
            "path": dir_name,
            "images": img_stats["total"],
            "readable": img_stats["readable"],
            "corrupted": img_stats["corrupted"],
            "resolutions": img_stats["resolutions"],
        })

    # Generate visualizations
    print("\n  Generating visualizations ...")

    n_generated = generate_samples(stage_paths, n_samples)
    print(f"  [SAMPLES] {n_generated} sample images saved")

    if any(report["class_distribution"].values()):
        generate_class_chart(report["class_distribution"])

    if report["all_bboxes_per_image"]:
        generate_bbox_histogram(report["all_bboxes_per_image"])

    # Compute summary stats
    bpi = report["all_bboxes_per_image"]
    if bpi:
        report["bbox_stats"] = {
            "min": min(bpi),
            "max": max(bpi),
            "avg": round(sum(bpi) / len(bpi), 2),
            "total": sum(bpi),
        }

    # Named class distribution
    report["class_distribution_named"] = {
        BEHAVIOR_CLASSES[k]: v
        for k, v in report["class_distribution"].items()
    }

    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EduVision — Verify dataset integrity and generate statistics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage", required=True,
        choices=["raw_frames", "annotated", "merged", "dataset"],
        help="Which dataset stage to verify"
    )
    parser.add_argument(
        "--samples", type=int, default=10,
        help="Number of sample images to generate with drawn bboxes (default: 10)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("-" * 60)
    print("  EduVision -- Dataset Verification Tool")
    print("-" * 60)
    print(f"  Stage    : {args.stage}")
    print(f"  Samples  : {args.samples}")
    print()

    t0 = time.time()
    report = run_verification(args.stage, args.samples)
    elapsed = time.time() - t0

    if not report:
        sys.exit(1)

    # Print summary
    print("\n" + "-" * 60)
    print("  VERIFICATION SUMMARY")
    print("-" * 60)
    print(f"  Stage             : {args.stage}")
    print(f"  Total images      : {report['total_images']:,}")
    print(f"  Total labels      : {report['total_labels']:,}")
    print(f"  Corrupted images  : {report['total_corrupted']}")
    print(f"  Empty labels      : {report['total_empty_labels']}")
    print(f"  Invalid labels    : {report['total_invalid_labels']}")

    if "bbox_stats" in report:
        bs = report["bbox_stats"]
        print(f"\n  Bounding boxes:")
        print(f"    Total           : {bs['total']:,}")
        print(f"    Per image (avg) : {bs['avg']:.1f}")
        print(f"    Per image (min) : {bs['min']}")
        print(f"    Per image (max) : {bs['max']}")

    if report.get("class_distribution_named"):
        print(f"\n  Class distribution:")
        total_boxes = sum(report["class_distribution"].values())
        for name, count in report["class_distribution_named"].items():
            pct = count / total_boxes * 100 if total_boxes else 0
            bar = "|" * int(pct / 2)
            print(f"    {name:20s} {count:6,}  ({pct:5.1f}%)  {bar}")

    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  Output: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")

    # Health check
    issues = []
    if report["total_corrupted"] > 0:
        issues.append(f"{report['total_corrupted']} corrupted images")
    if report["total_invalid_labels"] > 0:
        issues.append(f"{report['total_invalid_labels']} invalid label files")
    if report["total_empty_labels"] > report["total_labels"] * 0.1:
        issues.append(f"High empty label ratio: {report['total_empty_labels']}/{report['total_labels']}")

    if issues:
        print(f"\n  [!] Issues found:")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print(f"\n  [OK] No issues found. Dataset looks healthy!")

    print("-" * 60)

    # Save report
    # Remove large lists for JSON
    report_save = {k: v for k, v in report.items()
                   if k != "all_bboxes_per_image"}
    report_save["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    report_path = OUTPUT_DIR / "verification_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_save, f, ensure_ascii=False, indent=2)
    print(f"  [DONE] Report -> {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
