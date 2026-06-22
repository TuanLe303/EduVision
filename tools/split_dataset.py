"""
EduVision -- Dataset Split Tool
=================================
Splits the merged dataset (from tools/merge_datasets.py) or annotated
data into train / val / test sets and creates a YOLO26-ready dataset.

Default split: 70% train / 20% val / 10% test  (stratified by source prefix)

Output structure:
    data/
    └── dataset/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── val/
        │   ├── images/
        │   └── labels/
        ├── test/
        │   ├── images/
        │   └── labels/
        └── dataset.yaml

Usage:
    python tools/split_dataset.py                      # 70/20/10 default
    python tools/split_dataset.py --ratio 80 10 10
    python tools/split_dataset.py --seed 42
    python tools/split_dataset.py --help
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
# Default source: merged dataset (output of merge_datasets.py)
# Falls back to annotated/wide_shot if merged doesn't exist
MERGED_ROOT    = PROJECT_ROOT / "data" / "merged"
ANNOTATED_ROOT = PROJECT_ROOT / "data" / "annotated" / "wide_shot"
OUTPUT_ROOT    = PROJECT_ROOT / "data" / "dataset"

BEHAVIOR_CLASSES = [
    "person",
    "focused",
    "drowsy",
    "sleeping",
    "using_phone",
    "off_task",
    "side_talking",
    "away_from_seat",
    "raising_hand",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def collect_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    """Return (image_path, label_path) pairs that both exist."""
    pairs = []
    for img in sorted(images_dir.glob("*.jpg")):
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            print(f"  [WARN] No label for {img.name} — skipped")
    return pairs


def split_list(items: list, train_r: float, val_r: float, seed: int):
    """Split a list into (train, val, test) with given ratios."""
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    train = shuffled[:n_train]
    val   = shuffled[n_train:n_train + n_val]
    test  = shuffled[n_train + n_val:]
    return train, val, test


def copy_pairs(pairs: list[tuple[Path, Path]], split: str) -> int:
    """Copy image+label pairs to the output split directory."""
    img_out = OUTPUT_ROOT / split / "images"
    lbl_out = OUTPUT_ROOT / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for img, lbl in pairs:
        shutil.copy2(img, img_out / img.name)
        shutil.copy2(lbl, lbl_out / lbl.name)
    return len(pairs)


def write_yaml(nc: int, class_names: list[str]) -> None:
    """Write the final dataset.yaml for YOLO training."""
    yaml_path = OUTPUT_ROOT / "dataset.yaml"
    lines = [
        "# EduVision -- YOLO Training Dataset",
        f"# Split: train/val/test",
        "",
        f"path: {OUTPUT_ROOT.as_posix()}",
        "train: train/images",
        "val:   val/images",
        "test:  test/images",
        "",
        f"nc: {nc}",
        "names:",
    ]
    for i, name in enumerate(class_names):
        lines.append(f"  {i}: {name}")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [DONE] dataset.yaml -> {yaml_path.relative_to(PROJECT_ROOT)}")


def print_summary(counts: dict, total: int, train_r: float, val_r: float) -> None:
    test_r = 1.0 - train_r - val_r
    print("\n" + "-" * 60)
    print("  DATASET SPLIT SUMMARY")
    print("-" * 60)
    print(f"  Total pairs   : {total}")
    print(f"  Split ratio   : train {train_r*100:.0f}% / "
          f"val {val_r*100:.0f}% / test {test_r*100:.0f}%")
    print()
    for split, n in counts.items():
        pct = n / total * 100 if total else 0
        bar = "|" * int(pct / 2)
        print(f"  {split:8s}  {n:4d} frames  ({pct:.1f}%)  {bar}")
    print()
    print("  To start training (YOLO26):")
    print(f"    yolo train data={OUTPUT_ROOT.as_posix()}/dataset.yaml")
    print( "                   model=yolo26n.pt epochs=50 imgsz=640")
    print()
    print("  To verify dataset:")
    print("    python tools/verify_dataset.py --stage dataset")
    print("-" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EduVision -- Split annotated dataset into train/val/test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ratio", nargs=3, type=float, default=[70, 20, 10],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split percentages (must sum to 100). Default: 70 20 10"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--source", type=Path, default=None,
        help="Source data dir with images/ and labels/ subdirs. "
             "Default: data/merged/ if exists, else data/annotated/wide_shot/"
    )
    parser.add_argument(
        "--stratify", action="store_true", default=True,
        help="Stratify by camera angle prefix (default: True)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_pct, val_pct, test_pct = args.ratio
    if abs(train_pct + val_pct + test_pct - 100) > 0.01:
        print(f"[ERROR] Ratios must sum to 100, got {train_pct+val_pct+test_pct}")
        sys.exit(1)

    train_r = train_pct / 100
    val_r   = val_pct   / 100

    # Determine source directory
    if args.source is not None:
        source = args.source
    elif MERGED_ROOT.exists() and (MERGED_ROOT / "images").exists():
        source = MERGED_ROOT
    elif ANNOTATED_ROOT.exists() and (ANNOTATED_ROOT / "images").exists():
        source = ANNOTATED_ROOT
    else:
        print("[ERROR] No dataset found. Run one of:")
        print("  python tools/merge_datasets.py   (recommended)")
        print("  python tools/auto_annotate.py     (EduVision data only)")
        sys.exit(1)

    print("-" * 60)
    print("  EduVision -- Dataset Split Tool")
    print("-" * 60)
    print(f"  Source : {source.relative_to(PROJECT_ROOT)}")
    print(f"  Output : {OUTPUT_ROOT.relative_to(PROJECT_ROOT)}")
    print(f"  Ratio  : {train_pct:.0f}/{val_pct:.0f}/{test_pct:.0f}")
    print(f"  Seed   : {args.seed}")
    print()

    images_dir = source / "images"
    labels_dir = source / "labels"

    if not images_dir.exists():
        print(f"[ERROR] Images dir not found: {images_dir}")
        sys.exit(1)

    all_pairs = collect_pairs(images_dir, labels_dir)
    if not all_pairs:
        print("[ERROR] No image-label pairs found. Check the source directory.")
        sys.exit(1)

    print(f"  Found {len(all_pairs)} annotated pairs")

    # Clean output
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)

    if args.stratify:
        # Group by camera angle (prefix before __)
        groups: dict[str, list] = {}
        for pair in all_pairs:
            angle = pair[0].name.split("__")[0]
            groups.setdefault(angle, []).append(pair)

        train_all, val_all, test_all = [], [], []
        for angle, pairs in sorted(groups.items()):
            t, v, te = split_list(pairs, train_r, val_r, args.seed)
            train_all.extend(t)
            val_all.extend(v)
            test_all.extend(te)
            print(f"  [{angle}]  {len(t)} train / {len(v)} val / {len(te)} test")
    else:
        train_all, val_all, test_all = split_list(all_pairs, train_r, val_r, args.seed)

    print()
    n_train = copy_pairs(train_all, "train")
    print(f"  Copied {n_train} pairs -> train/")
    n_val   = copy_pairs(val_all,   "val")
    print(f"  Copied {n_val}   pairs -> val/")
    n_test  = copy_pairs(test_all,  "test")
    print(f"  Copied {n_test}  pairs -> test/")

    write_yaml(len(BEHAVIOR_CLASSES), BEHAVIOR_CLASSES)

    # Save split report
    report = {
        "seed": args.seed,
        "ratio": {"train": train_pct, "val": val_pct, "test": test_pct},
        "counts": {"train": n_train, "val": n_val, "test": n_test},
        "total": len(all_pairs),
    }
    report_path = OUTPUT_ROOT / "split_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_summary(
        {"train": n_train, "val": n_val, "test": n_test},
        len(all_pairs),
        train_r, val_r,
    )


if __name__ == "__main__":
    main()
