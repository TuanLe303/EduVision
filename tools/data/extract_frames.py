"""
EduVision — Frame Extraction Tool
===================================
Extracts frames from raw .MOV/.mov/.mp4 videos in the Data Collection folder
and organizes them into a structured dataset for training.

Output structure:
    data/
    ├── raw_frames/
    │   ├── wide_shot/
    │   │   ├── goc_cheo/         ← frame_0001.jpg, frame_0002.jpg, ...
    │   │   ├── goc_thang_phai/
    │   │   └── goc_thang_trai/
    │   └── expression/
    │       ├── pos1/
    │       │   ├── focused/
    │       │   ├── drowsy/
    │       │   ├── sleeping/
    │       │   ├── using_phone/
    │       │   ├── off_task/
    │       │   ├── side_talking/
    │       │   ├── away_from_seat/
    │       │   └── raising_hand/
    │       ├── pos2/
    │       └── pos3/
    └── extraction_report.json

Usage:
    python tools/extract_frames.py                        # default: 2 fps
    python tools/extract_frames.py --fps 5               # 5 fps
    python tools/extract_frames.py --fps 1 --quality 90  # 1 fps, JPEG quality 90
    python tools/extract_frames.py --help
"""

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

try:
    import cv2
except ImportError:
    print("[ERROR] opencv-python is not installed.")
    print("  Run:  pip install opencv-python")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_COLLECTION = PROJECT_ROOT / "Data Collection"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "raw_frames"

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".MOV", ".MP4"}

# Mapping of raw folder/file names → clean dataset names (slugified)
WIDE_SHOT_REMAP = {
    "góc chéo": "goc_cheo",
    "góc thẳng - phải": "goc_thang_phai",
    "góc thẳng - trái": "goc_thang_trai",
}

BEHAVIOR_REMAP = {
    "using_phone_1": "using_phone",
    "using_phone_2": "using_phone",
    "away_from_seat": "away_from_seat",
    "drowsy":         "drowsy",
    "focused":        "focused",
    "off_task":       "off_task",
    "raising_hand":   "raising_hand",
    "side_talking":   "side_talking",
    "sleeping":       "sleeping",
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def slugify(name: str) -> str:
    """Convert arbitrary string to ASCII filesystem-safe slug.
    
    Strips Unicode diacriticals (e.g., ó→o, é→e, ả→a) to ensure
    compatibility with cv2.imwrite() on Windows which can't handle
    non-ASCII paths reliably.
    """
    import unicodedata
    name = name.lower().strip()
    # Decompose Unicode characters and remove combining marks (diacriticals)
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    # Remove non-alphanumeric chars (except spaces and hyphens)
    name = re.sub(r"[^\w\s-]", "", name, flags=re.ASCII)
    name = re.sub(r"[\s-]+", "_", name)
    return name


def format_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def extract_frames(
    video_path: Path,
    output_dir: Path,
    target_fps: float,
    quality: int,
    prefix: str = "frame",
    start_index: int = 1,
) -> dict:
    """
    Extract frames from *video_path* at *target_fps* into *output_dir*.

    Returns a dict with extraction statistics.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration  = total_src / src_fps if src_fps else 0

    # How many source frames to skip between extracted frames
    frame_interval = max(1, round(src_fps / target_fps))

    output_dir.mkdir(parents=True, exist_ok=True)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    frame_idx   = 0   # source frame counter
    saved_count = 0   # extracted frame counter
    t_start     = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            filename = output_dir / f"{prefix}_{start_index + saved_count:05d}.jpg"
            cv2.imwrite(str(filename), frame, encode_params)
            saved_count += 1
        frame_idx += 1

    cap.release()
    elapsed = time.time() - t_start

    return {
        "video":          str(video_path.relative_to(PROJECT_ROOT)),
        "source_fps":     round(src_fps, 2),
        "target_fps":     target_fps,
        "frame_interval": frame_interval,
        "resolution":     f"{width}×{height}",
        "duration_sec":   round(duration, 2),
        "duration_fmt":   format_duration(duration),
        "total_src_frames": total_src,
        "extracted_frames": saved_count,
        "output_dir":     str(output_dir.relative_to(PROJECT_ROOT)),
        "elapsed_sec":    round(elapsed, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Processing helpers
# ──────────────────────────────────────────────────────────────────────────────

def process_wide_shot(fps: float, quality: int) -> list[dict]:
    """Extract frames from Wide Shot folder."""
    wide_dir = DATA_COLLECTION / "Wide Shot"
    results = []

    for video_file in sorted(wide_dir.iterdir()):
        if video_file.suffix not in VIDEO_EXTENSIONS:
            continue

        stem = video_file.stem
        # Try remap first (lowercase for case-insensitive match), fall back to slugify
        out_name = WIDE_SHOT_REMAP.get(stem.lower()) or slugify(stem)
        out_dir  = OUTPUT_ROOT / "wide_shot" / out_name

        print(f"  [VIDEO] {video_file.name}  ->  {out_dir.relative_to(PROJECT_ROOT)}")
        try:
            stats = extract_frames(video_file, out_dir, fps, quality)
            print(f"      [OK]  {stats['extracted_frames']} frames  "
                  f"({stats['duration_fmt']}, {stats['source_fps']} fps src)  "
                  f"in {stats['elapsed_sec']}s")
            results.append({**stats, "category": "wide_shot"})
        except Exception as exc:
            print(f"      [ERR] Error: {exc}")
            results.append({"video": str(video_file), "error": str(exc)})

    return results


def process_expression(fps: float, quality: int) -> list[dict]:
    """Extract frames from Expression Data folder (all positions)."""
    expr_dir = DATA_COLLECTION / "Expression Data"
    results  = []

    # Track per-behavior global frame index so using_phone_1 and _2 are
    # saved into the same folder without overwriting each other.
    behavior_counters: dict[str, dict[str, int]] = {}  # pos → behavior → count

    for pos_dir in sorted(expr_dir.iterdir()):
        if not pos_dir.is_dir():
            continue
        pos_slug = slugify(pos_dir.name)  # e.g. "pos_1"
        behavior_counters[pos_slug] = {}

        print(f"  [POS] {pos_dir.name}")

        for video_file in sorted(pos_dir.iterdir()):
            if video_file.suffix not in VIDEO_EXTENSIONS:
                continue

            stem     = video_file.stem
            behavior = BEHAVIOR_REMAP.get(stem) or slugify(stem)
            out_dir  = OUTPUT_ROOT / "expression" / pos_slug / behavior

            # Determine start index so using_phone_1 + using_phone_2 don't clash
            existing_count = behavior_counters[pos_slug].get(behavior, 0)

            print(f"    [VIDEO] {video_file.name}  ->  {out_dir.relative_to(PROJECT_ROOT)}")
            try:
                stats = extract_frames(
                    video_file, out_dir, fps, quality,
                    start_index=existing_count + 1
                )
                behavior_counters[pos_slug][behavior] = (
                    existing_count + stats["extracted_frames"]
                )
                print(f"        [OK]  {stats['extracted_frames']} frames  "
                      f"({stats['duration_fmt']}, {stats['source_fps']} fps src)  "
                      f"in {stats['elapsed_sec']}s")
                results.append({**stats, "category": "expression",
                                 "position": pos_slug, "behavior": behavior})
            except Exception as exc:
                print(f"        [ERR] Error: {exc}")
                results.append({"video": str(video_file), "error": str(exc)})

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Summary / report
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    ok      = [r for r in results if "error" not in r]
    errors  = [r for r in results if "error" in r]

    total_frames   = sum(r.get("extracted_frames", 0) for r in ok)
    total_duration = sum(r.get("duration_sec",     0) for r in ok)

    print("-" * 60)
    print("  EXTRACTION SUMMARY")
    print("-" * 60)
    print(f"  Videos processed  : {len(ok)}")
    print(f"  Errors            : {len(errors)}")
    print(f"  Total frames      : {total_frames:,}")
    print(f"  Total video time  : {format_duration(total_duration)}")
    print(f"  Output dir        : {OUTPUT_ROOT.relative_to(PROJECT_ROOT)}")

    # Per-behavior breakdown
    print("-" * 60)
    behavior_totals: dict[str, int] = {}
    for r in ok:
        b = r.get("behavior", r.get("category", "unknown"))
        behavior_totals[b] = behavior_totals.get(b, 0) + r.get("extracted_frames", 0)

    if behavior_totals:
        print("\n  Frames per class:")
        for b, count in sorted(behavior_totals.items()):
            bar = "|" * min(30, count // max(1, total_frames // 300))
            print(f"    {b:20s} {count:6,}  {bar}")

    if errors:
        print("\n  Errors:")
        for r in errors:
            print(f"    [ERR] {r['video']}: {r.get('error')}")
    print("-" * 60)


def save_report(results: list[dict], fps: float, quality: int) -> None:
    report = {
        "extraction_settings": {"target_fps": fps, "jpeg_quality": quality},
        "results": results,
    }
    report_path = OUTPUT_ROOT / "extraction_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  [DONE] Report saved -> {report_path.relative_to(PROJECT_ROOT)}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EduVision — Extract frames from collected video data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--fps", type=float, default=2.0,
        help="Frames per second to extract (default: 2.0)"
    )
    parser.add_argument(
        "--quality", type=int, default=85,
        help="JPEG quality 1–100 (default: 85)"
    )
    parser.add_argument(
        "--wide-only", action="store_true",
        help="Only process Wide Shot videos"
    )
    parser.add_argument(
        "--expr-only", action="store_true",
        help="Only process Expression Data videos"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("-" * 60)
    print("  EduVision -- Frame Extraction Tool")
    print("-" * 60)
    print(f"  Source   : {DATA_COLLECTION.relative_to(PROJECT_ROOT)}")
    print(f"  Output   : {OUTPUT_ROOT.relative_to(PROJECT_ROOT)}")
    print(f"  FPS      : {args.fps}")
    print(f"  Quality  : {args.quality}")
    print()

    if not DATA_COLLECTION.exists():
        print(f"[ERROR] Data Collection folder not found: {DATA_COLLECTION}")
        sys.exit(1)

    all_results: list[dict] = []
    t_total = time.time()

    if not args.expr_only:
        print("-- Wide Shot --------------------------------------------------")
        all_results.extend(process_wide_shot(args.fps, args.quality))

    if not args.wide_only:
        print("\n-- Expression Data --------------------------------------------")
        all_results.extend(process_expression(args.fps, args.quality))

    elapsed = time.time() - t_total
    print(f"  Total time: {elapsed:.1f}s")

    print_summary(all_results)
    save_report(all_results, args.fps, args.quality)


if __name__ == "__main__":
    main()
