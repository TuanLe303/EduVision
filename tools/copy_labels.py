import argparse
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Copy YOLO label file to multiple subsequent frames.")
    parser.add_argument("--source", required=True, help="Source label file (e.g., data/annotated/wide_shot/labels/goc_cheo__frame_00001.txt)")
    parser.add_argument("--start", type=int, required=True, help="Start frame number (e.g., 2)")
    parser.add_argument("--end", type=int, required=True, help="End frame number (inclusive) (e.g., 50)")
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"[ERROR] Source file not found: {source_path}")
        return

    # Extract prefix, e.g., 'goc_cheo__frame_00001.txt' -> 'goc_cheo__frame_'
    # Assuming format: {prefix}frame_{number:05d}.txt
    stem = source_path.stem
    if "frame_" not in stem:
        print("[ERROR] Unrecognized filename format. Must contain 'frame_'.")
        return

    prefix = stem.split("frame_")[0] + "frame_"
    parent_dir = source_path.parent

    print(f"Source: {source_path.name}")
    print(f"Targeting frames: {args.start:05d} to {args.end:05d}")
    print("-" * 40)

    count = 0
    for i in range(args.start, args.end + 1):
        target_name = f"{prefix}{i:05d}.txt"
        target_path = parent_dir / target_name
        
        # Only copy if we aren't overwriting the source itself
        if target_path != source_path:
            shutil.copy2(source_path, target_path)
            count += 1

    print(f"[OK] Successfully copied to {count} files.")

if __name__ == "__main__":
    main()
