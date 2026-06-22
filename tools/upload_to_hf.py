"""
EduVision -- HuggingFace Dataset Upload Script
================================================
Uploads the processed dataset (frames + annotations + splits) to HuggingFace.

What gets uploaded:
    data/raw_frames/      -> EduVision/raw_frames/
    data/annotated/       -> EduVision/annotated/
    data/dataset/         -> EduVision/dataset/

What does NOT get uploaded:
    Data Collection/      -> Raw videos (too large, already extracted)
    *.pt model weights    -> Not data

Usage:
    python tools/upload_to_hf.py --token YOUR_TOKEN
    python tools/upload_to_hf.py --token YOUR_TOKEN --repo annghoang/EduVision
    python tools/upload_to_hf.py --dry-run   # preview what would be uploaded
"""

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UPLOAD_FOLDERS = [
    ("data/raw_frames",  "raw_frames"),   # local_path -> hf_path
    ("data/annotated",   "annotated"),
    ("data/dataset",     "dataset"),
]

# Files/patterns to skip
SKIP_SUFFIXES = {".mov", ".mp4", ".avi", ".MOV", ".MP4", ".pt", ".pth", ".onnx"}
SKIP_NAMES    = {"__pycache__", ".DS_Store", "Thumbs.db"}


# ─────────────────────────────────────────────────────────────────────────────

def collect_files(local_root: Path) -> list[tuple[Path, str]]:
    """
    Collect all uploadable files under local_root.
    Returns list of (local_path, relative_path_for_hf).
    """
    files = []
    for f in sorted(local_root.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix in SKIP_SUFFIXES:
            continue
        if any(part in SKIP_NAMES for part in f.parts):
            continue
        rel = f.relative_to(local_root.parent)  # relative to data/
        files.append((f, str(rel).replace("\\", "/")))
    return files


def dry_run(pairs: list[tuple[Path, str, str]]) -> None:
    """Print what would be uploaded without doing it."""
    print("\n  DRY RUN -- Files that would be uploaded:")
    print(f"  {'Local file':<60} -> HF path")
    print("  " + "-" * 90)
    total_mb = 0
    for local, hf_path, _ in pairs:
        size_mb = local.stat().st_size / 1_048_576
        total_mb += size_mb
        rel = str(local.relative_to(PROJECT_ROOT))
        print(f"  {rel:<60} -> {hf_path}  ({size_mb:.2f} MB)")
    print(f"\n  Total: {len(pairs)} files, {total_mb:.1f} MB")


def upload(api: HfApi, repo_id: str, pairs: list[tuple[Path, str, str]]) -> None:
    """Upload all files with progress reporting."""
    total = len(pairs)
    ok = 0
    errors = []

    for i, (local, hf_path, folder_label) in enumerate(pairs, 1):
        size_mb = local.stat().st_size / 1_048_576
        print(f"  [{i:3d}/{total}] {folder_label}/{local.name:<40} ({size_mb:.2f} MB)", end=" ")
        try:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=hf_path,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Upload {hf_path}",
            )
            print("OK")
            ok += 1
        except Exception as e:
            print(f"ERROR: {e}")
            errors.append((hf_path, str(e)))

    print(f"\n  Uploaded: {ok}/{total} files")
    if errors:
        print(f"  Errors ({len(errors)}):")
        for path, err in errors:
            print(f"    {path}: {err}")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload EduVision dataset to HuggingFace Hub."
    )
    parser.add_argument(
        "--token", required=False,
        help="HuggingFace write token (hf_...)"
    )
    parser.add_argument(
        "--repo", default="annghoang/EduVision",
        help="HuggingFace repo ID (default: annghoang/EduVision)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview files without uploading"
    )
    parser.add_argument(
        "--folder", choices=["raw_frames", "annotated", "dataset", "all"],
        default="all",
        help="Which folder to upload (default: all)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("-" * 60)
    print("  EduVision -- HuggingFace Upload")
    print("-" * 60)
    print(f"  Repo   : {args.repo}")
    print(f"  Mode   : {'DRY RUN' if args.dry_run else 'UPLOAD'}")
    print()

    # Filter folders to upload
    folders = UPLOAD_FOLDERS
    if args.folder != "all":
        folders = [(l, h) for l, h in UPLOAD_FOLDERS if h == args.folder]

    # Collect all files
    all_pairs = []  # (local_path, hf_path, folder_label)
    for local_rel, hf_prefix in folders:
        local_dir = PROJECT_ROOT / local_rel
        if not local_dir.exists():
            print(f"  [WARN] Folder not found, skipping: {local_rel}")
            continue
        files = collect_files(local_dir)
        for local, rel in files:
            hf_path = rel  # keeps data/raw_frames/... structure under repo root
            all_pairs.append((local, hf_path, hf_prefix))
        print(f"  [{hf_prefix}]  {len(files)} files found")

    if not all_pairs:
        print("\n[ERROR] No files found to upload.")
        sys.exit(1)

    total_mb = sum(f.stat().st_size / 1_048_576 for f, _, _ in all_pairs)
    print(f"\n  Total: {len(all_pairs)} files, {total_mb:.1f} MB\n")

    if args.dry_run:
        dry_run(all_pairs)
        return

    if not args.token:
        print("[ERROR] --token is required for upload. Get it from:")
        print("  https://huggingface.co/settings/tokens")
        print("  (Create a token with 'Write' permission)")
        sys.exit(1)

    # Login
    print(f"  Logging in to HuggingFace...")
    try:
        login(token=args.token, add_to_git_credential=False)
        print("  Login OK\n")
    except Exception as e:
        print(f"  [ERROR] Login failed: {e}")
        sys.exit(1)

    api = HfApi()

    print(f"  Uploading {len(all_pairs)} files to {args.repo} ...\n")
    upload(api, args.repo, all_pairs)

    print(f"\n  Done! View at: https://huggingface.co/datasets/{args.repo}")
    print("-" * 60)


if __name__ == "__main__":
    main()
