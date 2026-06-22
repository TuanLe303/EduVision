"""
EduVision -- GitHub Push Script
=================================
Handles pushing new code files to GitHub, correctly dealing with the
case where the project was downloaded as a ZIP (not cloned via git).

What this script does:
  1. Clones the remote repo to a temp folder
  2. Identifies new/modified files in the local (ZIP-downloaded) copy
  3. Copies them to the cloned repo (skipping data, models, venv)
  4. Creates a new branch (default: 'data-pipeline')
  5. Commits and pushes

Files that will NEVER be pushed (per .gitignore + safety):
  - data/          (all data files)
  - *.pt / *.pth   (model weights)
  - ev/            (virtual environment)
  - Data Collection/  (raw videos)

Usage:
    python tools/push_to_github.py --token ghp_YOUR_TOKEN
    python tools/push_to_github.py --token ghp_... --branch my-branch
    python tools/push_to_github.py --dry-run   # preview only, no push
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GITHUB_REPO  = "https://github.com/TuanLe303/EduVision.git"

# Folders/patterns to NEVER copy to the GitHub repo
EXCLUDE_DIRS = {
    "data", "ev", ".git", "__pycache__",
    "Data Collection",          # raw videos
    "yolo11n.pt",               # downloaded model weights
}
EXCLUDE_SUFFIXES = {
    ".pt", ".pth", ".onnx", ".pkl", ".h5", ".bin", ".weights",
    ".mov", ".MOV", ".mp4", ".MP4", ".avi",
}
EXCLUDE_NAMES = {
    ".DS_Store", "Thumbs.db", "desktop.ini",
    "yolo11n.pt", "yolo11s.pt",
}


# ─────────────────────────────────────────────────────────────────────────────

def run(cmd: list[str], cwd: Path = None, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command, print it, raise on error."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None,
        capture_output=capture, text=True
    )
    if result.returncode != 0 and not capture:
        print(f"  [ERROR] Command failed with code {result.returncode}")
        sys.exit(1)
    return result


def find_git() -> str:
    """Find git executable on Windows (checks common install paths)."""
    candidates = [
        "git",
        r"C:\Program Files\Git\bin\git.exe",
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files (x86)\Git\bin\git.exe",
    ]
    for c in candidates:
        try:
            result = subprocess.run(
                [c, "--version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                return c
        except FileNotFoundError:
            continue
    return None


def collect_local_files(source: Path) -> list[Path]:
    """Collect all code files from the local ZIP-extracted project."""
    files = []
    for f in sorted(source.rglob("*")):
        if not f.is_file():
            continue
        # Check excluded dirs
        parts = set(f.relative_to(source).parts)
        if parts & EXCLUDE_DIRS:
            continue
        if f.name in EXCLUDE_NAMES:
            continue
        if f.suffix in EXCLUDE_SUFFIXES:
            continue
        files.append(f)
    return files


def copy_to_clone(files: list[Path], source: Path, dest: Path) -> list[Path]:
    """Copy files from local project to cloned repo, returning list of copied files."""
    copied = []
    for f in files:
        rel = f.relative_to(source)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        copied.append(rel)
    return copied


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push EduVision code to GitHub (handles ZIP download case)."
    )
    parser.add_argument(
        "--token", required=False,
        help="GitHub Personal Access Token (ghp_...)"
    )
    parser.add_argument(
        "--branch", default="data-pipeline",
        help="New branch name to create (default: data-pipeline)"
    )
    parser.add_argument(
        "--repo", default=GITHUB_REPO,
        help=f"GitHub repo URL (default: {GITHUB_REPO})"
    )
    parser.add_argument(
        "--user", default="",
        help="GitHub username (needed if token auth)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be pushed without actually pushing"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("-" * 60)
    print("  EduVision -- GitHub Push Tool")
    print("-" * 60)
    print(f"  Source  : {PROJECT_ROOT}")
    print(f"  Repo    : {args.repo}")
    print(f"  Branch  : {args.branch}")
    print(f"  Mode    : {'DRY RUN' if args.dry_run else 'PUSH'}")
    print()

    # Find git
    git = find_git()
    if not git:
        print("[ERROR] Git not found. Please install Git for Windows:")
        print("  https://git-scm.com/download/win")
        sys.exit(1)
    print(f"  Git: {git}")

    # Collect local files
    local_files = collect_local_files(PROJECT_ROOT)
    print(f"  Local code files to sync: {len(local_files)}")

    if args.dry_run:
        print("\n  DRY RUN -- Files that would be pushed:")
        for f in local_files:
            rel = f.relative_to(PROJECT_ROOT)
            print(f"    {rel}")
        print(f"\n  Total: {len(local_files)} files")
        print(f"  Branch: {args.branch}")
        return

    if not args.token:
        print("\n[ERROR] --token is required. Get one at:")
        print("  https://github.com/settings/tokens")
        print("  (Scopes needed: repo)")
        sys.exit(1)

    # Build authenticated URL
    repo_url = args.repo
    if "github.com" in repo_url and args.token:
        # Insert token: https://TOKEN@github.com/...
        repo_url = repo_url.replace("https://", f"https://oauth2:{args.token}@")

    # Clone into temp directory
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "EduVision"
        print(f"\n  Cloning {args.repo} ...")
        run([git, "clone", repo_url, str(tmp_path)])

        # Configure git identity (needed for commit)
        run([git, "config", "user.email", "eduvision-bot@local"], cwd=tmp_path)
        run([git, "config", "user.name",  "EduVision Bot"],        cwd=tmp_path)

        # Check existing branches
        result = run([git, "branch", "-r"], cwd=tmp_path, capture=True)
        remote_branches = result.stdout.strip()
        print(f"\n  Remote branches:\n{remote_branches}")

        # Create new branch
        print(f"\n  Creating branch: {args.branch}")
        run([git, "checkout", "-b", args.branch], cwd=tmp_path)

        # Copy local files to clone
        print(f"\n  Copying {len(local_files)} files ...")
        copied = copy_to_clone(local_files, PROJECT_ROOT, tmp_path)
        print(f"  Copied {len(copied)} files OK")

        # Show git status
        print("\n  Git status:")
        run([git, "status", "--short"], cwd=tmp_path)

        # Stage all changes
        run([git, "add", "-A"], cwd=tmp_path)

        # Check if there's anything to commit
        result = run([git, "diff", "--cached", "--stat"], cwd=tmp_path, capture=True)
        if not result.stdout.strip():
            print("\n  Nothing to commit — local files are identical to remote.")
            print("  No push needed.")
            return

        print(f"\n  Staged changes:\n{result.stdout[:2000]}")

        # Commit
        commit_msg = (
            f"feat: add data pipeline tools (extract_frames, auto_annotate, "
            f"split_dataset, upload_to_hf)\n\n"
            f"- tools/extract_frames.py: Extract frames from .MOV videos at configurable FPS\n"
            f"- tools/auto_annotate.py: Auto-annotate with YOLO11n person detection\n"
            f"- tools/split_dataset.py: Stratified train/val/test split\n"
            f"- tools/upload_to_hf.py: Upload dataset to HuggingFace Hub\n"
            f"- tools/README.md: Pipeline documentation\n\n"
            f"Dataset (frames + annotations) stored separately on HuggingFace:\n"
            f"  https://huggingface.co/datasets/annghoang/EduVision"
        )
        run([git, "commit", "-m", commit_msg], cwd=tmp_path)

        # Push
        print(f"\n  Pushing to origin/{args.branch} ...")
        run([git, "push", "-u", "origin", args.branch], cwd=tmp_path)

        print("\n" + "-" * 60)
        print("  Push completed!")
        print(f"  Branch: {args.branch}")
        print(f"  View:   https://github.com/TuanLe303/EduVision/tree/{args.branch}")
        print("  Next:   Open a Pull Request on GitHub to merge into main")
        print("-" * 60)


if __name__ == "__main__":
    main()
