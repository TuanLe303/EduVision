from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "services" / "behavior_detection" / "train.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the full-frame YOLO behavior detector.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = _load_yaml(args.config)
    dataset_path = _project_path(config["dataset"])
    _validate_dataset_config(dataset_path)

    from ultralytics import YOLO

    model = YOLO(str(config.get("pretrained_model", "yolo11n.pt")))
    device = config.get("device", "auto")
    model.train(
        data=str(dataset_path),
        epochs=int(config.get("epochs", 100)),
        batch=int(config.get("batch", 16)),
        imgsz=int(config.get("input_size", 640)),
        patience=int(config.get("patience", 20)),
        workers=int(config.get("workers", 4)),
        device=None if device == "auto" else device,
        project=str(_project_path(config.get("project", "runs/behavior_detection"))),
        name=str(config.get("run_name", "behavior_yolo")),
    )

    best_weight = Path(model.trainer.best)
    production_weight = _project_path(config.get("production_weight", "models/behavior_yolo.pt"))
    production_weight.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weight, production_weight)
    return 0


def _load_yaml(path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    if not resolved.is_file():
        raise FileNotFoundError(f"Training config not found: {resolved}")
    with resolved.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError("Training config must be a YAML mapping")
    return value


def _validate_dataset_config(path: Path) -> None:
    dataset = _load_yaml(path)
    names = dataset.get("names", {})
    labels = list(names.values()) if isinstance(names, dict) else list(names)
    expected = ["focused", "drowsy", "sleeping", "using_phone", "off_task", "side_talking"]
    if labels != expected:
        raise ValueError(f"Behavior dataset labels must be exactly {expected}, got {labels}")
    dataset_root = _project_path(dataset.get("path", "datasets/behavior"))
    for split in ("train", "val"):
        relative = dataset.get(split)
        if not relative or not (dataset_root / relative).exists():
            raise FileNotFoundError(f"Behavior dataset split not found: {dataset_root / str(relative)}")


def _project_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
