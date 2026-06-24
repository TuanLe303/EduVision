"""Convert sampled YOLO behavior labels to sparse evaluator JSONL."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2


CLASS_NAMES = (
    "focused",
    "drowsy",
    "sleeping",
    "using_phone",
    "off_task",
    "side_talking",
    "raising_hand",
)
FRAME_NUMBER = re.compile(r"frame_(\d+)$")


def _xyxy(
    center_x: float,
    center_y: float,
    box_width: float,
    box_height: float,
    image_width: int,
    image_height: int,
) -> list[float]:
    x1 = (center_x - box_width / 2.0) * image_width
    y1 = (center_y - box_height / 2.0) * image_height
    x2 = (center_x + box_width / 2.0) * image_width
    y2 = (center_y + box_height / 2.0) * image_height
    return [
        max(0.0, min(float(image_width), x1)),
        max(0.0, min(float(image_height), y1)),
        max(0.0, min(float(image_width), x2)),
        max(0.0, min(float(image_height), y2)),
    ]


def convert(
    labels_dir: Path,
    video_path: Path,
    output_path: Path,
    sample_fps: float,
) -> dict[str, float | int | str]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if source_fps <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"invalid video metadata: {video_path}")
    if sample_fps <= 0:
        raise ValueError("sample FPS must be positive")

    frame_interval = max(1, round(source_fps / sample_fps))
    records: list[dict[str, object]] = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        match = FRAME_NUMBER.search(label_path.stem)
        if match is None:
            raise ValueError(f"cannot parse sampled frame number: {label_path.name}")
        sampled_index = int(match.group(1))
        frame_index = (sampled_index - 1) * frame_interval + 1
        if frame_index > total_frames:
            raise ValueError(
                f"{label_path.name} maps to frame {frame_index}, beyond video frame count "
                f"{total_frames}; check --sample-fps"
            )

        students: list[dict[str, object]] = []
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{label_path}:{line_number}: expected 5 YOLO fields")
            class_id = int(fields[0])
            if not 0 <= class_id < len(CLASS_NAMES):
                raise ValueError(f"{label_path}:{line_number}: unknown class {class_id}")
            center_x, center_y, box_width, box_height = map(float, fields[1:])
            students.append({
                "state": CLASS_NAMES[class_id],
                "bbox": _xyxy(
                    center_x, center_y, box_width, box_height, width, height
                ),
            })
        records.append({
            "frame_index": frame_index,
            "timestamp": (frame_index - 1) / source_fps,
            "person_count": len(students),
            "person_count_complete": False,
            "box_annotation_complete": False,
            "students": students,
            "annotation_type": "anonymous_bbox_behavior",
            "source_label": label_path.name,
        })

    if not records:
        raise ValueError(f"no .txt labels found in {labels_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "label_files": len(records),
        "source_fps": source_fps,
        "sample_fps": sample_fps,
        "frame_interval": frame_interval,
        "width": width,
        "height": height,
        "first_frame_index": int(records[0]["frame_index"]),
        "last_frame_index": int(records[-1]["frame_index"]),
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert sampled YOLO behavior labels to sparse ground-truth JSONL."
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    args = parser.parse_args()
    summary = convert(args.labels, args.video, args.output, args.sample_fps)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
