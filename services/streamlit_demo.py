"""Local Streamlit harness for the EduVision pipeline up to report generation."""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import time

import cv2
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.report_generator.src.aggregator import aggregate_jsonl
from services.vision_ai.src.main import VisionPipeline, annotate_frame, build_parser

OUTPUT_DIR = ROOT / "outputs" / "streamlit"
DEFAULT_BEHAVIOR_WEIGHT = ROOT / "weights" / "behavior_yolo26n.pt"


def _pipeline_args(
    behavior_model: str,
    detector: str,
    tracker: str,
    person_input_size: int,
    device: str,
    skip_objects: bool,
    enrollment_path: str = "",
    start_class: bool = False,
):
    argv = [
        "--behavior-model", behavior_model,
        "--detector", detector,
        "--tracker", tracker,
        "--person-input-size", str(person_input_size),
        "--device", device,
    ]
    if skip_objects:
        argv.append("--skip-objects")
    if enrollment_path and Path(enrollment_path).is_file():
        argv.extend(["--enrollment-path", enrollment_path])
    if start_class:
        argv.append("--start-class")
    return build_parser().parse_args(argv)


def _save_upload(uploaded_file) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    content = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    digest = hashlib.sha256(content).hexdigest()[:12]
    path = OUTPUT_DIR / f"input_{digest}{suffix}"
    if not path.exists():
        path.write_bytes(content)
    return path


def _run_session(source: int | str, args, max_frames: int, display_every: int) -> None:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        st.error(f"Không mở được nguồn video/camera: {source}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUTPUT_DIR / f"session_{int(time())}.jsonl"
    frame_slot = st.empty()
    metrics_slot = st.empty()
    progress = st.progress(0.0 if max_frames else 0.01)
    pipeline = None
    processed = 0
    started_at = time()

    try:
        with st.spinner("Đang nạp model và khởi tạo pipeline..."):
            pipeline = VisionPipeline(args)
        with jsonl_path.open("w", encoding="utf-8") as output:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                processed += 1
                media_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                timestamp = time() if isinstance(source, int) else max(0.0, media_ms / 1000.0)
                result = pipeline.process_frame(frame, processed, timestamp)
                output.write(json.dumps(result, ensure_ascii=False) + "\n")

                if processed == 1 or processed % display_every == 0:
                    annotated = annotate_frame(frame, result)
                    frame_slot.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True,
                    )
                    metrics_slot.caption(
                        f"Frame {processed:,} · {len(result['tracks'])} track · "
                        f"{len(result['final_behavior'])} trạng thái đã ổn định"
                    )
                if max_frames:
                    progress.progress(min(processed / max_frames, 1.0))
                    if processed >= max_frames:
                        break
    except Exception as exc:
        st.exception(exc)
        return
    finally:
        capture.release()
        if pipeline is not None:
            pipeline.reset()

    if processed == 0:
        st.warning("Nguồn không trả về frame nào.")
        return

    summary = aggregate_jsonl(jsonl_path)
    summary_dict = asdict(summary)
    summary_path = jsonl_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    st.success(
        f"Hoàn tất {processed:,} frame trong {time() - started_at:.1f}s. "
        "Pipeline đã dừng tại session summary, chưa gọi report generator."
    )
    cols = st.columns(3)
    stats = summary.class_stats
    cols[0].metric("Tracks đủ dữ liệu", stats.total_students)
    cols[1].metric("Attention trung bình", f"{stats.avg_attention_score:.1%}")
    cols[2].metric("Thời lượng media", f"{stats.duration_seconds:.1f}s")
    st.subheader("Pre-report session summary")
    st.json(summary_dict)
    st.download_button(
        "Tải frame events (.jsonl)",
        jsonl_path.read_bytes(),
        file_name=jsonl_path.name,
        mime="application/x-ndjson",
    )
    st.download_button(
        "Tải session summary (.json)",
        summary_path.read_bytes(),
        file_name=summary_path.name,
        mime="application/json",
    )


def main() -> None:
    st.set_page_config(page_title="EduVision E2E", layout="wide")
    st.title("EduVision · E2E pipeline test")
    st.caption("Video/camera → tracking → behavior → temporal aggregation → pre-report summary")

    with st.sidebar:
        st.header("Pipeline")
        behavior_model = st.text_input("Behavior weight", str(DEFAULT_BEHAVIOR_WEIGHT))
        detector = st.selectbox("Person detector", ["yolo26n", "yolo11n", "yolo26s", "yolo11s"])
        tracker = st.selectbox("Tracker", ["bytetrack_classroom", "bytetrack", "botsort"])
        person_input_size = st.select_slider(
            "Person detector input size", options=[640, 960, 1280], value=1280
        )
        device = st.selectbox("Device", ["auto", "cpu", "cuda:0"])
        skip_objects = st.checkbox(
            "Bỏ qua off-task object detector", value=True,
            help="Nhanh hơn; behavior weight vẫn nhận diện using_phone nếu model đã học lớp này.",
        )
        enrollment_path = st.text_input("Enrollment file path (cho điểm danh)", "data/enrollments.json")
        start_class = st.checkbox("Bắt đầu lớp học (bật Face Recognition)", value=True)
        max_frames = st.number_input(
            "Giới hạn frame (0 = hết video)", min_value=0, value=300, step=30
        )
        display_every = st.number_input(
            "Cập nhật preview mỗi N frame", min_value=1, value=3, step=1
        )

    source_kind = st.radio("Nguồn input", ["Video upload", "Camera local"], horizontal=True)
    source: int | str | None = None
    if source_kind == "Video upload":
        uploaded = st.file_uploader("Chọn video", type=["mp4", "avi", "mov", "mkv", "webm"])
        if uploaded is not None:
            source = str(_save_upload(uploaded))
    else:
        source = int(st.number_input("Camera index", min_value=0, value=0, step=1))
        st.info("Camera được mở trên chính máy đang chạy Streamlit.")

    weight_ok = Path(behavior_model).is_file()
    if not weight_ok:
        st.error(f"Không tìm thấy behavior weight: {behavior_model}")
    if st.button("Chạy E2E", type="primary", disabled=source is None or not weight_ok):
        args = _pipeline_args(
            behavior_model, detector, tracker, int(person_input_size), device, skip_objects, enrollment_path, start_class
        )
        _run_session(source, args, int(max_frames), int(display_every))


if __name__ == "__main__":
    main()
