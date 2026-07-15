"""
EduVision — FastAPI backend.

Start with:
    uvicorn services.backend_api.main:app --reload --host 0.0.0.0 --port 8000

All /api/* routes map exactly to what services/frontend/src/api.js calls.
Avatar images are served at /api/avatars/<filename>.
"""

from __future__ import annotations

import json
import shutil
import traceback
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from services.backend_api.database import Database, _AVATARS_DIR, get_db
from services.backend_api.models import (
    AttendanceOut,
    EnrollRequest,
    EventOut,
    EventsBulkRequest,
    GenerateReportRequest,
    ReportOut,
    SessionOut,
    StartSessionRequest,
    StudentOut,
    SummaryOut,
    StartPipelineRequest,
    PipelineStatusOut,
)

# ---------------------------------------------------------------------------
# Global Pipeline State
# ---------------------------------------------------------------------------
class PipelineManager:
    process: Optional[subprocess.Popen] = None
    current_source: Optional[str] = None

pipeline_mgr = PipelineManager()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EduVision API",
    version="1.0.0",
    description="Backend API for the EduVision classroom monitoring system.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev: allow Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve avatar images statically
_AVATARS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/api/avatars", StaticFiles(directory=str(_AVATARS_DIR)), name="avatars")


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------


def db_dep() -> Database:
    return get_db()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


# ---------------------------------------------------------------------------
# Students  (/api/students)
# ---------------------------------------------------------------------------


@app.get("/api/students", response_model=list[StudentOut])
async def list_students(request: Request, db: Database = Depends(db_dep)):
    base = _base_url(request)
    return [StudentOut.from_db(r, base) for r in db.get_students()]


@app.get("/api/students/{student_id}", response_model=StudentOut)
async def get_student(
    student_id: str, request: Request, db: Database = Depends(db_dep)
):
    row = db.get_student(student_id)
    if not row:
        raise HTTPException(status_code=404, detail="Student not found")
    return StudentOut.from_db(row, _base_url(request))


@app.post("/api/students", response_model=StudentOut, status_code=status.HTTP_201_CREATED)
async def enroll_student(
    request: Request,
    student_id: str = Form(...),
    name: str = Form(...),
    email: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    db: Database = Depends(db_dep),
):
    """
    Register a student and optionally upload a face image.
    The face image is saved to data/avatars/ and used to generate an embedding
    via the vision pipeline's FaceRecognizer (if available).
    """
    # 1. Save avatar image if provided
    avatar_path: Optional[str] = None
    if image and image.filename:
        suffix = Path(image.filename).suffix.lower() or ".jpg"
        dest = _AVATARS_DIR / f"{student_id}{suffix}"
        with dest.open("wb") as f:
            shutil.copyfileobj(image.file, f)
        avatar_path = str(dest)

    # 2. Upsert student record
    db.upsert_student(student_id, name, email, avatar_path)

    # 3. Generate face embedding from uploaded image if possible
    if avatar_path:
        try:
            _enroll_face_from_image(db, student_id, avatar_path)
        except Exception:
            # Embedding is optional — student is still registered
            traceback.print_exc()

    row = db.get_student(student_id)
    return StudentOut.from_db(row, _base_url(request))  # type: ignore[arg-type]


@app.delete("/api/students/{student_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_student(student_id: str, db: Database = Depends(db_dep)):
    if not db.delete_student(student_id):
        raise HTTPException(status_code=404, detail="Student not found")


# ---------------------------------------------------------------------------
# Sessions  (/api/sessions)
# ---------------------------------------------------------------------------


@app.get("/api/sessions", response_model=list[SessionOut])
async def list_sessions(db: Database = Depends(db_dep)):
    return [SessionOut.from_db(r) for r in db.get_sessions()]


@app.get("/api/sessions/{session_id}", response_model=SessionOut)
async def get_session(session_id: int, db: Database = Depends(db_dep)):
    row = db.get_session(session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionOut.from_db(row)


@app.post("/api/sessions/start", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def start_session(body: StartSessionRequest, db: Database = Depends(db_dep)):
    row = db.create_session(body.class_name)
    return SessionOut.from_db(row)


@app.post("/api/sessions/{session_id}/end", response_model=SessionOut)
async def end_session(session_id: int, db: Database = Depends(db_dep)):
    row = db.end_session(session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Session not found or already ended")
    return SessionOut.from_db(row)


@app.get("/api/sessions/{session_id}/attendance", response_model=list[AttendanceOut])
async def get_attendance(session_id: int, db: Database = Depends(db_dep)):
    _require_session(db, session_id)
    return [AttendanceOut.from_db(r) for r in db.get_attendance(session_id)]


@app.get("/api/sessions/{session_id}/events", response_model=list[EventOut])
async def get_events(
    session_id: int,
    student_id: Optional[str] = None,
    limit: int = 200,
    db: Database = Depends(db_dep),
):
    _require_session(db, session_id)
    return [EventOut.from_db(r) for r in db.get_events(session_id, limit, student_id)]


@app.post("/api/sessions/{session_id}/events", status_code=status.HTTP_202_ACCEPTED)
async def push_events(
    session_id: int,
    body: EventsBulkRequest,
    db: Database = Depends(db_dep),
):
    """
    Internal endpoint called by the vision pipeline (event_pusher.py) to
    persist a batch of final_behavior records.
    """
    _require_session(db, session_id)
    inserted = db.log_events_bulk(session_id, body.events)
    return {"inserted": inserted}


@app.get("/api/sessions/{session_id}/summary", response_model=SummaryOut)
async def get_summary(session_id: int, db: Database = Depends(db_dep)):
    _require_session(db, session_id)
    return SummaryOut.from_db(db.get_summary(session_id))


# ---------------------------------------------------------------------------
# Reports  (/api/reports)
# ---------------------------------------------------------------------------


@app.post("/api/reports/generate", response_model=ReportOut)
async def generate_report(
    body: GenerateReportRequest, db: Database = Depends(db_dep)
):
    row = db.get_session(body.session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check if already generated
    existing = db.load_report(body.session_id)
    if existing:
        return ReportOut(**existing)

    # Build report content (summary-based template; swap for LLM call here)
    summary = db.get_summary(body.session_id)
    attendance = db.get_attendance(body.session_id)
    content = _render_report(row, summary, attendance, body.language)

    now = datetime.now(timezone.utc).isoformat()
    report_data = {
        "session_id": body.session_id,
        "provider": body.provider,
        "generated_at": now,
        "content": content,
    }
    db.save_report(body.session_id, report_data)
    return ReportOut(**report_data)


@app.get("/api/reports/{session_id}", response_model=ReportOut)
async def get_report(session_id: int, db: Database = Depends(db_dep)):
    report = db.load_report(session_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return ReportOut(**report)


# ---------------------------------------------------------------------------
# Pipeline Management (/api/pipeline)
# ---------------------------------------------------------------------------

@app.get("/api/pipeline/status", response_model=PipelineStatusOut)
async def get_pipeline_status():
    is_running = False
    pid = None
    if pipeline_mgr.process is not None:
        if pipeline_mgr.process.poll() is None:
            is_running = True
            pid = pipeline_mgr.process.pid
        else:
            pipeline_mgr.process = None
            pipeline_mgr.current_source = None
            
    return PipelineStatusOut(
        is_running=is_running,
        source=pipeline_mgr.current_source,
        pid=pid
    )

@app.post("/api/pipeline/start/{session_id}", response_model=PipelineStatusOut)
async def start_pipeline(session_id: int, body: StartPipelineRequest, db: Database = Depends(db_dep)):
    _require_session(db, session_id)
    
    # Check if already running
    if pipeline_mgr.process is not None and pipeline_mgr.process.poll() is None:
        raise HTTPException(status_code=400, detail="Pipeline is already running")
        
    python_exe = sys.executable
    cmd = [
        python_exe, "-m", "services.video_connection.realtime_demo",
        "--session-id", str(session_id),
        "--source", body.source,
        "--target-fps", str(body.target_fps),
        "--show",  # Pop up OpenCV window for demo
    ]
    
    try:
        pipeline_mgr.process = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).resolve().parents[3]) # Root of EduVision
        )
        pipeline_mgr.current_source = body.source
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start pipeline: {str(e)}")
        
    return PipelineStatusOut(
        is_running=True,
        source=pipeline_mgr.current_source,
        pid=pipeline_mgr.process.pid
    )

@app.post("/api/pipeline/stop", response_model=PipelineStatusOut)
async def stop_pipeline():
    if pipeline_mgr.process is not None:
        if pipeline_mgr.process.poll() is None:
            pipeline_mgr.process.terminate()
            try:
                pipeline_mgr.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pipeline_mgr.process.kill()
        pipeline_mgr.process = None
        pipeline_mgr.current_source = None
        
    return PipelineStatusOut(is_running=False)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_session(db: Database, session_id: int) -> None:
    if not db.get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")


def _enroll_face_from_image(db: Database, student_id: str, image_path: str) -> None:
    """Attempt to generate a face embedding from a saved avatar image."""
    import cv2

    from services.vision_ai.src.face_detection import FaceDetector
    from services.vision_ai.src.face_recognition import FaceRecognizer

    img = cv2.imread(image_path)
    if img is None:
        return

    detector = FaceDetector(backend="scrfd")
    faces = detector.detect(img)
    if not faces:
        return

    best = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
    recognizer = FaceRecognizer(backend="insightface")
    embedding = recognizer.encode(img, landmarks=best.landmarks if len(best.landmarks) == 5 else None)
    db.add_embedding(student_id, embedding.tolist())


def _render_report(
    session: dict,
    summary: dict,
    attendance: list[dict],
    language: str = "vi",
) -> str:
    dist = summary.get("behavior_distribution", {})
    total = summary.get("total_events", 1) or 1
    attn = summary.get("avg_attention_pct", 0.0)
    n_students = len(attendance)

    def pct(state: str) -> float:
        return round(100.0 * dist.get(state, 0) / total, 1)

    student_lines = "\n".join(
        f"- **{r['name'] or r['student_id']}** ({r['student_id']}): "
        f"tham gia {r.get('duration_min', 0):.0f} phút, {r.get('event_count', 0)} sự kiện"
        for r in attendance
        if r.get("student_id")
    )

    return f"""# Báo cáo Phiên Học — {session['class_name']}

## Tổng quan
Phiên học bắt đầu lúc **{session['start_time'][:16].replace('T', ' ')}**, kết thúc lúc **{(session.get('end_time') or 'đang diễn ra')[:16].replace('T', ' ')}**.
Có **{n_students} sinh viên** được ghi nhận. Tỷ lệ chú ý trung bình: **{attn:.1f}%**.

## Điểm danh
{student_lines or '_Không có dữ liệu điểm danh._'}

## Phân tích hành vi
- **Tập trung (focused)**: {pct('focused')}%
- **Buồn ngủ (drowsy)**: {pct('drowsy')}%
- **Ngủ (sleeping)**: {pct('sleeping')}%
- **Dùng điện thoại (using_phone)**: {pct('using_phone')}%
- **Không tập trung (off_task)**: {pct('off_task')}%
- **Nói chuyện riêng (side_talking)**: {pct('side_talking')}%
- **Giơ tay (raising_hand)**: {pct('raising_hand')}%
- **Rời chỗ (away_from_seat)**: {pct('away_from_seat')}%

## Tổng số sự kiện được ghi nhận
{total} sự kiện hành vi từ {n_students} sinh viên.
"""
