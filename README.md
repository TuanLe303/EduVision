# EduVision — AI Classroom Monitoring System

> An intelligent classroom monitoring system that uses computer vision and AI to automate attendance, track student engagement, analyze behaviors, and generate post-session reports.

---

## 1. Project Name

**EduVision**

---

## 2. Short Description

EduVision is an AI-powered classroom monitoring pipeline built for higher education environments. A camera mounted at an elevated position in the classroom feeds live video into the system, which automatically identifies students, tracks their positions, analyzes behavioral states (focused, drowsy, using phone, away from seat), monitors basic instructor activity, and produces a structured session report using a large language model (LLM). The goal is not to replace human judgment but to provide objective, data-driven observational support for instructors and administrators.

---

## 3. Key Features

| Feature | Description |
|---|---|
| **Automatic Attendance** | Identifies students via face recognition and logs entry/exit times |
| **Student Tracking** | Multi-object tracking assigns a persistent ID to each student throughout the session |
| **Behavior Analysis** | Classifies per-student states: focused, drowsy, using phone, absent from seat, side-talking |
| **Instructor Monitoring** | Tracks basic instructor movement and presence at the front of the class |
| **Session Statistics** | Aggregates per-student and class-wide engagement metrics over time |
| **LLM Report Generation** | Generates a human-readable session summary using Gemini or GPT (selectable at runtime) |
| **Dashboard / Notification** | Exposes metrics via a web dashboard and supports alert notifications |

---

## 4. System Architecture

```mermaid
flowchart TD
    CAM["IP Camera / Video File"]
    VC["Video Connection"]
    CAM --> VC

    subgraph VA["Vision AI Service"]
        direction TB
        PD["1. Person Detection\nyolo11n · yolo11s"]
        MOT["2. Multi-object Tracking\nByteTrack · BoT-SORT"]
        FD["3. Face Detection\nSCRFD · RetinaFace"]
        FR["4. Face Recognition / Attendance\nbuffalo_s · buffalo_l · ArcFace"]
        OD["5. Object Detection — off-task\nphone · laptop · book · bottle"]
        BD["6. Behavior Detection\nCustom YOLO\nfocused · drowsy · sleeping\nusing_phone · off_task · side_talking"]
        TA["7. Temporal Aggregation\nPer-track window · hysteresis"]
        SM["8. Seat Monitor\nFixed camera · identity · seat ROI\naway_from_seat"]
        BA["9. Final Behavior\nTemporal behavior + seat priority"]

        PD --> MOT
        MOT --> FD
        FD --> FR
        MOT --> OD
        MOT --> BD
        MOT --> SM
        FR --> SM
        OD --> BD
        BD --> TA
        TA --> BA
        SM --> BA
    end

    VC --> VA

    BA --> API

    subgraph BACK["Backend"]
        API["Backend API\nFastAPI + PostgreSQL"]
        RG["Report Generator\nGemini · GPT"]
        API --> RG
    end

    API --> FE["Web Dashboard\nREST API · WebSocket · Notifications"]
    RG  --> FE
```

The system is organized as **independent microservices** that communicate via REST API or message queues:

- **video_connection** — camera/stream ingestion and frame distribution
- **vision_ai** — core CV pipeline: detection → tracking → face recognition → object detection → behavior detection → temporal aggregation → seat monitoring
- **backend_api** — persists events to the database and exposes a REST API
- **report_generator** — queries the database and calls an LLM to produce session reports
- **frontend** — web dashboard for live monitoring and historical reports

---

## 5. Technologies Used

Each CV module supports multiple interchangeable backends. The active backend is selected via config or CLI flag, so the system can be tuned to match available hardware — from a CPU-only laptop to a GPU workstation.

### 5.1 Person Detection

| Option | Model | Notes |
|---|---|---|
| **`yolo11n`** *(default)* | YOLOv11-nano | Fastest, lowest VRAM — suitable for CPU or low-end GPU |
| **`yolo11s`** | YOLOv11-small | Better accuracy, moderate GPU recommended |

Selected via `--detector yolo11n` or `--detector yolo11s`.

### 5.2 Multi-object Tracking

| Option | Algorithm | Notes |
|---|---|---|
| **`bytetrack`** *(default)* | [ByteTrack](https://github.com/ifzhang/ByteTrack) | Fast, robust; start here |
| **`botsort`** | [BoT-SORT](https://github.com/NirAharon/BoT-SORT) | Better ID consistency when ByteTrack produces ID switches |

Selected via `--tracker bytetrack` or `--tracker botsort`. Switch to BoT-SORT only if frequent ID loss is observed.

### 5.3 Face Detection

| Option | Model | Notes |
|---|---|---|
| **`scrfd`** *(default)* | SCRFD (InsightFace) | Fast, lightweight, good for small faces from overhead camera |
| **`retinaface`** | RetinaFace (InsightFace) | More accurate, higher compute cost |

Selected via `--face-detector scrfd` or `--face-detector retinaface`.

### 5.4 Face Recognition / Attendance

| Option | Model | Notes |
|---|---|---|
| **`buffalo_s`** *(default)* | InsightFace `buffalo_s` | Lightweight, suitable for real-time on modest GPU |
| **`buffalo_l`** | InsightFace `buffalo_l` | Higher accuracy, requires more VRAM |

The recognition backend is selected with `--recognizer insightface`; choose the
model pack with `--recognition-model buffalo_s` or `--recognition-model buffalo_l`.

### 5.5 Object Detection (Off-task Behavior)

Runs a COCO object detector for off-task objects and associates each object with
the canonical student track. A detected phone can override the frame-level
behavior prediction for that track before temporal aggregation.

**Tracked object classes:**

| Category | Objects |
|---|---|
| Electronic devices | phone, laptop, tablet, earphone |
| Study materials | book, pen, paper |
| Other | backpack, food, bottle |

### 5.6 Behavior Detection and Temporal Aggregation

A custom YOLO detection model predicts one visual behavior box per visible
student. Behavior boxes are matched to canonical person tracks, then aggregated
independently per `track_id` using a temporal window, confidence thresholds,
priority rules, and hysteresis.

| Input signal | Source module |
|---|---|
| Visual behavior box | Custom behavior YOLO |
| Phone/object evidence | Object Detection |
| Track ID and person bbox | Multi-object Tracking |
| Prediction history | Temporal Aggregator |

**Output behavior states:**

| State | Trigger signals |
|---|---|
| `focused` | Sustained focused prediction |
| `drowsy` | Sustained drowsy prediction |
| `sleeping` | Sustained sleeping prediction |
| `using_phone` | Sustained model prediction or associated phone detection |
| `off_task` | Sustained off-task prediction |
| `side_talking` | Sustained side-talking prediction |
| `away_from_seat` | Fixed-camera seat monitor confirms the student elsewhere while the assigned seat remains empty |

Behavior thresholds are configured in
`configs/services/behavior_detection/yolo_behavior.yaml`.

### 5.7 Seat Monitoring / Attendance

At class start, recognized tracks calibrate a fixed seat ROI for each student.
Leaving the seat requires sustained spatial displacement, an empty assigned seat,
and face recognition confirming the same student at the new location. Missing or
occluded detections alone are not treated as `away_from_seat`.

---

### 5.8 Backend & Data

| Component | Technology |
|---|---|
| Backend Framework | [FastAPI](https://fastapi.tiangolo.com/) |
| Database | PostgreSQL (production) / SQLite (development) |
| Deep Learning Framework | PyTorch |
| Image Processing | OpenCV, Pillow |

### 5.9 LLM Report Generation

| Provider | Notes |
|---|---|
| **Google Gemini** | Selected via `--llm gemini` at runtime |
| **OpenAI GPT** | Selected via `--llm gpt` at runtime |

### 5.10 Frontend

| Component | Technology |
|---|---|
| Web Dashboard | React (planned) |
| Real-time updates | WebSocket |

### 5.11 Infrastructure

| Component | Technology |
|---|---|
| Environment | Python 3.10+, virtualenv |
| Containerization | Docker + Docker Compose (planned) |

---

## 6. Installation

### Prerequisites

- Python 3.10 or higher
- Git
- (Optional) NVIDIA GPU with CUDA for accelerated inference

### Steps

```powershell
# Install uv once, then create .venv and install the locked base environment.
uv sync

# Run tests.
uv run pytest

# Run the behavior pipeline without the optional face stack.
uv run python -m services.vision_ai.src.main --source classroom.mp4 `
  --behavior-model models/behavior_yolo.pt
```

Face recognition and identity-based seat monitoring are optional because the
PyPI InsightFace package must compile native code on Windows. Install Microsoft
Visual C++ Build Tools 14+ first, then use:

```powershell
uv sync --extra face
uv run python -m services.vision_ai.src.main --source classroom.mp4 `
  --behavior-model models/behavior_yolo.pt `
  --enrollment-path data/enrollments.json --start-class
```

```bash
# 1. Clone the repository
git clone <repository-url>
cd EduVision

# 2. Create and activate virtual environment
python -m venv ev

# Windows
ev\Scripts\activate

# macOS / Linux
source ev/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and configure environment variables
cp configs/.env.example configs/.env
# Edit configs/.env — set your database URL, LLM API keys, camera source, etc.
```

### Environment Variables

| Variable | Description | Example |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini API key | `AIza...` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `DATABASE_URL` | Database connection string | `postgresql://user:pass@localhost/eduvision` |
| `CAMERA_SOURCE` | Video source (file path or RTSP URL) | `0` or `rtsp://...` |
| `LLM_PROVIDER` | Default LLM provider | `gemini` or `gpt` |

---

## 7. Running the Demo

> **Note:** Full demo is currently under development. The steps below describe the intended workflow.

```bash
# Activate virtual environment first
ev\Scripts\activate   # Windows
source ev/bin/activate  # macOS / Linux

# Start the backend API
python -m services.backend_api.main

# Start the vision AI pipeline with a video file
python -m services.vision_ai.src.main --source path/to/video.mp4

# Start the vision AI pipeline with a webcam (device index 0)
python -m services.vision_ai.src.main --source 0

# Generate a session report using Gemini
python -m services.report_generator.main --session-id <SESSION_ID> --llm gemini

# Generate a session report using GPT
python -m services.report_generator.main --session-id <SESSION_ID> --llm gpt

# Start the web dashboard
python -m services.frontend.main
# Open http://localhost:3000 in your browser
```

---

## 8. Directory Structure

```
EduVision/
│
├── configs/                    # Configuration files and environment templates
│   ├── datasets/               # Dataset-specific configs
│   └── models/                 # Model hyperparameter configs
│
├── data/                       # Raw and processed data (gitignored)
│
├── database/                   # Database schema and migration scripts
│
├── docs/                       # Extended documentation
│   └── overview.md
│
├── models/                     # Pretrained model weights (gitignored)
│
├── monitoring/                 # System health monitoring configs
│
├── services/                   # Microservices
│   ├── video_connection/       # Camera / stream ingestion service
│   ├── vision_ai/              # Core computer vision pipeline
│   │   └── src/
│   │       ├── person_detection/   # YOLO-based person & face detection
│   │       ├── face_recognition/   # Student identification
│   │       ├── tracking/           # Multi-object tracking
│   │       ├── behavior/           # Engagement & behavior analysis
│   │       ├── e2e/                # End-to-end integration tests
│   │       └── main.py             # Pipeline entry point
│   ├── backend_api/            # FastAPI REST backend + database layer
│   ├── report_generator/       # LLM-based report generation (Gemini / GPT)
│   └── frontend/               # Web dashboard
│
├── tools/                      # Utility scripts (data preparation, evaluation)
│
├── ev/                         # Python virtual environment (gitignored)
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

---

## 9. Main APIs

> The API is under development. The endpoints below reflect the planned specification.

### Attendance

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/sessions` | List all recorded sessions |
| `GET` | `/api/sessions/{session_id}` | Get details for a specific session |
| `GET` | `/api/sessions/{session_id}/attendance` | Get attendance records for a session |

### Students

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/students` | List all registered students |
| `POST` | `/api/students` | Register a new student (with face images) |
| `GET` | `/api/students/{student_id}/sessions` | Get session history for a student |

### Behavior Events

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/sessions/{session_id}/events` | Get all behavior events for a session |
| `GET` | `/api/sessions/{session_id}/summary` | Get aggregated behavior statistics |

### Reports

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/reports/generate` | Trigger LLM report generation for a session |
| `GET` | `/api/reports/{session_id}` | Retrieve the generated report |

### Live Stream

| Method | Endpoint | Description |
|---|---|---|
| `WebSocket` | `/ws/stream` | Real-time annotated video stream |
| `WebSocket` | `/ws/events` | Real-time behavior event feed |

Full API documentation (Swagger UI) will be available at `http://localhost:8000/docs` when the backend is running.

---

## 10. Contributing

Contributions are welcome. Please follow these guidelines:

1. **Fork** the repository and create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Commit** your changes with clear, descriptive messages:
   ```bash
   git commit -m "feat: add drowsiness detection module"
   ```

3. **Test** your changes locally before submitting a pull request.

4. **Open a Pull Request** against the `main` branch. Include:
   - What the change does
   - Why it is needed
   - How it was tested

### Commit Message Convention

| Prefix | Use for |
|---|---|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `refactor:` | Code restructuring, no behavior change |
| `test:` | Adding or updating tests |
| `chore:` | Build scripts, configs, tooling |

### Code Style

- Python code follows [PEP 8](https://peps.python.org/pep-0008/)
- Use type hints for all function signatures
- Keep modules focused — one responsibility per file

---

## 11. Developers

| Name | Student ID | Role |
|---|---|---|
| *(to be filled)* | *(to be filled)* | *(to be filled)* |

---

*EduVision — Intelligent Classroom Monitoring, powered by Computer Vision and AI.*
