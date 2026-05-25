# EduVision — System Pipeline

> Hệ thống giám sát và phân tích tương tác lớp học thông minh bằng AI

---

## Luồng xử lý chính (2 dòng ngang)

```
ROW 1:  [INPUT]──────────────► [VIDEO PROCESSING]──────────────► [ANALYSIS]
        Camera 1/2/N              Frame Extraction               ┌─ Điểm danh
             │                    ObjDet (YOLO) ────────────────►├─ Theo dõi Vị trí
             ▼                    Face Recognition ─────────────►├─ Trạng thái SV
        Video Stream                                             └─ Giám sát GV
                                                                       │
                                             ┌─────────────────────────┘
                                             ▼
ROW 2:  [STORAGE]──────────► [AI / LLM]──────────────► [OUTPUT]──► Users
        Database (CSDL)       LLM Analysis Engine       Report      👤 Sinh viên
                                                         Web Portal  👨‍🏫 Giảng viên
                                                                     🏫 Nhà trường
                                                                     🏢 Phòng QL
```

---

## Layer 1 — INPUT `#dae8fc`

| Component | Mô tả |
|---|---|
| Camera 1..N | Camera IP/USB đặt trên cao trong phòng học |
| Video Stream | Luồng video thời gian thực từ tất cả camera |

---

## Layer 2 — VIDEO PROCESSING `#ffe6cc`

| Component | Mô tả |
|---|---|
| Video Ingestion & Frame Extraction | Tiếp nhận luồng video, trích xuất frame |
| Person Det. & Tracking (YOLO) | Phát hiện và theo dõi người (sinh viên, giảng viên) trong khung hình |
| Face Recognition Module | Nhận diện khuôn mặt sinh viên/giảng viên |
| Pose Estimation (YOLO-pose) | Ước lượng tư thế cơ thể sinh viên |
| Head Pose & Gaze (MediaPipe) | Xác định hướng nhìn và độ nghiêng đầu |
| Off-task Object Det (YOLO) | Phát hiện vật dụng ngoài bài học (điện thoại, máy tính...) |

---

## Layer 3 — ANALYSIS `#e1d5e7`

| Component | Mô tả |
|---|---|
| ✅ Điểm danh Thông minh | Ghi nhận trạng thái: có mặt / vắng / đi trễ / về sớm |
| 📍 Theo dõi Vị trí Thời gian thực | Phát hiện sinh viên rời chỗ / đổi chỗ quá thời gian |
| 🧠 Phân tích Trạng thái Sinh viên | Computer Vision: tập trung / buồn ngủ / mất tập trung |
| 👨‍🏫 Giám sát Hoạt động Giảng viên | Giảng bài / tương tác / có mặt / vắng mặt |

---

## Layer 4 — BACKEND & STORAGE `#f5f5f5`

| Component | Mô tả |
|---|---|
| 🔌 Backend API (FastAPI) | Tiếp nhận sự kiện từ các module AI và cung cấp API |
| 🗄️ Database (CSDL) | Lưu trữ toàn bộ dữ liệu tương tác, sự kiện, trạng thái (PostgreSQL/SQLite) |

---

## Layer 5 — AI / LLM `#fff2cc`

| Component | Mô tả |
|---|---|
| 🤖 LLM Analysis Engine | Tổng hợp dữ liệu từ CSDL, chuyển đổi số liệu kỹ thuật thành nhận định chất lượng lớp học |

---

## Layer 6 — OUTPUT `#d5e8d4`

| Component | Mô tả |
|---|---|
| 📄 Report Generation | Tạo báo cáo PDF / Dashboard tổng hợp |
| 🖥️ Cổng Thông tin Quản lý | Web Frontend: sơ đồ lớp, danh sách điểm danh, bảng đánh giá |

---

## Người dùng cuối

| Vai trò | Truy cập |
|---|---|
| 👤 Sinh viên | Cổng Thông tin Quản lý |
| 👨‍🏫 Giảng viên | Cổng Thông tin + Báo cáo |
| 🏫 Nhà trường | Báo cáo tổng hợp |
| 🏢 Phòng quản lý | Dashboard + Báo cáo |

---

## Màu sắc (draw.io)

| Layer | Fill | Stroke |
|---|---|---|
| INPUT | `#dae8fc` | `#6c8ebf` |
| VIDEO PROCESSING | `#ffe6cc` | `#d79b00` |
| ANALYSIS | `#e1d5e7` | `#9673a6` |
| STORAGE | `#f5f5f5` | `#666666` |
| AI / LLM | `#fff2cc` | `#d6b656` |
| OUTPUT | `#d5e8d4` | `#82b366` |
