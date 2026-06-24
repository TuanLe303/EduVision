// Mock data — used automatically when backend is unreachable

export const MOCK_STUDENTS = [
  { student_id: 'SV001', name: 'Nguyễn Văn An',    email: 'an.nv@email.com',    enrolled: true },
  { student_id: 'SV002', name: 'Trần Thị Bình',    email: 'binh.tt@email.com',  enrolled: true },
  { student_id: 'SV003', name: 'Lê Minh Cường',    email: 'cuong.lm@email.com', enrolled: true },
  { student_id: 'SV004', name: 'Phạm Thu Hà',      email: 'ha.pt@email.com',    enrolled: true },
  { student_id: 'SV005', name: 'Hoàng Đức Hùng',   email: 'hung.hd@email.com',  enrolled: false },
  { student_id: 'SV006', name: 'Vũ Ngọc Lan',      email: 'lan.vn@email.com',   enrolled: true },
  { student_id: 'SV007', name: 'Đặng Quốc Minh',   email: 'minh.dq@email.com',  enrolled: true },
  { student_id: 'SV008', name: 'Bùi Thị Nga',      email: 'nga.bt@email.com',   enrolled: false },
]

const now = Date.now()
const hour = 3600_000

export const MOCK_SESSIONS = [
  {
    id: 3, class_name: 'DPL302m - Nhóm 1', student_count: 8,
    start_time: new Date(now - 2 * hour).toISOString(),
    end_time: null,
    attention_pct: 72, has_report: false,
  },
  {
    id: 2, class_name: 'DPL302m - Nhóm 1', student_count: 8,
    start_time: new Date(now - 26 * hour).toISOString(),
    end_time: new Date(now - 24 * hour).toISOString(),
    attention_pct: 68, has_report: true,
  },
  {
    id: 1, class_name: 'DPL302m - Nhóm 1', student_count: 7,
    start_time: new Date(now - 50 * hour).toISOString(),
    end_time: new Date(now - 48 * hour).toISOString(),
    attention_pct: 81, has_report: true,
  },
]

export const MOCK_ATTENDANCE = [
  { student_id: 'SV001', name: 'Nguyễn Văn An',  entry_time: (now - 2 * hour) / 1000, exit_time: null, duration_min: 120 },
  { student_id: 'SV002', name: 'Trần Thị Bình',  entry_time: (now - 2 * hour) / 1000, exit_time: null, duration_min: 120 },
  { student_id: 'SV003', name: 'Lê Minh Cường',  entry_time: (now - 1.8 * hour) / 1000, exit_time: null, duration_min: 108 },
  { student_id: 'SV004', name: 'Phạm Thu Hà',    entry_time: (now - 2 * hour) / 1000, exit_time: (now - 0.5 * hour) / 1000, duration_min: 90 },
  { student_id: 'SV006', name: 'Vũ Ngọc Lan',    entry_time: (now - 2 * hour) / 1000, exit_time: null, duration_min: 120 },
  { student_id: 'SV007', name: 'Đặng Quốc Minh', entry_time: (now - 1.5 * hour) / 1000, exit_time: null, duration_min: 90 },
]

export const MOCK_EVENTS = [
  { track_id: 1, student_id: 'SV001', name: 'Nguyễn Văn An',  state: 'focused',     ts: (now - 120_000) / 1000 },
  { track_id: 2, student_id: 'SV002', name: 'Trần Thị Bình',  state: 'using_phone', ts: (now - 90_000)  / 1000, duration_s: 45 },
  { track_id: 3, student_id: 'SV003', name: 'Lê Minh Cường',  state: 'drowsy',      ts: (now - 60_000)  / 1000, duration_s: 30 },
  { track_id: 4, student_id: 'SV004', name: 'Phạm Thu Hà',    state: 'off_task',    ts: (now - 45_000)  / 1000 },
  { track_id: 6, student_id: 'SV006', name: 'Vũ Ngọc Lan',    state: 'side_talking',ts: (now - 30_000)  / 1000 },
  { track_id: 7, student_id: 'SV007', name: 'Đặng Quốc Minh', state: 'focused',     ts: (now - 15_000)  / 1000 },
  { track_id: 1, student_id: 'SV001', name: 'Nguyễn Văn An',  state: 'focused',     ts: (now - 5_000)   / 1000 },
]

export const MOCK_SUMMARY = {
  behavior_distribution: { focused: 45, drowsy: 8, using_phone: 5, off_task: 10, away_from_seat: 2, side_talking: 6 },
  avg_attention_pct: 72,
  total_events: 76,
}

export const MOCK_REPORT = {
  session_id: 2,
  provider: 'google',
  generated_at: new Date(now - 23 * hour).toISOString(),
  content: `# Báo cáo Phiên Học — DPL302m Nhóm 1

## Tổng quan
Phiên học diễn ra trong **2 giờ** với **8 sinh viên** tham gia. Tỷ lệ chú ý trung bình đạt **68%**, ở mức trung bình.

## Điểm danh
- **7/8 sinh viên** có mặt đúng giờ
- 1 sinh viên (SV004) rời lớp sớm sau 90 phút

## Phân tích hành vi
Phân bố hành vi trong phiên học:
- **Focused**: 59% thời gian — sinh viên tập trung vào bài giảng
- **Off-task**: 13% — xem điện thoại hoặc nhìn lung tung
- **Side-talking**: 8% — nói chuyện với bạn bên cạnh
- **Drowsy**: 10% — có dấu hiệu buồn ngủ (đặc biệt 30 phút cuối)
- **Using phone**: 7% — dùng điện thoại trong giờ học

## Nhận xét từng sinh viên
- **Nguyễn Văn An (SV001)**: Chú ý tốt, duy trì trạng thái focused hầu hết thời gian
- **Trần Thị Bình (SV002)**: Có 2 lần dùng điện thoại, cần nhắc nhở
- **Lê Minh Cường (SV003)**: Biểu hiện buồn ngủ sau giờ giải lao
- **Vũ Ngọc Lan (SV006)**: Hay nói chuyện với bạn bên cạnh

## Đề xuất
1. Tăng tương tác trong giờ học để giảm tỷ lệ off-task
2. Nhắc nhở về việc không dùng điện thoại trong lớp
3. Xem xét giờ giải lao ngắn giữa buổi để giảm drowsy
`,
}
