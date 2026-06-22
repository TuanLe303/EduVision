"""Build structured LLM prompts from a SessionSummary."""
from __future__ import annotations

from .aggregator import SessionSummary

_BEHAVIOR_LABELS_VI: dict[str, str] = {
    "focused": "Tập trung",
    "drowsy": "Buồn ngủ / Ngủ gật",
    "using_phone": "Dùng điện thoại",
    "off_task": "Mất tập trung",
    "away_from_seat": "Rời chỗ ngồi",
    "side_talking": "Nói chuyện riêng",
}

_BEHAVIOR_LABELS_EN: dict[str, str] = {
    "focused": "Focused",
    "drowsy": "Drowsy / Falling asleep",
    "using_phone": "Using phone",
    "off_task": "Off-task",
    "away_from_seat": "Away from seat",
    "side_talking": "Side-talking",
}


def build_prompt(summary: SessionSummary, language: str = "vi") -> str:
    """Return a prompt string ready to send to an LLM."""
    labels = _BEHAVIOR_LABELS_VI if language == "vi" else _BEHAVIOR_LABELS_EN
    lines: list[str] = []

    cs = summary.class_stats
    duration_str = _fmt_duration(cs.duration_seconds)

    if language == "vi":
        lines += [
            "Bạn là AI trợ lý giáo dục. Dưới đây là dữ liệu quan sát hành vi lớp học "
            "được thu thập tự động bằng hệ thống camera AI (EduVision). "
            "Hãy viết báo cáo buổi học **BẰNG TIẾNG VIỆT**, chuyên nghiệp, "
            "ngắn gọn và hữu ích cho giảng viên.",
            "",
            "=== TỔNG QUAN BUỔI HỌC ===",
            f"- Thời lượng quan sát: {duration_str}",
            f"- Số học sinh được ghi nhận: {cs.total_students}",
            f"- Tổng số frame phân tích: {cs.total_frames:,}",
            f"- Tỉ lệ tập trung trung bình toàn lớp: {cs.avg_attention_score * 100:.1f}%",
            "",
            "Phân bố hành vi toàn lớp (% thời gian):",
        ]
    else:
        lines += [
            "You are an AI educational assistant. Below is classroom behavior observation data "
            "collected automatically by an AI camera system (EduVision). "
            "Write a professional, concise session report useful for the instructor.",
            "",
            "=== SESSION OVERVIEW ===",
            f"- Observation duration: {duration_str}",
            f"- Students detected: {cs.total_students}",
            f"- Total frames analyzed: {cs.total_frames:,}",
            f"- Class average attention rate: {cs.avg_attention_score * 100:.1f}%",
            "",
            "Class-wide behavior distribution (% of time):",
        ]

    for state, frac in sorted(
        cs.behavior_distribution.items(), key=lambda x: -x[1]
    ):
        if frac > 0.001:
            lbl = labels.get(state, state)
            lines.append(f"  • {lbl}: {frac * 100:.1f}%")

    lines.append("")

    # Per-student breakdown
    heading = "=== DỮ LIỆU TỪNG HỌC SINH ===" if language == "vi" else "=== PER-STUDENT DATA ==="
    lines.append(heading)

    for s in summary.students:
        label = s.student_id or f"Track-{s.track_id}"
        name_part = f" ({s.name})" if s.name else ""
        dominant_lbl = labels.get(s.dominant_behavior, s.dominant_behavior)

        if language == "vi":
            lines += [
                f"Học sinh {label}{name_part}:",
                f"  - Số frame xuất hiện: {s.present_frames}",
                f"  - Tỉ lệ tập trung: {s.attention_score * 100:.1f}%",
                f"  - Hành vi chủ đạo: {dominant_lbl}",
                "  - Phân bố hành vi:",
            ]
        else:
            lines += [
                f"Student {label}{name_part}:",
                f"  - Present frames: {s.present_frames}",
                f"  - Attention rate: {s.attention_score * 100:.1f}%",
                f"  - Dominant behavior: {dominant_lbl}",
                "  - Behavior breakdown:",
            ]

        for state, frac in sorted(
            s.behavior_fractions.items(), key=lambda x: -x[1]
        ):
            if frac > 0.01:
                lbl = labels.get(state, state)
                lines.append(f"      {lbl}: {frac * 100:.1f}%")

        if s.events:
            unique_events = list(dict.fromkeys(s.events))
            event_str = ", ".join(unique_events[:8])
            key = "Sự kiện hành vi" if language == "vi" else "Behavior events"
            lines.append(f"  - {key}: {event_str}")

        lines.append("")

    # Instructions
    if language == "vi":
        lines += [
            "=== YÊU CẦU BÁO CÁO ===",
            "Hãy viết báo cáo buổi học gồm các phần sau (BẰNG TIẾNG VIỆT):",
            "",
            "1. **Tóm tắt buổi học** — 2–3 câu đánh giá tổng quan tình hình lớp học.",
            "2. **Mức độ tham gia** — Nhận xét về mức độ tập trung và tương tác của học sinh dựa trên dữ liệu.",
            "3. **Vấn đề nổi bật** — Các hành vi tiêu cực hoặc đáng lo ngại phổ biến nhất (nếu có).",
            "4. **Học sinh cần chú ý** — Liệt kê cụ thể học sinh có tỉ lệ tập trung thấp "
            "hoặc hành vi tiêu cực chiếm nhiều (>30%). Nêu rõ mã học sinh và % cụ thể.",
            "5. **Đề xuất cho giảng viên** — 2–3 gợi ý cụ thể, thiết thực để cải thiện "
            "chất lượng buổi học tiếp theo.",
            "",
            "Viết ngắn gọn, rõ ràng, mang tính xây dựng. "
            "Chỉ dựa trên dữ liệu được cung cấp, không bịa đặt.",
        ]
    else:
        lines += [
            "=== REPORT REQUIREMENTS ===",
            "Write a session report with the following sections:",
            "",
            "1. **Session Summary** — 2–3 sentences on the overall class situation.",
            "2. **Engagement Level** — Comments on student attention and participation based on data.",
            "3. **Notable Issues** — Most common negative or concerning behaviors (if any).",
            "4. **Students Needing Attention** — Specifically list students with low attention "
            "or high rates of negative behavior (>30%). Include student IDs and exact percentages.",
            "5. **Recommendations** — 2–3 specific, actionable suggestions to improve the next session.",
            "",
            "Be concise, constructive, and factual. Only use the data provided above.",
        ]

    return "\n".join(lines)


def _fmt_duration(seconds: float) -> str:
    if seconds <= 0:
        return "N/A"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"
