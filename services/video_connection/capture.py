"""
EduVision — Real-time optimized video capture.

Giải quyết các vấn đề chính khi stream RTSP từ iPhone:
1. RTSP buffer lag: OpenCV mặc định buffer nhiều frame → delay tích lũy
2. Frame drop khi pipeline xử lý chậm hơn FPS của camera
3. Resolution quá cao → GPU/CPU chạy không kịp
4. Reconnect khi mạng gián đoạn (WiFi/Tailscale flap)

Dùng mô hình producer-consumer với thread riêng cho capture.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class CaptureConfig:
    """Tuning knobs for real-time capture."""

    # Resize frame trước khi đưa vào pipeline (None = giữ nguyên)
    # 640 là tốt nhất cho YOLO11n/YOLO26n. Dùng 480 nếu CPU yếu.
    target_width: Optional[int] = 640

    # Chỉ giữ frame mới nhất trong buffer (tránh xử lý frame cũ)
    drop_stale_frames: bool = True

    # OpenCV internal buffer size — giữ thấp để giảm lag
    cv2_buffer_size: int = 1

    # Thời gian chờ giữa các lần reconnect (giây)
    reconnect_delay: float = 3.0

    # Số lần reconnect tối đa (0 = không giới hạn)
    max_reconnects: int = 0

    # RTSP transport protocol — tcp ổn định hơn udp qua Tailscale/NAT
    rtsp_transport: str = "tcp"

    # Timeout đọc frame (giây) — nếu không có frame trong thời gian này → reconnect
    read_timeout: float = 10.0


@dataclass
class FrameStats:
    """Runtime statistics for monitoring pipeline health."""
    captured: int = 0
    dropped: int = 0
    processed: int = 0
    reconnects: int = 0
    last_fps: float = 0.0
    _fps_t0: float = field(default_factory=time.time, repr=False)
    _fps_count: int = field(default=0, repr=False)

    def tick_capture(self) -> None:
        self.captured += 1
        self._fps_count += 1
        now = time.time()
        elapsed = now - self._fps_t0
        if elapsed >= 1.0:
            self.last_fps = self._fps_count / elapsed
            self._fps_count = 0
            self._fps_t0 = now

    def tick_drop(self) -> None:
        self.dropped += 1

    def tick_process(self) -> None:
        self.processed += 1

    @property
    def drop_rate(self) -> float:
        if self.captured == 0:
            return 0.0
        return self.dropped / self.captured


class ThreadedCapture:
    """
    Đọc frame từ RTSP/webcam trong thread riêng.

    Pipeline chính chỉ gọi read() để lấy frame mới nhất —
    không bao giờ bị block bởi network I/O.
    Tự động reconnect khi mất kết nối.

    Example
    -------
    cap = ThreadedCapture("rtsp://100.86.84.22:8554/live.sdp")
    cap.start()
    while cap.is_running:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        # xử lý frame...
    cap.stop()
    """

    def __init__(self, source: str | int, config: Optional[CaptureConfig] = None) -> None:
        self.source = source
        self.cfg = config or CaptureConfig()
        self.stats = FrameStats()

        self._frame: Optional[np.ndarray] = None
        self._frame_index: int = 0
        self._last_read_index: int = -1
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Video metadata (filled after first successful open)
        self.fps: float = 30.0
        self.width: int = 0
        self.height: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "ThreadedCapture":
        """Start the capture thread. Returns self for chaining."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="EduVision-Capture")
        self._thread.start()
        return self

    def stop(self) -> None:
        """Signal the capture thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        Return (True, frame) if a NEW frame is available since the last call.
        Return (False, None) if no new frame yet (caller should sleep briefly).
        """
        with self._lock:
            if self._frame is None or self._frame_index == self._last_read_index:
                return False, None
            frame = self._frame.copy()
            self._last_read_index = self._frame_index
            self.stats.tick_process()
            return True, frame

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    # ------------------------------------------------------------------
    # Internal capture loop (runs in background thread)
    # ------------------------------------------------------------------

    def _open(self) -> Optional[cv2.VideoCapture]:
        """Open the video source with optimised settings."""
        source = self.source
        if isinstance(self.source, str) and self.source.lower().startswith("rtsp://"):
            # Use FFMPEG backend for RTSP with UDP to prevent MSMF errors and connection hangs
            import os
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.cfg.cv2_buffer_size)
        else:
            cap = cv2.VideoCapture(self.source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.cfg.cv2_buffer_size)

        if not cap.isOpened():
            return None

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cap

    def _loop(self) -> None:
        reconnect_count = 0
        while not self._stop_event.is_set():
            cap = self._open()
            if cap is None:
                print(f"[Capture] Cannot open source: {self.source}. Retrying in {self.cfg.reconnect_delay}s...")
                self.stats.reconnects += 1
                self._stop_event.wait(self.cfg.reconnect_delay)
                continue

            print(f"[Capture] Connected: {self.source} ({self.width}x{self.height} @ {self.fps:.1f} fps)")
            consecutive_failures = 0
            deadline = time.time() + self.cfg.read_timeout

            while not self._stop_event.is_set():
                ok, raw = cap.read()
                if not ok:
                    consecutive_failures += 1
                    if time.time() > deadline or consecutive_failures > 30:
                        print(f"[Capture] Stream lost. Reconnecting...")
                        break
                    time.sleep(0.05)
                    continue

                consecutive_failures = 0
                deadline = time.time() + self.cfg.read_timeout

                # Resize if configured
                frame = self._resize(raw)

                with self._lock:
                    if self.cfg.drop_stale_frames and self._frame_index != self._last_read_index:
                        # Previous frame was never consumed — count as drop
                        self.stats.tick_drop()
                    self._frame = frame
                    self._frame_index += 1
                    self.stats.tick_capture()

            cap.release()
            reconnect_count += 1
            self.stats.reconnects += 1
            if self.cfg.max_reconnects and reconnect_count >= self.cfg.max_reconnects:
                print(f"[Capture] Max reconnects ({self.cfg.max_reconnects}) reached. Stopping.")
                self._stop_event.set()
                break
            if not self._stop_event.is_set():
                self._stop_event.wait(self.cfg.reconnect_delay)

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if self.cfg.target_width is None:
            return frame
        h, w = frame.shape[:2]
        if w == self.cfg.target_width:
            return frame
        scale = self.cfg.target_width / w
        new_w = self.cfg.target_width
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


class FrameSkipper:
    """
    Tính xem có nên xử lý frame này không, dựa trên target FPS.

    Ví dụ: camera 30fps, pipeline xử lý được 10fps
    → bỏ qua 2/3 frame để không bị lag tích lũy.
    """

    def __init__(self, target_fps: float = 10.0) -> None:
        self.target_fps = target_fps
        self._interval = 1.0 / max(target_fps, 1.0)
        self._last_process_time = 0.0

    def should_process(self) -> bool:
        now = time.time()
        if now - self._last_process_time >= self._interval:
            self._last_process_time = now
            return True
        return False

    def update_target(self, measured_fps: float) -> None:
        """Dynamically adjust target based on measured pipeline throughput."""
        self.target_fps = measured_fps
        self._interval = 1.0 / max(measured_fps, 1.0)
