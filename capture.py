"""
capture.py - SquadMind Core Low-Latency Video Capture
Implements a non-blocking, zero-buffer video capture worker to prevent frame lag.
"""

import cv2
import time
import threading
import queue
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("CaptureWorker")


class ZeroBufferCapture:
    """Continuously consumes video frames in a dedicated thread.
    Guarantees that `read()` always retrieves the most recently arrived frame
    without OS/OpenCV buffer accumulation.
    """
    def __init__(self, source=0, width=1920, height=1080, fps=60, mock_mode=False):
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = fps
        self.mock_mode = mock_mode

        self.q = queue.Queue(maxsize=1)
        self.running = False
        self.thread = None
        self.cap = None

        # Telemetry
        self.frames_captured = 0
        self.frames_dropped = 0
        self.last_frame_time = time.time()
        self.actual_fps = 0.0

    def start(self):
        if self.mock_mode:
            logger.info("Initializing synthetic frame generator (Mock Mode)...")
            self.running = True
            self.thread = threading.Thread(target=self._mock_reader, daemon=True)
            self.thread.start()
            return self

        logger.info(f"Opening capture source: {self.source} ({self.width}x{self.height} @ {self.target_fps}fps)")
        # CAP_DSHOW is pinned explicitly: OpenCV's default backend auto-selection
        # picks MSMF on Windows, which has been observed to hang for 90s+ (or
        # indefinitely) opening this capture card, while DSHOW opens it in <1s.
        backend = cv2.CAP_DSHOW if isinstance(self.source, int) else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.source, backend)

        if not self.cap.isOpened():
            logger.warning(f"Failed to open hardware capture source {self.source}. Falling back to Mock Mode.")
            self.mock_mode = True
            self.running = True
            self.thread = threading.Thread(target=self._mock_reader, daemon=True)
            self.thread.start()
            return self

        # Configure hardware capture properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        return self

    def _reader(self):
        fps_timer = time.time()
        frame_counter = 0

        while self.running:
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.05)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            self._enqueue_latest(frame)
            frame_counter += 1

            now = time.time()
            if now - fps_timer >= 1.0:
                self.actual_fps = frame_counter / (now - fps_timer)
                frame_counter = 0
                fps_timer = now

    def _mock_reader(self):
        """Generates synthetic game frames for testing pipelines without a capture card."""
        fps_timer = time.time()
        frame_counter = 0
        interval = 1.0 / max(1, self.target_fps)

        while self.running:
            t0 = time.time()
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # Draw a simulated game HUD
            # Scoreboard background
            cv2.rectangle(frame, (50, 40), (450, 110), (25, 25, 30), -1)
            cv2.rectangle(frame, (50, 40), (450, 110), (80, 80, 120), 2)
            cv2.putText(frame, "SQUADMIND SIMULATOR", (70, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Simulated dynamic clock & shot meter
            sim_time = time.strftime("%H:%M:%S")
            cv2.putText(frame, f"LIVE: {sim_time}", (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)

            # Simulated health/stamina bar
            bar_width = int(200 * ((np.sin(time.time()) + 1) / 2))
            cv2.rectangle(frame, (self.width - 300, 50), (self.width - 300 + bar_width, 70), (0, 220, 50), -1)
            cv2.rectangle(frame, (self.width - 300, 50), (self.width - 100, 70), (255, 255, 255), 1)

            self._enqueue_latest(frame)
            frame_counter += 1

            now = time.time()
            if now - fps_timer >= 1.0:
                self.actual_fps = frame_counter / (now - fps_timer)
                frame_counter = 0
                fps_timer = now

            elapsed = time.time() - t0
            sleep_duration = interval - elapsed
            if sleep_duration > 0:
                time.sleep(sleep_duration)

    def _enqueue_latest(self, frame):
        self.frames_captured += 1
        if not self.q.empty():
            try:
                self.q.get_nowait()
                self.frames_dropped += 1
            except queue.Empty:
                pass
        self.q.put((frame, time.time()))

    def read_latest(self):
        """Returns (frame, timestamp) or (None, None). Non-blocking."""
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None, None

    def release(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        logger.info(f"Capture released. Total captured: {self.frames_captured}, Dropped: {self.frames_dropped}")
