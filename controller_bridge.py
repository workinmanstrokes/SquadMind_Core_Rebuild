"""
controller_bridge.py - SquadMind Controller & Macro Bridge
Interfaces with Cronus Zen / Titan Two or serial/virtual controllers.
"""

import time
import queue
import threading
import logging

logger = logging.getLogger("ControllerBridge")


class ControllerBridge:
    """Manages asynchronous command dispatching to controller hardware
    (e.g., via USB serial to Cronus Zen / Titan Two GPC scripts or virtual controller).
    """
    def __init__(self, port=None, baudrate=115200, mock_mode=True):
        self.port = port
        self.baudrate = baudrate
        self.mock_mode = mock_mode
        self.serial_conn = None
        self.command_queue = queue.Queue()
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        if not self.mock_mode and self.port:
            try:
                import serial
                self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=0.1)
                logger.info(f"Connected to controller hardware on {self.port}")
            except Exception as e:
                logger.warning(f"Could not open serial port {self.port}: {e}. Running in Mock Mode.")
                self.mock_mode = True

        self.thread = threading.Thread(target=self._dispatcher, daemon=True)
        self.thread.start()
        return self

    def _dispatcher(self):
        while self.running:
            try:
                cmd = self.command_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self.mock_mode or not self.serial_conn:
                logger.info(f"[Hardware Emulation] Executed Controller Command: {cmd}")
            else:
                try:
                    payload = (cmd + "\n").encode("utf-8")
                    self.serial_conn.write(payload)
                except Exception as e:
                    logger.error(f"Serial transmission failed: {e}")

            self.command_queue.task_done()

    def send_command(self, command_str):
        """Enqueue a controller macro or GPC trigger."""
        self.command_queue.put(command_str)

    def trigger_shot_macro(self, duration_ms=480):
        """Triggers a calibrated shot timing macro."""
        cmd = f"TRIGGER_SHOT_{duration_ms}MS"
        self.send_command(cmd)

    def trigger_stamina_conservation(self):
        """Sends signal to release sprint burst modifier."""
        cmd = "RELEASE_SPRINT"
        self.send_command(cmd)

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.serial_conn:
            self.serial_conn.close()
        logger.info("Controller bridge stopped.")
