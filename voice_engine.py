"""
voice_engine.py - SquadMind Audio & Voice Comms Engine
Handles non-blocking tactical voice callouts with priority queuing and cooldowns.
"""

import time
import queue
import threading
import logging
import subprocess
import shutil

logger = logging.getLogger("VoiceEngine")


class VoiceEngine:
    """Dispatches real-time teammate callouts and tactical game audio without
    interfering with high-frequency vision processing.
    """
    def __init__(self, voice_enabled=True, min_interval_seconds=2.5):
        self.voice_enabled = voice_enabled
        self.min_interval = min_interval_seconds
        self.last_speech_time = 0.0
        self.speech_queue = queue.PriorityQueue()
        self.running = False
        self.thread = None

        # Check for system TTS binaries
        self.tts_backend = None
        if shutil.which("spd-say"):
            self.tts_backend = "spd-say"
        elif shutil.which("espeak"):
            self.tts_backend = "espeak"

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.thread.start()
        return self

    def speak(self, text, priority=2):
        """Enqueue speech text.
        Priority: 1 = Critical (interrupt/high priority), 2 = Standard, 3 = Low
        """
        if not self.voice_enabled:
            return
        # Stamped tuple: (priority, timestamp, text)
        self.speech_queue.put((priority, time.time(), text))

    def _speech_worker(self):
        while self.running:
            try:
                priority, timestamp, message = self.speech_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            now = time.time()
            # Drop low priority callouts if stale (> 3 seconds old)
            if priority > 1 and (now - timestamp) > 3.0:
                self.speech_queue.task_done()
                continue

            # Respect speech cooldown for non-critical alerts
            if priority > 1 and (now - self.last_speech_time) < self.min_interval:
                time.sleep(self.min_interval - (now - self.last_speech_time))

            self._deliver_speech(message)
            self.last_speech_time = time.time()
            self.speech_queue.task_done()

    def _deliver_speech(self, message):
        logger.info(f"[SquadMind Teammate Comms]: \"{message}\"")
        if self.tts_backend == "spd-say":
            try:
                subprocess.run(["spd-say", "-r", "15", message], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        elif self.tts_backend == "espeak":
            try:
                subprocess.run(["espeak", "-s", "165", message], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        logger.info("Voice engine stopped.")
