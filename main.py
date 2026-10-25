"""
main.py - SquadMind Core Orchestrator
Integrates zero-buffer capture, vision inference, state management,
tactical voice comms, and controller automation.
"""

import sys
import time
import yaml
import signal
import logging
from pathlib import Path

from capture import ZeroBufferCapture
from detector import SquadMindDetector
from state_machine import GameStateMachine
from controller_bridge import ControllerBridge
from voice_engine import VoiceEngine

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
logger = logging.getLogger("SquadMindOrchestrator")


class SquadMindSystem:
    def __init__(self, config_path="config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # 1. Initialize Subsystems
        cap_cfg = self.config.get("capture", {})
        self.capture = ZeroBufferCapture(
            source=cap_cfg.get("source", 0),
            width=cap_cfg.get("width", 1920),
            height=cap_cfg.get("height", 1080),
            fps=cap_cfg.get("fps", 60),
            mock_mode=cap_cfg.get("mock_mode", False)
        )

        det_cfg = self.config.get("detector", {})
        self.detector = SquadMindDetector(
            model_path=det_cfg.get("model_path"),
            conf_threshold=det_cfg.get("conf_threshold", 0.45)
        )

        fsm_cfg = self.config.get("state_machine", {})
        self.fsm = GameStateMachine(
            debounce_duration=fsm_cfg.get("debounce_seconds", 0.6),
            confidence_frames=fsm_cfg.get("confidence_frames", 3)
        )

        ctrl_cfg = self.config.get("controller", {})
        self.controller = ControllerBridge(
            port=ctrl_cfg.get("port"),
            baudrate=ctrl_cfg.get("baudrate", 115200),
            mock_mode=ctrl_cfg.get("mock_mode", True)
        )

        voice_cfg = self.config.get("voice", {})
        self.voice = VoiceEngine(
            voice_enabled=voice_cfg.get("enabled", True),
            min_interval_seconds=voice_cfg.get("min_interval_seconds", 3.0)
        )

        self._bind_tactical_events()
        self.running = False

    def _load_config(self):
        if not self.config_path.exists():
            logger.warning(f"Config file {self.config_path} not found. Using defaults.")
            return {}
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _bind_tactical_events(self):
        """Wires FSM state changes to audio comms and controller macros."""
        def on_low_stamina(event):
            self.voice.speak("Stamina low. Releasing turbo modifier.", priority=1)
            self.controller.trigger_stamina_conservation()

        def on_high_stamina(event):
            self.voice.speak("Full stamina. Lane clear.", priority=2)

        def on_menu(event):
            self.voice.speak("Menu screen. Tactical assist on standby.", priority=3)

        self.fsm.register_callback("ACTIVE_GAMEPLAY_LOW_STAMINA", on_low_stamina)
        self.fsm.register_callback("ACTIVE_GAMEPLAY_HIGH_STAMINA", on_high_stamina)
        self.fsm.register_callback("MENU_OR_LOBBY", on_menu)

    def start(self, duration=None):
        logger.info("Starting SquadMind Core engine...")
        self.running = True

        self.capture.start()
        self.controller.start()
        self.voice.start()

        self.voice.speak("SquadMind online. Systems active.", priority=1)

        start_time = time.time()
        processed_frames = 0
        perf_timer = time.time()

        try:
            while self.running:
                now = time.time()
                if duration and (now - start_time) >= duration:
                    logger.info(f"Execution duration ({duration}s) reached.")
                    break

                frame, timestamp = self.capture.read_latest()
                if frame is None:
                    time.sleep(0.005)
                    continue

                processed_frames += 1

                # Step 1: Detect HUD & Objects
                analysis = self.detector.analyze_frame(frame)

                # Step 2: Feed state machine
                self.fsm.update(analysis)

                # Periodic throughput telemetry
                now = time.time()
                if now - perf_timer >= 2.0:
                    fps = processed_frames / (now - perf_timer)
                    latency_ms = (now - timestamp) * 1000.0 if timestamp else 0.0
                    logger.info(f"[Perf] Processing FPS: {fps:.1f} | Frame Delay: {latency_ms:.1f}ms | Current State: {self.fsm.current_state}")
                    processed_frames = 0
                    perf_timer = now

                if duration and (now - start_time) >= duration:
                    logger.info(f"Execution duration ({duration}s) reached.")
                    break

        except KeyboardInterrupt:
            logger.info("Interrupt received.")
        finally:
            self.shutdown()

    def shutdown(self):
        self.running = False
        logger.info("Shutting down SquadMind subsystems...")
        self.capture.release()
        self.controller.stop()
        self.voice.stop()
        logger.info("All subsystems cleanly terminated.")


if __name__ == "__main__":
    duration = None
    if len(sys.argv) > 1:
        try:
            duration = float(sys.argv[1])
        except ValueError:
            pass

    app = SquadMindSystem()
    app.start(duration=duration)
