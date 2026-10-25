"""
detector.py - SquadMind Core Game State & Object Detection
Implements both Deep Learning (YOLOv8) and lightweight OpenCV HUD/ROI analysis.
"""

import cv2
import numpy as np
import logging
import time

from profiles import registry

logger = logging.getLogger("Detector")


class SquadMindDetector:
    """Combines model inference (YOLO) with fast computer-vision heuristics
    for HUD meters, scoreboards, and in-game UI elements.
    """
    def __init__(self, model_path=None, conf_threshold=0.45, rois=None):
        self.conf_threshold = conf_threshold
        self.rois = rois or {}
        self.model = None

        if model_path:
            self._load_yolo(model_path)

    def _load_yolo(self, model_path):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLOv8 model from {model_path}")
        except ImportError:
            logger.warning("ultralytics package not installed. YOLO model inference will be bypassed.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")

    def analyze_frame(self, frame):
        """Executes full frame analysis: model detections + HUD ROI evaluations."""
        results = {
            "timestamp": time.time(),
            "detections": [],
            "hud_states": {},
            "in_game": False,
            "detected_game": None,
        }

        if frame is None:
            return results

        h, w = frame.shape[:2]

        # 1. YOLO Model Inference (if active)
        if self.model:
            try:
                preds = self.model(frame, conf=self.conf_threshold, verbose=False)
                for box in preds[0].boxes:
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id]
                    conf = float(box.conf[0])
                    coords = box.xyxy[0].tolist()
                    results["detections"].append({
                        "label": label,
                        "confidence": conf,
                        "bbox": coords
                    })
            except Exception as e:
                logger.error(f"Error during YOLO inference: {e}")

        # 2. Identify which game is on screen, and delegate HUD reading to its
        # profile (each game's HUD elements live in different places / move
        # differently, e.g. NHL's stamina arc tracks the player, not a fixed
        # rectangle). Falls back to the generic fixed-ROI heuristic when no
        # profile's signature matches.
        game_name = registry.detect_game(frame)
        results["detected_game"] = game_name
        if game_name:
            hud_info = registry.analyze(frame, game_name)
        else:
            hud_info = self._analyze_hud_regions(frame, w, h)
        results["hud_states"] = hud_info

        # Determine overall game visibility heuristic
        results["in_game"] = hud_info.get("hud_detected", False)

        return results

    def _analyze_hud_regions(self, frame, width, height):
        """Analyzes standard HUD zones: health/stamina bar, radar, and scoreboard."""
        hud_data = {
            "hud_detected": False,
            "stamina_percentage": 0.0,
            "active_meter": False
        }

        # Example Stamina / Health bar ROI (default: top-right region)
        # Coordinates: x1=w-320, y1=45, x2=w-80, y2=75
        x1, y1 = max(0, width - 350), 40
        x2, y2 = width - 50, 80
        stamina_roi = frame[y1:y2, x1:x2]

        if stamina_roi.size > 0:
            hsv = cv2.cvtColor(stamina_roi, cv2.COLOR_BGR2HSV)
            # Green meter mask (typical for stamina, health, or green release meters)
            lower_green = np.array([35, 70, 70])
            upper_green = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)

            green_pixels = cv2.countNonZero(mask)
            total_pixels = mask.shape[0] * mask.shape[1]
            fill_ratio = green_pixels / max(1, total_pixels)

            if fill_ratio > 0.05:
                hud_data["hud_detected"] = True
                hud_data["stamina_percentage"] = min(1.0, fill_ratio * 3.5)
                hud_data["active_meter"] = True

        # Example Scoreboard / Text region ROI (top-left)
        sx1, sy1 = 40, 35
        sx2, sy2 = min(width, 460), 120
        scoreboard_roi = frame[sy1:sy2, sx1:sx2]

        if scoreboard_roi.size > 0:
            gray = cv2.cvtColor(scoreboard_roi, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            # If the scoreboard region contains clear edges / text
            if variance > 40.0:
                hud_data["hud_detected"] = True

        return hud_data
