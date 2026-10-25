import cv2
import sys
from detector import SquadMindDetector

frame = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else "hud_calibration.png")
det = SquadMindDetector(model_path=None)
result = det.analyze_frame(frame)
print("detected_game:", result["detected_game"])
print("in_game:", result["in_game"])
print("hud_states:", result["hud_states"])
