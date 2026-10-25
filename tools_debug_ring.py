import cv2
import numpy as np
import sys
from profiles.nhl import RED_LOW1, RED_HIGH1, RED_LOW2, RED_HIGH2

frame = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else "hud_calibration.png")
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, RED_LOW1, RED_HIGH1) | cv2.inRange(hsv, RED_LOW2, RED_HIGH2)
cv2.imwrite("debug_red_mask.png", mask)
print("mask nonzero pixels:", cv2.countNonZero(mask))

mask2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"found {len(contours)} contours")
areas = sorted([cv2.contourArea(c) for c in contours], reverse=True)
print("top areas:", areas[:10])

for c in contours:
    area = cv2.contourArea(c)
    if area < 100:
        continue
    if len(c) < 5:
        print(f"area={area:.0f} skipped (too few points: {len(c)})")
        continue
    (cx, cy), (ew, eh), angle = cv2.fitEllipse(c)
    aspect = max(ew, eh) / max(1, min(ew, eh))
    print(f"area={area:.0f} center=({cx:.0f},{cy:.0f}) ellipse=({ew:.0f}x{eh:.0f}) aspect={aspect:.2f}")
