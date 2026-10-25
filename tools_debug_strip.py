import cv2
import numpy as np
import sys
from profiles.nhl import _find_player_ring, _find_ring_bottom_edge, FILL_SAT_MIN, FILL_VAL_MIN, _STRIP_GAP_PX, _STRIP_HEIGHT_PX, _STRIP_HALF_WIDTH_FRAC

frame = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else "hud_calibration.png")
ring = _find_player_ring(frame)
rx, ry, rw, rh, mask = ring
print("ring bbox (x,y,w,h):", (rx, ry, rw, rh))

ring_bottom = _find_ring_bottom_edge(mask, rx, ry, rw, rh)
print("detected ring bottom edge (y):", ring_bottom, " (bbox bottom was:", ry + rh, ")")

cx = rx + rw / 2
half_w = rw * _STRIP_HALF_WIDTH_FRAC
strip_y1 = int(ring_bottom + _STRIP_GAP_PX)
strip_y2 = int(strip_y1 + _STRIP_HEIGHT_PX)
x1 = int(max(0, cx - half_w))
x2 = int(min(frame.shape[1], cx + half_w))
print(f"strip region: x=({x1},{x2}) y=({strip_y1},{strip_y2})")

strip = frame[strip_y1:strip_y2, x1:x2]
cv2.imwrite("debug_strip.png", strip)
big = cv2.resize(strip, (strip.shape[1] * 8, strip.shape[0] * 8), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("debug_strip_big.png", big)

hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
filled_mask = (hsv[:, :, 1] >= FILL_SAT_MIN) & (hsv[:, :, 2] >= FILL_VAL_MIN)
print("filled pixel count:", int(filled_mask.sum()), "/ total:", filled_mask.size)
if filled_mask.sum() > 0:
    hues = hsv[:, :, 0][filled_mask]
    print("hue distribution: min", hues.min(), "max", hues.max(), "mean", hues.mean())
    print("hue histogram:", np.bincount(hues, minlength=180)[:90])
