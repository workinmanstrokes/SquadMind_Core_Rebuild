"""One-off helper: print BGR and HSV pixel values at given coordinates in a
saved frame, to derive real color thresholds instead of guessing."""
import cv2
import sys

img = cv2.imread(sys.argv[1])
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

points = [
    ("red_ring_top", 415, 270),
    ("red_ring_bottom", 415, 450),
    ("red_ring_left", 220, 360),
    ("red_ring_right", 615, 360),
    ("green_bar", 500, 478),
    ("green_bar_2", 550, 478),
    ("gray_bar_empty", 300, 478),
    ("gray_bar_empty_2", 340, 478),
]

for label, x, y in points:
    b, g, r = img[y, x].tolist()
    h, s, v = hsv[y, x].tolist()
    print(f"{label:16s} (x={x},y={y}): BGR=({b:3d},{g:3d},{r:3d})  HSV=({h:3d},{s:3d},{v:3d})")
