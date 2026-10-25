"""One-off helper: grab a single live frame from the Mirabox and save it as a
PNG so the HUD ROI coordinates in detector.py can be calibrated visually."""
import cv2
import sys
import time

out_path = sys.argv[1] if len(sys.argv) > 1 else "frame.png"
index = int(sys.argv[2]) if len(sys.argv) > 2 else 2

cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

best_frame = None
best_brightness = -1
for _ in range(20):
    ret, frame = cap.read()
    if ret and frame is not None:
        b = frame.mean()
        if b > best_brightness:
            best_brightness = b
            best_frame = frame
    time.sleep(0.05)

cap.release()

if best_frame is None:
    print("Failed to capture a frame.")
    sys.exit(1)

cv2.imwrite(out_path, best_frame)
h, w = best_frame.shape[:2]
print(f"Saved {w}x{h} frame to {out_path}, mean brightness={best_brightness:.1f}")
