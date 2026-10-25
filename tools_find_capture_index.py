"""One-off helper: enumerate camera indices via the DSHOW backend only (MSMF
has repeatedly hung for 90s-plus on this Mirabox device, so it's skipped).
Reads several frames per index before judging brightness, and flushes output
immediately so progress is visible even when piped to a file."""
import cv2
import time

for i in range(4):
    print(f"probing index {i}...", flush=True)
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"index {i}: could not open", flush=True)
        cap.release()
        continue

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    best_brightness = 0.0
    ok_any = False
    for attempt in range(15):
        ret, frame = cap.read()
        if ret and frame is not None:
            ok_any = True
            best_brightness = max(best_brightness, frame.mean())
        time.sleep(0.05)

    status = "OK" if ok_any else "FAILED"
    print(f"index {i}: {status}, negotiated {int(w)}x{int(h)}, best mean brightness over 15 frames = {best_brightness:.1f}", flush=True)
    cap.release()
