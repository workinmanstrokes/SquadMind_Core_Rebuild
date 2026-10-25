"""One-off helper: crop a region out of a saved frame and save it, for
building game-signature template images."""
import cv2
import sys

src, dst, x1, y1, x2, y2 = sys.argv[1:7]
img = cv2.imread(src)
crop = img[int(y1):int(y2), int(x1):int(x2)]
cv2.imwrite(dst, crop)
print(f"Saved {crop.shape[1]}x{crop.shape[0]} crop to {dst}")
