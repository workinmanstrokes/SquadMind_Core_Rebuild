import cv2
import sys

src, dst, factor = sys.argv[1], sys.argv[2], int(sys.argv[3])
img = cv2.imread(src)
h, w = img.shape[:2]
big = cv2.resize(img, (w * factor, h * factor), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(dst, big)
print(f"Saved {big.shape[1]}x{big.shape[0]}")
