"""NHL (EA Sports) game profile.

Calibrated from real captured frames (see SquadMind session notes): the
on-screen "stamina" indicator is NOT a fixed HUD rectangle — it's a two-tone
arc (gray = empty, colored = filled) rendered in world space directly below
the red selection ring under the currently-controlled player's skates. Its
screen position moves with the player and the camera, so it must be located
each frame relative to the ring rather than read from a fixed ROI.

Color thresholds below were sampled directly from a real frame, not guessed:
  red ring:   HSV hue ~0/177, saturation 234-255, value 71-181
  green fill: HSV hue ~59,    saturation 235,      value 255
  empty seg:  saturation < 15 regardless of hue
Yellow/red (medium/low stamina) hue ranges are ESTIMATES based on standard
traffic-light convention (untested — no real frame has shown depleted
stamina yet). Re-tune HUE_MEDIUM/HUE_LOW against real footage once available.
"""
import os
import cv2
import numpy as np

SIGNATURE_PATH = os.path.join(os.path.dirname(__file__), "signatures", "nhl_ea_logo.png")
SIGNATURE_REGION = (42, 42, 130, 108)  # x1, y1, x2, y2 in a 1920x1080 frame
SIGNATURE_THRESHOLD = 0.7

SCOREBOARD_ROI = (40, 30, 470, 120)

# Red selection ring
RED_LOW1 = np.array([0, 140, 50])
RED_HIGH1 = np.array([8, 255, 255])
RED_LOW2 = np.array([172, 140, 50])
RED_HIGH2 = np.array([180, 255, 255])

# Stamina-bar fill color bands (green verified real; yellow/red estimated)
HUE_HIGH = (40, 80)    # green   -- verified
HUE_MEDIUM = (20, 39)  # yellow  -- ESTIMATED, not yet observed
HUE_LOW = (0, 12)      # red     -- ESTIMATED, not yet observed (overlaps ring hue;
                       # only sampled below the ring so should not collide in practice)
FILL_SAT_MIN = 140
FILL_VAL_MIN = 120

# Stamina bar sits just below the ring's true (mask-scanned) bottom edge.
_STRIP_GAP_PX = 1
_STRIP_HEIGHT_PX = 10
_STRIP_HALF_WIDTH_FRAC = 0.55

_signature_template = None


def _load_signature():
    global _signature_template
    if _signature_template is None:
        img = cv2.imread(SIGNATURE_PATH)
        _signature_template = img
    return _signature_template


def matches_signature(frame):
    """Template-match the EA SPORTS logo badge against its known screen region."""
    template = _load_signature()
    if template is None:
        return 0.0
    x1, y1, x2, y2 = SIGNATURE_REGION
    h, w = frame.shape[:2]
    x2, y2 = min(x2, w), min(y2, h)
    region = frame[y1:y2, x1:x2]
    th, tw = template.shape[:2]
    if region.shape[0] < th or region.shape[1] < tw:
        return 0.0
    result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
    return float(result.max())


def _find_player_ring(frame):
    """Locate the red selection ring under the controlled player.
    Returns an axis-aligned (x, y, w, h) bounding box, or None.

    Note: fitEllipse's returned axes are relative to the ellipse's own
    rotated frame, not the image's x/y axes, so they can't be used directly
    as image-space width/height without accounting for rotation. We use
    fitEllipse only to confirm ring-like shape (flattened, not circular/
    blob-like), then use boundingRect for actual image-space positioning.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, RED_LOW1, RED_HIGH1) | cv2.inRange(hsv, RED_LOW2, RED_HIGH2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # The ring is a thin outline, not a filled shape, so its contour AREA is
    # small and unreliable (anti-aliasing/occlusion by skates fragments it) -
    # size and shape of the fitted ellipse are the real filter, not area.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_size = 0
    for c in contours:
        if len(c) < 5:
            continue
        (cx, cy), (ew, eh), angle = cv2.fitEllipse(c)
        if ew < 15 or eh < 15 or ew > 250 or eh > 250:
            continue
        aspect = max(ew, eh) / max(1, min(ew, eh))
        if not (1.2 <= aspect <= 5.0):
            continue  # ring is a flattened ellipse from the broadcast camera angle
        x, y, w, h = cv2.boundingRect(c)
        size = w * h
        if size > best_size:
            best_size = size
            best = (x, y, w, h)
    if best is None:
        return None
    return (*best, mask)


def _find_ring_bottom_edge(mask, rx, ry, rw, rh):
    """The ring's contour bounding box under-reports its true extent (the
    outline gets fragmented by anti-aliasing/occlusion from the player's
    skates), which was confirmed by visually inspecting a debug crop: the
    stamina bar sample was still catching red ring pixels using the raw
    bbox bottom. Scan the mask itself, row by row below the bbox, for the
    real last row with meaningful red coverage across the ring's width.
    """
    search_bottom = min(mask.shape[0], ry + rh * 2)
    last_red_row = ry + rh - 1
    for row in range(ry, int(search_bottom)):
        row_slice = mask[row, rx:rx + rw]
        if row_slice.size == 0:
            continue
        coverage = np.count_nonzero(row_slice) / row_slice.size
        if coverage >= 0.12:
            last_red_row = row
    return last_red_row


def _sample_stamina_bar(frame, ring):
    """Given the ring's bounding box (+ its red mask), sample the fill bar
    just below its true bottom edge."""
    rx, ry, rw, rh, mask = ring
    cx = rx + rw / 2
    half_w = rw * _STRIP_HALF_WIDTH_FRAC
    ring_bottom = _find_ring_bottom_edge(mask, rx, ry, rw, rh)
    strip_y1 = int(ring_bottom + _STRIP_GAP_PX)
    strip_y2 = int(strip_y1 + _STRIP_HEIGHT_PX)
    x1 = int(max(0, cx - half_w))
    x2 = int(min(frame.shape[1], cx + half_w))
    if strip_y2 <= strip_y1 or x2 <= x1:
        return None

    strip = frame[strip_y1:strip_y2, x1:x2]
    if strip.size == 0:
        return None
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)

    filled_mask = (hsv[:, :, 1] >= FILL_SAT_MIN) & (hsv[:, :, 2] >= FILL_VAL_MIN)
    filled_px = int(filled_mask.sum())
    if filled_px < 5:
        return None  # not enough signal to trust this frame

    # Background ice is also low-saturation/grayish, which made counting
    # "empty" pixels directly unreliable (ice pixels get miscounted as empty
    # bar, drowning out the real ratio). Instead, measure what fraction of
    # the bar's horizontal span has a lit (filled-color) pixel in any row -
    # direction-agnostic and immune to background contamination.
    column_has_fill = filled_mask.any(axis=0)
    fill_ratio = float(column_has_fill.sum()) / max(1, column_has_fill.shape[0])

    hues = hsv[:, :, 0][filled_mask]
    mean_hue = float(np.mean(hues))
    state = "unknown"
    if HUE_HIGH[0] <= mean_hue <= HUE_HIGH[1]:
        state = "high"
    elif HUE_MEDIUM[0] <= mean_hue <= HUE_MEDIUM[1]:
        state = "medium"
    elif mean_hue <= HUE_LOW[1] or mean_hue >= 170:
        state = "low"

    return {"fill_ratio": fill_ratio, "state": state}


def analyze(frame):
    """Returns hud_states dict for the NHL profile."""
    h, w = frame.shape[:2]
    result = {
        "hud_detected": False,
        "stamina_percentage": 0.0,
        "active_meter": False,
        "player_ring_found": False,
    }

    sx1, sy1, sx2, sy2 = SCOREBOARD_ROI
    scoreboard_roi = frame[sy1:min(sy2, h), sx1:min(sx2, w)]
    if scoreboard_roi.size > 0:
        gray = cv2.cvtColor(scoreboard_roi, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        if variance > 40.0:
            result["hud_detected"] = True

    ring = _find_player_ring(frame)
    if ring:
        result["player_ring_found"] = True
        stamina = _sample_stamina_bar(frame, ring)
        if stamina:
            result["active_meter"] = True
            result["stamina_percentage"] = stamina["fill_ratio"]
            result["stamina_state"] = stamina["state"]

    return result
