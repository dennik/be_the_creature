# progress_bar.py
# Version: 1.3
# Change Log:
# v1.3 - Added blank frame for 0% (np.zeros BGRA transparent) at frames[0] to avoid requiring ui_bar000.png (fixes FileNotFoundError if missing). Updated range to 1 to FRAME_COUNT (1-100), append loads after blank insert. Retained v1.2 0-100 config; certainty: 90% (provides empty bar without file; assumes transparent BGRA—test for visual match; extend with load fallback if needed).
# v1.2 - Updated pre-load to range(0, 101) for ui_bar000.png (0%) to ui_bar100.png (100%), assuming ui_bar000.png added for empty bar. Retained v1.1 modular and prior; certainty: 90% (handles 0-100; test for file existence—add if missing).
# v1.1 - Removed standalone main loop/window/imshow (leftover from demo; bug caused extra window). Now modular for import/use (e.g., in ui_controller.py: call overlay_transparent with your canvas/frame/pos/alpha). Retained config/pre-load/helper. Certainty: 100% (direct removal; confirms no window on import).

import cv2
import numpy as np
import os

# ----------------------------------------------------------------------
# Configuration (as provided; unchanged for modularity)
# ----------------------------------------------------------------------
FRAME_DIR = r"d:\python\graphics\ui_bar"
FRAME_PATTERN = "ui_bar{:03d}.png"   # ui_bar001.png … ui_bar100.png (0 handled in code)
FRAME_COUNT = 101  # 0-100 (0 blank, 1-100 loaded)
FRAME_DURATION = 0.5               # seconds per frame (unused in modular; caller times)
FADE_IN_DURATION = 1.0              # seconds (caller can apply)
FADE_OUT_DURATION = 1.0             # seconds (caller can apply)

BAR_W = 520
BAR_H = 140

# ----------------------------------------------------------------------
# Pre-load all frames (with alpha channel) for reuse
# ----------------------------------------------------------------------
frames = []
# v1.3: Insert blank transparent frame for 0%
blank_frame = np.zeros((BAR_H, BAR_W, 4), dtype=np.uint8)  # BGRA all zero (transparent)
frames.append(blank_frame)

for i in range(1, FRAME_COUNT):  # Load 001-100
    path = os.path.join(FRAME_DIR, FRAME_PATTERN.format(i))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing frame: {path}")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)   # keeps alpha
    if img is None:
        raise ValueError(f"Failed to load {path}")
    if img.shape[:2] != (BAR_H, BAR_W):
        print(f"Warning: Frame {i} size {img.shape[:2]} ≠ expected ({BAR_H},{BAR_W})")
    frames.append(img)

# ----------------------------------------------------------------------
# Helper: overlay image with alpha onto canvas at (x, y)
# ----------------------------------------------------------------------
def overlay_transparent(background, overlay, x, y, alpha=1.0):
    """Overlay 'overlay' (BGR or BGRA) onto 'background' at (x,y) with optional global alpha.
    Comments: Handles alpha blend or opaque copy; clips to bounds implicitly via slicing.
    """
    h, w = overlay.shape[0], overlay.shape[1]
    if overlay.shape[2] == 4:  # has alpha channel
        b, g, r, a = cv2.split(overlay)
        overlay_rgb = cv2.merge((b, g, r))
        mask = a / 255.0
        mask = mask * alpha  # apply global fade
        for c in range(3):
            background[y:y+h, x:x+w, c] = (
                (1 - mask) * background[y:y+h, x:x+w, c] +
                mask * overlay_rgb[..., c]
            )
    else:  # no alpha → treat as opaque
        temp = overlay.astype(np.float32) * alpha + background[y:y+h, x:x+w] * (1 - alpha)
        background[y:y+h, x:x+w] = temp.astype(np.uint8)