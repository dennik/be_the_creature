# progress_bar.py
# Version: 1.4
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import PATHS

import cv2
import numpy as np

FRAME_DIR = PATHS['UI_BAR']
FRAME_PATTERN = "ui_bar{:03d}.png"
FRAME_COUNT = 101
FRAME_DURATION = 0.5
FADE_IN_DURATION = 1.0
FADE_OUT_DURATION = 1.0

BAR_W = 520
BAR_H = 140

frames = []
blank_frame = np.zeros((BAR_H, BAR_W, 4), dtype=np.uint8)
frames.append(blank_frame)

for i in range(1, FRAME_COUNT):
    path = os.path.join(FRAME_DIR, FRAME_PATTERN.format(i))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing frame: {path}")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load {path}")
    if img.shape[:2] != (BAR_H, BAR_W):
        print(f"Warning: Frame {i} size {img.shape[:2]} ≠ expected ({BAR_H},{BAR_W})")
    frames.append(img)

def overlay_transparent(background, overlay, x, y, alpha=1.0):
    h, w = overlay.shape[0], overlay.shape[1]
    if overlay.shape[2] == 4:
        b, g, r, a = cv2.split(overlay)
        overlay_rgb = cv2.merge((b, g, r))
        mask = a / 255.0
        mask = mask * alpha
        for c in range(3):
            background[y:y+h, x:x+w, c] = (
                (1 - mask) * background[y:y+h, x:x+w, c] +
                mask * overlay_rgb[..., c]
            )
    else:
        temp = overlay.astype(np.float32) * alpha + background[y:y+h, x:x+w] * (1 - alpha)
        background[y:y+h, x:x+w] = temp.astype(np.uint8)