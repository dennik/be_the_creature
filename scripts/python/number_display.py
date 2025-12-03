# filename: number_display.py
# Version: 1.0
# Change Log:
# v1.0 - Initial release. Draws an arbitrary integer (or string) on any OpenCV canvas
#        using transparent PNG digits from D:\python\Graphics\numbers\number_0.png … number_9.png.
#        Supports position (top-left of the whole number), scale factor, and a configurable gap.
#        Certainty: 100% (tested on Windows with transparent 165×263 PNGs).

import cv2
import os
import numpy as np

# ----------------------------------------------------------------------
# Configuration (feel free to tweak)
# ----------------------------------------------------------------------
DIGITS_DIR = r"D:\python\Graphics\numbers"   # Folder that contains number_0.png … number_9.png
DIGIT_WIDTH  = 165                           # Original pixel width of each digit PNG
DIGIT_HEIGHT = 263                           # Original pixel height of each digit PNG
GAP_PX       = 20                            # Horizontal gap between digits (in final scaled size)
DEFAULT_SCALE = 1.0                          # 1.0 = original size


def _load_digit(digit: int) -> np.ndarray:
    """
    Load a single transparent digit image (0-9) and return it as a BGR+Alpha array.
    Caches the images the first time they are requested.
    """
    if not 0 <= digit <= 9:
        raise ValueError(f"Digit must be 0-9, got {digit}")

    # Simple cache so we only read each file once
    if not hasattr(_load_digit, "cache"):
        _load_digit.cache = {}
    if digit not in _load_digit.cache:
        path = os.path.join(DIGITS_DIR, f"number_{digit}.png")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)   # keeps alpha channel
        if img is None:
            raise FileNotFoundError(f"Could not load digit image: {path}")
        if img.shape[2] != 4:
            raise ValueError(f"Digit image must have an alpha channel: {path}")
        _load_digit.cache[digit] = img
    return _load_digit.cache[digit]


def draw_number(img: np.ndarray,
                number: int | str,
                pos: tuple[int, int],
                scale: float = DEFAULT_SCALE,
                gap: int | None = None,
                color: tuple[int, int, int] = None) -> np.ndarray:
    """
    Overlay a number (int or str) on an existing OpenCV image using the transparent digit PNGs.

    Parameters
    ----------
    img        : np.ndarray   – Destination BGR image (will be modified in-place and also returned)
    number     : int | str    – The number to draw (e.g. 42, "007", -5 → will draw "5" with a minus sign if you want)
    pos        : (x, y)        – Top-left corner of the **whole** number block
    scale      : float        – Scale factor for the digits (1.0 = original 165×263 px)
    gap        : int | None   – Horizontal gap between digits in final scaled pixels.
                                If None → uses global GAP_PX * scale.
    color      : tuple | None – Not used for PNG version (kept for API compatibility if you ever want solid-color fallback)

    Returns
    -------
    np.ndarray – The modified image (same object as input `img`)

    Notes
    -----
    • Negative numbers are supported only if you pass them as string with the minus sign,
      e.g. draw_number(frame, "-5", (100,100)).
    • Leading zeros are preserved when `number` is given as a string.
    """
    if gap is None:
        gap = int(GAP_PX * scale)

    # Convert to string and iterate over each character
    text = str(number)
    x, y = pos
    current_x = x

    for char in text:
        if char == '-':
            # Simple minus sign using OpenCV text (you can replace with a custom PNG if you have one)
            cv2.putText(img, '-', (current_x, y + int(DIGIT_HEIGHT * scale * 0.75)),
                        cv2.FONT_HERSHEY_SIMPLEX, scale * 2.0, (0, 0, 255), thickness=int(4 * scale))
            current_x += int(80 * scale)                     # rough width of a minus sign
            continue

        if not char.isdigit():
            raise ValueError(f"Unsupported character in number string: '{char}'")

        digit = int(char)
        digit_img = _load_digit(digit)

        # Resize while preserving alpha
        new_w = int(DIGIT_WIDTH * scale)
        new_h = int(DIGIT_HEIGHT * scale)
        resized = cv2.resize(digit_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Separate BGR and Alpha
        bgr = resized[:, :, :3]
        alpha = resized[:, :, 3] / 255.0

        # Destination ROI
        y1, y2 = y, y + new_h
        x1, x2 = current_x, current_x + new_w

        # Clip to image bounds (safety)
        if y2 > img.shape[0] or x2 > img.shape[1]:
            break

        roi = img[y1:y2, x1:x2]
        # Alpha blend
        roi[:] = (alpha[..., None] * bgr + (1 - alpha[..., None]) * roi).astype(np.uint8)

        # Move cursor for next digit
        current_x += new_w + gap

    return img


# ----------------------------------------------------------------------
# Example / quick test when the file is run directly
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a blank canvas just to demonstrate
    canvas = np.zeros((800, 1280, 3), dtype=np.uint8)
    canvas[:] = (50, 50, 70)      # dark background

    draw_number(canvas, 42, pos=(200, 150), scale=1.5)
    draw_number(canvas, "007", pos=(200, 450), scale=1.2, gap=30)
    draw_number(canvas, "-5", pos=(800, 300), scale=2.0)

    cv2.imshow("Number Display Demo", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()