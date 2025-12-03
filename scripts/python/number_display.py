import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import PATHS

import cv2
import numpy as np

DIGITS_DIR = PATHS['NUMBERS']
DIGIT_WIDTH  = 165
DIGIT_HEIGHT = 263
GAP_PX       = 20
DEFAULT_SCALE = 1.0


def _load_digit(digit: int) -> np.ndarray:
    if not 0 <= digit <= 9:
        raise ValueError(f"Digit must be 0-9, got {digit}")

    if not hasattr(_load_digit, "cache"):
        _load_digit.cache = {}
    if digit not in _load_digit.cache:
        path = os.path.join(DIGITS_DIR, f"number_{digit}.png")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
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
    if gap is None:
        gap = int(GAP_PX * scale)

    text = str(number)
    x, y = pos
    current_x = x

    for char in text:
        if char == '-':
            cv2.putText(img, '-', (current_x, y + int(DIGIT_HEIGHT * scale * 0.75)),
                        cv2.FONT_HERSHEY_SIMPLEX, scale * 2.0, (0, 0, 255), thickness=int(4 * scale))
            current_x += int(80 * scale)
            continue

        if not char.isdigit():
            raise ValueError(f"Unsupported character in number string: '{char}'")

        digit = int(char)
        digit_img = _load_digit(digit)

        new_w = int(DIGIT_WIDTH * scale)
        new_h = int(DIGIT_HEIGHT * scale)
        resized = cv2.resize(digit_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        bgr = resized[:, :, :3]
        alpha = resized[:, :, 3] / 255.0

        y1, y2 = y, y + new_h
        x1, x2 = current_x, current_x + new_w

        if y2 > img.shape[0] or x2 > img.shape[1]:
            break

        roi = img[y1:y2, x1:x2]
        roi[:] = (alpha[..., None] * bgr + (1 - alpha[..., None]) * roi).astype(np.uint8)

        current_x += new_w + gap

    return img


if __name__ == "__main__":
    canvas = np.zeros((800, 1280, 3), dtype=np.uint8)
    canvas[:] = (50, 50, 70)

    draw_number(canvas, 42, pos=(200, 150), scale=1.5)
    draw_number(canvas, "007", pos=(200, 450), scale=1.2, gap=30)
    draw_number(canvas, "-5", pos=(800, 300), scale=2.0)

    cv2.imshow("Number Display Demo", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()