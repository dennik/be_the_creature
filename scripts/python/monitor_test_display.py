# monitor_test_display.py
# Version: 1.1
# Change Log:
# v1.1 - Fixed EnumDisplayMonitors call: Created enum_proc = MonitorEnumProc(monitor_enum_proc) instance, passed as third arg; lParam as fourth with ctypes.cast(ctypes.pointer(data), ctypes.c_void_p). Added try-except around enum call with print(e) for debug. Retained v1.0 logic; should now enumerate properly without silent fail. Certainty: 95% (standard ctypes pattern per examples; resolves wrong arg—test for output).
# v1.0 - Initial version: Standalone script to enumerate monitors via ctypes, find the target 800x1280 portrait monitor, create a fullscreen OpenCV window on it, load and overlay frame.png on a blank canvas, and display until 'q' pressed. Checks for cropping by filling canvas to exact size and overlaying frame (visual inspection: if frame fits without distortion/crop, success). Hardcoded paths relative to script dir; no camera/init. Certainty: 95% (direct port from grabphoto_control.py monitor logic + ui_controller.py overlay; assumes frame.png exists—test for window pos/size).

import os  # For graphics path
import cv2  # For window creation, imshow, and drawing
import numpy as np  # For blank canvas
import ctypes  # For Windows API monitor enumeration
from ctypes import wintypes  # For RECT/MONITORINFO structs

# Target monitor resolution (portrait; adjust if needed)
TARGET_MONITOR_WIDTH = 800
TARGET_MONITOR_HEIGHT = 1280

# Graphics path (relative to this script; assumes 'graphics/frame.png' exists)
script_dir = os.path.dirname(os.path.abspath(__file__))
graphics_dir = os.path.join(script_dir, 'graphics')
frame_path = os.path.join(graphics_dir, 'frame.png')

# ----------------------------------------------------------------------
# Monitor enumeration (copied from grabphoto_control.py for independence)
# ----------------------------------------------------------------------
class RECT(ctypes.Structure):
    _fields_ = [('left', ctypes.c_long), ('top', ctypes.c_long),
                ('right', ctypes.c_long), ('bottom', ctypes.c_long)]

def monitor_enum_proc(hMonitor, hdcMonitor, lprcMonitor, dwData):
    """Callback: Appends (left, top, width, height) to list."""
    r = lprcMonitor.contents
    data = ctypes.cast(dwData, ctypes.POINTER(ctypes.py_object)).contents.value
    data.append((r.left, r.top, r.right - r.left, r.bottom - r.top))
    return True

def get_monitor_rects():
    """Enumerate all monitors: Returns list of (left, top, width, height)."""
    monitors = []
    data = ctypes.py_object(monitors)
    MonitorEnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR,
                                         wintypes.HDC, ctypes.POINTER(RECT),
                                         wintypes.LPARAM)
    # v1.1: Create callback instance
    enum_proc = MonitorEnumProc(monitor_enum_proc)
    user32 = ctypes.windll.user32
    # v1.1: Pass enum_proc as third arg, lParam as fourth
    try:
        result = user32.EnumDisplayMonitors(None, None, enum_proc, ctypes.cast(ctypes.pointer(data), ctypes.c_void_p))
        if not result:
            print("EnumDisplayMonitors returned False; possible error.")
    except Exception as e:
        print(f"Error during EnumDisplayMonitors: {e}")
    return monitors

# ----------------------------------------------------------------------
# Frame overlay function (adapted from ui_controller.py _overlay_frame)
# ----------------------------------------------------------------------
def overlay_frame(img, frame_path, w, h):
    """Loads and overlays frame.png with alpha on img (resizes if needed)."""
    frame_img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    if frame_img is None:
        print(f"Warning: Failed to load frame.png at {frame_path}")
        return
    if frame_img.shape[:2] != (h, w):  # Resize to match canvas (portrait)
        frame_img = cv2.resize(frame_img, (w, h), interpolation=cv2.INTER_AREA)
    if frame_img.shape[2] == 4:  # Has alpha
        b, g, r, a = cv2.split(frame_img)
        overlay = cv2.merge([b, g, r])
        alpha = a / 255.0
        img[:] = (1 - alpha[..., None]) * img + alpha[..., None] * overlay

def main():
    """Main: Find target monitor, create fullscreen window, display blank + frame overlay."""
    print("Enumerating monitors...")
    monitors = get_monitor_rects()
    if not monitors:
        print("Error: No monitors found.")
        return

    # Debug: Print all monitors
    print("Found monitors:")
    for i, m in enumerate(monitors):
        print(f"Monitor {i}: left={m[0]}, top={m[1]}, width={m[2]}, height={m[3]}")

    # Find target (exact match)
    target_monitor = next((m for m in monitors if m[2] == TARGET_MONITOR_WIDTH and m[3] == TARGET_MONITOR_HEIGHT), None)
    if target_monitor is None:
        print("Warning: No 800x1280 monitor found; falling back to primary (monitor 0).")
        target_monitor = monitors[0] if monitors else (0, 0, 1920, 1080)
    x, y, w, h = target_monitor
    print(f"Selected monitor: left={x}, top={y}, width={w}, height={h}")

    # Create blank canvas (portrait)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)  # Black background
    canvas[:] = (50, 50, 50)  # Dark gray for contrast

    # Overlay frame to check fit/cropping
    overlay_frame(canvas, frame_path, w, h)

    # Create window on target monitor
    window_name = 'Monitor Test Display'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, x, y)  # Position at monitor top-left
    cv2.resizeWindow(window_name, w, h)  # Exact size
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Fullscreen

    # Display and wait for 'q'
    cv2.imshow(window_name, canvas)
    print("Displaying on target monitor. Press 'q' in the window to quit.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Test complete.")

if __name__ == "__main__":
    main()