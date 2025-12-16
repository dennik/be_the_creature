# monitor_debug.py
# Version: 1.0
# Changes:
# - v1.0 (2025-12-15): Initial version. Standalone script to enumerate and print resolutions and positions of all connected monitors using Windows API via ctypes.

import ctypes
from ctypes import wintypes

def get_monitor_rects():
    """
    Enumerates all connected monitors using Windows API via ctypes.
    Returns a list of tuples: (left, top, width, height) for each monitor.
    """
    monitors = []
    def callback(hMonitor, hdcMonitor, lprcMonitor, dwData):
        r = lprcMonitor.contents
        monitors.append((r.left, r.top, r.right - r.left, r.bottom - r.top))
        return True
    EnumDisplayMonitors = ctypes.windll.user32.EnumDisplayMonitors
    MonitorEnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(wintypes.RECT), ctypes.c_void_p)
    rect = wintypes.RECT()
    EnumDisplayMonitors(None, None, MonitorEnumProc(callback), 0)
    return monitors

if __name__ == "__main__":
    monitors = get_monitor_rects()
    if not monitors:
        print("No monitors detected.")
    else:
        for i, (left, top, width, height) in enumerate(monitors, start=1):
            print(f"Monitor {i}: Position (left={left}, top={top}), Resolution (width={width}, height={height})")