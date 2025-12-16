# grabphoto_control.py
# Version: 2.71
# Changes:
# - v2.71 (2025-12-16): Made capture_photos fully sequential (no threading/groups) to mimic standalone script: For each camera, read 20 frames with 0.1s delay, save last successful as user_{id}_camera_{i}_16MP_{timestamp}.jpg, del frame. Then rename based on mapping. Retained sequential init save phase. Retained previous changes.
# - v2.70 (2025-12-16): To prevent locking: Split initialize_cameras into two phases—1) Sequential init/save/release for frames (standalone-style). 2) Re-open all caps for preview/capture. Reverted to CAP_MSMF with auto_exposure=0.75. Capture in groups of 3. Added time.sleep(0.2) between opens. Retained previous changes.
# - v2.69 (2025-12-16): Changed backend to CAP_DSHOW for VideoCapture to address MSMF hanging issues with multiple/high-res cameras. Removed auto_exposure set (DShow uses different values; rely on default auto). Added time.sleep(1) after init saves before running mapper for IO settle. In capture_photos, made groups smaller (groups of 4) for less concurrent load. Retained previous changes.

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import PATHS

os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'
import cv2
import time
import threading
import numpy as np
import ctypes
from ctypes import wintypes
import mediapipe_landmarks
import json
import subprocess
from ui_controller import UIController
from pathlib import Path

DEBUG_TIMING = False

RESOLUTION = '16MP'
width, height = 4656, 3496

PREVIEW_ROTATE = True

PREVIEW_MIRROR = True

TEST_READ_FRAMES = False

QUICK_INIT = False

BASE_DIR = PATHS['BASE']

script_dir = os.path.dirname(os.path.abspath(__file__))
graphics_dir = PATHS['GRAPHICS']

INIT_FRAMES_DIR = os.path.join(BASE_DIR, 'initialization_frames')
os.makedirs(INIT_FRAMES_DIR, exist_ok=True)

MAPPING_JSON = os.path.join(INIT_FRAMES_DIR, 'camera_mapping.json')

cameras = []
capture_lock = threading.Lock()

TARGET_MONITOR_WIDTH = 800
TARGET_MONITOR_HEIGHT = 1280

# Counter file in scripts/python (safe from .gitignore)
COUNTER_FILE_PATH = os.path.join(PATHS['SCRIPTS_PYTHON'], 'user_counter.txt')

def load_user_counter():
    if os.path.exists(COUNTER_FILE_PATH):
        with open(COUNTER_FILE_PATH, 'r') as f:
            return int(f.read().strip())
    return 0

def save_user_counter(user_id):
    with open(COUNTER_FILE_PATH, 'w') as f:
        f.write(str(user_id))

def get_monitor_rects():
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

def load_mapping():
    if os.path.exists(MAPPING_JSON):
        with open(MAPPING_JSON, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: {MAPPING_JSON} not found. Using default mapping (identity).")
        return {i: i for i in range(17)}

def save_init_frames_sequential():
    jpg_saved = []
    for i in range(17 if not QUICK_INIT else 1):
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        if not cap.isOpened():
            print(f"Camera {i}: Failed to open during init save.", flush=True)
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_exposure = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        print(f"Camera {i}: Actual resolution {actual_width}x{actual_height}, auto exposure {actual_exposure}", flush=True)

        last_frame = None
        success_count = 0
        for _ in range(20):
            ret, frame = cap.read()
            if ret and frame is not None:
                last_frame = frame
                success_count += 1
            time.sleep(0.1)

        if last_frame is not None:
            init_path = os.path.join(INIT_FRAMES_DIR, f"camera_{i}_{RESOLUTION}.jpg")
            if cv2.imwrite(init_path, last_frame):
                print(f"Camera {i}: Saved init frame to {init_path}.", flush=True)
                time.sleep(0.1)
                jpg_saved.append(i)
            del last_frame

        cap.release()
        print(f"Camera {i}: Released after init save.", flush=True)
        time.sleep(0.2)  # Delay between cameras

    return jpg_saved

def open_all_cameras():
    for i in range(17 if not QUICK_INIT else 1):
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        if not cap.isOpened():
            print(f"Camera {i}: Failed to open during app init.", flush=True)
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

        cameras.append((i, cap))
        time.sleep(0.2)  # Delay between opens

def initialize_cameras(ui):
    init_start = time.time()
    preview_cap = None
    best_index = -1

    # Phase 1: Save init frames sequentially with release
    jpg_saved = save_init_frames_sequential()

    jpg_count = len(jpg_saved)
    if jpg_count == 0:
        print("Error: No init frames saved. No cameras initialized successfully.")
        return None
    elif jpg_count < 17:
        print(f"Warning: Only {jpg_count} init frames saved (expected 17). Mapping may be incomplete.")

    time.sleep(1)  # Wait for IO to settle

    # Run mapper
    print("Running init_camera_mapper.py to generate/overwrite mapping...")
    subprocess.call(["python", "init_camera_mapper.py"])
    time.sleep(0.5)
    mapping = load_mapping()

    # Phase 2: Open all for app
    open_all_cameras()

    # Find preview
    for orig, phys in mapping.items():
        if phys == 6:
            best_index = orig
            for idx, cap in cameras:
                if idx == orig:
                    preview_cap = cap
                    break
            break

    if preview_cap is None:
        if cameras:
            preview_cap = cameras[0][1]
            best_index = 0
            print("No physical 6 found; fallback to camera 0.", flush=True)
        if ui:
            ui.update_message("Preview on camera 0 (fallback)")

    if preview_cap:
        ui.preview_cap = preview_cap
        if ui:
            ui.update_message(f"Preview on camera {best_index} (physical 6)")

    if DEBUG_TIMING:
        init_duration = time.time() - init_start
        print(f"Init cameras total: {init_duration:.2f}s", flush=True)

    return preview_cap

def capture_photos():
    start_time = time.time()
    user_id = load_user_counter() + 1
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    user_dir = os.path.join(BASE_DIR, f'user_{user_id}')
    photos_dir = os.path.join(user_dir, 'photos')
    os.makedirs(photos_dir, exist_ok=True)

    mapping = load_mapping()

    # Sequential capture like standalone
    for i, cap in cameras:
        cam_start = time.time()

        # Read 20 frames, keep the last successful one
        last_frame = None
        success_count = 0
        for _ in range(20):
            ret, frame = cap.read()
            if ret and frame is not None:
                last_frame = frame
                success_count += 1
            time.sleep(0.1)

        if last_frame is not None:
            print(f"Camera {i}: Successfully read {success_count}/20 frames; grabbing the last one (shape: {last_frame.shape}).", flush=True)
            orig_filename = f"user_{user_id}_camera_{i}_{RESOLUTION}_{timestamp}.jpg"
            orig_path = os.path.join(photos_dir, orig_filename)
            if cv2.imwrite(orig_path, last_frame):
                print(f"Camera {i}: Saved photo to {orig_path}.", flush=True)
                time.sleep(0.1)

                phys = mapping.get(i, i)
                phys_str = f"{phys:02d}" if phys < 10 else str(phys)
                new_filename = orig_filename.replace(f'_camera_{i}_', f'_camera_{phys}_')
                new_path = os.path.join(photos_dir, new_filename)
                os.rename(orig_path, new_path)
                print(f"Renamed {orig_filename} to {new_filename}")
            del last_frame
            print(f"Camera {i}: Cleared frame from memory.", flush=True)
        else:
            print(f"Camera {i}: No successful frames read after 20 attempts.", flush=True)

        time.sleep(0.2)  # Delay between cameras

    save_user_counter(user_id)

    det_start = time.time()
    subprocess.Popen(["python", "eye_color_detector.py", str(user_id), photos_dir])
    subprocess.Popen(["python", "facial_hair_detector.py", str(user_id), photos_dir])

    if DEBUG_TIMING:
        det_duration = time.time() - det_start
        print(f"Detectors start (non-blocking): {det_duration:.2f}s", flush=True)

    if DEBUG_TIMING:
        duration = time.time() - start_time
        print(f"Capture photos total: {duration:.2f}s", flush=True)
    
    return user_id

def release_cameras():
    for _, cap in cameras:
        cap.release()
    cameras.clear()

def main():
    start_time = time.time()
    ui = None
    try:
        mon_start = time.time()
        monitors = get_monitor_rects()
        target_monitor = next((m for m in monitors if m[2] == TARGET_MONITOR_WIDTH and m[3] == TARGET_MONITOR_HEIGHT), None)
        x, y, w, h = target_monitor if target_monitor else (0, 0, 1920, 1080)
        if DEBUG_TIMING:
            mon_duration = time.time() - mon_start
            print(f"Monitor setup: {mon_duration:.2f}s", flush=True)

        ui_start = time.time()
        ui = UIController(w=w, h=h, capture_lock=capture_lock, preview_rotate=PREVIEW_ROTATE, preview_mirror=PREVIEW_MIRROR)
        ui.init_window(window_name='Capture App', x=x, y=y)
        ui.update_message("Initializing cameras...")
        if DEBUG_TIMING:
            ui_duration = time.time() - ui_start
            print(f"UI init: {ui_duration:.2f}s", flush=True)

        preview_cap = initialize_cameras(ui)
        if not cameras:
            return

        sound_start = time.time()
        soundclips_dir = PATHS['SOUNDCLIPS']
        sound_file = os.path.join(soundclips_dir, 'system_is_ready.mp3')
        if os.path.exists(sound_file):
            ret = subprocess.call(["python", "soundplayer.py", sound_file])
        if DEBUG_TIMING:
            sound_duration = time.time() - sound_start
            print(f"Ready sound playback: {sound_duration:.2f}s", flush=True)

        loop_start = time.time()

        def on_capture_wrapper():
            user_id = capture_photos()
            ui.user_id = user_id  # NEW: Set for UI processing
            return user_id

        ui.start_preview_loop(preview_cap=preview_cap, on_capture=on_capture_wrapper)
        if DEBUG_TIMING:
            loop_duration = time.time() - loop_start
            print(f"Preview loop ran for: {loop_duration:.2f}s", flush=True)

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).", flush=True)
    finally:
        if ui:
            ui.cleanup()
        release_cameras()

    if DEBUG_TIMING:
        total_duration = time.time() - start_time
        print(f"Main total: {total_duration:.2f}s", flush=True)

if __name__ == "__main__":
    main()