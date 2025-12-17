# grabphoto_control.py
# Version: 2.73
# Changes:
# - v2.73 (2025-12-16): Pivoted to SSIM-based mapping: Deleted all old remapping code/flags (init_camera_mapper.py, SKIP_REMAP). In initialize_cameras, after saving init frames, call ssim_camera_mapper.py to generate ssim_camera_mapping.json. Updated MAPPING_JSON to use this new file. Retained previous changes.
# - v2.72 (2025-12-16): Added copy_pre_aligned_xmp(photos_dir) to copy/rename XMPs from aligned_xmp dir after photos saved/renamed, before detectors. Retained previous changes.
# - v2.71 (2025-12-16): In capture_photos, added wait loop (up to 10s) after threads join to confirm >=17 .jpg files exist before launching detectors, ensuring they see complete set. Retained previous changes.

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
import shutil
import re

DEBUG_TIMING = False

RESOLUTION = '8MP'
width, height = 3840, 2160

MANUAL_EXPOSURE = 0.06

PREVIEW_ROTATE = False

PREVIEW_MIRROR = True

TEST_READ_FRAMES = False

QUICK_INIT = False

BASE_DIR = PATHS['BASE']

script_dir = os.path.dirname(os.path.abspath(__file__))
graphics_dir = PATHS['GRAPHICS']

INIT_FRAMES_DIR = os.path.join(BASE_DIR, 'initialization_frames')
os.makedirs(INIT_FRAMES_DIR, exist_ok=True)

MAPPING_JSON = os.path.join(INIT_FRAMES_DIR, 'ssim_camera_mapping.json')

ALIGNED_XMP_DIR = os.path.join(BASE_DIR, 'aligned_xmp')

cameras = []
capture_lock = threading.Lock()

TARGET_MONITOR_WIDTH = 800
TARGET_MONITOR_HEIGHT = 1280

# Counter file in BASE_DIR (photogrammetry)
COUNTER_FILE_PATH = os.path.join(BASE_DIR, 'user_counter.txt')

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
            data = json.load(f)
            return {int(k): int(v) for k, v in data.items()}
    else:
        print(f"Warning: {MAPPING_JSON} not found. Using default mapping (identity).")
        return {i: i for i in range(17)}

def initialize_cameras(ui):
    init_start = time.time()
    preview_cap = None
    best_index = -1

    mapping = load_mapping()
    
    for i in range(17 if not QUICK_INIT else 1):
        cam_start = time.time()
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            print(f"Camera {i} failed to open.", flush=True)
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_EXPOSURE, MANUAL_EXPOSURE)

        stable = False
        for _ in range(30):
            ret, frame = cap.read()
            if ret and frame is not None:
                stable = True
            time.sleep(0.033)

        if not stable:
            print(f"Camera {i} no stable frame after stabilization.", flush=True)
            cap.release()
            continue

        frame_saved = False
        for attempt in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                init_path = os.path.join(INIT_FRAMES_DIR, f"init_camera_{i}.jpg")
                if cv2.imwrite(init_path, frame):
                    print(f"Saved init frame for camera {i} to {init_path}", flush=True)
                    frame_saved = True
                else:
                    print(f"Failed to write init frame for camera {i} to {init_path}")
                break
            time.sleep(0.033)
        if not frame_saved:
            print(f"Failed to read/save frame for camera {i} after retries.", flush=True)

        cameras.append((i, cap))
        if DEBUG_TIMING:
            cam_duration = time.time() - cam_start
            print(f"Camera {i} init: {cam_duration:.2f}s", flush=True)

    # Check saved frames
    jpg_count = len(list(Path(INIT_FRAMES_DIR).glob("*.jpg")))
    if jpg_count == 0:
        print("Error: No init frames saved. No cameras initialized successfully.")
        return None
    elif jpg_count < 17:
        print(f"Warning: Only {jpg_count} init frames saved (expected 17). Mapping may be incomplete.")

    # Run SSIM mapper
    print("Running ssim_camera_mapper.py to generate mapping...")
    subprocess.call(["python", "ssim_camera_mapper.py"])
    mapping = load_mapping()  # Reload after generation

    # Find orig_idx for physical 6
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

def copy_pre_aligned_xmp(photos_dir):
    photos_dir = Path(photos_dir)
    aligned_dir = Path(ALIGNED_XMP_DIR)
    if not aligned_dir.exists():
        print(f"Warning: Aligned XMP directory {aligned_dir} does not exist. Skipping XMP copy.")
        return

    jpg_files = list(photos_dir.glob("*.jpg"))
    copied_count = 0
    for jpg_path in jpg_files:
        base_name = jpg_path.stem
        match = re.search(r'camera_(\d+)', base_name)
        if not match:
            print(f"Warning: No 'camera_#' found in {base_name}. Skipping XMP copy.")
            continue
        cam_idx_str = match.group(1)
        try:
            cam_idx = int(cam_idx_str)
        except ValueError:
            print(f"Warning: Could not parse integer from {cam_idx_str} in {base_name}. Skipping.")
            continue

        src_xmp = aligned_dir / f"camera_{cam_idx:02d}.xmp"
        if not src_xmp.exists():
            print(f"Warning: No source XMP {src_xmp} for camera {cam_idx}. Skipping.")
            continue

        dest_xmp = photos_dir / f"{base_name}.xmp"
        shutil.copy(src_xmp, dest_xmp)
        print(f"Copied {src_xmp.name} to {dest_xmp}")
        copied_count += 1

    print(f"Copied {copied_count} XMP files to {photos_dir}.")

def capture_photos():
    start_time = time.time()
    user_id = load_user_counter() + 1
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    user_dir = os.path.join(BASE_DIR, f'user_{user_id}')
    photos_dir = os.path.join(user_dir, 'photos')
    os.makedirs(photos_dir, exist_ok=True)

    # print(f"Capturing photos for user_{user_id} → {photos_dir}")  # Commented out debug print

    mapping = load_mapping()

    def capture_group(camera_list, lock):
        for i, cap in camera_list:
            cam_start = time.time()
            with capture_lock:
                ret, frame = cap.read()
            if ret and frame is not None:
                phys = mapping.get(i, i)  # Fallback to orig if not in mapping
                phys_str = f"{phys:02d}" if phys < 10 else str(phys)
                filename = f"user_{user_id}_camera_{phys_str}_{RESOLUTION}_{timestamp}_final.jpg"
                path = os.path.join(photos_dir, filename)
                cv2.imwrite(path, frame)
                print(f"Saved {filename}")
            if DEBUG_TIMING:
                cam_duration = time.time() - cam_start
                print(f"Camera {i} read/write: {cam_duration:.2f}s", flush=True)

    camera_groups = [cameras[0:9], cameras[9:17]]
    threads = []
    lock = threading.Lock()
    for group in camera_groups:
        if group:
            t = threading.Thread(target=capture_group, args=(group, lock))
            threads.append(t)
            t.start()
    for t in threads:
        t.join()

    # Wait until all photos are present
    expected_count = len(cameras)
    wait_start = time.time()
    while time.time() - wait_start < 10:
        jpg_count = len(list(Path(photos_dir).glob("*.jpg")))
        if jpg_count >= expected_count:
            break
        time.sleep(0.5)
    else:
        print(f"Warning: Only {jpg_count} photos found after wait (expected {expected_count}). Proceeding anyway.")

    # Copy pre-aligned XMPs after photos saved/renamed
    copy_pre_aligned_xmp(photos_dir)

    save_user_counter(user_id)

    det_start = time.time()
    subprocess.Popen(["python", "eye_color_detector.py", str(user_id), photos_dir])
    subprocess.Popen(["python", "facial_hair_detector.py", str(user_id), photos_dir])
    # REMOVED: No processor launch here—UI handles via class/queue

    if DEBUG_TIMING:
        det_duration = time.time() - det_start
        print(f"Detectors start (non-blocking): {det_duration:.2f}s", flush=True)

    if DEBUG_TIMING:
        duration = time.time() - start_time
        print(f"Capture photos total: {duration:.2f}s", flush=True)
    
    return user_id  # NEW: Return user_id for UI to start processor

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