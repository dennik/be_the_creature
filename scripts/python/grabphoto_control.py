# grabphoto_control.py
# Version: 2.54
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
from skimage.metrics import structural_similarity as ssim

DEBUG_TIMING = True

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

ref_image_path = os.path.join(graphics_dir, 'preview_photo.jpg')
ref_image = cv2.imread(ref_image_path, cv2.IMREAD_GRAYSCALE)
if ref_image is None:
    print(f"Warning: Failed to load reference image at {ref_image_path}. SSIM detection disabled.")
    ref_gray = None
else:
    ref_gray = ref_image

SAVE_INIT_PHOTOS = False
INIT_PHOTO_DIR = PATHS['INIT_PHOTOS']

cameras = []
capture_lock = threading.Lock()

TARGET_MONITOR_WIDTH = 800
TARGET_MONITOR_HEIGHT = 1280

COUNTER_FILE = os.path.join(PATHS['SCRIPTS_PYTHON'], 'user_counter.txt')

def load_user_counter():
    if os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, 'r') as f:
            return int(f.read().strip())
    return 0

def save_user_counter(user_id):
    with open(COUNTER_FILE, 'w') as f:
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

def initialize_cameras(ui):
    init_start = time.time()
    preview_cap = None
    best_ssim_score = -1
    best_index = -1
    best_cap = None

    if ref_gray is None:
        print("SSIM disabled; fallback to first camera for preview.")
    
    if SAVE_INIT_PHOTOS:
        os.makedirs(INIT_PHOTO_DIR, exist_ok=True)
    
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

        if ref_gray is not None:
            ret, frame = cap.read()
            if ret and frame is not None:
                if SAVE_INIT_PHOTOS:
                    init_path = os.path.join(INIT_PHOTO_DIR, f"init_camera_{i}.jpg")
                    cv2.imwrite(init_path, frame)
                    print(f"Saved init photo for camera {i} to {init_path}", flush=True)
                
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                score = ssim(ref_gray, gray_frame, data_range=gray_frame.max() - gray_frame.min())
                ssim_start = time.time() if DEBUG_TIMING else None
                if score > best_ssim_score:
                    best_ssim_score = score
                    best_index = i
                    best_cap = cap
                if DEBUG_TIMING:
                    print(f"Camera {i} SSIM: {score:.4f} ({time.time() - ssim_start:.2f}s)", flush=True)
            else:
                print(f"Camera {i} failed post-stab read for SSIM.", flush=True)

        if TEST_READ_FRAMES:
            ret, frame = cap.read()
            if ret and frame is not None:
                if np.mean(frame) < 10:
                    print(f"Camera {i} post-init frame blank (mean <10).", flush=True)
            else:
                print(f"Camera {i} failed test read.", flush=True)

        cameras.append((i, cap))
        if ui:
            ui.update_message(f"Initialized camera {i+1}/17")
        if DEBUG_TIMING:
            cam_duration = time.time() - cam_start
            print(f"Camera {i} init: {cam_duration:.2f}s", flush=True)

    if best_cap and best_ssim_score > 0.5:
        preview_cap = best_cap
        print(f"Preview camera: {best_index} (SSIM {best_ssim_score:.4f})", flush=True)
        if ui:
            ui.update_message(f"Preview on camera {best_index}")
    elif cameras:
        preview_cap = cameras[0][1]
        print("No SSIM match; fallback to camera 0 for preview.", flush=True)
        if ui:
            ui.update_message("Preview on camera 0 (fallback)")

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

    def capture_group(camera_list, lock):
        for i, cap in camera_list:
            cam_start = time.time()
            with capture_lock:
                ret, frame = cap.read()
            if ret and frame is not None:
                filename = f"user_{user_id}_camera_{i}_{RESOLUTION}_{timestamp}.jpg"
                cv2.imwrite(os.path.join(photos_dir, filename), frame)
            if DEBUG_TIMING:
                cam_duration = time.time() - cam_start
                print(f"Camera {i} read/write: {cam_duration:.2f}s", flush=True)

    camera_groups = [cameras[i:i+4] for i in range(0, len(cameras), 4)]
    threads = []
    lock = threading.Lock()
    for group in camera_groups:
        if group:
            t = threading.Thread(target=capture_group, args=(group, lock))
            threads.append(t)
            t.start()
    for t in threads:
        t.join()

    save_user_counter(user_id)

    det_start = time.time()
    subprocess.Popen(["python", "eye_color_detector.py", str(user_id), photos_dir])
    subprocess.Popen(["python", "facial_hair_detector.py", str(user_id), photos_dir])

    # UPDATED: Launch with stdout piping for progress monitoring
    processor_process = subprocess.Popen(
        ["python", "realityscan_processor.py", str(user_id)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        universal_newlines=True
    )

    if DEBUG_TIMING:
        det_duration = time.time() - det_start
        print(f"Detectors and processor start (non-blocking): {det_duration:.2f}s", flush=True)

    if DEBUG_TIMING:
        duration = time.time() - start_time
        print(f"Capture photos total: {duration:.2f}s", flush=True)
    
    # UPDATED: Return process for UI monitoring
    return user_id, processor_process

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
        ui.start_preview_loop(preview_cap=preview_cap, on_capture=capture_photos)
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