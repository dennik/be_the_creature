# grabphoto_control.py
# Version: 2.53
# Change Log:
# v2.53 - Added SAVE_INIT_PHOTOS global flag (default False): If True, during initialize_cameras, after stabilization and SSIM frame read, saves the frame to INIT_PHOTO_DIR (r"d:\python\graphics\initialization") as init_camera_{i}.jpg (creates dir if missing). No extra read (uses SSIM frame; skips if no ref_gray/SSIM disabled). Retained v2.52 user dir structure and prior; certainty: 95% (simple debug save; assumes SSIM frame rep for init—enable flag for testing).
# v2.52 - Updated to new directory structure: Replaced SAVE_DIR with BASE_DIR=r'd:\photogrammetry'; in capture_photos, create user_dir=BASE_DIR\user_{id}, photos_dir=user_dir\photos, save photos to photos_dir; call detectors with photos_dir (assumes they save txt to parent user_dir); added non-blocking call to photogrammetry_processor.py with user_id post-detectors for 3D processing. Retained v2.51 SSIM integration and prior; no other changes. Certainty: 95% (organizes per user; assumes modified detectors/RS for paths—test for creation/moves).
# v2.51 - Integrated SSIM preview detection into initialize_cameras: Pre-load ref_gray before loop; after stabilization reads, read one stable frame per camera, compute SSIM immediately, track best score/index/cap. After loop, set preview_cap and update UI based on best (fallback to first if no match). Removed separate find_preview_camera call in main (now baked in for ~17s savings by avoiding redundant reads). Retained v2.50 EnumDisplayMonitors fix, v2.49 DEBUG_TIMING, and all prior; no other changes. Certainty: 90% (eliminates duplicate reads; assumes post-stab frame is rep—test for exposure drift).

import os
import sys  # For flushing output to ensure real-time logging during long init/capture
# Suppress OpenCV warnings (e.g., backend issues) to clean stderr; set before cv2 import
os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'  # Only fatal errors; adjust to 'ERROR' if needed for debugging
import cv2  # Core library for camera capture, image processing, and GUI display
import time  # For timestamps in filenames and delays during exposure stabilization
import threading  # For parallel capture across camera groups to minimize sync delays
import numpy as np  # For image padding/cropping in preview display (e.g., letterboxing)
import ctypes  # For low-level Windows API calls in monitor enumeration and window styling
from ctypes import wintypes  # For monitor enumeration structures (e.g., RECT, MONITORINFO)
import mediapipe_landmarks  # Custom module for real-time face landmark detection and overlay on preview frames
import json  # For parsing PowerShell output in init (unused in this version but retained)
import subprocess  # For PowerShell in init (v2.22; unused) and detector calls
from ui_controller import UIController  # Modular UI handler for preview and buttons
from skimage.metrics import structural_similarity as ssim  # For SSIM in preview detection (pip install scikit-image)

# Global flag to toggle debug timing prints (set to True for verbose output during init/SSIM/capture)
DEBUG_TIMING = True  # Change to True to enable per-process time printouts

# Hardcode resolution to 8MP (3840x2160) - shared global for consistency across all cameras
RESOLUTION = '8MP'
width, height = 3840, 2160  # Defines the target capture resolution; Arducam IMX519 supports this

# Tunable manual exposure (seconds) if auto fails - shared global for all cameras
MANUAL_EXPOSURE = 0.06  # 60ms; adjust 0.01–0.1 if photos are too dark/bright under fixed lighting

# Optional preview rotation (180°) for camera orientation matching (e.g., for upside-down mounted cams like camera 7)
PREVIEW_ROTATE = False  # Set to True to enable 180° rotate in preview (MediaPipe/capture unaffected)

# Optional preview horizontal mirroring for natural self-view (mirrors person in real-time)
PREVIEW_MIRROR = True  # Set to False to disable horizontal flip in preview (MediaPipe/capture unaffected)

# Optional test read frames during init for debug (v2.24: logs if blank post-stabilization; set True for verbose check)
TEST_READ_FRAMES = False  # Default False to avoid overhead; True for debug like finder v1.4

# Quick init mode: Skip all but preview camera (first index) for fast testing of detections
QUICK_INIT = False  # Set to True for quick mode (only 1 cam); False for full 17-cam init

# Base directory (absolute path; creates user subdirs if missing) - where all user data is stored
BASE_DIR = r'd:\photogrammetry'  # Raw string for Windows paths to avoid escape issues

# Graphics directory (relative to script dir, for preview_photo.jpg and other assets)
script_dir = os.path.dirname(os.path.abspath(__file__))
graphics_dir = os.path.join(script_dir, 'graphics')

# Reference image for SSIM-based preview camera detection (grayscale, loaded once)
ref_image_path = os.path.join(graphics_dir, 'preview_photo.jpg')
ref_image = cv2.imread(ref_image_path, cv2.IMREAD_GRAYSCALE)
if ref_image is None:
    print(f"Warning: Failed to load reference image at {ref_image_path}. SSIM detection disabled.")
    ref_gray = None
else:
    ref_gray = ref_image  # Already gray

# v2.53: Init photo save flag and dir (for debug; saves post-stab SSIM frame per cam)
SAVE_INIT_PHOTOS = False  # Set to True to save initialization photos
INIT_PHOTO_DIR = r"d:\python\graphics\initialization"  # Dir for saved init photos (creates if missing)

# Global lists and locks (cameras: list of (index, cap); capture_lock for thread-safe reads)
cameras = []  # Filled during init: [(i, cap), ...]
capture_lock = threading.Lock()  # Shared lock for synchronized reads during capture

# Target monitor resolution for UI (adjust to match secondary monitor; fallback 1920x1080)
TARGET_MONITOR_WIDTH = 800
TARGET_MONITOR_HEIGHT = 1280

# Counter file for user ID increment (stored in BASE_DIR; creates if missing)
COUNTER_FILE = os.path.join(BASE_DIR, 'user_counter.txt')

def load_user_counter():
    """Load last user ID from file (starts at 0 if missing)."""
    if os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, 'r') as f:
            return int(f.read().strip())
    return 0

def save_user_counter(user_id):
    """Save current user ID to file."""
    with open(COUNTER_FILE, 'w') as f:
        f.write(str(user_id))

def get_monitor_rects():
    """Enumerate monitors via Windows API (ctypes); returns list of (x, y, w, h)."""
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
    """Init all 17 cameras: Open, set props, stabilize exposure, detect preview via SSIM (integrated). Returns preview_cap."""
    init_start = time.time()  # v2.47: Timing for init
    preview_cap = None
    best_ssim_score = -1
    best_index = -1
    best_cap = None

    # Pre-load ref_gray if exists (already done globally; check here for fallback)
    if ref_gray is None:
        print("SSIM disabled; fallback to first camera for preview.")
    
    # v2.53: Create init photo dir if flag True
    if SAVE_INIT_PHOTOS:
        os.makedirs(INIT_PHOTO_DIR, exist_ok=True)
    
    for i in range(17 if not QUICK_INIT else 1):
        cam_start = time.time()  # v2.47: Per-camera init timing
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            print(f"Camera {i} failed to open.", flush=True)
            continue

        # Set resolution and manual exposure (v2.3: via V4L2 backend props)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_EXPOSURE, MANUAL_EXPOSURE)  # Manual exposure (seconds)

        # Set stabilize exposure: Read ~30 frames (v2.4: adjustable; ~2s at 15fps)
        stable = False
        for _ in range(30):
            ret, frame = cap.read()
            if ret and frame is not None:
                stable = True
            time.sleep(0.033)  # ~30ms delay for settling

        if not stable:
            print(f"Camera {i} no stable frame after stabilization.", flush=True)
            cap.release()
            continue

        # v2.51: Integrated SSIM: Read one post-stab frame, compute gray/SSIM if ref
        if ref_gray is not None:
            ret, frame = cap.read()
            if ret and frame is not None:
                # v2.53: Save init photo if flag (uses this frame)
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

        # Optional test read (v2.24: verbose check)
        if TEST_READ_FRAMES:
            ret, frame = cap.read()
            if ret and frame is not None:
                if np.mean(frame) < 10:  # Arbitrary low mean for 'blank'
                    print(f"Camera {i} post-init frame blank (mean <10).", flush=True)
            else:
                print(f"Camera {i} failed test read.", flush=True)

        cameras.append((i, cap))
        if ui:
            ui.update_message(f"Initialized camera {i+1}/17")
        if DEBUG_TIMING:
            cam_duration = time.time() - cam_start
            print(f"Camera {i} init: {cam_duration:.2f}s", flush=True)

    # v2.51: Set preview_cap post-loop (best if SSIM > threshold else first)
    if best_cap and best_ssim_score > 0.5:  # Arbitrary threshold for 'match'
        preview_cap = best_cap
        print(f"Preview camera: {best_index} (SSIM {best_ssim_score:.4f})", flush=True)
        if ui:
            ui.update_message(f"Preview on camera {best_index}")
    elif cameras:
        preview_cap = cameras[0][1]  # Fallback to first
        print("No SSIM match; fallback to camera 0 for preview.", flush=True)
        if ui:
            ui.update_message("Preview on camera 0 (fallback)")

    if DEBUG_TIMING:
        init_duration = time.time() - init_start
        print(f"Init cameras total: {init_duration:.2f}s", flush=True)

    return preview_cap

def capture_photos():
    """Capture photos from all cameras, save with user_id prefix, call detectors/RS non-blocking."""
    start_time = time.time()  # v2.47: Start capture timing
    user_id = load_user_counter() + 1
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # v2.52: Create per-user dirs
    user_dir = os.path.join(BASE_DIR, f'user_{user_id}')
    photos_dir = os.path.join(user_dir, 'photos')
    os.makedirs(photos_dir, exist_ok=True)

    def capture_group(camera_list, lock):
        for i, cap in camera_list:
            cam_start = time.time()  # v2.47: Per-camera read/write timing
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

    # Detector scripts (non-blocking)
    det_start = time.time()  # v2.47: Timing for detectors
    subprocess.Popen(["python", "eye_color_detector.py", str(user_id), photos_dir])  # v2.48: Non-blocking
    subprocess.Popen(["python", "facial_hair_detector.py", str(user_id), photos_dir])  # v2.48: Non-blocking

    # v2.52: Call photogrammetry processor (non-blocking)
    subprocess.Popen(["python", "realityscan_processor.py", str(user_id)])


    if DEBUG_TIMING:
        det_duration = time.time() - det_start
        print(f"Detectors start (non-blocking): {det_duration:.2f}s", flush=True)

    if DEBUG_TIMING:
        duration = time.time() - start_time  # v2.47: End total capture timing
        print(f"Capture photos total: {duration:.2f}s", flush=True)
    return user_id  # Return to UI for starting processor with progress capture

def release_cameras():
    """Clean shutdown of all camera resources."""
    for _, cap in cameras:
        cap.release()
    cameras.clear()

def main():
    start_time = time.time()  # v2.47: Start overall main timing
    ui = None
    try:
        mon_start = time.time()  # v2.47: Timing for monitor setup
        monitors = get_monitor_rects()
        target_monitor = next((m for m in monitors if m[2] == TARGET_MONITOR_WIDTH and m[3] == TARGET_MONITOR_HEIGHT), None)
        x, y, w, h = target_monitor if target_monitor else (0, 0, 1920, 1080)
        if DEBUG_TIMING:
            mon_duration = time.time() - mon_start
            print(f"Monitor setup: {mon_duration:.2f}s", flush=True)

        ui_start = time.time()  # v2.47: Timing for UI init
        ui = UIController(w=w, h=h, capture_lock=capture_lock, preview_rotate=PREVIEW_ROTATE, preview_mirror=PREVIEW_MIRROR)  # v2.44: Pass flags
        ui.init_window(window_name='Capture App', x=x, y=y)
        ui.update_message("Initializing cameras...")
        if DEBUG_TIMING:
            ui_duration = time.time() - ui_start
            print(f"UI init: {ui_duration:.2f}s", flush=True)

        preview_cap = initialize_cameras(ui)  # v2.51: Now returns preview_cap (integrated detection)
        if not cameras:
            return

        # v2.36: Play ready sound post-init (relative path from script dir)
        sound_start = time.time()  # v2.47: Timing for sound
        soundclips_dir = os.path.join(os.path.dirname(__file__), 'soundclips')
        sound_file = os.path.join(soundclips_dir, 'system_is_ready.mp3')
        # print(f"Attempting to play sound at: {sound_file}")  # Debug log
        if os.path.exists(sound_file):
            ret = subprocess.call(["python", "soundplayer.py", sound_file])
            # print(f"Soundplayer return code: {ret}")  # Debug: 0 = success
        if DEBUG_TIMING:
            sound_duration = time.time() - sound_start
            print(f"Ready sound playback: {sound_duration:.2f}s", flush=True)

        loop_start = time.time()  # v2.47: Timing for preview loop (but loop runs until quit, so this is start only; add per-frame if needed)
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
        total_duration = time.time() - start_time  # v2.47: End overall main timing
        print(f"Main total: {total_duration:.2f}s", flush=True)

if __name__ == "__main__":
    main()