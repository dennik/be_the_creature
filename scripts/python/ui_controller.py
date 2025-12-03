# ui_controller.py
# v1.49 - Updated quit_button to image-based with 'quit.png'; removed drawn colors/text for image-only consistency. Certainty: 100% (assumes quit.png asset exists; no impact on other buttons).
# v1.48 - Made changes myself to remove import draw_rounded_rect from ui_button that was deleted
# v1.47 - Replaced time-based progress with real-time from processor: Removed processing_start/duration; added self.processor_process=None, self.progress_queue=queue.Queue(), self.current_progress=0. On entering processing, if process None, Popen realityscan_processor.py with user_id (assumed set post-capture), stdout=PIPE; start thread to read lines to queue. In loop, drain queue, update current_progress on "PROGRESS:" lines. Use current_progress for frame_idx (0-100) and percent draw. Retained v1.46 number placement and prior; certainty: 90% (non-blocking read; assumes user_id set—test for blocking/EOF).
# v1.46 - Adjusted number placement: Pre-calculate total_w = sum(new_w per digit) + gap*(len-1) at scale=0.5 (digits 82px w, gap=10). Centered: pos_x = bar_x + (bar_w - total_w)//2. Above bar: pos_y = bar_y - 20 - number_h (131). Retained v1.45 integrated bar/numbers and prior. Certainty: 95% (dynamic; assumes uniform digits—test for "1" vs "100").

import queue  # v1.47: For progress queue from subprocess
import cv2  # For window, drawing, imshow
import numpy as np  # For image ops (zeros, resize)
import time  # For fade timestamps, animation, and pending switch timing
import os  # v1.12: For sound file path; v1.15: For graphics paths
import subprocess  # v1.12: For calling soundplayer.py (retained but unused in v1.24 for button sound)
import mediapipe_landmarks  # For overlay_landmarks (imported here for preview processing)
from preview_enhancer import PreviewEnhancer  # v1.13: For readiness checks and overlays/sounds
from ui_button import Button  # Modular button class
import threading  # v1.11: For background capture thread
import pygame  # v1.24: For button sound playback
from progress_bar import overlay_transparent  # v1.45: For bar overlay (adapted helper)
from number_display import draw_number, DIGIT_WIDTH, DIGIT_HEIGHT, GAP_PX  # v1.45: For percentage text + constants

# v1.24: Initialize pygame mixer for button sound (global)
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

# ----------------------------------------------------------------------
# Global that the rest of the app can read after a creature is chosen
# ----------------------------------------------------------------------
creature_selection = None        # will be "sloth", "leopard" or "monkey"

class UIController:
    """
    Modular UI controller for grabphoto app: Handles canvas (800x1280 portrait), window setup, text messages (top-left, replace/fade),
    preview frame display (scaled to fit, with transforms/landmarks), and button management (via Button class).
    Init mode: Sequential message updates for progress (e.g., camera init), with optional fade/wait.
    Preview mode: Loop reads/processes frame, draws preview/buttons/msg, handles mouse for buttons.
    Modularity: Pass preview_cap and on_capture func; quit via internal flag. Cleanup destroys windows.
    v1.46: Dynamic number centering above bar.
    v1.45: Integrated progress bar + numbers.
    v1.44: Processing screen switch.
    """
    def __init__(self, w=800, h=1280, capture_lock=None, preview_rotate=False, preview_mirror=True):
        """
        Initialize canvas dimensions and internal state.
        Args:
            w (int): Canvas width (default 800 portrait).
            h (int): Canvas height (default 1280 portrait).
            capture_lock (threading.Lock or None): Shared lock for camera reads (v1.11).
            preview_rotate (bool): Enable 180° rotate in preview (default False).
            preview_mirror (bool): Enable horizontal flip in preview (default True).
        """
        self.w = w  # Canvas width
        self.h = h  # Canvas height
        self.preview_rotate = preview_rotate  # v1.19: Passed param for rotation
        self.preview_mirror = preview_mirror  # v1.19: Passed param for mirror
        self.window_name = None  # Set on init_window
        self.current_msg = None  # Current text message (str or None)
        self.fade_start = None  # Timestamp for fade (if set)
        self.fade_time = None  # Duration for fade (seconds)
        self.quit_flag = False  # Internal flag for quit (set by Quit button)
        self.on_capture = None  # Callback func for Capture button (set on start_preview_loop)
        self.preview_cap = None  # Set on start_preview_loop
        self.capture_thread = None  # v1.11: Local thread for async capture
        self.capture_lock = capture_lock  # v1.11: Shared lock for reads
        self.face_detected = False  # v1.12: State flag for face detection transitions
        self.preview_enhancer = PreviewEnhancer()  # v1.13: For readiness checks/overlays/sounds
        self.current_frame = None  # v1.20: Last full_img for immediate redraw on events
        self.current_preview = None  # v1.21: Last display_img (preview with border) for rebuild
        # v1.15: Graphics paths (relative to script dir)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        graphics_dir = os.path.join(script_dir, 'graphics')
        self.frame_path = os.path.join(graphics_dir, 'frame.png')
        self.sloth_path = os.path.join(graphics_dir, 'sloth.png')
        self.leopard_path = os.path.join(graphics_dir, 'leopard.png')
        self.monkey_path = os.path.join(graphics_dir, 'monkey.png')
        self.return_path = os.path.join(graphics_dir, 'return.png')
        self.button_path = os.path.join(graphics_dir, 'button.png')
        self.frame_img = None  # Loaded lazily with alpha
        # UI layout constants
        self.button_margin = 10  # Padding for text/button pos
        self.button_bottom_margin = 300  # v1.15: 300px from bottom to button bottom
        self.preview_height = self.h  # v1.16: Full height for preview
        
        # ------------------------------------------------------------------
        # State machine – which screen are we showing?
        # ------------------------------------------------------------------
        self.screen = "creature_select"          # "creature_select", "capture", or "processing"
        self.pending_switch = None  # v1.34: (target_screen, switch_time) for non-blocking delay
        
        # ------------------------------------------------------------------
        # Buttons – created here, shown/hidden by screen
        # ------------------------------------------------------------------
        # Creature selection buttons (custom sizes/positions per v1.30)
        # Sloth: size (508,340), top at 157px (centered x)
        self.sloth_btn = Button(
            pos=((self.w - 508) // 2, 157),
            size=(508, 340),
            on_press=lambda: self._choose_creature("sloth"),
            up_image_path=self.sloth_path,
            initial_scale=1.0,                       # image already correct size
            press_scale=0.95
        )
        # Leopard: size (500,296), top at 496px
        self.leopard_btn = Button(
            pos=((self.w - 500) // 2, 496),
            size=(500, 296),
            on_press=lambda: self._choose_creature("leopard"),
            up_image_path=self.leopard_path,
            initial_scale=1.0,
            press_scale=0.95
        )
        # Monkey: size (452,317), top at 795px
        self.monkey_btn = Button(
            pos=((self.w - 452) // 2, 795),
            size=(452, 317),
            on_press=lambda: self._choose_creature("monkey"),
            up_image_path=self.monkey_path,
            initial_scale=1.0,
            press_scale=0.95
        )

        # Existing buttons (only visible on capture screen)
        quit_pos = (self.w - 150 - self.button_margin, self.button_margin)  # Top-right: 150x40
        self.quit_button = Button(pos=quit_pos, size=(150, 40), on_press=self._on_quit_press,
                                  up_image_path=os.path.join(graphics_dir, 'quit.png'),
                                  initial_scale=1.0, press_scale=0.95)  # Image-based; assume quit.png exists
        # v1.31: Updated capture button size/pos per specs: (510,215) at top=896px (centered x)
        capture_pos = ((self.w - 510) // 2, 896)
        self.capture_button = Button(pos=capture_pos, size=(510, 215), on_press=self._on_capture_press,
                                     up_image_path=self.button_path, initial_scale=1.0, press_scale=0.95)

        # Back button (top-left, on top of frame.png; v1.33: size (302,213))
        self.return_button = Button(pos=(self.button_margin, self.button_margin),
                                    size=(302, 213),
                                    on_press=self._return_to_creature_select,
                                    up_image_path=self.return_path,
                                    initial_scale=1.0, press_scale=0.95)

        # ------------------------------------------------------------------
        # Helper to get the active button list for the current screen
        # ------------------------------------------------------------------
        self.creature_buttons = [self.sloth_btn, self.leopard_btn, self.monkey_btn]
        self.capture_buttons  = [self.quit_button, self.capture_button, self.return_button]
        self.processing_buttons = []  # v1.44: No buttons on processing
        
        # v1.33: Quit press counter (require 3 presses)
        self.quit_presses = 0
        
        # v1.36: Reminder timers and flags
        self.last_creature_reminder = 0
        self.last_capture_reminder = 0
        self.capture_pressed = False  # v1.36: Stop reminders
        self.capture_initiated = False  # v1.36: Enable alignment/auto after press
        self.capture_complete = False  # v1.39: Hide button/post-sound
        self.capture_start_time = None  # v1.40: For generation delay

        # v1.45: Progress bar (from progress_bar.py config)
        self.bar_frames = []  # Pre-load PNGs
        self.bar_frame_count = 100
        self.bar_frame_duration = 0.5  # Total ~50s, but loop or adjust
        self.bar_w = 520
        self.bar_h = 140
        self.bar_x = (self.w - self.bar_w) // 2
        self.bar_y = self.h // 2 - self.bar_h // 2
        self.progress_start_time = None
        self._load_bar_frames()  # Pre-load

        # v1.46: Number config (from number_display.py)
        self.number_scale = 0.5
        self.number_h = int(DIGIT_HEIGHT * self.number_scale)  # ~131
        self.number_gap = int(GAP_PX * self.number_scale)  # ~10
        
        self.user_id = None  # v1.47: Set post-capture for processor launch
        self.processor_process = None  # v1.47: Subprocess handle
        self.progress_queue = queue.Queue()  # v1.47: For lines from thread
        self.current_progress = 0  # v1.47: Current percent from PROGRESS lines

    def _load_bar_frames(self):
        """Pre-load progress bar PNGs from progress_bar.py config."""
        frame_dir = r"d:\python\graphics\ui_bar"
        frame_pattern = "ui_bar{:03d}.png"
        for i in range(1, self.bar_frame_count + 1):
            path = os.path.join(frame_dir, frame_pattern.format(i))
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Warning: Failed to load bar frame {path}")
                return
            self.bar_frames.append(img)

    # ----------------------------------------------------------------------
    # Screen switching logic
    # ----------------------------------------------------------------------
    def _choose_creature(self, name: str):
        global creature_selection
        creature_selection = name
        print(f"Creature selected: {creature_selection}")
        self.pending_switch = ("capture", time.time() + 1.0)  # v1.34: Non-blocking delay

    def _return_to_creature_select(self):
        self.pending_switch = ("creature_select", time.time() + 0.5)  # v1.34: Non-blocking

    # ----------------------------------------------------------------------
    # Mouse callback – dispatch to the buttons that are currently visible
    # ----------------------------------------------------------------------
    def _mouse_callback(self, event, mx, my, flags, param):
        buttons = self.creature_buttons if self.screen == "creature_select" else self.capture_buttons
        for btn in buttons:
            btn.handle_event(event, mx, my)

        # v1.35: Removed extra redraw/imshow here to avoid flicker; loop will handle next frame

    # ----------------------------------------------------------------------
    # The rest of the class (drawing, preview loop, etc.) only needed tiny tweaks
    # ----------------------------------------------------------------------
    def init_window(self, window_name='Capture App', x=0, y=0):
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # v1.32: Ensure true fullscreen
        cv2.moveWindow(window_name, x, y)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        # start with a black canvas + frame overlay
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self._overlay_frame(canvas)
        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)

    def start_preview_loop(self, preview_cap, on_capture):
        self.on_capture = on_capture
        self.preview_cap = preview_cap

        first_frame = True
        first_landmark = True  # For potential debug
        while not self.quit_flag:
            # v1.34: Check for pending screen switch
            if self.pending_switch:
                target, switch_time = self.pending_switch
                if time.time() >= switch_time:
                    self.screen = target
                    self.pending_switch = None

            # ------------------------------------------------------------------
            # Build canvas
            # ------------------------------------------------------------------
            full_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)

            # v1.37: Always process preview if cap (for detection on creature_select)
            results = None
            status = None
            is_face_detected = False
            display_img = None
            if preview_cap is not None:
                with self.capture_lock:
                    ret, frame = preview_cap.read()
                if ret and frame is not None:
                    # v1.19: Apply rotate/mirror if set
                    if self.preview_rotate:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    if self.preview_mirror:
                        frame = cv2.flip(frame, 1)
                    # Landmarks overlay with results return (for detection/ready check; no draw on creature_select)
                    frame, results = mediapipe_landmarks.overlay_landmarks(frame, return_results=True)
                    if not first_landmark:
                        first_landmark = True
                    # v1.12: Face detection sound trigger (moved to enhancer in v1.14)
                    is_face_detected = bool(results.multi_face_landmarks) if results else False
                    self.face_detected = is_face_detected  # Update state
                    # Scale to preview_height (keep aspect: resize height, crop/letterbox width)
                    scale = self.preview_height / frame.shape[0]
                    new_width = int(frame.shape[1] * scale)
                    resized = cv2.resize(frame, (new_width, self.preview_height), interpolation=cv2.INTER_AREA)
                    display_img = np.zeros((self.preview_height, self.w, 3), dtype=np.uint8)
                    if new_width > self.w:  # Crop center
                        start_x = (new_width - self.w) // 2
                        display_img[:, :] = resized[:, start_x:start_x + self.w]
                    else:  # Letterbox
                        start_x = (self.w - new_width) // 2
                        display_img[:, start_x:start_x + new_width] = resized
                    # v1.13: Readiness via PreviewEnhancer (compute always for auto-capture)
                    status = self.preview_enhancer.check_readiness(results, new_width, self.preview_height, self.w, start_x)
                    # v1.21: Store preview with border for rebuild (but only draw border on capture)
                    self.current_preview = display_img.copy()

            # Only draw preview/landmarks/border/sounds on capture screen
            if self.screen == "capture":
                if display_img is not None:
                    self.preview_enhancer.draw_readiness_border(display_img, status)
                    # v1.14: Position/size sound feedback via PreviewEnhancer (modified for post-press)
                    self.preview_enhancer.trigger_sounds(status, is_face_detected, capture_initiated=self.capture_initiated)
                    # Overlay on full canvas
                    full_img[:self.preview_height, :] = display_img

                # v1.36: Auto-capture if initiated and ready
                if self.capture_initiated and status and status['ready']:
                    self.capture_thread = threading.Thread(target=self.on_capture)
                    self.capture_thread.start()
                    self.capture_start_time = time.time()  # v1.40: Record for delay
                    self.capture_initiated = False  # Reset to prevent repeat

                # v1.39: Check for capture complete (thread done)
                if self.capture_thread and not self.capture_thread.is_alive() and not self.capture_complete:
                    current_time = time.time()
                    if current_time >= (self.capture_start_time or 0) + 4.0:
                        self._play_sound(os.path.join(os.path.dirname(__file__), 'soundclips', 'creature_generation_in_process.mp3'))
                        self.capture_complete = True
                        self.capture_initiated = False  # Stop cues
                        self.pending_switch = ("processing", time.time())  # v1.44: Immediate switch to processing

            # v1.45: Draw animated progress bar + number on processing screen
            if self.screen == "processing":
                if not self.progress_start_time:
                    self.progress_start_time = time.time()
                elapsed = time.time() - self.progress_start_time
                frame_idx = min(int(elapsed / 0.5), 99)  # 0-99 over ~50s, but cap at 99
                if self.bar_frames:
                    current_frame = self.bar_frames[frame_idx]
                    overlay_transparent(full_img, current_frame, self.bar_x, self.bar_y)
                # Draw number above bar with gap (centered X)
                percent = frame_idx + 1  # 1-100
                text = str(percent)
                # Pre-calc total width (digits * new_w + gaps)
                new_w = int(DIGIT_WIDTH * self.number_scale)
                total_w = len(text) * new_w + (len(text) - 1) * self.number_gap
                numbers_pos = (self.bar_x + (self.bar_w - total_w) // 2, self.bar_y - 20 - self.number_h)  # Gap 20 above bar top
                draw_number(full_img, percent, pos=numbers_pos, scale=self.number_scale)

            # ------------------------------------------------------------------
            # Draw buttons for the active screen
            # ------------------------------------------------------------------
            active_buttons = self.creature_buttons if self.screen == "creature_select" else self.capture_buttons if self.screen == "capture" else self.processing_buttons
            for btn in active_buttons:
                if btn != self.return_button and btn != self.capture_button:  # Draw non-return/capture before frame
                    btn.draw(full_img)
                elif btn == self.capture_button:  # v1.44: Draw always (no hide; switch handles)
                    btn.draw(full_img)

            # message (only used during init / creature screen if you want one)
            self._handle_fade()
            if self.current_msg:
                self._draw_text(full_img, self.current_msg,
                               pos=(self.button_margin, self.button_margin + 30))

            # frame overlay (after most buttons, before return)
            self._overlay_frame(full_img)

            # Draw return button last (on top of frame)
            if self.screen == "capture":
                self.return_button.draw(full_img)

            self.current_frame = full_img.copy()
            cv2.imshow(self.window_name, full_img)
            if first_frame:
                first_frame = False

            # v1.36: Handle reminders
            current_time = time.time()
            if self.screen == "creature_select" and self.face_detected:
                if current_time - self.last_creature_reminder > 10:
                    self._play_sound(os.path.join(os.path.dirname(__file__), 'soundclips', 'please_select_a_creature.mp3'))
                    self.last_creature_reminder = current_time
            elif self.screen == "capture" and not self.capture_pressed:
                if current_time - self.last_capture_reminder > 10:
                    self._play_sound(os.path.join(os.path.dirname(__file__), 'soundclips', 'press_button_to_capture.mp3'))
                    self.last_capture_reminder = current_time

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.quit_flag = True

        self.cleanup()

    def _play_sound(self, sound_file):
        """v1.36: Non-blocking sound play (simple Popen for reminders)."""
        if os.path.exists(sound_file):
            subprocess.Popen(["python", "soundplayer.py", sound_file])

    # ----------------------------------------------------------------------
    # unchanged helper methods (_draw_text, _overlay_frame, etc.)
    # ----------------------------------------------------------------------
    def _draw_text(self, img, text, pos=(10, 30), color=(255, 255, 255),
                   font_scale=0.7, thickness=2):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, pos, font, font_scale, color, thickness)

    def _overlay_frame(self, img):
        if self.frame_img is None:
            self.frame_img = cv2.imread(self.frame_path, cv2.IMREAD_UNCHANGED)
            if self.frame_img is None:
                print(f"Warning: Failed to load frame.png at {self.frame_path}")
                return
            if self.frame_img.shape[:2] != (self.h, self.w):
                self.frame_img = cv2.resize(self.frame_img, (self.w, self.h),
                                           interpolation=cv2.INTER_AREA)
        if self.frame_img.shape[2] == 4:
            b, g, r, a = cv2.split(self.frame_img)
            overlay = cv2.merge([b, g, r])
            alpha = a / 255.0
            img[:] = (1 - alpha[..., None]) * img + alpha[..., None] * overlay

    def _handle_fade(self):
        if self.fade_start and self.fade_time:
            elapsed = time.time() - self.fade_start
            if elapsed >= self.fade_time:
                self.current_msg = None
                self.fade_start = None

    def _on_quit_press(self):
        self.quit_presses += 1  # v1.33: Increment counter
        if self.quit_presses >= 3:
            self.quit_flag = True
            self.quit_presses = 0  # Reset for safety

    def _on_capture_press(self):
        self.capture_pressed = True  # v1.36: Stop reminders
        self.capture_initiated = True  # v1.36: Enable alignment/auto
        # v1.39: No immediate capture; wait for auto in loop

    def update_message(self, msg, fade_time=None):
        self.current_msg = msg
        self.fade_start = time.time() if fade_time else None
        self.fade_time = fade_time

    def cleanup(self):
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        if self.processor_process:
            self.processor_process.terminate()  # Cleanup if still running