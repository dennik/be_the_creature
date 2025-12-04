# ui_controller.py
# Version: 1.53
# Changes:
# - v1.53 (2025-12-04): Added completion sound + 2s pause + automatic return to creature selection menu
#                       when RealityScan reaches 100%. Sound plays via pygame (same as rest of UI).
# - v1.52: Immediate processing screen + real progress bar
# - v1.51: Queue-driven progress bar integration

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import PATHS

import queue
import cv2
import numpy as np
import time
import subprocess
import mediapipe_landmarks
from preview_enhancer import PreviewEnhancer
from ui_button import Button
import threading
import pygame
from progress_bar import overlay_transparent
from number_display import draw_number, DIGIT_WIDTH, DIGIT_HEIGHT, GAP_PX
from realityscan_processor import RealityScanProcessor

pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

creature_selection = None

class UIController:

    def __init__(self, w=800, h=1280, capture_lock=None, preview_rotate=False, preview_mirror=True):

        self.w = w
        self.h = h
        self.preview_rotate = preview_rotate
        self.preview_mirror = preview_mirror
        self.window_name = None
        self.current_msg = None
        self.fade_start = None
        self.fade_time = None
        self.quit_flag = False
        self.on_capture = None
        self.preview_cap = None
        self.capture_thread = None
        self.capture_lock = capture_lock
        self.face_detected = False
        self.preview_enhancer = PreviewEnhancer()
        self.current_frame = None
        self.current_preview = None
        self.frame_path = os.path.join(PATHS['GRAPHICS'], 'frame.png')
        self.sloth_path = os.path.join(PATHS['GRAPHICS'], 'sloth.png')
        self.leopard_path = os.path.join(PATHS['GRAPHICS'], 'leopard.png')
        self.monkey_path = os.path.join(PATHS['GRAPHICS'], 'monkey.png')
        self.return_path = os.path.join(PATHS['GRAPHICS'], 'return.png')
        self.button_path = os.path.join(PATHS['GRAPHICS'], 'button.png')
        self.frame_img = None
        self.button_margin = 10
        self.button_bottom_margin = 300
        self.preview_height = self.h
        
        self.screen = "creature_select"
        self.pending_switch = None
        
        # Creature buttons
        self.sloth_btn = Button(pos=((self.w - 508) // 2, 157), size=(508, 340),
                                on_press=lambda: self._choose_creature("sloth"), up_image_path=self.sloth_path,
                                initial_scale=1.0, press_scale=0.95)
        self.leopard_btn = Button(pos=((self.w - 500) // 2, 496), size=(500, 296),
                                  on_press=lambda: self._choose_creature("leopard"), up_image_path=self.leopard_path,
                                  initial_scale=1.0, press_scale=0.95)
        self.monkey_btn = Button(pos=((self.w - 452) // 2, 795), size=(452, 317),
                                 on_press=lambda: self._choose_creature("monkey"), up_image_path=self.monkey_path,
                                 initial_scale=1.0, press_scale=0.95)

        quit_pos = (self.w - 150 - self.button_margin, self.button_margin)
        self.quit_button = Button(pos=quit_pos, size=(150, 40), on_press=self._on_quit_press,
                                  up_image_path=os.path.join(PATHS['GRAPHICS'], 'quit.png'),
                                  initial_scale=1.0, press_scale=0.95)
        capture_pos = ((self.w - 510) // 2, 896)
        self.capture_button = Button(pos=capture_pos, size=(510, 215), on_press=self._on_capture_press,
                                     up_image_path=self.button_path, initial_scale=1.0, press_scale=0.95)

        self.return_button = Button(pos=(self.button_margin, self.button_margin),
                                    size=(302, 213), on_press=self._return_to_creature_select,
                                    up_image_path=self.return_path, initial_scale=1.0, press_scale=0.95)

        self.creature_buttons = [self.sloth_btn, self.leopard_btn, self.monkey_btn]
        self.capture_buttons  = [self.quit_button, self.capture_button, self.return_button]
        self.processing_buttons = []
        
        self.quit_presses = 0
        self.last_creature_reminder = 0
        self.last_capture_reminder = 0
        self.capture_pressed = False
        self.capture_initiated = False
        self.capture_complete = False
        self.capture_start_time = None

        self.bar_frames = []
        self.bar_frame_count = 100
        self.bar_w = 520
        self.bar_h = 140
        self.bar_x = (self.w - self.bar_w) // 2
        self.bar_y = self.h // 2 - self.bar_h // 2
        self._load_bar_frames()

        self.number_scale = 0.5
        self.number_h = int(DIGIT_HEIGHT * self.number_scale)
        self.number_gap = int(GAP_PX * self.number_scale)
        
        self.user_id = None
        self.processor = None
        self.progress_queue = queue.Queue()
        self.current_progress = 0
        self.completed_sound_played = False  # Track once-only playback

    def update_message(self, msg: str):
        self.current_msg = msg
        print(f"[UI] {msg}")

    def _load_bar_frames(self):
        frame_dir = PATHS['UI_BAR']
        frame_pattern = "ui_bar{:03d}.png"
        for i in range(1, self.bar_frame_count + 1):
            path = os.path.join(frame_dir, frame_pattern.format(i))
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Warning: Failed to load bar frame {path}")
                return
            self.bar_frames.append(img)

    def _choose_creature(self, name: str):
        global creature_selection
        creature_selection = name
        print(f"Creature selected: {creature_selection}")
        self.pending_switch = ("capture", time.time() + 1.0)

    def _return_to_creature_select(self):
        print("[UI] Returning to creature selection menu")
        self.pending_switch = ("creature_select", time.time() + 0.5)
        self.capture_complete = False
        self.capture_initiated = False
        self.processor = None
        self.current_progress = 0
        self.completed_sound_played = False
        while not self.progress_queue.empty():
            try:
                self.progress_queue.get_nowait()
            except queue.Empty:
                break
        self.current_msg = None

    def start_processing_phase(self, user_id):
        if self.processor:
            return
        user_dir = os.path.join(PATHS['BASE'], f"user_{user_id}")
        self.processor = RealityScanProcessor(user_dir, progress_queue=self.progress_queue)
        self.processor.start_photogrammetry()
        self.pending_switch = ("processing", time.time())
        self.update_message("Generating your creature...")
        self.current_progress = 0
        self.completed_sound_played = False

    def _on_quit_press(self):
        self.quit_presses += 1
        if self.quit_presses >= 2:
            self.quit_flag = True

    def _on_capture_press(self):
        if not self.capture_initiated and self.face_detected and not self.processor:
            self.capture_initiated = True
            self.capture_pressed = True

    def _mouse_callback(self, event, mx, my, flags, param):
        buttons = self.creature_buttons if self.screen == "creature_select" else \
                  self.capture_buttons if self.screen == "capture" else []
        for btn in buttons:
            btn.handle_event(event, mx, my)

    def init_window(self, window_name='Capture App', x=0, y=0):
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(window_name, x, y)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self._overlay_frame(canvas)
        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)

    def start_preview_loop(self, preview_cap, on_capture):
        self.on_capture = on_capture
        self.preview_cap = preview_cap
        first_frame = True

        while not self.quit_flag:
            if self.pending_switch:
                target, switch_time = self.pending_switch
                if time.time() >= switch_time:
                    self.screen = target
                    self.pending_switch = None

            full_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)

            # Preview handling (unchanged)
            if preview_cap is not None:
                with self.capture_lock:
                    ret, frame = preview_cap.read()
                if ret and frame is not None:
                    if self.preview_rotate:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    if self.preview_mirror:
                        frame = cv2.flip(frame, 1)
                    frame, results = mediapipe_landmarks.overlay_landmarks(frame, return_results=True)
                    self.face_detected = bool(results.multi_face_landmarks) if results else False
                    scale = self.preview_height / frame.shape[0]
                    new_width = int(frame.shape[1] * scale)
                    resized = cv2.resize(frame, (new_width, self.preview_height), interpolation=cv2.INTER_AREA)
                    display_img = np.zeros((self.preview_height, self.w, 3), dtype=np.uint8)
                    start_x = max(0, (new_width - self.w) // 2)
                    display_img[:, :] = resized[:, start_x:start_x + self.w]
                    status = self.preview_enhancer.check_readiness(results, new_width, self.preview_height, self.w, start_x)
                    self.preview_enhancer.draw_readiness_border(display_img, status)
                    self.current_preview = display_img.copy()

            if self.screen == "capture":
                if self.current_preview is not None:
                    self.preview_enhancer.trigger_sounds(status, self.face_detected, capture_initiated=self.capture_initiated)
                    full_img[:self.preview_height, :] = self.current_preview

                if self.capture_initiated and status and status['ready']:
                    self.capture_thread = threading.Thread(target=self.on_capture)
                    self.capture_thread.start()
                    self.capture_start_time = time.time()
                    self.capture_initiated = False

                if self.capture_thread and not self.capture_thread.is_alive() and not self.capture_complete:
                    if time.time() >= (self.capture_start_time or 0) + 4.0:
                        self._play_sound(os.path.join(PATHS['SOUNDCLIPS'], 'creature_generation_in_process.mp3'))
                        self.capture_complete = True
                        result = self.on_capture()
                        if result:
                            self.user_id = result
                        if self.user_id:
                            self.start_processing_phase(self.user_id)

            if self.screen == "processing":
                # Real progress from RealityScan
                try:
                    while True:
                        latest = self.progress_queue.get_nowait()
                        self.current_progress = latest
                except queue.Empty:
                    pass

                # NEW: Completion sound + auto-return when 100% reached
                if self.current_progress >= 100 and not self.completed_sound_played:
                    self.completed_sound_played = True
                    sound_path = os.path.join(PATHS['SOUNDCLIPS'], "your_creature_has_been_generated.mp3")
                    if os.path.exists(sound_path):
                        pygame.mixer.music.load(sound_path)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                    time.sleep(2.0)  # extra 2 seconds
                    self._return_to_creature_select()  # Auto back to main menu

                frame_idx = min(self.current_progress, 99)
                if self.bar_frames:
                    current_frame = self.bar_frames[frame_idx]
                    overlay_transparent(full_img, current_frame, self.bar_x, self.bar_y)
                percent = self.current_progress
                draw_number(full_img, percent, pos=(self.bar_x + 200, self.bar_y - 20 - self.number_h), scale=self.number_scale)

            # Button drawing
            active_buttons = self.creature_buttons if self.screen == "creature_select" else \
                            self.capture_buttons if self.screen == "capture" else self.processing_buttons
            for btn in active_buttons:
                btn.draw(full_img)

            if self.current_msg:
                self._draw_text(full_img, self.current_msg, pos=(self.button_margin, self.button_margin + 30))

            self._overlay_frame(full_img)
            if self.screen == "capture":
                self.return_button.draw(full_img)

            self.current_frame = full_img.copy()
            cv2.imshow(self.window_name, full_img)
            if first_frame:
                first_frame = False

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.quit_flag = True

        self.cleanup()

    def _play_sound(self, sound_file):
        if os.path.exists(sound_file):
            subprocess.Popen(["python", "soundplayer.py", sound_file])

    def _draw_text(self, img, text, pos=(10, 30), color=(255, 255, 255), font_scale=0.7, thickness=2):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    def _overlay_frame(self, img):
        if self.frame_img is None:
            self.frame_img = cv2.imread(self.frame_path, cv2.IMREAD_UNCHANGED)
            if self.frame_img is None:
                return
            if self.frame_img.shape[:2] != (self.h, self.w):
                self.frame_img = cv2.resize(self.frame_img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        if self.frame_img.shape[2] == 4:
            b, g, r, a = cv2.split(self.frame_img)
            overlay = cv2.merge([b, g, r])
            alpha = a / 255.0
            img[:] = (1 - alpha[..., None]) * img + alpha[..., None] * overlay

    def cleanup(self):
        cv2.destroyAllWindows()
        if self.preview_cap:
            self.preview_cap.release()
        pygame.mixer.quit()