# ui_controller.py
# v1.50
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
        
        self.sloth_btn = Button(
            pos=((self.w - 508) // 2, 157),
            size=(508, 340),
            on_press=lambda: self._choose_creature("sloth"),
            up_image_path=self.sloth_path,
            initial_scale=1.0,
            press_scale=0.95
        )
        self.leopard_btn = Button(
            pos=((self.w - 500) // 2, 496),
            size=(500, 296),
            on_press=lambda: self._choose_creature("leopard"),
            up_image_path=self.leopard_path,
            initial_scale=1.0,
            press_scale=0.95
        )
        self.monkey_btn = Button(
            pos=((self.w - 452) // 2, 795),
            size=(452, 317),
            on_press=lambda: self._choose_creature("monkey"),
            up_image_path=self.monkey_path,
            initial_scale=1.0,
            press_scale=0.95
        )

        quit_pos = (self.w - 150 - self.button_margin, self.button_margin)
        self.quit_button = Button(pos=quit_pos, size=(150, 40), on_press=self._on_quit_press,
                                  up_image_path=os.path.join(PATHS['GRAPHICS'], 'quit.png'),
                                  initial_scale=1.0, press_scale=0.95)
        capture_pos = ((self.w - 510) // 2, 896)
        self.capture_button = Button(pos=capture_pos, size=(510, 215), on_press=self._on_capture_press,
                                     up_image_path=self.button_path, initial_scale=1.0, press_scale=0.95)

        self.return_button = Button(pos=(self.button_margin, self.button_margin),
                                    size=(302, 213),
                                    on_press=self._return_to_creature_select,
                                    up_image_path=self.return_path,
                                    initial_scale=1.0, press_scale=0.95)

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
        self.bar_frame_duration = 0.5
        self.bar_w = 520
        self.bar_h = 140
        self.bar_x = (self.w - self.bar_w) // 2
        self.bar_y = self.h // 2 - self.bar_h // 2
        self.progress_start_time = None
        self._load_bar_frames()

        self.number_scale = 0.5
        self.number_h = int(DIGIT_HEIGHT * self.number_scale)
        self.number_gap = int(GAP_PX * self.number_scale)
        
        self.user_id = None
        self.processor_process = None
        self.progress_queue = queue.Queue()
        self.current_progress = 0

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
        self.pending_switch = ("creature_select", time.time() + 0.5)

    def _mouse_callback(self, event, mx, my, flags, param):
        buttons = self.creature_buttons if self.screen == "creature_select" else self.capture_buttons
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
        first_landmark = True
        while not self.quit_flag:
            if self.pending_switch:
                target, switch_time = self.pending_switch
                if time.time() >= switch_time:
                    self.screen = target
                    self.pending_switch = None

            full_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)

            results = None
            status = None
            is_face_detected = False
            display_img = None
            if preview_cap is not None:
                with self.capture_lock:
                    ret, frame = preview_cap.read()
                if ret and frame is not None:
                    if self.preview_rotate:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    if self.preview_mirror:
                        frame = cv2.flip(frame, 1)
                    frame, results = mediapipe_landmarks.overlay_landmarks(frame, return_results=True)
                    if not first_landmark:
                        first_landmark = True
                    is_face_detected = bool(results.multi_face_landmarks) if results else False
                    self.face_detected = is_face_detected
                    scale = self.preview_height / frame.shape[0]
                    new_width = int(frame.shape[1] * scale)
                    resized = cv2.resize(frame, (new_width, self.preview_height), interpolation=cv2.INTER_AREA)
                    display_img = np.zeros((self.preview_height, self.w, 3), dtype=np.uint8)
                    if new_width > self.w:
                        start_x = (new_width - self.w) // 2
                        display_img[:, :] = resized[:, start_x:start_x + self.w]
                    else:
                        start_x = (self.w - new_width) // 2
                        display_img[:, start_x:start_x + new_width] = resized
                    status = self.preview_enhancer.check_readiness(results, new_width, self.preview_height, self.w, start_x)
                    self.current_preview = display_img.copy()

            if self.screen == "capture":
                if display_img is not None:
                    self.preview_enhancer.draw_readiness_border(display_img, status)
                    self.preview_enhancer.trigger_sounds(status, is_face_detected, capture_initiated=self.capture_initiated)
                    full_img[:self.preview_height, :] = display_img

                if self.capture_initiated and status and status['ready']:
                    self.capture_thread = threading.Thread(target=self.on_capture)
                    self.capture_thread.start()
                    self.capture_start_time = time.time()
                    self.capture_initiated = False

                if self.capture_thread and not self.capture_thread.is_alive() and not self.capture_complete:
                    current_time = time.time()
                    if current_time >= (self.capture_start_time or 0) + 4.0:
                        self._play_sound(os.path.join(PATHS['SOUNDCLIPS'], 'creature_generation_in_process.mp3'))
                        self.capture_complete = True
                        self.capture_initiated = False
                        self.pending_switch = ("processing", time.time())

            if self.screen == "processing":
                if not self.progress_start_time:
                    self.progress_start_time = time.time()
                elapsed = time.time() - self.progress_start_time
                frame_idx = min(int(elapsed / 0.5), 99)
                if self.bar_frames:
                    current_frame = self.bar_frames[frame_idx]
                    overlay_transparent(full_img, current_frame, self.bar_x, self.bar_y)
                percent = frame_idx + 1
                text = str(percent)
                new_w = int(DIGIT_WIDTH * self.number_scale)
                total_w = len(text) * new_w + (len(text) - 1) * self.number_gap
                numbers_pos = (self.bar_x + (self.bar_w - total_w) // 2, self.bar_y - 20 - self.number_h)
                draw_number(full_img, percent, pos=numbers_pos, scale=self.number_scale)

            active_buttons = self.creature_buttons if self.screen == "creature_select" else self.capture_buttons if self.screen == "capture" else self.processing_buttons
            for btn in active_buttons:
                if btn != self.return_button and btn != self.capture_button:
                    btn.draw(full_img)
                elif btn == self.capture_button:
                    btn.draw(full_img)

            self._handle_fade()
            if self.current_msg:
                self._draw_text(full_img, self.current_msg,
                               pos=(self.button_margin, self.button_margin + 30))

            self._overlay_frame(full_img)

            if self.screen == "capture":
                self.return_button.draw(full_img)

            self.current_frame = full_img.copy()
            cv2.imshow(self.window_name, full_img)
            if first_frame:
                first_frame = False

            current_time = time.time()
            if self.screen == "creature_select" and self.face_detected:
                if current_time - self.last_creature_reminder > 10:
                    self._play_sound(os.path.join(PATHS['SOUNDCLIPS'], 'please_select_a_creature.mp3'))
                    self.last_creature_reminder = current_time
            elif self.screen == "capture" and not self.capture_pressed:
                if current_time - self.last_capture_reminder > 10:
                    self._play_sound(os.path.join(PATHS['SOUNDCLIPS'], 'press_button_to_capture.mp3'))
                    self.last_capture_reminder = current_time

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.quit_flag = True

        self.cleanup()

    def _play_sound(self, sound_file):
        if os.path.exists(sound_file):
            subprocess.Popen(["python", "soundplayer.py", sound_file])

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
        self.quit_presses += 1
        if self.quit_presses >= 3:
            self.quit_flag = True
            self.quit_presses = 0

    def _on_capture_press(self):
        self.capture_pressed = True
        self.capture_initiated = True

    def update_message(self, msg, fade_time=None):
        self.current_msg = msg
        self.fade_start = time.time() if fade_time else None
        self.fade_time = fade_time

    def cleanup(self):
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        if self.processor_process:
            self.processor_process.terminate()