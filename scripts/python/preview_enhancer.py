import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import PATHS

import mediapipe_landmarks
import cv2
import os
import subprocess
import time

class PreviewEnhancer:
    def __init__(self):
        self.last_ready = False
        self.last_too_far = False
        self.last_too_close = False
        self.last_too_high = False
        self.last_too_low = False
        self.last_face_detected = False
        self.lost_frames = 0
        self.lost_threshold = 3
        self.face_detect_start = None
        self.face_detect_duration = 2.0
        self.current_sound_process = None

    def check_readiness(self, results, new_width, preview_height, display_w, start_x):
        return mediapipe_landmarks.is_ready_for_capture(results, new_width, preview_height, display_w, start_x)

    def draw_readiness_border(self, display_img, status, color=(0, 255, 0), thickness=20):
        if status['ready']:
            h, w = display_img.shape[:2]
            cv2.rectangle(display_img, (0, 0), (w - 1, h - 1), color, thickness)

    def _play_sound(self, sound_file):
        if self.current_sound_process and self.current_sound_process.poll() is None:
            self.current_sound_process.terminate()
            self.current_sound_process.wait()

        if os.path.exists(sound_file):
            self.current_sound_process = subprocess.Popen(["python", "soundplayer.py", sound_file])

    def trigger_sounds(self, status, face_detected, capture_initiated=False):
        soundclips_dir = PATHS['SOUNDCLIPS']

        if face_detected:
            self.lost_frames = 0
            if not self.last_face_detected:
                self.face_detect_start = time.time()
            self.last_face_detected = True
        else:
            self.lost_frames += 1
            if self.lost_frames >= self.lost_threshold:
                self.last_face_detected = False
                self.face_detect_start = None

        if not face_detected:
            self.last_ready = False
            self.last_too_far = False
            self.last_too_close = False
            self.last_too_high = False
            self.last_too_low = False
            return

        if self.face_detect_start and (time.time() - self.face_detect_start < self.face_detect_duration):
            return

        if not capture_initiated:
            return

        if status['ready']:
            if not self.last_ready:
                sound_file = os.path.join(soundclips_dir, 'face_in_range.mp3')
                self._play_sound(sound_file)
                self.last_ready = True
            self.last_too_far = False
            self.last_too_close = False
            self.last_too_high = False
            self.last_too_low = False
        else:
            self.last_ready = False
            too_far = status['fill_h'] < 0.65 - 0.15
            too_close = status['fill_h'] > 0.85 + 0.15
            if too_far and not self.last_too_far:
                sound_file = os.path.join(soundclips_dir, 'face_too_far.mp3')
                self._play_sound(sound_file)
                self.last_too_far = True
            elif not too_far:
                self.last_too_far = False
            if too_close and not self.last_too_close:
                sound_file = os.path.join(soundclips_dir, 'face_too_close.mp3')
                self._play_sound(sound_file)
                self.last_too_close = True
            elif not too_close:
                self.last_too_close = False
            if not too_far and not too_close:
                too_high = status['center_y'] < 0.5 - 0.1
                too_low = status['center_y'] > 0.5 + 0.1
                if too_high and not self.last_too_high:
                    sound_file = os.path.join(soundclips_dir, 'face_is_too_high.mp3')
                    self._play_sound(sound_file)
                    self.last_too_high = True
                elif not too_high:
                    self.last_too_high = False
                if too_low and not self.last_too_low:
                    sound_file = os.path.join(soundclips_dir, 'face_is_too_low.mp3')
                    self._play_sound(sound_file)
                    self.last_too_low = True
                elif not too_low:
                    self.last_too_low = False
            else:
                self.last_too_high = False
                self.last_too_low = False