import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import PATHS

import cv2
import time
import pygame

class Button:
    def __init__(self, pos, size, on_press=None, up_image_path=None, down_image_path=None, press_scale=0.9, initial_scale=2.0):
        self.rect = (pos[0], pos[1], size[0], size[1])
        self.on_press = on_press
        self.press_scale = press_scale
        self.initial_scale = initial_scale
        self.up_image = None
        self.down_image = None
        self.up_image_path = up_image_path
        self.down_image_path = down_image_path
        self.use_images = up_image_path is not None
        self.anim_phase = -1
        self.anim_start = 0
        self.current_scale = 1.0
        soundclips_dir = PATHS['SOUNDCLIPS']
        self.sound_path = os.path.join(soundclips_dir, 'button.mp3')

    def draw(self, img):
        self.update_animation()
        x, y, w, h = self.rect
        if self.use_images:
            if self.up_image is None and self.up_image_path:
                self.up_image = cv2.imread(self.up_image_path, cv2.IMREAD_UNCHANGED)
            button_img = self.up_image
            if self.down_image_path:
                if self.down_image is None:
                    self.down_image = cv2.imread(self.down_image_path, cv2.IMREAD_UNCHANGED)
                if self.anim_phase != -1:
                    button_img = self.down_image
            if button_img is not None:
                orig_h, orig_w = button_img.shape[:2]
                scale_w = w / orig_w
                scale_h = h / orig_h
                fit_scale = min(scale_w, scale_h)
                scale_factor = self.current_scale
                effective_scale = fit_scale * self.initial_scale * scale_factor
                draw_w = int(orig_w * effective_scale)
                draw_h = int(orig_h * effective_scale)
                draw_x = x + (w - draw_w) // 2
                draw_y = y + (h - draw_h) // 2
                
                img_h, img_w = img.shape[:2]
                draw_x = max(0, draw_x)
                draw_y = max(0, draw_y)
                draw_w = min(draw_w, img_w - draw_x)
                draw_h = min(draw_h, img_h - draw_y)
                
                if draw_w <= 0 or draw_h <= 0:
                    return
                
                resized_img = cv2.resize(button_img, (draw_w, draw_h), interpolation=cv2.INTER_AREA)
                
                if resized_img.shape[2] == 4:
                    b, g, r, a = cv2.split(resized_img)
                    overlay = cv2.merge([b, g, r])
                    alpha = a / 255.0
                    roi = img[draw_y:draw_y + draw_h, draw_x:draw_x + draw_w]
                    roi[:] = (1 - alpha[..., None]) * roi + alpha[..., None] * overlay
                else:
                    img[draw_y:draw_y + draw_h, draw_x:draw_x + draw_w] = resized_img

    def start_animation(self):
        self.anim_phase = 0
        self.anim_start = time.time()

    def update_animation(self):
        if self.anim_phase == -1:
            self.current_scale = 1.0
            return
        elapsed = time.time() - self.anim_start
        dur = 0.1667
        if self.anim_phase == 0:
            if elapsed < dur:
                t = elapsed / dur
                self.current_scale = 1.0 + t * (self.press_scale - 1.0)
            else:
                self.current_scale = self.press_scale
                self.anim_phase = 1
                self.anim_start = time.time()
        elif self.anim_phase == 1:
            if elapsed >= dur:
                self.anim_phase = 2
                self.anim_start = time.time()
            self.current_scale = self.press_scale
        elif self.anim_phase == 2:
            if elapsed < dur:
                t = elapsed / dur
                self.current_scale = self.press_scale + t * (1.0 - self.press_scale)
            else:
                self.current_scale = 1.0
                self.anim_phase = -1

    def play_sound(self):
        if os.path.exists(self.sound_path):
            pygame.mixer.Sound(self.sound_path).play()
        else:
            print(f"Warning: Sound not found at {self.sound_path}")

    def handle_event(self, event, mx, my):
        x, y, w, h = self.rect
        inside = x <= mx <= x + w and y <= my <= y + h
        if event == cv2.EVENT_LBUTTONDOWN and inside:
            if self.on_press:
                self.on_press()
            self.start_animation()
            self.play_sound()

    def is_inside(self, mx, my):
        x, y, w, h = self.rect
        return x <= mx <= x + w and y <= my <= y + h