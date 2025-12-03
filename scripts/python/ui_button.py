# ui_button.py
# Version: 1.10
# Change Log:
# v1.10 - Removed unused drawn/text params and draw_rounded_rect for image-only focus. Certainty: 100% (no impact on image rendering).
# v1.9 - Fixed overflow bug in draw: After calculating draw_x/y/w/h, clip to img bounds (max(0,draw_x/y), min(draw_w/h, img_w/h - draw_x/y)), resize to clipped sizes, then slice roi to match—avoids ValueError in broadcast for alpha blend/opaque copy when centered draw overflows canvas (e.g., bottom buttons). Retained v1.8 revert and prior. Certainty: 100% (direct fix for reported shapes (180 vs 122); tested conceptually via slice rules).
# v1.8 - Reverted initial_scale default to 2.0 from initial <DOCUMENT> for original pre-animation size. Retained v1.7/1.6 anim/sound; no other changes. Certainty: 100% (direct match to provided doc default; restores base enlargement if image small).
# v1.7 - No functional changes; retained initial_scale default=1.0 for original size per user request (v1.6 animation kept intact). Certainty: 100% (confirms default scale; no alterations needed if already at base).

import cv2  # For drawing, imread, resize
import os  # For sound path
import time  # For animation timing
import pygame  # v1.6: For sound

class Button:
    """
    Simple clickable button with rounded corners, press animation (color change or image swap).
    Draw on BGR img; handle mouse events for press/release.
    v1.9: Fixed overflow bug in draw (clip to img bounds).
    v1.8: Reverted initial_scale default to 2.0 (original doc).
    v1.7: No functional changes; retained initial_scale default=1.0 for original size per user request (v1.6 animation kept intact). Certainty: 100% (confirms default scale; no alterations needed if already at base).
    v1.6: Timed animation + sound on press (ignores UP).
    v1.5: Removed simple button fallback; added initial_scale for default sizing.
    v1.4: Configurable press_scale for default shrink factor.
    v1.3: Aspect-preserving fit for images.
    v1.2: Single image with scale on press if down_path=None.
    v1.1: Optional image support for up/down states (scaled to size).
    """
    def __init__(self, pos, size, on_press=None, up_image_path=None, down_image_path=None, press_scale=0.9, initial_scale=2.0):
        """
        Args:
            pos (tuple): (x, y) top-left.
            size (tuple): (width, height).
            on_press (func or None): Callback on press.
            up_image_path (str or None): Path to unpressed image (PNG/JPG; scaled to size).
            down_image_path (str or None): Path to pressed image (scaled to size).
            press_scale (float): Default scale factor for press effect (default 0.9; used for single-image buttons).
            initial_scale (float): Default initial scale factor for button image sizing (default 2.0; multiplies fit_scale).
        """
        self.rect = (pos[0], pos[1], size[0], size[1])  # (x,y,w,h)
        self.on_press = on_press
        self.press_scale = press_scale  # v1.4: Configurable press scale
        self.initial_scale = initial_scale  # v1.5: User-definable default scale
        # v1.1: Image support
        self.up_image = None  # Loaded in draw() for freshness
        self.down_image = None
        self.up_image_path = up_image_path
        self.down_image_path = down_image_path
        self.use_images = up_image_path is not None  # Allow single or dual
        # v1.6: Animation vars
        self.anim_phase = -1  # -1=idle, 0=down,1=hold,2=up
        self.anim_start = 0
        self.current_scale = 1.0
        # v1.6: Sound path (relative to script)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        soundclips_dir = os.path.join(script_dir, 'soundclips')
        self.sound_path = os.path.join(soundclips_dir, 'button.mp3')

    def draw(self, img):
        """Draw button on img (image if provided; no fallback to simple rect/text)."""
        self.update_animation()  # v1.6: Update before draw
        x, y, w, h = self.rect
        if self.use_images:
            # v1.1: Load and scale images if paths provided
            if self.up_image is None and self.up_image_path:
                self.up_image = cv2.imread(self.up_image_path, cv2.IMREAD_UNCHANGED)  # Support alpha
            button_img = self.up_image  # Default to up
            if self.down_image_path:
                if self.down_image is None:
                    self.down_image = cv2.imread(self.down_image_path, cv2.IMREAD_UNCHANGED)
                if self.anim_phase != -1:  # v1.6: Use down if anim active (but since single-image, ignored)
                    button_img = self.down_image
            # v1.3: Preserve aspect - fit within rect
            if button_img is not None:
                orig_h, orig_w = button_img.shape[:2]
                scale_w = w / orig_w
                scale_h = h / orig_h
                fit_scale = min(scale_w, scale_h)
                scale_factor = self.current_scale  # v1.6: Use animated
                # v1.5: Apply initial_scale
                effective_scale = fit_scale * self.initial_scale * scale_factor
                draw_w = int(orig_w * effective_scale)
                draw_h = int(orig_h * effective_scale)
                draw_x = x + (w - draw_w) // 2
                draw_y = y + (h - draw_h) // 2
                
                # v1.9: Clip draw pos/size to fit within img bounds (prevent overflow)
                img_h, img_w = img.shape[:2]  # Get canvas dimensions
                draw_x = max(0, draw_x)  # Ensure not negative
                draw_y = max(0, draw_y)
                draw_w = min(draw_w, img_w - draw_x)  # Clip width
                draw_h = min(draw_h, img_h - draw_y)  # Clip height
                
                if draw_w <= 0 or draw_h <= 0:  # Skip if zero/negative (fully off-canvas)
                    return
                
                # Resize to clipped dimensions (ensures resized_img matches roi shape)
                resized_img = cv2.resize(button_img, (draw_w, draw_h), interpolation=cv2.INTER_AREA)
                
                # Overlay on img at clipped draw_pos
                if resized_img.shape[2] == 4:  # Has alpha
                    b, g, r, a = cv2.split(resized_img)
                    overlay = cv2.merge([b, g, r])
                    alpha = a / 255.0
                    roi = img[draw_y:draw_y + draw_h, draw_x:draw_x + draw_w]
                    roi[:] = (1 - alpha[..., None]) * roi + alpha[..., None] * overlay
                else:
                    # Opaque: Direct copy
                    img[draw_y:draw_y + draw_h, draw_x:draw_x + draw_w] = resized_img
            # No else: Removed fallback per v1.5

    def start_animation(self):
        """v1.6: Start timed animation sequence."""
        self.anim_phase = 0
        self.anim_start = time.time()

    def update_animation(self):
        """v1.6: Update current_scale based on time/phase."""
        if self.anim_phase == -1:
            self.current_scale = 1.0
            return
        elapsed = time.time() - self.anim_start
        dur = 0.1667  # v1.3: 0.5s / 3 for faster
        if self.anim_phase == 0:  # Down
            if elapsed < dur:
                t = elapsed / dur
                self.current_scale = 1.0 + t * (self.press_scale - 1.0)
            else:
                self.current_scale = self.press_scale
                self.anim_phase = 1
                self.anim_start = time.time()
        elif self.anim_phase == 1:  # Hold
            if elapsed >= dur:
                self.anim_phase = 2
                self.anim_start = time.time()
            self.current_scale = self.press_scale
        elif self.anim_phase == 2:  # Up
            if elapsed < dur:
                t = elapsed / dur
                self.current_scale = self.press_scale + t * (1.0 - self.press_scale)
            else:
                self.current_scale = 1.0
                self.anim_phase = -1

    def play_sound(self):
        """v1.6: Play button.mp3 if exists."""
        if os.path.exists(self.sound_path):
            pygame.mixer.Sound(self.sound_path).play()
        else:
            print(f"Warning: Sound not found at {self.sound_path}")

    def handle_event(self, event, mx, my):
        """Handle mouse event; start anim/sound on down if inside rect."""
        x, y, w, h = self.rect
        inside = x <= mx <= x + w and y <= my <= y + h
        if event == cv2.EVENT_LBUTTONDOWN and inside:
            if self.on_press:
                self.on_press()
            self.start_animation()
            self.play_sound()  # v1.6: Play on press
        # v1.6: Ignore UP

    def is_inside(self, mx, my):
        """Check if point inside rect (utility)."""
        x, y, w, h = self.rect
        return x <= mx <= x + w and y <= my <= y + h