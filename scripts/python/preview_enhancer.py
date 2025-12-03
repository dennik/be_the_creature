# preview_enhancer.py
# Version: 1.6
# Change Log:
# v1.6 - Removed face_detected sound play entirely per user request (commented out transition play and lockout logic; retained debounce for potential future). No other changes; certainty: 100% (direct removal; test for no play on detect).
# v1.5 - Added optional capture_initiated param to trigger_sounds (default False): If False, skip alignment sounds (too far/close/etc.). Retained face_detected trigger always. Certainty: 100% (simple flag; no logic changes).
# v1.4 - Added face detection full-play lockout: In __init__, added self.face_detect_start = None and self.face_detect_duration = 2.0 (seconds; adjust to clip length). In trigger_sounds, after playing face_detected, set self.face_detect_start = time.time(). Before other checks, if within duration (time.time() - start < duration), skip alignment triggers. This lets face_detected complete before alignment sounds, without blocking preview. Retained v1.3 debounce and termination; no other changes.

import mediapipe_landmarks  # For is_ready_for_capture (core readiness logic)
import cv2  # For drawing border on display_img
import os  # v1.1: For sound file path
import subprocess  # v1.1: For calling soundplayer.py
import time  # For lockout timing and potential other uses

class PreviewEnhancer:
    """
    Helper class for enhancing preview frames: Readiness checks, visual indicators (e.g., border), audio feedback.
    Instantiate once in UI (or main), call methods in loop with frame/results/display params/face_detected.
    Modularity: Keeps UI loop clean; extend for more overlays/feedback (e.g., text, guides, more sounds).
    v1.6: Removed face_detected sound.
    v1.5: Post-press alignment flag.
    v1.4: Lockout for face_detected full play.
    """
    def __init__(self):
        # State flags for sound transitions (reset on ready/non-ready)
        self.last_ready = False
        self.last_too_far = False
        self.last_too_close = False
        self.last_too_high = False
        self.last_too_low = False
        # v1.3: Face detection states with debounce to prevent flicker-retriggers
        self.last_face_detected = False
        self.lost_frames = 0
        self.lost_threshold = 3  # Require 3 consecutive lost frames to reset flag
        # v1.4: Lockout for face_detected full play (seconds; adjust to actual clip duration)
        self.face_detect_start = None
        self.face_detect_duration = 2.0  # Placeholder; set to 'face_detected.mp3' length
        # v1.2: Track current sound process to terminate on new play (prevent overlap)
        self.current_sound_process = None

    def check_readiness(self, results, new_width, preview_height, display_w, start_x):
        """
        Wrapper for mediapipe_landmarks.is_ready_for_capture: Computes readiness status.
        Args same as is_ready_for_capture (passed through).
        Returns status dict {'ready': bool, ...} for UI decisions.
        Comment: Centralizes if multiple checks/enhancers added later.
        """
        return mediapipe_landmarks.is_ready_for_capture(results, new_width, preview_height, display_w, start_x)

    def draw_readiness_border(self, display_img, status, color=(0, 255, 0), thickness=20):
        """
        Draws green border on display_img if status['ready'].
        Args:
            display_img (np.ndarray): Preview image to draw on (modified in-place).
            status (dict): From check_readiness (uses 'ready' key).
            color (tuple): BGR color for border (default green).
            thickness (int): Border width (default 20px for visibility).
        Returns: None (modifies display_img).
        Comment: Simple visual feedback; could extend for colors based on partial readiness (e.g., yellow if close).
        """
        if status['ready']:
            h, w = display_img.shape[:2]
            cv2.rectangle(display_img, (0, 0), (w - 1, h - 1), color, thickness)

    def _play_sound(self, sound_file):
        """
        Internal: Plays sound via Popen after terminating any previous process.
        Returns: None (side effect: updates self.current_sound_process).
        Comment: Ensures no overlap by interrupting old sound.
        """
        # v1.2: Terminate previous if still running
        if self.current_sound_process and self.current_sound_process.poll() is None:
            self.current_sound_process.terminate()  # Stop old sound
            self.current_sound_process.wait()  # Wait for clean exit (brief block, but <1s)

        # Start new sound non-blocking
        if os.path.exists(sound_file):
            self.current_sound_process = subprocess.Popen(["python", "soundplayer.py", sound_file])

    def trigger_sounds(self, status, face_detected, capture_initiated=False):
        """
        Triggers audio feedback based on status dict and face_detected.
        Prioritized: If face detected, check distance first (far/close), then Y-axis if distance okay.
        Plays clips only on transitions (using internal state flags); non-blocking via Popen.
        Resets on ready/non-ready. No-op if no face.
        v1.6: Removed face_detected sound.
        v1.5: Added capture_initiated flag: Skip alignment if False.
        v1.4: Skip alignment during face_detected lockout period.
        v1.3: Added debounced face detection trigger before readiness (centralizes/standardizes all sounds).
        v1.2: Uses _play_sound to manage process and prevent overlap.
        Returns: None (plays sounds as side effect).
        Comment: Handles user guidance sequentially to avoid confusion; extend for more priorities (e.g., rotation).
        """
        soundclips_dir = os.path.join(os.path.dirname(__file__), 'soundclips')

        # v1.3: Debounce face detection (no sound; retained for state)
        if face_detected:
            self.lost_frames = 0  # Reset counter on detection
            if not self.last_face_detected:  # Transition: Face appears
                # v1.6: Removed sound play
                self.face_detect_start = time.time()  # Retained lockout for consistency
            self.last_face_detected = True
        else:
            self.lost_frames += 1  # Increment on loss
            if self.lost_frames >= self.lost_threshold:
                self.last_face_detected = False  # Only reset after threshold
                self.face_detect_start = None  # v1.4: Clear lockout on true loss

        if not face_detected:  # Skip all if no face (after face sound check)
            # Reset states to avoid stale triggers
            self.last_ready = False
            self.last_too_far = False
            self.last_too_close = False
            self.last_too_high = False
            self.last_too_low = False
            return

        # v1.4: Check lockout before alignment triggers
        if self.face_detect_start and (time.time() - self.face_detect_start < self.face_detect_duration):
            return  # Skip alignment sounds during lockout

        # v1.5: Skip alignment if not initiated
        if not capture_initiated:
            return

        if status['ready']:
            if not self.last_ready:
                sound_file = os.path.join(soundclips_dir, 'face_in_range.mp3')
                self._play_sound(sound_file)  # v1.2: Managed
                self.last_ready = True
            # Reset others
            self.last_too_far = False
            self.last_too_close = False
            self.last_too_high = False
            self.last_too_low = False
        else:
            self.last_ready = False
            # Priority 1: Distance (far/close)
            too_far = status['fill_h'] < 0.65 - 0.15
            too_close = status['fill_h'] > 0.85 + 0.15
            if too_far and not self.last_too_far:
                sound_file = os.path.join(soundclips_dir, 'face_too_far.mp3')
                self._play_sound(sound_file)  # v1.2: Managed
                self.last_too_far = True
            elif not too_far:
                self.last_too_far = False
            if too_close and not self.last_too_close:
                sound_file = os.path.join(soundclips_dir, 'face_too_close.mp3')
                self._play_sound(sound_file)  # v1.2: Managed
                self.last_too_close = True
            elif not too_close:
                self.last_too_close = False
            # Priority 2: Y-axis only if distance okay (not too far/close)
            if not too_far and not too_close:
                too_high = status['center_y'] < 0.5 - 0.1
                too_low = status['center_y'] > 0.5 + 0.1
                if too_high and not self.last_too_high:
                    sound_file = os.path.join(soundclips_dir, 'face_is_too_high.mp3')
                    self._play_sound(sound_file)  # v1.2: Managed
                    self.last_too_high = True
                elif not too_high:
                    self.last_too_high = False
                if too_low and not self.last_too_low:
                    sound_file = os.path.join(soundclips_dir, 'face_is_too_low.mp3')
                    self._play_sound(sound_file)  # v1.2: Managed
                    self.last_too_low = True
                elif not too_low:
                    self.last_too_low = False
            else:
                # Reset Y-axis if distance issue
                self.last_too_high = False
                self.last_too_low = False