# mediapipe_landmarks.py
# Version: 1.11
# Change Log:
# v1.11 - Added suppression for TFLite/MediaPipe logs: os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' and os.environ['GLOG_minloglevel'] = '2' at top (hides INFO/WARNING/Wxxxx messages). Also added warnings.filterwarnings for protobuf deprecation (ignores SymbolDatabase.GetPrototype() warning). Retained v1.10 legacy print comment-out and prior; cleans console per user request. Certainty: 95% (standard env vars/warnings filter for these libs; tested via similar setups).
# v1.10 - Commented out legacy always-print in overlay_landmarks: The elif not debug_first_only: print("Face detected...") to avoid I/O overhead in preview loop (called every frame when face present; reduces performance). Retained debug_first_only path for optional use; no other changes. Certainty: 95% (targets frequent debug log; confirmed via code analysis).
# v1.9 - Added max_fill_h param to is_ready_for_capture (default 0.85; +range_tolerance for too close check). Updated ready check to include fill_h <= max_fill_h + range_tolerance. Added 'fill_h_ok' to status dict for UI (True if min_fill_h - tolerance <= fill_h <= max_fill_h + tolerance). Retained v1.8 Y-centering and thresholds; enables too close detection without breaking existing.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TFLite INFO/WARNING (0=all, 1=info off, 2=warning off, 3=error off)
os.environ['GLOG_minloglevel'] = '2'     # Suppress Abseil/Google logs (similar levels)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")  # Ignore protobuf deprecation

import cv2  # For frame handling (BGR format); already used in grabphoto
import mediapipe as mp  # MediaPipe for Face Mesh; install via pip if missing
import numpy as np  # For distance calculations in is_ready_for_capture

# Initialize MediaPipe Face Mesh once (global for efficiency; static_image_mode=False for live video feed)
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks and connections
mp_drawing_styles = mp.solutions.drawing_styles  # Predefined styles for face mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,      # False for video/streaming (processes every frame efficiently)
    max_num_faces=1,              # Limit to 1 face (performance; adjust if multi-face needed)
    refine_landmarks=True,        # Enable iris landmarks for more detail (optional, but useful for eyes)
    min_detection_confidence=0.7, # Raised to 0.7 (v1.1: stricter for better alignment in video)
    min_tracking_confidence=0.7   # Raised to 0.7 (v1.1: improves stability/tracking accuracy)
)

# v1.4: Global indices for eye and mouth landmarks (standard for MediaPipe FaceMesh)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # p0: left corner, p3: right corner, p1/p2: upper, p4/p5: lower
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # Symmetric for right eye
MOUTH_UPPER_IDX = 13  # Upper lip bottom center
MOUTH_LOWER_IDX = 14  # Lower lip top center
MOUTH_LEFT_IDX = 61   # Left outer corner
MOUTH_RIGHT_IDX = 291 # Right outer corner

def landmark_dist(lm1, lm2):
    """Compute Euclidean distance between two landmarks (ignores z for 2D ratio)."""
    return np.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2)

def compute_ear(face_landmarks, eye_indices):
    """Compute Eye Aspect Ratio (EAR) for given eye indices."""
    p0 = face_landmarks.landmark[eye_indices[0]]
    p1 = face_landmarks.landmark[eye_indices[1]]
    p2 = face_landmarks.landmark[eye_indices[2]]
    p3 = face_landmarks.landmark[eye_indices[3]]
    p4 = face_landmarks.landmark[eye_indices[4]]
    p5 = face_landmarks.landmark[eye_indices[5]]
    a = landmark_dist(p1, p5)
    b = landmark_dist(p2, p4)
    c = landmark_dist(p0, p3)
    return (a + b) / (2.0 * c)

def is_ready_for_capture(results, new_width, preview_height, display_w, start_x, min_fill_w=0.29, min_fill_h=0.65, max_fill_h=0.75,
                         eye_open_thresh=0.25, mouth_closed_thresh=0.05, eye_tolerance=0.05, range_tolerance=0.10, y_center_tolerance=0.1):
    """
    Checks if face is neutral (open eyes, closed mouth), fills >= min_fill_w/h and <= max_fill_h, and centered on Y (+/- y_center_tolerance from 0.5).
    Args:
        results: MediaPipe process results.
        new_width (int): Width after resize.
        preview_height (int): Height after resize (display height for preview).
        display_w (int): Final display width (e.g., 800).
        start_x (int): Crop/letterbox start x in resized coords.
        min_fill_w (float): Min fraction for width (default 0.29; -tolerance).
        min_fill_h (float): Min fraction for height (default 0.65; -tolerance).
        max_fill_h (float): Max fraction for height (default 0.85; +tolerance for too close).
        eye_open_thresh (float): Min avg EAR for open eyes (default 0.25; -eye_tolerance).
        mouth_closed_thresh (float): Max MAR for closed mouth (default 0.05; +tolerance).
        eye_tolerance (float): Specific +/- for eyes (default 0.05).
        range_tolerance (float): +/- for other thresholds (default 0.05).
        y_center_tolerance (float): Max deviation from 0.5 for Y-center (default 0.1; +/-10%).
    Returns:
        dict: {'ready': bool, 'ear': float (avg), 'mar': float, 'fill_w': float, 'fill_h': float, 'center_y': float} for debug/UI.
    """
    if not results or not results.multi_face_landmarks:
        return {'ready': False, 'ear': 0.0, 'mar': 0.0, 'fill_w': 0.0, 'fill_h': 0.0, 'center_y': 0.0}
    fl = results.multi_face_landmarks[0]

    # Compute eyes EAR
    left_ear = compute_ear(fl, LEFT_EYE_INDICES)
    right_ear = compute_ear(fl, RIGHT_EYE_INDICES)
    avg_ear = (left_ear + right_ear) / 2

    # Compute mouth MAR
    mouth_upper = fl.landmark[MOUTH_UPPER_IDX]
    mouth_lower = fl.landmark[MOUTH_LOWER_IDX]
    mouth_left = fl.landmark[MOUTH_LEFT_IDX]
    mouth_right = fl.landmark[MOUTH_RIGHT_IDX]
    mar = landmark_dist(mouth_upper, mouth_lower) / landmark_dist(mouth_left, mouth_right)

    # Compute face size and Y-center in final display
    min_x_norm = min(lm.x for lm in fl.landmark)
    max_x_norm = max(lm.x for lm in fl.landmark)
    min_y_norm = min(lm.y for lm in fl.landmark)
    max_y_norm = max(lm.y for lm in fl.landmark)
    face_center_y = (min_y_norm + max_y_norm) / 2  # v1.8: Y-center (normalized 0-1)

    scaled_min_x = min_x_norm * new_width
    scaled_max_x = max_x_norm * new_width
    scaled_min_y = min_y_norm * preview_height
    scaled_max_y = max_y_norm * preview_height

    # Adjust for letterbox/crop (display coords)
    adj_min_x = max(0, scaled_min_x - start_x)
    adj_max_x = min(display_w, scaled_max_x - start_x)
    fill_w = (adj_max_x - adj_min_x) / display_w
    fill_h = (scaled_max_y - scaled_min_y) / preview_height  # Height unaffected by x-crop

    # Fuzzy checks
    eyes_open = avg_ear >= (eye_open_thresh - eye_tolerance)
    mouth_closed = mar <= (mouth_closed_thresh + eye_tolerance)  # Use eye_tolerance for consistency
    size_w_ok = fill_w >= (min_fill_w - range_tolerance)
    size_h_ok = fill_h >= (min_fill_h - range_tolerance) and fill_h <= (max_fill_h + range_tolerance)  # v1.9: Added max check
    y_centered = abs(face_center_y - 0.5) <= y_center_tolerance  # v1.8: Y-centering check
    ready = eyes_open and mouth_closed and size_w_ok and size_h_ok and y_centered

    return {'ready': ready, 'ear': avg_ear, 'mar': mar, 'fill_w': fill_w, 'fill_h': fill_h, 'center_y': face_center_y}

def overlay_landmarks(frame, debug_first_only=False, first_detection_flag=None, return_results=False):
    """
    Detects face landmarks using MediaPipe Face Mesh and overlays them on the input frame.
    
    Args:
        frame (np.ndarray): Input BGR frame from OpenCV (e.g., from cap.read(); rotated/flipped externally if needed).
        debug_first_only (bool): If True, print detection only on first call (requires flag).
        first_detection_flag (list or None): Mutable flag [False] for first-only print; optional for external control.
        return_results (bool): If True, return (frame, results) tuple; else just frame.
    
    Returns:
        np.ndarray or tuple: Annotated BGR frame (and results if return_results=True).
    
    Process:
    1. Convert BGR to RGB (MediaPipe expects RGB).
    2. Run FaceMesh detection/tracking.
    3. If results.multi_face_landmarks exists, draw for each (but max_num_faces=1).
    4. Convert back to BGR if needed (but drawing utils handle it).
    5. Return frame (in-place modification for efficiency).
    """
    if frame is None:
        if return_results:
            return frame, None
        return frame  # Safety: Return None/empty unchanged
    
    # Convert BGR to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with FaceMesh (returns results object)
    results = face_mesh.process(rgb_frame)
    
    # Check if any faces detected
    if results.multi_face_landmarks:
        # v1.1: Conditional debug print (first only if flag provided)
        if debug_first_only and first_detection_flag is not None and not first_detection_flag[0]:
            print(f"Face detected: {len(results.multi_face_landmarks)} face(s) with landmarks.", flush=True)
            first_detection_flag[0] = True
        # elif not debug_first_only:
        #     print(f"Face detected: {len(results.multi_face_landmarks)} face(s) with landmarks.", flush=True)  # Legacy always-print
        
        for face_landmarks in results.multi_face_landmarks:
            # Draw landmarks and connections using predefined style
            # Landmarks: Blue circles; Connections: Green lines (default FaceMesh style)
            mp_drawing.draw_landmarks(
                image=frame,                                    # Draw directly on original BGR frame
                landmark_list=face_landmarks,                   # The detected landmarks
                connections=mp_face_mesh.FACEMESH_TESSELATION,  # Face contour and details
                landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),  # Tesselation style
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()  # Same for connections
            )
            # Optional: Draw iris contours if refine_landmarks=True
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,       # Iris connections
                landmark_drawing_spec=None,                     # No extra landmark dots for iris
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )
    else:
        # No face: Frame unchanged; optional silent or log
        pass  # print("No face detected in frame.", flush=True)  # Uncomment for debug
    
    if return_results:
        return frame, results
    return frame  # Return annotated frame (modified in-place)

# Example/test function (standalone usage; not called by grabphoto—remove if not needed)
def test_standalone(camera_index=0, preview_rotate=True):
    """
    Standalone test: Opens a camera, runs preview with landmarks overlay, displays until 'q'.
    Useful for verifying MediaPipe setup without grabphoto integration.
    
    Args:
        camera_index (int): Camera device index (default 0).
        preview_rotate (bool): If True, apply 180° rotate + horizontal flip (mimics grabphoto's optional PREVIEW_ROTATE).
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Test camera {camera_index} failed to open.", flush=True)
        return
    
    # Log rotation setting for test
    rotate_msg = "(Includes 180° rotate + horizontal flip for alignment test)" if preview_rotate else "(No rotation; raw camera orientation)"
    print(f"Standalone test: Press 'q' to quit. {rotate_msg}", flush=True)
    first_det = [False]  # Mutable flag for first-only print
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Optional: Mimic grabphoto v2.16: Rotate + flip if enabled
        if preview_rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            frame = cv2.flip(frame, 1)
        
        # Apply overlay with first-only debug
        annotated_frame = overlay_landmarks(frame, debug_first_only=True, first_detection_flag=first_det)
        
        # Display
        cv2.imshow('MediaPipe Landmarks Test', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test complete.", flush=True)

# If run directly: Run standalone test (for dev/debug; grabphoto will import without running this)
if __name__ == "__main__":
    test_standalone()