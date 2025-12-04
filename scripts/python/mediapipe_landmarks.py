# mediapipe_landmarks.py
# Version: 1.12
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import PATHS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['GLOG_minloglevel'] = '2'

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_UPPER_IDX = 13
MOUTH_LOWER_IDX = 14
MOUTH_LEFT_IDX = 61
MOUTH_RIGHT_IDX = 291

def landmark_dist(lm1, lm2):
    return np.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2)

def compute_ear(face_landmarks, eye_indices):
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
    if not results or not results.multi_face_landmarks:
        return {'ready': False, 'ear': 0.0, 'mar': 0.0, 'fill_w': 0.0, 'fill_h': 0.0, 'center_y': 0.0}
    fl = results.multi_face_landmarks[0]

    left_ear = compute_ear(fl, LEFT_EYE_INDICES)
    right_ear = compute_ear(fl, RIGHT_EYE_INDICES)
    avg_ear = (left_ear + right_ear) / 2

    mouth_upper = fl.landmark[MOUTH_UPPER_IDX]
    mouth_lower = fl.landmark[MOUTH_LOWER_IDX]
    mouth_left = fl.landmark[MOUTH_LEFT_IDX]
    mouth_right = fl.landmark[MOUTH_RIGHT_IDX]
    mar = landmark_dist(mouth_upper, mouth_lower) / landmark_dist(mouth_left, mouth_right)

    min_x_norm = min(lm.x for lm in fl.landmark)
    max_x_norm = max(lm.x for lm in fl.landmark)
    min_y_norm = min(lm.y for lm in fl.landmark)
    max_y_norm = max(lm.y for lm in fl.landmark)
    face_center_y = (min_y_norm + max_y_norm) / 2

    scaled_min_x = min_x_norm * new_width
    scaled_max_x = max_x_norm * new_width
    scaled_min_y = min_y_norm * preview_height
    scaled_max_y = max_y_norm * preview_height

    adj_min_x = max(0, scaled_min_x - start_x)
    adj_max_x = min(display_w, scaled_max_x - start_x)
    fill_w = (adj_max_x - adj_min_x) / display_w
    fill_h = (scaled_max_y - scaled_min_y) / preview_height

    eyes_open = avg_ear >= (eye_open_thresh - eye_tolerance)
    mouth_closed = mar <= (mouth_closed_thresh + eye_tolerance)
    size_w_ok = fill_w >= (min_fill_w - range_tolerance)
    size_h_ok = fill_h >= (min_fill_h - range_tolerance) and fill_h <= (max_fill_h + range_tolerance)
    y_centered = abs(face_center_y - 0.5) <= y_center_tolerance
    ready = eyes_open and mouth_closed and size_w_ok and size_h_ok and y_centered

    return {'ready': ready, 'ear': avg_ear, 'mar': mar, 'fill_w': fill_w, 'fill_h': fill_h, 'center_y': face_center_y}

def overlay_landmarks(frame, debug_first_only=False, first_detection_flag=None, return_results=False):
    if frame is None:
        if return_results:
            return frame, None
        return frame
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        if debug_first_only and first_detection_flag is not None and not first_detection_flag[0]:
            print(f"Face detected: {len(results.multi_face_landmarks)} face(s) with landmarks.", flush=True)
            first_detection_flag[0] = True
        
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )
    else:
        pass
    
    if return_results:
        return frame, results
    return frame

def test_standalone(camera_index=0, preview_rotate=True):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Test camera {camera_index} failed to open.", flush=True)
        return
    
    rotate_msg = "(Includes 180° rotate + horizontal flip for alignment test)" if preview_rotate else "(No rotation; raw camera orientation)"
    print(f"Standalone test: Press 'q' to quit. {rotate_msg}", flush=True)
    first_det = [False]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if preview_rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            frame = cv2.flip(frame, 1)
        
        annotated_frame = overlay_landmarks(frame, debug_first_only=True, first_detection_flag=first_det)
        
        cv2.imshow('MediaPipe Landmarks Test', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test complete.", flush=True)

if __name__ == "__main__":
    test_standalone()