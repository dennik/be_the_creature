import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import PATHS

import cv2
import mediapipe as mp
import numpy as np
from scipy import stats
import csv

EVALUATE_CAMERAS = False
SELECTED_CAMERAS = []

VISUALIZE = False

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def normalized_to_pixel(normalized_x, normalized_y, image_width, image_height):
    x_px = min(max(int(normalized_x * image_width), 0), image_width - 1)
    y_px = min(max(int(normalized_y * image_height), 0), image_height - 1)
    return (x_px, y_px)

class IrisColorExtractor:
    def __init__(self):
        self.v_min = 47
        self.s_min = 0
        self.v_max = 190
        self.black_v = 50
        self.gray_s = 30
        self.brown_max = 20
        self.hazel_max = 40
        self.green_max = 80
        self.blue_max = 140
        self.color_to_code = {'black': 0, 'brown': 1, 'hazel': 2, 'green': 3, 'blue': 4, 'gray': 5}
        self.code_to_color = {v: k for k, v in self.color_to_code.items()}
        self.color_bgr = {
            'black': (0, 0, 0), 'brown': (42, 42, 165), 'hazel': (42, 100, 165),
            'green': (0, 255, 0), 'blue': (255, 0, 0), 'gray': (128, 128, 128)
        }

    def extract_from_image(self, frame, viz_path=None):
        rotated = False
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            frame_rot = cv2.rotate(frame, cv2.ROTATE_180)
            rgb_rot = cv2.cvtColor(frame_rot, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_rot)
            if results.multi_face_landmarks:
                frame = frame_rot
                rotated = True
            else:
                return [], rotated

        colors = []
        viz_frame = frame.copy() if viz_path else None

        for face_landmarks in results.multi_face_landmarks:
            height, width, _ = frame.shape
            for iris_indices in [[474, 475, 476, 477], [469, 470, 471, 472]]:
                points = []
                for idx in iris_indices:
                    lm = face_landmarks.landmark[idx]
                    px = normalized_to_pixel(lm.x, lm.y, width, height)
                    points.append(px)
                if len(points) < 4:
                    continue

                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillConvexPoly(mask, np.array(points, dtype=np.int32), 255)

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
                iris_pixels = masked_hsv[mask > 0]

                if len(iris_pixels) < 10:
                    continue

                valid = (iris_pixels[:,2] >= self.v_min) & (iris_pixels[:,2] <= self.v_max) & (iris_pixels[:,1] >= self.s_min)
                valid_pixels = iris_pixels[valid]

                if len(valid_pixels) < 10:
                    continue

                hue_mode = stats.mode(valid_pixels[:,0], keepdims=False).mode
                sat_mean = np.mean(valid_pixels[:,1])
                val_mean = np.mean(valid_pixels[:,2])

                if val_mean < self.black_v:
                    color = 'black'
                elif sat_mean < self.gray_s:
                    color = 'gray'
                elif hue_mode < self.brown_max:
                    color = 'brown'
                elif hue_mode < self.hazel_max:
                    color = 'hazel'
                elif hue_mode < self.green_max:
                    color = 'green'
                elif hue_mode < self.blue_max:
                    color = 'blue'
                else:
                    color = 'gray'

                colors.append(color)

                if viz_path:
                    color_mask = np.zeros_like(frame)
                    for y, x in np.argwhere(mask > 0):
                        px_hsv = hsv[y, x]
                        h, s, v = px_hsv
                        if not (self.v_min <= v <= self.v_max and s >= self.s_min):
                            continue
                        if v < self.black_v:
                            px_color = 'black'
                        elif s < self.gray_s:
                            px_color = 'gray'
                        elif h < self.brown_max:
                            px_color = 'brown'
                        elif h < self.hazel_max:
                            px_color = 'hazel'
                        elif h < self.green_max:
                            px_color = 'green'
                        elif h < self.blue_max:
                            px_color = 'blue'
                        else:
                            px_color = 'gray'
                        color_mask[y, x] = self.color_bgr[px_color]
                    alpha = 0.5
                    viz_frame = cv2.addWeighted(viz_frame, 1 - alpha, color_mask, alpha, 0)

        if viz_path and len(colors) > 0:
            cv2.imwrite(viz_path, viz_frame)

        return colors, rotated

def compute_and_write_scores(user_id, save_dir, per_camera_colors, final_color):
    user_dir = os.path.dirname(save_dir)
    csv_path = os.path.join(user_dir, f"user_{user_id}_camera_performance.csv")
    scores = []
    for i in range(17):
        colors = per_camera_colors[i]
        if not colors:
            scores.append(0.0)
            continue
        match_count = sum(1 for c in colors if c == final_color)
        score = match_count / len(colors) if colors else 0.0
        scores.append(score)
        print(f"Camera {i}: Score {score:.2f} ({match_count}/{len(colors)} match {final_color})")

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if os.path.getsize(csv_path) == 0:
            writer.writerow(['user_id', 'final_color'] + [f'camera_{i}' for i in range(17)])
        writer.writerow([user_id, final_color] + scores)
    print(f"Performance scores appended to {csv_path}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python eye_color_detector.py <user_id> <save_dir>")
        sys.exit(1)

    user_id = int(sys.argv[1])
    save_dir = sys.argv[2]

    user_dir = os.path.dirname(save_dir)

    selected = SELECTED_CAMERAS if SELECTED_CAMERAS else list(range(17))

    photos = [f for f in os.listdir(save_dir) if f.startswith(f"user_{user_id}_camera_") and f.endswith(".jpg")
              and any(f"camera_{i}_" in f for i in selected)]
    if len(photos) < len(selected):
        print(f"Warning: Only {len(photos)} photos found for user {user_id} (expected {len(selected)}).")

    photos.sort(key=lambda f: int(f.split('_camera_')[1].split('_')[0]))

    extractor = IrisColorExtractor()
    all_colors = []
    per_camera_colors = {i: [] for i in range(17)}

    for photo in photos:
        image_path = os.path.join(save_dir, photo)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Warning: Failed to load {photo}")
            continue

        viz_path = os.path.join(save_dir, f"user_{user_id}_{photo.replace('.jpg', '_iris_viz.jpg')}") if VISUALIZE else None

        colors, rotated = extractor.extract_from_image(frame, viz_path=viz_path)
        all_colors.extend(colors)
        if colors:
            print(f"Colors from {photo}: {colors}")
        if rotated:
            print(f"Photo {photo} rotated 180° for face detection (likely upside-down camera).")

        try:
            cam_idx_str = photo.split('_camera_')[1].split('_')[0]
            cam_idx = int(cam_idx_str)
            if 0 <= cam_idx < 17:
                per_camera_colors[cam_idx].extend(colors)
        except Exception as e:
            print(f"Error parsing camera index for {photo}: {e}")

    if not all_colors:
        color = "unknown"
        print("No eye colors detected across selected photos.")
    else:
        codes = [extractor.color_to_code[c] for c in all_colors]
        mode_code = stats.mode(codes, keepdims=False).mode
        color = extractor.code_to_color[mode_code]
        print(f"Final eye color (majority vote): {color} from {all_colors}")

    output_file = os.path.join(user_dir, f"user_{user_id}_eye_color.txt")
    with open(output_file, 'w') as f:
        f.write(color)
    print(f"Eye color for user {user_id}: {color} (saved to {output_file}); detected from {len(all_colors)} irises across {len(photos)} photos.")

    if EVALUATE_CAMERAS:
        compute_and_write_scores(user_id, save_dir, per_camera_colors, color)

if __name__ == "__main__":
    main()