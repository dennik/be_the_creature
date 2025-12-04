# facial_hair_detector.py
# Version: 1.7

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import PATHS

import cv2
import mediapipe as mp
import numpy as np
import json
from scipy import stats
from collections import Counter
import csv

EVALUATE_CAMERAS = False
SELECTED_CAMERAS = []

VISUALIZE = False

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def normalized_to_pixel(normalized_x, normalized_y, image_width, image_height):
    x_px = min(max(int(normalized_x * image_width), 0), image_width - 1)
    y_px = min(max(int(normalized_y * image_height), 0), image_height - 1)
    return (x_px, y_px)

class FacialHairDetector:
    def __init__(self):
        self.density_presence = 0.1
        self.density_medium = 0.4
        self.density_long = 0.7
        self.color_to_code = {"black": 0, "brown": 1, "blonde": 2, "gray": 3, "red": 4, "unknown": 5}
        self.code_to_color = {v: k for k, v in self.color_to_code.items()}
        self.type_to_code = {"none": 0, "mustache": 1, "goatee": 2, "goatee with mustache": 3, "full beard": 4}
        self.code_to_type = {v: k for k, v in self.type_to_code.items()}
        self.length_to_code = {"short": 0, "medium": 1, "long": 2}
        self.code_to_length = {v: k for k, v in self.length_to_code.items()}
        self.region_colors = {
            "mustache": (255, 0, 0),
            "goatee": (0, 255, 0),
            "beard_left": (0, 0, 255),
            "beard_right": (0, 0, 255)
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
                return {"type": "none"}, rotated

        viz_img = frame.copy() if viz_path else None

        for face_landmarks in results.multi_face_landmarks:
            height, width, _ = frame.shape

            mustache_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
            mustache_pts = [normalized_to_pixel(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y, width, height) for i in mustache_indices]
            mustache_region = self._get_region_crop(frame, mustache_pts, padding=5)

            goatee_indices = [152, 176, 148, 377, 400, 378, 149, 150]
            goatee_pts = [normalized_to_pixel(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y, width, height) for i in goatee_indices]
            goatee_region = self._get_region_crop(frame, goatee_pts, padding=10)

            beard_left_indices = [234, 93, 132, 58, 172]
            beard_left_pts = [normalized_to_pixel(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y, width, height) for i in beard_left_indices]
            beard_left_region = self._get_region_crop(frame, beard_left_pts, padding=15)

            beard_right_indices = [454, 323, 361, 288, 397]
            beard_right_pts = [normalized_to_pixel(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y, width, height) for i in beard_right_indices]
            beard_right_region = self._get_region_crop(frame, beard_right_pts, padding=15)

            mustache_density = self._analyze_texture(mustache_region) if mustache_region is not None else 0.0
            goatee_density = self._analyze_texture(goatee_region) if goatee_region is not None else 0.0
            beard_left_density = self._analyze_texture(beard_left_region) if beard_left_region is not None else 0.0
            beard_right_density = self._analyze_texture(beard_right_region) if beard_right_region is not None else 0.0
            beard_density = (beard_left_density + beard_right_density) / 2 if beard_left_density + beard_right_density > 0 else 0.0

            has_mustache = mustache_density > self.density_presence
            has_goatee = goatee_density > self.density_presence
            has_beard = beard_density > self.density_presence

            if has_beard:
                hair_type = "full beard"
            elif has_goatee and has_mustache:
                hair_type = "goatee with mustache"
            elif has_goatee:
                hair_type = "goatee"
            elif has_mustache:
                hair_type = "mustache"
            else:
                hair_type = "none"

            if hair_type == "none":
                return {"type": "none"}, rotated

            max_density = max(mustache_density, goatee_density, beard_density)
            if max_density < self.density_medium:
                length = "short"
            elif max_density < self.density_long:
                length = "medium"
            else:
                length = "long"

            regions = [r for r in [mustache_region, goatee_region, beard_left_region, beard_right_region] if r is not None]
            colors = [self._analyze_color(r) for r in regions if r is not None]
            if colors:
                color_mode = Counter(colors).most_common(1)[0][0]
            else:
                color_mode = "unknown"

            result = {"type": hair_type, "length": length, "color": color_mode}

            if has_mustache:
                result["upper_lip_covered_by_mustache"] = mustache_density > self.density_presence
            if has_goatee:
                result["chin_covered_by_goatee"] = goatee_density > self.density_presence
            if has_beard:
                result["cheeks_covered_by_beard"] = beard_density > self.density_presence

            if viz_path:
                self._draw_viz_boxes(viz_img, mustache_pts, "mustache")
                self._draw_viz_boxes(viz_img, goatee_pts, "goatee")
                self._draw_viz_boxes(viz_img, beard_left_pts, "beard_left")
                self._draw_viz_boxes(viz_img, beard_right_pts, "beard_right")
                cv2.imwrite(viz_path, viz_img)

            return result, rotated

        return {"type": "none"}, rotated

    def _get_region_crop(self, frame, points, padding=10):
        if not points:
            return None
        min_x = max(0, min(p[0] for p in points) - padding)
        max_x = min(frame.shape[1], max(p[0] for p in points) + padding)
        min_y = max(0, min(p[1] for p in points) - padding)
        max_y = min(frame.shape[0], max(p[1] for p in points) + padding)
        if min_x >= max_x or min_y >= max_y:
            return None
        return frame[min_y:max_y, min_x:max_x]

    def _analyze_texture(self, region):
        if region is None or region.size == 0:
            return 0.0
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        variance = lap.var()
        return min(variance / 1000, 1.0)

    def _analyze_color(self, region):
        if region is None or region.size == 0:
            return "unknown"
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mask = (hsv[:,:,1] > 20) & (hsv[:,:,2] < 200)
        masked_hues = hsv[:,:,0][mask]
        if len(masked_hues) == 0:
            return "unknown"
        hue_mode = Counter(masked_hues).most_common(1)[0][0]
        if hue_mode < 20:
            return "brown"
        elif hue_mode < 40:
            return "blonde"
        elif hue_mode < 60:
            return "red"
        elif hue_mode > 120:
            return "gray"
        else:
            return "black"

    def _draw_viz_boxes(self, img, points, region_name):
        if not points:
            return
        pts = np.array(points, np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=self.region_colors.get(region_name, (0,255,255)), thickness=2)

def compute_and_write_scores(user_id, save_dir, per_camera_results, final_result):
    user_dir = os.path.dirname(save_dir)
    csv_path = os.path.join(user_dir, f"user_{user_id}_camera_performance.csv")
    scores = []
    for i in range(17):
        results = per_camera_results[i]
        if not results:
            scores.append(0.0)
            continue
        match_count = sum(1 for r in results if r == final_result)
        score = match_count / len(results) if results else 0.0
        scores.append(score)
        print(f"Camera {i}: Score {score:.2f} ({match_count}/{len(results)} match {final_result})")

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if os.path.getsize(csv_path) == 0:
            writer.writerow(['user_id', 'final_result'] + [f'camera_{i}' for i in range(17)])
        writer.writerow([user_id, final_result] + scores)
    print(f"Performance scores appended to {csv_path}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python facial_hair_detector.py <user_id> <save_dir>")
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

    detector = FacialHairDetector()
    all_results = []
    per_camera_results = {i: [] for i in range(17)}

    for photo in photos:
        image_path = os.path.join(save_dir, photo)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Warning: Failed to load {photo}")
            continue

        viz_path = os.path.join(save_dir, f"user_{user_id}_{photo.replace('.jpg', '_facial_hair_viz.jpg')}") if VISUALIZE else None

        result, rotated = detector.extract_from_image(frame, viz_path=viz_path)
        if result["type"] != "none":
            all_results.append(result)
        if rotated:
            print(f"Photo {photo} rotated 180° for face detection.")

        try:
            cam_idx = int(photo.split('_camera_')[1].split('_')[0])
            if 0 <= cam_idx < 17:
                per_camera_results[cam_idx].append(result)
        except:
            print(f"Error parsing camera for {photo}")

    if not all_results:
        final_result = {"type": "none"}
    else:
        types = [r["type"] for r in all_results]
        lengths = [r["length"] for r in all_results]
        colors = [r["color"] for r in all_results]
        upper_lips = [1 if "upper_lip_covered_by_mustache" in r and r["upper_lip_covered_by_mustache"] else 0 for r in all_results]
        chins = [1 if "chin_covered_by_goatee" in r and r["chin_covered_by_goatee"] else 0 for r in all_results]
        cheeks = [1 if "cheeks_covered_by_beard" in r and r["cheeks_covered_by_beard"] else 0 for r in all_results]

        type_codes = [detector.type_to_code[t] for t in types]
        length_codes = [detector.length_to_code[l] for l in lengths]
        color_codes = [detector.color_to_code[c] for c in colors]

        final_type = detector.code_to_type[stats.mode(type_codes, keepdims=False).mode]
        final_length = detector.code_to_length[stats.mode(length_codes, keepdims=False).mode]
        final_color = detector.code_to_color[stats.mode(color_codes, keepdims=False).mode]
        final_upper_lip = bool(stats.mode(upper_lips, keepdims=False).mode)
        final_chin = bool(stats.mode(chins, keepdims=False).mode)
        final_cheeks = bool(stats.mode(cheeks, keepdims=False).mode)

        final_result = {"type": final_type, "length": final_length, "color": final_color}
        if final_upper_lip:
            final_result["upper_lip_covered_by_mustache"] = True
        if final_chin:
            final_result["chin_covered_by_goatee"] = True
        if final_cheeks:
            final_result["cheeks_covered_by_beard"] = True

    output_file = os.path.join(user_dir, f"user_{user_id}_facial_hair.txt")
    with open(output_file, 'w') as f:
        json.dump(final_result, f)
    print(f"Facial hair for user {user_id}: {final_result} (saved to {output_file}); from {len(photos)} photos.")

    if EVALUATE_CAMERAS:
        compute_and_write_scores(user_id, save_dir, per_camera_results, final_result)

if __name__ == "__main__":
    main()