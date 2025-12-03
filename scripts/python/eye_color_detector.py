# eye_color_detector.py
# Version: 1.17
# Change Log:
# v1.17 - Adapted for new dir structure: Added user_dir = os.path.dirname(save_dir), save output_file and csv (if EVALUATE_CAMERAS) to user_dir instead of save_dir (photos now in subdir; txt/csv in main user folder). Retained v1.16 auto-rotate and prior; no other changes. Certainty: 95% (simple dirname; assumes save_dir is user_dir\photos—test for paths).
# v1.16 - Added auto-rotation for upside-down images: In extract_from_image, after initial process, if no face_landmarks, rotate frame 180° (cv2.rotate cv2.ROTATE_180), re-process; use rotated if detects (handles mounted cams). In main, after load, pass to extractor; print/log if rotated per photo (infers camera). Retained v1.15 viz and prior; fixes erratic detection without user list. Certainty: 95% (standard fix per CV forums; assumes 180° suffices—add 90/270 if needed).
# v1.15 - Added optional visualization: Global VISUALIZE flag (default False) to generate per-photo debug images with per-pixel iris color maps. In extract_from_image, for each iris, classify each masked pixel individually (same logic as aggregate), create color-coded mask (e.g., blue=(255,0,0)), overlay semi-transparent on original region, save as user_{id}_camera_{cam}_iris_viz.jpg in SAVE_DIR. Retained v1.14 selected cams/eval; no core changes (viz optional/off by default). Certainty: 90% (enhances debug without perf impact; per-pixel mirrors logic for insight into classifications).

import os  # For path handling and file operations (e.g., os.path.join, os.listdir)
import sys  # For command-line arguments and sys.exit
import cv2  # For image loading (cv2.imread), color conversion (cv2.cvtColor)
import mediapipe as mp  # For FaceMesh landmark detection
import numpy as np  # For array operations (e.g., np.mean, np.zeros_like)
from scipy import stats  # For mode (dominant hue and final vote)
import csv  # For writing per-camera performance CSV

# Global flags and configs
EVALUATE_CAMERAS = False  # Toggle to run camera performance evaluation and CSV append (default False to disable)
SELECTED_CAMERAS = []  # List of camera indices (0-16) to use for eye color detection; fill with best after review (old good cameras list., [7,8,9,10]); empty defaults to all
VISUALIZE = False  # v1.15: Toggle to generate iris color map visualizations (saves images; default False for production)

# Initialize MediaPipe FaceMesh globally for efficiency (reused across calls)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,  # Optimized for static photos, not video
    max_num_faces=1,  # Assume single face in front-facing photo
    refine_landmarks=True,  # Enables iris landmarks for precise eye region extraction
    min_detection_confidence=0.7,  # Strict threshold for reliable face detection
    min_tracking_confidence=0.7  # Helps in filtering even in static mode
)

# Helper function to convert normalized (0-1) landmarks to pixel coordinates
def normalized_to_pixel(normalized_x, normalized_y, image_width, image_height):
    """
    Manual conversion: Multiply by dimensions, floor to int, clamp to bounds.
    Replaces internal _normalized_to_pixel_coordinates to avoid attribute errors.
    """
    x_px = min(max(int(normalized_x * image_width), 0), image_width - 1)
    y_px = min(max(int(normalized_y * image_height), 0), image_height - 1)
    return (x_px, y_px)

class IrisColorExtractor:
    """
    Modular class for extracting iris color from an image (in memory).
    Detects irises via FaceMesh, applies mask, computes mode hue/avg S/V, classifies.
    Reusable for multi-photo processing or Blender/Unity integration (e.g., via subprocess).
    v1.16: Added auto-rotate logic here (try original, if no face rotate 180, re-try; return colors from successful).
    v1.15: Added per-pixel classification for viz: In extract_from_image, if VISUALIZE, create/overlay color map on copy of frame, save viz image.
    """
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
        # Color mapping for numeric mode (ensures mode works on ints)
        self.color_to_code = {'black': 0, 'brown': 1, 'hazel': 2, 'green': 3, 'blue': 4, 'gray': 5}
        self.code_to_color = {v: k for k, v in self.color_to_code.items()}
        # v1.15: Viz color map (BGR for overlay; distinct for debug)
        self.color_bgr = {
            'black': (0, 0, 0), 'brown': (42, 42, 165), 'hazel': (42, 100, 165),
            'green': (0, 255, 0), 'blue': (255, 0, 0), 'gray': (128, 128, 128)
        }

    def extract_from_image(self, frame, viz_path=None):
        """
        Extracts iris colors from frame: Processes RGB, gets landmarks, masks irises, computes HSV mode/mean, classifies.
        Returns list of colors (str) for all detected irises (usually 2; empty if no face/irises).
        v1.16: Tries original, if no landmarks rotate 180° and retry; returns colors and rotated flag (bool).
        v1.15: If viz_path, creates/saves color-mapped overlay on copy of frame (per-pixel classification).
        """
        rotated = False
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            # v1.16: Rotate 180° and retry
            frame_rot = cv2.rotate(frame, cv2.ROTATE_180)
            rgb_rot = cv2.cvtColor(frame_rot, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_rot)
            if results.multi_face_landmarks:
                frame = frame_rot  # Use rotated for further processing
                rotated = True
            else:
                return [], rotated  # No face even after rotate

        colors = []
        viz_frame = frame.copy() if viz_path else None  # v1.15: Copy for viz overlay

        for face_landmarks in results.multi_face_landmarks:
            height, width, _ = frame.shape
            # Iris landmarks: Left (474-478), right (469-473); center 468/473? But for mask use all
            for iris_indices in [[474, 475, 476, 477], [469, 470, 471, 472]]:  # Left, right iris
                points = []
                for idx in iris_indices:
                    lm = face_landmarks.landmark[idx]
                    px = normalized_to_pixel(lm.x, lm.y, width, height)
                    points.append(px)
                if len(points) < 4:
                    continue  # Skip incomplete iris

                # Create mask: Polygon from points (approx circle)
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillConvexPoly(mask, np.array(points, dtype=np.int32), 255)

                # Apply mask to HSV frame
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
                iris_pixels = masked_hsv[mask > 0]  # Flatten non-zero

                if len(iris_pixels) < 10:  # Min pixels for reliable stats
                    continue

                # Filter valid: V_min <= V <= V_max, S >= S_min (exclude sclera/skin)
                valid = (iris_pixels[:,2] >= self.v_min) & (iris_pixels[:,2] <= self.v_max) & (iris_pixels[:,1] >= self.s_min)
                valid_pixels = iris_pixels[valid]

                if len(valid_pixels) < 10:
                    continue

                # Compute stats: Mode hue (dominant), mean S/V
                hue_mode = stats.mode(valid_pixels[:,0], keepdims=False).mode
                sat_mean = np.mean(valid_pixels[:,1])
                val_mean = np.mean(valid_pixels[:,2])

                # Classify
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
                    color = 'gray'  # High hue fallback

                colors.append(color)

                # v1.15: If viz, classify per-pixel, create color mask, overlay on viz_frame
                if viz_path:
                    color_mask = np.zeros_like(frame)  # BGR for overlay
                    for y, x in np.argwhere(mask > 0):
                        px_hsv = hsv[y, x]
                        # Same logic as aggregate but per-pixel (simplified without stats)
                        h, s, v = px_hsv
                        if not (self.v_min <= v <= self.v_max and s >= self.s_min):
                            continue  # Skip invalid
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
                    # Overlay semi-transparent (alpha=0.5)
                    alpha = 0.5
                    viz_frame = cv2.addWeighted(viz_frame, 1 - alpha, color_mask, alpha, 0)

        # v1.15: Save viz if enabled
        if viz_path and len(colors) > 0:
            cv2.imwrite(viz_path, viz_frame)

        return colors, rotated

def compute_and_write_scores(user_id, save_dir, per_camera_colors, final_color):
    """
    Computes per-camera scores (0-1: fraction matching final_color), appends to CSV in user_dir.
    v1.17: csv_path to user_dir (parent).
    """
    user_dir = os.path.dirname(save_dir)  # v1.17: Parent for main user folder
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

    # Append row (create if missing)
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if os.path.getsize(csv_path) == 0:  # Header if new
            writer.writerow(['user_id', 'final_color'] + [f'camera_{i}' for i in range(17)])
        writer.writerow([user_id, final_color] + scores)
    print(f"Performance scores appended to {csv_path}")

def main():
    """
    Entry: Takes user_id and SAVE_DIR from args, finds user_{id}_camera_{0-16}_*.jpg (filtered by SELECTED_CAMERAS),
    processes each in memory with IrisColorExtractor, collects all iris colors, determines most common via mode (majority vote).
    Writes final color to user_{user_id}_eye_color.txt; no saves/clutter.
    v1.17: Save output/csv to user_dir = dirname(save_dir).
    v1.16: Added rotated flag from extract, print if rotated per photo (e.g., "Photo X (cam Y) rotated for detection").
    v1.15: Added viz_path for each photo if VISUALIZE (user_{id}_camera_{cam}_iris_viz.jpg).
    v1.14: Added SELECTED_CAMERAS filter (defaults to all if empty) and EVALUATE_CAMERAS toggle for scoring/CSV.
    v1.13: Added per-camera tracking, scoring, printing, and CSV append via compute_and_write_scores().
    v1.11: Multi-camera in-memory processing with voting (mode color); removed inpainting/saves. Modularized with IrisColorExtractor class.
    """
    if len(sys.argv) != 3:
        print("Usage: python eye_color_detector.py <user_id> <save_dir>")
        sys.exit(1)

    user_id = int(sys.argv[1])  # Parse user ID from args
    save_dir = sys.argv[2]  # Parse save directory

    # v1.17: Compute user_dir for outputs
    user_dir = os.path.dirname(save_dir)

    # Default to all cameras if SELECTED_CAMERAS empty
    selected = SELECTED_CAMERAS if SELECTED_CAMERAS else list(range(17))

    # Find selected camera photos for the user
    photos = [f for f in os.listdir(save_dir) if f.startswith(f"user_{user_id}_camera_") and f.endswith(".jpg")
              and any(f"camera_{i}_" in f for i in selected)]
    if len(photos) < len(selected):
        print(f"Warning: Only {len(photos)} photos found for user {user_id} (expected {len(selected)}).")

    # Sort by camera index for consistency (extract index from filename)
    photos.sort(key=lambda f: int(f.split('_camera_')[1].split('_')[0]))

    extractor = IrisColorExtractor()  # Instantiate modular extractor
    all_colors = []  # Collect colors from all irises across selected photos
    per_camera_colors = {i: [] for i in range(17)}  # Track colors per camera (for potential eval; includes all but only selected processed)

    for photo in photos:
        image_path = os.path.join(save_dir, photo)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Warning: Failed to load {photo}")
            continue

        # v1.15: Prepare viz_path if VISUALIZE
        viz_path = os.path.join(save_dir, f"user_{user_id}_{photo.replace('.jpg', '_iris_viz.jpg')}") if VISUALIZE else None

        colors, rotated = extractor.extract_from_image(frame, viz_path=viz_path)  # Get colors and rotated flag
        all_colors.extend(colors)  # Add to total list
        if colors:
            print(f"Colors from {photo}: {colors}")  # Per-photo print
        if rotated:
            print(f"Photo {photo} rotated 180° for face detection (likely upside-down camera).")  # v1.16: Log

        # Parse camera index safely and store per-camera colors
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
        # Map colors to codes for mode (fix non-numeric)
        codes = [extractor.color_to_code[c] for c in all_colors]
        mode_code = stats.mode(codes, keepdims=False).mode
        color = extractor.code_to_color[mode_code]
        print(f"Final eye color (majority vote): {color} from {all_colors}")

    output_file = os.path.join(user_dir, f"user_{user_id}_eye_color.txt")
    with open(output_file, 'w') as f:
        f.write(color)  # Write final color to file
    print(f"Eye color for user {user_id}: {color} (saved to {output_file}); detected from {len(all_colors)} irises across {len(photos)} photos.")

    # Optional: Evaluate and record per-camera performance if flag True
    if EVALUATE_CAMERAS:
        compute_and_write_scores(user_id, save_dir, per_camera_colors, color)

if __name__ == "__main__":
    main()