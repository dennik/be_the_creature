# ssim_camera_mapper.py
# Version: 1.9
# Changes:
# - v1.9 (2025-12-17): Made thumbnail generation optional via --thumbnails CLI flag (disabled by default).
# - v1.8 (2025-12-17): Added generation of 17 separate JPG thumbnail comparison files for matched pairs. Each file shows side-by-side thumbnails (resized to 400x400) of init_camera_## and the best-oriented camera_## (normal or 180° rotated based on higher SSIM). Saved as match_init_XX_ref_YY.jpg in INIT_PHOTOS. Labels included on images.
# - v1.7 (2025-12-17): Updated JSON output to be sorted by real-world index (values) using OrderedDict for preserved order in dump. Added print section for mapping sorted by real-world index.

import sys
import os
import json
import numpy as np
import cv2
import argparse
from skimage.metrics import structural_similarity as ssim
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict

# Dynamically resolve ROOT_DIR and INIT_PHOTOS (from config structure)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INIT_PHOTOS = os.path.join(ROOT_DIR, 'photogrammetry', 'initialization_frames')

MAX_SIDE = 800  # Good balance of accuracy and speed
THUMB_SIZE = (400, 400)  # Thumbnail size for comparisons

def center_crop_50(img):
    h, w = img.shape[:2]
    crop_h = int(h * 0.5)
    crop_w = int(w * 0.5)
    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2
    return img[start_y:start_y + crop_h, start_x:start_x + crop_w]

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cropped = center_crop_50(gray)
    h, w = cropped.shape
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        cropped = cv2.resize(cropped, (new_w, new_h))
    return cropped

def load_images(prefix, num_cams=17, ext='.jpg'):
    images = {}
    for i in range(num_cams):
        filename = f"{prefix}_{i:02d}{ext}"
        path = os.path.join(INIT_PHOTOS, filename)
        if not os.path.exists(path):
            if i < 10:
                filename_alt = f"{prefix}_{i}{ext}"
                path = os.path.join(INIT_PHOTOS, filename_alt)
                if not os.path.exists(path):
                    print(f"Warning: Missing {filename} and {filename_alt}")
                    continue
            else:
                print(f"Warning: Missing {filename}")
                continue
        img = cv2.imread(path)
        if img is None:
            print(f"Error: Failed to load {os.path.basename(path)}")
            continue
        images[i] = preprocess_image(img)
    return images

def load_original_image(prefix, idx, ext='.jpg'):
    filename = f"{prefix}_{idx:02d}{ext}"
    path = os.path.join(INIT_PHOTOS, filename)
    if not os.path.exists(path):
        if idx < 10:
            filename_alt = f"{prefix}_{idx}{ext}"
            path = os.path.join(INIT_PHOTOS, filename_alt)
            if not os.path.exists(path):
                return None
    img = cv2.imread(path)
    if img is None:
        print(f"Error loading original {filename}")
    return img

def compute_ssim_pair(args):
    init_idx, init_img, ref_idx, ref_normal, ref_rotated = args
    # Try both orientations of the reference image
    scores = []
    for ref_img in [ref_normal, ref_rotated]:
        ref_resized = cv2.resize(ref_img, (init_img.shape[1], init_img.shape[0]))
        score = ssim(init_img, ref_resized, data_range=ref_resized.max() - ref_resized.min())
        scores.append(score)
    best_score = max(scores)
    return init_idx, ref_idx, best_score

def compute_ssim_matrix(init_images, ref_images):
    num_cams = len(init_images)
    ssim_matrix = np.zeros((num_cams, num_cams))
    
    # Precompute 180° rotated versions of all reference images
    ref_rotated = {idx: cv2.rotate(img, cv2.ROTATE_180) for idx, img in ref_images.items()}
    
    tasks = []
    for init_idx, init_img in init_images.items():
        for ref_idx in ref_images:
            tasks.append((init_idx, init_img, ref_idx, ref_images[ref_idx], ref_rotated[ref_idx]))
    
    print(f"Computing {len(tasks)} SSIM comparisons (with auto-rotation handling) in parallel...")
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_ssim_pair, task) for task in tasks]
        for future in as_completed(futures):
            init_idx, ref_idx, score = future.result()
            ssim_matrix[init_idx, ref_idx] = score
            print(f"SSIM init_{init_idx:02d} vs ref_{ref_idx:02d} (best of normal/rotated): {score:.4f}")
    
    return ssim_matrix

def optimal_assignment(ssim_matrix):
    row_ind, col_ind = linear_sum_assignment(-ssim_matrix)
    mapping = {int(row): int(col) for row, col in zip(row_ind, col_ind)}
    return mapping

def validate_mapping(ssim_matrix, mapping):
    scores = [ssim_matrix[win, real] for win, real in mapping.items()]
    avg_score = np.mean(scores)
    print(f"\nValidation: Average SSIM of matched pairs: {avg_score:.4f}")
    if avg_score < 0.7:
        print("Warning: Average SSIM is low (<0.7). Mapping may be unreliable.")
    print("Top-3 matches per init camera:")
    for init_idx in range(ssim_matrix.shape[0]):
        row = ssim_matrix[init_idx]
        top3 = np.argsort(row)[-3:][::-1]
        print(f"  init_{init_idx:02d}: ref_{top3[0]:02d} ({row[top3[0]]:.4f}), ref_{top3[1]:02d} ({row[top3[1]]:.4f}), ref_{top3[2]:02d} ({row[top3[2]]:.4f})")

def generate_thumbnail_comparisons(mapping):
    print("\nGenerating thumbnail comparison images...")
    for win_idx, real_idx in sorted(mapping.items()):
        init_img = load_original_image("init_camera", win_idx)
        ref_img_normal = load_original_image("camera", real_idx)
        if init_img is None or ref_img_normal is None:
            print(f"Skipping match init_{win_idx:02d} ref_{real_idx:02d} due to load error")
            continue
        
        ref_img_rotated = cv2.rotate(ref_img_normal, cv2.ROTATE_180)
        
        # Compute SSIM for both to decide orientation (use small grayscale for speed)
        init_gray = cv2.resize(cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY), THUMB_SIZE)
        ref_normal_gray = cv2.resize(cv2.cvtColor(ref_img_normal, cv2.COLOR_BGR2GRAY), THUMB_SIZE)
        ref_rotated_gray = cv2.resize(cv2.cvtColor(ref_img_rotated, cv2.COLOR_BGR2GRAY), THUMB_SIZE)
        
        ssim_normal = ssim(init_gray, ref_normal_gray, data_range=ref_normal_gray.max() - ref_normal_gray.min())
        ssim_rotated = ssim(init_gray, ref_rotated_gray, data_range=ref_rotated_gray.max() - ref_rotated_gray.min())
        
        best_ref = ref_img_rotated if ssim_rotated > ssim_normal else ref_img_normal
        orientation_note = " (rotated)" if ssim_rotated > ssim_normal else ""
        
        # Resize to thumbnails
        init_thumb = cv2.resize(init_img, THUMB_SIZE)
        ref_thumb = cv2.resize(best_ref, THUMB_SIZE)
        
        # Add labels
        cv2.putText(init_thumb, f"Init Camera {win_idx:02d}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(ref_thumb, f"Ref Camera {real_idx:02d}{orientation_note}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Concat horizontally with a gap
        gap = np.zeros((THUMB_SIZE[1], 20, 3), dtype=np.uint8)  # 20px black gap
        comparison = np.hstack((init_thumb, gap, ref_thumb))
        
        output_file = os.path.join(INIT_PHOTOS, f"match_init_{win_idx:02d}_ref_{real_idx:02d}.jpg")
        cv2.imwrite(output_file, comparison)
        print(f"Saved {output_file}")

def main():
    parser = argparse.ArgumentParser(description="SSIM Camera Mapper")
    parser.add_argument('--thumbnails', action='store_true', help='Generate thumbnail comparison images for matched pairs')
    args = parser.parse_args()
    
    init_images = load_images("init_camera")
    ref_images = load_images("camera")
    
    if len(init_images) < 17 or len(ref_images) < 17:
        print("Error: Incomplete image sets. Ensure all 17 images exist for both sets.")
        sys.exit(1)
    
    ssim_matrix = compute_ssim_matrix(init_images, ref_images)
    
    mapping = optimal_assignment(ssim_matrix)
    
    validate_mapping(ssim_matrix, mapping)
    
    # Sort mapping by real-world index (values) for JSON
    sorted_items = sorted(mapping.items(), key=lambda x: x[1])
    ordered_mapping = OrderedDict(sorted_items)
    
    output_path = os.path.join(INIT_PHOTOS, 'ssim_camera_mapping.json')
    with open(output_path, 'w') as f:
        json.dump(ordered_mapping, f, indent=4)
    
    print(f"\nMapping saved to {output_path}")
    
    # Print sorted by Windows index
    print("Mapping sorted by Windows index:")
    for win_idx, real_idx in sorted(mapping.items()):
        score = ssim_matrix[win_idx, real_idx]
        print(f"  {win_idx:02d} -> {real_idx:02d} (SSIM: {score:.4f})")
    
    # Print sorted by real-world index
    print("\nMapping sorted by real-world index:")
    for win_idx, real_idx in sorted_items:
        score = ssim_matrix[win_idx, real_idx]
        print(f"  {win_idx:02d} -> {real_idx:02d} (SSIM: {score:.4f})")
    
    if args.thumbnails:
        generate_thumbnail_comparisons(mapping)
    else:
        print("\nThumbnail generation skipped. Use --thumbnails to enable.")

if __name__ == "__main__":
    main()