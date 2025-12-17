# init_camera_mapper.py
# Version: 1.5
# Changes:
# - v1.5 (2025-12-16): Reversed row_points sort by local_x (reverse=True) to flip X axis assignment per row (e.g., 0↔3, 1↔2). Retained previous.
# - v1.4 (2025-12-16): Updated parsing to use regex r'camera_(\d+)' for camera index from base_name, fixing skips for filenames like "camera_0_16MP". Removed rename_files_based_on_mapping since not needed (per initial request). Retained previous.
# - v1.3 (2025-12-16): Updated parsing to use regex r'camera_(\d+)' for camera index from base_name, fixing IndexError for filenames like "camera_0_16MP". Updated warning messages for clarity. Retained previous.

import sys
import os
import subprocess
import json
from pathlib import Path
import numpy as np
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import cut_tree
import xml.etree.ElementTree as ET
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import PATHS

# Paths
BASE_DIR = PATHS['BASE']
INIT_FRAMES_DIR = os.path.join(BASE_DIR, 'initialization_frames')
RS_PATH = r"C:\Program Files\Epic Games\RealityScan_2.0\RealityScan.exe"
MAPPING_JSON = os.path.join(INIT_FRAMES_DIR, 'camera_mapping.json')

def run_realityscan_align_and_export(photos_dir):
    # Clean old XMPs if any
    for xmp in Path(photos_dir).glob("*.xmp"):
        xmp.unlink()
        print(f"Removed old {xmp.name}")

    command = [
        RS_PATH,
        "-newScene",
        "-stdConsole",
        "-addFolder", photos_dir,
        "-align",
        "-exportXMP",
        "-quit"
    ]

    print(f"Running RealityScan command: {' '.join(command)}")

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    while process.poll() is None:
        line = process.stdout.readline().strip()
        if line:
            print(line)

    result = process.wait()
    if result == 0:
        print("RealityScan alignment and XMP export complete.")
    else:
        print(f"Error in RealityScan execution. Return code: {result}")
        sys.exit(1)

def parse_xmp(xmp_path):
    try:
        tree = ET.parse(xmp_path)
        root = tree.getroot()
        ns = {
            'x': 'adobe:ns:meta/',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'xcr': 'http://www.capturingreality.com/ns/xcr/1.1#'
        }
        desc = root.find('./rdf:RDF/rdf:Description', ns)
        if desc is None:
            return None, None

        rot_elem = desc.find('xcr:Rotation', ns)
        pos_elem = desc.find('xcr:Position', ns)

        # Fallback to attributes if elements not found (handles inconsistent formats)
        xcr_ns = '{http://www.capturingreality.com/ns/xcr/1.1#}'
        if rot_elem is None:
            rot_attr = desc.get(xcr_ns + 'Rotation')
            if rot_attr is not None:
                rot_values = list(map(float, rot_attr.split()))
            else:
                return None, None
        else:
            rot_values = list(map(float, rot_elem.text.split()))

        if pos_elem is None:
            pos_attr = desc.get(xcr_ns + 'Position')
            if pos_attr is not None:
                pos_values = list(map(float, pos_attr.split()))
            else:
                return None, None
        else:
            pos_values = list(map(float, pos_elem.text.split()))

        if len(rot_values) != 9 or len(pos_values) != 3:
            return None, None

        rot_matrix = np.reshape(np.array(rot_values), (3, 3))
        pos_vector = np.array(pos_values)

        return pos_vector, rot_matrix
    except Exception as e:
        print(f"Error parsing {xmp_path}: {e}")
        return None, None

def compute_mapping(photos_dir):
    photos_dir = Path(photos_dir)
    xmp_files = list(photos_dir.glob("*.xmp"))

    if not xmp_files:
        print(f"No .xmp files found in {photos_dir}.")
        return None

    print(f"DEBUG: Found {len(xmp_files)} XMP files: {[f.name for f in xmp_files]}")

    poses = {}
    for xmp_path in xmp_files:
        base_name = xmp_path.stem
        match = re.search(r'camera_(\d+)', base_name)
        if not match:
            print(f"Warning: No 'camera_#' found in {base_name}. Skipping.")
            continue
        cam_idx_str = match.group(1)
        try:
            orig_idx = int(cam_idx_str)
            print(f"DEBUG: Parsed orig_idx {orig_idx} from {base_name}")
        except ValueError:
            print(f"Warning: Could not parse integer from {cam_idx_str} in {base_name}. Skipping.")
            continue

        pos, rot = parse_xmp(xmp_path)
        if pos is None or rot is None:
            print(f"Warning: Invalid pose data in {xmp_path}. Skipping.")
            continue
        print(f"DEBUG: Successfully parsed pos {pos} and rot {rot} for orig_idx {orig_idx}")

        poses[orig_idx] = {'pos': pos, 'rot': rot}

    if len(poses) == 0:
        print("No valid camera poses found from XMP files.")
        return None

    print(f"DEBUG: Collected poses for orig_indices: {sorted(poses.keys())}")

    if len(poses) < 17:
        print(f"Warning: Only {len(poses)} valid XMP poses found (expected 17).")

    # Compute average forward and up directions
    forward_dirs = []
    up_dirs = []
    positions = []
    for orig_idx, data in poses.items():
        rot = data['rot']
        # Assuming camera forward is -Z, up is +Y
        forward_dir = rot @ np.array([0, 0, -1])
        up_dir = rot @ np.array([0, 1, 0])
        forward_norm = np.linalg.norm(forward_dir)
        up_norm = np.linalg.norm(up_dir)
        if forward_norm == 0 or up_norm == 0:
            print(f"Warning: Invalid direction norms for orig_idx {orig_idx}. Skipping normalization.")
            continue
        forward_dirs.append(forward_dir / forward_norm)
        up_dirs.append(up_dir / up_norm)
        positions.append(data['pos'])
        print(f"DEBUG: Added directions for orig_idx {orig_idx}: forward {forward_dir / forward_norm}, up {up_dir / up_norm}")

    if len(forward_dirs) == 0:
        print("No valid directions after normalization.")
        return None

    print(f"DEBUG: Processed {len(forward_dirs)} valid directions.")

    avg_forward = np.mean(forward_dirs, axis=0)
    forward_norm = np.linalg.norm(avg_forward)
    if forward_norm == 0 or np.any(np.isnan(avg_forward)):
        print("Error: Invalid average forward direction.")
        return None
    forward_axis = avg_forward / forward_norm
    print(f"DEBUG: avg_forward {avg_forward}, forward_axis {forward_axis}")

    # Set up_axis to world Z
    up_axis = np.array([0, 0, 1])
    print(f"DEBUG: up_axis {up_axis}")

    # Right axis: up x forward (right-handed)
    right_axis = np.cross(up_axis, forward_axis)
    print(f"DEBUG: right_axis {right_axis}")

    # Center position
    center_pos = np.mean(positions, axis=0)
    print(f"DEBUG: center_pos {center_pos}")

    # Compute local coordinates
    points = []
    for orig, data in poses.items():
        pos = data['pos']
        rel_pos = pos - center_pos
        local_x = np.dot(rel_pos, right_axis)
        local_y = np.dot(rel_pos, up_axis)
        local_z = np.dot(rel_pos, forward_axis)
        points.append({'orig': orig, 'local_x': local_x, 'local_y': local_y, 'local_z': local_z})
        print(f"DEBUG: Local coords for orig {orig}: x={local_x:.4f}, y={local_y:.4f}, z={local_z:.4f}")

    # Try clustering along both axes to find best match to expected row sizes
    all_configs = []
    if len(poses) == 17:
        expected_desc = [4, 5, 4, 4]
        expected_asc = [4, 4, 5, 4]
        prefer_max = 5
    else:
        expected_desc = [4, 4, 4, 4]
        expected_asc = [4, 4, 4, 4]
        prefer_max = 4

    for axis in ['y', 'x']:
        coord = [p[f'local_{axis}'] for p in points]
        coord_arr = np.array(coord)[:, np.newaxis]
        print(f"DEBUG: For axis {axis}, coordinates: {coord}")

        unique_coords = np.unique(coord_arr)
        print(f"DEBUG: Unique coords on {axis}: {unique_coords}")
        if len(unique_coords) < 4:
            print(f"Skipping axis {axis}: Fewer than 4 unique coordinates.")
            continue

        if np.std(coord_arr) < 1e-6:
            print(f"Skipping axis {axis}: Insufficient variance.")
            continue

        Z = hierarchy.linkage(coord_arr, method='ward')
        print(f"DEBUG: Linkage Z for {axis}: {Z}")

        labels = cut_tree(Z, n_clusters=4)[:, 0]
        print(f"DEBUG: Cluster labels for {axis}: {labels}")

        unique_labels = np.unique(labels)

        if len(unique_labels) < 4:
            print(f"Skipping axis {axis}: Could not form 4 clusters.")
            continue

        cluster_centers = [np.mean(coord_arr[labels == l]) for l in unique_labels]
        print(f"DEBUG: Cluster centers for {axis}: {cluster_centers}")

        for direction in ['desc', 'asc']:
            if direction == 'desc':
                sorted_idx = np.argsort(cluster_centers)[::-1]
                exp = expected_desc
            else:
                sorted_idx = np.argsort(cluster_centers)
                exp = expected_asc

            sorted_labels = np.array(unique_labels)[sorted_idx]
            counts = [np.sum(labels == l) for l in sorted_labels]
            diff = sum(abs(c - e) for c, e in zip(counts, exp))
            print(f"DEBUG: For {axis}-{direction}, sorted_labels {sorted_labels}, counts {counts}, diff {diff}")

            all_configs.append({
                'diff': diff,
                'max_count': max(counts),
                'config': {
                    'axis': axis,
                    'direction': direction,
                    'labels': labels,
                    'sorted_labels': sorted_labels,
                    'counts': counts
                }
            })

    if not all_configs:
        print("Error: No suitable clustering configurations found.")
        return None

    # Prefer 'y' axis
    configs_y = [c for c in all_configs if c['config']['axis'] == 'y']
    if configs_y:
        configs_pref = [c for c in configs_y if c['max_count'] == prefer_max] if prefer_max == 5 else [c for c in configs_y if c['max_count'] <= prefer_max]
        if configs_pref:
            best = min(configs_pref, key=lambda c: c['diff'])
        else:
            best = min(configs_y, key=lambda c: c['diff'])
    else:
        best = min(all_configs, key=lambda c: c['diff'])

    best_config = best['config']
    min_diff = best['diff']

    if min_diff > 2:
        print(f"Warning: Best clustering diff {min_diff} exceeds threshold; proceeding but check results.")

    print(f"Selected axis: {best_config['axis']}, direction: {best_config['direction']}, counts: {best_config['counts']}")

    labels = best_config['labels']
    sorted_labels = best_config['sorted_labels']

    phys_id_groups = [
        [0, 1, 2, 3],
        [4, 5, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]

    mapping = {}
    preview_assigned = False
    oversized_row_idx = None
    for row_idx, count in enumerate(best_config['counts']):
        if count > 4:
            oversized_row_idx = row_idx
            break

    if oversized_row_idx is not None:
        if len(poses) == 17:
            remove_outlier = True
        else:
            remove_outlier = False
            if oversized_row_idx == 1:
                phys_id_groups[1] = [4, 5, 6, 7, 8]
            else:
                print(f"Warning: Oversized row {oversized_row_idx+1} not middle; no phys adjustment.")

    for row_idx, lab in enumerate(sorted_labels):
        row_points = [p for p in points if labels[points.index(p)] == lab]
        actual_count = len(row_points)
        expected = len(phys_id_groups[row_idx])
        print(f"DEBUG: Row {row_idx+1} points orig: {[p['orig'] for p in row_points]}")

        if oversized_row_idx == row_idx and actual_count > 4:
            if remove_outlier:
                # Detect and remove outlier
                cluster_axis = best_config['axis']
                coords = np.array([p[f'local_{cluster_axis}'] for p in row_points])
                min_std = float('inf')
                outlier_idx = -1
                for i in range(actual_count):
                    remaining = np.delete(coords, i)
                    curr_std = np.std(remaining)
                    if curr_std < min_std:
                        min_std = curr_std
                        outlier_idx = i
                if outlier_idx != -1:
                    outlier_point = row_points.pop(outlier_idx)
                    mapping[outlier_point['orig']] = 6
                    preview_assigned = True
                    print(f"DEBUG: Detected outlier orig {outlier_point['orig']} in row {row_idx+1}, assigned phys 6")
                else:
                    print(f"Warning: Could not detect outlier in row {row_idx+1}.")

        if len(row_points) != expected:
            print(f"Warning: Row {row_idx+1} has {len(row_points)} cameras (expected {expected}). Proceeding with partial assignment.")

        # Sort by decreasing local_x to flip X axis (high x left)
        row_points.sort(key=lambda p: p['local_x'], reverse=True)
        print(f"DEBUG: Sorted row {row_idx+1} by local_x desc: orig {[p['orig'] for p in row_points]}, local_x {[p['local_x'] for p in row_points]}")

        phys_ids = phys_id_groups[row_idx]
        for i, point in enumerate(row_points):
            if i < len(phys_ids):
                mapping[point['orig']] = phys_ids[i]
                print(f"DEBUG: Assigned orig {point['orig']} to phys {phys_ids[i]}")
            else:
                print(f"Warning: Extra camera in row {row_idx+1} (orig {point['orig']}) not assigned.")

    if len(poses) == 17 and not preview_assigned:
        print("Warning: Preview camera not assigned (no oversized row or detection failed).")

    # Handle any unassigned
    assigned_phys = set(mapping.values())
    all_phys = set(range(17))
    missing_phys = all_phys - assigned_phys
    if missing_phys:
        print(f"Warning: Missing physical IDs: {missing_phys}")

    print(f"DEBUG: Final mapping keys (orig_indices): {sorted(mapping.keys())}")

    return mapping

if __name__ == "__main__":
    os.makedirs(INIT_FRAMES_DIR, exist_ok=True)

    # Step 1: Run RealityScan to align and export XMPs
    run_realityscan_align_and_export(INIT_FRAMES_DIR)

    # Step 2: Compute mapping
    mapping = compute_mapping(INIT_FRAMES_DIR)

    if mapping is None:
        print("Failed to compute mapping.")
        sys.exit(1)

    # Output for inspection
    print("Computed Mapping (original Windows index → physical ID):")
    for orig, phys in sorted(mapping.items(), key=lambda x: x[1]):
        print(f"camera {orig} becomes camera {phys}")

    # Save to JSON
    with open(MAPPING_JSON, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=4)
    print(f"Mapping saved to {MAPPING_JSON}")