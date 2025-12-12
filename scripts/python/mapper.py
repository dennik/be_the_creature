# camera_mapper.py
# Version: 1.8
# Changes:
# - v1.8 (2025-12-12): After computing and saving mapping, rename .jpg and .xmp files in photos_dir to "camera_{:02d}.jpg" and "camera_{:02d}.xmp" using physical IDs. Skips if no mapping or orig not in mapping; warns on errors.
# - v1.7 (2025-12-12): Flipped within-row sorting to increasing local_x (reverse=False) to correct left-right reversal when viewing from back. This assigns low phys IDs to low local_x (subject's left if right_axis points right).
# - v1.6 (2025-12-12): Improved row clustering: Now tries clustering along both local_y and local_x axes, selects the axis and sort direction (desc/asc) that best matches expected row sizes [4,5,4,4] or reverse. Uses 'ward' linkage and cut_tree to force exactly 4 clusters. Added check for <4 unique coordinates. This addresses cases where up_axis may be misaligned with row separation.

import sys
import os
import numpy as np
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import cut_tree
import xml.etree.ElementTree as ET
import json
from pathlib import Path

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

        if rot_elem is None or pos_elem is None:
            return None, None

        rot_values = list(map(float, rot_elem.text.split()))
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

    poses = {}
    for xmp_path in xmp_files:
        base_name = xmp_path.stem
        try:
            cam_idx_str = base_name.split('_camera_')[1].split('_')[0]
            orig_idx = int(cam_idx_str)
        except (IndexError, ValueError):
            print(f"Warning: Could not parse original index from {base_name}. Skipping.")
            continue

        pos, rot = parse_xmp(xmp_path)
        if pos is None or rot is None:
            print(f"Warning: Invalid pose data in {xmp_path}. Skipping.")
            continue

        poses[orig_idx] = {'pos': pos, 'rot': rot}

    if len(poses) == 0:
        print("No valid camera poses found from XMP files.")
        return None

    if len(poses) < 17:
        print(f"Warning: Only {len(poses)} valid XMP poses found (expected 17).")

    # Compute average forward and up directions
    forward_dirs = []
    up_dirs = []
    positions = []
    for data in poses.values():
        rot = data['rot']
        # Assuming camera forward is -Z, up is +Y
        forward_dir = rot @ np.array([0, 0, -1])
        up_dir = rot @ np.array([0, 1, 0])
        forward_norm = np.linalg.norm(forward_dir)
        up_norm = np.linalg.norm(up_dir)
        if forward_norm == 0 or up_norm == 0:
            print(f"Warning: Invalid direction norms for a camera. Skipping normalization.")
            continue
        forward_dirs.append(forward_dir / forward_norm)
        up_dirs.append(up_dir / up_norm)
        positions.append(data['pos'])

    if len(forward_dirs) == 0:
        print("No valid directions after normalization.")
        return None

    avg_forward = np.mean(forward_dirs, axis=0)
    forward_norm = np.linalg.norm(avg_forward)
    if forward_norm == 0 or np.any(np.isnan(avg_forward)):
        print("Error: Invalid average forward direction.")
        return None
    forward_axis = avg_forward / forward_norm

    avg_up = np.mean(up_dirs, axis=0)
    # Orthogonalize up to forward
    up_axis = avg_up - np.dot(avg_up, forward_axis) * forward_axis
    up_norm = np.linalg.norm(up_axis)
    if up_norm == 0 or np.any(np.isnan(up_axis)):
        print("Error: Up axis is zero after orthogonalization. Cameras may be misoriented.")
        return None
    up_axis = up_axis / up_norm

    # Right axis: up x forward (right-handed)
    right_axis = np.cross(up_axis, forward_axis)

    # Center position
    center_pos = np.mean(positions, axis=0)

    # Compute local coordinates
    points = []
    for orig, data in poses.items():
        pos = data['pos']
        rel_pos = pos - center_pos
        local_x = np.dot(rel_pos, right_axis)
        local_y = np.dot(rel_pos, up_axis)
        local_z = np.dot(rel_pos, forward_axis)
        points.append({'orig': orig, 'local_x': local_x, 'local_y': local_y, 'local_z': local_z})

    # Try clustering along both axes to find best match to expected row sizes
    best_config = None
    min_diff = float('inf')
    expected_desc = [4, 5, 4, 4]
    expected_asc = [4, 4, 5, 4]

    for axis in ['y', 'x']:
        coord = [p[f'local_{axis}'] for p in points]
        coord_arr = np.array(coord)[:, np.newaxis]

        unique_coords = np.unique(coord_arr)
        if len(unique_coords) < 4:
            print(f"Skipping axis {axis}: Fewer than 4 unique coordinates.")
            continue

        if np.std(coord_arr) < 1e-6:
            print(f"Skipping axis {axis}: Insufficient variance.")
            continue

        Z = hierarchy.linkage(coord_arr, method='ward')

        labels = cut_tree(Z, n_clusters=4)[:, 0]

        unique_labels = np.unique(labels)

        if len(unique_labels) < 4:
            print(f"Skipping axis {axis}: Could not form 4 clusters.")
            continue

        cluster_centers = [np.mean(coord_arr[labels == l]) for l in unique_labels]

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

            if diff < min_diff:
                min_diff = diff
                best_config = {
                    'axis': axis,
                    'direction': direction,
                    'labels': labels,
                    'sorted_labels': sorted_labels,
                    'counts': counts
                }

    if best_config is None:
        print("Error: Could not find a suitable clustering configuration.")
        return None

    if min_diff > 2:  # Allow some tolerance for missing cameras
        print(f"Warning: Best clustering diff {min_diff} exceeds threshold; proceeding but check results.")

    print(f"Selected axis: {best_config['axis']}, direction: {best_config['direction']}, counts: {best_config['counts']}")

    labels = best_config['labels']
    sorted_labels = best_config['sorted_labels']

    phys_id_groups = [
        [0, 1, 2, 3],
        [4, 5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]

    mapping = {}
    for row_idx, lab in enumerate(sorted_labels):
        row_points = [p for p in points if labels[points.index(p)] == lab]
        actual_count = len(row_points)
        expected = len(phys_id_groups[row_idx])
        if actual_count != expected:
            print(f"Warning: Row {row_idx+1} has {actual_count} cameras (expected {expected}). Proceeding anyway.")

        # Sort by increasing local_x (user's left: low x = low phys ID, user's right: high x = high phys ID)
        row_points.sort(key=lambda p: p['local_x'], reverse=False)

        phys_ids = phys_id_groups[row_idx]
        for i, point in enumerate(row_points):
            if i < len(phys_ids):
                mapping[point['orig']] = phys_ids[i]
            else:
                print(f"Warning: Extra camera in row {row_idx+1} (orig {point['orig']}) not assigned.")

    # Handle any unassigned (though unlikely)
    assigned_phys = set(mapping.values())
    all_phys = set(range(17))
    missing_phys = all_phys - assigned_phys
    if missing_phys:
        print(f"Warning: Missing physical IDs: {missing_phys}")

    return mapping

def rename_files(photos_dir, mapping):
    photos_dir = Path(photos_dir)
    jpg_files = list(photos_dir.glob("*.jpg"))

    renamed_count = 0
    for jpg_path in jpg_files:
        base_name = jpg_path.stem
        try:
            cam_idx_str = base_name.split('_camera_')[1].split('_')[0]
            orig_idx = int(cam_idx_str)
        except (IndexError, ValueError):
            print(f"Warning: Could not parse original index from {base_name}. Skipping rename.")
            continue

        if orig_idx not in mapping:
            print(f"Warning: Original index {orig_idx} not in mapping. Skipping rename.")
            continue

        phys_id = mapping[orig_idx]
        new_base = f"camera_{phys_id:02d}"
        new_jpg_path = photos_dir / f"{new_base}.jpg"

        try:
            jpg_path.rename(new_jpg_path)
            print(f"Renamed {jpg_path.name} to {new_jpg_path.name}")
            renamed_count += 1
        except Exception as e:
            print(f"Error renaming {jpg_path.name}: {e}")

        # Rename corresponding XMP if exists
        xmp_path = jpg_path.with_suffix('.xmp')
        if xmp_path.exists():
            new_xmp_path = photos_dir / f"{new_base}.xmp"
            try:
                xmp_path.rename(new_xmp_path)
                print(f"Renamed {xmp_path.name} to {new_xmp_path.name}")
            except Exception as e:
                print(f"Error renaming {xmp_path.name}: {e}")

    print(f"Renamed {renamed_count} image files (and their XMP if present).")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        print("Usage: python camera_mapper.py [<photos_dir>]")
        sys.exit(1)

    if len(sys.argv) == 2:
        photos_dir_arg = sys.argv[1]
    else:
        photos_dir_arg = os.getcwd()

    mapping = compute_mapping(photos_dir_arg)

    if mapping is None:
        print("Failed to compute mapping.")
        sys.exit(1)

    # Output for inspection
    print("Computed Mapping (original Windows index → physical ID):")
    for orig, phys in sorted(mapping.items(), key=lambda x: x[1]):
        print(f"Original {orig} → Physical {phys}")

    # Save to JSON
    photos_dir = Path(photos_dir_arg)
    user_dir = photos_dir.parent
    output_path = user_dir / "camera_mapping.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=4)
    print(f"Mapping saved to {output_path}")

    # Rename files
    rename_files(photos_dir, mapping)