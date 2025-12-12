import bpy
import bmesh
from mathutils import Vector, Matrix, Quaternion
import math
import os
import platform
import json

if platform.system() == 'Windows':
    os.system('cls')

# Script: 3Vertex Alignment with JSON Inputs
# Version: 1.3
# Description: Minor refactoring for better code organization and reusability.
# - Extracted apply_transforms function to avoid repetition.
# - Added constants for JSON file names.
# - Improved error messages and added index bounds checking.
# - Standardized print statements for verification.
# No functional changes.

ALIGN_JSON = "alignment_vertices.json"
HIGH_LANDMARK_JSON = "high_poly_landmark_indices.json"

def apply_transforms(obj):
    """Apply location, rotation, and scale transforms to the object."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    obj.data.update()
    bpy.context.view_layer.update()

def get_vertices_world(obj, indices):
    """Get world coordinates of vertices by indices, using evaluated depsgraph for robustness."""
    if any(i < 0 or i >= len(obj.data.vertices) for i in indices):
        raise ValueError(f"Vertex indices out of bounds for object '{obj.name}'")
    dg = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(dg)
    mesh_eval = obj_eval.to_mesh(preserve_all_data_layers=True, depsgraph=dg)
    world_positions = [obj_eval.matrix_world @ mesh_eval.vertices[i].co for i in indices]
    obj_eval.to_mesh_clear()  # Clear the temporary mesh
    return world_positions

def align_first_vertex(static_obj, movable_obj, static_v1, movable_v1):
    """Align the first vertex of the movable object to the first vertex of the static object."""
    translation = static_v1 - movable_v1
    movable_obj.location += translation
    bpy.context.view_layer.update()

def align_second_vertex(static_obj, movable_obj, static_verts, movable_verts):
    """Align the second vertex by rotating and scaling around the first vertex."""
    if len(static_verts) < 2 or len(movable_verts) < 2:
        return False
    static_v1, static_v2 = static_verts[0], static_verts[1]
    movable_v1, movable_v2 = movable_verts[0], movable_verts[1]
    pivot_point = static_v1
    static_vec = static_v2 - static_v1
    movable_vec = movable_v2 - movable_v1
    static_length = static_vec.length
    movable_length = movable_vec.length
    if movable_length < 1e-6 or static_length < 1e-6:
        return False
    scale_factor = static_length / movable_length
    rotation_quat = movable_vec.rotation_difference(static_vec)
    trans_to_origin = Matrix.Translation(-pivot_point)
    scale_matrix = Matrix.Scale(scale_factor, 4)
    rotation_matrix = rotation_quat.to_matrix().to_4x4()
    trans_back = Matrix.Translation(pivot_point)
    transform_matrix = trans_back @ rotation_matrix @ scale_matrix @ trans_to_origin
    movable_obj.matrix_world = transform_matrix @ movable_obj.matrix_world
    bpy.context.view_layer.update()
    return True

def align_third_vertex(static_obj, movable_obj, static_verts, movable_indices):
    """Rotate the movable object around the axis through the first two vertices to align the planes."""
    apply_transforms(movable_obj)
    movable_verts = get_vertices_world(movable_obj, movable_indices)
    if len(static_verts) < 3 or len(movable_verts) < 3:
        return False
    static_v1, static_v2, static_v3 = static_verts[0], static_verts[1], static_verts[2]
    movable_v1, movable_v2, movable_v3 = movable_verts[0], movable_verts[1], movable_verts[2]
    axis = (static_v2 - static_v1).normalized()
    pivot_point = static_v1
    mov_vec = movable_v3 - movable_v1
    stat_vec = static_v3 - static_v1
    mov_perp = mov_vec - (mov_vec.dot(axis) * axis)
    stat_perp = stat_vec - (stat_vec.dot(axis) * axis)
    if mov_perp.length < 1e-6 or stat_perp.length < 1e-6:
        return False
    if abs(mov_perp.length - stat_perp.length) > 1e-4:
        print(f"Warning: Perpendicular distances differ ({mov_perp.length:.6f} vs {stat_perp.length:.6f}), alignment may not be perfect.")
    mov_perp_n = mov_perp.normalized()
    stat_perp_n = stat_perp.normalized()
    dot = mov_perp_n.dot(stat_perp_n)
    angle = math.acos(max(min(dot, 1.0), -1.0))
    cross = mov_perp_n.cross(stat_perp_n)
    sign = math.copysign(1, cross.dot(axis))
    angle *= sign
    rotation_quat = Quaternion(axis, angle)
    trans_to_origin = Matrix.Translation(-pivot_point)
    rotation_matrix = rotation_quat.to_matrix().to_4x4()
    trans_back = Matrix.Translation(pivot_point)
    transform_matrix = trans_back @ rotation_matrix @ trans_to_origin
    movable_obj.matrix_world = transform_matrix @ movable_obj.matrix_world
    bpy.context.view_layer.update()
    updated_verts = get_vertices_world(movable_obj, movable_indices)
    distance = (static_v3 - updated_verts[2]).length
    print(f"Step 3 - Distance between third vertices after alignment: {distance:.6f}")
    return distance < 1e-4

def align_vertices():
    """Main function to align vertices of 'low_poly' and 'high_poly' using JSON inputs."""
    try:
        static_obj = bpy.data.objects["low_poly"]
        movable_obj = bpy.data.objects["high_poly"]
    except KeyError:
        print("Objects 'low_poly' or 'high_poly' not found. Aborting.")
        return
    if static_obj.type != 'MESH' or movable_obj.type != 'MESH':
        print("Both 'low_poly' and 'high_poly' must be meshes. Aborting.")
        return
    blend_dir = bpy.path.abspath("//")
    align_json_path = os.path.join(blend_dir, ALIGN_JSON)
    high_landmark_json_path = os.path.join(blend_dir, HIGH_LANDMARK_JSON)
    if not os.path.exists(align_json_path) or not os.path.exists(high_landmark_json_path):
        print("JSON files not found. Aborting.")
        return
    with open(align_json_path, 'r') as f:
        static_indices = json.load(f)
    with open(high_landmark_json_path, 'r') as f:
        high_landmarks = json.load(f)
    if len(static_indices) != 3 or len(high_landmarks) != 468:
        print("Invalid JSON data. Aborting.")
        return
    movable_indices = [high_landmarks[i] for i in static_indices]
    static_verts = get_vertices_world(static_obj, static_indices)
    movable_verts = get_vertices_world(movable_obj, movable_indices)
    if len(static_verts) < 3 or len(movable_verts) < 3:
        print("Insufficient vertices. Aborting.")
        return
    apply_transforms(movable_obj)
    movable_verts = get_vertices_world(movable_obj, movable_indices)
    align_first_vertex(static_obj, movable_obj, static_verts[0], movable_verts[0])
    apply_transforms(movable_obj)
    movable_verts = get_vertices_world(movable_obj, movable_indices)
    if not align_second_vertex(static_obj, movable_obj, static_verts, movable_verts):
        print("Step 2 alignment failed.")
        return
    updated_verts = get_vertices_world(movable_obj, movable_indices)
    distance = (static_verts[1] - updated_verts[1]).length
    print(f"Step 2 - Distance between second vertices after alignment: {distance:.6f}")
    if distance >= 1e-6:
        print("Step 2 verification failed.")
        return
    if not align_third_vertex(static_obj, movable_obj, static_verts, movable_indices):
        print("Step 3 alignment failed.")
        return
    final_verts = get_vertices_world(movable_obj, movable_indices)
    print(final_verts[1][:])

# Execute the alignment
align_vertices()