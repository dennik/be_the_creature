# Script: MeshFitter
# Version: 1.4.51
# Description: Fits low-poly Mediapipe mesh (468 vertices) to high_poly photoscan mesh using direct setting and smoothing,
# with adjustable HDRI lighting (brown_photostudio_02_4k.exr at 1.0 strength),
# single-angle render with existing 'Camera' (no changes to position/rotation), relaxed ray-casting fallback,
# default EEVEE renderer, and iterative topology-based smoothing with normal projection for all face landmarks,
# with bias to keep mouth, eye, nose, and eyebrow related vertices less changed. Operates directly on original low_poly mesh.
# Creates a material for low_poly, unwraps UVs, and projects texture from high_poly via baking with tunable cage parameters.
# Updated in version 1.4.38: Switched rigid alignment logic to align the high_poly mesh to the landmark locations of the low_poly mesh (pre-ICP/direct set stage),
# keeping the low_poly mesh in place. Added post-alignment update to target_landmarks for consistency. Also fixed a bug by scaling landmark positions after mesh scaling for numerical consistency.
# Updated in version 1.4.39: Removed the entire 1000x scaling logic to simplify the process, as it may not be necessary with current precision handling.
# Updated in version 1.4.40: Removed the pre-ICP rigid alignment code to further simplify the fitting process, assuming meshes are already roughly aligned.
# Updated in version 1.4.41: Refactored for modularity (functions), improved readability (naming/comments), removed unused imports/redundancy; no functional changes.
# Updated in version 1.4.42: Added missing get_ray_origin_and_direction function definition to fix NameError in project_landmarks_to_high_poly.
# Updated in version 1.4.43: Fixed NameError for 'cam_loc' in interpolate_edge_landmarks by passing 'cam' to the function and defining cam_loc inside.
# Updated in version 1.4.44: Fixed NameError for 'edge_landmarks' by defining it as a constant at the top of the script, making it accessible to all functions.
# Updated in version 1.4.45: Modified to call apply_shrinkwrap_without_mirror from shrinkwrap_1_2.py after smooth_low_poly, using "low_poly" as source_mesh, "high_poly" as target_mesh, "chin_vertices.json" as vertex_group, and 4 as smoothing_range.
# Updated in version 1.4.46: Added call to align_vertices from 3vertexalign.py before direct_set_and_deform to align high_poly to low_poly, replacing older rigid transformation logic.
# Updated in version 1.4.47: Added saving of high_poly vertex indices to 'high_poly_landmark_indices.json' after generation in project_landmarks_to_high_poly.
# Updated in version 1.4.48: Fixed missing return statement in get_ray_origin_and_direction for orthographic cameras by ensuring the return is outside the if-elif blocks.
# Updated in version 1.4.49: Added post-alignment update to target_landmarks by recomputing from vertex_indices after calling align_vertices. Also, re-render high_poly after alignment to update the texture image.
# Updated in version 1.4.50: Updated the shrinkwrap call to pass vg_expand_range parameter for controlling the shrinkwrap vertex group expansion.
# Updated in version 1.4.51: Added laplacian smoothing for vertices read from a json file
# Updated in version 1.4.52: Modified assign_materials_and_uvs to create a new UV map "FaceUV" and project UVs from camera view only on faces connected to Mediapipe vertices (0-467), expanded one time (i.e., include adjacent body faces). UV projection is now limited to faces with material_index == 0 (face material)

import bpy
import mathutils
from mathutils import Vector
import cv2
import mediapipe as mp
import os
import math
import numpy as np
from mediapipe.python.solutions import face_mesh_connections
import bpy_extras
import bmesh
import json
import importlib.util
import sys

# Load the shrinkwrap script
shrinkwrap_path = os.path.join(os.path.dirname(bpy.data.filepath), "shrinkwrap.py")
spec = importlib.util.spec_from_file_location("shrinkwrap", shrinkwrap_path)
shrinkwrap = importlib.util.module_from_spec(spec)
sys.modules["shrinkwrap"] = shrinkwrap
spec.loader.exec_module(shrinkwrap)

# Load the 3vertexalign script
align_path = os.path.join(os.path.dirname(bpy.data.filepath), "3vertexalign.py")
spec = importlib.util.spec_from_file_location("vertexalign", align_path)
vertexalign = importlib.util.module_from_spec(spec)
sys.modules["vertexalign"] = vertexalign
spec.loader.exec_module(vertexalign)

# Load the laplacian smooth script
laplacian_smooth_path = os.path.join(os.path.dirname(bpy.data.filepath), "laplacian_smooth.py")
spec = importlib.util.spec_from_file_location("lapsmooth", laplacian_smooth_path)
lapsmooth = importlib.util.module_from_spec(spec)
sys.modules["lapsmooth"] = lapsmooth
spec.loader.exec_module(lapsmooth)

# Constants and parameters
SMOOTHING_ITERATIONS = 5
EDGE_SMOOTH_FACTOR = 0.8
SMOOTHING_DEPTH = 2
PROCESS_NOISE = 0.01
MEASUREMENT_NOISE_BASE = 0.1
MEASUREMENT_NOISE_PROTECTED = 1.0
MEASUREMENT_NOISE_EDGE = 0.05
MEASUREMENT_NOISE_BODY = 0.2
FACE_TEXTURE_EXPAND = 1
BAKE_USE_CAGE = True
BAKE_CAGE_EXTRUSION = 0.15
BAKE_MAX_RAY_DISTANCE = 0.3
FALLOFF_DISTANCE = 0.05
REG_LAMBDA = 1.0  # Unused in direct setting
REG_PROTECTED = 10.0  # Unused
REG_NORMAL = 1.0  # Unused
BODY_FACTOR = 0.1
RENDER_ENGINE = "BLENDER_EEVEE_NEXT"
HDRI_PATH = r"C:\blenderscripts\brown_photostudio_02_4k.exr"
VG_EXPAND_RANGE = 1  # New variable for shrinkwrap vertex group expansion

# Edge landmarks (outer face contour) - defined here for global access
EDGE_LANDMARKS = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Simple Kalman Filter class for 3D position smoothing
class SimpleKalmanFilter:
    def __init__(self, initial_pos, process_noise, measurement_noise):
        self.x = np.array(initial_pos)
        self.P = np.eye(3)
        self.Q = process_noise * np.eye(3)
        self.R = measurement_noise * np.eye(3)
        self.F = np.eye(3)
        self.H = np.eye(3)

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

def get_objects():
    """Retrieve and validate required Blender objects."""
    try:
        low_poly = bpy.data.objects["low_poly"]
        high_poly = bpy.data.objects["high_poly"]
        cam = bpy.data.objects["Camera"]
    except KeyError as e:
        raise ValueError(f"Object {e} not found; ensure 'low_poly', 'high_poly', and 'Camera' exist")
    if low_poly.type != 'MESH' or high_poly.type != 'MESH':
        raise ValueError("Both 'low_poly' and 'high_poly' must be meshes")
    return low_poly, high_poly, cam

def prepare_high_poly(high_poly):
    """Add temp material if needed and subdivide high_poly."""
    if not high_poly.data.materials or not any(mat.node_tree.nodes.get('Image Texture') for mat in high_poly.data.materials if mat and mat.node_tree):
        mat = bpy.data.materials.new(name="TempHighPolyMat")
        mat.use_nodes = True
        mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = (0.8, 0.6, 0.5, 1)
        high_poly.data.materials.append(mat)
    bpy.context.view_layer.objects.active = high_poly
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=2)
    bpy.ops.object.mode_set(mode='OBJECT')

def setup_hdri_lighting(scene):
    """Set up HDRI lighting in the world shader."""
    world = scene.world
    world.use_nodes = True
    node_tree = world.node_tree
    node_tree.nodes.clear()
    bg_node = node_tree.nodes.new(type='ShaderNodeBackground')
    env_node = node_tree.nodes.new(type='ShaderNodeTexEnvironment')
    mapping_node = node_tree.nodes.new(type='ShaderNodeMapping')
    output_node = node_tree.nodes.new(type='ShaderNodeOutputWorld')
    if not os.path.exists(HDRI_PATH):
        raise FileNotFoundError(f"HDRI file not found at {HDRI_PATH}")
    env_node.image = bpy.data.images.load(HDRI_PATH)
    node_tree.links.new(mapping_node.outputs['Vector'], env_node.inputs['Vector'])
    node_tree.links.new(env_node.outputs['Color'], bg_node.inputs['Color'])
    node_tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])
    bg_node.inputs['Strength'].default_value = 1.0
    mapping_node.inputs['Rotation'].default_value[2] = math.radians(45)

def render_high_poly(scene, low_poly, high_poly):
    """Render high_poly to image for landmark detection."""
    scene.render.engine = RENDER_ENGINE
    if RENDER_ENGINE == "CYCLES":
        scene.cycles.samples = 128
        scene.cycles.use_denoising = True
    elif RENDER_ENGINE == "BLENDER_EEVEE_NEXT":
        scene.eevee.use_gtao = True
    scene.render.resolution_x = 2048
    scene.render.resolution_y = 2048
    scene.render.filepath = os.path.join(bpy.path.abspath("//"), "temp_render.png")
    low_poly.hide_render = True
    high_poly.hide_render = False
    bpy.ops.render.render(write_still=True)
    low_poly.hide_render = False
    return scene.render.filepath

def detect_landmarks(image_path):
    """Detect 468 Mediapipe landmarks from rendered image."""
    mp_face_mesh = mp.solutions.face_mesh
    ibug_mapping = list(range(468))
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load {image_path}")
    height, width = image.shape[:2]
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            raise ValueError("No face detected—check texture/lighting on high_poly mesh")
        landmarks_2d = []
        for idx in ibug_mapping:
            lm = results.multi_face_landmarks[0].landmark[idx]
            x = int(lm.x * width)
            y = int(lm.y * height)
            landmarks_2d.append((x, y))
    return landmarks_2d

def get_ray_origin_and_direction(cam, norm_x, norm_y):
    """Compute ray origin and direction from camera for normalized coords."""
    if cam.data.type == 'PERSP':
        frame = cam.data.view_frame()
        bottom_left = frame[2]
        bottom_right = frame[1]
        top_left = frame[3]
        top_right = frame[0]
        u = (1 - norm_x) * bottom_left + norm_x * bottom_right
        v = (1 - norm_x) * top_left + norm_x * top_right
        vec = (1 - norm_y) * u + norm_y * v
        direction = (cam.matrix_world.to_3x3() @ vec).normalized()
        origin = cam.matrix_world.translation
    elif cam.data.type == 'ORTHO':
        scale = cam.data.ortho_scale / 2.0
        shift_x = cam.data.shift_x * cam.data.ortho_scale
        shift_y = cam.data.shift_y * cam.data.ortho_scale
        right = cam.matrix_world.to_3x3() @ Vector((1, 0, 0))
        up = cam.matrix_world.to_3x3() @ Vector((0, 1, 0))
        x_offset = (norm_x - 0.5) * cam.data.ortho_scale + shift_x
        y_offset = (norm_y - 0.5) * cam.data.ortho_scale + shift_y
        origin = cam.matrix_world.translation + x_offset * right + y_offset * up
        direction = cam.matrix_world.to_3x3() @ Vector((0, 0, -1)).normalized()
    else:
        raise ValueError("Unsupported camera type")
    return origin, direction

def project_landmarks_to_high_poly(high_poly, cam, scene, landmarks_2d):
    """Project 2D landmarks to 3D on high_poly surface."""
    matrix = high_poly.matrix_world
    kd_high = mathutils.kdtree.KDTree(len(high_poly.data.vertices))
    for i, v in enumerate(high_poly.data.vertices):
        kd_high.insert(matrix @ v.co, i)
    kd_high.balance()
    vertex_indices = []
    target_landmarks = []
    res_x, res_y = scene.render.resolution_x, scene.render.resolution_y
    for i, (px, py_cv) in enumerate(landmarks_2d):
        norm_x = px / (res_x - 1)
        norm_y = 1 - (py_cv / (res_y - 1))
        global_origin, global_direction = get_ray_origin_and_direction(cam, norm_x, norm_y)
        local_origin = high_poly.matrix_world.inverted() @ global_origin
        local_direction = high_poly.matrix_world.to_3x3().inverted() @ global_direction
        local_direction.normalize()
        success, location_local, _, _ = high_poly.ray_cast(local_origin, local_direction, distance=1000.0)
        if success:
            location = matrix @ location_local
            _, index, _ = kd_high.find(location)
            vertex_indices.append(index)
            target_landmarks.append(location)
        else:
            vertex_indices.append(None)
            target_landmarks.append(None)
    # Save high_poly landmark indices to JSON
    blend_dir = os.path.dirname(bpy.data.filepath)
    json_path = os.path.join(blend_dir, "high_poly_landmark_indices.json")
    with open(json_path, 'w') as f:
        json.dump(vertex_indices, f, indent=4)
    return target_landmarks, vertex_indices, kd_high

def handle_failed_projections(vertex_indices, target_landmarks, landmarks_2d, cam, high_poly, scene, kd_high):
    """Handle failed ray casts by falling back to nearest vertex."""
    matrix = high_poly.matrix_world
    res_x, res_y = scene.render.resolution_x, scene.render.resolution_y
    for i in range(len(target_landmarks)):
        if target_landmarks[i] is None:
            px, py_cv = landmarks_2d[i]
            norm_x = px / (res_x - 1)
            norm_y = 1 - (py_cv / (res_y - 1))
            global_origin, global_direction = get_ray_origin_and_direction(cam, norm_x, norm_y)
            local_origin = high_poly.matrix_world.inverted() @ global_origin
            local_direction = high_poly.matrix_world.to_3x3().inverted() @ global_direction
            local_direction.normalize()
            # Relaxed fallback: find nearest vertex if ray_cast fails
            _, index, dist = kd_high.find(global_origin + global_direction * 1000)  # Arbitrary large distance
            if dist < 1.0:  # Arbitrary threshold
                location = matrix @ high_poly.data.vertices[index].co
                target_landmarks[i] = location
                vertex_indices[i] = index
    return vertex_indices, target_landmarks

def interpolate_edge_landmarks(vertex_indices, target_landmarks, kd_high, cam):
    """Interpolate positions for edge landmarks if needed."""
    cam_loc = cam.matrix_world.translation
    for i in EDGE_LANDMARKS:
        if target_landmarks[i] is None:
            # Find neighboring landmarks
            neighbors = [j for j in EDGE_LANDMARKS if j != i and target_landmarks[j] is not None]
            if neighbors:
                avg_pos = sum(target_landmarks[j] for j in neighbors) / len(neighbors)
                target_landmarks[i] = avg_pos
                _, index, _ = kd_high.find(avg_pos)
                vertex_indices[i] = index
    return vertex_indices, target_landmarks

def setup_protected_and_neighbors():
    """Setup protected landmarks and face neighbor dict."""
    mp_face_mesh = mp.solutions.face_mesh
    protected_set = set()
    for connection_set in [mp_face_mesh.FACEMESH_LIPS, mp_face_mesh.FACEMESH_LEFT_EYE, mp_face_mesh.FACEMESH_RIGHT_EYE, mp_face_mesh.FACEMESH_NOSE, mp_face_mesh.FACEMESH_LEFT_EYEBROW, mp_face_mesh.FACEMESH_RIGHT_EYEBROW]:
        for a, b in connection_set:
            protected_set.add(a)
            protected_set.add(b)
    face_neighbor_dict = {i: set() for i in range(468)}
    for a, b in face_mesh_connections.FACEMESH_TESSELATION:
        face_neighbor_dict[a].add(b)
        face_neighbor_dict[b].add(a)
    return protected_set, face_neighbor_dict

def smooth_target_landmarks(target_landmarks, face_neighbor_dict, protected_set, high_poly, cam, scene, kd_high, vertex_indices):
    """Smooth target landmarks using Kalman filters and re-project."""
    face_to_smooth = set(EDGE_LANDMARKS)
    kfs = {}
    for i in range(468):
        if target_landmarks[i] is not None:
            initial_pos = [target_landmarks[i].x, target_landmarks[i].y, target_landmarks[i].z]
            if i in EDGE_LANDMARKS:
                meas_noise = MEASUREMENT_NOISE_EDGE
            elif i in protected_set:
                meas_noise = MEASUREMENT_NOISE_PROTECTED
            else:
                meas_noise = MEASUREMENT_NOISE_BASE
            kfs[i] = SimpleKalmanFilter(initial_pos, PROCESS_NOISE, meas_noise)
    for iteration in range(SMOOTHING_ITERATIONS):
        for i in face_to_smooth:
            if target_landmarks[i] is None:
                continue
            neighbors = list(face_neighbor_dict[i])
            valid_neighbors = [target_landmarks[n] for n in neighbors if target_landmarks[n] is not None]
            if valid_neighbors:
                avg_pos = sum(valid_neighbors, Vector((0, 0, 0))) / len(valid_neighbors)
                kf = kfs[i]
                kf.predict()
                z = np.array([avg_pos.x, avg_pos.y, avg_pos.z])
                kf.update(z)
                target_landmarks[i] = Vector(kf.x)
        for i in range(468):
            if target_landmarks[i] is not None:
                world_pos = target_landmarks[i]
                cam_view = bpy_extras.object_utils.world_to_camera_view(scene, cam, world_pos)
                norm_x = cam_view.x
                norm_y = cam_view.y
                if 0 <= norm_x <= 1 and 0 <= norm_y <= 1:
                    global_origin, global_direction = get_ray_origin_and_direction(cam, norm_x, norm_y)
                    local_origin = high_poly.matrix_world.inverted() @ global_origin
                    local_direction = high_poly.matrix_world.to_3x3().inverted() @ global_direction
                    local_direction.normalize()
                    success, location_local, _, _ = high_poly.ray_cast(local_origin, local_direction, distance=1000.0)
                    if success:
                        new_location = high_poly.matrix_world @ location_local
                        target_landmarks[i] = new_location
                        _, new_vertex_idx, _ = kd_high.find(new_location)
                        vertex_indices[i] = new_vertex_idx
    return target_landmarks, vertex_indices

def extract_source_landmarks(low_poly):
    """Extract world positions of low_poly landmarks (0-467)."""
    matrix_low = low_poly.matrix_world
    ibug_mapping = list(range(468))
    source_landmarks = []
    low_poly_vertex_indices = []
    for i, idx in enumerate(ibug_mapping):
        if idx < len(low_poly.data.vertices):
            location = matrix_low @ low_poly.data.vertices[idx].co
            source_landmarks.append(location)
            low_poly_vertex_indices.append(idx)
        else:
            source_landmarks.append(None)
            low_poly_vertex_indices.append(None)
    return source_landmarks, low_poly_vertex_indices

def build_neighbor_dict(low_poly):
    """Build neighbor dictionary for low_poly vertices."""
    bpy.context.view_layer.objects.active = low_poly
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(low_poly.data)
    neighbor_dict = {i: set() for i in range(len(low_poly.data.vertices))}
    for edge in bm.edges:
        a = edge.verts[0].index
        b = edge.verts[1].index
        neighbor_dict[a].add(b)
        neighbor_dict[b].add(a)
    bpy.ops.object.mode_set(mode='OBJECT')
    return neighbor_dict

def build_to_smooth_set(face_neighbor_dict, neighbor_dict):
    """Determine vertices to smooth (edges + inward + outward)."""
    to_smooth = set(EDGE_LANDMARKS)
    inward = set()
    start_inward = set(EDGE_LANDMARKS)
    for _ in range(SMOOTHING_DEPTH):
        new_in = set()
        for lm in start_inward:
            in_neighbors = face_neighbor_dict.get(lm, set()) - to_smooth - inward
            new_in.update(in_neighbors)
        inward.update(new_in)
        start_inward = new_in
    to_smooth.update(inward)
    outward = set()
    start_outward = set(EDGE_LANDMARKS)
    for _ in range(SMOOTHING_DEPTH):
        new_out = set()
        for lm in start_outward:
            body_neighbors = neighbor_dict.get(lm, set()) & set(range(468, len(neighbor_dict))) - to_smooth - outward
            new_out.update(body_neighbors)
        outward.update(new_out)
        start_outward = new_out
    to_smooth.update(outward)
    return to_smooth

def direct_set_and_deform(low_poly, target_landmarks, vertex_indices, low_poly_vertex_indices, neighbor_dict):
    """Directly set face vertices and propagate to body."""
    valid_indices = [i for i, idx in enumerate(vertex_indices) if idx is not None and target_landmarks[i] is not None]
    pre_icp_cos = [v.co.copy() for v in low_poly.data.vertices]
    for j, idx in enumerate(valid_indices):
        if low_poly_vertex_indices[idx] is not None:
            v = low_poly.data.vertices[low_poly_vertex_indices[idx]]
            target_world = target_landmarks[idx]
            v.co = low_poly.matrix_world.inverted() @ target_world
    low_poly.data.update()
    bpy.context.view_layer.update()
    if len(low_poly.data.vertices) > 468:
        landmark_disps = [Vector((0,0,0)) for _ in range(468)]
        for lm_idx in range(468):
            if target_landmarks[lm_idx] is not None:
                landmark_disps[lm_idx] = low_poly.data.vertices[lm_idx].co - pre_icp_cos[lm_idx]
        matrix_low = low_poly.matrix_world
        edge_lm_indices = EDGE_LANDMARKS
        edge_world_pos = [matrix_low @ pre_icp_cos[lm_idx] for lm_idx in edge_lm_indices]
        kd_edge = mathutils.kdtree.KDTree(len(edge_lm_indices))
        for i, pos in enumerate(edge_world_pos):
            kd_edge.insert(pos, i)
        kd_edge.balance()
        effective_falloff = FALLOFF_DISTANCE
        for v_idx in range(468, len(low_poly.data.vertices)):
            v = low_poly.data.vertices[v_idx]
            world_pos = matrix_low @ pre_icp_cos[v_idx]
            nearest_pos, nearest_i, dist = kd_edge.find(world_pos)
            if dist <= effective_falloff:
                weight = 1 - (dist / effective_falloff)
                edge_lm_idx = edge_lm_indices[nearest_i]
                disp = landmark_disps[edge_lm_idx] * weight * BODY_FACTOR
                v.co += disp
        low_poly.data.update()
        bpy.context.view_layer.update()

def smooth_low_poly(low_poly, to_smooth, neighbor_dict, protected_set):
    """Smooth low_poly vertices using Kalman filters."""
    kfs = {}
    for i in to_smooth:
        initial_pos = [low_poly.data.vertices[i].co.x, low_poly.data.vertices[i].co.y, low_poly.data.vertices[i].co.z]
        if i in EDGE_LANDMARKS:
            meas_noise = MEASUREMENT_NOISE_EDGE
        elif i < 468 and i in protected_set:
            meas_noise = MEASUREMENT_NOISE_PROTECTED
        elif i >= 468:
            meas_noise = MEASUREMENT_NOISE_BODY
        else:
            meas_noise = MEASUREMENT_NOISE_BASE
        kfs[i] = SimpleKalmanFilter(initial_pos, PROCESS_NOISE, meas_noise)
    for iteration in range(SMOOTHING_ITERATIONS):
        for i in to_smooth:
            neighbors = list(neighbor_dict[i])
            valid_neighbors = [low_poly.data.vertices[n].co for n in neighbors]
            if valid_neighbors:
                avg_pos = sum(valid_neighbors, Vector((0, 0, 0))) / len(valid_neighbors)
                kf = kfs[i]
                kf.predict()
                z = np.array([avg_pos.x, avg_pos.y, avg_pos.z])
                kf.update(z)
                low_poly.data.vertices[i].co = Vector(kf.x)
        low_poly.data.update()
        bpy.context.view_layer.update()



def assign_materials_and_uvs(low_poly, neighbor_dict, cam, scene, rendered_img_path):
    """Assign materials, expand face if needed, and project UVs on a new 'FaceUV' map for face-connected faces."""
    face_mat = bpy.data.materials.new(name="FaceMat")
    face_mat.use_nodes = True
    body_mat = bpy.data.materials.new(name="BodyMat")
    body_mat.use_nodes = True
    body_mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1)
    low_poly.data.materials.clear()
    low_poly.data.materials.append(face_mat)
    low_poly.data.materials.append(body_mat)
    mediapipe_verts = set(range(468))
    face_verts = set(mediapipe_verts)  # Start with Mediapipe vertices
    # Expand once to include adjacent body vertices
    expand_start = set(EDGE_LANDMARKS)  # Use edge landmarks as starting point for expansion
    new_expand = set()
    for v in expand_start:
        neighbors = neighbor_dict.get(v, set()) & set(range(468, len(low_poly.data.vertices)))
        new_expand.update(neighbors - face_verts)
    face_verts.update(new_expand)
    bpy.context.view_layer.objects.active = low_poly
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(low_poly.data)
    for face in bm.faces:
        # Assign material to faces where at least one vertex is in mediapipe_verts (connected faces), expanded
        has_face_vert = any(v.index in face_verts for v in face.verts)
        face.material_index = 0 if has_face_vert else 1
    bmesh.update_edit_mesh(low_poly.data)
    # Create new UV layer "FaceUV"
    uv_layer = bm.loops.layers.uv.new("FaceUV")
    # Project UVs only on face material faces (material_index == 0)
    for face in bm.faces:
        if face.material_index == 0:  # Only project on face-connected faces
            for loop in face.loops:
                world_pos = low_poly.matrix_world @ loop.vert.co
                cam_view = bpy_extras.object_utils.world_to_camera_view(scene, cam, world_pos)
                loop[uv_layer].uv = (cam_view.x, cam_view.y) if 0 <= cam_view.x <= 1 and 0 <= cam_view.y <= 1 and cam_view.z > 0 else (0, 0)  # Removed Y-flip for corrected orientation
    bmesh.update_edit_mesh(low_poly.data)
    bpy.ops.object.mode_set(mode='OBJECT')
    rendered_img = bpy.data.images.load(rendered_img_path)
    face_mat.node_tree.nodes.clear()
    bsdf = face_mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
    output = face_mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
    face_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    img_node = face_mat.node_tree.nodes.new('ShaderNodeTexImage')
    img_node.image = rendered_img
    img_node.extension = 'CLIP'
    face_mat.node_tree.links.new(img_node.outputs['Color'], bsdf.inputs['Base Color'])
    # Set the new UV map as active for the face material
    low_poly.data.uv_layers.active = low_poly.data.uv_layers["FaceUV"]

# Main execution
low_poly, high_poly, cam = get_objects()
prepare_high_poly(high_poly)
setup_hdri_lighting(bpy.context.scene)
rendered_path = render_high_poly(bpy.context.scene, low_poly, high_poly)
landmarks_2d = detect_landmarks(rendered_path)
target_landmarks, vertex_indices, kd_high = project_landmarks_to_high_poly(high_poly, cam, bpy.context.scene, landmarks_2d)
vertex_indices, target_landmarks = handle_failed_projections(vertex_indices, target_landmarks, landmarks_2d, cam, high_poly, bpy.context.scene, kd_high)
vertex_indices, target_landmarks = interpolate_edge_landmarks(vertex_indices, target_landmarks, kd_high, cam)
protected_set, face_neighbor_dict = setup_protected_and_neighbors()
target_landmarks, vertex_indices = smooth_target_landmarks(target_landmarks, face_neighbor_dict, protected_set, high_poly, cam, bpy.context.scene, kd_high, vertex_indices)
source_landmarks, low_poly_vertex_indices = extract_source_landmarks(low_poly)
neighbor_dict = build_neighbor_dict(low_poly)
to_smooth = build_to_smooth_set(face_neighbor_dict, neighbor_dict)
vertexalign.align_vertices()  # Call 3vertexalign to align high_poly to low_poly before deformation
# Update target_landmarks after alignment
for i in range(468):
    if vertex_indices[i] is not None:
        target_landmarks[i] = high_poly.matrix_world @ high_poly.data.vertices[vertex_indices[i]].co
# Re-render high_poly for updated texture
rendered_path = render_high_poly(bpy.context.scene, low_poly, high_poly)
direct_set_and_deform(low_poly, target_landmarks, vertex_indices, low_poly_vertex_indices, neighbor_dict)
smooth_low_poly(low_poly, to_smooth, neighbor_dict, protected_set)
shrinkwrap.apply_shrinkwrap_without_mirror("low_poly", "high_poly", "forehead_chin_vertices.json", 0, vg_expand_range=VG_EXPAND_RANGE)
lapsmooth.apply_laplacian_smooth("smooth_vertices.json",15)

assign_materials_and_uvs(low_poly, neighbor_dict, cam, bpy.context.scene, rendered_path)