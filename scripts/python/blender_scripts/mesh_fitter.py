# mesh_fitter.py
# Version: 2.3.0
# Changes:
# - v2.3.0 (2025-12-11): Switched OBJ import to bpy.ops.wm.obj_import for Blender 4.0+ compatibility. Explicitly create world if None before HDRI setup. Added debug for world creation.
# - v2.2.0 (2025-12-11): Updated OBJ import to bpy.ops.wm.obj_import for Blender 4.0+ compatibility. Added -90° rotation on X axis to high_poly after import (with apply transforms). Added debug print for rotation.
# - v2.1.0 (2025-12-11): Added debug prints at every major step (start, imports, renders, alignments, exports). Print paths and success/failure. Errors now printed with details before exit.
# - v2.0.0 (2025-12-11): Complete refactor for integration with photogrammetry pipeline

import bpy
import sys
import os
import json
import math
import mathutils
from mathutils import Vector
import bmesh
import importlib.util

# -----------------------------
# CLI Argument: user_dir
# -----------------------------
if "--" not in sys.argv:
    print("DEBUG: No user directory provided. Exiting.")
    sys.exit(1)

argv = sys.argv[sys.argv.index("--") + 1:]
if len(argv) != 1:
    print("DEBUG: Invalid args. Usage: -- <user_dir>")
    sys.exit(1)

user_dir = argv[0]
if not os.path.exists(user_dir):
    print(f"DEBUG: User dir not found: {user_dir}. Exiting.")
    sys.exit(1)

print(f"DEBUG: Starting mesh_fitter for {user_dir}")

photos_dir = os.path.join(user_dir, "photos")
model_dir = os.path.join(user_dir, "3dmodel")
obj_path = os.path.join(model_dir, f"{os.path.basename(user_dir)}.obj")

if not os.path.exists(obj_path):
    print(f"DEBUG: .obj not found: {obj_path}. Exiting.")
    sys.exit(1)

print(f"DEBUG: Found .obj: {obj_path}")

# -----------------------------
# Paths
# -----------------------------
script_dir = os.path.dirname(os.path.realpath(__file__))

# Required helpers
helpers = ["3vertexalign.py", "shrinkwrap.py", "laplacian_smooth.py"]
for helper in helpers:
    path = os.path.join(script_dir, helper)
    if not os.path.exists(path):
        print(f"DEBUG: Missing helper: {path}. Exiting.")
        sys.exit(1)

print("DEBUG: All helpers found.")

# Load modules
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

vertexalign = load_module("vertexalign", os.path.join(script_dir, "3vertexalign.py"))
shrinkwrap = load_module("shrinkwrap", os.path.join(script_dir, "shrinkwrap.py"))
lapsmooth = load_module("lapsmooth", os.path.join(script_dir, "laplacian_smooth.py"))

print("DEBUG: Helpers loaded.")

# -----------------------------
# Config
# -----------------------------
HDRI_PATH = r"C:\blenderscripts\brown_photostudio_02_4k.exr"
RENDER_RES = 2048
VG_EXPAND_RANGE = 1
SMOOTHING_RANGE = 4

# -----------------------------
# Cleanup Scene
# -----------------------------
print("DEBUG: Resetting scene.")
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'

# Load base.blend
base_blend = os.path.join(script_dir, "base.blend")
if not os.path.exists(base_blend):
    print(f"DEBUG: base.blend missing: {base_blend}. Exiting.")
    sys.exit(1)

print(f"DEBUG: Loading from {base_blend}.")

with bpy.data.libraries.load(base_blend) as (data_from, data_to):
    data_to.objects = [name for name in data_from.objects if name in {"low_poly", "Camera"}]

for obj in data_to.objects:
    if obj:
        bpy.context.collection.objects.link(obj)

low_poly = bpy.data.objects.get("low_poly")
cam = bpy.data.objects.get("Camera")
if not low_poly or not cam:
    print("DEBUG: low_poly or Camera missing in base.blend. Exiting.")
    sys.exit(1)

print("DEBUG: Loaded low_poly and Camera.")

# -----------------------------
# Import High-Poly
# -----------------------------
print(f"DEBUG: Importing {obj_path}.")
bpy.ops.wm.obj_import(filepath=obj_path)
high_poly = None
for obj in bpy.context.selected_objects:
    if obj.type == 'MESH':
        high_poly = obj
        break
if not high_poly:
    print("DEBUG: Import failed. Exiting.")
    sys.exit(1)

high_poly.name = "high_poly"
print("DEBUG: Imported high_poly.")

# Rotate high_poly -90° on X
print("DEBUG: Rotating high_poly -90° on X.")
high_poly.rotation_euler = (math.radians(-90), 0, 0)
bpy.context.view_layer.objects.active = high_poly
bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
print("DEBUG: Rotation applied.")

# -----------------------------
# HDRI Lighting
# -----------------------------
print("DEBUG: Setting up HDRI.")
world = bpy.context.scene.world
if world is None:
    print("DEBUG: World is None - creating new.")
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
world.use_nodes = True
nodes = world.node_tree.nodes
links = world.node_tree.links
nodes.clear()

bg = nodes.new('ShaderNodeBackground')
env = nodes.new('ShaderNodeTexEnvironment')
mapping = nodes.new('ShaderNodeMapping')
out = nodes.new('ShaderNodeOutputWorld')

if not os.path.exists(HDRI_PATH):
    print(f"DEBUG: HDRI missing: {HDRI_PATH}. Using default.")
else:
    env.image = bpy.data.images.load(HDRI_PATH)
    print("DEBUG: HDRI loaded.")

links.new(mapping.outputs['Vector'], env.inputs['Vector'])
links.new(env.outputs['Color'], bg.inputs['Color'])
links.new(bg.outputs['Background'], out.inputs['Surface'])
bg.inputs['Strength'].default_value = 1.0

# -----------------------------
# Render High-Poly
# -----------------------------
def render_high_poly():
    print("DEBUG: Starting render.")
    bpy.context.scene.render.resolution_x = RENDER_RES
    bpy.context.scene.render.resolution_y = RENDER_RES
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    temp_path = os.path.join(model_dir, "temp_render.png")
    bpy.context.scene.render.filepath = temp_path
    low_poly.hide_render = True
    bpy.ops.render.render(write_still=True)
    low_poly.hide_render = False
    print(f"DEBUG: Rendered to {temp_path}.")
    return temp_path

rendered_path = render_high_poly()

# -----------------------------
# Detect Landmarks
# -----------------------------
import cv2
import mediapipe as mp

print("DEBUG: Starting landmark detection.")
mp_face_mesh = mp.solutions.face_mesh
image = cv2.imread(rendered_path)
if image is None:
    print(f"DEBUG: Render load failed: {rendered_path}. Exiting.")
    sys.exit(1)

h, w, _ = image.shape
with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        print("DEBUG: No face detected. Exiting.")
        sys.exit(1)
    lmks = results.multi_face_landmarks[0].landmark
    landmarks_2d = [(int(l.x * w), int(l.y * h)) for l in lmks[:468]]

print(f"DEBUG: Detected {len(landmarks_2d)} landmarks.")

# -----------------------------
# Project Landmarks
# -----------------------------
print("DEBUG: Projecting landmarks.")
def world_to_cam_view(pos):
    return bpy_extras.object_utils.world_to_camera_view(bpy.context.scene, cam, pos)

kd = mathutils.kdtree.KDTree(len(high_poly.data.vertices))
mw = high_poly.matrix_world
for i, v in enumerate(high_poly.data.vertices):
    kd.insert(mw @ v.co, i)
kd.balance()

target_landmarks = []
vertex_indices = []

for px, py in landmarks_2d:
    norm_x = px / w
    norm_y = 1 - (py / h)
    origin, direction = bpy_extras.object_utils.world_to_camera_view_ray(cam, norm_x, norm_y)
    origin_local = mw.inverted() @ origin
    direction_local = mw.to_3x3().inverted() @ direction
    hit, loc, _, idx = high_poly.closest_point_on_mesh(origin_local, direction=direction_local)
    if hit:
        world_hit = mw @ loc
        _, nearest_idx, _ = kd.find(world_hit)
        target_landmarks.append(world_hit)
        vertex_indices.append(nearest_idx)
    else:
        target_landmarks.append(None)
        vertex_indices.append(None)

print(f"DEBUG: Projected {len([i for i in vertex_indices if i is not None])} valid landmarks.")

# Save JSON
json_path = os.path.join(model_dir, "high_poly_landmark_indices.json")
with open(json_path, 'w') as f:
    json.dump(vertex_indices, f)
print(f"DEBUG: Saved landmark JSON: {json_path}")

# -----------------------------
# Align
# -----------------------------
print("DEBUG: Starting alignment.")
align_json = os.path.join(script_dir, "alignment_vertices.json")
if not os.path.exists(align_json):
    print("DEBUG: alignment_vertices.json missing. Exiting.")
    sys.exit(1)

import shutil
shutil.copy(align_json, model_dir)

vertexalign.align_vertices()
print("DEBUG: Alignment complete.")

# Re-render
rendered_path = render_high_poly()

# -----------------------------
# Deform
# -----------------------------
print("DEBUG: Starting deformation.")
mw_low = low_poly.matrix_world
for i in range(468):
    if target_landmarks[i] is not None and vertex_indices[i] is not None:
        v = low_poly.data.vertices[i]
        target_world = target_landmarks[i]
        v.co = mw_low.inverted() @ target_world

low_poly.data.update()
print("DEBUG: Deformation complete.")

# -----------------------------
# Shrinkwrap + Smooth
# -----------------------------
chin_json = os.path.join(script_dir, "forehead_chin_vertices.json")
if not os.path.exists(chin_json):
    print("DEBUG: forehead_chin_vertices.json missing – skipping shrinkwrap.")
else:
    shutil.copy(chin_json, model_dir)
    shrinkwrap.apply_shrinkwrap_without_mirror("low_poly", "high_poly", "forehead_chin_vertices.json", SMOOTHING_RANGE, vg_expand_range=VG_EXPAND_RANGE)
    print("DEBUG: Shrinkwrap applied.")

smooth_json = os.path.join(script_dir, "smooth_vertices.json")
if os.path.exists(smooth_json):
    shutil.copy(smooth_json, model_dir)
    lapsmooth.apply_laplacian_smooth("smooth_vertices.json", 15)
    print("DEBUG: Laplacian smooth applied.")

# -----------------------------
# Bake Texture
# -----------------------------
print("DEBUG: Starting texture bake.")
bpy.context.view_layer.objects.active = low_poly
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(low_poly.data)

uv_layer = bm.loops.layers.uv.new("FaceUV")

bpy.ops.mesh.select_all(action='DESELECT')
for i in range(468):
    bm.verts[i].select = True
bpy.ops.mesh.select_mode(type='VERT')
bpy.ops.mesh.select_more()

for face in bm.faces:
    for loop in face.loops:
        loop[uv_layer].uv = world_to_cam_view(low_poly.matrix_world @ loop.vert.co)[:2]

bmesh.update_edit_mesh(low_poly.data)
bpy.ops.object.mode_set(mode='OBJECT')

img = bpy.data.images.new("BakedFace", 2048, 2048)
for mat in low_poly.data.materials:
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    tex = nodes.new('ShaderNodeTexImage')
    tex.image = img
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    out = nodes.new('ShaderNodeOutputMaterial')
    links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

bake_node = low_poly.modifiers.new(name="Bake", type='BAKE')
bake_node.target = high_poly
bake_node.bake_type = 'DIFFUSE'
bake_node.bake_target = 'IMAGE_TEXTURES'
bake_node.use_selected_to_active = True
bake_node.cage_extrusion = 0.15
bake_node.max_ray_distance = 0.3
bake_node.use_cage = True

bpy.context.scene.render.bake.use_pass_direct = False
bpy.context.scene.render.bake.use_pass_indirect = False
bpy.context.scene.render.bake.use_pass_color = True
bpy.context.scene.render.bake.margin = 16

low_poly.data.uv_layers["FaceUV"].active = True
img_node = [n for n in low_poly.data.materials[0].node_tree.nodes if n.type == 'TEX_IMAGE'][0]
img_node.select = True
low_poly.data.materials[0].node_tree.nodes.active = img_node

bpy.ops.object.bake(type='DIFFUSE', use_selected_to_active=True, use_clear=True)
print("DEBUG: Bake complete.")

# -----------------------------
# Export
# -----------------------------
print("DEBUG: Exporting final model.")
output_fbx = os.path.join(model_dir, "final_model.fbx")
bpy.ops.export_scene.fbx(
    filepath=output_fbx,
    use_selection=True,
    object_types={'MESH', 'CAMERA', 'LIGHT'},
    bake_anim=False,
    path_mode='COPY',
    embed_textures=True
)

baked_tex_path = os.path.join(model_dir, "face_texture.png")
img.save_render(filepath=baked_tex_path)

print(f"DEBUG: Exported: {output_fbx} and {baked_tex_path}")
print("DEBUG: Mesh fitting complete!")