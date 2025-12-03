# standardize_face.py
# Version: 1.0
# Run with: blender --background --python standardize_face.py -- "D:\3Dmodels\user_42\user_42.obj"

import bpy
import sys
import os
import mathutils

# ------------------------------------------------------------------
# Get path from command line
# ------------------------------------------------------------------
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    raise Exception("Expected -- <path_to_obj>")

obj_path = argv[0]
folder    = os.path.dirname(obj_path)
obj_name  = os.path.basename(obj_path)

# ------------------------------------------------------------------
# Clear scene
# ------------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)

# ------------------------------------------------------------------
# Import the freshly exported OBJ
# ------------------------------------------------------------------
bpy.ops.import_scene.obj(filepath=obj_path)

# Find the mesh object (RealityScan always names it after the file)
obj = None
for o in bpy.data.objects:
    if o.type == 'MESH':
        obj = o
        break
if obj is None:
    raise Exception("No mesh found after import")

# ------------------------------------------------------------------
# 1. Re-orient so that the PREVIEW CAMERA points +Z (Blender convention)
# ------------------------------------------------------------------
# In your rig the preview camera is mounted looking straight at the face
# and is exactly at world origin (0,0,0) looking down -Z in OpenCV coordinates.
# RealityScan exports cameras in OpenCV convention → +Z is forward.
# So we simply rotate the whole mesh 180° around X to make the face look +Z.

obj.rotation_euler = (mathutils.Euler((math.radians(180), 0, 0), 'XYZ'))

# ------------------------------------------------------------------
# 2. Uniform scale so that eye-distance = 6.5 cm (average human)
#    This removes any scale variation between sessions
# ------------------------------------------------------------------
# Landmark indices for left/right outer eye corners (MediaPipe FaceMesh)
LEFT_EYE_OUTER  = 33
RIGHT_EYE_OUTER = 263

# Get the two vertices that correspond to these landmarks
# (RealityScan exports them with vertex groups or you can just use distance)
coords = [v.co for v in obj.data.vertices]
# Find the two points that are farthest apart on the X axis (quick & dirty but works
left  = min(coords, key=lambda co: co.x)
right = max(coords, key=lambda co: co.x)
current_eye_distance = (right - left).length

TARGET_EYE_DISTANCE = 0.065  # 6.5 cm in meters

scale_factor = TARGET_EYE_DISTANCE / current_eye_distance
obj.scale = (scale_factor, scale_factor, scale_factor)

# ------------------------------------------------------------------
# 3. Center on world origin
# ------------------------------------------------------------------
bpy.ops.object.origin_set(type='GEOMETRY_TO_ORIGIN')
obj.location = (0, 0, 0)

# ------------------------------------------------------------------
# 4. Export cleaned OBJ + MTL + PNG (overwrite)
# ------------------------------------------------------------------
export_path = obj_path  # overwrite the original
bpy.ops.export_scene.obj(
    filepath=export_path,
    use_selection=True,
    use_mesh_modifiers=False,
    use_edges=True,
    use_smooth_groups=True,
    use_normals=True,
    use_uvs=True,
    use_materials=True,
    use_triangles=True,
    keep_vertex_order=True,
    global_scale=1.0,
    path_mode='COPY'  # embeds textures
)

print(f"Standardized model saved: {export_path}")