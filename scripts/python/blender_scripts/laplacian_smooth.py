# Script: LaplacianSmoothSelected
# Version: 1.0
# Description: Applies Laplacian smoothing to vertices specified in a JSON file.
# The function takes the JSON file name (containing a list of vertex indices) and the number of smoothing iterations.
# Creates a vertex group from the indices and applies a Laplacian Smooth modifier limited to that group.
# Assumes the active object is the mesh to modify (e.g., 'low_poly').

import bpy
import bmesh
import json
import os

def apply_laplacian_smooth(json_filename, num_iterations):
    # Get the active object
    obj = bpy.context.active_object
    if not obj or obj.type != 'MESH':
        raise ValueError("Active object must be a mesh.")
    
    # Determine the path to the JSON file in the project directory
    blend_dir = os.path.dirname(bpy.data.filepath)
    json_path = os.path.join(blend_dir, json_filename)
    
    # Read vertex indices from JSON file
    with open(json_path, 'r') as f:
        indices = json.load(f)
        if not isinstance(indices, list):
            raise ValueError("JSON must contain a list of vertex indices.")
    
    # Switch to edit mode to create the vertex group from indices
    original_mode = bpy.context.mode
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type='VERT')
    bpy.ops.mesh.select_all(action='DESELECT')
    
    # Get BMesh for selection
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()  # Ensure index table is up-to-date
    for i in indices:
        if i < len(bm.verts):
            bm.verts[i].select = True
    bmesh.update_edit_mesh(obj.data)
    
    # Create and assign the vertex group
    vertex_group_name = "Laplacian_VG"
    vg = obj.vertex_groups.new(name=vertex_group_name)
    bpy.ops.object.vertex_group_assign()
    
    # Switch back to object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Add a Laplacian Smooth modifier
    laplacian_mod = obj.modifiers.new(name="LaplacianSmooth", type='LAPLACIANSMOOTH')
    laplacian_mod.vertex_group = vertex_group_name
    laplacian_mod.iterations = num_iterations
    laplacian_mod.lambda_factor = 1.0  # Default strength; adjust if needed
    laplacian_mod.use_normalized = False
    laplacian_mod.preserve_volume = True
    
    # Apply the modifier
    bpy.ops.object.modifier_apply(modifier=laplacian_mod.name)
    
    # Update the mesh
    obj.data.update()
    
    # Restore original mode
    bpy.ops.object.mode_set(mode=original_mode)