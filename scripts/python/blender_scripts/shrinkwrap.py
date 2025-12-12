# Script: ShrinkwrapWithoutMirror
# Version: 3.3
# Description: Modified version of ShrinkwrapWithMirror v2.0, removing mirroring functionality.
# Wrapped in a function with inputs for source_mesh, target_mesh, vertex_group (JSON file name
# containing vertex indices), and smoothing_range (int for selection expansion). Reads vertex
# indices from JSON, creates vertex group, applies shrinkwrap and smooth modifiers to the model.
# Updated in version 3.1: Fixed IndexError by adding bm.verts.ensure_lookup_table() before accessing vertices.
# Updated in version 3.2: Modified to exclude vertex_group vertices from smoothing_group by deselecting them before assigning the smoothing vertex group, ensuring only the transition area is smoothed.
# Updated in version 3.3: Added vg_expand_range parameter (default 0) to expand the shrinkwrap vertex group selection by that many steps. Changed deselect for smoothing group to use vertex_group_deselect for compatibility with expanded vg.

import bpy
import bmesh
import json
import os

def apply_shrinkwrap_without_mirror(source_mesh, target_mesh, vertex_group, smoothing_range, vg_expand_range=0):
    # Get the source and target objects
    source_obj = bpy.data.objects[source_mesh]
    target_obj = bpy.data.objects[target_mesh]
    
    # Apply transforms to source and target (rotation and scale)
    bpy.ops.object.select_all(action='DESELECT')
    source_obj.select_set(True)
    bpy.context.view_layer.objects.active = source_obj
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    target_obj.select_set(True)
    bpy.context.view_layer.objects.active = target_obj
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    
    # Determine the path to the JSON file in the project directory
    blend_dir = os.path.dirname(bpy.data.filepath)
    json_path = os.path.join(blend_dir, vertex_group)
    
    # Read vertex indices from JSON file
    with open(json_path, 'r') as f:
        indices = json.load(f)
    
    # Set vertex group names
    vertex_group_name = "Shrinkwrap_VG"
    smoothing_group_name = "Smoothing_VG"
    
    # Switch to edit mode to create the vertex group from indices
    bpy.context.view_layer.objects.active = source_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type='VERT')
    bpy.ops.mesh.select_all(action='DESELECT')
    
    # Get BMesh for selection
    bm = bmesh.from_edit_mesh(source_obj.data)
    bm.verts.ensure_lookup_table()  # Ensure index table is up-to-date
    for i in indices:
        if i < len(bm.verts):
            bm.verts[i].select = True
    bmesh.update_edit_mesh(source_obj.data)
    
    # Expand the selection for the shrinkwrap vertex group if vg_expand_range > 0
    for _ in range(vg_expand_range):
        bpy.ops.mesh.select_more()
    
    # Create and assign the vertex group
    vg = source_obj.vertex_groups.new(name=vertex_group_name)
    bpy.ops.object.vertex_group_assign()
    
    # Expand the selection by smoothing_range times
    for _ in range(smoothing_range):
        bpy.ops.mesh.select_more()
    
    # Deselect the shrinkwrap vertex group vertices
    bpy.ops.object.vertex_group_set_active(group=vertex_group_name)
    bpy.ops.object.vertex_group_deselect()
    
    # Create and assign the expanded smoothing vertex group
    smoothing_vg = source_obj.vertex_groups.new(name=smoothing_group_name)
    bpy.ops.object.vertex_group_assign()
    
    # Switch back to object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Add a Shrinkwrap modifier to the source object
    shrinkwrap_mod = source_obj.modifiers.new(name="Shrinkwrap", type='SHRINKWRAP')
    # Set the target object
    shrinkwrap_mod.target = target_obj
    # Assign the vertex group to limit the Shrinkwrap effect
    shrinkwrap_mod.vertex_group = vertex_group_name
    # Set wrap method to 'TARGET_PROJECT' for target normal projection
    shrinkwrap_mod.wrap_method = 'TARGET_PROJECT'
    # Enable both directions
    shrinkwrap_mod.use_positive_direction = True
    shrinkwrap_mod.use_negative_direction = True
    # Set a small offset to make deformation visible (adjust as needed)
    shrinkwrap_mod.offset = 0.01
    # Set snap mode (e.g., 'ABOVE_SURFACE' to keep above target)
    shrinkwrap_mod.wrap_mode = 'ABOVE_SURFACE'
    
    # Add a Smooth modifier after the Shrinkwrap, using the smoothing group
    smooth_mod = source_obj.modifiers.new(name="Smooth", type='SMOOTH')
    smooth_mod.vertex_group = smoothing_group_name
    smooth_mod.iterations = 40  # Adjust iterations for smoothing strength
    smooth_mod.factor = 1.0  # Adjust factor as needed
    
    # Apply the modifiers for destructive changes
    bpy.ops.object.modifier_apply(modifier=shrinkwrap_mod.name)
    bpy.ops.object.modifier_apply(modifier=smooth_mod.name)
    
    # Update the mesh
    source_obj.data.update()