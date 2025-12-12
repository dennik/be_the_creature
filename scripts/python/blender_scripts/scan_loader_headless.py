# Script: Scan Loader Headless
# Version: 1.1
# Description: Headless version of scan_loader.py. Assumes high_poly.obj exists, imports it, runs mesh_fitter, and exports low_poly_mapped.obj without timers or GUI.
# Updated to load FITTER_SCRIPT from root directory (C:\MyFolder\) instead of Output, to avoid issues with directory clearing.

import bpy
import os

# Customize these
DIRECTORY = r"C:\MyFolder\Output"  # Use raw string for Windows paths; for imports/exports
ROOT_DIRECTORY = r"C:\MyFolder"    # Root for persistent scripts like fitter
FILE_NAME = "high_poly.obj"
FULL_PATH = os.path.join(DIRECTORY, FILE_NAME)
FITTER_SCRIPT = os.path.join(ROOT_DIRECTORY, "mesh_fitter1_4_52.py")  # Updated to root
#FITTER_SCRIPT = os.path.join(ROOT_DIRECTORY, "mesh_fitter1_4_50.py")  # Updated to root
EXPORT_PATH = os.path.join(DIRECTORY, "low_poly_mapped.obj")
LOW_POLY_NAME = "low_poly"

if os.path.exists(FULL_PATH):
    # Import the file (assuming OBJ)
    bpy.ops.wm.obj_import(filepath=FULL_PATH)
    print(f"Imported: {FULL_PATH}")
    
    # Run mesh_fitter (assuming it's synchronous)
    if os.path.exists(FITTER_SCRIPT):
        with open(FITTER_SCRIPT, 'r') as f:
            code = f.read()
        exec(code, globals(), locals())
        print(f"Executed: {FITTER_SCRIPT}")
    else:
        print(f"Fitter script not found: {FITTER_SCRIPT}")
    
    # Export the low_poly object if it exists
    if LOW_POLY_NAME in bpy.data.objects:
        obj = bpy.data.objects[LOW_POLY_NAME]
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.ops.wm.obj_export(filepath=EXPORT_PATH, export_selected_objects=True)
        print(f"Exported: {EXPORT_PATH}")
    else:
        print(f"Object '{LOW_POLY_NAME}' not found for export.")
else:
    print(f"File not found: {FULL_PATH}")

# Optional: Quit Blender after processing
bpy.ops.wm.quit_blender()