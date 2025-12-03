# realityscan_processor.py
# Version: 1.16
# Change Log:
# v1.16 - Added print("PROGRESS: 0") before Popen to initialize progress at 0% for UI driving. Retained v1.15 count improvements and prior; certainty: 95% (simple addition; ensures bar starts empty—test for immediate output).
# v1.15 - Improved progress parse: Changed to if "progress" in line.lower() or "%" in line: to catch variations (case-insens, or percent-based). Added if count_mode and count==0 after: print warning, set count=80 (default). Retained v1.14 count file and prior; certainty: 80% (broadens detection; assumes progress has keywords—better with sample output; test for matches).
# v1.14 - Added STEP_COUNT_FILE = os.path.join(os.path.dirname(__file__), 'rs_step_count.txt'); if not exists, count_mode=True: run Popen, count lines with "Progress" in them (case-insens), at end save final_count to file. For normal, load total_steps from file (default 80 if missing). During run, percent = min(100, int((count / total_steps) * 100)) if total_steps >0 else 0; print "PROGRESS: {percent}". Retained v1.13 Popen/parse and prior; certainty: 85% (auto-counts on first run; assumes "Progress" per step—adjust parse if needed; test for accurate total).

import os
import shutil
import subprocess
import sys  # Added for CLI args (user_number)

# Main execution block: Parse user_number from CLI
# This section handles command-line input to determine the user ID for processing.
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python realityscan_processor.py <user_number>")
        sys.exit(1)
    
    try:
        user_number = int(sys.argv[1])  # Parse as int
    except ValueError:
        print("Error: user_number must be an integer.")
        sys.exit(1)
    
    PREFIX = f"user_{user_number}"  # Dynamic prefix based on user_number

# Set paths (adjust if needed)
# RS_PATH points to the RealityScan executable.
RS_PATH = r"C:\Program Files\Epic Games\RealityScan_2.0\RealityScan.exe"

# v1.15: Step count file (global in script dir)
STEP_COUNT_FILE = os.path.join(os.path.dirname(__file__), 'rs_step_count.txt')

# Dynamic user_dir and subdirs (photos for input, 3dmodel for output)
# user_dir is the main folder for this user, e.g., d:\photogrammetry\user_1
user_dir = os.path.join(r"d:\photogrammetry", PREFIX)
IMAGE_SOURCE_DIR = os.path.join(user_dir, "photos")  # Photos subfolder for input images (direct use)
TEMP_OUTPUT_DIR = os.path.join(user_dir, "temp_output")  # v1.10: Temp dir for RS output to isolate extras
PROJECT_NAME = PREFIX  # Project named after user prefix
PROJECT_FILE = os.path.join(TEMP_OUTPUT_DIR, f"{PROJECT_NAME}.rsproj")  # Project file in temp
OUTPUT_BASE = os.path.join(TEMP_OUTPUT_DIR, PROJECT_NAME)  # Base for exported .obj in temp
GENERATED_OBJ = f"{OUTPUT_BASE}.obj"  # Full path to generated OBJ file in temp

# Create temp output directory
# Ensures user_dir exists; creates TEMP_OUTPUT_DIR fresh for RS isolation.
os.makedirs(user_dir, exist_ok=True)
if os.path.exists(TEMP_OUTPUT_DIR):
    shutil.rmtree(TEMP_OUTPUT_DIR)
os.makedirs(TEMP_OUTPUT_DIR)

# Debug listing to verify images in source dir
print(f"Images in {IMAGE_SOURCE_DIR}:")
os.system(f'dir "{IMAGE_SOURCE_DIR}\\*.jpg"')  # Debug listing to verify files

# Check if images exist in source dir
# Warns if no images found in photos dir, pauses for user input.
if not any(f.endswith(".jpg") for f in os.listdir(IMAGE_SOURCE_DIR)):
    print(f"Warning: No .jpg files found in {IMAGE_SOURCE_DIR}.")
    input("Press Enter to continue...")

# v1.15: Load total_steps or set count_mode if missing (default 80)
count_mode = not os.path.exists(STEP_COUNT_FILE)
if not count_mode:
    try:
        with open(STEP_COUNT_FILE, 'r') as f:
            total_steps = int(f.read().strip())
    except:
        print(f"Error loading {STEP_COUNT_FILE}; using default 80.")
        total_steps = 80
else:
    total_steps = 80  # Placeholder during count
    print(f"First run: Counting steps; will save to {STEP_COUNT_FILE}.")

# First RealityScan call (no-XMP run, copied from bat)
# Constructs the command to process images in RealityScan, exporting to temp.
command = [
    RS_PATH,
    "-newScene",
    "-stdConsole",  # v1.12: Enable console redirection for standard output
    "-printProgress",  # v1.12: Print progress changes to CMD
    "-addFolder", IMAGE_SOURCE_DIR,  # v1.11: Direct use of photos dir
    "-generateAIMasks",
    "-align",
    "-setReconstructionRegionAuto",
    "-set", "mvsNormalDownscaleFactor=4",
    "-set", "mvsDefaultGroupingFactor=2",
    "-calculateNormalModel",
    "-selectMaximalComponent",
    "-cleanModel",
    "-set", "unwrapMaxTexResolution=4096",
    "-set", "txtImageDownscaleTexture=2",
    "-calculateTexture",
    "-exportSelectedModel", f"{OUTPUT_BASE}.obj",
    "-save", PROJECT_FILE,
    "-quit"
]

print(f"Running RealityScan command: {' '.join(command)}")

# v1.16: Initialize progress at 0% for UI
print("PROGRESS: 0")

# v1.15: Use Popen to capture output real-time
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
count = 0

# Read output lines, count progress, compute/print percent
while process.poll() is None:
    line = process.stdout.readline().strip()
    if line:
        print(line)  # Echo output
        if "progress" in line.lower() or "%" in line:  # Detect progress lines (case-insensitive or %)
            count += 1
            if total_steps > 0:
                percent = min(100, int((count / total_steps) * 100))
            else:
                percent = 0
            print(f"PROGRESS: {percent}")

# After completion, check return code
result = process.wait()
if result == 0:
    print("PROGRESS: 100")  # Final 100% on success
    # v1.15: If count_mode and count==0, warn and set to default
    if count_mode:
        if count == 0:
            print("Warning: No progress steps detected; using default 80.")
            count = 80
        with open(STEP_COUNT_FILE, 'w') as f:
            f.write(str(count))
        print(f"Saved step count {count} to {STEP_COUNT_FILE}.")
    print(f"Process complete. Generated model: {GENERATED_OBJ}")
    
    # Step 1: Create subfolder named '3dmodel'
    # Creates or clears the output subfolder for final model files.
    subfolder = os.path.join(user_dir, "3dmodel")
    os.makedirs(subfolder, exist_ok=True)
    
    # Clean any existing files in subfolder to remove old leftovers
    # Removes any prior files or subdirs in 3dmodel to start clean.
    for item in os.listdir(subfolder):
        item_path = os.path.join(subfolder, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
            print(f"Cleaned old file from subfolder: {item}")
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f"Cleaned old subdir from subfolder: {item}")
    
    # Step 2: Copy .obj, .mtl, .png files starting with prefix from TEMP_OUTPUT_DIR to subfolder
    # Selectively copies model files to 3dmodel subfolder (assumes files in TEMP_OUTPUT_DIR root).
    for file in os.listdir(TEMP_OUTPUT_DIR):
        if file.startswith(PREFIX) and file.endswith(('.obj', '.mtl', '.png')):
            src = os.path.join(TEMP_OUTPUT_DIR, file)
            dst = os.path.join(subfolder, file)
            shutil.copy(src, dst)
            print(f"Copied {file} to {subfolder}")
    
    # Step 3: Cleanup - remove TEMP_OUTPUT_DIR entirely (deletes all extras including potential subdirs)
    # Removes the entire temp output dir to eliminate any leftovers from RS.
    if os.path.exists(TEMP_OUTPUT_DIR):
        shutil.rmtree(TEMP_OUTPUT_DIR)
        print(f"Deleted temp output directory: {TEMP_OUTPUT_DIR}")
else:
    print(f"Error occurred during RealityScan execution. Return code: {result}")