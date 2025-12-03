import os
import shutil
import subprocess
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import PATHS

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python realityscan_processor.py <user_number>")
        sys.exit(1)
    
    try:
        user_number = int(sys.argv[1])
    except ValueError:
        print("Error: user_number must be an integer.")
        sys.exit(1)
    
    PREFIX = f"user_{user_number}"

RS_PATH = r"C:\Program Files\Epic Games\RealityScan_2.0\RealityScan.exe"

STEP_COUNT_FILE = os.path.join(PATHS['SCRIPTS_PYTHON'], 'rs_step_count.txt')

user_dir = os.path.join(PATHS['BASE'], PREFIX)
IMAGE_SOURCE_DIR = os.path.join(user_dir, "photos")
TEMP_OUTPUT_DIR = os.path.join(user_dir, "temp_output")
PROJECT_NAME = PREFIX
PROJECT_FILE = os.path.join(TEMP_OUTPUT_DIR, f"{PROJECT_NAME}.rsproj")
OUTPUT_BASE = os.path.join(TEMP_OUTPUT_DIR, PROJECT_NAME)
GENERATED_OBJ = f"{OUTPUT_BASE}.obj"

os.makedirs(user_dir, exist_ok=True)
if os.path.exists(TEMP_OUTPUT_DIR):
    shutil.rmtree(TEMP_OUTPUT_DIR)
os.makedirs(TEMP_OUTPUT_DIR)

print(f"Images in {IMAGE_SOURCE_DIR}:")
os.system(f'dir "{IMAGE_SOURCE_DIR}\\*.jpg"')

if not any(f.endswith(".jpg") for f in os.listdir(IMAGE_SOURCE_DIR)):
    print(f"Warning: No .jpg files found in {IMAGE_SOURCE_DIR}.")
    input("Press Enter to continue...")

count_mode = not os.path.exists(STEP_COUNT_FILE)
if not count_mode:
    try:
        with open(STEP_COUNT_FILE, 'r') as f:
            total_steps = int(f.read().strip())
    except:
        print(f"Error loading {STEP_COUNT_FILE}; using default 80.")
        total_steps = 80
else:
    total_steps = 80
    print(f"First run: Counting steps; will save to {STEP_COUNT_FILE}.")

command = [
    RS_PATH,
    "-newScene",
    "-stdConsole",
    "-printProgress",
    "-addFolder", IMAGE_SOURCE_DIR,
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

print("PROGRESS: 0")

process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
count = 0

while process.poll() is None:
    line = process.stdout.readline().strip()
    if line:
        print(line)
        if "progress" in line.lower() or "%" in line:
            count += 1
            if total_steps > 0:
                percent = min(100, int((count / total_steps) * 100))
            else:
                percent = 0
            print(f"PROGRESS: {percent}")

result = process.wait()
if result == 0:
    print("PROGRESS: 100")
    if count_mode:
        if count == 0:
            print("Warning: No progress steps detected; using default 80.")
            count = 80
        with open(STEP_COUNT_FILE, 'w') as f:
            f.write(str(count))
        print(f"Saved step count {count} to {STEP_COUNT_FILE}.")
    print(f"Process complete. Generated model: {GENERATED_OBJ}")
    
    subfolder = os.path.join(user_dir, "3dmodel")
    os.makedirs(subfolder, exist_ok=True)
    
    for item in os.listdir(subfolder):
        item_path = os.path.join(subfolder, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
            print(f"Cleaned old file from subfolder: {item}")
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f"Cleaned old subdir from subfolder: {item}")
    
    for file in os.listdir(TEMP_OUTPUT_DIR):
        if file.startswith(PREFIX) and file.endswith(('.obj', '.mtl', '.png')):
            src = os.path.join(TEMP_OUTPUT_DIR, file)
            dst = os.path.join(subfolder, file)
            shutil.copy(src, dst)
            print(f"Copied {file} to {subfolder}")
    
    if os.path.exists(TEMP_OUTPUT_DIR):
        shutil.rmtree(TEMP_OUTPUT_DIR)
        print(f"Deleted temp output directory: {TEMP_OUTPUT_DIR}")
else:
    print(f"Error occurred during RealityScan execution. Return code: {result}")