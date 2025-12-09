# realityscan_processor.py
# Version: 1.26
# Changes:
# - v1.26 (2025-12-09): Added -setReconstructionRegionAuto after -align to tighten reconstruction box around user. Commented out debug/console prints during session (retained queue.put and essential errors).
# - v1.25 (2025-12-09): Simplified CLI commands per user request: Removed -generateAIMasks, -setReconstructionRegionAuto, -selectMaximalComponent, -cleanModel, -save. Direct export to 3dmodel (no temp_output). Moved pre-clean to before Popen. This speeds up processing by skipping non-essential steps.
# - v1.24 (2025-12-04): To ensure progress always reaches 100 before completion (for visual purposes, even if count < total_steps), added a check after the loop: if current percent <100, explicitly put(100) before final put(100). Retained existing final put(100) for redundancy.

import os
import shutil
import subprocess
import sys
import threading
import time
import queue
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import PATHS

class RealityScanProcessor:
    def __init__(self, user_dir: str, progress_queue: queue.Queue = None):
        self.user_dir = Path(user_dir)
        self.prefix = self.user_dir.name  # e.g., "user_1"
        self.photos_dir = self.user_dir / "photos"
        self.output_dir = self.user_dir / "3dmodel"
        self.generated_obj = self.output_dir / f"{self.prefix}.obj"
        self.progress_queue = progress_queue  # NEW: For real-time UI updates
        self.step_count_file = Path(PATHS['SCRIPTS_PYTHON']) / 'rs_step_count.txt'
        self.rs_path = r"C:\Program Files\Epic Games\RealityScan_2.0\RealityScan.exe"

        # Ensure dirs
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self.photos_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

    def start_photogrammetry(self):
        threading.Thread(target=self._run_realityscan, daemon=True).start()

    def _run_realityscan(self):
        # print(f"Images in {self.photos_dir}:")  # Commented out debug print
        # os.system(f'dir "{self.photos_dir}\\*.jpg"')  # Commented out debug print

        # if not any(f.endswith(".jpg") for f in os.listdir(self.photos_dir)):
        #     print(f"Warning: No .jpg files found in {self.photos_dir}. Proceeding anyway...")  # Commented out debug print

        count_mode = not self.step_count_file.exists()
        if not count_mode:
            try:
                with open(self.step_count_file, 'r') as f:
                    total_steps = int(f.read().strip())
            except:
                # print(f"Error loading {self.step_count_file}; using known total 156.")  # Commented out debug print
                total_steps = 156
        else:
            total_steps = 156
            # print(f"First run: Counting #progress reports; will save to {self.step_count_file} (expected ~156).")  # Commented out debug print

        # Clean old files in output_dir
        for item in self.output_dir.iterdir():
            if item.is_file():
                item.unlink()
                # print(f"Cleaned old file from output_dir: {item.name}")  # Commented out debug print
            elif item.is_dir():
                shutil.rmtree(item)
                # print(f"Cleaned old subdir from output_dir: {item.name}")  # Commented out debug print

        command = [
            self.rs_path,
            "-newScene",
            "-stdConsole",
            "-printProgress",
            "-addFolder", str(self.photos_dir),
            "-align",
            "-setReconstructionRegionAuto",
            "-set", "mvsNormalDownscaleFactor=4",
            "-set", "mvsDefaultGroupingFactor=2",
            "-calculateNormalModel",
            "-set", "unwrapMaxTexResolution=4096",
            "-set", "txtImageDownscaleTexture=2",
            "-calculateTexture",
            "-exportSelectedModel", str(self.generated_obj),
            "-quit"
        ]

        # print(f"Running RealityScan command: {' '.join(command)}")  # Commented out debug print

        if self.progress_queue:
            self.progress_queue.put(0)
        # print("PROGRESS: 0")  # Commented out debug print

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        count = 0
        last_percent = 0

        while process.poll() is None:
            line = process.stdout.readline().strip()
            if line:
                # Filter out ONNX Runtime warnings to prevent CMD flooding
                if "[W:onnxruntime:" not in line:
                    # print(line)  # Commented out debug print
                    pass
                if "#progress" in line.lower():
                    count += 1
                    if total_steps > 0:
                        percent = min(100, int((count / total_steps) * 100))
                    else:
                        percent = 0
                    # print(f"PROGRESS: {percent}")  # Commented out debug print
                    if self.progress_queue:
                        self.progress_queue.put(percent)
                    last_percent = percent

        # Force to 100 if not already (visual fix)
        if last_percent < 100:
            if self.progress_queue:
                self.progress_queue.put(100)
            # print("PROGRESS: 100 (forced for completion)")  # Commented out debug print

        result = process.wait()  # Ensure exit (fixed)

        if result == 0:
            if self.progress_queue:
                self.progress_queue.put(100)
            # print("PROGRESS: 100")  # Commented out debug print
            if count_mode:
                if count == 0:
                    # print("Warning: No #progress reports detected; using known total 156.")  # Commented out debug print
                    count = 156
                with open(self.step_count_file, 'w') as f:
                    f.write(str(count))
                # print(f"Saved #progress count {count} to {self.step_count_file}.")  # Commented out debug print
            # print(f"Process complete. Generated model: {self.generated_obj}")  # Commented out debug print
        else:
            print(f"Error occurred during RealityScan execution. Return code: {result}")

# CLI mode for backward compatibility
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python realityscan_processor.py <user_number>")
        sys.exit(1)
    
    try:
        user_number = int(sys.argv[1])
    except ValueError:
        print("Error: user_number must be an integer.")
        sys.exit(1)
    
    user_dir = os.path.join(PATHS['BASE'], f"user_{user_number}")
    processor = RealityScanProcessor(user_dir)
    processor.start_photogrammetry()
    while threading.active_count() > 1:
        time.sleep(0.1)