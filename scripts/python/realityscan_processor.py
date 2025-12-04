# realityscan_processor.py
# Version: 1.19
# Changes:
# - v1.19: Added process.wait() to ensure RealityScan exits fully after each capture (prevents lingering processes).
#          CLI commands unchanged — same as baseline.
# - v1.18: Baseline from provided document (class with progress_queue).

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
        self.temp_output_dir = self.user_dir / "temp_output"
        self.project_file = self.temp_output_dir / f"{self.prefix}.rsproj"
        self.output_base = self.temp_output_dir / self.prefix
        self.generated_obj = f"{self.output_base}.obj"
        self.progress_queue = progress_queue  # NEW: For real-time UI updates
        self.step_count_file = Path(PATHS['SCRIPTS_PYTHON']) / 'rs_step_count.txt'
        self.rs_path = r"C:\Program Files\Epic Games\RealityScan_2.0\RealityScan.exe"

        # Ensure dirs
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self.photos_dir.mkdir(exist_ok=True)
        if self.temp_output_dir.exists():
            shutil.rmtree(self.temp_output_dir)
        self.temp_output_dir.mkdir(exist_ok=True)

    def start_photogrammetry(self):
        threading.Thread(target=self._run_realityscan, daemon=True).start()

    def _run_realityscan(self):
        print(f"Images in {self.photos_dir}:")
        os.system(f'dir "{self.photos_dir}\\*.jpg"')

        if not any(f.endswith(".jpg") for f in os.listdir(self.photos_dir)):
            print(f"Warning: No .jpg files found in {self.photos_dir}. Proceeding anyway...")

        count_mode = not self.step_count_file.exists()
        if not count_mode:
            try:
                with open(self.step_count_file, 'r') as f:
                    total_steps = int(f.read().strip())
            except:
                print(f"Error loading {self.step_count_file}; using default 80.")
                total_steps = 80
        else:
            total_steps = 80
            print(f"First run: Counting steps; will save to {self.step_count_file}.")

        command = [
            self.rs_path,
            "-newScene",
            "-stdConsole",
            "-printProgress",
            "-addFolder", str(self.photos_dir),
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
            "-exportSelectedModel", self.generated_obj,
            "-save", str(self.project_file),
            "-quit"
        ]

        print(f"Running RealityScan command: {' '.join(command)}")

        if self.progress_queue:
            self.progress_queue.put(0)
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
                    if self.progress_queue:
                        self.progress_queue.put(percent)

        result = process.wait()  # Ensure exit (fixed)

        if result == 0:
            if self.progress_queue:
                self.progress_queue.put(100)
            print("PROGRESS: 100")
            if count_mode:
                if count == 0:
                    print("Warning: No progress steps detected; using default 80.")
                    count = 80
                with open(self.step_count_file, 'w') as f:
                    f.write(str(count))
                print(f"Saved step count {count} to {self.step_count_file}.")
            print(f"Process complete. Generated model: {self.generated_obj}")
            
            subfolder = self.user_dir / "3dmodel"
            subfolder.mkdir(exist_ok=True)
            
            for item in subfolder.iterdir():
                if item.is_file():
                    item.unlink()
                    print(f"Cleaned old file from subfolder: {item.name}")
                elif item.is_dir():
                    shutil.rmtree(item)
                    print(f"Cleaned old subdir from subfolder: {item.name}")
            
            for file in self.temp_output_dir.iterdir():
                if file.name.startswith(self.prefix) and file.suffix in {'.obj', '.mtl', '.png'}:
                    shutil.copy(file, subfolder / file.name)
                    print(f"Copied {file.name} to {subfolder}")
            
            if self.temp_output_dir.exists():
                shutil.rmtree(self.temp_output_dir)
                print(f"Deleted temp output directory: {self.temp_output_dir}")
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