# realityscan_processor.py
# Version: 1.32
# Changes:
# - v1.32 (2025-12-12): Added retry logic in _run_realityscan: Up to 3 attempts on non-zero return code, with 5s delay between retries. Logs retry attempts. Proceeds only on success.
# - v1.31 (2025-12-11): Enhanced CLI mode with debug prints: args parsing, user_dir resolution, processor init, thread start, and loop monitoring. Prints thread status every 5s in CLI wait loop.
# - v1.30 (2025-12-11): Added extensive debug prints throughout: path resolutions, command prep, process monitoring, result handling, and Blender checks. Logs key variables and steps for troubleshooting.

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
        print(f"DEBUG: Initializing RealityScanProcessor for user_dir: {user_dir}")
        self.user_dir = Path(user_dir)
        self.prefix = self.user_dir.name  # e.g., "user_1"
        self.photos_dir = self.user_dir / "photos"
        self.output_dir = self.user_dir / "3dmodel"
        self.generated_obj = self.output_dir / f"{self.prefix}.obj"
        self.progress_queue = progress_queue  # For real-time UI updates
        self.step_count_file = Path(PATHS['SCRIPTS_PYTHON']) / 'rs_step_count.txt'
        self.rs_path = r"C:\Program Files\Epic Games\RealityScan_2.0\RealityScan.exe"

        # Blender paths
        self.blender_scripts_dir = Path(PATHS['SCRIPTS_PYTHON']) / "blender_scripts"
        self.base_blend = self.blender_scripts_dir / "base.blend"
        self.mesh_fitter_script = self.blender_scripts_dir / "mesh_fitter.py"

        # Debug log for Blender
        self.blender_log = self.user_dir / "blender_debug.log"

        print(f"DEBUG: Resolved paths - photos_dir: {self.photos_dir}, output_dir: {self.output_dir}, generated_obj: {self.generated_obj}")
        print(f"DEBUG: Blender paths - scripts_dir: {self.blender_scripts_dir}, base_blend: {self.base_blend}, mesh_fitter: {self.mesh_fitter_script}, log: {self.blender_log}")

        # Ensure dirs
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self.photos_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        print("DEBUG: Directories ensured.")

    def start_photogrammetry(self):
        print("DEBUG: Starting photogrammetry thread.")
        threading.Thread(target=self._run_realityscan, daemon=True).start()

    def _run_realityscan(self):
        print("DEBUG: Entering _run_realityscan.")
        count_mode = not self.step_count_file.exists()
        if not count_mode:
            try:
                with open(self.step_count_file, 'r') as f:
                    total_steps = int(f.read().strip())
                print(f"DEBUG: Loaded total_steps from file: {total_steps}")
            except Exception as e:
                print(f"DEBUG: Error loading step_count_file: {e}. Using default 156.")
                total_steps = 156
        else:
            total_steps = 156
            print(f"DEBUG: Count mode active - total_steps default: {total_steps}")

        # Clean old files in output_dir
        print("DEBUG: Cleaning output_dir.")
        for item in self.output_dir.iterdir():
            if item.is_file():
                item.unlink()
                print(f"DEBUG: Removed file: {item}")
            elif item.is_dir():
                shutil.rmtree(item)
                print(f"DEBUG: Removed dir: {item}")

        rsbox_path = os.path.join(PATHS['BASE'], "reconstructionregion.rsbox")
        print(f"DEBUG: rsbox_path: {rsbox_path}")

        command = [
            self.rs_path,
            "-newScene",
            "-stdConsole",
            "-printProgress",
            "-addFolder", str(self.photos_dir),
            "-align",
            "-setReconstructionRegion", rsbox_path,
            "-set", "mvsNormalDownscaleFactor=4",
            "-set", "mvsDefaultGroupingFactor=2",
            "-calculateNormalModel",
            "-set", "unwrapMaxTexResolution=4096",
            "-set", "txtImageDownscaleTexture=2",
            "-calculateTexture",
            "-exportSelectedModel", str(self.generated_obj),
            "-quit"
        ]
        print(f"DEBUG: RealityScan command: {' '.join(command)}")

        max_retries = 3  # Configurable retry limit
        for attempt in range(1, max_retries + 1):
            if self.progress_queue:
                self.progress_queue.put(0)
                print("DEBUG: Sent initial progress: 0")

            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(f"DEBUG: Started RealityScan process (PID: {process.pid}) - Attempt {attempt}/{max_retries}")
            count = 0
            last_percent = 0

            while process.poll() is None:
                line = process.stdout.readline().strip()
                if line:
                    if "[W:onnxruntime:" not in line:
                        pass
                    if "#progress" in line.lower():
                        count += 1
                        if total_steps > 0:
                            percent = min(100, int((count / total_steps) * 100))
                        else:
                            percent = 0
                        print(f"DEBUG: Progress update - count: {count}, percent: {percent}")
                        if self.progress_queue:
                            self.progress_queue.put(percent)
                        last_percent = percent

            if last_percent < 100:
                if self.progress_queue:
                    self.progress_queue.put(100)
                print("DEBUG: Forced final progress to 100")

            result = process.wait()
            print(f"DEBUG: RealityScan exited with code: {result}")

            if result == 0:
                print(f"DEBUG: RealityScan succeeded on attempt {attempt}")
                break  # Success - exit retry loop
            else:
                print(f"Error: RealityScan failed with return code {result} on attempt {attempt}")
                if self.progress_queue:
                    self.progress_queue.put(-1)
                if attempt < max_retries:
                    print(f"DEBUG: Retrying in 5 seconds...")
                    time.sleep(5)  # Delay before retry
                else:
                    print(f"DEBUG: Max retries ({max_retries}) reached. Aborting.")
                    return  # Or raise an exception if preferred

        # Proceed only on success
        if result == 0:
            if self.progress_queue:
                self.progress_queue.put(100)

            if count_mode and count > 0:
                with open(self.step_count_file, 'w') as f:
                    f.write(str(count))
                print(f"DEBUG: Saved step count: {count} to {self.step_count_file}")

            print(f"DEBUG: RealityScan success: {self.generated_obj}")

            # ———————————————————————
            # Launch Blender Mesh Fitter with Debug
            # ———————————————————————
            print("DEBUG: Checking for mesh fitting launch...")
            if not self.generated_obj.exists():
                print(f"DEBUG: Skipping Blender - .obj not found: {self.generated_obj}")
            elif not self.base_blend.exists():
                print(f"DEBUG: Skipping Blender - base.blend not found: {self.base_blend}")
            elif not self.mesh_fitter_script.exists():
                print(f"DEBUG: Skipping Blender - mesh_fitter.py not found: {self.mesh_fitter_script}")
            else:
                blender_cmd = [
                    "blender",
                    "--background",
                    str(self.base_blend),
                    "--python", str(self.mesh_fitter_script),
                    "--",
                    str(self.user_dir)
                ]
                print(f"DEBUG: Blender command: {' '.join(blender_cmd)}")
                print(f"DEBUG: Launching Blender... Output will log to {self.blender_log}")

                try:
                    # Launch with stdout/stderr redirect for debug
                    with open(self.blender_log, 'w') as log_file:
                        proc = subprocess.Popen(
                            blender_cmd,
                            stdout=log_file,
                            stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
                        )
                    print(f"DEBUG: Blender process started (PID: {proc.pid}). Check {self.blender_log} for output.")
                except Exception as e:
                    print(f"DEBUG: Failed to launch Blender: {e}")

        else:
            print(f"Error: RealityScan failed with return code {result}")
            if self.progress_queue:
                self.progress_queue.put(-1)


# CLI mode for backward compatibility
if __name__ == "__main__":
    print("DEBUG: Entering CLI mode.")
    print(f"DEBUG: CLI args: {sys.argv}")
    if len(sys.argv) != 2:
        print("Usage: python realityscan_processor.py <user_number>")
        sys.exit(1)
    
    try:
        user_number = int(sys.argv[1])
        print(f"DEBUG: Parsed user_number: {user_number}")
    except ValueError:
        print("Error: user_number must be an integer.")
        sys.exit(1)
    
    user_dir = os.path.join(PATHS['BASE'], f"user_{user_number}")
    print(f"DEBUG: CLI user_dir: {user_dir}")
    processor = RealityScanProcessor(user_dir)
    processor.start_photogrammetry()
    
    print("Processing started. Close this window only after both steps complete.")
    while threading.active_count() > 1:
        print(f"DEBUG: Threads active: {threading.active_count()}. Waiting...")
        time.sleep(5)