# usb_reset.py
# Version: 1.2
# Change Log:
# v1.2 - Added self-elevation for admin privileges: Imported ctypes and sys. In main(), check if running as admin via ctypes.windll.shell32.IsUserAnAdmin(); if not, relaunch self with admin using ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, __file__, None, 1). This automates elevation without manual intervention (exits original process after launch). Retained v1.1 delay and prior logic; no other changes. Certainty: 90% (standard Windows self-elevate pattern per StackOverflow/MS docs; assumes script called via python.exe—test for UAC prompt/compatibility; if denied, fails silently).
# v1.1 - Added 20 second delay between each port reset: Imported time, added time.sleep(20) after each subprocess.call in the loop (allows reset to propagate before next; sleeps after last but harmless as script ends). Retained v1.0 logic; no other changes. Certainty: 95% (assumes delay needed for USB enumeration/stability post-reset; adjust if too long/short via testing—20s per user request).
# v1.0 - Initial version: Defines PORT_CHAINS list at top for easy updates. Locates restartusbport.exe in utilities subdir. Sequentially calls exe for each chain via subprocess.call (blocking for safety, ensures each reset completes). Prints status for each call with return code. Certainty: 95% (assumes exe takes single chain arg like '1-19'; test with actual exe for args/sequencing—parallel if non-dependent).

import os  # For path handling (dirname, join, exists)
import subprocess  # For calling restartusbport.exe
import time  # v1.1: For delay between resets
import ctypes  # v1.2: For admin check and elevation
import sys  # v1.2: For sys.executable and __file__

# List of port chains to reset (update here as needed; format for exe arg)
PORT_CHAINS = ['1-19', '1-20', '2-3', '2-4', '1-25']

def main():
    """
    Main entry: Checks/runs as admin (self-elevates if not), locates exe, calls for each PORT_CHAINS sequentially with 20s delay after each.
    Prints status; exits on missing exe.
    """
    # v1.2: Check if running as admin; if not, relaunch with elevation
    if not ctypes.windll.shell32.IsUserAnAdmin():
        # Relaunch as admin (UAC prompt; exits current process)
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, __file__, None, 1)
        return  # Exit original non-admin instance
    
    # Get exe path: utilities dir relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    utilities_dir = os.path.join(script_dir, 'utilities')
    exe_path = os.path.join(utilities_dir, 'restartusbport.exe')
    
    # Check if exe exists
    if not os.path.exists(exe_path):
        print(f"Error: restartusbport.exe not found at {exe_path}")
        return  # Exit without reset
    
    # Call exe for each chain (blocking; assumes sequential safe)
    for chain in PORT_CHAINS:
        print(f"Resetting port chain: {chain}")
        try:
            ret = subprocess.call([exe_path, chain])  # Call with chain as arg
            print(f"Reset complete for {chain} (return code: {ret})")
        except Exception as e:
            print(f"Error resetting {chain}: {e}")
        
        # v1.1: Delay after reset to allow propagation (20s per user request)
        time.sleep(20)

if __name__ == "__main__":
    main()