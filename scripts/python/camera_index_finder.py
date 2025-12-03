# camera_index_finder.py
# Version: 1.6
# Change Log:
# v1.6 - Removed test frame read: Dropped v1.4's debug read per index to reduce overhead/multi-cam conflicts; retained sorted/filtered logic and InstanceId print for mismatch debug.
# v1.5 - Arducam filter: Updated ps_command to filter InstanceId -like "*VID_04B4&PID_0471*" to exclude virtual/non-Arducam cams, ensuring len match with OpenCV physical detections. Retained sorted by InstanceId and test frame debug.
# v1.4 - Enhanced mismatch debug: Added test frame read after detection to verify each index (logs if blank/all-white); returned sorted devices with InstanceId in match print. This helps correlate with grabphoto logs for manual offset adjustment if needed.

import subprocess
import json
import cv2

def get_camera_friendly_names():
    """
    Get camera friendly names using PowerShell on Windows.
    Runs PowerShell command to fetch active camera devices with FriendlyName and InstanceId.
    Returns list of dicts or empty list on error.
    v1.5: Filters to Arducam-specific VID/PID in InstanceId to exclude virtual cams.
    """
    try:
        # PowerShell command to get camera devices (filtered to Arducam)
        ps_command = """
        Get-PnpDevice -Class Camera | 
        Where-Object {$_.InstanceId -like "*VID_04B4&PID_0471*"} | 
        Where-Object {$_.Status -eq "OK"} | 
        Select-Object FriendlyName, InstanceId | 
        ConvertTo-Json
        """
        
        result = subprocess.run(
            ["powershell", "-Command", ps_command],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and result.stdout:
            devices = json.loads(result.stdout)
            
            # Handle single device (not in array)
            if isinstance(devices, dict):
                devices = [devices]
            
            return devices
        else:
            print(f"PowerShell error: {result.stderr}", flush=True)
            return []
            
    except Exception as e:
        print(f"Error getting camera names: {e}", flush=True)
        return []

def get_camera_index_by_name(target_name="Center_Camera"):
    """
    Find OpenCV camera index by friendly name.
    Queries devices via PowerShell, detects available OpenCV indices,
    matches target name (case-insensitive partial match), and returns index if found.
    Assumes enumeration order matches between PnP devices and OpenCV.
    Returns None if not found or no cameras available.
    v1.6: Removed test frame read for reduced overhead.
    v1.5: Aligns with filtered Arducam devices to match len and order.
    v1.4: Added test frame read per index (logs if opened and if frame is all-white/blank for debug); includes InstanceId in match print.
    v1.3: Sorts devices by 'InstanceId' to better match potential hardware enumeration order in OpenCV.
    v1.2: Reverted to default backend (no CAP_DSHOW) for compatibility.
    """
    # Get all available cameras with their friendly names (filtered)
    camera_devices = get_camera_friendly_names()
    
    if not camera_devices:
        print("No Arducam cameras found or unable to retrieve camera information.", flush=True)
        return None
    
    # v1.3: Sort by InstanceId (device path) to align order with OpenCV enumeration
    camera_devices = sorted(camera_devices, key=lambda d: d.get('InstanceId', ''))
    
    print("Available Arducam cameras from PowerShell (sorted by InstanceId):", flush=True)
    for i, device in enumerate(camera_devices):
        print(f"  {i}: {device.get('FriendlyName', 'Unknown')} (InstanceId: {device.get('InstanceId', 'N/A')})", flush=True)
    
    # Detect available OpenCV camera indices (up to 30 for safety with extra devices)
    max_cameras = 30
    available_indices = []
    
    for idx in range(max_cameras):
        cap = cv2.VideoCapture(idx)  # Default backend
        if cap.isOpened():
            available_indices.append(idx)
            cap.release()
    
    print(f"OpenCV (default) detected cameras at indices: {available_indices}", flush=True)
    
    # Match target name to sorted device list and map to OpenCV index (order-based)
    for i, device in enumerate(camera_devices):
        friendly_name = device.get('FriendlyName', '')
        if target_name.lower() in friendly_name.lower():
            if i < len(available_indices):
                matched_index = available_indices[i]
                print(f"Found '{target_name}' at index {matched_index} (full name: {friendly_name}, InstanceId: {device.get('InstanceId', 'N/A')})", flush=True)
                return matched_index
    
    print(f"Camera with name '{target_name}' not found.", flush=True)
    return None

# Standalone test (for dev/debug; grabphoto will import without running this)
if __name__ == "__main__":
    index = get_camera_index_by_name()
    if index is not None:
        print(f"Center_Camera index: {index}", flush=True)
    else:
        print("Center_Camera not found.", flush=True)