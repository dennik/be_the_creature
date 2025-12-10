# xmp_generator.py
# Version: 1.0
# Changes:
# - v1.0 (2025-12-10): Initial version. Generates XMP files for each captured image in photos_dir, using center_camera.xmp content for the preview camera image and other_cameras.xmp for others. Takes photos_dir and preview_index as arguments.
# Integration Note: Call this script right after image capture in grabphoto_control.py's capture_photos() function, before returning user_id. Example: subprocess.call(["python", "xmp_generator.py", str(photos_dir), str(best_index)]). Ensure best_index is accessible (e.g., return it from initialize_cameras and pass via globals or parameters).

import sys
import os
from pathlib import Path

# Hardcoded XMP contents from provided documents
CENTER_XMP = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#" xcr:Version="3"
       xcr:PosePrior="locked" xcr:Rotation="1 0 0 0 1 0 0 0 1" xcr:Position="0 0 0"
       xcr:Coordinates="absolute" xcr:DistortionModel="brown3" xcr:FocalLength35mm="33.2077297883518"
       xcr:Skew="0" xcr:AspectRatio="1" xcr:PrincipalPointU="-0.0133535273465855"
       xcr:PrincipalPointV="0.00185610175550173" xcr:CalibrationPrior="exact"
       xcr:CalibrationGroup="-1" xcr:DistortionGroup="-1" xcr:InTexturing="1"
       xcr:InMeshing="1">
      <xcr:DistortionCoeficients>0.249837427700882 -1.34632315213518 2.46705801609365 0 0 0</xcr:DistortionCoeficients>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""

OTHER_XMP = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xcr:Version="3" xcr:PosePrior="initial" xcr:Coordinates="absolute"
       xcr:DistortionModel="brown3" xcr:FocalLength35mm="33.1151127887442"
       xcr:Skew="0" xcr:AspectRatio="1" xcr:PrincipalPointU="-0.00554807087682733"
       xcr:PrincipalPointV="-0.00501429098610042" xcr:CalibrationPrior="initial"
       xcr:CalibrationGroup="-1" xcr:DistortionGroup="-1" xcr:InTexturing="1"
       xcr:InMeshing="1" xcr:latitude="179.998132327514924N" xcr:longitude="107.700327905279551E"
       xcr:version="2.2.0.0" xcr:altitude="643089440/10000" xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#">
      <xcr:Rotation>-0.92680322160703 -0.0395768153323533 0.373456107336564 -0.149519822352645 -0.873330868464748 -0.463613003387785 0.344499072774462 -0.485517115944184 0.803488344024217</xcr:Rotation>
      <xcr:Position>-1.71824608388901 5.38387013335462 1.39163328536584</xcr:Position>
      <xcr:DistortionCoeficients>0.267307715954277 -1.29801586768879 1.92025415106341 0 0 0</xcr:DistortionCoeficients>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""

def generate_xmp_files(photos_dir: str, preview_index: int):
    photos_dir = Path(photos_dir)
    if not photos_dir.exists() or not photos_dir.is_dir():
        print(f"Error: Photos directory {photos_dir} does not exist or is not a directory.")
        return

    jpg_files = list(photos_dir.glob("*.jpg"))
    if not jpg_files:
        print(f"Warning: No .jpg files found in {photos_dir}.")
        return

    for jpg_path in jpg_files:
        filename = jpg_path.name
        try:
            # Extract camera index from filename, e.g., user_1_camera_0_8MP_timestamp.jpg → 0
            cam_idx_str = filename.split('_camera_')[1].split('_')[0]
            cam_idx = int(cam_idx_str)
        except (IndexError, ValueError):
            print(f"Warning: Could not parse camera index from {filename}. Skipping.")
            continue

        # Determine which XMP content to use
        xmp_content = CENTER_XMP if cam_idx == preview_index else OTHER_XMP

        # Generate XMP filename: replace .jpg with .xmp
        xmp_path = jpg_path.with_suffix('.xmp')

        # Write the XMP file
        with open(xmp_path, 'w', encoding='utf-8') as f:
            f.write(xmp_content)

        print(f"Generated {xmp_path.name} for {filename} (using {'center' if cam_idx == preview_index else 'other'} XMP).")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python xmp_generator.py <photos_dir> <preview_index>")
        sys.exit(1)

    photos_dir_arg = sys.argv[1]
    try:
        preview_index_arg = int(sys.argv[2])
    except ValueError:
        print("Error: preview_index must be an integer.")
        sys.exit(1)

    generate_xmp_files(photos_dir_arg, preview_index_arg)