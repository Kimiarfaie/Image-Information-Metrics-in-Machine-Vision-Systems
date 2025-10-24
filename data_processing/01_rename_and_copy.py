import os
import subprocess
import json
import shutil

"""
    Script 01:
    Rename and copy captured images into organized RAW and JPG folders.
    
    This script will:
    - Extract metadata (FocalLength, ISO, EV) from image files using ExifTool.
    - Rename each file using the pattern:
        <object>_<focal>_<ISO>ISO_<EV>EV_<Dist>_<ISP>_<Focus>.<ext>
    - Copy renamed files to:
        RAW/  → for .CR3 files
        JPG/  → for .JPG files

    Folder Mapping:
        Each camera folder (e.g., 410CANON) corresponds to a specific capture configuration.
        The mapping below defines:
            Dist  → capture distance category (e.g., Dist1 or Dist2)
            ISP   → whether in-camera image processing was ON or OFF
      
            Focus → focus state (Focused, Defocus1, Defocus2)

    Usage:
        1. Set `object_name`, `root_directory`, and `target_directory` variables in the main block.
        2. Run the script: python 01_rename_and_copy.py
"""       
            
# ----------------------------------------------------------------------
# Folder mapping: camera folder → [Distance, ISP, Focus]
# ----------------------------------------------------------------------
FOLDER_MAP = {
    "410CANON": ["Dist1", "OFF", "Focused"],
    "411CANON": ["Dist1", "OFF", "Defocus1"],
    "412CANON": ["Dist1", "OFF", "Defocus2"],
    "413CANON": ["Dist1", "ON", "Focused"],
    "414CANON": ["Dist1", "ON", "Defocus1"],
    "415CANON": ["Dist1", "ON", "Defocus2"],
    "416CANON": ["Dist2", "OFF", "Focused"],
    "417CANON": ["Dist2", "OFF", "Defocus1"],
    "418CANON": ["Dist2", "OFF", "Defocus2"],
    "419CANON": ["Dist2", "ON", "Focused"],
    "420CANON": ["Dist2", "ON", "Defocus1"],
    "421CANON": ["Dist2", "ON", "Defocus2"],
    "422CANON": ["Dist1", "ON", "Focused"],
    "423CANON": ["Dist1", "ON", "Defocus1"],
    "424CANON": ["Dist1", "ON", "Defocus2"],
    "425CANON": ["Dist1", "OFF", "Focused"],
    "426CANON": ["Dist1", "OFF", "Defocus1"],
    "427CANON": ["Dist1", "OFF", "Defocus2"],
    "428CANON": ["Dist2", "OFF", "Focused"],
    "429CANON": ["Dist2", "OFF", "Defocus1"],
    "430CANON": ["Dist2", "OFF", "Defocus2"],
    "431CANON": ["Dist2", "ON", "Focused"],
    "432CANON": ["Dist2", "ON", "Defocus1"],
    "433CANON": ["Dist2", "ON", "Defocus2"]
}

def extract_metadata(image_path):
    """
    This function extracts Focal Length, ISO, and EV from an image using ExifTool.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        tuple: (focal_length, iso, ev)
    """
    try:
        result = subprocess.run(
            ["exiftool", "-json", "-FocalLength", "-ISO", "-ExposureCompensation", image_path],
            capture_output=True, text=True, check=True
        )
        metadata = json.loads(result.stdout)[0]

        focal_length = metadata.get("FocalLength", "Unknown")
        iso = metadata.get("ISO", "Unknown")
        ev = metadata.get("ExposureCompensation", "Unknown") 
        return focal_length, iso, ev
    
    except subprocess.CalledProcessError as e:
        print(f"Error reading metadata: {e}")
        return "Unknown", "Unknown", "Unknown"
    

def rename_and_copy(object_name, root_directory, target_directory):
    """
    Renames all CR3 and JPEG images and copy them to RAW/ and JPG/ in the target directory.
    
    Args:
        object_name (str): Name of the object captured (e.g., "chart", "cube").
        root_dir (str): Directory containing camera folders (e.g., 410CANON, 411CANON...).
        target_dir (str): Destination directory containing RAW/ and JPG/ subfolders.
    """
    # Define RAW and JPG folders inside target directory
    raw_folder = os.path.join(target_directory, "RAW")
    jpg_folder = os.path.join(target_directory, "JPG")

    # Create folders if they do not exist
    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(jpg_folder, exist_ok=True)

    print(f"\nChecking root directory: {root_directory}")

    for folder in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder)
        # Debugging: Print detected folders
        print(f"Found folder: {folder}")

        # Check if it's a valid CANON folder
        if not os.path.isdir(folder_path):
            print(f"Skipping {folder} (not a directory)")
            continue

        if folder not in FOLDER_MAP:
            print(f"Skipping {folder} (no mapping found)")
            continue

        # Get folder descriptor values
        dist, isp, focus = FOLDER_MAP[folder]

        # Process images inside the folder
        found_files = False
        for file in os.listdir(folder_path):
            if file.lower().endswith((".cr3", ".jpg")):
                old_path = os.path.join(folder_path, file)

                # Extract Metadata
                focal_length, iso, ev = extract_metadata(old_path)

                # Correct EV value
                if isinstance(ev, str) and "/" in ev:
                    try:
                        sign = "-" if "-" in ev else "+"
                        ev_clean = ev.replace("+", "").replace("-", "")
                        numerator, denominator = ev_clean.split("/")
                        ev_float = round(int(numerator) / int(denominator), 2)
                        ev = f"{sign}{ev_float}"
                    except Exception:
                        ev = ev.replace("/", "_")  # Fallback if parsing fails
                else:
                    ev = str(ev).replace("/", "_")

                # Construct new name
                extension = file.split(".")[-1]
                new_name = f"{object_name}_{focal_length}_{iso}ISO_{ev}EV_{dist}_{isp}_{focus}.{extension}"

                # Destination path
                new_path = os.path.join(raw_folder if extension.lower() == "cr3" else jpg_folder, new_name)

                # Copy and rename file
                try:
                    shutil.copy2(old_path, new_path)
                    print(f"Copied & Renamed: {file}")
                except Exception as e:
                    print(f"Error copying {file}: {e}")


if __name__ == "__main__":
    object = "chart" 
    root_directory = "/home/colourlabgpu4/Kimia/Thesis/captures/chart"
    target_directory = "/home/colourlabgpu4/Kimia/Datasets/Full Dataset/eSFR Chart"
    
    
    rename_and_copy(object, root_directory, target_directory)
    print("\nAll images renamed and copied successfully")