import os
import rawpy
import imageio
import numpy as np

def raw_to_tiff(root_directory):
    """
    Script 02:
    Converts all RAW CR3 images in RAW folder to 8-bit TIFF format using rawpy.

    This script will:
    - Read each .CR3 file from the RAW/ folder.- converts all Canon RAW (.CR3) files located in the "RAW" folder into 8-bit TIFF and PNG formats using the `rawpy` library.
    - The conversion applies camera white balance, disables auto-brightening, and uses a gamma curve consistent with sRGB output.
    
    Usage:
    1. Update `root_directory` in the main block.
    2. Run the script: python 02_raw_to_tiff.py
    """
        
    # Define folder paths
    raw_folder = os.path.join(root_directory, "RAW")
    tiff_folder = os.path.join(root_directory, "TIFF")
    png_folder = os.path.join(root_directory, "PNG")

    # Create TIFF and PNG directories if they do not exist
    os.makedirs(tiff_folder, exist_ok=True)
    os.makedirs(png_folder, exist_ok=True)

    # Safety check
    if not os.path.exists(raw_folder):
        print(f"RAW folder not found at: {raw_folder}")
        return
    
    # Process each RAW file
    for file in os.listdir(raw_folder):
        if file.lower().endswith(".cr3"):
            cr3_path = os.path.join(raw_folder, file)
            base_name = os.path.splitext(file)[0]
            tiff_path = os.path.join(tiff_folder, f"{base_name}.tiff")
            png_path = os.path.join(png_folder, f"{base_name}.png")

            print(f"Converting: {file}")

            try:
                with rawpy.imread(cr3_path) as raw:
                    # 8-bit TIFF conversion
                    tiff_8bit = raw.postprocess(
                        use_camera_wb=True, # Use camera white balance (-w)
                        no_auto_bright=True, # Don't brighten image (-W)
                        gamma=(2.4, 12.92), # Gamma curve
                        output_bps=8, # 8-bit TIFF output
                        output_color=rawpy.ColorSpace.sRGB, # Output in sRGB (-o 1)
                        half_size=False
                    )
                    imageio.imwrite(tiff_path, tiff_8bit)
                    print(f"Saved TIFF: {tiff_path}")

            except Exception as e:
                print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    root_directory = "/home/colourlabgpu4/Kimia/Datasets/Full Dataset/eSFR Chart/RAW"

    raw_to_tiff(root_directory)
    print("\nAll images converted to TIFF and PNG successfully.")
