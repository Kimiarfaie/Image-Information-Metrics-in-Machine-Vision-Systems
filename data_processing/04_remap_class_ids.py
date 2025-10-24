import json
import os

"""
    Script 04: Remap custom class IDs to official COCO class IDs.

    This script will:
    - Reads a COCO-format JSON file, updates each annotation's `category_id` 
    according to a predefined mapping.
    - Writes a new JSON file with the updated annotations. 
    - Optionally, it also updates the `categories` list to reflect only the remapped categories.

    Usage:
    1. Set `input_json` and `output_json` variables in the main block
    2. Run the script: python 04_remap_class_ids.py    
        """

def remap_class_ids(input_path, output_path, class_map, update_categories=True):
    """
    Remap custom category IDs in a COCO-style JSON file to official COCO IDs.

    Args:
        input_path (str): Path to the input JSON file.
        output_path (str): Path to save the updated JSON file.
        class_map (dict[int, int]): Mapping from custom IDs to COCO IDs.
        update_categories (bool): Whether to rewrite the `categories` field.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r") as f:
        coco_data = json.load(f)

    for ann in coco_data["annotations"]:
        old_id = ann["category_id"]
        if old_id in class_map:
            ann["category_id"] = class_map[old_id]
            changed += 1

    if update_categories:
        # Build the new categories list from the mapping
        coco_data["categories"] = [
            {"id": 2, "name": "car"},
            {"id": 32, "name": "sports ball"},
            {"id": 41, "name": "cup"},
            {"id": 58, "name": "potted plant"},
        ]

    with open(output_path, "w") as f:
        json.dump(coco_data, f, indent=4)

    print(f"Class IDs remapped successfully.")
    print(f"   Total annotations updated: {changed}")
    print(f"   Saved to: {output_path}")


if __name__ == "__main__":
    # --- Configuration ---
    custom_to_coco_map = {
        1: 58,  # potted plant
        2: 41,  # cup
        3: 32,  # sports ball
        4: 2,   # car
    }

    input_path = "/home/colourlabgpu4/Kimia/Thesis/DATA_PROCESSING/Annotations/Full_Annotation.json"
    output_path = "/home/colourlabgpu4/Kimia/Thesis/DATA_PROCESSING/Annotations/instances_val.json"

    remap_class_ids(input_path, output_path, custom_to_coco_map)
