import os
import json
from copy import deepcopy

"""
    Script 03:
    This script create COCO format annotation for all the images in the dataset from the 16 reference images which were annotated manually with CVAT.

    The script will:
    - Load the Reference_Annotation.json file which contains annotations for 16 reference images.
    - For each image in the dataset, it will find the corresponding reference image based on object name, focal length, and distance.
    - Copy the annotations from the reference image to the dataset images.

    Usage:
    1. Set `reference_path`, `dataset_folder`, and `output_path` variables in the main block.
    2. Run the script: python 03_generate_full_coco_annotations.py
"""

def extract_key(filename):
    """
    Extract the object name, focal length, and distance from the image file name.

    Args:
        filename (str): filename of the image.
    Returns:
        str: Key formatted as "<object>_<focal>_<distance>"
    """
    parts = filename.split("_")
    obj = parts[0]
    focal = parts[1]
    distance = parts[4]
    key = f"{obj}_{focal}_{distance}"
    return key

def generate_full_annotations(reference_path, dataset_folder, output_path):
    """
    Create full COCO-format annotations by mapping reference annotations
    to all dataset images that share the same object, focal length, and distance.

    Args:
        reference_path (str): Path to the reference annotation JSON.
        dataset_folder (str): Path to folder containing TIFF/JPG dataset images.
        output_path (str): Path to save the generated annotation file.
    """
    # Load the reference annotation
    with open(reference_path, "r") as f:
        reference_data = json.load(f)

    # Build mapping: key → (reference image metadata, annotations)
    ref_key_to_image = {}
    ref_anns_by_image_id = {}

    for img in reference_data["images"]:
        key = extract_key(img["file_name"])
        ref_key_to_image[key] = img
        ref_anns_by_image_id[img["id"]] = []

    for ann in reference_data["annotations"]:
        ref_anns_by_image_id[ann["image_id"]].append(ann)

    # Prepare new COCO structure
    new_data = {
        "licenses": reference_data.get("licenses", []),
        "info": reference_data.get("info", {}),
        "categories": reference_data.get("categories", []),
        "images": [],
        "annotations": []
    }

    # List dataset images
    all_images = sorted(
        f for f in os.listdir(dataset_folder)
        if f.lower().endswith((".jpg", ".tiff"))
    )

    next_image_id, next_ann_id = 1, 1

    for fname in all_images:
        key = extract_key(fname)
        if key not in ref_key_to_image:
            print(f"No matching reference found for {fname}, skipping.")
            continue

        ref_img = ref_key_to_image[key]
        ref_anns = ref_anns_by_image_id[ref_img["id"]]

        # Copy image metadata
        new_img = deepcopy(ref_img)
        new_img["id"] = next_image_id
        new_img["file_name"] = fname
        new_data["images"].append(new_img)

        # Copy annotations
        for ref_ann in ref_anns:
            new_ann = deepcopy(ref_ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = next_image_id
            new_data["annotations"].append(new_ann)
            next_ann_id += 1

        next_image_id += 1

    # Save the new JSON
    with open(output_path, "w") as f:
        json.dump(new_data, f, indent=2)

    print(f"\nFull annotation saved as: {output_path}")
    print(f"   Total images: {len(new_data['images'])}")
    print(f"   Total annotations: {len(new_data['annotations'])}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ann_dir = os.path.join(base_dir, "Annotations")

    reference_path = os.path.join(ann_dir, "Reference_Annotation.json")
    output_path = os.path.join(ann_dir, "Full_Annotation.json")
    dataset_folder = "/home/colourlabgpu4/Kimia/Datasets/Full Dataset/Images/TIFF"

    generate_full_annotations(reference_path, dataset_folder, output_path)