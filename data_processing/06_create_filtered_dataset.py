import os
import shutil
import json
from pathlib import Path

"""
Script 06:
Create a subset of the dataset based on filename keywords.

The script will::
    Create smaller filtered versions of an existing dataset by copying
    only those images (and their labels and annotations) that contain all specified
    keywords in the filename.

    It preserves COCO JSON structure and YOLO directory hierarchy:
        images/ → corresponding filtered images
        labels/ → matching YOLO .txt labels
        annotations/ → filtered COCO JSON
        data.yaml → updated dataset path

Inputs:
    - src_dataset_path : Path to the source dataset folder
    - base_output_dir  : Path where filtered datasets will be created
    - keywords         : List of keywords to match in filenames
    - split            : Dataset split (e.g., 'train', 'val', or 'test')

Outputs:
    - New dataset folder under `base_output_dir` named as:
          Dataset_<keyword1>_And_<keyword2>...
      containing only the filtered subset.

Usage:
    python 06_create_filtered_dataset.py

"""

def sanitize_name(s):
    """ Replace spaces with underscores for safe folder naming."""
    return s.replace(" ", "_")

def create_filtered_dataset(src_dataset_path, base_output_dir, keywords, split='val'):
    """
    Creates a new dataset subset filtered based on filename keywords.

    Args:
        src_dataset_path (str): Path to original dataset folder.
        base_output_dir (str): Base path where filtered datasets will be created.
        keywords (list of str): List of keywords to filter filenames (e.g., ['18.0 mm', 'Dist1']).
        split (str): Dataset split to filter (default: 'val').
    """

    src_dataset_path = Path(src_dataset_path)
    dst_name = "Dataset_" + "_And_".join([sanitize_name(k) for k in keywords])
    dst_dataset_path = Path(base_output_dir) / dst_name

    src_img_dir = src_dataset_path / 'images' / split
    src_lbl_dir = src_dataset_path / 'labels' / split
    src_ann_path = src_dataset_path / 'annotations' / f'instances_{split}.json'
    src_yaml_path = src_dataset_path / 'data.yaml'

    dst_img_dir = dst_dataset_path / 'images' / split
    dst_lbl_dir = dst_dataset_path / 'labels' / split
    dst_ann_dir = dst_dataset_path / 'annotations'
    dst_yaml_path = dst_dataset_path / 'data.yaml'

    # Create destination folders
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    dst_ann_dir.mkdir(parents=True, exist_ok=True)

    # --- Filter images and copy corresponding labels ---
    selected_images = []
    for img_file in src_img_dir.iterdir():
        if all(k in img_file.name for k in keywords):
            lbl_file = src_lbl_dir / (img_file.stem + '.txt')

            shutil.copy(img_file, dst_img_dir / img_file.name)
            if lbl_file.exists():
                shutil.copy(lbl_file, dst_lbl_dir / lbl_file.name)

            selected_images.append(img_file.name)

    print(f"Found {len(selected_images)} matching images with keywords {keywords}.")

    # --- Filter and copy COCO annotations ---
    with open(src_ann_path, 'r') as f:
        coco_data = json.load(f)

    filtered_images = [img for img in coco_data['images'] if img['file_name'] in selected_images]
    filtered_img_ids = {img['id'] for img in filtered_images}
    filtered_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in filtered_img_ids]

    new_coco = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": coco_data.get("categories", [])
    }

    with open(dst_ann_dir / f'instances_{split}.json', 'w') as f:
        json.dump(new_coco, f, indent=4)

    print(f"Filtered COCO JSON created with {len(filtered_images)} images and {len(filtered_annotations)} annotations.")

    with open(src_yaml_path, 'r') as f:
        yaml_lines = f.readlines()

    with open(dst_yaml_path, 'w') as f:
        for line in yaml_lines:
            if line.startswith('path:'):
                f.write(f'path: {dst_dataset_path}\n')
            else:
                f.write(line)

    print(f"New dataset YAML created at {dst_yaml_path}")

if __name__ == "__main__":

    src_dataset = '/home/colourlabgpu4/Kimia/Thesis/Datasets/Full Dataset'
    base_output_dir = '/home/colourlabgpu4/Kimia/Thesis/Datasets'
    
    # Example of combination of camera settings 
    keyword_sets = [
        ['55.0 mm', 'Dist2']
    ]

    # Example of one camera setting
    #keyword_sets = [['100ISO']]

    for keywords in keyword_sets:
        create_filtered_dataset(src_dataset, base_output_dir, keywords, split='val')
