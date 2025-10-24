import json
import os

def COCO2YOLO(json_path, output_path):
    """
    Script 05: 
    Convert COCO-format annotations to YOLO format.

    Args:
        json_path (str): Path to json annotation file.
        output_path (str): Path to the folder where labels will be created.

    Usage:
    python 05_coco_to_yolo.py

    """
    with open(json_path) as f:
        data = json.load(f)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for image in data['images']:
        image_id = image['id']
        file_name = image['file_name']
        width = image['width']
        height = image['height']

        # All annotations corresponding to this image
        annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]

        # Write YOLO-format label file
        with open(os.path.join(output_path, f"{os.path.splitext(file_name)[0]}.txt"), 'w') as f:
            for ann in annotations:
                category_id = ann['category_id'] # COCO start with 1, YOLO with 0
                bbox = ann['bbox']
                x_center = (bbox[0] + bbox[2] / 2) / width
                y_center = (bbox[1] + bbox[3] / 2) / height
                w = bbox[2] / width
                h = bbox[3] / height
                f.write(f"{category_id} {x_center} {y_center} {w} {h}\n")
    print(f"Conversion complete. YOLO labels saved to: {output_path}")

if __name__ == "__main__":
    json_path = "/home/colourlabgpu4/Kimia/Thesis/Data_Processing/instances_val.json"
    output_path = "/home/colourlabgpu4/Kimia/Thesis/Datasets/Full Dataset/labels/val"

    COCO2YOLO(json_path, output_path)