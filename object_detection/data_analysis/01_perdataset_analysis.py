import os
import json
from collections import defaultdict

"""
    Script 01:
    This script is for data analysis, when validation was done on a dataset (not in the thesis).
    
    The script takes predictions, annotations, and confidence threshold, and calculates detection rate, top, worst and missing detections.

    Usage:
    1. Set:
        - `keyword` → dataset split name (e.g., "Small", "1600ISO")
        - `base_output_path` → folder containing YOLO predictions
        - `base_dataset_path` → folder containing dataset annotations

    2. Run:
        python 01_perdataset_analysis.py

    3. The output JSON file will be created as:
        <base_output_path>/data_analysis.json
    
"""

# ----- Config -----
keyword = "Small"  # change to 1600ISO, etc.

# Paths
base_output_path = f"/home/colourlabgpu4/Kimia/Thesis/YOLO11n/per-dataset-validation/outputs/{keyword}"
base_dataset_path = f"/home/colourlabgpu4/Kimia/Thesis/Datasets/Dataset_{keyword}"

# Input and output files
predictions_json_path = os.path.join(base_output_path, "predictions.json")
annotations_json_path = os.path.join(base_dataset_path, "annotations", "instances_val.json")
output_path = os.path.join(base_output_path, "data_analysis.json")

# Analysis parameters
selected_class_ids = [2, 32, 41, 58]  # car, sports ball, cup, potted plant
CONF_THRESHOLD = 0.25
top_k = 10

# ----- Load files -----
with open(annotations_json_path, 'r') as f:
    annotations_data = json.load(f)

with open(predictions_json_path, 'r') as f:
    predictions = json.load(f)

# ----- Fix category_id indexing -----
for pred in predictions:
    if 1 <= pred['category_id'] <= 80:
        pred['category_id'] -= 1
    else:
        print(f"Unexpected category_id={pred['category_id']}")

# ----- Build mappings -----
id_to_filename = {img['id']: img['file_name'] for img in annotations_data['images']}
filename_to_id = {fname.replace('.tiff', ''): id for id, fname in id_to_filename.items()}

# ----- Build ground truth class-to-image mapping -----
gt_by_class = defaultdict(set)
for ann in annotations_data['annotations']:
    if ann['category_id'] in selected_class_ids:
        gt_by_class[ann['category_id']].add(ann['image_id'])

# ----- Build prediction mapping -----
preds_by_image = defaultdict(list)
for pred in predictions:
    img_id = filename_to_id.get(pred['image_id'])
    if img_id is not None:
        preds_by_image[img_id].append(pred)

# ----- Class-wise analysis -----
results = {}

for class_id in selected_class_ids:
    class_images = gt_by_class[class_id]
    detected = []  # [(img_id, max_score)]
    missed = []    # [(filename, max_score)]
    absent = []    # [filename]

    for img_id in class_images:
        preds_for_class = [p for p in preds_by_image.get(img_id, []) if p['category_id'] == class_id]

        if not preds_for_class:
            absent.append(id_to_filename[img_id])
        else:
            max_score = max(p['score'] for p in preds_for_class)
            if max_score >= CONF_THRESHOLD:
                detected.append((img_id, max_score))
            else:
                missed.append((id_to_filename[img_id], round(max_score, 4)))

    # Sort detected by max score for top-k
    top_k_detected = sorted(detected, key=lambda x: x[1], reverse=True)[:top_k]

    # Sort missed and take k examples
    missed_sorted = sorted(missed, key=lambda x: x[1])[:top_k]
    absent_sorted = absent[:top_k]

    results[class_id] = {
        "detection_rate": round(100 * len(detected) / len(class_images), 2),
        "num_images_with_class": len(class_images),
        "num_images_detected": len(detected),
        "num_images_missed": len(missed),
        "num_images_no_predictions": len(absent),
        "top_k_images": [(id_to_filename[i], round(s, 4)) for i, s in top_k_detected],
        "missed_images": missed_sorted,
        "absent_images": absent_sorted
    }

# ----- Save results -----
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Analysis complete. Results saved to {output_path}")
