import os
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

'''
Script 02:
Per-Image Detection Analysis Across Dataset Splits

This script evaluates per-image detection performance across multiple
split conditions (e.g., ISO levels, EV values, focus states, distances)
by aggregating detection rates from per-image validation outputs.

- Loads individual `predictions.json` files from each per-image result folder
- Loads ground-truth `instances_val.json` annotations
and computes average detection rates per condition and per class.

Usage:
1. Configure the following:
   - `split_conditions` → groups of conditions (e.g., ISO levels, focus states)
   - `split` → name of the analysis factor (e.g., "ISO", "EV", "focus", "size")
   - `network` → "YOLO11", "FasterRCNN", or other model name
   - `keyword` → dataset name ("Full Dataset", etc.)

2. Run:
   python 01_perimg_analysis.py

'''

# -------- CONFIG --------
ISO_split = [["100ISO"], ["1600ISO"], ["6400ISO"], ["25600ISO"]]

EV_split = [ ["-3EV"], ["-2EV"], ["-1EV"], ["0EV"], ["+1EV"]]

Focus_split = [ ["Focused"], ["Defocus1"], ["Defocus2"]]

Size_split = [["18.0 mm", "Dist1"], ["18.0 mm", "Dist2"], ["55.0 mm", "Dist1"], ["55.0 mm", "Dist2"]]

split_conditions = ISO_split  # or EV_split, Focus_split, etc.

split = "size"  # or "ISO", "EV", "focus", etc.
network = "FasterRCNN"  # or "FasterRCNN", "SSD", etc.
keyword = "Full Dataset" # name of the dataset
CONF_THRESHOLD = 0.25

# Class IDs depending on network
yolo_class_ids = [2, 32, 41, 58]      # from annotation files
other_class_ids = [3, 37, 47, 64]     # predicted by non-YOLO models
category_remap = {2: 3, 32: 37, 41: 47, 58: 64}  # for remapping GT annotations

if network.lower().startswith("yolo"):
    selected_class_ids = yolo_class_ids
else:
    selected_class_ids = other_class_ids

# -------- PATHS --------
base_output_path = f"/home/colourlabgpu4/Kimia/Thesis/Object_Detection/{network}/per-image-validation/outputs/{keyword}"
per_image_dir = os.path.join(base_output_path, "per_image_results")
annotations_path = f"/home/colourlabgpu4/Kimia/Thesis/Datasets/{keyword}/annotations/instances_val.json"
perimg_analysis_dir = os.path.join(base_output_path, "perimg_data_analysis")
output_analysis_path = os.path.join(base_output_path, f"data_analysis_{split}.json")

def normalize_string(s):
    return s.lower().replace(" ", "").replace("_", "")

# -------- Load Annotations --------
with open(annotations_path, "r") as f:
    annotations_data = json.load(f)

# Remap annotations if needed
for ann in annotations_data["annotations"]:
    if not network.lower().startswith("yolo") and ann["category_id"] in category_remap:
        ann["category_id"] = category_remap[ann["category_id"]]

# Map image IDs <-> filenames
id_to_filename = {img["id"]: img["file_name"] for img in annotations_data["images"]}
filename_to_id = {Path(fname).stem: id for id, fname in id_to_filename.items()}

# Ground truth mapping
gt_by_class = defaultdict(set)
for ann in annotations_data["annotations"]:
    cid = ann["category_id"]
    if cid in selected_class_ids:
        gt_by_class[cid].add(ann["image_id"])

# -------- Load Predictions --------
preds_by_image = defaultdict(list)
for folder in os.listdir(per_image_dir):
    pred_path = os.path.join(per_image_dir, folder, "predictions.json")
    if not os.path.exists(pred_path):
        continue

    with open(pred_path, "r") as f:
        preds = json.load(f)

    for p in preds:
        if network.lower().startswith("yolo"):
            key = p["image_id"]  # already filename string
            if not isinstance(p["category_id"], int):
                continue  # skip malformed entries
            original_id = p["category_id"]
            p["category_id"] -= 1  # convert from 1–80 → 0–79
        else:
            key = Path(p["image_name"]).stem

        preds_by_image[key].append(p)

# -------- Compute Per-Split Detection Rate --------
split_results = {}

for cond in split_conditions:
    label = " AND ".join(cond)

    matched_keys = [
        k for k in preds_by_image
        if all(normalize_string(c) in normalize_string(k) for c in cond)
    ]

    det_per_class = {cid: 0 for cid in selected_class_ids}
    total_per_class = {cid: 0 for cid in selected_class_ids}

    for k in matched_keys:

        img_id = filename_to_id.get(k)
        if img_id is None:
            continue

        for cid in selected_class_ids:
            if img_id not in gt_by_class[cid]:
                continue

            total_per_class[cid] += 1

            preds = [p for p in preds_by_image[k] if p["category_id"] == cid and p["score"] >= CONF_THRESHOLD]

            if preds:
                det_per_class[cid] += 1

    detection_rates = {
        str(cid): round(100 * det_per_class[cid] / total_per_class[cid], 2) if total_per_class[cid] > 0 else 0
        for cid in selected_class_ids
    }

    valid_rates = [v for v in detection_rates.values() if v > 0]
    average_rate = np.mean(valid_rates) if valid_rates else 0

    split_results[label] = {
        "avg_detection_rate": round(average_rate, 2),
        "per_class": detection_rates
    }

# -------- Save Results --------
with open(output_analysis_path, "w") as f:
    json.dump(split_results, f, indent=2)

print(f"\n Saved split-wise detection analysis to: {output_analysis_path}")
