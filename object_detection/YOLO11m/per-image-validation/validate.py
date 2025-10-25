#!/usr/bin/env python3
"""
Per-image YOLO validation for camera performance analysis.

The script will:
    - Eevaluate a YOLO model, individually on each image of a dataset.
    - For every image, it builds a temporary single-image dataset, 
    - runs `model.val()`,
    - collects metrics per image (mAP, precision, recall), and saves them in a CSV file.

Usage:
    1. Specify the settings below (model path, dataset path, output paths, selected classes).
    2. Run python validate.py

Usage:
    python validate.py --model yolo11m.pt --dataset "/home/.../Datasets/Full Dataset" --output "/home/.../YOLO11/per-image-validation/outputs" --classes 2 32 41 58
"""

import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import argparse
from utilities.save_metrics import save_metrics


def run_per_image_validation(model_path, dataset_path, output_base_dir, class_ids):
    """
    Run per-image validation and save results.

    Args:
        model_path (str): Path to YOLO weights (.pt file).
        dataset_path (str): Path to dataset (should contain images/, labels/, annotations/).
        output_base_dir (str): Folder to store per-image results.
        class_ids (list[int]): COCO IDs of classes to evaluate.
    """

    # ------------------- Derived paths -------------------
    images_dir = os.path.join(dataset_path, "images", "val")
    annotations_path = os.path.join(dataset_path, "annotations", "instances_val.json")

    dataset_name = os.path.basename(dataset_path)
    output_dir = os.path.join(output_base_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dataset_base = os.path.join(script_dir, "temp_datasets")

    # ------------------- Load model and data -------------------
    model = YOLO(model_path)

    with open(annotations_path, "r") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    # Map annotations by image ID
    ann_by_image = {}
    for ann in annotations:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    # ------------------- Prepare CSV -------------------
    per_image_root = os.path.join(output_dir, "per_image_results")
    os.makedirs(per_image_root, exist_ok=True)

    csv_path = os.path.join(output_dir, "all_images_metricsummary.csv")
    
    with open(csv_path, "w") as f:
        f.write("image,mAP50_95,mAP50,mAP75,Precision,Recall\n")

    # ------------------- Loop through images -------------------
    for img in tqdm(images, desc="Per-image validation"):
        image_id = img["id"]
        image_file = img["file_name"]
        img_name = Path(image_file).stem
        image_output_dir = os.path.join(per_image_root, img_name)

        # Temporary dataset structure
        temp_root = os.path.join(temp_dataset_base, img_name)
        temp_img_dir = os.path.join(temp_root, "images", "val")
        temp_lbl_dir = os.path.join(temp_root, "labels", "val")
        temp_ann_dir = os.path.join(temp_root, "annotations")
        os.makedirs(temp_img_dir, exist_ok=True)
        os.makedirs(temp_lbl_dir, exist_ok=True)
        os.makedirs(temp_ann_dir, exist_ok=True)

        # Copy YOLO label (if available)
        label_src = os.path.join(dataset_path, "labels", "val", f"{img_name}.txt")
        label_dst = os.path.join(temp_lbl_dir, f"{img_name}.txt")
        if os.path.exists(label_src):
            shutil.copy(label_src, label_dst)
        else:
            print(f"No label found for {img_name}")

        # Copy image
        shutil.copy(os.path.join(images_dir, image_file), os.path.join(temp_img_dir, image_file))

        # Single-image JSON
        ann_json = {
            "images": [img],
            "annotations": ann_by_image.get(image_id, []),
            "categories": categories,
        }
        with open(os.path.join(temp_ann_dir, "instances_val.json"), "w") as f:
            json.dump(ann_json, f, indent=2)

        # Rewrite data.yaml with correct path
        original_yaml = os.path.join(dataset_path, "data.yaml")
        new_yaml = os.path.join(temp_root, "data.yaml")

        with open(original_yaml, "r") as f:
            lines = f.readlines()
        with open(new_yaml, "w") as f:
            for line in lines:
                if line.strip().startswith("path:"):
                    f.write(f"path: {temp_root}\n")
                else:
                    f.write(line)

        # Run YOLO validation
        metrics = model.val(
            data=new_yaml,
            save_json=True,
            plots=False,
            classes=class_ids,
            project=image_output_dir,
            name=".",
            verbose=False,
            save_txt=False,
        )

        # Append results
        with open(csv_path, "a") as f:
            f.write(
                f"{image_file},{metrics.box.map:.4f},{metrics.box.map50:.4f},"
                f"{metrics.box.map75:.4f},{metrics.box.mp:.4f},{metrics.box.mr:.4f}\n"
            )

        # Save full metrics and clean up temp dataset
        save_metrics(metrics, image_output_dir, class_ids)
        shutil.rmtree(temp_root, ignore_errors=True)

    print("\n per-image validation complete.")
    print(f"Metrics saved to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run per-image YOLO validation on a dataset."
    )

    parser.add_argument(
        "--model", 
        type=str, 
        default="yolo11m.pt",
        help="Path to YOLO weights (.pt file)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/Users/kimiaarfaie/Github/Image-Information-Metrics-in-Machine-Vision-Systems/Datasets/Full Dataset",
        help="Path to dataset folder (must contain images/, labels/, annotations/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/Users/kimiaarfaie/Github/Image-Information-Metrics-in-Machine-Vision-Systems/object_detection/YOLO11m/per-image-validation/outputs",
        help="Base output directory for validation results",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[2, 32, 41, 58],
        help="List of COCO class IDs to evaluate, e.g., --classes 2 32 41 58",
    )

    args = parser.parse_args()

    run_per_image_validation(args.model, args.dataset, args.output, args.classes)