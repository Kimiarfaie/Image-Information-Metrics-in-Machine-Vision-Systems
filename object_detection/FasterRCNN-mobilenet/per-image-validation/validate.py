import os
import json
import torch
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from utilities.save_metrics import save_metrics
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.transforms.functional import to_tensor



# ----- Configuration -----
dataset_base_dir = "/Users/kimiaarfaie/Github/Image-Information-Metrics-in-Machine-Vision-Systems/Datasets"
dataset_name = "Full Dataset"
output_dir = "/Users/kimiaarfaie/Github/Image-Information-Metrics-in-Machine-Vision-Systems/object_detection/FasterRCNN/per-image-validation/outputs/Full Dataset"
selected_class_ids = [3, 37, 47, 64]  # Official xCOCO class IDs: car, sports ball, cup, potted plant
CONFIDENCE_THRESHOLD = 0.001  

# Mapping from torchvision model's class index to official COCO IDs
TORCHVISION_COCO_CATEGORY_MAPPING = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
    11: 11, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18,
    19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24, 25: 25,
    27: 27, 28: 28, 31: 31, 32: 32, 33: 33, 34: 34, 35: 35,
    36: 36, 37: 37, 38: 38, 39: 39, 40: 40, 41: 41, 42: 42,
    43: 43, 44: 44, 46: 46, 47: 47, 48: 48, 49: 49, 50: 50,
    51: 51, 52: 52, 53: 53, 54: 54, 55: 55, 56: 56, 57: 57,
    58: 58, 59: 59, 60: 60, 61: 61, 62: 62, 63: 63, 64: 64,
    65: 65, 67: 67, 70: 70, 72: 72, 73: 73, 74: 74, 75: 75,
    76: 76, 77: 77, 78: 78, 79: 79, 80: 80, 81: 81, 82: 82,
    84: 84, 85: 85, 86: 86, 87: 87, 88: 88, 89: 89, 90: 90
}

# Derived paths
dataset_path = os.path.join(dataset_base_dir, dataset_name)
images_dir = os.path.join(dataset_path, "images", "val")
annotations_path = os.path.join(dataset_path, "annotations", "instances_val.json")

# Load COCO annotations
# --- Load and temporarily remap COCO annotations ---
with open(annotations_path, "r") as f:
    coco_data = json.load(f)

# Remap wrong -> correct COCO class IDs
category_remap = {2: 3, 32: 37, 41: 47, 58: 64}

for ann in coco_data["annotations"]:
    if ann["category_id"] in category_remap:
        ann["category_id"] = category_remap[ann["category_id"]]

# Replace the categories list to match only remapped categories (optional but safe)
coco_data["categories"] = [
    {"id": 3, "name": "car"},
    {"id": 37, "name": "sports ball"},
    {"id": 47, "name": "cup"},
    {"id": 64, "name": "potted plant"},
]

# Save to a temporary JSON file just for evaluation
temp_ann_path = os.path.join(output_dir, "instances_val_TEMP.json")
with open(temp_ann_path, "w") as f:
    json.dump(coco_data, f)

# Load COCO using the temp file
coco = COCO(temp_ann_path)

images = coco.loadImgs(coco.getImgIds())

model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)

# Override the transform with custom resizing
custom_transform = GeneralizedRCNNTransform(
    min_size=427,  # shortest side: approximately what YOLO uses for 3:2 input
    max_size=640,
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225]
)

model.transform = custom_transform

model = model.eval().cuda()

# CSV setup
csv_path = os.path.join(output_dir, "all_images_metricsummary.csv")
results = []

# Per-image validation
for img_info in tqdm(images, desc="Per-image validation"):
    image_file = img_info["file_name"]
    image_name = os.path.splitext(image_file)[0]
    image_output_dir = os.path.join(output_dir, "per_image_results", image_name)
    os.makedirs(image_output_dir, exist_ok=True)

    image_path = os.path.join(images_dir, image_file)
    image = Image.open(image_path).convert("RGB")

    img_tensor = to_tensor(image).cuda()  # convert PIL to tensor
    with torch.no_grad():
        predictions = model([img_tensor])[0]

    # Convert predictions to COCO format
    coco_preds = []
    #(f"\nDebug for image: {image_name}")
    #print("Predictions from model:")
    for idx in range(len(predictions["boxes"])):
        score = predictions["scores"][idx].item()
        if score < CONFIDENCE_THRESHOLD:
            continue  # skip low-confidence detections
        bbox = predictions["boxes"][idx].cpu().numpy().tolist()
        bbox_xywh = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]  # xywh
        score = predictions["scores"][idx].item()
        model_category_id = predictions["labels"][idx].item()
        actual_coco_id = TORCHVISION_COCO_CATEGORY_MAPPING.get(model_category_id, None)

        #print(f" - Model Category ID: {model_category_id}, "
          #f"Mapped COCO ID: {actual_coco_id}, "
          #f"Score: {score:.3f}, BBox: {bbox_xywh}")
    
        if actual_coco_id in selected_class_ids:
            coco_preds.append({
                "image_id": img_info["id"],
                "image_name": image_file, 
                "category_id": actual_coco_id,
                "bbox": bbox_xywh,
                "score": score
            })

    #print(f"Filtered predictions matching selected classes ({selected_class_ids}): {len(coco_preds)} items.")

    # Save predictions
    predictions_path = os.path.join(image_output_dir, "predictions.json")
    with open(predictions_path, "w") as f:
        json.dump(coco_preds, f)

    # Check for empty predictions
    if not coco_preds:
        mAP = mAP50 = mAP75 = precision = recall = 0.0
    else:
        temp_pred_path = os.path.join(output_dir, "temp_predictions.json")
        # Temporary COCO predictions file
        #print("Prediction image IDs in temp_pred_path:")
        with open(temp_pred_path, "w") as f:
            json.dump(coco_preds, f)
            #print(set(p['image_id'] for p in preds))  # Image IDs in predictions
            #print("COCO GT image IDs:", set(coco.getImgIds()))  # Image IDs in annotations

        # Evaluate predictions
        cocoDt = coco.loadRes(temp_pred_path)
        cocoEval = COCOeval(coco, cocoDt, iouType='bbox')
        cocoEval.params.imgIds = [img_info["id"]]
        cocoEval.params.catIds = selected_class_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        mAP = cocoEval.stats[0]
        mAP50 = cocoEval.stats[1]
        mAP75 = cocoEval.stats[2]
        precision = cocoEval.stats[8]
        recall = cocoEval.stats[9]

        # Cleanup temp predictions
        os.remove(temp_pred_path)

    # Save metrics
    metrics_summary_path = os.path.join(image_output_dir, "metrics_summary.txt")
    with open(metrics_summary_path, "w") as f:
        f.write(f"mAP50_95: {mAP:.4f}\nmAP50: {mAP50:.4f}\nmAP75: {mAP75:.4f}\n")
        f.write(f"Precision: {precision:.4f}\nRecall: {recall:.4f}\n")

    results.append({
        "image": image_name,
        "mAP50_95": mAP,
        "mAP50": mAP50,
        "mAP75": mAP75,
        "Precision": precision,
        "Recall": recall
    })

# Write results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(csv_path, index=False)

print(f"\nDone. Per-image metrics saved to: {csv_path}")

# Cleanup temporary annotation file
if os.path.exists(temp_ann_path):
    os.remove(temp_ann_path)