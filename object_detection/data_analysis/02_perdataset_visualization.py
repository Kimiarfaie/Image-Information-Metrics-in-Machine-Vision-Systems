import os
import json
import cv2

# ------------ Utility Functions ------------

def load_annotations(annotations_path):
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    image_id_map = {img['id']: img['file_name'] for img in data['images']}
    filename_to_id = {img['file_name']: img['id'] for img in data['images']}
    return data['annotations'], image_id_map, filename_to_id

def load_predictions(predictions_path):
    with open(predictions_path, 'r') as f:
        preds = json.load(f)
    for pred in preds:
        pred['category_id'] -= 1
    return preds

def draw_bboxes(image, bboxes, color, class_label=None, draw_score=False):
    for box in bboxes:
        score = box.get("score", 1.0)
        x, y, w, h = box['bbox']
        x2, y2 = x + w, y + h
        label = f"{class_label} {score:.3f}" if draw_score and class_label else class_label or ""

        cv2.rectangle(image, (int(x), int(y)), (int(x2), int(y2)), color, thickness=6)

        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            thickness = 3
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            text_x, text_y = int(x), int(y) - 10

            cv2.rectangle(
                image,
                (text_x, text_y - text_height - 20),
                (text_x + text_width + 20, text_y),
                color,
                thickness=cv2.FILLED
            )
            cv2.putText(
                image,
                label,
                (text_x + 10, text_y - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness=thickness,
                lineType=cv2.LINE_AA
            )

def visualize_image(image_path, image_id, class_id, annotations, predictions, output_path_jpg, draw_gt=True, draw_preds=True, class_label=""):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    image_basename = os.path.splitext(os.path.basename(image_path))[0]

    if draw_gt:
        gt_boxes = [a for a in annotations if a['image_id'] == image_id and a['category_id'] == class_id]
        draw_bboxes(image, gt_boxes, (0, 255, 0), class_label="GT", draw_score=False)

    if draw_preds:
        pred_boxes = [p for p in predictions if p['image_id'] == image_basename and p['category_id'] == class_id]
        if pred_boxes:
            best_pred = max(pred_boxes, key=lambda x: x['score'])
            draw_bboxes(image, [best_pred], (0, 0, 255), class_label=class_label, draw_score=True)

    os.makedirs(os.path.dirname(output_path_jpg), exist_ok=True)
    #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path_jpg, image)

# ------------ Main ------------

keyword = "18.0_mm_And_Dist1"

base_dataset_path = f"/Users/kimiaarfaie/Github/Image-Information-Metrics-in-Machine-Vision-Systems/Dataset_{keyword}"
base_output_path = f"/Users/kimiaarfaie/Github/Image-Information-Metrics-in-Machine-Vision-Systems/object_detection/YOLO11n/per-dataset-validation/outputs/{keyword}"

data_analysis_path = os.path.join(base_output_path, "data_analysis.json")
annotations_path = os.path.join(base_dataset_path, "annotations", "instances_val.json")
predictions_path = os.path.join(base_output_path, "predictions.json")
images_dir = os.path.join(base_dataset_path, "images", "val")
output_dir = os.path.join(base_output_path, "visualization")

# Class mapping
class_id_to_name = {
    2: "car", 32: "sports ball", 41: "cup", 58: "potted plant"
}

# Load data
with open(data_analysis_path, 'r') as f:
    analysis = json.load(f)

annotations, image_id_map, filename_to_id = load_annotations(annotations_path)
predictions = load_predictions(predictions_path)

for class_id_str, class_data in analysis.items():
    class_id = int(class_id_str)
    class_label = class_id_to_name.get(class_id, f"class_{class_id}")
    top_images = class_data['top_k_images']
    missed_images = class_data.get('missed_images', [])
    absent_images = class_data.get('absent_images', [])

    for subset, image_list, draw_preds, draw_gt in [
        ("top", top_images, True, True),
        ("missed", [(img_name, None) for img_name, _ in missed_images], False, True),
        ("absent", [(img_name, None) for img_name in absent_images], False, True),
    ]:
        for image_name, _ in image_list:
            image_path = os.path.join(images_dir, image_name)
            output_path_jpg = os.path.join(output_dir, f"class_{class_id}", subset, image_name.replace(".tiff", ".jpg"))

            visualize_image(
                image_path=image_path,
                image_id=filename_to_id[image_name],
                class_id=class_id,
                annotations=annotations,
                predictions=predictions,
                output_path_jpg=output_path_jpg,
                draw_gt=draw_gt,
                draw_preds=draw_preds,
                class_label=class_label
            )

print("Visualization complete.")
