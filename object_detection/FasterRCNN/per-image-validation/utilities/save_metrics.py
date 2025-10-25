# utilities/save_metrics.py

import os

def save_metrics(metrics, output_dir, selected_class_ids):
    """
    Save important evaluation metrics to a text file.
    
    Args:
        metrics: Metrics object from model.val().
        output_dir (str): Directory to save the summary file.
        selected_class_ids (list): List of class IDs to save mAP for.
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "metrics_summary.txt")

    with open(summary_path, 'w') as f:
        f.write(f"mAP@[0.5:0.95]: {metrics.box.map:.4f}\n")
        f.write(f"mAP@0.5: {metrics.box.map50:.4f}\n")
        f.write(f"mAP@0.75: {metrics.box.map75:.4f}\n")
        f.write(f"Precision: {metrics.box.mp:.4f}\n")
        f.write(f"Recall: {metrics.box.mr:.4f}\n")
        f.write(f"Category mAP@[0.5:0.95]:\n")
        for class_id in selected_class_ids:
            f.write(f"  Class {class_id}: {metrics.box.maps[class_id]:.4f}\n")

    print(f"Metrics saved to {summary_path}")
