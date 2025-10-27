import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Script 05:
This script compares camerra image quality metrics and object detection performance (mAP) from multiple networks 
(YOLO, SSD, Faster R-CNN variants) against camera settings. 

The script will:
    • Load per-image detection results for each network.
    • Match each camera condition (e.g., “100ISO_Focused”) with its averaged Imatest metrics JSON file.
    • Extract key quality metrics such as MTF50, C4, Cmax, SNR, and SNRi.
    • Compute average mAP per condition.
    • Generate and save:
        - Combined plots showing mAP and quality metrics vs. camera settings plot.
        - Average SNRi (in dB) vs. camera setting plot.
"""

# --- CONFIGURATION ---
networks = ["YOLO11n", "YOLO11m", "SSD", "FasterRCNN-mobilenet", "FasterRCNN"]
keyword = "NOISE-ONLY"
xlabel = "SNR (dB)"
iqm_source = "Dataset"

ISO_split = [["100ISO"], ["1600ISO"], ["6400ISO"], ["25600ISO"]]
focus_split = [["Focused"], ["Defocus1"], ["Defocus2"]]
ISOfocuse_split = [["100ISO", "Focused"], ["1600ISO", "Focused"], ["6400ISO", "Focused"], ["25600ISO", "Focused"]]
size_split = [["18.0 mm", "Dist1"], ["18.0 mm", "Dist2"], ["55.0 mm", "Dist1"], ["55.0 mm", "Dist2"]]
EV_split = [["-3EV"], ["-2EV"], ["-1EV"], ["0EV"], ["+1EV"]]

split_conditions = ISOfocuse_split

imatest_summary_path = f"/Users/kimiaarfaie/Github/Image-Information-Metrics-in-Machine-Vision-Systems/Metric_Analysis/Metrics/{iqm_source}/Average Metrics+1"

custom_annotations = {
    "Chart": {
        "sharpness": ["0.151 Cy/Pxl", "0.033 Cy/Pxl", "0.017 Cy/Pxl"]
    },
    "Dataset": {
        "sharpness": ["0.155 Cy/Pxl", "0.042 Cy/Pxl", "0.024 Cy/Pxl"],
        "EV": ["8.21", "12.30", "18.04", "25.94", "35.71"],
        "size": ["size1", "size2", "size3", "size4"]
    }
}

def get_annotation(split, idx):
    if keyword in custom_annotations.get(iqm_source, {}):
        values = custom_annotations[iqm_source][keyword]
        if idx < len(values):
            return values[idx]
    return None

def add_vertical_annotations(splits, y_max, plot_type):
    vertical_offset = 0.03 * y_max
    for idx, split in enumerate(splits):
        ann = get_annotation(split, idx)
        label = ann if ann else None
        if not label and split in snr_values_for_ISO and plot_type in ["combined", "snri_square"]:
            label = f"{snr_values_for_ISO[split]:.1f} dB"
        if label:
            plt.axvline(x=split, color='gray', linestyle='--', linewidth=1)
            plt.text(split, y_max + vertical_offset, label, ha='center', va='bottom',
                     fontsize=12, color='gray', rotation=90)

for network in networks:
    print(f"Processing: {network}")
    csv_path = f"/Users/kimiaarfaie/Github/Image-Information-Metrics-in-Machine-Vision-Systems/object_detection/{network}/per-image-validation/outputs/Full Dataset/per_image_results.csv"
    analysis_dir = f"/Users/kimiaarfaie/Github/Image-Information-Metrics-in-Machine-Vision-Systems/Metric_Analysis/{network}_plots/{iqm_source}/perimg_plots/Metric_vs_Camerasetting"
    os.makedirs(analysis_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df["image"] = df["image"].str.replace(".tiff", "", regex=False)

    map_values, mtf50_values, c4_values, cmax_values, edge_snri_values = [], [], [], [], []
    valid_splits, snr_values_for_ISO = [], {}
    snri_data = {}
    snri_square_data = {}
    snri_square_dB_avg_values = []

    for idx, condition in enumerate(split_conditions):
        label = "_AND_".join(condition)
        matching_df = df[df["image"].apply(lambda x: all(k in x for k in condition))]

        if matching_df.empty:
            continue

        avg_map = matching_df["mAP50_95"].mean()
        map_values.append(avg_map)
        valid_splits.append(label)

        summary_file = label.replace(" ", "").replace(".", ".") + "_average_summary.json"
        summary_path = os.path.join(imatest_summary_path, summary_file)

        if not os.path.exists(summary_path):
            print(f"[!] Missing summary for: {label}")
            mtf50_values.append(None)
            c4_values.append(None)
            cmax_values.append(None)
            edge_snri_values.append(None)
            snri_square_dB_avg_values.append(None)
            continue

        with open(summary_path, "r") as f:
            summary = json.load(f)

        override_mtf = get_annotation(label, idx)
        if override_mtf and "Cy/Pxl" in override_mtf:
            mtf50_values.append(float(override_mtf.split()[0]))
        else:
            mtf50_values.append(summary.get("mtf50"))

        c4_values.append(summary.get("info_capacity_C_4_b_p"))
        cmax_values.append(summary.get("info_capacity_C_max_b_p"))

        edge_snri = summary.get("edge_SNRi_square", [])
        edge_snri_values.append(edge_snri[0] if edge_snri else None)

        if "ISO" in label or label in ["ON", "OFF"]:
            snr_val = summary.get("snr_dB_ISO15739_at_13pct_Lref")
            if snr_val is not None:
                snr_values_for_ISO[label] = snr_val

        snri_dB = summary.get("SNRi_square_dB", [])
        snri_x = summary.get("SNRi_box_width", [])
        if snri_dB and snri_x:
            snri_data[label] = (snri_x[:len(snri_dB)], snri_dB)
        snri_square_dB_avg_values.append(snri_dB[0] if snri_dB else None)

        snri_sq = summary.get("SNRi_square", [])
        if snri_sq and snri_x:
            snri_square_data[label] = (snri_x[:len(snri_sq)], snri_sq)

    # --- COMBINED PLOT ---
    plt.figure(figsize=(10, 8))
    plt.scatter(valid_splits, mtf50_values, label="MTF50_Y_horizontal", marker='o', s=80)
    plt.scatter(valid_splits, map_values, label="mAP@[0.50:0.95]", marker='^', s=160, color="k")
    plt.scatter(valid_splits, c4_values, label="C4", marker='s', s=80)
    plt.scatter(valid_splits, cmax_values, label="Cmax", marker='*', s=80)
    plt.scatter(valid_splits, edge_snri_values, label="Edge SNRi", marker='D', s=80)
    combined_values = map_values + mtf50_values + c4_values + cmax_values + edge_snri_values
    ymax = max([v for v in combined_values if v is not None]) * 1.05
    add_vertical_annotations(valid_splits, ymax, "combined")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(os.path.join(analysis_dir, f"{keyword}_metricscorr_plot.jpg"), dpi=300)
    plt.close()

    # --- SNRi Square dB Plot ---
    plt.figure(figsize=(10, 8))
    plt.scatter(valid_splits, snri_square_dB_avg_values, label="SNRi Square (dB per pixel^2)", marker='o', s=100, color='purple')
    ymax = max([v for v in snri_square_dB_avg_values if v is not None]) * 1
    add_vertical_annotations(valid_splits, ymax, "snri_square")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("SNRi Square (dB per pixel^2)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(os.path.join(analysis_dir, f"{keyword}_SNRi_square_dB_avg.jpg"), dpi=300)
    plt.close()

    # --- SNRi Square Curve Plot ---
    plt.figure(figsize=(10, 8))
    for split, (x, y) in snri_square_data.items():
        y_dB = [10 * np.log10(val) if val > 0 else float('nan') for val in y]
        plt.plot(x, y_dB, label=split.replace("AND", " & "), linewidth=2.0)
    plt.xlabel("SNRi Box Width", fontsize=12)
    plt.ylabel("SNRi Square (dB per pixel^2)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(os.path.join(analysis_dir, f"{keyword}_SNRiSquare_splits.jpg"), dpi=300)
    plt.close()
