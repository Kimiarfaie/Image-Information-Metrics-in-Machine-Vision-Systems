import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# --- CONFIGURATION ---
networks = ["YOLO11n", "YOLO11m", "SSD", "FasterRCNN-mobilenet", "FasterRCNN"]
keyword = "NOISE-ONLY"  # or "sharpness", "ISO", "EV", "size"
xlabel = "SNR (dB)"  # Adjusted for clarity
iqm_source = "Dataset"

split_conditions = [["100ISO", "Focused"], ["1600ISO", "Focused"], ["6400ISO", "Focused"], ["25600ISO", "Focused"]]

#split_conditions = [["18.0 mm", "Dist1"], ["18.0 mm", "Dist2"], ["55.0 mm", "Dist1"], ["55.0 mm", "Dist2"]]
#split_conditions = [["100ISO"], ["1600ISO"], ["6400ISO"], ["25600ISO"]]
#split_conditions = [["Focused"], ["Defocus1"], ["Defocus2"]]
#split_conditions = [["-3EV"], ["-2EV"], ["-1EV"], ["0EV"], ["+1EV"]]

imatest_summary_path = f"/home/colourlabgpu4/Kimia/Thesis/Metric_Analysis/Metrics/{iqm_source}/Average Metrics+1"
use_only_tag = None

custom_annotations = {
    "Chart": {
        "sharpness": ["0.151 Cy/Pxl", "0.033 Cy/Pxl", "0.017 Cy/Pxl"]
    },
    "Dataset": {
        "sharpness": ["0.155 Cy/Pxl", "0.042 Cy/Pxl", "0.024 Cy/Pxl"],
        "EV": ["8.21", "12.30", "18.04", "25.94", "35.71"],
        "size": ["640", "1206", "5198", "9770"]
    }
}

def get_annotation(split, idx):
    if keyword in custom_annotations.get(iqm_source, {}):
        values = custom_annotations[iqm_source][keyword]
        if idx < len(values):
            return values[idx]
    return None

def get_xvalues_from_keyword_v2(keyword, split_conditions, mtf50_values, snr_values_for_ISO, custom_annotations, iqm_source):
    x_vals = []
    if keyword == "NOISE-ONLY":
        for condition in split_conditions:
            label = "_AND_".join(condition)
            x_vals.append(snr_values_for_ISO.get(label, None))
    elif keyword == "sharpness":
        values = custom_annotations.get(iqm_source, {}).get("sharpness", [])
        for idx in range(len(split_conditions)):
            if idx < len(values):
                try:
                    mtf_val = float(values[idx].split()[0])
                    x_vals.append(mtf_val)
                except:
                    x_vals.append(mtf50_values[idx] if idx < len(mtf50_values) else None)
            else:
                x_vals.append(mtf50_values[idx] if idx < len(mtf50_values) else None)
    elif keyword == "EV":
        values = custom_annotations.get(iqm_source, {}).get("EV", [])
        for idx in range(len(split_conditions)):
            try:
                x_vals.append(float(values[idx]))
            except:
                x_vals.append(None)
    elif keyword == "size":
        values = custom_annotations.get(iqm_source, {}).get("size", [])
        for idx in range(len(split_conditions)):
            try:
                x_vals.append(float(values[idx]))
            except:
                x_vals.append(None)
    else:
        x_vals = list(range(1, len(split_conditions) + 1))
    return x_vals


for network in networks:
    print(f"Processing: {network}")
    csv_path = f"/home/colourlabgpu4/Kimia/Thesis/Object_Detection/{network}/per-image-validation/outputs/Full Dataset/per_image_results.csv"
    analysis_dir = f"/home/colourlabgpu4/Kimia/Thesis/Metric_Analysis/{network}_plots/{iqm_source}/perimg_Plots/New_Metric_Camerasetting"
    os.makedirs(analysis_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df["image"] = df["image"].str.replace(".tiff", "", regex=False)
    if use_only_tag:
        df = df[df["image"].str.contains(re.escape(use_only_tag))]

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
        if override_mtf and "Cyc/Pxl" in override_mtf:
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

    x_values = get_xvalues_from_keyword_v2(keyword, split_conditions, mtf50_values, snr_values_for_ISO, custom_annotations, iqm_source)

    # --- COMBINED PLOT ---
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.tick_params(labelsize=20)
    plt.scatter(x_values, mtf50_values, label="MTF50_Y_horizontal", marker='o', s=200)
    plt.scatter(x_values, map_values, label="mAP@[0.50:0.95]", marker='^', s=300, color="k")
    plt.scatter(x_values, c4_values, label="C4", marker='s', s=200)
    plt.scatter(x_values, cmax_values, label="Cmax", marker='*', s=300)
    plt.scatter(x_values, edge_snri_values, label="Edge SNRi", marker='D', s=200)
    plt.xlabel(xlabel, fontsize=22)
    plt.ylabel("Metric Value", fontsize=22)
    plt.legend(fontsize=22)
    plt.grid(True)
    plt.tight_layout()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(os.path.join(analysis_dir, f"{keyword}_metricscorr_plot.jpg"), dpi=300)
    plt.close()

    # --- SNRi Square dB Plot ---
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.tick_params(labelsize=20)
    plt.scatter(x_values, snri_square_dB_avg_values, label="SNRi Square (dB)", marker='o', s=300, color='purple')
    plt.xlabel(xlabel, fontsize=22)
    plt.ylabel("SNRi Square (dB)", fontsize=22)
    plt.legend(fontsize=22)
    plt.grid(True)
    plt.tight_layout()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(os.path.join(analysis_dir, f"{keyword}_SNRi_square_dB_avg.jpg"), dpi=300)
    plt.close()