import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

# -------- CONFIGURATION --------
iqm_source = "Dataset"
networks = ["FasterRCNN", "YOLO11m", "YOLO11n", "SSD", "FasterRCNN-mobilenet"]
sharpness_levels = {"Focused": 1.0, "Defocus1": 0.5, "Defocus2": 0.1}
cmap = plt.cm.plasma

object_size_markers = {
    "18.0 mm_And_Dist1": 25,
    "18.0 mm_And_Dist2": 100,
    "55.0 mm_And_Dist1": 400,
    "55.0 mm_And_Dist2": 1000,
}

EV_levels = ["-3EV", "-2EV", "-1EV", "0EV", "+1EV"]
iso_levels = ["100ISO", "1600ISO", "6400ISO", "25600ISO"]
class_keywords = ["car", "potted plant", "sports ball", "cup"]

iqm_keys_labels = {
    "SNRi_square_dB": "SNRi Square (dB)",
    "info_capacity_C_4_b_p": "C4"
}

def load_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    df["image"] = df["image"].str.replace(".tiff", "", regex=False)
    return df

def extract_iqm_value(data, iqm_key):
    if iqm_key == "SNRi_square_dB":
        val = data.get("noise_plot", {}).get("SNRi_square", [None])[0]
        if val is not None and val > 0:
            return 10 * np.log10(val)
        else:
            return None
    return data.get(iqm_key, None)

def create_plot(df, iqm_key, iqm_label, ev_filter, iso_filter, class_filter, extracted_metrics_dir, output_dir):
    fig, ax = plt.subplots(figsize=(10, 8))

    for _, row in df.iterrows():
        image_name = row["image"]
        if ev_filter not in image_name:
            continue

        if iso_filter and iso_filter not in image_name:
            continue

        if class_filter and class_filter not in image_name:
            continue

        sharpness_key = next((k for k in sharpness_levels if k in image_name), None)
        if not sharpness_key:
            continue
        sharpness_val = sharpness_levels[sharpness_key]

        size_key = next((k for k in object_size_markers if all(p in image_name for p in k.split("_And_"))), None)
        if not size_key:
            continue
        marker_size = object_size_markers[size_key]

        summary_file = image_name + "_summary.json"
        summary_path = os.path.join(extracted_metrics_dir, summary_file)
        if not os.path.exists(summary_path):
            continue

        with open(summary_path, "r") as f:
            data = json.load(f)

        iqm_val = extract_iqm_value(data, iqm_key)
        if iqm_val is None:
            continue

        color = cmap(sharpness_val)
        plt.scatter(iqm_val, row["mAP50_95"], c=[color], s=marker_size, alpha=0.6, edgecolors="k")

    plt.xlabel(iqm_label, fontsize=22)
    plt.ylabel("Average Precision", fontsize=22)
    plt.grid(True)
    plt.tight_layout()        
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=14) #2d
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    norm = plt.Normalize(vmin=min(sharpness_levels.values()), vmax=max(sharpness_levels.values()))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)

    cbar.set_ticks([sharpness_levels[k] for k in ["Defocus2", "Defocus1", "Focused"]])
    cbar.set_ticklabels(["Defocus2", "Defocus1", "Focused"])
    cbar.set_label("Sharpness", fontsize=18)

    iso_tag = f"_{iso_filter}" if iso_filter else "_allISO"
    class_tag = f"_{class_filter}" if class_filter else ""
    save_path = os.path.join(output_dir, f"{iso_tag}_{iqm_label.replace(' ', '')}_scatter{class_tag}_{ev_filter}.jpg")
    plt.savefig(save_path, dpi=300)
    plt.close()

def create_3d_plot(df, iqm_key, iqm_label, iso_filter, extracted_metrics_dir, output_dir):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for _, row in df.iterrows():
        image_name = row["image"]

        if iso_filter and iso_filter not in image_name:
            continue

        ev_val = next((int(ev.replace("EV", "")) for ev in EV_levels if ev in image_name), None)
        if ev_val is None:
            continue

        sharpness_key = next((k for k in sharpness_levels if k in image_name), None)
        if not sharpness_key:
            continue
        sharpness_val = sharpness_levels[sharpness_key]

        size_key = next((k for k in object_size_markers if all(p in image_name for p in k.split("_And_"))), None)
        if not size_key:
            continue
        marker_size = object_size_markers[size_key]

        summary_file = image_name + "_summary.json"
        summary_path = os.path.join(extracted_metrics_dir, summary_file)
        if not os.path.exists(summary_path):
            continue

        with open(summary_path, "r") as f:
            data = json.load(f)

        iqm_val = extract_iqm_value(data, iqm_key)
        if iqm_val is None:
            continue

        ax.scatter(iqm_val, row["mAP50_95"], ev_val, c=[cmap(sharpness_val)], s=marker_size, alpha=0.6, edgecolors="k")

    ax.set_xlabel(iqm_label, fontsize=18)
    ax.set_ylabel("mAP", fontsize=18)
    ax.set_zlabel("EV", fontsize=18)
    ax.set_zticks([-3, -2, -1, 0, +1])
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='z', labelsize=12)

    norm = plt.Normalize(vmin=min(sharpness_levels.values()), vmax=max(sharpness_levels.values()))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
    cbar.set_ticks([sharpness_levels[k] for k in ["Defocus2", "Defocus1", "Focused"]])
    cbar.set_ticklabels(["Defocus2", "Defocus1", "Focused"])
    cbar.set_label("Sharpness", fontsize=18)

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label=label,
               markerfacecolor='gray', markersize=np.sqrt(size / 5), alpha=0.5)
        for label, size in object_size_markers.items()
    ]
    ax.legend(handles=legend_handles, title='Object Size', fontsize=15, title_fontsize=16, loc='upper center', bbox_to_anchor=(0.2, 0.05), ncol=2, frameon=True)
    iso_tag = iso_filter if iso_filter else "allISO"
    save_path = os.path.join(output_dir, f"{iso_tag}_{iqm_label.replace(' ', '')}_3D.jpg")
    plt.savefig(save_path, dpi=300)
    plt.close()

# === MAIN LOOP ===
for network in networks:
    csv_path = f"/home/colourlabgpu4/Kimia/Thesis/Object_Detection/{network}/per-image-validation/outputs/Full Dataset/per_image_results.csv"
    extracted_metrics_dir = f"/home/colourlabgpu4/Kimia/Thesis/Metric_Analysis/Metrics/{iqm_source}/Extracted"
    output_dir = f"/home/colourlabgpu4/Kimia/Thesis/Metric_Analysis/{network}_plots/{iqm_source}/perimg_Plots/Multidimensional_EV"
    os.makedirs(output_dir, exist_ok=True)

    df = load_dataframe(csv_path)

    for iqm_key, iqm_label in iqm_keys_labels.items():
        for iso in iso_levels + [None]:
            create_3d_plot(df, iqm_key, iqm_label, iso, extracted_metrics_dir, output_dir)
            for ev in EV_levels:
                create_plot(df, iqm_key, iqm_label, ev, iso, None, extracted_metrics_dir, output_dir)
    print(f"[DONE] Finished all plots for network: {network}\n")