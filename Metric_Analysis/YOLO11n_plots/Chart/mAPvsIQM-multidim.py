import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

# -------- CONFIGURATION --------
csv_path = "/home/colourlabgpu4/Kimia/Thesis/Object_Detection/YOLO11/per-image-validation/outputs/Full Dataset/per_image_results.csv"
summary_dir = "/home/colourlabgpu4/Kimia/Thesis/Metric_Analysis/Chart/Extracted"
output_dir = "/home/colourlabgpu4/Kimia/Thesis/Metric_Analysis/Chart/perimg_Plots/Multidimensional"
os.makedirs(output_dir, exist_ok=True)

sharpness_levels = {"Focused": 1.0, "Defocus1": 0.5, "Defocus2": 0.1}
cmap = plt.cm.plasma
iso_levels = ["100ISO", "1600ISO", "6400ISO", "25600ISO"]
class_keywords = ["car", "potted plant", "sports ball", "cup"]

iso_snr_labels = {
    100: "100 (34.9 dB)",
    1600: "1600 (25.2 dB)",
    6400: "6400 (19.5 dB)",
    25600: "25600 (12.8 dB)"
}


iqm_names = ["info_capacity_C_4_b_p", "info_capacity_C_max_b_p", "mtf50", "edge_SNRi_square"]
iqm_labels = {
    "info_capacity_C_4_b_p": "C4",
    "info_capacity_C_max_b_p": "Cmax",
    "mtf50": "MTF50_Y",
    "edge_SNRi_square": "Edge SNRi"
}

df = pd.read_csv(csv_path)
df["image"] = df["image"].str.replace(".tiff", "", regex=False)

def find_chart_summary(image_name):
    for iso in iso_levels:
        if iso in image_name:
            for focus in sharpness_levels:
                if focus in image_name:
                    fl = "18.0 mm" if "18.0 mm" in image_name else "55.0 mm"
                    filename = f"chart_{fl}_{iso}_+1EV_Dist1_OFF_{focus}_summary.json"
                    return os.path.join(summary_dir, filename)
    return None

def create_2d_plot(iqm_key, iso_filter=None, class_filter=None):
    plt.figure(figsize=(10, 8))
    all_iqm_vals, all_mAP_vals, sharpness_vals = [], [], []

    for _, row in df.iterrows():
        image_name = row["image"]
        if "+1EV" not in image_name:
            continue
        if iso_filter and iso_filter not in image_name:
            continue
        if class_filter and class_filter not in image_name:
            continue

        summary_path = find_chart_summary(image_name)
        if not summary_path or not os.path.exists(summary_path):
            continue
        with open(summary_path, "r") as f:
            data = json.load(f)

        val = data.get("noise_plot", {}).get("edge_SNRi_square", [None])[0] if iqm_key == "edge_SNRi_square" else data.get(iqm_key)
        if val is None:
            continue

        sharpness_key = next((k for k in sharpness_levels if k in image_name), None)
        if not sharpness_key:
            continue

        all_iqm_vals.append(val)
        all_mAP_vals.append(row["mAP50_95"])
        sharpness_vals.append(sharpness_levels[sharpness_key])

    if all_iqm_vals:
        xmax = max(all_iqm_vals)
        ymax = max(all_mAP_vals)
        plt.xlim(0, xmax + 0.1)
        plt.ylim(0, ymax * 1.1)
        sc = plt.scatter(all_iqm_vals, all_mAP_vals, c=sharpness_vals, cmap=cmap, edgecolors="k", s=300, alpha=0.6)
        ax = plt.gca()
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.1, vmax=1.0))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label("Sharpness")
        cbar.set_ticks(list(sharpness_levels.values()))
        cbar.set_ticklabels(list(sharpness_levels.keys()))
        plt.xlabel(iqm_labels[iqm_key])
        plt.ylabel("mAP")
        plt.grid(True)
        plt.tight_layout()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        class_tag = f"_{class_filter}" if class_filter else ""
        iso_tag = f"_{iso_filter}" if iso_filter else ""
        plt.savefig(os.path.join(output_dir, f"{iqm_labels[iqm_key]}_scatter{class_tag}{iso_tag}.jpg"), dpi=300)
        plt.close()

def create_3d_plot(iqm_key, iqm_label):
    iqm_vals_3d, map_vals_3d, iso_vals_3d, sharpness_vals = [], [], [], []

    for _, row in df.iterrows():
        image_name = row["image"]
        if "+1EV" not in image_name:
            continue
        iso = next((int(i.replace("ISO", "")) for i in iso_levels if i in image_name), None)
        if iso is None:
            continue
        summary_path = find_chart_summary(image_name)
        if not summary_path or not os.path.exists(summary_path):
            continue
        with open(summary_path, "r") as f:
            data = json.load(f)

        val = data.get("noise_plot", {}).get("edge_SNRi_square", [None])[0] if iqm_key == "edge_SNRi_square" else data.get(iqm_key)
        if val is None:
            continue

        sharpness_key = next((k for k in sharpness_levels if k in image_name), None)
        if not sharpness_key:
            continue

        iqm_vals_3d.append(val)
        map_vals_3d.append(row["mAP50_95"])
        iso_vals_3d.append(iso)
        sharpness_vals.append(sharpness_levels[sharpness_key])

    if iqm_vals_3d:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(iqm_vals_3d, map_vals_3d, iso_vals_3d,
                        c=sharpness_vals, cmap=cmap, s=300, alpha=0.6, edgecolors="k")
        ax.set_xlabel(iqm_label)
        ax.set_ylabel("mAP")
        ax.set_zlabel("ISO")
        iso_ticks = [100, 1600, 6400, 25600]
        ax.set_zticks(iso_ticks)
        ax.set_zticklabels([iso_snr_labels[iso] for iso in iso_ticks], fontsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
        cbar.set_label("Sharpness")
        cbar.set_ticks(list(sharpness_levels.values()))
        cbar.set_ticklabels(list(sharpness_levels.keys()))
        ax.view_init(elev=30, azim=-45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{iqm_label}_3D_mixed.jpg"), dpi=300)
        plt.close()

# === RUN FOR ALL IQMs ===
for iqm in iqm_names:
    create_3d_plot(iqm, iqm_labels[iqm])
    for iso in iso_levels:
        create_2d_plot(iqm, iso_filter=iso)
        for cls in class_keywords:
            create_2d_plot(iqm, iso_filter=iso, class_filter=cls)
