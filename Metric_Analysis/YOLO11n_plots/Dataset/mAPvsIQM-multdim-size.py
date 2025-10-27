import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

# CONFIGURATION
csv_path = "/home/colourlabgpu4/Kimia/Thesis/Object_Detection/YOLO11/per-image-validation/outputs/Full Dataset/per_image_results.csv"
extracted_metrics_dir = "/home/colourlabgpu4/Kimia/Thesis/Metric_Analysis/Dataset/Extracted+1"
output_dir = "/home/colourlabgpu4/Kimia/Thesis/Metric_Analysis/Dataset/perimg_Plots/Multidimensional-size"
os.makedirs(output_dir, exist_ok=True)

use_only_tag = "+1EV"

sharpness_levels = {"Focused": 1.0, "Defocus1": 0.5, "Defocus2": 0.1}
object_size_map = {
    "18.0 mm_And_Dist1": 1,
    "18.0 mm_And_Dist2": 2,
    "55.0 mm_And_Dist1": 3,
    "55.0 mm_And_Dist2": 4,
}
class_keywords = ["car", "sports ball", "potted plant", "cup"]

iqm_names = ["info_capacity_C_4_b_p", "info_capacity_C_max_b_p", "mtf50", "edge_SNRi_square"]
iqm_labels = {
    "info_capacity_C_4_b_p": "C4",
    "info_capacity_C_max_b_p": "Cmax",
    "mtf50": "MTF50_Y",
    "edge_SNRi_square": "Edge SNRi"
}
cmap = plt.cm.plasma

df = pd.read_csv(csv_path)
df["image"] = df["image"].str.replace(".tiff", "", regex=False)

def get_size_level(name):
    for k, v in object_size_map.items():
        parts = k.split("_And_")
        if all(p in name for p in parts):
            return v
    return None

def get_sharpness(name):
    return next((v for k, v in sharpness_levels.items() if k in name), None)

def extract_iqm(image_name, iqm_key):
    path = os.path.join(extracted_metrics_dir, image_name + "_summary.json")
    if not os.path.exists(path): return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if iqm_key == "edge_SNRi_square":
            return data.get("noise_plot", {}).get("edge_SNRi_square", [None])[0]
        return data.get(iqm_key, None)
    except Exception:
        return None

def create_3d_plot(iqm_key, iqm_label):
    iqm_vals, map_vals, size_vals, sharpness_vals = [], [], [], []

    for _, row in df.iterrows():
        name = row["image"]
        if use_only_tag not in name:
            continue
        size = get_size_level(name)
        sharp = get_sharpness(name)
        if size is None or sharp is None:
            continue
        val = extract_iqm(name, iqm_key)
        if val is None: continue

        iqm_vals.append(val)
        map_vals.append(row["mAP50_95"])
        size_vals.append(size)
        sharpness_vals.append(sharp)

    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(iqm_vals, map_vals, size_vals, c=sharpness_vals, cmap=cmap, s=200, edgecolors="k", alpha=0.6)

    ax.set_xlabel(iqm_label)
    ax.set_ylabel("mAP")
    ax.set_zlabel("Object Size")
    ax.set_zticks([1, 2, 3, 4])
    ax.set_zticklabels(["Size1", "Size2", "Size3", "Size4"])
    ax.view_init(elev=30, azim=-45)

    cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label("Sharpness")
    cbar.set_ticks(list(sharpness_levels.values()))
    cbar.set_ticklabels(list(sharpness_levels.keys()))

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(os.path.join(output_dir, f"{iqm_label}_3D_Size.jpg"), dpi=300)
    plt.close()

def create_2d_slices(iqm_key, iqm_label, class_filter=None):
    for key, lvl in object_size_map.items():
        parts = key.split("_And_")
        iqm_vals, map_vals, sharpness_vals = [], [], []

        for _, row in df.iterrows():
            name = row["image"]
            if use_only_tag not in name or not all(p in name for p in parts):
                continue
            if class_filter and class_filter not in name:
                continue
            sharp = get_sharpness(name)
            val = extract_iqm(name, iqm_key)
            if sharp is None or val is None: continue

            iqm_vals.append(val)
            map_vals.append(row["mAP50_95"])
            sharpness_vals.append(sharp)

        if not iqm_vals:
            continue

        plt.figure(figsize=(10, 8), constrained_layout=True)
        plt.scatter(iqm_vals, map_vals, c=sharpness_vals, cmap=cmap, s=200, edgecolors="k", alpha=0.6)
        plt.xlabel(iqm_label)
        plt.ylabel("mAP")
        #plt.title(f"{iqm_label} vs mAP – {key}" + (f" – {class_filter}" if class_filter else ""))
        plt.grid(True)

        norm = plt.Normalize(vmin=min(sharpness_levels.values()), vmax=max(sharpness_levels.values()))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        ax = plt.gca()
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Sharpness")
        cbar.set_ticks(list(sharpness_levels.values()))
        cbar.set_ticklabels(list(sharpness_levels.keys()))

        class_tag = f"_{class_filter.replace(' ', '')}" if class_filter else ""
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.savefig(os.path.join(output_dir, f"{iqm_label}_Size{lvl}{class_tag}_2D.jpg"), dpi=300)
        plt.close()

# === RUN ===
for iqm in iqm_names:
    create_3d_plot(iqm, iqm_labels[iqm])
    create_2d_slices(iqm, iqm_labels[iqm])
    for cls in class_keywords:
        create_2d_slices(iqm, iqm_labels[iqm], class_filter=cls)
