import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

# -------- CONFIGURATION --------
csv_path = "/home/colourlabgpu4/Kimia/Thesis/Object_Detection/YOLO11/per-image-validation/outputs/Full Dataset/per_image_results.csv"
average_metrics_dir = "/home/colourlabgpu4/Kimia/Thesis/Metric_Analysis/Dataset/Average Metrics"
extracted_metrics_dir = "/home/colourlabgpu4/Kimia/Thesis/Metric_Analysis/Dataset/Extracted"
output_dir = "/home/colourlabgpu4/Kimia/Thesis/Metric_Analysis/Dataset/perimg_Plots/Multidimensional-EV"
os.makedirs(output_dir, exist_ok=True)

use_only_tag = None

sharpness_levels = {"Focused": 1.0, "Defocus1": 0.5, "Defocus2": 0.1}
cmap = plt.cm.plasma

object_size_markers = {
    "18.0 mm_And_Dist1": 25,
    "18.0 mm_And_Dist2": 100,
    "55.0 mm_And_Dist1": 400,
    "55.0 mm_And_Dist2": 1000,
}

color_mapping_size_values = {
    "18.0 mm_And_Dist1": 0.1,
    "18.0 mm_And_Dist2": 0.4,
    "55.0 mm_And_Dist1": 0.7,
    "55.0 mm_And_Dist2": 1.0,
}

EV_levels = ["-3EV", "-2EV", "-1EV", "0EV", "+1EV"]
class_keywords = ["car", "potted plant", "sports ball", "cup"]

# IQMs
iqm_names = ["info_capacity_C_4_b_p", "info_capacity_C_max_b_p", "mtf50", "edge_SNRi_square"]
iqm_labels = {
    "info_capacity_C_4_b_p": "C4",
    "info_capacity_C_max_b_p": "Cmax",
    "mtf50": "MTF50_Y",
    "edge_SNRi_square": "Edge SNRi"
}

# Load CSV
df = pd.read_csv(csv_path)
df["image"] = df["image"].str.replace(".tiff", "", regex=False)


def create_plot(iqm_key, color_mode, ev_filter=None, class_filter=None):
    plt.figure(figsize=(10, 8))
    all_iqm_vals = []
    all_mAP_vals = []

    for _, row in df.iterrows():
        image_name = row["image"]
        if use_only_tag and use_only_tag not in image_name:
            continue

        if ev_filter and ev_filter not in image_name:
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

        try:
            with open(summary_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue

        if iqm_key == "edge_SNRi_square":
            val = data.get("noise_plot", {}).get("edge_SNRi_square", None)
            if isinstance(val, list) and len(val) > 0:
                iqm_val = val[0]
            else:
                continue
        else:
            iqm_val = data.get(iqm_key, None)

        if iqm_val is None or not isinstance(iqm_val, (int, float)):
            continue

        all_iqm_vals.append(iqm_val)
        all_mAP_vals.append(row["mAP50_95"])

        base_marker_size = 300
        color = cmap(sharpness_val)
        plt.scatter(iqm_val, row["mAP50_95"], c=[color], s=marker_size, alpha=0.6, edgecolors="k")

    if all_iqm_vals and all_mAP_vals:
        xmax = max(all_iqm_vals)
        ymax = max(all_mAP_vals)
        plt.xlim(0, xmax + 0.1)
        plt.ylim(0, ymax * 1.1)

    plt.xlabel(iqm_labels[iqm_key], fontsize=14)
    plt.ylabel("Average Precision", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    norm = plt.Normalize(vmin=min(sharpness_levels.values()), vmax=max(sharpness_levels.values()))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Sharpness", fontsize=12)
    cbar.set_ticks(list(sharpness_levels.values()))
    cbar.set_ticklabels(list(sharpness_levels.keys()))

    class_tag = f"_{class_filter}" if class_filter else ""
    ev_tag = f"_{ev_filter}" if ev_filter else ""
    save_path = os.path.join(output_dir, f"{iqm_labels[iqm_key]}_scatter{class_tag}{ev_tag}.jpg")
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot: {save_path}")
    plt.close()


def create_3d_plot_mixed(iqm_key, iqm_label):
    iqm_vals_3d = []
    map_vals_3d = []
    ev_vals_3d = []
    sharpness_vals = []
    marker_sizes = []

    for _, row in df.iterrows():
        image_name = row["image"]
        if use_only_tag and use_only_tag not in image_name:
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

        try:
            with open(summary_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue

        if iqm_key == "edge_SNRi_square":
            val = data.get("noise_plot", {}).get("edge_SNRi_square", [None])[0]
        else:
            val = data.get(iqm_key, None)

        if val is None or not isinstance(val, (float, int)):
            continue

        iqm_vals_3d.append(val)
        map_vals_3d.append(row["mAP50_95"])
        ev_vals_3d.append(ev_val)
        sharpness_vals.append(sharpness_val)
        marker_sizes.append(marker_size)

    if not iqm_vals_3d:
        print(f"[SKIPPED] No data for 3D plot: {iqm_key}")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(iqm_vals_3d, map_vals_3d, ev_vals_3d,
                    c=sharpness_vals, cmap=cmap,
                    s=marker_sizes, alpha=0.6, edgecolors="k")

    ax.set_xlabel(iqm_label)
    ax.set_ylabel("mAP")
    ax.set_zlabel("EV")
    ax.set_zticks([-3, -2, -1, 0, +1])

    cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label("Sharpness")
    cbar.set_ticks(list(sharpness_levels.values()))
    cbar.set_ticklabels(list(sharpness_levels.keys()))

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label=label,
               markerfacecolor='gray', markersize=np.sqrt(size / 5), alpha=0.5)
        for label, size in object_size_markers.items()
    ]
    ax.legend(handles=legend_handles, loc='upper left', title='Object Size')
    ax.view_init(elev=30, azim=-45)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{iqm_label}_3D_mixed.jpg")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] 3D mixed plot for {iqm_label} → {save_path}")


# === RUN FOR ALL IQMs ===
for iqm in iqm_names:
    create_3d_plot_mixed(iqm, iqm_labels[iqm])
    for ev in EV_levels:
        # Plot with all classes combined
        create_plot(iqm, "mixed", ev_filter=ev, class_filter=None)

        # Then plot each class separately
        for cls in class_keywords:
            create_plot(iqm, "mixed", ev_filter=ev, class_filter=cls)
