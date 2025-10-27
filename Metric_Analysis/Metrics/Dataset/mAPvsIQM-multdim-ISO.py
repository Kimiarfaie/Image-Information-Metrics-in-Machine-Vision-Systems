import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

# -------- CONFIGURATION --------
iqm_source = "Dataset"
use_only_tag = "+1EV"

sharpness_levels = {"Focused": 1.0, "Defocus1": 0.5, "Defocus2": 0.1}
cmap = plt.cm.plasma

iso_snr_labels = {
    100: "100 (36.1 dB)",
    1600: "1600 (29.6 dB)",
    6400: "6400 (24.5 dB)",
    25600: "25600 (20.0 dB)"
}

object_size_markers = {
    "18.0 mm_And_Dist1": 25,
    "18.0 mm_And_Dist2": 100,
    "55.0 mm_And_Dist1": 400,
    "55.0 mm_And_Dist2": 1000,
}

iso_levels = ["100ISO", "1600ISO", "6400ISO", "25600ISO"]
class_keywords = ["car", "potted plant", "sports ball", "cup"]

# IQMs
iqm_names = ["info_capacity_C_4_b_p", "info_capacity_C_max_b_p", "mtf50", "edge_SNRi_square", "SNRi_square_dB"]
iqm_labels = {
    "info_capacity_C_4_b_p": "C4",
    "info_capacity_C_max_b_p": "Cmax",
    "mtf50": "MTF50_Y",
    "edge_SNRi_square": "Edge SNRi",
    "SNRi_square_dB": "SNRi square (dB)"
}

networks = ["FasterRCNN", "YOLO11m", "YOLO11n", "SSD", "FasterRCNN-mobilenet"]

for network in networks:
    print(f"\n[INFO] Processing network: {network}")

    csv_path = f"/home/colourlabgpu4/Kimia/Thesis/Object_Detection/{network}/per-image-validation/outputs/Full Dataset/per_image_results.csv"
    extracted_metrics_dir = f"/home/colourlabgpu4/Kimia/Thesis/Metric_Analysis/Metrics/{iqm_source}/Extracted+1"
    output_dir = f"/home/colourlabgpu4/Kimia/Thesis/Metric_Analysis/{network}_plots/{iqm_source}/perimg_Plots/Multidimensional_ISO"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df["image"] = df["image"].str.replace(".tiff", "", regex=False)

    def create_plot(iqm_key, color_mode, iso_filter=None, class_filter=None):
        plt.figure(figsize=(10, 8))
        all_iqm_vals = []
        all_mAP_vals = []

        for _, row in df.iterrows():
            image_name = row["image"]
            if use_only_tag and use_only_tag not in image_name:
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
            elif iqm_key == "SNRi_square_dB":
                val = data.get("noise_plot", {}).get("SNRi_square", [None])[0]
                if val is not None and val > 0:
                    iqm_val = 10 * np.log10(val)
                else:
                    continue
            else:
                iqm_val = data.get(iqm_key, None)

            if iqm_val is None or not isinstance(iqm_val, (int, float)):
                continue

            all_iqm_vals.append(iqm_val)
            all_mAP_vals.append(row["mAP50_95"])
            color = cmap(sharpness_val)
            plt.scatter(iqm_val, row["mAP50_95"], c=[color], s=marker_size, alpha=0.6, edgecolors="k")

        if all_iqm_vals and all_mAP_vals:
            xmax = max(all_iqm_vals)
            ymax = max(all_mAP_vals)
            plt.xlim(0, xmax + 0.1)
            plt.ylim(0, ymax * 1.1)

        plt.xlabel(iqm_labels[iqm_key], fontsize=22) #2d
        plt.ylabel("Average Precision", fontsize=22) #2d
        plt.grid(True)
        plt.tight_layout()
        ax = plt.gca()
        ax.tick_params(axis='both', labelsize=14) #2d
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        norm = plt.Normalize(vmin=min(sharpness_levels.values()), vmax=max(sharpness_levels.values()))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Sharpness", fontsize=18)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_ticks(list(sharpness_levels.values()))
        cbar.set_ticklabels(list(sharpness_levels.keys()))

        class_tag = f"_{class_filter}" if class_filter else ""
        iso_tag = f"_{iso_filter}" if iso_filter else ""
        save_path = os.path.join(output_dir, f"{iqm_labels[iqm_key]}_scatter{class_tag}{iso_tag}.jpg")
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot: {save_path}")
        plt.close()

    def create_3d_plot_mixed(iqm_key, iqm_label):
        iqm_vals_3d = []
        map_vals_3d = []
        iso_vals_3d = []
        sharpness_vals = []
        marker_sizes = []

        for _, row in df.iterrows():
            image_name = row["image"]
            if use_only_tag and use_only_tag not in image_name:
                continue

            iso_val = next((int(iso.replace("ISO", "")) for iso in iso_levels if iso in image_name), None)
            if iso_val is None:
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
            elif iqm_key == "SNRi_square_dB":
                val = data.get("noise_plot", {}).get("SNRi_square", [None])[0]
                if val is not None and val > 0:
                    val = 10 * np.log10(val)
                else:
                    continue
            else:
                val = data.get(iqm_key, None)

            if val is None or not isinstance(val, (float, int)):
                continue

            iqm_vals_3d.append(val)
            map_vals_3d.append(row["mAP50_95"])
            iso_vals_3d.append(iso_val)
            sharpness_vals.append(sharpness_val)
            marker_sizes.append(marker_size)

        if not iqm_vals_3d:
            print(f"[SKIPPED] No data for 3D plot: {iqm_key}")
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(iqm_vals_3d, map_vals_3d, iso_vals_3d,
                        c=sharpness_vals, cmap=cmap,
                        s=marker_sizes, alpha=0.6, edgecolors="k")

        ax.set_xlabel(iqm_label, fontsize=18) #3d
        ax.set_ylabel("mAP", fontsize=18) #3d
        ax.set_zlabel("ISO", fontsize=18) #3d
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='z', labelsize=12)
        iso_ticks = [100, 1600, 6400, 25600]
        ax.set_zticks(iso_ticks)
        ax.set_zticklabels([iso_snr_labels[iso] for iso in iso_ticks], fontsize=14)
        cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
        cbar.set_label("Sharpness", fontsize=18) #here
        cbar.ax.tick_params(labelsize=14)
        cbar.set_ticks(list(sharpness_levels.values()))
        cbar.set_ticklabels(list(sharpness_levels.keys()))

        legend_handles = [
            Line2D([0], [0], marker='o', color='w', label=label,
                   markerfacecolor='gray', markersize=np.sqrt(size / 5), alpha=0.5)
            for label, size in object_size_markers.items()
        ]
        ax.legend(handles=legend_handles, title='Object Size', fontsize=16, title_fontsize=16, loc='upper left', bbox_to_anchor=(0, 0), ncol=2, frameon=True)
        ax.view_init(elev=30, azim=-45)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{iqm_label}_3D_mixed.jpg")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[SAVED] 3D mixed plot for {iqm_label} → {save_path}")

    for iqm in iqm_names:
        create_3d_plot_mixed(iqm, iqm_labels[iqm])
        for iso in iso_levels:
            create_plot(iqm, "mixed", iso_filter=iso, class_filter=None)
            for cls in class_keywords:
                create_plot(iqm, "mixed", iso_filter=iso, class_filter=cls)
