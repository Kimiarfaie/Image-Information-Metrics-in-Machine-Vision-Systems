import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

# -------- CONFIG --------
networks = ["YOLO11n", "YOLO11m", "SSD", "FasterRCNN-mobilenet", "FasterRCNN"]
iqm_source = "Chart"
sharpness_levels = {"Focused": 1.0, "Defocus1": 0.5, "Defocus2": 0.1}
iso_levels = ["100ISO", "1600ISO", "6400ISO", "25600ISO"]
class_keywords = ["car", "potted plant", "sports ball", "cup"]

iso_snr_labels = {
    100: "100 (34.9 dB)",
    1600: "1600 (25.2 dB)",
    6400: "6400 (19.5 dB)",
    25600: "25600 (12.8 dB)"
}

iqm_names = ["info_capacity_C_4_b_p", "mtf50", "SNRi_square_dB"]
iqm_labels = {
    "info_capacity_C_4_b_p": "C4",
    "mtf50": "MTF50_Y",
    "SNRi_square_dB": "SNRi Square (dB)"
}

# -------- FUNCTIONS --------
def find_chart_summary(image_name, summary_dir):
    for iso in iso_levels:
        if iso in image_name:
            for focus in sharpness_levels:
                if focus in image_name:
                    fl = "18.0 mm" if "18.0 mm" in image_name else "55.0 mm"
                    filename = f"chart_{fl}_{iso}_+1EV_Dist1_OFF_{focus}_summary.json"
                    return os.path.join(summary_dir, filename)
    return None

def create_2d_plot(df, iqm_key, iqm_label, summary_dir, output_dir, iso_filter=None, class_filter=None):
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

        summary_path = find_chart_summary(image_name, summary_dir)
        if not summary_path or not os.path.exists(summary_path):
            continue
        with open(summary_path, "r") as f:
            data = json.load(f)

        if iqm_key == "SNRi_square_dB":
            val = data.get("noise_plot", {}).get("SNRi_square", [None])[0]
            if val is not None and val > 0:
                val = 10 * np.log10(val)
            else:
                continue
        else:
            val = data.get(iqm_key, None)

        if val is None:
            continue

        sharpness_key = next((k for k in sharpness_levels if k in image_name), None)
        if not sharpness_key:
            continue

        all_iqm_vals.append(val)
        all_mAP_vals.append(row["mAP50_95"])
        sharpness_vals.append(sharpness_levels[sharpness_key])

    if all_iqm_vals:
        plt.scatter(all_iqm_vals, all_mAP_vals, c=sharpness_vals, cmap=cm.plasma, edgecolors="k", s=300, alpha=0.6)
        sm = plt.cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(vmin=0.1, vmax=1.0))
        sm.set_array([])
        ax = plt.gca()
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Sharpness", fontsize=18)
        cbar.set_ticks(list(sharpness_levels.values()))
        cbar.set_ticklabels(list(sharpness_levels.keys()))
        plt.xlabel(iqm_label, fontsize=22)
        plt.ylabel("mAP", fontsize=22)
        plt.grid(True)
        plt.tight_layout()
        ax.tick_params(axis='both', labelsize=14) #2d
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        class_tag = f"_{class_filter}" if class_filter else ""
        iso_tag = f"_{iso_filter}" if iso_filter else ""
        plt.savefig(os.path.join(output_dir, f"{iqm_label}_scatter{class_tag}{iso_tag}.jpg"), dpi=300)
        plt.close()

def create_3d_plot(df, iqm_key, iqm_label, summary_dir, output_dir):
    iqm_vals_3d, map_vals_3d, iso_vals_3d, sharpness_vals = [], [], [], []

    for _, row in df.iterrows():
        image_name = row["image"]
        if "+1EV" not in image_name:
            continue
        iso = next((int(i.replace("ISO", "")) for i in iso_levels if i in image_name), None)
        if iso is None:
            continue
        summary_path = find_chart_summary(image_name, summary_dir)
        if not summary_path or not os.path.exists(summary_path):
            continue
        with open(summary_path, "r") as f:
            data = json.load(f)

        if iqm_key == "SNRi_square_dB":
            val = data.get("noise_plot", {}).get("SNRi_square", [None])[0]
            if val is not None and val > 0:
                val = 10 * np.log10(val)
            else:
                continue
        else:
            val = data.get(iqm_key, None)

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
                        c=sharpness_vals, cmap=cm.plasma, s=300, alpha=0.6, edgecolors="k")
        ax.set_xlabel(iqm_label, fontsize=18)
        ax.set_ylabel("mAP", fontsize=18)
        ax.set_zlabel("ISO", fontsize=18)
        iso_ticks = [100, 1600, 6400, 25600]
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='z', labelsize=12)
        ax.set_zticks(iso_ticks)
        ax.set_zticklabels([iso_snr_labels[iso] for iso in iso_ticks], fontsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
        cbar.set_label("Sharpness", fontsize=18)
        cbar.set_ticks(list(sharpness_levels.values()))
        cbar.set_ticklabels(list(sharpness_levels.keys()))
        ax.view_init(elev=30, azim=-45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{iqm_label}_3D_mixed.jpg"), dpi=300)
        plt.close()

# -------- MAIN LOOP --------
for network in networks:
    print(f"[INFO] Processing network: {network}")
    csv_path = f"/home/colourlabgpu4/Kimia/Thesis/Object_Detection/{network}/per-image-validation/outputs/Full Dataset/per_image_results.csv"
    summary_dir = f"/home/colourlabgpu4/Kimia/Thesis/Metric_Analysis/Metrics/{iqm_source}/Extracted"
    output_dir = f"/home/colourlabgpu4/Kimia/Thesis/Metric_Analysis/{network}_plots/{iqm_source}/perimg_Plots/Multidimensional"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df["image"] = df["image"].str.replace(".tiff", "", regex=False)

    for iqm_key in iqm_names:
        iqm_label = iqm_labels[iqm_key]
        create_3d_plot(df, iqm_key, iqm_label, summary_dir, output_dir)
        for iso in iso_levels:
            create_2d_plot(df, iqm_key, iqm_label, summary_dir, output_dir, iso_filter=iso)
            for cls in class_keywords:
                create_2d_plot(df, iqm_key, iqm_label, summary_dir, output_dir, iso_filter=iso, class_filter=cls)
