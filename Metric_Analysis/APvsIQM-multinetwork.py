import os
import json
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
networks = ["YOLO11n", "YOLO11m", "SSD", "FasterRCNN-mobilenet", "FasterRCNN"]
iqm_source = "Chart"
keyword = "sharpness"
splits = ["Focused", "Defocus1", "Defocus2"]
#splits = ["100ISO", "1600ISO", "6400ISO", "25600ISO"]
#splits = ["18.0 mm_AND_Dist1", "18.0 mm_AND_Dist2", "55.0 mm_AND_Dist1", "55.0 mm_AND_Dist2"]
#splits = ["-3EV", "-2EV", "-1EV", "0EV", "+1EV"]

network_colors = {
    "YOLO11m": "orange",
    "YOLO11n": "green",
    "SSD": "purple",
    "FasterRCNN-mobilenet": "red",
    "FasterRCNN": "royalblue"
}

imatest_summary_path = f"/home/colourlabgpu4/Kimia/Thesis/Metric_Analysis/Metrics/{iqm_source}/Average Metrics"

# --- PRELOAD IQMs ---
iqm_by_split = {}
for split in splits:
    summary_file = split.replace(" ", "").replace(".", ".") + "_average_summary.json"
    summary_path = os.path.join(imatest_summary_path, summary_file)
    if not os.path.exists(summary_path):
        continue

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    iqm_by_split[split] = {
        "mtf50": summary.get("mtf50"),
        "c4": summary.get("info_capacity_C_4_b_p"),
        "cmax": summary.get("info_capacity_C_max_b_p"),
        "edge_snri": summary.get("edge_SNRi_square", [None])[0],
        "snri_square_db": summary.get("SNRi_square_dB", [None])[0]
    }

# --- INIT STORAGE FOR PLOTTING ---
metric_names = ["mtf50", "c4", "cmax", "edge_snri", "snri_square_db"]
metric_labels = {
    "mtf50": "MTF50_Y_horizontal",
    "c4": "Info Capacity C4",
    "cmax": "Info Capacity Cmax",
    "edge_snri": "Edge SNRi",
    "snri_square_db": "SNRi Square (dB)"
}
network_data = {metric: {} for metric in metric_names}

# --- COLLECT mAP FOR EACH NETWORK ---
for network_name in networks:
    per_image_dir = f"/home/colourlabgpu4/Kimia/Thesis/Object_Detection/{network_name}/per-image-validation/outputs/Full Dataset/per_image_results"
    csv_path = f"{per_image_dir}.csv"
    if not os.path.exists(csv_path):
        continue

    with open(csv_path, 'r') as f:
        lines = f.readlines()[1:]
    image_map_dict = {
        line.split(',')[0].replace(".tiff", ""): float(line.split(',')[1])
        for line in lines
    }

    map_values_by_split = []
    for split in splits:
        matching_imgs = [k for k in image_map_dict if all(part in k for part in split.split("_AND_")) or split in k]
        if not matching_imgs:
            continue
        avg_map = np.mean([image_map_dict[k] for k in matching_imgs])
        map_values_by_split.append(avg_map)

    for metric in metric_names:
        iqm_vals = [iqm_by_split[split][metric] for split in splits if split in iqm_by_split and iqm_by_split[split][metric] is not None]
        if len(iqm_vals) == len(map_values_by_split):
            network_data[metric][network_name] = (iqm_vals, map_values_by_split)

# --- PLOT EACH IQM ACROSS ALL NETWORKS ---
output_dir = f"/home/colourlabgpu4/Kimia/Thesis/Metric_Analysis/Network_Comparison_APvsIQM/{iqm_source}"
os.makedirs(output_dir, exist_ok=True)

for metric in metric_names:
    plt.figure(figsize=(8, 6))

    # Draw split markers (based on first available network)
    base_network = next(iter(network_data[metric].values()), None)
    if base_network:
        x_vals_base, _ = base_network
        for i, x in enumerate(x_vals_base):
            if i < len(splits):
                plt.axvline(x=x, linestyle='dashed', color='gray', linewidth=0.5, alpha=0.7)
                plt.text(x, 1.02, splits[i], rotation=0, fontsize=8, color='black',
                         ha='center', va='bottom', transform=plt.gca().get_xaxis_transform())

    for network_name, (x_vals, y_vals) in network_data[metric].items():
        color = network_colors.get(network_name, None)
        plt.scatter(x_vals, y_vals, label=network_name, s=30, color=color)

        if len(x_vals) >= 2:
            try:
                if metric == "snri_square_db":
                    coeffs = np.polyfit(x_vals, y_vals, deg=1)
                    fit_fn = np.poly1d(coeffs)
                    x_fit = np.linspace(min(x_vals), max(x_vals), 100)
                    y_fit = fit_fn(x_fit)
                    y_pred = fit_fn(x_vals)
                    plt.plot(x_fit, y_fit, color=color, linestyle='--', linewidth=1.5)
                    plt.text(x_fit[-1], y_fit[-1], f"R²={1 - np.sum((np.array(y_vals) - y_pred) ** 2) / np.sum((np.array(y_vals) - np.mean(y_vals)) ** 2):.2f}",
                            color=color, fontsize=9, verticalalignment='bottom', horizontalalignment='left')
                else:
                    log_x = np.log10(x_vals)
                    coeffs = np.polyfit(log_x, y_vals, deg=1)
                    fit_fn = np.poly1d(coeffs)
                    x_fit = np.linspace(min(log_x), max(log_x), 100)
                    y_fit = fit_fn(x_fit)
                    y_pred = fit_fn(log_x)
                    plt.plot(10 ** x_fit, y_fit, color=color, linestyle='--', linewidth=1.5)
                    plt.text(10 ** x_fit[-1], y_fit[-1],
                            f"R²={1 - np.sum((np.array(y_vals) - y_pred) ** 2) / np.sum((np.array(y_vals) - np.mean(y_vals)) ** 2):.2f}",
                            color=color, fontsize=9, verticalalignment='bottom', horizontalalignment='left')

            except Exception as e:
                print(f"[!] Log-fit failed for {network_name} on {metric}: {e}")

    plt.xlabel(metric_labels[metric], fontsize=22)
    plt.ylabel("mAP", fontsize=22)
    plt.grid(True)
    plt.legend(fontsize=15, loc='lower left', frameon=True)
    plt.tight_layout()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=14) #2d
    plt.savefig(os.path.join(output_dir, f"{keyword}_AllNetworks_mAP_vs_{metric}.jpg"), dpi=300)
    plt.close()
