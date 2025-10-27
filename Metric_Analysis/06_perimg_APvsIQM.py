import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

# --- CONFIGURATION ---
networks = ["YOLO11n", "YOLO11m", "SSD", "FasterRCNN-mobilenet", "FasterRCNN"]
iqm_source = "Chart"  # or "Chart"
imatest_summary_path = f"/Users/kimiaarfaie/Github/Image-Information-Metrics-in-Machine-Vision-Systems/Metric_Analysis/Metrics/{iqm_source}/Average Metrics"

keyword = "sharpness"
ISOfocused_split = ["100ISO_AND_Focused", "1600ISO_AND_Focused", "6400ISO_AND_Focused", "25600ISO_AND_Focused"]
Focused_split = ["Focused", "Defocus1", "Defocus2"]
ISO_split = ["100ISO", "1600ISO", "6400ISO", "25600ISO"]
size_split = ["18.0 mm_AND_Dist1", "18.0 mm_AND_Dist2", "55.0 mm_AND_Dist1", "55.0 mm_AND_Dist2"]
EV_split = ["-3EV", "-2EV", "-1EV", "0EV", "+1EV"]

splits = ISO_split

category_colors = {
    "car": "blue",
    "sports ball": "orange",
    "cup": "green",
    "potted plant": "red"
}
category_ids = {
    3: "car",
    37: "sports ball",
    47: "cup",
    64: "potted plant"
}

# --- PRELOAD IQMs ---
iqm_by_split = {}
for split in splits:
    summary_file = split.replace(" ", "").replace(".", ".") + "_average_summary.json"
    summary_path = os.path.join(imatest_summary_path, summary_file)
    if not os.path.exists(summary_path):
        print(f"[!] Missing IQM summary: {summary_path}")
        continue

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    mtf50 = summary.get("mtf50")
    c4 = summary.get("info_capacity_C_4_b_p")
    cmax = summary.get("info_capacity_C_max_b_p")
    edge_snri_square = summary.get("edge_SNRi_square", [None])[0]
    snri_square_db = summary.get("SNRi_square_dB", [None])[0]

    if None in [mtf50, c4, cmax, edge_snri_square, snri_square_db]:
        print(f"[!] Incomplete IQMs for {split}")
        continue

    iqm_by_split[split] = {
        "mtf50": mtf50,
        "c4": c4,
        "cmax": cmax,
        "edge_snri square": edge_snri_square,
        "snri square (dB)": snri_square_db
    }

# --- LOOP OVER NETWORKS ---
for network_name in networks:
    print(f"\n===== Running for network: {network_name} =====")

    per_image_dir = f"/home/colourlabgpu4/Kimia/Thesis/Object_Detection/{network_name}/per-image-validation/outputs/Full Dataset/per_image_results"
    csv_path = f"{per_image_dir}.csv"

    analysis_dir = f"/home/colourlabgpu4/Kimia/Thesis/Metric_Analysis/{network_name}_plots/{iqm_source}/perimg_Plots/mAPvsIQM"
    os.makedirs(analysis_dir, exist_ok=True)

    with open(csv_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
    image_map_dict = {
        line.split(',')[0].replace(".tiff", ""): float(line.split(',')[1])
        for line in lines
    }

    map_values = []
    class_ap_values = {cid: [] for cid in category_ids}
    c4_values, cmax_values, mtf50_values, edge_snri_square_values, snri_square_values  = [], [], [], [], []
    valid_splits = []

    # --- PROCESS PER SPLIT ---
    for split in splits:
        if split not in iqm_by_split:
            continue

        matching_imgs = [k for k in image_map_dict if all(part in k for part in split.split("_AND_"))]
        if not matching_imgs:
            print(f"[!] No matching images for split: {split}")
            continue

        split_maps = [image_map_dict[k] for k in matching_imgs]
        map_values.append(np.mean(split_maps))

        for cid, cname in category_ids.items():
            class_imgs = [k for k in matching_imgs if cname in k]
            class_maps = [image_map_dict[k] for k in class_imgs]
            class_ap_values[cid].append(np.mean(class_maps) if class_maps else None)
    
        # Use preloaded IQMs
        mtf50_values.append(iqm_by_split[split]["mtf50"])
        c4_values.append(iqm_by_split[split]["c4"])
        cmax_values.append(iqm_by_split[split]["cmax"])
        edge_snri_square_values.append(iqm_by_split[split]["edge_snri square"])
        snri_square_values.append(iqm_by_split[split]["snri square (dB)"])
        valid_splits.append(split)

    # --- PLOTTING FUNCTION ---
    def plot_metric_vs_map(metric_values, metric_name, file_suffix):
        if not metric_values or not map_values:
            print(f"[!] Skipping {file_suffix}, missing values")
            return

        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        ax.tick_params(labelsize=20)
        seen_labels = set()
        all_aps = [ap for cid in category_ids for ap in class_ap_values[cid] if ap is not None] + map_values
        ymax = max(all_aps) * 1.05

        for i, split in enumerate(valid_splits):
            x = metric_values[i]
            y = map_values[i]
            plt.scatter(x, y, s=250, color='black', marker='^', edgecolors='white', linewidths=0.5,
                        label="mAP" if "mAP" not in seen_labels else "")
            seen_labels.add("mAP")
            plt.annotate(split, (x, y), fontsize=14, xytext=(6, 4), textcoords='offset points')

            for cid, label in category_ids.items():
                ap = class_ap_values[cid][i]
                if ap is not None:
                    lbl = label if label not in seen_labels else ""
                    plt.scatter(x, ap, s=100, label=lbl, color=category_colors[label], alpha=0.8)
                    seen_labels.add(label)

        x = np.array(metric_values)
        y = np.array(map_values)
        if len(x) >= 2:
            try:
                if file_suffix == "snri_square_db":
                    # LINEAR FIT for SNRi square (dB)
                    coeffs = np.polyfit(x, y, deg=1)
                    fit_fn = np.poly1d(coeffs)
                    x_fit = np.linspace(min(x), max(x), 100)
                    y_fit = fit_fn(x_fit)
                    y_pred = fit_fn(x)
                    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
                    plt.plot(x_fit, y_fit, color="black", linestyle="--", linewidth=1.0,
                             label=f"Linear Fit, R²={r2:.2f}")
                else:
                    # LOG FIT for all other metrics
                    log_x = np.log10(x)
                    coeffs = np.polyfit(log_x, y, deg=1)
                    fit_fn = np.poly1d(coeffs)
                    x_fit = np.linspace(min(log_x), max(log_x), 100)
                    y_fit = fit_fn(x_fit)
                    y_pred = fit_fn(log_x)
                    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
                    plt.plot(10 ** x_fit, y_fit, color="black", linestyle="--", linewidth=1.0,
                             label=f"Log Fit, R²={r2:.2f}")      
            except Exception as e:
                print(f"[!] Fit error ({metric_name}):", e)


        pearson = np.corrcoef(x, y)[0, 1]
        spearman_val, _ = spearmanr(x, y)
        kendall_val, _ = kendalltau(x, y)
        print(f"\n--- Correlation: {metric_name} vs mAP ({network_name}) ---")
        print(f"Pearson : {pearson:.4f}")
        print(f"Spearman: {spearman_val:.4f}")
        print(f"Kendall : {kendall_val:.4f}")

        plt.xlabel(metric_name, fontsize=22)
        plt.ylabel("mAP@[0.50:0.95]", fontsize=22)
        plt.grid(True)
        plt.legend(fontsize=18, loc='best')
        plt.tight_layout()
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.savefig(os.path.join(analysis_dir, f"{keyword}_mAP_vs_{file_suffix}.jpg"), dpi=300)
        plt.close()

    # --- PLOT CALLS ---
    plot_metric_vs_map(c4_values, "Info Capacity C4", "C4")
    plot_metric_vs_map(cmax_values, "Info Capacity Cmax", "Cmax")
    plot_metric_vs_map(mtf50_values, "MTF50_Y_horizontal", "MTF50")
    plot_metric_vs_map(edge_snri_square_values, "Edge SNRi square", "edge_snri_square")
    plot_metric_vs_map(snri_square_values, "SNRi square (dB)",  "snri_square_db")
