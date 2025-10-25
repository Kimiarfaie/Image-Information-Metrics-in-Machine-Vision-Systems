import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------- CONFIGURATION --------
networks = ["YOLO11n", "YOLO11m", "FasterRCNN", "SSD", "FasterRCNN-mobilenet"]
split = "EV"
xlabel = "Exposure Value"
base_dir = "/Users/kimiaarfaie/Github/Image-Information-Metrics-in-Machine-Vision-Systems/object_detection"
csv_filename = "all_images_metricsummary.csv"

# Define split keywords
#split_conditions = [["18.0 mm", "Dist1"], ["18.0 mm", "Dist2"], ["55.0 mm", "Dist1"], ["55.0 mm", "Dist2"]]
#split_conditions = [ ["100ISO"], ["1600ISO"], ["6400ISO"], ["25600ISO"]]

split_conditions = [ ["-3EV"], ["-2EV"], ["-1EV"], ["0EV"], ["+1EV"]]

#split_conditions = [ ["Focused"], ["Defocus1"], ["Defocus2"]]

# Output
output_dir = f"/Users/kimiaarfaie/Github/Image-Information-Metrics-in-Machine-Vision-Systems/Metric_Analysis/Network_Comparison_mAPvsSETTING"
os.makedirs(output_dir, exist_ok=True)
output_plot_path = os.path.join(output_dir, f"{split}_mAP_comparison.png")

# Colors for different networks
network_colors = {
    "YOLO11m": "orange",
    "YOLO11n": "green",
    "SSD": "purple",
    "FasterRCNN-mobilenet": "red",
    "FasterRCNN": "blue"
}

# -------- PROCESS & PLOT --------
split_labels = [" AND ".join(k) for k in split_conditions]
plt.figure(figsize=(10, 7))

for net in networks:
    net_csv = os.path.join(base_dir, net, "per-image-validation", "outputs", "Full Dataset", csv_filename)
    if not os.path.exists(net_csv):
        print(f"CSV file missing for {net}: {net_csv}")
        continue

    df = pd.read_csv(net_csv)
    df["image"] = df["image"].str.replace(".tiff", "", regex=False)

    avg_mAP = []
    for cond in split_conditions:
        matching = df[df["image"].apply(lambda x: all(k in x for k in cond))]
        mean_val = ( 
            matching["mAP50_95"].dropna().mean()
            if not matching["mAP50_95"].dropna().empty
            else None
            )
        avg_mAP.append(mean_val)

    if all(v is None for v in avg_mAP):
        continue

    color = network_colors.get(net, "black")
    
    print(f"{net} → mAP values:", avg_mAP)
    if all(v is None or np.isnan(v) for v in avg_mAP):
        print(f"[SKIPPED] {net}: all mAP values are None/NaN, so nothing to plot.")
        continue

    plt.plot(
        split_labels,
        avg_mAP,
        label=net,
        color=color,
        marker="o",
        linewidth=2,         # make line thicker
        markersize=8
    )

plt.xlabel(xlabel, fontsize=22)
plt.ylabel("mAP@[0.50:0.95]", fontsize=22)
plt.legend(fontsize=18)
plt.tight_layout()
plt.grid(True)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(output_plot_path)
plt.show()

print(f"Plot saved to: {output_plot_path}")
