import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------- CONFIGURATION --------
split = "size"
xlabel = "Object Size"
networks = ["YOLO11n", "YOLO11m", "FasterRCNN", "FasterRCNN-mobilenet", "SSD"]

# Define split conditions and labels
#split_conditions = [["100ISO"], ["1600ISO"], ["6400ISO"], ["25600ISO"]]

#split_conditions = [ ["-3EV"], ["-2EV"], ["-1EV"], ["0EV"], ["+1EV"]]

#split_conditions = [ ["Focused"], ["Defocus1"], ["Defocus2"]]

split_conditions = [ ["18.0 mm", "Dist1"], ["18.0 mm", "Dist2"], ["55.0 mm", "Dist1"], ["55.0 mm", "Dist2"]]

split_labels = [" AND ".join(conds) for conds in split_conditions]

for network in networks:
    print(f"Processing network: {network}")
    
    csv_path = f"/Users/kimiaarfaie/Github/Image-Information-Metrics-in-Machine-Vision-Systems/object_detection/{network}/per-image-validation/outputs/Full Dataset/pre_image_results/all_images_metricsummary.csv"
    base_output_dir = f"/Users/kimiaarfaie/Github/Image-Information-Metrics-in-Machine-Vision-Systems/object_detection/{network}/per-image-validation/outputs/Full Dataset"
    analysis_dir = os.path.join(base_output_dir, "metric_visualization")
    split_analysis_json = os.path.join(base_output_dir, f"data_analysis_{split}.json")

    os.makedirs(analysis_dir, exist_ok=True)

    output_plot_path_line = os.path.join(analysis_dir, f"{split}_line_plot_mAP.png")
    output_plot_path_bar = os.path.join(analysis_dir, f"{split}_grouped_bar_detection_rate.png")

    # Class labels and colors
    if network.lower().startswith("yolo"):
        class_ids_to_labels = {2: "car", 32: "sports ball", 41: "cup", 58: "potted plant"}
    else:
        class_ids_to_labels = {3: "car", 37: "sports ball", 47: "cup", 64: "potted plant"}

    avg_color = "red"
    category_colors = {
        list(class_ids_to_labels.keys())[0]: "#66C5CC",
        list(class_ids_to_labels.keys())[1]: "#F6CF71",
        list(class_ids_to_labels.keys())[2]: "#F89C74",
        list(class_ids_to_labels.keys())[3]: "#DCB0F2"
    }

    # -------- LOAD mAP DATA --------
    df = pd.read_csv(csv_path)
    df["image"] = df["image"].str.replace(".tiff", "", regex=False)

    avg_map = []
    class_map = {cid: [] for cid in class_ids_to_labels}

    for keywords in split_conditions:
        matching_df = df[df["image"].apply(lambda x: all(k in x for k in keywords))]
        avg_map.append(matching_df["mAP50_95"].mean())

        for cid, cname in class_ids_to_labels.items():
            class_imgs = matching_df[matching_df["image"].str.startswith(cname)]
            class_map[cid].append(class_imgs["mAP50_95"].mean() if not class_imgs.empty else None)

    # -------- LOAD SPLIT-BASED DETECTION DATA --------
    with open(split_analysis_json, "r") as f:
        detection_data = json.load(f)

    detection_rates = []
    class_detection = {cid: [] for cid in class_ids_to_labels}

    for label in split_labels:
        entry = detection_data.get(label)
        if not entry:
            detection_rates.append(0)
            for cid in class_ids_to_labels:
                class_detection[cid].append(0)
            continue

        detection_rates.append(entry.get("avg_detection_rate", 0))
        for cid in class_ids_to_labels:
            rate = entry["per_class"].get(str(cid), 0)
            class_detection[cid].append(rate)

    # -------- LINE PLOT --------
    plt.figure(figsize=(10, 7))
    plt.plot(split_labels, avg_map, label="mAP@[0.50:0.95]", color=avg_color, marker="o", linewidth=2)

    for cid in class_ids_to_labels:
        plt.plot(split_labels,
                 class_map[cid],
                 label=f"AP@[0.50:0.95] - {class_ids_to_labels[cid]}",
                 color=category_colors[cid],
                 linestyle="--",
                 marker="x",
                 linewidth=1.5)

    plt.xlabel(f"{xlabel}", fontsize=22)
    plt.ylabel("mAP@[0.50:0.95]", fontsize=22)
    plt.legend(loc='lower left', fontsize=18)
    plt.tight_layout()
    plt.grid(True)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(output_plot_path_line)
    plt.close()

    # -------- GROUPED BAR PLOT --------
    x = np.arange(len(split_labels))
    width = 0.15

    plt.figure(figsize=(12, 7))
    plt.bar(x - 2 * width, detection_rates, width=width, label="Avg Detection Rate", color=avg_color)

    for i, cid in enumerate(class_ids_to_labels):
        offset = (i - 1) * width
        plt.bar(
            x + offset,
            class_detection[cid],
            width=width,
            label=class_ids_to_labels[cid],
            color=category_colors[cid]
        )

    plt.xlabel(f"{xlabel}", fontsize=22)
    plt.ylabel("Detection Rate (%)", fontsize=22)
    plt.xticks(ticks=x, labels=split_labels)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(output_plot_path_bar)
    plt.close()

    print(f"Finished plotting for {network}\n")
