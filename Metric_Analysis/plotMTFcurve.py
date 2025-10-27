import os
import json
import matplotlib.pyplot as plt

# CONFIGURATION
source  = "Dataset"
imatest_summary_path = f"/home/colourlabgpu4/Kimia/Thesis/Metric_Analysis/Metrics/{source}/Average Metrics"
analysis_dir = f"/home/colourlabgpu4/Kimia/Thesis/Metric_Analysis/Metrics/{source}/MTF"
os.makedirs(analysis_dir, exist_ok=True)

# Choose your split type (e.g., ISO, EV, etc.)
keyword = "EV"
#splits = ["100ISO", "1600ISO", "6400ISO", "25600ISO"]
#splits = ["Focused", "Defocus1", "Defocus2"]
#splits = ["18.0 mm_AND_Dist1", "18.0 mm_AND_Dist2", "55.0 mm_AND_Dist1", "55.0 mm_AND_Dist2"]
splits = ["-3EV"]

# ---- PER SPLIT MTF PLOTS ONLY ----
for split in splits:
    summary_file = split.replace(" ", "").replace(".", ".") + "_average_summary.json"
    summary_path = os.path.join(imatest_summary_path, summary_file)

    if not os.path.exists(summary_path):
        print(f"Missing summary for {split}")
        continue

    with open(summary_path, 'r') as f:
        data = json.load(f)

    freq1 = data.get("freq1", [])
    freq_unit = data.get("freq1units", "cycles/pixel")

    plt.figure()
    ax = plt.gca()
    ax.tick_params(labelsize=14)
    for k, color, style, width in zip(
        ["mtf_r", "mtf_g", "mtf_b", "mtf_y"],
        ["r", "g", "b", "k"],
        ["--", "--", "--", "-"],
        [1.0, 1.0, 1.0, 2.0]
    ):
        vec = data.get(k, [])
        if vec:
            plt.plot(freq1[:len(vec)], vec, label=k, color=color, linestyle=style, linewidth=width)

    plt.xlabel(f"Frequency ({freq_unit})", fontsize=18)
    plt.xlim(0, 0.5)
    #plt.ylim(0, 1.5)
    plt.ylabel("MTF", fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.tight_layout()

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(os.path.join(analysis_dir, f"mtf_{split}.jpg"), dpi=300)
    plt.close()
