import os
import json
import numpy as np
import re
from glob import glob

'''
Script 02:
This script computes average camera image-quality metrics across a specified subset of the dataset from the Imatest
summary JSON files (produced by `extractdata.py`). Each *_summary.json file corresponds to one image and contains metrics such as:
MTF50, information capacities, SNR values, and MTF/SNRi curves.

The script allows users to calculate mean and standard deviation of these metrics
for a defined subset of images — for example, "100ISO" contains all images captured at 100 ISO.

The csript will:
1. Scans `summary_dir` for all *_summary.json files.
2. Filters files that contain all user-specified keywords in their filename.
   (e.g. `--keywords 100ISO` selects all summaries containing “100ISO” in their name.)
3. Loads all selected summaries and extracts metrics.
4. Computes:
      • Mean and standard deviation for the metrics
      • SNRi in dB scale for interpretability
6. Saves one combined average summary JSON to:
      <output_dir>/<keyword_group>_average_summary.json


'''
def average_summary_metrics(summary_dir, split_keywords, output_path):
    if isinstance(split_keywords, str):
        split_keywords = [split_keywords]

    all_files = glob(os.path.join(summary_dir, "*_summary.json"))
    filtered_files = [f for f in all_files if all(kw in os.path.basename(f) for kw in split_keywords)]

    if not filtered_files:
        print(f"No summary files found for: {split_keywords}")
        return

    print(f"Found {len(filtered_files)} files matching: {split_keywords}")

    metrics = {
        "mtf50": [], "mtf30": [], "info_capacity_C_4_b_p": [], "info_capacity_C_max_b_p": [],
        "snr_dB_ISO15739_at_13pct_Lref": [],
        "mtf_r": [], "mtf_g": [], "mtf_b": [],
        "mtf_y": [], "noise_power_spectrum": [], "noise_equivalent_quanta": [],
        "SNRi_square": [], "edge_SNRi_square": []
    }

    x_axis_fields = {
        "freq1": None, "freq1units": None,
        "NPS_NEQ_frequency": None, "SNRi_box_width": None
    }

    for file in filtered_files:
        with open(file, 'r') as f:
            data = json.load(f)

        mtf_plot = data.get("mtf_plot", {})
        noise_plot = data.get("noise_plot", {})

        for key in ["mtf50", "mtf30", "info_capacity_C_4_b_p", "info_capacity_C_max_b_p", "snr_dB_ISO15739_at_13pct_Lref"]:
            val = data.get(key)
            if isinstance(val, (float, int)):
                metrics[key].append(val)

        for key in ["mtf_r", "mtf_g", "mtf_b", "mtf_y"]:
            vec = mtf_plot.get(key)
            if vec and isinstance(vec, list) and all(isinstance(v, (float, int)) for v in vec):
                metrics[key].append(vec)

        for key in ["noise_power_spectrum", "noise_equivalent_quanta", "SNRi_square", "edge_SNRi_square"]:
            vec = noise_plot.get(key)
            if vec and isinstance(vec, list) and all(isinstance(v, (float, int)) for v in vec):
                metrics[key].append(vec)

        if x_axis_fields["freq1"] is None:
            x_axis_fields["freq1"] = mtf_plot.get("freq1", [])
            x_axis_fields["freq1units"] = mtf_plot.get("freq1units", "")
        if x_axis_fields["NPS_NEQ_frequency"] is None:
            x_axis_fields["NPS_NEQ_frequency"] = noise_plot.get("NPS_NEQ_frequency", [])
        if x_axis_fields["SNRi_box_width"] is None:
            x_axis_fields["SNRi_box_width"] = noise_plot.get("SNRi_box_width", [])

    result = {}

    for key in ["mtf50", "mtf30", "info_capacity_C_4_b_p", "info_capacity_C_max_b_p", "snr_dB_ISO15739_at_13pct_Lref"]:
        values = metrics[key]
        result[key] = float(np.mean(values)) if values else None
        result[f"{key}_std"] = float(np.std(values)) if values else None

    for key in ["mtf_r", "mtf_g", "mtf_b", "mtf_y", "noise_power_spectrum", "noise_equivalent_quanta", "SNRi_square", "edge_SNRi_square"]:
        vectors = metrics[key]
        if vectors:
            min_len = min(len(v) for v in vectors)
            trimmed = [v[:min_len] for v in vectors]
            stacked = np.stack(trimmed)
            avg = np.mean(stacked, axis=0)
            std = np.std(stacked, axis=0)
            result[key] = avg.tolist()
            result[f"{key}_std"] = std.tolist()

            if key == "SNRi_square":
                snri_dB = 10 * np.log10(np.maximum(avg, 1e-10))
                snri_dB_std = 10 * np.log10(np.maximum(avg + std, 1e-10)) - snri_dB
                result["SNRi_square_dB"] = snri_dB.tolist()
                result["SNRi_square_dB_std"] = snri_dB_std.tolist()
        else:
            result[key] = None
            result[f"{key}_std"] = None
            if key == "SNRi_square":
                result["SNRi_square_dB"] = None
                result["SNRi_square_dB_std"] = None

    result.update(x_axis_fields)

    os.makedirs(output_path, exist_ok=True)
    split_name = "_AND_".join([kw.replace(" ", "").replace(".", ".") for kw in split_keywords])
    output_file = os.path.join(output_path, f"{split_name}_average_summary.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Saved average summary to: {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Average and summarize metrics from *_summary.json files.")
    parser.add_argument(
        "--summary_dir", 
        type=str,
        help="Directory containing *_summary.json files.", 
        default="/home/colourlabgpu4/Kimia/Thesis/Metric_Analysis/Metrics/Dataset/Extracted+1",
    )
    parser.add_argument(
        "--keywords", 
        nargs="+", 
        required=True, 
        help="Keywords to filter files (AND logic)."
    )
    parser.add_argument(
        "--output_dir", 
        required=True, 
        default="/home/colourlabgpu4/Kimia/Thesis/Metric_Analysis/Metrics/Dataset/Average Metrics+1",
        )
    args = parser.parse_args()
    
    average_summary_metrics(args.summary_dir, args.keywords, args.output_dir)


# python 02_average_metrics.py --summary_dir /Users/kimiaarfaie/Github/Image-Information-Metrics-in-Machine-Vision-Systems/Metric_Analysis/Metrics/Dataset/Extracted+1 --keywords "100ISO" --output_dir /Users/kimiaarfaie/Github/Image-Information-Metrics-in-Machine-Vision-Systems/Metric_Analysis/Metrics/Dataset