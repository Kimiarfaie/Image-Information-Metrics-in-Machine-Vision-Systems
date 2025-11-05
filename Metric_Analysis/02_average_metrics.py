import os
import json
import numpy as np
import re
from glob import glob

"""
Script 02:
This script computes average camera image-quality metrics across a specified subset of the dataset from the Imatest
summary JSON files (produced by `extractdata.py`). The script also averages MTF (Modulation Transfer Function) curves across multiple Imatest summary JSON files.
   
    MTF curves are sampled at different frequency points for each image.
    Therefore, before averaging, all curves must be interpolated to a common frequency grid.
    This is necessary as each image may have MTF values sampled at different frequencies.

Usage:
    change necessary variables in the main section (source, summary_dir, keywords, output_dir) and run:
    python average_metrics_full.py
"""

def normalize(s):
    """Remove all non-alphanumeric characters and lowercase the string."""
    return re.sub(r"[^a-zA-Z0-9]", "", s).lower()

def interpolate_vectors(freq_lists, vec_lists, target_freq):
    """Interpolate each MTF curve to a shared frequency grid."""
    interpolated = []
    for freq, vec in zip(freq_lists, vec_lists):
        if len(freq) == len(vec) and len(freq) > 1:
            interp = np.interp(target_freq, freq, vec)
            interpolated.append(interp)
    if not interpolated:
        return None
    return np.stack(interpolated)

def average_summary_metrics(summary_dir, split_keywords, output_path):
    if isinstance(split_keywords, str):
        split_keywords = [split_keywords]

    normalized_keywords = [normalize(kw) for kw in split_keywords]
    all_files = glob(os.path.join(summary_dir, "*_summary.json"))
    filtered_files = [f for f in all_files if all(nkw in normalize(os.path.basename(f)) for nkw in normalized_keywords)]

    if not filtered_files:
        print(f"No matching summary files found for: {split_keywords}")
        return

    print(f"Found {len(filtered_files)} files for: {split_keywords}")

    # --- Containers ---
    metrics = {
        "mtf50": [], "mtf30": [], "info_capacity_C_4_b_p": [],
        "info_capacity_C_max_b_p": [], "snr_dB_ISO15739_at_13pct_Lref": [],
        "noise_power_spectrum": [], "noise_equivalent_quanta": [],
        "SNRi_square": [], "edge_SNRi_square": [],
        "mtf_r": [], "mtf_g": [], "mtf_b": [], "mtf_y": []
    }

    x_axis_fields = {
        "freq1": None, "freq1units": None, "all_freq1": [],
        "NPS_NEQ_frequency": None, "SNRi_box_width": None
    }

    # --- Load all JSON files ---
    for file in filtered_files:
        with open(file, "r") as f:
            data = json.load(f)

        mtf_plot = data.get("mtf_plot", {})
        noise_plot = data.get("noise_plot", {})

        # Scalars
        for key in ["mtf50", "mtf30", "info_capacity_C_4_b_p",
                    "info_capacity_C_max_b_p", "snr_dB_ISO15739_at_13pct_Lref"]:
            val = data.get(key)
            if isinstance(val, (float, int)):
                metrics[key].append(val)

        # MTF curves (R, G, B, Y)
        for key in ["mtf_r", "mtf_g", "mtf_b", "mtf_y"]:
            vec = mtf_plot.get(key)
            if vec and isinstance(vec, list):
                metrics[key].append(vec)

        # Frequency sampling (once per file)
        freq1 = mtf_plot.get("freq1", [])
        if freq1:
            x_axis_fields["all_freq1"].append(freq1)

        # Noise-related vectors
        for key in ["noise_power_spectrum", "noise_equivalent_quanta",
                    "SNRi_square", "edge_SNRi_square"]:
            vec = noise_plot.get(key)
            if vec and isinstance(vec, list):
                metrics[key].append(vec)

        # Units / axis info
        if x_axis_fields["freq1units"] is None:
            x_axis_fields["freq1units"] = mtf_plot.get("freq1units", "")
        if x_axis_fields["NPS_NEQ_frequency"] is None:
            x_axis_fields["NPS_NEQ_frequency"] = noise_plot.get("NPS_NEQ_frequency", [])
        if x_axis_fields["SNRi_box_width"] is None:
            x_axis_fields["SNRi_box_width"] = noise_plot.get("SNRi_box_width", [])

    # --- Start averaging ---
    result = {}

    # Scalars
    for key in ["mtf50", "mtf30", "info_capacity_C_4_b_p",
                "info_capacity_C_max_b_p", "snr_dB_ISO15739_at_13pct_Lref"]:
        vals = metrics[key]
        result[key] = float(np.mean(vals)) if vals else None

    # Interpolated MTF curves
    target_freq = np.linspace(0, 1.0, 100)
    freq_lists = x_axis_fields["all_freq1"]
    for key in ["mtf_r", "mtf_g", "mtf_b", "mtf_y"]:
        vecs = metrics[key]
        if vecs and freq_lists and len(vecs) == len(freq_lists):
            stacked = interpolate_vectors(freq_lists, vecs, target_freq)
            if stacked is not None:
                result[key] = np.mean(stacked, axis=0).tolist()

    result["freq1"] = target_freq.tolist()
    result["freq1units"] = x_axis_fields["freq1units"]

    # Other noise/SNRi vectors
    for key in ["noise_power_spectrum", "noise_equivalent_quanta",
                "SNRi_square", "edge_SNRi_square"]:
        vecs = metrics[key]
        if vecs:
            min_len = min(len(v) for v in vecs)
            trimmed = [v[:min_len] for v in vecs]
            stacked = np.stack(trimmed)
            avg = np.mean(stacked, axis=0)
            result[key] = avg.tolist()

            if key == "SNRi_square":
                snri_dB = 10 * np.log10(np.maximum(avg, 1e-10))
                result["SNRi_square_dB"] = snri_dB.tolist()

    # --- Save output ---
    os.makedirs(output_path, exist_ok=True)
    split_name = "_AND_".join([kw.replace(" ", "").replace(".", ".") for kw in split_keywords])
    output_file = os.path.join(output_path, f"{split_name}_average_summary.json")

    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Saved average summary → {output_file}")

if __name__ == "__main__":

    source = "Dataset" # this is the source of IQMs, Dataset or Chart
    import argparse
    parser = argparse.ArgumentParser(description="Average and summarize metrics including MTF curves.")
    parser.add_argument(
        "--summary_dir",
        type=str,
        default=f"/Users/kimiaarfaie/Github/Image-Information-Metrics-in-Machine-Vision-Systems/Metric_Analysis/Metrics/{source}/Extracted+1",
        help="Directory with *_summary.json files."
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=["100ISO"],
        help="Keywords to filter files (AND logic)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"/Users/kimiaarfaie/Github/Image-Information-Metrics-in-Machine-Vision-Systems/Metric_Analysis/Metrics/{source}/Average Metrics+1",
        help="Directory to save averaged JSON."
    )
    args = parser.parse_args()

    # Run for multiple independent keyword sets
    
    keyword_sets = [
        ["100ISO"],
        ["1600ISO"],
        ["6400ISO"],
        ["25600ISO"],
        ["Focused"],
        ["Defocus1"],
        ["Defocus2"],
        ["18.0 mm", "Dist1"],
        ["18.0 mm", "Dist2"],
        ["55.0 mm", "Dist1"],
        ["55.0 mm", "Dist2"],
        ["100ISO", "Focused"],
        ["1600ISO", "Focused"],
        ["6400ISO", "Focused"],
        ["25600ISO", "Focused"],
    ]

    # To alalyze EV, change the summary_dir and output_dir folder to Extracted and Average Metrics
    #keyword_sets = [["+1EV"], ["0EV"], ["-1EV"], ["-2EV"], ["-3EV"]]

    for kw in keyword_sets:
        print(f"\n--- Running for {kw} ---")
        average_summary_metrics(args.summary_dir, kw, args.output_dir)
    