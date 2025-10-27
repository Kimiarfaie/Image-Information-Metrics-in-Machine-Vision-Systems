import os
import json
from glob import glob

'''
Script 01:
This script extracts specific image quality metrics from Imatest JSON result files.

The script expects a folder structure where each subfolder under <base_data_dir> (such as "Dist1+18") contains:
    ├── Results/
    │     └── *_Y.json          → MTF and edge data from Imatest
    └── Results-noise/
          └── *_ColorToneAuto.json  → Noise and SNR data

For each subfolder, the script:
    - Loads the corresponding "_Y.json" and "_ColorToneAuto.json" files.
    - Extracts key metrics such as:
        • MTF50, MTF30
        • Information capacities (C_4_b_p, C_max_b_p)
        • Noise/SNRi data and power spectra
        • SNR ISO15739 value at 13% Lref
    - Saves one summary JSON file per image into the specified output directory.
           
Usage:
    python 01_extractdata.py <base_data_dir> [output_dir]

    Directory:
        - The script automatically searches for "Results" and "Results-noise"
        inside each first-level subfolder of <base_data_dir>.
        - You should therefore pass the parent directory that contains all
        your test scene folders (e.g., Dist1+18, Dist2+18, Dist1+55, etc.).

'''

def extract_metrics(base_dir, output_dir=None):
    if output_dir is None:
        output_dir = os.getcwd()

    os.makedirs(output_dir, exist_ok=True)
    dist_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for dist in dist_folders:
        result_dir = os.path.join(base_dir, dist, "Results")
        noise_dir = os.path.join(base_dir, dist, "Results-noise")

        print(f"\nChecking directory: {dist}")
        if not os.path.exists(result_dir):
            print(f"  Skipping: {result_dir} does not exist.")
            continue
        if not os.path.exists(noise_dir):
            print(f"  Skipping: {noise_dir} does not exist.")
            continue

        mtf_jsons = glob(os.path.join(result_dir, "*_Y.json"))
        if not mtf_jsons:
            print(f"  Warning: No *_Y.json files found in {result_dir}")
            continue

        for mtf_path in mtf_jsons:
            image_name = os.path.basename(mtf_path).replace("_Y.json", "")
            result = {}

            try:
                with open(mtf_path, 'r') as f:
                    mtf_data = json.load(f)["sfrResults"]

                mtf_section = mtf_data.get("MTF", {})
                edge_info = mtf_data.get("edge_info", {})
                roi_results = edge_info.get("ROI_results", {})

                result["mtf50"] = mtf_data.get("mtf50", [None])[0]
                result["mtf30"] = mtf_data.get("mtf30", [None])[0]
                result["info_capacity_C_4_b_p"] = mtf_data.get("edge_info_capacity_C_4_b_p", [[None]*4])[0][0]
                result["info_capacity_C_max_b_p"] = mtf_data.get("edge_info_capacity_C_max_b_p", [[None]*4])[0][0]

                result["mtf_plot"] = {
                    "freq1": mtf_section.get("freq1", []),
                    "freq1units": mtf_section.get("freq1units", ""),
                    "mtf_r": mtf_section.get("mtf_r", []),
                    "mtf_g": mtf_section.get("mtf_g", []),
                    "mtf_b": mtf_section.get("mtf_b", []),
                    "mtf_y": mtf_section.get("mtf_y", [])
                }

                result["noise_plot"] = {
                    "NPS_NEQ_frequency": edge_info.get("NPS_NEQ_frequency", []),
                    "SNRi_box_width": edge_info.get("SNRi_box_width", []),
                    "noise_power_spectrum": roi_results.get("noise_power_spectrum", []),
                    "noise_equivalent_quanta": roi_results.get("noise_equivalent_quanta", []),
                    "SNRi_square": roi_results.get("SNRi_square", []),
                    "edge_SNRi_square": roi_results.get("edge_SNRi_square", [])
                }

                noise_json_path = os.path.join(noise_dir, image_name + "_ColorToneAuto.json")
                if os.path.exists(noise_json_path):
                    with open(noise_json_path, 'r') as nf:
                        noise_data = json.load(nf)
                    snr_value = noise_data.get("jsonResults", {}).get("snr_dB_ISO15739_at_13pct_Lref", [None])[0]
                    result["snr_dB_ISO15739_at_13pct_Lref"] = snr_value
                else:
                    print(f"  Warning: Noise JSON not found for {image_name}")
                    result["snr_dB_ISO15739_at_13pct_Lref"] = None

                output_path = os.path.join(output_dir, f"{image_name}_summary.json")
                with open(output_path, 'w') as out_f:
                    json.dump(result, out_f, indent=4)

                print(f"  Saved: {output_path}")

            except Exception as e:
                print(f"  Failed to process {image_name}: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extract_metrics.py <base_data_dir> [output_dir]")
        sys.exit(1)

    base_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    extract_metrics(base_dir, output_dir)

'''
Example for dataset: 

01_extractdata.py /Users/kimiaarfaie/Github/Image-Information-Metrics-in-Machine-Vision-Systems/Metric_Analysis/Metrics/Dataset output_folder

'''