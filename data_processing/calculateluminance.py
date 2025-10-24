import os
import cv2
import numpy as np
from skimage import color

'''
This script computes average CIELAB L* (luminance) values for image groups by exposure value (EV).

It processes images corresponding to different exposure values (EVs) and computes the mean L* for each group.

Usage:
1. Specify the path to the image directory "image_dir", and the EV splits, "ev_splits", to analyze in the main block.
2. Run python calculateluminance.py
'''

def compute_average_L_star(image_dir, ev_splits):
    """
    Compute the average L* (luminance) values for each EV group.

    Args:
        image_dir (str): Path to the directory containing TIFF or PNG images.
        ev_splits (list[str]): List of EV values to group by (e.g., ['+1EV', '0EV', '-1EV']).

    Returns:
        dict: Mapping of EV -> average L* value.
    """
    l_star_values = {}

    for ev in ev_splits:
        ev_images = [img for img in os.listdir(image_dir) if ev in img]
        l_star_list = []

        for img_name in ev_images:
            img_path = os.path.join(image_dir, img_name)

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"⚠️  Warning: Couldn't read {img_path}")
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_lab = color.rgb2lab(img_rgb)
            mean_l_star = np.mean(img_lab[:, :, 0])
            l_star_list.append(mean_l_star)

        if l_star_list:
            avg_l_star = np.mean(l_star_list)
            l_star_values[ev] = avg_l_star
            print(f"EV {ev}: Avg L* = {avg_l_star:.2f}")
        else:
            print(f"EV {ev}: No images found.")

    return l_star_values


if __name__ == "__main__":
    image_dir = "/home/colourlabgpu4/Kimia/Thesis/Datasets/Full Dataset/images/val"
    ev_splits = ["-2EV", "-1EV", "0EV", "+1EV", "+2EV"]

    print(f"\nAnalyzing luminance for EV splits: {', '.join(ev_splits)}")
    l_star_values = compute_average_L_star(image_dir, ev_splits)

    print("\nFinal L* averages per EV split:")
    for ev, l_star in l_star_values.items():
        print(f"{ev}: {l_star:.2f}")