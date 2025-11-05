## Folders

"Metrics" folder contain the camera image quality metrics from the imatest software for the chart in the scenes ("Dataset" folder) and the eSFR ISO test chart ("Chart" folder). 

Under Metrics/Dataset : 

    Folders of "Dist1+18", "Dist2+18", "Dist1+55", and "Dist2+55" contain the raw outputs from Imatest software for both image information metrics (Results folder) and noise analysis (Results-noise folder).

    Images grouped in these four subsets were processed in batch mode by the software. These subsets were chosen as in each group, images had the same ROI of slanted edge. For example, images inside Dist1+18 had 18 mm focal length and Dist1 as the camera-to-scene distance, consequently having the same slanted edge ROI.

Under Metrics/Chart :

    "Captures" folder contain the raw outputs from Imatest software for both image information metrics ("Results" folder) and noise analysis ("Results-noise" folder).

## Scripts 

1. 01_extractdata.py is used to extract the needed metrics from the imatest results within the "Dist1+18", "Dist2+18", "Dist1+55", and "Dist2+55" folders for the dataset and from "Captures" folder for the chart. Results are saved to "Extracted" folder. Pay attention that there is an "Extracted+1" folder, which only has +1EV results from "Extracted" folder. In this project results of images with +1EV are mostly used, except for the time we are analysing EV changes. 


2. 02_average_metrics.py computes average camera image quality metrics across a specified subset of the dataset from the Imatest summary JSON files (produced by `extractdata.py`). Run the following command or chnage the defult arguments in the main section. 

    It also averages MTF curves across multiple Imatest summary JSON files. This is needed if we want to average the curve for multiple images, as the MTF curves are sampled at different frequency points for each image. Therefore, before averaging, all curves must be interpolated to a common frequency grid. 

3. Scripts 3-7 are for plotting MTF curves, Quality metrics vs Camera Settings, mAP vs IQMs, and 3D plots of mAP vs IQMs. 