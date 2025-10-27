"Metrics" folder contain the camera image quality metrics from the imatest software for the chart in the secne (Dataset folder) and the eSFR ISO test chart (Chart folder). 

    - Under Metrics/Dataset : 

        - Folders of "Dist1+18", "Dist2+18", "Dist1+55", and "Dist2+55" contain the raw outputs from Imatest software for both image information metrics (Results folder) and noise analysis (Results-noise folder).

        - Images were grouped in these four subsets were processed in batch by the software. These subsets were chosen as in each group, images had the same ROI of slanted edge. For example, images inside Dist1+18 had 18 mm focal length and Dist1 as the camera-to-scene distance, consequently having the same slanted edge ROI.

    - Under Metrics/Chart :

        - "Captures" folder contain the raw outputs from Imatest software for both image information metrics (Results folder) and noise analysis (Results-noise folder).

01_extractdata.py is used to extract the needed metrics from the imatest results within the "Dist1+18", "Dist2+18", "Dist1+55", and "Dist2+55" folders for the dataset and from "Captures" folder for the chart. 