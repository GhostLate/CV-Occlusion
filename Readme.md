# Tomato Segmentation & Depth Analysis

This project automates the counting of foreground tomatoes in farm images and handles occlusion issues to provide accurate depth estimation.

## Requirements

Ensure you have the following Python libraries installed:

```bash
pip install opencv-python imutils natsort numpy torch matplotlib scikit-learn
```

## Usage
Run the script by pointing it to your dataset directory:

```Bash
python main.py --data-path "./TaskDataset-Occlusion"
```

## Optional Arguments

* `--debug`: (bool) Set to True to see intermediate depth maps and clusters.
* `--kmean-score`: (float) Threshold for Silhouette Score to accept a cluster (default: 0.8).
* `--max-n-clusters`: (int) Max clusters to search for when separating foreground/background (default: 5).
* `--max-n-out`: (bool) Force output of max clusters (default: False).
* `--k-size`: (float) Cluster size ratio for occlusion filtering (default: 1.5).

## Methodology

### Task 1: Depth Threshold Determination (Foreground Counting)
 
**Goal:** Automatically determine a depth threshold to count only foreground tomatoes, excluding those deep in the background.

### Solution:

* Extract Depths: For every annotated tomato, we calculate a "base" distance using the median depth of the pixels within its mask.
* Global Clustering: We collect these distances into a list and apply KMeans clustering.
* Automatic Thresholding: The algorithm groups tomatoes into clusters (e.g., "Near/Foreground", "Middle", "Far/Background"). We sort these clusters by distance.
* Selection: The cluster with the smallest distance values is identified as the Foreground.
* Filtering: The script calculates the min and max depth of this foreground cluster and filters out any tomatoes falling outside this range.

### Bonus Task: Occlusion Handling
**Goal:** Correct depth estimation when a tomato is partially covered by a leaf. 

### Solution:

* Local Clustering: Instead of just averaging all pixels in a tomato mask, we perform KMeans clustering on the pixels inside the single mask.
* Occlusion Detection: If the mask contains two distinct depth layers (high Silhouette score for k=2), we assume one layer is the leaf and the other is the tomato.
* Cluster Selection: The algorithm analyzes the clusters. It generally selects the dominant cluster (the tomato surface) while discarding the smaller anomaly (the leaf edge or stem).
* Refined Centroid: The depth and the measurement point (centroid) are re-calculated using only the pixels from the "clean" tomato cluster, ignoring the leaf.

### Output
The script generates visualizations containing 4 plots for each image:

* `dist_base`: All detected tomatoes with naive depth estimation.
* `dist_base_foreground`: Only foreground tomatoes (Task 1 result).
* `dist_region`: All tomatoes with occlusion-corrected depth (Bonus Task).
* `dist_region_foreground`: Foreground tomatoes with occlusion-corrected depth (Final Result).
