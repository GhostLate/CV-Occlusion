import argparse
import os
import cv2
import natsort
import numpy as np

from core.io import load_annotation, load_depth
from core.utils import get_depth_masks, get_annot_dists, filter_annot_dists, get_annot_dists_k_mean
from core.visualization import draw_annot_dists, draw_image, draw_plots, draw_depth_masks


def parse_args():
    parser = argparse.ArgumentParser(description='Object Segmentation and Depth Estimation')
    parser.add_argument(
        '--data-path', default="./TaskDataset-Occlusion", help='Path to dataset folder')
    parser.add_argument(
        '--debug-vis', type=bool, default=False, help='Enable debug plotting')
    parser.add_argument(
        '--kmean-score', type=float, default=0.8, help='Silhouette score threshold for KMeans')
    parser.add_argument(
        '--max-n-clusters', type=int, default=5, help='Max clusters for depth separation')
    parser.add_argument(
        '--max-n-out', type=bool, default=False, help='Force output of max clusters')
    parser.add_argument(
        '--k-size', type=float, default=1.5, help='Cluster size ratio for occlusion filtering')
    args = parser.parse_args()
    return args


def depth_threshold(img, depth, annot):
    """
    Task 1 Wrapper (Colab):
    inputs : image, depth map, segmented tomato polygons
    outputs : depth threshold in m, foreground tomato count, image with only foreground tomatoes labeled
    """
    depth_masks = get_depth_masks(depth, annot)

    annot_dists_base = get_annot_dists(depth, depth_masks)
    annot_dists_base_fg, fg_min_max = filter_annot_dists(annot_dists_base, kmean_score, max_n_clusters, max_n_out)
    dist_base_foreground = draw_annot_dists(img, annot_dists_base_fg)
    return fg_min_max, len(annot_dists_base_fg), dist_base_foreground


def occ_tomato_depth(img, depth, annot) :
    """
    Bonus Task Wrapper (Colab):
    inputs : image, depth, segmented tomato polygons
    outputs : list of tomatoes and their depths, image with area used for depth estimation marked
    """
    depth_masks = get_depth_masks(depth, annot)

    annot_dists_region = get_annot_dists_k_mean(depth, depth_masks, k_size, kmean_score)
    annot_dists_region_fg, fg_min_max = filter_annot_dists(annot_dists_region, kmean_score, max_n_clusters, max_n_out)

    dists = []
    depths = []
    for dist, dist_idx, mask_idx in annot_dists_region_fg:
        mask = np.zeros(depth.shape[:2], dtype=np.bool)
        mask[mask_idx] = True

        new_depth = depth.copy()
        new_depth[~mask] = 0

        dists.append(dist)
        depths.append(new_depth)

    dist_base_foreground = draw_annot_dists(img, annot_dists_region_fg)
    return dists, depths, dist_base_foreground


if __name__ == '__main__':
    args = parse_args()

    data_path = args.data_path
    debug_vis = args.debug_vis

    kmean_score = args.kmean_score
    max_n_clusters = args.max_n_clusters
    max_n_out = args.max_n_out
    k_size = args.k_size

    filenames = natsort.natsorted(os.listdir(args.data_path))
    filenames = [filename for filename in filenames if filename.endswith(".png") or filename.endswith(".jpg")]

    for filename in filenames:
        img_file = filename
        annot_file = filename.split(".")[0] + "_annotation.pkl"
        depth_file = filename.split(".")[0] + "_depth.npy"

        print(f"\nProcessing {img_file}")
        try:
            img = cv2.imread(os.path.join(data_path, img_file))
            annot = load_annotation(os.path.join(data_path, annot_file))
            depth = load_depth(os.path.join(data_path, depth_file))
        except FileNotFoundError:
            print(f"Skipping {filename}: Annotations or Depth file missing.")
            continue

        annot_img = draw_image(img, annot)
        depth_masks = get_depth_masks(depth, annot)

        # --- Task 1: Base Processing (Global Depth Thresholding) ---
        print(f"Base processing...")
        annot_dists_base = get_annot_dists(depth, depth_masks)
        annot_dists_base_fg, _ = filter_annot_dists(annot_dists_base, kmean_score, max_n_clusters, max_n_out)
        # results = depth_threshold(img, depth, annot)

        # --- Bonus Task: Region Processing (Occlusion Handling) ---
        print(f"Region processing...")
        annot_dists_region = get_annot_dists_k_mean(depth, depth_masks, k_size, kmean_score)
        annot_dists_region_fg, _ = filter_annot_dists(annot_dists_region, kmean_score, max_n_clusters, max_n_out)
        # results = occ_tomato_depth(img, depth, annot)

        if debug_vis:
            # Visualize the comparison between basic and occlusion-aware methods (based on depth map)
            draw_plots(dict(
                annot=annot_img,
                masked_depth=draw_depth_masks(depth, depth_masks),
                depth_dist=draw_annot_dists(depth, annot_dists_base),
                depth_dist_foreground=draw_annot_dists(depth, annot_dists_base_fg),
                depth_dist_region=draw_annot_dists(depth, annot_dists_region),
                depth_dist_region_foreground=draw_annot_dists(depth, annot_dists_region_fg),
            ))
        else:
            # Visualize the comparison between basic and occlusion-aware methods
            draw_plots(dict(
                dist_base=draw_annot_dists(img, annot_dists_base),
                dist_base_foreground=draw_annot_dists(img, annot_dists_base_fg),
                dist_region=draw_annot_dists(img, annot_dists_region),
                dist_region_foreground=draw_annot_dists(img, annot_dists_region_fg),
            ))
