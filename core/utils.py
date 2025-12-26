import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def get_depth_masks(depth, annotations):
    """
    Extracts indices (coordinates) of pixels belonging to each annotation.
    Filters based on polygon inside the depth map.
    """
    depth_mask = depth > 0.01
    mask_tmp = np.zeros(depth.shape[:2], dtype=np.uint8)
    depth_masks = []
    for idx, annotation in annotations.items():
        mask = cv2.fillConvexPoly(mask_tmp.copy(), annotation['polygon'], color=255).astype(np.bool)
        mask = depth_mask & mask

        depth_masks.append(np.where(mask))
    return depth_masks


def get_mask(depth, depth_masks):
    """
    Reconstructs a full binary mask from a list of partial mask indices.
    """
    mask = np.zeros(depth.shape[:2], dtype=np.bool)
    for mask_idx in depth_masks:
        mask[mask_idx] = True
    return mask


def get_center_point(depth, mask_idx):
    """
    Calculates the median depth and the centroid (y, x) of the mask.
    """
    idx = np.rint(np.mean(mask_idx, axis=1)).astype(np.int64)
    return np.median(depth[mask_idx]), (idx[0], idx[1])


def get_annot_dists(depth, depth_masks):
    """
    Calculates the distance/depth for each tomato.
    Uses the median depth of all pixels within the mask.
    """
    annot_dists = [(*get_center_point(depth, mask_idx), mask_idx) for mask_idx in depth_masks]
    return annot_dists


def kmeans_best(data, best_score=0.85, debug=False, max_n_clusters=3, max_n_out=False):
    """
    Dynamically clusters 1D data (depths) to find groups using KMeans.
    The 'best' k (number of clusters) is determined by the Silhouette Score.
    """
    data_2d = data.reshape(-1, 1)
    possible_k_values = range(2, max_n_clusters + 1)

    best_labels = np.zeros_like(data.reshape(-1))
    best_centers = np.array([np.mean(data)])

    scores = []
    for k in possible_k_values:
        kmeans = KMeans(n_clusters=k, n_init='auto')
        labels = kmeans.fit_predict(data_2d)

        score = silhouette_score(data_2d, labels)
        scores.append(score)

        if score > best_score or max_n_out:
            best_score = score
            best_labels = kmeans.labels_
            best_centers = kmeans.cluster_centers_.flatten()
    if debug:
        print(f'Silhouette Scores: {scores}')
    return best_centers, best_labels


def find_best_cluster(centers, counts: np.ndarray, k_size=1.5):
    """
    Given multiple clusters of depth within a single mask, identify which cluster represents the actual object.
    Picks the cluster that is closer to the max size.
    """
    k_counts = counts.max() / counts
    idx = np.where(k_counts < k_size)[0]
    best_idx = idx[centers[idx].argmax()]
    return best_idx


def get_annot_dists_k_mean(depth, depth_masks, k_size=1.5, kmean_score=0.8):
    """
    Refines depth estimation by checking for occlusion within the mask.
    """
    annot_dists = []
    for depth_mask_idx in depth_masks:
        # Run KMeans on the pixels *inside* this specific mask
        centers, labels = kmeans_best(depth[depth_mask_idx], best_score=kmean_score)

        c_idx = np.argsort(centers)

        if len(centers) > 1:
            # Occlusion detected (multiple depth planes in one mask)
            c_centers = centers[c_idx]
            c_labels = [labels == i for i in c_idx]
            c_counts = np.count_nonzero(c_labels, axis=1)

            # Determine which cluster is the valid
            best_idx = find_best_cluster(c_centers, c_counts, k_size)
            best_label = c_labels[best_idx]

            # Update the mask index to only include the valid object
            best_mask_idx = (depth_mask_idx[0][best_label], depth_mask_idx[1][best_label])
            dist = c_centers[best_idx]
        else:
            # No occlusion detected
            best_mask_idx = depth_mask_idx
            dist = centers[0]

        # Recalculate center point based on the refined mask
        _, center_idx = get_center_point(depth, best_mask_idx)
        annot_dists.append((dist, center_idx, best_mask_idx))
    return annot_dists


def filter_annot_dists(annot_dists, kmean_score=0.8, max_n_clusters=5, max_n_out=False):
    """
    Automatically determines depth threshold to separate Foreground vs Background.
    Uses KMeans on the list of *all* tomato depths.
    """
    # Cluster the tomatoes based on their distance from camera
    dists = np.array([dist for dist, _, _ in annot_dists])
    centers, labels = kmeans_best(dists, best_score=kmean_score, max_n_clusters=max_n_clusters, max_n_out=max_n_out)

    c_idx = np.argsort(centers)
    c_labels = [labels == i for i in c_idx]
    c_counts = np.count_nonzero(c_labels, axis=1)

    # Assume the closest cluster (index 0 after sorting) is the Foreground
    f_counts = c_counts[0]
    f_label = c_labels[0]
    f_dists = dists[f_label]
    f_min_max = (f_dists.min(), f_dists.max())

    # All other clusters are Background
    if len(c_labels) > 1:
        b_counts = c_counts[1:].sum()
        b_label = np.any(c_labels[1:], axis=0)
        b_dists = dists[b_label]
        b_min_max = (b_dists.min(), b_dists.max())
    else:
        b_counts = 0
        b_min_max = (0, 0)  # Handle case where only foreground exists

    # Filter list to keep only foreground tomatoes
    f_annot_dists = [annot_dists[i] for i in np.where([f_label])[1]]

    print(f"[Foreground] From {f_min_max[0]:.2f} to {f_min_max[1]:.2f}, Counts: {f_counts}")
    print(f"[Background] From {b_min_max[0]:.2f} to {b_min_max[1]:.2f}, Counts: {b_counts}")
    return f_annot_dists, f_min_max
