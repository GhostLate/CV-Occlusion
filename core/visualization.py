import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from core.data import depth2color, mask2poly, mask2bbox
from core.utils import get_mask


def draw_plots(depths_dict: dict[str, torch.Tensor | np.ndarray]):
    """
    Helper to visualize multiple images/maps in a grid layout.
    """
    for name, depth in depths_dict.items():
        if isinstance(depth, torch.Tensor):
            depths_dict[name] = depth.detach().cpu().numpy()

    depth_len = len(depths_dict.keys())
    y = int(np.ceil(np.sqrt(depth_len)))
    x = int(np.ceil(depth_len / y))

    fig, axs = plt.subplots(x, y, layout='compressed')

    if x == 1 or y == 1:
        for i, (name, depth) in enumerate(depths_dict.items()):
            p = axs[i].imshow(depth)
            fig.colorbar(p, ax=axs[i])
            axs[i].set_title(name)
    else:
        for i, (name, depth) in enumerate(depths_dict.items()):
            xi = int(i // y)
            yi = int(i % y)
            p = axs[xi, yi].imshow(depth)
            fig.colorbar(p, ax=axs[xi, yi])
            axs[xi, yi].set_title(name)

    plt.show()


def draw_annotation(image, bbox, polygon, thickness=2):
    """
    Draws bounding box and polygon contour on the image.
    """
    top_left = (int(bbox[0]), int(bbox[1]))
    bot_right = (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(image, top_left, bot_right, (239,255, 0), thickness)
    cv2.drawContours(image, polygon, -1, (0,255,255), thickness)
    return image


def draw_image(image, annotations):
    """
    labels annotations (bbox and masks) on the given image
    """
    image = image.copy()
    for idx, annotation in annotations.items():
        bbox = annotation['bbox']
        polygon = annotation['polygon']
        image = draw_annotation(image, bbox, polygon)
    return image


def draw_point(image, value, coord):
    """
    Draws the depth value and a dot at the centroid.
    """
    point_coord = (coord[1], coord[0])
    text_coord = (coord[1] + 10, coord[0])
    cv2.circle(image, point_coord, radius=5, color=(0, 0, 255), thickness=-1)
    cv2.putText(image, f'[{value:.2f}]', text_coord,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


def draw_annot_dists(image, annot_dists):
    """
    Draws annotations, bounding boxes, and depth values on the image.
    """
    image = image.copy()

    # If input is a depth map (2D), colorize it first
    if image.ndim == 2:
        depth_masks = [mask_idx for _, _, mask_idx in annot_dists]
        mask = get_mask(image, depth_masks)
        image[~mask] = 0
        image = depth2color(image)

    for dist, dist_idx, mask_idx in annot_dists:
        mask = np.zeros(image.shape[:2], dtype=np.bool)
        mask[mask_idx] = True

        polygon = mask2poly(mask)
        bbox = mask2bbox(mask)

        draw_annotation(image, bbox, polygon)
        draw_point(image, dist, dist_idx)
    return image


def draw_depth_masks(depth, depth_masks):
    """
    Visualizes the extracted masks on the depth map.
    """
    depth = depth.copy()
    mask = np.zeros(depth.shape[:2], dtype=np.bool)
    for depth_mask_idx in depth_masks:
        mask[depth_mask_idx] = True
    depth[~mask] = 0
    return depth2color(depth)