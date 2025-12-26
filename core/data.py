import cv2
import imutils
import numpy as np


def mask2poly(mask):
    """
    Converts a binary mask to a polygon contour.
    """
    mask = np.array(mask).astype(np.uint8)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts):
        c = max(cnts, key=cv2.contourArea)
    else:
        c = []
    return c


def mask2bbox(mask):
    """
    Converts a binary mask to a bounding box [xmin, ymin, xmax, ymax].
    """
    m = mask.astype(bool)
    ys, xs = np.where(m)
    if ys.size == 0:
        return None
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return [x_min, y_min, x_max, y_max]


def poly2mask(polygon, im_shape):
    """
    Converts a polygon contour to a binary mask.
    """
    mask = np.zeros(im_shape)
    cv2.drawContours(mask, [polygon], -1, color=(255,255,255), thickness=-1)
    return mask.astype(bool)


def depth2color(depth_map):
    """
    To visualise the depthmap, the function colors it like a heatmap
    INPUTS
    depth_map (np.array): The depthmap to be colored
    OUTPUTS
    depth_colored (np.array): Colored depthmap
    """
    depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=10), cv2.COLORMAP_JET)
    return depth_colored