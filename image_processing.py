# Our custom image processing functions, which replace certain OpenCV functions
# These are mostly copied/modified from our in-class assignments A1-7

import cv2
import numpy as np
import shapes

def otsu_threshold(img):
    """Otsu's threshold from its definition. Returns an int; pixels > t are class 1."""
    #  1. normalised histogram p over 256 levels      (np.bincount, minlength=256)
    hist = np.bincount(img.flat, minlength=256)
    norm = hist / np.sum(hist)
    levels = np.linspace(0, 255, 256)
    #  2. w0 = cumsum(p);  mu = cumsum(p * levels);  mu_T = mu[-1]
    w0 = np.cumsum(norm)
    mu = np.cumsum(norm * levels)
    mu_T = mu[-1]
    #  3. sigma_B^2 for every t at once, with empty classes set to 0
    with np.errstate(divide='ignore', invalid='ignore'): #added this because it kept yelling at me
        sigma2 = np.where((w0 > 0) & (w0 < 1), np.power((mu_T * w0) - mu, 2) / (w0 * (1 - w0)), 0.0)
    #  4. return int(argmax)
    return int(np.argmax(sigma2))

def clean_mask(mask, open=2, close=2):
    """Open with disk(r_open), then close with disk(r_close). Returns bool."""
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, shapes.rectangle(open, open/2))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, shapes.rectangle(close, close/2))
    return closed.astype(bool)

def count_objects(mask, connectivity=2):
    """Number of connected components (1 = 4-connected, 2 = 8-connected)."""
    num_labels, _ = cv2.connectedComponents(
        mask.astype(np.uint8), connectivity=connectivity * 4
    )
    return num_labels - 1