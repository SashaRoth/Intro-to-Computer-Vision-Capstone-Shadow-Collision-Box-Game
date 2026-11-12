# Our custom image processing functions, which replace certain OpenCV functions
# These are mostly copied/modified from our in-class assignments A1-7

import cv2
import numpy as np
import shapes

def otsu_threshold(img):
    """Otsu's threshold from its definition. Returns int."""
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
    """Open with disk(r_open), then close with disk(r_close). Returns binary mask."""
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, shapes.rectangle(open, open/2))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, shapes.rectangle(close, close/2))
    return closed.astype(bool)

def keep_largest_contour(mask):
    """Keep only the largest contour in a binary mask. Returns binary mask"""
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = max(contours, key=cv2.contourArea)
    out = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(out, [cont], -1, 255, cv2.FILLED)
    return np.bitwise_and(mask, out)

def background_div(mask, kernel_size=51):
    """Comment"""
    background = cv2.GaussianBlur(mask, (kernel_size,kernel_size), 0)
    img_f= mask.astype(np.float32)
    bg_f = background.astype(np.float32) + 1.0
    result = img_f/bg_f
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)

def hole_removal(mask, open_size=5, close_size=3):
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size,open_size))

    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)
    return cleaned