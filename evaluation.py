# Module for all of our test functions, which returns quantitative metrics for the pipeline.py processing steps
from math import dist

import numpy as np
import cv2
from scipy import ndimage as ndi
import matplotlib.pyplot as plt


def mean_edge_distance(result, benchmark):
    """Compute the mean edge distance between two contours."""
    intersection = abs(result.astype(np.int32) - benchmark.astype(np.int32))
    mask, distances = ndi.distance_transform_edt(intersection, sampling=None, return_distances=True, return_indices=False, distances=None, indices=None) # as used in activity_07 task 3.2

    plt.figure(figsize=(4, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(result, cmap='gray')
    plt.axis('off')
    plt.title('result')

    plt.subplot(1, 4, 2)
    plt.imshow(benchmark, cmap='gray')
    plt.axis('off')
    plt.title('benchmark')

    plt.subplot(1, 4, 3)
    plt.imshow(intersection, cmap='gray')
    plt.axis('off')
    plt.title('intersection')

    plt.subplot(1, 4, 4)
    plt.imshow(distances, cmap='gray')
    plt.axis('off')
    plt.title('distances')
    
    plt.show()
    
    return None

def IoU(result, benchmark):
    """Compute the Intersection over Union (IoU) between two binary masks."""
    intersection = np.logical_and(result, benchmark)
    union = np.logical_or(result, benchmark)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score