# Module for all of our test functions, which returns quantitative metrics for the pipeline.py processing steps
from math import dist
#from unittest import result

import numpy as np
import cv2
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

import display_results as disp


def mean_edge_distance(result, benchmark):
    """Compute the mean edge distance between two contours."""
    result_mask = np.asarray(result) != 0
    benchmark_mask = np.asarray(benchmark) != 0
    diff  = np.abs(result_mask.astype(int) - benchmark_mask.astype(int))

    result_edge = result_mask & ~ndi.binary_erosion(result_mask)
    benchmark_edge = benchmark_mask & ~ndi.binary_erosion(benchmark_mask)

    to_benchmark = ndi.distance_transform_edt(~benchmark_edge)
    to_result = ndi.distance_transform_edt(~result_edge)

    kernel = np.ones((6, 6), dtype=np.int8)
    superimposed = cv2.dilate(result_edge.astype(np.uint8), kernel) | cv2.dilate(benchmark_edge.astype(np.uint8), kernel)

    edge_distances = np.concatenate((
        to_benchmark[result_edge],
        to_result[benchmark_edge],
    ))
    mean_distance = edge_distances.mean()

    disp.images([result, benchmark, diff, superimposed],
                ['Result', 'Benchmark', 'Difference', 'Superimposed, mean distance: {:.2f}'.format(mean_distance)])
    
    return mean_distance

def IoU(result, benchmark):
    """Compute the Intersection over Union (IoU) between two binary masks."""
    intersection = np.logical_and(result, benchmark)
    union = np.logical_or(result, benchmark)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score