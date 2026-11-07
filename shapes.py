
# Module that defines various structuring elements for morphological operations.

import numpy as np

def disk(radius):
    """Return a disk-shaped structuring element with the given radius."""
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    return mask.astype(np.uint8)

def rectangle(width, height):
    """Return a rectangle-shaped structuring element with the given width and height."""
    width = int(width)
    height = int(height)
    return np.ones((height, width), dtype=np.uint8)