# standard imports
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


# custom imports
import cvat
import shapes
import image_processing as ip
import display_results as disp
import evaluation as eval

# parameters
sigma = 10.0
blur_size = (int(sigma*6+1), int(sigma*6+1))
structure_element_size = 100
open_intensity = 70
close_intensity = 200

# Accept an image filename with or without its extension.
image_argument = sys.argv[1] if len(sys.argv) > 1 else "testshadow"
filename, extension = os.path.splitext(os.path.basename(image_argument))
supported_extensions = (".png", ".jpg", ".jpeg")
if extension and extension.lower() not in supported_extensions:
    raise ValueError(f"Unsupported image extension: {extension}")
extensions = (extension,) if extension else supported_extensions
image_path = next(
    (os.path.join("images", filename + ext) for ext in extensions
     if os.path.isfile(os.path.join("images", filename + ext))),
    None,
)
if image_path is None:
    raise FileNotFoundError(f"No matching image found for {image_argument!r} in 'images'.")

img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
if img is None:
    raise ValueError(f"OpenCV could not read image: {image_path}")
benchmark = cvat.load_image(filename)

# Check the result
print(type(img))        # <class 'numpy.ndarray'>
print(img.shape)        # (height, width, channels)
print(img.dtype)        # usually uint8
print(f"Value range: [{img.min()}, {img.max()}]")

# step #1: preprocess the image (blur + grayscale)
img_blur = cv2.GaussianBlur(img, blur_size, sigma)
img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

# step #2: Background division <-- SAGE ADD HERE
...

# step #3: Global OTSU thresholding
threshold = ip.otsu_threshold(img_gray) #<-- custom function to find OTSU threshold
img_thresh = (img_gray < threshold) * 255

# step #4: Find contours, keep only the largest one
img_cont = ip.keep_largest_contour(img_thresh) #<-- custom function to keep only largest object

# step #5: Remove holes to create 1 continuous region <-- SAGE ADD HERE
img_filled = img_cont

# step #6: Trace with polylines
img_poly = img_filled

# Display each step of the pipeline
disp.images([img, img_gray, img_thresh, img_cont, img_filled, img_poly],
            ['Original', 'Grayscale', 'Thresholded', 'Contour', 'Filled', 'Polyline'])

mean_dist = eval.mean_edge_distance(img_poly, benchmark)
IoU_score = eval.IoU(img_poly, benchmark)

print(f"Mean edge distance: {mean_dist:.2f}")
print(f"Intersection over Union: {IoU_score:.2f}")