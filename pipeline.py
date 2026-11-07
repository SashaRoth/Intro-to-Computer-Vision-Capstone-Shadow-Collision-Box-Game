# standard imports
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


# custom imports
import cvat
import shapes
import image_processing as ip
import display_results as disp

# parameters
sigma = 10.0
blur_size = (int(sigma*6+1), int(sigma*6+1))
structure_element_size = 100
Clahe_Cliplimit = 2.0
Clahe_Size = 30
open_intensity = 70
close_intensity = 200

# parse arguments -- the only argument is the filename of the image to process, without extension (e.g. "testshadow" for "testshadow.png")
filename = sys.argv[1] if len(sys.argv) > 1 else "testshadow"

# Read image
img = cv2.imread(f"images/{filename}.png", cv2.IMREAD_UNCHANGED)

# Check the result
print(type(img))        # <class 'numpy.ndarray'>
print(img.shape)        # (height, width, channels)
print(img.dtype)        # usually uint8
print(f"Value range: [{img.min()}, {img.max()}]")

# step #1: preprocess the image (blur + resharpen, grayscale, CLAHE)
img_blur = cv2.GaussianBlur(img, blur_size, sigma)
img_sharp = cv2.addWeighted(img, 1.5, img_blur, -0.5, 0)
img_gray = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2GRAY)

# step #2: Global OTSU thresholding
threshold = ip.otsu_threshold(img_gray)
img_thresh = (img_gray < threshold) * 255

# step #3: Clean up with close->open
img_thresh = ip.clean_mask(img_thresh, open_intensity, close_intensity)

# step #4: Reduce to only 1 foreground and 1 background object
num_labels, overlapping = cv2.connectedComponents(img_thresh.astype(np.uint8), connectivity=4)
print(f"Number of objects: {num_labels}")
overlapping = (overlapping == np.max(overlapping)) * 255

# Display the original, greyscaled/processed, and thresholded images
disp.images(img, img_gray, img_thresh, overlapping)