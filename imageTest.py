import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import label

# parameters
sigma = 10.0
blur_size = (int(sigma*6+1), int(sigma*6+1))
structure_element_size = 100
Clahe_Cliplimit = 2.0
Clahe_Size = 30
open_intensity = 70
close_intensity = 200

# functions
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
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, rectangle(open, open/2))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, rectangle(close, close/2))
    return closed.astype(bool)

def count_objects(mask, connectivity=2):
    """Number of connected components (1 = 4-connected, 2 = 8-connected)."""
    # TODO
    array, count = label(mask, connectivity=connectivity, return_num=True)
    return count

img = cv2.imread("images/shadow2.png", cv2.IMREAD_UNCHANGED)

# Check the result
print(type(img))        # <class 'numpy.ndarray'>
print(img.shape)        # (height, width, channels)
print(img.dtype)        # usually uint8
print(f"Value range: [{img.min()}, {img.max()}]")

# step #1: preprocess the image (blur + resharpen, grayscale, CLAHE)
img_blur = cv2.GaussianBlur(img, blur_size, sigma)
img_sharp = cv2.addWeighted(img, 1.5, img_blur, -0.5, 0)
img_gray = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2GRAY)
#clahe = cv2.createCLAHE(clipLimit=Clahe_Cliplimit, tileGridSize=(Clahe_Size, Clahe_Size))
#img_gray = clahe.apply(img_gray)

# NOTE FOR SAGE: I tried using tophat first, but it fucked up the image cause the shadow takes up half the frame lol, the structuring element has to be wayyy too huge

# step #2: Global OTSU thresholding
threshold = otsu_threshold(img_gray)
img_thresh = (img_gray < threshold) * 255

# step #3: Clean up with close->open
img_thresh = clean_mask(img_thresh, open_intensity, close_intensity)

# step #4: Reduce to only 1 foreground and 1 background object
num_labels, overlapping = cv2.connectedComponents(img_thresh.astype(np.uint8), connectivity=4)
print(f"Number of objects: {num_labels}")
overlapping = (overlapping == np.max(overlapping)) * 255

# Display the original, CLAHE, and thresholded images
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('Original')

plt.subplot(1, 4, 2)
plt.imshow(img_gray, cmap='gray')
plt.axis('off')
plt.title('Processed')

plt.subplot(1, 4, 3)
plt.imshow(img_thresh, cmap='gray')
plt.axis('off')
plt.title('Thresholded')

plt.subplot(1, 4, 4)
plt.imshow(overlapping, cmap='gray')
plt.axis('off')
plt.title('Connected Components')

plt.show()