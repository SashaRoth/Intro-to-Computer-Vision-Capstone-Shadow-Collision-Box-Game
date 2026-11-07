# Module to display/print results of the pipeline.py processing steps, for debugging and visualization purposes.

import matplotlib.pyplot as plt

def images(img, img_gray, img_thresh, overlapping):
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