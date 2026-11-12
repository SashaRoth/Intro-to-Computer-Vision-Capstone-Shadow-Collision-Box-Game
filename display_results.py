# Module to display/print results of the pipeline.py processing steps, for debugging and visualization purposes.

import matplotlib.pyplot as plt

def images(img, img_gray, img_thresh, img_cont, img_filled, img_poly):
    plt.figure(figsize=(16, 4))

    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('Original')

    plt.subplot(2, 3, 2)
    plt.imshow(img_gray, cmap='gray')
    plt.axis('off')
    plt.title('Processed')

    plt.subplot(2, 3, 3)
    plt.imshow(img_thresh, cmap='gray')
    plt.axis('off')
    plt.title('Thresholded')

    plt.subplot(2, 3, 4)
    plt.imshow(img_cont, cmap='gray')
    plt.axis('off')
    plt.title('Largest Contour')

    plt.subplot(2, 3, 5)
    plt.imshow(img_filled, cmap='gray')
    plt.axis('off')
    plt.title('Filled')

    plt.subplot(2, 3, 6)
    plt.imshow(img_poly, cmap='gray')
    plt.axis('off')
    plt.title('Polylines')

    plt.show()