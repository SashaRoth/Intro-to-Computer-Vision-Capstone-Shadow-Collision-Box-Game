# Module to display/print results of the pipeline.py processing steps, for debugging and visualization purposes.

import matplotlib.pyplot as plt

def images(imgs, titles):

    num = len(imgs)

    plt.figure(figsize=(4*num, 4))

    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.imshow(imgs[i], cmap='gray')
        plt.axis('off')
        plt.title(titles[i])

    plt.tight_layout()
    plt.show()