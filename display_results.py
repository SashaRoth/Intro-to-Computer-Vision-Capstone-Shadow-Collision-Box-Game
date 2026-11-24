# Module to display/print results of the pipeline.py processing steps, for debugging and visualization purposes.

import matplotlib.pyplot as plt

def images(imgs, titles):

    num = len(imgs)
    rows = 2
    cols = num//rows + (num % rows > 0)

    plt.figure(figsize=(2*num, 6))

    for i in range(num):
        plt.subplot(rows, cols, i+1)
        plt.imshow(imgs[i], cmap='gray')
        plt.axis('off')
        plt.title(titles[i])

    plt.tight_layout()
    plt.show()