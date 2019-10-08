# -*- coding: utf-8 -*-



from skimage import io, measure, filters, segmentation, morphology, color, exposure
from skimage.feature import blob_dog, blob_log, blob_doh, peak_local_max

from math import sqrt
import numpy as np
from scipy import ndimage




def invert_image(image):
    return (image - 1 )*(-1)


def crop_circle(image, shrinkage_ratio=0.95):

    x, y = image.shape
    center = (int(x/2), int(y/2))
    diameter = min(center)

    threshold = (diameter*shrinkage_ratio)**2

    # initialize mask
    mask = np.zeros_like(image)

    # crop as circle.
    for x_ in range(x):
        for y_ in range(y):
            dist = (x_ - center[0])**2 + (y_ - center[1])**2
            mask[x_, y_] = (dist < threshold)

    return image*mask

def background_subtraction(image, sigma=1, verbose=True):

    image_ = ndimage.gaussian_filter(image, sigma)
    #seed = np.copy(image_)
    #seed[1:-1, 1:-1] = image_.min()
    #image_ = image.copy()
    seed = image_ - 0.4
    mask = image_
    dilated = morphology.reconstruction(seed, mask, method='dilation')
    result = image_ - dilated

    if verbose:
        plt.subplot(1,4,1)
        plt.imshow(image)

        plt.subplot(1,4,2)
        plt.imshow(image_)

        plt.subplot(1,4,3)
        plt.imshow(dilated)

        plt.subplot(1,4,4)
        plt.imshow(result)
        plt.show()
    return result

def search_for_blobs(image, min_size=3, max_size=15, threshold=0.02, verbose=True):

    # detect blobs
    blobs_log = blob_log(image, max_sigma=max_size, min_sigma=min_size,
                         threshold=threshold, log_scale=True)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    if verbose:
        ax = plt.axes()
        plt.imshow(image)
        plot_circles(circle_list=blobs_log, ax=ax)

    return blobs_log
