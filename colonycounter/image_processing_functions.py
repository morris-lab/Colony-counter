# -*- coding: utf-8 -*-



from skimage import io, measure, filters, segmentation, morphology, color, exposure
from skimage.draw import circle

from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny

from skimage.feature import blob_dog, blob_log, blob_doh, peak_local_max

from math import sqrt
import numpy as np
from scipy import ndimage




def invert_image(image):
    return (image - 1 )*(-1)

def detect_circle_by_canny(image_bw, radius=395, n_peaks=20):
    edges = canny(image_bw, sigma=2)
    hough_res = hough_circle(edges, [radius])
    accums, cx, cy, radii = hough_circle_peaks(hough_res, [radius],
                                               total_num_peaks=n_peaks)

    label = np.zeros_like(image_bw)
    ind = 1
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle(center_y, center_x, radius,
                                        shape=image_bw.shape)
        label[circy, circx] = ind
        ind += 1

    return label.astype(np.int)

def _get_radius(bbox):
    minr, minc, maxr, maxc = bbox
    r = np.array([[maxr - minr], [maxc - minc]]).mean()/2
    return r

def _bbox_to_center(bbox):
    minr, minc, maxr, maxc = bbox
    c = int(np.array([minc, maxc]).mean())
    r = int(np.array([minr, maxr]).mean())
    return r, c

def make_circle_label(bb_list, img_shape):

    # get radious
    radius = np.median([_get_radius(i) for i in bb_list])

    # draw circles in an image
    label = np.zeros(img_shape)
    id = 1
    for bb in bb_list:
        # get centroid
        r, c = _bbox_to_center(bb)

        # draw circle
        rr, cc = circle(r, c, radius)
        label[rr, cc] = id
        id += 1

    return label.astype(np.int)

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

def search_for_blobs(image, min_size=3, max_size=15, num_sigma=10, overlap=0.5, threshold=0.02, verbose=True):

    # detect blobs
    blobs_log = blob_log(image, max_sigma=max_size, min_sigma=min_size, num_sigma=num_sigma, overlap=overlap,
                         threshold=threshold, log_scale=True)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    if verbose:
        ax = plt.axes()
        plt.imshow(image)
        plot_circles(circle_list=blobs_log, ax=ax)

    return blobs_log
