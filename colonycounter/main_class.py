# -*- coding: utf-8 -*-

# import libraries
from skimage import io, measure, filters, segmentation, morphology, color, exposure
from skimage.feature import blob_dog, blob_log, blob_doh, peak_local_max

from math import sqrt
import numpy as np
import pandas as pd

from scipy import ndimage

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['image.cmap'] = 'inferno'

from .plotting_functions import (plot_bboxs, plot_texts, plot_circles,
                                 easy_sub_plot)
from .image_processing_functions import (invert_image, crop_circle,
                                         background_subtraction,
                                         search_for_blobs)


class Counter():
    """
    The main class that stores all data and process images.
    Images in a Numpy array type can be used for the initiation.
    Or you can directly import images by entering image file path.

     Attributes:
        detected_blobs (list of array): information about detected colonies
        image_bw (numpy.array): bw input image
        image_inverted_bw (numpy.array): inversed bw image
        image_raw (numpy.array): raw input image
        labeled (numpy.array): labels of image in segmentation
        props (dict) : dictionary that store region prop information (=detected objects)
        quantification_results (pandas.DataFrame): table that store quantification results (i.e. cordinates of each colony)
        quantification_summary (pandas.DataFrame): table that store colony counts
        sample_image_bw (list of array): white-bleck image for each sample
        sample_image_for_quantification (list of array): inversed images for quantification.
        sample_image_inversed_bw (list of array): inversed image

    """

    def __init__(self, image_path=None, image_array=None, verbose=True):

        """
        Images in a Numpy array type can be used for the initiation.
        Or you can directly import images by entering image file path.

        Args:
            image_path (str): file path for image data.

            image_array (numpy.array): 2d (white-black) or 3d (RGB) image array.

            verbose (bool): if True it plot the image

        """
        self.props = {}

        if not image_path is None:
            self.load_from_path(image_path, verbose=verbose)
        if not image_array is None:
            self.load_image(image_array, verbose=verbose)

    def load_from_path(self, image_path, verbose=True):
        """
        You can directly import images by entering image file path.

        Args:
            image_path (str): file path for image data.

            verbose (bool): if True it plot the image

        """
        image = io.imread(image_path)
        self.load_image(image, verbose=verbose)

    def load_image(self, image_array, verbose=True):
        """
        Images in a Numpy array type can be used for the initiation.

        Args:
            image_array (numpy.array): 2d (white-black) or 3d (RGB) image array.

            verbose (bool): if True it plot the image
        """
        self.image_raw = image_array.copy()
        self.image_bw = color.rgb2gray(self.image_raw)
        self.image_inverted_bw = invert_image(self.image_bw)

        if verbose:
            plt.imshow(self.image_raw)
            plt.show()


    def detect_area(self, verbose=True):
        """
        The method detects sample area in input image.
        Large, white and circle-like object in the input image will be
        detected as sample area.

        Args:
            verbose (bool): if True it plot the detection results

        """
        if verbose:
            print("detecting sample area...")
        # 1. Segmentation
        bw = self.image_bw.copy()
        # get elevation map
        elevation_map = filters.sobel(bw)

        # annotate marker
        markers = np.zeros_like(bw)
        markers[bw < 0.5] = 1
        markers[bw > 0.95] = 2

        # watershed
        segmentation = morphology.watershed(elevation_map, markers)

        segmentation = ndimage.binary_fill_holes(segmentation - 1)
        labeled, _ = ndimage.label(segmentation)
        self.labeled = labeled

        if verbose:
            plt.title("segmentation")
            plt.imshow(labeled)
            plt.show()

        # 2. region props
        props = np.array(measure.regionprops(label_image=labeled, intensity_image=self.image_bw))
        bboxs = np.array([prop.bbox for prop in props])
        areas = np.array([prop.area for prop in props])
        cordinates = np.array([prop.centroid for prop in props])
        eccentricities = np.array([prop.eccentricity for prop in props])


        # 3. filter object

        selected = (areas >= np.percentile(areas, 90)) & (eccentricities < 0.3)

        self._props = props[selected]
        self.props["bboxs"] = bboxs[selected]
        self.props["areas"] = areas[selected]
        self.props["cordinates"] = cordinates[selected]
        self.props["eccentricities"] = eccentricities[selected]
        self.props["names"] = [f"sample_{i}" for i in range(len(self.props["areas"]))]

        if verbose:
            print(str(len(self.props['areas'])) +" samples were detected")
            ax = plt.axes()
            plt.title("detected samples")
            ax.imshow(self.image_raw)
            plot_bboxs(bbox_list=self.props["bboxs"], ax=ax)
            plot_texts(text_list=self.props["names"], cordinate_list=self.props["bboxs"], ax=ax, shift=[0, -60])
            plt.show()




    def crop_samples(self, shrinkage_ratio=0.9):
        """
        The function will crop sample area and make indivisual picture for each sample.
        Sample area, which are supposed to be circle-like shape, will be selected.
        Signal intensity in unselected area (outside the circle) will be converted into zero.
        If you set shrinkage_ratio less than 1, the circle radius will be shrinked according to the ratio.
        In general, undesired staining background signal might appear in the edge area,
        decreasing quantification accuracy.
        The shrinkage is useful to remove the edge are of cell culture dish from quantification.

        Args:
            shrinkage_ratio (float): shrinkage_ratio to crop image. This number should be between 0 and 1.

        """
        self.sample_image_bw = [crop_circle(i.intensity_image, shrinkage_ratio) for i in self._props]
        self.sample_image_inversed_bw = [crop_circle(invert_image(i.intensity_image), shrinkage_ratio) for i in self._props]
        self.sample_image_for_quantification = self.sample_image_inversed_bw.copy()

    def plot_cropped_samples(self, inverse=False, col_num=4):
        """
        The function plots the cropped area.

        Args:
            inverse (bool): if True the inversed image will be plotted.

            col_num (int): the number of column in subplot.

        """

        if not inverse:
            image_list = self.sample_image_bw
            vmax = _get_vmax(image_list)
            easy_sub_plot(image_list, col_num, self.props["names"], args={"cmap": "gray", "vmin": 0, "vmax": vmax})

        if inverse:
            image_list = self.sample_image_inversed_bw
            vmax = _get_vmax(image_list)
            easy_sub_plot(image_list, col_num, self.props["names"], args={"vmin": 0, "vmax": vmax})


    def adjust_contrast(self, verbose=True, reset_image=False):
        """
        Function for contrast adjustment.
        This step is not necessary if signal contrast is high enough.
        Use this function just when the quantification does not work because of low contrast.

        Args:
            verbose (bool): if True, images before and after the this process will be plotted.

            reset_image (bool): if True the image will be reset before contrast adjustment.
                if it is not True, the function will use pre-processed image by another function (i.g. background subtraction).
        """
        if reset_image:
            self.sample_image_for_quantification = self.sample_image_inversed_bw.copy()
        if verbose:
            print("before_contrast_adjustment")
            vmax = _get_vmax(self.sample_image_for_quantification)
            easy_sub_plot(self.sample_image_for_quantification, 4, self.props["names"], {"vmin":0, "vmax": vmax})
            plt.show()


        for i, image in enumerate(self.sample_image_for_quantification):
            result = exposure.adjust_log(image, 1)
            #result = result - result[0,0]
            self.sample_image_for_quantification[i] = result

        if verbose:
            print("after_contrast_adjustment")
            vmax = _get_vmax(self.sample_image_for_quantification)
            easy_sub_plot(self.sample_image_for_quantification, 4, self.props["names"], {"vmin":0, "vmax": vmax})
            plt.show()


    def subtract_background(self, sigma=1, verbose=True, reset_image=True):
        """
        Function for background subtraction.
        This step is not necessary if the image does not contain high background noise.
        Use this function just when the quantification does not work because of background noise.

        Args:
            verbose (bool): if True, images before and after the this process will be plotted.

            reset_image (bool): if True the image will be reset before background adjustment.

        """
        if reset_image:
            self.sample_image_for_quantification = self.sample_image_inversed_bw.copy()
        if verbose:
            print("before_background_subtraction")
            vmax = _get_vmax(self.sample_image_for_quantification)
            easy_sub_plot(self.sample_image_for_quantification, 4, self.props["names"], {"vmin":0, "vmax": vmax})
            plt.show()

        for i, image in enumerate(self.sample_image_for_quantification):
            result = background_subtraction(image=image, sigma=sigma, verbose=False)
            result = result - result[0,0]
            result[result<0] = 0
            self.sample_image_for_quantification[i] = result

        if verbose:
            print("after_background_subtraction")
            vmax = _get_vmax(self.sample_image_for_quantification)
            easy_sub_plot(self.sample_image_for_quantification, 4, self.props["names"],  {"vmin":0, "vmax": vmax})
            plt.show()

    def detect_colonies(self, min_size=5, max_size=15, threshold=0.02, verbose=True):
        """
        Function for colony detection.
        Using inversed sample image, this function will detect particles with Laplacian of Gaussian method.
        (https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html)
        It is important to set appropriate colony size (minimun and maximum size).
        The size depends on the imaging pixel size.
        It is also important to set appropriate threshold, which is threshold value for signal intensity.
        It is highly recommended to try several values searching for appropriate parameters.

        Args:
            min_size (int, float) : minimum colony size.

            max_size (int, float) : maximum colony size.

            threshold (float) : threshold for local contrast signal intensity. You will get more colony if you decrease this value.

            verbose (bool): if True, the resutls wil be shown.

        """



        self.detected_blobs = []
        for image in self.sample_image_for_quantification:
            blobs = search_for_blobs(image=image, min_size=min_size, max_size=max_size,
                                     threshold=threshold, verbose=False)
            self.detected_blobs.append(blobs)

        # save result as a dataFrame
        result = []
        for i, blobs in enumerate(self.detected_blobs):
            df = pd.DataFrame(blobs, columns=["x", "y", "radius"])
            df["sample"] = self.props["names"][i]
            result.append(df)
        result = pd.concat(result, axis=0)
        self.quantification_results = result

        # summarize results
        summary = result.groupby('sample').count()
        summary = summary[["x"]]
        summary.columns = ["colony_count"]
        self.quantification_summary = summary

        if verbose:
            self.plot_detected_colonies()

    def plot_detected_colonies(self, plot="final", col_num=4):
        """
        Function to plot detected colonies detection.

        Args:
            plot_raw (bool) : if True, the white-black image will be shown. If it is not True, inversed image will be used for the plotting.

            col_num (int): the number of column in subplot.

        """
        if plot == "raw":
            image_list = self.sample_image_bw
        elif plot == "final":
            image_list = self.sample_image_for_quantification
        elif plot == "raw_inversed":
            image_list = self.sample_image_inversed_bw


        vmax = _get_vmax(image_list)

        for i, image in enumerate(image_list):

            k = (i%col_num + 1)
            ax = plt.subplot(1, col_num, k)
            blobs = self.detected_blobs[i]
            if plot == "raw":
                plt.imshow(image, cmap="gray", vmin=0, vmax=vmax)
                plot_circles(circle_list=blobs, ax=ax, args={"color": "black"})

            else:
                plt.imshow(image, vmin=0, vmax=vmax)
                plot_circles(circle_list=blobs, ax=ax)

            name = self.props["names"][i]
            plt.title(f"{name}: {len(blobs)} colonies")
            if (k == col_num) | (i == len(image_list)):
                plt.show()

def _get_vmax(image_list):
    vmax = []
    for i in image_list:
        vmax.append(i.max())
    vmax = np.max(vmax)
    return vmax
