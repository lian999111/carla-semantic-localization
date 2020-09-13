# Implementation of lane extraction from semantic segmentation images

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class LaneMarkingDetector(object):
    """ 
    Class for lane marking detection from semantic segmentation images. 

    It performs inverse perspective mapping (IPM) to obtain the bird's eye view, 
    then uses 2 approaches to detect lane marking pixels:
        - Sliding window: Used when there was no lane marking detected in previous images.
        - CTRV motion model: Used to predict the current positions of lane markings. 
          It requires lane markings to be detected in previous images.
    """

    def __init__(self, M, warped_size=(400, 600), valid_mask=None):
        """ 
        Constructor method. 

        Input:
            M: np.array of 3x3 matrix for IPM transform
            warped_size: image size tuple of IPM image (width, height).
            valid_mask: np.array of booleans to mask out edge pixels caused by the border of the FOV.
        """
        self.ipm_tform = M  # ipm stands for inverse perspective mapping
        self.warped_size = warped_size
        self.valid_mask = valid_mask

    def _get_bev_edge(self, lane_image):
        """ 
        Get edge image in the bird's eye view.

        Input:
            lane_image: OpenCV image with supported data type (e.g. np.uint8). The image should have non-zero values only
            at lane-related pixels, which is easy to obtained from a semantic image.
        Output:
            Edge image in the bird's eye view.
        """

        # IPM transform
        warped_image = cv2.warpPerspective(
            lane_image, self.ipm_tform, self.warped_size)

        # Extract edges
        # Vertical Sobel filter is enough in our case, since we have crisp clear semantic image and only care about vertical edges
        edge_image = cv2.Sobel(warped_image, cv2.CV_64F, 1, 0, ksize=3)
        edge_image[edge_image != 0] = 1     # turn every non-zero pixels into 1

        # Remove edges caused by the border of the FOV using a predefined mask
        if self.valid_mask is not None:
            edge_image = edge_image.astype(np.uint8) & self.valid_mask

        return edge_image
