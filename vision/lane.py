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

    def __init__(self, M, px_per_meter_x, px_per_meter_y, warped_size=(400, 600), valid_mask=None):
        """ 
        Constructor method. 

        Input:
            M: Numpy.array of 3x3 matrix for IPM transform.
            px_per_meter_x: Pixels per meters in x in the Bird's eye view.
            px_per_meter_y: Pixels per meters in y in the Bird's eye view.
            warped_size: Image size tuple of IPM image (width, height).
            valid_mask: Numpy.array of booleans to mask out edge pixels caused by the border of the FOV.
        """
        self.ipm_tform = M  # ipm stands for inverse perspective mapping
        self.warped_size = warped_size
        self.valid_mask = valid_mask
        self.px_per_meters_x = px_per_meter_x
        self.px_per_meters_y = px_per_meter_y

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

    def _find_histo_peaks(histo, required_height=500):
    """ Find at most 2 peaks as lane marking searching bases from histogram. """
    # Find peaks above required height
    peaks, _ = find_peaks(histogram, height=required_height)

    # Remove peaks that are too close to their precedents
    # diff = np.diff(peaks)
    # delete_mask = np.concatenate(([False], diff < 40))
    # peaks = np.delete(peaks, delete_mask)

    # Find at most 2 peaks from the middle towards left and 2 towards right
    half_idx = histo.shape[0]/2
    left_base = peaks[peaks < half_idx][-1] if peaks[peaks < half_idx].size != 0 else None
    right_base = peaks[peaks >= half_idx][0] if peaks[peaks >= half_idx].size != 0 else None

    return left_base, right_base

    
