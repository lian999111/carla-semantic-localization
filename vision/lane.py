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

    def __init__(self, M, px_per_meter_x, px_per_meter_y, warped_size, valid_mask, lane_config_args):
        """ 
        Constructor method. 

        Input:
            M: Numpy.array of 3x3 matrix for IPM transform.
            px_per_meter_x: Pixels per meters in x in the Bird's eye view.
            px_per_meter_y: Pixels per meters in y in the Bird's eye view.
            warped_size: Image size tuple of IPM image (width, height).
            valid_mask: Numpy.array of booleans to mask out edge pixels caused by the border of the FOV.
            lane_config_args: Dict object storing algorithm related parameters.
        """
        self.ipm_tform = M  # ipm stands for inverse perspective mapping
        self.warped_size = warped_size
        self.valid_mask = valid_mask
        self.px_per_meters_x = px_per_meter_x
        self.px_per_meters_y = px_per_meter_y

        # Algorithm related parameters
        # Dilation
        self.dilation_iter = lane_config_args['dilation']['n_iters']
        # Rotation
        self.hough_thres = lane_config_args['rotation']['hough_thres']
        # Histogram
        self.required_height = lane_config_args['histo']['required_height']
        self.n_bins = lane_config_args['histo']['n_bins']
        # Sliding window
        self.n_windows = lane_config_args['sliding_window']['n_windows']
        self.margin = lane_config_args['sliding_window']['margin']
        self.recenter_minpix = lane_config_args['sliding_window']['recenter_minpix']

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

    def _try_rotate_image(self, edge_image):
        """
        Try rotating edge image if the major lines in the lower half of edge image are tilted.

        This method uses Hough transform to find major lines. If the median angle of lines found 
        is noticable, rotate the image around the center of image's lower bottom to make it more vertical.
        """
        # When the heading angle difference between ego vehicle and lane is too high, using histogram to find
        # starting search points is prone to fail. This method uses Hough transform to find major lines. If the
        # median angle of lines found is noticable, rotate the image around the center of image's lower bottom.
        # After rotation, lane markings should be more vertical and thus histogram is less likely to fail.
        lines = cv2.HoughLines(
            edge_image[edge_image.shape[0]//2:, :], 2, np.pi/180, self.hough_thres)

        # When hough transform can't find lines in the lower half of image,
        # it's an indication that there is no good lines to detect
        if lines is not None:
            # Convert the angle range from (0, pi) to (-pi/2, pi/2)
            # Those with angle larger than pi/2 are also those with negative rhos.
            lines[lines[:, :, 0] < 0, 1] -= np.pi
            # Convert to degree
            rot_angle = np.median(lines[:, :, 1]) * 180 / np.pi
        else:
            rot_angle = None

        # %% Rotate image
        if rot_angle and abs(rot_angle) > 5:
            rot_center = (self.warped_size[0]//2, self.warped_size[1])
            M_rot = cv2.getRotationMatrix2D(rot_center, rot_angle, scale=1)
            rot_image = cv2.warpAffine(edge_image, M_rot, self.warped_size)
        else:
            rot_image = None

        return rot_angle, rot_image

    def _get_histo(self, edge_image):
        """ 
        Get histogram of edge image.
        
        The peaks in histogram is then used as starting points for sliding window search. 
        """
        # Only the lower third image is used since we focus on the starting points
        histogram, _ = np.histogram(edge_image[int(edge_image.shape[0]/3):, :].nonzero()[1], bins=self.n_bins, range=(0, self.warped_size[0]))
        bin_width = edge_image.shape[1] / self.n_bins

        return histogram, bin_width

    def _find_histo_peaks(self, histogram):
        """ Find at most 2 peaks as lane marking searching bases from histogram. """
        # Find peaks above required height
        peaks, _ = find_peaks(histogram, height=self.required_height)

        # Remove peaks that are too close to their precedents
        # diff = np.diff(peaks)
        # delete_mask = np.concatenate(([False], diff < 40))
        # peaks = np.delete(peaks, delete_mask)

        # Find at most 2 peaks from the middle towards left and 2 towards right
        half_idx = histogram.shape[0]/2
        left_base_bin = peaks[peaks < half_idx][-1] if peaks[peaks <
                                                        half_idx].size != 0 else None
        right_base_bin = peaks[peaks >= half_idx][0] if peaks[peaks >=
                                                        half_idx].size != 0 else None

        return left_base_bin, right_base_bin

    def _sliding_window_search(self, edge_image, left_base, right_base):
        """ 
        Find lane marking edge pixels using sliding window. 
        
        Input:
            edge_image: Bird's eye image of edges.
            left_base: Starting point to search for left marking.
            right_base: Starting point to search for right marking.
        Output:
            left_idc: Indices of possible left marking points.
            right_idc: Indices of possible right marking points.
            nonzerox: X coordinates of non-zero pixels in the edge image.
            nonzeroy: Y coordinates of non-zero pixels in the edge image.
        """
        if __debug__:
            # Create an output image to draw on and  visualize the result
            debug_img = edge_image.copy()
        # Set height of windows
        window_height = np.int(edge_image.shape[0]/n_windows*2/3)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = edge_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Create empty lists to store left and right lane pixel indices
        left_idc = []
        right_idc = []

        # TODO: remove this?
        # # Local function to set initial window's y position
        # def set_init_window_y(window_x):
        #     img_height = edge_image.shape[0]
        #     img_width = edge_image.shape[1]
        #     window_y = img_height
        #     if window_x < invalid_tri_w:
        #         window_y = int(
        #             img_height + (window_x / invalid_tri_w - 1) * invalid_tri_h)
        #     elif window_x > img_width - invalid_tri_w:
        #         window_y = int(img_height + ((img_width - window_x) /
        #                                     invalid_tri_w - 1) * invalid_tri_h)
        #     return window_y

        shift = 0

        if left_base:
            # Initial windows' positions
            leftx_curr = int(left_base)

            # TODO: remove this?
            # if invalid_tri_w:
            #     lefty_curr = set_init_window_y(leftx_curr)
            # else:
            #     lefty_curr = edge_image.shape[0]
            lefty_curr = edge_image.shape[0]

            # rightx_shift = 0
            search_left = True
        else:
            search_left = False

        if right_base:
            rightx_curr = int(right_base)

            # TODO: remove this?
            # if invalid_tri_w:
            #     righty_curr = set_init_window_y(rightx_curr)
            # else:
            #     righty_curr = edge_image.shape[0]
            righty_curr = edge_image.shape[0]

            # leftx_shift = 0
            search_right = True
        else:
            search_right = False

        for win_count in range(self.n_windows):
            # Left markings
            if search_left:
                # Vertical
                win_yleft_low = lefty_curr - (win_count + 1) * window_height
                win_yleft_high = lefty_curr - win_count * window_height
                # Horizontal
                win_xleft_low = leftx_curr - self.margin
                win_xleft_high = leftx_curr + self.margin
                good_left_idc = ((nonzeroy >= win_yleft_low) & (nonzeroy < win_yleft_high) &
                                (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]

                # TODO: Remove this?
                # if good_left_idc.size == 0 or min(nonzeroy[good_left_idc]) > (win_yleft_low + 0.1*window_height):
                #     win_xleft_low = int(leftx_curr - scale * self.margin)
                #     win_xleft_high = int(leftx_curr + scale * self.margin)
                #     good_left_idc = ((nonzeroy >= win_yleft_low) & (nonzeroy < win_yleft_high) &
                #                     (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]

                if __debug__:
                    cv2.rectangle(debug_img, (win_xleft_low, win_yleft_low),
                                (win_xleft_high, win_yleft_high), 1, 2)

                left_idc += list(good_left_idc)
                # Recenter next window if enough points found in current window
                if len(good_left_idc) > self.recenter_minpix:
                    newx = np.int(np.mean(nonzerox[good_left_idc]))
                    shift = newx - leftx_curr
                    leftx_curr = newx
                else:
                    leftx_curr += shift

                if leftx_curr > edge_image.shape[1] - self.margin or leftx_curr < 0:
                    search_left = False

            # Right markings
            if search_right:
                # Vertical
                win_yright_low = righty_curr - (win_count + 1) * window_height
                win_yright_high = righty_curr - win_count * window_height
                # Horizontal
                win_xright_low = rightx_curr - self.margin
                win_xright_high = rightx_curr + self.margin
                good_right_idc = ((nonzeroy >= win_yright_low) & (nonzeroy < win_yright_high) &
                                (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

                # TODO: Remove this?
                # if good_right_idc.size == 0 or min(nonzeroy[good_right_idc]) > (win_yright_low + 0.1*window_height):
                #     win_xright_low = int(rightx_curr - scale * self.margin)
                #     win_xright_high = int(rightx_curr + scale * self.margin)
                #     good_right_idc = ((nonzeroy >= win_yright_low) & (nonzeroy < win_yright_high) &
                #                     (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                if __debug__:
                    cv2.rectangle(debug_img, (win_xright_low, win_yright_low),
                                (win_xright_high, win_yright_high), 1, 2)

                right_idc += list(good_right_idc)
                # Recenter next window if enough points found in current window
                if len(good_right_idc) > self.recenter_minpix:
                    newx = np.int(np.mean(nonzerox[good_right_idc]))
                    shift = newx - rightx_curr
                    rightx_curr = newx
                else:
                    rightx_curr += shift

                if rightx_curr > edge_image.shape[1] - self.margin or rightx_curr < 0:
                    search_right = False
        if __debug__:
            plt.imshow(debug_img)
            plt.show()

        return left_idc, right_idc, nonzerox, nonzeroy
