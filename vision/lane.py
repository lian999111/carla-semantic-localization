# Implementation of lane extraction from semantic segmentation images

import os
import pickle
import argparse
import yaml
import numpy as np
import cv2
import glob
from scipy.signal import find_peaks
from math import sin, cos

if __debug__:
    import matplotlib.pyplot as plt


def image_side_by_side(leftImg, leftTitle, rightImg, rightTitle, figsize=(20, 10), leftCmap=None, rightCmap=None):
    """
    Display the images `leftImg` and `rightImg` side by side with image titles.
    """
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    if leftCmap == None:
        axes[0].imshow(leftImg)
    else:
        axes[0].imshow(leftImg, cmap=leftCmap)
    axes[0].set_title(leftTitle)

    if rightCmap == None:
        axes[1].imshow(rightImg)
    else:
        axes[1].imshow(rightImg, cmap=rightCmap)
    axes[1].set_title(rightTitle)


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
        self.histo_region = lane_config_args['histo']['histo_region']
        self.required_height = lane_config_args['histo']['required_height']
        self.n_bins = lane_config_args['histo']['n_bins']
        # Sliding window
        self.search_region = lane_config_args['sliding_window']['search_region']
        self.n_windows = lane_config_args['sliding_window']['n_windows']
        self.margin = lane_config_args['sliding_window']['margin']
        self.recenter_minpix = lane_config_args['sliding_window']['recenter_minpix']
        # Fitting
        self.sampling_ratio = lane_config_args['fitting']['sampling_ratio']

    def find_marking_points(self, lane_image):
        """
        Find marking points using the lane semantic image.

        Input:
            lane_image: OpenCV image with supported data type (e.g. np.uint8). The image should have non-zero values only
            at lane-related pixels, which is easy to obtained from a semantic image.
        Output:
            left_coords_ego: x-y coordinates of detected left lane marking points in ego frame (z-up)
            right_coords_ego: x-y coordinates of detected right lane marking points in ego frame (z-up)
        """

        # Get bird's eye view edge image
        # Sidewalk borders are also extracted in this step
        edge_img = self._get_bev_edge(lane_image)
        # Dilate edge image so markings are thicker and easier to detect.
        dilated_edge_image = cv2.dilate(
            edge_img, kernel=np.ones((3, 3), np.uint8), iterations=self.dilation_iter)

        if __debug__:
            image_side_by_side(edge_img, 'Edge Image',
                               dilated_edge_image, 'Dilated Edge Image')
            plt.show(block=False)

        # When the heading angle difference between ego vehicle and lane is too high, using histogram to find
        # starting search points is prone to fail. This method uses Hough transform to find major lines. If the
        # median angle of lines found is noticable, rotate the image around the center of image's bottom.
        # After rotation, lane markings should be more vertical and thus histogram is less likely to fail.
        rot_image, rot_angle = self._try_rotate_image(dilated_edge_image)

        # Get histogram peaks and corresponding bases for sliding window search
        histo, bin_width = self._get_histo(rot_image)

        if __debug__:
            plt.figure()
            plt.plot(histo)
            plt.show(block=False)

        left_base_bin, right_base_bin = self._find_histo_peaks(histo)
        left_base = left_base_bin*bin_width + bin_width/2 if left_base_bin else None
        right_base = right_base_bin*bin_width + bin_width/2 if right_base_bin else None

        # Sliding window search
        left_idc, right_idc, nonzerox, nonzeroy = self._sliding_window_search(
            rot_image, left_base, right_base)

        # Recenter coordinates at the center of the image's bottom
        # Now the coordinates are aligned with ego frame (x forwards, y leftwards)
        left_coords = np.array([edge_img.shape[0] - nonzeroy[left_idc],
                                edge_img.shape[1]//2 - nonzerox[left_idc]])

        right_coords = np.array([edge_img.shape[0] - nonzeroy[right_idc],
                                 edge_img.shape[1]//2 - nonzerox[right_idc]])

        # Down sampling to reduce number of points to process
        if left_coords.size != 0:
            left_coords = left_coords[:, 0::int(1/self.sampling_ratio)]
        if right_coords.size != 0:
            right_coords = right_coords[:, 0::int(1/self.sampling_ratio)]

        # Rotate coordinates back to the original orientation if image was rotated
        if rot_angle:
            sin_minus_rot_angle = sin(-rot_angle)
            cos_minus_rot_angle = cos(-rot_angle)
            rotm = np.array([[cos_minus_rot_angle, -sin_minus_rot_angle],
                             [sin_minus_rot_angle, cos_minus_rot_angle]])
            left_coords = rotm @ left_coords
            right_coords = rotm @ right_coords

        # Map from pixels to meters
        left_coords_ego = left_coords.astype(np.float)
        right_coords_ego = right_coords.astype(np.float)
        left_coords_ego[0, :] = left_coords_ego[0, :] / self.px_per_meters_x
        left_coords_ego[1, :] = left_coords_ego[1, :] / self.px_per_meters_y
        right_coords_ego[0, :] = right_coords_ego[0, :] / self.px_per_meters_x
        right_coords_ego[1, :] = right_coords_ego[1, :] / self.px_per_meters_y

        if __debug__:
            _, ax = plt.subplots(ncols=2)
            ax[0].plot(-left_coords[1], left_coords[0], '.')
            ax[0].plot(-right_coords[1], right_coords[0], '.')
            ax[0].set_aspect('equal')
            ax[0].set_xlim(-self.warped_size[0]/2, self.warped_size[0]/2)
            ax[0].set_ylim(0, self.warped_size[1])
            ax[0].set_title('Detected Marking Pixels')

            ax[1].plot(-left_coords_ego[1], left_coords_ego[0], '.')
            ax[1].plot(-right_coords_ego[1], right_coords_ego[0], '.')
            ax[1].set_aspect('equal')
            ax[1].set_xlim(-self.warped_size[0]/2 / self.px_per_meters_y, self.warped_size[0]/2 / self.px_per_meters_y)
            ax[1].set_ylim(0, self.warped_size[1] / self.px_per_meters_x)
            ax[1].set_title('Detected Marking Points in Ego Frame')
            plt.show(block=False)

        return left_coords_ego, right_coords_ego

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
        is noticable, rotate the image around the center of image's  bottom to make it more vertical.

        Input:
            edge_image: Edge image in the bird's eye view.

        Output:
            rot_image: Image after rotation. It is the same as input edge image if no rotation is needed.
            rot_angle: Rotation angle in rad. It is None if no rotation is needed.
        """

        lines = cv2.HoughLines(
            edge_image[edge_image.shape[0]//2:, :], 2, np.pi/180, self.hough_thres)

        # When hough transform can't find lines in the lower half of image,
        # it's an indication that there is no good lines to detect
        if lines is not None:
            # Convert the angle range from (0, pi) to (-pi/2, pi/2)
            # Those with angle larger than pi/2 are also those with negative rhos.
            lines[lines[:, :, 0] < 0, 1] -= np.pi
            rot_angle = np.median(lines[:, :, 1])   # (rad)
        else:
            rot_angle = None

        # Rotate image when angle larger than 5 degrees
        if rot_angle and abs(rot_angle) > 5 * np.pi / 180:
            rot_center = (self.warped_size[0]//2, self.warped_size[1])
            M_rot = cv2.getRotationMatrix2D(
                rot_center, rot_angle * 180 / np.pi, scale=1)
            rot_image = cv2.warpAffine(edge_image, M_rot, self.warped_size)
        else:
            rot_image = edge_image
            rot_angle = None

        return rot_image, rot_angle

    def _get_histo(self, edge_image):
        """ 
        Get histogram of edge image.

        The peaks in histogram is then used as starting points for sliding window search. 
        """
        # Only the lower third image is used since we focus on the starting points
        histogram, _ = np.histogram(edge_image[int(edge_image.shape[0]*self.histo_region):, :].nonzero()[
                                    1], bins=self.n_bins, range=(0, self.warped_size[0]))
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
        # Return None if no peaks found
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
        window_height = np.int(edge_image.shape[0]/self.n_windows*self.search_region)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = edge_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Create empty lists to store left and right lane pixel indices
        left_idc = []
        right_idc = []

        shift = 0

        if left_base:
            # Initial windows' positions
            leftx_curr = int(left_base)
            lefty_curr = edge_image.shape[0]
            search_left = True
        else:
            search_left = False

        if right_base:
            rightx_curr = int(right_base)
            righty_curr = edge_image.shape[0]
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

        if __debug__:
            plt.figure()
            plt.plot(nonzerox[left_idc], nonzeroy[left_idc], '.')
            plt.plot(nonzerox[right_idc], nonzeroy[right_idc], '.')
            plt.imshow(debug_img)
            plt.show(block=False)

        return left_idc, right_idc, nonzerox, nonzeroy


def main():
    argparser = argparse.ArgumentParser(
        description='Lane Detection using Semantic Images')
    argparser.add_argument('vision_config', type=argparse.FileType(
        'r'), help='configuration yaml file for vision algorithms')
    args = argparser.parse_args()

    # Read configurations from yaml file to config_args
    with args.vision_config as vision_config_file:
        vision_config_args = yaml.safe_load(vision_config_file)

    # Load parameters for inverse projection
    perspective_tform_data = np.load(
        'vision/perspective_tform_vanish_pt_ideal.npz')
    M = perspective_tform_data['M']
    warped_size = tuple(perspective_tform_data['bev_size'])
    valid_mask = perspective_tform_data['valid_mask']
    px_per_meter_x = float(perspective_tform_data['px_per_meter_x'])
    px_per_meter_y = float(perspective_tform_data['px_per_meter_y'])

    # Load data
    folder_name = 'small_roundabout'
    mydir = os.path.join('recordings', folder_name)
    with open(os.path.join(mydir, 'lane_images'), 'rb') as image_file:
        images = pickle.load(image_file)
    with open(os.path.join(mydir, 'yaw_rate'), 'rb') as yaw_rate_file:
        yaw_rates = pickle.load(yaw_rate_file)
    with open(os.path.join(mydir, 'in_junction'), 'rb') as in_junction_file:
        in_junction = pickle.load(in_junction_file)

    lane_detector = LaneMarkingDetector(
        M, px_per_meter_x, px_per_meter_y, warped_size, valid_mask, vision_config_args['lane'])
    left_coords, right_coords = lane_detector.find_marking_points(images[250].astype(np.uint8))
    if __debug__:
        plt.show()

if __name__ == "__main__":
    main()
