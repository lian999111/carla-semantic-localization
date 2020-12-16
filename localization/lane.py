""" Implementation of lane-related factors """

import numpy as np
from scipy.stats.distributions import chi2
from minisam import Factor, key, DiagonalLoss, CauchyLoss

from carlasim.carla_tform import Transform
from carlasim.utils import get_fbumper_location
from .utils import ExpectedLaneExtractor

# TODO: Add docstring


class GeoLaneBoundaryFactor(Factor):
    """ Basic lane boundary factor. """
    gate = chi2.ppf(0.99, df=2)

    def __init__(self, key, lane_detection, pose_uncert, dist_raxle_to_fbumper, geo_lane_factor_config, expected_lane_extractor):
        self.lane_detection = lane_detection
        self.pose_uncert = pose_uncert
        self.px = dist_raxle_to_fbumper
        self.config = geo_lane_factor_config
        self.noise_cov = np.diag([geo_lane_factor_config['stddev_c0'],
                                  geo_lane_factor_config['stddev_c1']])

        # For getting expected lane marking measurements
        self._expected_lane_extractor = expected_lane_extractor

        self.expected_left_coeffs = None
        self.expected_right_coeffs = None
        self._left_null_hypo = False
        self._right_null_hypo = False

        loss = DiagonalLoss.Sigmas(np.array(
            [geo_lane_factor_config['stddev_c0'],
             geo_lane_factor_config['stddev_c1'],
             geo_lane_factor_config['stddev_c0'],
             geo_lane_factor_config['stddev_c1']]))

        Factor.__init__(self, 1, [key], loss)

    def copy(self):
        return GeoLaneBoundaryFactor(self.keys()[0], self.lane_detection, self.pose_uncert, self.px, self.config, self._expected_lane_extractor)

    def error(self, variables):
        ########## Expectation ##########
        pose = variables.at(self.keys()[0])
        location = np.append(pose.translation(), 0)  # append z = 0
        orientation = np.array([0, 0, pose.so2().theta()])
        fbumper_location = get_fbumper_location(location, orientation, self.px)

        in_junction, me_format_lane_markings = self._expected_lane_extractor.extract(
            fbumper_location, orientation)

        # List of expected markings' coefficients
        # Each element will be an np.ndarray of c0 and c1 values
        coeffs_list = []
        # List of each expected marking's innovation matrix
        innov_matrices = []
        for expected_marking in me_format_lane_markings:
            coeffs = expected_marking.get_c0c1_list()

            # Compute innovation matrix for current expected marking
            a, b, c, alpha = self._compute_normal_form_line_coeffs(coeffs[0],
                                                                   coeffs[1])
            h13 = -self.px + (-a*c + a*a*self.px)/b**2
            H = np.array(
                [[coeffs[1], -1, h13], [0, 0, -1/np.cos(alpha)**2]])
            innov = H @ self.pose_uncert @ H.T + self.noise_cov
            innov_matrices.append(innov)

            # Add coefficients to the list
            coeffs_list.append(coeffs)

        ########## Measurement ##########
        left_marking_detection = self.lane_detection.left_marking_detection
        right_marking_detection = self.lane_detection.right_marking_detection

        self._left_null_hypo = True
        # Left marking measurement
        if left_marking_detection is not None and abs(left_marking_detection.coeffs[-1]) < 3.5:
            left_coeffs = np.asarray(
                left_marking_detection.get_c0c1_list()).reshape(2, -1)

            # Nearest neighbor association
            mahala_dists = []
            errors = []
            for expected_coeffs, innov in zip(coeffs_list, innov_matrices):
                e = np.asarray(expected_coeffs).reshape(2, -1) - left_coeffs
                mahala_dists.append(e.T @ np.linalg.inv(innov) @ e)
                errors.append(e)

            if mahala_dists:
                mahala_dists = np.asarray(mahala_dists)
                marking_idx = np.argmin(mahala_dists)

                self.expected_left_coeffs = coeffs_list[marking_idx]

                if mahala_dists[marking_idx] <= self.gate and not in_junction:
                    self._left_null_hypo = False

        if self._left_null_hypo:
            e_left = np.zeros((2,))
        else:
            # e_left = np.zeros((2,))
            e_left = errors[marking_idx].squeeze()
            # e_left = (np.asarray(self.expected_left_coeffs).reshape(2, -1) - left_coeffs).squeeze()

        self._right_null_hypo = True
        # Right marking measurement
        if right_marking_detection is not None and abs(right_marking_detection.coeffs[-1]) < 3.5:
            right_coeffs = np.asarray(
                right_marking_detection.get_c0c1_list()).reshape(2, -1)

            # Nearest neighbor association
            mahala_dists = []
            errors = []
            for expected_coeffs, innov in zip(coeffs_list, innov_matrices):
                e = np.asarray(expected_coeffs).reshape(2, -1) - right_coeffs
                mahala_dists.append(e.T @ np.linalg.inv(innov) @ e)
                errors.append(e)

            if mahala_dists:
                mahala_dists = np.asarray(mahala_dists)
                marking_idx = np.argmin(mahala_dists)

                self.expected_right_coeffs = coeffs_list[marking_idx]

                if mahala_dists[marking_idx] <= self.gate and not in_junction:
                    self._right_null_hypo = False

        if self._right_null_hypo:
            e_right = np.zeros((2,))
        else:
            e_right = errors[marking_idx].squeeze()

        return np.concatenate((e_left, e_right), axis=None)
        # return e_left
        # return e_right

    def jacobians(self, variables):
        # Left marking
        if self.expected_left_coeffs is None:
            jacob_left = np.array([[1, 1, 1], [0, 0, 1]]) * 1e-1
        else:
            expected_c0 = self.expected_left_coeffs[0]
            expected_c1 = self.expected_left_coeffs[1]

            a, b, c, alpha = self._compute_normal_form_line_coeffs(
                expected_c0, expected_c1)

            h13 = -self.px + (-a*c + a*a*self.px)/b**2

            jacob_left = np.array([[expected_c1, -1, h13], [0, 0, -1/np.cos(alpha)**2]])

            if self._left_null_hypo:
                jacob_left *= 0.01

        # Right marking
        if self.expected_right_coeffs is None:
            jacob_right = np.array([[1, 1, 1], [0, 0, 1]]) * 1e-5
        else:
            expected_c0 = self.expected_right_coeffs[0]
            expected_c1 = self.expected_right_coeffs[1]

            a, b, c, alpha = self._compute_normal_form_line_coeffs(
                expected_c0, expected_c1)

            h13 = -self.px + (-a*c + a*a*self.px)/b**2

            jacob_right = np.array([[expected_c1, -1, h13], [0, 0, -1/np.cos(alpha)**2]])

            if self._right_null_hypo:
                jacob_right *= 0.01

        # return [np.concatenate((jacob_left, jacob_right), axis=0)]
        return [jacob_left]
        # return [jacob_right]

    def _compute_normal_form_line_coeffs(self, expected_c0, expected_c1):
        alpha = np.arctan(expected_c1)
        a_L = -np.sin(alpha)
        b_L = np.cos(alpha)
        c_L = a_L*self.px + b_L*expected_c0

        return a_L, b_L, c_L, alpha
