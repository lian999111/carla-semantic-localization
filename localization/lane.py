# Implementation of lane-related factors

import numpy as np
from scipy.stats.distributions import chi2
from minisam import Factor, key, DiagonalLoss, CauchyLoss

from carlasim.carla_tform import Transform

# TODO: Add docstring
class GeometricLaneBoundaryFactor(Factor):
    """ Basic lane boundary factor. """

    def __init__(self, key, coeffs, dist_raxle_to_fbumper, lane_gt_extractor, geometric_lane_factor_config):
        self.coeffs = coeffs    # [c0, c1]
        self.px = dist_raxle_to_fbumper
        self.config = geometric_lane_factor_config

        # For getting expected c0 and c1 measurements
        self._lane_gt_extractor = lane_gt_extractor
        self.expected_lane_detection = None
        self.ex_coeffs = None

        loss = DiagonalLoss.Sigmas(np.array(
            [geometric_lane_factor_config['stddev_c0'],
             geometric_lane_factor_config['stddev_c1']]))

        Factor.__init__(self, 1, [key], loss)

    def copy(self):
        return GeometricLaneBoundaryFactor(self.keys()[0], self.coeffs, self.px, self._lane_gt_extractor, self.config)

    def error(self, variables):
        pose = variables.at(self.keys()[0])
        location = np.append(pose.translation(), 0)  # append z = 0
        rotation = np.array([0, 0, pose.so2().theta()])
        tform = Transform.from_conventional(location, rotation)
        fbumper_location = tform.tform_e2w_numpy_array(
            np.array([self.px, 0, 0])).squeeze()
        self.expected_lane_detection = self._lane_gt_extractor.update(
            fbumper_location, rotation)

        ex_left_coeffs = self.expected_lane_detection['left_marking_coeffs']
        left_marking = self.expected_lane_detection['left_marking']

        ex_next_left_coeffs = self.expected_lane_detection['next_left_marking_coeffs']
        next_left_marking = self.expected_lane_detection['next_left_marking']

        ex_right_coeffs = self.expected_lane_detection['right_marking_coeffs']
        right_marking = self.expected_lane_detection['right_marking']

        ex_next_right_coeffs = self.expected_lane_detection['next_right_marking_coeffs']
        next_right_marking = self.expected_lane_detection['next_right_marking']

        markings = [next_left_marking, left_marking,
                    right_marking, next_right_marking]

        c0_vals = np.array([ex_next_left_coeffs[0],
                            ex_left_coeffs[0],
                            ex_right_coeffs[0],
                            ex_next_right_coeffs[0]])

        c1_vals = np.array([ex_next_left_coeffs[1],
                            ex_left_coeffs[1],
                            ex_right_coeffs[1],
                            ex_next_right_coeffs[1]])

        v = np.concatenate((c0_vals, c1_vals)).reshape(2, -1)
        d = np.asarray(self.coeffs).reshape(2, -1) - v
        mahala_dists = np.diag(
            d.T @ np.linalg.inv(np.diag((0.5**2, 0.5**2))) @ d)
        marking_idx = np.argmin(mahala_dists)
        # marking_idx = np.argmin(np.abs(c0_vals - self.coeffs[0]))

        self.ex_coeffs = [c0_vals[marking_idx], c1_vals[marking_idx]]

        if markings[marking_idx] is not None and abs(self.ex_coeffs[0] - self.coeffs[0]) < 2:
            e = np.asarray(self.ex_coeffs) - np.asarray(self.coeffs)
        else:
            e = np.array([0, 0])

        return e

    def jacobians(self, variables):
        expected_c0 = self.ex_coeffs[0]
        expected_c1 = self.ex_coeffs[1]

        a, b, c = self._compute_normal_form_line_coeffs(expected_c0, expected_c1)

        h13 = -self.px + (-a*c + a*a*self.px)/b**2

        return [np.array([[expected_c1, -1, h13], [0, 0, -1/np.cos(np.arctan(expected_c1))**2]])]

    def _compute_normal_form_line_coeffs(self, expected_c0, expected_c1):
        alpha = np.arctan(expected_c1)
        a_L = -np.sin(alpha)
        b_L = np.cos(alpha)
        c_L = a_L*self.px + b_L*expected_c0

        return a_L, b_L, c_L

    # def lossFunction(self):
    #     if self.keys()[0] > key('x', 220):
    #         return DiagonalLoss.Sigmas(np.array([10, 10]))
    #     else:
    #         return DiagonalLoss.Sigmas(np.array([0.2, 0.2]))
