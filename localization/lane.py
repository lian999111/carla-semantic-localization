""" Implementation of lane-related factors """

import numpy as np
from scipy.stats.distributions import chi2
from minisam import Factor, key, DiagonalLoss, CauchyLoss

from carlasim.carla_tform import Transform
from carlasim.utils import get_fbumper_location
from .utils import ExpectedLaneExtractor


def compute_normal_form_line_coeffs(px, expected_c0, expected_c1):
    """Compute normal form of lane boundary given expected c0 and c1 coefficients.

    Normal form parameters of a line are a, b, c, and alpha, which describe the line
    with respect to the referecen frame in the form:
        ax + by = c
        alpha: Relative heading of the line.

    Args:
        px: Logitudinal distance from the local frame to the front bumper. 
        expected_c0: Expected c0 coefficient of lane boundary.
        expected_c1: Expected c1 coefficient of lane boundary.

    Returns:
        Normal form parameters a, b, c, and alpha.
    """
    alpha = np.arctan(expected_c1)
    a_L = -np.sin(alpha)
    b_L = np.cos(alpha)
    c_L = a_L*px + b_L*expected_c0

    return a_L, b_L, c_L, alpha


def compute_H(px, expected_c0, expected_c1):
    """Compute H matrix given expected c0 and c1.

    H matrix is the jacobian of the measurement model wrt the pose.

    Args:
        px: Logitudinal distance from the local frame to the front bumper. 
        expected_c0: Expected c0 coefficient of lane boundary.
        expected_c1: Expected c1 coefficient of lane boundary.

    Returns:
        H matrix as np.ndarray.
    """
    a, b, c, alpha = compute_normal_form_line_coeffs(px,
                                                     expected_c0,
                                                     expected_c1)

    h13 = -px + (-a*c + a*a*px)/b**2

    H = np.array([[expected_c1, -1, h13],
                  [0, 0, -1/np.cos(alpha)**2]])

    return H

# TODO: Add docstring


class GeoLaneBoundaryFactor(Factor):
    """ Basic lane boundary factor. """
    gate = chi2.ppf(0.99, df=2)
    expected_lane_extractor = None

    def __init__(self, key, lane_detection, pose_uncert, dist_raxle_to_fbumper, geo_lane_factor_config):
        if self.expected_lane_extractor is None:
            raise RuntimeError('Extractor for expected lane should be initialized first.')

        self.lane_detection = lane_detection
        self.pose_uncert = pose_uncert
        self.px = dist_raxle_to_fbumper
        self.config = geo_lane_factor_config
        self.noise_cov = np.diag([geo_lane_factor_config['stddev_c0'],
                                  geo_lane_factor_config['stddev_c1']])

        # For getting expected lane marking measurements
        self._expected_lane_extractor = expected_lane_extractor

        # bool: True to activate static mode
        self.static = self.config['static']

        self.in_junction = False
        self.into_junction = False
        # List of MELaneDetection: Describing expected markings in mobileye-like formats
        self.me_format_expected_markings = None

        # Attributes for static expected lane boundary extraction
        # bool: True if expected lane marking already extracted
        self._extracted = False
        # Transform: Transform of initially guessed pose
        self._init_tform = None
        # ndarray: RPY of initially guessed pose
        self._init_orientation = None

        # Attributes for static expected lane boundary extraction
        # bool: True if error is computed the first time
        self._first_time = True
        # tuple: a, b, c, and alpha describing the lines extracted using initially guessed pose
        self._init_normal_forms = None

        self.expected_left_coeffs = None
        self.expected_right_coeffs = None
        self._left_null_hypo = False
        self._right_null_hypo = False

        loss = DiagonalLoss.Sigmas(np.array(
            [self.config['stddev_c0'],
             self.config['stddev_c1'],
             self.config['stddev_c0'],
             self.config['stddev_c1']]))

        Factor.__init__(self, 1, [key], loss)

    def copy(self):
        return GeoLaneBoundaryFactor(self.keys()[0], self.lane_detection, self.pose_uncert, self.px, self.config)

    def error(self, variables):
        ########## Expectation ##########
        pose = variables.at(self.keys()[0])
        location = np.append(pose.translation(), 0)  # append z = 0
        orientation = np.array([0, 0, pose.so2().theta()])
        fbumper_location = get_fbumper_location(location, orientation, self.px)

        if self._first_time:
            # Store the initially guessed pose when computing error the first time
                self._init_tform = Transform.from_conventional(
                    location, orientation)
                self._init_orientation = orientation

        if self.static:
            # Static mode
            if self._first_time:
                # First time extracting expected land boundaries
                fbumper_location = get_fbumper_location(location, orientation, self.px)
                self.in_junction, self.into_junction, self.me_format_expected_markings = self.expected_lane_extractor.extract(
                    fbumper_location, orientation)

                expected_coeffs_list = [expected.get_c0c1_list()
                                        for expected in self.me_format_expected_markings]

                # The snapshot is stored in their normal forms; i.e. a, b, c, and alpha describing the lines
                self._init_normal_forms = [compute_normal_form_line_coeffs(self.px, c[0], c[1])
                                           for c in expected_coeffs_list]

                self._first_time = False
            else:
                # Not first time, use snapshot of lane boundaries extracted the first time to compute error
                # Pose difference must be wrt local frame
                delta = self._init_tform.tform_w2e_numpy_array(
                    location).squeeze()
                dx, dy = delta[0], delta[1]
                dtheta = orientation[2] - self._init_orientation[2]

                # Compute expected lane boundary coefficients using the snapshot
                expected_coeffs_list = []
                for normal_form in self._init_normal_forms:
                    a, b, c, alpha = normal_form
                    c0 = (c - a*dx - a*self.px*np.cos(dtheta) - b*dy - b*self.px*np.sin(dtheta)) \
                        / (-a*np.sin(dtheta) + b*np.cos(dtheta))
                    c1 = np.tan(alpha - dtheta)
                    expected_coeffs_list.append((c0, c1))
        else:
            # Not static mode
            # Extract ground truth from the Carla server
            fbumper_location = get_fbumper_location(location, orientation, self.px)
            self.in_junction, self.into_junction, self.me_format_expected_markings = self.expected_lane_extractor.extract(
                fbumper_location, orientation)

            # List of expected markings' coefficients
            expected_coeffs_list = [expected.get_c0c1_list()
                                    for expected in self.me_format_expected_markings]
        
        # List of each expected marking's innovation matrix
        innov_matrices = []
        for expected_coeffs in expected_coeffs_list:
            # Compute innovation matrix for current expected marking
            expected_c0, expected_c1 = expected_coeffs
            H = compute_H(self.px, expected_c0, expected_c1)
            innov = H @ self.pose_uncert @ H.T + self.noise_cov
            innov_matrices.append(innov)

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
            for expected_coeffs, innov in zip(expected_coeffs_list, innov_matrices):
                e = np.asarray(expected_coeffs).reshape(2, -1) - left_coeffs
                mahala_dists.append(e.T @ np.linalg.inv(innov) @ e)
                errors.append(e)

            if mahala_dists:
                mahala_dists = np.asarray(mahala_dists)
                asso_idx = np.argmin(mahala_dists)

                self.expected_left_coeffs = expected_coeffs_list[asso_idx]

                if mahala_dists[asso_idx] <= self.gate and not (self.in_junction or self.into_junction):
                    self._left_null_hypo = False

        # TODO: Better handling of null hypothesis using null normal form
        if self._left_null_hypo:
            e_left = np.zeros((2,))
        else:
            e_left = errors[asso_idx].squeeze()

        self._right_null_hypo = True
        # Right marking measurement
        if right_marking_detection is not None and abs(right_marking_detection.coeffs[-1]) < 3.5:
            right_coeffs = np.asarray(
                right_marking_detection.get_c0c1_list()).reshape(2, -1)

            # Nearest neighbor association
            mahala_dists = []
            errors = []
            for expected_coeffs, innov in zip(expected_coeffs_list, innov_matrices):
                e = np.asarray(expected_coeffs).reshape(2, -1) - right_coeffs
                mahala_dists.append(e.T @ np.linalg.inv(innov) @ e)
                errors.append(e)

            if mahala_dists:
                mahala_dists = np.asarray(mahala_dists)
                asso_idx = np.argmin(mahala_dists)

                self.expected_right_coeffs = expected_coeffs_list[asso_idx]

                if mahala_dists[asso_idx] <= self.gate and not (self.in_junction or self.into_junction):
                    self._right_null_hypo = False

        if self._right_null_hypo:
            e_right = np.zeros((2,))
        else:
            e_right = errors[asso_idx].squeeze()

        return np.concatenate((e_left, e_right), axis=None)

    def jacobians(self, variables):
        # Left marking
        if self.expected_left_coeffs is None:
            jacob_left = np.array([[1, 1, 1], [0, 0, 1]]) * 1e-1
        else:
            expected_c0, expected_c1 = self.expected_left_coeffs
            jacob_left = compute_H(self.px, expected_c0, expected_c1)

            if self._left_null_hypo:
                jacob_left *= 0.01

        # Right marking
        if self.expected_right_coeffs is None:
            jacob_right = np.array([[1, 1, 1], [0, 0, 1]]) * 1e-5
        else:
            expected_c0, expected_c1 = self.expected_right_coeffs
            jacob_right = compute_H(self.px, expected_c0, expected_c1)

            if self._right_null_hypo:
                jacob_right *= 0.01

        return [np.concatenate((jacob_left, jacob_right), axis=0)]

    @classmethod
    def set_expected_lane_extractor(cls, expected_lane_extractor):
        cls.expected_lane_extractor = expected_lane_extractor
