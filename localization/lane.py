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

    def __init__(self, key, detected_marking, pose_uncert, dist_raxle_to_fbumper, geo_lane_factor_config):
        if self.expected_lane_extractor is None:
            raise RuntimeError(
                'Extractor for expected lane should be initialized first.')

        self.detected_marking = detected_marking
        self.pose_uncert = pose_uncert
        self.px = dist_raxle_to_fbumper
        self.config = geo_lane_factor_config
        self.noise_cov = np.diag([geo_lane_factor_config['stddev_c0'],
                                  geo_lane_factor_config['stddev_c1']])

        # bool: True to activate static mode
        self.static = self.config['static']

        self.in_junction = False
        self.into_junction = False
        # List of MELaneDetection: Describing expected markings in mobileye-like formats
        self.me_format_expected_markings = None

        # tuple: Normal form for null hypothesis
        # It is a vertical line passing through the origin of the local frame
        self._null_hypo_normal_form = (0, 1, 0, 0)

        # Transform: Transform of initially guessed pose
        self._init_tform = None
        # ndarray: RPY of initially guessed pose
        self._init_orientation = None

        # Attributes for static expected lane boundary extraction
        # bool: True if error is computed the first time
        self._first_time = True
        # tuple: a, b, c, and alpha describing the lines extracted using initially guessed pose
        self._init_normal_forms = None

        self.expected_coeffs = None
        self._null_hypo = False

        loss = DiagonalLoss.Sigmas(np.array(
            [self.config['stddev_c0'],
             self.config['stddev_c1']]))

        Factor.__init__(self, 1, [key], loss)

    def copy(self):
        return GeoLaneBoundaryFactor(self.keys()[0], self.detected_marking, self.pose_uncert, self.px, self.config)

    def error(self, variables):
        ########## Expectation ##########
        pose = variables.at(self.keys()[0])
        location = np.append(pose.translation(), 0)  # append z = 0
        orientation = np.array([0, 0, pose.so2().theta()])

        if self._first_time:
            # Store the initially guessed pose when computing error the first time
            self._init_tform = Transform.from_conventional(
                location, orientation)
            self._init_orientation = orientation

        if self.static:
            # Static mode
            if self._first_time:
                # First time extracting expected land boundaries
                fbumper_location = get_fbumper_location(
                    location, orientation, self.px)
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
                # Pose difference is wrt local frame
                pose_diff = self._get_pose_diff(location, orientation)

                # Compute expected lane boundary coefficients using the snapshot
                expected_coeffs_list = []
                for normal_form in self._init_normal_forms:
                    c0c1 = self._compute_expected_c0c1(normal_form, pose_diff)
                    expected_coeffs_list.append(c0c1)
        else:
            # Not static mode
            # Extract ground truth from the Carla server
            fbumper_location = get_fbumper_location(
                location, orientation, self.px)
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
        measured_coeffs = np.asarray(
            self.detected_marking.get_c0c1_list()).reshape(2, -1)

        if self.in_junction or self.into_junction:
            self._null_hypo = True
        elif not expected_coeffs_list:
            self._null_hypo = True
        else:
            # Nearest neighbor association
            squared_mahala_dists = []
            errors = []
            gated_coeffs_list = []
            for expected_coeffs, innov in zip(expected_coeffs_list, innov_matrices):
                e = np.asarray(expected_coeffs).reshape(
                    2, -1) - measured_coeffs
                squared_mahala_dist = e.T @ np.linalg.inv(innov) @ e
                # Gating
                if squared_mahala_dist <= self.gate:
                    squared_mahala_dists.append(e.T @ np.linalg.inv(innov) @ e)
                    errors.append(e)
                    gated_coeffs_list.append(expected_coeffs)

            # Check if any valid mahalanobis distance exists after gating
            if squared_mahala_dists:
                # Find smallest mahalanobis distance
                squared_mahala_dists = np.asarray(squared_mahala_dists)
                asso_idx = np.argmin(squared_mahala_dists)

                self.expected_coeffs = gated_coeffs_list[asso_idx]
                error = errors[asso_idx]
                self._null_hypo = False
            else:
                self._null_hypo = True

        if self._null_hypo:
            # Null hypothesis
            pose_diff = self._get_pose_diff(location, orientation)
            null_c0c1 = self._compute_expected_c0c1(
                self._null_hypo_normal_form, pose_diff)
            self.expected_coeffs = null_c0c1
            error = (np.asarray(null_c0c1).reshape(
                2, -1) - measured_coeffs) * 1e-10

        return error

    def jacobians(self, variables):
        # Left marking
        expected_c0, expected_c1 = self.expected_coeffs
        jacob = compute_H(self.px, expected_c0, expected_c1)

        if self._null_hypo:
            jacob *= 1e-10

        return [jacob]

    def _get_pose_diff(self, location, orientation):
        """Get pose difference from the initial guess."""
        if self._init_tform is None or self._init_orientation is None:
            raise RuntimeError('Initial pose not initialized yet.')

        delta = self._init_tform.tform_w2e_numpy_array(
            location).squeeze()
        dx, dy = delta[0], delta[1]
        dtheta = orientation[2] - self._init_orientation[2]
        return dx, dy, dtheta

    def _compute_expected_c0c1(self, normal_form, pose_diff):
        """Compute exptected c0 and c1 using normal form and pose difference."""
        a, b, c, alpha = normal_form
        dx, dy, dtheta = pose_diff
        c0 = (c - a*dx - a*self.px*np.cos(dtheta) - b*dy - b*self.px*np.sin(dtheta)) \
            / (-a*np.sin(dtheta) + b*np.cos(dtheta))
        c1 = np.tan(alpha - dtheta)
        return (c0, c1)

    @classmethod
    def set_expected_lane_extractor(cls, expected_lane_extractor):
        """Set class attribute expected lane extractor.

        This must be called before instantiating any of this class."""
        cls.expected_lane_extractor = expected_lane_extractor
