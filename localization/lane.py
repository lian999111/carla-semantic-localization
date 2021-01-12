""" Implementation of lane-related factors """

import numpy as np
from scipy.stats import chi2
from minisam import Factor, DiagonalLoss

from carlasim.carla_tform import Transform
from carlasim.utils import get_fbumper_location

from .utils import multivariate_normal_pdf


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
    a_l = -np.sin(alpha)
    b_l = np.cos(alpha)
    c_l = a_l*px + b_l*expected_c0

    return a_l, b_l, c_l, alpha


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


class LaneBoundaryFactor(Factor):
    """ Lane boundary factor. """
    geo_gate = chi2.ppf(0.99999999, df=2)
    sem_gate = 0.9
    expected_lane_extractor = None

    def __init__(self, key, detected_marking, z, pose_uncert, px, lane_factor_config):
        """Constructor.

        Args:
            key: Key to the pose node.
            detected_marking: Detected rs stop wrt front bumper.
            z: z coordinate for extracting ground truth lane boundaries at the correct height.
            pose_uncert: Covariance matrix of pose.
            px: Distance from rear axle to front bumper.
            lane_factor_config: Configuraiont for lane boundary factor.
        """
        if self.expected_lane_extractor is None:
            raise RuntimeError(
                'Extractor for expected lane should be initialized first.')

        self.detected_marking = detected_marking
        self.z = z
        self.pose_uncert = pose_uncert
        self.px = px
        self.config = lane_factor_config
        self.noise_cov = np.diag([lane_factor_config['stddev_c0']**2,
                                  lane_factor_config['stddev_c1']**2])
        self.prob_null = lane_factor_config['prob_null']

        # bool: True to turn on semantic association
        self.semantic = self.config['semantic']
        # bool: True to activate static mode
        self.static = self.config['static']
        # bool: True to ignore lane boundary detection in junction areas
        self.ignore_junction = self.config['ignore_junction']
        # float: Scale for noise cov for null hypothesis
        # It is just used to make very large covariance matrix when computing
        # the weight of null hypothesis.
        self.null_std_scale = self.config['null_std_scale']

        # bool: True if current pose is in junction area.
        self.in_junction = False
        # bool: True if current pose is driving into junction area.
        self.into_junction = False

        # List of MELaneDetection: Describing expected markings in mobileye-like formats
        self.me_format_expected_markings = None

        # Transform: Transform of initially guessed pose
        self._init_tform = None
        # ndarray: RPY of initially guessed pose
        self._init_orientation = None

        # Attributes for static expected lane boundary extraction
        # bool: True if error is computed the first time
        self._first_time = True
        # tuple: a, b, c, and alpha describing the lines extracted using initially guessed pose
        self._init_normal_forms = None
        self._init_types = None

        # list: Stores chosen c0 and c1 of chosen expected lane boundary
        self.expected_coeffs = None
        # float: Scale for the chosen Gaussian mode based on its association weight
        self._scale = 1.0
        # bool: True if null hypothesis is chosen
        self._null_hypo = False

        loss = DiagonalLoss.Sigmas(np.array(
            [self.config['stddev_c0'],
             self.config['stddev_c1']]))

        Factor.__init__(self, 1, [key], loss)

    def copy(self):
        return LaneBoundaryFactor(self.keys()[0],
                                  self.detected_marking,
                                  self.z,
                                  self.pose_uncert,
                                  self.px,
                                  self.config)

    def error(self, variables):
        ########## Expectation ##########
        pose = variables.at(self.keys()[0])
        location = np.append(pose.translation(), self.z)  # append z
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
                expected_type_list = [expected.type
                                      for expected in self.me_format_expected_markings]

                # The snapshot is stored in their normal forms; i.e. a, b, c, and alpha describing the lines
                self._init_normal_forms = [compute_normal_form_line_coeffs(self.px, c[0], c[1])
                                           for c in expected_coeffs_list]
                # Snapshot of lane boundary types
                self._init_types = expected_type_list

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
                # Retrieve lane boundary types from snapshot
                expected_type_list = self._init_types
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

            # List of expected markings' type
            expected_type_list = [expected.type
                                  for expected in self.me_format_expected_markings]

        ########## Measurement ##########
        measured_coeffs = np.asarray(
            self.detected_marking.get_c0c1_list()).reshape(2, -1)
        measured_type = self.detected_marking.type

        # Null hypothesis
        # Use the measurement itself at every optimization iteration as the null hypothesis.
        # This is, of course, just a trick.
        # This means the error for null hypothesis is always zeros.
        null_expected_c0c1 = measured_coeffs.squeeze().tolist()
        null_error = np.zeros((2, 1))

        # Compute innovation matrix for the null hypo
        null_noise_cov = self.noise_cov * self.null_std_scale**2

        # Compute measurement likelihood weighted by null probability
        null_weighted_meas_likelihood = self.prob_null * \
            multivariate_normal_pdf(null_error.squeeze(), cov=null_noise_cov)

        # In this implementation, scaling down error and jacobian is done to achieve
        # the same effect of tuning the information matrix online.
        # Here, however, scale down error for null hypo; i.e.
        # null_error /= self.null_std_scale
        # is not necessary, since its always zero.
        # Zero error and jacobian effectively result in zero information matrix as well.

        if self.ignore_junction and (self.in_junction or self.into_junction):
            self._null_hypo = True
        elif not expected_coeffs_list:
            self._null_hypo = True
        else:
            # Data association
            errors = [null_error]
            gated_coeffs_list = [null_expected_c0c1]
            asso_probs = []
            meas_likelihoods = []
            for exp_coeffs, exp_type in zip(expected_coeffs_list, expected_type_list):
                # Compute innovation matrix
                expected_c0, expected_c1 = exp_coeffs
                H = compute_H(self.px, expected_c0, expected_c1)
                innov = H @ self.pose_uncert @ H.T + self.noise_cov

                # Compute squared mahalanobis distance
                error = np.asarray(exp_coeffs).reshape(
                    2, -1) - measured_coeffs
                squared_mahala_dist = error.T @ np.linalg.inv(innov) @ error

                # Semantic likelihood
                if self.semantic:
                    # Conditional probability on type
                    sem_likelihood = self._conditional_prob_type(
                        exp_type, measured_type)
                else:
                    # Truning off semantic association is equivalent to always
                    # set semantic likelihood to 1.0
                    sem_likelihood = 1.0

                # Gating (geometric and semantic)
                # Reject both geometrically and semantically unlikely associations
                # Note:
                # Since location distribution across lanes is essentially multimodal,
                # geometric gating often used when assuming location is unimodal is not
                # very reasonable and will inevitablly reject possible associations.
                # Here the geometric gate is set very large so we can preserve associations that
                # are a bit far from the current mode (the only mode that exists in the optimization)
                # but still possible when the multimodal nature is concerned.
                # Or, we can simply give up geometric gating and use semantic gating only.
                # The large geometric gate is an inelegant remedy after all.
                # if squared_mahala_dist <= self.geo_gate and sem_likelihood > self.sem_gate:
                if sem_likelihood > self.sem_gate:
                    gated_coeffs_list.append(exp_coeffs)
                    errors.append(error)

                    # Measurement likelihood (based on noise cov)
                    meas_likelihood = multivariate_normal_pdf(
                        error.reshape(-1), cov=self.noise_cov)
                    meas_likelihoods.append(meas_likelihood)

                    # Geometric likelihood (based on innov)
                    geo_likelihood = multivariate_normal_pdf(
                        error.reshape(-1), cov=innov)
                    asso_prob = geo_likelihood * sem_likelihood
                    asso_probs.append(asso_prob)

            # Check if any possible association exists after gating
            if asso_probs:
                asso_probs = np.asarray(asso_probs)
                meas_likelihoods = np.asarray(meas_likelihoods)

                # Compute weights based on total probability theorem
                weights = (1-self.prob_null) * \
                    (asso_probs/np.sum(asso_probs))
                # Weight measurement likelihoods
                weighted_meas_likelihood = weights*meas_likelihoods

                # Add weight and weighted likelihood of null hypothesis
                weights = np.insert(weights, 0, self.prob_null)
                weighted_meas_likelihood = np.insert(
                    weighted_meas_likelihood, 0, null_weighted_meas_likelihood)

                asso_idx = np.argmax(weighted_meas_likelihood)

                if asso_idx == 0:
                    self._null_hypo = True
                else:
                    self._null_hypo = False
                    self.expected_coeffs = gated_coeffs_list[asso_idx]
                    # Scale down the hypothesis to account for target uncertainty
                    # This form is empirically chosen
                    self._scale = weights[asso_idx]**1
                    # Scale down the error based on weight
                    # This is to achieve the same effect of scaling infomation matrix during optimzation
                    chosen_error = errors[asso_idx] * self._scale
            else:
                self._null_hypo = True

        if self._null_hypo:
            # Null hypothesis
            self.expected_coeffs = null_expected_c0c1
            chosen_error = null_error

        return chosen_error

    def jacobians(self, variables):
        if self._null_hypo:
            # In this implementation, scaling down error and jacobian is done to achieve
            # the same effect of tuning the information matrix online.
            # Here, however, computing jacobian of null hypothesis is not necessary.
            # Zero error and jacobian together effectively result in zero information matrix as well.
            jacob = np.zeros((2, 3))
        else:
            expected_c0, expected_c1 = self.expected_coeffs
            jacob = compute_H(self.px, expected_c0, expected_c1)
            # Scale down jacobian matrix based on weight
            # This is to achieve the same effect of scaling infomation matrix during optimzation
            jacob *= self._scale

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

    @staticmethod
    def _conditional_prob_type(expected_type, measured_type):
        if expected_type == measured_type:
            return 0.95
        else:
            return 0.0045

    @classmethod
    def set_expected_lane_extractor(cls, expected_lane_extractor):
        """Set class attribute expected lane extractor.

        This must be called before instantiating any of this class."""
        cls.expected_lane_extractor = expected_lane_extractor
