""" Implementation of lane-related factors """

import numpy as np
from scipy.stats import chi2, multivariate_normal
from scipy.optimize import linear_sum_assignment as lsa
from minisam import Factor, DiagonalLoss

from carlasim.carla_tform import Transform
from carlasim.utils import get_fbumper_location


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
    """ Max-mixture PDA Lane boundary factor. """
    # float: Geometric gate
    geo_gate = chi2.ppf(0.99999999, df=2)
    # float: Semantic gate
    sem_gate = 0.9

    # Attributes that needs to be initialized.
    # ExpectedLaneExtractor: Extractor for expected lane boundaries.
    expected_lane_extractor = None
    # float: Longitudinal distance from rear axle to front bumper.
    px = None

    def __init__(self, key, detected_marking, z, pose_uncert, lane_factor_config):
        """Constructor.

        Args:
            key: Key to the pose node.
            detected_marking (MELaneMarking): Detected lane marking.
            z: z coordinate for extracting ground truth lane boundaries at the correct height.
            pose_uncert: Covariance matrix of pose.
            lane_factor_config: Configuraiont for lane boundary factor.
        """
        if self.expected_lane_extractor is None:
            raise RuntimeError(
                'LaneBoundaryFactor should be initialized first.')

        self.detected_marking = detected_marking
        self.z = z
        self.pose_uncert = pose_uncert
        self.config = lane_factor_config
        self.noise_cov = np.diag([lane_factor_config['stddev_c0']**2,
                                  lane_factor_config['stddev_c1']**2])

        # bool: True to turn on semantic association
        self.semantic = self.config['semantic']
        # bool: True to activate static mode
        self.static = self.config['static']
        # bool: True to ignore lane boundary detection in junction areas
        self.ignore_junction = self.config['ignore_junction']

        # float: Null hypothesis probability
        self.prob_null = self.config['prob_null']
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
        self.chosen_expected_coeffs = None
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
            multivariate_normal.pdf(null_error.squeeze(), cov=null_noise_cov)

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
                    meas_likelihood = multivariate_normal.pdf(
                        error.reshape(-1), cov=self.noise_cov)

                    # Geometric likelihood (based on innov)
                    geo_likelihood = multivariate_normal.pdf(
                        error.reshape(-1), cov=innov)

                    # Due to numerical errors, likelihood can become exactly 0.0
                    # in some very rare cases.
                    # When it happens, simply ignore it.
                    if meas_likelihood > 0.0 and geo_likelihood > 0.0:
                        meas_likelihoods.append(meas_likelihood)
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
                    self.chosen_expected_coeffs = gated_coeffs_list[asso_idx]
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
            self.chosen_expected_coeffs = null_expected_c0c1
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
            expected_c0, expected_c1 = self.chosen_expected_coeffs
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
    def initialize(cls, expected_lane_extractor, px):
        """Initialize lane boundary factor.

        This must be called before instantiating any of this class.

        Args:
            expected_lane_extractor: Expected lane extractor.
            px: Logitudinal distance from the local frame to the front bumper.
        """
        cls.expected_lane_extractor = expected_lane_extractor
        cls.px = px


# TODO: This factor is not used in my thesis due to the time constraint.
#       Worth investigating in the future. Maybe consider making it a max-mixture JPDA factor.
class GNNLaneBoundaryFactor(Factor):
    """ GNN Lane boundary factor.

    This factor considers 2 lane boundary detections at the same time.
    GNN association is performed at every optimization iteration, so this factor
    is still a max-mixture factor in some sense. The max operation is just replaced
    by the linear sum assignment operation, which can be seen as an variant of max
    operation for multiple distributions.
    """
    # float: Geometric gate
    geo_gate = chi2.ppf(0.99999999, df=2)
    # float: Semantic gate
    sem_gate = 0.9

    # Attributes that needs to be initialized.
    # ExpectedLaneExtractor: Extractor for expected lane boundaries.
    expected_lane_extractor = None
    # float: Longitudinal distance from rear axle to front bumper.
    px = None

    def __init__(self, key, lane_marking_detection, z, pose_uncert, lane_factor_config):
        """Constructor.

        Args:
            key: Key to the pose node.
            lane_marking_detection: Lane marking detection.
            z: z coordinate for extracting ground truth lane boundaries at the correct height.
            pose_uncert: Covariance matrix of pose.
            lane_factor_config: Configuraiont for lane boundary factor.
        """
        if self.expected_lane_extractor is None:
            raise RuntimeError(
                'LaneBoundaryFactor should be initialized first.')

        self.lane_marking_detection = lane_marking_detection
        self.left_marking = lane_marking_detection.left_marking_detection
        self.right_marking = lane_marking_detection.right_marking_detection
        self.z = z
        self.pose_uncert = pose_uncert
        self.config = lane_factor_config
        self.noise_cov = np.diag([lane_factor_config['stddev_c0']**2,
                                  lane_factor_config['stddev_c1']**2])

        # bool: True to turn on semantic association
        self.semantic = self.config['semantic']
        # bool: True to activate static mode
        self.static = self.config['static']
        # bool: True to ignore lane boundary detection in junction areas
        self.ignore_junction = self.config['ignore_junction']

        # float: Null hypothesis probability
        self.prob_null = self.config['prob_null']
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
        self.chosen_expected_coeffs_left = None
        self.chosen_expected_coeffs_right = None
        # float: Scale for the chosen Gaussian mode based on its association weight
        self._scale_left = 1.0
        self._scale_right = 1.0
        # bool: True if null hypothesis is chosen
        self._null_hypo_left = False
        self._null_hypo_right = False

        loss = DiagonalLoss.Sigmas(np.array(
            [self.config['stddev_c0'],
                self.config['stddev_c1'],
                self.config['stddev_c0'],
                self.config['stddev_c1']]))

        Factor.__init__(self, 1, [key], loss)

    def copy(self):
        return GNNLaneBoundaryFactor(self.keys()[0],
                                     self.lane_marking_detection,
                                     self.z,
                                     self.pose_uncert,
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
        if self.left_marking:
            measured_coeffs_left = np.asarray(
                self.left_marking.get_c0c1_list()).reshape(2, -1)
            measured_type_left = self.left_marking.type

        if self.right_marking:
            measured_coeffs_right = np.asarray(
                self.right_marking.get_c0c1_list()).reshape(2, -1)
            measured_type_right = self.right_marking.type

        # Null hypothesis
        # Use the measurement itself at every optimization iteration as the null hypothesis.
        # This is, of course, just a trick.
        # This means the error for null hypothesis is always zeros.
        null_expected_c0c1_left = self.left_marking.get_c0c1_list()
        null_expected_c0c1_right = self.right_marking.get_c0c1_list()
        null_error = np.zeros((2, 1))   # same for both left and right

        # Compute innovation matrix for the null hypo
        null_noise_cov = self.noise_cov * self.null_std_scale**2

        # Compute measurement likelihood weighted by null probability
        null_weighted_meas_likelihood = self.prob_null * \
            multivariate_normal.pdf(null_error.squeeze(), cov=null_noise_cov)

        # In this implementation, scaling down error and jacobian is done to achieve
        # the same effect of tuning the information matrix online.
        # Here, however, scale down error for null hypo; i.e.
        # null_error /= self.null_std_scale
        # is not necessary, since its always zero.
        # Zero error and jacobian effectively result in zero information matrix as well.

        if self.ignore_junction and (self.in_junction or self.into_junction):
            self._null_hypo_left = True
            self._null_hypo_right = True
        elif not expected_coeffs_list:
            self._null_hypo_left = True
            self._null_hypo_right = True
        else:
            # Data association
            num_expected_markings = len(self.me_format_expected_markings)
            asso_table = np.zeros((2, num_expected_markings+2))

            # Left lane marking
            errors_left = []
            asso_probs = []
            meas_likelihoods = []
            for exp_coeffs, exp_type in zip(expected_coeffs_list, expected_type_list):
                # Compute innovation matrix
                expected_c0, expected_c1 = exp_coeffs
                H = compute_H(self.px, expected_c0, expected_c1)
                innov = H @ self.pose_uncert @ H.T + self.noise_cov

                # Compute squared mahalanobis distance
                error = np.asarray(exp_coeffs).reshape(
                    2, -1) - measured_coeffs_left
                squared_mahala_dist = error.T @ np.linalg.inv(
                    innov) @ error

                # Semantic likelihood
                if self.semantic:
                    # Conditional probability on type
                    sem_likelihood = self._conditional_prob_type(
                        exp_type, measured_type_left)
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
                    errors_left.append(error)

                    # Measurement likelihood (based on noise cov)
                    meas_likelihood = multivariate_normal.pdf(
                        error.reshape(-1), cov=self.noise_cov)

                    # Geometric likelihood (based on innov)
                    geo_likelihood = multivariate_normal.pdf(
                        error.reshape(-1), cov=innov)

                    meas_likelihoods.append(meas_likelihood)
                    asso_prob = geo_likelihood * sem_likelihood
                    asso_probs.append(asso_prob)
                else:
                    errors_left.append(None)
                    meas_likelihoods.append(0)
                    asso_probs.append(0)

            asso_probs = np.asarray(asso_probs)
            meas_likelihoods = np.asarray(meas_likelihoods)

            # Compute weights based on total probability theorem
            if asso_probs.sum():
                weights_left = (1-self.prob_null) * \
                    (asso_probs/np.sum(asso_probs))
            else:
                weights_left = np.zeros(asso_probs.shape)

            # Weight measurement likelihoods
            weighted_meas_likelihood = weights_left*meas_likelihoods

            # Add weight and weighted likelihood of null hypothesis
            weights_left = np.insert(weights_left, 0, [self.prob_null, 0])
            weighted_meas_likelihood = np.insert(
                weighted_meas_likelihood, 0, [null_weighted_meas_likelihood, 0])

            asso_table[0, :] = weighted_meas_likelihood

            # Right marking
            errors_right = []
            asso_probs = []
            meas_likelihoods = []
            for exp_coeffs, exp_type in zip(expected_coeffs_list, expected_type_list):
                # Compute innovation matrix
                expected_c0, expected_c1 = exp_coeffs
                H = compute_H(self.px, expected_c0, expected_c1)
                innov = H @ self.pose_uncert @ H.T + self.noise_cov

                # Compute squared mahalanobis distance
                error = np.asarray(exp_coeffs).reshape(
                    2, -1) - measured_coeffs_right
                squared_mahala_dist = error.T @ np.linalg.inv(
                    innov) @ error

                # Semantic likelihood
                if self.semantic:
                    # Conditional probability on type
                    sem_likelihood = self._conditional_prob_type(
                        exp_type, measured_type_right)
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
                    errors_right.append(error)

                    # Measurement likelihood (based on noise cov)
                    meas_likelihood = multivariate_normal.pdf(
                        error.reshape(-1), cov=self.noise_cov)

                    # Geometric likelihood (based on innov)
                    geo_likelihood = multivariate_normal.pdf(
                        error.reshape(-1), cov=innov)

                    meas_likelihoods.append(meas_likelihood)
                    asso_prob = geo_likelihood * sem_likelihood
                    asso_probs.append(asso_prob)
                else:
                    errors_right.append(None)
                    meas_likelihoods.append(0)
                    asso_probs.append(0)

            asso_probs = np.asarray(asso_probs)
            meas_likelihoods = np.asarray(meas_likelihoods)

            # Compute weights based on total probability theorem
            if asso_probs.sum():
                weights_right = (1-self.prob_null) * \
                    (asso_probs/np.sum(asso_probs))
            else:
                weights_right = np.zeros(asso_probs.shape)

            # Weight measurement likelihoods
            weighted_meas_likelihood = weights_right*meas_likelihoods

            # Add weight and weighted likelihood of null hypothesis
            weights_right = np.insert(weights_right, 0, [0., self.prob_null])
            weighted_meas_likelihood = np.insert(
                weighted_meas_likelihood, 0, [0., null_weighted_meas_likelihood, ])

            asso_table[1, :] = weighted_meas_likelihood

            # GNN association
            # This is performed at every optimization step, so this factor is essentially
            # a max-mixture factor. It's just now the max operation is replaced by the
            # linear sum assignment operation.

            # Take log so the association result maximizes the product of likelihoods
            # of the associations of the both sides
            log_asso_table = np.log(asso_table)
            _, col_idc = lsa(log_asso_table, maximize=True)

            asso_idx_left = col_idc[0]
            asso_idx_right = col_idc[1]

            # Left assocation
            if asso_idx_left == 0:
                # Null hypothesis
                self._null_hypo_left = True
            else:
                self._null_hypo_left = False
                chosen_error_left = errors_left[asso_idx_left-2]
                self.chosen_expected_coeffs_left = expected_coeffs_list[asso_idx_left-2]
                self._scale_left = weights_left[asso_idx_left]

            # Right assocation
            if asso_idx_right == 1:
                # Null hypothesis
                self._null_hypo_right = True
            else:
                self._null_hypo_right = False
                chosen_error_right = errors_right[asso_idx_right-2]
                self.chosen_expected_coeffs_right = expected_coeffs_list[asso_idx_right-2]
                self._scale_right = weights_right[asso_idx_right]

        if self._null_hypo_left:
            chosen_error_left = null_error
            self.chosen_expected_coeffs_left = null_expected_c0c1_left
        if self._null_hypo_right:
            chosen_error_right = null_error
            self.chosen_expected_coeffs_right = null_expected_c0c1_right

        chosen_error = np.concatenate((chosen_error_left, chosen_error_right))

        return chosen_error

    def jacobians(self, variables):
        if self._null_hypo_left:
            # In this implementation, scaling down error and jacobian is done to achieve
            # the same effect of tuning the information matrix online.
            # Here, however, computing jacobian of null hypothesis is not necessary.
            # Zero error and jacobian together effectively result in zero information matrix as well.
            jacob_left = np.zeros((2, 3))
        else:
            expected_c0, expected_c1 = self.chosen_expected_coeffs_left
            jacob_left = compute_H(self.px, expected_c0, expected_c1)
            # Scale down jacobian matrix based on weight
            # This is to achieve the same effect of scaling infomation matrix during optimzation
            jacob_left *= self._scale_left

        if self._null_hypo_right:
            # In this implementation, scaling down error and jacobian is done to achieve
            # the same effect of tuning the information matrix online.
            # Here, however, computing jacobian of null hypothesis is not necessary.
            # Zero error and jacobian together effectively result in zero information matrix as well.
            jacob_right = np.zeros((2, 3))
        else:
            expected_c0, expected_c1 = self.chosen_expected_coeffs_right
            jacob_right = compute_H(self.px, expected_c0, expected_c1)
            # Scale down jacobian matrix based on weight
            # This is to achieve the same effect of scaling infomation matrix during optimzation
            jacob_right *= self._scale_right

        jacob = np.concatenate((jacob_left, jacob_right))

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
    def initialize(cls, expected_lane_extractor, px):
        """Initialize lane boundary factor.

        This must be called before instantiating any of this class.

        Args:
            expected_lane_extractor: Expected lane extractor.
            px: Logitudinal distance from the local frame to the front bumper.
        """
        cls.expected_lane_extractor = expected_lane_extractor
        cls.px = px
