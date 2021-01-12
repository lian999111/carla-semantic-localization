""" Implementation of pole factor. """

import math

import numpy as np
from scipy.stats import chi2
from minisam import Factor, DiagonalLoss

from .utils import multivariate_normal_pdf


def compute_H(prc, exp_x, exp_y):
    """Compute H matrix given expected c0 and c1.

    H matrix is the jacobian of the measurement model wrt the pose.

    Args:
        prc: Logitudinal distance from the local frame to the front bumper.
        exp_x: Expected x of pole wrt camera frame.
        exp_y: Expected y of pole wrt camera frame.

    Returns:
        H matrix as np.ndarray.
    """
    exp_r = math.sqrt(exp_x**2 + exp_y**2)

    H = np.zeros((2, 3))

    H[0, 0] = - exp_x / exp_r
    H[0, 1] = - exp_y / exp_r
    H[0, 2] = - exp_y*prc / exp_r

    H[1, 0] = exp_y / (exp_x**2 + exp_y**2)
    H[1, 1] = exp_x / (exp_x**2 + exp_y**2)
    H[1, 2] = -prc*exp_x / (exp_x**2 + exp_y**2) - 1

    return H


class PoleFactor(Factor):
    """Pole factor."""
    geo_gate = chi2.ppf(0.9, df=2)
    sem_gate = 0.9

    # Attributes that needs to be initialized.
    # float: Longitudinal distance from rear axle to front bumper.
    px = None
    # float: Longitudinal distance from camera to front bumper.
    pcf = None

    def __init__(self, key, detected_pole, neighbor_poles, pose_uncert, pole_factor_config):
        """Constructor.

        Args:
            key: Key to the pose node.
            detected_pole: Detected pole wrt front bumper.
            neighboring_poles: List of poles in the neighborhood wrt world frame.
            pose_uncert: Covariance matrix of pose.
            px: Distance from rear axle to front bumper.
            pcf: Distance from front camera to front bumper.
            pole_factor_config: Configuraiont for pole factor.
        """
        if self.px is None:
            raise RuntimeError(
                'PoleFactor should be initialized first.')

        self.detected_pole = detected_pole
        self.neighbor_poles = neighbor_poles
        self.pose_uncert = pose_uncert
        self.config = pole_factor_config

        self.noise_cov = np.diag([pole_factor_config['stddev_r']**2,
                                  pole_factor_config['stddev_phi']**2])

        # float: Null hypothesis probability
        self.prob_null = pole_factor_config['prob_null']
        # float: To scale up standard deviation for null hypothesis
        self.null_std_scale = pole_factor_config['null_std_scale']

        # bool: True to turn on semantic association
        self.semantic = self.config['semantic']

        self.expected_xy = None
        self._null_hypo = False
        self._std_scale = 1.0
        self._scale = 1.0

        loss = DiagonalLoss.Sigmas(np.array(
            [self.config['stddev_r'],
             self.config['stddev_phi']]))

        Factor.__init__(self, 1, [key], loss)

    def copy(self):
        return PoleFactor(self.keys()[0],
                          self.detected_pole,
                          self.neighbor_poles,
                          self.pose_uncert,
                          self.config)

    def error(self, variables):
        ########## Expectation ##########
        pose = variables.at(self.keys()[0])
        yaw = pose.so2().theta()

        # Need to transform map poles in the neighborhood into ego (rear axle) frame
        # Matrix to transform points in world frame to ego (rear axle) frame
        tform_w2e = np.zeros((3, 3))
        rotm = np.array([[math.cos(yaw), -math.sin(yaw)],
                         [math.sin(yaw), math.cos(yaw)]])
        tform_w2e[0:2, 0:2] = rotm.T
        tform_w2e[0:2, 2] = -(rotm.T @ pose.translation())
        tform_w2e[2, 2] = 1

        # Extract map poles into lists
        pole_x_world = [pole.x for pole in self.neighbor_poles]
        pole_y_world = [pole.y for pole in self.neighbor_poles]
        pole_types = [pole.type for pole in self.neighbor_poles]

        # Transform map pole coordinates in world frame to ego (rear axle) frame
        pole_homo_coords_world = np.asarray(
            [pole_x_world, pole_y_world, np.ones((len(pole_x_world),))])
        pole_homo_coords_ego = tform_w2e @ pole_homo_coords_world

        # Get coordinates in camera's frame by offsetting x values
        pole_homo_coords_cam = pole_homo_coords_ego
        pole_homo_coords_cam[0, :] -= (self.px - self.pcf)

        # Remove map poles behind the camera
        exp_pole_types = [t for t, coord_cam in zip(
            pole_types, pole_homo_coords_cam.T) if coord_cam[0] > 10]
        pole_homo_coords_cam = pole_homo_coords_cam[:,
                                                    pole_homo_coords_cam[0] > 10]

        ########## Measurements ##########
        # Range and azimuth wrt camera
        meas_xy_cam = [self.detected_pole.x+self.pcf, self.detected_pole.y]
        meas_r, meas_phi = self.detected_pole.get_r_phi(-self.pcf)
        meas_type = self.detected_pole.type

        # Null hypothesis
        # Use the measurement itself at every optimization iteration as the null hypothesis.
        # This is, of course, just a trick.
        # This means the error for null hypothesis is always zeros.
        null_expected_xy_cam = meas_xy_cam
        null_error = np.zeros((2, 1))

        # Compute innovation matrix for the null hypo
        null_noise_cov = self.noise_cov*self.null_std_scale**2

        # Compute measurement likelihood weighted by null probability
        null_weighted_meas_likelihood = self.prob_null \
            * multivariate_normal_pdf(null_error.squeeze(), cov=null_noise_cov)

        # In this implementation, scaling down error and jacobian is done to achieve
        # the same effect of having a very small information matrix during optimzation.
        # Here, however, scale down error for null hypo; i.e.
        # null_error /= self.null_std_scale
        # is not necessary, since its always zero.

        ########## Data Association ##########
        squared_mahala_dists = []
        errors = [null_error]
        gated_xy_list = [null_expected_xy_cam]
        std_scales = [1]
        # errors = []
        # gated_xy_list = []
        # std_scales = []
        asso_probs = []
        meas_likelihoods = []
        for (exp_x, exp_y), exp_type in zip(pole_homo_coords_cam[0:2, :].T, exp_pole_types):
            r = math.sqrt(exp_x**2 + exp_y**2)
            phi = math.atan2(exp_y, exp_x)

            # Scale noise standard deviation based on range
            std_scale = max(0.001*r**2, 1)
            # std_scale = 1

            H = compute_H(self.px-self.pcf, exp_x, exp_y)
            scaled_noise_cov = self.noise_cov * std_scale**2
            innov = H @ self.pose_uncert @ H.T + scaled_noise_cov

            error = (np.array([r, phi]) -
                     np.array([meas_r, meas_phi])).reshape(2, -1)
            squared_mahala_dist = error.T @ np.linalg.inv(innov) @ error

            # Semantic likelihood
            if self.semantic:
                # Conditional probability on type
                sem_likelihood = self._conditional_prob_type(
                    exp_type, meas_type)
            else:
                # Truning off semantic association is equivalent to always
                # set semantic likelihood to 1.0
                sem_likelihood = 1.0

            # if squared_mahala_dist < self.geo_gate and sem_likelihood > self.sem_gate:
            if sem_likelihood > self.sem_gate:
                squared_mahala_dists.append(squared_mahala_dist)
                errors.append(error)
                std_scales.append(std_scale)
                gated_xy_list.append([exp_x, exp_y])

                # Measurement likelihood
                meas_likelihood = multivariate_normal_pdf(error.squeeze(),
                                                          cov=scaled_noise_cov)
                meas_likelihoods.append(meas_likelihood)

                # Geometric likelihood
                geo_likelihood = multivariate_normal_pdf(error.squeeze(),
                                                         cov=innov)

                asso_prob = geo_likelihood * sem_likelihood
                asso_probs.append(asso_prob)

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
                self.expected_xy = gated_xy_list[asso_idx]
                self._std_scale = std_scales[asso_idx]
                # To scale down the hypothesis to account for target uncertainty
                # This form is empirically chosen
                self._scale = weights[asso_idx]**1
                chosen_error = errors[asso_idx]

                # Scale down the error based on range
                chosen_error /= self._std_scale

                # Scale down the error based on weight
                chosen_error *= self._scale
        else:
            self._null_hypo = True

        if self._null_hypo:
            # Null hypothesis
            self.expected_xy = null_expected_xy_cam
            chosen_error = null_error

        return chosen_error

    def jacobians(self, variables):

        if self._null_hypo:
            jacob = np.zeros((2, 3))
        else:
            exp_x, exp_y = self.expected_xy
            jacob = compute_H(self.px-self.pcf, exp_x, exp_y)
            # Scale down jacobian matrix based on range
            jacob *= 1/self._std_scale
            # Scale down jacobian matrix based on weight
            jacob *= self._scale
        return [jacob]

    @staticmethod
    def _conditional_prob_type(expected_type, measured_type):
        if expected_type == measured_type:
            return 0.95
        else:
            return 0.0125

    @classmethod
    def initialize(cls, px, pcf):
        """Initialize lane boundary factor.

        This must be called before instantiating any of this class.

        Args:
            px: Logitudinal distance from rear axle to front bumper.
            pcf: Logitudinal distance from camera to front bumper.
        """
        cls.px = px
        cls.pcf = pcf
