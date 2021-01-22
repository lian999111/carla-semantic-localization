"""Implementation of max-mixture prior factor."""

import math

import numpy as np
from minisam import Factor, GaussianLoss

from .utils import multivariate_normal_pdf


class MMPriorFactor(Factor):
    """Max-mixture prior factor."""

    def __init__(self, key, prior_pose, prior_config, prior_cov=None):
        """Constructor.

        This factor uses a narrow (normal) mode and a wide mode internally.
        When the error is large, the wide mode will be selected by the max-mixture
        scheme and loose the constraint, so wrong prior belief does not geopardize
        the later estimations.

        Args:
            key: Key to the pose node.
            prior_pose (sophus.SE2): Pose object of prior.
            prior_config (dict): Configurations.
            prior_cov (np.ndarray): Covariance matrix of prior.
        """
        self.prior_pose = prior_pose
        self.config = prior_config
        self.wide_std_scale = self.config['wide_std_scale']
        self._wide_mode = False

        # Homogeneous tform matrix for performing optimization in the local frame
        loc_x = self.prior_pose.translation()[0]
        loc_y = self.prior_pose.translation()[1]
        theta = prior_pose.so2().theta()
        self._tform_w2e = np.zeros((3, 3))
        rotm = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
        self._tform_w2e[0:2, 0:2] = rotm.T
        self._tform_w2e[0:2, 2] = -(rotm.T @ np.array([loc_x, loc_y]))
        self._tform_w2e[2, 2] = 1

        if prior_cov is None:
            # Use default prior noise
            self.prior_cov = np.diag([self.config['stddev_x']**2,
                                      self.config['stddev_y']**2,
                                      self.config['stddev_theta']**2])
            self.prior_noise_model = GaussianLoss.Covariance(self.prior_cov)
        else:
            self.prior_cov = prior_cov
            self.prior_noise_model = GaussianLoss.Covariance(self.prior_cov)

        Factor.__init__(self, 1, [key], self.prior_noise_model)

    def copy(self):
        return MMPriorFactor(self.keys()[0],
                             self.prior_pose,
                             self.config,
                             self.prior_cov)

    def error(self, variables):
        prior_theta = self.prior_pose.so2().theta()

        curr_pose = variables.at(self.keys()[0])
        curr_loc_x = curr_pose.translation()[0]
        curr_loc_y = curr_pose.translation()[1]
        curr_theta = curr_pose.so2().theta()

        trans_error = self._tform_w2e @ np.array([curr_loc_x, curr_loc_y, 1])
        rot_error = curr_theta - prior_theta
        # Wrt the local frame
        error = np.array([[trans_error[0]],
                          [trans_error[1]],
                          [rot_error]])

        # Narrow mode
        prob_narrow = multivariate_normal_pdf(error.squeeze(), self.prior_cov)

        # Wide mode
        prob_wide = multivariate_normal_pdf(
            error.squeeze(), self.prior_cov*self.wide_std_scale**2)

        # If choose wide mode, scale down error; i.e. scale up info matrix
        if prob_wide > prob_narrow and prob_narrow > 0.0:
            error /= self.wide_std_scale
            self._wide_mode = True
        else:
            self._wide_mode = False

        return error

    def jacobians(self, variables):
        jacob = np.diag([1, 1, 1])
        if self._wide_mode:
            # In this implementation, scaling down error and jacobian is done to achieve
            # the same effect of tuning the information matrix online.
            # Here, however, computing jacobian of null hypothesis is not necessary.
            # Zero error and jacobian together effectively result in zero information matrix as well.
            jacob = jacob / self.wide_std_scale

        return [jacob]
