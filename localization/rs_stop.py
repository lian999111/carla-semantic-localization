"""Implementation of road surface stop sign factor."""

import math

import numpy as np
from scipy.stats import chi2
from minisam import Factor, DiagonalLoss

from carlasim.utils import get_fbumper_location
from .utils import univariate_normal_pdf


class RSStopFactor(Factor):
    """Road surface stop sign factor."""
    # Attributes that needs to be initialized.
    # ExpectedRSStopExtractor: Extractor for expected road surface stop signs.
    expected_rs_stop_extractor = None
    # float: Longitudinal distance from rear axle to front bumper.
    px = None

    def __init__(self, key, detected_rs_stop_dist, z, pose_uncert, rs_stop_factor_config):
        """Constructor.

        Args:
            key: Key to the pose node.
            detected_rs_stop_dist: Longitudinal distance to detected rs stop wrt front bumper.
            z: z coordinate for extracting ground truth at the correct height.
            pose_uncert: Covariance matrix of pose.
            px: Distance from rear axle to front bumper.
            rs_stop_factor_config: Configuraiont for rs stop factor.
        """
        if self.expected_rs_stop_extractor is None:
            raise RuntimeError(
                'Extractor for expected road surface stop sign should be initialized first.')

        self.detected_rs_stop_dist = detected_rs_stop_dist
        self.z = z
        self.pose_uncert = pose_uncert
        self.config = rs_stop_factor_config

        self.noise_var = rs_stop_factor_config['stddev_dist']**2

        # list: Contains all extracted expected rs stop sign distances
        self.expected_rs_stop_dists = None

        # float: Null hypothesis probability
        self.prob_null = self.config['prob_null']
        # float: Scale for noise cov for null hypothesis
        # It is just used to make very large covariance matrix when computing
        # the weight of null hypothesis.
        self.null_std_scale = self.config['null_std_scale']

        self.chosen_expected_dist = None
        self._null_hypo = False
        self._scale = 1.0

        loss = DiagonalLoss.Sigmas(np.array(
            [self.config['stddev_dist']]))

        Factor.__init__(self, 1, [key], loss)

    def copy(self):
        return RSStopFactor(self.keys()[0],
                            self.detected_rs_stop_dist,
                            self.z,
                            self.pose_uncert,
                            self.config)

    def error(self, variables):
        ########## Expectation ##########
        pose = variables.at(self.keys()[0])

        # Append 0 as z since this factor is in 2D space
        location = np.append(pose.translation(), 0)
        orientation = np.array([0, 0, pose.so2().theta()])

        fbumper_location = get_fbumper_location(location,
                                                orientation,
                                                self.px)

        self.expected_rs_stop_dists = self.expected_rs_stop_extractor.extract(fbumper_location,
                                                                              orientation)

        ########## Measurement ##########
        # No need for parsing measurement since it is so simple

        # Null hypothesis
        # Use the measurement itself at every optimization iteration as the null hypothesis.
        # This is, of course, just a trick.
        # This means the error for null hypothesis is always zeros.
        null_expected_dist = self.detected_rs_stop_dist
        null_error = np.zeros((1, 1))

        # Compute innovation matrix for the null hypo
        null_noise_var = self.noise_var * self.null_std_scale**2

        # Compute measurement likelihood weighted by null probability
        null_weighted_meas_likelihood = self.prob_null \
            * univariate_normal_pdf(null_error.squeeze(), var=null_noise_var)

        ########## Data Association ##########
        if not self.expected_rs_stop_dists:
            self._null_hypo = True
        else:
            # Note: No gating is performed
            errors = [null_error]
            exp_dist_list = [null_expected_dist]
            meas_likelihoods = []
            asso_probs = []
            for exp_dist in self.expected_rs_stop_dists:
                H = np.array([[-1.0, 0, 0]])
                innov = (H @ self.pose_uncert @ H.T + self.noise_var).squeeze()

                # Compute squared mahalanobis distance
                error = np.asarray(exp_dist).reshape(
                    1, -1) - self.detected_rs_stop_dist

                errors.append(error)
                exp_dist_list.append(exp_dist)

                # Measurement likelihood (based on noise cov)
                meas_likelihood = univariate_normal_pdf(error.squeeze(),
                                                        var=self.noise_var)
                meas_likelihoods.append(meas_likelihood)

                # Geometric likelihood (based on innov)
                geo_likelihood = univariate_normal_pdf(error.squeeze(),
                                                       var=innov)
                asso_probs.append(geo_likelihood)

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
                self.chosen_expected_dist = exp_dist_list[asso_idx]
                # Scale down the hypothesis to account for target uncertainty
                # This form is empirically chosen
                self._scale = weights[asso_idx]**1
                # Scale down the error based on weight
                # This is to achieve the same effect of scaling infomation matrix during optimzation
                chosen_error = errors[asso_idx] * self._scale

        if self._null_hypo:
            # Null hypothesis
            self.chosen_expected_dist = null_expected_dist
            chosen_error = null_error

        return chosen_error

    def jacobians(self, variables):
        if self._null_hypo:
            # In this implementation, scaling down error and jacobian is done to achieve
            # the same effect of tuning the information matrix online.
            # Here, however, computing jacobian of null hypothesis is not necessary.
            # Zero error and jacobian together effectively result in zero information matrix as well.
            jacob = np.zeros((1, 3))
        else:
            jacob = np.array([[-1.0, 0, 0]])
            # Scale down jacobian matrix based on weight
            # This is to achieve the same effect of scaling infomation matrix during optimzation
            jacob *= self._scale

        return [jacob]

    @classmethod
    def initialize(cls, expected_rs_stop_extractor, px):
        """initialize road surface stop sign factor.

        This must be called before instantiating any of this class.

        Args:
            expected_rs_stop_extractor: Expected road surface stop sign extractor.
            px: Logitudinal distance from the local frame to the front bumper.
        """
        cls.expected_rs_stop_extractor = expected_rs_stop_extractor
        cls.px = px
