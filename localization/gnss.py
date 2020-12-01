# Implementation of GNSS minisam factor

import numpy as np
from math import sin, cos

from minisam import Factor, keyString, DiagonalLoss

class GNSSFactor(Factor):
    """
    GNSS factor.
    """

    def __init__(self, key, point, gnss_factor_config):
        """Constructor.

        Args:
            key:    Key to the connected node.
            point:  1D numpy array of GNSS x-y point measurement.
            gnss_factor_config (dict):   Configurations for GNSS factor.
        """
        self.p_ = point
        self.config = gnss_factor_config

        loss = DiagonalLoss.Sigmas(np.array([self.config['stddev_x'], self.config['stddev_y']]))
        Factor.__init__(self, 1, [key], loss)

    def copy(self):
        """
        Deep copy.
        """
        return GNSSFactor(self.keys()[0], self.p_, self.config)

    def error(self, variables):
        """
        Compute error.
        """
        pose = variables.at(self.keys()[0])
        return pose.translation() - self.p_

    def jacobians(self, variables):
        """
        Compute the jacobian at the current linearization point.
        """
        # GTSAM and thus minisam requires Jacobian to be defined wrt the body frame
        pose = variables.at(self.keys()[0])
        theta = pose.so2().theta()
        return [np.array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0]])]

    # optional print function
    def __repr__(self):
        return 'GPS Factor on SE(2):\nprior = ' + self.p_.__repr__() + ' on ' + keyString(self.keys()[0]) + '\n'