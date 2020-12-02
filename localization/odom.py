""" Implementation of CTRV odometry factor """

import numpy as np
from minisam import BetweenFactor, key, GaussianLoss, SE2, SO2

from model.ctrv import predict_motion_from_ego_frame


def create_ctrv_between_factor(key1, key2, vx, yaw_rate, delta_t, ctrv_config):
    """Create a BetweenFactor based on CTRV motion model.

    Args:
        key1:   Key to the first connected pose node.
        key2 :  Key to the second connected pose node.
            The keys should be generated using minisam.key() function.
        vx:     Velocity in x. (m/s)
        yaw_rate: Yaw rate. (rad/s)
        delta_t:  Time difference. (sec)
        ctrv_config (dict): Configurations for ctrv factor.
    """
    # Initizlize noise covariance matrix of vx and yaw_rate
    Q = np.zeros((2, 2))
    Q[0, 0] = ctrv_config['stddev_vx']**2
    Q[1, 1] = ctrv_config['stddev_yaw_rate']**2

    delta_x, delta_y, delta_theta, cov = predict_motion_from_ego_frame(
        vx, yaw_rate, delta_t, Q)
    ctrv_noise = GaussianLoss.Covariance(cov)

    between_factor = BetweenFactor(key1, key2, SE2(
        SO2(delta_theta), np.array([delta_x, delta_y])), ctrv_noise)

    return between_factor, (delta_x, delta_y, delta_theta, cov)
