# Implementation of CTRV model

from math import sin, cos
import numpy as np


def predict_motion_from_ego_frame(vx, yaw_rate, delta_t, Q=None):
    """ 
    Predict the expected CTRV motion with respect to current ego frame given the control and time difference.

    That is, get the expected CTRV motion while viewed from the current pose.    
    When yaw rate is very small, CV model is used instead to avoid division-by-zero.

    Note: The resultant covarinace matrix of delta_x, delta_y, delta_theta is obtained by transforming 
    the covariance matrix of vx and yaw rate based on the Jacobian of the CTRV model. This method results
    in a rank-deficient matrix and causes problems during optimization. Thus, a small diagonal matrix is
    added in order to make it full-rank and stabilize the optimization.

    Args:
        vx:         Velocity in x-axis of the ego frame. (m/s)
        yaw_rate:   Yaw rate. (rad/s)
        delta_t:    Time difference. (sec)
        Q:          2-by-2 noise covariance matrix of vx and yaw_rate. Defaluts to None.
                    When given, the noise covariance matrix wrt delta_x, delta_y, and delta_theta is computed and returned.
    Return:
        delta_x: Difference in x.
        delta_y: Difference in y.
        delta_theta: Difference in heading.
        cov: 3-by-3 noise covariance matrix.
    """
    # Initialize L, the 3-by-2 Jacobian of displacement wrt noise in vx and yaw_rate
    if Q is not None:
        L = np.zeros((3, 2))

    if yaw_rate > 1e-3:
        yaw_rate_T = yaw_rate * delta_t
        r = vx/yaw_rate
        delta_x = r * np.sin(yaw_rate_T)
        delta_y = r * (-np.cos(yaw_rate_T) + 1)
        delta_theta = yaw_rate_T

        if Q is not None:
            L[0, 0] = 1/yaw_rate * np.sin(yaw_rate_T)
            L[0, 1] = -r/yaw_rate * \
                np.sin(yaw_rate_T) + r*delta_t*np.cos(yaw_rate_T)
            L[1, 0] = 1/yaw_rate * (-np.cos(yaw_rate_T) + 1)
            L[1, 1] = -r/yaw_rate * \
                (-np.cos(yaw_rate_T) + 1) + r*delta_t*np.sin(yaw_rate_T)
            L[2, 1] = delta_t
    else:
        delta_x = vx*delta_t
        delta_y = 0.0

        if Q is not None:
            L[0, 0] = delta_t

        if vx > 0.1:
            delta_theta = yaw_rate * delta_t
            if Q is not None:
                L[2, 1] = delta_t
        else:
            # Do not update theta if vx is small
            delta_theta = 0.0

    if Q is not None:
        # LQL.T along gives a rank-deficient (only rank 2) covariance, which makes it impossible to use
        # since inverse must be taken during optimization.
        # As a compensation, add a small diagnol matrix to make it full-rank again.
        cov = L @ Q @ L.T + np.eye(3)*1e-4
    else:
        cov = None

    return delta_x, delta_y, delta_theta, cov


def compute_F(theta, vx, yaw_rate, delta_t):
    """Compute linearized F matrix for ctrv model.

    F is independent of x and y.
    
    Args:
        theta:      Current heading. (rad)
        vx:         Velocity in x-axis of the ego frame. (m/s)
        yaw_rate:   Yaw rate. (rad/s)
        delta_t:    Time difference. (sec)

    Returns:
        F: Linearized F matrix.
    """
    F = np.eye(3)

    if yaw_rate > 1e-3:
        r = vx/yaw_rate
        theta_plus_yaw_rate_dt = theta + yaw_rate * delta_t
        F[0, 2] = r * (cos(theta_plus_yaw_rate_dt) - cos(theta))
        F[1, 2] = r * (sin(theta_plus_yaw_rate_dt) - sin(theta))
    else:
        F[0, 2] = -vx * sin(theta) * delta_t
        F[1, 2] = vx * cos(theta) * delta_t

    return F