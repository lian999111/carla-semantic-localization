# Implementation of CTRV model

from math import sin, cos
import numpy as np

def predict_motion_from_ego_frame(vx, yaw_rate, delta_t, Q=None):
    """ 
    Predict the expected CTRV motion with respect to current ego frame given the control and time difference.
    
    That is, get the expected CTRV motion while viewed from the current pose.    
    When yaw rate is very small, CV model is used instead to avoid division-by-zero.

    Input:
        vx:         velocity in x-axis of the ego frame. (m/s)
        yaw_rate:   yaw rate. (rad/s)
        delta_t:    time difference. (sec)
        Q:          2-by-2 noise covariance matrix of vx and yaw_rate. 
                    When given, the noise covariance matrix wrt delta_x, delta_y, and delta_theta is computed and returned.
    Output:
        delta_x, delta_y, delta_theta, cov
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
            L[0, 1] = -r/yaw_rate * np.sin(yaw_rate_T) + r*delta_t*np.cos(yaw_rate_T)
            L[1, 0] = 1/yaw_rate * (-np.cos(yaw_rate_T) + 1)
            L[1, 1] = -r/yaw_rate * (-np.cos(yaw_rate_T) + 1) + r*delta_t*np.sin(yaw_rate_T)
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
