# Implementation of CTRV model

from math import sin, cos

def predict_motion_from_ego_frame(velocity, yaw_rate, dt):
    """ 
    Predict the expected CTRV motion with respect to current ego frame given the control and time difference.
    That is, get the expected CTRV motion while viewed from the current pose.
    The control includes:
        velocity: m/s
        yaw rate: rad/s
    
    When yaw rate is very small, CV model is used instead to avoid division-by-zero.
    """
    if abs(yaw_rate) < 1e-3:
        # Use CV model
        delta_x = velocity * dt
        delta_y = 0
        delta_theta = 0
    else:
        # CTRV
        r = velocity / yaw_rate
        yaw_rate_dt = yaw_rate * dt     # to avoid recomputing it
        delta_x = r * sin(yaw_rate_dt)
        delta_y = r * (-cos(yaw_rate_dt) + 1)
        delta_theta = yaw_rate_dt

    return delta_x, delta_y, delta_theta
