"""Utility functions for localization evaluation"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.linalg import sqrtm


def compute_errors(pose_estis, loc_gt_seq, ori_gt_seq):
    """Compute longitudinal, lateral, and yaw errors.

    Args:
        pose_estis (list of sophus.SE2): List of pose esitmations.
        loc_gt_seq (list): List of ground truth locations.
        ori_gt_seq (list): List of ground truth orientations.
    Returns:
        longitudinal_errors (list): List of longitudinal errors
        lateral_errors (list): List of lateral errors
        yaw_errors (list): List of yaw errors
    """
    longitudinal_errors = []
    lateral_errors = []
    yaw_errors = []
    for loc_gt, ori_gt, pose_esit in zip(loc_gt_seq, ori_gt_seq, pose_estis):
        x_gt = loc_gt[0]
        y_gt = loc_gt[1]
        yaw_gt = ori_gt[2]

        # Translational error
        # Matrix that transforms a point in ego frame to world frame
        tform_e2w = np.array([[math.cos(yaw_gt), -math.sin(yaw_gt), x_gt],
                              [math.sin(yaw_gt), math.cos(yaw_gt), y_gt],
                              [0, 0, 1]])
        tform_w2e = np.linalg.inv(tform_e2w)

        trvec_world = np.append(pose_esit.translation(), 1)
        trvec_ego = tform_w2e @ trvec_world

        longitudinal_errors.append(trvec_ego[0])
        lateral_errors.append(trvec_ego[1])

        # Rotational error
        yaw = pose_esit.so2().theta()
        yaw_error = yaw - yaw_gt
        # Since yaw angle is in a cyclic space, when the amount of error is larger than 180 degrees,
        # we need to correct it.
        if yaw_error > math.pi:
            yaw_error = 2*math.pi - yaw_error
        elif yaw_error < -math.pi:
            yaw_error = 2*math.pi + yaw_error
        yaw_errors.append(yaw_error)

    return longitudinal_errors, lateral_errors, yaw_errors


def world_to_pixel(location, map_info, offset=(0, 0)):
    """Convert the world coordinates to pixel coordinates"""
    x = map_info['scale'] * map_info['pixels_per_meter'] * \
        (location.x - map_info['world_offset_x'])
    y = map_info['scale'] * map_info['pixels_per_meter'] * \
        (location.y - map_info['world_offset_y'])
    return [int(x - offset[0]), int(y - offset[1])]


def plot_se2_with_cov(ax, pose, cov, vehicle_size=0.5, line_color='k', vehicle_color='r', confidence=0.99):
    # Plot a triangle representing the ego vehicle
    p1 = pose.translation() + pose.so2() * np.array([1, 0]) * vehicle_size
    p2 = pose.translation() + pose.so2() * \
        np.array([-0.5, -0.5]) * vehicle_size
    p3 = pose.translation() + pose.so2() * np.array([-0.5, 0.5]) * vehicle_size

    line = plt.Polygon([p1, p2, p3], closed=True, fill=True,
                       edgecolor=line_color, facecolor=vehicle_color, zorder=2)
    triangle = ax.add_line(line)

    # Plot covariance
    # Reference: https://gist.github.com/CarstenSchelp/b992645537660bda692f218b562d0712#gistcomment-3465086
    # This approach of ploting covariance has the benefit
    # that it allows to draw ellipse of different confidence
    # regions.
    cov_2d = cov[0:2, 0:2]
    circle_count = 50
    r = np.sqrt(chi2.ppf(confidence, df=2))
    t = np.linspace(0, 2*np.pi, num=circle_count)
    circle = r * np.vstack((np.cos(t), np.sin(t)))

    # Need to rotate the ellipse because the covariance is wrt the local frame
    yaw = pose.so2().theta()
    rotm = np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])
    pts = rotm @ sqrtm(cov_2d) @ circle + pose.translation().reshape(2, 1)

    line = plt.Polygon(pts.T, closed=True, fill=False,
                       edgecolor=line_color, zorder=2)

    ellipse = ax.add_line(line)
    return triangle, ellipse


def adjust_figure(fig, aspect, size=7):
    """Adjust figure size to fit.

    Args:
        fig: Figure object.
        ax: Axes of graph.
        aspect: Height-to-width ratio of graph. 
        size: Maximum size (inch).
    """
    if aspect < 1:
        height = size * aspect
        width = size
    else:
        height = size
        width = size / aspect

    l = fig.subplotpars.left
    r = fig.subplotpars.right
    t = fig.subplotpars.top
    b = fig.subplotpars.bottom
    fig.set_size_inches(width/(r-l)+1, height/(t-b))
