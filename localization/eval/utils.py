"""Utility functions for localization evaluation"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.linalg import sqrtm


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
