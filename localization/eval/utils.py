"""Utility functions for localization evaluation"""
import numpy as np
import math
import matplotlib.pyplot as plt


def world_to_pixel(location, map_info, offset=(0, 0)):
    """Converts the world coordinates to pixel coordinates"""
    x = map_info['scale'] * map_info['pixels_per_meter'] * \
        (location.x - map_info['world_offset_x'])
    y = map_info['scale'] * map_info['pixels_per_meter'] * \
        (location.y - map_info['world_offset_y'])
    return [int(x - offset[0]), int(y - offset[1])]


def plot_se2_with_cov(ax, pose, cov, vehicle_size=0.5, line_color='k', vehicle_color='r'):
    # plot vehicle
    p1 = pose.translation() + pose.so2() * np.array([1, 0]) * vehicle_size
    p2 = pose.translation() + pose.so2() * \
        np.array([-0.5, -0.5]) * vehicle_size
    p3 = pose.translation() + pose.so2() * np.array([-0.5, 0.5]) * vehicle_size

    yaw = pose.so2().theta()
    rotm = np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])

    line = plt.Polygon([p1, p2, p3], closed=True, fill=True,
                       edgecolor=line_color, facecolor=vehicle_color, zorder=2)
    triangle = ax.add_line(line)
    # plot cov
    ps = []
    circle_count = 50
    for i in range(circle_count):
        t = float(i) / float(circle_count) * math.pi * 2.0
        cp = pose.translation() + \
            rotm @ np.matmul(cov[0:2, 0:2],
                             np.array([math.cos(t), math.sin(t)]))
        ps.append(cp)
    line = plt.Polygon(ps, closed=True, fill=False,
                       edgecolor=line_color, zorder=2)
    ellipse = ax.add_line(line)
    return triangle, ellipse


def adjust_figure(fig, ax, aspect, size=7):
    """Add color bar and adjust figure size to fit.

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
