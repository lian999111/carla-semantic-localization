"""Utility functions for localization evaluation"""

import math
import sys
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.stats import chi2
from scipy.linalg import sqrtm

try:
    sys.path.append(glob.glob('./carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


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
    """Convert the world coordinates to pixel coordinates in a map image.

    Since the map images are generated using Carla's demo codes, the conversion
    requires the location to be represented in Carla's coordinate system.
    
    Args:
        location (carla.Location): Location of interest.
        map_info (dict): Metainfo of the map image.
    Returns:
        Pixel coordinate of the given carla.Location.
    """
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


def get_local_map_image(loc_gt_seq, pose_estimations, map_image, map_info, margin=25):
    """Get local map image given the trajectory.

    Args:
        loc_gt_seq (list): List of ground truth locations.
        pose_estimations (list of sophus.SE2): List of pose esitmations.
        map_image (np.ndarray): Map image.
        map_info (dict): Metainfo of the map image.
        margin (int): Margin to be included around the trajectory.
    Returns:
        local_map_image: Local map image.
        extent: Extent of the local map image for imshow().
    """
    loc_gts = np.asarray(loc_gt_seq)
    x_estimations = [pose.translation()[0] for pose in pose_estimations]
    y_estimations = [pose.translation()[1] for pose in pose_estimations]
    x_min = min(loc_gts[:, 0].min(), min(x_estimations)) - margin
    x_max = max(loc_gts[:, 0].max(), max(x_estimations)) + margin
    y_min = min(loc_gts[:, 1].min(), min(y_estimations)) - margin
    y_max = max(loc_gts[:, 1].max(), max(y_estimations)) + margin
    extent = [x_min, x_max, y_min, y_max]

    x_center = (x_max + x_min)/2
    y_center = (y_max + y_min)/2
    x_half_width = (x_max - x_min)/2
    y_half_width = (y_max - y_min)/2

    map_center = world_to_pixel(
        carla.Location(x_center, -y_center, 0), map_info)
    left_idx = map_center[0] - int(x_half_width*map_info['pixels_per_meter'])
    right_idx = map_center[0] + int(x_half_width*map_info['pixels_per_meter'])
    bottom_idx = map_center[1] + int(y_half_width*map_info['pixels_per_meter'])
    top_idx = map_center[1] - int(y_half_width*map_info['pixels_per_meter'])
    local_map_image = map_image[top_idx:bottom_idx,
                                left_idx:right_idx]

    return local_map_image, extent


def gen_colored_error_plot(title, errors, loc_gt_seq, pose_estimations, sign_pole_coords, general_pole_coords, local_map_img, extent):
    # Prepare path segments
    x_estimations = [pose.translation()[0] for pose in pose_estimations]
    y_estimations = [pose.translation()[1] for pose in pose_estimations]
    points = np.array([x_estimations, y_estimations]).T.reshape(-1, 1, 2)
    segments = np.concatenate((points[:-1], points[1:]), axis=1)

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    # Ground truth path
    loc_gts = np.asarray(loc_gt_seq)
    ax.plot(loc_gts[:, 0], loc_gts[:, 1], '-o', color='limegreen', ms=1, zorder=0)
    # Ground truth poles
    ax.plot(sign_pole_coords[:, 0], sign_pole_coords[:, 1],
            'o', color='crimson', ms=3, zorder=1)
    ax.plot(general_pole_coords[:, 0], general_pole_coords[:, 1],
            'o', color='midnightblue', ms=3, zorder=1)
    # Resultant path with color
    norm = plt.Normalize(0, 3)
    lc = LineCollection(segments, cmap='gnuplot2', norm=norm)
    # Set the values used for colormapping
    lc.set_array(errors)
    lc.set_linewidth(3)
    line = ax.add_collection(lc)

    # Background map image
    # Height-to-width aspect ratio
    aspect = float(local_map_img.shape[0])/local_map_img.shape[1]
    ax.imshow(local_map_img,
              extent=extent,
              alpha=0.5)
    adjust_figure(fig, aspect)

    # Add color bar
    # Create an axes for colorbar. The position of the axes is calculated based on the position of ax.
    fig_width = fig.get_size_inches()[0]
    cax = fig.add_axes([ax.get_position().x1+0.05/fig_width,
                        ax.get_position().y0,
                        0.1/fig_width,
                        ax.get_position().height])
    fig.colorbar(line, cax=cax)


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
