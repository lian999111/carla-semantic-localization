import os
import argparse
import yaml
import pickle
import numpy as np
import math
import sys
import glob

try:
    sys.path.append(glob.glob('./carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

import minisam as ms

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import pygame

from carlasim.groundtruth import LaneGTExtractor
from localization.graph_manager import SlidingWindowGraphManager
from localization.utils import ExpectedLaneExtractor

from map_image import MapImage
from PIL import Image


def world_to_pixel(location, map_info, offset=(0, 0)):
    """Converts the world coordinates to pixel coordinates"""
    x = map_info['scale'] * map_info['pixels_per_meter'] * \
        (location.x - map_info['world_offset_x'])
    y = map_info['scale'] * map_info['pixels_per_meter'] * \
        (location.y - map_info['world_offset_y'])
    return [int(x - offset[0]), int(y - offset[1])]


def plotSE2WithCov(ax, pose, cov, vehicle_size=0.5, line_color='k', vehicle_color='r'):
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


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(
            f"readable_dir: {path} is not a valid path")


def main():
    # Parse arguments
    argparser = argparse.ArgumentParser(
        description='Visualization of Offline Detections')
    argparser.add_argument('recording_dir',
                           type=dir_path,
                           help='directory of recording')
    argparser.add_argument('localization_config',
                           type=argparse.FileType('r'),
                           help='yaml file for localization configuration')
    args = argparser.parse_args()

    # Load data in the recording folder
    with open(os.path.join(args.recording_dir, 'sensor_data.pkl'), 'rb') as f:
        sensor_data = pickle.load(f)
    with open(os.path.join(args.recording_dir, 'gt_data.pkl'), 'rb') as f:
        gt_data = pickle.load(f)
    with open(os.path.join(args.recording_dir, 'detections.pkl'), 'rb') as f:
        detections = pickle.load(f)
    # with open(os.path.join(args.recording_dir, 'pole_map.pkl'), 'rb') as f:
    #     pole_map = pickle.load(f)

    # Read carla simulation configs of the recording for dist_raxle_to_fbumper
    path_to_config = os.path.join(args.recording_dir, 'settings/config.yaml')
    with open(path_to_config, 'r') as f:
        carla_config = yaml.safe_load(f)
    dist_raxle_to_fbumper = carla_config['ego_veh']['raxle_to_fbumper']

    # Read configurations for localization
    with args.localization_config as f:
        localization_config = yaml.safe_load(f)

    # Retrieve required data
    timestamp_seq = sensor_data['gnss']['timestamp']
    gnss_x_seq = sensor_data['gnss']['x']
    gnss_y_seq = sensor_data['gnss']['y']
    gnss_z_seq = sensor_data['gnss']['z']

    vx_seq = sensor_data['imu']['vx']
    gyro_z_seq = sensor_data['imu']['gyro_z']

    raxle_locations = gt_data['seq']['pose']['raxle_location']
    raxle_orientations = gt_data['seq']['pose']['raxle_orientation']

    lane_id_seq = gt_data['seq']['lane']['lane_id']
    left_marking_coeffs_seq = gt_data['seq']['lane']['left_marking_coeffs']
    left_marking_seq = gt_data['seq']['lane']['left_marking']
    right_marking_coeffs_seq = gt_data['seq']['lane']['right_marking_coeffs']
    right_marking_seq = gt_data['seq']['lane']['right_marking']

    lane_detection_seq = detections['lane']

    # Connect to Carla server
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    carla_world = client.load_world(carla_config['world']['map'])

    settings = carla_world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = 0.0
    settings.no_rendering_mode = False
    carla_world.apply_settings(settings)

    lane_gt_extractor = LaneGTExtractor(carla_world, {'radius': 10}, True)
    expected_lane_extractor = ExpectedLaneExtractor(lane_gt_extractor)

    np.random.seed(2)

    init_idx = 10
    end_idx = 1000

    ############### Load map image ###############
    dirname = os.path.join("cache", "map_images")
    filename = carla_config['world']['map'] + '.jpg'
    full_path = str(os.path.join(dirname, filename))

    # If map image does not exist, create it
    if not os.path.isfile(full_path):
        # pygame is needed for map rendering
        pygame.init()
        display = pygame.display.set_mode(
            (600, 200),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        # Show loading screen
        display.fill(pygame.Color(0, 0, 0))
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        text_surface = font.render('Rendering map using pygame...',
                                   True,
                                   pygame.Color(255, 255, 255))
        display.blit(text_surface,
                     text_surface.get_rect(center=(300, 100)))
        pygame.display.flip()

        # MapImage class was part of example code "no_rendering_mode.py"
        # It is just borrowed (with a little modifications) here to create and store map image.
        MapImage(carla_world,
                 carla_world.get_map(),
                 pixels_per_meter=20,
                 show_triggers=False,
                 show_connections=False,
                 show_spawn_points=False)
        pygame.quit()

    map_image = plt.imread(full_path)

    info_filename = carla_config['world']['map'] + '_info.yaml'
    info_full_path = str(os.path.join(dirname, info_filename))
    with open(info_full_path, 'r') as f:
        map_info = yaml.safe_load(f)

    ############### Prepare figure ###############
    fig, ax = plt.subplots()

    loc_gt_seq = raxle_locations[init_idx:end_idx+1]
    ori_gt_seq = raxle_orientations[init_idx:end_idx+1]
    location_gt = np.asarray(loc_gt_seq)
    loc_x_gt = location_gt[:, 0]
    loc_y_gt = location_gt[:, 1]

    # Path ground truth
    gt_path = ax.plot(loc_x_gt, loc_y_gt, '-o', ms=2)

    # Create a dummy map background
    map_im = ax.imshow(np.zeros((1, 1, 3), dtype=int),
                       extent=[0, 0, 0, 0],
                       alpha=0.5)

    ax.axis('equal')
    plt.show(block=False)

    ############### Sliding window graph ###############
    sw_graph = SlidingWindowGraphManager(
        localization_config, expected_lane_extractor, first_node_idx=init_idx)

    optimized_poses = []

    for idx, timestamp in enumerate(timestamp_seq):

        if idx < init_idx:
            continue

        delta_t = timestamp - timestamp_seq[idx-1]
        vx = vx_seq[idx] + np.random.normal(0, 0.0)
        yaw_rate = gyro_z_seq[idx] + np.random.normal(0, 0.0)

        lane_id = lane_id_seq[idx]
        left_marking_coeffs = np.asarray(left_marking_coeffs_seq[idx])
        right_marking_coeffs = np.asarray(right_marking_coeffs_seq[idx])

        # Detection
        lane_detection = lane_detection_seq[idx]
        left_marking_detection = lane_detection.left_marking_detection
        if left_marking_detection is not None:
            left_detection_coeffs = np.flip(
                left_marking_detection.coeffs[1:3]).tolist()
        else:
            left_detection_coeffs = None

        gnss_x = gnss_x_seq[idx]
        gnss_y = gnss_y_seq[idx]
        gnss_z = gnss_z_seq[idx]
        noised_gnss_x = gnss_x + np.random.normal(-0.0, 1.0)
        noised_gnss_y = gnss_y + np.random.normal(-0.0, 1.0)

        yaw_gt = raxle_orientations[idx][2]

        # Add prior factor
        if idx == init_idx:
            sw_graph.add_prior_factor(noised_gnss_x, noised_gnss_y, yaw_gt)

        if idx > init_idx:
            # Add CTRV between factor
            sw_graph.add_ctrv_between_factor(
                vx, yaw_rate, delta_t, add_init_guess=True)
            # Add GNSS factor
            sw_graph.add_gnss_factor(
                np.array([noised_gnss_x, noised_gnss_y]), add_init_guess=False)
            # Add lane factor
            if idx - init_idx > 20:
                if lane_detection.left_marking_detection is not None:
                    c0 = lane_detection.left_marking_detection.get_c0c1_list()[
                        0]
                    if abs(c0) <= 3.5:
                        sw_graph.add_lane_factor(
                            lane_detection.left_marking_detection, gnss_z)
                if lane_detection.right_marking_detection is not None:
                    c0 = lane_detection.right_marking_detection.get_c0c1_list()[
                        0]
                    if abs(c0) <= 3.5:
                        sw_graph.add_lane_factor(
                            lane_detection.right_marking_detection, gnss_z)

        sw_graph.try_move_sliding_window_forward()

        # Optimize graph
        sw_graph.solve_one_step()

        last_pose = sw_graph.last_optimized_se2
        last_loc = last_pose.translation()
        optimized_poses.append(last_pose)

        ######## Visualize ########
        half_width = 15  # half width of background map
        half_width_px = half_width * map_info['pixels_per_meter']

        # Get image coordinate of
        image_coord = world_to_pixel(carla.Location(
            last_loc[0], -last_loc[1]), map_info)
        local_map_image = map_image[image_coord[1]-half_width_px:image_coord[1]+half_width_px,
                                    image_coord[0]-half_width_px:image_coord[0]+half_width_px]
        left = (last_loc[0]-half_width)
        right = (last_loc[0]+half_width)
        bottom = (last_loc[1]-half_width)
        top = (last_loc[1]+half_width)

        map_im.set_data(local_map_image)
        map_im.set_extent([left, right, bottom, top])

        pose_plots = []
        for idx in sw_graph.get_idc_in_graph():
            cov = sw_graph.get_marignal_cov_matrix(idx)
            pose_plots.append(plotSE2WithCov(
                ax, sw_graph.get_result(idx), cov))

        ax.set_xlim((last_loc[0]-half_width, last_loc[0]+half_width))
        ax.set_ylim((last_loc[1]-half_width, last_loc[1]+half_width))

        plt.pause(0.001)

        for triangle, ellipse in pose_plots:
            triangle.remove()
            ellipse.remove()

        if idx >= end_idx:
            break

    ############### Evaluate ###############
    opti_loc_x = []
    opti_loc_y = []
    opti_yaw = []
    longitudinal_errors = []
    lateral_errors = []
    yaw_errors = []
    for loc_gt, ori_gt, opti_pose in zip(loc_gt_seq, ori_gt_seq, optimized_poses):
        x_gt = loc_gt[0]
        y_gt = loc_gt[1]
        yaw_gt = ori_gt[2]

        # Translational error
        # Matrix that transforms a point in ego frame to world frame
        tform_e2w = np.array([[math.cos(yaw_gt), -math.sin(yaw_gt), x_gt],
                              [math.sin(yaw_gt), math.cos(yaw_gt), y_gt],
                              [0, 0, 1]])
        tform_w2e = np.linalg.inv(tform_e2w)

        trvec_world = np.append(opti_pose.translation(), 1)
        trvec_ego = tform_w2e @ trvec_world

        longitudinal_errors.append(trvec_ego[0])
        lateral_errors.append(trvec_ego[1])

        # Rotational error
        yaw = opti_pose.so2().theta()
        yaw_error = yaw - yaw_gt
        # Since yaw angle is in a cyclic space, when the amount of error is larger than 180 degrees,
        # we need to correct it.
        if yaw_error > math.pi:
            yaw_error = 2*math.pi - yaw_error
        elif yaw_error < -math.pi:
            yaw_error = 2*math.pi + yaw_error
        yaw_errors.append(yaw_error)

        # Localization results
        opti_loc_x.append(opti_pose.translation()[0])
        opti_loc_y.append(opti_pose.translation()[1])
        opti_yaw.append(yaw)

    # Absolute errors
    abs_longitudinal_errors = np.abs(np.asarray(longitudinal_errors))
    abs_lateral_errors = np.abs(np.asarray(lateral_errors))
    abs_yaw_errors = np.abs(np.asarray(yaw_errors))

    ############### Visualize errors ###############
    # Prepare background
    loc_gts = np.asarray(loc_gt_seq)
    margin = 7  # (m)
    x_min = loc_gts[:, 0].min() - margin
    x_max = loc_gts[:, 0].max() + margin
    y_min = loc_gts[:, 1].min() - margin
    y_max = loc_gts[:, 1].max() + margin
    x_center = (x_max + x_min)/2
    y_center = (y_max + y_min)/2
    x_half_width = (x_max - x_min)/2
    y_half_width = (y_max - y_min)/2
    aspect = y_half_width/x_half_width  # height-to-width aspect ratio

    map_center = world_to_pixel(
        carla.Location(x_center, -y_center, 0), map_info)
    left_idx = map_center[0] - int(x_half_width*map_info['pixels_per_meter'])
    right_idx = map_center[0] + int(x_half_width*map_info['pixels_per_meter'])
    bottom_idx = map_center[1] + int(y_half_width*map_info['pixels_per_meter'])
    top_idx = map_center[1] - int(y_half_width*map_info['pixels_per_meter'])
    local_map_image = map_image[top_idx:bottom_idx,
                                left_idx:right_idx]

    # Prepare path segments
    points = np.array([opti_loc_x, opti_loc_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate((points[:-1], points[1:]), axis=1)

    # Longitudinal error
    fig, ax = plt.subplots()
    ax.set_title('Longitudinal Error (m)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    # Ground truth path
    ax.plot(loc_x_gt, loc_y_gt, '-o', ms=1, zorder=0)
    # Resultant path with color
    norm = plt.Normalize(0, 3)
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(abs_longitudinal_errors)
    lc.set_linewidth(3)
    line = ax.add_collection(lc)
    ax.imshow(local_map_image,
              extent=[x_min, x_max, y_min, y_max],
              alpha=0.5)
    adjust_figure(fig, ax, aspect)

    # Add color bar
    # Create an axes for colorbar. The position of the axes is calculated based on the position of ax.
    fig_width = fig.get_size_inches()[0]
    cax = fig.add_axes([ax.get_position().x1+0.02,
                        ax.get_position().y0,
                        0.1/fig_width,
                        ax.get_position().height])
    fig.colorbar(line, cax=cax)

    # Lateral error
    fig, ax = plt.subplots()
    ax.set_title('Lateral Error (m)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    # Ground truth path
    ax.plot(loc_x_gt, loc_y_gt, '-o', ms=1, zorder=0)
    # Resultant path with color
    norm = plt.Normalize(0, 1)
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(abs_lateral_errors)
    lc.set_linewidth(3)
    line = ax.add_collection(lc)
    ax.imshow(local_map_image,
              extent=[x_min, x_max, y_min, y_max],
              alpha=0.5)
    adjust_figure(fig, ax, aspect)

    # Add color bar
    # Create an axes for colorbar. The position of the axes is calculated based on the position of ax.
    fig_width = fig.get_size_inches()[0]
    cax = fig.add_axes([ax.get_position().x1+0.02,
                        ax.get_position().y0,
                        0.1/fig_width,
                        ax.get_position().height])
    fig.colorbar(line, cax=cax)

    # Yaw error
    fig, ax = plt.subplots()
    ax.set_title('Yaw Error (rad)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    # Ground truth path
    ax.plot(loc_x_gt, loc_y_gt, '-o', ms=1, zorder=0)
    # Resultant path with color
    norm = plt.Normalize(0, 0.5)
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(abs_yaw_errors)
    lc.set_linewidth(3)
    line = ax.add_collection(lc)
    ax.imshow(local_map_image,
              extent=[x_min, x_max, y_min, y_max],
              alpha=0.5)

    adjust_figure(fig, ax, aspect)

    # Add color bar
    # Create an axes for colorbar. The position of the axes is calculated based on the position of ax.
    fig_width = fig.get_size_inches()[0]
    cax = fig.add_axes([ax.get_position().x1+0.02,
                        ax.get_position().y0,
                        0.1/fig_width,
                        ax.get_position().height])
    fig.colorbar(line, cax=cax)

    plt.show()


if __name__ == "__main__":
    main()
