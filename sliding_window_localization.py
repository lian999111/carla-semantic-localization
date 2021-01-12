"""Localization using graph-based sliding window."""

import os
import argparse
import pickle
import math
import sys
import glob

import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import imageio
import pygame

from carlasim.groundtruth import LaneGTExtractor, RSStopGTExtractor
from carlasim.utils import TrafficSignType
from localization.graph_manager import SlidingWindowGraphManager
from localization.utils import ExpectedLaneExtractor, ExpectedPoleExtractor, ExpectedRSStopExtractor
from localization.eval.map_image import MapImage
from localization.eval.utils import world_to_pixel, plot_se2_with_cov, adjust_figure

try:
    sys.path.append(glob.glob('./carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(
            f"readable_dir: {path} is not a valid path")


def main():
    """Main function"""
    ############### Parse arguments ###############
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
    with open(os.path.join(args.recording_dir, 'pole_map.pkl'), 'rb') as f:
        pole_map = pickle.load(f)

    # Read carla simulation configs of the recording for dist_raxle_to_fbumper
    path_to_config = os.path.join(args.recording_dir, 'settings/config.yaml')
    with open(path_to_config, 'r') as f:
        carla_config = yaml.safe_load(f)
    dist_raxle_to_fbumper = carla_config['ego_veh']['raxle_to_fbumper']
    dist_cam_to_fbumper = dist_raxle_to_fbumper \
        - carla_config['sensor']['front_camera']['pos_x'] \
        - carla_config['ego_veh']['raxle_to_cg']

    # Read configurations for localization
    with args.localization_config as f:
        localization_config = yaml.safe_load(f)

    ############### Retrieve required data ###############
    # Ground truth
    # Pose ground truth
    raxle_locations = gt_data['seq']['pose']['raxle_location']
    raxle_orientations = gt_data['seq']['pose']['raxle_orientation']

    # Traffic sign ground truth
    traffic_signs = gt_data['static']['traffic_sign']

    # Lane marking ground truth
    left_marking_coeffs_seq = gt_data['seq']['lane']['left_marking_coeffs']
    left_marking_seq = gt_data['seq']['lane']['left_marking']
    right_marking_coeffs_seq = gt_data['seq']['lane']['right_marking_coeffs']
    right_marking_seq = gt_data['seq']['lane']['right_marking']

    # Sensor data
    timestamp_seq = sensor_data['gnss']['timestamp']
    gnss_x_seq = sensor_data['gnss']['x']
    gnss_y_seq = sensor_data['gnss']['y']
    gnss_z_seq = sensor_data['gnss']['z']

    vx_seq = sensor_data['imu']['vx']
    gyro_z_seq = sensor_data['imu']['gyro_z']

    # Simulated detections
    lane_detection_seq = detections['lane']
    pole_detection_seq = detections['pole']
    rs_stop_detection_seq = detections['rs_stop']

    # Indices to clip the recording
    # Only data between the indices are used
    init_idx = 5
    end_idx = 2000

    ############### Connect to Carla server ###############
    client = carla.Client('localhost', 2000)
    client.set_timeout(3.0)
    carla_world = client.load_world(carla_config['world']['map'])

    settings = carla_world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = 0.0
    settings.no_rendering_mode = False
    carla_world.apply_settings(settings)

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
                       alpha=0.5)
    # Dot of gnss measurement
    gnss_dot = ax.plot([], [], 'o', color='gold', ms=4, zorder=3)[0]
    # Lane boundary detection
    left_lb = ax.plot([], [])[0]
    right_lb = ax.plot([], [])[0]

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.show(block=False)

    ############### Sliding window graph ###############
    # Expected measurement extractors must be created in advance and given to
    # the sliding window graph manager. They serve as interfaces between the graph
    # manager wnd the underlying map data.

    # ExpectedLaneExtractor uses a LaneGTExtractor internally to do the queries.
    # Note: The extracted lane boundaries are wrt the query point.
    lane_gt_extractor = LaneGTExtractor(carla_world,
                                        localization_config['lane']['lane_gt_extractor'],
                                        debug=False)
    expected_lane_extractor = ExpectedLaneExtractor(lane_gt_extractor)

    # ExpectedPoleExtractor extracts map poles given query points.
    # Note: The extracted poles are wrt the world frame. Transformation
    # is taken care of by the graph manager before feeding them to pole factors.
    expected_pole_extractor = ExpectedPoleExtractor(pole_map)

    # ExpectedRSStopExtractor uses a RSStopGTExtractor internally to do the queries.
    # Note: The extracted lane boundaries are wrt the query point.
    rs_stop_gt_extractor = RSStopGTExtractor(traffic_signs,
                                             localization_config['rs_stop']['rs_stop_gt_extractor'])
    expected_rs_stop_extractor = ExpectedRSStopExtractor(rs_stop_gt_extractor)

    # Create a sliding window graph manager
    sw_graph = SlidingWindowGraphManager(dist_raxle_to_fbumper,
                                         dist_cam_to_fbumper,
                                         localization_config,
                                         expected_lane_extractor,
                                         expected_pole_extractor,
                                         expected_rs_stop_extractor,
                                         first_node_idx=init_idx)

    ############### Loop through recorded data ###############
    # Fix the seed for noise added afterwards
    np.random.seed(0)

    # List for storing pose of each time step after optimization
    optimized_poses = []
    gif_image_seq = []
    for idx, timestamp in enumerate(timestamp_seq):

        if idx < init_idx:
            continue

        delta_t = timestamp - timestamp_seq[idx-1]
        vx = vx_seq[idx] + np.random.normal(0, 0.2)
        yaw_rate = gyro_z_seq[idx] + np.random.normal(0, 0.1)

        left_marking_coeffs_gt = np.asarray(left_marking_coeffs_seq[idx])
        right_marking_coeffs_gt = np.asarray(right_marking_coeffs_seq[idx])

        # Detection
        lane_detection = lane_detection_seq[idx]
        pole_detection = pole_detection_seq[idx]
        rs_stop_detection = rs_stop_detection_seq[idx]

        gnss_x = gnss_x_seq[idx]
        gnss_y = gnss_y_seq[idx]
        gnss_z = gnss_z_seq[idx]
        noised_gnss_x = gnss_x + np.random.normal(3.0, 3.0)
        noised_gnss_y = gnss_y + np.random.normal(3.0, 3.0)

        raxle_loacation_gt = raxle_locations[idx]
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

            # Add pole factor after 3 steps for init estimation to converge
            if idx - init_idx > 3:
                if pole_detection is not None:
                    for detected_pole in pole_detection:
                        # if detected_pole.x < 50 and detected_pole.type != TrafficSignType.Unknown:
                        if detected_pole.x < 50 and abs(detected_pole.y) < 25:
                            sw_graph.add_pole_factor(detected_pole)

            # Add lane factor after 3 steps for init estimation to converge
            if idx - init_idx > 6:
                if lane_detection.left_marking_detection is not None:
                    sw_graph.add_lane_factor(
                        lane_detection.left_marking_detection, gnss_z)
                if lane_detection.right_marking_detection is not None:
                    sw_graph.add_lane_factor(
                        lane_detection.right_marking_detection, gnss_z)

            if rs_stop_detection is not None:
                sw_graph.add_rs_stop_factor(rs_stop_detection, gnss_z)

        # Truncate the graph if necessary
        sw_graph.try_move_sliding_window_forward()

        # Optimize graph
        sw_graph.solve_one_step()

        # Record the lastest pose after optimization
        last_pose = sw_graph.last_optimized_se2
        optimized_poses.append(last_pose)

        ##### Visualize current step #####
        half_width = 15  # (m) half width of background map
        half_width_px = half_width * map_info['pixels_per_meter']

        ### background map ###
        # Get image coordinate of the latest pose on the map image
        last_loc = last_pose.translation()
        image_coord = world_to_pixel(carla.Location(
            last_loc[0], -last_loc[1]), map_info)

        # Crop the map image for display
        local_map_image = map_image[image_coord[1]-half_width_px:image_coord[1]+half_width_px,
                                    image_coord[0]-half_width_px:image_coord[0]+half_width_px]

        # Past the cropped map image to the correct place
        left = last_loc[0] - half_width
        right = last_loc[0] + half_width
        bottom = last_loc[1] - half_width
        top = last_loc[1] + half_width
        map_im.set_data(local_map_image)
        map_im.set_extent([left, right, bottom, top])

        ### Visualize poses ###
        pose_plots = []
        for node_idx in sw_graph.get_idc_in_graph():
            pose = sw_graph.get_result(node_idx)
            cov = sw_graph.get_marignal_cov_matrix(node_idx)
            pose_plots.append(plot_se2_with_cov(
                ax, pose, cov, confidence=0.999))

        ### Visualize GNSS ###
        gnss_dot.set_data(noised_gnss_x, noised_gnss_y)

        ### Visualize Lane boundary detection ###
        last_se2 = sw_graph.last_optimized_se2
        last_x = last_se2.translation()[0]
        last_y = last_se2.translation()[1]
        last_yaw = last_se2.so2().theta()
        # Matrix to transform points in ego frame to world frame
        tform_e2w = np.array([[math.cos(last_yaw), -math.sin(last_yaw), last_x],
                              [math.sin(last_yaw), math.cos(last_yaw), last_y],
                              [0, 0, 1]])
        # Matrix to transform points in front bumper to ego frame
        tfrom_fbumper2raxel = np.array([[1, 0, dist_raxle_to_fbumper],
                                        [0, 1, 0],
                                        [0, 0, 1]])
        # Matrix to transform points in front bumper to world frame
        tform_fbumper2w = tform_e2w @ tfrom_fbumper2raxel
        lb_x = np.linspace(0, 10, 10)
        if lane_detection.left_marking_detection:
            lb_y = lane_detection.left_marking_detection.compute_y(lb_x)
            lb_pts_wrt_fbumper = np.array(
                [lb_x, lb_y, np.ones((lb_x.shape[0],))])
            lb_pts_world = tform_fbumper2w @ lb_pts_wrt_fbumper
            # Update plot
            left_lb.set_data(lb_pts_world[0], lb_pts_world[1])
        else:
            # Update plot
            left_lb.set_data([], [])

        if lane_detection.right_marking_detection:
            lb_y = lane_detection.right_marking_detection.compute_y(lb_x)
            lb_pts_wrt_fbumper = np.array(
                [lb_x, lb_y, np.ones((lb_x.shape[0],))])
            lb_pts_world = tform_fbumper2w @ lb_pts_wrt_fbumper
            # Update plot
            right_lb.set_data(lb_pts_world[0], lb_pts_world[1])
        else:
            # Update plot
            right_lb.set_data([], [])

        ax.set_title(idx)
        plt.pause(0.001)

        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        gif_image_seq.append(image)

        # Remove artists for poses
        for triangle, ellipse in pose_plots:
            triangle.remove()
            ellipse.remove()

        if idx >= end_idx:
            break

    # imageio.mimsave('./localization.gif', gif_image_seq, fps=10)

    # TODO: Make following functions in eval package
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
    x_min = min(loc_gts[:, 0].min(), min(opti_loc_x)) - margin
    x_max = max(loc_gts[:, 0].max(), max(opti_loc_x)) + margin
    y_min = min(loc_gts[:, 1].min(), min(opti_loc_y)) - margin
    y_max = max(loc_gts[:, 1].max(), max(opti_loc_y)) + margin
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

    ## Longitudinal error ##
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
    adjust_figure(fig, aspect)

    # Add color bar
    # Create an axes for colorbar. The position of the axes is calculated based on the position of ax.
    fig_width = fig.get_size_inches()[0]
    cax = fig.add_axes([ax.get_position().x1+0.05/fig_width,
                        ax.get_position().y0,
                        0.1/fig_width,
                        ax.get_position().height])
    fig.colorbar(line, cax=cax)

    ## Lateral error ##
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
    adjust_figure(fig, aspect)

    # Add color bar
    # Create an axes for colorbar. The position of the axes is calculated based on the position of ax.
    fig_width = fig.get_size_inches()[0]
    cax = fig.add_axes([ax.get_position().x1+0.05/fig_width,
                        ax.get_position().y0,
                        0.1/fig_width,
                        ax.get_position().height])
    fig.colorbar(line, cax=cax)

    ## Yaw error ##
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

    adjust_figure(fig, aspect)

    # Add color bar
    # Create an axes for colorbar. The position of the axes is calculated based on the position of ax.
    fig_width = fig.get_size_inches()[0]
    cax = fig.add_axes([ax.get_position().x1+0.05/fig_width,
                        ax.get_position().y0,
                        0.1/fig_width,
                        ax.get_position().height])
    fig.colorbar(line, cax=cax)

    plt.show()


if __name__ == "__main__":
    main()
