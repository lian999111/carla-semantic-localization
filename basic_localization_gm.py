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

from carlasim.groundtruth import LaneGTExtractor
from localization.graph_manager import SlidingWindowGraphManager
from localization.utils import ExpectedLaneExtractor


def plotSE2WithCov(pose, cov, vehicle_size=0.5, line_color='k', vehicle_color='r'):
    # plot vehicle
    p1 = pose.translation() + pose.so2() * np.array([1, 0]) * vehicle_size
    p2 = pose.translation() + pose.so2() * \
        np.array([-0.5, -0.5]) * vehicle_size
    p3 = pose.translation() + pose.so2() * np.array([-0.5, 0.5]) * vehicle_size

    yaw = pose.so2().theta()
    rotm = np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])

    line = plt.Polygon([p1, p2, p3], closed=True, fill=True,
                       edgecolor=line_color, facecolor=vehicle_color)
    plt.gca().add_line(line)
    # plot cov
    ps = []
    circle_count = 50
    for i in range(circle_count):
        t = float(i) / float(circle_count) * math.pi * 2.0
        cp = pose.translation() + \
            rotm @ np.matmul(cov[0:2, 0:2],
                             np.array([math.cos(t), math.sin(t)]))
        ps.append(cp)
    line = plt.Polygon(ps, closed=True, fill=False, edgecolor=line_color)
    plt.gca().add_line(line)


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
    settings.no_rendering_mode = True
    carla_world.apply_settings(settings)

    lane_gt_extractor = LaneGTExtractor(carla_world, {'radius': 10})
    expected_lane_extractor = ExpectedLaneExtractor(lane_gt_extractor)

    np.random.seed(2)

    init_idx = 30
    end_idx = 70

    # Prepare figure
    fig, ax = plt.subplots()

    location_gt = np.asarray(raxle_locations)
    loc_x_gt = location_gt[init_idx:end_idx+1, 0]
    loc_y_gt = location_gt[init_idx:end_idx+1, 1]

    ax.plot(loc_x_gt, loc_y_gt, '-o', ms=2)
    plt.show(block=False)

    sw_graph = SlidingWindowGraphManager(
        localization_config, first_node_idx=init_idx)

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
        noised_gnss_x = gnss_x + np.random.normal(-0.0, 0.0)
        noised_gnss_y = gnss_y + np.random.normal(-0.0, 0.0)

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
            # Add geometric lane factor factor
            sw_graph.add_geo_lane_factor(lane_detection, expected_lane_extractor)
            # sw_graph.add_geo_lane_factor(lane_detection.left_marking_detection, expected_lane_extractor)
            # sw_graph.add_geo_lane_factor(lane_detection.right_marking_detection, expected_lane_extractor)
            # if idx - init_idx > 2:
            #     # Add geometric lane factor factor
            #     sw_graph.add_geo_lane_factor(lane_detection, expected_lane_extractor)

        sw_graph.try_move_sliding_window_forward()

        # Optimize graph
        sw_graph.solve_one_step()

        plt.cla()
        ax.plot(loc_x_gt, loc_y_gt, '-o', ms=2)
        plt.show(block=False)
        for idx in sw_graph.get_idc_in_graph():
            cov = sw_graph.get_marignal_cov_matrix(idx)
            # print(cov)
            plotSE2WithCov(sw_graph.get_result(idx), cov)
        
        plt.axis('equal')

        last_pos = sw_graph.last_optimized_se2.translation()
        ax.set_xlim((last_pos[0]-10, last_pos[0]+10))
        ax.set_ylim((last_pos[1]-10, last_pos[1]+10))
        

        # plt.axis('equal')
        plt.pause(0.001)

        if idx >= end_idx:
            break

    plt.show()


if __name__ == "__main__":
    main()
