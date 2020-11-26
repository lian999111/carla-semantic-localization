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

from minisam import FactorGraph, Factor, PriorFactor, BetweenFactor, GaussianLoss, DiagonalLoss, ScaleLoss, key, keyString, Variables
from minisam import LevenbergMarquardtOptimizerParams, LevenbergMarquardtOptimizer, NonlinearOptimizerVerbosityLevel
from minisam import GaussNewtonOptimizerParams, GaussNewtonOptimizer
from minisam import NonlinearOptimizerVerbosityLevel, NonlinearOptimizationStatus, MarginalCovarianceSolver, MarginalCovarianceSolverStatus
from minisam.sophus import SE2, SO2

import minisam as ms

import matplotlib.pyplot as plt

from carlasim.groundtruth import LaneGTExtractor
from localization.gnss import GNSSFactor
from localization.lane import GeometricLaneBoundaryFactor
from localization.odom import create_ctrv_between_factor


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
    # with open(os.path.join(args.recording_dir, 'detections.pkl'), 'rb') as f:
    #     detections = pickle.load(f)
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

    # Connect to Carla server
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    carla_world = client.load_world('Town04')

    settings = carla_world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    settings.no_rendering_mode = True
    carla_world.apply_settings(settings)

    lane_gt_extractor = LaneGTExtractor(carla_world, {'radius': 10})

    # Create a factor graph
    graph = FactorGraph()
    initials = Variables()
    win_size = localization_config['graph']['win_size']

    # Prepare solver
    # Use LM method optimizes the initial values
    opt_param = LevenbergMarquardtOptimizerParams()
    opt_param.verbosity_level = NonlinearOptimizerVerbosityLevel.ITERATION
    # opt_param.max_iterations = 1
    opt = LevenbergMarquardtOptimizer(opt_param)
    # opt_param = GaussNewtonOptimizerParams()
    # opt_param.verbosity_level = NonlinearOptimizerVerbosityLevel.ITERATION
    # opt = GaussNewtonOptimizer(opt_param)

    # Prepare cov solver
    mcov_solver = MarginalCovarianceSolver()

    np.random.seed(2)

    init_idx = 30
    end_idx = 70

    graph_init_idx = init_idx
    graph_end_idx = init_idx + win_size
    num_poses = 0

    # Prepare figure
    fig, ax = plt.subplots()

    location_gt = np.asarray(raxle_locations)
    loc_x_gt = location_gt[init_idx:end_idx+1, 0]
    loc_y_gt = location_gt[init_idx:end_idx+1, 1]

    ax.plot(loc_x_gt, loc_y_gt, '-o', ms=2)
    plt.show(block=False)

    for idx, timestamp in enumerate(timestamp_seq):
        if idx < init_idx or idx > end_idx:
            continue
        delta_t = timestamp_seq[idx+1] - timestamp
        vx = vx_seq[idx] + np.random.normal(0, 0.1)
        yaw_rate = gyro_z_seq[idx] + np.random.normal(0, 0.1)

        lane_id = lane_id_seq[idx]
        left_marking_coeffs = np.asarray(left_marking_coeffs_seq[idx])

        gnss_x = gnss_x_seq[idx]
        gnss_y = gnss_y_seq[idx]
        noised_gnss_x = gnss_x + np.random.normal(1.0, 0.0)
        noised_gnss_y = gnss_y + np.random.normal(-0.0, 0.0)

        yaw_gt = raxle_orientations[idx][2]

        # Odom factor
        if idx > init_idx:
            graph.add(create_ctrv_between_factor(key('x', idx-1),
                                                 key('x', idx),
                                                 vx, yaw_rate,
                                                 delta_t,
                                                 localization_config['ctrv']))

        if idx >= init_idx:
            # GNSS factor
            graph.add(GNSSFactor(key('x', idx),
                                 np.array([noised_gnss_x, noised_gnss_y]),
                                 localization_config['gnss']))

            # Lane factor
            graph.add(GeometricLaneBoundaryFactor(key('x', idx),
                                                  left_marking_coeffs,
                                                  dist_raxle_to_fbumper,
                                                  lane_gt_extractor,
                                                  localization_config['geometric_lane']))

        # Initials
        initials.add(key('x', idx), SE2(SO2(yaw_gt+np.random.normal(0.0, 0.0)),
                                        np.array([noised_gnss_x, noised_gnss_y])))

        if graph[len(graph)-1].keys()[-1] == key('x', graph_end_idx):
            while True:
                first_factor = graph[0]
                if first_factor.keys()[0] == key('x', graph_init_idx):
                    graph.erase(0)
                    initials.erase(key('x', graph_init_idx))
                else:
                    break

            graph_init_idx += 1
            graph_end_idx += 1
        else:
            num_poses += 1

        if num_poses < 2:
            continue

        results = Variables()
        status = opt.optimize(graph, initials, results)
        print(results)
        initials = results

        if status != NonlinearOptimizationStatus.SUCCESS:
            print("optimization error: ", status)

        # Calculate marginal covariances for all poses
        status = mcov_solver.initialize(graph, results)
        if status != MarginalCovarianceSolverStatus.SUCCESS:
            print("maginal covariance error")
            print(status)

        plt.cla()
        ax.plot(loc_x_gt, loc_y_gt, '-o', ms=2)
        plt.show(block=False)
        for idx in range(graph_init_idx, graph_init_idx + num_poses):
            cov = mcov_solver.marginalCovariance(key('x', idx))
            # print(cov)
            plotSE2WithCov(results.at(key('x', idx)), cov)
        plt.axis('equal')
        plt.pause(0.001)

        if idx >= end_idx:
            break

    plt.show()


if __name__ == "__main__":
    main()
