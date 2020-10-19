# This script implements offline detection generation and pole map creation

import glob
import os
import datetime
import sys

try:
    sys.path.append(glob.glob('./carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

import argparse
import yaml
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from detection.vision.lane import LaneMarkingDetector
from detection.vision.pole import PoleDetector
from detection.vision.camproj import im2world_known_z, im2world_known_x
from detection.vision.utils import find_pole_bases, decode_depth
from detection.rs_stop import RSStopDetectionSimulator
from detection.utils import Pole, MELaneMarking, MELaneMarkingType
from detection.pole_map import gen_pole_map
from carlasim.utils import get_fbumper_location, Transform, LaneMarkingColor


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(
            f"readable_dir: {path} is not a valid path")


def main():
    # Parse passed-in config yaml file
    argparser = argparse.ArgumentParser(
        description='Offline Detection and Pole Map Generator')
    argparser.add_argument('recording_dir', type=dir_path,
                           help='directory of recording')
    argparser.add_argument('vision_config', type=argparse.FileType(
        'r'), help='yaml file for vision algorithm parameters')
    argparser.add_argument('sim_detection_config', type=argparse.FileType(
        'r'), help='yaml file for simulated detection configuration')
    argparser.add_argument('pole_map_config', type=argparse.FileType(
        'r'), help='yaml file for pole map generation configuration')
    args = argparser.parse_args()

    # Read carla simulation configs of the recording
    path_to_config = os.path.join(args.recording_dir, 'config.yaml')
    with open(path_to_config, 'r') as f:
        carla_config = yaml.safe_load(f)

    # Read configurations for detection simulation
    with args.vision_config as f:
        vision_config = yaml.safe_load(f)
    with args.sim_detection_config as f:
        sim_detection_config = yaml.safe_load(f)
    with args.pole_map_config as f:
        pole_map_config = yaml.safe_load(f)

    # Load camera parameters
    with open('calib_data.pkl', 'rb') as f:
        calib_data = pickle.load(f)
    K = calib_data['K']
    R = calib_data['R']
    x0 = calib_data['x0']
    # 3-by-4 calibration matrix
    P = K @ R @ np.concatenate((np.eye(3), -x0), axis=1)

    # Load parameters for bird's eye view projection
    with open('ipm_data.pkl', 'rb') as f:
        ipm_data = pickle.load(f)
    M = ipm_data['M']
    warped_size = ipm_data['bev_size']
    valid_mask = ipm_data['valid_mask']
    px_per_meter_x = ipm_data['px_per_meter_x']
    px_per_meter_y = ipm_data['px_per_meter_y']
    dist_cam_to_intersect = ipm_data['dist_to_intersect']

    # Distance from rear axle to front bumper
    dist_raxle_to_fbumper = carla_config['ego_veh']['raxle_to_fbumper']

    # Distance from camera to front bumper
    dist_cam_to_fbumper = (dist_raxle_to_fbumper
                           - carla_config['sensor']['front_camera']['pos_x']
                           - carla_config['ego_veh']['raxle_to_cg'])

    # For lane detector
    # Distance from front bumper to intersection point between camera's vertical FOV and ground surface
    dist_fbumper_to_intersect = dist_cam_to_intersect - dist_cam_to_fbumper

    # Load recorded data that are stored in dictionaries
    with open(os.path.join(args.recording_dir, 'sensor_data.pkl'), 'rb') as f:
        sensor_data = pickle.load(f)
    with open(os.path.join(args.recording_dir, 'gt_data.pkl'), 'rb') as f:
        gt_data = pickle.load(f)

    # Retrieve required sensor data
    ss_images = sensor_data['semantic_camera']['ss_image']
    depth_buffers = sensor_data['depth_camera']['depth_buffer']
    yaw_rates = sensor_data['imu']['gyro_z']
    # Retrieve required ground truth data
    traffic_signs = gt_data['static']['traffic_sign']
    raxle_locations = gt_data['seq']['pose']['raxle_location']
    raxle_orientations = gt_data['seq']['pose']['raxle_orientation']

    left_marking_coeffs_seq = gt_data['seq']['lane']['left_marking_coeffs']
    left_marking_seq = gt_data['seq']['lane']['left_marking']
    right_marking_coeffs_seq = gt_data['seq']['lane']['right_marking_coeffs']
    right_marking_seq = gt_data['seq']['lane']['right_marking']
    lane_id_seq = gt_data['seq']['lane']['lane_id']

    # Create detector objects
    pole_detector = PoleDetector(K, R, x0, vision_config['pole'])
    lane_detector = LaneMarkingDetector(M, px_per_meter_x, px_per_meter_y,
                                        warped_size, valid_mask,
                                        dist_fbumper_to_intersect,
                                        vision_config['lane'])
    rs_stop_detector = RSStopDetectionSimulator(
        traffic_signs, sim_detection_config['rs_stop_gt_extractor'], sim_detection_config['rs_stop'])

    # Create a pole detector that has a stronger ability to extract poles, which is for building the pole map
    # It is stronger in that it allows smaller region of poles to be extracted so the pole map can contain more poles than those
    # detected by pole_detector
    pole_detector_for_pole_map = pole_detector = PoleDetector(
        K, R, x0, pole_map_config['pole_extraction'])

    # Container to accumulate accurate world coordinates of poles at each step
    all_accurate_poles = []

    # Containers for detection sequences
    left_lane_makring_detections = []
    right_lane_marking_detections = []

    # Loop over recorded data
    for image_idx, (ss_image, depth_buffer) in enumerate(zip(ss_images, depth_buffers)):
        # Retrieve data at current step
        ss_image_copy = ss_image
        pole_image = (ss_image == 5).astype(np.uint8)
        lane_image = (ss_image == 6) | (ss_image == 8).astype(np.uint8)
        depth_image = decode_depth(depth_buffer)
        # The pose of rear axle at current step
        raxle_location = raxle_locations[image_idx]
        raxle_orientation = raxle_orientations[image_idx]
        # Front bumper's pose at current step
        fbumper_location = get_fbumper_location(
            raxle_location, raxle_orientation, dist_raxle_to_fbumper)
        fbumper_orientation = raxle_orientation
        # Lane markings
        left_coeffs_gt = left_marking_coeffs_seq[image_idx]
        left_marking_gt = left_marking_seq[image_idx]
        right_coeffs_gt = right_marking_coeffs_seq[image_idx]
        right_marking_gt = right_marking_seq[image_idx]
        lane_id = lane_id_seq[image_idx]

        # Pole detection (wrt front bumper)
        # x-y coordinates assuming z = 0
        poles_xy_z0 = pole_detector.update_poles(pole_image, z=0)
        # Accurate x-y coordinates using ground truth depth image
        accurate__detected_pole_xyz = pole_detector.get_pole_xyz_from_depth(
            depth_image, dist_cam_to_fbumper)

        # Lane detection (wrt front bumper)
        left_coeffs, right_coeffs = lane_detector.update_lane_coeffs(
            lane_image, yaw_rates[image_idx])
        
        # Add color and type properties as part of lane detections
        # The simulated detections are first compared to the ground truth. If the coefficients are close to the ground truth,
        # the recorded marking properties are used as the classification result. 
        # It should be noted that due to the lane detection algorithm implementation details, when the ego vehicle makes a lane
        # change, the detected lane markings will switch to the next lane before the front bumper's center actually enters the next lane.
        # That is, there will be some time points when the lane detection returns the next lane already while the recorded ground truth
        # remains in the current lane since it is based on the front bumper's actual location. In this case, detections will be created
        # with "Other" color and "Unknown" type.
        if left_coeffs is None:
            left_lane_makring_detections.append(None)
        elif abs(left_coeffs[-1] - left_coeffs_gt[0]) < 0.5 and abs(left_coeffs[-2] - left_coeffs_gt[1]) < 0.5:
            left_lane_makring_detections.append(MELaneMarking.from_lane_marking(
                left_coeffs, left_marking_gt, lane_id, 0.0))
        else:
            left_lane_makring_detections.append(MELaneMarking(
                left_coeffs, LaneMarkingColor.Other, MELaneMarkingType.Unknown))

        if right_coeffs is None:
            right_lane_marking_detections.append(None)
        elif abs(right_coeffs[-1] - right_coeffs_gt[0]) < 0.5 and abs(right_coeffs[-2] - right_coeffs_gt[1]) < 0.5:
            right_lane_marking_detections.append(MELaneMarking.from_lane_marking(
                right_coeffs, right_marking_gt, lane_id, 0.0))
        else:
            right_lane_marking_detections.append(MELaneMarking(
                right_coeffs, LaneMarkingColor.Other, MELaneMarkingType.Unknown))

        # RS stop sign detection (wrt front bumper)
        longi_dist_to_rs_stop = rs_stop_detector.update_rs_stop(
            fbumper_location, fbumper_orientation)

        # Accurate poles for building pole map
        # The bases in image are stored internally in the detector
        pole_detector_for_pole_map.find_pole_bases(pole_image)
        accurate_pole_xyz = pole_detector_for_pole_map.get_pole_xyz_from_depth(
            depth_image, dist_cam_to_fbumper)
        # Filter out points that are too high to focus on the lower part of poles
        accurate_pole_xyz = accurate_pole_xyz[:, accurate_pole_xyz[2, :] < 0.5]

        # Instantiate a Transform object to make transformation between front bumper and world
        fbumper2world_tfrom = Transform.from_conventional(
            fbumper_location, fbumper_orientation)
        accurate_pole_xyz_world = fbumper2world_tfrom.tform_e2w_numpy_array(
            accurate_pole_xyz)

        all_accurate_poles.append(accurate_pole_xyz_world)
        print(image_idx)

    # Pole map generation
    # Concatenate all poles as an np.array
    all_accurate_poles_xy = np.concatenate(
        all_accurate_poles, axis=1)[0:2, :]  # 2-by-N

    pole_map = gen_pole_map(all_accurate_poles_xy,
                            traffic_signs, pole_map_config)

    # TODO: Add pole classification


if __name__ == "__main__":
    main()
