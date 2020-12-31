"""Implements offline detection generation and pole map creation"""

import os
import argparse
import pickle
from shutil import copyfile

import yaml
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

from detection.vision.lane import LaneMarkingDetector
from detection.vision.pole import PoleDetector
from detection.vision.utils import decode_depth
from detection.rs_stop import RSStopDetectionSimulator
from detection.utils import Pole, MELaneMarking, MELaneMarkingType, MELaneDetection
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
    path_to_config = os.path.join(args.recording_dir, 'settings/config.yaml')
    with open(path_to_config, 'r') as f:
        carla_config = yaml.safe_load(f)

    # Read configurations for detection simulation
    with args.vision_config as f:
        vision_config = yaml.safe_load(f)
    with args.sim_detection_config as f:
        sim_detection_config = yaml.safe_load(f)
    with args.pole_map_config as f:
        pole_map_config = yaml.safe_load(f)

    # Retrieve configs of detection simulation
    pole_detection_sim_config = sim_detection_config['pole']
    lane_detection_sim_config = sim_detection_config['lane']
    rs_stop_detection_sim_config = sim_detection_config['rs_stop']

    # Load camera parameters
    with open('calib_data.pkl', 'rb') as f:
        calib_data = pickle.load(f)
    K = calib_data['K']
    R = calib_data['R']
    x0 = calib_data['x0']

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
    rs_stop_detector = RSStopDetectionSimulator(traffic_signs,
                                                rs_stop_detection_sim_config)

    # Create a pole detector that has a stronger ability to extract poles, which is for building the pole map
    # It is stronger in that it allows smaller region of poles to be extracted so the pole map can contain more poles than those
    # detected by pole_detector
    pole_detector_for_pole_map = PoleDetector(K, R, x0,
                                              pole_map_config['pole_extraction'])

    # Container to accumulate accurate world coordinates of poles at each step
    all_accurate_poles = []

    # Containers for detection sequences
    # Sequence of pole detecions
    pole_detection_seq = []
    # The accurate version of detected poles in world frame
    accurate_pole_detections_in_world_seq = []
    lane_marking_detection_seq = []
    rs_stop_detecion_seq = []

    # Loop over recorded data
    for image_idx, (ss_image, depth_buffer) in enumerate(zip(ss_images, depth_buffers)):
        # Retrieve data at current step
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

        # Instantiate a Transform object to make transformation between front bumper and world
        fbumper2world_tfrom = Transform.from_conventional(
            fbumper_location, fbumper_orientation)

        ############ Pole detection (wrt front bumper) ############
        # x-y coordinates assuming z = 0
        poles_xy_z0 = pole_detector.update_poles(pole_image, z=0)
        # Accurate x-y-z coordinates using ground truth depth image
        accurate_detected_pole_xyz = pole_detector.get_pole_xyz_from_depth(
            depth_image, dist_cam_to_fbumper)

        if poles_xy_z0 is not None:
            pole_detection_seq.append(
                [Pole(coord[0], coord[1]) for coord in poles_xy_z0.T])
            # Add accurate detected poles to the container. This is for proximity-based labelling later.
            accurate_detected_in_world = fbumper2world_tfrom.tform_e2w_numpy_array(
                accurate_detected_pole_xyz)
            accurate_pole_detections_in_world_seq.append(
                accurate_detected_in_world)
        else:
            pole_detection_seq.append(None)
            accurate_pole_detections_in_world_seq.append(None)

        ############ Lane detection (wrt front bumper) ############
        left_coeffs, right_coeffs = lane_detector.update_lane_coeffs(
            lane_image, yaw_rates[image_idx])

        # Add color and type properties as part of lane detections
        # The simulated detections are first compared to the ground truth. If the coefficients are close to the ground truth,
        # the recorded marking properties are used as the classification result.
        # It should be noted that due to the lane detection algorithm implementation details, when the ego vehicle makes a lane
        # change, the detected lane markings will switch to the next lane before the front bumper's center actually enters the next lane.
        # That is, there will be some time points when the lane detection returns the next lane already while the recorded ground truth
        # remains in the current lane since it is based on the front bumper's actual location. In this case, detections will be created
        # with "Other" color and "Unknown" type. This is also the case when there is a false positive detection as the coefficients are
        # not consistent with the ground truth.

        # Coeffs from detector are in descending order, while those from ground truth are in ascending order
        if left_coeffs is None:
            # No detection
            left_detection = None
        elif left_marking_gt is None or (abs(left_coeffs[-1] - left_coeffs_gt[0]) > lane_detection_sim_config['c0_thres'] or
                                          abs(left_coeffs[-2] - left_coeffs_gt[1]) > lane_detection_sim_config['c1_thres']):
            # False positive
            left_detection = MELaneMarking(
                left_coeffs, LaneMarkingColor.Other, MELaneMarkingType.Unknown)
        else:
            # True positive. Thresholding doesn't guarantee definite true positive nevertheless.
            left_detection = MELaneMarking.from_lane_marking(
                left_coeffs, left_marking_gt, lane_id)
            # Perturb lane marking type
            left_detection.perturb_type(lane_detection_sim_config['fc_prob'])

        if right_coeffs is None:
            # No detection
            right_detection = None
        elif right_marking_gt is None or (abs(right_coeffs[-1] - right_coeffs_gt[0]) > lane_detection_sim_config['c0_thres'] or
                                          abs(right_coeffs[-2] - right_coeffs_gt[1]) > lane_detection_sim_config['c1_thres']):
            # False positive
            right_detection = MELaneMarking(
                right_coeffs, LaneMarkingColor.Other, MELaneMarkingType.Unknown)
        else:
            # True positive. Thresholding doesn't guarantee definite true positive nevertheless.
            right_detection = MELaneMarking.from_lane_marking(
                right_coeffs, right_marking_gt, lane_id)
            # Perturb lane marking type
            right_detection.perturb_type(lane_detection_sim_config['fc_prob'])

        lane_marking_detection_seq.append(
            MELaneDetection(left_detection, right_detection))

        ############ RS stop sign detection (wrt front bumper) ############
        longi_dist_to_rs_stop = rs_stop_detector.update_rs_stop(
            fbumper_location, fbumper_orientation)
        rs_stop_detecion_seq.append(longi_dist_to_rs_stop)

        ############ Accurate poles for building pole map ############
        # The bases in image are stored internally in the detector
        pole_detector_for_pole_map.find_pole_bases(pole_image)
        accurate_pole_xyz = pole_detector_for_pole_map.get_pole_xyz_from_depth(
            depth_image, dist_cam_to_fbumper)
        if accurate_pole_xyz is not None:
            # Filter out points that are too high to focus on the lower part of poles
            accurate_pole_xyz = accurate_pole_xyz[:, accurate_pole_xyz[2, :] < 0.5]

            # Transform accurate poles to world frame
            accurate_pole_xyz_world = fbumper2world_tfrom.tform_e2w_numpy_array(
                accurate_pole_xyz)

            all_accurate_poles.append(accurate_pole_xyz_world)
        print(image_idx)

    ############ Pole map generation ############
    # Concatenate all poles as an np.array
    all_accurate_poles_xy = np.concatenate(
        all_accurate_poles, axis=1)[0:2, :]  # 2-by-N

    pole_map = gen_pole_map(all_accurate_poles_xy,
                            traffic_signs, pole_map_config)

    ############ Proximity-based pole detecion labelling ############
    if __debug__:
        for poles in accurate_pole_detections_in_world_seq:
            if poles is not None:
                plt.plot(poles[0, :], poles[1, :], 'ro', ms=1)
        plt.title('Accurate Detected Poles')

    # Make a kd-tree out of pole map for later queries
    pole_map_coords = np.asarray([[pole.x, pole.y] for pole in pole_map])
    kd_poles = KDTree(pole_map_coords)

    # Loop over accurate detection sequence and try to associate it to the nearest pole landmark in the pole map
    for detections, accurate_detections in zip(pole_detection_seq, accurate_pole_detections_in_world_seq):
        if accurate_detections is not None:
            for pole_idx, accurate_detection in enumerate(accurate_detections.T):
                nearest_dist, nearest_idx = kd_poles.query(
                    accurate_detection[0:2])
                if nearest_dist < pole_detection_sim_config['max_dist']:
                    detections[pole_idx].type = pole_map[nearest_idx].type
                    detections[pole_idx].perturb_type(
                        pole_detection_sim_config['fc_prob'])

    ############ Save data ############
    # Copy configuration files for future reference
    dst = os.path.join(args.recording_dir, 'settings/vision.yaml')
    copyfile(args.vision_config.name, dst)
    dst = os.path.join(args.recording_dir, 'settings/sim_detection.yaml')
    copyfile(args.sim_detection_config.name, dst)
    dst = os.path.join(args.recording_dir, 'settings/pole_map.yaml')
    copyfile(args.pole_map_config.name, dst)

    # Save detections
    detections = {}
    detections['pole'] = pole_detection_seq
    detections['lane'] = lane_marking_detection_seq
    detections['rs_stop'] = rs_stop_detecion_seq
    with open(os.path.join(args.recording_dir, 'detections.pkl'), 'wb') as f:
        pickle.dump(detections, f)

    # Save pole map
    with open(os.path.join(args.recording_dir, 'pole_map.pkl'), 'wb') as f:
        pickle.dump(pole_map, f)


if __name__ == "__main__":
    main()
