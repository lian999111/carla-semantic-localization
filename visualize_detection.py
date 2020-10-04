# Visualize the process of detection using recorded data

import os
import argparse
import yaml
import numpy as np

import pickle
import matplotlib.pyplot as plt
import cv2

from detection.vision.lane import LaneMarkingDetector
from detection.vision.pole import PoleDetector
from detection.vision.camproj import im2world_known_x
from detection.vision.utils import convert_semantic_color, decode_depth

def visualize(folder_name):
    """
    Visualize detection process on recorded data.

    Input:
        folder_name: The folder containing recorded data under the folder "recording".
    """
    argparser = argparse.ArgumentParser(
        description='Visualize detection process on recorded data.')
    argparser.add_argument('config', type=argparse.FileType(
        'r'), help='yaml file for carla configuration')
    argparser.add_argument('vision_params', type=argparse.FileType(
        'r'), help='yaml file for vision algorithm parameters')
    args = argparser.parse_args()

    # Read configurations from yaml file
    with args.config as config_file:
        carla_config = yaml.safe_load(config_file)
    with args.vision_params as vision_params_file:
        vision_params = yaml.safe_load(vision_params_file)

    # For correcting pole ground truth
    dist_cam_to_fbumper = (carla_config['ego_veh']['raxle_to_fbumper']
                           - carla_config['sensor']['front_camera']['pos_x']
                           - carla_config['ego_veh']['raxle_to_cg'])

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

    # Calculate distance from camera to front bumper
    dist_cam_to_fbumper = (carla_config['ego_veh']['raxle_to_fbumper']
                           - carla_config['sensor']['front_camera']['pos_x']
                           - carla_config['ego_veh']['raxle_to_cg'])

    # For lane detector
    # Calculate distance from front bumper to intersection point of FOV and ground surface
    dist_fbumper_to_intersect = dist_cam_to_intersect - dist_cam_to_fbumper

    # Load data
    mydir = os.path.join('recordings', folder_name)
    with open(os.path.join(mydir, 'sensor_data.pkl'), 'rb') as f:
        sensor_data = pickle.load(f)
    with open(os.path.join(mydir, 'gt_data.pkl'), 'rb') as f:
        gt_data = pickle.load(f)

    ss_images = sensor_data['semantic_camera']['ss_image']
    depth_buffers = sensor_data['depth_camera']['depth_buffer']
    yaw_rates = sensor_data['imu']['gyro_z']
    raxle_locations = gt_data['seq']['pose']['raxle_location']
    raxle_orientations = gt_data['seq']['pose']['raxle_orientation']

    # Detector objects
    pole_detector = PoleDetector(K, R, x0, vision_params['pole'])
    lane_detector = LaneMarkingDetector(M, px_per_meter_x, px_per_meter_y,
                                        warped_size, valid_mask,
                                        dist_fbumper_to_intersect,
                                        vision_params['lane'])

    # Init figure
    fig, ax = plt.subplots(1, 2)
    # Camera view
    im = ax[0].imshow(np.ones((ss_images[0].shape[0], ss_images[0].shape[1], 3)).astype(
        np.uint8), vmin=0, vmax=255)
    left_lane = ax[0].plot([], [], ms=0.5)[0]
    right_lane = ax[0].plot([], [], ms=0.5)[0]

    # Bird's eye view
    pole0 = ax[1].plot([], [], '.', label='z = 0')[0]
    pole_gt = ax[1].plot([], [], '.', label='GT')[0]
    left_lane_bev = ax[1].plot([], [], linewidth=0.5)[0]
    right_lane_bev = ax[1].plot([], [], linewidth=0.5)[0]
    ax[1].set_xlim((30, -30))
    ax[1].set_ylim((0, 50))
    plt.legend()
    plt.show(block=False)

    # For lane points
    x = np.linspace(0, 20, 10)

    # Loop over data
    for image_idx, (ss_image, depth_buffer) in enumerate(zip(ss_images, depth_buffers)):
        # Prepare images
        ss_image_copy = convert_semantic_color(ss_image)
        pole_image = (ss_image == 5).astype(np.uint8)
        lane_image = (ss_image == 6) | (ss_image == 8).astype(np.uint8)

        # Tried multi-threading here, the overhead had worsen the speed
        # Pole detection
        poles_xy_z0 = pole_detector.update_poles(pole_image, z=0)

        # Lane detection
        left_coeffs, right_coeffs = lane_detector.update_lane_coeffs(
            lane_image, yaw_rates[image_idx])

        pole_bases_uv = pole_detector.pole_bases_uv

        # Visualize poles
        if poles_xy_z0 is not None:
            # Ground truth
            depth_image = decode_depth(depth_buffer)
            x_world = depth_image[pole_bases_uv[1],
                                  pole_bases_uv[0]] - dist_cam_to_fbumper
            poles_gt_xyz = im2world_known_x(
                pole_detector.H, pole_detector.x0, pole_detector.pole_bases_uv, x_world)

            # Visualization
            for base_coord in pole_detector.pole_bases_uv.T:
                ss_image_copy = cv2.circle(
                    ss_image_copy, (base_coord[0], base_coord[1]), 10, color=[12, 0, 0], thickness=5)

            pole0.set_data(poles_xy_z0[1, :], poles_xy_z0[0, :])
            pole_gt.set_data(poles_gt_xyz[1, :], poles_gt_xyz[0, :])
        else:
            pole0.set_data([], [])
            pole_gt.set_data([], [])

        # Visualize lane
        # Left marking
        if left_coeffs is not None:
            y = np.zeros(x.shape)
            for idx, coeff in enumerate(reversed(left_coeffs)):
                y += coeff * x**idx

            # Project lane marking to image
            homo_img_coords = P @ np.array([x, y,
                                            np.zeros(x.shape), np.ones(x.shape)])
            u = homo_img_coords[0, :] / homo_img_coords[2, :]
            v = homo_img_coords[1, :] / homo_img_coords[2, :]

            left_lane.set_data(u, v)
            left_lane_bev.set_data(y, x)
        else:
            left_lane.set_data([], [])
            left_lane_bev.set_data([], [])

        # Right marking
        if right_coeffs is not None:
            y = np.zeros(x.shape)
            for idx, coeff in enumerate(reversed(right_coeffs)):
                y += coeff * x**idx

            # Project lane marking to image
            homo_img_coords = P @ np.array([x, y,
                                            np.zeros(x.shape), np.ones(x.shape)])
            u = homo_img_coords[0, :] / homo_img_coords[2, :]
            v = homo_img_coords[1, :] / homo_img_coords[2, :]

            right_lane.set_data(u, v)
            right_lane_bev.set_data(y, x)
        else:
            right_lane.set_data([], [])
            right_lane_bev.set_data([], [])

        im.set_data(ss_image_copy)
        ax[1].set_title(image_idx)
        plt.pause(0.001)

        print(image_idx)


if __name__ == "__main__":
    visualize('town03_highway')
