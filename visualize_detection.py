# Visualize the process of detection using recorded data

import os
import argparse
import yaml
import numpy as np

import pickle
import matplotlib.pyplot as plt
import cv2

from vision.lane import LaneMarkingDetector
from vision.pole import PoleDetector
from vision.camproj import im2world_known_x
from vision import vutils


def visualize(folder_name):
    """
    Visualize detection process on recorded data.

    Input:
        folder_name: The folder containing recorded data under the folder "recording".
    """
    argparser = argparse.ArgumentParser(
        description='Visualize detection process on recorded data.')
    argparser.add_argument('config', type=argparse.FileType(
        'r'), help='configuration yaml file for carla env setup')
    argparser.add_argument('vision_config', type=argparse.FileType(
        'r'), help='configuration yaml file for vision algorithms')
    args = argparser.parse_args()

    # Read configurations from yaml file
    with args.config as config_file:
        config_args = yaml.safe_load(config_file)
    with args.vision_config as vision_config_file:
        vision_config_args = yaml.safe_load(vision_config_file)

    # For correcting pole ground truth
    dist_cam_to_fbumper = (config_args['ego_veh']['raxle_to_fbumper']
                           - config_args['sensor']['front_camera']['pos_x']
                           - config_args['ego_veh']['raxle_to_cg'])

    # Load camera parameters for pole detection
    calib_data = np.load('vision/calib_data.npz')
    K = calib_data['K']
    R = calib_data['R']
    x0 = calib_data['x0']
    # 3-by-4 calibration matrix
    P = K @ R @ np.concatenate((np.eye(3), -x0), axis=1)

    # Load bird's eye view projection parameters for lane detection
    perspective_tform_data = np.load('vision/ipm_data.npz')
    M = perspective_tform_data['M']
    warped_size = tuple(perspective_tform_data['bev_size'])
    valid_mask = perspective_tform_data['valid_mask']
    px_per_meter_x = float(perspective_tform_data['px_per_meter_x'])
    px_per_meter_y = float(perspective_tform_data['px_per_meter_y'])
    dist_cam_to_intersect = float(perspective_tform_data['dist_to_intersect'])

    # For lane detector
    dist_fbumper_to_intersect = dist_cam_to_intersect - dist_cam_to_fbumper

    # Load data
    mydir = os.path.join('recordings', folder_name)
    with open(os.path.join(mydir, 'ss_images'), 'rb') as image_file:
        ss_images = pickle.load(image_file)
    with open(os.path.join(mydir, 'depth_buffers'), 'rb') as image_file:
        depth_buffers = pickle.load(image_file)
    with open(os.path.join(mydir, 'yaw_rate'), 'rb') as yaw_rate_file:
        yaw_rates = pickle.load(yaw_rate_file)

    # Detector objects
    pole_detector = PoleDetector(K, R, x0, vision_config_args['pole'])
    lane_detector = LaneMarkingDetector(M, px_per_meter_x, px_per_meter_y,
                                        warped_size, valid_mask,
                                        dist_fbumper_to_intersect,
                                        vision_config_args['lane'])

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
        ss_image_copy = vutils.convert_semantic_color(ss_image)
        pole_image = (ss_image == 5).astype(np.uint8)
        lane_image = (ss_image == 6) | (ss_image == 8).astype(np.uint8)
        depth_image = vutils.decode_depth(depth_buffer)

        # Pole detection
        pole_detector.update_poles(pole_image, z=0)

        # Lane detection
        lane_detector.update_lane_coeffs(lane_image, yaw_rates[image_idx])

        pole_bases_uv = pole_detector.pole_bases_uv

        # Visualize poles
        if pole_bases_uv is not None:
            poles_xy_z0 = pole_detector.pole_bases_xy
            
            # Ground truth
            x_world = depth_image[pole_bases_uv[1],
                                  pole_bases_uv[0]] - dist_cam_to_fbumper
            poles_gt_xyz = im2world_known_x(
                pole_detector.H, pole_detector.x0, pole_detector.pole_bases_uv, x_world)

            # Visualization
            for base_coord in pole_detector.pole_bases_uv.T:
                ss_image_copy = cv2.circle(
                    ss_image_copy, (base_coord[0], base_coord[1]), 10, color=[12, 0, 0], thickness=10)

            pole0.set_data(poles_xy_z0[1, :], poles_xy_z0[0, :])
            pole_gt.set_data(poles_gt_xyz[1, :], poles_gt_xyz[0, :])
        else:
            pole0.set_data([], [])
            pole_gt.set_data([], [])

        # Visualize lane
        # Left marking
        if lane_detector.left_coeffs is not None:
            coeffs = lane_detector.left_coeffs
            y = np.zeros(x.shape)
            for idx, coeff in enumerate(reversed(coeffs)):
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
        if lane_detector.right_coeffs is not None:
            coeffs = lane_detector.right_coeffs
            y = np.zeros(x.shape)
            for idx, coeff in enumerate(reversed(coeffs)):
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


if __name__ == "__main__":
    visualize('true_highway')
