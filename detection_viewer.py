# This script implements the visualization of offline detection results

import os
import argparse
import pickle

import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

from detection.vision.utils import convert_semantic_color
from detection.vision.camproj import world2im
from carlasim.utils import Transform, TrafficSignType


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
    argparser.add_argument('recording_dir', type=dir_path,
                           help='directory of recording')
    argparser.add_argument('-i', '--index', dest='image_idx', type=int,
                           default=None,
                           help='specific index to visualization')
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

    # Load camera parameters
    with open('calib_data.pkl', 'rb') as f:
        calib_data = pickle.load(f)
    K = calib_data['K']
    R = calib_data['R']
    x0 = calib_data['x0']
    # 3-by-4 calibration matrix
    P = K @ R @ np.concatenate((np.eye(3), -x0), axis=1)

    # Retrieve required data
    ss_images = sensor_data['semantic_camera']['ss_image']
    raxle_locations = gt_data['seq']['pose']['raxle_location']
    raxle_orientations = gt_data['seq']['pose']['raxle_orientation']

    pole_detection_seq = detections['pole']
    lane_detection_seq = detections['lane']
    rs_stop_detecion_seq = detections['rs_stop']

    # Make a kd-tree out of pole map for later queries
    # make it 3D for later transform
    pole_map_coords = np.asarray([[pole.x, pole.y, 0] for pole in pole_map]).T
    # TODO: Implement 2D tform

    # Init figure
    fig, ax = plt.subplots(1, 2)
    # Camera view
    im = ax[0].imshow(np.ones((ss_images[0].shape[0], ss_images[0].shape[1], 3)).astype(
        np.uint8), vmin=0, vmax=255)
    left_lane = ax[0].plot([], [], ms=0.5)[0]
    right_lane = ax[0].plot([], [], ms=0.5)[0]
    left_lane_type = ax[0].text(20, 580, 'None', fontsize=8)
    right_lane_type = ax[0].text(
        780, 580, 'None', fontsize=8, horizontalalignment='right')

    # Bird's eye view
    mustang = ax[1].add_patch(
        patches.Rectangle((-0.866, -0.814), 1.732, 4.614))
    pole_landmarks = ax[1].plot([], [], '.', ms=2, label='lm')[0]
    pole0 = ax[1].plot([], [], '.', ms=2, label='z=0')[0]
    left_lane_bev = ax[1].plot([], [], linewidth=0.5)[0]
    right_lane_bev = ax[1].plot([], [], linewidth=0.5)[0]
    rs_stop = ax[1].plot([], [], label='rs_stop')[0]
    ax[1].set_xlim((20, -20))
    ax[1].set_ylim((-2, 50))
    ax[1].set_aspect('equal', adjustable='box')
    plt.legend(loc='upper right', fontsize='x-small')
    plt.show(block=False)

    # For lane points
    x = np.linspace(0, 20, 10)

    image_idx = args.image_idx

    if image_idx is not None:
        # Prepare data at current time step
        ss_image = ss_images[image_idx]
        ss_image_copy = convert_semantic_color(ss_image)

        pole_detections = pole_detection_seq[image_idx]
        lane_detection = lane_detection_seq[image_idx]
        rs_stop_detection = rs_stop_detecion_seq[image_idx]

        raxle_location = raxle_locations[image_idx]
        raxle_orientation = raxle_orientations[image_idx]
        raxle_tform = Transform.from_conventional(
            raxle_location, raxle_orientation)

        # Visualize pole landmarks
        pole_map_coords_ego = raxle_tform.tform_w2e_numpy_array(
            pole_map_coords)
        pole_landmarks.set_data(pole_map_coords_ego[1, :],
                                pole_map_coords_ego[0, :])

        # Visualize pole detections
        if pole_detections is not None:
            # Pole detections wrt front bumper
            detections_wrt_fbumper = np.asarray(
                [[pole.x, pole.y, 0] for pole in pole_detections]).T
            bases_in_image = world2im(P, detections_wrt_fbumper)

            # Pole detections wrt rear axle
            detections_wrt_raxle = detections_wrt_fbumper
            detections_wrt_raxle[0, :] += dist_raxle_to_fbumper

            # Visualization
            for pole_idx, base_coord in enumerate(bases_in_image.T):
                if pole_detections[pole_idx].type == TrafficSignType.Unknown:
                    color = [0, 0, 255]
                else:
                    color = [255, 0, 0]
                ss_image_copy = cv2.circle(
                    ss_image_copy, (base_coord[0], base_coord[1]), 10, color=color, thickness=5)

            pole0.set_data(
                detections_wrt_raxle[1, :], detections_wrt_raxle[0, :])
        else:
            pole0.set_data([], [])

        # Visualize lane
        left_marking_detection = lane_detection.left_marking_detection
        right_marking_detection = lane_detection.right_marking_detection

        # Left marking
        if left_marking_detection is not None:
            # x-y is wrt front bumper
            y = np.zeros(x.shape)
            for idx, coeff in enumerate(reversed(left_marking_detection.coeffs)):
                y += coeff * x**idx

            # Project lane marking to image
            homo_img_coords = P @ np.array([x, y,
                                            np.zeros(x.shape), np.ones(x.shape)])
            u = homo_img_coords[0, :] / homo_img_coords[2, :]
            v = homo_img_coords[1, :] / homo_img_coords[2, :]

            left_lane.set_data(u, v)
            left_lane_type.set_text(left_marking_detection.type.name)
            # Must add an offset to make it wrt to rear axle
            left_lane_bev.set_data(y, x + dist_raxle_to_fbumper)
            # Set text color
            if left_marking_detection.type.name == 'Unknown':
                left_lane_type.set_color([1, 0, 0])
            else:
                left_lane_type.set_color([0, 0, 0])
        else:
            left_lane.set_data([], [])
            left_lane_bev.set_data([], [])
            left_lane_type.set_text('None')
            left_lane_type.set_color([0, 0, 0])

        # Right marking
        if right_marking_detection is not None:
            y = np.zeros(x.shape)
            for idx, coeff in enumerate(reversed(right_marking_detection.coeffs)):
                y += coeff * x**idx

            # Project lane marking to image
            homo_img_coords = P @ np.array([x, y,
                                            np.zeros(x.shape), np.ones(x.shape)])
            u = homo_img_coords[0, :] / homo_img_coords[2, :]
            v = homo_img_coords[1, :] / homo_img_coords[2, :]

            right_lane.set_data(u, v)
            right_lane_type.set_text(right_marking_detection.type.name)
            # Must add an offset to make it wrt to rear axle
            right_lane_bev.set_data(y, x + dist_raxle_to_fbumper)
            # Set text color
            if right_marking_detection.type.name == 'Unknown':
                right_lane_type.set_color([1, 0, 0])
            else:
                right_lane_type.set_color([0, 0, 0])
        else:
            right_lane.set_data([], [])
            right_lane_bev.set_data([], [])
            right_lane_type.set_text('None')
            right_lane_type.set_color([0, 0, 0])

        # Visualize rs stop sign
        if rs_stop_detection is not None:
            rs_stop_wrt_raxle = rs_stop_detection + dist_raxle_to_fbumper
            rs_stop.set_data(
                [-1.75, 1.75], [rs_stop_wrt_raxle, rs_stop_wrt_raxle])
        else:
            rs_stop.set_data([], [])

        im.set_data(ss_image_copy)
        ax[1].set_title(image_idx)

        print(image_idx)
    else:
        # Loop over data
        for image_idx, ss_image in enumerate(ss_images):
            # Prepare data at current time step
            ss_image_copy = convert_semantic_color(ss_image)

            pole_detections = pole_detection_seq[image_idx]
            lane_detection = lane_detection_seq[image_idx]
            rs_stop_detection = rs_stop_detecion_seq[image_idx]

            raxle_location = raxle_locations[image_idx]
            raxle_orientation = raxle_orientations[image_idx]
            raxle_tform = Transform.from_conventional(
                raxle_location, raxle_orientation)

            # Visualize pole landmarks
            pole_map_coords_ego = raxle_tform.tform_w2e_numpy_array(
                pole_map_coords)
            pole_landmarks.set_data(pole_map_coords_ego[1, :],
                                    pole_map_coords_ego[0, :])

            # Visualize pole detections
            if pole_detections is not None:
                # Pole detections wrt front bumper
                detections_wrt_fbumper = np.asarray(
                    [[pole.x, pole.y, 0] for pole in pole_detections]).T
                bases_in_image = world2im(P, detections_wrt_fbumper)

                # Pole detections wrt rear axle
                detections_wrt_raxle = detections_wrt_fbumper
                detections_wrt_raxle[0, :] += dist_raxle_to_fbumper

                # Visualization
                for pole_idx, base_coord in enumerate(bases_in_image.T):
                    if pole_detections[pole_idx].type == TrafficSignType.Unknown:
                        color = [50, 50, 50]
                    else:
                        color = [255, 0, 0]
                    ss_image_copy = cv2.circle(
                        ss_image_copy, (base_coord[0], base_coord[1]), 10, color=color, thickness=5)

                pole0.set_data(
                    detections_wrt_raxle[1, :], detections_wrt_raxle[0, :])
            else:
                pole0.set_data([], [])

            # Visualize lane
            left_marking_detection = lane_detection.left_marking_detection
            right_marking_detection = lane_detection.right_marking_detection

            # Left marking
            if left_marking_detection is not None:
                # x-y is wrt front bumper
                y = np.zeros(x.shape)
                for idx, coeff in enumerate(reversed(left_marking_detection.coeffs)):
                    y += coeff * x**idx

                # Project lane marking to image
                homo_img_coords = P @ np.array([x, y,
                                                np.zeros(x.shape), np.ones(x.shape)])
                u = homo_img_coords[0, :] / homo_img_coords[2, :]
                v = homo_img_coords[1, :] / homo_img_coords[2, :]

                left_lane.set_data(u, v)
                left_lane_type.set_text(left_marking_detection.type.name)
                # Must add an offset to make it wrt to rear axle
                left_lane_bev.set_data(y, x + dist_raxle_to_fbumper)
                # Set text color
                if left_marking_detection.type.name == 'Unknown':
                    left_lane_type.set_color([1, 0, 0])
                else:
                    left_lane_type.set_color([0, 0, 0])
            else:
                left_lane.set_data([], [])
                left_lane_bev.set_data([], [])
                left_lane_type.set_text('None')
                left_lane_type.set_color([0, 0, 0])

            # Right marking
            if right_marking_detection is not None:
                y = np.zeros(x.shape)
                for idx, coeff in enumerate(reversed(right_marking_detection.coeffs)):
                    y += coeff * x**idx

                # Project lane marking to image
                homo_img_coords = P @ np.array([x, y,
                                                np.zeros(x.shape), np.ones(x.shape)])
                u = homo_img_coords[0, :] / homo_img_coords[2, :]
                v = homo_img_coords[1, :] / homo_img_coords[2, :]

                right_lane.set_data(u, v)
                right_lane_type.set_text(right_marking_detection.type.name)
                # Must add an offset to make it wrt to rear axle
                right_lane_bev.set_data(y, x + dist_raxle_to_fbumper)
                # Set text color
                if right_marking_detection.type.name == 'Unknown':
                    right_lane_type.set_color([1, 0, 0])
                else:
                    right_lane_type.set_color([0, 0, 0])
            else:
                right_lane.set_data([], [])
                right_lane_bev.set_data([], [])
                right_lane_type.set_text('None')
                right_lane_type.set_color([0, 0, 0])

            # Visualize rs stop sign
            if rs_stop_detection is not None:
                rs_stop_wrt_raxle = rs_stop_detection + dist_raxle_to_fbumper
                rs_stop.set_data(
                    [-1.75, 1.75], [rs_stop_wrt_raxle, rs_stop_wrt_raxle])
            else:
                rs_stop.set_data([], [])

            im.set_data(ss_image_copy)
            ax[1].set_title(image_idx)
            plt.pause(0.001)

            print(image_idx)
    plt.show()


if __name__ == "__main__":
    main()
