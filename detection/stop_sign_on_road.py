# Implementation of road surface stop sign detection

# The following boilerplate is required if .egg is not installed
# See: https://carla.readthedocs.io/en/latest/build_system/
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

import numpy as np
from scipy.spatial import KDTree
import pickle
import argparse
import yaml
import matplotlib.pyplot as plt

from carlasim.carla_tform import CarlaW2ETform
from carlasim.utils import TrafficSignType


class StopSignOnRoadDetector(object):
    """
    Class for road surface stop sign detection.
    """

    def __init__(self, traffic_signs):
        """
        Constructor.

        Input:
            traffic_signs: List of TrafficSigns.
        """
        # A list of TrafficSigns that are stop signs on road
        self.stop_signs_on_road = [
            sign for sign in traffic_signs if sign.type == TrafficSignType.StopOnRoad]
        # A 2D array where each row is a 2D coordinate of a stop sign on road
        self.stop_signs_on_road_coords = np.asarray([
            [sign.x, sign.y, sign.z] for sign in traffic_signs if sign.type == TrafficSignType.StopOnRoad])
        self._kd_coords = KDTree(self.stop_signs_on_road_coords)

        # Placeholder for the stop sign on road that is in sight
        self.in_sight = None

    def update_stop_sign_in_sight(self, fbumper_location, fbumper_orientation):
        """
        Update detection of stop sign on road that is in sight. 
        """
        # Find stop signs on road in the neighborhood
        # TODO: config r
        nearest_idc = self._kd_coords.query_ball_point(
            fbumper_location[0:3], r=30)

        if not nearest_idc:
            self.in_sight = None
            return self.in_sight

        # Pick out coordinates of interest
        coords_of_interest = []
        for idx in nearest_idc:
            # Compare yaw angles. Only pick those with small yaw difference from ego's.
            curr_sign = self.stop_signs_on_road[idx]
            yaw_diff = abs(curr_sign.yaw - fbumper_orientation[2])

            # TODO: config thresholds
            if (yaw_diff < 0.18) or (np.pi-0.18 < yaw_diff < np.pi+0.18) or (yaw_diff > 2*np.pi-0.18):
                coords_of_interest.append(self.stop_signs_on_road_coords[idx])

        # Make it a numpy array where each row is a 3D point
        coords_of_interest = np.asarray(coords_of_interest)

        # Transform the coordinates
        w2e_tform = CarlaW2ETform.from_conventional(
            fbumper_location, fbumper_orientation)
        # Each row is a 3D point in ego frame
        coords_of_interest_in_ego = w2e_tform.tform_w2e_numpy_array(
            coords_of_interest.T).T

        # Filter out signs behind
        # TODO: config threshold
        coords_of_interest_in_ego = coords_of_interest_in_ego[
            coords_of_interest_in_ego[:, 0] > 1.7, :]
        # Filter out signs with large lateral offsets
        # TODO: config threshold
        coords_of_interest_in_ego = coords_of_interest_in_ego[np.abs(
            coords_of_interest_in_ego[:, 1]) < 1.75, :]

        # Pick the farthest one if multiple stops signs still remain
        if coords_of_interest_in_ego.shape[0] > 1:
            chosen_idx = np.argmax(coords_of_interest_in_ego[:, 0])
            self.in_sight = coords_of_interest_in_ego[chosen_idx]
        elif coords_of_interest_in_ego.shape[0] == 1:
            self.in_sight = coords_of_interest_in_ego[0]
        else:
            self.in_sight = None

        return self.in_sight

# TODO: Add noise


def main(folder_name, idx=None):
    argparser = argparse.ArgumentParser(
        description='Stop Sign on Road Detection using Ground Truth')
    argparser.add_argument('config', type=argparse.FileType(
        'r'), help='yaml file for carla configuration')
    args = argparser.parse_args()

    # Load camera calibration parameters
    with open('calib_data.pkl', 'rb') as f:
        calib_data = pickle.load(f)
    K = calib_data['K']
    R = calib_data['R']
    x0 = calib_data['x0']
    # 3-by-4 calibration matrix for visualization
    P = K @ R @ np.concatenate((np.eye(3), -x0), axis=1)

    # Read configurations from yaml file
    with args.config as config_file:
        carla_config = yaml.safe_load(config_file)

    dist_raxle_to_fbumper = carla_config['ego_veh']['raxle_to_fbumper']

    # Load recorded data
    path_to_folder = os.path.join('recordings', folder_name)
    with open(os.path.join(path_to_folder, 'sensor_data.pkl'), 'rb') as f:
        sensor_data = pickle.load(f)
    with open(os.path.join(path_to_folder, 'gt_data.pkl'), 'rb') as f:
        gt_data = pickle.load(f)

    ss_images = sensor_data['semantic_camera']['ss_image']

    traffic_signs = gt_data['static']['traffic_sign']
    raxle_locations = gt_data['seq']['pose']['raxle_location']
    raxle_orientations = gt_data['seq']['pose']['raxle_orientation']

    # Prepare visualization
    _, ax = plt.subplots(1, 2)
    im = ax[0].imshow(np.ones(ss_images[0].shape).astype(
        np.uint8), vmin=0, vmax=12)
    stop_on_road_img = ax[0].plot([], [], 's')[0]
    stop_on_road = ax[1].plot([], [], 's')[0]
    ax[1].set_xlim((30, -30))
    ax[1].set_ylim((-5, 60))
    plt.show(block=False)

    if idx is not None:
        raxle_location = raxle_locations[idx]
        raxle_orientation = raxle_orientations[idx]

        stop_sign_on_road_detector = StopSignOnRoadDetector(traffic_signs)

        in_sight = stop_sign_on_road_detector.update_stop_sign_in_sight(
            raxle_location, raxle_orientation)

        ss_image = ss_images[idx]

        if in_sight is not None:
            in_sight[0] -= dist_raxle_to_fbumper
            in_sight[2] = 0     # put it on road surface (some actors are buried underground)
            homo_img_coord = P @ np.append(in_sight, 1)
            u = homo_img_coord[0] / homo_img_coord[2]
            v = homo_img_coord[1] / homo_img_coord[2]
            
            stop_on_road_img.set_data(u, v)
            stop_on_road.set_data(in_sight[1], in_sight[0])
        else:
            stop_on_road_img.set_data([], [])
        im.set_data(ss_image)

        print(in_sight)
    
    else:
        for idx, ss_image in enumerate(ss_images):
            raxle_location = raxle_locations[idx]
            raxle_orientation = raxle_orientations[idx]

            stop_sign_on_road_detector = StopSignOnRoadDetector(traffic_signs)

            in_sight = stop_sign_on_road_detector.update_stop_sign_in_sight(
                raxle_location, raxle_orientation)

            ss_image = ss_images[idx]

            if in_sight is not None:
                in_sight[0] -= dist_raxle_to_fbumper
                in_sight[2] = 0     # put it on road surface (some actors are buried underground)
                homo_img_coord = P @ np.append(in_sight, 1)
                u = homo_img_coord[0] / homo_img_coord[2]
                v = homo_img_coord[1] / homo_img_coord[2]
                
                stop_on_road_img.set_data(u, v)
                stop_on_road.set_data(in_sight[1], in_sight[0])
            else:
                stop_on_road_img.set_data([], [])
                stop_on_road.set_data([], [])
            im.set_data(ss_image)

            print(in_sight)
            plt.pause(0.001)
    plt.show()


if __name__ == "__main__":
    main('stop_sign2', 50)
