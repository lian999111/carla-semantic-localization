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

from carlasim.groundtruth import RSStopGTExtractor
from carlasim.carla_tform import Transform
from carlasim.utils import TrafficSignType, get_fbumper_location


class RSStopDetectionSimulator(object):
    """
    Class for road surface stop sign detection simulation.
    """

    def __init__(self, traffic_signs, rs_stop_gt_config, sim_detection_config):
        """
        Constructor.

        Input:
            traffic_signs: List of TrafficSigns.
            rs_stop_gt_config: Dict of configurations for RSStopGTExtractor.
            rs_stop_sim_detection_config: Dict of configurations for simulated detection.
        """
        self.rs_stop_gt_extractor = RSStopGTExtractor(
            traffic_signs, rs_stop_gt_config)

        self._scale = sim_detection_config['scale']
        self._noise_bias = sim_detection_config['noise_bias']
        self._noise_stddev = sim_detection_config['noise_stddev']

        # Placeholder for the detected road surface stop sign
        self.detected_rs_stop_gt = None
        self.longitudinal_dist = None

    def update_rs_stop(self, fbumper_location, fbumper_orientation):
        """
        Update simulated road surface stop sign detection given front bumper's pose. 
        """

        # Get updated gt of road surface stop signs that are likely visible
        # Each column is a 3D coordinate
        visible_rs_stop_signs_gt = self.rs_stop_gt_extractor.update(
            fbumper_location, fbumper_orientation)['visible_rs_stop']

        if visible_rs_stop_signs_gt is None:
            self.detected_rs_stop_gt = None
            self.longitudinal_dist = None
            return self.longitudinal_dist

        elif visible_rs_stop_signs_gt.shape[1] > 1:
            # Pick the nearest one if multiple road surface stop signs are extracted
            chosen_idx = np.argmin(visible_rs_stop_signs_gt[0, :])
            self.detected_rs_stop_gt = visible_rs_stop_signs_gt[:, chosen_idx].squeeze(
            )
            self.longitudinal_dist = self.detected_rs_stop_gt[0]
        elif visible_rs_stop_signs_gt.shape[1] == 1:
            self.detected_rs_stop_gt = visible_rs_stop_signs_gt[:, 0].squeeze()
            self.longitudinal_dist = self.detected_rs_stop_gt[0]

        self._add_noise()
        return self.longitudinal_dist

    def _add_noise(self):
        self.longitudinal_dist *= self._scale
        self.longitudinal_dist += np.random.normal(self._noise_bias,
                                            self._noise_stddev)


def main(folder_name, idx=None):
    argparser = argparse.ArgumentParser(
        description='Stop Sign on Road Detection using Ground Truth')
    argparser.add_argument('config', type=argparse.FileType(
        'r'), help='yaml file for carla configuration')
    argparser.add_argument('sim_detection_config', type=argparse.FileType(
        'r'), help='yaml file for simulated detection configuration')
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
    with args.config as f:
        carla_config = yaml.safe_load(f)
    with args.sim_detection_config as f:
        sim_detection_config = yaml.safe_load(f)

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
    rs_stop_in_img = ax[0].plot([], [], 's')[0]
    rs_stop_gt = ax[1].plot([], [], 's', label='rs stop gt')[0]
    rs_stop_detect = ax[1].plot([], [], 's', ms=3, label='rs stop detction')[0]
    ax[1].set_xlim((30, -30))
    ax[1].set_ylim((-5, 60))
    ax[1].legend()
    plt.show(block=False)

    rs_stop_detector = RSStopDetectionSimulator(
        traffic_signs, sim_detection_config['rs_stop_gt_extractor'], sim_detection_config['rs_stop'])

    if idx is not None:
        raxle_location = raxle_locations[idx]
        raxle_orientation = raxle_orientations[idx]

        # Get front bumper's pose
        fbumper_location = get_fbumper_location(
            raxle_location, raxle_orientation, dist_raxle_to_fbumper)
        fbumper_orientation = raxle_orientation

        # Get detection
        longi_dist = rs_stop_detector.update_rs_stop(
            fbumper_location, fbumper_orientation)
        detected_rs_stop_gt = rs_stop_detector.detected_rs_stop_gt

        ss_image = ss_images[idx]

        if detected_rs_stop_gt is not None:
            # put it on road surface (some actors are buried underground)
            detected_rs_stop_gt[2] = 0
            homo_img_coord = P @ np.append(detected_rs_stop_gt, 1)
            u = homo_img_coord[0] / homo_img_coord[2]
            v = homo_img_coord[1] / homo_img_coord[2]

            rs_stop_in_img.set_data(u, v)
            rs_stop_gt.set_data(detected_rs_stop_gt[1], detected_rs_stop_gt[0])
            rs_stop_detect.set_data(detected_rs_stop_gt[1], longi_dist)
        else:
            rs_stop_in_img.set_data([], [])
            rs_stop_detect.set_data([], [])
        im.set_data(ss_image)

        print(detected_rs_stop_gt)

    else:
        for idx, ss_image in enumerate(ss_images):
            raxle_location = raxle_locations[idx]
            raxle_orientation = raxle_orientations[idx]

            # Get front bumper's pose
            fbumper_location = get_fbumper_location(
                raxle_location, raxle_orientation, dist_raxle_to_fbumper)
            fbumper_orientation = raxle_orientation

            # Get detection
            longi_dist = rs_stop_detector.update_rs_stop(
                fbumper_location, fbumper_orientation)
            detected_rs_stop_gt = rs_stop_detector.detected_rs_stop_gt

            ss_image = ss_images[idx]

            if detected_rs_stop_gt is not None:
                # put it on road surface (some actors are buried underground)
                detected_rs_stop_gt[2] = 0
                homo_img_coord = P @ np.append(detected_rs_stop_gt, 1)
                u = homo_img_coord[0] / homo_img_coord[2]
                v = homo_img_coord[1] / homo_img_coord[2]

                rs_stop_in_img.set_data(u, v)
                rs_stop_gt.set_data(detected_rs_stop_gt[1], detected_rs_stop_gt[0])
                rs_stop_detect.set_data(detected_rs_stop_gt[1], longi_dist)
            else:
                rs_stop_in_img.set_data([], [])
                rs_stop_gt.set_data([], [])
                rs_stop_detect.set_data([], [])
            im.set_data(ss_image)

            print(detected_rs_stop_gt)
            plt.pause(0.001)
    plt.show()


if __name__ == "__main__":
    main('stop_sign')
