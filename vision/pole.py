import os
import argparse
import yaml
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

import pickle
import matplotlib.pyplot as plt

from camproj import im2world_known_z, im2world_known_x
import utils


class PoleDetector(object):
    """ 
    Class for pole-like objects detection from semantic segmentation images. 

    It takes advantage of the calibration matrices of the camera and projects 2D points in a image
    to the 3D world with known z coordinates. The bases of poles are used since they often stem from 
    the ground plane (z = 0). The obtained world coordinates are wrt the frame at the center of the front bumper.
    This relationship is embedded in the extrinsic parameters already.
    """

    def __init__(self, K, R, x0, pole_config_args):
        """ 
        Constructor method. 

        Input:
            K: Numpy.array of 3x3 matrix for intrinsic calibration matrix.
            R: Numpy.array of 3x3 matrix for rotation matrix of the reference frame wrt the camera frame.
            x0: 3-by-1 Numpy.array representing camera's origin (principal point) wrt the front bumper's frame.
            pole_config_args: Dict object storing algorithm related parameters.
            vp: Array-like representing 2D vanish point in the image. If not given, image center is used.
        """
        self.K = K
        self.R = R
        self.x0 = x0

        # For convenience of 2D-to-3D projection
        self.H = R.T @ np.linalg.inv(K)

        # u-v coordinates of pole bases in image
        self.pole_bases_uv = None
        # x-y coordinates of pole bases wrt the front bumper
        self.poles_xy = None

        # For finding valid bounding boxes around pole-labelled pixels
        # Minimum bound box height (to ignore those too short)
        self._min_height = pole_config_args['min_height']
        # Maximum bound box width (to ignore those too wide)
        self._max_width = pole_config_args['max_width']

    def update_poles(self, pole_image, horizon=None, z=0):
        """
        Update measurement of poles.

        Input: 
            pole_image: OpenCV image with supported data type (e.g. np.uint8). The image should have non-zero values only
                        at pole pixels, which is easy to obtained from a semantic image.
            horizon: Position of the horizon in the image (wrt the top of image). If not given, half point of image is used.
            z: Assumed z coordinates perpendicular to ground of corresponding points
        """
        self._find_pole_bases(pole_image, horizon)
        self._get_pole_xy_fbumper(z)

    def _find_pole_bases(self, pole_image, horizon=None):
        """
        Find bases of poles in the given image.

        This method first finds connected pixels labelled as pole. Then the bottom center of their 
        bound boxes are extracted. Only image below the horizon is searched to avoid poles that are 
        too far. A pole taller than the camera is bound to cross the horizon, so this strategy can 
        find all near poles that are taller than the camera.

        Input: 
            pole_image: OpenCV image with supported data type (e.g. np.uint8). The image should have non-zero values only
                        at pole pixels, which is easy to obtained from a semantic image.
            horizon: Position of the horizon in the image (wrt the top of image). If not given, half point of image is used.
        Output:
            pole_bases_uv: Image coordiantes (u-v) of detected pole bases.
        """
        self.pole_bases_uv = utils.find_pole_bases(
            pole_image, self._max_width, self._min_height, use_bbox_center=False, horizon=horizon)

    def _get_pole_xy_fbumper(self, z=0):
        """
        Get x-y coordinates of poles in the front bumper's frame.

        This method assumes the poles stem from the ground plane (z = 0), then project
        the pole bases in the image to the 3D world. The obtained world coordinates are
        wrt the frame at the center of the front bumper.

        Input:
            pole_bases_uv: Image coordiantes (u-v) of detected pole bases.
            z: Assumed z coordinates perpendicular to ground of corresponding points
        """

        pole_bases_xyz = im2world_known_z(
            self.H, self.x0, self.pole_bases_uv, z_world=z)
        self.pole_bases_xy = pole_bases_xyz[0:2, :]


def single(folder_name, image_idx):
    argparser = argparse.ArgumentParser(
        description='Pole Detection using Semantic Images')
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

    dist_cam_to_fbumper = (config_args['ego_veh']['raxle_to_fbumper']
                           - config_args['sensor']['front_camera']['pos_x']
                           - config_args['ego_veh']['raxle_to_cg'])

    # Load parameters for inverse projection
    calib_data = np.load('vision/calib_data.npz')
    H = calib_data['H']
    K = calib_data['K']
    R = calib_data['R']
    x0 = calib_data['x0']

    # Load data
    mydir = os.path.join('recordings', folder_name)
    with open(os.path.join(mydir, 'ss_images'), 'rb') as image_file:
        ss_images = pickle.load(image_file)
    with open(os.path.join(mydir, 'depth_buffers'), 'rb') as image_file:
        depth_buffers = pickle.load(image_file)

    # Extract pole-relevant semantic labels
    ss_image = ss_images[image_idx]
    pole_image = (ss_image == 5).astype(np.uint8)

    depth_buffer = depth_buffers[image_idx]
    depth_image = utils.decode_depth(depth_buffer)

    pole_detector = PoleDetector(K, R, x0, vision_config_args['pole'])
    pole_detector.update_poles(pole_image, z=0)
    poles_xy_z0 = pole_detector.pole_bases_xy
    pole_detector.update_poles(pole_image, z=0.1)
    poles_xy_z1 = pole_detector.pole_bases_xy

    pole_bases_uv = pole_detector.pole_bases_uv

    x_world = depth_image[pole_bases_uv[1], pole_bases_uv[0]] - dist_cam_to_fbumper
    poles_gt_xyz = im2world_known_x(
        pole_detector.H, pole_detector.x0, pole_detector.pole_bases_uv, x_world)

    # Visualization
    ss_image_copy = ss_image.copy()
    for base_coord in pole_detector.pole_bases_uv.T:
        ss_image_copy = cv2.circle(
            ss_image_copy, (base_coord[0], base_coord[1]), 10, color=[1, 0, 0], thickness=10)

    _, ax = plt.subplots(1, 2)
    ax[0].imshow(ss_image_copy)
    # ax[1].axis('equal')
    ax[1].plot(poles_xy_z0[1, :], poles_xy_z0[0, :], '.', label='z = 0')
    ax[1].plot(poles_xy_z1[1, :], poles_xy_z1[0, :], '.', label='z = 1')
    ax[1].plot(poles_gt_xyz[1, :], poles_gt_xyz[0, :], '.', label='GT')
    ax[1].set_xlim((30, -30))
    ax[1].set_ylim((-5, 60))

    plt.legend()
    plt.show()


if __name__ == "__main__":
    single('depth', 20)
