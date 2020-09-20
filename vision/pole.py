import os
import argparse
import yaml
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

import pickle
import matplotlib.pyplot as plt

from camproj import im2world_known_z
import utils


class PoleDetector(object):
    """ 
    Class for pole-like objects detection from semantic segmentation images. 

    It takes advantage of the calibration matrices of the camera and projects 2D points in a image
    to the 3D world with known z coordinates. The bases of poles are used since they often stem from 
    the ground plane (z = 0). The obtained world coordinates are wrt the frame at the center of the front bumper.
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

        # x-y coordinates of poles wrt the front bumper
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
        pole_bases_uv = self._find_pole_bases(pole_image, horizon)
        self.poles_xy = self._get_pole_xy_fbumper(pole_bases_uv, z)

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
        pole_bases_uv = utils.find_pole_bases(pole_image, self._max_width, self._min_height, horizon=horizon)
        
        return pole_bases_uv

    def _get_pole_xy_fbumper(self, pole_bases_uv, z=0):
        """
        Get x-y coordinates of poles in the front bumper's frame.

        This method assumes the poles stem from the ground plane (z = 0), then project
        the pole bases in the image to the 3D world. The obtained world coordinates are
        wrt the frame at the center of the front bumper.

        Input:
            pole_bases_uv: Image coordiantes (u-v) of detected pole bases.
            z: Assumed z coordinates perpendicular to ground of corresponding points
        Output:
            pole_xy: x-y coordinates of pole bases wrt the front bumper.
        """

        pole_bases_xyz = im2world_known_z(self.H, self.x0, pole_bases_uv, z_world=z)
        pole_bases_xy = pole_bases_xyz[0:2, :]

        return pole_bases_xy


def single(folder_name, image_idx):
    argparser = argparse.ArgumentParser(
        description='Lane Detection using Semantic Images')
    argparser.add_argument('vision_config', type=argparse.FileType(
        'r'), help='configuration yaml file for vision algorithms')
    args = argparser.parse_args()

    # Read configurations from yaml file
    with args.vision_config as vision_config_file:
        vision_config_args = yaml.safe_load(vision_config_file)

    # Load parameters for inverse projection
    calib_data = np.load('vision/calib_data.npz')
    K = calib_data['K']
    R = calib_data['R']
    x0 = calib_data['x0']

    # Load data
    mydir = os.path.join('recordings', folder_name)
    with open(os.path.join(mydir, 'ss_images'), 'rb') as image_file:
        ss_images = pickle.load(image_file)

    # Extract pole-relevant semantic labels
    ss_image = ss_images[image_idx]
    pole_image = (ss_image == 5).astype(np.uint8)

    pole_detector = PoleDetector(K, R, x0, vision_config_args['pole'])

    pole_detector.update_poles(pole_image, z=0)
    poles_xy_z0 = pole_detector.poles_xy
    pole_detector.update_poles(pole_image, z=0.1)
    poles_xy_z1 = pole_detector.poles_xy

    # Visualization
    pole_bases_uv = pole_detector._find_pole_bases(pole_image)
    ss_image_copy = ss_image.copy()
    for base_coord in pole_bases_uv.T:
        ss_image_copy = cv2.circle(
            ss_image_copy, (base_coord[0], base_coord[1]), 10, color=[1, 0, 0], thickness=10)

    _, ax = plt.subplots(1, 2)
    ax[0].imshow(ss_image_copy)
    # ax[1].axis('equal')
    ax[1].plot(poles_xy_z0[1, :], poles_xy_z0[0, :], '.')
    ax[1].plot(poles_xy_z1[1, :], poles_xy_z1[0, :], '.')
    ax[1].set_xlim((20, -20))
    ax[1].set_ylim((-10, 60))

    plt.show()


if __name__ == "__main__":
    single('highway', 20)