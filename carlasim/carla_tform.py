"""Implements world-to-ego transformations."""

# %%
# The following boilerplate is required if .egg is not installed
# See: https://carla.readthedocs.io/en/latest/build_system/
import glob
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

try:
    sys.path.append(glob.glob('./carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


class Transform:
    """ 
    Helper class to perform world-to-ego (w2e) and ego-to-world (e2w) transformations. 

    Carla provides only an API to transform a location from ego to world frame.
    This class provides functionalities to perform both w2e and e2w transfromations.

    Important note:
     Carla (Unreal Engine) uses:
     - Left-handed coordinate system for locations (x-forward, y-rightward, z-up)
     - Right-handed z-down coordinate (airplane like) coordinate system for rotations (roll-pitch-yaw)

    This class automatically converts given carla.Transform and carla.Location so the transformation results
    will always be in the right-handed z-up coordinate system convention.
    """

    def __init__(self, carla_tform: carla.Transform):
        """ 
        Constructor method using lazy initialization. 

        Input:
            carla_tform: Carla.Transform object of ego frame.
        """
        self._carla_tform = carla_tform
        # rotation matrix to rotate a point from world to ego frame
        self._rotm_w2e = None
        # homogeneous transformation matrix to transform a point from world frame to ego frame
        self._tform_w2e = None
        # rotation matrix to rotate a point from ego frame to world frame
        self._rotm_e2w = None
        # homogeneous transformation matrix to transform a point from ego frame to world frame
        self._tform_e2w = None

    @classmethod
    def from_conventional(cls, location, orientation):
        """
        Create a Transform from given location and orientation in conventional right-handed z-up coordinate system.

        The input location and orientation should follow the right-handed z-up coordinate system.

        Input:
            location: Array-like (x, y, z) coordinate.
            orientation: Array-like (roll, pitch, yaw) in rad.
        Output:
            An instance of Transform.
        """
        # Make sure inputs are np.ndarray with right shape
        if isinstance(location, np.ndarray):
            location = location.reshape((-1)).astype(np.float)
        else:
            location = np.array(location, dtype=np.float)

        if isinstance(orientation, np.ndarray):
            orientation = orientation.reshape((-1)).astype(np.float)
        else:
            orientation = np.array(orientation, dtype=np.float)

        carla_location = carla.Location(x=location[0],
                                        y=-location[1],
                                        z=location[2])
        carla_rotation = carla.Rotation(roll=orientation[0]*180/np.pi,
                                        pitch=-orientation[1]*180/np.pi,
                                        yaw=-orientation[2]*180/np.pi)

        return cls(carla.Transform(carla_location, carla_rotation))

    def rot_w2e_carla_vector3D(self, vector3D: carla.Vector3D):
        """ 
        Rotate the given carla.Vector3D from world to ego frame.

        Note the return np 3D vector already follows the right-handed z-up coordinate system.

        Input:
            vector3D: Carla.Vector3D which follows carla's coordinate system (z-down system).
        Output:
            1D Numpy.array representing a 3D point in the right-handed z-up coordinate system.
        """
        if self._rotm_w2e is None:
            self._init_rotm_w2e()
        # Need to convert from left-handed to right-handed before apply the rotation
        np_vec = np.array([vector3D.x, -vector3D.y, vector3D.z])
        return self._rotm_w2e.dot(np_vec)

    def tform_w2e_carla_vector3D(self, vector3D: carla.Vector3D):
        """ 
        Homogeneous transform the given carla.Vector3D from world to ego frame.

        Note the return np 3D vector already follows right-handed z-up coordinate system.

        Input:
            vector3D: Carla.Vector3D which follows carla's coordinate system (z-down system).
        Output:
            1D Numpy.array representing a 3D point in the right-handed z-up coordinate system.
        """
        if self._tform_w2e is None:
            self._init_tform_w2e()
        # Need to convert from left-handed to right-handed before apply the homogeneous transformation
        np_homo_vec = np.array([vector3D.x, -vector3D.y, vector3D.z, 1])
        return self._tform_w2e.dot(np_homo_vec)[0:3]

    def rot_w2e_numpy_array(self, np_points):
        """
        Rotate the given set of numpy points from world to ego frame.

        Note the input should follows the right-handed z-up coordinate system.

        Input:
            np_points: 3-by-N numpy array, where each column is the (x, y, z) 
                       coordinates of a point.
        Returns:
            An 3-by-N numpy array of rotated points in the right-handed z-up coordinate system.
        """
        # Make sure shape is 2D
        if np_points.ndim != 2:
            np_points = np_points.reshape((3, -1))    # 3-by-N
        if self._rotm_w2e is None:
            self._init_rotm_w2e()
        return self._rotm_w2e @ np_points

    def tform_w2e_numpy_array(self, np_points):
        """
        Transform the given set of numpy points from world to ego frame.

        Note the input should follows the right-handed z-up coordinate system.

        Input:
            np_points: 3-by-N numpy array, where each column is the (x, y, z) 
                       coordinates of a point.
        Returns:
            An 3-by-N numpy array of rotated points in the right-handed z-up coordinate system.
        """
        if np_points.ndim != 2:
            np_points = np_points.reshape((3, -1))    # 3-by-N
        if self._tform_w2e is None:
            self._init_tform_w2e()

        # Number of points
        n_pts = np_points.shape[1]

        np_homo_vec = np.concatenate((np_points, np.ones((1, n_pts))), axis=0)
        return (self._tform_w2e @ np_homo_vec)[0:3, :]

    def rot_e2w_carla_vector3D(self, vector3D: carla.Vector3D):
        """ 
        Rotate the given carla.Vector3D from ego to world frame.

        Note the return np 3D vector already follows the right-handed z-up coordinate system.

        Input:
            vector3D: Carla.Vector3D which follows carla's coordinate system (z-down system).
        Output:
            1D Numpy.array representing a 3D point in the right-handed z-up coordinate system.
        """
        if self._rotm_e2w is None:
            self._init_rotm_e2w()
        # Need to convert from left-handed to right-handed before apply the rotation
        np_vec = np.array([vector3D.x, -vector3D.y, vector3D.z])
        return self._rotm_e2w.dot(np_vec)

    def tform_e2w_carla_vector3D(self, vector3D: carla.Vector3D):
        """ 
        Homogeneous transform the given carla.Vector3D from ego to world frame.

        Note the return np 3D vector already follows right-handed z-up coordinate system.

        Input:
            vector3D: Carla.Vector3D which follows carla's coordinate system (z-down system).
        Output:
            1D Numpy.array representing a 3D point in the right-handed z-up coordinate system.
        """
        if self._tform_e2w is None:
            self._init_tform_e2w()
        # Need to convert from left-handed to right-handed before apply the homogeneous transformation
        np_homo_vec = np.array([vector3D.x, -vector3D.y, vector3D.z, 1])
        return self._tform_e2w.dot(np_homo_vec)[0:3]

    def rot_e2w_numpy_array(self, np_points):
        """
        Rotate the given set of numpy points from ego to world frame.

        Note the input should follows the right-handed z-up coordinate system.

        Input:
            np_points: 3-by-N numpy array, where each column is the (x, y, z) 
                       coordinates of a point.
        Returns:
            An 3-by-N numpy array of rotated points in the right-handed z-up coordinate system.
        """
        # Make sure shape is 2D
        if np_points.ndim != 2:
            np_points = np_points.reshape((3, -1))    # 3-by-N
        if self._rotm_e2w is None:
            self._init_rotm_e2w()
        return self._rotm_e2w @ np_points

    def tform_e2w_numpy_array(self, np_points):
        """
        Transform the given set of numpy points from ego to world frame.

        Note the input should follows the right-handed z-up coordinate system.

        Input:
            np_points: 3-by-N numpy array, where each column is the (x, y, z) 
                       coordinates of a point.
        Returns:
            An 3-by-N numpy array of rotated points in the right-handed z-up coordinate system.
        """
        if np_points.ndim != 2:
            np_points = np_points.reshape((3, -1))    # 3-by-N
        if self._tform_e2w is None:
            self._init_tform_e2w()

        # Number of points
        n_pts = np_points.shape[1]

        np_homo_vec = np.concatenate((np_points, np.ones((1, n_pts))), axis=0)
        return (self._tform_e2w @ np_homo_vec)[0:3, :]

    def _init_rotm_w2e(self):
        """ Helper method to create rotation matrix _rotm_w2e. """
        if self._rotm_e2w is not None:
            self._rotm_w2e = self._rotm_e2w.T
            return

        rotation = self._carla_tform.rotation
        # Need to convert from right-handed z-down to right-handed z-up system when building up the rotation matrix
        # Transpose so it is the rotation of the world frame wrt the ego frame
        self._rotm_w2e = Rotation.from_euler(
            'zyx', [-rotation.yaw, -rotation.pitch, rotation.roll], degrees=True).as_matrix().T

    def _init_tform_w2e(self):
        """ Helper method to create homogeneous transform matrix _tform_w2e. """
        if self._rotm_w2e is None:
            self._init_rotm_w2e()
        self._tform_w2e = np.zeros((4, 4), dtype=np.float)
        self._tform_w2e[3, 3] = 1
        self._tform_w2e[0:3, 0:3] = self._rotm_w2e
        trvec = np.array([self._carla_tform.location.x,
                          -self._carla_tform.location.y,
                          self._carla_tform.location.z])
        self._tform_w2e[0:3, 3] = - self._rotm_w2e.dot(trvec)
    
    def _init_rotm_e2w(self):
        """ Helper method to create rotation matrix _rotm_e2w. """
        if self._rotm_w2e is not None:
            self._rotm_e2w = self._rotm_w2e.T
            return

        rotation = self._carla_tform.rotation
        # Need to convert from right-handed z-down to right-handed z-up system when building up the rotation matrix
        # Transpose so it is the rotation of the world frame wrt the ego frame
        self._rotm_e2w = Rotation.from_euler(
            'zyx', [-rotation.yaw, -rotation.pitch, rotation.roll], degrees=True).as_matrix()

    def _init_tform_e2w(self):
        """ Helper method to create homogeneous transform matrix _tform_e2w. """
        if self._rotm_e2w is None:
            self._init_rotm_e2w()
        self._tform_e2w = np.zeros((4, 4), dtype=np.float)
        self._tform_e2w[3, 3] = 1
        self._tform_e2w[0:3, 0:3] = self._rotm_e2w
        self._tform_e2w[0:3, 3] = np.array([self._carla_tform.location.x,
                          -self._carla_tform.location.y,
                          self._carla_tform.location.z])
