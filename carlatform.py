# %%
# The following boilerplate is required if .egg is not installed
# See: https://carla.readthedocs.io/en/latest/build_system/
import glob
import os
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
from scipy.spatial.transform import Rotation

# Carla provides only a transform API from ego to world frame.
# This class complement the transfromation abilities from world to ego frame.

# Note:
# Carla uses z-down coordinate system (left-handed) while we use z-up coordinate system (right-handed).
# This class automatically convert given carla.Transform and carla.Location so the transformation results
# will be in the "ego z-up coordinate system".

class CarlaW2ETform:
    """ Helper class to perform world-to-ego transformation for Carla """

    def __init__(self, ego_transform: carla.Transform):
        """ 
        Constructor method using lazy initialization. 
        
        Input:
            ego_transform: a carla.Transform object of ego frame.
        """
        self._ego_veh_transform = ego_transform
        # rotation matrix to transform a vector from world frame to ego frame
        self._rotm_w2e = None
        # homogeneous transformation matrix to transform a vector from world frame to ego frame
        self._tform_w2e = None

    def rotm_world_to_ego(self, vector3D: carla.Vector3D):
        """ 
        Rotationally transform the given carla.Vector3D in ego frame to world frame.

        Note the return np 3D vector already follows z-up coordinate system.
        
        Input:
            vector3D: a carla.Vector3D which follows carla's coordinate system (z-down system).
        Output:
            np.array representing a 3D coordinate in the z-up coordinate system.
        """
        if self._rotm_w2e is None:
            self._init_rotm_w2e()
        # Need to convert from z-down to z-up coordiante
        np_vec = np.array([vector3D.x, -vector3D.y, -vector3D.z]).T
        return self._rotm_w2e.dot(np_vec)

    def tform_world_to_ego(self, vector3D: carla.Vector3D):
        """ 
        Homogeneous transform the given carla.Vector3D in ego frame to world frame.
        Note the return np 3D vector already follows z-up coordinate system.
        Input:
            vector3D: a carla.Vector3D which follows carla's coordinate system (z-down system).
        Output:
            np.array representing a 3D coordinate in the z-up coordinate system.
        """
        if self._tform_w2e is None:
            self._init_tform_w2e()
        # Need to convert from z-down to z-up coordiante
        np_homo_vec = np.array([vector3D.x, -vector3D.y, -vector3D.z, 1]).T
        return self._tform_w2e.dot(np_homo_vec)[0:3]
        

    def _init_rotm_w2e(self):
        """ Helper method to create rotation matrix _rotm_e2w. """
        rotation = self._ego_veh_transform.rotation
        # Carla uses z-down coordinate system (z towards down)
        # Ref: https://carla.readthedocs.io/en/latest/python_api/#carlarotation
        # Convert to z-up coordinate system when building up the rotation matrix
        self._rotm_w2e = Rotation.from_euler(
            'zyx', [-rotation.yaw, -rotation.pitch, rotation.roll], degrees=True).as_matrix().T

    def _init_tform_w2e(self):
        """ Helper method to create rotation matrix _tform_e2w. """
        if self._rotm_w2e is None:
            self._init_rotm_w2e()
        self._tform_w2e = np.zeros((4, 4), dtype=np.float)
        self._tform_w2e[3, 3] = 1
        self._tform_w2e[0:3, 0:3] = self._rotm_w2e
        trvec = np.array([self._ego_veh_transform.location.x,
                          -self._ego_veh_transform.location.y, 
                          -self._ego_veh_transform.location.z]).T
        self._tform_w2e[0:3, 3] = - self._rotm_w2e.dot(trvec)

# %%
