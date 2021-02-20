# Implementations of utilities for carla simuation
import glob
import os
import sys
from enum import Enum

import numpy as np

from carlasim.carla_tform import Transform

try:
    sys.path.append(glob.glob('./carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


class LaneMarkingType(Enum):
    """
    Enum that defines the lane marking types according to Carla's definition.

    The is to make a pickle-able LaneMarkingType type.
    """
    Other = 0
    Broken = 1
    Solid = 2
    SolidSolid = 3
    SolidBroken = 4
    BrokenSolid = 5
    BrokenBroken = 6
    BottsDot = 7
    Grass = 8
    Curb = 9
    NONE = 10


class LaneMarkingColor(Enum):
    """
    Enum that defines the lane marking colors according to Carla's definition.

    The is to make a pickle-able LaneMarkingColor type.
    Experiements show that 0 is White althoutgh it is 4 in official documentation.
    """
    White = 0
    Blue = 1
    Green = 2
    Red = 3
    Yellow = 4
    Other = 5


class LaneMarking(object):
    """
    Used to represent a lane marking.

    This class is to create a pickle-able class for lane marking.
    This class is modified from pylot project.
    """

    def __init__(self, marking_color, marking_type):
        """
        Input:
            marking_color: The color of the lane marking.
            marking_type: The type of the lane marking.
        """
        self.color = LaneMarkingColor(marking_color)
        self.type = LaneMarkingType(marking_type)

    @classmethod
    def from_carla_lane_marking(cls, lane_marking):
        """
        Creates a LaneMarking from a CARLA lane marking.

        Input:
            lane_marking: An instance of a Carla.LaneMarking.
        Output:
            LaneMarking object.
        """
        return cls(lane_marking.color, lane_marking.type)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "LaneMarking(color: {}, type: {})".format(
            self.color, self.type)


class TrafficSignType(Enum):
    """
    Enum that defines the traffic sign types.
    """
    Stop = 0
    RSStop = 1  # Road surface stop sign
    Yield = 2
    SpeedLimit = 3
    TrafficLight = 4
    Unknown = 5


class TrafficSign(object):
    """
    Used to represent a traffic sign in Carla.

    The internal data follows the right-handed z-up coordinate system.
    """

    def __init__(self, traffic_sign_landmark, traffic_sign_type):
        """
        Constructor.

        Input:
            traffic_sign_landmark: Carla.Landmark obj of the traffic sign.
            traffic_sign_type: TrafficSignType obj.
        """
        carla_location = traffic_sign_landmark.transform.location
        carla_rotation = traffic_sign_landmark.transform.rotation
        self.x = carla_location.x
        self.y = - carla_location.y   # convert to right-handed frame
        self.z = carla_location.z
        self.type = traffic_sign_type
        # Convert from z-down to z-up frame
        self.yaw = - carla_rotation.yaw * np.pi / 180     # to rad


def get_fbumper_location(raxle_location, raxle_orientation, dist_raxle_to_fbumper):
    """
    Helper function to get the front bumper's location in right-haned z-up frame given the pose of the rear axle.

    Input:
        location: Array-like (x, y, z) coordinate.
        orientation: Array-like (roll, pitch, yaw) in rad.
    Output:
        1D numpy.array representing a 3D point in the right-handed z-up coordinate system.
    """
    tform = Transform.from_conventional(raxle_location, raxle_orientation)
    fbumper_pt_in_ego = np.array([dist_raxle_to_fbumper, 0, 0])
    # make it 1D
    return tform.tform_e2w_numpy_array(fbumper_pt_in_ego).squeeze()
