# Implementations of utilities for carla simuation
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

from enum import Enum
import numpy as np


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


class MELaneMarkingType(Enum):
    """
    Enum that defines the lane marking types according to ME's definition.
    """
    Unknown = 0
    Undecided = 1
    SolidMarker = 2
    DashedMarker = 3
    DoubleLine_Unspecific = 4
    DoubleLine_LeftDashed = 5
    DoubleLine_RightDashed = 6
    DoubleLine_BothSolid = 7
    DoubleLine_BothDashed = 8
    BottsDotts = 9
    RoadEdge = 10
    Barrier = 11


def to_me_lane_marking_type(lane_id, carla_lane_marking_type):
    """
    Convert carla.LaneMarkingType to ME's definition.
    """
    if carla_lane_marking_type == carla.LaneMarkingType.NONE:
        return MELaneMarkingType.Unknown
    elif carla_lane_marking_type == carla.LaneMarkingType.Other:
        return MELaneMarkingType.Unknown
    elif carla_lane_marking_type == carla.LaneMarkingType.Broken:
        return MELaneMarkingType.DashedMarker
    elif carla_lane_marking_type == carla.LaneMarkingType.Solid:
        return MELaneMarkingType.SolidMarker
    elif carla_lane_marking_type == carla.LaneMarkingType.SolidSolid:
        return MELaneMarkingType.DoubleLine_BothSolid
    elif carla_lane_marking_type == carla.LaneMarkingType.SolidBroken:
        if lane_id < 0:
            return MELaneMarkingType.DoubleLine_RightDashed
        else:
            return MELaneMarkingType.DoubleLine_LeftDashed
    elif carla_lane_marking_type == carla.LaneMarkingType.BrokenSolid:
        if lane_id < 0:
            return MELaneMarkingType.DoubleLine_LeftDashed
        else:
            return MELaneMarkingType.DoubleLine_LeftDashed
    elif carla_lane_marking_type == carla.LaneMarkingType.BottsDot:
        return MELaneMarkingType.BottsDotts
    elif carla_lane_marking_type == carla.LaneMarkingType.Grass:
        return MELaneMarkingType.RoadEdge
    elif carla_lane_marking_type == carla.LaneMarkingType.Curb:
        return MELaneMarkingType.RoadEdge


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
        """Creates a pylot LaneMarking from a CARLA lane marking.
        Args:
            lane_marking (:py:class:`carla.LaneMarking`): An instance of a
                CARLA lane marking.
        Returns:
            :py:class:`.LaneMarking`: A pylot lane-marking.
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
    StopOnRoad = 1  # the big STOP sign printed on road surface
    Yield = 2
    SpeedLimit = 3
    TrafficLight = 4


class TrafficSign(object):
    """
    Used to represent a traffic sign in Carla.

    The internal data follows the right-handed z-up coordinate system.
    """

    def __init__(self, traffic_sign_actor, traffic_sign_type):
        """
        Constructor.

        Input:
            traffic_sign_actor: Carla.Actor obj of the traffic sign.
            traffic_sign_type: TrafficSignType obj.
        """
        carla_location = traffic_sign_actor.get_location()
        carla_rotation = traffic_sign_actor.get_transform().rotation
        self.x = carla_location.x
        self.y = - carla_location.y   # convert to right-handed frame
        self.z = carla_location.z
        self.type = traffic_sign_type
        self.yaw = carla_rotation.yaw * np.pi / 180     # to rad

