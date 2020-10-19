# Implementations of utilities for detection
import glob
import os
import sys
from enum import Enum
import random

try:
    sys.path.append(glob.glob('./carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

from carlasim.utils import TrafficSignType, LaneMarkingColor
from carlasim.carla_tform import Transform


class Pole(object):
    """
    Class to represent a pole for detection and pole map.

    The internal data follows the right-handed z-up coordinate system.
    """

    def __init__(self, x, y, traffic_sign_type=TrafficSignType.NONE):
        """
        Constructor.

        Input:
            x: x coordinate of pole.
            y: y coordinate of pole.
            traffic_sign_type: TrafficSignType obj.
        """
        self.x = x
        self.y = y
        self.type = traffic_sign_type


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

    Input:
        carla_lane_marking_type: An instance of Carla.LaneMarkingType.
    """
    if carla_lane_marking_type == carla.LaneMarkingType.NONE:
        return MELaneMarkingType.Unknown
    if carla_lane_marking_type == carla.LaneMarkingType.Other:
        return MELaneMarkingType.Unknown
    if carla_lane_marking_type == carla.LaneMarkingType.Broken:
        return MELaneMarkingType.DashedMarker
    if carla_lane_marking_type == carla.LaneMarkingType.Solid:
        return MELaneMarkingType.SolidMarker
    if carla_lane_marking_type == carla.LaneMarkingType.SolidSolid:
        return MELaneMarkingType.DoubleLine_BothSolid
    if carla_lane_marking_type == carla.LaneMarkingType.SolidBroken:
        if lane_id < 0:
            return MELaneMarkingType.DoubleLine_RightDashed
        else:
            return MELaneMarkingType.DoubleLine_LeftDashed
    if carla_lane_marking_type == carla.LaneMarkingType.BrokenSolid:
        if lane_id < 0:
            return MELaneMarkingType.DoubleLine_LeftDashed
        else:
            return MELaneMarkingType.DoubleLine_RightDashed
    if carla_lane_marking_type == carla.LaneMarkingType.BottsDot:
        return MELaneMarkingType.BottsDotts
    if carla_lane_marking_type == carla.LaneMarkingType.Grass:
        return MELaneMarkingType.RoadEdge
    if carla_lane_marking_type == carla.LaneMarkingType.Curb:
        return MELaneMarkingType.RoadEdge


class MELaneMarking(object):
    """
    Class for mobileye-like lane marking.
    """

    def __init__(self, coeffs, marking_color, me_lane_marking_type):
        """
        Constructor.

        Input:
            coeffs: List containing coefficients of the polynomial.
            marking_color: The color of the lane marking.
            me_lane_marking_type: The ME type of the lane marking.
        """
        self.coeffs = coeffs
        self.color = LaneMarkingColor(marking_color)
        self.type = MELaneMarkingType(me_lane_marking_type)

    @classmethod
    def from_carla_lane_marking(cls, coeffs, lane_marking, lane_id, fc_prob):
        """
        Creates an MELaneMarking from a CARLA lane marking.

        Input:
            coeffs: List containing coefficients of the polynomial.
            lane_marking: An instance of a Carla.LaneMarking.
            lane_id: Lane ID the ego vehicle is on.
            fc_prob: false classification probability. If nonzero, the lane type is assigned randomly to a wrong type.
        Output:
            MELaneMarking object.
        """
        me_lane_marking_type = to_me_lane_marking_type(
            lane_id, lane_marking.type)

        # Perturb lane marking type
        if random.random() < fc_prob:
            while True:
                wrong_type = random.choice(list(MELaneMarkingType))
                if wrong_type != me_lane_marking_type:
                    me_lane_marking_type = wrong_type
                    break

        return cls(coeffs, lane_marking.color, me_lane_marking_type)
