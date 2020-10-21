# Implementations of utilities for detection
from enum import Enum
import random

from carlasim.utils import TrafficSignType, LaneMarkingType, LaneMarkingColor, LaneMarking
from carlasim.carla_tform import Transform


class Pole(object):
    """
    Class to represent a pole for detection and pole map.

    The internal data follows the right-handed z-up coordinate system.
    """

    def __init__(self, x, y, traffic_sign_type=TrafficSignType.Unknown):
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


def to_me_lane_marking_type(lane_id, lane_marking_type):
    """
    Convert LaneMarkingType to MELaneMarkingType.

    Input:
        lane_marking_type: An instance of LaneMarkingType.
    """
    if lane_marking_type == LaneMarkingType.NONE:
        return MELaneMarkingType.Unknown
    if lane_marking_type == LaneMarkingType.Other:
        return MELaneMarkingType.Unknown
    if lane_marking_type == LaneMarkingType.Broken:
        return MELaneMarkingType.DashedMarker
    if lane_marking_type == LaneMarkingType.Solid:
        return MELaneMarkingType.SolidMarker
    if lane_marking_type == LaneMarkingType.SolidSolid:
        return MELaneMarkingType.DoubleLine_BothSolid
    if lane_marking_type == LaneMarkingType.SolidBroken:
        if lane_id < 0:
            return MELaneMarkingType.DoubleLine_RightDashed
        else:
            return MELaneMarkingType.DoubleLine_LeftDashed
    if lane_marking_type == LaneMarkingType.BrokenSolid:
        if lane_id < 0:
            return MELaneMarkingType.DoubleLine_LeftDashed
        else:
            return MELaneMarkingType.DoubleLine_RightDashed
    if lane_marking_type == LaneMarkingType.BottsDot:
        return MELaneMarkingType.BottsDotts
    if lane_marking_type == LaneMarkingType.Grass:
        return MELaneMarkingType.RoadEdge
    if lane_marking_type == LaneMarkingType.Curb:
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
    def from_lane_marking(cls, coeffs, lane_marking, lane_id, fc_prob=0.0):
        """
        Creates an MELaneMarking with a LaneMarking and other information.

        Input:
            coeffs: List containing coefficients of the polynomial.
            lane_marking: An instance of LaneMarking.
            lane_id: ID of the lane the ego vehicle is on.
            fc_prob: False classification probability. If nonzero, the lane type is assigned randomly to a wrong type.
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
