"""Implementations of utilities for detection"""
from enum import Enum
import random
import math

import numpy as np

from carlasim.utils import TrafficSignType, LaneMarkingType, LaneMarkingColor


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

    def perturb_type(self, fc_prob):
        """
        Perturb the type with the given probability.

        Input:
            fc_prob: Probability of false classification.
        """
        if random.random() < fc_prob:
            while True:
                wrong_type = random.choice(list(TrafficSignType))
                if wrong_type != self.type and wrong_type != TrafficSignType.RSStop:
                    self.type = wrong_type
                    break

    def get_r_phi(self, x_offset=0.0):
        """Get range (r) and azimuth angle (phi) wrt the offset point.

        Mobileye's reference point for measurements is the front bumper.
        An x offset can be given to compute the desired r and phi wrt to, for example,
        the camera. The sign of the offset is defined wrt the front bumper's frame; 
        i.e. x pointing forwards.
        e.g. A camera placed on the windshield should have a negative x_offset.

        Args:
            x_offset (float): Offset in x-axis.
        Returns:
            r: Range of the pole.
            phi: Azimuth angle of the pole.
        """
        corrected_x = self.x - x_offset
        r = math.sqrt(corrected_x**2 + self.y**2)
        phi = math.atan2(self.y, corrected_x)
        return r, phi


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
    Class for mobileye-like lane marking entity.
    """

    def __init__(self, coeffs, marking_color, me_lane_marking_type):
        """
        Constructor.

        Input:
            coeffs: List containing coefficients of the polynomial.
            marking_color: The color of the lane marking.
            me_lane_marking_type: The ME type of the lane marking.
        """
        self.coeffs = coeffs    # in descending order
        self.color = LaneMarkingColor(marking_color)
        self.type = MELaneMarkingType(me_lane_marking_type)

    @classmethod
    def from_lane_marking(cls, coeffs, lane_marking, lane_id):
        """
        Creates an MELaneMarking with a LaneMarking and other information.

        Input:
            coeffs: List containing coefficients of the polynomial.
            lane_marking: An instance of LaneMarking.
            lane_id: ID of the lane the ego vehicle is on.
        Output:
            MELaneMarking object.
        """
        me_lane_marking_type = to_me_lane_marking_type(
            lane_id, lane_marking.type)

        return cls(coeffs, lane_marking.color, me_lane_marking_type)

    def perturb_type(self, fc_prob):
        """
        Perturb lane marking type with uniform distribution.

        Input:
            fc_prob: Probability of false classification.
        """
        # Perturb lane marking type
        if random.random() < fc_prob:
            while True:
                wrong_type = random.choice(list(MELaneMarkingType))
                if wrong_type != self.type and wrong_type != MELaneMarkingType.Unknown:
                    self.type = wrong_type
                    break

    def perturb_type_common(self, fc_prob):
        """
        Perturb lane marking type with commonly seen types.

        Input:
            fc_prob: Probability of false classification.
        """
        # Perturb lane marking type
        if random.random() < fc_prob:
            while True:
                commons = [MELaneMarkingType.DashedMarker,
                           MELaneMarkingType.SolidMarker,
                           MELaneMarkingType.DoubleLine_BothDashed,
                           MELaneMarkingType.DoubleLine_BothSolid,
                           MELaneMarkingType.DoubleLine_LeftDashed,
                           MELaneMarkingType.DoubleLine_RightDashed]
                wrong_type = random.choice(commons)
                if wrong_type != self.type and wrong_type != MELaneMarkingType.Unknown:
                    self.type = wrong_type
                    break

    def get_c0c1_list(self):
        """Extract c0 and c1 coefficients as a list in ascending order."""
        return self.coeffs[-1:-3:-1]

    def compute_y(self, x):
        """Given a list of x coordinates, compute the corresponding y coordinates.

        Args:
            x: The x coordinate(s).
        Returns:
            y: The corresponding y coordinate(s) along the lane marking detection.
        """
        x = np.asarray(x)
        coeffs = np.asarray(self.coeffs)
        y = np.zeros(x.shape)
        for idx, coeff in enumerate(reversed(coeffs)):
            y += coeff * x**idx
        return y


class MELaneDetection(object):
    """
    Class for mobileye-like lane detecion.

    This class wraps the left and right MELaneMarking objects as a whole. 
    """

    def __init__(self, left_me_marking, right_me_marking):
        """
        Constructor.

        Input:
            left_me_marking: An instance of MELaneMarking for left marking.
            right_me_marking: An instance of MELaneMarking for right marking.
        """
        self.left_marking_detection = left_me_marking
        self.right_marking_detection = right_me_marking
