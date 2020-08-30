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
from carlatform import CarlaW2ETform


class Direction(Enum):
    """ Enum for specifying lane searching direction """
    Left = 1
    Right = 2


class GroundTruthExtractor(object):
    """ Class for ground truth extraction """

    def __init__(self, ego_veh, carla_map, actor_list, config_args):
        """ Constructor method """
        # Ego vehicle
        # Distance from rear axle to front bumper
        self.raxle_to_fbumper = config_args['ego_veh']['raxle_to_fbumper']
        # Distance from rear axle to center of gravity
        self.raxle_to_cg = config_args['ego_veh']['raxle_to_cg']

        self.ego_veh = ego_veh
        self.ego_veh_tform = ego_veh.get_transform()
        
        # Front bumper location in Carla's coordinate system (z-down) as a carla.Vector3D object
        # It in carla's z-down world frame so querying waypoints using carla's APIs is more straightforward
        self._fbumper_location = self.ego_veh_tform.transform(
            carla.Location(x=self.raxle_to_fbumper - self.raxle_to_cg))
        
        # Rear axle in Carla's coordinate system (z-down) as a carla.Vector3D object
        raxle_location = self.ego_veh_tform.transform(
            carla.Location(x=-self.raxle_to_cg))
        # Rear axle in our coordinate system (z-up) as a numpy array
        self.raxle_gt_location = np.array([raxle_location.x,
                                           -raxle_location.y,
                                           -raxle_location.z])

        # Simulation environment
        self.map = carla_map
        self.actor_list = actor_list
        # Ignore landmarks now since carla built-in maps don't have them defined
        # self.landmarks = None

        # Lanes
        # Search radius
        self._radius = config_args['gt']['lane']['radius']
        # In some OpenDrive definitions, ego lane may have no visible lane boundaries.
        # Ground truth extractor then tries to get closest visible lane markings towards both sides and store the
        # corresponding waypoints into waypoint_left_marking and waypoint_right_marking. These 2 waypoints
        # are just for the convenience to extract closes left and right markings.
        self.waypoint = None
        self.waypoint_left_marking = None
        self.waypoint_right_marking = None
        self.waypoint_next_left_marking = None
        self.waypoint_next_right_marking = None

        self.left_marking = None
        self.next_left_marking = None
        self.right_marking = None
        self.next_right_marking = None
        # c0 and c1 of lane markings
        self.left_marking_param = [0, 0]
        self.next_left_marking_param = [0, 0]
        self.right_marking_param = [0, 0]
        self.next_right_marking_param = [0, 0]

    def update(self):
        """ Update ground truth at the current tick """
        self.ego_veh_tform = self.ego_veh.get_transform()
        # carla.Location.transform() returns just a carla.Vector3D object
        self._fbumper_location = carla.Location(self.ego_veh_tform.transform(
            carla.Location(x=self.raxle_to_fbumper - 1.4)))
        self._raxle_location = self.ego_veh_tform.transform(
            carla.Location(x=-self.raxle_to_cg))

        # Find a waypoint on the nearest lane (any lane type except NONE)
        # So when ego vehicle is driving abnormally (e.g. on shoulder or parking), lane markings can still be obtained.
        # Some strange results may happen in extreme cases though (e.g. car drives on rail or sidewalk).
        self.waypoint = self.map.get_waypoint(
            self._fbumper_location, lane_type=carla.LaneType.Any)

        # When the query point (front bumper) is farther from the obtained waypoint than its half lane width,
        # the ego vehicle is likely  to be off road, then no lane info is further extracted.
        if self._fbumper_location.distance(self.waypoint.transform.location) >= self.waypoint.lane_width/2:
            self.waypoint = None

        if self.waypoint is not None:
            # Find left and right markings of ego lane
            self._get_left_lane_marking()
            self._get_right_lane_marking()

            # Find next lane markings unless a curb is aleady by the ego lane
            # Next left
            if self.left_marking is not None and self.left_marking.type != carla.LaneMarkingType.Curb:
                self._get_next_left_lane_marking()
            else:
                self.waypoint_next_left_marking = None
                self.next_left_marking = None

            # Next right
            if self.right_marking is not None and self.right_marking.type != carla.LaneMarkingType.Curb:
                self._get_next_right_lane_marking()
            else:
                self.waypoint_next_right_marking = None
                self.next_right_marking = None

            # TODO: lane marking parameters
        else:
            self.waypoint_left_marking = None
            self.waypoint_right_marking = None
            self.waypoint_next_left_marking = None
            self.waypoint_next_right_marking = None
            self.left_marking = None
            self.right_marking = None
            self.next_left_marking = None
            self.next_right_marking = None

    def _get_left_lane_marking(self):
        """
        Get left visible lane marking.
        """
        left_waypt = self._find_visible_lane_marking(direction=Direction.Left)

        # Updatee left lane marking
        if left_waypt is not None:
            self.waypoint_left_marking = left_waypt
            self.left_marking = self._get_lane_marking(
                left_waypt, direction=Direction.Left)
        else:
            self.waypoint_left_marking = None
            self.left_marking = None

    def _get_right_lane_marking(self):
        """
        Get right visible lane marking.
        """
        right_waypt = self._find_visible_lane_marking(direction=Direction.Right)

        # Updatee right lane marking
        if right_waypt is not None:
            self.waypoint_right_marking = right_waypt
            self.right_marking = self._get_lane_marking(
                right_waypt, direction=Direction.Right)
        else:
            self.waypoint_right_marking = None
            self.right_marking = None

    def _find_visible_lane_marking(self, direction):
        """
        Helper method for finding the waypoint with visible lane marking.
        """
        # Use original ego waypoint if it already has visible marking
        if direction == Direction.Left:
            if self.waypoint.left_lane_marking.type != carla.LaneMarkingType.NONE:
                return self.waypoint
        else:
            if self.waypoint.right_lane_marking.type != carla.LaneMarkingType.NONE:
                return self.waypoint

        # No visible marking in the direction of interest, start searching
        curr_waypt = self.waypoint

        # Search until visible lane marking found
        while True:
            curr_waypt = self._get_next_lane(curr_waypt, direction)

            if curr_waypt is None:
                return None

            # Do not search across non-drivable space (e.g. across a middle island)
            # If so, set left lane as None
            # Not sure if this strategy is realistic
            if (curr_waypt.lane_type == carla.LaneType.Median
                    and curr_waypt.lane_type == carla.LaneType.Sidewalk
                    and curr_waypt.lane_type == carla.LaneType.Rail):
                return None

            if self._get_lane_marking(curr_waypt, direction).type != carla.LaneMarkingType.NONE:
                # Found
                return curr_waypt

    def _get_next_left_lane_marking(self):
        """
        Get next left visible lane marking.
        Many transition lanes often without visible lane boundaries (e.g. Shoulder) are defined in OpenDrive.
        This method tries to find the lanes corresponding to visible lane boundaries (e.g. curb) but may not 
        be directly adjacent to current lane.
        """
        next_left_waypt = self._find_next_visible_lane_marking(direction=Direction.Left)

        # Updatee next left lane marking
        if next_left_waypt is not None:
            self.waypoint_next_left_marking = next_left_waypt
            self.next_left_marking = self._get_lane_marking(
                next_left_waypt, direction=Direction.Left)
        else:
            self.waypoint_next_left_marking = None
            self.next_left_marking = None

    def _get_next_right_lane_marking(self):
        """
        Get next right visible lane marking.
        Many transition lanes often without visible lane boundaries (e.g. Shoulder) are defined in OpenDrive.
        This method tries to find the lanes corresponding to visible lane boundaries (e.g. curb) but may not 
        be directly adjacent to current lane.
        """
        next_right_waypt = self._find_next_visible_lane_marking(direction=Direction.Right)

        # Update next right lane marking
        if next_right_waypt is not None:
            self.waypoint_next_right_marking = next_right_waypt
            self.next_right_marking = self._get_lane_marking(
                next_right_waypt, direction=Direction.Right)
        else:
            self.waypoint_next_right_marking = None
            self.next_right_marking = None

    def _find_next_visible_lane_marking(self, direction):
        """
        Helper method for finding the waypoint with the visible next lane marking.
        """
        # Initialize curr_next_waypt with the waypoint in the corresponding direction
        if direction == Direction.Left:
            curr_next_waypt = self.waypoint_left_marking
        else:
            curr_next_waypt = self.waypoint_right_marking

        # If the starting waypoint to search is aleady None, return None
        if curr_next_waypt is None:
            return None

        # Search until visible lane marking found
        while True:
            # Go to the next lane
            curr_next_waypt = self._get_next_lane(curr_next_waypt, direction)

            if curr_next_waypt is None:
                return None

            # Do not search across non-drivable space (e.g. across a middle island)
            # If so, set left lane as None
            # Not sure if this strategy is realistic
            if (curr_next_waypt.lane_type == carla.LaneType.Median
                    or curr_next_waypt.lane_type == carla.LaneType.Sidewalk
                    or curr_next_waypt.lane_type == carla.LaneType.Rail):
                return None

            if self._get_lane_marking(curr_next_waypt, direction).type != carla.LaneMarkingType.NONE:
                # Found
                return curr_next_waypt

    def _check_same_direction_as_ego_lane(self, waypoint_of_interest):
        """ Check if the direction of the waypoint of interest is the same as the ego lane """
        if waypoint_of_interest is None:
            return None
        return (self.waypoint.lane_id * waypoint_of_interest.lane_id) > 0

    def _get_next_lane(self, waypoint_of_interest, direction):
        """ Get waypoint of next lane in specified direction with respect to ego lane """
        if waypoint_of_interest is None:
            return None
        if direction == Direction.Left:
            if self._check_same_direction_as_ego_lane(waypoint_of_interest):
                return waypoint_of_interest.get_left_lane()
            else:
                return waypoint_of_interest.get_right_lane()
        else:
            if self._check_same_direction_as_ego_lane(waypoint_of_interest):
                return waypoint_of_interest.get_right_lane()
            else:
                return waypoint_of_interest.get_left_lane()

    def _get_lane_marking(self, waypoint_of_interest, direction):
        """ Get lane marking of given waypoint in the specified direction with respect to ego lane """
        if waypoint_of_interest is None:
            return None
        if direction == Direction.Left:
            if self._check_same_direction_as_ego_lane(waypoint_of_interest):
                return waypoint_of_interest.left_lane_marking
            else:
                return waypoint_of_interest.right_lane_marking
        else:
            if self._check_same_direction_as_ego_lane(waypoint_of_interest):
                return waypoint_of_interest.right_lane_marking
            else:
                return waypoint_of_interest.left_lane_marking

    def _get_poles(self):
        # TODO: use semantic lidar or just actors?
        pass
