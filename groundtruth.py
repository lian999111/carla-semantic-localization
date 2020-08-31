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
    Forward = 3
    Backward = 4


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
        # A flag indicating ego vehicle is in junction
        self.in_junction = False

        # Rear axle in Carla's coordinate system (z-down) as a carla.Vector3D object
        raxle_location = self.ego_veh_tform.transform(
            carla.Location(x=-self.raxle_to_cg))
        # Rear axle's location in our coordinate system (z-up) as a numpy array
        # This is the ground truth of the rear axle's location
        self.raxle_gt_location = np.array([raxle_location.x,
                                           -raxle_location.y,
                                           -raxle_location.z])
        # Rear axle's orientation in our coordinate system (z-up) as a numpy array (roll, pitch, yaw)
        # This is the ground truth of the rear axle's orientation
        self.raxle_gt_orientation = np.array([self.ego_veh_tform.rotation.roll,
                                              -self.ego_veh_tform.rotation.pitch,
                                              -self.ego_veh_tform.rotation.yaw])

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

        # Update front bumper (carla.Vector3D in z-down frame)
        self._fbumper_location = self.ego_veh_tform.transform(
            carla.Location(x=self.raxle_to_fbumper - self.raxle_to_cg))  # carla.Location.transform() returns just a carla.Vector3D object

        # Update rear axle (np.array in z-up frame)
        raxle_location = self.ego_veh_tform.transform(
            carla.Location(x=-self.raxle_to_cg))
        self.raxle_gt_location = np.array([raxle_location.x,
                                           -raxle_location.y,
                                           -raxle_location.z])
        self.raxle_gt_orientation = np.array([self.ego_veh_tform.rotation.roll,
                                              -self.ego_veh_tform.rotation.pitch,
                                              -self.ego_veh_tform.rotation.yaw])

        # TODO: modify comment to Drivee
        # Find a waypoint on the nearest lane (any lane type except NONE)
        # So when ego vehicle is driving abnormally (e.g. on shoulder or parking), lane markings can still be obtained.
        # Some strange results may happen in extreme cases though (e.g. car drives on rail or sidewalk).
        self.waypoint = self.map.get_waypoint(
            self._fbumper_location, lane_type=carla.LaneType.Driving)

        # Update in_junction flag if current waypoint is junction
        self.in_junction = self.waypoint.is_junction
            

        # When the query point (front bumper) is farther from the obtained waypoint than its half lane width,
        # the ego vehicle is likely  to be off road, then no lane info is further extracted.
        # make a carla.Location object
        fbumper_loc = carla.Location(self._fbumper_location)
        if fbumper_loc.distance(self.waypoint.transform.location) >= self.waypoint.lane_width/2:
            self.waypoint = None

        if self.waypoint is not None:
            # Find candidate visible markings within the specified radius
            # We get a list of 3D points of candidate markings in ego vehicle's frame (z-up)
            # as well as a list containing the corresponding marking types.
            candidate_markings_in_ego, candidate_marking_types = self._find_candidate_markings()
            
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

    def _find_candidate_markings(self):
        """ 
        Find candidate lane markings withing a radius at the current time step.
        Return a list of 3D points of candidate markings and a list of their corresponding marking type.
        """
        fbumper_transform = carla.Transform(
            self._fbumper_location, self.ego_veh.get_transform().rotation)
        # Object for transforming a carla.Location in carla's world frame (z-down) into our ego vehicle's frame (z-up)
        tform_w2e = CarlaW2ETform(fbumper_transform)
        waypt_ego_frame = tform_w2e.tform_world_to_ego(
            self.waypoint.transform.location)

        # Container lists
        # A list of points of each candidate marking
        # Each candidate marking has a list of 3D points in ego vehicle's frame (z-up)
        candidate_markings_in_ego = []
        # A list of type of each candidate marking point
        candidate_marking_types = []

        # Left
        # Initilization with current waypoint
        left_marking_waypt = self.waypoint
        cum_dist = waypt_ego_frame[1] + \
            0.5 * left_marking_waypt.lane_width
        # Search left till cumulative distance exceeds radius
        while cum_dist < self._radius:
            # Get left marking of current waypoint
            left_marking = self._get_lane_marking(
                left_marking_waypt, Direction.Left)
            if left_marking.type != carla.LaneMarkingType.NONE:
                # A candidate found
                marking_pts, marking_types = self._get_marking_pts(
                    left_marking_waypt, tform_w2e, Direction.Left)

                candidate_markings_in_ego.append(marking_pts)
                candidate_marking_types.append(marking_types)

            # Stop when reaching a curb
            if left_marking.type == carla.LaneMarkingType.Curb:
                break
            # Go to the next left lane
            left_marking_waypt = self._get_next_lane(
                left_marking_waypt, Direction.Left)
            if left_marking_waypt is not None:
                cum_dist += left_marking_waypt.lane_width
            else:
                # Stop when next lane waypoint is None
                break

        # Right
        # Initilization with current waypoint
        right_marking_waypt = self.waypoint
        cum_dist = waypt_ego_frame[1] + \
            0.5 * right_marking_waypt.lane_width
        # Search right till cumulative distance exceeds radius
        while cum_dist < self._radius:
            # Get right marking of current waypoint
            right_marking = self._get_lane_marking(
                right_marking_waypt, Direction.Right)
            if right_marking.type != carla.LaneMarkingType.NONE:
                # A candidate found
                marking_pts, marking_types = self._get_marking_pts(
                    right_marking_waypt, tform_w2e, Direction.Right)

                candidate_markings_in_ego.append(marking_pts)
                candidate_marking_types.append(marking_types)

            # Stop when reaching a curb
            if right_marking.type == carla.LaneMarkingType.Curb:
                break
            # Go to the next right lane
            right_marking_waypt = self._get_next_lane(
                right_marking_waypt, Direction.Right)
            if right_marking_waypt is not None:
                cum_dist += right_marking_waypt.lane_width
            else:
                # Stop when next lane waypoint is None
                break

        return candidate_markings_in_ego, candidate_marking_types

    def _get_marking_pts(self, waypoint: carla.Waypoint, world_to_ego: CarlaW2ETform, direction):
        """ Get marking points along the lane in ego frame (z-up) for given waypoint and direction """

        # Local helper functions
        def get_lane_marking_pt_in_ego_frame(waypoint_of_interest):
            """ Get the corresponding marking point in ego frame given a waypoint of interest """
            if self._check_same_direction_as_ego_lane(waypoint_of_interest):
                half_width = 0.5 * waypoint_of_interest.lane_width
            else:
                half_width = -0.5 * waypoint_of_interest.lane_width

            if direction == Direction.Left:
                lane_pt_in_world = waypoint_of_interest.transform.transform(
                    carla.Location(y=-half_width))
            else:
                lane_pt_in_world = waypoint_of_interest.transform.transform(
                    carla.Location(y=half_width))
            return world_to_ego.tform_world_to_ego(lane_pt_in_world)

        # Local helper functions
        def get_next_waypoint(waypoint_of_interest, distance, direction):
            """ 
            Get the next waypoint of the waypont of interest.
            The direction is with respect to the ego lane.
            Returns None if not found.
            """
            next_waypt = None
            if direction == Direction.Forward:
                if self._check_same_direction_as_ego_lane(waypoint_of_interest):
                    # next() returns a list with at most one element
                    new_waypt = waypoint_of_interest.next(distance)
                else:
                    # previous() returns a list with at most one element
                    new_waypt = waypoint_of_interest.previous(distance)
                if len(new_waypt) != 0:
                    next_waypt = new_waypt[0]
            elif direction == Direction.Backward:
                if self._check_same_direction_as_ego_lane(waypoint_of_interest):
                    new_waypt = waypoint_of_interest.previous(distance)
                else:
                    new_waypt = waypoint_of_interest.next(distance)
                if len(new_waypt) != 0:
                    next_waypt = new_waypt[0]

            return next_waypt

        # Containers for lane marking points and their types
        # Stored in ascending order (from back to forth)
        lane_pts_in_ego = []    # a list 3D points in ego frame (z-up)
        lane_types = []         # a list of marking type of each point
        # Previous waypointes of the given waypoint
        # carla's waypoint.previous_until_lane_start() has bugs with lane type like Border and Parking.
        # The method simply stops the whole program without any warning or error when called with a waypoint of the types above.
        # Here a for loop is used instead of previous_until_lane_start() for finding waypoints backwards.
        # One advantage is we can just add the marking points with valid types.
        for distance in reversed(range(1, 11)):
            backward_waypt = get_next_waypoint(
                waypoint, distance, Direction.Backward)
            if backward_waypt is not None:
                lane_type = self._get_lane_marking(
                    backward_waypt, direction).type
                # Add only points with visible marking types
                if lane_type != carla.LaneMarkingType.NONE:
                    lane_pts_in_ego.append(
                        get_lane_marking_pt_in_ego_frame(backward_waypt))
                    lane_types.append(lane_type)
            else:
                continue

        # The given waypoint
        lane_type = self._get_lane_marking(waypoint, direction).type
        if lane_type != carla.LaneMarkingType.NONE:
            lane_pts_in_ego.append(get_lane_marking_pt_in_ego_frame(waypoint))
            lane_types.append(lane_type)

        # Next waypointes of the given waypoint
        # Here a for loop is used instead of next_until_lane_end() for finding waypoints forwards.
        # One advantage is we can just add the marking points with valid types.
        for distance in range(1, 11):
            forward_waypt = get_next_waypoint(
                waypoint, distance, Direction.Forward)
            if forward_waypt is not None:
                lane_type = self._get_lane_marking(
                    forward_waypt, direction).type
                if lane_type != carla.LaneMarkingType.NONE:
                    lane_pts_in_ego.append(
                        get_lane_marking_pt_in_ego_frame(forward_waypt))
                    lane_types.append(lane_type)
            else:
                continue

        return lane_pts_in_ego, lane_types

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
        right_waypt = self._find_visible_lane_marking(
            direction=Direction.Right)

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
        next_left_waypt = self._find_next_visible_lane_marking(
            direction=Direction.Left)

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
        next_right_waypt = self._find_next_visible_lane_marking(
            direction=Direction.Right)

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
        """ Get carla.LaneMarking object of given waypoint in the specified direction with respect to ego lane """
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
