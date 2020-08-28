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

class GroundTruthExtractor(object):
    """ Class for ground truth extraction """

    def __init__(self, carla_map, ego_veh, actor_list, config_args=None):
        """ Constructor method """
        # Ego vehicle
        self.raxle_to_fbumper = config_args['ego_veh']['raxle_to_fbumper']
        self.ego_veh = ego_veh
        self.ego_veh_tform = ego_veh.get_transform()
        self.fbumper_location = self.ego_veh_tform.transform(
            carla.Location(x=self.raxle_to_fbumper - 1.4))  # -1.4: cg to rear axle only valid for carla's mustang

        # Simulation environment
        self.map = carla_map
        self.actor_list = actor_list
        # Ignore landmarks now since carla built-in maps don't have them defined
        # self.landmarks = None

        # Lanes
        # In some OpenDrive definitions, ego lane may have no visible lane boundaries.
        # Ground truth extractor then tries to get closest visible lane markings towards both sides and store the
        # corresponding waypoints into waypoint_left_marking and waypoint_right_marking. These 2 waypoints
        # are just for the convenience to extract closes left and right markings.
        self.waypoint = None
        self.waypoint_left_marking = None
        self.waypoint_right_marking = None
        self.waypoint_next_left_marking = None
        self.waypoint_next_right_marking = None

        self.left_marking_type = None
        self.next_left_marking_type = None
        self.right_marking_type = None
        self.next_right_marking_type = None
        # c0 and c1 of lane markings
        self.left_marking_param = [0, 0]
        self.next_left_marking_param = [0, 0]
        self.right_marking_param = [0, 0]
        self.next_right_marking_param = [0, 0]

    def update(self):
        """ Update ground truth at the current tick """
        self.ego_veh_tform = self.ego_veh.get_transform()
        # carla.Location.transform() returns just a carla.Vector3D object
        self.fbumper_location = carla.Location(self.ego_veh_tform.transform(
            carla.Location(x=self.raxle_to_fbumper - 1.4)))
        # Find a waypoint on the nearest lane (any lane type except NONE)
        # So when ego vehicle is driving abnormally (e.g. on shoulder or parking), lane markings can still be obtained.
        # Some strange results may happen in extreme cases though (e.g. car drives on rail or sidewalk).
        self.waypoint = self.map.get_waypoint(
            self.fbumper_location, lane_type=carla.LaneType.Any)

        # When the query point (front bumper) is farther from the obtained waypoint than its half lane width,
        # the ego vehicle is likely  to be off road, then no lane info is further extracted.
        if self.fbumper_location.distance(self.waypoint.transform.location) >= self.waypoint.lane_width/2:
            self.waypoint = None

        # and (self.waypoint.lane_type == carla.LaneType.Driving):
        if self.waypoint is not None:
            # Find left and right markings of ego lane
            self._get_left_lane_marking()
            self._get_right_lane_marking()

            # Find next lane markings unless a curb is aleady by the ego lane
            # Next left
            if self.left_marking_type != carla.LaneMarkingType.Curb:
                self._get_next_left_lane_marking()
            else:
                self.left_waypoint = None
                self.next_left_marking_type = None

            # Next right
            if self.right_marking_type != carla.LaneMarkingType.Curb:
                self._get_next_right_lane_marking()
            else:
                self.right_waypoint = None
                self.next_right_marking_type = None

            # TODO: lane marking parameters
        else:
            self.waypoint_left_marking = None
            self.waypoint_right_marking = None
            self.waypoint_next_left_marking = None
            self.waypoint_next_right_marking = None
            self.left_marking_type = None
            self.right_marking_type = None
            self.next_left_marking_type = None
            self.next_right_marking_type = None

    def _get_left_lane_marking(self):
        """
        Get left visible lane marking.
        """
        left_waypt = self._find_visible_lane_marking(to_left=True)

        # Updatee left lane marking
        if left_waypt is not None:
            self.waypoint_left_marking = left_waypt
            if self.waypoint.lane_id * left_waypt.lane_id >= 0:
                self.left_marking_type = left_waypt.left_lane_marking.type
            else:
                self.left_marking_type = left_waypt.right_lane_marking.type
        else:
            self.waypoint_left_marking = None
            self.left_marking_type = None

    def _get_right_lane_marking(self):
        """
        Get right visible lane marking.
        """
        right_waypt = self._find_visible_lane_marking(to_left=False)

        # Updatee right lane marking
        if right_waypt is not None:
            self.waypoint_right_marking = right_waypt
            if self.waypoint.lane_id * right_waypt.lane_id >= 0:
                self.right_marking_type = right_waypt.right_lane_marking.type
            else:
                self.right_marking_type = right_waypt.left_lane_marking.type
        else:
            self.waypoint_right_marking = None
            self.right_marking_type = None

    def _find_visible_lane_marking(self, to_left=True):
        """
        Helper method for finding the waypoint with visible lane marking.
        """
        # Use original ego waypoint if it already has visible marking
        if to_left:
            if self.waypoint.left_lane_marking.type != carla.LaneMarkingType.NONE:
                return self.waypoint
        else:
            if self.waypoint.right_lane_marking.type != carla.LaneMarkingType.NONE:
                return self.waypoint

        # No visible marking in the direction of interest, start searching
        curr_waypt = self.waypoint
        # Init previous lane id. It is used to check if a change in direction has occured.
        prev_lane_id = curr_waypt.lane_id
        # Search until visible lane marking found
        while True:
            if to_left:
                curr_waypt = curr_waypt.get_left_lane()
            else:
                curr_waypt = curr_waypt.get_right_lane()

            if curr_waypt is None:
                return None

            # Do not search across non-drivable space (e.g. across a middle island)
            # If so, set left lane as None
            # Not sure if this strategy is realistic
            if (curr_waypt.lane_type == carla.LaneType.Median
                    and curr_waypt.lane_type == carla.LaneType.Sidewalk
                    and curr_waypt.lane_type == carla.LaneType.Rail):
                return None

            if to_left:
                if curr_waypt.left_lane_marking.type != carla.LaneMarkingType.NONE:
                    # Found
                    break
            else:
                if curr_waypt.right_lane_marking.type != carla.LaneMarkingType.NONE:
                    # Found
                    break

            # Check if two adjacent lanes have same direction
            if prev_lane_id * curr_waypt.lane_id < 0:
                # Start searching using the opposite direction next time since the direction of the lane has changed
                to_left = not to_left

            prev_lane_id = curr_waypt.lane_id
        return curr_waypt

    def _get_next_left_lane_marking(self):
        """
        Get next left visible lane marking.
        Many transition lanes often without visible lane boundaries (e.g. Shoulder) are defined in OpenDrive.
        This method tries to find the lanes corresponding to visible lane boundaries (e.g. curb) but may not 
        be directly adjacent to current lane.
        """
        next_left_waypt = self._find_next_visible_lane_marking(to_left=True)

        # Updatee next left lane marking
        if next_left_waypt is not None:
            self.waypoint_next_left_marking = next_left_waypt
            if self.waypoint.lane_id * next_left_waypt.lane_id >= 0:
                self.next_left_marking_type = next_left_waypt.left_lane_marking.type
            else:
                self.next_left_marking_type = next_left_waypt.right_lane_marking.type
        else:
            self.waypoint_next_left_marking = None
            self.next_left_marking_type = None

    def _get_next_right_lane_marking(self):
        """
        Get next right visible lane marking.
        Many transition lanes often without visible lane boundaries (e.g. Shoulder) are defined in OpenDrive.
        This method tries to find the lanes corresponding to visible lane boundaries (e.g. curb) but may not 
        be directly adjacent to current lane.
        """
        next_right_waypt = self._find_next_visible_lane_marking(to_left=False)

        # Update next right lane marking
        if next_right_waypt is not None:
            self.waypoint_next_right_marking = next_right_waypt
            if self.waypoint.lane_id * next_right_waypt.lane_id >= 0:
                self.next_right_marking_type = next_right_waypt.right_lane_marking.type
            else:
                self.next_right_marking_type = next_right_waypt.left_lane_marking.type
        else:
            self.waypoint_next_right_marking = None
            self.next_right_marking_type = None

    def _find_next_visible_lane_marking(self, to_left=True):
        """
        Helper method for finding the waypoint with the visible next lane marking.
        """
        # Initialize curr_next_waypt with the waypoint in the corresponding direction
        if to_left:
            curr_next_waypt = self.waypoint_left_marking
        else:
            curr_next_waypt = self.waypoint_right_marking

        # If the starting waypoint to search is aleady None, return None
        if curr_next_waypt is None:
            return None

        # Check if searching has crossed the line that changes the direction of driving
        # If so, revert search direction
        if self.waypoint.lane_id * curr_next_waypt.lane_id < 0:
            to_left = not to_left

        # Go to the next lane
        if to_left:
            curr_next_waypt = curr_next_waypt.get_left_lane()
        else:
            curr_next_waypt = curr_next_waypt.get_right_lane()

        if curr_next_waypt is None:
            return None

        # Init previous lane id. It is used to check if a change in direction has occured.
        prev_lane_id = curr_next_waypt.lane_id
        # Search until visible lane marking found
        while True:
            if to_left:
                # Check the left lane marking
                if curr_next_waypt.left_lane_marking.type != carla.LaneMarkingType.NONE:
                    # Found
                    break
                else:
                    # Lane marking not visible. Go to the next one.
                    curr_next_waypt = curr_next_waypt.get_left_lane()
            else:
                # Check the right lane marking
                if curr_next_waypt.right_lane_marking.type != carla.LaneMarkingType.NONE:
                    # Found
                    break
                else:
                    # Lane marking not visible. Go to the next one.
                    curr_next_waypt = curr_next_waypt.get_right_lane()

            if curr_next_waypt is None:
                return None

            # Do not search across non-drivable space (e.g. across a middle island)
            # If so, set left lane as None
            # Not sure if this strategy is realistic
            if (curr_next_waypt.lane_type == carla.LaneType.Median
                    and curr_next_waypt.lane_type == carla.LaneType.Sidewalk
                    and curr_next_waypt.lane_type == carla.LaneType.Rail):
                curr_next_waypt = None
                break

            # Check if two adjacent lanes have same direction
            if prev_lane_id * curr_next_waypt.lane_id < 0:
                # Start searching using the opposite direction next time since the direction of the lane has changed
                to_left = not to_left
            prev_lane_id = curr_next_waypt.lane_id

        return curr_next_waypt

    def _get_poles(self):
        # TODO: use semantic lidar or just actors?
        pass
