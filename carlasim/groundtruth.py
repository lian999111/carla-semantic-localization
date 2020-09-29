"""
Implements ground truth of neighboring objects extraction from Carla simulation environment.
"""

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
import queue
from carlasim.carla_tform import CarlaW2ETform
from vision.utils import decode_depth
from carlasim.utils import LaneMarking, TrafficSignType, TrafficSign


class Direction(Enum):
    """ Enum for specifying lane searching direction. """
    Left = 1
    Right = 2
    Forward = 3
    Backward = 4

# TODO: extract traffic signs, especially stop signs
class GroundTruthExtractor(object):
    """ 
    Class for ground truth extraction during Carla simulation. 

    This class uses a dict "all_gt" as the sole buffer to store all ground truth data.
    This buffer is divided into 2 sub-elements: 
        - static: Stores static ground truth data that don't change across time, such as traffic sign locations.
        - seq: Stores sequential ground truth data that changes over time, such as the pose of the ego vehicle.
    """

    def __init__(self, ego_veh: carla.Actor, gt_config: dict):
        """
        Constructor method. 

        Input:
            ego_veh: Carla.Actor obj of the ego vehicle. Carla.Actor provides a get_world() method to get the 
                     Carla.World it belongs to; thus, ego vehicle actor is sufficient to retrieve other info.
        """
        # Retrieve other necessary objects from ego vehicle Carla.Actor object
        carla_world = ego_veh.get_world()
        carla_map = carla_world.get_map()

        # Dict as an buffer to store all ground truth data of interest
        # Using dict helps automate data selection during recording since data can be queried by keys
        # This buffer is automatically updated when the child ground truth extractors update their buffer
        self.all_gt = {'static': {}, 'seq': {}}

        # Traffic signs
        self.all_gt['static']['traffic_sign'] = self.get_traffic_signs(carla_world)

        # Front bumper's transform in Carla's coordinate system
        # It's for the convenience of querying waypoints for lane using carla's APIs
        # TODO: Try put this frame at the intersection of camera FOV and ground surface?
        self._fbumper_carla_tform = None

        # Pose ground truth extractor
        self.pose_gt = PoseGTExtractor(ego_veh, gt_config['pose'])
        # Lane ground truth extractor
        self.lane_gt = LaneGTExtractor(carla_map, gt_config['lane'])

        # Set up buffer for sequential data
        # Point to pose ground truth extractor's gt buffer
        self.all_gt['seq']['pose'] = self.pose_gt.gt
        # Point to lane ground truth extractor's gt buffer
        self.all_gt['seq']['lane'] = self.lane_gt.gt

    def update(self):
        """ Update ground truth at the current tick. """
        # Update pose
        self.pose_gt.update()
        self._fbumper_carla_tform = self.pose_gt.get_fbumper_carla_tform()
        # Update lanes
        self.lane_gt.update_using_carla_transform(self._fbumper_carla_tform)
    
    @staticmethod
    def get_traffic_signs(carla_world):
        """
        Get a list of of TrafficSign objects from the given Carla.World object.
        """
        actor_list = carla_world.get_actors()
        carla_map = carla_world.get_map()
        traffic_signs = []
        
        # Stop signs
        for actor in actor_list.filter('traffic.stop'):
            location = actor.get_location()
            closest_waypt = carla_map.get_waypoint(location)
            
            # If the stop sign is within 1 meter from the closest waypoint, it is most likely on the road surface
            if location.distance(closest_waypt.transform.location) < 1:
                traffic_signs.append(TrafficSign(actor, TrafficSignType.StopOnRoad))
                ########################
                carla_world.debug.draw_arrow(location, location + carla.Location(z=50))
            else:
                traffic_signs.append(TrafficSign(actor, TrafficSignType.Stop))
                ########################
                carla_world.debug.draw_arrow(location, location + carla.Location(z=50), color=carla.Color(0, 255, 0), thickness=2)

        # Yield signs
        for actor in actor_list.filter('traffic.yield'):
            traffic_signs.append(TrafficSign(actor, TrafficSignType.Yield))

            location = actor.get_location()
            carla_world.debug.draw_arrow(location, location + carla.Location(z=50), color=carla.Color(0, 0, 255))

        # Speed limit signs
        for actor in actor_list.filter('traffic.speed_limit.*'):
            traffic_signs.append(TrafficSign(actor, TrafficSignType.SpeedLimit))

            location = actor.get_location()
            carla_world.debug.draw_arrow(location, location + carla.Location(z=50), color=carla.Color(255, 0, 255))

        # Traffic lights
        for actor in actor_list.filter('traffic.traffic_light'):
            traffic_signs.append(TrafficSign(actor, TrafficSignType.TrafficLight))

            location = actor.get_location()
            carla_world.debug.draw_arrow(location, location + carla.Location(z=50), color=carla.Color(255, 255, 0))

        return traffic_signs
            



class PoseGTExtractor(object):
    """ Class for rear axle's pose extraction. """

    def __init__(self, ego_veh: carla.Actor, pose_config: dict):
        # Ego vehicle
        self.ego_veh = ego_veh
        self.ego_veh_tform = ego_veh.get_transform()
        # Distance from rear axle to front bumper
        self.raxle_to_fbumper = pose_config['raxle_to_fbumper']
        # Distance from rear axle to center of gravity
        self.raxle_to_cg = pose_config['raxle_to_cg']

        self.gt = {}
        # Rear axle in Carla's coordinate system (left-handed z-up) as a carla.Vector3D object
        raxle_location = self.ego_veh_tform.transform(
            carla.Location(x=-self.raxle_to_cg))
        # Rear axle's location in our coordinate system (right-handed z-up) as a numpy array
        # This is the ground truth of the rear axle's location
        self.gt['raxle_location'] = np.array([raxle_location.x,
                                              -raxle_location.y,
                                              raxle_location.z])
        # Rear axle's orientation in our coordinate system (right-handed z-up) as a numpy array (roll, pitch, yaw)
        # This is the ground truth of the rear axle's orientation (rad)
        self.gt['raxle_orientation'] = np.array([self.ego_veh_tform.rotation.roll,
                                                 -self.ego_veh_tform.rotation.pitch,
                                                 -self.ego_veh_tform.rotation.yaw]) * np.pi / 180

    def update(self):
        """ Update ground truth at the current tick. """
        self.ego_veh_tform = self.ego_veh.get_transform()

        # Update rear axle's location and orientation (np.array in right-handed z-up frame)
        raxle_location = self.ego_veh_tform.transform(
            carla.Location(x=-self.raxle_to_cg))
        self.gt['raxle_location'] = np.array([raxle_location.x,
                                              -raxle_location.y,
                                              raxle_location.z])
        self.gt['raxle_orientation'] = np.array([self.ego_veh_tform.rotation.roll,
                                                 -self.ego_veh_tform.rotation.pitch,
                                                 -self.ego_veh_tform.rotation.yaw])

    def get_fbumper_carla_tform(self):
        """ Get front bumper's carla.Transform. """
        fbumper_location = self.ego_veh_tform.transform(
            carla.Location(x=self.raxle_to_fbumper - self.raxle_to_cg))
        # Update front bumper's transform in Carla's coordinate system
        fbumper_carla_tform = carla.Transform(
            carla.Location(fbumper_location), self.ego_veh_tform.rotation)
        return fbumper_carla_tform


class LaneGTExtractor(object):
    """ 
    Class for lane ground truth extraction. 
    
    This class is intended to be used not only during data collection, but also localization.
    It can be updated with an specified pose in the right-handed z-up convention and extract the lane ground truth.
    """

    def __init__(self, carla_map: carla.Map, lane_gt_config):
        self.map = carla_map

        # carla.Transform of current query point
        self._carla_tform = None

        # Search radius
        self._radius = lane_gt_config['radius']

        self.gt = {}
        # Current waypoint
        self.waypoint = None
        # Flag indicating ego vehicle is in junction
        self.gt['in_junction'] = False
        # Current lane id (to know the order of double marking types)
        self.gt['lane_id'] = None
        # Carla.LaneMarking object of each marking
        self.gt['left_marking'] = None
        self.gt['next_left_marking'] = None
        self.gt['right_marking'] = None
        self.gt['next_right_marking'] = None
        # c0 and c1 of lane markings
        self.gt['left_marking_coeffs'] = [0, 0]
        self.gt['next_left_marking_coeffs'] = [0, 0]
        self.gt['right_marking_coeffs'] = [0, 0]
        self.gt['next_right_marking_coeffs'] = [0, 0]

    def update(self, location, rotation):
        """
        Update lane ground truth at the current tick. 

        Input:
            location: Array-like 3D query point in world (right-handed z-up).
            rotation: Array-like roll pitch yaw rotation representation in rad (right-handed z-up).
        """
        # Convert the passed-in location into Carla's frame (left-handed z-up).
        carla_location = carla.Location(
            x=location[0], y=-location[1], z=location[2])
        rotation = rotation * 180/np.pi  # to deg
        carla_rotation = carla.Rotation(
            roll=rotation[0], pitch=rotation[1], yaw=rotation[2])
        self.update_using_carla_transform(
            carla.Transform(carla_location, carla_rotation))

    def update_using_carla_transform(self, carla_tform: carla.Transform):
        """ 
        Update lane ground truth using a carla.Location at the current tick. 

        Input:
            carla_tform: carla.Transform query point following Carla's coordinate convention.
        """
        self._carla_tform = carla_tform
        carla_location = self._carla_tform.location

        # Update lanes
        # Find a waypoint on the nearest lane (any lane type except NONE)
        # So when ego vehicle is driving abnormally (e.g. on shoulder or parking), lane markings can still be obtained.
        # Some strange results may happen in extreme cases though (e.g. car drives on rail or sidewalk).
        self.waypoint = self.map.get_waypoint(
            carla_location, lane_type=carla.LaneType.Any)

        self.gt['in_junction'] = self.waypoint.is_junction
        self.gt['lane_id'] = self.waypoint.lane_id

        if carla_location.distance(self.waypoint.transform.location) >= self._radius:
            self.waypoint = None

        if self.waypoint is not None:
            # Find candidate visible markings within the specified radius
            # We get a list of 3D points of candidate markings in front bumper's frame (right-handed z-up)
            # as well as a list containing the corresponding marking types.
            candidate_markings_in_ego, candidate_markings = self._find_candidate_markings()
            candidates = []
            for idx, candidate in enumerate(candidate_markings_in_ego):
                # Extract only x and y coordinates of this candidate marking
                candidate_2D = np.array(candidate)[:, 0:2]
                # Find the index where the x value goes from negative to positive
                # That's where the lane marking intersect the y-axis of the front bumper
                # Find the idx where the sign is about to change
                sign_change_idx = np.where(
                    candidate_2D[:-1, 0] * candidate_2D[1:, 0] < 0)[0]
                # Skip this candidate if there is no sign change at all
                if sign_change_idx.size == 0:
                    continue
                last_neg_idx = sign_change_idx[0]
                first_pos_idx = last_neg_idx + 1
                pt1 = candidate_2D[last_neg_idx, :]
                pt2 = candidate_2D[first_pos_idx, :]

                # TODO: address when pt1 or pt2 don't exist

                # If the 2 points across the y-axis are too far, there are no lane marking defined in between
                # then skip this candidate
                if np.linalg.norm(pt1-pt2) > 10:
                    break

                # Compute c0 and c1
                c1 = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
                c0 = pt1[1] - c1*pt1[0]

                # Use the marking of pt2
                candidate_marking_obj = candidate_markings[idx][first_pos_idx]

                # append the tuple ([c0, c1], marking_type)
                candidates.append(([c0, c1], candidate_marking_obj))

            # Sort candidate markings with c0 in descending order (left to right)
            candidates.sort(reverse=True)

            # Init indices
            left_idx = None
            next_left_idx = None
            right_idx = None
            next_right_idx = None

            # Get the candidate with the smallest positive c0, which is the left lane marking
            left_idx = (sum(item[0][0] >= 0 for item in candidates)-1) if (
                sum(item[0][0] >= 0 for item in candidates) > 0) else None

            # Try to find next left and right by shifting from left_idx
            if left_idx is not None:
                next_left_idx = left_idx-1 if left_idx-1 >= 0 else None
                right_idx = (left_idx+1) if (left_idx+1 <
                                             len(candidates)) else None
            # If the right marking is not decided yet
            if right_idx is None:
                # Get the candidate with the largest negative c0 (smallest absolute value), which is the right lane marking
                right_idx = len(candidates)-sum(item[0][0] < 0 for item in candidates) if (
                    sum(item[0][0] < 0 for item in candidates) > 0) else None
            # Try to find next right by shifting from right_idx
            if right_idx is not None:
                next_right_idx = (right_idx+1) if (right_idx +
                                                   1 < len(candidates)) else None

            # c0 and c1 coefficients
            self.gt['left_marking_coeffs'] = candidates[left_idx][0] if left_idx is not None else [
                0, 0]
            self.gt['next_left_marking_coeffs'] = candidates[next_left_idx][0] if next_left_idx is not None else [
                0, 0]
            self.gt['right_marking_coeffs'] = candidates[right_idx][0] if right_idx is not None else [
                0, 0]
            self.gt['next_right_marking_coeffs'] = candidates[next_right_idx][0] if next_right_idx is not None else [
                0, 0]

            # Lane marking objects (replaced with our pickle-able LaneMarking)
            left_marking_obj = candidates[left_idx][1] if left_idx is not None else None
            next_left_lane_marking_obj = candidates[next_left_idx][1] if next_left_idx is not None else None
            right_marking_obj = candidates[right_idx][1] if right_idx is not None else None
            next_right_marking_obj = candidates[next_right_idx][1] if next_right_idx is not None else None

            self.gt['left_marking'] = LaneMarking.from_carla_lane_marking(
                left_marking_obj) if left_marking_obj else None
            self.gt['next_left_marking'] = LaneMarking.from_carla_lane_marking(
                next_left_lane_marking_obj) if next_left_lane_marking_obj is not None else None
            self.gt['right_marking'] = LaneMarking.from_carla_lane_marking(
                right_marking_obj) if right_marking_obj is not None else None
            self.gt['next_right_marking'] = LaneMarking.from_carla_lane_marking(
                next_right_marking_obj) if next_right_marking_obj is not None else None

        else:
            self.gt['left_marking_coeffs'] = [0, 0]
            self.gt['next_left_marking_coeffs'] = [0, 0]
            self.gt['right_marking_coeffs'] = [0, 0]
            self.gt['next_right_marking_coeffs'] = [0, 0]
            self.gt['left_marking'] = None
            self.gt['next_left_marking'] = None
            self.gt['right_marking'] = None
            self.gt['next_right_marking'] = None

    def _find_candidate_markings(self):
        """ 
        Find candidate lane markings withing a radius at the current time step.

        Output:
            candidate_markings_in_ego:
                List of candidate lane markings. Each lane marking consists of a list of 
                3D points in the ego frame (right-handed z-up).
            candidate_markings:
                List containing carla.LaneMarking objects corresponding to the points
                in candidate_markings_in_ego.
        """
        # Object for transforming a carla.Location in carla's world frame (left-handed z-up)
        # into our front bumper's frame (right-handed z-up)
        tform_w2e = CarlaW2ETform(self._carla_tform)
        waypt_ego_frame = tform_w2e.tform_world_to_ego(
            self.waypoint.transform.location)

        # Container lists
        # A list of points of each candidate marking
        # Each candidate marking has a list of 3D points in front bumper's frame (z-up)
        candidate_markings_in_ego = []
        # A 2D list of carla.LandMarking object of each candidate marking point
        candidate_markings = []

        # Left
        # Initilization with current waypoint
        left_marking_waypt = self.waypoint
        cum_dist = waypt_ego_frame[1] + \
            0.5 * left_marking_waypt.lane_width
        # Search left till cumulative distance exceeds radius or a None waypoint is reached
        while cum_dist < self._radius:
            # Get left marking of current waypoint
            left_marking = self._get_lane_marking(
                left_marking_waypt, Direction.Left)
            if left_marking.type != carla.LaneMarkingType.NONE:
                # A candidate found
                marking_pts, marking_objs = self._get_marking_pts(
                    left_marking_waypt, tform_w2e, Direction.Left)

                candidate_markings_in_ego.append(marking_pts)
                candidate_markings.append(marking_objs)

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
        # Search right till cumulative distance exceeds radius or a None waypoint is reached
        while cum_dist < self._radius:
            # Get right marking of current waypoint
            right_marking = self._get_lane_marking(
                right_marking_waypt, Direction.Right)
            if right_marking.type != carla.LaneMarkingType.NONE:
                # A candidate found
                marking_pts, marking_objs = self._get_marking_pts(
                    right_marking_waypt, tform_w2e, Direction.Right)

                candidate_markings_in_ego.append(marking_pts)
                candidate_markings.append(marking_objs)

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

        return candidate_markings_in_ego, candidate_markings

    def _get_marking_pts(self, waypoint: carla.Waypoint, world_to_ego: CarlaW2ETform, side: Direction):
        """ 
        Get marking points along the lane marking of specified side in ego frame (right-handed z-up).

        Input:
            waypoint: Carla.Waypont object of the lane marking of interest.
            world_to_ego: CarlaW2ETform object that performs world-to-ego transformation.
            side: Direction object specifying the side of interest.
        Output:
            lane_pts_in_ego: A list of lane markings 3D points in ego frame (right-handed z-up).
            lane_markings: A list of lane marking objects corresponding to the list of 3D points.
        """

        # Local helper functions
        def get_lane_marking_pt_in_ego_frame(waypoint_of_interest):
            """ 
            Get the corresponding marking point in ego frame given a waypoint of interest. 

            It obtains the point of lane marking by projecting the half lane width.
            """
            if self._check_same_direction_as_ego_lane(waypoint_of_interest):
                half_width = 0.5 * waypoint_of_interest.lane_width
            else:
                half_width = -0.5 * waypoint_of_interest.lane_width

            if side == Direction.Left:
                lane_pt_in_world = waypoint_of_interest.transform.transform(
                    carla.Location(y=-half_width))
            else:
                lane_pt_in_world = waypoint_of_interest.transform.transform(
                    carla.Location(y=half_width))
            return world_to_ego.tform_world_to_ego(lane_pt_in_world)

        # Local helper functions
        def get_next_waypoint(waypoint_of_interest, distance, direction):
            """ 
            Get the next waypoint in the specified direction (forward or backward).

            Input:
                waypoint_of_interest: Carla.Waypoint object of interest.
                distance: Distance in meters to query the next waypoint.
                direction: Direction object specifying forward or backward with respect to the ego lane.
            Output:
                carla.Waypoint object. None if not found. 
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
        # a list 3D points in ego frame (right-handed z-up)
        lane_pts_in_ego = []
        lane_markings = []      # a list of carla.LaneMarking of each point
        # Previous waypointes of the given waypoint
        # carla's waypoint.previous_until_lane_start() has bugs with lane type like Border and Parking.
        # The method simply stops the whole program without any warning or error when called with a waypoint of the types above.
        # Here a for loop is used instead of previous_until_lane_start() for finding waypoints backwards.
        # One advantage is we can just add the marking points with valid types.
        for distance in reversed(range(1, 11)):
            backward_waypt = get_next_waypoint(
                waypoint, distance, Direction.Backward)
            if backward_waypt is not None:
                lane_marking = self._get_lane_marking(
                    backward_waypt, side)
                # Add only points with visible marking types
                if lane_marking.type != carla.LaneMarkingType.NONE:
                    lane_pts_in_ego.append(
                        get_lane_marking_pt_in_ego_frame(backward_waypt))
                    lane_markings.append(lane_marking)
            else:
                continue

        # The given waypoint
        lane_marking = self._get_lane_marking(waypoint, side)
        if lane_marking.type != carla.LaneMarkingType.NONE:
            lane_pts_in_ego.append(get_lane_marking_pt_in_ego_frame(waypoint))
            lane_markings.append(lane_marking)

        # Next waypointes of the given waypoint
        # Here a for loop is used instead of next_until_lane_end() for finding waypoints forwards.
        # One advantage is we can just add the marking points with valid types.
        for distance in range(1, 11):
            forward_waypt = get_next_waypoint(
                waypoint, distance, Direction.Forward)
            if forward_waypt is not None:
                lane_marking = self._get_lane_marking(
                    forward_waypt, side)
                if lane_marking.type != carla.LaneMarkingType.NONE:
                    lane_pts_in_ego.append(
                        get_lane_marking_pt_in_ego_frame(forward_waypt))
                    lane_markings.append(lane_marking)
            else:
                continue

        return lane_pts_in_ego, lane_markings

    def _check_same_direction_as_ego_lane(self, waypoint_of_interest):
        """ Check if the direction of the given waypoint is the same as the ego lane. """
        if waypoint_of_interest is None:
            return None
        return (self.waypoint.lane_id * waypoint_of_interest.lane_id) > 0

    def _get_next_lane(self, waypoint_of_interest, direction):
        """ Get waypoint of next lane in specified direction (left or right) with respect to ego lane. """
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
        """ Get carla.LaneMarking object of given waypoint in the specified direction with respect to ego lane. """
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
