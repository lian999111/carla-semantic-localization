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
from vision.vutils import decode_depth


class Direction(Enum):
    """ Enum for specifying lane searching direction. """
    Left = 1
    Right = 2
    Forward = 3
    Backward = 4


class GroundTruthExtractor(object):
    """ Class for ground truth extraction. """

    def __init__(self, ego_veh, carla_map, actor_list, config):
        """ Constructor method. """
        # Ego vehicle
        # Distance from rear axle to front bumper
        self.raxle_to_fbumper = config['ego_veh']['raxle_to_fbumper']
        # Distance from rear axle to center of gravity
        self.raxle_to_cg = config['ego_veh']['raxle_to_cg']

        self.ego_veh = ego_veh
        self.ego_veh_tform = ego_veh.get_transform()

        # TODO: Try put this frame at the intersect of camera FOV and ground surface?
        # Front bumper location in Carla's coordinate system (left-handed z-up) as a carla.Vector3D object
        # It's in carla's left-handed world frame so querying waypoints using carla's APIs is more straightforward
        self._fbumper_location = self.ego_veh_tform.transform(
            carla.Location(x=self.raxle_to_fbumper - self.raxle_to_cg))

        # Rear axle in Carla's coordinate system (left-handed z-up) as a carla.Vector3D object
        raxle_location = self.ego_veh_tform.transform(
            carla.Location(x=-self.raxle_to_cg))
        # Rear axle's location in our coordinate system (right-handed z-up) as a numpy array
        # This is the ground truth of the rear axle's location
        self.raxle_location = np.array([raxle_location.x,
                                           -raxle_location.y,
                                           raxle_location.z])
        # Rear axle's orientation in our coordinate system (right-handed z-up) as a numpy array (roll, pitch, yaw)
        # This is the ground truth of the rear axle's orientation (rad)
        self.raxle_orientation = np.array([self.ego_veh_tform.rotation.roll,
                                              -self.ego_veh_tform.rotation.pitch,
                                              -self.ego_veh_tform.rotation.yaw]) * np.pi / 180

        # Simulation environment
        self.map = carla_map
        self.actor_list = actor_list
        # Ignore landmarks now since carla built-in maps don't have them defined
        # self.landmarks = None

        # Lanes
        # Search radius
        self._radius = config['gt']['lane']['radius']

        # Current waypoint
        self.waypoint = None
        # Flag indicating ego vehicle is in junction
        self.in_junction = False
        # Current lane id (to know the order of double marking types)
        self.lane_id = None
        # Carla.LaneMarking object of each marking
        self.left_marking = None
        self.next_left_marking = None
        self.right_marking = None
        self.next_right_marking = None
        # c0 and c1 of lane markings
        self.left_marking_coeffs = [0, 0]
        self.next_left_marking_coeffs = [0, 0]
        self.right_marking_coeffs = [0, 0]
        self.next_right_marking_coeffs = [0, 0]

    def update(self):
        """ Update ground truth at the current tick. """
        self.ego_veh_tform = self.ego_veh.get_transform()

        # Update front bumper's location (carla.Vector3D in left-handed frame)
        self._fbumper_location = self.ego_veh_tform.transform(
            carla.Location(x=self.raxle_to_fbumper - self.raxle_to_cg))  # carla.Location.transform() returns just a carla.Vector3D object

        # Update rear axle's location and orientation (np.array in right-handed z-up frame)
        raxle_location = self.ego_veh_tform.transform(
            carla.Location(x=-self.raxle_to_cg))
        self.raxle_location = np.array([raxle_location.x,
                                           -raxle_location.y,
                                           raxle_location.z])
        self.raxle_orientation = np.array([self.ego_veh_tform.rotation.roll,
                                              -self.ego_veh_tform.rotation.pitch,
                                              -self.ego_veh_tform.rotation.yaw])

        # Update lanes
        # Find a waypoint on the nearest lane (any lane type except NONE)
        # So when ego vehicle is driving abnormally (e.g. on shoulder or parking), lane markings can still be obtained.
        # Some strange results may happen in extreme cases though (e.g. car drives on rail or sidewalk).
        self.waypoint = self.map.get_waypoint(
            self._fbumper_location, lane_type=carla.LaneType.Any)

        self.in_junction = self.waypoint.is_junction
        self.lane_id = self.waypoint.lane_id

        # When the query point (front bumper) is farther from the obtained waypoint than searching radius,
        # the ego vehicle is likely  to be off road, then no lane info is further extracted.
        # make a carla.Location object
        fbumper_loc = carla.Location(self._fbumper_location)
        if fbumper_loc.distance(self.waypoint.transform.location) >= self._radius:
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

            self.left_marking_coeffs = candidates[left_idx][0] if left_idx is not None else [
                0, 0]
            self.left_marking = candidates[left_idx][1] if left_idx is not None else None
            self.next_left_marking_coeffs = candidates[next_left_idx][0] if next_left_idx is not None else [
                0, 0]
            self.next_left_marking = candidates[next_left_idx][1] if next_left_idx is not None else None
            self.right_marking_coeffs = candidates[right_idx][0] if right_idx is not None else [
                0, 0]
            self.right_marking = candidates[right_idx][1] if right_idx is not None else None
            self.next_right_marking_coeffs = candidates[next_right_idx][0] if next_right_idx is not None else [
                0, 0]
            self.next_right_marking = candidates[next_right_idx][1] if next_right_idx is not None else None

        else:
            self.left_marking = None
            self.right_marking = None
            self.next_left_marking = None
            self.next_right_marking = None
            self.left_marking_coeffs = [0, 0]
            self.next_left_marking_coeffs = [0, 0]
            self.right_marking_coeffs = [0, 0]
            self.next_right_marking_coeffs = [0, 0]

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
        fbumper_transform = carla.Transform(
            self._fbumper_location, self.ego_veh.get_transform().rotation)
        # Object for transforming a carla.Location in carla's world frame (left-handed z-up)
        # into our front bumper's frame (right-handed z-up)
        tform_w2e = CarlaW2ETform(fbumper_transform)
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


class ObjectGTExtractor(object):
    """
    Class for object ground truth extraction.

    It uses semantic segmentation together depth cameras that can freely move around the environment 
    to extract ground truth of objects of interest.
    """

    def __init__(self, carla_world, transform, obj_gt_config, attach_to=None):
        """
        Constructor method.

        Input:
            carla_world: Carla.World object of the simulation environment.
            transform: Carla.Transform representating the location and orientation 
                       the caremras will be spawned with..
            obj_gt_config: Configurations for object ground truth extraction.
            attach_to: Parent actor that the camera will follow around.
        """
        # Queues to store images from Carla cameras
        self._ss_queue = queue.Queue()
        self._depth_queue = queue.Queue()

        # Images
        self.ss_image = None
        self.depth_image = None

        # xyz coordinates of poles wrt the reference frame
        # The relation ship
        self.poles_xyz = None

        # Initialize semantic camera
        ss_cam_bp = carla_world.get_blueprint_library().find(
            'sensor.camera.semantic_segmentation')
        ss_cam_bp.set_attribute('image_size_x', obj_gt_config['res_h'])
        ss_cam_bp.set_attribute('image_size_y', obj_gt_config['res_v'])
        ss_cam_bp.set_attribute('fov', obj_gt_config['fov'])

        print("Spawning semantic camera sensor for ground truth.")
        self.sensor = carla_world.spawn_actor(
            ss_cam_bp, transform, attach_to=attach_to)
        self.sensor.listen(lambda image: self._ss_queue.put(image))

        # Initialize depth camera
        depth_cam_bp = carla_world.get_blueprint_library().find(
            'sensor.camera.depth')
        depth_cam_bp.set_attribute(
            'image_size_x', obj_gt_config['res_h'])
        depth_cam_bp.set_attribute(
            'image_size_y', obj_gt_config['res_v'])
        depth_cam_bp.set_attribute('fov', obj_gt_config['fov'])

        print("Spawning depth camera sensor for ground truth.")
        self.sensor = carla_world.spawn_actor(
            depth_cam_bp, transform, attach_to=attach_to)
        self.sensor.listen(lambda image: self._depth_queue.put(image))

    def update(self):
        """
        Update object-related ground truth.

        This method first calls _update_images() then use the new images to extract object ground truth. 
        """

    def _update_images(self):
        """
        Update semantic and depth images at the current tick.

        Must be called at each Carla tick to get the latest images.
        """
        # Update semantic image
        image = self._ss_queue.get()
        np_img = np.frombuffer(image.raw_data, dtype=np.uint8)
        # Reshap to BGRA format
        np_img = np.reshape(np_img, (image.height, image.width, -1))
        # Semantic info is stored only in the R channel
        # Since np_img is from the buffer, which is reused by Carla
        # Making a copy makes sure ss_img is not subject to side-effect when the underlying buffer is modified
        self.ss_img = np_img[:, :, 2].copy()

        # Update depth image
        image = self._depth_queue.get()
        np_img = np.frombuffer(image.raw_data, dtype=np.uint8)
        # Reshap to BGRA format
        # The depth info is encoded by the BGR channels using the so-called depth buffer.
        # Decoding is required before use.
        # Ref: https://carla.readthedocs.io/en/latest/ref_sensors/#depth-camera
        np_img = np.reshape(np_img, (image.height, image.width, -1))
        # Since np_img is from the buffer, which is reused by Carla
        # Making a copy makes sure depth_buffer is not subject to side-effect when the underlying buffer is modified
        depth_buffer = np_img[:, :, 0:3]    # get just BGR channels
        self.depth_image = decode_depth(depth_buffer)

    def _update_poles_xyz(self):
        """
        Update xyz coordinates of poles.

        Must be called at each Carla tick to get the latest images.
        """
        pass
