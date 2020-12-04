#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains a local planner to perform
low-level waypoint following based on PID controllers. """

from collections import deque
from enum import Enum
import numpy as np
import math

import carla
from agents.navigation.controller import VehiclePIDController
from agents.tools.misc import distance_vehicle, draw_waypoints


class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations
    when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class LocalPlanner(object):
    """
    LocalPlanner implements the basic behavior of following a trajectory
    of waypoints that is generated on-the-fly.
    The low-level motion of the vehicle is computed by using two PID controllers,
    one is used for the lateral control
    and the other for the longitudinal control (cruise speed).

    When multiple paths are available (intersections)
    this local planner makes a random choice.
    """

    # Minimum distance to target waypoint as a percentage
    # (e.g. within 80% of total distance)

    # FPS used for dt
    FPS = 10

    def __init__(self, agent):
        """
        :param agent: agent that regulates the vehicle
        :param vehicle: actor to apply to local planner logic onto
        """
        self._vehicle = agent.vehicle
        self._map = agent.vehicle.get_world().get_map()

        self._target_speed = None
        self.sampling_radius = None
        self._min_distance = None
        self._current_waypoint = None
        self.target_road_option = None
        self._next_waypoints = None
        self.target_waypoint = None
        self._vehicle_controller = None
        self._global_plan = None
        self._pid_controller = None
        # queue with tuples of (waypoint, RoadOption)
        self.waypoints_queue = deque(maxlen=20000)
        self._buffer_size = 10
        self._waypoint_buffer = deque(maxlen=self._buffer_size)
        self._prev_waypoint = None

        self._init_controller()  # initializing controller

    def reset_vehicle(self):
        """Reset the ego-vehicle"""
        self._vehicle = None
        print("Resetting ego-vehicle!")

    def _init_controller(self):
        """
        Controller initialization.

        dt -- time difference between physics control in seconds.
        This is can be fixed from server side
        using the arguments -benchmark -fps=F, since dt = 1/F
_prev_waypoint
        target_speed -- desired cruise speed in km/h

        min_distance -- minimum distance to remove waypoint from queue

        lateral_dict -- dictionary of arguments to setup the lateral PID controller
                            {'K_P':, 'K_D':, 'K_I':, 'dt'}

        longitudinal_dict -- dictionary of arguments to setup the longitudinal PID controller
                            {'K_P':, 'K_D':, 'K_I':, 'dt'}
        """
        # Default parameters
        self.args_lat_hw_dict = {
            'K_P': 0.75,
            'K_D': 0.02,
            'K_I': 0.4,
            'dt': 1.0 / self.FPS}
        self.args_lat_city_dict = {
            'K_P': 0.58,
            'K_D': 0.02,
            'K_I': 0.5,
            'dt': 1.0 / self.FPS}
        self.args_long_hw_dict = {
            'K_P': 0.37,
            'K_D': 0.024,
            'K_I': 0.032,
            'dt': 1.0 / self.FPS}
        self.args_long_city_dict = {
            'K_P': 0.15,
            'K_D': 0.05,
            'K_I': 0.07,
            'dt': 1.0 / self.FPS}

        self._current_waypoint = self._map.get_waypoint(
            self._vehicle.get_location())
        self._prev_waypoint = self._current_waypoint

        self._global_plan = False

        self._target_speed = self._vehicle.get_speed_limit()

        self._min_distance = 3

    def set_speed(self, speed):
        """
        Request new target speed.

            :param speed: new target speed in km/h
        """

        self._target_speed = speed

    def set_global_plan(self, current_plan, clean=False):
        """
        Sets new global plan.

            :param current_plan: list of waypoints in the actual plan
        """
        for elem in current_plan:
            self.waypoints_queue.append(elem)

        if clean:
            self._waypoint_buffer.clear()
            for _ in range(self._buffer_size):
                if self.waypoints_queue:
                    self._waypoint_buffer.append(
                        self.waypoints_queue.popleft())
                else:
                    break

        # Put waypoints from global queue to buffer
        self._buffer_waypoints()

        self._global_plan = True

    def get_incoming_waypoint_and_direction(self, steps=3):
        """
        Returns direction and waypoint at a distance ahead defined by the user.

            :param steps: number of steps to get the incoming waypoint.
        """
        if len(self._waypoint_buffer) > steps:
            return self._waypoint_buffer[steps]

        else:
            try:
                wpt, direction = self._waypoint_buffer[-1]
                return wpt, direction
            except IndexError as i:
                print(i)
                return None, RoadOption.VOID
        return None, RoadOption.VOID

    def run_step(self, target_speed=None, debug=False):
        """
        Execute one step of local planning which involves
        running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

            :param target_speed: desired speed
            :param debug: boolean flag to activate waypoints debugging
            :return: control
        """

        if target_speed is not None:
            self._target_speed = target_speed
        else:
            self._target_speed = self._vehicle.get_speed_limit()

        # Buffering the waypoints
        self._buffer_waypoints(debug=debug)

        if len(self._waypoint_buffer) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False
            return control

        # Current vehicle waypoint
        self._current_waypoint = self._map.get_waypoint(
            self._vehicle.get_location())

        veh_vel = self._vehicle.get_velocity()
        speed = math.sqrt(veh_vel.x**2 + veh_vel.y**2) * 3.6    # kph
        look_ahead = max(1, speed / 5)

        # Target waypoint
        self.target_waypoint, self.target_road_option = self._waypoint_buffer[0]

        look_ahead_loc = self._get_look_ahead_location(look_ahead)

        if target_speed > 50:
            args_lat = self.args_lat_hw_dict
            args_long = self.args_long_hw_dict
        else:
            args_lat = self.args_lat_city_dict
            args_long = self.args_long_city_dict

        if not self._pid_controller:
            self._pid_controller = VehiclePIDController(self._vehicle,
                                                        args_lateral=args_lat,
                                                        args_longitudinal=args_long)

        control = self._pid_controller.run_step(
            self._target_speed, look_ahead_loc)

        # Purge the queue of obsolete waypoints
        vehicle_transform = self._vehicle.get_transform()
        max_index = -1

        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            if distance_vehicle(
                    waypoint, vehicle_transform) < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                if i == max_index:
                    self._prev_waypoint = self._waypoint_buffer.popleft()[0]
                else:
                    self._waypoint_buffer.popleft()

        if debug:
            draw_waypoints(self._vehicle.get_world(),
                           [look_ahead_loc], 1.0)
        return control

    def _buffer_waypoints(self, debug=False):
        """Put waypoints into the buffer."""
        num_waypoints_to_add = self._buffer_size - len(self._waypoint_buffer)
        for _ in range(num_waypoints_to_add):
            if self.waypoints_queue:
                next_waypoint = self.waypoints_queue.popleft()
                self._waypoint_buffer.append(next_waypoint)
                if debug:
                    carla_world = self._vehicle.get_world()
                    carla_world.debug.draw_line(next_waypoint[0].transform.location,
                                                next_waypoint[0].transform.location +
                                                carla.Location(z=0.5),
                                                color=carla.Color(255, 0, 255))
            else:
                break

    def waypoints_in_buffer(self):
        """True if waypoints exist in buffer."""
        return len(self._waypoint_buffer) > 0

    def _get_projection(self):
        """Get the projection of current vehicle position between prev and target waypoints.

        Returns:
            carla.Location: Location of the current projection point.
            numpy.ndarray: 3D vector formed by the prev and incoming waypoints.
        """
        # Vector between prev and target waypoints
        waypt_location_diff = self.target_waypoint.transform.location - \
            self._prev_waypoint.transform.location
        vec_waypoints = np.array(
            [waypt_location_diff.x, waypt_location_diff.y, waypt_location_diff.z])

        # Vector between prev waypoint and current vehicle's location
        veh_location_diff = self._vehicle.get_location(
        ) - self._prev_waypoint.transform.location
        vec_vehicle = np.array(
            [veh_location_diff.x, veh_location_diff.y, veh_location_diff.z])

        proj_pt = vec_vehicle @ vec_waypoints * \
            vec_waypoints / np.linalg.norm(vec_waypoints)**2

        prev_waypt_loc = self._prev_waypoint.transform.location

        proj_loc = carla.Location(proj_pt[0] + prev_waypt_loc.x,
                                  proj_pt[1] + prev_waypt_loc.y,
                                  proj_pt[2] + prev_waypt_loc.z)

        return proj_loc, vec_waypoints

    def _get_look_ahead_location(self, look_ahead):
        """Get location of look ahead point along path formed by waypoints."""
        proj_loc, vec_waypoints = self._get_projection()
        target_loc = self.target_waypoint.transform.location

        # Distance between current projection point and the incoming target waypoint
        dist_to_next_waypoint = proj_loc.distance(target_loc)

        if look_ahead <= dist_to_next_waypoint:
            # Vector scaled by look ahead distance
            vec_look_ahead = vec_waypoints / \
                np.linalg.norm(vec_waypoints) * look_ahead

            look_ahead_location = carla.Location(proj_loc.x + vec_look_ahead[0],
                                                 proj_loc.y +
                                                 vec_look_ahead[1],
                                                 proj_loc.z + vec_look_ahead[2])

        else:
            # Loop over buffered waypoints to find the section where the look ahead point
            # lies in, then compute the location of the look ahead point
            idx = 0
            while True:
                # If out of waypoints in buffer, use the last waypoint in buffer as look ahead point
                if idx+2 > len(self._waypoint_buffer):
                    look_ahead_location = self._waypoint_buffer[-1][0].transform.location
                    break

                # Comput look ahead distacne in the current section
                look_ahead -= dist_to_next_waypoint

                # Vector formed by start and end waypoints of the current section
                vec = self._waypoint_buffer[idx+1][0].transform.location - \
                    self._waypoint_buffer[idx][0].transform.location
                vec = np.array([vec.x, vec.y, vec.z])

                dist_to_next_waypoint = np.linalg.norm(vec)

                # If look ahead distance exceeds length of current section, go to next one
                if look_ahead > dist_to_next_waypoint:
                    idx += 1
                    continue

                # Section found, compute look ahead location
                else:
                    # Vector scaled by remaining look ahead distance
                    vec_look_ahead = vec / dist_to_next_waypoint * look_ahead

                    base_waypt_loc = self._waypoint_buffer[idx][0].transform.location
                    look_ahead_location = carla.Location(
                        base_waypt_loc.x + vec_look_ahead[0],
                        base_waypt_loc.y + vec_look_ahead[1],
                        base_waypt_loc.z + vec_look_ahead[2])
                    break

        return look_ahead_location
