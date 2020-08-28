# %%
# The following boilerplate is required if .egg is not installed
# See: https://carla.readthedocs.io/en/latest/build_system/
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

import argparse
import yaml
import carla
import re
import random
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import queue

# %% ================= Global function =================


def find_weather_presets():
    """ Method to find weather presets """
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters)
               if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

# %% ================= Geo2Location =================

# TODO: Replace this with pyproj?
class Geo2Location(object):
    """
    Helper class for homogeneous transform from geolocation used by gnss to Cartisian location.
    This transform is not provided by Carla, but it can be solved using 4 chosen points.
    """

    def __init__(self, carla_map):
        """ Constructor method """
        self._map = carla_map
        # Pick 4 points of carla.Location
        loc1 = carla.Location(0, 0, 0)
        loc2 = carla.Location(1, 0, 0)
        loc3 = carla.Location(0, 1, 0)
        loc4 = carla.Location(0, 0, 1)
        # Get the corresponding carla.GeoLocation points using carla's transform_to_geolocation()
        geoloc1 = self._map.transform_to_geolocation(loc1)
        geoloc2 = self._map.transform_to_geolocation(loc2)
        geoloc3 = self._map.transform_to_geolocation(loc3)
        geoloc4 = self._map.transform_to_geolocation(loc4)
        # Solve the transform from geolocation to location (geolocation_to_location)
        l = np.array([[loc1.x, loc2.x, loc3.x, loc4.x],
                      [loc1.y, loc2.y, loc3.y, loc4.y],
                      [loc1.z, loc2.z, loc3.z, loc4.z],
                      [1, 1, 1, 1]], dtype=np.float)
        g = np.array([[geoloc1.latitude, geoloc2.latitude, geoloc3.latitude, geoloc4.latitude],
                      [geoloc1.longitude, geoloc2.longitude,
                          geoloc3.longitude, geoloc4.longitude],
                      [geoloc1.altitude, geoloc2.altitude,
                          geoloc3.altitude, geoloc4.altitude],
                      [1, 1, 1, 1]], dtype=np.float)
        # Tform = (G*L^-1)^-1
        self._tform = np.linalg.inv(g.dot(np.linalg.inv(l)))

    def transform(self, geolocation):
        """ 
        Transform from carla.GeoLocation to carla.Location.
        Numerical error may exist. Experiments show error is about under 1 cm in Town03.
        """
        geoloc = np.array(
            [geolocation.latitude, geolocation.longitude, geolocation.altitude, 1])
        loc = self._tform.dot(geoloc.T)
        return carla.Location(loc[0], loc[1], loc[2])

    def get_matrix(self):
        """ Get the 4-by-4 transform matrix """
        return self._tform


# %% ================= World =================


class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, traffic_manager, config_args, spawn_point=None):
        """ 
        Constructor method. 
        If spawn_point not given, choose random spawn point recommended by carla.
        """
        self.carla_world = carla_world
        self.tm = traffic_manager
        self.spectator = carla_world.get_spectator()
        try:
            self.map = self.carla_world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print(
                '  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self._weather_presets = find_weather_presets()
        self._weather_index = config_args['world']['weather']
        self.carla_world.set_weather(
            self._weather_presets[self._weather_index][0])
        self.ego_veh = None
        self.imu = None
        self.gnss = None
        self.semantic_camera = None
        self.virtual_odom = None

        self.ground_truth = None

        # Start simuation
        self.restart(config_args, spawn_point)
        # Tick the world to bring the actors into effect
        self.step_forward()

    def restart(self, config_args, spawn_point=None):
        # Set up carla engine using config
        settings = self.carla_world.get_settings()
        settings.no_rendering_mode = config_args['world']['no_rendering']
        settings.synchronous_mode = config_args['world']['sync_mode']
        settings.fixed_delta_seconds = config_args['world']['delta_seconds']
        self.carla_world.apply_settings(settings)

        # Spawn the ego vehicle as a cool mustang
        ego_veh_bp = self.carla_world.get_blueprint_library().find('vehicle.mustang.mustang')
        print("Spawning the ego vehicle.")
        if self.ego_veh is not None:
            # Destroy previously spawned actors
            self.destroy()
            if spawn_point is None:
                spawn_point = self.ego_veh.get_transform()
                spawn_point.location.z += 2.0
                spawn_point.rotation.roll = 0.0
                spawn_point.rotation.pitch = 0.0
                self.ego_veh = self.carla_world.try_spawn_actor(
                    ego_veh_bp, spawn_point)
            else:
                self.ego_veh = self.carla_world.try_spawn_actor(
                    ego_veh_bp, spawn_point)
                if self.ego_veh is None:
                    print('Chosen spawn transform failed.')

        while self.ego_veh is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(
                spawn_points) if spawn_points else carla.Transform()
            self.ego_veh = self.carla_world.try_spawn_actor(
                ego_veh_bp, spawn_point)

        # Point the spectator to the ego vehicle
        self.see_ego_veh()

        # Spawn the sensors
        self.gnss = GNSS(self.ego_veh, config_args['sensor']['gnss'])
        self.imu = IMU(self.ego_veh, config_args['sensor']['imu'])
        self.semantic_camera = SemanticCamera(
            self.ego_veh, config_args['sensor']['semantic_image'])
        self.virtual_odom = VirtualOdometry(
            self.ego_veh, config_args['sensor']['virtual_odom'])

        # Ground truth extractor
        self.ground_truth = GroundTruthExtractor(
            self.map, self.ego_veh, actor_list=None, config_args=config_args)

    def set_ego_autopilot(self, active, autopilot_config_args=None):
        """ Set traffic manager and register ego vehicle to it """
        if autopilot_config_args is not None:
            self.tm.auto_lane_change(
                self.ego_veh, autopilot_config_args['auto_lane_change'])
            self.tm.ignore_lights_percentage(
                self.ego_veh, autopilot_config_args['ignore_lights_percentage'])
            self.tm.vehicle_percentage_speed_difference(
                self.ego_veh, autopilot_config_args['vehicle_percentage_speed_difference'])
        self.ego_veh.set_autopilot(active, self.tm.get_port())

    def force_lane_change(self, to_left):
        """ 
        Force ego vehicle to change the lane regardless collision with other vehicles. 
        However, it only allows lane changes in the possible direction. 
        Performing a left lane change on the left-most lane is not possible.
        """
        # carla uses true for right
        self.tm.force_lane_change(self.ego_veh, not to_left)

    def step_forward(self):
        """ Tick carla world to take simulation one step forward """
        self.carla_world.tick()
        self.imu.update()
        self.gnss.update()
        self.semantic_camera.update()
        self.virtual_odom.update()
        self.ground_truth.update()
    
    def see_ego_veh(self, following_dist=5, height=5, tilt_ang=-30):
        """ Aim the spectator down to the ego vehicle """
        spect_location = carla.Location(x=-following_dist)
        self.ego_veh.get_transform().transform(spect_location)  # it modifies passed-in location
        ego_rotation = self.ego_veh.get_transform().rotation
        self.spectator.set_transform(carla.Transform(spect_location + carla.Location(z=height),
                                                     carla.Rotation(pitch=tilt_ang, yaw=ego_rotation.yaw)))


    def allow_free_run(self):
        """ Allow carla engine to run asynchronously and freely """
        settings = self.carla_world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = 0.0
        self.carla_world.apply_settings(settings)

    def destroy(self):
        """ Destroy spawned actors in carla world """
        if self.ego_veh is not None:
            print("Destroying the ego vehicle.")
            self.ego_veh.destroy()
            self.ego_veh = None
        if self.imu is not None:
            print("Destroying IMU sensor.")
            self.imu.destroy()
            self.imu = None
        if self.gnss is not None:
            print("Destroying gnss sensor.")
            self.gnss.destroy()
            self.gnss = None
        if self.semantic_camera is not None:
            print("Destroying semantic camera sensor.")
            self.semantic_camera.destroy()
            self.semantic_camera = None

# %% ================= Sensor Base =================


class CarlaSensor(object):
    """ Base class for sensors provided by carla """

    def __init__(self, parent_actor):
        """ Constructor method """
        self.sensor = None
        self.timestamp = 0.0
        self._parent = parent_actor
        # The callback method in listen() to retrieve data used widely in official tutorials has a data race problem.
        # The callback will likely not finish before data get accessed from the main loop, causing inconsistent data.
        # Here the queue is expected to be used in listen() instead. The callback simply puts the sensor data into the queue,
        # then the data can be obtained in update() using get() which blocks and make sure synchronization.
        self._queue = queue.Queue()

    def update(self):
        """ Wait for sensro event to be put in queue and update data """
        raise NotImplementedError()

    def destroy(self):
        """ Destroy sensor actor """
        if self.sensor is not None:
            self.sensor.destroy()
            self.sensor = None


# %% ================= IMU Sensor =================


class IMU(CarlaSensor):
    """ Class for IMU sensor"""

    def __init__(self, parent_actor, imu_config_args):
        """ Constructor method """
        super().__init__(parent_actor)
        self.accelerometer = None
        self.gyro = None

        world = self._parent.get_world()
        imu_bp = world.get_blueprint_library().find('sensor.other.imu')

        imu_bp.set_attribute('noise_accel_stddev_x',
                             imu_config_args['noise_accel_stddev_x'])
        imu_bp.set_attribute('noise_accel_stddev_y',
                             imu_config_args['noise_accel_stddev_y'])
        imu_bp.set_attribute('noise_accel_stddev_z',
                             imu_config_args['noise_accel_stddev_z'])
        imu_bp.set_attribute('noise_gyro_bias_x',
                             imu_config_args['noise_gyro_bias_x'])
        imu_bp.set_attribute('noise_gyro_bias_y',
                             imu_config_args['noise_gyro_bias_y'])
        imu_bp.set_attribute('noise_gyro_bias_z',
                             imu_config_args['noise_gyro_bias_z'])
        imu_bp.set_attribute('noise_gyro_stddev_x',
                             imu_config_args['noise_gyro_stddev_x'])
        imu_bp.set_attribute('noise_gyro_stddev_y',
                             imu_config_args['noise_gyro_stddev_y'])
        imu_bp.set_attribute('noise_gyro_stddev_z',
                             imu_config_args['noise_gyro_stddev_z'])

        print("Spawning IMU sensor.")
        self.sensor = world.spawn_actor(imu_bp, carla.Transform(carla.Location(x=0.0, z=0.0)),
                                        attach_to=self._parent)

        self.sensor.listen(lambda event: self._queue.put(event))

    def update(self):
        """ Wait for IMU measurement and update data """
        # get() blocks the script so synchronization is guaranteed
        event = self._queue.get()
        self.timestamp = event.timestamp
        self.accelerometer = event.accelerometer
        self.gyro = event.gyroscope


# %% ================= GNSS Sensor =================


class GNSS(CarlaSensor):
    """ Class for GNSS sensor"""

    def __init__(self, parent_actor, gnss_config_args):
        """ Constructor method """
        super().__init__(parent_actor)
        self.lat = 0.0
        self.lon = 0.0
        self.alt = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        world = self._parent.get_world()
        gnss_bp = world.get_blueprint_library().find('sensor.other.gnss')

        gnss_bp.set_attribute(
            'noise_alt_bias', gnss_config_args['noise_alt_bias'])
        gnss_bp.set_attribute('noise_alt_stddev',
                              gnss_config_args['noise_alt_stddev'])
        gnss_bp.set_attribute(
            'noise_lat_bias', gnss_config_args['noise_lat_bias'])
        gnss_bp.set_attribute('noise_lat_stddev',
                              gnss_config_args['noise_lat_stddev'])
        gnss_bp.set_attribute(
            'noise_lon_bias', gnss_config_args['noise_lon_bias'])
        gnss_bp.set_attribute('noise_lon_stddev',
                              gnss_config_args['noise_lon_stddev'])

        print("Spawning GNSS sensor.")
        self.sensor = world.spawn_actor(gnss_bp, carla.Transform(carla.Location(x=0.0, z=0.0)),
                                        attach_to=self._parent, attachment_type=carla.AttachmentType.Rigid)
        self._geo2location = Geo2Location(world.get_map())

        self.sensor.listen(lambda event: self._queue.put(event))

    def update(self):
        """ Wait for GNSS measurement and update data """
        # get() blocks the script so synchronization is guaranteed
        event = self._queue.get()
        self.timestamp = event.timestamp
        self.lat = event.latitude
        self.lon = event.longitude
        self.alt = event.altitude

        # Get transform from geolocation to location
        location = self._geo2location.transform(
            carla.GeoLocation(self.lat, self.lon, self.alt))
        self.x = location.x
        self.y = location.y
        self.z = location.z

# %% ================= Semantic Camera =================


class SemanticCamera(CarlaSensor):
    """ Class for semantic camera """

    def __init__(self, parent_actor, ss_cam_config_args):
        """ Constructor method"""
        super().__init__(parent_actor)
        self.lane_img = None
        self.pole_img = None

        world = self._parent.get_world()
        ss_cam_bp = world.get_blueprint_library().find(
            'sensor.camera.semantic_segmentation')
        ss_cam_bp.set_attribute('image_size_x', ss_cam_config_args['res_x'])
        ss_cam_bp.set_attribute('image_size_y', ss_cam_config_args['res_y'])
        ss_cam_bp.set_attribute('fov', ss_cam_config_args['fov'])

        print("Spawning semantic camera sensor.")
        self.sensor = world.spawn_actor(ss_cam_bp, carla.Transform(carla.Location(x=0.6, z=1.5)),
                                        attach_to=self._parent)

        self.sensor.listen(lambda image: self._queue.put(image))

    def update(self):
        """ Wait for semantic image and update data """
        image = self._queue.get()
        self.timestamp = image.timestamp
        np_img = np.frombuffer(image.raw_data, dtype=np.uint8)
        # Reshap to BGRA format
        # Semantic info is stored only in the R channel
        np_img = np.reshape(np_img, (image.height, image.width, -1))
        # Lane lines and sidewalks are considered
        self.lane_img = (np_img[:, :, 2] == 6) | (np_img[:, :, 2] == 8)
        # Pole-like objects
        self.pole_img = np_img[:, :, 2] == 5

# %% ================= Simulated Odometry =================


class VirtualOdometry(object):
    """ 
    Class for virtual velocity and yaw rate measurement.
    This is done by adding noise to ego vehilce's velocities in Carla.
    """

    def __init__(self, parent_actor, virtual_odom_config):
        """ Constructor method """
        self._parent = parent_actor
        self._noise_vx_bias = virtual_odom_config['noise_vx_bias']
        self._noise_vy_bias = virtual_odom_config['noise_vy_bias']
        self._noise_yaw_rate_bias = virtual_odom_config['noise_yaw_rate_bias']
        self._noise_vx_stddev = virtual_odom_config['noise_vx_stddev']
        self._noise_vy_stddev = virtual_odom_config['noise_vy_stddev']
        self._noise_yaw_rate_stddev = virtual_odom_config['noise_yaw_rate_stddev']

        self.vx = 0.0    # m/s
        self.vy = 0.0    # m/s
        self.yaw_rate = 0.0  # deg/s

    def update(self):
        """ Update virtual odometry """
        vel = self._parent.get_velocity()
        # Carla uses SAE coordinate system (z towards down)
        # Ref: https://carla.readthedocs.io/en/latest/python_api/#carlarotation
        # Convert to ISO coordinate system when building up the vector
        vel_vec = np.array([vel.x, -vel.y, -vel.z]).T

        rotation = self._parent.get_transform().rotation
        # Also convert to ISO coordinate system when creating rotation matrix
        tform_ego2world = Rotation.from_euler(
            'zyx', [-rotation.yaw, -rotation.pitch, rotation.roll], degrees=True).as_matrix().T

        ego_vel = tform_ego2world.dot(vel_vec)
        self.vx = ego_vel[0]
        self.vy = ego_vel[1]
        self.yaw_rate = -self._parent.get_angular_velocity().z
        self._add_noise()

    def _add_noise(self):
        self.vx += np.random.normal(self._noise_vx_bias, self._noise_vx_stddev)
        self.vy += np.random.normal(self._noise_vy_bias, self._noise_vy_stddev)
        self.yaw_rate += np.random.normal(self._noise_yaw_rate_bias,
                                          self._noise_yaw_rate_stddev)


# TODO: stop sign measurement

# %% ================= Ground Truth  =================


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
        self.waypoint = None
        self.left_waypoint = None
        self.right_waypoint = None
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
        self.fbumper_location = self.ego_veh_tform.transform(
            carla.Location(x=self.raxle_to_fbumper - 1.4))
        self.waypoint = self.map.get_waypoint(self.fbumper_location)
        # TODO: handle waypoint to far from ego location (off road)
        if self.waypoint.lane_type == carla.LaneType.Driving:
            self.left_marking_type = self.waypoint.left_lane_marking.type
            self.right_marking_type = self.waypoint.right_lane_marking.type
            self._get_useful_next_lanes()
            # TODO: lane marking parameters
        else:
            self.left_marking_type = None
            self.right_marking_type = None
            self.left_waypoint = None
            self.right_waypoint = None
            self.next_left_marking_type = None
            self.next_right_marking_type = None

    def _get_useful_next_lanes(self):
        """
        Get next left and right lanes with visible lane boundaries.
        Many transition lanes often without visible lane boundaries (e.g. Shoulder) are defined in OpenDrive.
        This method tries to find the lanes corresponding to visible lane boundaries (e.g. curb) but may not 
        be directly adjacent to current lane.
        """
        left_waypt = self._find_next_visible_lane_marking(to_left=True)
        right_waypt = self._find_next_visible_lane_marking(to_left=False)

        # Updatee next left lane
        if left_waypt is not None:
            self.left_waypoint = left_waypt
            if self.waypoint.lane_id * left_waypt.lane_id >= 0:
                self.next_left_marking_type = left_waypt.left_lane_marking.type
            else:
                self.next_left_marking_type = left_waypt.right_lane_marking.type
        else:
            self.left_waypoint = None
            self.next_left_marking_type = None

        # Update next right lane
        if right_waypt is not None:
            self.right_waypoint = right_waypt
            if self.waypoint.lane_id * right_waypt.lane_id >= 0:
                self.next_right_marking_type = right_waypt.right_lane_marking.type
            else:
                self.next_right_marking_type = right_waypt.left_lane_marking.type
        else:
            self.right_waypoint = None
            self.next_right_marking_type = None

    def _find_next_visible_lane_marking(self, to_left=True):
        """
        Helper method for finding the waypoint with the visible next lane marking.
        """
        curr_lane_id = self.waypoint.lane_id
        if to_left:
            curr_next_waypt = self.waypoint.get_left_lane()
        else:
            curr_next_waypt = self.waypoint.get_right_lane()

        # Search until visible lane marking found
        while True:
            # Return if next waypoint doesn't exist
            if curr_next_waypt is None:
                return curr_next_waypt

            # TODO: Remove the following?
            # Do not search across non-drivable space (e.g. across a middle island)
            # If so, set left lane as None
            # Not sure if this strategy is realistic
            # if (curr_next_waypt.lane_type != carla.LaneType.Driving
            #         and curr_next_waypt.lane_type != carla.LaneType.Shoulder
            #         and curr_next_waypt.lane_type != carla.LaneType.Bidirectional
            #         and curr_next_waypt.lane_type != carla.LaneType.Parking):
            #     curr_next_waypt = None
            #     break
            if (curr_next_waypt.lane_type == carla.LaneType.Median
                    and curr_next_waypt.lane_type == carla.LaneType.Sidewalk
                    and curr_next_waypt.lane_type == carla.LaneType.Rail):
                curr_next_waypt = None
                break

            # Check if two lanes have same direction
            if curr_lane_id * curr_next_waypt.lane_id < 0:
                # Start searching using the opposite direction next time since the direction of the lane has changed
                to_left = not to_left

            curr_lane_id = curr_next_waypt.lane_id

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
        return curr_next_waypt

    def _get_poles(self):
        # TODO: use semantic lidar or just actors?
        pass


# %%
def main():
    # Parse passed-in config yaml file
    argparser = argparse.ArgumentParser(
        description='CARLA Roaming Data Collector')
    argparser.add_argument('config', type=argparse.FileType(
        'r'), help='configuration yaml file for carla env setup')
    args = argparser.parse_args()

    # Read configurations from yaml file to config_args
    with args.config as config_file:
        config_args = yaml.safe_load(config_file)

    # Initialize world
    world = None
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        # Create a World obj with a built-in map
        world = World(client.load_world(
            config_args['world']['map']), client.get_trafficmanager(), config_args)

        # Launch autopilot for ego vehicle
        world.set_ego_autopilot(True, config_args['autopilot'])

        n_ticks = int(config_args['sim_duration'] /
                      config_args['world']['delta_seconds'])

        # Simulation loop
        to_left = True
        for idx in range(n_ticks):
            world.step_forward()
            world.see_ego_veh()
            # print('vx: {}'.format(world.virtual_odom.vx))
            # print('vy: {}'.format(world.virtual_odom.vy))
            # print('w: {}'.format(world.virtual_odom.yaw_rate))
            print('{}   {}   {}   {}'.format(
                world.ground_truth.next_left_marking_type,
                world.ground_truth.left_marking_type,
                world.ground_truth.right_marking_type,
                world.ground_truth.next_right_marking_type))

            if idx%int(5/config_args['world']['delta_seconds']) == 0:
                world.force_lane_change(to_left=to_left)
                to_left = not to_left

    finally:
        if world is not None:
            world.set_ego_autopilot(False)
            world.destroy()
            # Allow carla engine to run freely so it doesn't just hang there
            world.allow_free_run()


# %%
if __name__ == '__main__':
    main()
