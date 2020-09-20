# %%
# The following boilerplate is required if .egg is not installed
# See: https://carla.readthedocs.io/en/latest/build_system/
import glob
import os
import datetime
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
import matplotlib.pyplot as plt
import queue
from groundtruth import GroundTruthExtractor
from carlatform import CarlaW2ETform
from math import pi
import pickle

# %% ================= Global function =================


def find_weather_presets():
    """ Method to find weather presets. """
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters)
               if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

# %% ================= Geo2Location =================

# TODO: Replace this with pyproj?


class Geo2Location(object):
    """
    Helper class for homogeneous transform from geolocation 

    This class is used by GNSS class to transform from carla.GeoLocation to carla.Location.
    This transform is not provided by Carla, but it can be solved using 4 chosen points.
    Note that carla.Location is in the left-handed coordinate system.
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
    """ Class representing the surrounding environment. """

    def __init__(self, carla_world, traffic_manager, config_args, spawn_point=None):
        """ 
        Constructor method. 

        If spawn_point not given, choose random spawn point recommended by the map.
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
        self.depth_camera = None

        self.ground_truth = None

        # Start simuation
        self.restart(config_args, spawn_point)
        # Tick the world to bring the actors into effect
        self.step_forward()

    def restart(self, config_args, spawn_point=None):
        """
        Start the simulation with the configuration arguments.

        It spawn the actors including ego vehicle and sensors. If the ego vehicle exists already,
        it respawn the vehicle either at the same location or at the designated location.
        """
        # Set up carla engine using config
        settings = self.carla_world.get_settings()
        settings.no_rendering_mode = config_args['world']['no_rendering']
        settings.synchronous_mode = config_args['world']['sync_mode']
        settings.fixed_delta_seconds = config_args['world']['delta_seconds']
        self.carla_world.apply_settings(settings)

        # Spawn a Mustang as the ego vehicle (not stolen from John Wick, don't panic)
        ego_veh_bp = self.carla_world.get_blueprint_library().find('vehicle.mustang.mustang')

        if self.ego_veh:
            if spawn_point is None:
                print("Respawning ego vehicle.")
                spawn_point = self.ego_veh.get_transform()
            else:
                print("Respawning ego vehicle at assigned point.")
            # Destroy previously spawned actors
            self.destroy()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.ego_veh = self.carla_world.try_spawn_actor(
                ego_veh_bp, spawn_point)
            if self.ego_veh is None:
                print('Chosen spawn point failed.')

        else:
            if spawn_point:
                print("Spawning new ego vehicle at assigned point.")
                spawn_point.location.z += 2.0
                self.ego_veh = self.carla_world.try_spawn_actor(
                    ego_veh_bp, spawn_point)

        while self.ego_veh is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                sys.exit(1)
            print("Spawning new ego vehicle at a random point.")
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
            self.ego_veh, config_args['sensor']['front_camera'])
        self.depth_camera = DepthCamera(
            self.ego_veh, config_args['sensor']['front_camera'])

        # Ground truth extractor
        self.ground_truth = GroundTruthExtractor(
            self.ego_veh, self.map, actor_list=None, config_args=config_args)

    def set_ego_autopilot(self, active, autopilot_config_args=None):
        """ Set traffic manager and register ego vehicle to it. """
        if autopilot_config_args:
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

        It only allows lane changes in the possible direction. 
        Performing a left lane change on the left-most lane is not possible.
        Carla's traffic manager doesn't seem to make car change to a left turn lane in built-in town (tested in Town03)
        """
        # carla uses true for right
        self.tm.force_lane_change(self.ego_veh, not to_left)

    def step_forward(self):
        """ Tick carla world to take simulation one step forward. """
        self.carla_world.tick()
        self.imu.update()
        self.gnss.update()
        self.semantic_camera.update()
        self.depth_camera.update()
        self.ground_truth.update()

    def see_ego_veh(self, following_dist=5, height=5, tilt_ang=-30):
        """ Aim the spectator down to the ego vehicle. """
        spect_location = carla.Location(x=-following_dist)
        self.ego_veh.get_transform().transform(
            spect_location)  # it modifies passed-in location
        ego_rotation = self.ego_veh.get_transform().rotation
        self.spectator.set_transform(carla.Transform(spect_location + carla.Location(z=height),
                                                     carla.Rotation(pitch=tilt_ang, yaw=ego_rotation.yaw)))

    def allow_free_run(self):
        """ Allow carla engine to run asynchronously and freely. """
        settings = self.carla_world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = 0.0
        self.carla_world.apply_settings(settings)

    def destroy(self):
        """ Destroy spawned actors in carla world. """
        if self.ego_veh:
            print("Destroying the ego vehicle.")
            self.ego_veh.destroy()
            self.ego_veh = None
        if self.imu:
            print("Destroying IMU sensor.")
            self.imu.destroy()
            self.imu = None
        if self.gnss:
            print("Destroying gnss sensor.")
            self.gnss.destroy()
            self.gnss = None
        if self.semantic_camera:
            print("Destroying semantic camera sensor.")
            self.semantic_camera.destroy()
            self.semantic_camera = None
        if self.depth_camera:
            print("Destroying depth camera sensor.")
            self.depth_camera.destroy()
            self.depth_camera = None

# %% ================= Sensor Base =================


class CarlaSensor(object):
    """ Base class for sensors provided by carla. """

    def __init__(self, parent_actor):
        """ Constructor method. """
        self.sensor = None
        self.timestamp = 0.0
        self._parent = parent_actor
        # The callback method in listen() to retrieve data used widely in official tutorials has a data race problem.
        # The callback will likely not finish before data get accessed from the main loop, causing inconsistent data.
        # Here the queue is expected to be used in listen() instead. The callback simply puts the sensor data into the queue,
        # then the data can be obtained in update() using get() which blocks and make sure synchronization.
        self._queue = queue.Queue()

    def update(self):
        """ Wait for sensro event to be put in queue and update data. """
        raise NotImplementedError()

    def destroy(self):
        """ Destroy sensor actor. """
        if self.sensor:
            self.sensor.destroy()
            self.sensor = None


# %% ================= IMU Sensor =================


class IMU(CarlaSensor):
    """ 
    Class for IMU sensor. 

    This class wraps the IMU sensor provided by Carla that gives accelerations and angular velocities in each axis.
    To fulfill the need for also velocities, the actor's velocities are extracted. Experiments show that actor's dynamics
    lag the IMU sensor by one simulation tick, so a compensation using the acceleration multiplied with the time step is added.

        Important note:
        Carla uses left-handed (x-forward, y-right, z-up) coordinate system for locations 
        and right-handed z-down (airplane-like) coordinate system for rotations (roll-pitch-yaw).
        This wrapper class automatically convert to right-handed z-up coordinate system to match our convention.
    """

    def __init__(self, parent_actor, imu_config_args):
        """ Constructor method. """
        super().__init__(parent_actor)
        # In right-handed z-up coordinate system
        # Accelerations
        self.accel_x = 0.0      # m/s^2
        self.accel_y = 0.0      # m/s^2
        self.accel_z = 0.0      # m/s^2
        # Angular velocities
        self.gyro_x = 0.0       # rad/s
        self.gyro_y = 0.0       # rad/s
        self.gyro_z = 0.0       # rad/s

        # Velocities (virtual odometry)
        self.vx = 0.0           # m/s
        self.vy = 0.0           # m/s
        # Virtual odometry uses velocities of ego vehicle's actor directly,
        # which is found to lag behind Carla's IMU by 1 simulation step.
        # To recover that, virtual odometry's velocities are added with acceleration times simulation step
        self._delta_seconds = imu_config_args['delta_seconds']

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

        self._noise_vx_bias = imu_config_args['noise_vx_bias']
        self._noise_vy_bias = imu_config_args['noise_vy_bias']
        self._noise_vx_stddev = imu_config_args['noise_vx_stddev']
        self._noise_vy_stddev = imu_config_args['noise_vy_stddev']

        print("Spawning IMU sensor.")
        self.sensor = world.spawn_actor(imu_bp,
                                        carla.Transform(carla.Location(
                                            x=imu_config_args['pos_x'], z=0.0)),
                                        attach_to=self._parent)

        self.sensor.listen(lambda event: self._queue.put(event))

    def update(self):
        """ Wait for IMU measurement and update data. """
        # get() blocks the script so synchronization is guaranteed
        event = self._queue.get()

        # Convert to right-handed z-up frame
        self.timestamp = event.timestamp
        self.accel_x = event.accelerometer.x
        self.accel_y = - event.accelerometer.y
        self.accel_z = event.accelerometer.z
        self.gyro_x = event.gyroscope.x
        self.gyro_y = - event.gyroscope.y
        self.gyro_z = - event.gyroscope.z

        # Velocities
        vel = self._parent.get_velocity()
        tform_w2e = CarlaW2ETform(self._parent.get_transform())
        # Transform velocities from Carla world frame (left-handed) to ego frame (right-handed)
        ego_vel = tform_w2e.rotm_world_to_ego(vel)  # an np 3D vector
        self.vx = ego_vel[0] + self._delta_seconds * self.accel_x
        self.vy = ego_vel[1] + self._delta_seconds * self.accel_y
        self._add_velocity_noise()

    def _add_velocity_noise(self):
        self.vx += np.random.normal(self._noise_vx_bias, self._noise_vx_stddev)
        self.vy += np.random.normal(self._noise_vy_bias, self._noise_vy_stddev)


# %% ================= GNSS Sensor =================


class GNSS(CarlaSensor):
    """ 
    Class for GNSS sensor.

    Carla uses left-handed coordinate system  while we use right-handed coordinate system.
    Ref: https://subscription.packtpub.com/book/game_development/9781784394905/1/ch01lvl1sec18/the-2d-and-3d-coordinate-systems
    This class already converts the GNSS measurements into our right-handed z-up coordinate system.
    """

    def __init__(self, parent_actor, gnss_config_args):
        """ Constructor method. """
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
        self.sensor = world.spawn_actor(gnss_bp,
                                        carla.Transform(carla.Location(
                                            x=gnss_config_args['pos_x'], z=0.0)),
                                        attach_to=self._parent)
        self._geo2location = Geo2Location(world.get_map())

        self.sensor.listen(lambda event: self._queue.put(event))

    def update(self):
        """ Wait for GNSS measurement and update data. """
        # get() blocks the script so synchronization is guaranteed
        event = self._queue.get()
        self.timestamp = event.timestamp
        self.lat = event.latitude
        self.lon = event.longitude
        self.alt = event.altitude

        # Get transform from geolocation to location
        location = self._geo2location.transform(
            carla.GeoLocation(self.lat, self.lon, self.alt))

        # y must be flipped to match the right-handed convention we use.
        self.x = location.x
        self.y = - location.y
        self.z = location.z

# %% ================= Semantic Camera =================


class SemanticCamera(CarlaSensor):
    """ Class for semantic camera. """

    def __init__(self, parent_actor, ss_cam_config_args):
        """ Constructor method. """
        super().__init__(parent_actor)
        self.ss_img = None

        world = self._parent.get_world()
        ss_cam_bp = world.get_blueprint_library().find(
            'sensor.camera.semantic_segmentation')
        ss_cam_bp.set_attribute('image_size_x', ss_cam_config_args['res_h'])
        ss_cam_bp.set_attribute('image_size_y', ss_cam_config_args['res_v'])
        ss_cam_bp.set_attribute('fov', ss_cam_config_args['fov'])

        print("Spawning semantic camera sensor.")
        self.sensor = world.spawn_actor(ss_cam_bp,
                                        carla.Transform(
                                            carla.Location(x=ss_cam_config_args['pos_x'], z=ss_cam_config_args['pos_z'])),
                                        attach_to=self._parent)

        self.sensor.listen(lambda image: self._queue.put(image))

    def update(self):
        """ Wait for semantic image and update data. """
        image = self._queue.get()
        self.timestamp = image.timestamp
        np_img = np.frombuffer(image.raw_data, dtype=np.uint8)
        # Reshap to BGRA format
        np_img = np.reshape(np_img, (image.height, image.width, -1))
        # Semantic info is stored only in the R channel
        # Since np_img is from the buffer, which is reused by Carla
        # Making a copy makes sure ss_img is not subject to side-effect when the underlying buffer is modified
        self.ss_img = np_img[:, :, 2].copy()

# %% ================= Depth Camera =================


class DepthCamera(CarlaSensor):
    """ Class for semantic camera. """

    def __init__(self, parent_actor, depth_cam_config_args):
        """ Constructor method. """
        super().__init__(parent_actor)
        self.depth_buffer = None

        world = self._parent.get_world()
        depth_cam_bp = world.get_blueprint_library().find(
            'sensor.camera.depth')
        depth_cam_bp.set_attribute(
            'image_size_x', depth_cam_config_args['res_h'])
        depth_cam_bp.set_attribute(
            'image_size_y', depth_cam_config_args['res_v'])
        depth_cam_bp.set_attribute('fov', depth_cam_config_args['fov'])

        print("Spawning semantic camera sensor.")
        self.sensor = world.spawn_actor(depth_cam_bp,
                                        carla.Transform(
                                            carla.Location(x=depth_cam_config_args['pos_x'], z=depth_cam_config_args['pos_z'])),
                                        attach_to=self._parent)

        self.sensor.listen(lambda image: self._queue.put(image))

    def update(self):
        """ Wait for semantic image and update data. """
        image = self._queue.get()
        self.timestamp = image.timestamp
        np_img = np.frombuffer(image.raw_data, dtype=np.uint8)
        # Reshap to BGRA format
        # The depth info is encoded by the BGR channels using the so-called depth buffer.
        # Decoding is required before use.
        # Ref: https://carla.readthedocs.io/en/latest/ref_sensors/#depth-camera
        # Since np_img is from the buffer, which is reused by Carla
        # Making a copy makes sure depth_buffer is not subject to side-effect when the underlying buffer is modified
        np_img = np.reshape(np_img, (image.height, image.width, -1))
        self.depth_buffer = np_img.copy()

# TODO: stop sign measurement

# %%
def main():
    # Parse passed-in config yaml file
    argparser = argparse.ArgumentParser(
        description='CARLA Roaming Data Collector')
    argparser.add_argument('config', type=argparse.FileType(
        'r'), help='configuration yaml file for carla env setup')
    argparser.add_argument('-r', '--record', default=False,
                           action='store_true', help='record collected data')
    args = argparser.parse_args()

    # Read configurations from yaml file to config_args
    with args.config as config_file:
        config_args = yaml.safe_load(config_file)

    # Initialize world
    world = None
    spawn_point = None
    # Assign spawn point for ego vehicle
    spawn_point = carla.Transform(carla.Location(
        229.67, 80.99, 0.59), carla.Rotation(yaw=91.39))
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        # Create a World obj with a built-in map
        world = World(client.load_world(
            config_args['world']['map']), client.get_trafficmanager(), config_args, spawn_point=spawn_point)

        # Launch autopilot for ego vehicle
        world.set_ego_autopilot(True, config_args['autopilot'])

        n_ticks = int(config_args['sim_duration'] /
                      config_args['world']['delta_seconds'])

        ss_images = []
        vx = []
        yaw_rate = []
        in_junction = []

        # Simulation loop
        to_left = True
        for idx in range(n_ticks):
            world.step_forward()
            world.see_ego_veh()

            ss_images.append(world.semantic_camera.ss_img)
            vx.append(world.imu.vx)
            yaw_rate.append(world.imu.gyro_z)
            in_junction.append(world.ground_truth.in_junction)

            print('vx: {:3.2f}, vy: {:3.2f}, w: {:3.2f}'.format(
                world.imu.vx, world.imu.vy, world.imu.gyro_z * 180 / pi))
            print('in junction: {}'.format(world.ground_truth.in_junction))
            # c0
            print('{}   {}   {}   {}'.format(
                world.ground_truth.next_left_marking_param[0] if world.ground_truth.next_left_marking else None,
                world.ground_truth.left_marking_param[0] if world.ground_truth.left_marking else None,
                world.ground_truth.right_marking_param[0] if world.ground_truth.right_marking else None,
                world.ground_truth.next_right_marking_param[0] if world.ground_truth.next_right_marking else None))
            # Marking type
            print('{}   {}   {}   {}'.format(
                world.ground_truth.next_left_marking.type if world.ground_truth.next_left_marking else None,
                world.ground_truth.left_marking.type if world.ground_truth.left_marking else None,
                world.ground_truth.right_marking.type if world.ground_truth.right_marking else None,
                world.ground_truth.next_right_marking.type if world.ground_truth.next_right_marking else None))

            if (idx+1) % int(3/config_args['world']['delta_seconds']) == 0:
                world.force_lane_change(to_left=to_left)
                to_left = not to_left

        # Store data
        if args.record:
            mydir = os.path.join(os.getcwd(), 'recordings',
                                 datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            os.makedirs(mydir)

            with open(os.path.join(mydir, 'ss_images'), 'wb') as image_file:
                pickle.dump(ss_images, image_file)
            with open(os.path.join(mydir, 'vx'), 'wb') as vx_file:
                pickle.dump(vx, vx_file)
            with open(os.path.join(mydir, 'yaw_rate'), 'wb') as yaw_rate_file:
                pickle.dump(yaw_rate, yaw_rate_file)
            with open(os.path.join(mydir, 'in_junction'), 'wb') as in_junction_file:
                pickle.dump(in_junction, in_junction_file)

    finally:
        if world:
            world.set_ego_autopilot(False)
            world.destroy()
            # Allow carla engine to run freely so it doesn't just hang there
            world.allow_free_run()


# %%
if __name__ == '__main__':
    main()
