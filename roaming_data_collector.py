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
import queue
import re
import weakref
import random
import numpy as np
import matplotlib.pyplot as plt

# %% ================= Global function =================


def find_weather_presets():
    """ Method to find weather presets """
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters)
               if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

# %% ================= World =================


class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, config_args):
        """ Constructor """
        self.carla_world = carla_world
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

        # Start simuation
        self.restart(config_args)

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
        self.spectator.set_transform(carla.Transform(spawn_point.location + carla.Location(z=25),
                                                     carla.Rotation(pitch=-90)))

        # Set up the sensors
        self.gnss = GNSS(self.ego_veh, config_args['sensor']['gnss'])
        self.imu = IMU(self.ego_veh, config_args['sensor']['imu'])
        self.semantic_camera = SemanticCamera(
            self.ego_veh, config_args['sensor']['semantic_image'])

        self.carla_world.tick()

    def allow_free_run(self):
        """ Allows carla engine to run asynchronously and freely """
        settings = self.carla_world.get_settings()
        settings.synchronous_mode = False
        self.carla_world.apply_settings(settings)

    def destroy(self):
        pass
        # TODO: destroy actors in world


# %% ================= IMU Sensor =================


class IMU(object):
    """ Class for IMU sensor"""

    def __init__(self, parent_actor, imu_config_args):
        """ Constructor method """
        self.sensor = None
        self._parent = parent_actor
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
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: IMU._on_imu_event(weak_self, event))

    @staticmethod
    def _on_imu_event(weak_self, event):
        """ IMU method """
        self = weak_self()
        if not self:
            return
        self.accelerometer = event.accelerometer
        self.gyro = event.gyroscope

# %% ================= GNSS Sensor =================


class GNSS(object):
    """ Class for GNSS sensor"""

    def __init__(self, parent_actor, gnss_config_args):
        """ Constructor method """
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
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
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: GNSS._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """ GNSS method """
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# %% ================= Semantic Camera =================


class SemanticCamera(object):
    """ Class for semantic camera """

    def __init__(self, parent_actor, ss_cam_config_args):
        """ Constructor method"""
        self.sensor = None
        self._parent = parent_actor
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

        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda image: SemanticCamera._parse_semantic_image(weak_self, image))

    @staticmethod
    def _parse_semantic_image(weak_self, image):
        """ Parse semantic image raw data on its arrival """
        self = weak_self()
        np_img = np.frombuffer(image.raw_data, dtype=np.uint8)
        # Reshap to BGRA format
        np_img = np.reshape(np_img, (image.height, image.width, -1))
        self.lane_img = (np_img[:, :, 2] == 6) | (np_img[:, :, 2] == 8)
        self.pole_img = np_img[:, :, 2] == 5

# %% ================= Ground Truth  =================


class GroundTruth(object):
    """ Class for ground truth extraction """

    def __init(self, config_args):
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
            config_args['world']['map']), config_args)

        # Launch autopilot using traffic manager
        tm = client.get_trafficmanager()
        tm_port = tm.get_port()
        tm.auto_lane_change(
            world.ego_veh, config_args['autopilot']['auto_lane_change'])
        tm.ignore_lights_percentage(
            world.ego_veh, config_args['autopilot']['ignore_lights_percentage'])
        tm.vehicle_percentage_speed_difference(
            world.ego_veh, config_args['autopilot']['vehicle_percentage_speed_difference'])
        world.ego_veh.set_autopilot(True, tm_port)

        n_ticks = int(config_args['sim_duration'] /
                      config_args['world']['delta_seconds'])
        for idx in range(n_ticks):
            world.carla_world.tick()

    finally:
        if world is not None:
            world.destroy()
            world.allow_free_run()


# %%
if __name__ == '__main__':
    main()
