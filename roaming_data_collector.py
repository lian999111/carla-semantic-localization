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
import random
import numpy as np
import matplotlib.pyplot as plt

# %% ================= Global function =================


def find_weather_presets():
    """Method to find weather presets"""
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
        self.carla_world.set_weather(self._weather_presets[self._weather_index][0])
        self.ego_veh = None
        self.gnss = None
        self.imu = None
        self.semantic_camera = None

        # Set up carla engine using config
        settings = self.carla_world.get_settings()
        settings.no_rendering_mode = config_args['world']['no_rendering']
        settings.synchronous_mode = config_args['world']['sync_mode']
        settings.fixed_delta_seconds = config_args['world']['delta_seconds']
        self.carla_world.apply_settings(settings)

        # Start simuation
        self.restart(config_args)

    def restart(self, config_args, spawn_point=None):
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
        self.spectator.set_transform(carla.Transform(spawn_point.location + carla.Location(z=30),
                                                     carla.Rotation(pitch=-90)))
        
        # Set up the sensors

        self.carla_world.tick()

    def destroy(self):
        pass

# %% ================= Camera Manager =================


class GNSS(object):
    """ Class for GNSS sensor"""

    def __init__(self, config_args):
        pass

# %% ================= Camera Manager =================


class CameraManager(object):
    """ Class for semantic camera"""

    def __init__(self, config_args):
        pass


# %%
def main():
    # Parse passed-in config yaml file
    argparser = argparse.ArgumentParser(
        description='CARLA Roaming Data Collector')
    argparser.add_argument('config', type=argparse.FileType(
        'r'), help='configuration yaml file for carla env setup')
    args = argparser.parse_args()

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

    finally:
        if world is not None:
            world.destroy()


# %%
if __name__ == '__main__':
    main()
