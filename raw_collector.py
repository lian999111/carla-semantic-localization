# This implements a raw data (no high-level detection) collectior car.

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
import carla

import argparse
import yaml
import pickle
from math import pi
from shutil import copyfile

from carlasim.data_collect import World, IMU, GNSS, SemanticCamera, DepthCamera
from carlasim.record import SequentialRecorder, StaticAndSequentialRecorder

# TODO: Add Carla recorder


def main():
    # Parse passed-in config yaml file
    argparser = argparse.ArgumentParser(
        description='CARLA Roaming Data Collector')
    argparser.add_argument('config', type=argparse.FileType(
        'r'), help='yaml file for carla simulation config')
    argparser.add_argument('-r', '--record', default=False, action='store_true',
                           help='record data selected in config file')
    args = argparser.parse_args()

    # Read configurations from yaml file to config
    with args.config as f:
        config = yaml.safe_load(f)

    # Initialize world
    world = None
    spawn_point = None
    # Assign spawn point for ego vehicle
    spawn_point = carla.Transform(carla.Location(
        229.67, 80.99, 0.59), carla.Rotation(yaw=91))
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        # Create a World obj with a built-in map
        world = World(client.load_world(config['world']['map']),
                      client.get_trafficmanager(),
                      config,
                      spawn_point=spawn_point)

        # Add Carla sensors
        ego_veh = world.ego_veh
        world.add_carla_sensor(IMU('imu', config['sensor']['imu'], ego_veh))
        world.add_carla_sensor(GNSS('gnss', config['sensor']['gnss'], ego_veh))
        world.add_carla_sensor(SemanticCamera(
            'semantic_camera', config['sensor']['front_camera'], ego_veh))
        world.add_carla_sensor(DepthCamera(
            'depth_camera', config['sensor']['front_camera'], ego_veh))

        # Set up recorders
        if args.record:
            sensor_recorder = SequentialRecorder(config['recorder']['sensor'])
            gt_recorder = StaticAndSequentialRecorder(config['recorder']['gt'])

            # Record static ground truth data
            gt_recorder.record_static(world.ground_truth.all_gt['static'])

        # Launch autopilot for ego vehicle
        world.set_ego_autopilot(True, config['autopilot'])

        n_ticks = int(config['sim_duration'] /
                      config['world']['delta_seconds'])

        # Simulation loop
        to_left = True
        for idx in range(n_ticks):
            world.step_forward()
            world.see_ego_veh()

            # Record sequential data
            if args.record:
                sensor_recorder.record_seq(world.all_sensor_data)
                gt_recorder.record_seq(world.ground_truth.all_gt['seq'])

            # Print data of interest
            imu_data = world.all_sensor_data['imu']
            lane_gt = world.ground_truth.all_gt['seq']['lane']

            print('vx: {:3.2f}, vy: {:3.2f}, w: {:3.2f}'.format(
                imu_data['vx'], imu_data['vy'], imu_data['gyro_z'] * 180 / pi))
            print('in junction: {}'.format(
                lane_gt['in_junction']))
            # c0
            print('{:.2f}   {:.2f}   {:.2f}   {:.2f}'.format(
                lane_gt['next_left_marking_coeffs'][0] if lane_gt['next_left_marking'] else -10,
                lane_gt['left_marking_coeffs'][0] if lane_gt['left_marking'] else -10,
                lane_gt['right_marking_coeffs'][0] if lane_gt['right_marking'] else -10,
                lane_gt['next_right_marking_coeffs'][0] if lane_gt['next_right_marking'] else -10))
            # Marking type
            print('{}   {}   {}   {}'.format(
                lane_gt['next_left_marking'].type if lane_gt['next_left_marking'] else None,
                lane_gt['left_marking'].type if lane_gt['left_marking'] else None,
                lane_gt['right_marking'].type if lane_gt['right_marking'] else None,
                lane_gt['next_right_marking'].type if lane_gt['next_right_marking'] else None))

            if (idx+1) % int(3/config['world']['delta_seconds']) == 0:
                world.force_lane_change(to_left=to_left)
                to_left = not to_left

        if args.record:
            # Save recorded data
            mydir = os.path.join(os.getcwd(), 'recordings',
                                 datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            os.makedirs(mydir)
            
            sensor_recorder.save(mydir, 'sensor_data')
            gt_recorder.save(mydir, 'gt_data')

            # Copy config files to folder for future reference
            target_dir = os.path.join(mydir, 'config.yaml')
            copyfile(args.config.name, target_dir)

    finally:
        if world:
            world.set_ego_autopilot(False)
            world.destroy()
            # Allow carla engine to run freely so it doesn't just hang there
            world.allow_free_run()


# %%
if __name__ == '__main__':
    main()
