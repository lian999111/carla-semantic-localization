"""This implements a raw data (no high-level detection) collectior car."""

# The following boilerplate is required if .egg is not installed
# See: https://carla.readthedocs.io/en/latest/build_system/
import glob
import os
import datetime
import sys
import argparse
from math import pi
from shutil import copyfile

import yaml
from carlasim.data_collect import World, IMU, GNSS, SemanticCamera, DepthCamera
from carlasim.record import SequentialRecorder, StaticAndSequentialRecorder

try:
    sys.path.append(glob.glob('./carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


def main():
    """Main function."""
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

    # Placeholder for World obj
    world = None

    # Assign spawn point for ego vehicle according to config
    if config['ego_veh']['use_random_spawn_point']:
        spawn_point = None
    else:
        spawn_location = carla.Location(config['ego_veh']['spawn_location'][0],
                                        config['ego_veh']['spawn_location'][1],
                                        config['ego_veh']['spawn_location'][2])
        spawn_orientation = carla.Rotation(config['ego_veh']['spawn_orientation'][0],
                                           config['ego_veh']['spawn_orientation'][1],
                                           config['ego_veh']['spawn_orientation'][2])
        spawn_point = carla.Transform(spawn_location, spawn_orientation)

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

        # Launch the controller for ego vehicle
        # Currently, 2 methods to control the ego vehicle are provided:
        # 1. Autopilot:
        #    Use the traffic manager built in Carla. It allows for some basic behavior settings,
        #    but the car can only roam around randomly.
        #    For more info: https://github.com/carla-simulator/carla/issues/2966
        # 2. Behavior Agent:
        #    Use the BehaviorAgent class defined in the "agent" package found in Carla's repository.
        #    This utility class is used in several example codes for demonstrations.
        #    However, it is not officially documented and its use here is solely based on the 
        #    examination of the example codes with some very minor changes.
        #    It has the advantage that it can accept a set of waypoints to follow. 

        if 'autopilot' in config.keys():
            world.set_ego_autopilot(True, config['autopilot'])
        if 'behavior_agent' in config.keys():
            world.set_behavior_agent(config['behavior_agent'])

        if args.record:
            recording_folder = os.path.join(os.getcwd(), 'recordings',
                                            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            os.makedirs(recording_folder)
            # Set up Carla's recorder for replaying if activated by config
            if config['carla_recorder']:
                file_name = os.path.join(
                    recording_folder, 'carla_recording.log')
                client.start_recorder(file_name)

        n_ticks = int(config['sim_duration'] /
                      config['world']['delta_seconds'])

        # Simulation loop
        for idx in range(n_ticks):
            keep_running = world.step_forward()
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
            print('into junction: {}'.format(
                lane_gt['into_junction']))
            # c0
            print('{:.2f}   {:.2f}   {:.2f}   {:.2f}'.format(
                lane_gt['next_left_marking_coeffs'][0] if lane_gt['next_left_marking'] else -10,
                lane_gt['left_marking_coeffs'][0] if lane_gt['left_marking'] else -10,
                lane_gt['right_marking_coeffs'][0] if lane_gt['right_marking'] else -10,
                lane_gt['next_right_marking_coeffs'][0] if lane_gt['next_right_marking'] else -10))
            # c1
            print('{:.2f}   {:.2f}   {:.2f}   {:.2f}'.format(
                lane_gt['next_left_marking_coeffs'][1] if lane_gt['next_left_marking'] else -10,
                lane_gt['left_marking_coeffs'][1] if lane_gt['left_marking'] else -10,
                lane_gt['right_marking_coeffs'][1] if lane_gt['right_marking'] else -10,
                lane_gt['next_right_marking_coeffs'][1] if lane_gt['next_right_marking'] else -10))
            # Marking type
            print('{}   {}   {}   {}'.format(
                lane_gt['next_left_marking'].type.name if lane_gt['next_left_marking'] else None,
                lane_gt['left_marking'].type.name if lane_gt['left_marking'] else None,
                lane_gt['right_marking'].type.name if lane_gt['right_marking'] else None,
                lane_gt['next_right_marking'].type.name if lane_gt['next_right_marking'] else None))

            if not keep_running:
                print("Final goal reached, mission accomplished...")
                break

        if args.record:
            # Stop Carla's recorder
            if config['carla_recorder']:
                client.stop_recorder()
            # Save recorded data
            sensor_recorder.save(recording_folder, 'sensor_data')
            gt_recorder.save(recording_folder, 'gt_data')
            # Copy config files to recording folder for future reference
            settings_dir = os.path.join(recording_folder, 'settings')
            os.mkdir(settings_dir)
            target_dir = os.path.join(settings_dir, 'config.yaml')
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
