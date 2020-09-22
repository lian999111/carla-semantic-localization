# This implements a data collection car with camera detection algorithms running online.
# IMPORTANT: This can run very slow.

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

from carlasim.data_collect import World, FrontSmartCamera


def main():
    # Parse passed-in config yaml file
    argparser = argparse.ArgumentParser(
        description='CARLA Roaming Data Collector')
    argparser.add_argument('config', type=argparse.FileType(
        'r'), help='configuration yaml file for carla env setup')
    argparser.add_argument('vision_params', type=argparse.FileType(
        'r'), help='yaml file for vision algorithm parameters')
    argparser.add_argument('-r', '--record', default=False,
                           action='store_true', help='record collected data')
    args = argparser.parse_args()

    # Read configurations from yaml file to config
    with args.config as config_file:
        config = yaml.safe_load(config_file)
    with args.vision_params as vision_params_file:
        vision_params = yaml.safe_load(vision_params_file)

    # Load camera parameters
    with open('vision/calib_data.pkl', 'rb') as f:
        calib_data = pickle.load(f)
    # Load parameters for bird's eye view projection
    with open('vision/ipm_data.pkl', 'rb') as f:
        ipm_data = pickle.load(f)

    # Calculate distance from camera to front bumper
    dist_cam_to_fbumper = (config['ego_veh']['raxle_to_fbumper']
                           - config['sensor']['front_camera']['pos_x']
                           - config['ego_veh']['raxle_to_cg'])

    # Initialize world
    world = None
    spawn_point = None
    # Assign spawn point for ego vehicle
    spawn_point = carla.Transform(carla.Location(
        218.67, 59.18, 0.59), carla.Rotation(yaw=180))
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        # Create a World obj with a built-in map
        world = World(client.load_world(config['world']['map']),
                      client.get_trafficmanager(),
                      config,
                      spawn_point=spawn_point)

        # Craete a smart camera that extract high-level detections
        front_smart_camera = FrontSmartCamera(world.semantic_camera,
                                              world.imu,
                                              vision_params,
                                              calib_data,
                                              ipm_data)

        # Launch autopilot for ego vehicle
        world.set_ego_autopilot(True, config['autopilot'])

        n_ticks = int(config['sim_duration'] /
                      config['world']['delta_seconds'])

        if args.record:
            ss_images = []
            depth_buffers = []
            vx = []
            yaw_rate = []
            in_junction = []

        # Simulation loop
        to_left = True
        for idx in range(n_ticks):
            world.step_forward()
            world.see_ego_veh()
            front_smart_camera.update()

            if args.record:
                ss_images.append(world.semantic_camera.ss_image)
                depth_buffers.append(world.depth_camera.depth_buffer)
                vx.append(world.imu.vx)
                yaw_rate.append(world.imu.gyro_z)
                in_junction.append(world.ground_truth.in_junction)

            print('vx: {:3.2f}, vy: {:3.2f}, w: {:3.2f}'.format(
                world.imu.vx, world.imu.vy, world.imu.gyro_z * 180 / pi))
            print('in junction: {}'.format(world.ground_truth.in_junction))
            # c0
            print('{:.2f}   {:.2f}   {:.2f}   {:.2f}'.format(
                world.ground_truth.next_left_marking_param[0] if world.ground_truth.next_left_marking else -10,
                world.ground_truth.left_marking_param[0] if world.ground_truth.left_marking else -10,
                world.ground_truth.right_marking_param[0] if world.ground_truth.right_marking else -10,
                world.ground_truth.next_right_marking_param[0] if world.ground_truth.next_right_marking else -10))
            # Marking type
            print('{}   {}   {}   {}'.format(
                world.ground_truth.next_left_marking.type if world.ground_truth.next_left_marking else None,
                world.ground_truth.left_marking.type if world.ground_truth.left_marking else None,
                world.ground_truth.right_marking.type if world.ground_truth.right_marking else None,
                world.ground_truth.next_right_marking.type if world.ground_truth.next_right_marking else None))

            if (idx+1) % int(3/config['world']['delta_seconds']) == 0:
                world.force_lane_change(to_left=to_left)
                to_left = not to_left

    finally:
        if world:
            world.set_ego_autopilot(False)
            world.destroy()
            # Allow carla engine to run freely so it doesn't just hang there
            world.allow_free_run()

        # Store data
        if args.record:
            mydir = os.path.join(os.getcwd(), 'recordings',
                                 datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            os.makedirs(mydir)

            with open(os.path.join(mydir, 'ss_images'), 'wb') as image_file:
                pickle.dump(ss_images, image_file)
            with open(os.path.join(mydir, 'depth_buffers'), 'wb') as depth_file:
                pickle.dump(depth_buffers, depth_file)
            with open(os.path.join(mydir, 'vx'), 'wb') as vx_file:
                pickle.dump(vx, vx_file)
            with open(os.path.join(mydir, 'yaw_rate'), 'wb') as yaw_rate_file:
                pickle.dump(yaw_rate, yaw_rate_file)
            with open(os.path.join(mydir, 'in_junction'), 'wb') as in_junction_file:
                pickle.dump(in_junction, in_junction_file)


if __name__ == "__main__":
    main()
