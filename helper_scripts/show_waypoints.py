# Helper script to show waypoints throughout the specified map

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

import random
import time
import matplotlib.pyplot as plt


def main():
    try:
        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)

        client.load_world('Town04')

        # Once we have a client we can retrieve the world that is currently
        # running.
        world = client.get_world()

        carla_map = world.get_map()

        waypoints = carla_map.generate_waypoints(1)
        for w in waypoints:
            half_width = w.lane_width * 0.5
            left_marking = w.left_lane_marking
            right_marking = w.right_lane_marking
            world.debug.draw_point(w.transform.location,
                                   color=carla.Color(r=255, g=0, b=0))
            if left_marking.type is not carla.LaneMarkingType.NONE:
                left_marking_loc = w.transform.transform(
                    carla.Location(y=-half_width))
                world.debug.draw_line(left_marking_loc, left_marking_loc+carla.Location(z=1),
                                      color=carla.Color(r=0, g=255, b=0), thickness=0.2)
            if right_marking.type is not carla.LaneMarkingType.NONE:
                right_marking_loc = w.transform.transform(
                    carla.Location(y=half_width))
                world.debug.draw_line(right_marking_loc, right_marking_loc+carla.Location(z=2),
                                      color=carla.Color(r=255, g=0, b=255), thickness=0.1)

        x = [w.transform.location.x for w in waypoints]
        y = [-w.transform.location.y for w in waypoints]
        plt.plot(x, y, '.', ms=1)
        plt.show()

    finally:
        print('done.')


if __name__ == '__main__':

    main()
