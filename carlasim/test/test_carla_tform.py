# Run it from project root with: python -m unittest carlasim.test.test_carla_tform

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

import unittest
import numpy as np

from carlasim.carla_tform import CarlaW2ETform


class TestCarlaW2ETform(unittest.TestCase):
    def test_rot_w2e_carla_vector3D(self):
        # In left-handed z-up coordinate system
        location = carla.Location(10, 0, 0)
        # In right-handed z-down coordinate system
        rotation = carla.Rotation(yaw=90)
        carla_transform = carla.Transform(location, rotation)

        w2e_tform = CarlaW2ETform(carla_transform)

        # In left-handed z-up coordinate system
        location_in_world = carla.Location(0, 10, 0)

        # In right-handed z-up coordinate system
        np_pt_in_ego = w2e_tform.rot_w2e_carla_vector3D(location_in_world)

        self.assertAlmostEqual(np_pt_in_ego[0], 10.0)
        self.assertAlmostEqual(np_pt_in_ego[1], 0.0)
        self.assertAlmostEqual(np_pt_in_ego[2], 0.0)

    def test_tform_w2e_carla_vector3D(self):
        # In left-handed z-up coordinate system
        location = carla.Location(10, 0, 0)
        # In right-handed z-down coordinate system
        rotation = carla.Rotation(yaw=90)
        carla_transform = carla.Transform(location, rotation)

        w2e_tform = CarlaW2ETform(carla_transform)

        # In left-handed z-up coordinate system
        location_in_world = carla.Location(0, 10, 0)

        # In right-handed z-up coordinate system
        np_pt_in_ego = w2e_tform.tform_w2e_carla_vector3D(location_in_world)

        self.assertAlmostEqual(np_pt_in_ego[0], 10.0)
        self.assertAlmostEqual(np_pt_in_ego[1], -10.0)
        self.assertAlmostEqual(np_pt_in_ego[2], 0.0)

    def test_rot_w2e_carla_numpy_point(self):
        # In left-handed z-up coordinate system
        location = carla.Location(10, 0, 0)
        # In right-handed z-down coordinate system
        rotation = carla.Rotation(yaw=90)
        carla_transform = carla.Transform(location, rotation)

        w2e_tform = CarlaW2ETform(carla_transform)

        # In right-handed z-up coordinate system
        np_pts_in_world = np.array([0, -10, 0])
        # In right-handed z-up coordinate system
        np_pts_in_ego = w2e_tform.rot_w2e_numpy_array(np_pts_in_world)

        np.testing.assert_array_almost_equal(
            np_pts_in_ego, np.array([10, 0, 0]).reshape((3, -1)))

    def test_rot_w2e_carla_numpy_points(self):
        # In left-handed z-up coordinate system
        location = carla.Location(10, 0, 0)
        # In right-handed z-down coordinate system
        rotation = carla.Rotation(yaw=90)
        carla_transform = carla.Transform(location, rotation)

        w2e_tform = CarlaW2ETform(carla_transform)

        # In right-handed z-up coordinate system
        np_pts_in_world = np.array([[0, -10, 0], [10, 10, 0]]).T
        # In right-handed z-up coordinate system
        np_pts_in_ego = w2e_tform.rot_w2e_numpy_array(np_pts_in_world)

        np.testing.assert_array_almost_equal(
            np_pts_in_ego, np.array([[10, 0, 0], [-10, 10, 0]]).T.reshape((3, -1)))

    def test_tform_w2e_carla_numpy_point(self):
        # In left-handed z-up coordinate system
        location = carla.Location(10, 0, 0)
        # In right-handed z-down coordinate system
        rotation = carla.Rotation(yaw=90)
        carla_transform = carla.Transform(location, rotation)

        w2e_tform = CarlaW2ETform(carla_transform)

        # In right-handed z-up coordinate system
        np_pts_in_world = np.array([0, -10, 0])
        # In right-handed z-up coordinate system
        np_pts_in_ego = w2e_tform.tform_w2e_numpy_array(np_pts_in_world)

        np.testing.assert_array_almost_equal(
            np_pts_in_ego, np.array([10, -10, 0]).reshape((3, -1)))

    def test_tform_w2e_carla_numpy_points(self):
        # In left-handed z-up coordinate system
        location = carla.Location(10, 0, 0)
        # In right-handed z-down coordinate system
        rotation = carla.Rotation(yaw=90)
        carla_transform = carla.Transform(location, rotation)

        w2e_tform = CarlaW2ETform(carla_transform)

        # In right-handed z-up coordinate system
        np_pts_in_world = np.array([[0, -10, 0], [10, 10, 0]]).T
        # In right-handed z-up coordinate system
        np_pts_in_ego = w2e_tform.tform_w2e_numpy_array(np_pts_in_world)

        np.testing.assert_array_almost_equal(
            np_pts_in_ego, np.array([[10, -10, 0], [-10, 0, 0]]).T.reshape((3, -1)))

    def test_from_conventional(self):
        # In right-handed z-up coordinate system
        conventional_location = [0, 10, 0]
        conventional_orientation = [0, 0, np.pi/2]
        w2e_tform = CarlaW2ETform.from_conventional(
            conventional_location, conventional_orientation)
        
        # In right-handed z-up coordinate system
        np_pts_in_world = np.array([0, -10, 0])
        # In right-handed z-up coordinate system
        np_pts_in_ego = w2e_tform.tform_w2e_numpy_array(np_pts_in_world)

        np.testing.assert_array_almost_equal(
            np_pts_in_ego, np.array([-20, 0, 0]).reshape((3, -1)))

    def test_from_conventional_with_column_vector(self):
        # In right-handed z-up coordinate system
        conventional_location = np.array([0, 10, 0]).reshape((3, 1))
        conventional_orientation = np.array([0, 0, np.pi/2]).reshape((3, 1))
        w2e_tform = CarlaW2ETform.from_conventional(
            conventional_location, conventional_orientation)
        
        # In right-handed z-up coordinate system
        np_pts_in_world = np.array([0, -10, 0])
        # In right-handed z-up coordinate system
        np_pts_in_ego = w2e_tform.tform_w2e_numpy_array(np_pts_in_world)

        np.testing.assert_array_almost_equal(
            np_pts_in_ego, np.array([-20, 0, 0]).reshape((3, -1)))


if __name__ is '__main__':
    unittest.main()
