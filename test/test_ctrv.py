# Unit test for ctrv.py

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import model.ctrv as ctrv

class TestCTRV(unittest.TestCase):
    def test_predict_motion_from_ego_frame_bit_yaw_rate(self):
        dx, dy, dtheta = ctrv.predict_motion_from_ego_frame(1, 1, 1)
        self.assertAlmostEqual(dx, 0.8414710)
        self.assertAlmostEqual(dy, 0.4596977)
        self.assertAlmostEqual(dtheta, 1.0)

    def test_predict_motion_from_ego_frame_small_yaw_rate(self):
        dx, dy, dtheta = ctrv.predict_motion_from_ego_frame(1, 1e-4, 1)
        self.assertAlmostEqual(dx, 1.)
        self.assertAlmostEqual(dy, 0.)
        self.assertAlmostEqual(dtheta, 0.)