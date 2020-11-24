# Run it from project root with: python -m unittest model.test.test_ctrv

import unittest
import model.ctrv as ctrv

class TestCTRV(unittest.TestCase):
    def test_predict_motion_from_ego_frame_big_yaw_rate(self):
        dx, dy, dtheta, cov = ctrv.predict_motion_from_ego_frame(1, 1, 1)
        self.assertAlmostEqual(dx, 0.8414710)
        self.assertAlmostEqual(dy, 0.4596977)
        self.assertAlmostEqual(dtheta, 1.0)
        self.assertIsNone(cov)

    def test_predict_motion_from_ego_frame_small_yaw_rate(self):
        dx, dy, dtheta, cov = ctrv.predict_motion_from_ego_frame(1, 1e-4, 1)
        self.assertAlmostEqual(dx, 1.)
        self.assertAlmostEqual(dy, 0.)
        self.assertAlmostEqual(dtheta, 1e-4)
        self.assertIsNone(cov)

    def test_predict_motion_from_ego_frame_small_yaw_rate_velocity(self):
        dx, dy, dtheta, cov = ctrv.predict_motion_from_ego_frame(1e-4, 1e-4, 1)
        self.assertAlmostEqual(dx, 1e-4)
        self.assertAlmostEqual(dy, 0.)
        self.assertAlmostEqual(dtheta, 0.)
        self.assertIsNone(cov)

if __name__ is '__main__':
    unittest.main()