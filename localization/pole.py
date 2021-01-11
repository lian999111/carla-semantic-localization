""" Implementation of pole factor. """

import math

import numpy as np
from scipy.stats import chi2
from minisam import Factor, DiagonalLoss


def compute_H(prc, exp_x, exp_y):
    """Compute H matrix given expected c0 and c1.

    H matrix is the jacobian of the measurement model wrt the pose.

    Args:
        prc: Logitudinal distance from the local frame to the front bumper.
        exp_x: Expected x of pole.
        exp_y: Expected y angle of pole.

    Returns:
        H matrix as np.ndarray.
    """
    exp_r = math.sqrt(exp_x**2 + exp_y**2)
    exp_raxle = exp_x + prc

    H = np.zeros((2, 3))

    H[0, 0] = - (exp_raxle - prc)/exp_r
    H[0, 1] = - exp_y/exp_r
    H[0, 2] = - (exp_y * prc)/exp_r

    H[1, 0] = exp_y / ((exp_raxle-prc)**2 + exp_y**2)
    H[1, 1] = (exp_raxle - prc) / ((exp_raxle-prc)**2 + exp_y**2)
    H[1, 2] = -prc*(exp_raxle-prc) / ((exp_raxle-prc)**2 + exp_y**2) - 1

    return H


class PoleFactor(Factor):
    """Pole factor."""
    geo_gate = chi2.ppf(0.8, df=2)
    sem_gate = 0.9

    def __init__(self, key, detected_pole, neighbor_poles, pose_uncert, px, pcf, pole_factor_config):
        """Constructor.

        Args:
            key: Key to the pose node.
            detected_pole: Detected pole.
            neighboring_poles: List of poles in the neighborhood.
            pose_uncert: Covariance matrix of pose.
            px: Distance from rear axle to front bumper.
            pcf: Distance from front camera to front bumper.
            pole_factor_config: Configuraiont for pole factor.
        """
        self.detected_pole = detected_pole
        self.neighbor_poles = neighbor_poles
        self.pose_uncert = pose_uncert
        self.px = px
        self.pcf = pcf
        self.config = pole_factor_config

        # r = math.sqrt((detected_pole.x+pcf)**2 + detected_pole.y**2)
        # scaled_stddev_r = pole_factor_config['stddev_r'] + 0.3*r
        self.noise_cov = np.diag([pole_factor_config['stddev_r']**2,
                                  pole_factor_config['stddev_phi']**2])

        self.expected_xy = None
        self._null_hypo = False
        self._std_scale = None

        loss = DiagonalLoss.Sigmas(np.array(
            [self.config['stddev_r'],
             self.config['stddev_phi']]))

        Factor.__init__(self, 1, [key], loss)

    def copy(self):
        return PoleFactor(self.keys()[0],
                          self.detected_pole,
                          self.neighbor_poles,
                          self.pose_uncert,
                          self.px,
                          self.pcf,
                          self.config)

    def error(self, variables):
        ########## Expectation ##########
        pose = variables.at(self.keys()[0])
        yaw = pose.so2().theta()

        # Matrix to transform points in world frame to ego (rear axle) frame
        tform_w2e = np.zeros((3, 3))
        rotm = np.array([[math.cos(yaw), -math.sin(yaw)],
                        [math.sin(yaw), math.cos(yaw)]])
        tform_w2e[0:2, 0:2] = rotm.T
        tform_w2e[0:2, 2] = -(rotm.T @ pose.translation())
        tform_w2e[2, 2] = 1

        # Extract pole landmarks into lists
        pole_x_world = [pole.x for pole in self.neighbor_poles]
        pole_y_world = [pole.y for pole in self.neighbor_poles]
        pole_types = [pole.type for pole in self.neighbor_poles]

        # Transform pole coordinates in world frame to ego (rear axle) frame
        pole_homo_coords_world = np.asarray(
            [pole_x_world, pole_y_world, np.ones((len(pole_x_world),))])
        pole_homo_coords_ego = tform_w2e @ pole_homo_coords_world

        # Get coordinates in camera's frame by offsetting x values
        pole_homo_coords_cam = pole_homo_coords_ego
        pole_homo_coords_cam[0, :] -= (self.px - self.pcf)

        # Remove pole landmarks behind the camera
        exp_pole_types = [t for t, coord_cam in zip(
            pole_types, pole_homo_coords_cam) if coord_cam[0] > 10]
        pole_homo_coords_cam = pole_homo_coords_cam[:,
                                                    pole_homo_coords_cam[0] > 10]

        ########## Measurements ##########
        # Range and azimuth wrt camera
        meas_r, meas_phi = self.detected_pole.get_r_phi(-self.pcf)

        ########## Data Association ##########
        errors = []
        squared_mahala_dists = []
        std_scales = []
        for x, y in pole_homo_coords_cam[0:2, :].T:
            r = math.sqrt(x**2 + y**2)
            phi = math.atan2(y, x)

            H = compute_H(self.px-self.pcf, x, y)
            innov = H @ self.pose_uncert @ H.T + \
                self.noise_cov * std_scale**2

            error = (np.array([r, phi]) -
                     np.array([meas_r, meas_phi])).reshape(2, -1)
            squared_mahala_dist = error.T @ np.linalg.inv(innov) @ error

            if squared_mahala_dist < self.geo_gate:
                squared_mahala_dists.append(squared_mahala_dist)
                errors.append(error)

                # Scale noise standard deviation based on range
                std_scale = max(0.001*r**2, 1)
                std_scales.append(std_scale)

        if squared_mahala_dists:
            self._null_hypo = False
            squared_mahala_dists = np.asarray(squared_mahala_dists)
            asso_idx = np.argmin(squared_mahala_dists)
            self.expected_xy = (pole_homo_coords_cam[0, asso_idx],
                                pole_homo_coords_cam[1, asso_idx])

            self._std_scale = std_scales[asso_idx]
            chosen_error = errors[asso_idx] * 1/self._std_scale
        else:
            self._null_hypo = True
            chosen_error = np.zeros((2, 1))

        return chosen_error

    def jacobians(self, variables):
        if self._null_hypo:
            jacob = np.zeros((2, 3))
        else:
            exp_x, exp_y = self.expected_xy
            jacob = compute_H(self.px-self.pcf, exp_x, exp_y)
            jacob *= 1/self._std_scale
        return [jacob]
