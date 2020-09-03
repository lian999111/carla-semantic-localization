import numpy as np
from math import sin, cos


class PlanarState(object):
    """ Class for SE(2) state containing the position and heading (yaw) of an object on a 2D plane """

    def __init__(self, x=0.0, y=0.0, theta=0.0):
        """ Constructor method """
        self._x = x
        self._y = y
        self._theta = theta  # (rad) yaw 

        # The homogeneous transform of the pose
        self._tform = None  # lazy initialization
        self._tform_ready = False

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value
        # Update tform
        if self._tform_ready:
            self._tform[0, 2] = self._x

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value
        # Update tform
        if self._tform_ready:
            self._tform[1, 2] = self._y

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        self._theta = value
        # Update tform
        if self._tform_ready:
            sin_theta = sin(self._theta)
            cos_theta = cos(self._theta)
            self._tform[0, 0] = self._tform[1, 1] = cos_theta
            self._tform[0, 1] = - sin_theta
            self._tform[1, 0] = sin_theta

    @property
    def tform(self):
        """ Get homogeneous transform of current pose """
        if not self._tform_ready:
            self._tform = np.zeros((3, 3))
            self._tform[2, 2] = 1.
            sin_theta = sin(self._theta)
            cos_theta = cos(self._theta)
            # Rotation
            self._tform[0, 0] = self._tform[1, 1] = cos_theta
            self._tform[0, 1] = - sin_theta
            self._tform[1, 0] = sin_theta
            # Translation
            self._tform[0, 2] = self._x
            self._tform[1, 2] = self._y

            self._tform_ready = True

        return self._tform


    def get_state_vec(self):
        """ Get numpy vector of state """
        return np.array([self._x, self._y, self._theta])


class PlanarCov(object):
    """ Class for covariance matrix of PlanarState """

    def __init__(self, var_x, var_y, var_theta, cov_xy=0., cov_xtheta=0., cov_ytheta=0.):
        """ Constructor method """
        self.var_x = var_x
        self.var_y = var_y
        self.var_theta = var_theta
        self.cov_xy = cov_xy
        self.cov_xtheta = cov_xtheta
        self.cov_ytheta = cov_ytheta

    def get_cov_matrix(self):
        """ Get numpy matrix for the covariance matrix """
        # Follows the order of x y yaw
        cov_mat = np.zeros((3, 3))
        cov_mat[0, 0] = self.var_x
        cov_mat[1, 1] = self.var_y
        cov_mat[2, 2] = self.var_theta
        cov_mat[0, 1] = cov_mat[1, 0] = self.cov_xy
        cov_mat[0, 2] = cov_mat[2, 0] = self.cov_xtheta
        cov_mat[1, 3] = cov_mat[3, 1] = self.cov_ytheta
        return 

# s = PlanarState(10, 20, 1)
# s.x = 100
# print(s.tform)
# s.theta = 0
# print(s.tform)