"""Implementation of sliding window graph manager."""

from math import atan2, sin, cos
from collections import deque

import numpy as np
import minisam as ms
import minisam.sophus as sophus

from model.ctrv import compute_F
from .odom import create_ctrv_between_factor
from .gnss import GNSSFactor
from .lane import LaneBoundaryFactor
from .utils import copy_se2


class SlidingWindowGraphManager(object):
    """Class for management and optimization of sliding window factor graph."""

    def __init__(self, px, config, expected_lane_extractor, first_node_idx=0):
        """Constructor method.

        Args:
            px (float): Distance from rear axle to front bumper.
            config (dict): Container for all configurations related to factor graph based localization.
                This should be read from the configuration .yaml file.
            expected_lane_extractor: Expected lane extractor for lane boundary factor.
                This is for lane boundary factors to query the map for expected lane boundaries.
            first_node_idx (int): Index of the first node.
                Sometimes it is more convenient to let indices consistent with recorded data.
                Especially when performing localization using only part of data that don't start from 
                the beginning of the recording.
        """
        self.px = px
        # Config
        self.config = config

        # Set expected lane extractor for the lane boundary factor
        LaneBoundaryFactor.set_expected_lane_extractor(
            expected_lane_extractor)

        # Factor graph
        self.graph = ms.FactorGraph()
        # Initial guess
        self.initials = ms.Variables()
        # Window size
        self.win_size = config['graph']['win_size']
        #: bool: True to use previous a posteriori as a priori
        self.use_prev_posteriori = config['graph']['use_prev_posteriori']

        # Optimizer parameter
        self.opt_params = ms.LevenbergMarquardtOptimizerParams()
        self.opt_params.verbosity_level = ms.NonlinearOptimizerVerbosityLevel.ITERATION
        self.opt_params.max_iterations = 5
        # Optimizer
        self.opt = ms.LevenbergMarquardtOptimizer(self.opt_params)

        # # Optimizer parameter
        # self.opt_params = ms.GaussNewtonOptimizerParams()
        # self.opt_params.verbosity_level = ms.NonlinearOptimizerVerbosityLevel.ITERATION
        # self.opt_params.max_iterations = 5
        # # Optimizer
        # self.opt = ms.GaussNewtonOptimizer(self.opt_params)

        # Marginal covarnaince solver
        self.mcov_solver = ms.MarginalCovarianceSolver()

        #: int: Index for the prior node
        self.prior_node_idx = first_node_idx
        #: int: Index for the next node to be added
        self.next_node_idx = first_node_idx + 1

        #: deque of pose node indices (including prior node)
        self._idc_in_graph = deque()
        #: deque of tuples: Each contains the state and noise model of the a posteriori of the last pose of each step
        # Tuple: (index, ms.SE2, ms.GaussianModel)
        self._history_a_posteriori = deque(maxlen=self.win_size-1)

        #: ms.Variables(): Result of optimization
        self.optimized_results = None
        #: ms.sophus.SE2: Last optimized SE2 pose
        self.last_optimized_se2 = None
        #: np.ndarray: Covariance matrix of last optimized pose
        self.last_optimized_cov = None

        #: np.ndarray: Stores the covariance matrix of the new node predicted using CTRV model.
        # This is used for gating and computing weights for data association
        self.pred_cov = None

        #: bool: True if new pose node already has a corresponding initial guess
        self.new_node_guessed = False
        #: bool: True if odom factor has been added at the current time step.
        # As in the current implementation, a odom factor should be added as the first operation
        # at every time step to introduce a new node. This flag is used to check the abovementioned
        # is not violated.
        # This flag is set to False after optimization is carried out in the end of every time step.
        # When trying to add a factor other than a odom factor without already adding a odom factor,
        # an Exceptio will be raised.
        self.odom_added = False

    def add_prior_factor(self, x, y, theta, prior_noise=None):
        """Add prior factor to first pose node.

        The first pose node, so-called prior node, in the sliding window only has a prior factor apart 
        from the between factor, which connects the prior node and the second pose node. 
        Initial guess is automatically added to the prior node using the same values as the prior.

        Args:
            x: X coordinate. (m)
            y: Y coordinate. (m)
            theta: Heading (rad)
            prior_noise: Noise model for prior.
        """
        if prior_noise is None:
            # Use default prior noise
            prior_noise = ms.DiagonalLoss.Sigmas(np.array([self.config['prior']['stddev_x'],
                                                           self.config['prior']['stddev_y'],
                                                           self.config['prior']['stddev_theta']]))

        prior_node_key = ms.key('x', self.prior_node_idx)
        self.graph.add(ms.PriorFactor(prior_node_key,
                                      ms.sophus.SE2(
                                          ms.sophus.SO2(theta), np.array([x, y])),
                                      prior_noise))

        # Add prior node's index into queue if it is the first node added
        if not self._idc_in_graph:
            self._idc_in_graph.appendleft(self.prior_node_idx)

        # Increment prior node index for later use if a posteriori is to be used as a priori later on
        if self.use_prev_posteriori:
            self.prior_node_idx += 1

        # Add initial guess
        self.initials.add(prior_node_key,
                          sophus.SE2(sophus.SO2(theta), np.array([x, y])))
        self.new_node_guessed = True

    def add_ctrv_between_factor(self, vx, yaw_rate, delta_t, add_init_guess=True):
        """Add CTRV based between factor.

        This method extends the graph by adding a new ctrv between factor to the last pose node.
        This method should be the first operation at every time step to introduce a new node into 
        the graph in most cases. An exception is the very first time step where a prior node should
        be added beforehand.

        Args:
            vx: Velocity in x. (m/s)
            yaw_rate: Yaw rate. (rad/s)
            delta_t: Timd difference. (sec)
            add_init_guess (bool): True to use the motion model as the source of initial guess.
                This should be True for most cases. For first node where odom or imu is not applicable,
                choose to add init guess using with gnss factor instead.
        """
        if not self._idc_in_graph:
            raise RuntimeError(
                'Between factor cannot be added to an empty graph!')
        if self.odom_added:
            raise RuntimeError(
                'Between factor already added for this time step. \
                    Optimize the current factor graph to proceed to the next time step.')

        last_node_idx = self._idc_in_graph[-1]
        last_node_key = ms.key('x', last_node_idx)
        new_node_key = ms.key('x', self.next_node_idx)

        # Create a BetweenFactor obj based on CTRV model and add it to the graph
        between, motion = create_ctrv_between_factor(last_node_key, new_node_key,
                                                     vx, yaw_rate, delta_t,
                                                     self.config['ctrv'])
        self.graph.add(between)

        # Add the new node index into the queue
        self._idc_in_graph.append(self.next_node_idx)

        # Increment the index of the next node to be added
        self.odom_added = True
        self.next_node_idx += 1
        self.new_node_guessed = False

        # Predict uncertainty of the new pose node from last pose and odom.
        # This will be used for gating and data association for other types of factors.
        # Note since minisam always considers uncertainty wrt local frames, theta used for
        # computing F is 0 as the last pose has the coordinate (x=0, y=0, theta=0) wrt its own frame.
        F = compute_F(0., vx, yaw_rate, delta_t)
        motion_uncert = motion[3]
        self.pred_cov = F @ self.last_optimized_cov @ F.T + motion_uncert

        # Add CTRV based initial guess for the new pose node
        if add_init_guess:
            # Predict pose a priori as initial guess using CTRV model
            last_trans = self.last_optimized_se2.translation()
            last_x, last_y = last_trans[0], last_trans[1]
            last_theta = self.last_optimized_se2.so2().theta()
            # The delta motion is wrt the local frame of the last pose, must make it wrt the global frame
            delta_x, delta_y, delta_theta = motion[0:3]
            rotm = np.array([[cos(last_theta), -sin(last_theta)],
                             [sin(last_theta), cos(last_theta)]])
            delta = rotm @ np.array([delta_x, delta_y])
            delta_x_global = delta[0]
            delta_y_global = delta[1]

            # Predict pose a priori
            prior_x = last_x + delta_x_global
            prior_y = last_y + delta_y_global
            prior_theta = last_theta + delta_theta

            # Use prior as initial guess
            guessed_pose = sophus.SE2(sophus.SO2(
                prior_theta), np.array([prior_x, prior_y]))
            self.initials.add(new_node_key, guessed_pose)
            self.new_node_guessed = True

    def add_gnss_factor(self, point, add_init_guess=False):
        """Add GNSS factor to the last pose node (excluding the prior node). 

        Args:
            point: Numpy.ndarray of measured x-y coordinate.
            add_init_guess (bool): True to use gnss as initial guess.
                This can be used when odom or imu is not applicable.
        """
        if not self.odom_added:
            raise RuntimeError(
                'Between (odom) factor should be added first at every time step.')

        node_key = ms.key('x', self._idc_in_graph[-1])

        self.graph.add(GNSSFactor(node_key,
                                  point,
                                  self.config['gnss']))

        if add_init_guess:
            if self.last_optimized_se2 is not None:
                # Use the difference from last pose to guess heading
                last_trans = self.last_optimized_se2.translation()
                trans_diff = point - last_trans
                theta_guess = atan2(trans_diff[1], trans_diff[0])
            else:
                # Simply guess 0 as heading
                theta_guess = 0.0

            self.initials.add(node_key,
                              sophus.SE2(sophus.SO2(theta_guess), point))
            self.new_node_guessed = True

    def add_lane_factor(self, detected_marking, z):
        """
        TODO: Add docstring
        """
        if not self.odom_added:
            raise RuntimeError(
                'Between (odom) factor should be added first at every time step.')

        node_key = ms.key('x', self._idc_in_graph[-1])

        self.graph.add(LaneBoundaryFactor(node_key,
                                          detected_marking,
                                          z,
                                          self.pred_cov,
                                          self.px,
                                          self.config['lane']))

    def solve_one_step(self):
        """Solve the graph and corresponding covariance matrices for the current step.

        This method should be called at every step to obtain the solution.
        """
        self._optimize_graph()
        self._solve_marginal_cov()
        last_noise_model = ms.GaussianLoss.Covariance(self.last_optimized_cov)
        self._history_a_posteriori.append(
            (self._idc_in_graph[-1], self.last_optimized_se2, last_noise_model))

    def get_result(self, idx):
        """Get optimied SE2 pose result of the node with the specified index."""
        return copy_se2(self.optimized_results.at(ms.key('x', idx)))

    def get_marignal_cov_matrix(self, idx):
        """Get marginal covariance matrix of the node with the specified index.."""
        return self.mcov_solver.marginalCovariance(ms.key('x', idx))

    def get_graph_size(self):
        """Get number of nodes in current sliding window graph."""
        return len(self._idc_in_graph)

    def get_idc_in_graph(self):
        """Get pose node indices in the current graph."""
        return list(self._idc_in_graph)

    def try_move_sliding_window_forward(self):
        """Move sliding window forward if necessary."""
        # Perform truncation if number of nodes exceeds the specified sliding window size
        if self.get_graph_size() > self.win_size:
            # Remove all factor related to the first node
            self._truncate_first_node()

            # The following are for the case where previous a posteriori is to be used as a priori
            if self.use_prev_posteriori:
                # Now the original second node becomes the first node.
                # Remove all unary factors related to it as they are already taken into account
                # in previous steps and are absorbed into the prior. That is, the prior already
                # involves the information of these unary factors.
                self._remove_unary_factors_from_first_node()

                # Add prior factor back to the graph
                prior_idx, prior_pose, prior_noise_model = self._history_a_posteriori.popleft()
                prior_trans = prior_pose.translation()
                prior_x, prior_y = prior_trans[0], prior_trans[1]
                prior_theta = prior_pose.so2().theta()

                if prior_idx != self.prior_node_idx:
                    raise RuntimeError(
                        'First node index does not match the one of stored prior!')
                self.add_prior_factor(
                    prior_x, prior_y, prior_theta, prior_noise_model)

    def _optimize_graph(self):
        """Optimize the factor graph."""
        if not self.new_node_guessed:
            raise RuntimeError('Missing initial guess for newly added node!')

        self.optimized_results = ms.Variables()
        status = self.opt.optimize(self.graph,
                                   self.initials,
                                   self.optimized_results)

        # Record last optimized SE2 (pose)
        self.last_optimized_se2 = self.get_result(self._idc_in_graph[-1])

        # Use the results as the initials for the next iteration
        self.initials = self.optimized_results

        # After optimization at the current time step, set this flag to false so the first
        # factor added at the next time step can be ensured to be a odom (between) factor
        self.odom_added = False

        if status != ms.NonlinearOptimizationStatus.SUCCESS:
            print("optimization error: ", status)

    def _solve_marginal_cov(self):
        """Solve marginal covariance matrices."""
        status = self.mcov_solver.initialize(
            self.graph, self.optimized_results)

        # Record noise model (covariance) of last pose
        self.last_optimized_cov = self.get_marignal_cov_matrix(
            self._idc_in_graph[-1])

        if status != ms.MarginalCovarianceSolverStatus.SUCCESS:
            print("maginal covariance error")
            print(status)

    def _truncate_first_node(self):
        """Truncate the first node in the current graph.

        Used when the graph size exceeds the specified window size. All factors
        related to the first node are simply deleted. It is done by creating a new 
        FactorGraph without factors connected to the first pose node.
        """
        # Remove initial for the prior node
        first_node_key = ms.key('x', self._idc_in_graph[0])
        self.initials.erase(first_node_key)

        # Create a new graph without factors connected to the first node
        new_graph = ms.FactorGraph()
        for factor in self.graph:
            keep = True

            # Check if this factor connects to the first node
            for k in factor.keys():
                if k == first_node_key:
                    keep = False
                    break

            if keep:
                new_graph.add(factor)

        # Replace the new graph
        self.graph = new_graph

        # Remove the left most node index from queue
        self._idc_in_graph.popleft()

    def _remove_unary_factors_from_first_node(self):
        """Remove unary factors related to the first node.

        If the a posteriori of the current first node is recorded in a previous step, where it
        was the last node, and used as a priori in this step, all factors already taken into 
        account to obtain the a posteriori must be removed. Otherwise, the same information will 
        contirbutes twice and leads to over-confident estimation.

        Since the current implementation supports only in order factor adding, factors to be removed
        are supposed to be unary in this localization case.

        It is done by creating a new FactorGraph without factors that have to be removed.
        """
        # Remove initial for the prior node
        first_node_key = ms.key('x', self._idc_in_graph[0])
        self.initials.erase(first_node_key)

        # Create a new graph without factors connected to the first node
        new_graph = ms.FactorGraph()
        for factor in self.graph:
            keep = True

            # Check if it's a unary factor and connects to the first node
            if len(factor.keys()) == 1 and first_node_key == factor.keys()[0]:
                keep = False

            if keep:
                new_graph.add(factor)

        # Replace the new graph
        self.graph = new_graph

        # No need to remove the left most node index from queue since the node
        # still exists and is connected with at least to a between factor
