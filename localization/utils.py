import numpy as np
import minisam as ms

from detection.utils import MELaneMarking


def copy_se2(se2):
    """Deep copy SE2 object.

    Args:
        se2: SE2 object.
    """
    trans = se2.translation()
    so2 = se2.so2()
    return ms.sophus.SE2(so2, trans)


class ExpectedLaneExtractor(object):
    """Class for expected lane detection extraction.

    It is built upon a lane ground truth extractor."""

    def __init__(self, lane_gt_extractor):
        """Constructor.

        Args:
            lane_gt_extractor (LaneGTExtractor): Lane ground truth extractor that interacts
                with Carla to get exptected lane information.
        """
        self._lane_gt_extractor = lane_gt_extractor

        # Attributes for query history
        self.location = None
        self.orientation = None
        self.in_junction = None
        self.into_junction = None
        self.me_format_lane_markings = None

    def extract(self, location, orientation):
        """Extract lane information given location and orientation.

        Args:
            location: Array-like 3D query point in world (right-handed z-up).
            rotation: Array-like roll pitch yaw rotation representation in rad (right-handed z-up).

        Returns:
            in_junction: True if is in junction area.
            into_junction: True if is into junction area.
            me_format_lane_markings: List of ME format lane markings
        """
        # Only extract ground truth if new pose is given
        if (self.location is None) or not np.array_equal(location, self.location) or not np.array_equal(orientation, self.orientation):
            expected_lane_gt = self._lane_gt_extractor.update(
                location, orientation)

            self.location = location
            self.orientation = orientation

            self.in_junction = expected_lane_gt['in_junction']
            self.into_junction = expected_lane_gt['into_junction']
            lane_id = expected_lane_gt['lane_id']

            coeffs_keys = ['next_left_marking_coeffs', 'left_marking_coeffs',
                           'right_marking_coeffs', 'next_right_marking_coeffs']
            marking_obj_keys = ['next_left_marking', 'left_marking',
                                'right_marking', 'next_right_marking']

            # List of GT lane markings as MELaneMarking objects
            self.me_format_lane_markings = []
            for _, (coeffs_key, marking_obj_key) in enumerate(zip(coeffs_keys, marking_obj_keys)):
                lane_marking = expected_lane_gt[marking_obj_key] if expected_lane_gt[marking_obj_key] is not None else None
                if lane_marking is not None:
                    # make it in descending order
                    coeffs = expected_lane_gt[coeffs_key][::-1]
                    self.me_format_lane_markings.append(
                        MELaneMarking.from_lane_marking(coeffs, lane_marking, lane_id))

        return self.in_junction, self.into_junction, self.me_format_lane_markings
