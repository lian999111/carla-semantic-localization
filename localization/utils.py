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

    def extract(self, location, orientation):
        """Extract lane information given location and orientation.

        Args:
            location: Array-like 3D query point in world (right-handed z-up).
            rotation: Array-like roll pitch yaw rotation representation in rad (right-handed z-up).

        Returns:
            in_junction: True if is in junction area.
            me_format_lane_markings: List of ME format lane markings
        """
        expected_lane_gt = self._lane_gt_extractor.update(
            location, orientation)

        in_junction = expected_lane_gt['in_junction']
        lane_id = expected_lane_gt['lane_id']

        coeffs_keys = ['next_left_marking_coeffs', 'left_marking_coeffs',
                       'right_marking_coeffs', 'next_right_marking_coeffs']
        marking_obj_keys = ['next_left_marking', 'left_marking',
                            'right_marking', 'next_right_marking']

        # List of GT lane markings as MELaneMarking objects
        me_format_lane_markings = []
        for _, (coeffs_key, marking_obj_key) in enumerate(zip(coeffs_keys, marking_obj_keys)):
            lane_marking = expected_lane_gt[marking_obj_key] if expected_lane_gt[marking_obj_key] is not None else None
            if lane_marking is not None:
                coeffs = expected_lane_gt[coeffs_key][::-1] # make it in descending order
                me_format_lane_markings.append(MELaneMarking.from_lane_marking(coeffs, lane_marking, lane_id))

        return in_junction, me_format_lane_markings
