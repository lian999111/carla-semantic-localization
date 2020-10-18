# Implementations of utilities for detection

from carlasim.utils import TrafficSignType
from carlasim.carla_tform import Transform

class Pole(object):
    """
    Used to represent a pole for detection and pole map.

    The internal data follows the right-handed z-up coordinate system.
    """

    def __init__(self, x, y, traffic_sign_type=TrafficSignType.NONE):
        """
        Constructor.

        Input:
            x: x coordinate of pole.
            y: y coordinate of pole.
            traffic_sign_type: TrafficSignType obj.
        """
        self.x = x
        self.y = y
        self.type = traffic_sign_type

class