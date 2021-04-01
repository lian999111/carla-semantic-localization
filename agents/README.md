This package is modified with some bug fixes and improved from CARLA's official repository.

CARLA is under the MIT license.

Bug fix:
- The PID controller is no longer created and deleted at every control cycle. This behavior in CARLA's official implementation made I- and D-control totally not in action.

Improvement:
- Lateral control now refer to a lookahead point with a distance based on the current speed instead of the next waypoint directly. 


