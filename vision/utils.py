# Implementations of depth image related functions
import numpy as np


def decode_depth(depth_buffer):
    """
    Decode the depth buffer into a depth image.

    Input:
        depth_buffer: Numpy.array of a BGR image where depth info is encoded.
    Output:
        depth_image: Numpy.array of depth image with np.float64 numbers representating depths of pixels.
    """

    b = depth_buffer[:, :, 0]
    g = depth_buffer[:, :, 1]
    r = depth_buffer[:, :, 2]
    # Decode depth buffer (convert to uint32 to avoid overflow)
    normalized = (r + g*256 + b.astype(np.float32)*65535) / 16777215
    depth_image = normalized * 1000
    return depth_image
