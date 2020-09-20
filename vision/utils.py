# Implementations of vision-related utilities functions
# that can be used from different modules
import numpy as np
import cv2


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
    normalized = (r + g*256 + b.astype(np.uint32)*65535) / 16777215
    depth_image = normalized * 1000
    return depth_image


def find_pole_bases(pole_image, max_width, min_height, use_center=True, horizon=None):
    """
    Find bases of poles in the given image.

    This function first finds connected pole pixels. Then the bottom center of their 
    bound boxes are extracted. Assuming flat ground, a pole stemming from the ground 
    is bound to appear under the horizon. Thus, only image below the horizon is searched 
    to avoid cluttering caused by poles that are too far. 

    Input: 
        pole_image: OpenCV image with supported data type (e.g. np.uint8). The image should have non-zero values only
                    at pole pixels, which is easy to obtained from a semantic image.
        max_width: Maximum width of bound box under which a pole object is considered.
        min_height: Minimum height of bound box under which a pole object is considered.
        use_center: Bool whether to use the bottom center of the bounding box as a pole's base.
                    If False, the pixel with the largest v coordinate (lowest in image) is used.
        horizon: Position of the horizon in the image (wrt the top of image). If not given, half point of image is used.
    Output:
        pole_bases_uv: Image coordiantes (u-v) of detected pole bases.
    """
    # Use half height of image is horizon not given
    if not horizon:
        horizon = pole_image.shape[0] // 2

    _, labels, stats, centroids = cv2.connectedComponentsWithStats(
        pole_image[horizon:, :])

    # Find components fulfilling the criteria
    selected = np.logical_and(
        stats[:, 2] < max_width, stats[:, 3] > min_height)

    n_pole_detected = sum(selected)
    # Coordinates of pole bases in image (2-by-N)
    pole_bases_uv = np.zeros((2, n_pole_detected))

    if use_center:
        # u
        pole_bases_uv[0, :] = centroids[selected, 0]
        # v
        pole_bases_uv[1, :] = horizon + \
            stats[selected, 1] + stats[selected, 3]
    else:
        # Nonzero coordiantes
        nonzerou = labels.nonzero()[1]
        nonzerov = labels.nonzero()[0]
        for pole_idx, stat in enumerate(stats[selected]):
            # Get the bounding box
            bbox_u_low = stat[0]
            bbox_v_low = stat[1]
            bbox_u_high = bbox_u_low + stat[2]
            bbox_v_high = bbox_v_low + stat[3]

            # Get mask in the bbox
            mask = np.logical_and.reduce((bbox_u_low < nonzerou,
                                          nonzerou < bbox_u_high,
                                          bbox_v_low < nonzerov,
                                          nonzerov < bbox_v_high))

            # Get the base coordinate
            selected_u, selected_v = nonzerou[mask], nonzerov[mask]
            max_v = selected_v.max()     # largest v coordinate (lowest in image)
            # If there are multiple pixels with v = max_v, pick the one in the middle
            center_max_v_idx = int(np.median((selected_v == max_v).nonzero()))
            pole_bases_uv[0, pole_idx] = selected_u[center_max_v_idx]
            pole_bases_uv[1, pole_idx] = selected_v[center_max_v_idx] + horizon

    return pole_bases_uv.astype(np.int)
