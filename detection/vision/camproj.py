# Implementations of projections between image and world
import numpy as np


def im2world_known_x(H, x0, image_pts, x_world):
    """
    Project 2D image points to 3D world points with known x in the world.

    With known camera intrinsic and extrinsic parameters, 
     - K: Camera instrinsic matrix
     - R: Camera rotation matrix
     - x0: Principal point
    one can easily project a 2D point on the image back to the 3D world, if one dimension in the world is fixed.
    This function allows projection of multiple points at once.

    Important: Singularity happens when the points lies on the u=0, v=0, w=0 in the image's coordinate system.

    Input:
        H: Numpy 3-by-3 matrix representating R.T @ K^-1.
        x0: Array-like 3D vector representating principal point's coordinate in the world frame.
        image_pts: 2-by-N Numpy.array of N 2D u-v points in the image.
        x_world: 1-by-N Numpy.array of N known x-coordinates in the world.
    Output:
        3-by-N Numpy.array of N 3D coordinates in the world frame.
    """
    if not isinstance(image_pts, np.ndarray):
        image_pts = np.array(image_pts)

    # Make sure shape is right
    if image_pts.ndim != 2:
        image_pts = image_pts.reshape((2, -1))    # 2-by-N

    # Number of points
    n_pts = image_pts.shape[1]
    
    if isinstance(x_world, (int, float)):
        x_world = x_world * np.ones((1, n_pts))
    elif not isinstance(x_world, np.ndarray):
        x_world = np.array(x_world)
    
    # Make sure shape is right
    if x_world.ndim != 2:
        x_world = x_world.reshape((1, -1))        # 1-by-N

    image_pt_homo = np.concatenate((image_pts, np.ones((1, n_pts))), axis=0)
    b = H @ image_pt_homo
    W_prime = b[0] / (x_world - x0[0])

    y_world = b[1] / W_prime + x0[1]
    z_world = b[2] / W_prime + x0[2]

    return np.concatenate((x_world, y_world, z_world), axis=0)


def im2world_known_z(H, x0, image_pts, z_world):
    """
    Project 2D image points to 3D world points with known z in the world.

    With known camera intrinsic and extrinsic parameters, 
     - K: Camera instrinsic matrix
     - R: Camera rotation matrix
     - x0: Principal point
    one can easily project a 2D point on the image back to the 3D world, if one dimension in the world is fixed.
    This function allows projection of multiple points at once.

    Important: Singularity happens when the points lies on the u=0, v=0, w=0 in the image's coordinate system.

    Input:
        H: Numpy 3-by-3 matrix representating R.T @ K^-1.
        x0: Array-like 3D vector representating principal point's coordinate in the world frame.
        image_pts: 2-by-N Numpy.array of N 2D u-v points in the image.
        z_world: 1-by-N Numpy.array of N known z-coordinates in the world.
    Output:
        3-by-N Numpy.array of N 3D coordinates in the world frame.
    """
    if not isinstance(image_pts, np.ndarray):
        image_pts = np.array(image_pts)

    # Make sure shapes are right
    if image_pts.ndim != 2:
        image_pts = image_pts.reshape((2, -1))    # 2-by-N

    # Number of points
    n_pts = image_pts.shape[1]
    
    if isinstance(z_world, (int, float)):
        z_world = z_world * np.ones((1, n_pts))
    elif not isinstance(z_world, np.ndarray):
        z_world = np.array(z_world)
    
    # Make sure shapes are right
    if z_world.ndim != 2:
        z_world = z_world.reshape((1, -1))        # 1-by-N

    image_pt_homo = np.concatenate((image_pts, np.ones((1, n_pts))), axis=0)
    b = H @ image_pt_homo
    W_prime = b[2] / (z_world - x0[2])

    x_world = b[0] / W_prime + x0[0]
    y_world = b[1] / W_prime + x0[1]

    return np.concatenate((x_world, y_world, z_world), axis=0)


def world2im(P, world_pts):
    """
    Project 3D world points to 2D image points.

    Input:
        P: Numpy 3-by-4 matrix representation camera calibration matrix.
        world_pts: 3-by-N Numpy.array of N 3D points in the world frame.
    Output:
        image_pts: 2-by-N Numpy.array of N 2D points in the image.
    """
    # Make sure shapes is right
    if not isinstance(world_pts, np.ndarray):
        world_pts = np.array(world_pts)
    if world_pts.ndim != 2:
        world_pts = world_pts.reshape((3, -1))
    n_pts = world_pts.shape[1]

    world_pts_homo = np.concatenate((world_pts, np.ones((1, n_pts))), axis=0)
    image_pts_homo = P @ world_pts_homo

    return (image_pts_homo[0:2, :] / image_pts_homo[2, :]).astype(np.int)
