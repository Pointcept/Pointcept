import cv2
import math
import numpy as np


def eulerAnglesToRotationMatrix(theta):
    """Euler rotation matrix with clockwise logic.
    Rotation

    Args:
        theta: list of float
            [theta_x, theta_y, theta_z]
    Returns:
        R: np.array (3, 3)
            rotation matrix of Rz*Ry*Rx
    """
    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta[0]), -math.sin(theta[0])],
            [0, math.sin(theta[0]), math.cos(theta[0])],
        ]
    )

    R_y = np.array(
        [
            [math.cos(theta[1]), 0, math.sin(theta[1])],
            [0, 1, 0],
            [-math.sin(theta[1]), 0, math.cos(theta[1])],
        ]
    )

    R_z = np.array(
        [
            [math.cos(theta[2]), -math.sin(theta[2]), 0],
            [math.sin(theta[2]), math.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def upright_camera_relative_transform(pose):
    """Generate pose matrix with z-dim as height

    Args:
        pose: np.array (4, 4)
    Returns:
        urc: (4, 4)
        urc_inv: (4, 4)
    """

    # take viewing direction in camera local coordiantes (which is simply unit vector along +z)
    view_dir_camera = np.asarray([0, 0, 1])
    R = pose[0:3, 0:3]
    t = pose[0:3, 3]

    # convert to world coordinates
    view_dir_world = np.dot(R, view_dir_camera)

    # compute heading
    view_dir_xy = view_dir_world[0:2]
    heading = math.atan2(view_dir_xy[1], view_dir_xy[0])

    # compute rotation around Z to align heading with +Y
    zRot = -heading + math.pi / 2

    # translation first, back to camera point
    urc_t = np.identity(4)
    urc_t[0:2, 3] = -1 * t[0:2]

    # compute rotation matrix
    urc_r = np.identity(4)
    urc_r[0:3, 0:3] = eulerAnglesToRotationMatrix([0, 0, zRot])

    urc = np.dot(urc_r, urc_t)
    urc_inv = np.linalg.inv(urc)

    return urc, urc_inv


def rotate_pc(pc, rotmat):
    """Rotation points w.r.t. rotmat
    Args:
        pc: np.array (n, 3)
        rotmat: np.array (4, 4)
    Returns:
        pc: (n, 3)
    """
    pc_4 = np.ones([pc.shape[0], 4])
    pc_4[:, 0:3] = pc
    pc_4 = np.dot(pc_4, np.transpose(rotmat))

    return pc_4[:, 0:3]


def rotate_points_along_z(points, angle):
    """Rotation clockwise
    Args:
        points: np.array of np.array (B, N, 3 + C) or
            (N, 3 + C) for single batch
        angle: np.array of np.array (B, )
            or (, ) for single batch
            angle along z-axis, angle increases x ==> y
    Returns:
        points_rot:  (B, N, 3 + C) or (N, 3 + C)

    """
    single_batch = len(points.shape) == 2
    if single_batch:
        points = np.expand_dims(points, axis=0)
        angle = np.expand_dims(angle, axis=0)
    cosa = np.expand_dims(np.cos(angle), axis=1)
    sina = np.expand_dims(np.sin(angle), axis=1)
    zeros = np.zeros_like(cosa)  # angle.new_zeros(points.shape[0])
    ones = np.ones_like(sina)  # angle.new_ones(points.shape[0])

    rot_matrix = np.concatenate(
        (cosa, -sina, zeros, sina, cosa, zeros, zeros, zeros, ones), axis=1
    ).reshape(-1, 3, 3)

    # print(rot_matrix.view(3, 3))
    points_rot = np.matmul(points[:, :, :3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)

    if single_batch:
        points_rot = points_rot.squeeze(0)

    return points_rot


def convert_angle_axis_to_matrix3(angle_axis):
    """Return a Matrix3 for the angle axis.
    Arguments:
        angle_axis {Point3} -- a rotation in angle axis form.
    """
    matrix, jacobian = cv2.Rodrigues(angle_axis)
    return matrix
