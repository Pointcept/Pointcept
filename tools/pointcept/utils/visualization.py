"""
Visualization Utils

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import open3d as o3d
import numpy as np
import torch


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().cpu().numpy()
    assert isinstance(x, np.ndarray)
    return x


def save_point_cloud(coord, color=None, file_path="pc.ply", logger=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    coord = to_numpy(coord)
    if color is not None:
        color = to_numpy(color)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(
        np.ones_like(coord) if color is None else color
    )
    o3d.io.write_point_cloud(file_path, pcd)
    if logger is not None:
        logger.info(f"Save Point Cloud to: {file_path}")


def save_bounding_boxes(
    bboxes_corners, color=(1.0, 0.0, 0.0), file_path="bbox.ply", logger=None
):
    bboxes_corners = to_numpy(bboxes_corners)
    # point list
    points = bboxes_corners.reshape(-1, 3)
    # line list
    box_lines = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 0],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
    )
    lines = []
    for i, _ in enumerate(bboxes_corners):
        lines.append(box_lines + i * 8)
    lines = np.concatenate(lines)
    # color list
    color = np.array([color for _ in range(len(lines))])
    # generate line set
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_line_set(file_path, line_set)

    if logger is not None:
        logger.info(f"Save Boxes to: {file_path}")


def save_lines(
    points, lines, color=(1.0, 0.0, 0.0), file_path="lines.ply", logger=None
):
    points = to_numpy(points)
    lines = to_numpy(lines)
    colors = np.array([color for _ in range(len(lines))])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_line_set(file_path, line_set)

    if logger is not None:
        logger.info(f"Save Lines to: {file_path}")
