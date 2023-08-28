# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import torch
import numpy as np
import math
import glob, os
import argparse
import open3d as o3d


def make_open3d_point_cloud(xyz, color=None, voxel_size=None):
    if np.isnan(xyz).any():
        return None

    xyz = xyz[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size)

    return pcd


def compute_overlap_ratio(pcd0, pcd1, voxel_size):
    pcd0_down = pcd0.voxel_down_sample(voxel_size)
    pcd1_down = pcd1.voxel_down_sample(voxel_size)
    matching01 = get_matching_indices(pcd0_down, pcd1_down, voxel_size * 1.5, 1)
    matching10 = get_matching_indices(pcd1_down, pcd0_down, voxel_size * 1.5, 1)
    overlap0 = float(len(matching01)) / float(len(pcd0_down.points))
    overlap1 = float(len(matching10)) / float(len(pcd1_down.points))
    return max(overlap0, overlap1)


def get_matching_indices(source, pcd_tree, search_voxel_size, K=None):
    match_inds = []
    for i, point in enumerate(source.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds


def compute_full_overlapping(data_root, scene_id, voxel_size=0.05):
    _points = [
        (
            pcd_name,
            make_open3d_point_cloud(
                torch.load(pcd_name)["coord"], voxel_size=voxel_size
            ),
        )
        for pcd_name in glob.glob(os.path.join(data_root, scene_id, "pcd", "*.pth"))
    ]
    points = [(pcd_name, pcd) for (pcd_name, pcd) in _points if pcd is not None]
    print(
        "load {} point clouds ({} invalid has been filtered), computing matching/overlapping".format(
            len(points), len(_points) - len(points)
        )
    )

    matching_matrix = np.zeros((len(points), len(points)))
    for i, (pcd0_name, pcd0) in enumerate(points):
        print("matching to...{}".format(pcd0_name))
        pcd0_tree = o3d.geometry.KDTreeFlann(copy.deepcopy(pcd0))
        for j, (pcd1_name, pcd1) in enumerate(points):
            if i == j:
                continue
            matching_matrix[i, j] = float(
                len(get_matching_indices(pcd1, pcd0_tree, 1.5 * voxel_size, 1))
            ) / float(len(pcd1.points))

    # write to file
    with open(os.path.join(data_root, scene_id, "pcd", "overlap.txt"), "w") as f:
        for i, (pcd0_name, pcd0) in enumerate(points):
            for j, (pcd1_name, pcd1) in enumerate(points):
                if i < j:
                    overlap = max(matching_matrix[i, j], matching_matrix[j, i])
                    f.write(
                        "{} {} {}\n".format(
                            pcd0_name.replace(data_root, ""),
                            pcd1_name.replace(data_root, ""),
                            overlap,
                        )
                    )
