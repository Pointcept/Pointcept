"""
Preprocessing Script for ScanNet 20/200

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import warnings

import torch

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import argparse
import glob
import json
import plyfile
import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

# Load external constants
from meta_data.scannet200_constants import VALID_CLASS_IDS_200, VALID_CLASS_IDS_20

CLOUD_FILE_PFIX = "_vh_clean_2"
SEGMENTS_FILE_PFIX = ".0.010000.segs.json"
AGGREGATIONS_FILE_PFIX = ".aggregation.json"
CLASS_IDS200 = VALID_CLASS_IDS_200
CLASS_IDS20 = VALID_CLASS_IDS_20
IGNORE_INDEX = -1


def read_plymesh(filepath):
    """Read ply file and return it as numpy array. Returns None if emtpy."""
    with open(filepath, "rb") as f:
        plydata = plyfile.PlyData.read(f)
    if plydata.elements:
        vertices = pd.DataFrame(plydata["vertex"].data).values
        faces = np.stack(plydata["face"].data["vertex_indices"], axis=0)
        return vertices, faces


# Map the raw category id to the point cloud
def point_indices_from_group(seg_indices, group, labels_pd):
    group_segments = np.array(group["segments"])
    label = group["label"]

    # Map the category name to id
    label_id20 = labels_pd[labels_pd["raw_category"] == label]["nyu40id"]
    label_id20 = int(label_id20.iloc[0]) if len(label_id20) > 0 else 0
    label_id200 = labels_pd[labels_pd["raw_category"] == label]["id"]
    label_id200 = int(label_id200.iloc[0]) if len(label_id200) > 0 else 0

    # Only store for the valid categories
    if label_id20 in CLASS_IDS20:
        label_id20 = CLASS_IDS20.index(label_id20)
    else:
        label_id20 = IGNORE_INDEX

    if label_id200 in CLASS_IDS200:
        label_id200 = CLASS_IDS200.index(label_id200)
    else:
        label_id200 = IGNORE_INDEX

    # get points, where segment indices (points labelled with segment ids) are in the group segment list
    point_idx = np.where(np.isin(seg_indices, group_segments))[0]
    return point_idx, label_id20, label_id200


def face_normal(vertex, face):
    v01 = vertex[face[:, 1]] - vertex[face[:, 0]]
    v02 = vertex[face[:, 2]] - vertex[face[:, 0]]
    vec = np.cross(v01, v02)
    length = np.sqrt(np.sum(vec**2, axis=1, keepdims=True)) + 1.0e-8
    nf = vec / length
    area = length * 0.5
    return nf, area


def vertex_normal(vertex, face):
    nf, area = face_normal(vertex, face)
    nf = nf * area

    nv = np.zeros_like(vertex)
    for i in range(face.shape[0]):
        nv[face[i]] += nf[i]

    length = np.sqrt(np.sum(nv**2, axis=1, keepdims=True)) + 1.0e-8
    nv = nv / length
    return nv


def handle_process(
    scene_path, output_path, labels_pd, train_scenes, val_scenes, parse_normals=True
):
    scene_id = os.path.basename(scene_path)
    mesh_path = os.path.join(scene_path, f"{scene_id}{CLOUD_FILE_PFIX}.ply")
    segments_file = os.path.join(
        scene_path, f"{scene_id}{CLOUD_FILE_PFIX}{SEGMENTS_FILE_PFIX}"
    )
    aggregations_file = os.path.join(scene_path, f"{scene_id}{AGGREGATIONS_FILE_PFIX}")
    info_file = os.path.join(scene_path, f"{scene_id}.txt")

    if scene_id in train_scenes:
        output_file = os.path.join(output_path, "train", f"{scene_id}.pth")
        split_name = "train"
    elif scene_id in val_scenes:
        output_file = os.path.join(output_path, "val", f"{scene_id}.pth")
        split_name = "val"
    else:
        output_file = os.path.join(output_path, "test", f"{scene_id}.pth")
        split_name = "test"

    print(f"Processing: {scene_id} in {split_name}")

    vertices, faces = read_plymesh(mesh_path)
    coords = vertices[:, :3]
    colors = vertices[:, 3:6]
    save_dict = dict(coord=coords, color=colors, scene_id=scene_id)

    # # Rotating the mesh to axis aligned
    # info_dict = {}
    # with open(info_file) as f:
    #     for line in f:
    #         (key, val) = line.split(" = ")
    #         info_dict[key] = np.fromstring(val, sep=' ')
    #
    # if 'axisAlignment' not in info_dict:
    #     rot_matrix = np.identity(4)
    # else:
    #     rot_matrix = info_dict['axisAlignment'].reshape(4, 4)
    # r_coords = coords.transpose()
    # r_coords = np.append(r_coords, np.ones((1, r_coords.shape[1])), axis=0)
    # r_coords = np.dot(rot_matrix, r_coords)
    # coords = r_coords

    # Parse Normals
    if parse_normals:
        save_dict["normal"] = vertex_normal(coords, faces)

    # Load segments file
    if split_name != "test":
        with open(segments_file) as f:
            segments = json.load(f)
            seg_indices = np.array(segments["segIndices"])

        # Load Aggregations file
        with open(aggregations_file) as f:
            aggregation = json.load(f)
            seg_groups = np.array(aggregation["segGroups"])

        # Generate new labels
        semantic_gt20 = np.ones((vertices.shape[0])) * IGNORE_INDEX
        semantic_gt200 = np.ones((vertices.shape[0])) * IGNORE_INDEX
        instance_ids = np.ones((vertices.shape[0])) * IGNORE_INDEX
        for group in seg_groups:
            point_idx, label_id20, label_id200 = point_indices_from_group(
                seg_indices, group, labels_pd
            )

            semantic_gt20[point_idx] = label_id20
            semantic_gt200[point_idx] = label_id200
            instance_ids[point_idx] = group["id"]

        semantic_gt20 = semantic_gt20.astype(int)
        semantic_gt200 = semantic_gt200.astype(int)
        instance_ids = instance_ids.astype(int)

        save_dict["semantic_gt20"] = semantic_gt20
        save_dict["semantic_gt200"] = semantic_gt200
        save_dict["instance_gt"] = instance_ids

        # Concatenate with original cloud
        processed_vertices = np.hstack((semantic_gt200, instance_ids))

        if np.any(np.isnan(processed_vertices)) or not np.all(
            np.isfinite(processed_vertices)
        ):
            raise ValueError(f"Find NaN in Scene: {scene_id}")

    # Save processed data
    torch.save(save_dict, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the ScanNet dataset containing scene folders",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val folders will be located",
    )
    parser.add_argument(
        "--parse_normals", default=True, type=bool, help="Whether parse point normals"
    )
    config = parser.parse_args()

    # Load label map
    labels_pd = pd.read_csv(
        "pointcept/datasets/preprocessing/scannet/meta_data/scannetv2-labels.combined.tsv",
        sep="\t",
        header=0,
    )

    # Load train/val splits
    with open(
        "pointcept/datasets/preprocessing/scannet/meta_data/scannetv2_train.txt"
    ) as train_file:
        train_scenes = train_file.read().splitlines()
    with open(
        "pointcept/datasets/preprocessing/scannet/meta_data/scannetv2_val.txt"
    ) as val_file:
        val_scenes = val_file.read().splitlines()

    # Create output directories
    train_output_dir = os.path.join(config.output_root, "train")
    os.makedirs(train_output_dir, exist_ok=True)
    val_output_dir = os.path.join(config.output_root, "val")
    os.makedirs(val_output_dir, exist_ok=True)
    test_output_dir = os.path.join(config.output_root, "test")
    os.makedirs(test_output_dir, exist_ok=True)

    # Load scene paths
    scene_paths = sorted(glob.glob(config.dataset_root + "/scans*/scene*"))

    # Preprocess data.
    print("Processing scenes...")
    pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
    # pool = ProcessPoolExecutor(max_workers=1)
    _ = list(
        pool.map(
            handle_process,
            scene_paths,
            repeat(config.output_root),
            repeat(labels_pd),
            repeat(train_scenes),
            repeat(val_scenes),
            repeat(config.parse_normals),
        )
    )
