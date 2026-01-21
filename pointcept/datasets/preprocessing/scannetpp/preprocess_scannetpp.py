"""
Preprocessing Script for ScanNet++
modified from official preprocess code.

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import warnings
import argparse
import json
import torch
import numpy as np
import pandas as pd
import open3d as o3d
import multiprocessing as mp
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

try:
    import pointseg
except:
    # Pointseg is located in libs/pointseg
    warnings.warn("Pointseg is not installed, superpoint segmentation will be skipped.")
    pointseg = None


def parse_scene(
    name,
    split,
    dataset_root,
    output_root,
    label_mapping,
    class2idx,
    ignore_index=-1,
):
    print(f"Parsing scene {name} in {split} split")
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    scene_path = dataset_root / "data" / name / "scans"
    mesh_path = scene_path / "mesh_aligned_0.05.ply"
    segs_path = scene_path / "segments.json"
    anno_path = scene_path / "segments_anno.json"

    # load mesh vertices and colors
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    # extract mesh information
    mesh.compute_vertex_normals(normalized=True)
    coord = np.array(mesh.vertices).astype(np.float32)
    color = (np.array(mesh.vertex_colors) * 255).astype(np.uint8)
    normal = np.array(mesh.vertex_normals).astype(np.float32)

    # extract superpoint information
    if pointseg is not None:
        vertices_sp = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
        faces_sp = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
        superpoint = pointseg.segment_mesh(vertices_sp, faces_sp).numpy()
    else:
        superpoint = None

    save_path = output_root / split / name
    save_path.mkdir(parents=True, exist_ok=True)
    np.save(save_path / "coord.npy", coord)
    np.save(save_path / "color.npy", color)
    np.save(save_path / "normal.npy", normal)
    if superpoint is not None:
        np.save(save_path / "superpoint.npy", superpoint)

    if split == "test":
        return

    # get label on vertices
    # load segments = vertices per segment ID
    with open(segs_path) as f:
        segments = json.load(f)
    # load anno = (instance, groups of segments)
    with open(anno_path) as f:
        anno = json.load(f)
    seg_indices = np.array(segments["segIndices"], dtype=np.uint32)
    num_vertices = len(seg_indices)
    assert num_vertices == len(coord)
    semantic_gt = np.ones((num_vertices, 3), dtype=np.int16) * ignore_index
    instance_gt = np.ones((num_vertices, 3), dtype=np.int16) * ignore_index

    # number of labels are used per vertex. initially 0
    # increment each time a new label is added
    instance_size = np.ones((num_vertices, 3), dtype=np.int16) * np.inf

    # keep track of the size of the instance (#vertices) assigned to each vertex
    # later, keep the label of the smallest instance for major label of vertices
    # store inf initially so that we can pick the smallest instance
    labels_used = np.zeros(num_vertices, dtype=np.int16)

    for idx, instance in enumerate(anno["segGroups"]):
        label = instance["label"]
        instance["label_orig"] = label
        # remap label
        instance["label"] = label_mapping.get(label, None)
        instance["label_index"] = class2idx.get(instance["label"], ignore_index)

        if instance["label_index"] == ignore_index:
            continue
        # get all the vertices with segment index in this instance
        # and max number of labels not yet applied
        mask = np.isin(seg_indices, instance["segments"]) & (labels_used < 3)
        size = mask.sum()
        if size == 0:
            continue

        # get the position to add the label - 0, 1, 2
        label_position = labels_used[mask]
        semantic_gt[mask, label_position] = instance["label_index"]
        # store all valid instance (include ignored instance)
        instance_gt[mask, label_position] = instance["objectId"]
        instance_size[mask, label_position] = size
        labels_used[mask] += 1

    # major label is the label of smallest instance for each vertex
    # use major label for single class segmentation
    # shift major label to the first column
    mask = labels_used > 1
    if mask.sum() > 0:
        major_label_position = np.argmin(instance_size[mask], axis=1)

        major_semantic_label = semantic_gt[mask, major_label_position]
        semantic_gt[mask, major_label_position] = semantic_gt[:, 0][mask]
        semantic_gt[:, 0][mask] = major_semantic_label

        major_instance_label = instance_gt[mask, major_label_position]
        instance_gt[mask, major_label_position] = instance_gt[:, 0][mask]
        instance_gt[:, 0][mask] = major_instance_label

    np.save(save_path / "segment.npy", semantic_gt)
    np.save(save_path / "instance.npy", instance_gt)


def filter_map_classes(mapping, mapping_type):
    if mapping_type == "semantic":
        map_key = "semantic_map_to"
    elif mapping_type == "instance":
        map_key = "instance_map_to"
    else:
        raise NotImplementedError
    # create a dict with classes to be mapped
    # classes that don't have mapping are entered as x->x
    # otherwise x->y
    map_dict = OrderedDict()

    for i in range(mapping.shape[0]):
        row = mapping.iloc[i]
        class_name = row["class"]
        map_target = row[map_key]

        # map to None or some other label -> don't add this class to the label list
        try:
            if len(map_target) > 0:
                # map to None -> don't use this class
                if map_target == "None":
                    pass
                else:
                    # map to something else -> use this class
                    map_dict[class_name] = map_target
        except TypeError:
            # nan values -> no mapping, keep label as is
            if class_name not in map_dict:
                map_dict[class_name] = class_name

    return map_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the ScanNet++ dataset containing data/metadata/splits.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val/test folders will be located.",
    )
    parser.add_argument(
        "--ignore_index",
        default=-1,
        type=int,
        help="Default ignore index.",
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )
    config = parser.parse_args()

    print("Loading meta data...")
    config.dataset_root = Path(config.dataset_root)
    config.output_root = Path(config.output_root)

    train_list = np.loadtxt(
        config.dataset_root / "splits" / "nvs_sem_train.txt",
        dtype=str,
    )
    print("Num samples in training split:", len(train_list))

    val_list = np.loadtxt(
        config.dataset_root / "splits" / "nvs_sem_val.txt",
        dtype=str,
    )
    print("Num samples in validation split:", len(val_list))

    test_list = np.loadtxt(
        config.dataset_root / "splits" / "sem_test.txt",
        dtype=str,
    )
    print("Num samples in testing split:", len(test_list))

    data_list = np.concatenate([train_list, val_list, test_list])
    split_list = np.concatenate(
        [
            np.full_like(train_list, "train"),
            np.full_like(val_list, "val"),
            np.full_like(test_list, "test"),
        ]
    )

    # Parsing label information and mapping
    segment_class_names = np.loadtxt(
        config.dataset_root / "metadata" / "semantic_benchmark" / "top100.txt",
        dtype=str,
        delimiter=".",  # dummy delimiter to replace " "
    )
    print("Num classes in segment class list:", len(segment_class_names))

    instance_class_names = np.loadtxt(
        config.dataset_root / "metadata" / "semantic_benchmark" / "top100_instance.txt",
        dtype=str,
        delimiter=".",  # dummy delimiter to replace " "
    )
    print("Num classes in instance class list:", len(instance_class_names))

    label_mapping = pd.read_csv(
        config.dataset_root / "metadata" / "semantic_benchmark" / "map_benchmark.csv"
    )
    label_mapping = filter_map_classes(label_mapping, mapping_type="semantic")
    class2idx = {
        class_name: idx for (idx, class_name) in enumerate(segment_class_names)
    }

    print("Processing scenes...")
    pool = ProcessPoolExecutor(max_workers=config.num_workers)
    _ = list(
        pool.map(
            parse_scene,
            data_list,
            split_list,
            repeat(config.dataset_root),
            repeat(config.output_root),
            repeat(label_mapping),
            repeat(class2idx),
            repeat(config.ignore_index),
        )
    )
    pool.shutdown()
    # parse_scene(
    #     data_list[0],
    #     split_list[0],
    #     config.dataset_root,
    #     config.output_root,
    #     label_mapping,
    #     class2idx,
    #     config.ignore_index,
    # )
