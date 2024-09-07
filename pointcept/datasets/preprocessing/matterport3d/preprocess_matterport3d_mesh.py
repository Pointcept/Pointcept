"""
Preprocessing Matterport3D
adatpted from https://github.com/pengsongyou/openscene/blob/main/scripts/preprocess/preprocess_3d_matterport.py
"""

import os
import argparse
import glob
import plyfile
import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
import torch

MATTERPORT_CLASS_REMAP = np.zeros(41)
MATTERPORT_CLASS_REMAP[1] = 1
MATTERPORT_CLASS_REMAP[2] = 2
MATTERPORT_CLASS_REMAP[3] = 3
MATTERPORT_CLASS_REMAP[4] = 4
MATTERPORT_CLASS_REMAP[5] = 5
MATTERPORT_CLASS_REMAP[6] = 6
MATTERPORT_CLASS_REMAP[7] = 7
MATTERPORT_CLASS_REMAP[8] = 8
MATTERPORT_CLASS_REMAP[9] = 9
MATTERPORT_CLASS_REMAP[10] = 10
MATTERPORT_CLASS_REMAP[11] = 11
MATTERPORT_CLASS_REMAP[12] = 12
MATTERPORT_CLASS_REMAP[14] = 13
MATTERPORT_CLASS_REMAP[16] = 14
MATTERPORT_CLASS_REMAP[22] = 21  # DIFFERENCE TO SCANNET!
MATTERPORT_CLASS_REMAP[24] = 15
MATTERPORT_CLASS_REMAP[28] = 16
MATTERPORT_CLASS_REMAP[33] = 17
MATTERPORT_CLASS_REMAP[34] = 18
MATTERPORT_CLASS_REMAP[36] = 19
MATTERPORT_CLASS_REMAP[39] = 20

MATTERPORT_LABELS_21 = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "other",
    "ceiling",
)
MATTERPORT_ALLOWED_NYU_CLASSES = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    14,
    16,
    22,
    24,
    28,
    33,
    34,
    36,
    39,
]


def handle_process(mesh_path, output_path, mapping, train_scenes, val_scenes):
    # Get the scene id and region name from the mesh path
    scene_id = Path(mesh_path).parent.parent.name
    region_name = Path(mesh_path).stem

    # Check which split the scene belongs to (train, val, or test)
    if scene_id in train_scenes:
        output_folder = os.path.join(output_path, "train", scene_id)
        output_file = os.path.join(output_path, "train", scene_id, f"{region_name}.pth")
        split_name = "train"
    elif scene_id in val_scenes:
        output_folder = os.path.join(output_path, "val", scene_id)
        output_file = os.path.join(output_path, "val", scene_id, f"{region_name}.pth")
        split_name = "val"
    else:
        output_folder = os.path.join(output_path, "test", scene_id)
        output_file = os.path.join(output_path, "test", scene_id, f"{region_name}.pth")
        split_name = "test"

    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"Processing: {scene_id} in {split_name}")

    # Load the vertex data
    with open(mesh_path, "rb") as f:
        plydata = plyfile.PlyData.read(f)
    vertex_data = plydata["vertex"].data

    # Get the coordinates, colors, and normals from the vertex data
    coords = np.vstack([vertex_data["x"], vertex_data["y"], vertex_data["z"]]).T
    colors = np.vstack(
        [vertex_data["red"], vertex_data["green"], vertex_data["blue"]]
    ).T
    normals = np.vstack([vertex_data["nx"], vertex_data["ny"], vertex_data["nz"]]).T

    # Prepare the data to be saved
    save_dict = dict(
        coord=coords.astype("float32"),
        color=colors.astype("uint8"),
        normal=normals.astype("float32"),
        scene_id=scene_id + "_" + region_name,
    )

    # Load the face data
    face_data = plydata["face"].data
    category_id = face_data["category_id"]

    # Replace -1 with 0 in category_id
    category_id[category_id == -1] = 0

    # Map the labels according to NYU40ID
    mapped_labels = mapping[category_id]

    # Replace labels not in MATTERPORT_ALLOWED_NYU_CLASSES with 0
    mapped_labels[
        np.logical_not(np.isin(mapped_labels, MATTERPORT_ALLOWED_NYU_CLASSES))
    ] = 0

    # Remap the labels to ScanNet 20 categories + ceiling
    remapped_labels = MATTERPORT_CLASS_REMAP[mapped_labels].astype(int)

    # Calculate per-vertex labels
    triangles = face_data["vertex_indices"]
    vertex_labels = np.zeros((coords.shape[0], 22), dtype=np.int32)
    # calculate per-vertex labels
    for row_id in range(triangles.shape[0]):
        for i in range(3):
            vertex_labels[triangles[row_id][i], remapped_labels[row_id]] += 1

    # Get the most frequent label for each vertex
    vertex_labels = np.argmax(vertex_labels, axis=1)
    vertex_labels -= 1

    # Add the vertex labels to the data to be saved
    save_dict["semantic_gt21"] = vertex_labels.astype("int16")

    # Save processed data
    torch.save(save_dict, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the Matterport3D dataset containing scene folders",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val folders will be located",
    )
    opt = parser.parse_args()

    # Load label map
    category_mapping = pd.read_csv(
        "pointcept/datasets/preprocessing/matterport3d/meta_data/category_mapping.tsv",
        sep="\t",
        header=0,
    )
    mapping = np.insert(
        category_mapping[["nyu40id"]].to_numpy().astype(int).flatten(), 0, 0, axis=0
    )

    # Load train/val splits
    with open(
        "pointcept/datasets/preprocessing/matterport3d/meta_data/train.txt"
    ) as train_file:
        train_scenes = train_file.read().splitlines()
    with open(
        "pointcept/datasets/preprocessing/matterport3d/meta_data/val.txt"
    ) as val_file:
        val_scenes = val_file.read().splitlines()
    with open(
        "pointcept/datasets/preprocessing/matterport3d/meta_data/test.txt"
    ) as test_file:
        test_scenes = test_file.read().splitlines()

    # Create output directories
    os.makedirs(opt.output_root, exist_ok=True)
    train_output_dir = os.path.join(opt.output_root, "train")
    os.makedirs(train_output_dir, exist_ok=True)
    val_output_dir = os.path.join(opt.output_root, "val")
    os.makedirs(val_output_dir, exist_ok=True)
    test_output_dir = os.path.join(opt.output_root, "test")
    os.makedirs(test_output_dir, exist_ok=True)

    # Load scene paths
    scene_paths = sorted(glob.glob(opt.dataset_root + "/*/region_segmentations/*.ply"))

    # Preprocess data.
    pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
    # pool = ProcessPoolExecutor(max_workers=1)
    print("Processing scenes...")
    _ = list(
        pool.map(
            handle_process,
            scene_paths,
            repeat(opt.output_root),
            repeat(mapping),
            repeat(train_scenes),
            repeat(val_scenes),
        )
    )
