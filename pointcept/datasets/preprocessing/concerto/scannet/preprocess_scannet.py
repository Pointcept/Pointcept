"""
Preprocessing Script for ScanNet 20/200

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import camtools as ct
import open3d as o3d
from scipy.spatial import cKDTree
import struct
import zlib
import imageio
import cv2
import argparse
import glob
import json
import plyfile
import numpy as np
import pandas as pd
import multiprocessing as mp
from pathlib import Path

# Load external constants
import sys

sys.path.append("pointcept/datasets/preprocessing/scannet/meta_data")
from scannet200_constants import VALID_CLASS_IDS_200, VALID_CLASS_IDS_20

CLOUD_FILE_PFIX = "_vh_clean_2"
SEGMENTS_FILE_PFIX = ".0.010000.segs.json"
AGGREGATIONS_FILE_PFIX = ".aggregation.json"
CLASS_IDS200 = VALID_CLASS_IDS_200
CLASS_IDS20 = VALID_CLASS_IDS_20
IGNORE_INDEX = -1

COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {
    -1: "unknown",
    0: "raw_ushort",
    1: "zlib_ushort",
    2: "occi_ushort",
}


class RGBDFrame:
    def load(self, file_handle):
        self.camera_to_world = np.asarray(
            struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack("Q", file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.color_data = b"".join(
            struct.unpack(
                "c" * self.color_size_bytes, file_handle.read(self.color_size_bytes)
            )
        )
        self.depth_data = b"".join(
            struct.unpack(
                "c" * self.depth_size_bytes, file_handle.read(self.depth_size_bytes)
            )
        )

    def decompress_depth(self, compression_type):
        if compression_type == "zlib_ushort":
            return self.decompress_depth_zlib()
        else:
            raise

    def decompress_depth_zlib(self):
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        if compression_type == "jpeg":
            return self.decompress_color_jpeg()
        else:
            raise

    def decompress_color_jpeg(self):
        return imageio.imread(self.color_data)


class SensorData:
    def __init__(self, filename):
        self.version = 4
        self.load(filename)

    def load(self, filename):
        with open(filename, "rb") as f:
            version = struct.unpack("I", f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack("Q", f.read(8))[0]
            self.sensor_name = b"".join(struct.unpack("c" * strlen, f.read(strlen)))
            self.intrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.intrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[
                struct.unpack("i", f.read(4))[0]
            ]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[
                struct.unpack("i", f.read(4))[0]
            ]
            self.color_width = struct.unpack("I", f.read(4))[0]
            self.color_height = struct.unpack("I", f.read(4))[0]
            self.depth_width = struct.unpack("I", f.read(4))[0]
            self.depth_height = struct.unpack("I", f.read(4))[0]
            self.depth_shift = struct.unpack("f", f.read(4))[0]
            num_frames = struct.unpack("Q", f.read(8))[0]
            self.frames = []
            for i in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)

    def export_depth_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(
            "exporting", len(self.frames) // frame_skip, " depth frames to", output_path
        )
        for f in range(0, len(self.frames), frame_skip):
            if os.path.exists((os.path.join(output_path, str(f) + ".png"))):
                continue
            if f % 100 == 0:
                print(
                    "exporting",
                    f,
                    "th depth frames to",
                    os.path.join(output_path, str(f) + ".png"),
                )

            depth_data = self.frames[f].decompress_depth(self.depth_compression_type)
            depth = np.fromstring(depth_data, dtype=np.uint16).reshape(
                self.depth_height, self.depth_width
            )
            if image_size is not None:
                depth = cv2.resize(
                    depth,
                    (image_size[1], image_size[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            imageio.imwrite(os.path.join(output_path, str(f) + ".png"), depth)

    def export_color_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(
            "exporting", len(self.frames) // frame_skip, "color frames to", output_path
        )
        for f in range(0, len(self.frames), frame_skip):
            if os.path.exists((os.path.join(output_path, str(f) + ".png"))):
                continue
            if f % 100 == 0:
                print(
                    "exporting",
                    f,
                    "th color frames to",
                    os.path.join(output_path, str(f) + ".png"),
                )
            color = self.frames[f].decompress_color(self.color_compression_type)
            if image_size is not None:
                color = cv2.resize(
                    color,
                    (image_size[1], image_size[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            # imageio.imwrite(os.path.join(output_path, str(f) + '.jpg'), color)
            imageio.imwrite(os.path.join(output_path, str(f) + ".png"), color)

    def save_mat_to_file(self, matrix, filename):
        with open(filename, "w") as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt="%f")

    def export_poses(self, output_path, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(
            "exporting", len(self.frames) // frame_skip, "camera poses to", output_path
        )
        for f in range(0, len(self.frames), frame_skip):
            np.save(
                os.path.join(output_path, str(f) + ".npy"),
                self.frames[f].camera_to_world,
            )

    def export_intrinsics(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print("exporting camera intrinsics to", output_path)
        np.save(os.path.join(output_path, "intrinsic.npy"), self.intrinsic_color)


def reader(
    filename,
    output_path,
    frame_skip,
    export_color_images=False,
    export_depth_images=False,
    export_poses=False,
    export_intrinsics=False,
):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # load the data
    print("loading %s..." % filename)
    sd = SensorData(filename)
    if export_depth_images:
        sd.export_depth_images(
            os.path.join(output_path, "depth"), frame_skip=frame_skip
        )
    if export_color_images:
        sd.export_color_images(
            os.path.join(output_path, "color"), frame_skip=frame_skip
        )
    if export_poses:
        sd.export_poses(os.path.join(output_path, "pose"), frame_skip=frame_skip)
    if export_intrinsics:
        sd.export_intrinsics(os.path.join(output_path, "intrinsic"))
    return sd.color_height, sd.color_width


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


def correspondenceGet(mesh, K, T, img_size, coords_gt):
    height, width = img_size
    if np.isnan(T).any() or np.isinf(T).any():
        return None

    depth = ct.raycast.mesh_to_im_depth(
        mesh=mesh, K=K, T=np.linalg.inv(T), height=height, width=width
    )

    pixel = np.transpose(np.indices((width, height)), (2, 1, 0))
    pixel = pixel.reshape((-1, 2))
    pixel = np.hstack((pixel, np.ones((pixel.shape[0], 1))))
    depth = depth.reshape((-1, 1))
    valid = ~np.isinf(depth).squeeze(-1)
    coord = np.zeros_like(pixel, dtype=np.float32)
    coord[valid] = depth[valid] * (np.linalg.inv(K) @ pixel[valid].T).T  # coord_camera
    coord[valid] = coord[valid] @ T[:3, :3].T + T[:3, 3]  # column then row

    pixel = pixel[valid]
    coord = coord[valid]
    if coord.shape[0] == 0:
        return None
    pixel = pixel[:, :2]
    coord_dict = {"pixel": pixel, "coord": coord}
    return coord_dict


def correspondenceSave(mesh, scene_dir, coords_gt, output_dir, img_size):
    os.makedirs(output_dir, exist_ok=True)
    scene_dir = Path(scene_dir)
    index_gt = [
        img_name.split(".")[0]
        for img_name in os.listdir(str(scene_dir / "color"))
        if img_name.endswith(".png")
    ]
    index_gt = sorted(index_gt, key=lambda x: int(x))

    Ks_path = str(scene_dir / "intrinsic" / "intrinsic.npy")
    Ts_path = str(scene_dir / "pose")
    Ts_files = sorted(
        [f for f in os.listdir(Ts_path) if f.endswith(".npy")],
        key=lambda x: int(x.split(".")[0]),
    )

    print(f"total pose num:{len(Ts_files)}")
    Ts = []
    for Ts_file in Ts_files:
        file_path = os.path.join(Ts_path, Ts_file)
        Ts_ = np.load(file_path)
        Ts.append(Ts_)
    Ts = np.stack(Ts)
    Ks = np.load(Ks_path)

    Ks = np.tile(Ks, (Ts.shape[0], 1, 1))
    Ks = Ks[:, :3, :3]
    coords_gt_ = coords_gt
    pixels_ = []
    coords_ = []

    for i, (K, T) in enumerate(zip(Ks, Ts)):
        coord_dict = correspondenceGet(mesh, K, T, img_size, coords_gt)
        if coord_dict is None:
            correspondences = -np.ones((1, 3))
        else:
            pixels_ = coord_dict["pixel"]
            coords_ = coord_dict["coord"]
            tree = cKDTree(coords_gt_)
            dis, idx = tree.query(coords_, k=1)
            idx_valid = idx[dis < 0.01]
            pixels_valid = pixels_[dis < 0.01]
            correspondences = np.hstack((pixels_valid, idx_valid.reshape(-1, 1)))
        np.save(Path(output_dir) / "{}.npy".format(index_gt[i]), correspondences)


def handle_process(
    scene_path,
    output_path,
    pointclouds_root,
    labels_pd,
    train_scenes,
    val_scenes,
    frame_gap=75,
    parse_pointclouds=True,
    parse_normals=True,
    export_depth_images=True,
):
    pc_output_path = output_path
    im_output_path = os.path.join(output_path, "images")
    scene_id = os.path.basename(scene_path)
    mesh_path = os.path.join(scene_path, f"{scene_id}{CLOUD_FILE_PFIX}.ply")
    segments_file = os.path.join(
        scene_path, f"{scene_id}{CLOUD_FILE_PFIX}{SEGMENTS_FILE_PFIX}"
    )
    aggregations_file = os.path.join(scene_path, f"{scene_id}{AGGREGATIONS_FILE_PFIX}")

    if scene_id in train_scenes:
        pc_output_path = os.path.join(pc_output_path, "train", f"{scene_id}")
        pc_input_path = os.path.join(pointclouds_root, "train", f"{scene_id}")
        im_output_path = os.path.join(im_output_path, "train", f"{scene_id}")
        split_name = "train"
    elif scene_id in val_scenes:
        pc_output_path = os.path.join(pc_output_path, "val", f"{scene_id}")
        pc_input_path = os.path.join(pointclouds_root, "val", f"{scene_id}")
        im_output_path = os.path.join(im_output_path, "val", f"{scene_id}")
        split_name = "val"
    else:
        pc_output_path = os.path.join(pc_output_path, "test", f"{scene_id}")
        pc_input_path = os.path.join(pointclouds_root, "test", f"{scene_id}")
        im_output_path = os.path.join(im_output_path, "test", f"{scene_id}")
        split_name = "test"

    print(f"Processing: {scene_id} in {split_name}")

    if parse_pointclouds:
        vertices, faces = read_plymesh(mesh_path)
        coords = vertices[:, :3]
        colors = vertices[:, 3:6]
        save_dict = dict(
            coord=coords.astype(np.float32),
            color=colors.astype(np.uint8),
        )
        # Parse Normals
        if parse_normals:
            save_dict["normal"] = vertex_normal(coords, faces).astype(np.float32)

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
            semantic_gt20 = np.ones((vertices.shape[0]), dtype=np.int16) * IGNORE_INDEX
            semantic_gt200 = np.ones((vertices.shape[0]), dtype=np.int16) * IGNORE_INDEX
            instance_ids = np.ones((vertices.shape[0]), dtype=np.int16) * IGNORE_INDEX
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

            save_dict["segment20"] = semantic_gt20
            save_dict["segment200"] = semantic_gt200
            save_dict["instance"] = instance_ids

            # Concatenate with original cloud
            processed_vertices = np.hstack((semantic_gt200, instance_ids))

            if np.any(np.isnan(processed_vertices)) or not np.all(
                np.isfinite(processed_vertices)
            ):
                raise ValueError(f"Find NaN in Scene: {scene_id}")

        # Save pointcloud data
        os.makedirs(pc_output_path, exist_ok=True)
        for key in save_dict.keys():
            np.save(os.path.join(pc_output_path, f"{key}.npy"), save_dict[key])
    else:
        coords = np.load(os.path.join(pc_input_path, "coord.npy"))
        save_dict = dict(
            coord=coords.astype(np.float32),
        )

    # Save img data
    os.makedirs(im_output_path, exist_ok=True)
    sens_dir = os.path.join(scene_path, scene_id + ".sens")
    print(f"Parsing sens data{sens_dir}")
    h, w = reader(
        sens_dir,
        im_output_path,
        frame_gap,
        export_color_images=True,
        export_depth_images=export_depth_images,
        export_poses=True,
        export_intrinsics=True,
    )
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    correspondenceSave(
        mesh,
        im_output_path,
        save_dict["coord"],
        os.path.join(im_output_path, "correspondence"),
        (h, w),
    )


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
        "--pointclouds_root",
        default="data/scannet",
        type=str,
        help="Input path where previous pointclouds folder located",
    )
    parser.add_argument(
        "--frame_gap", default=75, type=int, help="Frame gap for processing"
    )
    parser.add_argument(
        "--parse_pointclouds", action="store_true", help="Whether parse point clouds"
    )
    parser.add_argument(
        "--parse_normals", action="store_true", help="Whether parse point normals"
    )
    parser.add_argument(
        "--parse_depths", action="store_true", help="Whether parse depths"
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )
    parser.add_argument(
        "--thread_id",
        default=0,
        type=int,
        help="Thread id for parallel processing",
    )
    config = parser.parse_args()
    meta_root = Path("pointcept/datasets/preprocessing/scannet") / "meta_data"

    # Load label map
    labels_pd = pd.read_csv(
        meta_root / "scannetv2-labels.combined.tsv",
        sep="\t",
        header=0,
    )

    # Load train/val splits
    with open(meta_root / "scannetv2_train.txt") as train_file:
        train_scenes = train_file.read().splitlines()
    with open(meta_root / "scannetv2_val.txt") as val_file:
        val_scenes = val_file.read().splitlines()

    # Load scene paths
    scene_paths = sorted(glob.glob(config.dataset_root + "/scans*/scene*"))
    scene_paths_list = np.array_split(scene_paths, config.num_workers)
    scene_paths_ = scene_paths_list[config.thread_id]
    # Preprocess data.
    print("Processing scenes...")
    for scene_paths_i in scene_paths_:
        handle_process(
            scene_paths_i,
            config.output_root,
            config.pointclouds_root,
            labels_pd,
            train_scenes,
            val_scenes,
            config.frame_gap,
            config.parse_pointclouds,
            config.parse_normals,
            config.parse_depths,
        )
