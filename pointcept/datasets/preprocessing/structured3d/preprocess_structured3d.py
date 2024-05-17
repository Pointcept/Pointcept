"""
Preprocessing Script for Structured3D

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import argparse
import io
import os
import PIL
from PIL import Image
import cv2
import zipfile
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat


VALID_CLASS_IDS_25 = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    11,
    14,
    15,
    16,
    17,
    18,
    19,
    22,
    24,
    25,
    32,
    34,
    35,
    38,
    39,
    40,
)
CLASS_LABELS_25 = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "picture",
    "desk",
    "shelves",
    "curtain",
    "dresser",
    "pillow",
    "mirror",
    "ceiling",
    "refrigerator",
    "television",
    "nightstand",
    "sink",
    "lamp",
    "otherstructure",
    "otherfurniture",
    "otherprop",
)


def normal_from_cross_product(points_2d: np.ndarray) -> np.ndarray:
    xyz_points_pad = np.pad(points_2d, ((0, 1), (0, 1), (0, 0)), mode="symmetric")
    xyz_points_ver = (xyz_points_pad[:, :-1, :] - xyz_points_pad[:, 1:, :])[:-1, :, :]
    xyz_points_hor = (xyz_points_pad[:-1, :, :] - xyz_points_pad[1:, :, :])[:, :-1, :]
    xyz_normal = np.cross(xyz_points_hor, xyz_points_ver)
    xyz_dist = np.linalg.norm(xyz_normal, axis=-1, keepdims=True)
    xyz_normal = np.divide(
        xyz_normal, xyz_dist, out=np.zeros_like(xyz_normal), where=xyz_dist != 0
    )
    return xyz_normal


class Structured3DReader:
    def __init__(self, files):
        super().__init__()
        if isinstance(files, str):
            files = [files]
        self.readers = [zipfile.ZipFile(f, "r") for f in files]
        self.names_mapper = dict()
        for idx, reader in enumerate(self.readers):
            for name in reader.namelist():
                self.names_mapper[name] = idx

    def filelist(self):
        return list(self.names_mapper.keys())

    def listdir(self, dir_name):
        dir_name = dir_name.lstrip(os.path.sep).rstrip(os.path.sep)
        file_list = list(
            np.unique(
                [
                    f.replace(dir_name + os.path.sep, "", 1).split(os.path.sep)[0]
                    for f in self.filelist()
                    if f.startswith(dir_name + os.path.sep)
                ]
            )
        )
        if "" in file_list:
            file_list.remove("")
        return file_list

    def read(self, file_name):
        split = self.names_mapper[file_name]
        return self.readers[split].read(file_name)

    def read_camera(self, camera_path):
        z2y_top_m = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
        cam_extr = np.fromstring(self.read(camera_path), dtype=np.float32, sep=" ")
        cam_t = np.matmul(z2y_top_m, cam_extr[:3] / 1000)
        if cam_extr.shape[0] > 3:
            cam_front, cam_up = cam_extr[3:6], cam_extr[6:9]
            cam_n = np.cross(cam_front, cam_up)
            cam_r = np.stack((cam_front, cam_up, cam_n), axis=1).astype(np.float32)
            cam_r = np.matmul(z2y_top_m, cam_r)
            cam_f = cam_extr[9:11]
        else:
            cam_r = np.eye(3, dtype=np.float32)
            cam_f = None
        return cam_r, cam_t, cam_f

    def read_depth(self, depth_path):
        depth = cv2.imdecode(
            np.frombuffer(self.read(depth_path), np.uint8), cv2.IMREAD_UNCHANGED
        )[..., np.newaxis]
        depth[depth == 0] = 65535
        return depth

    def read_color(self, color_path):
        color = cv2.imdecode(
            np.frombuffer(self.read(color_path), np.uint8), cv2.IMREAD_UNCHANGED
        )[..., :3][..., ::-1]
        return color

    def read_segment(self, segment_path):
        segment = np.array(PIL.Image.open(io.BytesIO(self.read(segment_path))))[
            ..., np.newaxis
        ]
        return segment


def parse_scene(
    scene,
    dataset_root,
    output_root,
    ignore_index=-1,
    grid_size=None,
    fuse_prsp=True,
    fuse_pano=True,
    vis=False,
):
    assert fuse_prsp or fuse_pano
    reader = Structured3DReader(
        [
            os.path.join(dataset_root, f)
            for f in os.listdir(dataset_root)
            if f.endswith(".zip")
        ]
    )
    scene_id = int(os.path.basename(scene).split("_")[-1])
    if scene_id < 3000:
        split = "train"
    elif 3000 <= scene_id < 3250:
        split = "val"
    else:
        split = "test"

    print(f"Processing: {scene} in {split}")
    rooms = reader.listdir(os.path.join("Structured3D", scene, "2D_rendering"))
    for room in rooms:
        room_path = os.path.join("Structured3D", scene, "2D_rendering", room)
        coord_list = list()
        color_list = list()
        normal_list = list()
        segment_list = list()
        if fuse_prsp:
            prsp_path = os.path.join(room_path, "perspective", "full")
            frames = reader.listdir(prsp_path)

            for frame in frames:
                try:
                    cam_r, cam_t, cam_f = reader.read_camera(
                        os.path.join(prsp_path, frame, "camera_pose.txt")
                    )
                    depth = reader.read_depth(
                        os.path.join(prsp_path, frame, "depth.png")
                    )
                    color = reader.read_color(
                        os.path.join(prsp_path, frame, "rgb_rawlight.png")
                    )
                    segment = reader.read_segment(
                        os.path.join(prsp_path, frame, "semantic.png")
                    )
                except:
                    print(
                        f"Skipping {scene}_room{room}_frame{frame} perspective view due to loading error"
                    )
                else:
                    fx, fy = cam_f
                    height, width = depth.shape[0], depth.shape[1]
                    pixel = np.transpose(np.indices((width, height)), (2, 1, 0))
                    pixel = pixel.reshape((-1, 2))
                    pixel = np.hstack((pixel, np.ones((pixel.shape[0], 1))))
                    k = np.diag([1.0, 1.0, 1.0])

                    k[0, 2] = width / 2
                    k[1, 2] = height / 2

                    k[0, 0] = k[0, 2] / np.tan(fx)
                    k[1, 1] = k[1, 2] / np.tan(fy)
                    coord = (
                        depth.reshape((-1, 1)) * (np.linalg.inv(k) @ pixel.T).T
                    ).reshape(height, width, 3)
                    coord = coord @ np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
                    normal = normal_from_cross_product(coord)

                    # Filtering invalid points
                    view_dist = np.maximum(
                        np.linalg.norm(coord, axis=-1, keepdims=True), float(10e-5)
                    )
                    cosine_dist = np.sum(
                        (coord * normal / view_dist), axis=-1, keepdims=True
                    )
                    cosine_dist = np.abs(cosine_dist)
                    mask = ((cosine_dist > 0.15) & (depth < 65535) & (segment > 0))[
                        ..., 0
                    ].reshape(-1)

                    coord = np.matmul(coord / 1000, cam_r.T) + cam_t
                    normal = normal_from_cross_product(coord)

                    if sum(mask) > 0:
                        coord_list.append(coord.reshape(-1, 3)[mask])
                        color_list.append(color.reshape(-1, 3)[mask])
                        normal_list.append(normal.reshape(-1, 3)[mask])
                        segment_list.append(segment.reshape(-1, 1)[mask])
                    else:
                        print(
                            f"Skipping {scene}_room{room}_frame{frame} perspective view due to all points are filtered out"
                        )

        if fuse_pano:
            pano_path = os.path.join(room_path, "panorama")
            try:
                _, cam_t, _ = reader.read_camera(
                    os.path.join(pano_path, "camera_xyz.txt")
                )
                depth = reader.read_depth(os.path.join(pano_path, "full", "depth.png"))
                color = reader.read_color(
                    os.path.join(pano_path, "full", "rgb_rawlight.png")
                )
                segment = reader.read_segment(
                    os.path.join(pano_path, "full", "semantic.png")
                )
            except:
                print(f"Skipping {scene}_room{room} panorama view due to loading error")
            else:
                p_h, p_w = depth.shape[:2]
                p_a = np.arange(p_w, dtype=np.float32) / p_w * 2 * np.pi - np.pi
                p_b = np.arange(p_h, dtype=np.float32) / p_h * np.pi * -1 + np.pi / 2
                p_a = np.tile(p_a[None], [p_h, 1])[..., np.newaxis]
                p_b = np.tile(p_b[:, None], [1, p_w])[..., np.newaxis]
                p_a_sin, p_a_cos, p_b_sin, p_b_cos = (
                    np.sin(p_a),
                    np.cos(p_a),
                    np.sin(p_b),
                    np.cos(p_b),
                )
                x = depth * p_a_cos * p_b_cos
                y = depth * p_b_sin
                z = depth * p_a_sin * p_b_cos
                coord = np.concatenate([x, y, z], axis=-1) / 1000
                normal = normal_from_cross_product(coord)

                # Filtering invalid points
                view_dist = np.maximum(
                    np.linalg.norm(coord, axis=-1, keepdims=True), float(10e-5)
                )
                cosine_dist = np.sum(
                    (coord * normal / view_dist), axis=-1, keepdims=True
                )
                cosine_dist = np.abs(cosine_dist)
                mask = ((cosine_dist > 0.15) & (depth < 65535) & (segment > 0))[
                    ..., 0
                ].reshape(-1)
                coord = coord + cam_t

                if sum(mask) > 0:
                    coord_list.append(coord.reshape(-1, 3)[mask])
                    color_list.append(color.reshape(-1, 3)[mask])
                    normal_list.append(normal.reshape(-1, 3)[mask])
                    segment_list.append(segment.reshape(-1, 1)[mask])
                else:
                    print(
                        f"Skipping {scene}_room{room} panorama view due to all points are filtered out"
                    )

        if len(coord_list) > 0:
            coord = np.concatenate(coord_list, axis=0)
            coord = coord @ np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
            color = np.concatenate(color_list, axis=0)
            normal = np.concatenate(normal_list, axis=0)
            normal = normal @ np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
            segment = np.concatenate(segment_list, axis=0)
            segment25 = np.ones_like(segment, dtype=np.int64) * ignore_index
            for idx, value in enumerate(VALID_CLASS_IDS_25):
                mask = np.all(segment == value, axis=-1)
                segment25[mask] = idx

            data_dict = dict(
                coord=coord.astype(np.float32),
                color=color.astype(np.uint8),
                normal=normal.astype(np.float32),
                segment=segment25.astype(np.int16),
            )
            # Grid sampling data
            if grid_size is not None:
                grid_coord = np.floor(coord / grid_size).astype(int)
                _, idx = np.unique(grid_coord, axis=0, return_index=True)
                coord = coord[idx]
                for key in data_dict.keys():
                    data_dict[key] = data_dict[key][idx]

            # Save data
            save_path = os.path.join(
                output_root, split, os.path.basename(scene), f"room_{room}"
            )
            os.makedirs(save_path, exist_ok=True)
            for key in data_dict.keys():
                np.save(os.path.join(save_path, f"{key}.npy"), data_dict[key])

            if vis:
                from pointcept.utils.visualization import save_point_cloud

                os.makedirs("./vis", exist_ok=True)
                save_point_cloud(
                    coord, color / 255, f"./vis/{scene}_room{room}_color.ply"
                )
                save_point_cloud(
                    coord, (normal + 1) / 2, f"./vis/{scene}_room{room}_normal.ply"
                )
        else:
            print(f"Skipping {scene}_room{room} due to no valid points")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the ScanNet dataset containing scene folders.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val folders will be located.",
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )
    parser.add_argument(
        "--grid_size", default=None, type=float, help="Grid size for grid sampling."
    )
    parser.add_argument("--ignore_index", default=-1, type=float, help="Ignore index.")
    parser.add_argument(
        "--fuse_prsp", action="store_true", help="Whether fuse perspective view."
    )
    parser.add_argument(
        "--fuse_pano", action="store_true", help="Whether fuse panorama view."
    )
    config = parser.parse_args()

    reader = Structured3DReader(
        [
            os.path.join(config.dataset_root, f)
            for f in os.listdir(config.dataset_root)
            if f.endswith(".zip")
        ]
    )

    scenes_list = reader.listdir("Structured3D")
    scenes_list = sorted(scenes_list)
    os.makedirs(os.path.join(config.output_root, "train"), exist_ok=True)
    os.makedirs(os.path.join(config.output_root, "val"), exist_ok=True)
    os.makedirs(os.path.join(config.output_root, "test"), exist_ok=True)

    # Preprocess data.
    print("Processing scenes...")
    pool = ProcessPoolExecutor(max_workers=config.num_workers)
    _ = list(
        pool.map(
            parse_scene,
            scenes_list,
            repeat(config.dataset_root),
            repeat(config.output_root),
            repeat(config.ignore_index),
            repeat(config.grid_size),
            repeat(config.fuse_prsp),
            repeat(config.fuse_pano),
        )
    )
    pool.shutdown()
