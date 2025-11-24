"""
nuScenes Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Zheng Zhang
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
from collections.abc import Sequence
import pickle
from PIL import Image
import open3d as o3d
import torch

from .builder import DATASETS
from .defaults import DefaultDataset, DefaultImagePointDataset

os.environ["OMP_NUM_THREADS"] = "1"


@DATASETS.register_module()
class NuScenesDataset(DefaultDataset):
    def __init__(self, sweeps=10, ignore_index=-1, **kwargs):
        self.sweeps = sweeps
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        super().__init__(ignore_index=ignore_index, **kwargs)

    def get_info_path(self, split):
        assert split in ["train", "val", "test"]
        if split == "train":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_train.pkl"
            )
        elif split == "val":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_val.pkl"
            )
        elif split == "test":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_test.pkl"
            )
        else:
            raise NotImplementedError

    def get_data_list(self):
        if isinstance(self.split, str):
            info_paths = [self.get_info_path(self.split)]
        elif isinstance(self.split, Sequence):
            info_paths = [self.get_info_path(s) for s in self.split]
        else:
            raise NotImplementedError
        data_list = []
        for info_path in info_paths:
            with open(info_path, "rb") as f:
                info = pickle.load(f)
                data_list.extend(info)
        return data_list

    def get_data(self, idx):
        data = self.data_list[idx % len(self.data_list)]
        lidar_path = os.path.join(self.data_root, "raw", data["lidar_path"])
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape(
            [-1, 5]
        )
        coord = points[:, :3]
        strength = points[:, 3].reshape([-1, 1]) / 255  # scale strength to [0, 1]

        if "gt_segment_path" in data.keys():
            gt_segment_path = os.path.join(
                self.data_root, "raw", data["gt_segment_path"]
            )
            segment = np.fromfile(
                str(gt_segment_path), dtype=np.uint8, count=-1
            ).reshape([-1])
            segment = np.vectorize(self.learning_map.__getitem__)(segment).astype(
                np.int64
            )
        else:
            segment = np.ones((points.shape[0],), dtype=np.int64) * self.ignore_index
        data_dict = dict(
            coord=coord,
            strength=strength,
            segment=segment,
            name=self.get_data_name(idx),
        )
        return data_dict

    def get_data_name(self, idx):
        # return data name for lidar seg, optimize the code when need to support detection
        return self.data_list[idx % len(self.data_list)]["lidar_token"]

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,
            1: ignore_index,
            2: 6,
            3: 6,
            4: 6,
            5: ignore_index,
            6: 6,
            7: ignore_index,
            8: ignore_index,
            9: 0,
            10: ignore_index,
            11: ignore_index,
            12: 7,
            13: ignore_index,
            14: 1,
            15: 2,
            16: 2,
            17: 3,
            18: 4,
            19: ignore_index,
            20: ignore_index,
            21: 5,
            22: 8,
            23: 9,
            24: 10,
            25: 11,
            26: 12,
            27: 13,
            28: 14,
            29: ignore_index,
            30: 15,
            31: ignore_index,
        }
        return learning_map


@DATASETS.register_module()
class NuScenesColorNormalDataset(NuScenesDataset):
    @staticmethod
    def estimate_normals(points, center=np.array([0, 0, 0])):
        normals = points - center[None, :]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / norms
        return normals

    def get_data(self, idx):
        data = self.data_list[idx % len(self.data_list)]
        lidar_path = os.path.join(self.data_root, "raw", data["lidar_path"])
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape(
            [-1, 5]
        )
        coord = points[:, :3]
        normal = self.estimate_normals(coord)
        if "gt_segment_path" in data.keys():
            gt_segment_path = os.path.join(
                self.data_root, "raw", data["gt_segment_path"]
            )
            segment = np.fromfile(
                str(gt_segment_path), dtype=np.uint8, count=-1
            ).reshape([-1])
            segment = np.vectorize(self.learning_map.__getitem__)(segment).astype(
                np.int64
            )
        else:
            segment = np.ones((points.shape[0],), dtype=np.int64) * self.ignore_index
        data_dict = dict(
            coord=coord,
            color=np.zeros_like(coord),  # placeholder for color
            normal=np.zeros_like(coord),  # placeholder for normal
            segment=segment,
            name=self.get_data_name(idx),
        )
        return data_dict


@DATASETS.register_module()
class NuScenesImagePointDataset(DefaultImagePointDataset):
    CAMERA_TYPES = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    def __init__(
        self,
        if_img=False,
        if_sweep=False,
        sweeps_max=10,
        sweeps=10,
        sweep_gap=1,
        ignore_index=-1,
        img_num=4,
        **kwargs,
    ):
        self.sweeps = sweeps
        self.sweep_gap = sweep_gap
        self.sweeps_max = sweeps_max
        self.if_sweep = if_sweep
        self.if_img = if_img
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.img_ratio = img_num / (6 * sweeps)
        super().__init__(ignore_index=ignore_index, **kwargs)

    @staticmethod
    def project_lidar_to_image_with_color(
        lidar_points,  # shape: (N, 3) or (N, 4)
        image,  # shape: (H, W, 3), uint8 RGB
        cam_intrinsic,  # shape: (3, 3)
        lidar_to_cam,  # shape: (4, 4)
        lidar_colors,
    ):
        """
        Projects LiDAR points to the image, fetches pixel color and pixel coordinates.
        Returns:
            filtered_points: (M, 3) - 3D points in camera frame that project onto the image.
            colors:          (M, 3) - RGB colors at projected 2D locations.
            uv_coords:       (M, 2) - Integer pixel coordinates (u, v) on the image.
            mask:            (N,)   - (optional) Boolean mask indicating which lidar points are used.
        """
        lidar_uv_coords = np.full(
            (lidar_points.shape[0], 2), -1, dtype=int
        )  # Default to (-1, -1)
        lidar_points_coord = lidar_points[:, :3]
        ones = np.ones((lidar_points_coord.shape[0], 1))
        lidar_hom = np.concatenate([lidar_points_coord, ones], axis=1)  # (N, 4)

        points_cam = (lidar_to_cam @ lidar_hom.T).T  # (N, 4)

        valid = points_cam[:, 2] > 0
        points_cam = points_cam[valid]

        pts_2d = (cam_intrinsic @ points_cam[:, :3].T).T  # (N, 3)
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]  # (N, 2) - pixel (u, v)

        H, W = image.shape[:2]
        u, v = pts_2d[:, 0], pts_2d[:, 1]
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)

        u = u[inside].astype(int)
        v = v[inside].astype(int)

        mask = np.zeros(lidar_points.shape[0], dtype=bool)
        mask[np.where(valid)[0][inside]] = True

        lidar_colors[mask] = image[v, u, :]
        lidar_uv_coords[mask] = np.stack([u, v], axis=1)  # (M, 2)
        return lidar_colors, lidar_uv_coords, mask

    def get_info_path(self, split):
        assert split in ["train", "val", "test"]
        if split == "train":
            return os.path.join(
                self.data_root,
                "info",
                f"nuscenes_infos_{self.sweeps_max}sweeps_train.pkl",
            )
        elif split == "val":
            return os.path.join(
                self.data_root,
                "info",
                f"nuscenes_infos_{self.sweeps_max}sweeps_val.pkl",
            )
        elif split == "test":
            return os.path.join(
                self.data_root,
                "info",
                f"nuscenes_infos_{self.sweeps_max}sweeps_test.pkl",
            )
        else:
            raise NotImplementedError

    def get_data_list(self):
        split_list = {}
        if isinstance(self.split, str):
            info_paths = [self.get_info_path(self.split)]
            split = [self.split]
        elif isinstance(self.split, Sequence):
            split = self.split
            info_paths = [self.get_info_path(s) for s in self.split]
        else:
            raise NotImplementedError
        data_list = []
        for info_path, split_i in zip(info_paths, split):
            with open(info_path, "rb") as f:
                info = pickle.load(f)
                data_list.extend(info)
                split_list[split_i] = list([i["token"] for i in info])
        return data_list, split_list

    def get_data(self, idx):
        data = self.data_list[idx % len(self.data_list)]
        lidar_path = os.path.join(self.data_root, "raw", data["lidar_path"])
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape(
            [-1, 5]
        )

        imgs = []
        cam_coords = []
        cam_normals = []
        cam_colors = []
        cam_strengths = []
        cam_correspondences = []
        correspondence_start = 0
        frame_pcd_offset = []
        lidar_colors = np.zeros((points.shape[0], 3), dtype=int)  # Default to black
        for id, cam_name in enumerate(self.CAMERA_TYPES):
            cam_info = data["cams"][cam_name]
            cam_intrinsic = cam_info["camera_intrinsics"]
            cam_image = Image.open(
                os.path.join(self.data_root, "raw", data["cams"][cam_name]["data_path"])
            )
            cam_image_np = np.array(cam_image)
            sensor2lidar = np.eye(4)
            sensor2lidar[:3, :3] = cam_info["sensor2lidar_rotation"]
            sensor2lidar[:3, 3] = cam_info["sensor2lidar_translation"]
            lidar2sensor = np.linalg.inv(sensor2lidar)
            lidar_colors, correspondence_info, _ = (
                self.project_lidar_to_image_with_color(
                    points, cam_image_np, cam_intrinsic, lidar2sensor, lidar_colors
                )
            )
            correspondence_point_id = (
                np.array(range(correspondence_info.shape[0])).reshape((-1, 1))
                + correspondence_start
            )
            correspondence_info = np.hstack(
                [correspondence_info, correspondence_point_id]
            )
            if np.random.rand() < self.img_ratio:
                cam_correspondences.append(correspondence_info)
                imgs.append(cam_image)
        correspondence_start += points.shape[0]
        cam_coord = points[:, :3]
        cam_center = np.array([0, 0, 0])
        cam_normal = self.get_normals(cam_center, cam_coord)
        cam_normals.append(cam_normal)
        cam_strength = points[:, 3].reshape([-1, 1]) / 255
        cam_coords.append(cam_coord)
        cam_colors.append(lidar_colors)
        cam_strengths.append(cam_strength)

        if self.if_sweep:
            frame_pcd_offset.append(points.shape[0])
            for id, sweep in enumerate(
                data["sweeps"][: (self.sweep_gap * self.sweeps) : self.sweep_gap]
            ):
                lidar_path = os.path.join(self.data_root, "raw", sweep["lidar_path"])
                points = np.fromfile(
                    str(lidar_path), dtype=np.float32, count=-1
                ).reshape([-1, 5])
                lidar_colors = np.zeros(
                    (points.shape[0], 3), dtype=int
                )  # Default to black
                cam_lidar_tm = (
                    sweep["transform_matrix"]
                    if sweep["transform_matrix"] is not None
                    else np.eye(4)
                )
                for id, cam_name in enumerate(self.CAMERA_TYPES):
                    cam_info = sweep["cams"][cam_name]
                    # cam_image_np = np.array(imgs[id])
                    cam_intrinsic = cam_info["camera_intrinsics"]
                    cam_image = Image.open(
                        os.path.join(self.data_root, "raw", cam_info["data_path"])
                    )
                    cam_image_np = np.array(cam_image)
                    sensor2lidar = np.eye(4)
                    sensor2lidar[:3, :3] = cam_info["sensor2lidar_rotation"]
                    sensor2lidar[:3, 3] = cam_info["sensor2lidar_translation"]
                    # sensor2lidar = cam_lidar_tm @ sensor2lidar
                    lidar2sensor = np.linalg.inv(sensor2lidar)
                    lidar_colors, correspondence_info, _ = (
                        self.project_lidar_to_image_with_color(
                            points,
                            cam_image_np,
                            cam_intrinsic,
                            lidar2sensor,
                            lidar_colors,
                        )
                    )
                    correspondence_point_id = (
                        np.array(range(correspondence_info.shape[0])).reshape((-1, 1))
                        + correspondence_start
                    )
                    correspondence_info = np.hstack(
                        [correspondence_info, correspondence_point_id]
                    )
                    if np.random.rand() < self.img_ratio:
                        cam_correspondences.append(correspondence_info)
                        imgs.append(cam_image)
                correspondence_start += correspondence_info.shape[0]
                frame_pcd_offset.append(correspondence_start)
                cam_coord = points[:, :3]
                cam_center = np.array([0, 0, 0])
                cam_normal = self.get_normals(cam_center, cam_coord)
                cam_normals.append(cam_normal)
                ones = np.ones((points.shape[0], 1))
                cam_coord_hom = np.concatenate([cam_coord, ones], axis=1)  # (N, 4)
                cam_coord = cam_coord_hom @ cam_lidar_tm.T
                cam_coord = cam_coord[:, :3]
                cam_strength = points[:, 3].reshape([-1, 1]) / 255
                cam_coords.append(cam_coord)
                cam_colors.append(lidar_colors)
                cam_strengths.append(cam_strength)

        coord = np.vstack(cam_coords)
        color = np.vstack(cam_colors)
        normal = np.vstack(cam_normals)
        strength = np.vstack(cam_strengths)
        frame_pcd_offset = np.array(frame_pcd_offset)

        car_from_ref = np.linalg.inv(data["ref_from_car"])
        coord_homo = np.hstack((coord, np.ones((coord.shape[0], 1))))
        coord_homo = coord_homo @ car_from_ref.T
        coord = coord_homo[:, :3]

        img_assets = dict()
        if self.if_img:
            if len(imgs) > 0:
                img_width, img_height = imgs[0].size
                div_w = img_width // self.patch_w
                div_h = img_height // self.patch_h
                div_min = max(min(div_w, div_h), 1)
                crop_img_width = div_min * self.patch_w
                crop_img_height = div_min * self.patch_h
                left = int((img_width - crop_img_width) / 2)
                top = int((img_height - crop_img_height) / 2)
                right = int((img_width + crop_img_width) / 2)
                bottom = int((img_height + crop_img_height) / 2)
                imgs = [img.crop((left, top, right, bottom)) for img in imgs]
                imgs = [self.transform_img(img) for img in imgs]
                imgs_list = torch.stack(imgs)
                img_assets["images"] = imgs_list.float()
            else:
                img_assets["images"] = torch.empty(
                    (
                        0,
                        3,
                        self.patch_h * self.patch_size,
                        self.patch_w * self.patch_size,
                    )
                )
            img_assets["img_num"] = np.array(
                [img_assets["images"].shape[0]], dtype=np.int32
            )

            correspondence_infos = np.ones(
                (coord.shape[0], len(cam_correspondences), 2), dtype=np.int32
            ) * (-1)
            for id, correspondence_info in enumerate(cam_correspondences):
                correspondence_info = self.resize_correspondence_info(
                    correspondence_info,
                    (self.patch_h * self.patch_size, self.patch_w * self.patch_size),
                    (img_height, img_width),
                    (left, top, right, bottom),
                    self.patch_size,
                )
                correspondence_infos[correspondence_info[:, -1], id, :] = (
                    correspondence_info[:, :-1]
                )
            img_assets["correspondence"] = correspondence_infos
        if "gt_segment_path" in data.keys():
            gt_segment_path = os.path.join(
                self.data_root, "raw", data["gt_segment_path"]
            )
            segment = np.fromfile(
                str(gt_segment_path), dtype=np.uint8, count=-1
            ).reshape([-1])
            segment = np.vectorize(self.learning_map.__getitem__)(segment).astype(
                np.int64
            )
        else:
            segment = np.ones((points.shape[0],), dtype=np.int64) * self.ignore_index

        color = color.astype(np.float32)
        if self.if_sweep:
            data_dict = dict(
                coord=coord,
                color=color,
                normal=normal,
                strength=strength,
                segment=segment,
                frame_pcd_offset=frame_pcd_offset,
                name=self.get_data_name(idx),
            )
        else:
            data_dict = dict(
                coord=coord,
                color=color,
                normal=normal,
                strength=strength,
                segment=segment,
                name=self.get_data_name(idx),
            )
        data_dict.update(img_assets)
        return data_dict

    def get_data_name(self, idx):
        return self.data_list[idx % len(self.data_list)]["lidar_token"]

    @staticmethod
    def get_normals(cam_center, coords):
        Cs = np.repeat(cam_center.reshape((1, -1)), coords.shape[0], axis=0)
        view_dirs = coords - Cs
        view_dirs = view_dirs / np.linalg.norm(view_dirs, axis=-1, keepdims=True)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.estimate_normals()
        normals = np.asarray(pcd.normals)
        dot_product = np.sum(normals * view_dirs, axis=-1)
        flip_mask = dot_product > 0
        normals[flip_mask] = -normals[flip_mask]
        normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
        return normals

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,
            1: ignore_index,
            2: 6,
            3: 6,
            4: 6,
            5: ignore_index,
            6: 6,
            7: ignore_index,
            8: ignore_index,
            9: 0,
            10: ignore_index,
            11: ignore_index,
            12: 7,
            13: ignore_index,
            14: 1,
            15: 2,
            16: 2,
            17: 3,
            18: 4,
            19: ignore_index,
            20: ignore_index,
            21: 5,
            22: 8,
            23: 9,
            24: 10,
            25: 11,
            26: 12,
            27: 13,
            28: 14,
            29: ignore_index,
            30: 15,
            31: ignore_index,
        }
        return learning_map
